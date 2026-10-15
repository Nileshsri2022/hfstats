"""Full streaming benchmark for working model:provider pairs."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    build_chat_request,
    categorize_error_text,
    categorize_http_status,
    gh_annotation,
    load_json,
    write_json,
)

CLASSIFIED_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
OUT_PATH = os.environ.get("RESULTS_PATH", "scripts/results.json")
MODEL_GROUP = os.environ.get("MODEL_GROUP", "group1")
MAX_BENCHMARK_PAIRS = int(os.environ.get("MAX_BENCHMARK_PAIRS", "50"))
PROMPT = (
    "Write a short paragraph explaining the concept of neural networks "
    "in machine learning. Be concise but informative."
)
MAX_TOKENS = 500
REQUEST_TIMEOUT = 120
# Only a preview of each answer is stored: full answers were never displayed
# but would have grown history.db (downloaded by every dashboard visitor) by
# ~45 MB over the retention window.
RESPONSE_PREVIEW_CHARS = 300
GROUPS = ("group1", "group2", "group3")


def _delta_parts(choice) -> tuple:
    """Return ``(content, reasoning)`` text from a streamed choice delta.

    Reasoning models (DeepSeek-R1, gpt-oss, Qwen3, ...) stream their thinking
    as ``reasoning_content``/``reasoning`` before any ``content``; those are
    generated tokens too and must count toward TTFT and throughput.
    """
    delta = choice.get("delta") if isinstance(choice, dict) else None
    if not isinstance(delta, dict):
        return "", ""
    content = delta.get("content")
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else "",
    )


def benchmark_pair(model: str, provider: str) -> dict:
    req = build_chat_request(model, provider, PROMPT, MAX_TOKENS, stream=True)

    start = time.perf_counter()
    first_token_at = None
    token_chunks = 0
    completion_tokens = None
    total_tokens = None
    content_parts = []
    reasoning_parts = []
    stream_error = None

    def failure(error, category, response_time):
        return {
            "model": model,
            "provider": provider,
            "success": False,
            "error": error,
            "error_category": category,
            "response_time": response_time,
            "ttft": int((first_token_at - start) * 1000) if first_token_at else None,
            "tokens_generated": completion_tokens or token_chunks,
            "total_tokens": total_tokens,
            "throughput": None,
            "response": "",
        }

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue

                # Some providers report errors inside a 200 OK stream.
                if chunk.get("error"):
                    stream_error = str(chunk["error"])[:500]
                    break

                got_token = False
                for c in chunk.get("choices") or []:
                    content, reasoning = _delta_parts(c)
                    if content:
                        content_parts.append(content)
                        got_token = True
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        got_token = True
                if got_token:
                    token_chunks += 1
                    if first_token_at is None:
                        # TTFT = time until the first generated token, not until
                        # the HTTP headers arrive (which is almost immediate).
                        first_token_at = time.perf_counter()

                # Usage arrives in the final chunk (requested via stream_options).
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens") or completion_tokens
                    total_tokens = usage.get("total_tokens") or total_tokens

        end = time.perf_counter()
        response_time = int((end - start) * 1000)

        if stream_error:
            return failure(stream_error, categorize_error_text(stream_error), response_time)
        if token_chunks == 0:
            return failure("No tokens generated", "provider_bug", response_time)

        # Prefer the provider's real token count; chunk counts undercount when a
        # provider packs several tokens into one chunk.
        tokens_generated = completion_tokens or token_chunks
        total_tokens = total_tokens or tokens_generated

        # Output speed: tokens after the first one / time after the first token.
        gen_seconds = end - first_token_at
        if tokens_generated > 1 and gen_seconds > 0:
            throughput = round((tokens_generated - 1) / gen_seconds, 2)
        elif end - start > 0:
            throughput = round(tokens_generated / (end - start), 2)
        else:
            throughput = 0

        response_text = "".join(content_parts)
        if not response_text and reasoning_parts:
            response_text = "[reasoning] " + "".join(reasoning_parts)

        return {
            "model": model,
            "provider": provider,
            "success": True,
            "error": None,
            "error_category": "working",
            "response_time": response_time,
            "ttft": int((first_token_at - start) * 1000),
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "response": response_text[:RESPONSE_PREVIEW_CHARS],
        }

    except urllib.error.HTTPError as e:
        response_time = int((time.perf_counter() - start) * 1000)
        body = e.read().decode("utf-8", errors="replace")[:500]
        return failure(f"HTTP {e.code}: {body}", categorize_http_status(e.code), response_time)

    except urllib.error.URLError as e:
        response_time = int((time.perf_counter() - start) * 1000)
        reason = str(e.reason).lower()
        category = "cold_start" if ("timeout" in reason or "timed out" in reason) else "provider_bug"
        return failure(reason, category, response_time)

    except TimeoutError:
        # Read timeouts while waiting for headers or mid-stream are raised as
        # TimeoutError, not URLError.
        response_time = int((time.perf_counter() - start) * 1000)
        return failure(f"timed out after {REQUEST_TIMEOUT}s", "cold_start", response_time)

    except Exception as e:
        response_time = int((time.perf_counter() - start) * 1000)
        return failure(str(e), "provider_bug", response_time)


def main():
    print(f"Loading classified pairs from {CLASSIFIED_PATH} ...")
    classified = load_json(CLASSIFIED_PATH) or {}

    working = [p for p in classified.get("pairs", []) if p.get("status") == "working"]
    working.sort(key=lambda x: x.get("downloads") or 0, reverse=True)
    capped = working[:MAX_BENCHMARK_PAIRS]

    # Split into 3 groups (round-robin so every group gets popular pairs)
    groups = {g: [] for g in GROUPS}
    for i, p in enumerate(capped):
        groups[GROUPS[i % len(GROUPS)]].append(p)

    if MODEL_GROUP not in groups:
        print(
            f"ERROR: invalid MODEL_GROUP={MODEL_GROUP!r}; expected one of "
            f"{sorted(groups)}",
            file=sys.stderr,
        )
        sys.exit(1)

    my_pairs = groups[MODEL_GROUP]
    print(f"Group {MODEL_GROUP}: {len(my_pairs)} pairs to benchmark")

    results = []
    for i, p in enumerate(my_pairs):
        model = p["model"]
        provider = p["provider"]
        print(f"[{i+1}/{len(my_pairs)}] Benchmarking {model}:{provider} ...")
        r = benchmark_pair(model, provider)
        status = "OK" if r["success"] else r.get("error_category", "fail")
        print(
            f"  -> {status} time={r.get('response_time')}ms ttft={r.get('ttft')}ms "
            f"tok={r.get('tokens_generated')} tok/s={r.get('throughput')}"
        )
        results.append(r)

    payload = {
        "timestamp": classified.get("timestamp"),
        "group": MODEL_GROUP,
        "prompt": PROMPT,
        "results": results,
    }

    write_json(OUT_PATH, payload)

    print(f"Wrote {len(results)} results to {OUT_PATH}")

    ok = [r for r in results if r.get("success")]
    if results:
        ttfts = sorted(r["ttft"] for r in ok if r.get("ttft") is not None)
        tps = sorted(r["throughput"] for r in ok if r.get("throughput") is not None)
        msg = f"{len(ok)}/{len(results)} succeeded"
        if ttfts:
            msg += f"; median TTFT {ttfts[len(ttfts) // 2]} ms"
        if tps:
            msg += f"; median throughput {tps[len(tps) // 2]} tok/s"
        gh_annotation("notice", msg, title=f"Benchmark {MODEL_GROUP}")
        failed = Counter(r.get("error_category") or "fail" for r in results if not r.get("success"))
        if failed:
            gh_annotation(
                "warning",
                ", ".join(f"{n} {cat}" for cat, n in failed.most_common()),
                title=f"Benchmark {MODEL_GROUP} failures",
            )


if __name__ == "__main__":
    main()
