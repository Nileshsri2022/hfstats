"""Full streaming benchmark for working model:provider pairs."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    build_chat_request,
    categorize_http_status,
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


def benchmark_pair(model: str, provider: str) -> dict:
    req = build_chat_request(model, provider, PROMPT, MAX_TOKENS, stream=True)

    start = time.perf_counter()
    ttft = None
    tokens_generated = 0
    total_tokens = None
    response_chunks = []
    error = None
    error_category = None

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            first_byte = time.perf_counter()
            ttft = int((first_byte - start) * 1000)

            for line in resp:
                line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Collect choices
                choices = chunk.get("choices", [])
                for c in choices:
                    delta = c.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        response_chunks.append(content)
                        tokens_generated += 1

                # Usage may appear in the last chunk
                usage = chunk.get("usage")
                if usage and isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens")

        end = time.perf_counter()
        response_time = int((end - start) * 1000)
        response_text = "".join(response_chunks)

        if tokens_generated == 0:
            return {
                "model": model,
                "provider": provider,
                "success": False,
                "error": "No tokens generated",
                "error_category": "provider_bug",
                "response_time": response_time,
                "ttft": ttft,
                "tokens_generated": 0,
                "total_tokens": total_tokens,
                "response": response_text,
            }

        total_tokens = total_tokens or tokens_generated
        throughput = round(tokens_generated / (response_time / 1000), 2) if response_time > 0 else 0

        return {
            "model": model,
            "provider": provider,
            "success": True,
            "error": None,
            "error_category": "working",
            "response_time": response_time,
            "ttft": ttft,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "response": response_text,
        }

    except urllib.error.HTTPError as e:
        end = time.perf_counter()
        response_time = int((end - start) * 1000)
        body = e.read().decode("utf-8", errors="replace")[:500]
        status = e.code
        error_category = categorize_http_status(status)
        return {
            "model": model,
            "provider": provider,
            "success": False,
            "error": f"HTTP {status}: {body}",
            "error_category": error_category,
            "response_time": response_time,
            "ttft": ttft,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "response": "",
        }

    except urllib.error.URLError as e:
        end = time.perf_counter()
        response_time = int((end - start) * 1000)
        reason = str(e.reason).lower()
        error_category = "cold_start" if "timeout" in reason else "provider_bug"
        return {
            "model": model,
            "provider": provider,
            "success": False,
            "error": reason,
            "error_category": error_category,
            "response_time": response_time,
            "ttft": ttft,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "response": "",
        }

    except Exception as e:
        end = time.perf_counter()
        response_time = int((end - start) * 1000)
        return {
            "model": model,
            "provider": provider,
            "success": False,
            "error": str(e),
            "error_category": "provider_bug",
            "response_time": response_time,
            "ttft": ttft,
            "tokens_generated": tokens_generated,
            "total_tokens": total_tokens,
            "response": "",
        }


def main():
    print(f"Loading classified pairs from {CLASSIFIED_PATH} ...")
    classified = load_json(CLASSIFIED_PATH) or {}

    working = [p for p in classified.get("pairs", []) if p.get("status") == "working"]
    working.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    capped = working[:MAX_BENCHMARK_PAIRS]

    # Split into 3 groups
    groups = {"group1": [], "group2": [], "group3": []}
    for i, p in enumerate(capped):
        groups[["group1", "group2", "group3"][i % 3]].append(p)

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
        print(f"  -> {status} time={r.get('response_time')}ms ttft={r.get('ttft')}ms tok={r.get('tokens_generated')}")
        results.append(r)

    payload = {
        "timestamp": classified.get("timestamp"),
        "group": MODEL_GROUP,
        "prompt": PROMPT,
        "results": results,
    }

    write_json(OUT_PATH, payload)

    print(f"Wrote {len(results)} results to {OUT_PATH}")


if __name__ == "__main__":
    main()
