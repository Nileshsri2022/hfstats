"""Full streaming benchmark for working model:provider pairs."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

CLASSIFIED_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
OUT_PATH = os.environ.get("RESULTS_PATH", "scripts/results.json")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_GROUP = os.environ.get("MODEL_GROUP", "group1")
MAX_BENCHMARK_PAIRS = int(os.environ.get("MAX_BENCHMARK_PAIRS", "50"))
INFERENCE_URL = "https://api-inference.huggingface.co/models/"
PROMPT = (
    "Write a short paragraph explaining the concept of neural networks "
    "in machine learning. Be concise but informative."
)
MAX_TOKENS = 500
REQUEST_TIMEOUT = 120


def benchmark_pair(model: str, provider: str) -> dict:
    url = f"{INFERENCE_URL}{model}:{provider}"
    payload = json.dumps(
        {
            "model": f"{model}:{provider}",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {HF_TOKEN}",
        },
        method="POST",
    )

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
        if status == 429:
            error_category = "rate_limited"
        elif status == 503:
            error_category = "loading"
        elif status == 529:
            error_category = "overloaded"
        elif status == 402:
            error_category = "quota_exceeded"
        elif status == 404:
            error_category = "not_found"
        elif status == 400:
            error_category = "unsupported"
        else:
            error_category = "provider_bug"
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
    with open(CLASSIFIED_PATH, "r", encoding="utf-8") as f:
        classified = json.load(f)

    working = [p for p in classified.get("pairs", []) if p.get("status") == "working"]
    working.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    capped = working[:MAX_BENCHMARK_PAIRS]

    # Split into 3 groups
    groups = {"group1": [], "group2": [], "group3": []}
    for i, p in enumerate(capped):
        groups[["group1", "group2", "group3"][i % 3]].append(p)

    my_pairs = groups.get(MODEL_GROUP, [])
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

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(results)} results to {OUT_PATH}")


if __name__ == "__main__":
    main()
