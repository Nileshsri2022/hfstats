"""Classify model:provider pairs with tiny inference test."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

CANDIDATES_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")
OUT_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INFERENCE_URL = "https://api-inference.huggingface.co/models/"
PROMPT = "Hello, count to three: 1,"
TINY_MAX_TOKENS = 5
REQUEST_TIMEOUT = 25
RETRY_DELAY = 5
BACKOFF_DELAY = 10


def classify_pair(model: str, provider: str) -> dict:
    url = f"{INFERENCE_URL}{model}:{provider}"
    payload = json.dumps(
        {
            "model": f"{model}:{provider}",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": TINY_MAX_TOKENS,
            "temperature": 0.7,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            # Check for JSON array which is valid HF chat-completion streaming response
            # but for non-streaming we should get a JSON object
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return {"status": "provider_bug", "detail": "malformed json"}

            # If it's an error object from provider
            if isinstance(data, dict) and data.get("error"):
                err = str(data["error"]).lower()
                if "rate limit" in err or "too many requests" in err:
                    return {"status": "rate_limited", "detail": err}
                if "quota" in err or "insufficient" in err:
                    return {"status": "quota_exceeded", "detail": err}
                if "not found" in err or "unknown" in err:
                    return {"status": "not_found", "detail": err}
                return {"status": "provider_bug", "detail": err}

            if isinstance(data, dict) and "choices" in data:
                return {"status": "working", "detail": "ok"}
            if isinstance(data, list) and len(data) > 0:
                # Could be streaming chunks aggregated somehow
                return {"status": "working", "detail": "list_response"}

            return {"status": "provider_bug", "detail": "unexpected structure"}

    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8", errors="replace")
        detail = body[:200]
        if status == 503:
            return {"status": "loading", "detail": detail}
        if status == 529:
            return {"status": "overloaded", "detail": detail}
        if status == 429:
            return {"status": "rate_limited", "detail": detail}
        if status == 402:
            return {"status": "quota_exceeded", "detail": detail}
        if status == 404:
            return {"status": "not_found", "detail": detail}
        if status == 400:
            return {"status": "unsupported", "detail": detail}
        return {"status": "provider_bug", "detail": f"http {status}: {detail}"}

    except urllib.error.URLError as e:
        reason = str(e.reason).lower()
        if "timeout" in reason or "timed out" in reason:
            return {"status": "cold_start", "detail": reason}
        return {"status": "provider_bug", "detail": reason}

    except Exception as e:
        return {"status": "provider_bug", "detail": str(e)}


def main():
    print("Loading candidates...")
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    pairs = candidates.get("pairs", [])
    results = []
    counts = {
        "working": 0,
        "loading": 0,
        "overloaded": 0,
        "rate_limited": 0,
        "quota_exceeded": 0,
        "cold_start": 0,
        "provider_bug": 0,
        "not_found": 0,
        "unsupported": 0,
    }

    for i, p in enumerate(pairs):
        model = p["model"]
        provider = p["provider"]
        print(f"[{i+1}/{len(pairs)}] Classifying {model}:{provider} ...", end=" ")
        info = classify_pair(model, provider)
        status = info["status"]

        # Retry loading once
        if status == "loading":
            print(f"{status} → retrying in {RETRY_DELAY}s...", end=" ")
            time.sleep(RETRY_DELAY)
            info = classify_pair(model, provider)
            status = info["status"]

        # Backoff for rate limited
        if status == "rate_limited":
            print(f"{status} → backing off {BACKOFF_DELAY}s...", end=" ")
            time.sleep(BACKOFF_DELAY)
            info = classify_pair(model, provider)
            status = info["status"]

        counts[status] = counts.get(status, 0) + 1
        print(status)

        results.append(
            {
                "model": model,
                "provider": provider,
                "provider_id": p.get("provider_id"),
                "downloads": p.get("downloads"),
                "likes": p.get("likes"),
                "status": status,
                "detail": info.get("detail"),
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "counts": counts,
        "pairs": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote classified pairs to {OUT_PATH}")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
