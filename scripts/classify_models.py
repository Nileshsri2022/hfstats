"""Classify model:provider pairs with tiny inference test."""
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
    utc_now_iso,
    write_json,
)

CANDIDATES_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")
OUT_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
PROMPT = "Hello, count to three: 1,"
TINY_MAX_TOKENS = 5
REQUEST_TIMEOUT = 25
RETRY_DELAY = 5
BACKOFF_DELAY = 10


def classify_pair(model: str, provider: str) -> dict:
    req = build_chat_request(
        model, provider, PROMPT, TINY_MAX_TOKENS, provider_in_url=True
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
        category = categorize_http_status(status)
        if category == "provider_bug":
            detail = f"http {status}: {detail}"
        return {"status": category, "detail": detail}

    except urllib.error.URLError as e:
        reason = str(e.reason).lower()
        if "timeout" in reason or "timed out" in reason:
            return {"status": "cold_start", "detail": reason}
        return {"status": "provider_bug", "detail": reason}

    except Exception as e:
        return {"status": "provider_bug", "detail": str(e)}


def main():
    print("Loading candidates...")
    candidates = load_json(CANDIDATES_PATH) or {}

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
        "timestamp": utc_now_iso(),
        "total": len(results),
        "counts": counts,
        "pairs": results,
    }

    write_json(OUT_PATH, payload)

    print(f"Wrote classified pairs to {OUT_PATH}")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
