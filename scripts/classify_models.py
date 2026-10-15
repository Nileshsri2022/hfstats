"""Classify model:provider pairs with tiny inference test."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import (
    build_chat_request,
    categorize_error_text,
    categorize_http_status,
    gh_annotation,
    load_json,
    utc_now_iso,
    write_json,
)

CANDIDATES_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")
OUT_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
PROMPT = "Hello, count to three: 1,"
TOKEN_HINT = (
    "HF_TOKEN must be a Hugging Face access token with the "
    "'Make calls to Inference Providers' permission, on an account with "
    "inference credits left."
)
TINY_MAX_TOKENS = 5
REQUEST_TIMEOUT = 25
RETRY_DELAY = 5
BACKOFF_DELAY = 10
# Providers are classified in parallel, but each provider only ever sees one
# request at a time from us, so we never trip provider-side concurrency limits.
CONCURRENCY = max(1, int(os.environ.get("CLASSIFY_CONCURRENCY", "6")))
STATUSES = (
    "working",
    "loading",
    "overloaded",
    "rate_limited",
    "quota_exceeded",
    "cold_start",
    "auth_error",
    "provider_bug",
    "not_found",
    "unsupported",
)


def classify_pair(model: str, provider: str) -> dict:
    req = build_chat_request(model, provider, PROMPT, TINY_MAX_TOKENS)

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return {"status": "provider_bug", "detail": "malformed json"}

        # Error object returned with a 2xx status
        if isinstance(data, dict) and data.get("error"):
            err = str(data["error"]).lower()
            return {"status": categorize_error_text(err), "detail": err[:200]}

        if isinstance(data, dict) and "choices" in data:
            return {"status": "working", "detail": "ok"}
        if isinstance(data, list) and len(data) > 0:
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

    except TimeoutError:
        # Slow responses time out while waiting for headers/body; urllib does
        # not wrap these in URLError.
        return {"status": "cold_start", "detail": f"timed out after {REQUEST_TIMEOUT}s"}

    except Exception as e:
        return {"status": "provider_bug", "detail": str(e)}


def classify_with_retries(model: str, provider: str) -> dict:
    info = classify_pair(model, provider)
    # Retry loading once
    if info["status"] == "loading":
        time.sleep(RETRY_DELAY)
        info = classify_pair(model, provider)
    # Backoff for rate limited
    if info["status"] == "rate_limited":
        time.sleep(BACKOFF_DELAY)
        info = classify_pair(model, provider)
    return info


def classify_all(pairs: list) -> list:
    """Classify every pair; returns infos in the same order as ``pairs``."""
    infos = [None] * len(pairs)
    queues = {}
    for idx, p in enumerate(pairs):
        queues.setdefault(p["provider"], []).append(idx)

    lock = threading.Lock()
    done = [0]

    def run_queue(indices):
        for idx in indices:
            p = pairs[idx]
            info = classify_with_retries(p["model"], p["provider"])
            infos[idx] = info
            with lock:
                done[0] += 1
                n = done[0]
            print(f"[{n}/{len(pairs)}] {p['model']}:{p['provider']} -> {info['status']}", flush=True)

    # Longest queues first so the slowest provider starts immediately.
    ordered = sorted(queues.values(), key=len, reverse=True)
    workers = max(1, min(CONCURRENCY, len(ordered)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_queue, ordered))
    return infos


def main():
    print("Loading candidates...")
    candidates = load_json(CANDIDATES_PATH) or {}

    pairs = candidates.get("pairs", [])
    if not pairs:
        print(f"ERROR: no candidate pairs in {CANDIDATES_PATH}; nothing to classify.", file=sys.stderr)
        sys.exit(1)

    if not common.HF_TOKEN:
        # Without a token the router rejects every request with 401; fail fast
        # with an actionable message instead of sending hundreds of requests.
        msg = f"HF_TOKEN is not set (add it as a repository secret). {TOKEN_HINT}"
        print(f"ERROR: {msg}", file=sys.stderr)
        gh_annotation("error", msg, title="Missing HF_TOKEN")
        sys.exit(1)

    providers = {p["provider"] for p in pairs}
    print(f"Classifying {len(pairs)} pairs across {len(providers)} providers (concurrency={CONCURRENCY})...")
    infos = classify_all(pairs)

    results = []
    counts = {status: 0 for status in STATUSES}
    for p, info in zip(pairs, infos):
        status = info["status"]
        counts[status] = counts.get(status, 0) + 1
        results.append(
            {
                "model": p["model"],
                "provider": p["provider"],
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
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)

    if counts["working"] == 0:
        # Every pair failing at once is (almost) never the providers' fault: it
        # means a broken endpoint, a bad token or exhausted credits. Fail the
        # workflow instead of quietly recording another empty run.
        print("ERROR: no pair is working. Most common failures:", file=sys.stderr)
        gh_annotation("error", f"No working pair among {len(results)}: {summary}. {TOKEN_HINT}",
                      title="Classification failed")
        failures = Counter((r["status"], str(r["detail"] or "")[:300]) for r in results)
        for (status, detail), n in failures.most_common(5):
            print(f"  {n:4d} x {status}: {detail}", file=sys.stderr)
            gh_annotation("error", f"{n} x {status}: {detail}", title=f"Top failure: {status}")
        print(f"Check that {TOKEN_HINT}", file=sys.stderr)
        sys.exit(1)

    gh_annotation("notice", f"{len(results)} pairs: {summary}", title="Classification")


if __name__ == "__main__":
    main()
