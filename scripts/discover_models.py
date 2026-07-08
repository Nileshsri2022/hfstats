"""HF API discovery with provider-level extraction."""
import json
import os
import sys
import time
import re
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import utc_now_iso, write_json

API_URL = "https://huggingface.co/api/models"
MIN_DOWNLOADS = int(os.environ.get("MIN_DOWNLOADS", "10000"))
LIMIT = int(os.environ.get("DISCOVER_LIMIT", "500"))
OUT_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")
MAX_ATTEMPTS = 5
TRANSIENT_BACKOFF = 5


def fetch_models():
    all_models = []
    next_url = f"{API_URL}?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=100&expand=inferenceProviderMapping"
    batch_count = 0

    while len(all_models) < LIMIT and next_url:
        req = urllib.request.Request(next_url, headers={"Accept": "application/json"})
        data = None
        page_url = next_url
        next_url = None
        last_error = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
                    link = resp.headers.get("Link", "")
                    data = json.loads(raw)

                    m = re.search(r'<([^>]+)>[^<]*rel="next"', link)
                    if m:
                        next_url = m.group(1)
                    else:
                        next_url = None
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:
                    wait = int(e.headers.get("Retry-After", 30))
                    print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
                    time.sleep(wait)
                else:
                    raise
            except urllib.error.URLError as e:
                # Transient network/timeout error: retry with backoff.
                last_error = e
                print(f"Network error ({e.reason}), retrying in {TRANSIENT_BACKOFF}s (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
                time.sleep(TRANSIENT_BACKOFF)

        # If we exhausted all attempts without a successful fetch, fail loudly
        # instead of silently returning a partial/empty candidate list.
        if data is None:
            raise RuntimeError(
                f"Failed to fetch models after {MAX_ATTEMPTS} attempts for {page_url}: {last_error}"
            )
        if not isinstance(data, list):
            raise RuntimeError(
                f"Unexpected response structure from HF API (expected a list, got {type(data).__name__}) for {page_url}"
            )

        batch_count += 1

        for m in data:
            if m.get("gated") is True:
                continue
            if m.get("private") is True:
                continue
            downloads = m.get("downloads", 0) or 0
            if downloads < MIN_DOWNLOADS:
                continue
            mapping = m.get("inferenceProviderMapping", [])
            if mapping and any(isinstance(x, dict) and x.get("status") == "live" for x in mapping):
                all_models.append(m)

        print(f"batch={batch_count}, candidates={len(all_models)}, has_next={next_url is not None}")
        if len(all_models) >= LIMIT:
            break

    return all_models[:LIMIT]


def extract_pairs(models):
    pairs = []
    for m in models:
        model_id = m["id"]
        downloads = m.get("downloads", 0) or 0
        likes = m.get("likes", 0) or 0
        mapping = m.get("inferenceProviderMapping", [])
        for info in mapping:
            if not isinstance(info, dict) or info.get("status") != "live":
                continue
            provider = info.get("provider")
            if not provider:
                continue
            pairs.append(
                {
                    "model": model_id,
                    "provider": provider,
                    "provider_id": info.get("providerId", model_id),
                    "downloads": downloads,
                    "likes": likes,
                }
            )
    return pairs


def main():
    print("Discovering models from HF API...")
    models = fetch_models()
    pairs = extract_pairs(models)

    payload = {
        "timestamp": utc_now_iso(),
        "total_models": len(models),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }
    write_json(OUT_PATH, payload)

    print(f"Wrote {len(models)} models, {len(pairs)} pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()