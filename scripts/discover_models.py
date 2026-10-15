"""HF API discovery with provider-level extraction."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import gh_annotation, hub_headers, utc_now_iso, write_json

API_URL = "https://huggingface.co/api/models"
MIN_DOWNLOADS = int(os.environ.get("MIN_DOWNLOADS", "10000"))
LIMIT = int(os.environ.get("DISCOVER_LIMIT", "500"))
MAX_PAGES = int(os.environ.get("DISCOVER_MAX_PAGES", "100"))
SKIP_GATED = os.environ.get("SKIP_GATED", "").strip().lower() in ("1", "true", "yes")
OUT_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")
PAGE_SIZE = 100
MAX_ATTEMPTS = 5
TRANSIENT_BACKOFF = 5
DEFAULT_RATE_LIMIT_WAIT = 30
MAX_RATE_LIMIT_WAIT = 310  # Hub rate limits are counted over 5-minute windows.
TRANSIENT_HTTP_CODES = {502, 503, 504}

# When ``expand`` is used, the Hub API returns *only* the listed properties
# (plus ids), so every field the filters below rely on must be requested.
# Without this, likes were always 0 and the gated/private filters never fired.
EXPAND_FIELDS = ("inferenceProviderMapping", "downloads", "likes", "gated", "private")


def first_page_url() -> str:
    params = [
        ("pipeline_tag", "text-generation"),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", str(PAGE_SIZE)),
    ] + [("expand", field) for field in EXPAND_FIELDS]
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def rate_limit_wait(headers, default: int = DEFAULT_RATE_LIMIT_WAIT) -> int:
    """Seconds to wait after a 429 response.

    The Hub sends an IETF-draft ``RateLimit`` header such as
    ``"api";r=0;t=55`` where ``t`` is the number of seconds until the window
    resets. Fall back to ``Retry-After`` and finally to ``default``.
    """
    headers = headers or {}
    match = re.search(r"\bt=(\d+)", str(headers.get("RateLimit") or ""))
    if match:
        return min(int(match.group(1)) + 1, MAX_RATE_LIMIT_WAIT)
    retry_after = str(headers.get("Retry-After") or "").strip()
    if retry_after.isdigit():
        return min(int(retry_after), MAX_RATE_LIMIT_WAIT)
    return default


def is_gated(model: dict) -> bool:
    """The Hub reports ``gated`` as False, "auto" or "manual" (older payloads: True)."""
    return model.get("gated") not in (None, False)


def fetch_page(url: str):
    """Fetch one listing page. Returns ``(models, next_url)``; raises on failure."""
    req = urllib.request.Request(url, headers=hub_headers())
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                link = resp.headers.get("Link", "") or ""
            data = json.loads(raw)
            if not isinstance(data, list):
                raise RuntimeError(
                    "Unexpected response structure from HF API (expected a list, "
                    f"got {type(data).__name__}) for {url}"
                )
            m = re.search(r'<([^>]+)>[^<]*rel="next"', link)
            return data, (m.group(1) if m else None)
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                wait = rate_limit_wait(e.headers)
                print(f"Rate limited by the Hub API, retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})...")
            elif e.code in TRANSIENT_HTTP_CODES:
                wait = TRANSIENT_BACKOFF * attempt
                print(f"Hub API returned HTTP {e.code}, retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})...")
            else:
                raise
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            # Transient network problem or read timeout: retry with backoff.
            last_error = e
            print(f"Network error ({getattr(e, 'reason', e)}), retrying in {TRANSIENT_BACKOFF}s (attempt {attempt}/{MAX_ATTEMPTS})...")
            time.sleep(TRANSIENT_BACKOFF)

    # Fail loudly instead of silently returning a partial/empty candidate list.
    raise RuntimeError(f"Failed to fetch models after {MAX_ATTEMPTS} attempts for {url}: {last_error}")


def fetch_models():
    all_models = []
    next_url = first_page_url()
    pages = 0

    while next_url and len(all_models) < LIMIT:
        if pages >= MAX_PAGES:
            print(f"Reached DISCOVER_MAX_PAGES={MAX_PAGES}; stopping pagination.")
            break
        data, next_url = fetch_page(next_url)
        pages += 1

        for m in data:
            if m.get("private") is True:
                continue
            if SKIP_GATED and is_gated(m):
                continue
            downloads = m.get("downloads", 0) or 0
            if downloads < MIN_DOWNLOADS:
                continue
            mapping = m.get("inferenceProviderMapping") or []
            if any(isinstance(x, dict) and x.get("status") == "live" for x in mapping):
                all_models.append(m)

        print(f"page={pages}, candidates={len(all_models)}, has_next={next_url is not None}")

        # Results are sorted by downloads (descending), so once a page ends below
        # the threshold no later page can contain a qualifying model. Without this
        # stop, discovery paged through the entire catalogue until the Hub's rate
        # limiter cut it off -- which is what made the workflow fail every run.
        if not data or (data[-1].get("downloads") or 0) < MIN_DOWNLOADS:
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
            # We call the chat-completions endpoint, which only serves
            # "conversational" mappings (absent task = accept, for safety).
            if info.get("task", "conversational") != "conversational":
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
    try:
        models = fetch_models()
    except RuntimeError as e:
        gh_annotation("error", str(e)[:1000], title="Model discovery failed")
        raise
    pairs = extract_pairs(models)

    payload = {
        "timestamp": utc_now_iso(),
        "total_models": len(models),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }
    write_json(OUT_PATH, payload)

    print(f"Wrote {len(models)} models, {len(pairs)} pairs to {OUT_PATH}")
    if not pairs:
        print("ERROR: discovery found no model:provider pairs.", file=sys.stderr)
        gh_annotation("error", f"No model:provider pairs among {len(models)} models.",
                      title="Model discovery failed")
        sys.exit(1)
    providers = len({p["provider"] for p in pairs})
    gh_annotation("notice", f"{len(models)} models, {len(pairs)} pairs across {providers} providers",
                  title="Discovery")


if __name__ == "__main__":
    main()
