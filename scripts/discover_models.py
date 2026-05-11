"""HF API discovery with provider-level extraction."""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

API_URL = "https://huggingface.co/api/models"
MIN_DOWNLOADS = int(os.environ.get("MIN_DOWNLOADS", "10000"))
LIMIT = int(os.environ.get("DISCOVER_LIMIT", "500"))
OUT_PATH = os.environ.get("CANDIDATES_PATH", "scripts/candidates.json")


def fetch_models():
    all_models = []
    offset = 0
    batch = 100

    while len(all_models) < LIMIT:
        url = (
            f"{API_URL}?pipeline_tag=text-generation"
            f"&sort=downloads&direction=-1&limit={batch}&offset={offset}"
            f"&expand[]=inferenceProviderMapping"
        )
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not isinstance(data, list) or not data:
            break

        for m in data:
            if m.get("gated", False):
                continue
            if m.get("private", True):
                continue
            downloads = m.get("downloads", 0) or 0
            if downloads < MIN_DOWNLOADS:
                continue
            # Only keep models that have at least one live provider mapping
            mapping = m.get("inferenceProviderMapping", [])
            if not mapping:
                continue
            if not any(isinstance(x, dict) and x.get("status") == "live" for x in mapping):
                continue
            all_models.append(m)

        print(f"Fetched batch offset={offset}, total candidates so far: {len(all_models)}")
        offset += batch
        if len(data) < batch:
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
            if not isinstance(info, dict):
                continue
            if info.get("status") != "live":
                continue
            provider = info.get("provider")
            provider_id = info.get("providerId", model_id)
            if not provider:
                continue
            pairs.append(
                {
                    "model": model_id,
                    "provider": provider,
                    "provider_id": provider_id,
                    "downloads": downloads,
                    "likes": likes,
                }
            )
    return pairs


def main():
    print("Discovering models from HF API...")
    models = fetch_models()
    pairs = extract_pairs(models)

    outfile = OUT_PATH
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_models": len(models),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(models)} models, {len(pairs)} pairs to {outfile}")


if __name__ == "__main__":
    main()
