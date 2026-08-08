"""Shared utilities for HFStats benchmark scripts.

Consolidates code that was previously duplicated across the pipeline scripts:
HuggingFace inference API access, HTTP-status-to-error-category mapping,
JSON file I/O, and UTC timestamp formatting.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

INFERENCE_URL = "https://api-inference.huggingface.co/models/"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# HTTP status code -> HFStats error category. Any status not listed here is
# treated as a generic provider bug.
_HTTP_STATUS_CATEGORIES = {
    400: "unsupported",
    402: "quota_exceeded",
    404: "not_found",
    429: "rate_limited",
    503: "loading",
    529: "overloaded",
}


def categorize_http_status(status: int) -> str:
    """Map an HTTP status code to an HFStats error category."""
    return _HTTP_STATUS_CATEGORIES.get(status, "provider_bug")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def build_chat_request(
    model: str,
    provider: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool = False,
    provider_in_url: bool = False,
    temperature: float = 0.7,
) -> urllib.request.Request:
    """Build a POST Request for the HF chat-completions inference API.

    ``provider_in_url`` appends ``:provider`` to the URL path (used by the
    classifier); ``stream`` toggles server-sent-event streaming and the
    corresponding ``Accept`` header (used by the benchmarker).
    """
    path = f"{model}:{provider}" if provider_in_url else model
    body = {
        "model": f"{model}:{provider}",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        body["stream"] = True

    return urllib.request.Request(
        f"{INFERENCE_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
        },
        method="POST",
    )


def load_json(path: str):
    """Load JSON from ``path``, returning ``None`` if the file is missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload) -> None:
    """Write ``payload`` as indented JSON, creating parent dirs as needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
