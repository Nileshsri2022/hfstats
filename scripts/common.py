"""Shared utilities for HFStats benchmark scripts.

Consolidates code used across the pipeline scripts: Hugging Face Inference
Providers access, HTTP-status / error-text to error-category mapping, JSON file
I/O, and UTC timestamp formatting.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

# OpenAI-compatible chat-completions endpoint of the Hugging Face Inference
# Providers router. The provider is selected with a ``model:provider`` suffix in
# the request body. The legacy https://api-inference.huggingface.co host has
# been retired and answers every request with HTTP 410 Gone -- calling it is why
# no pair was ever classified as "working" before this endpoint was adopted.
CHAT_COMPLETIONS_URL = os.environ.get(
    "HF_CHAT_COMPLETIONS_URL", "https://router.huggingface.co/v1/chat/completions"
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
USER_AGENT = "hfstats/3 (+https://github.com/Nileshsri2022/hfstats)"

# HTTP status code -> HFStats error category. Any status not listed here is
# treated as a generic provider bug.
_HTTP_STATUS_CATEGORIES = {
    400: "unsupported",
    401: "auth_error",
    402: "quota_exceeded",
    403: "auth_error",
    404: "not_found",
    410: "not_found",
    422: "unsupported",
    429: "rate_limited",
    503: "loading",
    529: "overloaded",
}


def categorize_http_status(status: int) -> str:
    """Map an HTTP status code to an HFStats error category."""
    return _HTTP_STATUS_CATEGORIES.get(status, "provider_bug")


def categorize_error_text(text) -> str:
    """Map a provider error message (e.g. an ``{"error": ...}`` body) to a category."""
    err = str(text).lower()
    if any(k in err for k in ("unauthorized", "invalid token", "invalid credentials",
                              "permission", "forbidden")):
        return "auth_error"
    if "rate limit" in err or "too many requests" in err:
        return "rate_limited"
    if "quota" in err or "insufficient" in err or "credits" in err:
        return "quota_exceeded"
    if "not found" in err or "unknown" in err:
        return "not_found"
    return "provider_bug"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def hub_headers() -> dict:
    """Headers for Hugging Face Hub API calls (authenticated when a token is set).

    Authenticated requests get a much larger Hub rate-limit budget than
    anonymous, per-IP requests from shared CI runners.
    """
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def build_chat_request(
    model: str,
    provider: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool = False,
    temperature: float = 0.7,
) -> urllib.request.Request:
    """Build a POST Request for the HF router's chat-completions endpoint.

    ``stream`` toggles server-sent-event streaming (used by the benchmarker) and
    asks for a final usage chunk so real token counts can be recorded.
    """
    body = {
        "model": f"{model}:{provider}",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    return urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
            "User-Agent": USER_AGENT,
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
