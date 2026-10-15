"""Tests for scripts/benchmark_models.py."""
import json

import benchmark_models
from conftest import FakeResponse, make_http_error


def _sse(obj):
    return f"data: {json.dumps(obj)}\n"


def _patch_urlopen(monkeypatch, resp=None, exc=None):
    def fake_urlopen(*a, **k):
        if exc is not None:
            raise exc
        return resp

    monkeypatch.setattr(
        benchmark_models.urllib.request, "urlopen", fake_urlopen
    )


def test_benchmark_success_aggregates_tokens_and_usage(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " world"}}]}),
        _sse({"choices": [{"delta": {}}], "usage": {"total_tokens": 42}}),
        "data: [DONE]\n",
    ]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is True
    assert r["error_category"] == "working"
    assert r["tokens_generated"] == 2
    assert r["total_tokens"] == 42
    assert r["response"] == "Hello world"
    assert r["ttft"] is not None
    assert "throughput" in r


def test_benchmark_ignores_non_data_lines_and_bad_json(monkeypatch):
    lines = [
        ": comment line\n",
        "data: not-json\n",
        _sse({"choices": [{"delta": {"content": "X"}}]}),
        "data: [DONE]\n",
    ]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is True
    assert r["tokens_generated"] == 1
    assert r["response"] == "X"


def test_benchmark_no_tokens_generated(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {}}]}),
        "data: [DONE]\n",
    ]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error"] == "No tokens generated"
    assert r["error_category"] == "provider_bug"
    assert r["tokens_generated"] == 0


def test_benchmark_total_tokens_defaults_to_generated(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {"content": "a"}}]}),
        _sse({"choices": [{"delta": {"content": "b"}}]}),
        "data: [DONE]\n",
    ]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    # No usage chunk -> total_tokens falls back to tokens_generated.
    assert r["total_tokens"] == 2


def test_benchmark_http_error_mapping(monkeypatch):
    cases = {
        429: "rate_limited",
        503: "loading",
        529: "overloaded",
        402: "quota_exceeded",
        404: "not_found",
        400: "unsupported",
        500: "provider_bug",
    }
    for code, expected in cases.items():
        _patch_urlopen(monkeypatch, exc=make_http_error(code, "boom"))
        r = benchmark_models.benchmark_pair("m", "p")
        assert r["success"] is False
        assert r["error_category"] == expected
        assert r["error"].startswith(f"HTTP {code}")


def test_benchmark_urlerror_timeout(monkeypatch):
    import urllib.error

    _patch_urlopen(monkeypatch, exc=urllib.error.URLError("request timeout"))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error_category"] == "cold_start"


def test_benchmark_urlerror_generic(monkeypatch):
    import urllib.error

    _patch_urlopen(monkeypatch, exc=urllib.error.URLError("connection refused"))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error_category"] == "provider_bug"


def test_benchmark_unexpected_exception(monkeypatch):
    _patch_urlopen(monkeypatch, exc=RuntimeError("kaboom"))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error_category"] == "provider_bug"
    assert r["error"] == "kaboom"


def test_main_splits_groups_and_writes(monkeypatch, tmp_path):
    classified = tmp_path / "classified.json"
    out = tmp_path / "results.json"
    pairs = [
        {"model": f"m{i}", "provider": "p", "status": "working", "downloads": i}
        for i in range(6)
    ]
    # Add a non-working pair that must be filtered out.
    pairs.append({"model": "skip", "provider": "p", "status": "loading", "downloads": 999})
    classified.write_text(json.dumps({"timestamp": "t", "pairs": pairs}))

    monkeypatch.setattr(benchmark_models, "CLASSIFIED_PATH", str(classified))
    monkeypatch.setattr(benchmark_models, "OUT_PATH", str(out))
    monkeypatch.setattr(benchmark_models, "MODEL_GROUP", "group1")
    monkeypatch.setattr(benchmark_models, "MAX_BENCHMARK_PAIRS", 50)
    monkeypatch.setattr(
        benchmark_models,
        "benchmark_pair",
        lambda model, provider: {
            "model": model,
            "provider": provider,
            "success": True,
            "error_category": "working",
            "response_time": 1,
            "ttft": 1,
            "tokens_generated": 1,
        },
    )

    benchmark_models.main()
    data = json.loads(out.read_text())
    assert data["group"] == "group1"
    # 6 working pairs, round-robin into 3 groups -> group1 gets 2.
    assert len(data["results"]) == 2
    assert "skip" not in {r["model"] for r in data["results"]}


class _ClockedStream:
    """Streams SSE lines while advancing a fake clock to each line's timestamp."""

    def __init__(self, clock, timeline):
        self.clock = clock
        self.timeline = timeline
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for t, line in self.timeline:
            self.clock["now"] = t
            yield line.encode("utf-8")


def _run_clocked(monkeypatch, timeline):
    clock = {"now": 0.0}
    monkeypatch.setattr(benchmark_models.time, "perf_counter", lambda: clock["now"])
    _patch_urlopen(monkeypatch, _ClockedStream(clock, timeline))
    return benchmark_models.benchmark_pair("m", "p")


def test_benchmark_request_targets_router_with_streaming_usage(monkeypatch):
    seen = {}

    def fake_urlopen(req, *a, **k):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse(lines=[_sse({"choices": [{"delta": {"content": "x"}}]}), "data: [DONE]\n"])

    monkeypatch.setattr(benchmark_models.urllib.request, "urlopen", fake_urlopen)
    benchmark_models.benchmark_pair("org/model", "groq")
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["body"]["model"] == "org/model:groq"
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert seen["body"]["max_tokens"] == benchmark_models.MAX_TOKENS


def test_ttft_is_measured_at_first_token_not_at_headers(monkeypatch):
    r = _run_clocked(
        monkeypatch,
        [
            (0.4, _sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})),
            (1.0, _sse({"choices": [{"delta": {"content": "Hello"}}]})),
            (1.5, _sse({"choices": [{"delta": {"content": " big"}}]})),
            (2.0, _sse({"choices": [{"delta": {"content": " world"}}]})),
            (2.0, _sse({"choices": [], "usage": {"completion_tokens": 7, "total_tokens": 20}})),
            (2.0, "data: [DONE]\n"),
        ],
    )
    assert r["success"] is True
    assert r["ttft"] == 1000            # first content token, not the role-only chunk
    assert r["response_time"] == 2000
    assert r["tokens_generated"] == 7    # provider usage preferred over chunk count
    assert r["total_tokens"] == 20
    assert r["throughput"] == 6.0        # (7 - 1) tokens / 1.0 s after first token
    assert r["response"] == "Hello big world"


def test_reasoning_tokens_count_as_generated(monkeypatch):
    r = _run_clocked(
        monkeypatch,
        [
            (0.5, _sse({"choices": [{"delta": {"reasoning_content": "Let me think"}}]})),
            (1.0, _sse({"choices": [{"delta": {"reasoning": " about it"}}]})),
            (1.0, "data: [DONE]\n"),
        ],
    )
    # Reasoning models that spend the token budget thinking used to be
    # reported as "No tokens generated" failures.
    assert r["success"] is True
    assert r["ttft"] == 500
    assert r["tokens_generated"] == 2
    assert r["response"].startswith("[reasoning] Let me think")


def test_data_lines_without_space_are_parsed(monkeypatch):
    lines = ['data:{"choices":[{"delta":{"content":"X"}}]}\n', "data:[DONE]\n"]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is True
    assert r["response"] == "X"


def test_error_inside_stream_is_a_failure(monkeypatch):
    lines = [_sse({"error": {"message": "Rate limit reached for model"}}), "data: [DONE]\n"]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error_category"] == "rate_limited"
    assert r["throughput"] is None


def test_benchmark_read_timeout_is_cold_start(monkeypatch):
    _patch_urlopen(monkeypatch, exc=TimeoutError("timed out"))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["success"] is False
    assert r["error_category"] == "cold_start"


def test_benchmark_auth_error_mapping(monkeypatch):
    _patch_urlopen(monkeypatch, exc=make_http_error(401, "Invalid credentials"))
    r = benchmark_models.benchmark_pair("m", "p")
    assert r["error_category"] == "auth_error"


def test_stored_response_is_truncated_to_preview(monkeypatch):
    long_text = "x" * 5000
    lines = [_sse({"choices": [{"delta": {"content": long_text}}]}), "data: [DONE]\n"]
    _patch_urlopen(monkeypatch, FakeResponse(lines=lines))
    r = benchmark_models.benchmark_pair("m", "p")
    assert len(r["response"]) == benchmark_models.RESPONSE_PREVIEW_CHARS


def test_main_handles_missing_downloads(monkeypatch, tmp_path):
    classified = tmp_path / "classified.json"
    out = tmp_path / "results.json"
    pairs = [
        {"model": "a", "provider": "p", "status": "working", "downloads": None},
        {"model": "b", "provider": "p", "status": "working", "downloads": 10},
    ]
    classified.write_text(json.dumps({"timestamp": "t", "pairs": pairs}))
    monkeypatch.setattr(benchmark_models, "CLASSIFIED_PATH", str(classified))
    monkeypatch.setattr(benchmark_models, "OUT_PATH", str(out))
    monkeypatch.setattr(benchmark_models, "MODEL_GROUP", "group1")
    monkeypatch.setattr(
        benchmark_models,
        "benchmark_pair",
        lambda model, provider: {"model": model, "provider": provider, "success": True},
    )
    benchmark_models.main()  # used to raise TypeError comparing None and int
    assert json.loads(out.read_text())["results"][0]["model"] == "b"
