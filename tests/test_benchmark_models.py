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
