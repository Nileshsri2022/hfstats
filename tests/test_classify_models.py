"""Tests for scripts/classify_models.py."""
import json

import classify_models
from conftest import FakeResponse, make_http_error


def _patch_urlopen(monkeypatch, resp=None, exc=None):
    def fake_urlopen(*a, **k):
        if exc is not None:
            raise exc
        return resp

    monkeypatch.setattr(
        classify_models.urllib.request, "urlopen", fake_urlopen
    )


def test_classify_working_choices(monkeypatch):
    _patch_urlopen(monkeypatch, FakeResponse(json.dumps({"choices": [{}]})))
    assert classify_models.classify_pair("m", "p")["status"] == "working"


def test_classify_working_list_response(monkeypatch):
    _patch_urlopen(monkeypatch, FakeResponse(json.dumps([{"x": 1}])))
    info = classify_models.classify_pair("m", "p")
    assert info["status"] == "working"
    assert info["detail"] == "list_response"


def test_classify_malformed_json(monkeypatch):
    _patch_urlopen(monkeypatch, FakeResponse("not json{"))
    info = classify_models.classify_pair("m", "p")
    assert info["status"] == "provider_bug"
    assert info["detail"] == "malformed json"


def test_classify_unexpected_structure(monkeypatch):
    _patch_urlopen(monkeypatch, FakeResponse(json.dumps({"foo": "bar"})))
    info = classify_models.classify_pair("m", "p")
    assert info["status"] == "provider_bug"
    assert info["detail"] == "unexpected structure"


def test_classify_empty_list_is_provider_bug(monkeypatch):
    _patch_urlopen(monkeypatch, FakeResponse(json.dumps([])))
    assert classify_models.classify_pair("m", "p")["status"] == "provider_bug"


def test_classify_error_object_rate_limit(monkeypatch):
    _patch_urlopen(
        monkeypatch, FakeResponse(json.dumps({"error": "Rate limit exceeded"}))
    )
    assert classify_models.classify_pair("m", "p")["status"] == "rate_limited"


def test_classify_error_object_quota(monkeypatch):
    _patch_urlopen(
        monkeypatch, FakeResponse(json.dumps({"error": "quota reached"}))
    )
    assert classify_models.classify_pair("m", "p")["status"] == "quota_exceeded"


def test_classify_error_object_not_found(monkeypatch):
    _patch_urlopen(
        monkeypatch, FakeResponse(json.dumps({"error": "model not found"}))
    )
    assert classify_models.classify_pair("m", "p")["status"] == "not_found"


def test_classify_error_object_generic(monkeypatch):
    _patch_urlopen(
        monkeypatch, FakeResponse(json.dumps({"error": "something odd"}))
    )
    assert classify_models.classify_pair("m", "p")["status"] == "provider_bug"


def test_classify_http_status_mapping(monkeypatch):
    cases = {
        503: "loading",
        529: "overloaded",
        429: "rate_limited",
        402: "quota_exceeded",
        404: "not_found",
        400: "unsupported",
        500: "provider_bug",
    }
    for code, expected in cases.items():
        _patch_urlopen(monkeypatch, exc=make_http_error(code, "body"))
        assert classify_models.classify_pair("m", "p")["status"] == expected


def test_classify_urlerror_timeout(monkeypatch):
    import urllib.error

    _patch_urlopen(monkeypatch, exc=urllib.error.URLError("connection timed out"))
    assert classify_models.classify_pair("m", "p")["status"] == "cold_start"


def test_classify_urlerror_generic(monkeypatch):
    import urllib.error

    _patch_urlopen(monkeypatch, exc=urllib.error.URLError("refused"))
    assert classify_models.classify_pair("m", "p")["status"] == "provider_bug"


def test_classify_unexpected_exception(monkeypatch):
    _patch_urlopen(monkeypatch, exc=RuntimeError("boom"))
    info = classify_models.classify_pair("m", "p")
    assert info["status"] == "provider_bug"
    assert info["detail"] == "boom"


def test_main_classifies_and_writes(monkeypatch, tmp_path):
    candidates = tmp_path / "candidates.json"
    out = tmp_path / "classified.json"
    candidates.write_text(
        json.dumps(
            {
                "pairs": [
                    {"model": "m1", "provider": "p1", "downloads": 5, "likes": 1},
                    {"model": "m2", "provider": "p2", "downloads": 3, "likes": 0},
                ]
            }
        )
    )
    monkeypatch.setattr(classify_models, "CANDIDATES_PATH", str(candidates))
    monkeypatch.setattr(classify_models, "OUT_PATH", str(out))
    monkeypatch.setattr(
        classify_models,
        "classify_pair",
        lambda model, provider: {"status": "working", "detail": "ok"},
    )

    classify_models.main()

    data = json.loads(out.read_text())
    assert data["total"] == 2
    assert data["counts"]["working"] == 2
    assert {p["model"] for p in data["pairs"]} == {"m1", "m2"}


def test_main_retries_loading_then_rate_limited(monkeypatch, tmp_path):
    candidates = tmp_path / "candidates.json"
    out = tmp_path / "classified.json"
    candidates.write_text(
        json.dumps({"pairs": [{"model": "m", "provider": "p"}]})
    )
    monkeypatch.setattr(classify_models, "CANDIDATES_PATH", str(candidates))
    monkeypatch.setattr(classify_models, "OUT_PATH", str(out))
    monkeypatch.setattr(classify_models.time, "sleep", lambda s: None)

    statuses = iter(["loading", "rate_limited", "working"])
    monkeypatch.setattr(
        classify_models,
        "classify_pair",
        lambda model, provider: {"status": next(statuses), "detail": "d"},
    )

    classify_models.main()
    data = json.loads(out.read_text())
    # After a loading retry and a rate_limited backoff, final status is working.
    assert data["pairs"][0]["status"] == "working"
    assert data["counts"]["working"] == 1


def test_classify_calls_router_with_model_provider_in_body(monkeypatch):
    seen = {}

    def fake_urlopen(req, *a, **k):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["method"] = req.get_method()
        return FakeResponse(json.dumps({"choices": [{}]}))

    monkeypatch.setattr(classify_models.urllib.request, "urlopen", fake_urlopen)
    assert classify_models.classify_pair("org/model", "novita")["status"] == "working"
    # Regression: the retired api-inference host answered everything with 410.
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert "api-inference" not in seen["url"]
    assert seen["method"] == "POST"
    assert seen["body"]["model"] == "org/model:novita"
    assert seen["body"]["max_tokens"] == classify_models.TINY_MAX_TOKENS
    assert "stream" not in seen["body"]


def test_classify_auth_and_gone_statuses(monkeypatch):
    cases = {401: "auth_error", 403: "auth_error", 410: "not_found", 422: "unsupported"}
    for code, expected in cases.items():
        _patch_urlopen(monkeypatch, exc=make_http_error(code, "body"))
        assert classify_models.classify_pair("m", "p")["status"] == expected


def test_classify_error_object_auth(monkeypatch):
    _patch_urlopen(
        monkeypatch, FakeResponse(json.dumps({"error": "Invalid credentials in Authorization header"}))
    )
    assert classify_models.classify_pair("m", "p")["status"] == "auth_error"


def test_classify_read_timeout_is_cold_start(monkeypatch):
    # Timeouts while waiting for the response are raised as TimeoutError (not
    # URLError) and used to be misfiled as provider_bug.
    _patch_urlopen(monkeypatch, exc=TimeoutError("timed out"))
    assert classify_models.classify_pair("m", "p")["status"] == "cold_start"


def test_main_fails_loudly_when_no_pair_works(monkeypatch, tmp_path, capsys):
    import pytest

    candidates = tmp_path / "candidates.json"
    out = tmp_path / "classified.json"
    candidates.write_text(
        json.dumps({"pairs": [{"model": f"m{i}", "provider": "p"} for i in range(3)]})
    )
    monkeypatch.setattr(classify_models, "CANDIDATES_PATH", str(candidates))
    monkeypatch.setattr(classify_models, "OUT_PATH", str(out))
    monkeypatch.setattr(
        classify_models,
        "classify_pair",
        lambda model, provider: {"status": "not_found", "detail": "gone for good"},
    )

    with pytest.raises(SystemExit) as exc:
        classify_models.main()
    assert exc.value.code == 1
    # The classification is still written (useful for debugging) ...
    assert json.loads(out.read_text())["counts"]["not_found"] == 3
    # ... and the most common failure is printed.
    assert "3 x not_found: gone for good" in capsys.readouterr().err


def test_main_exits_when_no_candidates(monkeypatch, tmp_path):
    import pytest

    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"pairs": []}))
    monkeypatch.setattr(classify_models, "CANDIDATES_PATH", str(candidates))
    monkeypatch.setattr(classify_models, "OUT_PATH", str(tmp_path / "classified.json"))
    with pytest.raises(SystemExit) as exc:
        classify_models.main()
    assert exc.value.code == 1


def test_classify_all_preserves_order_and_serializes_each_provider(monkeypatch):
    import threading
    import time as _time

    monkeypatch.setattr(classify_models, "CONCURRENCY", 4)
    pairs = [{"model": f"m{i}", "provider": f"p{i % 3}"} for i in range(12)]
    active = {}
    lock = threading.Lock()
    overlaps = []

    def fake_classify(model, provider):
        with lock:
            active[provider] = active.get(provider, 0) + 1
            if active[provider] > 1:
                overlaps.append(provider)
        _time.sleep(0.01)
        with lock:
            active[provider] -= 1
        return {"status": "working", "detail": model}

    monkeypatch.setattr(classify_models, "classify_pair", fake_classify)
    infos = classify_models.classify_all(pairs)
    assert [i["detail"] for i in infos] == [p["model"] for p in pairs]
    assert overlaps == []  # never two in-flight requests to the same provider
