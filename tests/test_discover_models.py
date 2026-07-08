"""Tests for scripts/discover_models.py."""
import json

import discover_models
from conftest import FakeResponse, make_http_error


def _model(model_id, downloads=50000, likes=10, status="live", **extra):
    m = {
        "id": model_id,
        "downloads": downloads,
        "likes": likes,
        "inferenceProviderMapping": [
            {"provider": "together", "providerId": "tid", "status": status}
        ],
    }
    m.update(extra)
    return m


def test_extract_pairs_basic():
    models = [_model("org/model-a")]
    pairs = discover_models.extract_pairs(models)
    assert pairs == [
        {
            "model": "org/model-a",
            "provider": "together",
            "provider_id": "tid",
            "downloads": 50000,
            "likes": 10,
        }
    ]


def test_extract_pairs_defaults_provider_id_to_model():
    models = [
        {
            "id": "org/m",
            "downloads": 1,
            "likes": 0,
            "inferenceProviderMapping": [
                {"provider": "hf-inference", "status": "live"}
            ],
        }
    ]
    pairs = discover_models.extract_pairs(models)
    assert pairs[0]["provider_id"] == "org/m"


def test_extract_pairs_skips_non_live_and_missing_provider():
    models = [
        {
            "id": "org/m",
            "downloads": 1,
            "likes": 0,
            "inferenceProviderMapping": [
                {"provider": "a", "status": "staging"},
                {"provider": None, "status": "live"},
                "not-a-dict",
                {"provider": "good", "status": "live"},
            ],
        }
    ]
    pairs = discover_models.extract_pairs(models)
    assert [p["provider"] for p in pairs] == ["good"]


def test_extract_pairs_handles_none_downloads_and_likes():
    models = [
        {
            "id": "org/m",
            "downloads": None,
            "likes": None,
            "inferenceProviderMapping": [{"provider": "p", "status": "live"}],
        }
    ]
    pairs = discover_models.extract_pairs(models)
    assert pairs[0]["downloads"] == 0
    assert pairs[0]["likes"] == 0


def test_fetch_models_filters_gated_private_and_low_downloads(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 10000)
    monkeypatch.setattr(discover_models, "LIMIT", 500)

    payload = [
        _model("keep/model", downloads=20000),
        _model("gated/model", gated=True),
        _model("private/model", private=True),
        _model("low/model", downloads=5),
        _model("nolive/model", status="staging"),
    ]
    resp = FakeResponse(body=json.dumps(payload), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )

    models = discover_models.fetch_models()
    assert [m["id"] for m in models] == ["keep/model"]


def test_fetch_models_respects_limit(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(discover_models, "LIMIT", 2)
    payload = [_model(f"org/m{i}", downloads=100) for i in range(10)]
    resp = FakeResponse(body=json.dumps(payload), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )
    models = discover_models.fetch_models()
    assert len(models) == 2


def test_fetch_models_stops_on_non_list_response(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    resp = FakeResponse(body=json.dumps({"error": "boom"}), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )
    assert discover_models.fetch_models() == []


def test_fetch_models_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(discover_models, "LIMIT", 500)
    sleeps = []
    monkeypatch.setattr(discover_models.time, "sleep", lambda s: sleeps.append(s))

    ok = FakeResponse(
        body=json.dumps([_model("org/m", downloads=100)]), headers={"Link": ""}
    )
    calls = {"n": 0}

    def fake_urlopen(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_http_error(429, "rate", {"Retry-After": "7"})
        return ok

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    models = discover_models.fetch_models()
    assert [m["id"] for m in models] == ["org/m"]
    assert sleeps == [7]


def test_fetch_models_reraises_non_429_http_error(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)

    def fake_urlopen(*a, **k):
        raise make_http_error(500, "server error")

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    try:
        discover_models.fetch_models()
    except Exception as e:  # noqa: BLE001
        assert getattr(e, "code", None) == 500
    else:
        raise AssertionError("expected HTTPError to propagate")


def test_fetch_models_follows_link_header(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(discover_models, "LIMIT", 500)

    page1 = FakeResponse(
        body=json.dumps([_model("org/a", downloads=100)]),
        headers={"Link": '<https://next.test/page2>; rel="next"'},
    )
    page2 = FakeResponse(
        body=json.dumps([_model("org/b", downloads=100)]),
        headers={"Link": ""},
    )
    responses = [page1, page2]
    monkeypatch.setattr(
        discover_models.urllib.request,
        "urlopen",
        lambda *a, **k: responses.pop(0),
    )
    models = discover_models.fetch_models()
    assert [m["id"] for m in models] == ["org/a", "org/b"]


def test_main_writes_candidates_json(monkeypatch, tmp_path):
    out = tmp_path / "candidates.json"
    monkeypatch.setattr(discover_models, "OUT_PATH", str(out))
    monkeypatch.setattr(
        discover_models, "fetch_models", lambda: [_model("org/m", downloads=100)]
    )
    discover_models.main()
    data = json.loads(out.read_text())
    assert data["total_models"] == 1
    assert data["total_pairs"] == 1
    assert data["pairs"][0]["model"] == "org/m"
    assert "timestamp" in data
