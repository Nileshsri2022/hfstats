"""Tests for scripts/discover_models.py."""
import json

import pytest

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


def _gated_payload():
    return [
        _model("keep/model", downloads=90000),
        _model("gated/model", gated="manual"),
        _model("autogated/model", gated="auto"),
        _model("private/model", private=True),
        _model("nolive/model", status="staging"),
        _model("low/model", downloads=5),
    ]


def test_fetch_models_filters_private_and_low_downloads_keeps_gated_by_default(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 10000)
    monkeypatch.setattr(discover_models, "LIMIT", 500)
    monkeypatch.setattr(discover_models, "SKIP_GATED", False)

    resp = FakeResponse(body=json.dumps(_gated_payload()), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )

    models = discover_models.fetch_models()
    # Gated models (e.g. Llama, Gemma) are served by providers, so they are kept.
    assert [m["id"] for m in models] == ["keep/model", "gated/model", "autogated/model"]


def test_fetch_models_skip_gated_handles_real_hub_values(monkeypatch):
    # The Hub reports gated as "auto"/"manual", never True; the old `is True`
    # check could therefore never skip anything.
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 10000)
    monkeypatch.setattr(discover_models, "LIMIT", 500)
    monkeypatch.setattr(discover_models, "SKIP_GATED", True)

    resp = FakeResponse(body=json.dumps(_gated_payload()), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )

    models = discover_models.fetch_models()
    assert [m["id"] for m in models] == ["keep/model"]


def test_first_page_url_requests_every_field_the_filters_use():
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(discover_models.first_page_url()).query)
    assert query["pipeline_tag"] == ["text-generation"]
    assert query["sort"] == ["downloads"]
    assert set(query["expand"]) >= {
        "inferenceProviderMapping", "downloads", "likes", "gated", "private"
    }


def test_fetch_models_sends_token_when_available(monkeypatch):
    import common

    monkeypatch.setattr(common, "HF_TOKEN", "hf_secret")
    seen = {}

    def fake_urlopen(req, *a, **k):
        seen["auth"] = req.get_header("Authorization")
        return FakeResponse(body=json.dumps([]), headers={"Link": ""})

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    discover_models.fetch_models()
    assert seen["auth"] == "Bearer hf_secret"


def test_fetch_models_stops_once_downloads_fall_below_threshold(monkeypatch):
    # Sorted by downloads: a page ending below MIN_DOWNLOADS means no later page
    # can qualify, so the next page must not be requested.
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 10000)
    monkeypatch.setattr(discover_models, "LIMIT", 500)
    page1 = FakeResponse(
        body=json.dumps([_model("org/big", downloads=80000), _model("org/small", downloads=900)]),
        headers={"Link": '<https://next.test/page2>; rel="next"'},
    )
    calls = []

    def fake_urlopen(req, *a, **k):
        calls.append(req.full_url)
        if len(calls) > 1:
            raise AssertionError("fetched a page past the downloads threshold")
        return page1

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    models = discover_models.fetch_models()
    assert [m["id"] for m in models] == ["org/big"]
    assert len(calls) == 1


def test_fetch_models_respects_max_pages(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(discover_models, "LIMIT", 500)
    monkeypatch.setattr(discover_models, "MAX_PAGES", 2)
    counter = {"n": 0}

    def fake_urlopen(*a, **k):
        counter["n"] += 1
        return FakeResponse(
            body=json.dumps([_model(f"org/m{counter['n']}", downloads=100)]),
            headers={"Link": '<https://next.test/more>; rel="next"'},
        )

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    models = discover_models.fetch_models()
    assert counter["n"] == 2
    assert len(models) == 2


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


def test_fetch_models_raises_on_non_list_response(monkeypatch):
    # An unexpected payload must fail the run loudly instead of producing an
    # empty candidate list that gets recorded as a valid run.
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    resp = FakeResponse(body=json.dumps({"error": "boom"}), headers={"Link": ""})
    monkeypatch.setattr(
        discover_models.urllib.request, "urlopen", lambda *a, **k: resp
    )
    with pytest.raises(RuntimeError, match="expected a list"):
        discover_models.fetch_models()


def test_fetch_models_raises_after_exhausting_rate_limit_retries(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    sleeps = []
    monkeypatch.setattr(discover_models.time, "sleep", lambda s: sleeps.append(s))

    def fake_urlopen(*a, **k):
        raise make_http_error(429, "slow down")

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Failed to fetch models"):
        discover_models.fetch_models()
    assert len(sleeps) == discover_models.MAX_ATTEMPTS


def test_rate_limit_wait_prefers_hub_ratelimit_header():
    # Hub format: "api";r=<remaining>;t=<seconds until reset>
    assert discover_models.rate_limit_wait({"RateLimit": '"api";r=0;t=42'}) == 43
    assert discover_models.rate_limit_wait({"Retry-After": "7"}) == 7
    # HTTP-date Retry-After values used to crash int(); now they fall back.
    assert discover_models.rate_limit_wait({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}) == 30
    assert discover_models.rate_limit_wait({}) == 30
    assert discover_models.rate_limit_wait(None) == 30
    # Waits are capped to one rate-limit window.
    assert discover_models.rate_limit_wait({"RateLimit": '"api";r=0;t=9999'}) == discover_models.MAX_RATE_LIMIT_WAIT


def test_fetch_models_uses_ratelimit_reset_on_429(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    sleeps = []
    monkeypatch.setattr(discover_models.time, "sleep", lambda s: sleeps.append(s))
    ok = FakeResponse(body=json.dumps([_model("org/m", downloads=100)]), headers={"Link": ""})
    calls = {"n": 0}

    def fake_urlopen(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_http_error(429, "rate", {"RateLimit": '"api";r=0;t=12'})
        return ok

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    assert [m["id"] for m in discover_models.fetch_models()] == ["org/m"]
    assert sleeps == [13]


def test_fetch_models_retries_transient_gateway_errors(monkeypatch):
    monkeypatch.setattr(discover_models, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(discover_models.time, "sleep", lambda s: None)
    ok = FakeResponse(body=json.dumps([_model("org/m", downloads=100)]), headers={"Link": ""})
    errors = [make_http_error(503, "unavailable"), make_http_error(504, "timeout")]

    def fake_urlopen(*a, **k):
        if errors:
            raise errors.pop(0)
        return ok

    monkeypatch.setattr(discover_models.urllib.request, "urlopen", fake_urlopen)
    assert [m["id"] for m in discover_models.fetch_models()] == ["org/m"]


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


def test_main_exits_when_nothing_discovered(monkeypatch, tmp_path):
    monkeypatch.setattr(discover_models, "OUT_PATH", str(tmp_path / "candidates.json"))
    monkeypatch.setattr(discover_models, "fetch_models", lambda: [])
    with pytest.raises(SystemExit) as exc:
        discover_models.main()
    assert exc.value.code == 1


def test_extract_pairs_keeps_only_conversational_mappings():
    models = [
        {
            "id": "org/m",
            "downloads": 50000,
            "inferenceProviderMapping": [
                {"provider": "a", "status": "live", "task": "conversational"},
                {"provider": "b", "status": "live", "task": "text-generation"},
                {"provider": "c", "status": "live"},  # no task field: kept
                {"provider": "d", "status": "error", "task": "conversational"},
            ],
        }
    ]
    assert [p["provider"] for p in discover_models.extract_pairs(models)] == ["a", "c"]
