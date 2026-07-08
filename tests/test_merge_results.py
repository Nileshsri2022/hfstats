"""Tests for scripts/merge_results.py."""
import json

import pytest

import merge_results


def test_load_json_missing_returns_none(tmp_path):
    assert merge_results.load_json(str(tmp_path / "nope.json")) is None


def test_load_json_reads_file(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"a": 1}))
    assert merge_results.load_json(str(p)) == {"a": 1}


def _write_inputs(tmp_path, monkeypatch, results_groups, classified):
    classified_path = tmp_path / "classified.json"
    classified_path.write_text(json.dumps(classified))
    monkeypatch.setattr(merge_results, "CLASSIFIED_PATH", str(classified_path))

    paths = []
    for i, grp in enumerate(results_groups, start=1):
        gp = tmp_path / f"results-group{i}.json"
        gp.write_text(json.dumps(grp))
        paths.append(str(gp))
    monkeypatch.setattr(merge_results, "GROUP_PATHS", paths)


def test_main_merges_results_into_db(tmp_path, monkeypatch):
    import db_utils

    db_file = tmp_path / "history.db"
    monkeypatch.setattr(db_utils, "DB_PATH", str(db_file))

    results_groups = [
        {
            "prompt": "explain",
            "results": [
                {"model": "m1", "provider": "p", "success": True, "response_time": 300},
                {"model": "m2", "provider": "p", "success": False, "response_time": 100},
            ],
        },
        {
            "results": [
                {"model": "m3", "provider": "q", "success": True, "response_time": 150},
            ]
        },
    ]
    classified = {
        "counts": {"working": 2, "loading": 1, "rate_limited": 1, "not_found": 1},
        "pairs": [
            {"model": "m1", "provider": "p", "status": "working", "downloads": 10, "likes": 1},
            {"model": "m3", "provider": "q", "status": "working", "downloads": 20, "likes": 2},
        ],
    }
    _write_inputs(tmp_path, monkeypatch, results_groups, classified)

    merge_results.main()

    conn = db_utils.get_conn()
    try:
        run = conn.execute("SELECT * FROM runs").fetchone()
        mr = conn.execute("SELECT COUNT(*) AS c FROM model_results").fetchone()["c"]
        ds = conn.execute("SELECT COUNT(*) AS c FROM discovery_snapshots").fetchone()["c"]
    finally:
        conn.close()

    assert run["success_count"] == 2
    assert run["total_pairs"] == 3
    assert run["prompt"] == "explain"
    # Fastest successful pair is m3 at 150ms.
    assert run["fastest_pair"] == "m3:q"
    assert run["fastest_time"] == 150
    assert run["candidates_found"] == 2
    assert run["pairs_working"] == 2
    # unsupported = unsupported + not_found + quota_exceeded = 0 + 1 + 0.
    assert run["pairs_unsupported"] == 1
    assert mr == 3
    assert ds == 2


def test_main_exits_when_classified_missing(tmp_path, monkeypatch):
    import db_utils

    monkeypatch.setattr(db_utils, "DB_PATH", str(tmp_path / "h.db"))
    monkeypatch.setattr(
        merge_results, "CLASSIFIED_PATH", str(tmp_path / "missing.json")
    )
    monkeypatch.setattr(merge_results, "GROUP_PATHS", [])

    with pytest.raises(SystemExit) as exc:
        merge_results.main()
    assert exc.value.code == 1


def test_main_handles_no_successful_results(tmp_path, monkeypatch):
    import db_utils

    monkeypatch.setattr(db_utils, "DB_PATH", str(tmp_path / "h.db"))
    results_groups = [
        {"results": [{"model": "m", "provider": "p", "success": False}]}
    ]
    classified = {"counts": {}, "pairs": []}
    _write_inputs(tmp_path, monkeypatch, results_groups, classified)

    merge_results.main()

    conn = db_utils.get_conn()
    try:
        run = conn.execute("SELECT * FROM runs").fetchone()
    finally:
        conn.close()
    assert run["success_count"] == 0
    assert run["fastest_pair"] == ""
    # fastest_time None coalesced to 0 on insert.
    assert run["fastest_time"] == 0
