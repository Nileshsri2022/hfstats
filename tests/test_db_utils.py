"""Tests for scripts/db_utils.py."""
import sqlite3


def _sample_result(**overrides):
    base = {
        "model": "meta-llama/Llama-3",
        "provider": "together",
        "success": True,
        "error": None,
        "error_category": "working",
        "response_time": 1234,
        "ttft": 100,
        "tokens_generated": 50,
        "total_tokens": 55,
        "response": "hello world",
    }
    base.update(overrides)
    return base


def test_get_conn_enables_foreign_keys_and_row_factory(temp_db):
    conn = temp_db.get_conn()
    try:
        assert conn.row_factory is sqlite3.Row
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
    finally:
        conn.close()


def test_init_db_creates_all_tables(temp_db):
    conn = temp_db.get_conn()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r["name"] for r in rows}
    finally:
        conn.close()
    assert {"runs", "model_results", "discovery_snapshots"} <= names


def test_init_db_is_idempotent(temp_db):
    # Calling a second time should not raise (uses IF NOT EXISTS).
    temp_db.init_db()


def test_insert_run_returns_incrementing_ids(temp_db):
    first = temp_db.insert_run("p", 1, 2, "a:b", 10, 5, 1, 0, 0, 0)
    second = temp_db.insert_run("p", 1, 2, "a:b", 10, 5, 1, 0, 0, 0)
    assert first == 1
    assert second == 2


def test_insert_run_persists_fields_and_timestamp(temp_db):
    run_id = temp_db.insert_run(
        prompt="explain neural nets",
        success_count=3,
        total_pairs=4,
        fastest_pair="m:together",
        fastest_time=999,
        candidates_found=10,
        pairs_working=3,
        pairs_loading=1,
        pairs_rate_limited=2,
        pairs_unsupported=4,
    )
    conn = temp_db.get_conn()
    try:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    finally:
        conn.close()
    assert row["prompt"] == "explain neural nets"
    assert row["success_count"] == 3
    assert row["fastest_pair"] == "m:together"
    assert row["pairs_unsupported"] == 4
    # timestamp is an ISO-8601 string with timezone info.
    assert row["timestamp"].endswith("+00:00")


def test_insert_model_result_maps_success_to_int(temp_db):
    run_id = temp_db.insert_run("p", 0, 0, "", 0, 0, 0, 0, 0, 0)
    temp_db.insert_model_result(run_id, _sample_result(success=True))
    temp_db.insert_model_result(run_id, _sample_result(success=False, model="m2"))
    conn = temp_db.get_conn()
    try:
        rows = conn.execute(
            "SELECT model, success FROM model_results ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert [(r["model"], r["success"]) for r in rows] == [
        ("meta-llama/Llama-3", 1),
        ("m2", 0),
    ]


def test_insert_model_result_uses_get_for_missing_keys(temp_db):
    run_id = temp_db.insert_run("p", 0, 0, "", 0, 0, 0, 0, 0, 0)
    # Only the required-ish fields present; the rest default to None via .get().
    temp_db.insert_model_result(run_id, {"model": "m", "provider": "p"})
    conn = temp_db.get_conn()
    try:
        row = conn.execute("SELECT * FROM model_results").fetchone()
    finally:
        conn.close()
    assert row["model"] == "m"
    assert row["success"] == 0
    assert row["response_time"] is None


def test_insert_discovery_snapshot_defaults_status(temp_db):
    run_id = temp_db.insert_run("p", 0, 0, "", 0, 0, 0, 0, 0, 0)
    temp_db.insert_discovery_snapshot(
        run_id, {"model": "m", "provider": "prov"}
    )
    conn = temp_db.get_conn()
    try:
        row = conn.execute("SELECT * FROM discovery_snapshots").fetchone()
    finally:
        conn.close()
    assert row["status"] == "unknown"
    assert row["downloads"] is None


def test_prune_old_runs_keeps_only_recent(temp_db):
    for _ in range(5):
        rid = temp_db.insert_run("p", 0, 0, "", 0, 0, 0, 0, 0, 0)
        temp_db.insert_model_result(rid, _sample_result())
        temp_db.insert_discovery_snapshot(rid, {"model": "m", "provider": "p"})

    temp_db.prune_old_runs(max_runs=2)

    conn = temp_db.get_conn()
    try:
        run_ids = [r["id"] for r in conn.execute("SELECT id FROM runs").fetchall()]
        mr = conn.execute("SELECT DISTINCT run_id FROM model_results").fetchall()
    finally:
        conn.close()
    # Only the 2 most recent runs (ids 4 and 5) survive.
    assert run_ids == [4, 5]
    assert {r["run_id"] for r in mr} == {4, 5}


def test_prune_old_runs_noop_when_under_limit(temp_db):
    for _ in range(3):
        temp_db.insert_run("p", 0, 0, "", 0, 0, 0, 0, 0, 0)
    temp_db.prune_old_runs(max_runs=10)
    conn = temp_db.get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
    finally:
        conn.close()
    assert count == 3


def test_dump_db_to_json_returns_all_tables(temp_db):
    rid = temp_db.insert_run("p", 1, 1, "m:p", 5, 1, 1, 0, 0, 0)
    temp_db.insert_model_result(rid, _sample_result())
    temp_db.insert_discovery_snapshot(rid, {"model": "m", "provider": "p"})

    data = temp_db.dump_db_to_json()

    assert set(data.keys()) == {"runs", "model_results", "discovery_snapshots"}
    assert len(data["runs"]) == 1
    assert len(data["model_results"]) == 1
    assert data["model_results"][0]["model"] == "meta-llama/Llama-3"
