"""SQLite helpers for HFStats benchmark database."""
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import utc_now_iso

DB_PATH = os.environ.get("DB_PATH", "history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt TEXT,
    success_count INTEGER,
    total_pairs INTEGER,
    fastest_pair TEXT,
    fastest_time INTEGER,
    candidates_found INTEGER,
    pairs_working INTEGER,
    pairs_loading INTEGER,
    pairs_rate_limited INTEGER,
    pairs_unsupported INTEGER
);

CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    success INTEGER NOT NULL,
    error TEXT,
    error_category TEXT,
    response_time INTEGER,
    ttft INTEGER,
    tokens_generated INTEGER,
    total_tokens INTEGER,
    response TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS discovery_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    downloads INTEGER,
    likes INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def prune_old_runs(max_runs: int = 360):
    """Keep only the most recent max_runs runs to cap DB size."""
    conn = get_conn()
    cur = conn.execute(
        "SELECT id FROM runs ORDER BY id DESC LIMIT 1 OFFSET ?", (max_runs,)
    )
    row = cur.fetchone()
    if row:
        cutoff = row["id"]
        conn.execute("DELETE FROM model_results WHERE run_id <= ?", (cutoff,))
        conn.execute("DELETE FROM discovery_snapshots WHERE run_id <= ?", (cutoff,))
        conn.execute("DELETE FROM runs WHERE id <= ?", (cutoff,))
        conn.commit()
    conn.close()


def insert_run(
    prompt: str,
    success_count: int,
    total_pairs: int,
    fastest_pair: str,
    fastest_time: int,
    candidates_found: int,
    pairs_working: int,
    pairs_loading: int,
    pairs_rate_limited: int,
    pairs_unsupported: int,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO runs
        (timestamp, prompt, success_count, total_pairs, fastest_pair, fastest_time,
         candidates_found, pairs_working, pairs_loading, pairs_rate_limited, pairs_unsupported)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            prompt,
            success_count,
            total_pairs,
            fastest_pair,
            fastest_time,
            candidates_found,
            pairs_working,
            pairs_loading,
            pairs_rate_limited,
            pairs_unsupported,
        ),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def insert_model_result(run_id: int, result: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO model_results
        (run_id, model, provider, success, error, error_category,
         response_time, ttft, tokens_generated, total_tokens, response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            result.get("model"),
            result.get("provider"),
            1 if result.get("success") else 0,
            result.get("error"),
            result.get("error_category"),
            result.get("response_time"),
            result.get("ttft"),
            result.get("tokens_generated"),
            result.get("total_tokens"),
            result.get("response"),
        ),
    )
    conn.commit()
    conn.close()


def insert_discovery_snapshot(run_id: int, snapshot: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO discovery_snapshots
        (run_id, model, provider, status, downloads, likes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            snapshot["model"],
            snapshot["provider"],
            snapshot.get("status", "unknown"),
            snapshot.get("downloads"),
            snapshot.get("likes"),
        ),
    )
    conn.commit()
    conn.close()


def dump_db_to_json():
    """Export full DB as JSON for potential debugging or backup."""
    conn = get_conn()
    data = {}
    for table in ("runs", "model_results", "discovery_snapshots"):
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        data[table] = [dict(r) for r in rows]
    conn.close()
    return data
