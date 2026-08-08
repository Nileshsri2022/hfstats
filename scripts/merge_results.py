"""Merge parallel benchmark results and discovery snapshots into SQLite DB."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import db_utils
from common import load_json

CLASSIFIED_PATH = os.environ.get("CLASSIFIED_PATH", "scripts/classified.json")
GROUP_PATHS = [
    os.environ.get("RESULTS_GROUP1", "scripts/results-group1.json"),
    os.environ.get("RESULTS_GROUP2", "scripts/results-group2.json"),
    os.environ.get("RESULTS_GROUP3", "scripts/results-group3.json"),
]


def main():
    db_utils.init_db()

    classified = load_json(CLASSIFIED_PATH)
    if not classified:
        print(f"ERROR: {CLASSIFIED_PATH} not found")
        sys.exit(1)

    all_results = []
    prompt = ""
    missing_groups = []
    for path in GROUP_PATHS:
        data = load_json(path)
        if data:
            all_results.extend(data.get("results", []))
            if data.get("prompt"):
                prompt = data["prompt"]
        else:
            missing_groups.append(path)

    if missing_groups:
        print(
            "WARNING: missing benchmark result files (their pairs will be absent "
            f"from this run): {', '.join(missing_groups)}",
            file=sys.stderr,
        )
    if not all_results:
        print(
            "ERROR: no benchmark results found in any group file; refusing to "
            "record an empty run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute aggregate stats
    success_count = sum(1 for r in all_results if r.get("success"))
    total_pairs = len(all_results)

    fastest_pair = ""
    fastest_time = None
    for r in all_results:
        if r.get("success"):
            t = r.get("response_time")
            if t is not None and (fastest_time is None or t < fastest_time):
                fastest_time = t
                fastest_pair = f"{r['model']}:{r['provider']}"

    # Count classified categories
    pairs = classified.get("pairs", [])
    counts = classified.get("counts", {})
    candidates_found = len(pairs)
    pairs_working = counts.get("working", 0)
    pairs_loading = counts.get("loading", 0)
    pairs_rate_limited = counts.get("rate_limited", 0)
    pairs_unsupported = (
        counts.get("unsupported", 0)
        + counts.get("not_found", 0)
        + counts.get("quota_exceeded", 0)
    )

    run_id = db_utils.insert_run(
        prompt=prompt,
        success_count=success_count,
        total_pairs=total_pairs,
        fastest_pair=fastest_pair,
        fastest_time=fastest_time or 0,
        candidates_found=candidates_found,
        pairs_working=pairs_working,
        pairs_loading=pairs_loading,
        pairs_rate_limited=pairs_rate_limited,
        pairs_unsupported=pairs_unsupported,
    )

    # Insert model results
    for r in all_results:
        db_utils.insert_model_result(run_id, r)

    # Insert discovery snapshots
    for p in pairs:
        db_utils.insert_discovery_snapshot(
            run_id,
            {
                "model": p["model"],
                "provider": p["provider"],
                "status": p.get("status", "unknown"),
                "downloads": p.get("downloads"),
                "likes": p.get("likes"),
            },
        )

    db_utils.prune_old_runs(max_runs=360)

    print(f"Merged {total_pairs} results into run_id={run_id}")
    print(f"  Success: {success_count}/{total_pairs}")
    print(f"  Fastest: {fastest_pair} ({fastest_time}ms)")
    print(f"  DB: {db_utils.DB_PATH}")


if __name__ == "__main__":
    main()
