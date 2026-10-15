# HFStats v3 — Provider-Level HuggingFace Benchmark Dashboard

Benchmark HuggingFace Inference Providers models **per provider** (`model:provider` pair) to enable deterministic comparisons, routing visibility, and uptime tracking.

## Architecture

```
HF Hub API (/api/models, sorted by downloads)
    │
    ▼
Discover Candidates (≥ MIN_DOWNLOADS, at least one "live" provider)
    ▼
Extract model:provider Pairs (from inferenceProviderMapping)
    ▼
Classify Each Pair (tiny test, max_tokens=5)          ─┐  HF Inference Providers router
    ▼                                                   ├─ POST https://router.huggingface.co/v1/chat/completions
Benchmark Working Pairs (stream=true, max_tokens=500)  ─┘  body: {"model": "<model>:<provider>", ...}
    ▼
SQLite (history.db) → Static Dashboard (7 tabs, sql.js in the browser)
```

There is no server: GitHub Actions runs the pipeline, `history.db` is committed
back to the repo, and `index.html` loads it in the browser.

## Project Structure

```
hfstats/
├── .github/
│   ├── workflows/benchmark.yml   # scheduled pipeline (every 6 hours)
│   ├── workflows/tests.yml       # unit tests on every push / PR
│   └── dependabot.yml
├── scripts/
│   ├── common.py                 # router requests, error categories, JSON helpers
│   ├── discover_models.py
│   ├── classify_models.py
│   ├── benchmark_models.py
│   ├── merge_results.py
│   └── db_utils.py
├── tests/                        # pytest suite (network fully mocked)
├── index.html                    # the dashboard
├── history.db                    # benchmark history (written by CI)
└── requirements-dev.txt
```

## Quickstart

1. Install Python 3.11+ (the pipeline uses only the standard library).
2. Create a HuggingFace token with the **"Make calls to Inference Providers"** permission and export it as `HF_TOKEN`.
3. Run the pipeline locally:

```bash
python3 scripts/discover_models.py
python3 scripts/classify_models.py
for g in group1 group2 group3; do
  MODEL_GROUP=$g RESULTS_PATH=scripts/results-$g.json python3 scripts/benchmark_models.py
done
python3 scripts/merge_results.py
python3 -m http.server 8000
# Open http://localhost:8000
```

Run the tests with `pip install -r requirements-dev.txt && python -m pytest`.

## Dashboard Tabs

1. **Overview** — KPIs, success trend, top performers
2. **Leaderboard** — Sortable `model:provider` pairs with sparklines
3. **Explorer** — Per-pair response-time / TTFT history, error-category donut, run history
4. **Timeline** — Expandable run cards with per-pair tables
5. **Compare** — Head-to-head comparison of two pairs (aligned by run)
6. **Providers** — Provider-level uptime, 24h success rate, models served
7. **Discovery** — Classification breakdown, volatility analytics

Metrics: **TTFT** is the time until the first generated token (reasoning tokens
count); **throughput** is output tokens per second after the first token, using
the provider's reported token usage when available.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (Inference Providers permission) |
| `MIN_DOWNLOADS` | 10000 | Minimum monthly downloads for discovery |
| `DISCOVER_LIMIT` | 500 | Maximum number of candidate models |
| `DISCOVER_MAX_PAGES` | 100 | Safety cap on Hub API pages per run |
| `SKIP_GATED` | off | Set to `1` to skip gated models (Llama, Gemma, …) |
| `CLASSIFY_CONCURRENCY` | 6 | Providers classified in parallel (one request at a time per provider) |
| `MAX_BENCHMARK_PAIRS` | 50 | Cap on pairs to benchmark |
| `MODEL_GROUP` | group1 | Which shard to benchmark (group1/2/3) |
| `MAX_RUNS` | 360 | Runs kept in `history.db` (~90 days at 4 runs/day) |

## Deployment & schedule

The workflow runs **every 6 hours** (`23 */6 * * *`), benchmarks in parallel
across 3 runners, merges results into `history.db` and commits it back. Serve
`index.html` + `history.db` from any static host (GitHub Pages recommended).

Why a schedule at all: uptime, success trends and churn are time series, so the
pipeline samples providers around the clock. Every run makes real, billed
inference calls (~400 tiny classification requests and up to 50 streamed
500-token completions), so the frequency is a cost/resolution trade-off — edit
the cron line to change it. The pipeline fails loudly (instead of recording an
empty run) when no pair works, which usually means an invalid token, missing
permission or exhausted inference credits.

GitHub disables scheduled workflows in repositories with no activity for 60
days; if that happens, re-enable it from the Actions tab.

## License

MIT
