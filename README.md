# HFStats v3 — Provider-Level HuggingFace Benchmark Dashboard

Benchmark HuggingFace Inference API models **per provider** (`model:provider` pair) to enable deterministic comparisons, routing visibility, and uptime tracking.

## Architecture

```
HF API (/api/models)
    │
    ▼
Discover Candidates (paginated, filtered)
    ▼
Extract model:provider Pairs (from inferenceProviderMapping)
    ▼
Classify Each Pair (tiny test, max_tokens=5)
    ▼
Benchmark Working Pairs (stream=true, max_tokens=500)
    ▼
SQLite (history.db) → Static Dashboard (7 tabs)
```

## Project Structure

```
huggingface-stat/
├── .github/
│   ├── workflows/benchmark.yml
│   └── dependabot.yml
├── scripts/
│   ├── db_utils.py
│   ├── discover_models.py
│   ├── classify_models.py
│   ├── benchmark_models.py
│   └── merge_results.py
├── index.html
├── history.db
└── README.md
```

## Quickstart

1. Install Python 3.11+
2. Set `HF_TOKEN` environment variable with a valid HuggingFace token
3. Run the pipeline locally:

```bash
python3 scripts/discover_models.py
HF_TOKEN=$HF_TOKEN python3 scripts/classify_models.py
HF_TOKEN=$HF_TOKEN MODEL_GROUP=group1 python3 scripts/benchmark_models.py
python3 scripts/merge_results.py
python3 -m http.server 8000
# Open http://localhost:8000
```

## Dashboard Tabs

1. **Overview** — KPIs, success trends, top performers
2. **Leaderboard** — Sortable `model:provider` pairs with sparklines
3. **Explorer** — Per-pair time/TTFT history, error donut, heatmap
4. **Timeline** — Expandable run cards with per-pair tables
5. **Compare** — Head-to-head comparison of two pairs
6. **Providers** — Provider-level uptime, success rate, models served
7. **Discovery** — Classification breakdown, volatility analytics

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `MIN_DOWNLOADS` | 10000 | Minimum downloads for discovery |
| `MAX_BENCHMARK_PAIRS` | 50 | Cap on pairs to benchmark |
| `MODEL_GROUP` | group1 | Which shard to benchmark (group1/2/3) |

## Deployment

The GitHub Actions workflow runs every 2 hours, benchmarks in parallel across 3 runners, merges results into `history.db`, and commits it back. Serve `index.html` + `history.db` from any static host (GitHub Pages recommended).

## License

MIT
