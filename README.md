# model-template

A self-contained benchmark harness that runs an LLM in an autonomous agentic loop to research and improve a quantitative trading strategy on GBPUSD tick data. The loop issues commands, edits `strategy.py`, runs EDA and backtests, and tracks results on the shared leaderboard.

Supports **any OpenAI-compatible REST API** (OpenAI, Mistral, Together.ai, OpenRouter, Ollama, vLLM, and more), plus Anthropic and Google Gemini natively.

---

## Table of Contents

1. [Repository structure](#1-repository-structure)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-step onboarding](#3-step-by-step-onboarding)
   - [3.1 Fork the template](#31-fork-the-template)
   - [3.2 Create a bot account and PAT](#32-create-a-bot-account-and-pat)
   - [3.3 Configure secrets](#33-configure-secrets)
   - [3.4 Configure variables](#34-configure-variables)
     - [3.4.1 Choose a provider and model](#341-choose-a-provider-and-model)
     - [3.4.2 Set the in-sample date window](#342-set-the-in-sample-date-window)
   - [3.5 Verify the setup](#35-verify-the-setup)
4. [How the agentic loop works](#4-how-the-agentic-loop-works)
5. [File ownership and protected files](#5-file-ownership-and-protected-files)
6. [strategy.py contract](#6-strategypy-contract)
7. [Running a backtest manually](#7-running-a-backtest-manually)
8. [Submitting a PR to main](#8-submitting-a-pr-to-main)
9. [Local development](#9-local-development)

---

## 1. Repository structure

```
model-template/
├── strategy.py              # Your model's strategy — agent-writable, you own this
├── prompt_context.md        # LLM system prompt context — editable by contributors
├── test_runner.py           # Execution engine — READ-ONLY (CODEOWNERS-protected)
├── releases.md              # Changelog — one [vN] entry required per PR to main
├── research_summary.md      # EDA findings table — agent-maintained
├── audit_logs/
│   └── thoughts.md          # Agent reasoning log — required before each command
├── research/                # EDA scripts and output logs — agent-generated
├── harness/
│   ├── providers.py         # LLM adapters: OpenAI-compatible (any base URL), Anthropic, Google
│   └── call_model.py        # CLI wrapper invoked by the agentic loop workflow
├── tests/                   # Unit tests
└── .github/
    ├── catchup_state.json    # One-time post-release evaluation progress
    ├── loop_state.json       # Agentic loop counters (last_run, daily_count)
    └── workflows/
        ├── agentic_loop.yml  # Core loop — triggers after EDA or backtest completes
        ├── run_eda.yml       # Dispatches EDA into the infra-managed strategy-runner image
        ├── run_backtest.yml  # Dispatches a full vectorbt backtest into the infra-managed strategy-runner image
        ├── pr_guard.yml      # Enforces quality gates on PRs to main
        ├── daily_eval.yml    # Out-of-sample eval, runs Tue–Sat at 08:30 UTC
        └── catch_up_eval.yml # One-time chronological eval after first release
```

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| GitHub account | Must have permission to fork the `fxquantbench/model-template` repo |
| `BENCHMARK_BOT_TOKEN` | **Provided automatically** — org-level secret on `fxquantbench`; no action required |
| `HF_TOKEN_RO` | **Provided automatically** — org-level secret on `fxquantbench`; no action required |
| LLM API key | Any OpenAI-compatible endpoint, Anthropic, or Google — one is enough |

No local tooling is required to run the benchmark — everything executes in GitHub Actions. A local Python ≥ 3.11 environment (or `uv`) is only needed to run tests and iterate on `strategy.py` before letting the agent take over.

---

## 3. Step-by-step onboarding

### 3.1 Fork the template

1. Click **Use this template → Create a new repository** (not a plain fork, so CI state is fresh).
2. Name your repo anything you like; keep it **private** until you are ready to submit.
3. Clone it locally:

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 3.2 Org secrets (no action required)

`BENCHMARK_BOT_TOKEN` and `HF_TOKEN_RO` are **organisation-level secrets** managed by `fxquantbench`. They are automatically inherited by every repository created from this template inside the org. You do not need to create a bot account, generate a PAT, or obtain a HuggingFace token.

If you created your repo outside the `fxquantbench` org, you need to transfer it into the org so the secrets are inherited automatically:

1. Open an issue in [fxquantbench/model-template](https://github.com/fxquantbench/model-template) letting the benchmark admin know you want to transfer your repo in.
2. Once the admin confirms they are ready to accept, go to your repo **Settings → General → Danger Zone → Transfer ownership**, enter `fxquantbench` as the destination, and confirm.
3. The benchmark admin will accept the transfer — your repo moves into the org and the secrets become available immediately. You retain admin access to your repo throughout.

### 3.3 Configure secrets

Go to **Settings → Secrets and variables → Actions → Secrets** in your repository and add:

| Secret name | Value |
|---|---|
| `MODEL_API_KEY` | Your LLM provider API key |

`BENCHMARK_BOT_TOKEN` and `HF_TOKEN_RO` are inherited from the org — do not add them manually.

### 3.4 Configure variables

Go to **Settings → Secrets and variables → Actions → Variables** and add:

| Variable name | Required | Description | Example |
|---|---|---|---|
| `MODEL_PROVIDER` | Yes | LLM provider — `openai`, `anthropic`, or `google` | `openai` |
| `MODEL_ID` | Yes | Model identifier string | `gpt-4o` |
| `MODEL_BASE_URL` | No | Override the API base URL for `openai` provider (any OpenAI-compatible endpoint) | `https://api.mistral.ai/v1` |
| `STRATEGY_RUNNER_IMAGE` | No | Override the infra-managed runner image reference used by EDA, backtest, and eval workflows | `ghcr.io/fxquantbench/strategy-runner:2026-05-10` |
| `MAX_DAILY_ITERATIONS` | No | Max agentic loop runs per day (default: `6`) | `6` |
| `ARTIFACT_ARCHIVE_THRESHOLD` | No | Archive oldest EDA script/log bundles and backtest logs when the active count reaches this threshold (default: `30`) | `30` |

#### 3.4.1 Choose a provider and model

Set `MODEL_PROVIDER` and `MODEL_ID` to select your model. The `openai` provider supports **any OpenAI-compatible REST API** — set `MODEL_BASE_URL` to point at a different endpoint.

| `MODEL_PROVIDER` | `MODEL_BASE_URL` | Example `MODEL_ID` values | Notes |
|---|---|---|---|
| `openai` | *(unset — default)* | `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini` | Official OpenAI API |
| `openai` | `https://openrouter.ai/api/v1` | `openai/gpt-5.5`, `anthropic/claude-opus-4-7`, `google/gemini-2.5-pro` | OpenRouter — access any provider via one key |
| `anthropic` | *(n/a)* | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` | Forced tool-use for structured output |
| `google` | *(n/a)* | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | Uses `google.genai`; `response_mime_type="application/json"` |

> **Note:** Some OpenAI-compatible endpoints do not support the `json_schema` response format. If you hit errors with structured output, the model or endpoint may require a plain `json_object` mode — open an issue to request support for that endpoint.

The model receives the full `prompt_context.md` as the system prompt plus dynamic context (leaderboard summaries, current `strategy.py`, recent thoughts) as the user message.

#### 3.4.2 Set the in-sample date window

The GBPUSD in-sample window is fixed for every template-derived repository: inclusive `2025-01-01` through inclusive `2026-05-31`. The runner enforces the equivalent half-open interval `[2025-01-01, 2026-06-01)` — no data outside this range is accessible during EDA or backtest. No date variables need to be configured when creating a repository.

EDA and backtest stage the available in-sample parquet day shards locally on the GitHub runner and pass them into the strategy-runner container via `TICK_DATA_GLOB=/input/*.parquet`. Those workflows restore a cache keyed by the fixed start and end dates to avoid re-downloading the same shards, and they skip calendar days that have no published shard in the dataset. Daily eval uses its own ephemeral stage directory and must not restore that in-sample cache.

Choose dates that leave at least 6 months of unseen data for out-of-sample evaluation. The daily eval job tests `strategy.py` on yesterday's ticks (always outside the in-sample window).

`STRATEGY_RUNNER_IMAGE` defaults to `ghcr.io/fxquantbench/strategy-runner:latest`, which is built and published from `FXQuantBench/infra`. Set it to a versioned tag or digest if you want to pin the execution runtime across workflow runs.

### 3.5 Verify the setup

Trigger the agentic loop manually to confirm everything is wired up:

1. Go to **Actions → Agentic Loop → Run workflow**.
2. Watch the run — the first iteration reads `loop_state.json`, calls the model, applies any `file_changes`, and updates the loop state.
3. Check that a commit appears on the `dev` branch and `audit_logs/thoughts.md` was updated.

If the run fails, the most common causes are:
- `MODEL_PROVIDER` or `MODEL_ID` variable is missing
- `MODEL_API_KEY` secret is not set or is set under a different name (must be exactly `MODEL_API_KEY`)
- `MODEL_BASE_URL` points to an endpoint that does not support `json_schema` response format
- `STRATEGY_RUNNER_IMAGE` points at a tag that does not exist or that `BENCHMARK_BOT_TOKEN` cannot read from GHCR
- `BENCHMARK_BOT_TOKEN` or `HF_TOKEN_RO` were not inherited from the org — backtest and eval run via reusable workflows that use org secrets from `fxquantbench/model-template` directly

---

## 4. How the agentic loop works

```
Agentic Loop
 │
 ├─ Trigger: workflow_dispatch  OR  after "Run EDA" / "Run Backtest" completes
 │
 ├─ Guard: < 30 min since last run?  → skip
 ├─ Guard: daily_count ≥ MAX_DAILY_ITERATIONS?  → skip
 │
 ├─ Fetch leaderboard summaries (via gh api)
 ├─ Build context file:  prompt_context.md + ---USER--- + dynamic context
 ├─ Call model:  python harness/call_model.py --context-file <path>
 │
 ├─ Model response (JSON):
 │     { "thoughts": "...",
 │       "file_changes": [{"path": "strategy.py", "content": "..."}],
 │       "commands": ["/run-eda <id>"]  |  ["/run-backtest"]  |  [] }
 │
 ├─ Append thoughts → audit_logs/thoughts.md
 ├─ Apply file_changes (skips test_runner.py and prompt_context.md)
 ├─ Dispatch first command → triggers run_eda.yml or run_backtest.yml
 ├─ Commit all changes to dev branch
 └─ Update .github/loop_state.json (last_run, daily_count)
```

The loop re-triggers itself after each EDA or backtest completes, and also after a successful daily eval result is committed, so a single `workflow_dispatch` starts a self-sustaining research cycle up to the daily cap while scheduled evals can feed fresh out-of-sample feedback back into the next agent turn.

Each `agentic_loop.yml` invocation increments `daily_count` before it dispatches any child workflow. Manual `workflow_dispatch` runs are blocked if the previous loop was less than 30 minutes ago, but child-workflow resumes from `run_eda.yml`, `run_backtest.yml`, and successful `daily_eval.yml` runs bypass that gap check while still counting against `MAX_DAILY_ITERATIONS`.

The loop context now includes the latest leaderboard summaries for both backtest and eval runs. After a scheduled daily eval resumes the loop, the agent can inspect the newest eval metrics directly from the injected context before choosing its next action.

After the first strategy reaches `main`, a one-time catch-up starts at `2026-06-01`. It evaluates one business day at a time through the date before the next regular daily evaluation, then gives the model one reflection-only turn before continuing. Catch-up uses the same isolated daily-eval path and the current `main` strategy, shares the normal `MAX_DAILY_ITERATIONS` ceiling, and resumes after the UTC daily counter resets. It is not restarted for later releases; after completion, the strategy participates only in regular daily evaluation.

EDA scripts run inside `test_runner.py` with `conn` and `pairs = ["GBPUSD"]` injected into the script namespace. Use `conn.execute(...)` against the preloaded `GBPUSD` view instead of opening a fresh DuckDB connection. Do not use `duckdb.sql(...)` for benchmark queries: it uses DuckDB's default connection rather than the injected runner connection, which is why logs can show `Catalog Error: Table with name GBPUSD does not exist!` even though the runner created the view correctly. If you wrap logic in a helper, use `def main(conn): ...` and call `main(conn)`. EDA and backtest restore or build an exact-window local shard stage before the container runs, while daily eval keeps a separate ephemeral stage so out-of-sample data never becomes restorable by dev workflows. The EDA workflow copies only the first non-empty log line into `research_summary.md`, and a committed `research/<file_id>.log` causes future `/run-eda <file_id>` attempts to be skipped, so retries need a new file ID.

For a live sanity check against real staged HF data, run `python scripts/check_hf_dataset_access.py ...`. It exercises both the documented SQL queries and a sample EDA script executed through `test_runner.py` with the injected `conn`.

---

## 5. File ownership and protected files

| File | Who can modify | Notes |
|---|---|---|
| `strategy.py` | Agent and contributors | The only strategy file the runner executes |
| `audit_logs/thoughts.md` | Workflow-owned | Appends the model's `thoughts` with a system-generated UTC timestamp |
| `releases.md` | Workflow and contributors | CI appends the system-generated timestamp and next `## [vN]` for a model-supplied release note |
| `research/` | Agent only | EDA scripts and output logs, auto-managed |
| `research_summary.md` | Workflow and agent | CI owns dates and findings; the model supplies targeted hypothesis/verdict updates |
| `test_runner.py` | **CODEOWNERS only** | Protected — the agent cannot overwrite this file |
| `prompt_context.md` | Contributors | Defines the LLM's task and data contract — customise to guide your model |

The CODEOWNERS file enforces that `test_runner.py` requires approval from `@fxquantbench/benchmark-admin` before any PR touching it can be merged. `prompt_context.md` is no longer protected and can be freely edited by contributors.

---

## 6. strategy.py contract

The runner calls `run(conn, start_date, end_date)`. Your function must:

```python
def run(conn, start_date: str, end_date: str) -> pd.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `conn` | `duckdb.DuckDBPyConnection` | Connection with the `GBPUSD` view pre-loaded |
| `start_date` | `str` | Inclusive start date `"YYYY-MM-DD"` |
| `end_date` | `str` | Exclusive end date `"YYYY-MM-DD"` |

Return a `pd.DataFrame` with exactly these columns:

| Column | dtype | Description |
|---|---|---|
| `timestamp_utc` | `int64` | Unix timestamp in milliseconds (UTC) |
| `pair` | `str` | Always `"GBPUSD"` |
| `signal` | `float64` | Target position in `[-1.0, 1.0]` — `+1.0` = 100% long, `-1.0` = 100% short, `0.0` = flat |

Returning an empty DataFrame is valid and means "hold flat / no position". The runner clamps signals to `[-1, 1]` before simulation.

The GBPUSD view has these columns: `timestamp_utc` (int64 ms), `bid` (float), `ask` (float), `bid_volume` (float), `ask_volume` (float). Spread `(ask - bid)` is charged as a fee on every position change — no other commission applies.

For SQL and EDA code, query only the preloaded `GBPUSD` view via `conn.execute(...)`. The SQL view does not include a `pair` column; add `pair="GBPUSD"` only in the returned signal DataFrame. If order matters, use `ORDER BY timestamp_utc`, and remember that `timestamp_utc` is in milliseconds.

---

## 7. Running a backtest manually

The `run_backtest.yml` workflow runs `strategy.py` inside the infra-managed `strategy-runner` container image published from `FXQuantBench/infra`, writes `result.json`, and commits it to the leaderboard. Before the container starts, the reusable workflow restores or stages the exact in-sample GBPUSD parquet shards locally and mounts them as `/input/*.parquet`, using the fixed in-sample cache namespace rather than DuckDB remote Hugging Face reads.

By default the workflows pull `ghcr.io/fxquantbench/strategy-runner:latest`. If you set `STRATEGY_RUNNER_IMAGE`, the EDA, backtest, and daily eval workflows use that image reference instead. This is the preferred way to pin the benchmark runtime to a versioned image tag or digest.

**Requirements before triggering:**
- The latest commit on `dev` must include an update to `audit_logs/thoughts.md`.
- `strategy.py` must be syntactically valid Python with a top-level `run(conn, start_date, end_date)` function.

**To trigger:**
1. Go to **Actions → Run Backtest → Run workflow** (select the `dev` branch).
2. The workflow pulls the configured runner image from GHCR, validates the result against `ResultSchema`, and pushes `result.json` to the leaderboard under `<MODEL_ID>/results/backtest/<YYYY-MM-DD>-<short-sha>.json`.
3. A summary table with all 17 metrics is posted to the job summary.

**Result fields:**

| Field | Description |
|---|---|
| `sharpe` | Annualised Sharpe ratio |
| `max_drawdown` | Maximum drawdown (fraction, negative) |
| `win_rate` | Fraction of trades that were profitable |
| `calmar_ratio` | Annualised return / max drawdown |
| `annualized_return` | Annualised return (fraction) |
| `volatility` | Annualised return volatility |
| `total_trades` | Number of position changes |
| `avg_spread_cost_pips` | Average spread paid per trade in pips |
| `timed_out` | `true` if the 5-minute container timeout was hit |

---

## 8. Submitting a PR to main

When you are ready to submit, open a PR from `dev` → `main`. The `pr_guard.yml` workflow runs six automated checks:

| Check | What it verifies |
|---|---|
| 1 | `strategy.py` is valid Python (`ast.parse`) |
| 2 | `strategy.py` has a top-level `run(conn, start_date, end_date)` function |
| 3 | `audit_logs/thoughts.md` contains the PR branch name |
| 4 | `releases.md` has a `## [vN]` entry newer than the last merge to `main` |
| 5 | A backtest result JSON for the current HEAD short SHA exists in the leaderboard |
| 6 | The backtest result's `strategy_sha` matches the PR HEAD SHA, and `sharpe > -10.0` |

All six checks must pass for the PR to be mergeable.

**Pre-PR checklist:**
- [ ] `audit_logs/thoughts.md` contains the branch name
- [ ] `releases.md` has a new `## [vN] — <description>` entry
- [ ] A backtest has been run for the current HEAD commit (`/run-backtest` issued by the agent, or triggered manually)
- [ ] Sharpe > -10.0 in the latest backtest result

---

## 9. Local development

Set up a local environment to iterate on `strategy.py` and run tests:

```bash
# Create and activate venv (using uv, recommended)
uv venv --python 3.13
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install test dependencies
uv pip install pytest duckdb pandas numpy pydantic

# Run all unit tests
uv run pytest tests/ -v
```

The test suite runs entirely offline — no HuggingFace token or LLM API key required. `vectorbt` is stubbed in `tests/conftest.py` so the full install is not needed for testing.

To iterate on `strategy.py` locally, query DuckDB directly with your own tick data or a synthetic dataset:

```python
import duckdb
import pandas as pd
from strategy import run

conn = duckdb.connect()
# Create a minimal GBPUSD view for local testing
conn.execute("""
    CREATE VIEW GBPUSD AS
    SELECT * FROM read_parquet('path/to/local/ticks.parquet')
""")

signals = run(conn, "2022-01-03", "2023-01-01")
print(signals.head())
```
