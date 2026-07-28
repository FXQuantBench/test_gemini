"""Regression tests for prompt/docs/workflow guidance that steers the agent loop."""

import duckdb

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class TestPromptContextGuidance:
    def test_mentions_loop_budget_and_resume_flow(self):
        text = _read("prompt_context.md")
        assert "MAX_DAILY_ITERATIONS" in text
        assert "run_eda.yml" in text
        assert "run_backtest.yml" in text
        assert "daily_eval.yml" in text
        assert "latest leaderboard summaries for both backtest results and eval results" in text
        assert "first non-empty line" in text
        assert "new `file_id`" in text

    def test_mentions_injected_conn_and_gbpusd_query_rules(self):
        text = _read("prompt_context.md")
        assert "Use the injected `conn` object" in text
        assert "There is no `pair` column in the `GBPUSD` SQL view" in text
        assert "Do not open a fresh DuckDB connection" in text
        assert "Do not use module-level `duckdb.sql(...)`" in text
        assert "Table with name GBPUSD does not exist" in text
        assert "def main(conn): ..." in text
        assert "The `GBPUSD` view is already filtered to `[2025-01-01, 2026-06-01)" in text
        assert "record_updates" in text
        assert "do not include dates, release versions, or markdown headings" in text
        assert "catch_up_eval" in text

    def test_documented_query_examples_execute_against_gbpusd_view(self):
        text = _read("prompt_context.md")
        assert "(bid + ask) / 2.0 AS mid" in text
        assert "timestamp_utc - (timestamp_utc % 60000) AS minute_bucket_utc_ms" in text

        conn = duckdb.connect()
        conn.execute("""
            CREATE VIEW GBPUSD AS
            SELECT *
            FROM (
                VALUES
                    (1704067201000::BIGINT, 1.2700::DOUBLE, 1.2702::DOUBLE, 1000.0::DOUBLE, 1200.0::DOUBLE),
                    (1704067215000::BIGINT, 1.2701::DOUBLE, 1.2703::DOUBLE, 1100.0::DOUBLE, 1300.0::DOUBLE),
                    (1704067261000::BIGINT, 1.2702::DOUBLE, 1.2704::DOUBLE, 1200.0::DOUBLE, 1400.0::DOUBLE)
            ) AS ticks(timestamp_utc, bid, ask, bid_volume, ask_volume)
        """)

        sample = conn.execute(
            """
            SELECT
              timestamp_utc,
              bid,
              ask,
              (bid + ask) / 2.0 AS mid,
              ask - bid AS spread
            FROM GBPUSD
            ORDER BY timestamp_utc
            LIMIT 10
            """
        ).df()

        minute_stats = conn.execute(
            """
            SELECT
              timestamp_utc - (timestamp_utc % 60000) AS minute_bucket_utc_ms,
              COUNT(*) AS tick_count,
              AVG((bid + ask) / 2.0) AS avg_mid,
              AVG(ask - bid) AS avg_spread
            FROM GBPUSD
            GROUP BY 1
            ORDER BY 1
            LIMIT 20
            """
        ).df()

        assert len(sample) == 3
        assert list(sample.columns) == ["timestamp_utc", "bid", "ask", "mid", "spread"]
        assert len(minute_stats) == 2
        assert list(minute_stats.columns) == [
            "minute_bucket_utc_ms",
            "tick_count",
            "avg_mid",
            "avg_spread",
        ]


class TestReadmeGuidance:
    def test_mentions_daily_limit_and_eda_contract(self):
        text = _read("README.md")
        assert "MAX_DAILY_ITERATIONS" in text
        assert "conn.execute(...)" in text
        assert "first non-empty log line" in text
        assert "new file ID" in text
        assert "does not include a `pair` column" in text
        assert "inclusive `2025-01-01` through inclusive `2026-05-31`" in text
        assert "No date variables need to be configured" in text
        assert "ARTIFACT_ARCHIVE_THRESHOLD" in text
        assert "one-time catch-up starts at `2026-06-01`" in text
        assert "Daily eval uses its own ephemeral stage directory" in text
        assert "skip calendar days that have no published shard in the dataset" in text
        assert "successful daily eval result is committed" in text
        assert "latest leaderboard summaries for both backtest and eval runs" in text
        assert "Do not use `duckdb.sql(...)` for benchmark queries" in text
        assert "def main(conn): ..." in text
        assert "scripts/check_hf_dataset_access.py" in text


class TestEDAWorkflowGuidance:
    def test_wrapper_passes_in_sample_window(self):
        text = _read(".github/workflows/run_eda.yml")
        assert "in_sample_start: '2025-01-01'" in text
        assert "in_sample_end: '2026-06-01'" in text
        assert "vars.IN_SAMPLE_START" not in text
        assert "vars.IN_SAMPLE_END" not in text
        assert "artifact_archive_threshold" in text

    def test_reusable_workflow_preloads_view_and_flags_empty_logs(self):
        text = _read(".github/workflows/_run_eda.yml")
        assert "-e START_DATE=\"${{ inputs.in_sample_start }}\"" in text
        assert "-e END_DATE=\"${{ inputs.in_sample_end }}\"" in text
        assert "EDA_EMPTY_LOG: 'false'" in text
        assert "Fail workflow on empty EDA log" in text
        assert "printf '%s\\n'" in text
        assert "cat <<'EOF'" not in text
        assert "Record EDA finding in research_summary.md" in text
        assert "Archive old EDA artifacts when threshold is reached" in text
        assert "ARTIFACT_ARCHIVE_THRESHOLD" in text

    def test_backtest_uses_fixed_window_and_shared_archive_threshold(self):
        text = _read(".github/workflows/run_backtest.yml")
        reusable = _read(".github/workflows/_run_backtest.yml")
        assert "in_sample_start: '2025-01-01'" in text
        assert "in_sample_end: '2026-06-01'" in text
        assert "artifact_archive_threshold" in text
        assert "Archive old backtest artifacts when threshold is reached" in reusable
        assert "ARTIFACT_ARCHIVE_THRESHOLD" in reusable
        assert "backtest_results" in reusable


class TestWorkflowDataStaging:
    def test_eda_and_backtest_stage_local_in_sample_shards(self):
        for relative_path in (
            ".github/workflows/_run_eda.yml",
            ".github/workflows/_run_backtest.yml",
        ):
            text = _read(relative_path)
            assert "INSAMPLE_CACHE_NAMESPACE: 'gbpusd-insample-v1'" in text
            assert "actions/cache/restore@v4" in text
            assert "actions/cache/save@v4" in text
            assert "Stage in-sample shards locally" in text
            assert "cache_key=${{ env.INSAMPLE_CACHE_NAMESPACE }}-${{ runner.os }}-${{ inputs.in_sample_start }}-${{ inputs.in_sample_end }}" in text
            assert "-e TICK_DATA_GLOB=\"/input/*.parquet\"" in text
            assert ":/input:ro" in text
            assert "manifest.txt" in text
            assert "HfFileSystem" in text
            assert "Skipping unavailable shard day" in text
            assert 'repo_id="FXQuantBench/fx-ticks"' in text
            assert "Write in-sample stage summary" in text
            assert "In-sample shard stage: cache_hit=${CACHE_HIT_VALUE} shard_count=${SHARD_COUNT}/${EXPECTED_COUNT} skipped_missing_days=${SKIPPED_COUNT_VALUE} cache_complete=${CACHE_COMPLETE_VALUE}" in text
            assert '>> "$GITHUB_STEP_SUMMARY"' in text

    def test_daily_eval_stays_ephemeral_and_does_not_share_dev_cache(self):
        text = _read(".github/workflows/_daily_eval.yml")
        assert "EVAL_STAGE_ISOLATION: 'ephemeral_only'" in text
        assert "Stage eval shards locally (ephemeral only)" in text
        assert "no shared cache restore/save is allowed" in text
        assert "actions/cache/restore@v4" not in text
        assert "actions/cache/save@v4" not in text


class TestEvalLoopResume:
    def test_daily_eval_wrapper_resumes_agent_loop_after_fresh_success(self):
        text = _read(".github/workflows/daily_eval.yml")
        assert "actions: write" in text
        assert "resume_loop:" in text
        assert "needs.eval.outputs.resume_loop == 'true'" in text
        assert "Resume agent loop after eval" in text
        assert "-f trigger_source=daily_eval" in text
        assert "-f trigger_details=\"eval_date=${{ needs.eval.outputs.eval_date }}\"" in text

    def test_reusable_daily_eval_exposes_resume_outputs(self):
        text = _read(".github/workflows/_daily_eval.yml")
        assert "outputs:" in text
        assert "resume_loop:" in text
        assert "value: ${{ jobs.eval.outputs.resume_loop }}" in text
        assert "value: ${{ jobs.eval.outputs.eval_date }}" in text
        assert "eval_date: ${{ steps.date.outputs.eval_date }}" in text
        assert "id: resume_gate" in text
        assert 'echo "resume_loop=false" >> "$GITHUB_OUTPUT"' in text
        assert 'echo "resume_loop=true" >> "$GITHUB_OUTPUT"' in text
        assert "result_status:" in text
        assert 'echo "result_status=created" >> "$GITHUB_OUTPUT"' in text
        assert 'echo "result_status=already_exists" >> "$GITHUB_OUTPUT"' in text

    def test_agentic_loop_includes_eval_results_in_context(self):
        text = _read(".github/workflows/agentic_loop.yml")
        assert "workflow_dispatch:" in text
        assert "trigger_source:" in text
        assert "trigger_details:" in text
        assert "model_results/${{ vars.MODEL_ID }}/results/${RESULT_TYPE}" in text
        assert "fetch_summaries backtest" in text
        assert "fetch_summaries eval" in text
        assert "## Last 3 eval summaries" in text
        assert "steps.leaderboard.outputs.eval_summaries" in text
        assert "Trigger: child workflow ${TRIGGER_SOURCE}" in text
        assert '"audit_logs/thoughts.md"' in text
        assert '"releases.md"' in text
        assert '"research_summary.md"' in text
        assert "Catch-up reflection complete" in text

    def test_agentic_loop_keeps_record_updates_as_a_separate_step(self):
        text = _read(".github/workflows/agentic_loop.yml")
        assert "\n      - name: Apply structured record updates\n" in text
        assert "\n          - name: Apply structured record updates\n" not in text


class TestCatchUpWorkflow:
    def test_catch_up_is_one_time_serialized_and_reflective(self):
        text = _read(".github/workflows/catch_up_eval.yml")
        assert "branches: [main]" in text
        assert "resume_after_reflection" in text
        assert "model-eval-${{ github.repository }}" in text
        assert "catchup_state.json" in text
        assert "plan_catch_up" in text
        assert "needs.evaluate.outputs.result_status" in text
        assert "trigger_source=catch_up_eval" in text
        assert "next_business_day" in text