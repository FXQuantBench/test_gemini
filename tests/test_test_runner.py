"""tests/test_test_runner.py — unit tests for test_runner.py.

Tests run without the HF dataset or Docker. External dependencies (vectorbt,
duckdb httpfs) are mocked or replaced with in-memory equivalents.

Covers:
  - _ms_from_date conversion
  - _create_view filter correctness (AC1: strict [start_ms, end_ms) window)
  - _validate_signals column validation and clamping
  - ResultSchema pydantic validation
  - _compute_metrics via mocked vectorbt
  - Timeout partial-result write
"""

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pandas as pd
import pytest

import test_runner
from test_runner import (
    HF_DATASET_GLOB,
    LOCAL_DATASET_GLOB_ENV,
    ResultSchema,
    _hf_day_parquet_path,
    _ms_from_date,
    _remote_tick_data_paths,
    _tick_data_glob,
    _validate_signals,
)


# ---------------------------------------------------------------------------
# _ms_from_date
# ---------------------------------------------------------------------------

class TestMsFromDate:
    def test_known_value(self):
        """2024-01-01 UTC midnight = 1704067200000 ms."""
        assert _ms_from_date("2024-01-01") == 1704067200000

    def test_always_utc_midnight(self):
        ms = _ms_from_date("2024-06-15")
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_consecutive_days_differ_by_86400000(self):
        d1 = _ms_from_date("2024-03-10")
        d2 = _ms_from_date("2024-03-11")
        assert d2 - d1 == 86_400_000


class TestHFDatasetGlob:
    def test_uses_nested_gbpusd_hf_layout(self):
        assert HF_DATASET_GLOB == "s3://datasets/FXQuantBench/fx-ticks/GBPUSD/*/*/*/*.parquet"


class TestHFDayParquetPaths:
    def test_formats_single_day_path(self):
        assert _hf_day_parquet_path(datetime(2024, 1, 2).date()) == (
            "s3://datasets/FXQuantBench/fx-ticks/GBPUSD/2024/01/02/ticks_2024-01-02.parquet"
        )

    def test_builds_exclusive_end_window(self):
        assert _remote_tick_data_paths("2024-01-01", "2024-01-03") == [
            "s3://datasets/FXQuantBench/fx-ticks/GBPUSD/2024/01/01/ticks_2024-01-01.parquet",
            "s3://datasets/FXQuantBench/fx-ticks/GBPUSD/2024/01/02/ticks_2024-01-02.parquet",
        ]


class TestTickDataSourceSelection:
    def test_defaults_to_hf_dataset_glob(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCAL_DATASET_GLOB_ENV, None)
            assert _tick_data_glob() == HF_DATASET_GLOB

    def test_local_override_takes_precedence(self):
        with patch.dict(os.environ, {LOCAL_DATASET_GLOB_ENV: "/input/*.parquet"}, clear=False):
            assert _tick_data_glob() == "/input/*.parquet"

    def test_build_connection_allows_local_override_without_hf_token(self):
        with patch.dict(os.environ, {LOCAL_DATASET_GLOB_ENV: "/input/*.parquet"}, clear=False):
            conn = test_runner._build_connection("")
            try:
                assert conn.execute("SELECT 1").fetchone() == (1,)
            finally:
                conn.close()

    def test_build_connection_uses_path_style_for_hf_endpoint(self):
        fake_conn = MagicMock()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCAL_DATASET_GLOB_ENV, None)
            with patch("test_runner.duckdb.connect", return_value=fake_conn):
                result = test_runner._build_connection("hf-test-token")

        assert result is fake_conn
        executed_sql = [call.args[0] for call in fake_conn.execute.call_args_list]
        assert "SET s3_endpoint='huggingface.co';" in executed_sql
        assert "SET s3_url_style='path';" in executed_sql


# ---------------------------------------------------------------------------
# AC1 — view filter strictly excludes rows at or beyond end_ms
# ---------------------------------------------------------------------------

class TestCreateView:
    """Tests for _create_view using an in-memory DuckDB with synthetic tick data."""

    START = "2024-01-01"
    END   = "2024-01-02"

    def _build_conn(self) -> duckdb.DuckDBPyConnection:
        """Return an in-memory conn with a raw_ticks table patched into HF_DATASET."""
        conn = duckdb.connect()
        start_ms = _ms_from_date(self.START)
        end_ms   = _ms_from_date(self.END)

        # Insert: one row inside window, one exactly at end_ms, one after
        conn.execute("""
            CREATE TABLE raw_ticks (
                timestamp_utc BIGINT,
                bid           DOUBLE,
                ask           DOUBLE,
                bid_volume    DOUBLE,
                ask_volume    DOUBLE
            )
        """)
        conn.execute(f"""
            INSERT INTO raw_ticks VALUES
              ({start_ms + 1000}, 1.27, 1.2701, 1000, 1000),
              ({end_ms},          1.28, 1.2801, 1000, 1000),
              ({end_ms + 1000},   1.29, 1.2901, 1000, 1000)
        """)
        # Apply the same filter logic as _create_view without touching HF
        conn.execute(f"""
            CREATE OR REPLACE VIEW GBPUSD AS
            SELECT * FROM raw_ticks
            WHERE timestamp_utc >= {_ms_from_date(self.START)}
              AND timestamp_utc <  {_ms_from_date(self.END)}
        """)
        return conn

    def test_row_inside_window_is_accessible(self):
        conn = self._build_conn()
        rows = conn.execute("SELECT * FROM GBPUSD").fetchall()
        assert len(rows) == 1

    def test_row_at_end_ms_is_excluded(self):
        """AC1: A row with timestamp_utc == end_ms must not appear in the view."""
        conn = self._build_conn()
        end_ms = _ms_from_date(self.END)
        rows = conn.execute("SELECT * FROM GBPUSD").fetchall()
        timestamps = [r[0] for r in rows]
        assert end_ms not in timestamps

    def test_row_after_end_ms_is_excluded(self):
        conn = self._build_conn()
        end_ms = _ms_from_date(self.END)
        rows = conn.execute("SELECT * FROM GBPUSD").fetchall()
        assert all(r[0] < end_ms for r in rows)

    def test_view_uses_documented_tick_columns(self):
        conn = self._build_conn()
        rows = conn.execute("PRAGMA table_info('GBPUSD')").fetchall()
        columns = [row[1] for row in rows]
        assert columns == ["timestamp_utc", "bid", "ask", "bid_volume", "ask_volume"]

    def test_empty_window_returns_zero_rows(self):
        """A window where start == end must return no rows."""
        conn = duckdb.connect()
        test_runner._create_view(conn, "2024-01-01", "2024-01-01")
        assert conn.execute("SELECT COUNT(*) FROM GBPUSD").fetchone()[0] == 0

    def test_create_view_uses_explicit_remote_day_paths(self):
        conn = MagicMock()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCAL_DATASET_GLOB_ENV, None)
            test_runner._create_view(conn, "2024-01-01", "2024-01-03")

        sql = conn.execute.call_args.args[0]
        assert "ticks_2024-01-01.parquet" in sql
        assert "ticks_2024-01-02.parquet" in sql
        assert HF_DATASET_GLOB not in sql

    def test_create_view_reads_from_local_override_glob(self):
        conn = duckdb.connect()
        start_ms = _ms_from_date(self.START)
        end_ms = _ms_from_date(self.END)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_dir = Path(tmpdir)
            parquet_glob = f"{parquet_dir.as_posix()}/*.parquet"
            parquet_path = parquet_dir / "ticks_2024-01-01.parquet"

            conn.execute("""
                CREATE TABLE raw_ticks (
                    timestamp_utc BIGINT,
                    bid           DOUBLE,
                    ask           DOUBLE,
                    bid_volume    DOUBLE,
                    ask_volume    DOUBLE
                )
            """)
            conn.execute(f"""
                INSERT INTO raw_ticks VALUES
                  ({start_ms + 1000}, 1.27, 1.2701, 1000, 1000),
                  ({end_ms},          1.28, 1.2801, 1000, 1000),
                  ({end_ms + 1000},   1.29, 1.2901, 1000, 1000)
            """)
            conn.execute(f"COPY raw_ticks TO '{parquet_path.as_posix()}' (FORMAT PARQUET)")
            conn.execute("DROP TABLE raw_ticks")

            with patch.dict(os.environ, {LOCAL_DATASET_GLOB_ENV: parquet_glob}, clear=False):
                test_runner._create_view(conn, self.START, self.END)

            rows = conn.execute("SELECT * FROM GBPUSD ORDER BY timestamp_utc").fetchall()
            assert len(rows) == 1
            assert rows[0][0] == start_ms + 1000

    def test_create_view_projects_only_documented_columns_from_parquet(self):
        conn = duckdb.connect()

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_dir = Path(tmpdir)
            parquet_glob = f"{parquet_dir.as_posix()}/*.parquet"
            parquet_path = parquet_dir / "ticks_2024-01-01.parquet"

            conn.execute("""
                CREATE TABLE raw_ticks (
                    timestamp_utc BIGINT,
                    bid DOUBLE,
                    ask DOUBLE,
                    bid_volume DOUBLE,
                    ask_volume DOUBLE,
                    is_interpolated BOOLEAN
                )
            """)
            conn.execute(f"""
                INSERT INTO raw_ticks VALUES
                  ({_ms_from_date(self.START) + 1000}, 1.27, 1.2701, 1000, 1000, TRUE)
            """)
            conn.execute(f"COPY raw_ticks TO '{parquet_path.as_posix()}' (FORMAT PARQUET)")
            conn.execute("DROP TABLE raw_ticks")

            with patch.dict(os.environ, {LOCAL_DATASET_GLOB_ENV: parquet_glob}, clear=False):
                test_runner._create_view(conn, self.START, self.END)

            columns = [row[1] for row in conn.execute("PRAGMA table_info('GBPUSD')").fetchall()]
            assert columns == ["timestamp_utc", "bid", "ask", "bid_volume", "ask_volume"]


# ---------------------------------------------------------------------------
# _validate_signals
# ---------------------------------------------------------------------------

class TestValidateSignals:
    def _make_df(self, **overrides) -> pd.DataFrame:
        data = {"timestamp_utc": [1704067200000], "pair": ["GBPUSD"], "signal": [0.5]}
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid_df_passes(self):
        df = self._make_df()
        result = _validate_signals(df)
        assert list(result.columns) == ["timestamp_utc", "pair", "signal"]

    def test_signal_clamped_above_1(self):
        df = self._make_df(signal=[1.5])
        result = _validate_signals(df)
        assert result["signal"].iloc[0] == pytest.approx(1.0)

    def test_signal_clamped_below_minus1(self):
        df = self._make_df(signal=[-2.0])
        result = _validate_signals(df)
        assert result["signal"].iloc[0] == pytest.approx(-1.0)

    def test_signal_within_range_unchanged(self):
        df = self._make_df(signal=[-0.75])
        result = _validate_signals(df)
        assert result["signal"].iloc[0] == pytest.approx(-0.75)

    def test_timestamp_utc_cast_to_int64(self):
        df = self._make_df(timestamp_utc=[1704067200000.0])
        result = _validate_signals(df)
        assert result["timestamp_utc"].dtype == np.int64

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp_utc": [0], "pair": ["GBPUSD"]})  # missing signal
        with pytest.raises(ValueError, match="missing columns"):
            _validate_signals(df)

    def test_extra_columns_preserved(self):
        df = self._make_df()
        df["extra"] = 99
        result = _validate_signals(df)
        assert "extra" in result.columns

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame(columns=["timestamp_utc", "pair", "signal"])
        result = _validate_signals(df)
        assert result.empty


# ---------------------------------------------------------------------------
# ResultSchema
# ---------------------------------------------------------------------------

def _valid_result(**overrides) -> dict:
    base = {
        "run_id": "2024-01-01-abc1234",
        "mode": "backtest",
        "model_id": "test-model",
        "strategy_sha": "a" * 40,
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "sharpe": 1.23,
        "max_drawdown": 0.05,
        "win_rate": 0.55,
        "calmar_ratio": 2.0,
        "annualized_return": 0.12,
        "volatility": 0.08,
        "total_trades": 42,
        "avg_spread_cost_pips": 1.5,
        "runtime_seconds": 30.0,
        "completed_at": "2024-01-01T12:00:00+00:00",
        "timed_out": False,
    }
    base.update(overrides)
    return base


class TestResultSchema:
    def test_valid_result_passes(self):
        r = ResultSchema(**_valid_result())
        assert r.sharpe == pytest.approx(1.23)

    def test_all_17_fields_required(self):
        for field in _valid_result().keys():
            data = _valid_result()
            del data[field]
            with pytest.raises(Exception):
                ResultSchema(**data)

    def test_nan_sharpe_coerced_to_zero(self):
        r = ResultSchema(**_valid_result(sharpe=float("nan")))
        assert r.sharpe == 0.0

    def test_inf_max_drawdown_coerced_to_zero(self):
        r = ResultSchema(**_valid_result(max_drawdown=float("inf")))
        assert r.max_drawdown == 0.0

    def test_timed_out_false_by_default(self):
        r = ResultSchema(**_valid_result(timed_out=False))
        assert r.timed_out is False

    def test_serialises_to_json(self):
        r = ResultSchema(**_valid_result())
        j = json.loads(json.dumps(r.model_dump()))
        assert j["run_id"] == "2024-01-01-abc1234"


# ---------------------------------------------------------------------------
# _compute_metrics (vectorbt mocked)
# ---------------------------------------------------------------------------

def _make_tick_df(n: int = 5, start_ms: int = 1704067200000) -> pd.DataFrame:
    """Minimal synthetic tick DataFrame."""
    timestamps = [start_ms + i * 60_000 for i in range(n)]
    bid = [1.2700 + i * 0.0001 for i in range(n)]
    ask = [b + 0.0002 for b in bid]
    return pd.DataFrame({
        "timestamp_utc": timestamps,
        "pair": ["GBPUSD"] * n,
        "bid": bid,
        "ask": ask,
        "bid_volume": [1000.0] * n,
        "ask_volume": [1000.0] * n,
    })


def _make_signal_df(n: int = 5, start_ms: int = 1704067200000, value: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_utc": [start_ms + i * 60_000 for i in range(n)],
        "pair": ["GBPUSD"] * n,
        "signal": [value] * n,
    })


def _mock_portfolio(sharpe=1.5, max_dd=0.1, total_ret=0.15, n_trades=10, win_rate=0.6):
    """Build a MagicMock that mimics the vbt.Portfolio API."""
    pf = MagicMock()
    pf.sharpe_ratio.return_value = sharpe
    pf.max_drawdown.return_value = max_dd
    pf.total_return.return_value = total_ret
    pf.trades.count.return_value = n_trades
    # trades.records_readable with a pnl column
    trades_df = pd.DataFrame({"PnL": [10.0, -5.0, 8.0, 3.0, -2.0,
                                       9.0, -1.0, 4.0, 6.0, -3.0]})
    pf.trades.records_readable = trades_df
    # returns() — a small Series of per-bar returns
    pf.returns.return_value = pd.Series([0.001, -0.0005, 0.0008, 0.0002, -0.0003])
    return pf


class TestComputeMetrics:
    def test_empty_signals_returns_zeros(self):
        empty_sig = pd.DataFrame(columns=["timestamp_utc", "pair", "signal"])
        result = test_runner._compute_metrics(empty_sig, _make_tick_df())
        assert result["total_trades"] == 0
        assert result["sharpe"] == 0.0

    def test_empty_ticks_returns_zeros(self):
        empty_tick = pd.DataFrame(columns=["timestamp_utc", "pair", "bid", "ask",
                                            "bid_volume", "ask_volume"])
        result = test_runner._compute_metrics(_make_signal_df(), empty_tick)
        assert result["total_trades"] == 0

    @patch.object(test_runner, "vbt")
    def test_metrics_forwarded_from_portfolio(self, mock_vbt):
        pf = _mock_portfolio(sharpe=2.0, max_dd=0.08, total_ret=0.20, n_trades=5)
        mock_vbt.Portfolio.from_orders.return_value = pf

        result = test_runner._compute_metrics(_make_signal_df(), _make_tick_df())

        assert result["sharpe"] == pytest.approx(2.0)
        assert result["max_drawdown"] == pytest.approx(0.08)
        assert result["total_trades"] == 5

    @patch.object(test_runner, "vbt")
    def test_win_rate_computed_from_trades(self, mock_vbt):
        pf = _mock_portfolio(win_rate=0.6)
        mock_vbt.Portfolio.from_orders.return_value = pf

        result = test_runner._compute_metrics(_make_signal_df(), _make_tick_df())
        # 6 positive pnl out of 10 non-zero = 0.6
        assert result["win_rate"] == pytest.approx(0.6, abs=0.01)

    @patch.object(test_runner, "vbt")
    def test_nan_inf_values_coerced_to_zero(self, mock_vbt):
        pf = _mock_portfolio()
        pf.sharpe_ratio.return_value = float("nan")
        pf.max_drawdown.return_value = float("inf")
        mock_vbt.Portfolio.from_orders.return_value = pf

        result = test_runner._compute_metrics(_make_signal_df(), _make_tick_df())
        assert result["sharpe"] == 0.0
        assert result["max_drawdown"] == 0.0

    @patch.object(test_runner, "vbt")
    def test_from_orders_called_with_targetpercent(self, mock_vbt):
        pf = _mock_portfolio()
        mock_vbt.Portfolio.from_orders.return_value = pf

        test_runner._compute_metrics(_make_signal_df(value=0.75), _make_tick_df())

        call_kwargs = mock_vbt.Portfolio.from_orders.call_args
        assert call_kwargs.kwargs.get("size_type") == "targetpercent"

    @patch.object(test_runner, "vbt")
    def test_spread_cost_passed_as_fees(self, mock_vbt):
        pf = _mock_portfolio()
        mock_vbt.Portfolio.from_orders.return_value = pf

        ticks = _make_tick_df()
        test_runner._compute_metrics(_make_signal_df(), ticks)

        call_kwargs = mock_vbt.Portfolio.from_orders.call_args
        fees_arg = call_kwargs.kwargs.get("fees")
        # fees series should have same length as tick data
        assert fees_arg is not None
        assert len(fees_arg) == len(ticks)

    @patch.object(test_runner, "vbt")
    def test_avg_spread_pips_nonzero_when_position_changes(self, mock_vbt):
        pf = _mock_portfolio(n_trades=2)
        mock_vbt.Portfolio.from_orders.return_value = pf

        # Signal that changes: 0.5 then 0.0
        sig = pd.DataFrame({
            "timestamp_utc": [1704067200000, 1704067260000,
                               1704067320000, 1704067380000, 1704067440000],
            "pair": ["GBPUSD"] * 5,
            "signal": [0.5, 0.5, 0.0, 0.0, 0.0],
        })
        result = test_runner._compute_metrics(sig, _make_tick_df())
        assert result["avg_spread_cost_pips"] >= 0.0


# ---------------------------------------------------------------------------
# Timeout — partial result written on SIGALRM
# ---------------------------------------------------------------------------

class TestTimeoutResult:
    def test_partial_timeout_result_has_timed_out_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OUTPUT_DIR": tmpdir}):
                test_runner._partial_timeout_result(
                    run_id="test-run",
                    mode="backtest",
                    model_id="m",
                    strategy_sha="s",
                    start_date="2024-01-01",
                    end_date="2024-02-01",
                    t_start=0.0,
                )
                result_path = Path(tmpdir) / "result.json"
                assert result_path.exists()
                data = json.loads(result_path.read_text())
                assert data["timed_out"] is True
                assert data["run_id"] == "test-run"

    def test_partial_timeout_result_validates_against_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OUTPUT_DIR": tmpdir}):
                test_runner._partial_timeout_result(
                    run_id="tr",
                    mode="eval",
                    model_id="m",
                    strategy_sha="s",
                    start_date="2024-01-01",
                    end_date="2024-01-02",
                    t_start=0.0,
                )
                data = json.loads((Path(tmpdir) / "result.json").read_text())
                # Must not raise
                ResultSchema(**data)


# ---------------------------------------------------------------------------
# EDA mode — stdout/stderr captured to EDA_OUTPUT
# ---------------------------------------------------------------------------

class TestEDAMode:
    def test_stdout_captured_to_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "eda_test.py"
            output = Path(tmpdir) / "eda.log"
            script.write_text("print('hello from eda')\n")

            conn = duckdb.connect()
            with patch.dict(os.environ, {
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }):
                test_runner.run_eda(conn)

            assert output.exists()
            assert "hello from eda" in output.read_text()

    def test_stderr_captured_to_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "eda_err.py"
            output = Path(tmpdir) / "eda.log"
            script.write_text("import sys; print('err line', file=sys.stderr)\n")

            conn = duckdb.connect()
            with patch.dict(os.environ, {
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }):
                test_runner.run_eda(conn)

            assert "err line" in output.read_text()

    def test_conn_and_pairs_injected_into_namespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "eda_ns.py"
            output = Path(tmpdir) / "eda.log"
            # Script that verifies it received conn and pairs
            script.write_text(
                "import sys\n"
                "print(type(conn).__name__)\n"
                "print(pairs)\n"
            )

            conn = duckdb.connect()
            with patch.dict(os.environ, {
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }):
                test_runner.run_eda(conn)

            text = output.read_text()
            assert "DuckDBPyConnection" in text
            assert "GBPUSD" in text

    def test_injected_conn_can_query_precreated_gbpusd_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_dir = Path(tmpdir) / "ticks"
            parquet_dir.mkdir()
            parquet_path = parquet_dir / "ticks_2024-01-01.parquet"
            output = Path(tmpdir) / "eda.log"
            script = Path(tmpdir) / "eda_view.py"

            builder = duckdb.connect()
            builder.execute("""
                CREATE TABLE raw_ticks (
                    timestamp_utc BIGINT,
                    bid           DOUBLE,
                    ask           DOUBLE,
                    bid_volume    DOUBLE,
                    ask_volume    DOUBLE
                )
            """)
            builder.execute(f"""
                INSERT INTO raw_ticks VALUES
                  ({_ms_from_date('2024-01-01') + 1000}, 1.27, 1.2701, 1000, 1000)
            """)
            builder.execute(f"COPY raw_ticks TO '{parquet_path.as_posix()}' (FORMAT PARQUET)")
            builder.close()

            script.write_text(
                "rows = conn.execute(\"SELECT COUNT(*) FROM GBPUSD\").fetchone()[0]\n"
                "print(f'rows={rows}')\n"
                "sample = conn.execute(\"SELECT timestamp_utc, bid, ask FROM GBPUSD ORDER BY timestamp_utc LIMIT 1\").df()\n"
                "print(sample.to_string(index=False))\n"
            )

            conn = duckdb.connect()
            with patch.dict(os.environ, {
                LOCAL_DATASET_GLOB_ENV: f"{parquet_dir.as_posix()}/*.parquet",
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }, clear=False):
                test_runner._create_view(conn, "2024-01-01", "2024-01-02")
                test_runner.run_eda(conn)

            text = output.read_text()
            assert "rows=1" in text
            assert "timestamp_utc" in text

    def test_duckdb_sql_default_connection_cannot_see_injected_gbpusd_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "eda_bad_duckdb_sql.py"
            output = Path(tmpdir) / "eda.log"
            script.write_text(
                "import duckdb\n"
                "try:\n"
                "    duckdb.sql(\"SELECT COUNT(*) FROM GBPUSD\").fetchone()\n"
                "except Exception as exc:\n"
                "    print(type(exc).__name__)\n"
                "    print(exc)\n"
            )

            conn = duckdb.connect()
            conn.execute("""
                CREATE OR REPLACE VIEW GBPUSD AS
                SELECT
                    1704067200000::BIGINT AS timestamp_utc,
                    1.27::DOUBLE AS bid,
                    1.2701::DOUBLE AS ask,
                    1000.0::DOUBLE AS bid_volume,
                    1000.0::DOUBLE AS ask_volume
            """)

            with patch.dict(os.environ, {
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }):
                test_runner.run_eda(conn)

            text = output.read_text()
            assert "CatalogException" in text or "Catalog Error" in text
            assert "GBPUSD" in text

    def test_main_guarded_script_executes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "eda_main.py"
            output = Path(tmpdir) / "eda.log"
            script.write_text(
                "if __name__ == '__main__':\n"
                "    print(__file__.endswith('eda_main.py'))\n"
                "    print('main guard ran')\n"
            )

            conn = duckdb.connect()
            with patch.dict(os.environ, {
                "EDA_SCRIPT": str(script),
                "EDA_OUTPUT": str(output),
            }):
                test_runner.run_eda(conn)

            text = output.read_text()
            assert "True" in text
            assert "main guard ran" in text
