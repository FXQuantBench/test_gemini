"""tests/test_strategy.py — tests for the baseline strategy."""

import numpy as np
import pandas as pd
import pytest

from strategy import run


class TestBaselineStrategy:
    def _conn(self):
        """Return an in-memory DuckDB connection with an empty GBPUSD view."""
        import duckdb
        conn = duckdb.connect()
        conn.execute("""
            CREATE VIEW GBPUSD AS
            SELECT 0::BIGINT AS timestamp_utc,
                   1.27      AS bid,
                   1.2701    AS ask,
                   1000.0    AS bid_volume,
                   1000.0    AS ask_volume
            WHERE 1 = 0
        """)
        return conn

    def test_returns_dataframe(self):
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        assert set(df.columns) >= {"timestamp_utc", "pair", "signal"}

    def test_baseline_returns_empty(self):
        """The baseline do-nothing strategy must return zero rows."""
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        assert len(df) == 0

    def test_timestamp_utc_is_int64(self):
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        assert df["timestamp_utc"].dtype == np.int64

    def test_pair_is_object_or_str(self):
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        dtype_str = str(df["pair"].dtype)
        assert df["pair"].dtype == object or dtype_str in ("string", "str")

    def test_signal_is_float64(self):
        df = run(self._conn(), "2024-01-01", "2024-02-01")
        assert df["signal"].dtype == np.float64

    def test_signature_accepts_conn_start_end(self):
        """run() must accept exactly (conn, start_date, end_date)."""
        import inspect
        sig = inspect.signature(run)
        params = list(sig.parameters.keys())
        assert params == ["conn", "start_date", "end_date"]
