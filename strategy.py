"""strategy.py — baseline do-nothing strategy (agent-writable)."""

import pandas as pd


def run(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Return trading signals for the given date window.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Connection with the GBPUSD view pre-loaded.
    start_date : str
        Inclusive start date "YYYY-MM-DD".
    end_date : str
        Exclusive end date "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_utc (int64), pair (str), signal (float ∈ [-1, 1]).
        Empty DataFrame is valid and means "flat / no position".
    """
    return pd.DataFrame(
        columns=["timestamp_utc", "pair", "signal"]
    ).astype({"timestamp_utc": "int64", "pair": "str", "signal": "float64"})
