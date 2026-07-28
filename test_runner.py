"""
test_runner.py — read-only execution engine (protected by CODEOWNERS).

Environment variables
---------------------
HF_TOKEN            : HuggingFace read-only token for dataset access
TICK_DATA_GLOB      : optional local parquet glob mounted into the runner container
MODE                : "eda" | "backtest" | "eval"
EDA_SCRIPT          : path to EDA script (MODE=eda only)
EDA_OUTPUT          : path where EDA stdout/stderr are written (MODE=eda only)
BACKTEST_LOG_OUTPUT : path where backtest stdout/stderr are written (MODE=backtest/eval only)
START_DATE          : ISO date string "YYYY-MM-DD"  (backtest/eval)
END_DATE            : ISO date string "YYYY-MM-DD"  (backtest/eval; exclusive upper bound)
RUN_ID              : run identifier string (backtest/eval)
MODEL_ID            : model identifier (backtest/eval)
OUTPUT_DIR          : directory where result.json is written (backtest/eval)
STRATEGY_SHA        : git SHA of strategy.py at time of run
"""

import contextlib
import io
import json
import math
import os
import signal
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import vectorbt as vbt

# ---------------------------------------------------------------------------
# Result validation helpers — all fields required, no nulls
# ---------------------------------------------------------------------------

RESULT_FLOAT_FIELDS = (
    "sharpe",
    "max_drawdown",
    "win_rate",
    "calmar_ratio",
    "annualized_return",
    "volatility",
    "avg_spread_cost_pips",
    "runtime_seconds",
)

RESULT_REQUIRED_FIELDS = {
    "run_id",
    "mode",
    "model_id",
    "strategy_sha",
    "start_date",
    "end_date",
    *RESULT_FLOAT_FIELDS,
    "total_trades",
    "completed_at",
    "timed_out",
}

RESULT_FIELD_ORDER = (
    "run_id",
    "mode",
    "model_id",
    "strategy_sha",
    "start_date",
    "end_date",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "calmar_ratio",
    "annualized_return",
    "volatility",
    "total_trades",
    "avg_spread_cost_pips",
    "runtime_seconds",
    "completed_at",
    "timed_out",
)


def _normalize_float(value: Any) -> float:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def _validate_result(payload: dict[str, Any]) -> dict[str, Any]:
    missing = RESULT_REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Result payload missing fields: {sorted(missing)}")

    normalized = dict(payload)
    for field in RESULT_FLOAT_FIELDS:
        normalized[field] = _normalize_float(normalized[field])

    normalized["total_trades"] = int(normalized["total_trades"])
    normalized["timed_out"] = bool(normalized["timed_out"])
    return normalized


class ResultSchema:
    def __init__(self, **payload: Any) -> None:
        normalized = _validate_result(payload)
        for field in RESULT_FIELD_ORDER:
            setattr(self, field, normalized[field])

    def model_dump(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in RESULT_FIELD_ORDER}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 300  # 5 minutes
HF_DATASET_ROOT = "s3://datasets/FXQuantBench/fx-ticks/GBPUSD"
HF_DATASET_GLOB = f"{HF_DATASET_ROOT}/*/*/*/*.parquet"
LOCAL_DATASET_GLOB_ENV = "TICK_DATA_GLOB"


def _tick_data_glob() -> str:
    return os.environ.get(LOCAL_DATASET_GLOB_ENV, HF_DATASET_GLOB)


def _using_local_tick_data() -> bool:
    return bool(os.environ.get(LOCAL_DATASET_GLOB_ENV))


def _sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def _sql_string_list_literal(values: list[str]) -> str:
    return "[" + ", ".join(f"'{_sql_string_literal(value)}'" for value in values) + "]"


def _ms_from_date(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to Unix milliseconds (midnight UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _hf_day_parquet_path(day: date) -> str:
    return f"{HF_DATASET_ROOT}/{day:%Y/%m/%d}/ticks_{day.isoformat()}.parquet"


def _remote_tick_data_paths(start_date: str, end_date: str) -> list[str]:
    window_start = date.fromisoformat(start_date)
    window_end = date.fromisoformat(end_date)
    paths: list[str] = []

    day = window_start
    while day < window_end:
        paths.append(_hf_day_parquet_path(day))
        day += timedelta(days=1)

    return paths


def _build_connection(hf_token: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    if _using_local_tick_data():
        return conn
    if not hf_token:
        raise RuntimeError(
            f"HF_TOKEN must be set when {LOCAL_DATASET_GLOB_ENV} is not provided"
        )
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"SET s3_endpoint='huggingface.co';")
    conn.execute("SET s3_url_style='path';")
    conn.execute(f"SET s3_access_key_id='user';")
    conn.execute(f"SET s3_secret_access_key='{hf_token}';")
    conn.execute(f"SET s3_session_token='';")
    try:
        conn.execute(f"SET hf_token='{hf_token}';")
    except duckdb.Error as exc:
        if 'unrecognized configuration parameter "hf_token"' not in str(exc):
            raise
    return conn


def _create_view(conn: duckdb.DuckDBPyConnection, start_date: str, end_date: str) -> None:
    """Create a view GBPUSD filtered strictly to [start_date, end_date).

    No row with timestamp_utc outside [start_ms, end_ms) is accessible.
    """
    start_ms = _ms_from_date(start_date)
    end_ms = _ms_from_date(end_date)

    if start_ms >= end_ms:
        conn.execute("""
            CREATE OR REPLACE VIEW GBPUSD AS
            SELECT
                CAST(NULL AS BIGINT) AS timestamp_utc,
                CAST(NULL AS DOUBLE) AS bid,
                CAST(NULL AS DOUBLE) AS ask,
                CAST(NULL AS DOUBLE) AS bid_volume,
                CAST(NULL AS DOUBLE) AS ask_volume
            WHERE 1 = 0
        """)
        return

    if _using_local_tick_data():
        dataset_source_sql = f"'{_sql_string_literal(_tick_data_glob())}'"
    else:
        dataset_source_sql = _sql_string_list_literal(
            _remote_tick_data_paths(start_date, end_date)
        )

    try:
        conn.execute(f"""
            CREATE OR REPLACE VIEW GBPUSD AS
            SELECT
                timestamp_utc,
                bid,
                ask,
                bid_volume,
                ask_volume
            FROM read_parquet({dataset_source_sql})
            WHERE timestamp_utc >= {start_ms}
              AND timestamp_utc < {end_ms}
        """)
    except duckdb.Error as exc:
        if _using_local_tick_data():
            raise
        raise RuntimeError(
            "Failed to load GBPUSD ticks from Hugging Face. "
            "Set TICK_DATA_GLOB to a mounted local parquet glob or verify HF_TOKEN "
            "and DuckDB httpfs/S3 access."
        ) from exc


def _output_dir() -> Path:
    d = Path(os.environ.get("OUTPUT_DIR", "/output"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_result(payload: dict[str, Any]) -> None:
    out = _output_dir() / "result.json"
    out.write_text(json.dumps(payload, indent=2))


def _partial_timeout_result(run_id: str, mode: str, model_id: str,
                             strategy_sha: str, start_date: str, end_date: str,
                             t_start: float) -> None:
    payload = {
        "run_id": run_id,
        "mode": mode,
        "model_id": model_id,
        "strategy_sha": strategy_sha,
        "start_date": start_date,
        "end_date": end_date,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "calmar_ratio": 0.0,
        "annualized_return": 0.0,
        "volatility": 0.0,
        "total_trades": 0,
        "avg_spread_cost_pips": 0.0,
        "runtime_seconds": round(time.time() - t_start, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "timed_out": True,
    }
    _write_result(payload)


def _write_eda_failure_log(message: str) -> None:
    eda_output = Path(os.environ.get("EDA_OUTPUT", "/output/eda.log"))
    eda_output.parent.mkdir(parents=True, exist_ok=True)
    eda_output.write_text(message)


def _write_backtest_failure_log(message: str) -> None:
    backtest_log = Path(os.environ.get("BACKTEST_LOG_OUTPUT", "/output/backtest.log"))
    backtest_log.parent.mkdir(parents=True, exist_ok=True)
    backtest_log.write_text(message)


def _handle_failure(run_id: str, mode: str, model_id: str,
                    strategy_sha: str, start_date: str, end_date: str,
                    t_start: float, error_text: str | None = None) -> None:
    if mode == "eda":
        message = error_text or f"ERROR: EDA runner failed after {round(time.time() - t_start, 3)}s.\n"
        _write_eda_failure_log(message)
        return

    # backtest / eval: write the failure log AND a zeroed-out result.json
    message = error_text or f"ERROR: {mode} runner failed after {round(time.time() - t_start, 3)}s.\n"
    _write_backtest_failure_log(message)
    _partial_timeout_result(run_id, mode, model_id, strategy_sha,
                            start_date, end_date, t_start)


# ---------------------------------------------------------------------------
# EDA mode
# ---------------------------------------------------------------------------

def run_eda(conn: duckdb.DuckDBPyConnection) -> None:
    eda_script = os.environ["EDA_SCRIPT"]
    eda_output = os.environ["EDA_OUTPUT"]

    Path(eda_output).parent.mkdir(parents=True, exist_ok=True)
    script_globals = {
        "__name__": "__main__",
        "__file__": eda_script,
        "conn": conn,
        "pairs": ["GBPUSD"],
    }

    with open(eda_output, "w") as out_file:
        with contextlib.redirect_stdout(out_file), contextlib.redirect_stderr(out_file):
            with open(eda_script, "r") as f:
                source = f.read()
            exec(compile(source, eda_script, "exec"), script_globals)  # noqa: S102


# ---------------------------------------------------------------------------
# Backtest / Eval mode
# ---------------------------------------------------------------------------

def _validate_signals(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_utc", "pair", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"strategy.run() returned DataFrame missing columns: {missing}")

    # Enforce dtypes
    df = df.copy()
    df["timestamp_utc"] = df["timestamp_utc"].astype(np.int64)
    df["pair"] = df["pair"].astype(str)
    df["signal"] = df["signal"].astype(float)

    # Clamp signal to [-1, 1]
    df["signal"] = df["signal"].clip(-1.0, 1.0)

    return df


def _compute_metrics(signals: pd.DataFrame, tick_data: pd.DataFrame,
                     initial_capital: float = 10_000.0) -> dict[str, Any]:
    """
    Simulate using vectorbt Portfolio.from_orders with size_type='targetpercent'.

    Signal semantics (per brief):
      +1.0 = 100% long, -1.0 = 100% short, 0.0 = flat.
      When signal changes, vectorbt closes the current position and opens a new
      one sized to abs(signal) * capital.  Spread cost (ask - bid) is applied
      as a per-order fee on every open and close.
    """
    _ZERO = {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "calmar_ratio": 0.0,
        "annualized_return": 0.0,
        "volatility": 0.0,
        "total_trades": 0,
        "avg_spread_cost_pips": 0.0,
    }

    if signals.empty or tick_data.empty:
        return _ZERO

    # Merge signals onto tick data (backward fill)
    tick = tick_data.sort_values("timestamp_utc").reset_index(drop=True)
    sig = signals.sort_values("timestamp_utc").reset_index(drop=True)

    merged = pd.merge_asof(
        tick,
        sig[["timestamp_utc", "signal"]],
        on="timestamp_utc",
        direction="backward",
    )
    merged["signal"] = merged["signal"].fillna(0.0)

    mid = (merged["bid"] + merged["ask"]) / 2.0
    spread = merged["ask"] - merged["bid"]

    # vectorbt requires a DatetimeIndex
    idx = pd.to_datetime(merged["timestamp_utc"], unit="ms", utc=True)
    close_s = pd.Series(mid.values, index=idx, name="close")
    size_s  = pd.Series(merged["signal"].values, index=idx, name="size")
    # Fee = spread / mid price (fraction of trade value), applied on every order
    fees_s  = pd.Series(
        (spread / mid.replace(0, np.nan).fillna(1.0)).values,
        index=idx,
        name="fees",
    )

    try:
        pf = vbt.Portfolio.from_orders(
            close=close_s,
            size=size_s,
            size_type="targetpercent",
            fees=fees_s,
            init_cash=initial_capital,
            freq="infer",
        )
    except Exception as exc:
        raise RuntimeError(f"vectorbt simulation failed: {exc}") from exc

    # --- Core metrics from vectorbt ---
    def _safe(val: Any, default: float = 0.0) -> float:
        try:
            v = float(val)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    sharpe  = _safe(pf.sharpe_ratio())
    max_dd  = _safe(abs(pf.max_drawdown()))
    n_trades = int(pf.trades.count())
    total_ret = _safe(pf.total_return())

    # Win rate from individual trades
    try:
        trades_df = pf.trades.records_readable
        pnl_col = next((c for c in trades_df.columns if c.lower() == "pnl"), None)
        if pnl_col and len(trades_df) > 0:
            nonzero = trades_df[pnl_col][trades_df[pnl_col] != 0]
            win_rate = _safe((nonzero > 0).mean())
        else:
            win_rate = 0.0
    except Exception:
        win_rate = 0.0

    # Annualized return over the actual date range
    duration_years = (idx.iloc[-1] - idx.iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    if duration_years > 0 and total_ret > -1:
        try:
            ann_return = _safe((1.0 + total_ret) ** (1.0 / duration_years) - 1.0)
        except OverflowError:
            ann_return = 0.0
    else:
        ann_return = 0.0

    # Annualized volatility from per-bar returns
    try:
        rets = pf.returns()
        median_ms = merged["timestamp_utc"].diff().median()
        median_period_s = max(1.0, float(median_ms) / 1000.0)
        periods_per_year = 365.25 * 24.0 * 3600.0 / median_period_s
        ann_vol = _safe(float(rets.std()) * math.sqrt(periods_per_year))
    except Exception:
        ann_vol = 0.0

    calmar = _safe(ann_return / max_dd) if max_dd != 0 else 0.0

    # Average spread cost in pips (position changes only; 1 pip = 0.0001 for GBPUSD)
    position_changed = merged["signal"].diff().fillna(0) != 0
    spread_on_changes = spread[position_changed]
    avg_spread_pips = _safe(spread_on_changes.mean() / 0.0001) if position_changed.any() else 0.0

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "calmar_ratio": calmar,
        "annualized_return": ann_return,
        "volatility": ann_vol,
        "total_trades": n_trades,
        "avg_spread_cost_pips": avg_spread_pips,
    }


def run_backtest_eval(conn: duckdb.DuckDBPyConnection, mode: str, run_id: str,
                      model_id: str, strategy_sha: str,
                      start_date: str, end_date: str, t_start: float) -> None:
    backtest_log = Path(os.environ.get("BACKTEST_LOG_OUTPUT", "/output/backtest.log"))
    backtest_log.parent.mkdir(parents=True, exist_ok=True)

    with open(backtest_log, "w") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            import strategy  # noqa: PLC0415  (intentional late import from /sandbox)

            signals_df = strategy.run(conn, start_date, end_date)

            if not isinstance(signals_df, pd.DataFrame):
                raise TypeError(f"strategy.run() must return pd.DataFrame, got {type(signals_df)}")

            signals_df = _validate_signals(signals_df)

            tick_df = conn.execute("SELECT * FROM GBPUSD ORDER BY timestamp_utc").df()

            metrics = _compute_metrics(signals_df, tick_df)

    runtime = round(time.time() - t_start, 3)
    result = ResultSchema(**{
        "run_id": run_id,
        "mode": mode,
        "model_id": model_id,
        "strategy_sha": strategy_sha,
        "start_date": start_date,
        "end_date": end_date,
        "runtime_seconds": runtime,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "timed_out": False,
        **metrics,
    })

    _write_result(result.model_dump())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    mode = os.environ.get("MODE", "backtest")
    hf_token = os.environ.get("HF_TOKEN", "")
    run_id = os.environ.get("RUN_ID", "local")
    model_id = os.environ.get("MODEL_ID", "unknown")
    strategy_sha = os.environ.get("STRATEGY_SHA", "unknown")
    start_date = os.environ.get("START_DATE", "")
    end_date = os.environ.get("END_DATE", "")

    # --- Timeout handler ---
    def _on_timeout(signum, frame):  # noqa: ARG001
        _handle_failure(run_id, mode, model_id, strategy_sha,
                        start_date, end_date, t_start,
                        f"ERROR: {mode} runner timed out after {TIMEOUT_SECONDS}s.\n")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        conn = _build_connection(hf_token)

        if mode == "eda":
            # EDA: no date filtering — script controls its own queries,
            # but the view is still created so the script can use it.
            if start_date and end_date:
                _create_view(conn, start_date, end_date)
            run_eda(conn)
        elif mode in ("backtest", "eval"):
            if not start_date or not end_date:
                raise ValueError("START_DATE and END_DATE must be set for backtest/eval mode")
            _create_view(conn, start_date, end_date)
            run_backtest_eval(conn, mode, run_id, model_id, strategy_sha,
                              start_date, end_date, t_start)
        else:
            raise ValueError(f"Unknown MODE: {mode!r}")

    except Exception:  # noqa: BLE001
        error_text = traceback.format_exc()
        print(error_text, file=sys.stderr, end="")
        _handle_failure(run_id, mode, model_id, strategy_sha,
                        start_date, end_date, t_start, error_text)
        sys.exit(1)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
