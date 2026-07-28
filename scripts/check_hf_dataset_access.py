"""Real connectivity check for the Hugging Face dataset access paths used in this repo.

Usage:
    uv run --with duckdb --with huggingface_hub --with pyarrow --with fsspec --with pandas --with numpy \
        python scripts/check_hf_dataset_access.py --start-date 2024-01-01 --end-date 2024-01-03

If `HF_TOKEN_RO` or `HF_TOKEN` is available, the script also exercises authenticated-only checks
and DuckDB remote `s3://datasets/...` access. Without a token, it still validates the public
local staging path and a real EDA script executed through `test_runner.py`.

This script exercises the same real connection/query styles used by the workflows:
  - Hugging Face Hub auth (`HfApi.auth_check`)
  - Hugging Face filesystem existence checks (`HfFileSystem.exists`)
  - PyArrow dataset queries against HF-hosted parquet files
  - `hf_hub_download` staging to local parquet files
  - DuckDB remote `s3://datasets/...` access with the same S3 settings as `test_runner.py`
  - DuckDB local parquet-glob access over the staged files
    - A real EDA script executed through `test_runner.py` using the injected `conn`
  - The prompt-documented DuckDB sample and minute-bucket queries
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import duckdb
import pyarrow.dataset as ds
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from pyarrow import compute as pc
from pyarrow import fs as pafs


HF_REPO_ID = "FXQuantBench/fx-ticks"
PAIR = "GBPUSD"
HF_DATASET_ROOT = f"s3://datasets/{HF_REPO_ID}/{PAIR}"


def _log(message: str) -> None:
    print(message, flush=True)


def _ok(message: str) -> None:
    _log(f"[PASS] {message}")


def _fail(message: str) -> None:
    _log(f"[FAIL] {message}")


def _sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def _sql_string_list_literal(values: list[str]) -> str:
    return "[" + ", ".join(f"'{_sql_string_literal(value)}'" for value in values) + "]"


def _ms_from_date(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _iter_days(start_date: str, end_date: str) -> list[date]:
    window_start = date.fromisoformat(start_date)
    window_end = date.fromisoformat(end_date)
    days: list[date] = []

    day = window_start
    while day < window_end:
        days.append(day)
        day += timedelta(days=1)

    return days


def _hf_fs_day_path(day: date) -> str:
    return f"datasets/{HF_REPO_ID}/{PAIR}/{day:%Y/%m/%d}/ticks_{day.isoformat()}.parquet"


def _hf_download_day_path(day: date) -> str:
    return f"{PAIR}/{day:%Y/%m/%d}/ticks_{day.isoformat()}.parquet"


def _hf_s3_day_path(day: date) -> str:
    return f"{HF_DATASET_ROOT}/{day:%Y/%m/%d}/ticks_{day.isoformat()}.parquet"


def _build_hf_duckdb_connection(hf_token: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_endpoint='huggingface.co';")
    conn.execute("SET s3_url_style='path';")
    conn.execute("SET s3_access_key_id='user';")
    conn.execute(f"SET s3_secret_access_key='{_sql_string_literal(hf_token)}';")
    conn.execute("SET s3_session_token='';")
    try:
        conn.execute(f"SET hf_token='{_sql_string_literal(hf_token)}';")
    except duckdb.Error as exc:
        if 'unrecognized configuration parameter "hf_token"' not in str(exc):
            raise
    return conn


def _create_remote_view(conn: duckdb.DuckDBPyConnection, start_date: str, end_date: str) -> None:
    start_ms = _ms_from_date(start_date)
    end_ms = _ms_from_date(end_date)
    dataset_source_sql = _sql_string_list_literal(
        [_hf_s3_day_path(day) for day in _iter_days(start_date, end_date)]
    )
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


def _create_local_view(
    conn: duckdb.DuckDBPyConnection, start_date: str, end_date: str, parquet_glob: str
) -> None:
    start_ms = _ms_from_date(start_date)
    end_ms = _ms_from_date(end_date)
    conn.execute(f"""
        CREATE OR REPLACE VIEW GBPUSD AS
        SELECT
            timestamp_utc,
            bid,
            ask,
            bid_volume,
            ask_volume
        FROM read_parquet('{_sql_string_literal(parquet_glob)}')
        WHERE timestamp_utc >= {start_ms}
          AND timestamp_utc < {end_ms}
    """)


def _run_repo_queries(conn: duckdb.DuckDBPyConnection, verbose: bool) -> dict[str, int]:
    columns = [row[1] for row in conn.execute("PRAGMA table_info('GBPUSD')").fetchall()]
    expected_columns = ["timestamp_utc", "bid", "ask", "bid_volume", "ask_volume"]
    if columns != expected_columns:
        raise RuntimeError(f"Unexpected GBPUSD columns: {columns}")

    total_rows = int(conn.execute("SELECT COUNT(*) FROM GBPUSD").fetchone()[0])
    if total_rows <= 0:
        raise RuntimeError("GBPUSD view returned zero rows for the requested window")

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
        LIMIT 5
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
        LIMIT 5
        """
    ).df()

    if verbose:
        _log("sample query result:")
        _log(sample.to_string(index=False))
        _log("minute bucket query result:")
        _log(minute_stats.to_string(index=False))

    return {
        "total_rows": total_rows,
        "sample_rows": len(sample),
        "minute_rows": len(minute_stats),
    }


def _load_test_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "test_runner.py"

    def _exec_module():
        spec = importlib.util.spec_from_file_location("_live_test_runner", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        return _exec_module()
    except ModuleNotFoundError as exc:
        if exc.name != "vectorbt":
            raise

        original_vectorbt = sys.modules.get("vectorbt")
        sys.modules["vectorbt"] = types.ModuleType("vectorbt")
        try:
            return _exec_module()
        finally:
            if original_vectorbt is None:
                sys.modules.pop("vectorbt", None)
            else:
                sys.modules["vectorbt"] = original_vectorbt


def _run_real_eda_runner_check(
    stage_dir: Path, start_date: str, end_date: str, verbose: bool
) -> dict[str, int]:
    test_runner = _load_test_runner_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        eda_script = tmp_path / "eda_real_query.py"
        eda_output = tmp_path / "eda.log"

        eda_script.write_text(
            dedent(
                '''
                def main(conn):
                    summary = conn.execute(
                        """
                        SELECT
                          COUNT(*) AS row_count,
                          MIN(timestamp_utc) AS min_ts,
                          MAX(timestamp_utc) AS max_ts,
                          AVG(ask - bid) AS avg_spread
                        FROM GBPUSD
                        """
                    ).df()
                    print("eda_conn_query_ok")
                    print(summary.to_string(index=False))

                    sample = conn.execute(
                        """
                        SELECT timestamp_utc, bid, ask
                        FROM GBPUSD
                        ORDER BY timestamp_utc
                        LIMIT 3
                        """
                    ).df()
                    print(sample.to_string(index=False))

                if __name__ == "__main__":
                    main(conn)
                '''
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        env_updates = {
            "MODE": "eda",
            "START_DATE": start_date,
            "END_DATE": end_date,
            "TICK_DATA_GLOB": f"{stage_dir.as_posix()}/*.parquet",
            "EDA_SCRIPT": str(eda_script),
            "EDA_OUTPUT": str(eda_output),
        }
        previous_env = {key: os.environ.get(key) for key in env_updates}

        conn = duckdb.connect()
        try:
            os.environ.update(env_updates)
            test_runner._create_view(conn, start_date, end_date)
            test_runner.run_eda(conn)
        finally:
            conn.close()
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        log_text = eda_output.read_text(encoding="utf-8") if eda_output.exists() else ""
        if "eda_conn_query_ok" not in log_text:
            raise RuntimeError(f"EDA log missing success marker: {log_text!r}")
        if "Table with name GBPUSD does not exist" in log_text:
            raise RuntimeError(f"EDA log hit missing GBPUSD view error: {log_text!r}")

        if verbose:
            _log("EDA runner log:")
            _log(log_text)

        return {
            "log_lines": len([line for line in log_text.splitlines() if line.strip()]),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real HF dataset connectivity and query checks for this repo."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date in YYYY-MM-DD format. Choose a small window you know exists.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN_RO",
        help="Environment variable name containing the HF token. Defaults to HF_TOKEN_RO.",
    )
    parser.add_argument(
        "--stage-dir",
        default="",
        help="Optional directory for downloaded parquet shards. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the DuckDB sample and minute-bucket query results.",
    )
    parser.add_argument(
        "--skip-remote-s3",
        action="store_true",
        help="Skip the DuckDB remote s3://datasets/... check and validate only the staged local path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    token = os.environ.get(args.token_env) or os.environ.get("HF_TOKEN", "")

    days = _iter_days(args.start_date, args.end_date)
    if not days:
        _fail("The requested window is empty. Use start-date < end-date.")
        return 2

    _log(
        "[INFO] Testing HF dataset access for "
        f"{args.start_date} -> {args.end_date} ({len(days)} day(s)) using {args.token_env if os.environ.get(args.token_env) else 'HF_TOKEN'}"
    )
    if not token:
        _log(
            "[INFO] No HF token found; skipping authenticated-only checks and remote DuckDB S3 access."
        )

    hf_fs_paths = [_hf_fs_day_path(day) for day in days]
    hf_s3_paths = [_hf_s3_day_path(day) for day in days]
    failures: list[tuple[str, str]] = []
    hffs: HfFileSystem | None = None
    stage_dir: Path | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    remote_stats: dict[str, int] | None = None
    local_stats: dict[str, int] | None = None
    eda_runner_stats: dict[str, int] | None = None

    try:
        if token:
            try:
                api = HfApi(token=token)
                api.auth_check(HF_REPO_ID, repo_type="dataset")
                _ok(f"HfApi.auth_check succeeded for {HF_REPO_ID}")
            except Exception as exc:  # noqa: BLE001
                failures.append(("HfApi.auth_check", str(exc)))
                _fail(f"HfApi.auth_check failed: {exc}")
        else:
            _log("[INFO] Skipping HfApi.auth_check because no token was provided.")

        try:
            hffs = HfFileSystem(token=token or None)
            for hf_fs_path in hf_fs_paths:
                if not hffs.exists(hf_fs_path):
                    raise RuntimeError(f"Missing HF shard: {hf_fs_path}")
            _ok(f"HfFileSystem.exists succeeded for {len(hf_fs_paths)} shard path(s)")
        except Exception as exc:  # noqa: BLE001
            failures.append(("HfFileSystem.exists", str(exc)))
            _fail(f"HfFileSystem.exists failed: {exc}")
            hffs = None

        if hffs is not None:
            try:
                dataset = ds.dataset(
                    hf_fs_paths,
                    filesystem=pafs.PyFileSystem(pafs.FSSpecHandler(hffs)),
                    format="parquet",
                )
                day_counts: dict[str, int] = {}
                for day in days:
                    day_start_ms = _ms_from_date(day.isoformat())
                    day_end_ms = day_start_ms + 86_400_000
                    count = int(
                        dataset.count_rows(
                            filter=(
                                (pc.field("timestamp_utc") >= day_start_ms)
                                & (pc.field("timestamp_utc") < day_end_ms)
                            )
                        )
                    )
                    if count <= 0:
                        raise RuntimeError(
                            f"HF PyArrow dataset returned zero rows for {day.isoformat()}"
                        )
                    day_counts[day.isoformat()] = count
                _ok(f"PyArrow dataset count_rows succeeded: {day_counts}")
            except Exception as exc:  # noqa: BLE001
                failures.append(("PyArrow dataset count_rows", str(exc)))
                _fail(f"PyArrow dataset count_rows failed: {exc}")

        try:
            if args.stage_dir:
                stage_dir = Path(args.stage_dir).resolve()
                stage_dir.mkdir(parents=True, exist_ok=True)
            else:
                temp_dir = tempfile.TemporaryDirectory()
                stage_dir = Path(temp_dir.name)

            for day in days:
                cached_path = Path(
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        filename=_hf_download_day_path(day),
                        token=token or None,
                    )
                )
                target_path = stage_dir / cached_path.name
                shutil.copyfile(cached_path, target_path)

            staged_files = sorted(stage_dir.glob("*.parquet"))
            if len(staged_files) != len(days):
                raise RuntimeError(
                    f"Expected {len(days)} staged parquet file(s), found {len(staged_files)}"
                )
            _ok(f"hf_hub_download succeeded and staged {len(staged_files)} parquet file(s)")
        except Exception as exc:  # noqa: BLE001
            failures.append(("hf_hub_download", str(exc)))
            _fail(f"hf_hub_download failed: {exc}")
            stage_dir = None

        if args.skip_remote_s3:
            _log("[INFO] Skipping DuckDB remote S3 access because --skip-remote-s3 was set.")
        elif token:
            try:
                remote_conn = _build_hf_duckdb_connection(token)
                try:
                    _create_remote_view(remote_conn, args.start_date, args.end_date)
                    remote_stats = _run_repo_queries(remote_conn, args.verbose)
                finally:
                    remote_conn.close()
                _ok(
                    "DuckDB remote S3 access succeeded: "
                    f"rows={remote_stats['total_rows']} sample_rows={remote_stats['sample_rows']} minute_rows={remote_stats['minute_rows']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(("DuckDB remote S3 access", str(exc)))
                _fail(f"DuckDB remote S3 access failed: {exc}")
        else:
            _log("[INFO] Skipping DuckDB remote S3 access because no token was provided.")

        if stage_dir is not None:
            try:
                local_conn = duckdb.connect()
                try:
                    parquet_glob = f"{stage_dir.as_posix()}/*.parquet"
                    _create_local_view(local_conn, args.start_date, args.end_date, parquet_glob)
                    local_stats = _run_repo_queries(local_conn, args.verbose)
                finally:
                    local_conn.close()
                _ok(
                    "DuckDB local staged parquet access succeeded: "
                    f"rows={local_stats['total_rows']} sample_rows={local_stats['sample_rows']} minute_rows={local_stats['minute_rows']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(("DuckDB local staged parquet access", str(exc)))
                _fail(f"DuckDB local staged parquet access failed: {exc}")

            try:
                eda_runner_stats = _run_real_eda_runner_check(
                    stage_dir, args.start_date, args.end_date, args.verbose
                )
                _ok(
                    "test_runner EDA injected-conn access succeeded: "
                    f"log_lines={eda_runner_stats['log_lines']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(("test_runner EDA injected-conn access", str(exc)))
                _fail(f"test_runner EDA injected-conn access failed: {exc}")

        if remote_stats is not None and local_stats is not None:
            if local_stats != remote_stats:
                message = f"Remote/local DuckDB stats differ: remote={remote_stats} local={local_stats}"
                failures.append(("DuckDB remote/local comparison", message))
                _fail(message)
            else:
                _ok("Remote and local DuckDB query summaries matched")

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    _log("[INFO] Remote S3 paths used by DuckDB:")
    for hf_s3_path in hf_s3_paths:
        _log(f"  - {hf_s3_path}")

    if failures:
        _log("[INFO] Failure summary:")
        for name, message in failures:
            _log(f"  - {name}: {message}")
        return 1

    _ok("All real HF dataset connection and query checks completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())