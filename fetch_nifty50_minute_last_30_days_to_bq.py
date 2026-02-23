import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance
DEFAULT_INTERVAL = "1m"
DEFAULT_DAYS = 30
DEFAULT_TIMEZONE = "Asia/Kolkata"


def configure_logging(log_file: Path, verbose: bool = False) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logging.basicConfig(level=level, handlers=[file_handler, stream_handler])


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def _ensure_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    try:
        if index.tz is None:
            index = index.tz_localize("UTC")
        return index.tz_convert(tz)
    except Exception as exc:
        logging.warning("Timezone conversion failed (%s). Keeping original index. Error=%s", tz, exc)
        return index


def _download_chunk(
    ticker: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int,
    retry_sleep_secs: float,
) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            return _flatten_yf_columns(df)
        except Exception as exc:
            last_err = exc
            logging.warning(
                "Chunk download failed (attempt %s/%s). start=%s end=%s err=%s",
                attempt,
                retries,
                start,
                end,
                exc,
            )
            time.sleep(retry_sleep_secs)
    raise RuntimeError(f"Failed to download chunk after {retries} attempts: {last_err}")


def fetch_intraday_nifty(
    ticker: str,
    days: int,
    interval: str,
    chunk_days: int,
    timezone: str,
    retries: int,
    retry_sleep_secs: float,
) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("--days must be >= 1")

    logging.warning(
        "Yahoo Finance 1m data is limited to recent days (typically ~30 days). Older data may be unavailable."
    )

    end_utc = (pd.Timestamp.utcnow().floor("min") + pd.Timedelta(days=1)).tz_localize(None)
    start_utc = (end_utc - pd.Timedelta(days=days)).tz_localize(None)

    logging.info("Requested date range (UTC) | start=%s end=%s", start_utc, end_utc)
    logging.info("Interval=%s | chunk_days=%s", interval, chunk_days)

    chunks: list[pd.DataFrame] = []

    cursor_end = end_utc
    empty_streak = 0
    while cursor_end > start_utc:
        cursor_start = max(cursor_end - pd.Timedelta(days=chunk_days), start_utc)
        logging.info("Downloading chunk | start=%s end=%s", cursor_start, cursor_end)

        df = _download_chunk(
            ticker=ticker,
            interval=interval,
            start=cursor_start,
            end=cursor_end,
            retries=retries,
            retry_sleep_secs=retry_sleep_secs,
        )

        if df.empty:
            empty_streak += 1
            logging.warning(
                "No data returned for chunk | start=%s end=%s (empty_streak=%s)",
                cursor_start,
                cursor_end,
                empty_streak,
            )
            if chunks and empty_streak >= 3:
                logging.warning(
                    "Stopping early due to repeated empty chunks after having data (likely source retention limit)."
                )
                break
        else:
            empty_streak = 0
            chunks.append(df)

        cursor_end = cursor_start

    if not chunks:
        raise RuntimeError("No data returned. Yahoo Finance intraday history is limited.")

    out = pd.concat(chunks).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    out.index = _ensure_tz(pd.to_datetime(out.index), timezone)
    out.index.name = "Datetime"
    out = out.reset_index()

    actual_start = out["Datetime"].min()
    actual_end = out["Datetime"].max()
    requested_start_local = _ensure_tz(pd.DatetimeIndex([start_utc]), timezone)[0]
    requested_end_local = _ensure_tz(pd.DatetimeIndex([end_utc]), timezone)[0]
    if actual_start > requested_start_local:
        logging.warning(
            "Returned data starts later than requested. Requested_start=%s Actual_start=%s (likely source retention).",
            requested_start_local,
            actual_start,
        )
    logging.info("Returned date range (%s) | start=%s end=%s", timezone, actual_start, actual_end)

    return out


def _prepare_dataframe(df: pd.DataFrame, timezone: str, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)

    dt = pd.to_datetime(df["Datetime"])
    dt = _ensure_tz(pd.DatetimeIndex(dt), timezone)
    dt = dt.tz_localize(None)
    df["Datetime"] = dt
    df["Date"] = df["Datetime"].dt.date
    df["Time"] = df["Datetime"].dt.time
    df["Timezone"] = timezone
    df["Ticker"] = ticker

    columns = [
        "Datetime",
        "Date",
        "Time",
        "Timezone",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]
    return df[columns]


def _parse_bq_table(table_ref: str) -> str:
    parts = table_ref.split(".")
    if len(parts) != 3:
        raise ValueError("--bq-table must be in the form project.dataset.table")
    return table_ref


def load_csv_to_bq(csv_path: Path, table_ref: str, write_disposition: str) -> None:
    try:
        from google.cloud import bigquery
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-bigquery is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    table_ref = _parse_bq_table(table_ref)
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("Datetime", "DATETIME"),
        bigquery.SchemaField("Date", "DATE"),
        bigquery.SchemaField("Time", "TIME"),
        bigquery.SchemaField("Timezone", "STRING"),
        bigquery.SchemaField("Ticker", "STRING"),
        bigquery.SchemaField("Open", "FLOAT"),
        bigquery.SchemaField("High", "FLOAT"),
        bigquery.SchemaField("Low", "FLOAT"),
        bigquery.SchemaField("Close", "FLOAT"),
        bigquery.SchemaField("Adj_Close", "FLOAT"),
        bigquery.SchemaField("Volume", "INTEGER"),
    ]

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=write_disposition,
        schema=schema,
    )

    with csv_path.open("rb") as handle:
        job = client.load_table_from_file(handle, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    logging.info("Loaded %s rows into %s", table.num_rows, table_ref)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NIFTY 50 1m data (last 30 days) and load into BigQuery."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="How many days of history")
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="yfinance interval (default 1m).",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Chunk size in days for repeated downloads (default 7).",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="Timezone to convert timestamps to (default Asia/Kolkata).",
    )
    parser.add_argument(
        "--output",
        default="data/nifty50_1m_last_30d.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-file",
        default="logs/nifty50_1m_fetch.log",
        help="Log file path",
    )
    parser.add_argument(
        "--bq-table",
        default="nift108.nift108_ds.nifty50_1m_yahoo_last30d",
        help="BigQuery table (project.dataset.table).",
    )
    parser.add_argument(
        "--bq-write-disposition",
        default="WRITE_TRUNCATE",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
        help="BigQuery write disposition.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-secs", type=float, default=2.0, help="Sleep between retries")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    configure_logging(log_path, args.verbose)

    logging.info("Job started")
    logging.info(
        "Parameters | ticker=%s | days=%s | interval=%s | output=%s | bq_table=%s",
        args.ticker,
        args.days,
        args.interval,
        Path(args.output).resolve(),
        args.bq_table,
    )

    data = fetch_intraday_nifty(
        ticker=args.ticker,
        days=args.days,
        interval=args.interval,
        chunk_days=args.chunk_days,
        timezone=args.timezone,
        retries=args.retries,
        retry_sleep_secs=args.retry_sleep_secs,
    )

    data = _prepare_dataframe(data, args.timezone, args.ticker)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logging.info("Saved %s rows to: %s", len(data), output_path.resolve())

    logging.info("Loading to BigQuery: %s", args.bq_table)
    load_csv_to_bq(output_path, args.bq_table, args.bq_write_disposition)
    logging.info("Job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Job failed")
        raise
