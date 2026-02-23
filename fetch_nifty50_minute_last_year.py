import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance
DEFAULT_INTERVAL = "1m"
DEFAULT_YEARS = 1

DEFAULT_CHUNK_DAYS_BY_INTERVAL = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 180,
    "90m": 180,
    "1h": 180,
}


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


def _default_chunk_days(interval: str) -> int:
    return DEFAULT_CHUNK_DAYS_BY_INTERVAL.get(interval, 30)


def fetch_intraday_nifty(
    ticker: str,
    years: int,
    interval: str,
    chunk_days: int,
    timezone: str,
    retries: int,
    retry_sleep_secs: float,
) -> pd.DataFrame:
    if years <= 0:
        raise ValueError("--years must be >= 1")

    if interval.endswith("m"):
        logging.warning(
            "Yahoo Finance intraday retention is limited. For 1m data, you typically only get recent days."
        )

    # Crawl backwards from now, stop if we hit repeated empty chunks.
    end_utc = (pd.Timestamp.utcnow().floor("min") + pd.Timedelta(days=1)).tz_localize(None)
    start_utc = (end_utc - pd.DateOffset(years=years)).tz_localize(None)

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
        raise RuntimeError(
            "No data returned. Yahoo Finance often limits intraday history (especially for 1m)."
        )

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


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("GCS destination must start with gs://")
    path = uri[len("gs://") :]
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    if not bucket or not blob:
        raise ValueError("GCS destination must include bucket and object path (gs://bucket/path)")
    return bucket, blob


def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    try:
        from google.cloud import storage
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-storage is required for upload. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NIFTY 50 intraday data (default 1m) for the last N years."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="How many years of history")
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="yfinance interval (e.g. 1m). Intraday retention limits may apply.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=None,
        help="Chunk size in days for repeated downloads (defaults based on interval).",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Kolkata",
        help="Timezone to convert timestamps to (default Asia/Kolkata).",
    )
    parser.add_argument(
        "--output",
        default="nifty50_minute_last_year.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-file",
        default="logs/nifty50_minute_fetch.log",
        help="Log file path",
    )
    parser.add_argument(
        "--gcs-dest",
        default=None,
        help="Optional GCS destination (e.g. gs://bucket/path.csv).",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-secs", type=float, default=2.0, help="Sleep between retries")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    chunk_days = args.chunk_days or _default_chunk_days(args.interval)

    log_path = Path(args.log_file)
    configure_logging(log_path, args.verbose)

    logging.info("Job started")
    logging.info(
        "Parameters | ticker=%s | years=%s | interval=%s | chunk_days=%s | output=%s",
        args.ticker,
        args.years,
        args.interval,
        chunk_days,
        Path(args.output).resolve(),
    )

    data = fetch_intraday_nifty(
        ticker=args.ticker,
        years=args.years,
        interval=args.interval,
        chunk_days=chunk_days,
        timezone=args.timezone,
        retries=args.retries,
        retry_sleep_secs=args.retry_sleep_secs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    logging.info("Saved %s rows to: %s", len(data), output_path.resolve())

    if args.gcs_dest:
        logging.info("Uploading to GCS: %s", args.gcs_dest)
        upload_to_gcs(output_path, args.gcs_dest)
        logging.info("Upload complete")

    logging.info("Job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Job failed")
        raise
