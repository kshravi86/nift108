import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance
DEFAULT_INTERVAL = "60m"  # hourly


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
    except Exception as e:
        logging.warning("Timezone conversion failed (%s). Keeping original index. Error=%s", tz, e)
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
        except Exception as e:
            last_err = e
            logging.warning(
                "Chunk download failed (attempt %s/%s). start=%s end=%s err=%s",
                attempt,
                retries,
                start,
                end,
                e,
            )
            time.sleep(retry_sleep_secs)
    raise RuntimeError(f"Failed to download chunk after {retries} attempts: {last_err}")


def fetch_hourly_nifty(
    ticker: str,
    years: int,
    interval: str,
    chunk_days: int,
    timezone: str,
    retries: int,
    retry_sleep_secs: float,
) -> pd.DataFrame:
    # Use UTC boundaries for consistent chunking.
    end_utc = pd.Timestamp.utcnow() + pd.Timedelta(days=1)
    start_utc = end_utc - pd.DateOffset(years=years)

    logging.info("Requested date range (UTC) | start=%s end=%s", start_utc, end_utc)
    logging.info("Interval=%s | chunk_days=%s", interval, chunk_days)

    chunks: list[pd.DataFrame] = []

    cursor = start_utc
    empty_streak = 0
    while cursor < end_utc:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_utc)
        logging.info("Downloading chunk | start=%s end=%s", cursor, chunk_end)

        df = _download_chunk(
            ticker=ticker,
            interval=interval,
            start=cursor,
            end=chunk_end,
            retries=retries,
            retry_sleep_secs=retry_sleep_secs,
        )

        if df.empty:
            empty_streak += 1
            logging.warning("No data returned for chunk | start=%s end=%s", cursor, chunk_end)
            # Yahoo often returns empty for intraday ranges outside their retention window.
            # If we get repeated empties, stop early.
            if empty_streak >= 3:
                logging.warning("Stopping early due to repeated empty chunks (likely retention limit).")
                break
        else:
            empty_streak = 0
            chunks.append(df)

        cursor = chunk_end

    if not chunks:
        raise RuntimeError(
            "No data returned. Yahoo Finance often limits intraday history (e.g. ~730 days for 60m)."
        )

    out = pd.concat(chunks).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    out.index = _ensure_tz(pd.to_datetime(out.index), timezone)
    out.index.name = "Datetime"
    out = out.reset_index()

    # Report coverage vs request.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NIFTY 50 hourly (60m) OHLCV data for the last N years (with chunking)."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker")
    parser.add_argument("--years", type=int, default=3, help="How many years of history")
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="yfinance interval (e.g. 60m). Note: intraday retention limits may apply.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=180,
        help="Chunk size in days for repeated downloads (helps reliability).",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Kolkata",
        help="Timezone to convert timestamps to (default Asia/Kolkata).",
    )
    parser.add_argument(
        "--output",
        default="nifty50_hourly_last_3_years.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-file",
        default="logs/nifty50_hourly_fetch.log",
        help="Log file path",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-secs", type=float, default=2.0, help="Sleep between retries")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    configure_logging(log_path, args.verbose)

    logging.info("Job started")
    logging.info(
        "Parameters | ticker=%s | years=%s | interval=%s | output=%s",
        args.ticker,
        args.years,
        args.interval,
        Path(args.output).resolve(),
    )

    data = fetch_hourly_nifty(
        ticker=args.ticker,
        years=args.years,
        interval=args.interval,
        chunk_days=args.chunk_days,
        timezone=args.timezone,
        retries=args.retries,
        retry_sleep_secs=args.retry_sleep_secs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    logging.info("Saved %s rows to: %s", len(data), output_path.resolve())
    logging.info("Job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Job failed")
        raise

