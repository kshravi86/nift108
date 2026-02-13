import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance


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


def fetch_nifty_data(ticker: str, years: int, full_history: bool = False) -> pd.DataFrame:
    if full_history:
        df = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    else:
        end = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(years=years)
        df = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

    if df.empty:
        raise RuntimeError(f"No data returned for ticker '{ticker}'.")

    # yfinance may return MultiIndex columns when using a ticker symbol.
    # Flatten to simple OHLCV column names for clean CSV output.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "Date"
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NIFTY 50 daily OHLCV data for the last N years."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker")
    parser.add_argument("--years", type=int, default=3, help="How many years of history")
    parser.add_argument(
        "--output",
        default="nifty50_last_3_years.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log-file",
        default="logs/nifty50_fetch.log",
        help="Log file path",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Download full available history from source (period=max).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    configure_logging(log_path, args.verbose)

    logging.info("Job started")
    logging.info(
        "Parameters | ticker=%s | years=%s | output=%s | log_file=%s",
        args.ticker,
        args.years,
        args.output,
        log_path.resolve(),
    )

    data = fetch_nifty_data(args.ticker, args.years, full_history=args.full_history)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    logging.info("Saved %s rows to: %s", len(data), output_path.resolve())
    logging.info(
        "Date range | start=%s | end=%s",
        data["Date"].min(),
        data["Date"].max(),
    )
    logging.info("Job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Job failed")
        raise
