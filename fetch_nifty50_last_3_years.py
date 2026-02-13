import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance


def fetch_nifty_data(ticker: str, years: int) -> pd.DataFrame:
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
    args = parser.parse_args()

    data = fetch_nifty_data(args.ticker, args.years)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Saved {len(data)} rows to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
