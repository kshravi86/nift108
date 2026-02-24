import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


NAKSHATRA_LORDS = {
    "Ashwini": "Ketu",
    "Bharani": "Venus",
    "Krittika": "Sun",
    "Rohini": "Moon",
    "Mrigashirsha": "Mars",
    "Ardra": "Rahu",
    "Punarvasu": "Jupiter",
    "Pushya": "Saturn",
    "Ashlesha": "Mercury",
    "Magha": "Ketu",
    "Purva Phalguni": "Venus",
    "Uttara Phalguni": "Sun",
    "Hasta": "Moon",
    "Chitra": "Mars",
    "Swati": "Rahu",
    "Vishakha": "Jupiter",
    "Anuradha": "Saturn",
    "Jyeshtha": "Mercury",
    "Mula": "Ketu",
    "Purva Ashadha": "Venus",
    "Uttara Ashadha": "Sun",
    "Shravana": "Moon",
    "Dhanishta": "Mars",
    "Shatabhisha": "Rahu",
    "Purva Bhadrapada": "Jupiter",
    "Uttara Bhadrapada": "Saturn",
    "Revati": "Mercury",
}


def parse_date(value: str) -> date:
    try:
        return pd.to_datetime(value).date()
    except Exception as exc:
        raise SystemExit(f"Invalid --date '{value}'. Use YYYY-MM-DD.") from exc


def load_data(table: str) -> pd.DataFrame:
    client = bigquery.Client()
    query = f"""
        SELECT
            Datetime,
            Date,
            Time,
            Ticker,
            Open,
            High,
            Low,
            Close,
            Moon_Longitude,
            Moon_Nakshatra,
            Moon_Pada
        FROM `{table}`
    """
    return client.query(query).to_dataframe()


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Date", "Datetime"]).copy()
    df["prev_close"] = df.groupby("Date")["Close"].shift(1)
    df["delta"] = df["Close"] - df["prev_close"]
    df["pct_delta"] = df["delta"] / df["prev_close"]
    df["direction"] = pd.cut(
        df["delta"],
        bins=[-float("inf"), -1e-9, 1e-9, float("inf")],
        labels=["DOWN", "FLAT", "UP"],
    )
    return df


def add_moon_lord(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Moon_Lord"] = df["Moon_Nakshatra"].map(NAKSHATRA_LORDS).fillna("Unknown")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["Moon_Nakshatra", "Moon_Lord"])
        .agg(
            minutes=("Datetime", "count"),
            avg_delta=("delta", "mean"),
            avg_pct_delta=("pct_delta", "mean"),
            up_rate=("direction", lambda s: (s == "UP").mean()),
            down_rate=("direction", lambda s: (s == "DOWN").mean()),
            flat_rate=("direction", lambda s: (s == "FLAT").mean()),
        )
        .reset_index()
        .sort_values(["minutes", "Moon_Nakshatra"], ascending=[False, True])
    )
    return out


def load_csv_to_bq(csv_path: Path, table_ref: str, write_disposition: str) -> None:
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("Datetime", "DATETIME"),
        bigquery.SchemaField("Date", "DATE"),
        bigquery.SchemaField("Time", "TIME"),
        bigquery.SchemaField("Ticker", "STRING"),
        bigquery.SchemaField("Open", "FLOAT"),
        bigquery.SchemaField("High", "FLOAT"),
        bigquery.SchemaField("Low", "FLOAT"),
        bigquery.SchemaField("Close", "FLOAT"),
        bigquery.SchemaField("delta", "FLOAT"),
        bigquery.SchemaField("pct_delta", "FLOAT"),
        bigquery.SchemaField("direction", "STRING"),
        bigquery.SchemaField("Moon_Longitude", "FLOAT"),
        bigquery.SchemaField("Moon_Nakshatra", "STRING"),
        bigquery.SchemaField("Moon_Pada", "INTEGER"),
        bigquery.SchemaField("Moon_Lord", "STRING"),
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
    print(f"Loaded {table.num_rows} rows into {table_ref}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze Moon Nakshatra/Lord patterns vs NIFTY 1m moves."
    )
    p.add_argument("--date", required=True, help="Date to analyze (YYYY-MM-DD).")
    p.add_argument(
        "--table",
        default="nift108.nift108_ds.nifty50_1m_yahoo_last30d_with_astro",
        help="BigQuery table to use.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory for CSVs.",
    )
    p.add_argument(
        "--bq-out-table",
        default=None,
        help="BigQuery table for per-minute output (project.dataset.table).",
    )
    p.add_argument(
        "--bq-write-disposition",
        default="WRITE_TRUNCATE",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
        help="BigQuery write disposition.",
    )
    args = p.parse_args()

    target_date = parse_date(args.date)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.bq_out_table:
        date_tag = target_date.strftime("%Y%m%d")
        args.bq_out_table = f"nift108.nift108_ds.nifty50_1m_moon_pattern_{date_tag}"

    df = load_data(args.table)
    df = add_returns(df)
    df = add_moon_lord(df)

    df_day = df[df["Date"] == target_date].copy()
    if df_day.empty:
        raise SystemExit(f"No rows found for date {target_date} in {args.table}.")

    df_day["Datetime"] = pd.to_datetime(df_day["Datetime"])
    df_day["Date"] = df_day["Datetime"].dt.date
    df_day["Time"] = df_day["Datetime"].dt.time

    per_minute_cols = [
        "Datetime",
        "Date",
        "Time",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "delta",
        "pct_delta",
        "direction",
        "Moon_Longitude",
        "Moon_Nakshatra",
        "Moon_Pada",
        "Moon_Lord",
    ]
    per_minute = df_day[per_minute_cols]

    per_minute_path = out_dir / f"nifty_moon_pattern_{target_date}.csv"
    per_minute.to_csv(per_minute_path, index=False)

    summary_day = summarize(df_day)
    summary_path = out_dir / f"nifty_moon_pattern_{target_date}_summary.csv"
    summary_day.to_csv(summary_path, index=False)

    print(f"Saved per-minute rows: {per_minute_path}")
    print(f"Saved date summary: {summary_path}")
    print("\nTop Moon Nakshatra/Lord for the date:")
    print(summary_day.head(10).to_string(index=False))

    load_csv_to_bq(per_minute_path, args.bq_out_table, args.bq_write_disposition)


if __name__ == "__main__":
    main()
