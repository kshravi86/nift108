import argparse
import logging
from pathlib import Path

import pandas as pd

from generate_nakshatra_for_nifty_hourly import (
    DEFAULT_TZ,
    MUMBAI_LAT,
    MUMBAI_LON,
    add_planet_nakshatras,
    configure_logging,
    load_hourly_nifty,
)


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
        bigquery.SchemaField("Place", "STRING"),
        bigquery.SchemaField("Latitude", "FLOAT"),
        bigquery.SchemaField("Longitude", "FLOAT"),
        bigquery.SchemaField("Sun_Longitude", "FLOAT"),
        bigquery.SchemaField("Sun_Nakshatra", "STRING"),
        bigquery.SchemaField("Sun_Pada", "INTEGER"),
        bigquery.SchemaField("Moon_Longitude", "FLOAT"),
        bigquery.SchemaField("Moon_Nakshatra", "STRING"),
        bigquery.SchemaField("Moon_Pada", "INTEGER"),
        bigquery.SchemaField("Mars_Longitude", "FLOAT"),
        bigquery.SchemaField("Mars_Nakshatra", "STRING"),
        bigquery.SchemaField("Mars_Pada", "INTEGER"),
        bigquery.SchemaField("Mercury_Longitude", "FLOAT"),
        bigquery.SchemaField("Mercury_Nakshatra", "STRING"),
        bigquery.SchemaField("Mercury_Pada", "INTEGER"),
        bigquery.SchemaField("Jupiter_Longitude", "FLOAT"),
        bigquery.SchemaField("Jupiter_Nakshatra", "STRING"),
        bigquery.SchemaField("Jupiter_Pada", "INTEGER"),
        bigquery.SchemaField("Venus_Longitude", "FLOAT"),
        bigquery.SchemaField("Venus_Nakshatra", "STRING"),
        bigquery.SchemaField("Venus_Pada", "INTEGER"),
        bigquery.SchemaField("Saturn_Longitude", "FLOAT"),
        bigquery.SchemaField("Saturn_Nakshatra", "STRING"),
        bigquery.SchemaField("Saturn_Pada", "INTEGER"),
        bigquery.SchemaField("Rahu_Longitude", "FLOAT"),
        bigquery.SchemaField("Rahu_Nakshatra", "STRING"),
        bigquery.SchemaField("Rahu_Pada", "INTEGER"),
        bigquery.SchemaField("Ketu_Longitude", "FLOAT"),
        bigquery.SchemaField("Ketu_Nakshatra", "STRING"),
        bigquery.SchemaField("Ketu_Pada", "INTEGER"),
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
    p = argparse.ArgumentParser(
        description="Generate 1m NIFTY 50 astro features and load into BigQuery."
    )
    p.add_argument(
        "--nifty-csv",
        default="data/nifty50_1m_last_30d.csv",
        help="Input NIFTY 1m CSV path (must include a Datetime column).",
    )
    p.add_argument(
        "--datetime-col",
        default="Datetime",
        help="Datetime column name in input CSV.",
    )
    p.add_argument(
        "--input-tz",
        default=DEFAULT_TZ,
        help="Timezone to assume if input datetimes are naive.",
    )
    p.add_argument(
        "--output",
        default="data/nifty50_1m_astro_last30d.csv",
        help="Output CSV path.",
    )
    p.add_argument("--latitude", type=float, default=MUMBAI_LAT, help="Latitude (Mumbai default).")
    p.add_argument("--longitude", type=float, default=MUMBAI_LON, help="Longitude (Mumbai default).")
    p.add_argument("--place", default="Mumbai", help="Place name for metadata.")
    p.add_argument("--timezone", default=DEFAULT_TZ, help="Timezone name for metadata.")
    p.add_argument(
        "--no-longitudes",
        action="store_true",
        help="Exclude sidereal longitudes in the output (Nakshatra/Pada only).",
    )
    p.add_argument("--log-every", type=int, default=500, help="Log progress every N rows (0 disables).")
    p.add_argument("--log-file", default="logs/nifty50_1m_astro.log", help="Log file path.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    p.add_argument(
        "--bq-table",
        default="nift108.nift108_ds.nifty50_1m_astro_yahoo_last30d",
        help="BigQuery table (project.dataset.table).",
    )
    p.add_argument(
        "--bq-write-disposition",
        default="WRITE_TRUNCATE",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
        help="BigQuery write disposition.",
    )
    args = p.parse_args()

    configure_logging(Path(args.log_file), args.verbose)
    logging.info("1m Astro job started")
    logging.info(
        "Parameters | nifty_csv=%s | output=%s | datetime_col=%s | tz=%s | place=%s | bq_table=%s",
        args.nifty_csv,
        args.output,
        args.datetime_col,
        args.timezone,
        args.place,
        args.bq_table,
    )

    nifty_df = load_hourly_nifty(Path(args.nifty_csv), args.datetime_col, args.input_tz)
    logging.info("Loaded %s rows from %s", len(nifty_df), Path(args.nifty_csv).resolve())

    if "Timezone" in nifty_df.columns:
        nifty_df = nifty_df.drop(columns=["Timezone"])

    out = add_planet_nakshatras(
        df=nifty_df,
        datetime_col=args.datetime_col,
        latitude=args.latitude,
        longitude=args.longitude,
        place=args.place,
        timezone_name=args.timezone,
        include_longitudes=not args.no_longitudes,
        log_every=args.log_every,
    )

    dt = pd.to_datetime(out[args.datetime_col])
    dt = dt.dt.tz_convert(args.timezone).dt.tz_localize(None)
    out[args.datetime_col] = dt
    out["Date"] = dt.dt.date
    out["Time"] = dt.dt.time
    out["Timezone"] = args.timezone

    ordered_cols = [
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
        "Place",
        "Latitude",
        "Longitude",
        "Sun_Longitude",
        "Sun_Nakshatra",
        "Sun_Pada",
        "Moon_Longitude",
        "Moon_Nakshatra",
        "Moon_Pada",
        "Mars_Longitude",
        "Mars_Nakshatra",
        "Mars_Pada",
        "Mercury_Longitude",
        "Mercury_Nakshatra",
        "Mercury_Pada",
        "Jupiter_Longitude",
        "Jupiter_Nakshatra",
        "Jupiter_Pada",
        "Venus_Longitude",
        "Venus_Nakshatra",
        "Venus_Pada",
        "Saturn_Longitude",
        "Saturn_Nakshatra",
        "Saturn_Pada",
        "Rahu_Longitude",
        "Rahu_Nakshatra",
        "Rahu_Pada",
        "Ketu_Longitude",
        "Ketu_Nakshatra",
        "Ketu_Pada",
    ]

    missing_cols = [col for col in ordered_cols if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns before export: {missing_cols}")

    out = out[ordered_cols]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logging.info("Saved %s rows to: %s", len(out), out_path.resolve())
    logging.info(
        "Datetime range | start=%s | end=%s",
        out[args.datetime_col].min(),
        out[args.datetime_col].max(),
    )

    logging.info("Loading to BigQuery: %s", args.bq_table)
    load_csv_to_bq(out_path, args.bq_table, args.bq_write_disposition)
    logging.info("1m Astro job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("1m Astro job failed")
        raise
