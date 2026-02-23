import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import swisseph as swe

MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777
MUMBAI_ALT_M = 14.0
DEFAULT_TZ = "Asia/Kolkata"

# 27 Nakshatras in order from 0 Aries onward.
NAKSHATRAS = [
    "Ashwini",
    "Bharani",
    "Krittika",
    "Rohini",
    "Mrigashirsha",
    "Ardra",
    "Punarvasu",
    "Pushya",
    "Ashlesha",
    "Magha",
    "Purva Phalguni",
    "Uttara Phalguni",
    "Hasta",
    "Chitra",
    "Swati",
    "Vishakha",
    "Anuradha",
    "Jyeshtha",
    "Mula",
    "Purva Ashadha",
    "Uttara Ashadha",
    "Shravana",
    "Dhanishta",
    "Shatabhisha",
    "Purva Bhadrapada",
    "Uttara Bhadrapada",
    "Revati",
]

PLANETS = [
    ("Sun", swe.SUN),
    ("Moon", swe.MOON),
    ("Mars", swe.MARS),
    ("Mercury", swe.MERCURY),
    ("Jupiter", swe.JUPITER),
    ("Venus", swe.VENUS),
    ("Saturn", swe.SATURN),
    ("Rahu", swe.MEAN_NODE),
]


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


def longitude_to_nakshatra(longitude_deg: float) -> tuple[str, int]:
    longitude_deg = longitude_deg % 360.0
    nak_len = 360.0 / 27.0
    pada_len = nak_len / 4.0

    nak_index = int(longitude_deg // nak_len)
    nak_name = NAKSHATRAS[nak_index]
    pada = int((longitude_deg % nak_len) // pada_len) + 1
    return nak_name, pada


def calc_sidereal_longitude(jd_ut: float, planet_id: int, flags: int) -> float:
    xx, _ = swe.calc_ut(jd_ut, planet_id, flags)
    return xx[0] % 360.0


def datetime_to_jd_ut(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    hour_ut = (
        dt_utc.hour
        + dt_utc.minute / 60.0
        + dt_utc.second / 3600.0
        + dt_utc.microsecond / 3_600_000_000.0
    )
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour_ut, swe.GREG_CAL)


def load_hourly_nifty(nifty_csv: Path, datetime_col: str, input_tz: str) -> pd.DataFrame:
    if not nifty_csv.exists():
        raise FileNotFoundError(f"NIFTY csv not found: {nifty_csv}")

    df = pd.read_csv(nifty_csv)
    if datetime_col not in df.columns:
        raise ValueError(f"Input CSV must contain '{datetime_col}' column.")

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    if dt.isna().all():
        raise ValueError(f"No valid datetimes parsed from column '{datetime_col}'.")

    tz = ZoneInfo(input_tz)
    if getattr(dt.dt, "tz", None) is None:
        dt_local = dt.dt.tz_localize(tz)
    else:
        dt_local = dt.dt.tz_convert(tz)

    df[datetime_col] = dt_local
    df = df.dropna(subset=[datetime_col]).copy()
    return df


def add_planet_nakshatras(
    df: pd.DataFrame,
    datetime_col: str,
    latitude: float,
    longitude: float,
    place: str,
    timezone_name: str,
    include_longitudes: bool,
    log_every: int,
) -> pd.DataFrame:
    tz = ZoneInfo(timezone_name)

    swe.set_sid_mode(swe.SIDM_LAHIRI, 0.0, 0.0)
    swe.set_topo(longitude, latitude, MUMBAI_ALT_M)
    flags = swe.FLG_MOSEPH | swe.FLG_SIDEREAL | swe.FLG_TOPOCTR

    dts_local = pd.to_datetime(df[datetime_col]).dt.tz_convert(tz)
    dts_utc = dts_local.dt.tz_convert("UTC")

    out_rows = []
    total = len(df)
    iter_dts = dts_utc.dt.to_pydatetime() if hasattr(dts_utc, "dt") else dts_utc.to_pydatetime()
    for i, dt_utc in enumerate(iter_dts, start=1):
        jd_ut = datetime_to_jd_ut(dt_utc)

        row = {
            "Place": place,
            "Latitude": latitude,
            "Longitude": longitude,
            "Timezone": timezone_name,
        }

        for planet_name, planet_id in PLANETS:
            plon = calc_sidereal_longitude(jd_ut, planet_id, flags)
            nak, pada = longitude_to_nakshatra(plon)
            if include_longitudes:
                row[f"{planet_name}_Longitude"] = round(plon, 6)
            row[f"{planet_name}_Nakshatra"] = nak
            row[f"{planet_name}_Pada"] = pada

        # Ketu opposite Rahu by 180 degrees.
        rahu_lon = row["Rahu_Longitude"] if include_longitudes else calc_sidereal_longitude(jd_ut, swe.MEAN_NODE, flags)
        ketu_lon = (float(rahu_lon) + 180.0) % 360.0
        ketu_nak, ketu_pada = longitude_to_nakshatra(ketu_lon)
        if include_longitudes:
            row["Ketu_Longitude"] = round(ketu_lon, 6)
        row["Ketu_Nakshatra"] = ketu_nak
        row["Ketu_Pada"] = ketu_pada

        out_rows.append(row)

        if log_every > 0 and (i % log_every == 0 or i == total):
            logging.info("Processed %s/%s timestamps", i, total)

    astro_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), astro_df], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate planet Nakshatra/Pada for each timestamp in NIFTY hourly CSV "
            "(Mumbai, Lahiri ayanamsha)."
        )
    )
    p.add_argument(
        "--nifty-csv",
        default="nifty50_hourly_last_3_years.csv",
        help="Input NIFTY hourly CSV path (must include a Datetime column).",
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
        default="nifty50_hourly_with_nakshatra_mumbai_lahiri.csv",
        help="Output CSV path.",
    )
    p.add_argument("--latitude", type=float, default=MUMBAI_LAT, help="Latitude (Mumbai default).")
    p.add_argument("--longitude", type=float, default=MUMBAI_LON, help="Longitude (Mumbai default).")
    p.add_argument("--place", default="Mumbai", help="Place name for metadata.")
    p.add_argument("--timezone", default=DEFAULT_TZ, help="Timezone name for metadata.")
    p.add_argument(
        "--include-longitudes",
        action="store_true",
        help="Include sidereal longitudes in the output (adds columns).",
    )
    p.add_argument("--log-every", type=int, default=500, help="Log progress every N rows (0 disables).")
    p.add_argument("--log-file", default="logs/nifty50_hourly_nakshatra.log", help="Log file path.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = p.parse_args()

    configure_logging(Path(args.log_file), args.verbose)
    logging.info("Hourly Nakshatra job started")
    logging.info(
        "Parameters | nifty_csv=%s | output=%s | datetime_col=%s | tz=%s | place=%s",
        args.nifty_csv,
        args.output,
        args.datetime_col,
        args.timezone,
        args.place,
    )

    nifty_df = load_hourly_nifty(Path(args.nifty_csv), args.datetime_col, args.input_tz)
    logging.info("Loaded %s rows from %s", len(nifty_df), Path(args.nifty_csv).resolve())

    out = add_planet_nakshatras(
        df=nifty_df,
        datetime_col=args.datetime_col,
        latitude=args.latitude,
        longitude=args.longitude,
        place=args.place,
        timezone_name=args.timezone,
        include_longitudes=args.include_longitudes,
        log_every=args.log_every,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logging.info("Saved %s rows to: %s", len(out), out_path.resolve())
    logging.info(
        "Datetime range | start=%s | end=%s",
        out[args.datetime_col].min(),
        out[args.datetime_col].max(),
    )
    logging.info("Hourly Nakshatra job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Hourly Nakshatra job failed")
        raise
