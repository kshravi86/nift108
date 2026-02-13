import argparse
import logging
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import swisseph as swe

MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777
MUMBAI_ALT_M = 14.0
IST_ZONE = "Asia/Kolkata"

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


def load_nifty_dates(nifty_csv: Path) -> list:
    if not nifty_csv.exists():
        raise FileNotFoundError(f"NIFTY csv not found: {nifty_csv}")

    df = pd.read_csv(nifty_csv)
    if "Date" not in df.columns:
        raise ValueError("Input CSV must contain a 'Date' column.")

    dates = pd.to_datetime(df["Date"], errors="coerce").dropna().dt.date.unique().tolist()
    dates.sort()
    if not dates:
        raise ValueError("No valid dates found in input CSV.")
    return dates


def generate_nakshatra_data(
    dates: list,
    hour: int,
    minute: int,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    tz = ZoneInfo(IST_ZONE)

    # Lahiri ayanamsha for Vedic sidereal calculations.
    swe.set_sid_mode(swe.SIDM_LAHIRI, 0.0, 0.0)
    swe.set_topo(longitude, latitude, MUMBAI_ALT_M)

    # MOSEPH avoids external ephemeris files and works out-of-the-box.
    flags = swe.FLG_MOSEPH | swe.FLG_SIDEREAL | swe.FLG_TOPOCTR

    rows = []
    total = len(dates)
    for i, d in enumerate(dates, start=1):
        local_dt = datetime.combine(d, time(hour, minute), tzinfo=tz)
        utc_dt = local_dt.astimezone(timezone.utc)
        hour_ut = (
            utc_dt.hour
            + utc_dt.minute / 60.0
            + utc_dt.second / 3600.0
            + utc_dt.microsecond / 3_600_000_000.0
        )
        jd_ut = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, hour_ut, swe.GREG_CAL)

        row = {
            "Date": d.isoformat(),
            "Time": f"{hour:02d}:{minute:02d}:00",
            "Timezone": IST_ZONE,
            "Place": "Mumbai",
            "Latitude": latitude,
            "Longitude": longitude,
        }

        for planet_name, planet_id in PLANETS:
            plon = calc_sidereal_longitude(jd_ut, planet_id, flags)
            nak, pada = longitude_to_nakshatra(plon)
            row[f"{planet_name}_Longitude"] = round(plon, 6)
            row[f"{planet_name}_Nakshatra"] = nak
            row[f"{planet_name}_Pada"] = pada

        # Ketu opposite Rahu by 180 degrees.
        ketu_lon = (row["Rahu_Longitude"] + 180.0) % 360.0
        ketu_nak, ketu_pada = longitude_to_nakshatra(ketu_lon)
        row["Ketu_Longitude"] = round(ketu_lon, 6)
        row["Ketu_Nakshatra"] = ketu_nak
        row["Ketu_Pada"] = ketu_pada

        rows.append(row)

        if i % 500 == 0 or i == total:
            logging.info("Processed %s/%s dates", i, total)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Nakshatra data (10:00 AM Mumbai by default) for each NIFTY date "
            "and each major Vedic planet/graha."
        )
    )
    parser.add_argument(
        "--nifty-csv",
        default="data/nifty50_full_history.csv",
        help="Input NIFTY CSV path (must include Date column).",
    )
    parser.add_argument(
        "--output",
        default="data/nifty50_nakshatra_10am_mumbai.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--hour", type=int, default=10, help="Local hour (0-23).")
    parser.add_argument("--minute", type=int, default=0, help="Local minute (0-59).")
    parser.add_argument("--latitude", type=float, default=MUMBAI_LAT, help="Latitude.")
    parser.add_argument("--longitude", type=float, default=MUMBAI_LON, help="Longitude.")
    parser.add_argument(
        "--log-file",
        default="logs/nifty50_nakshatra.log",
        help="Log file path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    configure_logging(log_path, args.verbose)

    logging.info("Nakshatra job started")
    logging.info(
        "Parameters | nifty_csv=%s | output=%s | time=%02d:%02d | place=Mumbai | lat=%s | lon=%s",
        args.nifty_csv,
        args.output,
        args.hour,
        args.minute,
        args.latitude,
        args.longitude,
    )

    input_path = Path(args.nifty_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dates = load_nifty_dates(input_path)
    logging.info("Loaded %s unique NIFTY dates from %s", len(dates), input_path.resolve())

    result = generate_nakshatra_data(
        dates=dates,
        hour=args.hour,
        minute=args.minute,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    result.to_csv(output_path, index=False)

    logging.info("Saved %s rows to: %s", len(result), output_path.resolve())
    logging.info(
        "Date range | start=%s | end=%s",
        result["Date"].min(),
        result["Date"].max(),
    )
    logging.info("Nakshatra job completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Nakshatra job failed")
        raise
