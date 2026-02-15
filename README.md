# nift108

Python script to download NIFTY 50 (`^NSEI`) daily data for the last 3 years and save it as CSV.

## Files

- `fetch_nifty50_last_3_years.py` - downloads NIFTY 50 data from Yahoo Finance.
- `fetch_nifty50_hourly_last_3_years.py` - downloads NIFTY 50 hourly (60m) data from Yahoo Finance (note: intraday retention limits apply; you may only get ~730 days).
- `generate_nakshatra_for_nifty_dates.py` - generates planetary Nakshatra/Pada data per NIFTY date.
- `generate_nakshatra_for_nifty_hourly.py` - generates planetary Nakshatra/Pada data per NIFTY hourly timestamp (Mumbai, Lahiri ayanamsha).
- `vertex_pipeline/` - Vertex AI pipeline for astrology-only neural network training/evaluation.
- `requirements.txt` - python dependencies.

## Usage

```bash
pip install -r requirements.txt
python fetch_nifty50_last_3_years.py --output nifty50_last_3_years.csv
```

Hourly example:

```bash
python fetch_nifty50_hourly_last_3_years.py --years 3 --output nifty50_hourly_last_3_years.csv
python generate_nakshatra_for_nifty_hourly.py \
  --nifty-csv nifty50_hourly_last_3_years.csv \
  --output nifty50_hourly_with_nakshatra_mumbai_lahiri.csv
```

Optional arguments:

- `--ticker` (default: `^NSEI`)
- `--years` (default: `3`)
- `--output` (default: `nifty50_last_3_years.csv`)

Example:

```bash
python fetch_nifty50_last_3_years.py --years 3 --output data/nifty50_last_3_years.csv
```
