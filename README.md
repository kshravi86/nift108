# nift108

Python script to download NIFTY 50 (`^NSEI`) daily data for the last 3 years and save it as CSV.

## Files

- `fetch_nifty50_last_3_years.py` - downloads NIFTY 50 data from Yahoo Finance.
- `generate_nakshatra_for_nifty_dates.py` - generates planetary Nakshatra/Pada data per NIFTY date.
- `vertex_pipeline/` - Vertex AI pipeline for astrology-only neural network training/evaluation.
- `requirements.txt` - python dependencies.

## Usage

```bash
pip install -r requirements.txt
python fetch_nifty50_last_3_years.py --output nifty50_last_3_years.csv
```

Optional arguments:

- `--ticker` (default: `^NSEI`)
- `--years` (default: `3`)
- `--output` (default: `nifty50_last_3_years.csv`)

Example:

```bash
python fetch_nifty50_last_3_years.py --years 3 --output data/nifty50_last_3_years.csv
```
