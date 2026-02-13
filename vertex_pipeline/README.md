# Vertex Pipeline: Astrology-only Neural Network

This folder contains a Vertex AI Pipeline that:

- Reads astrology + target data from BigQuery
- Uses all astrology features (`*_Nakshatra`, `*_Pada`) only
- Trains an embedding-based neural network
- Splits data as:
  - Training: rows before latest 1 year
  - Validation: latest 1 year
- Produces:
  - Model artifact
  - Validation predictions artifact
  - Validation metrics (`mae`, `rmse`, `r2`, `direction_accuracy`)

## Prerequisites

- Active gcloud project: `nift108`
- Enabled APIs:
  - `aiplatform.googleapis.com`
  - `artifactregistry.googleapis.com`
  - `cloudbuild.googleapis.com`
  - `bigquery.googleapis.com`
- A GCS path for pipeline root, for example:
  - `gs://nift108-bucket/vertex-pipelines/astrology-nn`

## Install dependencies (local machine)

```bash
pip install -r vertex_pipeline/requirements.txt
```

## Compile and run

```bash
python vertex_pipeline/run_pipeline.py \
  --project nift108 \
  --location us-central1 \
  --pipeline-root gs://nift108-bucket/vertex-pipelines/astrology-nn \
  --bq-table nift108.nift108_ds.nifty50_full_with_nakshatra_10am_mumbai
```

Use `--sync` if you want the command to wait until completion.
