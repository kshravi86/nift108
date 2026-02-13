"""Compile and submit the astrology NN Vertex AI Pipeline."""

import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import google.auth
from google.cloud import aiplatform
from google.oauth2.credentials import Credentials
from kfp import compiler

from astrology_nn_pipeline import ASTROLOGY_FEATURES, astrology_nn_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and run the astrology-only NIFTY Vertex pipeline."
    )
    parser.add_argument("--project", default="nift108", help="GCP project id")
    parser.add_argument("--location", default="us-central1", help="Vertex region")
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="GCS path for pipeline artifacts (e.g. gs://nift108-bucket/pipeline-root)",
    )
    parser.add_argument(
        "--bq-table",
        default="nift108.nift108_ds.nifty50_full_with_nakshatra_10am_mumbai",
        help="Fully-qualified BigQuery table name",
    )
    parser.add_argument(
        "--target-column",
        default="open_close_diff",
        help="Target column name in BQ table",
    )
    parser.add_argument(
        "--feature-columns-csv",
        default=",".join(ASTROLOGY_FEATURES),
        help="Comma-separated feature columns",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--service-account",
        default="",
        help="Optional service account email for the pipeline run",
    )
    parser.add_argument(
        "--display-name",
        default=f"nifty-astrology-nn-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        help="Vertex PipelineJob display name",
    )
    parser.add_argument(
        "--template-path",
        default=str(Path(__file__).resolve().parent / "astrology_nn_pipeline.json"),
        help="Compiled pipeline JSON path",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline JSON, do not submit a run",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Wait until pipeline completes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = str(Path(args.template_path).resolve())
    compiler.Compiler().compile(
        pipeline_func=astrology_nn_pipeline,
        package_path=template_path,
    )
    print(f"Compiled pipeline template: {template_path}")

    if args.compile_only:
        return

    credentials = None
    try:
        credentials, _ = google.auth.default()
    except Exception:
        gcloud_bin = shutil.which("gcloud") or shutil.which("gcloud.cmd")
        if not gcloud_bin:
            default_gcloud = (
                Path.home()
                / "AppData"
                / "Local"
                / "Google"
                / "Cloud SDK"
                / "google-cloud-sdk"
                / "bin"
                / "gcloud.cmd"
            )
            if default_gcloud.exists():
                gcloud_bin = str(default_gcloud)
        if not gcloud_bin:
            raise RuntimeError("gcloud CLI not found in PATH.")

        env = dict(os.environ)
        token = (
            subprocess.run(
                [gcloud_bin, "auth", "print-access-token"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            .stdout.strip()
        )
        if not token:
            raise RuntimeError(
                "No ADC found and no gcloud access token available. "
                "Run `gcloud auth login` first."
            )
        credentials = Credentials(token=token)

    aiplatform.init(
        project=args.project,
        location=args.location,
        credentials=credentials,
    )
    parameter_values = {
        "project_id": args.project,
        "bq_table": args.bq_table,
        "target_column": args.target_column,
        "feature_columns_csv": args.feature_columns_csv,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
    }

    job = aiplatform.PipelineJob(
        display_name=args.display_name,
        template_path=template_path,
        pipeline_root=args.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=False,
        credentials=credentials,
    )

    run_kwargs = {"sync": args.sync}
    if args.service_account.strip():
        run_kwargs["service_account"] = args.service_account.strip()

    job.run(**run_kwargs)
    resource_name = None
    try:
        resource_name = job.resource_name
    except Exception:
        resource_name = getattr(getattr(job, "_gca_resource", None), "name", None)

    if resource_name:
        print(f"Pipeline started: {resource_name}")
    try:
        print(f"Pipeline state: {job.state}")
    except Exception:
        pass
    print(
        f"Vertex URL: https://console.cloud.google.com/vertex-ai/locations/{args.location}/pipelines/runs"
        f"?project={args.project}"
    )


if __name__ == "__main__":
    main()
