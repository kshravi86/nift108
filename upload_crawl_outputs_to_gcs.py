#!/usr/bin/env python3
"""
Upload crawl outputs (e.g. JSONL docs files) to a folder (prefix) in a GCS bucket.

Examples:
  python3 upload_crawl_outputs_to_gcs.py --gcs-prefix gs://my-bucket/vertex-docs/ gcp_docs_pmle_hybrid_*.jsonl
  python3 upload_crawl_outputs_to_gcs.py --gcs-prefix gs://my-bucket/vertex-docs/run1/ --pattern "gcp_docs_pmle_hybrid_*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_gcs_prefix(gcs_prefix: str) -> tuple[str, str]:
    m = re.match(r"^gs://([^/]+)(?:/(.*))?$", gcs_prefix.strip())
    if not m:
        raise ValueError(f"Invalid --gcs-prefix: {gcs_prefix!r} (expected gs://bucket/prefix)")
    bucket = m.group(1)
    prefix = (m.group(2) or "").strip("/")
    return bucket, prefix


def is_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def expand_sources(sources: list[str]) -> list[Path]:
    out: list[Path] = []
    for src in sources:
        if is_glob(src):
            matches = [Path(p) for p in glob.glob(src)]
            out.extend(matches)
        else:
            out.append(Path(src))
    return out


def collect_files(sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        if src.is_file():
            files.append(src)
        elif src.is_dir():
            for root, dirs, filenames in os.walk(src):
                # Skip common non-artifact dirs.
                dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__"}]
                for fn in filenames:
                    p = Path(root) / fn
                    if p.is_file():
                        files.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {src}")
    # De-dupe while preserving order.
    seen = set()
    uniq: list[Path] = []
    for p in files:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def dest_object_name(prefix: str, src: Path, base_dir: Path | None) -> str:
    if base_dir is not None:
        rel = src.resolve().relative_to(base_dir.resolve()).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
    else:
        key = f"{prefix}/{src.name}" if prefix else src.name
    return key.lstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload crawl outputs to GCS.")
    ap.add_argument(
        "--gcs-prefix",
        required=True,
        help="Destination prefix, e.g. gs://my-bucket/folder/subfolder",
    )
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern(s) to include (evaluated by the script). Can be repeated.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel uploads (files), default 4.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded, but do not upload.",
    )
    ap.add_argument(
        "--base-dir",
        default="",
        help="Optional base dir to preserve relative paths when uploading.",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Files/dirs/globs to upload. If omitted, defaults to gcp_docs_pmle_hybrid_*.jsonl in cwd.",
    )
    args = ap.parse_args()

    try:
        bucket_name, prefix = parse_gcs_prefix(args.gcs_prefix)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    sources: list[str] = []
    if args.paths:
        sources.extend(args.paths)
    if args.pattern:
        sources.extend(args.pattern)
    if not sources:
        sources = ["gcp_docs_pmle_hybrid_*.jsonl"]

    expanded = expand_sources(sources)
    try:
        files = collect_files(expanded)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not files:
        print("No files matched.")
        return 0

    base_dir = Path(args.base_dir) if args.base_dir else None
    if base_dir is not None and not base_dir.exists():
        print(f"ERROR: --base-dir does not exist: {base_dir}", file=sys.stderr)
        return 2

    try:
        from google.cloud import storage  # type: ignore
    except Exception:
        print(
            "ERROR: Missing dependency google-cloud-storage.\n"
            "Install it with: python3 -m pip install google-cloud-storage\n",
            file=sys.stderr,
        )
        return 2

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    plan = [(src, dest_object_name(prefix, src, base_dir)) for src in files]

    total_bytes = sum(p.stat().st_size for p, _ in plan)
    print(f"Uploading {len(plan)} file(s) to gs://{bucket_name}/{prefix or ''}")
    print(f"Total bytes: {total_bytes}")

    if args.dry_run:
        for src, obj in plan:
            print(f"DRY_RUN: {src} -> gs://{bucket_name}/{obj}")
        return 0

    def _upload_one(src_obj: tuple[Path, str]) -> tuple[Path, str]:
        src, obj = src_obj
        blob = bucket.blob(obj)
        # Resumable uploads for larger files.
        if src.stat().st_size >= 8 * 1024 * 1024:
            blob.chunk_size = 8 * 1024 * 1024
        blob.upload_from_filename(str(src))
        return src, obj

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_upload_one, item): item for item in plan}
        for fut in as_completed(futs):
            src, obj = futs[fut]
            try:
                fut.result()
                print(f"OK: {src} -> gs://{bucket_name}/{obj}")
            except Exception as e:
                failures += 1
                print(f"FAIL: {src} -> gs://{bucket_name}/{obj}: {e}", file=sys.stderr)

    if failures:
        print(f"Completed with failures: {failures}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

