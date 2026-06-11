"""Load raw baseball CSVs, clean them, and save processed Parquet files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.clean import clean_dataset, summarize_dataframe, validate_cleaned_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_MANIFEST = PROJECT_ROOT / "legacy" / "static_link_ingestion" / "data_sources.csv"

TRUTHY_VALUES = {"1", "true", "yes", "y"}
URL_SOURCE_TYPES = {"url_csv", "static_csv", "csv_url"}

DATASET_NAME_ALIASES = {
    "standard_batting_05_06": "batters_standard_05_06",
    "advanced_batting_05_06": "batters_advanced_05_06",
    "statcast_batting_05_06": "batters_statcast_05_06",
}


def normalize_dataset_name(raw_name: object) -> str:
    """Turn a raw file or manifest name into a stable dataset name."""
    name = str(raw_name).lstrip("\ufeff").strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = re.sub(r"_(\d)_(\d{2})$", r"_0\1_\2", name)
    return DATASET_NAME_ALIASES.get(name, name)


def infer_dataset_name(csv_path: Path) -> str:
    """Infer a clean dataset name from the raw filename."""
    return normalize_dataset_name(csv_path.stem)


def download_configured_sources(
    manifest_path: Path = SOURCE_MANIFEST,
    raw_dir: Path = RAW_DIR,
) -> int:
    """Download enabled static CSV sources from the source manifest."""
    if not manifest_path.exists():
        print(f"No source manifest found at {manifest_path}")
        return 0

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        for row in reader:
            if not _is_enabled(row.get("enabled")):
                continue

            dataset_name = normalize_dataset_name(row.get("dataset_name", ""))
            source_type = str(row.get("source_type", "url_csv")).strip().lower()
            source_url = str(row.get("url", "")).strip()
            raw_filename = str(row.get("raw_filename", "")).strip()

            if source_type not in URL_SOURCE_TYPES:
                print(f"Skipping {dataset_name}: unsupported source_type={source_type}")
                continue
            if not source_url:
                print(f"Skipping {dataset_name}: enabled but url is blank")
                continue

            if not raw_filename:
                raw_filename = f"{dataset_name}.csv"
            raw_path = raw_dir / raw_filename

            if _download_csv(source_url, raw_path):
                downloaded += 1

    print(f"\nDownloaded {downloaded} enabled CSV source(s).")
    return downloaded


def process_csv(csv_path: Path, processed_dir: Path = PROCESSED_DIR) -> bool:
    """Process one CSV and continue gracefully if it fails."""
    dataset_name = infer_dataset_name(csv_path)
    output_path = processed_dir / f"{dataset_name}.parquet"

    try:
        raw_df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        cleaned_df = clean_dataset(raw_df)
        cleaned_df.to_parquet(output_path, index=False, engine="pyarrow")

        summary = summarize_dataframe(cleaned_df)
        print(f"{csv_path.name} -> {output_path}")
        print(f"  rows: {summary['rows']}")
        print(f"  columns: {summary['columns']}")
        print(f"  missing values: {summary['missing_values']}")

        for warning in validate_cleaned_dataset(cleaned_df, dataset_name):
            print(f"  WARNING: {warning}")

        return True
    except Exception as exc:
        print(f"ERROR: failed to process {csv_path.name}: {exc}")
        return False


def run(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    download: bool = False,
    manifest_path: Path = SOURCE_MANIFEST,
) -> None:
    """Run ingestion for every CSV in the raw data directory."""
    if download:
        download_configured_sources(manifest_path, raw_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    successes = 0
    for csv_path in csv_files:
        if process_csv(csv_path, processed_dir):
            successes += 1

    print(f"\nDone. Processed {successes} of {len(csv_files)} CSV files.")


def _download_csv(source_url: str, raw_path: Path) -> bool:
    request = Request(
        source_url,
        headers={"User-Agent": "fantasy-baseball-project/1.0"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            raw_path.write_bytes(response.read())
        print(f"Downloaded {raw_path.name}")
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"ERROR: failed to download {raw_path.name}: {exc}")
        return False


def _is_enabled(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY_VALUES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download optional CSV sources, then clean raw CSVs to Parquet."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download enabled URLs from the legacy static-link manifest before processing.",
    )
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    run(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download=args.download,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
