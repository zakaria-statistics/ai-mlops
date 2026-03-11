"""
Step 1: Data Acquisition
  - Fetch Credit Card Fraud Detection dataset from Kaggle or OpenML
  - Save to data/raw/transactions.csv
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).parent))
import config


def _fetch_from_kaggle() -> pd.DataFrame:
    """Download credit card fraud dataset from Kaggle."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: kaggle package not installed.\n"
            "  pip install kaggle\n"
            "  Then place your API token at ~/.kaggle/kaggle.json\n"
            "  See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        sys.exit(
            "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Place the downloaded kaggle.json in ~/.kaggle/\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )

    download_dir = config.RAW_DIR / "kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    csv_path = download_dir / "creditcard.csv"

    if not csv_path.exists():
        print(f"  → Downloading {config.KAGGLE_DATASET} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET,
             "-p", str(download_dir), "--unzip"],
            check=True,
        )
    else:
        print(f"  → Using cached download at {csv_path}")

    return pd.read_csv(csv_path)


def _fetch_from_openml() -> pd.DataFrame:
    """Download credit card fraud dataset from OpenML."""
    data = fetch_openml(data_id=config.OPENML_DATASET_ID, as_frame=True, parser="auto")
    return data.frame


def fetch_transactions(source: str = "kaggle") -> pd.DataFrame:
    """Download credit card fraud dataset."""
    print(f"[generate] Fetching transactions from {source}...")

    if source == "kaggle":
        df = _fetch_from_kaggle()
    else:
        df = _fetch_from_openml()

    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def main(source: str = "kaggle"):
    config.ensure_dirs()

    transactions = fetch_transactions(source)
    transactions.to_csv(config.RAW_TRANSACTIONS_CSV, index=False)

    csv_size = config.RAW_TRANSACTIONS_CSV.stat().st_size / (1024 * 1024)
    print(f"  → Saved to {config.RAW_TRANSACTIONS_CSV} ({csv_size:.1f} MB, {len(transactions):,} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: Download raw data")
    parser.add_argument("--source", choices=["kaggle", "openml"], default="kaggle",
                        help="Dataset source (default: kaggle)")
    args = parser.parse_args()
    main(source=args.source)
