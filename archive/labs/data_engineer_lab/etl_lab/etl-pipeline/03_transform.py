"""
Step 3: Transform
  - Clean: fill nulls, remove duplicates, validate ranges
  - Rename columns for ML readability
  - Feature engineering: hour_of_day, amount_bin, log_amount
  - Save prepared CSV to data/processed/
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from schemas.validation import PreparedSchema


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction data: fill nulls, deduplicate, validate."""
    print("[transform] Cleaning...")
    initial_rows = len(df)

    # Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  → Filled {n_filled} nulls in '{col}' with median={median_val:.4f}")

    # Remove exact duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    if dupes_removed > 0:
        print(f"  → Removed {dupes_removed} duplicate rows")

    # Validate ranges
    df["Amount"] = df["Amount"].clip(lower=0)
    df["Class"] = df["Class"].astype(int)
    assert df["Class"].isin([0, 1]).all(), "Invalid Class values found"

    print(f"  → {len(df):,} rows after cleaning")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and add derived features."""
    print("[transform] Feature engineering...")

    # Rename for ML readability
    df = df.rename(columns={
        "Class": "is_fraud",
        "Time": "time_seconds",
        "Amount": "amount",
    })

    # hour_of_day: Time is seconds from first transaction, map to 24h cycle
    df["hour_of_day"] = ((df["time_seconds"] % 86400) / 3600).astype(int).clip(0, 23)

    # amount_bin
    bins = [0, 10, 50, 200, 1000, float("inf")]
    labels = ["0-10", "10-50", "50-200", "200-1000", "1000+"]
    df["amount_bin"] = pd.cut(df["amount"], bins=bins, labels=labels, include_lowest=True).astype(str)

    # log_amount (log1p to handle 0s)
    df["log_amount"] = np.log1p(df["amount"])

    print(f"  → Added: hour_of_day, amount_bin, log_amount")
    print(f"  → Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def main():
    config.ensure_dirs()
    t0 = time.time()

    print("[transform] Loading staged data...")
    df = pd.read_csv(config.STAGED_TRANSACTIONS)
    print(f"  → {len(df):,} rows")

    df = clean(df)
    df = transform(df)

    # Validate
    print("[transform] Validating PreparedSchema...")
    PreparedSchema.validate(df, lazy=True)
    print("  → Validation passed")

    # Save
    df.to_csv(config.PREPARED_CSV, index=False)
    csv_size = config.PREPARED_CSV.stat().st_size / (1024 * 1024)
    print(f"  → Saved: {config.PREPARED_CSV} ({csv_size:.1f} MB)")

    # Report
    elapsed = round(time.time() - t0, 2)
    report = {
        "stage": "transform",
        "elapsed_seconds": elapsed,
        "rows": len(df),
        "columns": len(df.columns),
        "csv_size_mb": round(csv_size, 2),
        "fraud_rate": round(df["is_fraud"].mean(), 6),
        "schema_validation": "passed",
    }

    report_path = config.REPORTS_DIR / "transform_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[transform] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
