"""
Step 2: Extract
  - Read raw CSV
  - Profile (row/col counts, dtypes, null %, memory)
  - Validate against Pandera schema
  - Stage as validated CSV
  - Write extract report
"""

import json
import sys
import time
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from schemas.validation import RawTransactionSchema


def profile_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Generate profiling stats for a DataFrame."""
    null_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(memory_mb, 2),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_pct": {k: round(v, 2) for k, v in null_pct.items() if v > 0},
        "column_list": list(df.columns),
    }


def main():
    config.ensure_dirs()
    t0 = time.time()

    print("[extract] Reading transactions CSV...")
    df = pd.read_csv(config.RAW_TRANSACTIONS_CSV)

    profile = profile_dataframe(df, "transactions")
    print(f"  → {profile['rows']:,} rows, {profile['columns']} cols, "
          f"{profile['memory_mb']:.1f} MB")

    # Validate against schema
    print("  → Validating against RawTransactionSchema...")
    RawTransactionSchema.validate(df, lazy=True)
    print("  → Schema validation passed")

    # Stage as validated CSV
    df.to_csv(config.STAGED_TRANSACTIONS, index=False)
    csv_size = config.STAGED_TRANSACTIONS.stat().st_size / (1024 * 1024)
    print(f"  → Staged: {config.STAGED_TRANSACTIONS} ({csv_size:.1f} MB)")

    # Write extract report
    elapsed = round(time.time() - t0, 2)
    report = {
        "stage": "extract",
        "elapsed_seconds": elapsed,
        "profile": {**profile, "staged_size_mb": round(csv_size, 2)},
        "schema_validation": "passed",
    }

    report_path = config.REPORTS_DIR / "extract_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[extract] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
