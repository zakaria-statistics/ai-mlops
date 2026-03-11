"""
Fraud Detection ETL Pipeline — Orchestrator
  Runs steps 01→04 sequentially.

Usage:
  python pipeline.py                    # Full run (Kaggle + Azure or local)
  python pipeline.py --source openml    # Use OpenML instead of Kaggle
  python pipeline.py --local-only       # Skip Azure upload, local hub only
  python pipeline.py --skip-download    # Skip step 01 (data already exists)
"""

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


def run_step(name: str, func, *args, **kwargs):
    """Run a pipeline step with timing."""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    func(*args, **kwargs)
    elapsed = time.time() - t0

    print(f"\n  completed in {elapsed:.1f}s")
    return elapsed


def validate_outputs(step: str, paths: list[Path]):
    """Check that expected output files exist after a step."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"\n[pipeline] ERROR after {step}:")
        for p in missing:
            print(f"  ! Missing: {p}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection ETL Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip step 01 (use existing raw data)")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip Azure upload, save to local hub only")
    parser.add_argument("--source", choices=["kaggle", "openml"], default="kaggle",
                        help="Dataset source (default: kaggle)")
    args = parser.parse_args()

    config.ensure_dirs()
    timings = {}
    pipeline_start = time.time()

    print("=" * 60)
    print(" FRAUD DETECTION ETL PIPELINE")
    print("=" * 60)
    print(f"  Source     : {args.source}")
    print(f"  Local-only : {args.local_only}")
    print(f"  Skip DL    : {args.skip_download}")
    print(f"  Azure      : {'configured' if config.azure_configured() else 'not configured (local fallback)'}")

    # Step 01: Download Data
    if not args.skip_download:
        gen = import_module("01_generate_data")
        timings["01_download"] = run_step("Step 01: Download Data", gen.main, source=args.source)
        validate_outputs("01_download", [config.RAW_TRANSACTIONS_CSV])
    else:
        print("\n[pipeline] Skipping step 01 (--skip-download)")
        validate_outputs("01_download (skipped)", [config.RAW_TRANSACTIONS_CSV])

    # Step 02: Extract
    extract = import_module("02_extract")
    timings["02_extract"] = run_step("Step 02: Extract", extract.main)
    validate_outputs("02_extract", [config.STAGED_TRANSACTIONS])

    # Step 03: Transform
    transform = import_module("03_transform")
    timings["03_transform"] = run_step("Step 03: Transform", transform.main)
    validate_outputs("03_transform", [config.PREPARED_CSV])

    # Step 04: Load
    if args.local_only:
        original = config.AZURE_CONNECTION_STRING
        config.AZURE_CONNECTION_STRING = ""
        load = import_module("04_load")
        timings["04_load"] = run_step("Step 04: Load (local only)", load.main)
        config.AZURE_CONNECTION_STRING = original
    else:
        load = import_module("04_load")
        timings["04_load"] = run_step("Step 04: Load", load.main)

    validate_outputs("04_load", [config.LOAD_MANIFEST])

    # Summary
    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f" PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Step Timings:")
    for step, elapsed in timings.items():
        print(f"    {step:<20s} {elapsed:>6.1f}s")
    print(f"    {'─'*28}")
    print(f"    {'Total':<20s} {total:>6.1f}s")

    print(f"\n  Outputs:")
    for label, path in [
        ("Raw CSV", config.RAW_TRANSACTIONS_CSV),
        ("Staged", config.STAGED_TRANSACTIONS),
        ("Prepared", config.PREPARED_CSV),
        ("Hub", config.HUB_FINAL),
        ("Manifest", config.LOAD_MANIFEST),
    ]:
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"    {label:<12s} {str(path):<50s} {size:>7.2f} MB")

    print()


if __name__ == "__main__":
    main()
