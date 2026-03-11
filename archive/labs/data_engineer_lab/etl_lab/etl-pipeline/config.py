"""Central configuration for the Fraud Detection ETL Pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from pipeline directory
_pipeline_dir = Path(__file__).parent
load_dotenv(_pipeline_dir / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = _pipeline_dir
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
PROCESSED_DIR = DATA_DIR / "processed"
HUB_DIR = DATA_DIR / "hub"
REPORTS_DIR = BASE_DIR / "reports"

# Raw source file
RAW_TRANSACTIONS_CSV = RAW_DIR / "transactions.csv"

# Staged CSV
STAGED_TRANSACTIONS = STAGING_DIR / "transactions.csv"

# Processed output
PREPARED_CSV = PROCESSED_DIR / "fraud_prepared.csv"

# Hub artifact
HUB_FINAL = HUB_DIR / "fraud_dataset.csv"
LOAD_MANIFEST = HUB_DIR / "load_manifest.json"

# ── Azure ──────────────────────────────────────────────────────────────
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = "fraud-etl-pipeline"

# ── Dataset Parameters ─────────────────────────────────────────────────
OPENML_DATASET_ID = 1597           # Credit Card Fraud Detection
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"  # Same dataset on Kaggle
RANDOM_SEED = 42

# ── Helpers ────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [RAW_DIR, STAGING_DIR, PROCESSED_DIR, HUB_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def azure_configured() -> bool:
    """Check if Azure credentials are available."""
    return bool(AZURE_CONNECTION_STRING) and "YOUR_ACCOUNT" not in AZURE_CONNECTION_STRING
