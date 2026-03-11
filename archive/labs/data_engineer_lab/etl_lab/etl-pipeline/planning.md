# Fraud Detection ETL Pipeline

> Simple ETL pipeline: download Credit Card Fraud dataset from Kaggle, clean & prepare, load to Azure Blob Storage as CSV for ML consumption.

## Table of Contents

1. [Architecture Overview](#architecture-overview) - Pipeline flow and components
2. [Data Architecture](#data-architecture) - Schema evolution across stages
3. [Component Breakdown](#component-breakdown) - What each script does
4. [Data Zone Progression](#data-zone-progression) - raw → staging → processed → hub
5. [Configuration & Environment](#configuration--environment) - Config, Azure, Kaggle setup
6. [Validation Strategy](#validation-strategy) - Pandera schemas per stage
7. [Running the Pipeline](#running-the-pipeline) - Dependencies, CLI usage

---

## Architecture Overview

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        pipeline.py                              │
│                    (orchestrator + CLI)                          │
└──────┬──────────────┬────────────────┬─────────────────┬────────┘
       │              │                │                 │
       ▼              ▼                ▼                 ▼
 ┌───────────┐  ┌───────────┐  ┌────────────┐   ┌────────────┐
 │    01      │  │    02      │  │     03      │   │     04      │
 │  Download  │  │  Extract   │  │  Transform  │   │    Load     │
 │   Data     │→ │  & Stage   │→ │  & Prepare  │→  │  to Azure   │
 └───────────┘  └───────────┘  └────────────┘   └────────────┘
       │              │                │                 │
       ▼              ▼                ▼                 ▼
   data/raw/     data/staging/   data/processed/    data/hub/
                 + reports/      + reports/         + manifest
```

### Component Relationships

```
config.py ◄──────── All scripts import paths, params, Azure config
    │
    ├── .env ──────► AZURE_STORAGE_CONNECTION_STRING
    │                (loaded via python-dotenv)
    │
schemas/
    └── validation.py ◄── 02_extract.py  (RawTransactionSchema)
                      ◄── 03_transform.py (PreparedSchema)
```

### Data Source

```
Source: Kaggle (mlg-ulb/creditcardfraud)
┌──────────────────────────────┐
│ Credit Card Fraud Detection  │
│ 284,807 transactions         │
│ 31 columns (V1-V28 + meta)  │
│ Format: CSV                  │
│ Labelled: Class (0/1)        │
└──────────────────────────────┘
```

---

## Data Architecture

### Schema Evolution Across Stages

```
RAW (as-downloaded)              STAGED (validated CSV)           PREPARED (cleaned + features)
─────────────────────            ─────────────────────            ──────────────────────────
Time           float             Time           float        ──►  time_seconds     float    (renamed)
V1..V28        float             V1..V28        float             V1..V28          float
Amount         float             Amount         float        ──►  amount           float    (renamed)
Class          int               Class          int          ──►  is_fraud         int      (renamed)
                                                                  hour_of_day      int      (derived)
                                                                  amount_bin       str      (derived)
                                                                  log_amount       float    (derived)
```

### Derived Features

| Feature       | Logic                                      | Purpose                    |
|---------------|--------------------------------------------|-----------------------------|
| `hour_of_day` | `(time_seconds % 86400) / 3600` → int 0-23| Temporal fraud patterns     |
| `amount_bin`  | 5 buckets: 0-10, 10-50, 50-200, 200-1000, 1000+ | Amount-based analysis |
| `log_amount`  | `log1p(amount)`                            | Normalize skewed amounts    |

---

## Component Breakdown

### `config.py` — Central Configuration

Single source of truth for the entire pipeline:
- **Paths:** all `data/` subdirectories, file locations, report paths
- **Azure:** connection string loaded from `.env`, container name
- **Dataset:** Kaggle dataset ID, OpenML fallback ID
- **Helpers:** `ensure_dirs()` creates all directories, `azure_configured()` checks credentials

### `schemas/validation.py` — Pandera Schemas

| Schema                 | Used by        | Key checks                                       |
|------------------------|----------------|--------------------------------------------------|
| `RawTransactionSchema` | `02_extract`   | Amount >= 0, Class in {0,1}, V1-V28 float        |
| `PreparedSchema`       | `03_transform` | Renamed cols, derived features exist, ranges valid|

### `01_generate_data.py` — Data Download

```
Kaggle API (default) ──► creditcard.csv ──► data/raw/transactions.csv
   or
OpenML API (fallback) ──► DataFrame ──► data/raw/transactions.csv
```

- Default: `--source kaggle` using `kaggle datasets download`
- Fallback: `--source openml` using `sklearn.datasets.fetch_openml`
- Kaggle requires `~/.kaggle/kaggle.json` credentials

### `02_extract.py` — Extract & Profile

```
data/raw/
  transactions.csv  ──► profile ──► validate ──► data/staging/transactions.csv
                                                        │
                                               reports/extract_report.json
```

- **Profile:** row/col counts, dtypes, null %, memory usage
- **Validate:** Pandera RawTransactionSchema (lazy mode)
- **Stage:** validated CSV copy to staging zone

### `03_transform.py` — Clean & Prepare

```
STAGED CSV
     │
     ▼
┌─────────────────────────────────────┐
│  Phase 1: CLEAN                     │
│  • Numeric nulls → median           │
│  • Remove exact duplicates          │
│  • Clip Amount >= 0                 │
│  • Assert Class in {0, 1}           │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│  Phase 2: TRANSFORM                 │
│  • Rename: Class→is_fraud           │
│  •         Time→time_seconds        │
│  •         Amount→amount            │
│  • Derive: hour_of_day              │
│  • Derive: amount_bin (5 buckets)   │
│  • Derive: log_amount               │
└─────────────┬───────────────────────┘
              ▼
  data/processed/fraud_prepared.csv
  + reports/transform_report.json
```

### `04_load.py` — Load to Azure Blob Storage

```
                    ┌─── Azure configured? ───┐
                    │                         │
                   YES                        NO
                    │                         │
      ┌─────────────▼──────────────┐  ┌──────▼──────────┐
      │ BlobServiceClient          │  │ Local fallback   │
      │ • Create container         │  │ • Copy to hub/   │
      │ • Upload raw/              │  │ • Write manifest  │
      │ • Upload staging/          │  └─────────────────┘
      │ • Upload processed/        │
      │ • Upload hub/              │
      │ • List blobs (verify)      │
      │ • Write manifest w/ URLs   │
      └────────────────────────────┘
```

- **Manifest (`load_manifest.json`):** blob URLs, MD5 checksums, row counts, timestamps
- **Container:** `fraud-etl-pipeline` (auto-created)
- **Blob prefixes:** `raw/`, `staging/`, `processed/`, `hub/`

---

## Data Zone Progression

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│   RAW    │───►│ STAGING  │───►│ PROCESSED  │───►│   HUB    │
│          │    │          │    │            │    │          │
│ Original │    │ Validated│    │ Cleaned +  │    │ Final    │
│ CSV from │    │ CSV      │    │ features   │    │ ML-ready │
│ Kaggle   │    │          │    │ added      │    │ CSV      │
└──────────┘    └──────────┘    └────────────┘    └──────────┘
  Mutable        Immutable       Immutable         Immutable
  (download)     (extract)       (transform)       (load)
```

| Zone       | Path              | Format  | Contents                                |
|------------|-------------------|---------|-----------------------------------------|
| Raw        | `data/raw/`       | CSV     | Original Kaggle download                |
| Staging    | `data/staging/`   | CSV     | Validated copy                          |
| Processed  | `data/processed/` | CSV     | Cleaned + feature-engineered dataset    |
| Hub        | `data/hub/`       | CSV     | Final ML-ready dataset + manifest       |
| Reports    | `reports/`        | JSON    | Profiling + validation reports per stage|

---

## Configuration & Environment

### Parameters

| Parameter             | Value                    | Purpose                          |
|-----------------------|--------------------------|----------------------------------|
| `KAGGLE_DATASET`     | `mlg-ulb/creditcardfraud`| Kaggle dataset identifier         |
| `OPENML_DATASET_ID`  | `1597`                   | OpenML fallback dataset           |
| `RANDOM_SEED`        | `42`                     | Reproducibility                   |

### Kaggle Setup

```
1. pip install kaggle
2. Go to kaggle.com/settings → API → Create Legacy API Key
3. Place kaggle.json at ~/.kaggle/kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json
```

### Azure Setup

```
1. cp .env.example .env
2. Edit .env → paste your Azure Storage connection string
3. Pipeline auto-creates container "fraud-etl-pipeline"
4. If no .env / invalid creds → local hub fallback (no errors)
```

---

## Validation Strategy

Pandera schemas enforce contracts at stage boundaries:

```
01_download  ──►  data/raw/
                     │
02_extract   ──►  RawTransactionSchema  ──►  data/staging/
                     │
03_transform ──►  PreparedSchema        ──►  data/processed/
                     │
04_load      ──►  (manifest checksum)   ──►  data/hub/ + Azure
```

- **Lazy validation:** collects all schema errors before failing
- **Strict=False:** allows extra columns (V1-V28 pass through without individual rules)
- **Coerce=True:** auto-casts compatible types rather than failing

---

## Running the Pipeline

### Install Dependencies

```bash
pip install pandas pandera kaggle azure-storage-blob python-dotenv scikit-learn
```

### Individual Steps

```bash
cd lab/etl-pipeline
python 01_generate_data.py              # → data/raw/transactions.csv (~150MB)
python 02_extract.py                    # → data/staging/transactions.csv + report
python 03_transform.py                  # → data/processed/fraud_prepared.csv + report
python 04_load.py                       # → Azure blobs or data/hub/ (local fallback)
```

### Full Pipeline

```bash
python pipeline.py                      # End-to-end (Kaggle + Azure or local)
python pipeline.py --source openml      # Use OpenML instead of Kaggle
python pipeline.py --local-only         # Force local hub, skip Azure
python pipeline.py --skip-download      # Reuse existing data/raw/
```

### Verification

| Step | Command                         | Expected Output                              |
|------|---------------------------------|----------------------------------------------|
| 01   | `python 01_generate_data.py`    | `data/raw/transactions.csv`                  |
| 02   | `python 02_extract.py`          | `data/staging/transactions.csv` + report      |
| 03   | `python 03_transform.py`        | `data/processed/fraud_prepared.csv` + report  |
| 04   | `python 04_load.py`             | Azure blobs or `data/hub/` + manifest        |
| All  | `python pipeline.py`            | Full run with timing summary                 |
