"""
Step 4: Load to Azure Blob Storage
  - Upload prepared dataset to Azure Blob Storage
  - Generate load manifest with checksums
  - Falls back to local data/hub/ if Azure not configured
"""

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config


def file_checksum(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def row_count(path: Path) -> int:
    """Get row count from a CSV file (excluding header)."""
    return len(pd.read_csv(path))


def load_to_azure():
    """Upload prepared dataset to Azure Blob Storage."""
    from azure.storage.blob import BlobServiceClient

    print("[load] Connecting to Azure Blob Storage...")
    client = BlobServiceClient.from_connection_string(config.AZURE_CONNECTION_STRING)

    # Create container if needed
    container = client.get_container_client(config.AZURE_CONTAINER_NAME)
    try:
        container.get_container_properties()
        print(f"  → Container '{config.AZURE_CONTAINER_NAME}' exists")
    except Exception:
        container.create_container()
        print(f"  → Created container '{config.AZURE_CONTAINER_NAME}'")

    # Upload files
    upload_plan = [
        (config.RAW_TRANSACTIONS_CSV, "raw/"),
        (config.STAGED_TRANSACTIONS, "staging/"),
        (config.PREPARED_CSV, "processed/"),
    ]

    manifest_entries = []
    for local_path, prefix in upload_plan:
        if not local_path.exists():
            print(f"  ! Skipping {local_path.name} (not found)")
            continue

        blob_name = f"{prefix}{local_path.name}"
        blob_client = container.get_blob_client(blob_name)

        print(f"  → Uploading {local_path.name} → {blob_name}...")
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        size_mb = local_path.stat().st_size / (1024 * 1024)
        manifest_entries.append({
            "blob_name": blob_name,
            "blob_url": blob_client.url,
            "size_mb": round(size_mb, 2),
            "checksum_md5": file_checksum(local_path),
            "rows": row_count(local_path),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })

    # Upload hub artifact (final dataset for ML)
    print("\n[load] Creating hub artifact...")
    shutil.copy2(config.PREPARED_CSV, config.HUB_FINAL)
    hub_blob = f"hub/{config.HUB_FINAL.name}"
    blob_client = container.get_blob_client(hub_blob)
    with open(config.HUB_FINAL, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    manifest_entries.append({
        "blob_name": hub_blob,
        "blob_url": blob_client.url,
        "size_mb": round(config.HUB_FINAL.stat().st_size / (1024 * 1024), 2),
        "checksum_md5": file_checksum(config.HUB_FINAL),
        "rows": row_count(config.HUB_FINAL),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })

    # Verify
    print("\n[load] Verifying uploaded blobs...")
    blobs = list(container.list_blobs())
    print(f"  → {len(blobs)} blobs in container:")
    for blob in blobs:
        print(f"    - {blob.name} ({blob.size / (1024*1024):.2f} MB)")

    return manifest_entries


def load_local_fallback():
    """Copy artifacts to local hub directory when Azure is not configured."""
    print("[load] Azure not configured — using local fallback")

    manifest_entries = []

    if not config.PREPARED_CSV.exists():
        print("  ! prepared CSV not found")
        return manifest_entries

    # Copy to hub
    shutil.copy2(config.PREPARED_CSV, config.HUB_FINAL)
    size_mb = config.HUB_FINAL.stat().st_size / (1024 * 1024)
    print(f"  → Copied to {config.HUB_FINAL} ({size_mb:.2f} MB)")

    manifest_entries.append({
        "blob_name": f"hub/{config.HUB_FINAL.name}",
        "local_path": str(config.HUB_FINAL),
        "size_mb": round(size_mb, 2),
        "checksum_md5": file_checksum(config.HUB_FINAL),
        "rows": row_count(config.HUB_FINAL),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    return manifest_entries


def main():
    config.ensure_dirs()
    t0 = time.time()

    if not config.PREPARED_CSV.exists():
        print("[load] ERROR: Prepared CSV not found. Run 03_transform.py first.")
        sys.exit(1)

    if config.azure_configured():
        manifest_entries = load_to_azure()
        destination = "azure"
    else:
        manifest_entries = load_local_fallback()
        destination = "local"

    # Write manifest
    elapsed = round(time.time() - t0, 2)
    manifest = {
        "pipeline": "fraud-detection-etl",
        "destination": destination,
        "elapsed_seconds": elapsed,
        "artifacts": manifest_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(config.LOAD_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[load] Manifest saved to {config.LOAD_MANIFEST}")
    print(f"[load] Done in {elapsed:.1f}s — {len(manifest_entries)} artifacts loaded to {destination}")


if __name__ == "__main__":
    main()
