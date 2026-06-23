"""Step 1 — Download ISIC 2018 Task 1 and build a manifest.

Downloads the segmentation images + ground-truth masks, pairs them by ISIC id,
and records the license/attribution for every image (a hard requirement before
any of this data is ever served). The big input zip is ~10 GB; the mask zip is
small. Use --enrich-licenses to fetch per-image licenses from the ISIC API.
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request

import pandas as pd

from config import (ISIC_IMAGES_URL, ISIC_MASKS_URL, RAW_DIR, DATA_DIR,
                    MANIFEST_CSV, DATASET_LICENSE, DATASET_ATTRIBUTION, ISIC_API)
from utils import get_logger, ensure_dir, isic_id

log = get_logger("download")

IMAGES_DIR = os.path.join(RAW_DIR, "ISIC2018_Task1-2_Training_Input")
MASKS_DIR = os.path.join(RAW_DIR, "ISIC2018_Task1_Training_GroundTruth")


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, block_num * block_size * 100 // total_size)
    if block_num % 200 == 0 or pct >= 100:
        done = block_num * block_size / 1e6
        log.info("  ... %d%% (%.0f MB)", pct, done)


def fetch_zip(url: str, dest_zip: str, extract_to: str) -> None:
    ensure_dir(RAW_DIR)
    marker = extract_to
    if os.path.isdir(marker) and os.listdir(marker):
        log.info("Already extracted: %s", marker)
        return
    if not os.path.exists(dest_zip):
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, dest_zip, _progress)
    log.info("Extracting %s", dest_zip)
    with zipfile.ZipFile(dest_zip) as z:
        z.extractall(RAW_DIR)


def enrich_license(iid: str):
    """Fetch (license, attribution) for one ISIC id from the public API."""
    import json
    url = f"{ISIC_API}{iid}/"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return (data.get("copyright_license") or DATASET_LICENSE,
                data.get("attribution") or DATASET_ATTRIBUTION)
    except Exception as e:  # noqa: BLE001 - best-effort enrichment
        log.warning("license fetch failed for %s: %s", iid, e)
        return DATASET_LICENSE, DATASET_ATTRIBUTION


def build_manifest(enrich: bool) -> pd.DataFrame:
    rows = []
    masks = sorted(f for f in os.listdir(MASKS_DIR) if f.endswith(".png"))
    log.info("Pairing %d masks with images ...", len(masks))
    for mfile in masks:
        iid = isic_id(mfile)
        if not iid:
            continue
        img_path = os.path.join(IMAGES_DIR, f"{iid}.jpg")
        if not os.path.exists(img_path):
            continue
        lic, attr = (enrich_license(iid) if enrich
                     else (DATASET_LICENSE, DATASET_ATTRIBUTION))
        rows.append({
            "isic_id": iid,
            "image_path": img_path,
            "mask_path": os.path.join(MASKS_DIR, mfile),
            "license": lic,
            "attribution": attr,
        })
    df = pd.DataFrame(rows)
    log.info("Manifest: %d image/mask pairs", len(df))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich-licenses", action="store_true",
                    help="fetch per-image license from the ISIC API (slow)")
    args = ap.parse_args()

    ensure_dir(DATA_DIR)
    fetch_zip(ISIC_MASKS_URL, os.path.join(RAW_DIR, "masks.zip"), MASKS_DIR)
    fetch_zip(ISIC_IMAGES_URL, os.path.join(RAW_DIR, "images.zip"), IMAGES_DIR)
    df = build_manifest(args.enrich_licenses)
    df.to_csv(MANIFEST_CSV, index=False)
    log.info("Wrote -> %s", MANIFEST_CSV)
    log.info("License summary: %s", df["license"].value_counts().to_dict())


if __name__ == "__main__":
    main()
