"""Step 2 — Trivial Otsu baseline (the bar the U-Net must clear).

Dermoscopy lesions are typically darker than surrounding skin, so a global Otsu
threshold on the inverted grayscale image is a reasonable no-learning baseline.
If the trained model can't beat THIS, the whole platform thesis is in trouble —
which is exactly why we compute it first.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (MANIFEST_CSV, BASELINE_METRICS, REPORTS_DIR,
                    THRESH_JACCARD_CUTOFF, IMG_SIZE)
from dataset import load_manifest, split_manifest, raw_pair
from metrics import aggregate
from utils import get_logger, ensure_dir

log = get_logger("baseline")


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu: the 0-255 threshold maximizing inter-class variance."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, best_var, best_t = 0.0, 0.0, -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def otsu_lesion_mask(rgb: np.ndarray) -> np.ndarray:
    """Lesion = darker pixels (inverted Otsu).

    `otsu_threshold` accumulates hist[t] into the dark class, so the split is
    (gray <= t) dark vs (gray > t) light; the lesion is the dark class.
    """
    gray = rgb.mean(axis=2).astype(np.uint8)
    t = otsu_threshold(gray)
    return (gray <= t).astype(np.uint8)


def main() -> None:
    ensure_dir(REPORTS_DIR)
    df = load_manifest(MANIFEST_CSV)
    _, val_df = split_manifest(df)
    log.info("Running Otsu baseline on %d val images ...", len(val_df))

    preds, gts = [], []
    for _, row in val_df.iterrows():
        rgb, gt = raw_pair(row, IMG_SIZE)
        preds.append(otsu_lesion_mask(rgb))
        gts.append(gt)

    metrics = aggregate(preds, gts, THRESH_JACCARD_CUTOFF)
    metrics["method"] = "otsu_inverted"
    log.info("Baseline: Dice=%.4f IoU=%.4f thrJaccard=%.4f",
             metrics["dice"], metrics["iou"], metrics["threshold_jaccard"])
    with open(BASELINE_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote -> %s", BASELINE_METRICS)


if __name__ == "__main__":
    main()
