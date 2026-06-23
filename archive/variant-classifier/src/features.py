"""Step 3 — Join labels + annotations into a model-ready feature table.

XGBoost handles missing values natively, so we keep NaNs rather than imputing.
We add a couple of light engineered features and a log-transformed allele
frequency.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import LABELED_CSV, ANNOTATIONS_CSV, FEATURES_CSV, DATA_DIR
from utils import get_logger, ensure_dir

log = get_logger("features")

NUMERIC_FEATURES = [
    "cadd_phred", "gerp", "sift_score", "polyphen_score", "revel_score",
    "log_gnomad_af", "is_transition",
]
META_COLS = ["variant_id", "gene", "label"]


def main() -> None:
    ensure_dir(DATA_DIR)
    labeled = pd.read_csv(LABELED_CSV)
    ann = pd.read_csv(ANNOTATIONS_CSV)
    df = labeled.merge(ann, on="variant_id", how="left")

    # rare-variant signal: pathogenic variants tend to be very rare.
    df["gnomad_af"] = df["gnomad_af"].fillna(0.0)
    df["log_gnomad_af"] = np.log10(df["gnomad_af"] + 1e-7)

    # drop rows with no functional annotation at all (nothing to learn from)
    score_cols = ["cadd_phred", "gerp", "sift_score", "polyphen_score",
                  "revel_score"]
    before = len(df)
    df = df[df[score_cols].notna().any(axis=1)]
    log.info("Dropped %d variants with zero functional annotations",
             before - len(df))

    out = df[META_COLS + NUMERIC_FEATURES].copy()
    log.info("Feature table: %d rows x %d features", len(out),
             len(NUMERIC_FEATURES))
    log.info("Class balance: %s", out["label"].value_counts().to_dict())
    out.to_csv(FEATURES_CSV, index=False)
    log.info("Wrote -> %s", FEATURES_CSV)


if __name__ == "__main__":
    main()
