"""Step 4 — Train an XGBoost pathogenicity classifier.

Key methodology choice: we split by GENE (GroupShuffleSplit) so no gene appears
in both train and test. This prevents the model from simply memorizing which
genes are disease genes and forces it to learn from the functional features —
a far more honest estimate of real-world performance.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from config import (FEATURES_CSV, TEST_CSV, MODEL_PATH, FEATURE_COLS_PATH,
                    METRICS_PATH, MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
                    TEST_SIZE, VALID_SIZE)
from features import NUMERIC_FEATURES
from utils import get_logger, ensure_dir

log = get_logger("train")


def grouped_split(df, group_col, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    train_idx, test_idx = next(
        splitter.split(df, df["label"], groups=df[group_col])
    )
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def main() -> None:
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    df = pd.read_csv(FEATURES_CSV)

    train_df, test_df = grouped_split(df, "gene", TEST_SIZE, RANDOM_SEED)
    # carve a validation set out of the *training genes* for early stopping
    val_frac = VALID_SIZE / (1 - TEST_SIZE)
    train_df, val_df = grouped_split(train_df, "gene", val_frac, RANDOM_SEED)
    log.info("Split -> train %d / val %d / test %d genes-disjoint",
             len(train_df), len(val_df), len(test_df))

    X_tr, y_tr = train_df[NUMERIC_FEATURES], train_df["label"]
    X_val, y_val = val_df[NUMERIC_FEATURES], val_df["label"]
    X_te, y_te = test_df[NUMERIC_FEATURES], test_df["label"]

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    def report(name, X, y):
        p = model.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        log.info("%-5s  AUROC=%.4f  AUPRC=%.4f", name, auroc, auprc)
        return {"auroc": auroc, "auprc": auprc, "n": int(len(y))}

    metrics = {
        "train": report("train", X_tr, y_tr),
        "valid": report("valid", X_val, y_val),
        "test": report("test", X_te, y_te),
        "best_iteration": int(model.best_iteration),
        "scale_pos_weight": scale_pos_weight,
    }

    model.save_model(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(NUMERIC_FEATURES, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    test_df.to_csv(TEST_CSV, index=False)

    log.info("Saved model -> %s", MODEL_PATH)
    log.info("Saved test set -> %s", TEST_CSV)
    log.info("Saved metrics -> %s", METRICS_PATH)


if __name__ == "__main__":
    main()
