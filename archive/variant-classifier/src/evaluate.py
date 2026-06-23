"""Step 5 — Evaluate the model and benchmark it against CADD alone.

Produces, in `reports/`:
  - roc_curve.png      (our model vs. CADD-only baseline)
  - pr_curve.png
  - confusion_matrix.png
  - shap_summary.png   (feature attributions, if `shap` is installed)
  - eval_metrics.json

The headline research result is the ROC/PR comparison: does combining
predictors beat the single best established score (CADD) on a gene-disjoint
test set?
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, confusion_matrix,
                             f1_score)
import xgboost as xgb

from config import (TEST_CSV, MODEL_PATH, FEATURE_COLS_PATH, REPORTS_DIR)
from utils import get_logger, ensure_dir

log = get_logger("evaluate")


def load():
    with open(FEATURE_COLS_PATH) as f:
        features = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    test = pd.read_csv(TEST_CSV)
    return model, features, test


def plot_roc(y, p_model, p_cadd, path):
    plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y, p_model)
    plt.plot(fpr, tpr, label=f"Our model (AUROC={roc_auc_score(y, p_model):.3f})")
    mask = ~np.isnan(p_cadd)
    if mask.sum() > 0:
        fpr_c, tpr_c, _ = roc_curve(y[mask], p_cadd[mask])
        plt.plot(fpr_c, tpr_c, "--",
                 label=f"CADD only (AUROC={roc_auc_score(y[mask], p_cadd[mask]):.3f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — pathogenicity classification"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_pr(y, p_model, path):
    prec, rec, _ = precision_recall_curve(y, p_model)
    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, label=f"AUPRC={average_precision_score(y, p_model):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Pathogenic"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Pathogenic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (threshold 0.5)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_shap(model, X, path):
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP plot")
        return
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    shap.summary_plot(values, X, show=False)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main() -> None:
    ensure_dir(REPORTS_DIR)
    model, features, test = load()
    X = test[features]
    y = test["label"].values
    p_model = model.predict_proba(X)[:, 1]
    pred = (p_model >= 0.5).astype(int)
    p_cadd = test["cadd_phred"].values.astype(float)

    auroc = roc_auc_score(y, p_model)
    auprc = average_precision_score(y, p_model)
    f1 = f1_score(y, pred)
    mask = ~np.isnan(p_cadd)
    cadd_auroc = roc_auc_score(y[mask], p_cadd[mask]) if mask.sum() else None

    metrics = {
        "model_auroc": auroc, "model_auprc": auprc, "model_f1": f1,
        "cadd_only_auroc": cadd_auroc,
        "improvement_over_cadd": (auroc - cadd_auroc) if cadd_auroc else None,
        "n_test": int(len(y)),
    }
    log.info("Model AUROC=%.4f  AUPRC=%.4f  F1=%.4f", auroc, auprc, f1)
    if cadd_auroc is not None:
        log.info("CADD-only AUROC=%.4f  (model improvement: %+.4f)",
                 cadd_auroc, auroc - cadd_auroc)

    plot_roc(y, p_model, p_cadd, os.path.join(REPORTS_DIR, "roc_curve.png"))
    plot_pr(y, p_model, os.path.join(REPORTS_DIR, "pr_curve.png"))
    plot_confusion(y, pred, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plot_shap(model, X, os.path.join(REPORTS_DIR, "shap_summary.png"))

    with open(os.path.join(REPORTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote plots + metrics to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
