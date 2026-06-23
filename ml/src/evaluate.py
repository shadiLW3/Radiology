"""Step 4 — Evaluate the trained U-Net and compare it to the Otsu baseline.

Writes reports/eval_metrics.json (model vs baseline, the Phase 0 GATE) and a
grid of overlay images so you can eyeball that masks line up with lesions.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (MANIFEST_CSV, MODEL_PATH, EVAL_METRICS, BASELINE_METRICS,
                    REPORTS_DIR, IMG_SIZE, THRESH_JACCARD_CUTOFF, PRED_THRESHOLD,
                    pick_device)
from dataset import load_manifest, split_manifest, ISICDataset, raw_pair
from metrics import aggregate
from utils import get_logger, ensure_dir, overlay_mask

log = get_logger("evaluate")


def predict_all(model, val_df, device):
    import torch
    ds = ISICDataset(val_df, IMG_SIZE, augment=False)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(x[None].to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            preds.append((prob >= PRED_THRESHOLD).astype(np.uint8))
            gts.append(y[0].numpy().astype(np.uint8))
    return preds, gts


def save_overlays(val_df, preds, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(val_df))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        rgb, gt = raw_pair(val_df.iloc[i], IMG_SIZE)
        axes[i, 0].imshow(rgb); axes[i, 0].set_title("image")
        axes[i, 1].imshow(overlay_mask(rgb, gt, (0, 255, 0))); axes[i, 1].set_title("ground truth")
        axes[i, 2].imshow(overlay_mask(rgb, preds[i], (255, 0, 0))); axes[i, 2].set_title("model")
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    out = os.path.join(REPORTS_DIR, "overlays.png")
    plt.savefig(out, dpi=110); plt.close()
    log.info("Saved overlays -> %s", out)


def main() -> None:
    import torch
    from train import build_model
    ensure_dir(REPORTS_DIR)
    device = pick_device()

    df = load_manifest(MANIFEST_CSV)
    _, val_df = split_manifest(df)

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    preds, gts = predict_all(model, val_df, device)
    model_metrics = aggregate(preds, gts, THRESH_JACCARD_CUTOFF)
    model_metrics["method"] = "unet"

    baseline = {}
    if os.path.exists(BASELINE_METRICS):
        with open(BASELINE_METRICS) as f:
            baseline = json.load(f)

    result = {"model": model_metrics, "baseline": baseline}
    if baseline:
        result["dice_improvement_over_baseline"] = (
            model_metrics["dice"] - baseline.get("dice", 0.0))
        result["GATE_passed"] = bool(
            model_metrics["dice"] > baseline.get("dice", 0.0))

    log.info("Model   Dice=%.4f  IoU=%.4f  thrJaccard=%.4f",
             model_metrics["dice"], model_metrics["iou"],
             model_metrics["threshold_jaccard"])
    if baseline:
        log.info("Baseline Dice=%.4f  (improvement: %+.4f)  GATE %s",
                 baseline.get("dice", 0.0),
                 result["dice_improvement_over_baseline"],
                 "PASSED" if result["GATE_passed"] else "FAILED")

    with open(EVAL_METRICS, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote -> %s", EVAL_METRICS)
    save_overlays(val_df, preds)


if __name__ == "__main__":
    main()
