"""Segmentation metrics (pure numpy, no torch) so they're independently testable.

Convention for the degenerate "no lesion" case:
  - both pred and gt empty  -> perfect (1.0)
  - exactly one empty        -> 0.0
This matters because some dermoscopy images legitimately have tiny/empty masks.
"""
import numpy as np


def _binarize(a: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(a) >= threshold).astype(np.uint8)


def dice_coef(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p, g = _binarize(pred, threshold), _binarize(gt, 0.5)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    return float(2.0 * inter / (ps + gs))


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p, g = _binarize(pred, threshold), _binarize(gt, 0.5)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0  # both empty
    inter = np.logical_and(p, g).sum()
    return float(inter / union)


def threshold_jaccard(pred: np.ndarray, gt: np.ndarray,
                      cutoff: float = 0.65, threshold: float = 0.5) -> float:
    """ISIC 2018 official metric: per-image IoU, zeroed when IoU < cutoff."""
    j = iou(pred, gt, threshold)
    return j if j >= cutoff else 0.0


def pixel_sensitivity(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p, g = _binarize(pred, threshold), _binarize(gt, 0.5)
    pos = g.sum()
    if pos == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / pos)


def aggregate(preds, gts, cutoff: float = 0.65, threshold: float = 0.5) -> dict:
    """Mean metrics over a list/iterable of (pred, gt) arrays."""
    dices, ious, tjs, sens = [], [], [], []
    for p, g in zip(preds, gts):
        dices.append(dice_coef(p, g, threshold))
        ious.append(iou(p, g, threshold))
        tjs.append(threshold_jaccard(p, g, cutoff, threshold))
        sens.append(pixel_sensitivity(p, g, threshold))
    n = max(len(dices), 1)
    return {
        "n": len(dices),
        "dice": sum(dices) / n,
        "iou": sum(ious) / n,
        "threshold_jaccard": sum(tjs) / n,
        "pixel_sensitivity": sum(sens) / n,
    }
