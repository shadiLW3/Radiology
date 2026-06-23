"""Mask scoring for the human-vs-model comparison.

Pure numpy for Dice/IoU/threshold-Jaccard (with the same empty-mask conventions as
ml/src/metrics.py). Hausdorff95 (a boundary metric — the research flagged that border
accuracy, not just area, is the clinically meaningful signal) uses scipy if available
and degrades to None otherwise.
"""
import base64
import io

import numpy as np

GRID = 256  # all masks are compared on a 256x256 grid


def _binarize(a, threshold=0.5):
    return (np.asarray(a) >= threshold).astype(np.uint8)


def dice_coef(pred, gt, threshold=0.5):
    p, g = _binarize(pred, threshold), _binarize(gt, 0.5)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / (ps + gs))


def iou(pred, gt, threshold=0.5):
    p, g = _binarize(pred, threshold), _binarize(gt, 0.5)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def threshold_jaccard(pred, gt, cutoff=0.65, threshold=0.5):
    j = iou(pred, gt, threshold)
    return j if j >= cutoff else 0.0


def hausdorff95(pred, gt, threshold=0.5):
    """Symmetric 95th-percentile boundary distance, in pixels (256 grid).

    Returns 0.0 when both masks are empty, None when exactly one is empty (undefined)
    or scipy is unavailable.
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        return None
    p, g = _binarize(pred, threshold).astype(bool), _binarize(gt, 0.5).astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return None
    pb = p & ~binary_erosion(p)
    gb = g & ~binary_erosion(g)
    d_to_g = distance_transform_edt(~gb)
    d_to_p = distance_transform_edt(~pb)
    dists = np.concatenate([d_to_g[pb], d_to_p[gb]])
    return float(np.percentile(dists, 95))


def all_metrics(pred, gt):
    return {
        "dice": round(dice_coef(pred, gt), 4),
        "iou": round(iou(pred, gt), 4),
        "threshold_jaccard": round(threshold_jaccard(pred, gt), 4),
        "hausdorff95": (round(h, 2) if (h := hausdorff95(pred, gt)) is not None else None),
    }


def decode_mask_png(b64, size=GRID):
    """Decode a base64 PNG (optionally a data URL) of the user's drawing layer into a
    binary [size, size] mask. Any painted (non-transparent) pixel counts as lesion.
    """
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    arr = np.asarray(img)
    alpha = arr[..., 3]
    rgb_painted = arr[..., :3].sum(axis=2) > 0
    return ((alpha > 10) | rgb_painted).astype(np.uint8)
