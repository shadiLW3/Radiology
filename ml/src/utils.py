"""Shared helpers: logging, seeding, ISIC id parsing, mask overlays."""
import logging
import os
import re
from typing import Optional

import numpy as np

_ISIC_RE = re.compile(r"(ISIC_\d+)")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def isic_id(filename: str) -> Optional[str]:
    """Extract 'ISIC_0000000' from any image/mask filename."""
    m = _ISIC_RE.search(os.path.basename(filename))
    return m.group(1) if m else None


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask over an RGB uint8 image for sanity visualization."""
    img = image.astype(np.float32).copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    m = mask.astype(bool)
    for c in range(3):
        img[..., c][m] = (1 - alpha) * img[..., c][m] + alpha * color[c]
    return np.clip(img, 0, 255).astype(np.uint8)
