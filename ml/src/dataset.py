"""ISIC segmentation Dataset + train/val split.

Kept independent of segmentation_models_pytorch (uses plain ImageNet
normalization) so it can be exercised without the full model stack.
"""
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image

from config import (IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, VAL_FRACTION,
                    RANDOM_SEED, MAX_IMAGES)


def load_manifest(manifest_csv: str, max_images=MAX_IMAGES) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if max_images:
        df = df.sample(n=min(max_images, len(df)), random_state=RANDOM_SEED)
    return df.reset_index(drop=True)


def split_manifest(df: pd.DataFrame, val_fraction=VAL_FRACTION,
                   seed=RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Image-level random split (ISIC 2018 Task1 has one lesion per image)."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled.iloc[n_val:].reset_index(drop=True), \
        shuffled.iloc[:n_val].reset_index(drop=True)


def _resize(arr: Image.Image, size: int, mode: int) -> Image.Image:
    return arr.resize((size, size), mode)


class ISICDataset:
    """Returns (image[3,H,W] float32, mask[1,H,W] float32) torch tensors."""

    def __init__(self, df: pd.DataFrame, size: int = IMG_SIZE, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.augment = augment
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        img = _resize(img, self.size, Image.BILINEAR)
        mask = _resize(mask, self.size, Image.NEAREST)

        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1].copy()

        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1))          # [3,H,W]
        y = torch.from_numpy(y[None, :, :])                  # [1,H,W]
        return x, y


def raw_pair(row, size: int = IMG_SIZE):
    """Return (rgb_uint8, gt_binary) at `size` — for baseline + overlays."""
    img = Image.open(row["image_path"]).convert("RGB").resize((size, size), Image.BILINEAR)
    mask = Image.open(row["mask_path"]).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8), (np.asarray(mask) > 127).astype(np.uint8)
