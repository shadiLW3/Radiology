"""Step 3 — Train a U-Net (EfficientNet-B0 encoder) for lesion segmentation.

Loss = Dice + BCE (standard, robust to the mild class imbalance in dermoscopy).
Tracks validation Dice each epoch and saves the best checkpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (MANIFEST_CSV, MODEL_PATH, MODELS_DIR, REPORTS_DIR, TRAIN_CURVE,
                    ENCODER, ENCODER_WEIGHTS, BATCH_SIZE, EPOCHS, LR, IMG_SIZE,
                    RANDOM_SEED, pick_device)
from dataset import load_manifest, split_manifest, ISICDataset
from metrics import dice_coef
from utils import get_logger, ensure_dir, set_seed

log = get_logger("train")


def build_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3, classes=1, activation=None)


def make_loss():
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    return lambda logits, y: dice(logits, y) + bce(logits, y)


def evaluate_dice(model, loader, device) -> float:
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = y.numpy()
            for p, g in zip(probs, gts):
                scores.append(dice_coef(p[0], g[0]))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(RANDOM_SEED)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    device = pick_device()
    log.info("Device: %s | encoder=%s weights=%s", device, ENCODER, ENCODER_WEIGHTS)

    df = load_manifest(MANIFEST_CSV)
    train_df, val_df = split_manifest(df)
    log.info("train=%d  val=%d", len(train_df), len(val_df))

    train_ds = ISICDataset(train_df, IMG_SIZE, augment=True)
    val_ds = ISICDataset(val_df, IMG_SIZE, augment=False)
    nw = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)

    model = build_model().to(device)
    loss_fn = make_loss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = {"train_loss": [], "val_dice": []}
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        tl = float(np.mean(losses)) if losses else 0.0
        vd = evaluate_dice(model, val_dl, device)
        history["train_loss"].append(tl)
        history["val_dice"].append(vd)
        flag = ""
        if vd > best:
            best = vd
            torch.save(model.state_dict(), MODEL_PATH)
            flag = " *saved"
        log.info("epoch %2d/%d  loss=%.4f  val_dice=%.4f%s", epoch, EPOCHS, tl, vd, flag)

    log.info("Best val Dice=%.4f -> %s", best, MODEL_PATH)
    _plot_curve(history)


def _plot_curve(history) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["train_loss"], "C0-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(history["val_dice"], "C1-", label="val Dice")
    ax2.set_ylabel("val Dice", color="C1")
    plt.title("U-Net training")
    fig.tight_layout(); plt.savefig(TRAIN_CURVE, dpi=130); plt.close()
    log.info("Saved curve -> %s", TRAIN_CURVE)


if __name__ == "__main__":
    main()
