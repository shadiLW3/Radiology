"""End-to-end smoke test on tiny SYNTHETIC data — no download, no pretrained
weights, runs on CPU in seconds. Proves the whole pipeline is wired together
(dataset -> train -> evaluate -> ONNX export) before you spend hours on real data.

Run: python src/_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image

from utils import get_logger, ensure_dir

log = get_logger("smoke")
N = 24
SIZE = 64


def make_synthetic(root: str) -> str:
    """Random images with a bright disc; mask = the disc. A U-Net should fit this."""
    img_dir = os.path.join(root, "img")
    msk_dir = os.path.join(root, "msk")
    ensure_dir(img_dir)
    ensure_dir(msk_dir)
    rows = []
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for i in range(N):
        iid = f"ISIC_{i:07d}"
        cx, cy, r = rng.randint(20, 44), rng.randint(20, 44), rng.randint(8, 16)
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        img = (rng.rand(SIZE, SIZE, 3) * 60).astype(np.uint8)
        img[disc] = np.clip(img[disc] + 160, 0, 255)
        ip = os.path.join(img_dir, f"{iid}.jpg")
        mp = os.path.join(msk_dir, f"{iid}_segmentation.png")
        Image.fromarray(img).save(ip)
        Image.fromarray((disc * 255).astype(np.uint8)).save(mp)
        rows.append({"isic_id": iid, "image_path": ip, "mask_path": mp,
                     "license": "SYNTHETIC", "attribution": "synthetic"})
    csv = os.path.join(root, "manifest.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    from dataset import load_manifest, split_manifest, ISICDataset
    from metrics import dice_coef

    tmp = tempfile.mkdtemp(prefix="isic_smoke_")
    log.info("Synthetic data in %s", tmp)
    csv = make_synthetic(tmp)

    df = load_manifest(csv, max_images=None)
    train_df, val_df = split_manifest(df, val_fraction=0.25)
    train_dl = DataLoader(ISICDataset(train_df, SIZE, augment=True), batch_size=8, shuffle=True)
    val_ds = ISICDataset(val_df, SIZE, augment=False)

    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1,
                     decoder_attention_type="scse")
    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        log.info("epoch %d done (loss=%.3f)", epoch + 1, loss.item())

    model.eval()
    with torch.no_grad():
        x, y = val_ds[0]
        prob = torch.sigmoid(model(x[None]))[0, 0].numpy()
    d = dice_coef(prob, y[0].numpy())
    log.info("val[0] Dice after 3 epochs = %.3f", d)

    onnx_path = os.path.join(tmp, "smoke.onnx")
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), onnx_path, opset_version=17)
    import onnxruntime as ort
    ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    log.info("ONNX export + load OK")
    log.info("SMOKE TEST PASSED ✓  (pipeline is wired correctly)")


if __name__ == "__main__":
    main()
