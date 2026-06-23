"""Populate the case set + precompute model masks.

Two modes:
  --synthetic N   generate N synthetic cases (no real data needed) so the app runs immediately.
  --bundle DIR    load a real ISIC bundle: DIR/images/{id}.png, DIR/gt/{id}.png, DIR/labels.csv
                  (columns: case_id,benign_malignant). See export_case_bundle_colab.py / load_bundle.py.

For every case we run the U-Net once and store image/gt/model PNGs under web/data/cases/{id}/.
"""
import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image

from db import init_db, get_conn, reset_cases, CASES_DIR

GRID = 256


def _save(case_id, image, gt, model_mask):
    d = os.path.join(CASES_DIR, case_id)
    os.makedirs(d, exist_ok=True)
    image.convert("RGB").resize((GRID, GRID)).save(os.path.join(d, "image.png"))
    Image.fromarray((np.asarray(gt) > 0).astype(np.uint8) * 255).resize(
        (GRID, GRID), Image.NEAREST).save(os.path.join(d, "gt.png"))
    Image.fromarray((model_mask > 0).astype(np.uint8) * 255).save(os.path.join(d, "model.png"))
    return d


def _insert(conn, case_id, gt_diag, model_diag):
    d = os.path.join(CASES_DIR, case_id)
    conn.execute(
        "INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?)",
        (case_id, os.path.join(d, "image.png"), os.path.join(d, "gt.png"),
         os.path.join(d, "model.png"), gt_diag, model_diag, GRID, GRID),
    )


def seed_synthetic(n):
    """Random bright blob on noise; gt = the blob; random benign/malignant label."""
    from model import predict_mask, predict_diagnosis
    init_db(); reset_cases()
    if os.path.isdir(CASES_DIR):
        shutil.rmtree(CASES_DIR)
    conn = get_conn()
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    for i in range(n):
        cid = f"syn_{i:04d}"
        cx, cy, r = rng.randint(80, 176), rng.randint(80, 176), rng.randint(40, 80)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        img = (rng.rand(GRID, GRID, 3) * 70).astype(np.uint8)
        img[blob] = np.clip(img[blob] + 150, 0, 255)
        pil = Image.fromarray(img)
        model_mask = predict_mask(pil)
        _save(cid, pil, blob.astype(np.uint8), model_mask)
        _insert(conn, cid, rng.choice(["benign", "malignant"]), predict_diagnosis(pil))
        if (i + 1) % 5 == 0:
            print(f"  seeded {i + 1}/{n}")
    conn.commit(); conn.close()
    print(f"Done: {n} synthetic cases.")


def seed_bundle(bundle_dir):
    from model import predict_mask, predict_diagnosis
    init_db(); reset_cases()
    labels = {}
    with open(os.path.join(bundle_dir, "labels.csv")) as f:
        for row in csv.DictReader(f):
            labels[row["case_id"]] = row.get("benign_malignant", "").strip().lower()
    conn = get_conn()
    img_dir = os.path.join(bundle_dir, "images")
    gt_dir = os.path.join(bundle_dir, "gt")
    n = 0
    for fn in sorted(os.listdir(img_dir)):
        cid = os.path.splitext(fn)[0]
        gt_path = os.path.join(gt_dir, cid + ".png")
        if not os.path.exists(gt_path):
            continue
        diag = labels.get(cid)
        if diag not in ("benign", "malignant"):
            continue  # skip cases without a clear diagnosis label
        pil = Image.open(os.path.join(img_dir, fn))
        gt = Image.open(gt_path)
        _save(cid, pil, gt, predict_mask(pil))
        _insert(conn, cid, diag, predict_diagnosis(pil))
        n += 1
        if n % 10 == 0:
            print(f"  seeded {n}")
    conn.commit(); conn.close()
    print(f"Done: {n} ISIC cases from {bundle_dir}.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int, metavar="N")
    g.add_argument("--bundle", type=str, metavar="DIR")
    args = ap.parse_args()
    if args.synthetic:
        seed_synthetic(args.synthetic)
    else:
        seed_bundle(args.bundle)


if __name__ == "__main__":
    main()
