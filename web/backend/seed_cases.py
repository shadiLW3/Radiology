"""Populate cases for a MODALITY + precompute that modality's model outputs.

  --modality {dermoscopy,chest_xray,...}   default dermoscopy
  --synthetic N    N synthetic cases (no real data; exercises the loop)
  --bundle DIR     real bundle: DIR/images/{id}.png, DIR/gt/{id}.png, DIR/labels.csv
                   (label column may be 'diagnosis' or 'benign_malignant')

Runs the modality's segmentation + (optional) classifier per case. Reseeding one
modality leaves the others intact (cases coexist).
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

import modalities
from db import init_db, get_conn, reset_modality, CASES_DIR

GRID = 256


def _save(case_id, image, gt, model_mask):
    d = os.path.join(CASES_DIR, case_id)
    os.makedirs(d, exist_ok=True)
    image.convert("RGB").resize((GRID, GRID)).save(os.path.join(d, "image.png"))
    Image.fromarray((np.asarray(gt) > 0).astype(np.uint8) * 255).resize(
        (GRID, GRID), Image.NEAREST).save(os.path.join(d, "gt.png"))
    Image.fromarray((model_mask > 0).astype(np.uint8) * 255).save(os.path.join(d, "model.png"))


def _insert(conn, case_id, gt_diag, model_diag, modality):
    d = os.path.join(CASES_DIR, case_id)
    conn.execute(
        "INSERT OR REPLACE INTO cases (case_id, image_path, gt_mask_path, model_mask_path, "
        "gt_diagnosis, model_diagnosis, width, height, modality) VALUES (?,?,?,?,?,?,?,?,?)",
        (case_id, os.path.join(d, "image.png"), os.path.join(d, "gt.png"),
         os.path.join(d, "model.png"), gt_diag, model_diag, GRID, GRID, modality),
    )


def _label_from_row(row):
    for col in ("diagnosis", "benign_malignant", "label"):
        if row.get(col):
            return row[col].strip().lower()
    return None


def seed_synthetic(n, modality):
    from model import predict_mask, predict_diagnosis
    init_db(); reset_modality(modality)
    conn = get_conn()
    diags = modalities.get(modality)["diagnoses"]
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    for i in range(n):
        cid = f"syn_{modality}_{i:04d}"
        cx, cy, r = rng.randint(80, 176), rng.randint(80, 176), rng.randint(40, 80)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        img = (rng.rand(GRID, GRID, 3) * 70).astype(np.uint8)
        img[blob] = np.clip(img[blob] + 150, 0, 255)
        pil = Image.fromarray(img)
        _save(cid, pil, blob.astype(np.uint8), predict_mask(pil, modality))
        _insert(conn, cid, rng.choice(diags), predict_diagnosis(pil, modality), modality)
        if (i + 1) % 5 == 0:
            print(f"  seeded {i + 1}/{n}")
    conn.commit(); conn.close()
    print(f"Done: {n} synthetic '{modality}' cases.")


def seed_bundle(bundle_dir, modality):
    from model import predict_mask, predict_diagnosis
    init_db(); reset_modality(modality)
    valid = set(modalities.get(modality)["diagnoses"])
    labels = {}
    with open(os.path.join(bundle_dir, "labels.csv")) as f:
        for row in csv.DictReader(f):
            labels[row["case_id"]] = _label_from_row(row)
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
        if diag not in valid:
            continue  # skip cases whose label isn't in this modality's vocabulary
        pil = Image.open(os.path.join(img_dir, fn))
        gt = Image.open(gt_path)
        _save(cid, pil, gt, predict_mask(pil, modality))
        _insert(conn, cid, diag, predict_diagnosis(pil, modality), modality)
        n += 1
        if n % 10 == 0:
            print(f"  seeded {n}")
    conn.commit(); conn.close()
    print(f"Done: {n} '{modality}' cases from {bundle_dir}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="dermoscopy")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int, metavar="N")
    g.add_argument("--bundle", type=str, metavar="DIR")
    args = ap.parse_args()
    if not modalities.is_valid(args.modality):
        raise SystemExit(f"unknown modality '{args.modality}'; known: {list(modalities.MODALITIES)}")
    if args.synthetic:
        seed_synthetic(args.synthetic, args.modality)
    else:
        seed_bundle(args.bundle, args.modality)


if __name__ == "__main__":
    main()
