"""PASTE THIS INTO A COLAB CELL at the end of phase0_colab.ipynb (after `model`, `val_df`,
and the helper functions exist). It exports ~120 validation cases as a small bundle the
backend can seed from — images + GT masks + benign/malignant labels (fetched from the ISIC
API). The model masks are NOT exported here; the backend recomputes them from unet.onnx.

It produces case_bundle.zip and downloads it. Then locally:
    python load_bundle.py case_bundle.zip
    python seed_cases.py --bundle ../data/bundle
"""

CELL = r'''
import os, io, csv, json, zipfile, urllib.request, shutil
from PIL import Image

N_CASES = 120
OUT = "bundle"
if os.path.isdir(OUT): shutil.rmtree(OUT)
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/gt", exist_ok=True)

def benign_malignant(isic_id):
    # ISIC API now uses diagnosis_1 = "Benign"/"Malignant"/"Indeterminate" (no more benign_malignant)
    url = f"https://api.isic-archive.com/api/v2/images/{isic_id}/"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            meta = json.loads(r.read())
        d1 = ((meta.get("metadata", {}).get("clinical", {}) or {}).get("diagnosis_1") or "").lower()
        return d1 if d1 in ("benign", "malignant") else None
    except Exception:
        return None

rows = []
sub = val_df.head(N_CASES)
for _, r in sub.iterrows():
    cid = r["isic_id"]
    label = benign_malignant(cid)
    if label not in ("benign", "malignant"):
        continue  # skip cases without a clear label
    img = Image.open(r["image_path"]).convert("RGB").resize((256, 256))
    img.save(f"{OUT}/images/{cid}.png")
    gt = Image.open(r["mask_path"]).convert("L").resize((256, 256), Image.NEAREST)
    gt.save(f"{OUT}/gt/{cid}.png")
    rows.append({"case_id": cid, "benign_malignant": label})

with open(f"{OUT}/labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case_id", "benign_malignant"])
    w.writeheader(); w.writerows(rows)

with zipfile.ZipFile("case_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in files:
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, "."))

print(f"exported {len(rows)} labeled cases -> case_bundle.zip")
from google.colab import files
files.download("case_bundle.zip")
'''

if __name__ == "__main__":
    print(CELL)
