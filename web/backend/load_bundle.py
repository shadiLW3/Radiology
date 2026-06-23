"""Unzip + validate a case_bundle.zip exported from Colab, ready for seed_cases --bundle.

Usage: python load_bundle.py path/to/case_bundle.zip [out_dir]
Then:  python seed_cases.py --bundle <out_dir>
"""
import csv
import os
import sys
import zipfile


def main():
    if len(sys.argv) < 2:
        print("usage: python load_bundle.py case_bundle.zip [out_dir]")
        sys.exit(1)
    zip_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "bundle")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    # the zip may contain a top-level 'bundle/' folder; normalize
    root = out_dir
    if os.path.isdir(os.path.join(out_dir, "bundle")) and not os.path.isdir(os.path.join(out_dir, "images")):
        root = os.path.join(out_dir, "bundle")

    images = os.path.join(root, "images")
    gt = os.path.join(root, "gt")
    labels = os.path.join(root, "labels.csv")
    problems = []
    if not os.path.isdir(images): problems.append("missing images/")
    if not os.path.isdir(gt): problems.append("missing gt/")
    if not os.path.exists(labels): problems.append("missing labels.csv")
    if problems:
        print("INVALID bundle:", "; ".join(problems))
        sys.exit(1)

    n_img = len([f for f in os.listdir(images) if f.endswith(".png")])
    n_gt = len([f for f in os.listdir(gt) if f.endswith(".png")])
    with open(labels) as f:
        n_lab = sum(1 for _ in csv.DictReader(f))
    print(f"OK -> {root}")
    print(f"  images: {n_img} | gt: {n_gt} | labels: {n_lab}")
    print(f"Next: python seed_cases.py --bundle {root}")


if __name__ == "__main__":
    main()
