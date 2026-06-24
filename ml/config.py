"""Configuration for the Phase 0 dermoscopy segmentation spike.

Goal of Phase 0 (per docs/DESIGN.md): prove the ML half in isolation BEFORE any
web code. Train one U-Net on ISIC 2018 Task 1, beat a trivial Otsu baseline on a
held-out split, export to ONNX, and verify CPU inference. No web app yet.
"""
import os

# --- Paths -------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "unet.pt")
ONNX_PATH = os.path.join(MODELS_DIR, "unet.onnx")
BASELINE_METRICS = os.path.join(REPORTS_DIR, "baseline_metrics.json")
EVAL_METRICS = os.path.join(REPORTS_DIR, "eval_metrics.json")
TRAIN_CURVE = os.path.join(REPORTS_DIR, "train_curve.png")

# --- Data source (ISIC 2018 Task 1 challenge release) ------------------------
# Canonical S3 mirror. If these 404, see https://challenge.isic-archive.com/data/#2018
ISIC_IMAGES_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2018/"
    "ISIC2018_Task1-2_Training_Input.zip"
)
ISIC_MASKS_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2018/"
    "ISIC2018_Task1_Training_GroundTruth.zip"
)
# ISIC 2018 is governed by CC-BY-NC 4.0 (per-image licenses vary; enrich optionally).
DATASET_LICENSE = "CC-BY-NC-4.0"
DATASET_ATTRIBUTION = (
    "ISIC 2018 Challenge (Codella et al. 2019; Tschandl et al. 2018, HAM10000). "
    "https://challenge.isic-archive.com/"
)
ISIC_API = "https://api.isic-archive.com/api/v2/images/"

# --- Model + training knobs --------------------------------------------------
IMG_SIZE = 256
BASE = 32                             # base channel width of the from-scratch Attention U-Net
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-3
VAL_FRACTION = 0.2
RANDOM_SEED = 42
# Cap images used (None = all ~2594). Lower for a fast first iteration.
MAX_IMAGES = None

# ImageNet normalization (matches the pretrained encoder).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ISIC 2018 official segmentation metric: IoU is zeroed below this cutoff.
THRESH_JACCARD_CUTOFF = 0.65
# Probability threshold to binarize model output.
PRED_THRESHOLD = 0.5


def pick_device():
    """cuda > mps (Apple Silicon) > cpu, resolved at runtime."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
