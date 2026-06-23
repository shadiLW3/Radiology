"""ONNX wrappers, both precomputed at SEED time (live play does no inference):

  - U-Net SEGMENTATION  -> predict_mask  (ml/models/unet.onnx, required)
  - benign/malignant CLASSIFIER -> predict_diagnosis (ml/models/classifier.onnx, OPTIONAL)

If classifier.onnx is absent, predict_diagnosis returns None and the model simply
abstains on the diagnosis channel (the segmentation head-to-head still works).
"""
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "ml", "models"))
ONNX_PATH = os.path.join(MODELS_DIR, "unet.onnx")
CLF_ONNX_PATH = os.path.join(MODELS_DIR, "classifier.onnx")

IMG_SIZE = 256
MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)

_seg = None
_clf = None


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _preprocess(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # [1,3,256,256]


def predict_mask(pil_image, threshold=0.5):
    """PIL RGB -> binary uint8 [256,256] lesion mask."""
    global _seg
    if _seg is None:
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                f"Model not found at {ONNX_PATH}. Run the Phase 0 spike / Colab export first."
            )
        _seg = _session(ONNX_PATH)
    x = _preprocess(pil_image)
    logits = _seg.run(None, {_seg.get_inputs()[0].name: x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    return (prob >= threshold).astype(np.uint8)


def has_classifier():
    return os.path.exists(CLF_ONNX_PATH)


def predict_diagnosis(pil_image, threshold=0.5):
    """PIL RGB -> 'benign' | 'malignant', or None if no classifier is installed.

    Expects a single-logit classifier where sigmoid(logit) = P(malignant).
    """
    global _clf
    if not has_classifier():
        return None
    if _clf is None:
        _clf = _session(CLF_ONNX_PATH)
    x = _preprocess(pil_image)
    out = _clf.run(None, {_clf.get_inputs()[0].name: x})[0]
    logit = float(np.asarray(out).flatten()[0])
    p = 1.0 / (1.0 + np.exp(-logit))
    return "malignant" if p >= threshold else "benign"
