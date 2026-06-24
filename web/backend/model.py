"""ONNX serving — generic over modality (reads the registry, never branches on it).

predict_mask / predict_diagnosis take a modality id, look up its model filenames in
modalities.py, and lazy-load + cache one InferenceSession per file. Both are called
at SEED time only (live play is a DB/file read).
"""
import os

import numpy as np

import modalities

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "ml", "models"))

IMG_SIZE = 256
MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)

_sessions = {}  # filename -> ort.InferenceSession


def _session(filename):
    """Lazy-load + cache an ONNX session by filename; None if the file is absent."""
    if filename not in _sessions:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            _sessions[filename] = None
        else:
            import onnxruntime as ort
            _sessions[filename] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sessions[filename]


def _preprocess(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # [1,3,256,256]


def predict_mask(pil_image, modality, threshold=0.5):
    """PIL image + modality -> binary uint8 [256,256] segmentation mask."""
    spec = modalities.get(modality)
    sess = _session(spec["seg_model"])
    if sess is None:
        raise FileNotFoundError(
            f"Segmentation model {spec['seg_model']} missing in {MODELS_DIR} for '{modality}'.")
    x = _preprocess(pil_image)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    return (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(np.uint8)


def predict_diagnosis(pil_image, modality, threshold=0.5):
    """PIL image + modality -> the modality's positive/negative diagnosis label,
    or None if that modality's classifier isn't installed (model abstains)."""
    spec = modalities.get(modality)
    sess = _session(spec["clf_model"])
    if sess is None:
        return None
    x = _preprocess(pil_image)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    p = 1.0 / (1.0 + np.exp(-float(np.asarray(out).flatten()[0])))  # P(positive)
    neg, pos = spec["diagnoses"]
    return pos if p >= threshold else neg
