"""Run the exported ONNX model on a single image (CPU). Mirrors how the future
web backend will produce a predicted mask.

Usage: python src/infer.py path/to/lesion.jpg [out_overlay.png]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from config import (ONNX_PATH, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                    PRED_THRESHOLD)
from utils import get_logger, overlay_mask

log = get_logger("infer")


def preprocess(path: str):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return rgb, x.transpose(2, 0, 1)[None]


def predict_mask(path: str):
    import onnxruntime as ort
    rgb, x = preprocess(path)
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"input": x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    return rgb, (prob >= PRED_THRESHOLD).astype(np.uint8)


def main() -> None:
    if len(sys.argv) < 2:
        log.error("usage: python src/infer.py <image> [out.png]")
        sys.exit(1)
    image_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "prediction_overlay.png"
    rgb, mask = predict_mask(image_path)
    cover = 100.0 * mask.mean()
    log.info("Predicted lesion covers %.1f%% of the image", cover)
    Image.fromarray(overlay_mask(rgb, mask)).save(out)
    log.info("Saved overlay -> %s", out)


if __name__ == "__main__":
    main()
