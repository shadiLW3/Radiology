"""Step 5 — Export the trained U-Net to ONNX and verify CPU inference.

This is the contract with the (future) web backend: the platform will serve
predictions via onnxruntime on CPU, never a live GPU. We export and then check
that ONNX output matches PyTorch within tolerance.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import MODEL_PATH, ONNX_PATH, MODELS_DIR, IMG_SIZE
from utils import get_logger, ensure_dir

log = get_logger("export")


def main() -> None:
    import torch
    from train import build_model
    ensure_dir(MODELS_DIR)

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("Exported -> %s", ONNX_PATH)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    log.info("Max |torch - onnx| = %.2e", max_diff)
    if max_diff < 1e-3:
        log.info("CPU inference verified ✓")
    else:
        log.warning("ONNX output diverges from PyTorch (%.2e) — investigate", max_diff)


if __name__ == "__main__":
    main()
