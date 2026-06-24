"""Modality registry — the single source of truth for what differs per modality.

Everything else (model loading, diagnosis vocabulary, UI strings, scoring) is
GENERIC and reads a spec from here instead of branching on the modality name.
Adding a modality = add one entry below + drop its ONNX models in ml/models/.

Each spec:
  label        human-readable name (UI)
  draw_target  what the user traces (UI: "trace {draw_target}")
  seg_model    segmentation ONNX filename in ml/models/  (required to play)
  clf_model    diagnosis ONNX filename in ml/models/      (optional; abstains if absent)
  diagnoses    [negative_label, positive_label]; classifier outputs P(positive)
"""

MODALITIES = {
    "dermoscopy": {
        "label": "Dermoscopy (skin lesion)",
        "draw_target": "the lesion border",
        "seg_model": "unet.onnx",
        "clf_model": "classifier.onnx",
        "diagnoses": ["benign", "malignant"],
    },
    "chest_xray": {
        "label": "Chest X-ray (lungs)",
        "draw_target": "the lung fields",
        "seg_model": "unet_cxr.onnx",
        "clf_model": "classifier_cxr.onnx",
        "diagnoses": ["normal", "tb"],
    },
}

DEFAULT_MODALITY = "dermoscopy"


def get(modality):
    """Spec for a modality id, falling back to the default if unknown."""
    return MODALITIES.get(modality) or MODALITIES[DEFAULT_MODALITY]


def is_valid(modality):
    return modality in MODALITIES


def public_list():
    """Registry as a list for the frontend (id + everything it needs to render)."""
    return [
        {"id": mid, "label": s["label"], "draw_target": s["draw_target"],
         "diagnoses": s["diagnoses"]}
        for mid, s in MODALITIES.items()
    ]
