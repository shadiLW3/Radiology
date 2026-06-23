# Phase 0 — Dermoscopy Segmentation Spike

The **offline ML derisking step** for [MedVS-AI](../docs/DESIGN.md). Before any web
code exists, this proves the core question: *can a U-Net segment skin lesions well
enough to be worth building a platform around?*

> **NOT FOR CLINICAL USE.** Research/educational only. No output is a diagnosis.

## The gate

Train one U-Net on **ISIC 2018 Task 1** and beat a trivial **Otsu baseline** on a
held-out split, with an honestly reported single-fold operating point. If it can't
clear that bar, fix the ML before building anything else.

## Pipeline

```
download_isic.py → baseline.py → train.py → evaluate.py → export_onnx.py
  images+masks      Otsu floor    U-Net      model vs        ONNX + CPU
  + license CSV                   (smp)      baseline GATE   verify
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make setup

# 1. prove the wiring in seconds on synthetic data (no download):
make smoke

# 2. run the real spike (downloads ISIC; input zip is ~10 GB):
make spike
```

Or step by step: `make download` → `make baseline` → `make train` → `make evaluate` → `make export`.

**No local GPU?** Use [`notebooks/phase0_colab.ipynb`](notebooks/phase0_colab.ipynb) — open it in
[Google Colab](https://colab.research.google.com/), set a T4 GPU runtime, and Run All. It runs the
same spike end-to-end on a free GPU and lets you download the trained `unet.onnx` back into `models/`.

Run inference with the exported model:
```bash
python src/infer.py path/to/lesion.jpg overlay.png
```

## Outputs (`reports/`)

- `baseline_metrics.json` — Otsu Dice/IoU/threshold-Jaccard
- `eval_metrics.json` — **model vs baseline + `GATE_passed`**
- `train_curve.png`, `overlays.png` — training curve and side-by-side sanity overlays

## Configuration (`config.py`)

- `MAX_IMAGES` — cap dataset size for a fast first iteration (default: all ~2594)
- `IMG_SIZE` (256), `ENCODER` (`efficientnet-b0`), `EPOCHS` (25), `BATCH_SIZE` (16), `LR`
- `ENCODER_WEIGHTS=None` to train from scratch
- Device auto-detects CUDA → Apple MPS → CPU

## Layout

```
ml/
  config.py
  src/
    download_isic.py  # data + per-image license manifest
    dataset.py        # ISIC Dataset + image-level split
    baseline.py       # Otsu floor
    train.py          # U-Net (segmentation_models_pytorch), Dice+BCE
    evaluate.py       # metrics vs baseline + overlays (the GATE)
    export_onnx.py    # ONNX export + CPU-inference verification
    infer.py          # single-image inference CLI
    metrics.py        # Dice / IoU / threshold-Jaccard (pure numpy)
    utils.py          # logging, seeding, overlays
    _smoke.py         # synthetic end-to-end test
```

## Data & licensing notes

ISIC 2018 is **CC-BY-NC 4.0** with a per-image license mix; `download_isic.py`
records license + attribution per image (and `--enrich-licenses` fetches exact
per-image terms from the ISIC API). The non-commercial terms apply to anything
built on it — see [docs/DESIGN.md](../docs/DESIGN.md) §7.

## Next (Phase 1)

Once the gate passes: precompute predictions for ~100 cases and build the minimal
web loop (FastAPI + react-konva, SQLite, self-reported badge) — diagnose + draw →
lock → reveal model vs you. See the roadmap in [docs/DESIGN.md](../docs/DESIGN.md) §6.
