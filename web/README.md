# MedVS-AI — Phase 1 web loop

The minimal **human vs. model** loop: you're shown a dermoscopy lesion, you **draw the lesion**
(lasso or brush) and **pick a diagnosis**, you **lock**, then you see your mask vs. the model's vs.
ground truth — scored with Dice / IoU / Hausdorff95 — and a leaderboard by background.

> **NOT FOR CLINICAL USE.** Research / educational only. No output is a diagnosis.

Design grounded in [docs/ANNOTATION_RESEARCH.md](../docs/ANNOTATION_RESEARCH.md): blinded
draw→lock→reveal (anti-anchoring), freehand-lasso-first drawing, boundary-aware scoring, and
inter-expert context so a 0.78 Dice reads as "as good as two experts agree", not a failure.

## Run it

```bash
cd web/backend
python -m venv .venv && source .venv/bin/activate   # or reuse one
pip install -r requirements.txt

# seed 20 SYNTHETIC cases — runs with zero real data, just to exercise the loop:
python seed_cases.py --synthetic 20

# start the server:
python app.py            # or: uvicorn app:app --reload
# open http://127.0.0.1:8000
```

Requires the Phase 0 model at `ml/models/unet.onnx` (the seed step runs it once per case).

## Loading real ISIC cases (option 1)

The synthetic cases prove the plumbing; for real dermoscopy, run
[`ml/notebooks/phase1_isic_bundle_colab.ipynb`](../ml/notebooks/phase1_isic_bundle_colab.ipynb) in
Colab (no GPU) → it downloads ISIC Task 1 and produces `case_bundle.zip` (~120 images + GT masks +
benign/malignant labels). Then locally:

```bash
cd web/backend
python load_bundle.py ~/Downloads/case_bundle.zip      # unzips + validates -> ../data/bundle
python seed_cases.py --bundle ../data/bundle           # runs the model per case
python app.py
```

(`backend/export_case_bundle_colab.py` is the same export as a paste-in cell, if you'd rather add it
to the Phase 0 notebook.)

## Diagnosis head-to-head — the classifier (option 2)

The U-Net is segmentation-only, so the model **abstains** on diagnosis until you add a benign/malignant
classifier. Train one with [`ml/notebooks/phase1_classifier_colab.ipynb`](../ml/notebooks/phase1_classifier_colab.ipynb)
(T4 GPU, ~10 min on ISIC Task 3 / HAM10000) → download `classifier.onnx`, drop it at
`ml/models/classifier.onnx`, and reseed:

```bash
cd web/backend
python seed_cases.py --bundle ../data/bundle    # or --synthetic 20
python app.py
```

Now the reveal shows the model's benign/malignant call, whether it was right, and whether you agreed.
No classifier present → it just abstains (the segmentation game still works).

## Chest X-ray & other modalities

The platform is **modality-agnostic** via a registry ([`backend/modalities.py`](backend/modalities.py)):
each modality is one declarative entry (label, draw target, model filenames, diagnosis vocabulary) and the
backend/frontend *read the spec* instead of branching on the name. To add **chest X-ray** (lung-field
segmentation + normal/TB):

1. Run [`ml/notebooks/phase4_chest_xray_colab.ipynb`](../ml/notebooks/phase4_chest_xray_colab.ipynb) →
   `unet_cxr.onnx`, `classifier_cxr.onnx`, `case_bundle_cxr.zip`.
2. Put the two ONNX files in `ml/models/`, then:
   ```bash
   python load_bundle.py ~/Downloads/case_bundle_cxr.zip
   # load_bundle prints the exact --bundle path (for CXR it's ../data/bundle/bundle_cxr):
   python seed_cases.py --modality chest_xray --bundle ../data/bundle/bundle_cxr
   ```
3. Reseeding one modality leaves the others intact; the UI's **Modality** picker switches between them.

Adding a 3rd modality = one new entry in `modalities.py` + its models + a bundle. No other code changes.

## Consensus ground truth (STAPLE)

Each locked drawing is stored, and [`backend/consensus.py`](backend/consensus.py) fuses the reference
mask + every human drawing for a case into a probabilistic **consensus truth** via STAPLE (Warfield et
al. 2004), weighting each annotator by its estimated reliability (sensitivity/specificity, learned by EM).
After you lock, the reveal shows your Dice vs. this consensus plus the **inter-annotator agreement** (the
"even experts disagree" floor) — a fairer target than one noisy mask, and it sharpens as more people play.
`GET /api/consensus/{case_id}`.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | the play page |
| GET | `/api/next_case?session_id=` | an unseen case (never returns the model/GT answer) |
| GET | `/cases/{id}/{image\|gt\|model}.png` | case assets |
| POST | `/api/attempt` | score the locked answer + return the reveal |
| GET | `/api/leaderboard` | stats by badge |

## Layout

```
web/
  backend/
    app.py                     # FastAPI app + endpoints
    db.py                      # sqlite schema/helpers (stdlib)
    scoring.py                 # Dice/IoU/threshold-Jaccard + Hausdorff95 + mask decode
    model.py                   # ONNX loader -> predict_mask (used at seed time)
    seed_cases.py              # --synthetic N | --bundle DIR
    load_bundle.py             # unzip+validate a Colab case_bundle.zip
    export_case_bundle_colab.py# the cell to paste into Colab
    requirements.txt
  static/                      # index.html + app.js + style.css (vanilla + Konva CDN)
  data/                        # app.sqlite + cases/  (generated, gitignored)
```

## Known limitations / next

- The diagnosis head-to-head needs `ml/models/classifier.onnx` (see "Diagnosis head-to-head" above);
  without it the model abstains on diagnosis and only the segmentation game runs.
- Deferred (see research doc): SAM-style click-to-segment (opt-in practice mode only), zoom/loupe,
  multi-annotator/STAPLE consensus ground truth, NPI credential badging, 3D modalities.
