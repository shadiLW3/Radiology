# 🧬 Variant Pathogenicity Classifier

Predict whether a human genetic variant is **pathogenic** (disease-causing) or
**benign**, using machine learning on public genomics data — and benchmark the
model against an established clinical predictor (CADD).

> **Why this matters.** When a patient's genome is sequenced, clinicians find
> thousands of variants. Deciding which ones actually cause disease is one of
> the central bottlenecks in clinical genetics. This project builds a
> reproducible ML pipeline for that triage problem.

---

## Highlights (what makes this portfolio + research grade)

- **Real clinical data** — labels from [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/),
  functional annotations from [myvariant.info](https://myvariant.info) (CADD,
  conservation, SIFT, PolyPhen, REVEL, gnomAD allele frequency).
- **Leakage-aware evaluation** — train/test split is **grouped by gene**, so no
  gene is in both sets. This is the difference between a model that *memorizes
  disease genes* and one that actually learns variant-level biology. Most naive
  tutorials get this wrong.
- **An honest baseline** — we benchmark against **CADD alone** (a widely used
  published score). The research question: *does combining predictors beat the
  single best one on unseen genes?*
- **Interpretability** — SHAP values explain which features drive predictions.
- **Class-imbalance handling** — `scale_pos_weight` + AUPRC as the early-stop
  metric.

---

## Pipeline

```
download_clinvar.py  →  annotate.py  →  features.py  →  train.py  →  evaluate.py
   labels (ClinVar)     functional      model-ready    XGBoost +     metrics +
                        scores (API)    table          gene-split    plots
```

## Quickstart

```bash
# 1. install deps (use a virtualenv)
python -m venv .venv && source .venv/bin/activate
make setup

# 2. run the whole thing (downloads ~80 MB, annotation step needs internet)
make pipeline
```

Or run a single stage: `make download`, `make annotate`, `make features`,
`make train`, `make evaluate`.

Outputs land in `reports/`:
- `roc_curve.png` — your model vs. CADD-only
- `pr_curve.png`, `confusion_matrix.png`, `shap_summary.png`
- `eval_metrics.json` — AUROC / AUPRC / F1 and the improvement over CADD

## Configuration

Tune `src/config.py`:
- `MAX_VARIANTS` — cap dataset size (default 40k) for faster runs
- `MIN_REVIEW_STARS` — ClinVar confidence filter (default 1 = "criteria provided")
- `TEST_SIZE` / `VALID_SIZE` — gene-grouped split fractions

## Project layout

```
src/
  config.py            # paths + knobs
  utils.py             # logging, HGVS ids, nested-dict extraction
  download_clinvar.py  # step 1: labels
  annotate.py          # step 2: functional scores via myvariant.info
  features.py          # step 3: feature table
  train.py             # step 4: XGBoost, gene-grouped split
  evaluate.py          # step 5: metrics, plots, CADD benchmark
data/    models/    reports/   # generated (gitignored)
```

## The research angle (turn this into a write-up)

A few questions worth investigating once the baseline runs:

1. **Does ensembling beat CADD on gene-disjoint test data, and by how much?**
   Report ΔAUROC with confidence intervals (bootstrap the test set).
2. **Where does it fail?** Slice performance by gene, variant consequence, or
   allele frequency. Are mistakes concentrated in specific gene families?
3. **Generalization across diseases** — train on one disease area, test on
   another (split genes by phenotype).
4. **Feature ablation** — how much does each predictor contribute? (SHAP +
   leave-one-feature-out.)

## Level-ups (roadmap)

- Add conservation (phyloP/phastCons) and more dbNSFP scores.
- Calibrate probabilities (Platt / isotonic) and report calibration curves.
- Compare XGBoost vs. logistic regression vs. a small MLP.
- Wrap the model in a Streamlit app: paste a variant → get a probability + SHAP.
- Cross-validate with `GroupKFold` for tighter performance estimates.

## Data & ethics

ClinVar and gnomAD are public, de-identified aggregate resources. This is a
**research/educational tool** and is **not** validated for clinical use — never
use it to make medical decisions.

## License

MIT (add a LICENSE file if you publish this).
