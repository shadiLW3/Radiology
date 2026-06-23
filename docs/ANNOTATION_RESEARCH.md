# How clinicians annotate medical images — and what it means for MedVS-AI

_Research synthesis for the MedVS-AI annotation UI. Sources are real and cited; a few
exact numbers are flagged **[medium confidence]** where the source agent couldn't open
the primary PDF. MVP modality = 2D dermoscopy; chest X-ray and CT/MRI come later._

> **NOT FOR CLINICAL USE.** This tool is research/educational.

---

## TL;DR — the decisions this forces

1. **Blind the human first.** Showing an AI answer before the human commits causes *catastrophic*
   anchoring — in a mammography RCT a wrong AI suggestion dropped even very-experienced radiologists
   from 82% → 46% correct, and inexperienced ones from 80% → 20%. Our **draw → lock → reveal** order
   isn't just nice-to-have; it's the thing that makes the head-to-head scientifically honest.
2. **Primary drawing tool = smooth freehand/lasso contour, *not* a click-the-vertices polygon.** Organic
   lesion borders favor smooth contours; polygons punish non-experts and create vertex-density noise.
   Add a round **brush + erase** for cleanup.
3. **Don't expect Dice > 0.9.** Expert-vs-expert Dice on lesion borders is only **~0.75–0.81**, and
   disagreement is *worse on malignant lesions* (fuzzy borders). Calibrate "you beat the AI" against
   that realistic ceiling and surface *"even experts disagree here."*
4. **Score border, not just area.** Report Dice/IoU **plus a boundary metric** (Hausdorff / boundary-F1
   / surface Dice), because border irregularity is the clinically dominant signal.
5. **AI "click-to-segment" (SAM/MedSAM) is opt-in only — never in the blinded competitive read.** A
   model-prefilled mask reintroduces anchoring and makes the contest meaningless.

---

## 1. How clinicians actually annotate (the real workflow + reference tools)

The de-facto standard tools converge on a **small, stable primitive set**. **3D Slicer's Segment
Editor** (the research reference) offers: **Paint** (round brush, drag), **Draw** (trace a freehand
outline / click polygon vertices, right-click to close-and-fill), **Erase**, **Level Tracing**
(semi-auto isocontour), **Scissors** (clip), plus post-processing (smoothing, grow/shrink margin,
keep-largest-island, logical ops). **ITK-SNAP** and **MITK** expose the same effects;
**OHIF/Cornerstone3D**, **CVAT**, **MONAI Label**, and **Supervisely** prove the *entire vocabulary
runs in a browser* (CVAT ships rectangle, polygon, polyline, ellipse, points, brush-mask, and SAM 2/3).
A radiologist's mental model of "annotating" **includes semi-automated assists**, not just dragging a
freehand loop.

**For dermoscopy specifically, the task is uniquely simple:** outline **one closed region = the lesion
border**. There is *no* slice scrolling, *no* DICOM window/level (it's already 8-bit RGB), and *no* 3D.
The clinically meaningful judgement is **border irregularity** (ragged/notched edges, abrupt pigment
cut-off) under the **ABCD rule** (Asymmetry, Border, Color, Diameter > 6 mm), with **pattern analysis**
as the dominant diagnostic method. ISIC's own expert masks were made by exactly three pipelines:
auto + expert review, semi-automatic flood-fill, or manual polygon — i.e. expert "truth" is itself
tool-dependent and noisy.

## 2. The primitive set that matters

| Primitive | What it's for | MVP (2D dermoscopy) | Later (CXR / CT-MRI) |
|---|---|---|---|
| **Freehand / lasso contour** (drag, auto-close) | Trace an organic border | ✅ **primary tool** | ✅ |
| **Round brush + erase** (adjustable radius) | Paint/cleanup a region | ✅ | ✅ |
| **Editable polygon** (click vertices, drag, close) | Precise straight-ish boundaries | ⚠️ optional | ✅ |
| **Bounding box** | Localize a finding | ❌ | ✅ (X-ray) |
| **Ellipse / linear (RECIST)** | Longest-diameter tumor measurement | ❌ (only ABCD "D" if scale known) | ✅ (CT/MRI) |
| **Scribble / seed points** | Steer a region-grow / AI assist | ❌ (assist only) | ✅ |
| **Window/Level (contrast)** | See faint grayscale structure | ❌ (use brightness/contrast slider) | ✅ **mandatory** |
| **Slice scroll / MPR / interpolation** | Navigate + propagate in a volume | ❌ | ✅ **3D only** |

All three MVP primitives **output the same binary lesion mask**, so a single Dice/IoU comparison works
regardless of which tool the human used. Design the MVP around producing **one clean closed mask**, not
a multi-class/multi-object editor.

**Viewer ergonomics to ship in the MVP** (these separate a credible tool from a toy): scroll/pinch
**zoom**, space-drag **pan**, a hover **magnifier loupe**, **brush size on one key/scroll**
(ITK-SNAP uses `-`/`+`), one-key **undo/redo**, a **brightness/contrast/saturation** slider, and
lightweight post-processing (**contour smoothing**, **keep-largest-island**). Keyboard-first for the
high-frequency actions — never bury brush-size or undo in a menu.

**Input devices:** mouse is the floor; **stylus/Apple Pencil should be first-class** (pointer events,
pressure). On a finger/phone, *do not demand pixel-perfect borders* — relax scoring tolerance or steer
touch users to an assisted mode.

## 3. AI-assisted "click-to-segment" — and our policy

The 2025 state of the art is a unified prompt vocabulary — **points (fg/bg), bounding box, scribbles** —
feeding a "propose → correct" loop. **MedSAM** (SAM fine-tuned for 2D medical images) finds a **single
box prompt beats ~10 point clicks**; SAM 2/3 are now in CVAT; **MONAI Label** offers DeepGrow/DeepEdit/
Scribbles; **nnInteractive** (CVPR 2025 winner) does 3D-from-2D-prompts. Zero-shot SAM IoU is wildly
task-dependent (**0.11 spine MRI → 0.87 hip X-ray**), so it's an *accelerator*, not a guarantee.

**Policy for MedVS-AI:** keep an AI-assist out of the **blinded competitive read** entirely (it
re-introduces anchoring and would make "human vs model" meaningless). Reserve it for a **separate
practice/assisted mode** or **post-reveal refinement**, clearly labeled, and prefer a **box prompt**.
A classic **intelligent-scissors / live-wire** edge-snapping contour is a good non-black-box middle tier
for dermoscopy's high-contrast borders if we want assist without a heavy model.

## 4. Ground truth & inter-observer reality → what "fair" scoring means

There is **no single canonical truth**, and the field treats that as a feature:

- **LIDC-IDRI** (4 thoracic radiologists, blinded then unblinded read) deliberately **preserved 4
  separate opinions** rather than forcing consensus.
- **BraTS** uses *annotate → neuroradiologist approver iterates → STAPLE-fuse* to make a defensible
  single truth.
- **STAPLE** (Warfield et al., IEEE TMI 2004) is the canonical EM algorithm for fusing multiple
  segmentations, weighting each rater by estimated sensitivity/specificity.
- **DRIVE/CHASE_DB1** ship **two** observer masks — observer-1 as truth, observer-2 as the human
  benchmark.
- **Dermoscopy inter-observer variability is large and structured:** mean inter-annotator Dice
  ~**0.79 benign / 0.75 malignant**, some images with *zero* overlap; a 12-dermatologist / 77-image
  study found ~**15% mean pairwise XOR** disagreement (>28% in the worst 10%) **[medium confidence on
  exact figures]**; lower agreement is statistically tied to malignancy (p < 0.001). Cohen's κ ~0.67.

**Implication for scoring:** Dice/IoU as the headline, but framed honestly — *"you scored 0.74, within
the range two experts disagree by."* Where multiple annotations exist, **fuse with STAPLE / a soft
probability band** and score against that, optionally weighting *harder* (low-agreement) images more.
Even our own ISIC masks are single, mixed-provenance, and noisy — so don't treat any one mask as
absolute.

**Formats / provenance** (defer the heavy bits, but design for them): masks are either **raster
(DICOM-SEG)** or **vector contours (RTSTRUCT)**; the diagnosis/measurement layer on top is **DICOM SR
(TID 1500/1410)** or **AIM**, which explicitly models *who annotated, with what, and adjudication*.
Capture **provenance with every annotation** (user + expertise tier + tool + timestamp + diagnosis) now;
offering "exports to DICOM-SEG/RTSTRUCT/SR" later is a credibility signal to clinicians.

**Structured diagnosis vocabulary:** dermoscopy → the **ISIC/HAM 7-class taxonomy** (mel, nv, bcc,
akiec, bkl, df, vasc) and/or **ABCD** criteria; chest X-ray/breast → **BI-RADS / Lung-RADS**; oncologic
CT/MRI → **RECIST 1.1**. Real reporting categories beat a free-text guess.

## 5. Gamification precedent (this is not a toy idea)

**DiagnosUs** (lung ultrasound, JMIR 2024): public phone users + ~30-min training + immediate feedback +
a rolling per-user **quality score** (only ≥80% counted). Crowd consensus **87.9%** matched expert mean
**85.0%**; for segmentation, **crowd consensus beat individual experts on Dice (0.755 vs 0.643)**.
Lesson: individuals rarely beat an expert, but the **aggregate** does — so a quality-weighted crowd is
both the engagement loop *and* a real ground-truth data engine. Manage **fatigue**: timed reads make
people *faster but less accurate*, so cap session length and reward accuracy over raw speed.

## 6. Recommended MedVS-AI annotation plan

**MVP (2D dermoscopy) — build:**
- **Draw → Lock → Reveal** flow (blinded; enforce server-side). ✅ already in our design.
- **Freehand/lasso contour (primary) + round brush/erase**, both → one binary mask.
- Viewer: **zoom, pan, magnifier loupe, brightness/contrast slider**, brush-size + undo on hotkeys.
- Post-process: **smoothing + keep-largest-island**.
- Scoring: **Dice + IoU + one boundary metric**; show the **inter-expert context** ("experts ~0.78").
- Diagnosis half: start **benign/malignant + confidence** (our data has it); design the field so it can
  grow into the **7-class taxonomy / ABCD**.
- Capture **provenance** (badge tier, tool used, draw time) per attempt.

**Defer:**
- SAM/MedSAM assist → opt-in practice mode only (never the competitive read).
- Polygon/bbox/ellipse/RECIST, real window/level presets → chest-X-ray phase.
- Slice scrolling, MPR, fill-between-slices/3D propagation, STAPLE-3D, DICOM-SEG/RTSTRUCT/SR export →
  CT/MRI phase. Architect the data model so a "slice" is just the 2D case we already handle.

## Open decisions (these shape the build)

1. **Drawing tools in v1:** freehand-lasso + brush/erase (recommended) — or add an editable polygon too?
2. **Diagnosis granularity:** benign/malignant binary (simplest, matches our ISIC metadata) vs the
   7-class taxonomy vs ABCD-structured.
3. **Ground truth:** we currently have *single* ISIC Task-1 masks (no consensus). Score against that, or
   invest in a multi-annotator/STAPLE consensus later?
4. **"Beat the AI" bar:** absolute Dice, or calibrated against the ~0.78 inter-expert ceiling?
5. **Boundary metric:** which one (Hausdorff95 vs surface-Dice vs boundary-F1)?

## Sources

- 3D Slicer Segment Editor — https://slicer.readthedocs.io/en/v4.11/user_guide/modules/segmenteditor.html
- ISIC multi-annotator / inter-observer (IMA++) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10407008/ · https://arxiv.org/abs/2508.09381
- ABCD rule / dermoscopy — https://dermoscopedia.org/ABCD_rule
- RECIST 1.1 — https://radiologyassistant.nl/more/recist-1-1/recist-1-1-1
- Live-wire / intelligent scissors — https://en.wikipedia.org/wiki/Livewire_Segmentation_Technique
- MedSAM (box-vs-point) — https://www.mathworks.com/help/medical-imaging/ug/get-started-with-medsam-in-medical-image-labeler.html
- nnInteractive — https://github.com/MIC-DKFZ/nnInteractive
- CVAT brush/mask + SAM — https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-brush-tool/
- Automation/anchoring bias RCT (mammography) — https://pubs.rsna.org/doi/10.1148/radiol.222176
- DiagnosUs gamified crowdsourcing — https://www.jmir.org/2024/1/e51397
- Input device (stylus>mouse>finger) — https://www.researchgate.net/publication/256840923
- Reader fatigue — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9975365/
- SAM zero-shot medical range — https://www.sciencedirect.com/science/article/abs/pii/S1361841523001780
- DICOM-SEG (highdicom) — https://highdicom.readthedocs.io/en/latest/seg.html
- AIM model — https://pmc.ncbi.nlm.nih.gov/articles/PMC2837161/
- LIDC-IDRI protocol — https://www.cancerimagingarchive.net/collection/lidc-idri/
- BraTS annotation pipeline — https://pmc.ncbi.nlm.nih.gov/articles/PMC11945730/
- STAPLE (Warfield 2004) — https://pubmed.ncbi.nlm.nih.gov/15250643/
- HAM10000 / ISIC 2018 masks — https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/
- Inter-observer Dice baselines — https://pmc.ncbi.nlm.nih.gov/articles/PMC10709549/
- BI-RADS / Lung-RADS / RECIST — https://pmc.ncbi.nlm.nih.gov/articles/PMC7251936/
