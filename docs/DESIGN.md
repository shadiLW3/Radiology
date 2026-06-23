# MedVS-AI — Design Document

**A research/educational platform where anyone can compete against per-modality deep-learning models on medical-image diagnosis and region annotation, with human performance stratified by NPI-verified expertise.**

> **NOT FOR CLINICAL USE.** This is a portfolio/research tool. No output is a diagnosis. This disclaimer is non-negotiable and must persist across every screen, the API docs, and the README.

---

## 1. Vision & Elevator Pitch

MedVS-AI is an open-access web platform that turns medical-image AI from a spectator sport into a competition. A user is shown a real medical image, submits a **diagnosis** (present/absent or class) and **draws the finding** (a segmentation mask), then sees their answer scored head-to-head against a deep-learning model *and* the expert ground truth. Anyone can play, but every account carries a credential **badge** — *layperson*, *MD*, or *radiologist* (specialty-aware) — verified where possible against the free public US NPPES NPI Registry. Because each modality gets a deliberately **different model architecture** (a 2D transfer-learned U-Net for chest X-ray, a 3D nnU-Net for brain MRI, a 3D detection framework for lung CT, a patch-trained U-Net for retinal vessels), the platform doubles as a living illustration of *why one architecture never fits all of medical imaging*. The research payload: a badge-stratified, statistically rigorous answer to "**does the model agree with experts as well as experts agree with each other — and where does it beat or lose to humans?**"

---

## 2. Per-Modality Model Matrix

The core architectural thesis: **the right model is a function of the data's dimensionality, the target's sparsity/shape, and the clinical readout.** Below, each modality earns a genuinely distinct architecture and training regime — not cosmetic variation.

| Modality | ML Task | Recommended Model | Primary Dataset (license) | Headline Metric | Annotation Style |
|---|---|---|---|---|---|
| **Chest X-ray** (2D grayscale) | Binary segmentation (pneumothorax) + present/absent diagnosis | **2D U-Net, EfficientNet-B4 encoder** (ImageNet-pretrained, via `segmentation_models_pytorch`); Dice+BCE/focal loss | SIIM-ACR Pneumothorax (research/edu + attribution; *images* are NIH = no-restriction) | **Dice** (empty-mask = 1.0 convention); diagnosis: sensitivity/AUROC | 2D canvas brush/polygon + brightness/contrast sliders |
| **Brain MRI** (3D, 4-channel) | Multi-class 3D semantic segmentation (WT/TC/ET nested regions) | **3D U-Net via nnU-Net `3d_fullres`** (residual encoder, deep supervision, Dice+CE); SegResNet fallback | BraTS 2021 (**CC BY 4.0**, TCIA) | **Dice + HD95 per region**; lesion-wise Dice if using 2024 data | Volumetric slice-scroll voxel painting (NiiVue), GrowCut-assisted |
| **Lung CT** (3D volumetric) | **3D object detection** (nodule finding) + auxiliary mask | **nnDetection / Retina U-Net** (3D RetinaNet + U-Net FPN, focal loss, aux seg head) | LUNA16 (**CC BY 4.0**); LIDC-IDRI (**CC BY 3.0**) for boundary GT | **FROC / CPM** (SOTA ref ~0.93) | Axial scroll + click-to-detect; local brush-contour for found nodules (Cornerstone3D) |
| **Dermoscopy** (2D RGB) — *flagship/MVP* | Lesion segmentation **+** benign/malignant classification | **U-Net (EfficientNet-B0 encoder)** for seg **+ separate EfficientNet-B0 classifier** | ISIC 2018 Task 1 + Task 3/HAM10000 (**CC-BY-NC**, per-image mix) | **Threshold-Jaccard** (IoU<0.65→0); class: balanced acc / melanoma sensitivity | 2D polygon/brush blob (react-konva or Cornerstone3D) |
| **Retinal fundus** (2D RGB) | Thin-structure vessel segmentation | **Shallow 2D U-Net trained from scratch on 48×48 patches** (deliberately *not* transfer-learned) | **CHASE_DB1 (CC-BY 4.0 — preferred)**; DRIVE/STARE research-only (flagged) | Dice/F1 + **clDice** (topology); evaluate inside FOV only | Bounded-patch brush (don't trace whole tree) |

**The case for per-modality diversity, made concrete:**
- **2D vs 3D is non-negotiable.** A pneumothorax is one flat image; a glioma's sub-compartments are defined by signal *contrast across four co-registered sequences* and *continuity across slices* — a 2D RGB CNN literally cannot represent the input. Brain MRI is the one place a heavy 3D multi-channel network earns its cost.
- **Segmentation vs detection.** Lung nodules are tiny (3–30 mm) sparse targets in a ~30M-voxel volume with extreme background dominance; the LUNA16 reference standard and FROC scoring are *detection-based* (center-within-radius), and focal-loss detection handles the imbalance that voxel Dice handles poorly. This is the one modality where a detection framework is correct.
- **Same skeleton, opposite training regime.** Skin uses a *pretrained encoder on whole resized images* (lesions are large, ImageNet features transfer). Retina uses a *from-scratch patch-trained* U-Net (20 images → thousands of 48×48 tiles; the net needs local ridge detection, not ImageNet semantics). Identical U-Net family, fundamentally different recipe — the clearest teaching example on the platform.
- **Transfer learning is what makes it solo-dev-feasible.** Pretrained encoders converge in a few epochs on a single free GPU; nnU-Net/nnDetection self-configure spacing/normalization/patch-size, removing the #1 solo-dev failure mode.

**Build order (compute-cheapest first):** Dermoscopy (laptop/Colab, <2 hrs) → Chest X-ray (Kaggle T4, a weekend) → Lung CT and Brain MRI (heaviest; ~1–2 GPU-days each, precompute all model masks offline).

---

## 3. Credential-Badging System

### Three tiers — *labels, never gates*

Every account can fully diagnose and annotate. The badge is purely additive metadata for stratifying the leaderboard.

1. **Layperson** (default at signup). Sub-states for honesty: *Unverified* (no NPI) and *Self-reported clinician* (toggled but not NPI-verified — rendered visually distinct, lower trust).
2. **MD / Physician** — NPI record is type **NPI-1** (individual), status **"A"** (active), **and ≥1 taxonomy code starts with `207` or `208`** (NUCC "Allopathic & Osteopathic Physicians" grouping), **and** the entered name reasonably matches `basic.first_name`/`last_name`.
3. **Specialist (specialty-aware, strict subset of MD)** — among that physician's taxonomies, a code prefix matches the modality's specialty set. **Crucial correction from the research:** dermoscopy and fundus are *not radiology* — badge users as the **matching specialist for the modality they compete on**:
   - **Radiologist:** `2085*` (e.g. `2085R0202X` Diagnostic Radiology, `2085R0204X` Vascular/IR, `2085N0700X` Neuroradiology, `2085B0100X` Body Imaging…) → for CXR / CT / MRI.
   - **Dermatologist:** `207N00000X` → for skin.
   - **Ophthalmologist:** `207W00000X` → for retina.
   - Optionally split *diagnostic-imaging* radiologists from *therapeutic/physics* roles (`2085R0001X`/`2085R0203X`/`2085R0205X`) when stratifying segmentation.

### Verification flow

1. Signup → badge = `layperson`, `npi_verified = false`. User plays immediately.
2. Optional **"Verify clinician status"** CTA: enter NPI (10 digits) + name. Client-side validate the **NPI Luhn check digit** (prefix `80840`) to reject typos before any API call.
3. **Server-side** call (never from the browser — keeps the user's IP off CMS, enables caching/throttling):
   `GET https://npiregistry.cms.hhs.gov/api/?version=2.1&number=<NPI>&enumeration_type=NPI-1`
   No API key. `version=2.1` is **mandatory** (omitting it errors).
4. Parse `results[0]`: check `result_count`, `basic.status == "A"`; normalize-and-compare name; iterate `taxonomies[]` matching on **`code`** (never `desc` — `desc` can be `null` for codes like `2085N0700X`; `basic.credential` is inconsistent free text, used only as a weak corroborator).
5. Assign badge enum (`specialist > md > layperson`), persist `verified_at`, matched taxonomy codes JSON snapshot, and `name_match_score`. Provide a **re-verify** button (NPPES updates ~weekly).

### NPI API facts (verified live)
- Base: `https://npiregistry.cms.hhs.gov/api/` · no auth · `version=2.1` required · `limit` 1–1200 (default 10) · no published rate limit (best-effort; cache ~30 days).
- Authoritative field is `taxonomies[].code`; `enumeration_type` separates NPI-1 (person) from NPI-2 (org).

### Honest limitations (state these in-UI)
- **NPPES does not authenticate identity.** Anyone can type a real radiologist's *public* NPI. The badge means *"self-asserted, name-matched to a real NPI,"* never *"verified identity."*
- NPPES does **not** confirm active licensure or good standing; taxonomy is **self-selected** (can be stale/wrong).
- Many legitimate experts (students, residents, foreign-trained) **have no NPI** — tiers must degrade gracefully to self-reported.
- **Tiered trust + optional friction:** mark auto-checks as *"NPI-matched"* (medium trust) vs an optional *"manually reviewed"* (license/board-cert upload, or .edu/.gov/hospital-email challenge) for the specialist tier (high trust); show the distinction on the leaderboard. Run **verified-only sensitivity analyses** and statistically down-weight/flag outliers rather than trusting any badge absolutely.

### Privacy
NPI data is public but PII-adjacent. **Data minimization:** persist only badge enum, matched taxonomy codes, `verified_at`, `name_match_score`, and at most NPI **last-4 or a salted hash**; discard returned addresses/phone/email immediately. Never display NPI/license/address publicly — only the badge label. Consent copy at the NPI form; right-to-delete reverts to layperson and cascade-deletes the snapshot. NPPES is US-only — non-US clinicians must not be penalized in UX.

---

## 4. The Human-vs-Model Comparison Engine

Two independent scoring channels, captured as **separate explicit steps** so "didn't see it" is never conflated with "couldn't outline it." The human's diagnosis and mask are **locked before the model/GT answer is revealed** (anti-anchoring), and a user never re-sees the same image (learning effect).

### Channel A — Diagnosis agreement
- Capture: fixed dataset-specific ontology (radio buttons matching the model's classes) + a **required confidence** (0–100 slider or 5-point ordinal — needed for ROC) + a distinct *"cannot assess"* escape hatch + start/submit timestamps.
- Score: `diag_user_vs_gt`, `diag_model_vs_gt`, `diag_user_vs_model`.
- Metrics: balanced accuracy, **sensitivity/specificity with 95% CIs**, Cohen's κ (quadratic-weighted for ordinal), **AUROC/AUPRC**, MCC. Lead with sensitivity/specificity/AUROC; caveat that PPV/NPV don't transfer to clinical base rates.

### Channel B — Complex region annotation
- Score the human mask against **both** the model output **and** ground truth, with identical metrics, so the leaderboard can independently report *human-vs-GT*, *model-vs-GT*, and *human-vs-model*.
- Mask metrics: **Dice** (primary, matches U-Net loss) + IoU (note: deterministic function of Dice, not independent evidence) + **HD95** + **ASSD** (spacing-aware, in **mm** for 3D) + **NSD** + pixel sensitivity/precision (exposes over/under-segmentation Dice hides) + per-lesion F1/FROC for multifocal/detection tasks.
- Modality-specific metric traps: **threshold-Jaccard** for skin (not raw Dice); **lesion-wise** Dice/HD95 for BraTS-2024; **FROC/CPM** for lung CT; evaluate retina **inside the FOV mask** only.

### Empty-mask & coordinate discipline (the silent score-corrupters)
- **Empty-mask math:** correct "no finding" (empty GT + empty human) must score as a *success* (Volumetric Similarity = 1 or a defined convention); distinguish *"annotated nothing"* from *"did not annotate."* Special-case the empty-GT Dice=1.0 convention everywhere (78% of SIIM CXRs, frequently-absent ET in brain).
- **Never store screen/canvas pixels.** Persist masks in image/voxel space + the affine (IPP, direction cosines, spacing). Resample human, model, and GT masks onto the **GT's native grid** before any metric — a silent resize mismatch inflates/deflates every score.

### Expertise stratification (the whole point)
- Badge is a fixed grouping variable snapshotted **at time of submission** (`badge_at_time`) so later verification can't retroactively rewrite the leaderboard. Always render the *self-reported/unverified* cohort separately.
- Report **every** metric per badge with per-stratum 95% CIs (never pooled-only — Simpson's paradox). Test the **expertise gradient** (Cochran-Armitage for binary, Jonckheere-Terpstra for continuous Dice). Report the human-vs-model gap *separately per badge* — "the model beats laypeople but not radiologists" is the headline contrast.

### Statistics — treat it as a reader study, not a bake-off
You have **crossed structure** (many users × many images), so naive per-submission tests violate independence and overstate significance.
- **Core model:** mixed-effects / GEE with **crossed random intercepts for user and image**, fixed effects for rater-type and badge (mixed logistic for diagnosis; LMM for Dice/HD).
- **ROC comparison:** compare the model curve to a human *point* at a **matched operating point** (model sensitivity at the human's specificity); DeLong for two AUROCs on the same cases; **Obuchowski-Rockette-Hillis MRMC** (R `MRMCaov`) for the formal "readers-as-group vs model, generalizing to new readers *and* cases."
- **Inter-rater reliability:** Krippendorff's α (tolerates the missing-data design open participation guarantees); **ICC** for continuous mask agreement — the key claim *"the model agrees with experts as well as experts agree with each other"* is **human-model ICC vs human-human ICC**.
- **Uncertainty:** **cluster/hierarchical bootstrap** (resample images, then readers within — never flat resample submissions), 1k–10k resamples.
- **Hygiene:** pre-register the primary endpoint and inclusion rules (min time, non-degenerate mask, diagnosis present); correct multiplicity (Holm/BH) across the metric×badge grid; for "as good as experts" use **equivalence/TOST**, not a non-significant difference test.
- **Ground truth is the weakest link:** public GT is itself observer-variable. Use **multi-annotator consensus** (STAPLE/majority — LIDC 4-reader, ISIC IMA++, DRIVE/CHASE second observers) and report **inter-expert variability as the noise floor**; frame model performance relative to that floor, and surface it in-UI as a "how hard was this one" indicator so a human legitimately beating imperfect GT reads correctly.

---

## 5. System Architecture, Data Model & Repo Structure

### Stack
- **Backend:** FastAPI (Python — same language as the ML pipeline, models import cleanly). SQLAlchemy + Pydantic + Alembic. JWT/session auth.
- **Frontend:** React + Vite + TypeScript SPA. **Annotation engine: Cornerstone3D** (MIT) as the single engine for *both* 2D and 3D — BrushTool / PlanarFreehandROI (polygon) / RectangleROI, native DICOM-SEG + NIfTI I/O, world↔voxel transforms handled. Use **react-konva** *only* for the simplest PNG-only 2D path. Use **NiiVue** for the 3D brain MRI experience (sequence blending + GrowCut). **Borrow Cornerstone3D, do not self-host OHIF** (too heavy for a timed game loop). Tailwind for styling. Persistent "NOT for clinical use" banner.
- **Model serving:** export every model to **ONNX** (`torch.onnx.export`), run via **onnxruntime on CPU** (TorchScript fallback). `model_server.py` keeps an **LRU cache** of `InferenceSession`s keyed by `model_id`, lazy-loaded from `artifact_uri`; default `CPUExecutionProvider`, swap to CUDA EP if present. **Precompute all predictions offline** for the seed case set (live play = a DB read, never GPU inference); tiny ~50ms-coalescing async batch queue for on-demand new cases.
- **Ground-truth prep (offline only):** MONAI Label / 3D Slicer or Label Studio Community — for building consensus reference masks, never the live game UI.
- **Scoring (server-side, Python):** MONAI metrics (Dice/Hausdorff/SurfaceDistance — spacing-aware), scikit-learn (κ/sensitivity/AUROC), pingouin/statsmodels/scipy (ICC, GEE, mixed models, bootstrap).
- **Storage:** images/masks in object storage with **free egress** (Cloudflare R2 or Backblaze B2; or HF Datasets) — DB stores only URIs. Masks as single-channel 8-bit PNG (matching ISIC 0/255) or compact RLE + optional polygon GeoJSON, **always at native case resolution**. SQLite (Phase 0/1) → Postgres (Neon/Supabase, Phase 2+; JSONB + materialized leaderboard view).

### Data model (key entities)
`users` · `credentials` (claimed_badge, npi_last4/hash, verification_status, verified_taxonomy_code, verified_at, raw snapshot for audit) · `modalities` (task_type, default_model) · `models` (architecture, framework, artifact_uri, input_h/w, normalization, class_labels) · `cases` (**mandatory `license` + `attribution`**, gt_diagnosis_label/code, gt_mask_uri) · `model_predictions` (`UNIQUE(case_id, model_id)`, precomputed) · `attempts` (groups one diagnosis + one annotation per user×case×model, enabling re-scoring after retrain) · `diagnoses` (chosen_label, confidence_self, response_ms) · `human_annotations` (mask + polygon, tool, draw_ms) · `comparison_results` (**`badge_at_time` snapshotted**, all diag + Dice/IoU fields, composite_score) · `leaderboard_view` (materialized, stratified by badge × modality).

### Repo structure
```
medvs-ai/
  README.md            # prominent "NOT for clinical use"
  ml/                  # standalone; download → dataset → train → baseline → export_onnx → infer → precompute_preds
    src/ ...           # download scripts RECORD per-image license+attribution
  api/                 # FastAPI: routers/{cases,predictions,attempts,leaderboard,credentials}
    app/services/{npi.py, scoring.py, model_server.py}, storage.py
    alembic/
  web/                 # React+Vite+TS: pages/{Play,Result,Leaderboard,Profile}
    components/{ImageViewer,AnnotationCanvas,DiagnosisPicker,BadgeChip,ScoreOverlay}
  infra/               # fly.toml / render.yaml / vercel.json
  docs/                # this spec, data licenses, NPI taxonomy notes
```

---

## 6. Phased Roadmap

**MVP (crisp definition):** ONE modality (**dermoscopy / ISIC 2018**), ONE U-Net exported to ONNX served on CPU, predictions **precomputed for ~200–500 seed cases**. A single user: signs up → picks claimed badge → optionally NPI-verifies → gets a random unseen case (with license/attribution shown) → picks a diagnosis **and** draws the lesion → submits → sees **side-by-side user mask | model mask | ground truth** with Dice/IoU + "you agreed with the model: y/n" + "you beat the model on Dice: y/n" → a leaderboard grouping average Dice and diagnosis accuracy **by badge**. This one slice exercises every core table and both integrations (NPI + ISIC) at ~$0 infra and proves the entire thesis.

| Phase | Goal | Key deliverables |
|---|---|---|
| **0 — Data + model spike** (offline) | Prove the ML half on one modality | Download ISIC (recording per-image license), train U-Net (smp/MONAI), export ONNX, verify CPU inference, Dice/IoU script, **+ a trivial Otsu baseline** so "human vs model vs baseline" is a real research result |
| **1 — MVP web slice** | Smallest full-vision loop | FastAPI (cases/predict-cached/attempt/score), React canvas + diagnosis dropdown, signup + optional NPI verify, side-by-side result, one-table leaderboard |
| **2 — Multi-model + UX** | Several models per modality, hardened inference | Model registry + LRU ONNX cache + micro-batch queue, confidence/heatmap overlay, **badge-stratified** leaderboards, per-image attribution rendering |
| **3 — Second modality + research depth** | Demonstrate per-modality generality | Add **chest X-ray (pneumothorax)** behind the same abstractions; **cluster-bootstrap CIs** on human-vs-model Dice deltas, badge stratification, failure-mode slicing, public write-up. (Lung CT / Brain MRI follow as heavier 3rd/4th modalities with NiiVue/Cornerstone3D volume UX.) |
| **4 — Polish + deploy** | Portfolio-grade public demo on a budget | Dockerize; web on Vercel/Netlify, API on Fly.io/Render free tier, Postgres on Neon/Supabase, blobs on R2/B2; rate limiting, abuse guards, NPI cache, disclaimers everywhere |

---

## 7. Risks, Ethics & "Not for Clinical Use"

**Framing (load-bearing).** Subtle real findings (an apical pneumothorax, a small malignant melanoma, a missed lung nodule) are dangerous if missed. The platform must **never present model output as a diagnosis** and must surface the "research/educational — NOT for clinical use" disclaimer **persistently** across UI, API docs, and README. A layperson "beating the model" on one image must never read as triage capability.

**Licensing (verify each ReadMe before launch — flagged as the biggest non-technical risk).**
- **Safe to serve publicly:** NIH ChestX-ray14 (no use restrictions + attribution — the only CXR set safe to host; the SIIM masks sit on these images); **LUNA16 (CC BY 4.0)**, **LIDC-IDRI (CC BY 3.0)**, **BraTS 2021 (CC BY 4.0)**, **CHASE_DB1 (CC BY 4.0)** — all permit use with attribution + required TCIA citation/DOI + NCI/FNIH acknowledgement.
- **Non-commercial, keeps the whole platform non-commercial:** **ISIC/HAM10000 (CC-BY-NC, per-image mix of CC0/CC-BY/CC-BY-NC)** — store per-image license + attribution, render it, and filter to CC0/CC-BY if commercialization is ever considered.
- **Do NOT host publicly:** **CheXpert** (non-commercial RUA, no redistribution — train/benchmark privately only); **NLM Montgomery/Shenzhen** ("do not redistribute outside your group"); **DRIVE/STARE** (research-only, no explicit open license — **prefer CHASE_DB1 as the primary retina set**, treat DRIVE/STARE as eval-only); **LNDb** (registration-gated, eval-only). **BraTS 2024** is a *combined* pack with **per-component** terms in `LICENSES.md` — do not advertise the whole BraTS corpus as "CC BY."

**ML & scientific-integrity risks.**
- **Split hygiene / leakage:** split by **patient** (NIH/CheXpert) and by **lesion** (`lesion_id` in HAM10000); respect LUNA16's patient-wise 10-fold; the SIIM stage-1 public test leaked — make your own patient-disjoint split; ensure the served competition images were **not** in the model's training set.
- **Class imbalance everywhere** (78% empty CXRs, tiny nodule/tumor voxel fractions, 67% nevi): use Dice+focal/Tversky, oversampling, balanced metrics — never lead with raw accuracy.
- **Performance theater:** published SOTA (~0.93 CPM, ~88 Dice) come from multi-day ensembles. Report **your actual single-fold operating point honestly**, not the SOTA.
- **Coordinate/normalization bugs:** MRI per-channel z-scoring within brain mask; CT HU windowing + world↔voxel conversion; SIIM RLE is column-major/1-indexed — validate by overlaying decoded masks before training.
- **Pseudo-replication in stats** (the single most common reader-study error): crossed random effects + cluster bootstrap, always.

**Identity & fairness.** Badges are best-effort, not identity proof (§3); run verified-only sensitivity analyses, label trust tiers honestly, and never penalize non-US or NPI-less experts. Open participation guarantees abandoned/low-effort submissions — pre-register inclusion rules and never filter post-hoc to flatter the model.

**Privacy.** Server-side NPI calls only; store minimal NPI data (last-4/hash + taxonomy snapshot), never display NPI/license/address; right-to-delete to layperson.

**Confidence notes carried from research:** model choices, task framing, metrics, the NPI API mechanics, and BraTS-2021/LUNA16/LIDC/CHASE/ISIC/CheXpert/NIH licenses are **high-confidence/verified**. **Re-verify before launch:** SIIM-ACR competition-rules wording, NLM Montgomery/Shenzhen ReadMes, BraTS-2024 `LICENSES.md` + Synapse `syn53708249`, DRIVE/STARE/LNDb terms, and the NUCC radiology code set (updated twice yearly, Jan/Jul).

---

# Appendix A — Adversarial Review & Recommended MVP Cut

_A skeptical principal-engineer pass over the design above._

**Overall feasibility:** Feasible as a portfolio piece IF aggressively descoped to ~one or two 2D modalities; over-scoped as written. The document is unusually high-quality and self-aware — it already verified most of its load-bearing technical claims (NPI mechanics, taxonomy codes, LUNA16/LIDC/BraTS/ISIC licenses) and they hold up on independent check. The core thesis (per-modality architectures + human-vs-model + badge stratification) is sound and the per-modality reasoning is genuinely correct (2D-vs-3D, segmentation-vs-detection, transfer-vs-from-scratch). The real risk is not wrongness but breadth: 5 modalities, 4-5 distinct architectures (incl. 3D nnU-Net + nnDetection), 2 annotation engines, full NPI integration, and a reader-study-grade statistics program is 6-12+ months for a solo dev and most of it doesn't pay rent on validating the idea. A single trained model could take a weekend; five production-grade models with honest splits, plus the web platform, plus the stats, is a different order of magnitude. Cut to one modality end-to-end, prove the loop, then expand. The two substantive license corrections (SIIM masks are non-redistributable per Kaggle rules; BraTS has component-level controlled-access despite the CC-BY headline) must be fixed before any public hosting — those are the genuine legal landmines, and the doc currently understates both.

### Scope risks

- FIVE modalities with FOUR-to-FIVE genuinely distinct architectures (2D U-Net, 3D nnU-Net, nnDetection/Retina U-Net, from-scratch patch U-Net, EfficientNet classifier) — each 3D model is ~1-2 GPU-days plus self-configuration debugging. This is the single largest over-scope; cut to 1-2 2D modalities to validate.
- Two annotation engines (Cornerstone3D + NiiVue + react-konva) is three viewer integrations. Each is a substantial frontend effort, especially the 3D voxel-painting/GrowCut UX. Speculative for MVP.
- Full NPPES verification (server-side call, Luhn check, name normalization/matching, taxonomy parsing, tiered trust, manual-review path, re-verify, right-to-delete cascade) is a complete subsystem with legal surface — sequenced into MVP but proves nothing until there is a user population.
- Reader-study-grade statistics (crossed mixed-effects/GEE, DeLong, Obuchowski-Rockette-Hillis MRMC, Krippendorff alpha, ICC, hierarchical bootstrap, TOST equivalence, multiplicity correction) is a research project unto itself and is almost certainly underpowered at the realistic solo-project n.
- Infra sprawl: Postgres (Neon/Supabase) + object storage (R2/B2) + ONNX LRU cache + micro-batch queue + materialized leaderboard view — all introduced before there's traffic to justify them. SQLite + local disk covers MVP.
- Coordinate/affine discipline (resample human/model/GT onto GT native grid, store image/voxel-space masks + affine, never canvas pixels, empty-mask Dice=1.0 convention) is correct and critical — but it is also a meticulous, bug-prone area that will consume real time; budget for it explicitly rather than assuming it's free.
- 'Precompute all predictions offline' is the right call and de-risks live GPU serving — but the on-demand micro-batch queue for new cases reintroduces the exact live-inference complexity that precomputation was meant to avoid. Drop the on-demand path until genuinely needed.

### Missing pieces

- No abuse/sybil mitigation despite open access: nothing stops one person creating many anon sessions to farm the leaderboard or pollute the research data. At minimum rate-limit and flag duplicate-IP/device patterns; this also threatens statistical validity (pseudo-replication at the user level).
- No data-collection consent / IRB-or-exemption consideration. Even an educational tool collecting human diagnostic performance stratified by professional credential may need an IRB determination (likely exempt, but get the determination) if you intend to publish the human-vs-model study. Not mentioned anywhere.
- No image-safety/PHI scrub step on ingest. Public derm/CXR sets are de-identified, but burned-in text, DICOM header PHI, or identifiable skin features (tattoos, faces in some derm sets) should be checked before public serving. No pipeline step for this.
- No model-versioning/re-scoring story beyond a schema hint (attempts table 'enabling re-scoring after retrain'). If you retrain, every past comparison_result silently references a different model — needs explicit model_id pinning per comparison and a re-score migration plan.
- No handling of the 'human edits then model is revealed' anti-anchoring enforcement on the backend — the doc says lock-before-reveal, but doesn't specify server-side enforcement (client could reveal early). Must be server-authoritative or the anti-anchoring claim is unverifiable in the data.
- No accessibility / mobile story for an annotation (drawing) task — precise mask drawing on a phone touchscreen is hard; this affects who can realistically participate and may bias the 'layperson' cohort toward desktop users.
- No cost ceiling / kill switch for the (deferred) on-demand inference path or NPPES calls — undocumented endpoints + free-tier compute can surprise a solo dev with either outages or bills.
- No definition of 'consensus GT' construction effort for ISIC specifically — the doc invokes STAPLE/majority generally, but ISIC Task 1 ships single-annotator masks for many images; the inter-expert noise floor (a headline framing device) may not be computable for the MVP modality without extra annotation data. Verify ISIC has multi-annotator subsets before promising the noise-floor narrative.

### Legal / ethics landmines

- SIIM-ACR pneumothorax masks: Kaggle competition rules prohibit redistribution to non-participants. You may train on them privately but must NOT serve them as ground-truth to users. The doc's 'safe' framing is wrong on this point — verified against current competition rules.
- BraTS 2021: 'CC BY 4.0' is the aggregate label but several source collections (TCGA-GBM/LGG, ACRIN-FMISO-Brain, CPTAC-GBM) require NIH Controlled Data Access agreements, and TCIA Data Usage Policy applies. Confirm per-component terms before any public hosting; include the exact Baid et al. 2023 citation (DOI 10.7937/jc8x-9874) + six underlying-collection citations + TCIA acknowledgement.
- ISIC/HAM10000 is a per-image license MIX (CC0, CC-BY, CC-BY-NC). The CC-BY-NC images make the ENTIRE platform non-commercial as long as they are served. Must store + render per-image license/attribution, and filter to CC0/CC-BY only if commercialization is ever contemplated. This constrains monetization permanently if not segregated.
- NPI/NPPES data is public but PII-adjacent. Data minimization is mandatory: persist only badge enum, matched taxonomy codes, verified_at, name_match_score, and at most NPI last-4 or a salted hash. Discard returned name/address/phone/email immediately. Never display NPI/license/address publicly. Right-to-delete must cascade-delete the snapshot and revert to layperson (GDPR/CCPA exposure if a real clinician's data is mishandled).
- NPPES does NOT authenticate identity — anyone can enter a real radiologist's public NPI. The badge is 'self-asserted, name-matched to a real NPI,' never 'verified identity.' This must be stated in-UI; misrepresenting it could be a defamation/impersonation vector (someone badged as a specific MD they are not).
- 'NOT FOR CLINICAL USE' must be persistent and load-bearing across UI/API/README, AND the framing must prevent 'a layperson beat the model on this image' from reading as triage capability. Missed subtle real findings (apical pneumothorax, small melanoma, missed nodule) are the ethical core — never present model output as a diagnosis.
- Do NOT host publicly: CheXpert (non-commercial RUA, no redistribution — train/benchmark privately only), NLM Montgomery/Shenzhen ('do not redistribute outside your group'), DRIVE/STARE (research-only, no open license — prefer CHASE_DB1 CC-BY 4.0 as primary retina set), LNDb (registration-gated). The doc gets these right; they remain hard constraints.
- Open participation guarantees low-effort/abandoned submissions. Pre-register inclusion rules (min time, non-degenerate mask, diagnosis present) BEFORE data collection; never filter post-hoc to flatter the model — that would be scientific misconduct in a portfolio piece you intend to publicize.

### Recommended MVP cut

Build a SINGLE-modality, SINGLE-user, NO-account, precomputed slice and cut almost everything in the doc's "MVP" definition. Concretely:

THE RIGHT FIRST CUT (target: 1-2 weekends, ~$0):
1. Modality: Dermoscopy ONLY (ISIC 2018 Task 1 + Task 3). Correct choice — cheapest compute, 2D RGB, lesions are large so transfer learning converges fast.
2. Offline ML spike FIRST, fully decoupled from the web app (Phase 0 as written is right): download ISIC recording per-image license, train one U-Net (segmentation_models_pytorch, EfficientNet-B0 encoder), export ONNX, verify CPU inference, write a Dice/IoU + threshold-Jaccard script, AND a trivial Otsu/color-threshold baseline. STOP here and confirm your single-fold operating point is honest before touching any web code. If the model is junk, the whole platform is junk — derisk this in isolation.
3. Web slice: precompute model masks for ~100-200 ISIC cases (start at 100, not 500). Single page: show image (with license + attribution rendered) -> user picks benign/malignant + confidence + draws lesion on a react-konva canvas (NOT Cornerstone3D for MVP — see corrections) -> lock answer -> reveal user mask | model mask | GT side-by-side with Dice/IoU + "agreed with model y/n" + "beat model on Dice y/n".
4. Persistence: SQLite. A localStorage anon session id is enough to enforce "never re-see an image." 

DEFER ENTIRELY out of MVP (the doc folds these in too early):
- NPI verification / accounts / badges. This is a whole second integration with legal/PII surface area and it proves NOTHING about the core thesis until you have a multi-user population. Ship a hardcoded self-reported badge dropdown (layperson/MD/specialist, unverified) so the leaderboard schema exists, and wire real NPPES verification only in Phase 2 once the loop works and you actually have humans.
- The materialized leaderboard view, multi-model registry, ONNX LRU cache + micro-batch queue, Postgres, object storage. All Phase 2+. For 100-200 precomputed cases, masks-on-local-disk + SQLite is fine.
- ALL 3D modalities (CT, MRI) and their heavy viewers (NiiVue, Cornerstone3D volume UX). These are 1-2 GPU-days each AND the hardest UI; they are Phase 3-4, not validation.
- The full statistics stack (mixed-effects, MRMC, cluster bootstrap). You cannot run a reader study with n=1 user. The MVP's job is to prove the LOOP works end-to-end and the model is non-trivial vs baseline — not to produce a publishable result.

The single sentence test the MVP must pass: "A stranger can load a derm image, diagnose + draw it, and see a fair, correctly-aligned head-to-head score against a real trained model and human-consensus GT." That exercises the data licensing, the ML half, the annotation capture, and the scoring/coordinate discipline — the four things most likely to be broken. Everything else is decoration on top of that loop.

### Sequencing advice

Strict serialization, derisk the riskiest+cheapest things first:

1. Phase 0 (offline, no web): ISIC download with per-image license capture -> train one derm U-Net -> Otsu baseline -> ONNX export -> CPU inference verify -> Dice/IoU/threshold-Jaccard script. GATE: model meaningfully beats Otsu on a patient/lesion-disjoint split, with an honestly-reported single-fold operating point. If it doesn't, fix the ML before building anything else. This is the cheapest place to discover the idea doesn't work.

2. Phase 0.5 (offline): Nail coordinate/affine discipline and empty-mask conventions on the derm masks in isolation, with overlay-visualization sanity checks. This is the silent score-corrupter; prove it on disk before it's entangled with a UI.

3. Phase 1 (MVP web slice): precompute ~100 case predictions -> FastAPI (cases / cached-predict / attempt / score) -> react-konva canvas + diagnosis dropdown + confidence -> server-authoritative lock-before-reveal -> side-by-side result -> SQLite -> one-table leaderboard grouped by a self-reported (unverified) badge dropdown. No accounts, no NPI. GATE: a stranger completes the full loop and gets a correct, aligned score.

4. Phase 2: real NPPES verification (async, non-blocking, cache, degrade-to-self-reported), real accounts, badge-at-time snapshotting, badge-stratified leaderboard, per-image attribution rendering, then Postgres + object storage ONLY when SQLite/local actually hurts.

5. Phase 3: SECOND 2D modality (retina/CHASE_DB1 CC-BY is cleaner than CXR given the SIIM redistribution problem — consider retina before CXR, or CXR with model-output-only/no-GT-mask display). Add descriptive stats + cluster-bootstrap CIs once you have >1 modality and some users.

6. Phase 4+: 3D modalities (CT detection, MRI segmentation) with the heavy viewers, only after the 2D platform is proven and you've confirmed BraTS component licenses. These are the most expensive and least idea-validating; they come last.

Throughout: pre-register inclusion rules and the primary endpoint BEFORE collecting human data; get an IRB exemption determination before publicizing any human-performance study.

### Top corrections

- **SIIM-ACR Pneumothorax masks CANNOT be redistributed. The doc claims they are 'research/edu + attribution' and that 'the images are NIH = no-restriction' so it's safe. Verified false on the redistribution point: the Kaggle competition rules explicitly state participants agree 'not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition.' The SIIM-provided pneumothorax MASKS/annotations are Competition Data and are NOT redistributable, even though the underlying NIH ChestX-ray14 images are.** → For Phase 3 CXR: you may host the NIH images and TRAIN on SIIM masks privately, but you may NOT serve the SIIM ground-truth masks to users. Either (a) generate your own GT by having the masks come from a redistributable source, (b) serve only your model's output + diagnosis labels (no GT mask display) for CXR, or (c) drop pneumothorax segmentation as a public-GT modality and use it train-only. Re-read the current Kaggle rules at signup before launch — they are the binding terms, not the dataset paper.
- **BraTS 2021 is described as flatly 'CC BY 4.0, TCIA' and listed under 'Safe to serve publicly.' Verified more nuanced: the aggregate dataset is CC BY 4.0 and the challenge pack is freely downloadable, BUT several source collections (TCGA-GBM, TCGA-LGG, ACRIN-FMISO-Brain, CPTAC-GBM) fall under the NIH Controlled Data Access Policy requiring a usage agreement, and TCIA requires abiding by its Data Usage Policy. Redistribution-as-a-public-web-corpus is not unambiguously granted by 'CC BY 4.0' alone.** → Treat BraTS as train/benchmark-locally first. Before hosting ANY BraTS image publicly, confirm the specific component collections you serve are the CC-BY ones, not the controlled-access ones, and include the exact required citation (Baid et al. 2023, DOI 10.7937/jc8x-9874) plus the six underlying-collection citations and the TCIA acknowledgement. Do not advertise 'BraTS = CC BY' as a blanket claim.
- **Cornerstone3D is specified as the single MVP annotation engine for both 2D and 3D. This is over-engineering for the MVP and a likely time sink. Cornerstone3D is a heavy DICOM/medical-imaging toolkit; for a single 2D PNG dermoscopy canvas with a brush/polygon it is far more integration surface than needed, and it is the kind of dependency that eats a solo dev's first weekend.** → Use react-konva (which the doc already names as the 'simplest PNG-only 2D path') for the dermoscopy and other 2D-PNG MVP work. Introduce Cornerstone3D / NiiVue only when you reach a modality that genuinely needs world<->voxel transforms, DICOM-SEG, or slice scrolling (CT/MRI, Phase 3+). Adopt the heavy engine when its features are required, not speculatively.
- **The credential-badging system is positioned as core and built into the MVP loop, but it cannot validate anything with a single user and adds the project's largest legal/PII surface (handling NPI data, identity claims, GDPR/CCPA right-to-delete). It is sequenced too early.** → Defer real NPPES integration to Phase 2. In MVP, ship only a self-reported, unverified badge dropdown so the schema/leaderboard grouping exists. The doc's own NPI design is otherwise solid and verified: base https://npiregistry.cms.hhs.gov/api/, no auth, version=2.1 required, JSON top-level is {result_count, results}, authoritative field is taxonomies[].code, NPI-1 vs NPI-2 via enumeration_type. The taxonomy codes check out (207N00000X dermatology, 207W00000X ophthalmology, 2085R0202X diagnostic radiology). Build it as specified — just later.
- **The doc claims 'no published rate limit (best-effort)' for NPPES and plans a ~30-day cache. There is genuinely no published rate limit, so the claim is technically accurate, but building on an undocumented, unguaranteed best-effort endpoint with no SLA for a user-facing signup-blocking call is a reliability risk that isn't flagged.** → Make NPI verification fully asynchronous and non-blocking (the doc already says badge is a label, never a gate — enforce that in the impl: user plays immediately, verification resolves in the background and can fail/retry silently). Cache aggressively and degrade to self-reported on any API failure. Never let an NPPES outage block signup or play.
- **Statistics section is excellent but scoped as if a reader study population already exists (MRMC/Obuchowski-Rockette-Hillis, crossed mixed-effects, cluster bootstrap). For a portfolio project the realistic n is tiny and heavily self-selected/abandoned. The plan risks an underpowered study dressed in heavy machinery, and 'verified-only sensitivity analysis' may leave near-zero verified specialists.** → Be explicit that the research deliverable is exploratory/descriptive with honest CIs, not a powered MRMC claim, unless you can actually recruit a panel of verified radiologists (you likely cannot, solo). Lead with the model-vs-consensus-GT result + inter-expert-variability noise floor, which needs no live human panel. Treat the human-vs-model comparison as a secondary, sample-size-caveated finding. Pre-register inclusion rules as the doc says, but right-size the stats to the n you'll plausibly get (likely <50 engaged users).
