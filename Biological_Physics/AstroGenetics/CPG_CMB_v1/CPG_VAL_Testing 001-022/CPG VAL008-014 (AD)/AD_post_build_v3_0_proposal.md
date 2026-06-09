# AD-immune Card v2.2 → v3.0 — Post-Build Design Proposal

**Status:** PROPOSAL — awaiting Heath's review and approval before any execution
**Date:** 2026-06-02
**Pattern:** Strict additive bump (same model used for breast v2.3 → v3.0)
**Author:** Walther
**Source documents read:** `ad-immune_card_v2.2.json`, `ad-immune_README.md`, VAL-091 outcome, VAL-057 prereg, RETIRED AD folder

---

## §1. What carries forward from v2.2 (preserved byte-for-byte in v3.0)

The biological substance of the v2.2 card is sound and stays. The strict-additive principle applies: nothing in v2.2 gets rewritten or deleted; v3.0 layers post-build evidence on top.

### Operational blocks preserved verbatim
- `stage_1_immune_flag` — including the 7-CpG Rule A directional panel, h_min_immune = 0.838889, A_dir scoring method, tier framework (BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH), all panel selection rationale, freeze timestamp
- `stage_2_localization` — Loyfer/Moss array atlas, expected NULL solid-organ + cortical-neuron at HC floor, differential-diagnosis tile (AD vs glioma at 0.5% cortical-neuron threshold)
- `stage_3_subcomposition` — EpiDISH 6-cell exploratory descriptive layer
- `report_contents`, `validation_tier`, `known_limitations` (all 11), `universal_reference` (full-inline), `lessons_learned`
- `validation_anchors`: VAL-049, VAL-050, VAL-051, VAL-052, VAL-053, VAL-054a (NON-TEST), VAL-054b, VAL-040, VAL-057 — all retained byte-for-byte

### Biological substance carrying forward
- **Disease:** Alzheimer's disease, cross-sectional detection from peripheral blood (G30)
- **Substrate:** buffy coat DNA, immune class
- **Three frozen cohorts:** AIBL GSE153712 (EPIC, n=726), AddNeuroMed GSE144858 (450K, n=300), GSE53740 GIFT (450K, n=383 specificity arm with PSP/CBD/FTD)
- **The Directional-Score Principle:** AD has bidirectional per-CpG drift; pooled entropy is NULL; directional weighting recovers d=+0.62. This finding is central and stays.
- **Sex stratification mandatory** (AIBL female d=+0.71 vs male d=+0.51 on unified panel)
- **Age regression mandatory** (R²=0.26 in AddNeuroMed)
- **PSP/CBD crossover signature** documented at d=-0.51 cortical-neuron (BELOW_NORMAL tier) per VAL-091

---

## §2. Limitations of v2.2 the post-build instrument can address

The v2.2 card documents 11 known limitations. The post-build instrument doesn't fix all of them, but it materially addresses 5:

| v2.2 limitation | Post-build capability that addresses it |
|---|---|
| **L1.** Per-patient sensitivity 25-30% at 95% spec (modest signal on 7-CpG panel) | Mahalanobis hyper-volume in 115-cell A-score space — universal departure-from-HC summary that surfaces signal across ALL cells, not just 7 CpGs |
| **L2.** R²=0.26 age confound; ad-hoc regression reduces effect 38% | `age_axis_foreground.py` + `IAMAtlas_age_layer.csv` (8,199 CpGs, 100% MCMC convergence) — principled IAMAtlas-native age subtraction at the β level BEFORE A-scoring |
| **L4.** EpiDISH Stage 3 6-cell sub-composition exploratory only | 115-cell A-score fan-out native to one Walther pass — finer than 6-cell at no extra cost. Sub-types like aTreg, exhausted CD8, B-cell memory subsets become individually visible |
| **L8.** Pooled entropy is bidirectional-blind | PC2 T-cell suppression axis (per breast CPG-VAL-005, d=-0.67/-0.58) is a covariance-axis decomposition that captures bidirectional T-cell senescence by construction — AD's known T-cell signature should hit this axis hard |
| **L10.** No formal null suite — only HC-internal permutation | L9 7-test null suite per VAL: synthetic-null, label-shuffle, panel-permutation, atlas-shuffle, etc. (the same suite that gave 7/7 PASS on CPG-VAL-001/002/003/005/007) |

The post-build run will tell us which of these *actually* improves AD performance vs which are theoretical gains. That's what the VALs are for.

### What the post-build does NOT change
- The bidirectional drift is real biology, not instrument artifact. The new instrument doesn't make pooled entropy work on AD.
- The age confound is real biology (AD patients ARE older). Age subtraction reveals what's left after removing age — could be more or less than the 7-CpG panel showed.
- The sex effect is real biology and stays in the report.
- Per-patient sensitivity may remain modest. EDEAR-AD is cohort screening + trajectory monitoring, not single-timepoint diagnosis. That clinical framing stays.

---

## §3. Proposed v3.0 design — strict additive bump

Mirroring the breast v2.3 → v3.0 pattern:

### File layout (matches breast pattern)
```
walther_clinical_runtime/DISEASE_MAPS_CARDS/AD_immune/
├── ad_immune_card_json/
│   ├── ad-immune_card_v3_0.json          ← NEW — strict additive over v2.2
│   ├── ad-immune_README.md               ← v2.2 README (Heath's prose, unchanged)
│   ├── ad-immune_v3_0_release_notes.md   ← NEW — what changed, what didn't
│   └── OLD/
│       └── ad-immune_card_v2.2.json      ← v2.2 archived verbatim
└── ad_immune_residual_maps/
    ├── ad_immune_residual_map_chr_annotated.csv   ← from AD VAL analog of CPG-VAL-003
    ├── ad_immune_bimodality_map.csv               ← from AD VAL analog of CPG-VAL-004
    ├── ad_immune_pca_projections.csv              ← from AD VAL analog of CPG-VAL-005
    └── README_AD_immune_residual_maps.md
```

### What changes in the v3.0 JSON
- `card_version`: v2.2 → v3.0
- `card_date`: 2026-04-26 → (date of post-build sealing)
- `supersedes`: extended supersedes string (v2.2 archived verbatim)
- `validation_evidence_summary` (currently `validation_anchors`): EXTENDED with new post-build CPG-VAL entries (analogous to how breast went from 4 entries → 9 entries)
- NEW top-level: `cpg_native_post_build_addendum` — explains the post-build evidence layer
- NEW top-level: `v30_changes` — change list

### Versioning question for Heath
The MASTER_TRACKER and the RETIRED README I pushed earlier both said the post-build card would be called "v1.0" (a clean reset, signaling "first card under the new instrument"). The actual breast pattern was v2.3 → v3.0 (continuing the sequence). Two options:

- **Option A — Continue the sequence (v3.0):** mirrors breast exactly, preserves audit trail of the version history
- **Option B — Clean reset (v1.0):** the post-build pipeline is new enough that starting fresh at v1.0 signals the discontinuity loudly

**My recommendation: Option A (v3.0).** Same as breast. Cleaner audit trail. The discontinuity is documented in the `cpg_native_post_build_addendum` block + the OLD/ archive.

But this is Heath's call.

---

## §4. Proposed VAL series for AD

Following the Family A foundation pattern (CPG-VAL-001-007 on breast cohort), the AD Family B series for the AD-immune card would be a structured analog. Each VAL tests one specific post-build capability against the existing AD cohorts.

### Proposed slot allocation

The MASTER_TRACKER currently reserves CPG-VAL-008-014 for "breast Family B (deferred)." Two options:

- **Option A — AD takes CPG-VAL-008-014:** since breast Family B is deferred and AD is the next active series, the next-available numbered slots go to AD. Breast Family B gets pushed to CPG-VAL-022+ when it resumes.
- **Option B — AD starts at CPG-VAL-015:** preserve the originally-reserved breast slots even if deferred. AD starts at the next clean range.

**My recommendation: Option A.** Slot numbers should reflect actual sequencing. Deferred reservations don't earn slot priority. The MASTER_TRACKER + v4 inventory get updated to reflect this.

### Proposed VAL design (using CPG-VAL-008-014 per Option A)

| VID | Test | Cohort | Capability validated | Predicted outcome |
|---|---|---|---|---|
| CPG-VAL-008 | Per-cell-type A-score fan-out — AD | AIBL GSE153712 (161 AD / 471 HC, EPIC) | 115-cell A-scores reveal sub-cellular AD signature | Expected: T-cell subtypes (CD4+ memory, exhausted CD8) show effect; possibly B-cell memory and senescent monocytes; basophils/microglia/BE near HC (specificity check) |
| CPG-VAL-009 | Mahalanobis hyper-volume — AD AIBL | AIBL (same cohort) | Single-number universal departure-from-HC | Expected: Cohen's d in range ~+0.7 to +1.5 (vs +0.624 for 7-CpG panel) — testing whether the universal summary beats the disease-trained panel |
| CPG-VAL-010 | Mahalanobis cross-platform — AD AddNeuroMed | AddNeuroMed GSE144858 (93 AD / 96 HC, 450K) | Cross-platform replication of CPG-VAL-009 | Expected: d preserved within ±0.3 of AIBL (cross-platform attenuation pattern seen in breast) |
| CPG-VAL-011 | Age-axis foreground subtraction comparison | AIBL + AddNeuroMed | Test whether age_axis_foreground.py outperforms ad-hoc R²=0.26 regression | Expected: signal preserved or boosted after age subtraction (analogous to breast CPG-VAL-007 +0.255 Mahalanobis gain) |
| CPG-VAL-012 | PC2 T-cell suppression axis — AD | AIBL + AddNeuroMed | Test whether PC2 T-cell axis (per breast CPG-VAL-005) captures AD's T-cell senescence | Expected: AD cases shift on PC2 in the T-cell-suppression direction (analogous to breast d=-0.67/-0.58) |
| CPG-VAL-013 | Per-CpG residual map — AD | AIBL + AddNeuroMed | Derive IAMAtlas-native AD panel candidate (analog of CPG_breast_panel_v1) | Expected: a directional-CpG set, NOT a uniform-direction set (bidirectional drift hypothesis), feeding into a candidate CPG_ad_panel_v1 for future |
| CPG-VAL-014 | GSE53740 specificity arm — full reproduction | GSE53740 GIFT (15 AD / 128 FTD / 44 PSP/CBD / 193 HC) | Post-build instrument's read of the FTD/PSP-vs-AD signal | Expected: PSP/CBD differentiation preserved; tauopathy-class question remains open |

**These 7 VALs would establish whether the post-build instrument materially improves AD detection vs the v2.2 baseline.** The card's operational scoring stays on the 7-CpG Rule A panel UNTIL these VALs sealed AND the data say the new methods are better. v3.0 is documentation + addendum; the operational Stage 1 panel does NOT change in this bump.

### Disease matrix
New rows in `disease_cell_signature_matrix_v1_5.csv` for `disease_id=alzheimers`:
- `phase=at_dx` (cross-sectional, AIBL/AddNeuroMed)
- `phase=tauopathy_class` (the FTD/PSP/CBD column, from GSE53740)
- substrate: `whole_blood_buffy_coat`
- severity_class: `ACTIVE` for at_dx; `RELATED_TAUOPATHY` for the cross-reactive arm

Matrix bumps v1.5 → v1.6 when these rows are added.

---

## §5. Cohort data acquisition plan

This is the practical question that determines timing.

### What's needed
Full β matrices (all ~485K CpGs on 450K, ~865K on EPIC) per sample for all three cohorts. NOT just the 18-CpG or 7-CpG subset that v2.2 used.

### What's available in repo (NOT sufficient)
- `validation_runs/val_050_aibl/aibl_imm_betas.json` — 18-CpG subset only (pre-build extraction)
- `validation_runs/val_052_addneuromed/addneuromed_imm_betas.json` — 18-CpG subset only
- `validation_runs/extract_aibl.py`, `stream_aibl.py` — extraction scripts (these CAN be adapted to pull all CpGs but were originally written for the 18-CpG subset)

### What's needed from GEO
| GSE | Cohort | n samples | Platform | Approx raw size |
|---|---|---|---|---|
| GSE153712 | AIBL | 726 | EPIC 850K | ~7 GB IDATs |
| GSE144858 | AddNeuroMed | 300 | 450K | ~2 GB IDATs |
| GSE53740 | GIFT | 384 | 450K | ~2.5 GB IDATs |

### Two acquisition paths

**Path A: Re-extract from GEO IDATs.** Most rigorous. Each cohort downloaded fresh from GEO, methylprep run, full β matrix produced, sealed with SHA-256, anchored at `Biological_Physics/validation_runs/ad_immune_cohorts/`. Provides clean reproducibility chain.

**Path B: Use Heath's existing AIBL/AddNeuroMed/GIFT β matrices if available locally.** If Heath has already extracted full β matrices on his Apple laptop or gaming PC from earlier VAL work (the existing series matrix extractors suggest this is possible), upload those, SHA-verify, anchor in repo.

**My recommendation: Path B first, fall back to Path A.** Heath: do you have full β matrices for AIBL/AddNeuroMed/GSE53740 already extracted? If yes, that saves 11+ GB of download + methylprep compute time.

---

## §6. Proposed sequencing

After Heath approves this proposal, the execution order:

1. **Cohort acquisition** (Path B if possible, A otherwise) — produces 3 β-matrix CSVs anchored in repo with SHA-256, cohort_manifest.json per cohort
2. **Run AIBL through SOP v1.2 Stages 0-10** using `walther_clinical_runtime/` modules end-to-end. This is the first time the full chain is exercised on a non-breast disease.
3. **CPG-VAL-008 sealed** — per-cell-type fan-out (the foundation result that everything else builds on)
4. **CPG-VAL-009 sealed** — Mahalanobis hyper-volume on AIBL
5. **CPG-VAL-010 sealed** — Mahalanobis cross-platform on AddNeuroMed
6. **CPG-VAL-011 sealed** — age-axis foreground subtraction
7. **CPG-VAL-012 sealed** — PC2 T-cell suppression axis
8. **CPG-VAL-013 sealed** — per-CpG residual map → CPG_ad_panel_v1 seed candidate
9. **CPG-VAL-014 sealed** — GSE53740 specificity arm
10. **Card v3.0 sealed** with all 7 CPG-VAL refs appended to validation_evidence_summary + `cpg_native_post_build_addendum` block
11. **Disease matrix v1.5 → v1.6** with 2 new `alzheimers` rows
12. **Residual maps** (chr-annotated + bimodality + PCA) produced + README
13. **MASTER_TRACKER §2 + §5 + §7 updated** to reflect AD completion

**Compute estimate:** AIBL alone (726 samples × all CpGs through Walther + IAMAtlas REBUILD) is the heaviest step — probably ~12-24 hours on Heath's 16-core gaming PC for Stage 2. The rest is downstream of that.

**Timeline estimate:** If cohort acquisition is Path B (β matrices already exist locally) and Heath approves the design: 5-7 days of compute + analysis + sealing. If Path A (download + methylprep): add 2-3 days.

---

## §7. Decision points for Heath

Before any execution begins, I need explicit decisions on:

1. **Versioning (§3):** v3.0 (continue sequence — recommended) or v1.0 (clean reset)?
2. **VAL slot allocation (§4):** AD takes CPG-VAL-008-014 (recommended) or starts at CPG-VAL-015?
3. **Cohort acquisition (§5):** Do you have full β matrices for AIBL / AddNeuroMed / GSE53740 already extracted locally that you can upload? Or do we go Path A (download from GEO)?
4. **The 7-VAL series as proposed (§4):** approve as-is, modify (which ones to add/remove/reorder), or skip any?
5. **Disease matrix rows (§4):** approve `at_dx` + `tauopathy_class` rows as proposed, or different phase structure?
6. **Operational scoring change:** confirmed that v3.0 does NOT change the Stage 1 operational scoring (still 7-CpG Rule A panel) — the post-build evidence is documented in the addendum, but operational logic is unchanged. A later v3.1 or v4.0 may switch operational scoring to Mahalanobis IF the VALs show it's better. Agree?
7. **Anything in §1 (carries forward) that should actually be reconsidered?** Anything in §2 (limitations addressed) you'd add or remove?

---

## §8. What I am NOT doing until you approve

- No data downloads
- No β matrix processing
- No deconvolution runs
- No card edits
- No file moves
- No commits
- No pushes

**Read-only proposal. Awaiting your sign-off.**

When approved, I'll proceed with the cohort acquisition step (Path B if Heath has the data, A otherwise) and pause again before the first Stage 2 run on AIBL — that's the first heavy compute step and worth a checkpoint before launching.
