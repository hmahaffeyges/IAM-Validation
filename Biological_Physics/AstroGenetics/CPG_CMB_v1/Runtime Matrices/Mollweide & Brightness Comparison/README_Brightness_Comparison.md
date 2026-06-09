# Brightness_Comparison — Stage 4.6 module

**Module:** `patient_brightness_comparison.py`
**Stage:** 4.6 — Per-class healthy brightness comparison + patient Mollweide projection
**Reference:** CPG Plate 1 (Cosmic Methylome Background) at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png`

## What this module does

For each of the 8 architectural classes, compute the patient's per-CpG z-score departure from the class's healthy reference (μ, σ from the brightness CSVs), then project those z-scores onto the same HEALPix NSIDE=128 Mollweide grid as Plate 1. The output is the patient's personal Cosmic Methylome Background — an 8-panel sky map of where their methylation pattern departs from the per-class healthy baseline.

## Inputs

1. **Patient β values** — Stage 3 foreground-cleaned β matrix as a pd.Series indexed by cpg_id
2. **Per-class brightness references** — loaded from `Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives/{class}_v0_1_REBUILD.tar.xz`, inner path `{class}/iamatlas_v0_1_{class}_brightness.csv`. Columns: cpg_id, class, mean, sd, ci_lo, ci_hi.
3. **CpG-to-HEALPix mapping** — `iamatlas_cpg_to_healpix_nside128.npy` (built at IAMAtlas build time using genomic-order pixel assignment; pending repo addition).

## Outputs

1. **PatientBrightnessReport** — per-class z-score arrays + summary statistics
2. **`{patient_id}_brightness_comparison_summary.json`** — engine-internal summary
3. **`{patient_id}_{class}_z_scores.csv`** — per-class z-score CSV (one per class, 8 total)
4. **`{patient_id}_cosmic_methylome.png`** — 8-panel Mollweide PNG (the customer-facing visualization endpoint)

## Engine integration

Called from `walther_clinical.py` at Stage 4.6 (after Stage 3 foreground subtraction, before Stage 5 Mahalanobis):

```python
from patient_brightness_comparison import (
    load_all_8_class_references,
    compute_all_8_class_departures,
    render_patient_cosmic_methylome,
    save_brightness_report,
)

references = load_all_8_class_references(
    "Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives"
)
report = compute_all_8_class_departures(
    patient_beta=patient_beta_cleaned,
    references=references,
    patient_id=patient_metadata["patient_id"],
)
cpg_to_pixel = np.load("iamatlas_cpg_to_healpix_nside128.npy")
render_patient_cosmic_methylome(
    report, cpg_to_pixel,
    out_path=f"reports/{patient_id}_cosmic_methylome.png",
)
save_brightness_report(report, out_dir=f"reports/{patient_id}/brightness/")
```

## Patient intake covariate usage in this module

**This module does NOT consume patient intake questionnaire fields.** Stage 4.6 is a pure β-vs-reference comparison — the math is patient_β minus class_μ over class_σ. Covariate adjustment happens upstream (Stage 3 foreground subtraction for age, sex, smoking, batch, ancestry) so the patient_β values arriving at Stage 4.6 are already covariate-cleaned.

Intake covariates that DO get used elsewhere in the chain (per BUILD_SPEC v1.2 §4.3 + §4.5):
- **Stage 3 foreground:** chronological_age, sex_at_birth, smoking_status, smoking_bin
- **Stage 7 tier override:** known_autoimmune_condition, current_immunosuppression, transplant_status, current_cancer_in_treatment, current_pregnancy_with_trimester, hrt_status, trt_status, current_glp1_or_weight_loss_medication, hiv_status
- **Stage 8 card matching:** hpv_status, prior_chemotherapy_history, prior_radiation_history, prior_cancer_history, menopause_status, recent_illness_within_3_months, recent_vaccination_within_3_months

## Conventions (mirror Plate 1)

- **HEALPix NSIDE=128** (npix = 196,608)
- **Mollweide projection** (equal-area, full-sky)
- **CpG-to-pixel:** genomic order (chr1 → chrY, then MAPINFO within), sequential assignment
- **Multi-CpG-per-pixel:** averaged per pixel for the z-map
- **Colormap (patient z-map):** RdBu_r diverging, centered at z=0, range [-3, +3]
  - Red = β_patient significantly above class μ (hypermethylated departure)
  - Blue = β_patient significantly below class μ (hypomethylated departure)
  - Neutral = within healthy variance
- **Mask:** CpGs with σ < 1e-4 (singular posterior) are masked. Stromal galactic mask CpGs are likewise masked. Masked pixels render BLACK (matches Plate 1 stromal panel convention).

## Lineage

- **Per-class brightness CSVs:** outputs of the IAMAtlas REBUILD MCMC, frozen 2026-04-06. 483,093 CpGs per class. Each CSV is the data behind one panel of Plate 1.
- **IAMAtlasREBUILD.csv:** merged per-class brightness with per-cell-type columns, 264 columns total.
- **Plates:** see `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/README_CPG_Plates.md`

## H_min anchors (frozen 2026-04-06)

| Class | H_min |
|---|---|
| terminal | 0.7728 |
| immune | 0.838889 |
| secretory | 0.8433 |
| progenitor | 0.8522 |
| cycling | 0.8561 |
| stromal | 0.8630 |
| stem_adult | 0.8737 |
| stem_pluri | 0.9822 |

## Validation

The module produces no scientific output by itself — its correctness is validated by:
1. Smoke-test: `python patient_brightness_comparison.py --smoke-test` (loads all 8 references, reports n_cpgs and mean β/σ)
2. Round-trip test: synthetic patient β with known per-class offset → recovered z-score per class within tolerance (to be added as N7-extension test)
3. Visual test: render the reference itself (patient_β = class_μ) → expect all z=0 → uniform neutral Mollweide projection
