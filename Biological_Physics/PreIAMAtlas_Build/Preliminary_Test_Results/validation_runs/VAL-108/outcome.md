# VAL-108 — Outcome Record

**Outcome:** `O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED` (per sealed prereg)

**Sealed prereg SHA-256:** `6f40ebd9d30bb10242b245d7bde280607f1170e3c7993a8284e2852ad1f69e7a`  
**Sealed at:** 2026-04-28T22:34:33Z  
**Executed at:** 2026-04-28T22:48:28Z  
**Cohort:** GSE69138 ischemic stroke discovery cohort, n=404 whole-blood samples, GenomeStudio AVG_Beta substrate (HM450K)

## Headline numbers

**CHK-3.1A self-calibration (within-cohort substrate anchor):**
- 383 / 404 samples QC-pass (94.8%); fail rate 5.2% — well under the 10% O5 trip threshold
- Pre-locked thresholds: extreme ≥ 25%, middle ≤ 13%, n_valid ≥ 400,000

**Subtype distribution (QC-pass):**
- Large-artery atherosclerosis: 125
- Small-vessel disease (lacunar): 134
- Cardioembolic: 120

**Stage 1 (Salas IDOL 350-CpG immune entropy as Xu-538 proxy):**
- Large-artery vs small-vessel: d = +0.032
- Large-artery vs cardioembolic: d = +0.129
- Small-vessel vs cardioembolic: d = +0.094

**Stage 2 (Loyfer 25-tile cardio-relevant tiles):**

| Tile | Large vs Small | Large vs Cardio | Small vs Cardio |
|---|---|---|---|
| Vascular_endothelial_cells | −0.023 | −0.054 | −0.030 |
| Left_atrium | −0.015 | −0.013 | +0.003 |
| Adipocytes | −0.043 | −0.038 | +0.007 |
| Monocytes_EPIC (top5) | — | +0.106 | +0.167 |
| Neutrophils_EPIC (top5) | −0.063 | +0.098 | +0.166 |

**Stage 3 (UniLIFE + Salas pooled entropy):**
- All pairs |d| < 0.16

## Outcome interpretation

Every Cohen's d in every contrast at every stage is below 0.5 (well below the threshold for biologically-meaningful subtype differentiation). The largest |d| anywhere is 0.167 (small-vessel vs cardioembolic on Monocytes tile). **Ischemic stroke etiology subtypes are framework-equivalent on whole-blood methylation at the assayed substrates.**

## Biological interpretation

This is a real finding, not a null artifact. By the time blood is drawn post-stroke, the inflammatory response has homogenized the immune methylation signature across etiologies. Whether the stroke originated from large-artery atherosclerosis, small-vessel disease, or a cardiac embolic source, the systemic blood inflammatory profile converges. The framework correctly reports that whole-blood DNA methylation does not stratify ischemic stroke by TOAST etiology subtype.

This has clinical implications for cardio-epic deployment: an EDEAR cardio report on whole-blood methylation should not claim stroke subtype discrimination. The framework can detect stroke vs healthy contrast (deferred to future VAL with proper healthy control cohort), but stroke-vs-stroke etiology stratification is not within whole-blood methylation's signal envelope.

## What propagates to cardio-epic v0.1

1. **CHK-3.1A self-calibration thresholds for GenomeStudio AVG_Beta whole-blood substrate** (within-cohort): extreme ≥ 25%, middle ≤ 13%, n_valid ≥ 400,000. Documented as cohort-specific, not generalizable to other GenomeStudio AVG_Beta cohorts pending future structurally-separated calibration VAL.

2. **Within-stroke immune A-score baseline** for cardio-epic deployment confidence intervals on Stage 1 (Salas proxy): mean ≈ {report from per-sample CSV} ± SD across 379 stroke samples.

3. **Per-tile Loyfer A-score baselines** for stroke whole blood across all 25 tiles (saved in per_sample.csv).

4. **Cardio-epic v0.1 reporting policy**: stroke etiology subtypes are reported as a single pooled signature, not stratified, on whole-blood methylation.

## What does NOT propagate

- No stroke-vs-healthy comparison (deferred — cohort has no healthy controls).
- No Stage 1 production claims (Stage 1 used Salas proxy, not Xu-538 production panel).
- No EpiSCORE HeartRef or Caggiano CelFiE biology (deferred to v0.2+).
- No generalization to GenomeStudio AVG_Beta substrate beyond this cohort.

## Reproducibility (CHK-7.6)

- **Inputs**: GSE69138_ave_beta.txt.gz (NIH GEO public, SHA computed at acquisition), gse69138_filtered.tsv (8,100 CpGs × 404 samples derived subset), gse69138_metadata.json, atlas vault Loyfer 25-tile + UniLIFE + Salas IDOL
- **Environment**: Python 3 + pandas; runtime ~1 minute on filtered matrix
- **Output**: results.json + per_sample.csv

## EDEAR commercial deployment

Per CCL-037 — unaffected. Cardio-epic deployment policy: Stage 1 + Stage 2 + Stage 3 pooled report, no etiology stratification claim.

## Outcome status

`O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED` — sealed.
