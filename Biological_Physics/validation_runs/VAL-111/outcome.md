# VAL-111 Outcome — EpiSCORE HeartRef on Cardio-Epic Cohorts

**Sealed:** 2026-04-29
**Outcome:** **O3_TISSUE_FLOOR_DOMINATED**
**Atlas:** EpiSCORE HeartRef (Zhu et al. Nat Commun 2022 13:3895), bridged to 450K CpGs, 3,727 atlas CpGs × 5 cardiac cell types (CM, EC, FB, MP, SMC).
**Atlas SHA-256:** `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`
**Atlas vault path:** `/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/`

---

## Summary

Cohort-internal A-score ranges across cardiac tiles fail the pre-declared discrimination threshold (≥0.10) on all three cohorts. Cohort means cluster ~0.46–0.51 across all five tiles in all three cohorts, regardless of tissue substrate (cultured pulmonary endothelial cells, ascending aortic tissue, peripheral blood). The blood-control cohort shows the same mean range as the tissue cohorts, indicating that EpiSCORE HeartRef — a gene-promoter-based reference — does not produce the substrate-specific contrast that tile-coverage atlases (Loyfer/Moss) produce on Stage 2.

The outcome is consistent with the prior expectation that promoter-based references generate flat ~0.5 readings across heterogeneous β panels because gene promoters that distinguish cardiac cell types are not differentially methylated outside cardiac tissue, and within cardiac tissue the promoter signal is averaged across many CpGs that carry mixed methylation directions for the same cell type. The atlas is methodologically sound for its design purpose (EpiDISH-style proportion estimation in heart tissue) but does not transfer to A-score tile reading on β panels at the resolution required for cardio-epic Stage 2.

This is consistent with the result, not a refutation of EpiSCORE.

---

## Cohort-level numbers

| Cohort | n samples | Atlas CpGs intersected | β shape |
|---|---|---|---|
| GSE69138 (stroke blood + control blood) | 589 | 3,727 | (3727, 589) |
| GSE84395 (PAH cultured PEC) | 39 | 3,727 | (3727, 39) |
| GSE84274 (ascending aorta dissection / BAV / normal) | 24 | 3,408 | (3408, 24) |

No O4_BRIDGE_FAILURE — atlas∩cohort intersection > 500 in all three cohorts.

## Stratified A-score means

### GSE84274 — ascending aorta tissue (by disease state)

| Group | n | A_CM | A_EC | A_FB | A_MP | A_SMC |
|---|---|---|---|---|---|---|
| aortic dissection | 12 | 0.4802 | 0.4995 | 0.4973 | 0.5012 | **0.5192** |
| BAV with aorta dilation | 6 | 0.4728 | 0.4924 | 0.4901 | 0.4931 | **0.5131** |
| normal | 6 | 0.4669 | 0.4855 | 0.4845 | 0.4860 | **0.5072** |
| **range** |  | 0.0133 | 0.0140 | 0.0128 | **0.0152** | 0.0120 |

Direction is biologically sensible: dissection > BAV > normal monotonically across all five tiles; SMC tile always highest (consistent with aortic media SMC content). However, all five tile ranges fall well below the 0.10 discrimination threshold.

### GSE84395 — PAH cultured pulmonary endothelial cells (by subject status)

| Group | n | A_CM | A_EC | A_FB | A_MP | A_SMC |
|---|---|---|---|---|---|---|
| control | 18 | 0.4618 | 0.4924 | 0.4828 | 0.4880 | 0.4923 |
| hPAH (heritable) | 10 | 0.4599 | 0.4971 | 0.4829 | 0.4902 | 0.4936 |
| iPAH (idiopathic) | 11 | 0.4648 | **0.4995** | 0.4872 | 0.4934 | 0.4980 |
| **range** |  | 0.0049 | **0.0070** | 0.0044 | 0.0055 | 0.0057 |

EC tile is the highest-range tile (consistent with cultured endothelial substrate) but the range is 0.0070 — far below the 0.10 threshold.

### GSE69138 — peripheral blood (cohort means)

All five cardiac tiles read 0.477–0.511 across the 589-sample blood cohort. **Blood floor breached on 5/5 tiles** (cohort mean > 0.10 floor on every tile).

| Tile | Cohort mean A | Floor breach? |
|---|---|---|
| CM  | 0.4770 | ✓ breach |
| EC  | 0.5025 | ✓ breach |
| FB  | 0.4905 | ✓ breach |
| MP  | 0.5109 | ✓ breach |
| SMC | 0.5064 | ✓ breach |

Stroke subtype stratification (within stroke samples) shows no tile differentiation across cardioembolic, lacunar, large-artery atherosclerosis, atherothrombotic, or small-vessel disease subtypes — all subtype means within ±0.002 of cohort mean.

## Tissue discrimination ranges (max across cohorts × tiles)

| Range | Value |
|---|---|
| GSE84274_MP_range | 0.0152 (largest) |
| GSE84274_EC_range | 0.0140 |
| GSE84274_CM_range | 0.0133 |
| GSE84274_FB_range | 0.0128 |
| GSE84274_SMC_range | 0.0120 |
| GSE84395_EC_range | 0.0070 |
| GSE84395_MP_range, SMC_range | ~0.0055 |
| GSE84395_CM_range, FB_range | ~0.0045 |

**Maximum tissue discrimination:** 0.0152 (GSE84274 MP tile, dissection − normal). Required: ≥0.10. **Discrimination ratio: 15%.**

---

## Outcome selector logic

- O4_BRIDGE_FAILURE? **No.** All three cohorts had atlas∩cohort ≥ 3,408 CpGs.
- Any tissue tile range ≥ 0.10? **No.** Maximum tissue range = 0.0152 (15% of threshold).
- → **O3_TISSUE_FLOOR_DOMINATED.**

---

## Card-level disposition

- **EpiSCORE HeartRef → atlases_deferred for cardio-epic v0.3** with rationale: "promoter-based reference, gene-promoter substrate insufficient for A-score tile discrimination on heterogeneous β panels at current bridging resolution; tile means cluster 0.46–0.51 across tissues and substrates regardless of disease state".
- **Caggiano CelFiE TIM remains in atlases_deferred for cardio-epic v0.3** (already there, blocked on HM450 hg19 manifest).
- **No additional Stage 2 atlas added to cardio-epic v0.2 atlases_run.** Cardio-epic v0.2 ships with VAL-108/109/110 sealed structural results plus VAL-111 sealed atlas-deferred outcome and no atlas attribution at Stage 2.

## Card-level limitation block (text for cardio-epic v0.2)

```
## EpiSCORE HeartRef atlas — deferred to v0.3 (VAL-111 O3_TISSUE_FLOOR_DOMINATED)

VAL-111 ran the EpiSCORE HeartRef atlas (Zhu et al. Nat Commun 2022 13:3895; 3,727 450K CpGs × 5 cardiac cell types CM/EC/FB/MP/SMC) on three cardio cohorts (GSE69138 stroke blood n=589, GSE84395 PAH cultured PEC n=39, GSE84274 ascending aorta dissection/BAV/normal n=24). All tile A-scores read in the 0.46–0.51 range across all three cohorts, with within-cohort tissue discrimination ≤0.0152 (max GSE84274 MP, dissection − normal) — well below the 0.10 pre-declared threshold. The blood-control cohort shows the same A-score range as tissue cohorts, indicating gene-promoter reference panels do not produce substrate-specific contrast at A-score-tile resolution on heterogeneous β data. EpiSCORE HeartRef is methodologically sound for its design purpose (EpiDISH-style proportion estimation in heart tissue) but does not transfer to cardio-epic Stage 2 A-score tile reading. Deferred to v0.3 pending alternative bridging or a tile-coverage cardiac atlas. Caggiano CelFiE TIM remains queued for v0.3 (blocked on HM450 hg19 manifest).
```

## Reproducibility triple (CHK-7.6)

- **Inline source:** `val_111.py`, `restratify.py` (this directory)
- **Inputs:**
    - GSE69138 series matrix (GEO FTP, `GSE69138_series_matrix.txt`, ~2 GB, 590 samples)
    - GSE84395 series matrix (GEO FTP, `GSE84395_series_matrix.txt`, 39 samples)
    - GSE84274 series matrix (GEO FTP, `GSE84274_series_matrix.txt`, 24 samples)
    - EpiSCORE HeartRef bridged CSV (atlas vault, SHA-256 above)
- **Environment:** Python 3.12, pandas, numpy
- **Expected headline outputs:** per-sample A-score CSVs (3 files), `results.json` with sealed outcome, this `outcome.md`.

## Language discipline check

- ✗ "validates", "confirms", "resolves", "proves" — not used
- ✓ "consistent with", "indicating", "fails the pre-declared threshold", "deferred", "queued"
- ✗ Recipient-work-incomplete framing — not used (EpiSCORE is described as methodologically sound for its design purpose)
- ✓ Numbers reported, threshold reported, outcome selected by pre-declared rule

---

**Sealed by:** VAL-111 prereg (SHA-256: `172c6ae2a11345935c176b4a1fc57d30009ad4bac9bb9cdeeb9c8226035b78a6`).
**No biology downstream of cardio-epic v0.2 architecture decisions changes from VAL-111.** VAL-108/109/110 sealed structural results remain.
