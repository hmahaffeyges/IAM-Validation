# VAL-119 — Pre-Registration

**VAL ID:** VAL-119
**Card target:** bladder-epic v0.1 (Phase B calibration anchor)
**Atlas under calibration:** EpiSCORE BladderRef (CpG-bridged), 2,696 CpGs × 4 bladder cell types
**Calibration cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (same cohort used for VAL-106/107/112/113 cardio + VAL-117 prostate calibration anchor)
**Substrate:** TCGA HM450K sesame Level 3
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β access:** YES (this prereg.md is sealed before val119 script reads any β values)

---

## Question

Does the EpiSCORE BladderRef CpG-bridged matrix produce per-tile A-score readings on a structurally-separated healthy substrate-matched cohort (TCGA HM450K sesame Level 3 adjacent-normal) that pass CHK-3.1A (full-genome substrate baseline) AND CHK-3.1B (atlas-subset coverage) AND CHK-3.1C (atlas dedup)? If yes, the per-tile healthy-floor distributions sealed here become the calibration anchor for BladderRef in bladder-epic v0.1 production scoring on EPIC 850K and HM450K substrates.

This is the analog of VAL-117 prostate calibration: same cohort, same CHK-3.1A/B/C protocol, same substrate-matched-healthy-reference principle, same gene-promoter atlas family. Different atlas (BladderRef vs ProstateRef) and different cell-type list (4 cell types: EC/Epi/Fib/IC vs 6 cell types: BE/EC/Fib/LE/Leu/SM).

---

## Why this calibration matters operationally

The bladder-epic v0.1 card scores Stage 2 against the layered Moss+Loyfer atlas, which has a single `Bladder` tile (urothelial bulk reference, n=5 sorted bladder epithelium WGBS samples per Loyfer 2023 Nature). EpiSCORE BladderRef adds bladder sub-cell-type resolution beyond that single bulk tile:

- **EC**  = Vascular Endothelial Cells (intra-bladder vasculature)
- **Epi** = Urothelial Epithelium — **the bladder cancer cell of origin**
- **Fib** = Fibroblasts (stromal)
- **IC**  = Immune Cells (intra-bladder)

For the patient flow (Stage 1 immune red flag → Stage 2 cell-of-origin localization → Stage 3 immune fine-tune), separating Epi from Fib/EC/IC matters: bladder urothelial carcinoma is an epithelial-origin disease analogous to prostate luminal-origin disease. A trajectory score that reads Epi specifically — separated from intra-bladder Fib stromal, from vascular EC, from intra-bladder IC immune — is a higher-resolution trajectory than a single bulk `Bladder` tile.

This calibration VAL is what unblocks integration of BladderRef into bladder-epic v0.1 Phase C scoring.

---

## Atlas inventory + provenance

### EpiSCORE BladderRef
- **Source paper:** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: 10.1038/s41592-022-01412-7
- **Repository:** https://github.com/aet21/EpiSCORE
- **Source matrix:** `BladderRef.rda` → `mrefBladder.m` (163 Entrez Gene IDs × 4 cell types + weight)
  - SHA-256: `a357383a492ebd6ec6262cb0bfba45f970c6a266ef2a1b83f813f31164a42135` (BladderRef.rda)
  - SHA-256: `f73fbeab74dfbe5aec2829303757908df569bb969101180c2875a46505a3e758` (atlas_vault BladderRef__mrefBladder_m.csv)
- **Bridge methodology:** Same as VAL-094 BreastRef + VAL-111 HeartRef + VAL-117 ProstateRef — Entrez Gene IDs broadcast to all 450K CpGs mapping to that gene via EpiSCORE's `probeInfo450k.lv`.
  - probeInfo450k.rda SHA-256: `1b4d0bb8ebd0de3a5bd8b1c9cbf170599fce920da399076182070bdd93b57ca8`
- **Bridged CpG matrix:** `episcore_bladderref_cpg_bridged.csv`
  - 2,696 unique 450K CpGs × 4 cell types
  - 158 of 163 source Entrez IDs covered (5 had no probeInfo450k mapping: 1880, 2252, 26521, 51699, 54829)
  - SHA-256: `26b7ee3cb7254e28c1dab5bb4bd2c405f35c46f856f429b40aeab087d7f2ca16`
- **License:** GPL-2 per EpiSCORE repository

### Atlas family classification
**Gene-promoter atlas family** (same as VAL-111 HeartRef, VAL-117 ProstateRef; NOT tile-coverage-WGBS-derived). Per DISC-CARDIO-004 + DISC-PROSTATE-001 (atlas family fitness depends on per-tissue cell-type distinctness), gene-promoter atlases produce useful within-cohort A-score variance ONLY when the cell-type set spans markedly different gene-promoter methylation profiles for the marker genes in that tissue.

**The per-tissue test record so far (third entry):**

| Atlas | Tissue | n cell types | Outcome | Max within-cohort tile range |
|---|---|---|---|---|
| HeartRef (VAL-111) | cardiac | 5 (CM/EC/FB/MP/SMC) | O3_TISSUE_FLOOR_DOMINATED | 0.0152 |
| ProstateRef (VAL-117) | prostate | 6 (BE/EC/Fib/LE/Leu/SM) | O1 sealed | 0.0597 |
| **BladderRef (VAL-119)** | **bladder** | **4 (EC/Epi/Fib/IC)** | **TBD by this VAL** | **TBD** |

**Hypothesis going in.** BladderRef's 4-cell-type set is sparser than ProstateRef's 6-cell-type set — fewer architectural distinctions to expose per-tissue methylation difference. However, the urothelial epithelium (Epi) vs the IC immune compartment vs Fib stromal vs EC vascular ARE four anatomically distinct compartments at the gene-promoter level (urothelium has barrier-secretory-epithelial gene programs; IC has immune-cell programs; Fib has stromal programs; EC has vascular programs). The hypothesis is that BladderRef separates closer to ProstateRef than to HeartRef. The actual outcome is what the data says when this prereg is sealed and val119 script runs.

### CHK-3.1C dedup pre-check
BladderRef bridged matrix has 2,696 rows with 0 duplicate probeIDs at bridge time (verified at bridge_bladderref_to_array.py execution 2026-04-30). val119 script will re-verify CHK-3.1C dedup gate on load.

---

## Calibration cohort

- **TCGA-KIRC adjacent-normal:** n=160, sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/KIRC/`
- **TCGA-PRAD adjacent-normal:** n=50, sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/PRAD/`
- **Combined n:** 210 (same as VAL-106/107/112/113/117 calibration anchor)
- **Substrate:** TCGA HM450K sesame Level 3 (canonical substrate baseline, f_extreme 55.87% ± 2.44% per VAL-106 sealed)

**Structural separation note (CCL-041).** TCGA-KIRC and TCGA-PRAD adjacent-normal are histologically normal kidney/prostate tissue from cancer patients. They are NOT bladder tissue. For a calibration cohort intended to set healthy-floor thresholds on a bladder-specific atlas, this is BY DESIGN — the goal is to set thresholds against substrate-matched **healthy non-bladder tissue**, so that when the same atlas reads bladder cohorts (TCGA-BLCA tumor + adjacent-normal at Phase C), differential signal can be attributed to bladder biology rather than substrate noise. This is the standard cookbook calibration discipline established by VAL-106/107/112/113 and re-used for VAL-117 prostate (where TCGA-PRAD adjacent-normal was inside the calibration set; for VAL-119 bladder, neither cohort tissue is bladder, which is structurally cleaner).

---

## Pre-locked outcomes

Per CHK-2.1 (all outcomes pre-locked, none added post-hoc):

### O1 — `BLADDERREF_CALIBRATION_SEALED`

CHK-3.1A passes on ≥190/210 samples (≥90%); CHK-3.1B passes on ≥200/210 samples (≥95% under threshold of 80% per-sample atlas-subset coverage, per VAL-117 amendment-formalized CHK-2.8 substrate-floor finding for TCGA HM450K small-atlas-subsets); CHK-3.1C passes (no duplicate probeIDs in bridged matrix); per-tile healthy-floor A-score distributions (mean, sd, n, q2.5, q5, q50, q95, q97.5) sealed for all 4 tiles (EC, Epi, Fib, IC); CHK-3.1B q5 threshold sealed; **maximum within-cohort tile range ≥ 0.02** (per DISC-CARDIO-004 + DISC-PROSTATE-001 tissue-floor-dominated threshold).

**Outcome:** BladderRef enters bladder-epic v0.1 atlases_run with calibration anchor VAL-119.

### O2 — `BLADDERREF_CALIBRATION_PARTIAL`

CHK-3.1A passes on 75-90% OR CHK-3.1B passes on 85-95% (substrate edge case). Calibration sealed but flagged as partial; v0.1 atlases_run inclusion deferred pending platform-specific re-calibration on a second healthy cohort.

### O3 — `BLADDERREF_TISSUE_FLOOR_DOMINATED`

Per-tile healthy-floor A-scores all cluster within a tight band with within-cohort tissue discrimination max < 0.02 (analog to VAL-111 HeartRef pattern). Seals as gene-promoter-atlas-family floor finding for bladder tissue. Atlas → atlases_deferred for bladder-epic next version with explicit unblock dependency. Logged as DISC-BLADDER-NNN finding propagating to LESSONS_LEARNED.md per CCL-043 + DISC-CARDIO-004 generalization. **This outcome would be the third data point on the per-tissue gene-promoter atlas family rule and would tilt the rule toward "fewer-cell-type gene-promoter atlases collapse" rather than "specific-tissue-only collapses."**

### O4 — `BLADDERREF_BRIDGE_FAILURE`

CHK-3.1A or CHK-3.1B fails on >25% of samples; or CHK-3.1C dedup fails (duplicate probeIDs in bridged matrix); or smoke test produces all-NaN tiles or all-zero CpG-intersection failures. Bridge engineering bug — defer atlas, log DISC-BLADDER finding, propagate to LESSONS_LEARNED.md.

### O5 — `BLADDERREF_UNEXPECTED`

Anything else not anticipated in O1-O4. Per CCL-032 (data integrity → biology → framework), classify as O5 if data integrity is uncertain or result contradicts expected gene-promoter-atlas behavior. Convene with Heath before sealing direction.

---

## Pre-locked thresholds (CHK-2.1)

Per VAL-106/107/112/113 cardio + VAL-117 prostate precedent on the same calibration cohort:

| Threshold | Pre-locked value | Source |
|---|---|---|
| CHK-3.1A f_extreme baseline | ≥ 50.0% | VAL-106 sealed at 55.87% ± 2.44% |
| CHK-3.1A f_middle ceiling | ≤ 12.0% | VAL-106 sealed at 7.42% ± 0.75% |
| CHK-3.1A pass rate | ≥ 190/210 (≥90%) | VAL-106/107/117 envelope |
| CHK-3.1B atlas-subset coverage threshold | **≥ 80% per sample** | VAL-117 CCL-041 amendment formalized as CHK-2.8 substrate-floor for TCGA HM450K (VAL-117 observed 80.18-88.13%, q5=86.1%) |
| CHK-3.1B atlas pass rate | ≥ 200/210 (≥95%) | VAL-117 envelope |
| CHK-3.1C dedup | 0 duplicate probeIDs in bridged matrix | Hard gate (bridge already verified PASS at 2026-04-30) |
| Tissue-floor-dominated threshold | within-cohort tile range < 0.02 | VAL-111 (HeartRef cluster 0.46-0.51, max range 0.0152, FIRED O3); VAL-117 (ProstateRef max range 0.0597, did NOT fire O3) |

**Per CHK-2.7 magnitude-direction discipline.** This calibration outcome is direction-agnostic — A-score distributions on substrate-matched healthy reference are not pre-locked positive or negative. Outcome thresholds are MAGNITUDES (range thresholds, pass-rate percentages, dedup gates) without direction labels. Direction-aware Phase C scoring against bladder-disease cohorts comes later in VAL-121.

---

## Reproducibility triple (CHK-7.6)

### Source code
`val119_bladderref_calibrate.py` — Python 3.12 stdlib + numpy. Loads bridged BladderRef CSV, loads TCGA β files from KIRC + PRAD adjacent-normal directories, computes per-sample CHK-3.1A on full genome + per-sample CHK-3.1B on atlas subset + per-tile A-scores using H_min anchors per cell-type class assignment.

H_min anchor list (frozen 2026-04-06 G-003b MCMC posteriors):
- **EC**  → stromal class    H_min = 0.862950
- **Epi** → secretory class  H_min = 0.843264
- **Fib** → stromal class    H_min = 0.862950
- **IC**  → immune class     H_min = 0.838889

### Inputs
1. **Bridged atlas matrix:** `episcore_bladderref_cpg_bridged.csv`
   - Path: `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv`
   - SHA-256: `26b7ee3cb7254e28c1dab5bb4bd2c405f35c46f856f429b40aeab087d7f2ca16`
   - Size: 167 KB (estimated; actual measured at script execution and recorded in outcome.md)
   - 2,696 CpGs × 4 cell types + weight column
2. **TCGA-KIRC adjacent-normal:** sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/KIRC/`; 160 files; ~12-13 MB each
3. **TCGA-PRAD adjacent-normal:** sesame Level 3 β files at `/home/claude/edear_working/VAL-106/calibration_betas/PRAD/`; 50 files; ~12-13 MB each
4. **Total cohort:** 210 samples; ~2.7 GB total

### Environment
- Python 3.12.3
- numpy 2.4.4
- No pandas dependency for the calibration script (csv stdlib only — same as VAL-117)
- Expected runtime: ~30-60 seconds for n=210 cohort
- Expected memory: ~500 MB peak (210 β vectors of ~485k CpGs each held simultaneously)

### Expected headline output
- `VAL-119_calibration_results.json` — per-tile mean, sd, n, q2.5, q5, q50, q95, q97.5
- `VAL-119_per_sample_calibration.csv` — per-sample CHK-3.1A f_extreme, f_middle, n_subset_valid, per-tile A-scores
- `outcome.md` — sealed outcome class with comparison to VAL-111 + VAL-117 precedent

---

## RNG seed

20260420 (cookbook standard).

---

## SHA-256 of this prereg

To be computed at seal time and recorded in `PREREG_SEAL.txt` before val119 script reads any β files.

---

## Pre-registered audit chain

This prereg seals against:
- The bladder-epic v0.1 Phase 0 cohort survey signed off 2026-04-30
- Calibration TODO v0.5 Phase B requirement
- Guardrail #11 (calibration before testing is the inviolable order)
- CCL-041 (prereg locked before β read)
- CCL-046 (atlas selection traces to canonical-document-named candidates per CHK-5.12)
- CHK-2.7 (magnitude-based thresholds for direction-ambiguous outcomes)
- CHK-2.8 (TCGA HM450K substrate-floor for atlas-subset coverage threshold ≥80%)

val119 script execution begins ONLY after this prereg.md is sealed and SHA-hashed. Outcome sealed against pre-locked thresholds above.

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**
