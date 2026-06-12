# VAL-090 Pre-registration

**Title:** Cortical-neuron cfDNA fraction in glioma peripheral blood — direct deconvolution against published Loyfer/Moss array atlas

**Card:** glioma-epic v0.2
**Date sealed:** 2026-04-25
**Pre-reg author:** Heath W. Mahaffey (with Walther)
**Run trigger:** Per glioma-LL-005 — Heath's direction "the chance we can detect neuron cells in blood is not zero and they are VERY loud, so we should proceed like we did with the others" — Walther had previously deferred Loyfer/Caggiano integration to "v0.2 future task" without justification. Heath caught the unjustified deferral and directed immediate integration.

---

## Hypothesis

The published `nloyfer/meth_atlas/reference_atlas.csv` (Loyfer/Kaplan group, Hebrew University of Jerusalem, distributed alongside Loyfer 2023 Nature 613:355) contains 26 cell-type references including a sorted-cell `Cortical_neurons` reference indexed to Illumina 450K/EPIC CpG IDs. When applied directly to publicly-deposited glioma peripheral blood methylation data via standard non-negative least squares (NNLS) deconvolution, this reference will allow direct quantification of the cortical-neuron cfDNA fraction in glioma plasma vs healthy reference.

**Expected result going in:** unknown direction and magnitude. Standard cfDNA biology suggests brain contribution to plasma at healthy baseline is approximately 0.5% — at or near the typical NNLS noise floor for cell-of-origin deconvolution methods. Glioma may or may not produce detectable elevation above this floor in standard EPIC/450K array data; the only way to know is to run the analysis. The hypothesis is **descriptive, not directional**: characterize the cortical-neuron fraction distribution in glioma plasma vs healthy reference and report what the data show.

---

## Cohorts

| Role | Cohort | Accession | Platform | n | Source |
|---|---|---|---|---|---|
| Healthy reference | EPIC-Italy menarche cohort, cancer-free subset | GSE51057 | HM450 | 177 | GEO public |
| Glioma plasma test | Salas/Wiencke 2022 EPIC peripheral blood | GSE180683 | EPIC 850K | 76 | GEO public |
| Brain tissue test | Lai 2015 brain tissue 450K | GSE60274 | HM450 | 77 (64 GBM_primary + 4 GBM_recurrent + 4 spheres + 5 NTB) | GEO public |

**Cancer-free subset of GSE51057** is defined as samples with empty `cancer type (icd-10)` characteristic field (cancer-free at follow-up).

---

## Method

1. Download `nloyfer/meth_atlas/reference_atlas.csv` from GitHub at commit current as of 2026-04-25 (file SHA-256 will be recorded in results JSON).
2. Extract per-sample β matrices from each GEO series matrix file as plain text CSV (CpG ID column + per-sample columns).
3. Run `nloyfer/meth_atlas/deconvolve.py` with `reference_atlas.csv` against each input — produces per-sample × per-cell-type fraction matrix.
4. Extract `Cortical_neurons` row across samples per cohort.
5. Compute per-cohort mean, SD, range, percentiles. Compute Cohen's d between cohorts.
6. Stratify glioma plasma cohort by treatment status (pre-surgery treatment-naive vs post-treatment) and histological grade (LGG vs GBM) from manifest `time.point` and `histological.group` fields.
7. Stratify brain tissue cohort by sample title parsing (NTB vs GBM_primary vs GBM_recurrent vs sphere).
8. Generate distribution figure (boxplot + scatter) with both blood and tissue panels.

**No parameter tuning, no panel selection, no post-hoc adjustment.** The reference atlas is fixed external; the NNLS solver is `scipy.optimize.nnls`; the analysis pipeline is the published one from `nloyfer/meth_atlas`.

---

## Decision criteria

- **O1_PASS** — Cohort separation in cortical-neuron fraction with Cohen's d ≥ +1.0 (glioma plasma vs healthy reference) AND consistent direction in pre-surgery treatment-naive subset.
- **O2_PARTIAL** — d ∈ [+0.5, +1.0) OR direction flips in pre-surgery subset.
- **O3_NULL** — d < +0.5 in either pooled or pre-surgery analysis. Brain cfDNA is below detection floor in standard array peripheral blood; framework predicts this and the result confirms it.
- **O4_NEGATIVE** — d < −0.5 (glioma plasma reads lower cortical-neuron fraction than healthy). Would be biologically anomalous; would prompt re-examination of cohort metadata for confounds.
- **O5_UNEXPECTED** — Any result that doesn't fit O1–O4 (e.g., distributions look right but variance pattern unexpected; LGG > GBM or other counterintuitive sub-finding requiring follow-up).

For the brain tissue arm:
- Expected: NTB controls show high cortical-neuron fraction (>40%, consistent with cerebral cortex composition); GBM_primary shows lower cortical-neuron fraction (tumor displaces normal architecture); GBM_recurrent at or below GBM_primary; cultured spheres at or below NTB.
- If NTB cortical-neuron fraction reads <30%, the deconvolution is not capturing brain architecture and the pipeline must be re-validated.

---

## Reproducibility (CHK-7.6)

**Inputs:**
- `nloyfer/meth_atlas` reference: `https://github.com/nloyfer/meth_atlas/blob/master/reference_atlas.csv` (~16 MB)
- GSE180683 processed matrix: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180683/suppl/GSE180683_Matrix_processed.txt.gz` (~482 MB)
- GSE51057 series matrix: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51057/matrix/GSE51057_series_matrix.txt.gz` (~1.2 GB)
- GSE60274 series matrix: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60274/matrix/GSE60274_series_matrix.txt.gz` (~309 MB)
- Manifest: `GSE180683_manifest.json` (already in repo from VAL-088)

**Environment:**
- Python 3.12
- numpy ≥1.20, pandas ≥2.0, scipy ≥1.10, matplotlib ≥3.7
- RNG seed: 20260425 (none needed — NNLS is deterministic)

**Expected runtime:** ~5 min after downloads (NNLS deconvolution is ~30 s per cohort on a single CPU; bottleneck is pandas CSV I/O for the EPIC matrix).

**Expected memory:** ~6 GB peak (loading EPIC β-matrix as float64).

**Expected headline output:** Glioma plasma cortical-neurons fraction mean ≈ 0.01 (i.e., ~1%), healthy reference mean ≈ 0.003 (~0.3%), Cohen's d in the range [+1.0, +3.0]. Brain tissue NTB cortical-neurons mean ≈ 0.6 (60%), GBM primary ≈ 0.4 (40%).

---

**Pre-registration seal SHA-256:** computed at commit time, stored in `VAL-090_PREREG_SEAL.txt`.
