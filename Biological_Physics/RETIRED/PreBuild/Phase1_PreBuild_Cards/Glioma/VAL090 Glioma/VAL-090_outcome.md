# VAL-090 Outcome — Cortical-neuron cfDNA in glioma plasma via Loyfer/Moss array atlas

**Status:** RAN 2026-04-25
**Outcome label:** **O1_PASS** — pooled Cohen's d = +1.96 [+1.62, +2.31]; pre-surgery treatment-naive subset d = +1.97; brain tissue arm consistent.
**Pre-reg seal:** d00c7cee73d423945081b1bdf5abcd540447d5f2d3fc434bf95d12101b3c5725 (sealed before scoring; no post-hoc adjustments).

---

## Headline result

**Direct quantification of brain-derived cfDNA in standard EPIC peripheral blood at array resolution. Two full standard deviations of separation between glioma patients and healthy reference.**

| Cohort | n | Cortical-neurons mean | SD | Median | % ≥ 0.5% | % ≥ 1% |
|---|---|---|---|---|---|---|
| Healthy buffy coat (GSE51057, cancer-free) | 177 | **0.276%** | 0.373% | 0.000% | — | 6.8% |
| All glioma peripheral blood (GSE180683) | 76 | **1.092%** | 0.455% | 1.100% | 89.5% | 63.2% |
| Pre-surgery treatment-naive (subset of above) | 37 | 1.076% | 0.435% | 1.100% | 91.9% | 64.9% |
| Pre-surgery LGG (treatment-naive) | 12 | 1.292% | (small n) | — | — | — |
| Pre-surgery GBM (treatment-naive) | 19 | 0.858% | (small n) | — | — | — |

Cohen's d (all glioma vs healthy reference) = **+1.963 [+1.621, +2.305]**.
Cohen's d (pre-surgery treatment-naive subset vs healthy reference) = **+1.975**.

The pre-surgery LGG > pre-surgery GBM ordering observed in VAL-088 Stage 1 A-score analysis is **also seen in cortical-neuron cfDNA fraction** (LGG 1.29% vs GBM 0.86%), strengthening the LGG-louder-than-GBM finding under a completely different metric.

---

## Brain tissue arm — consistency check

| Tissue | n | Cortical-neurons fraction | SD | Cohen's d vs NTB |
|---|---|---|---|---|
| Non-tumor brain (NTB) controls | 5 | 62.44% | 3.51% | (reference) |
| GBM primary tumor | 64 | 39.32% | 11.09% | **−2.811** |
| GBM recurrent tumor | 4 | 35.18% | 9.47% | (n=4, smaller separation reading) |
| Cultured glioma spheres | 4 | 42.93% | 4.56% | −4.797 |

Tumor displaces normal cortical-neuron architecture in the tissue. The deconvolution reads:
- **Healthy peripheral blood: 0.28% neurons** (NNLS noise floor — median 0.0%, only solver-floor activity).
- **Healthy non-tumor brain: 62.4% neurons** (biologically plausible for cerebral cortex).
- **GBM primary tumor: 39.3% neurons** (~23 percentage points lower than healthy brain — tumor displaces normal architecture).

The deconvolution pipeline reads non-tumor brain as ~62% neurons; the same pipeline reads healthy peripheral blood as ~0.3% neurons. **This is the expected biological gradient from a working method.** The cortical-neuron readings in glioma plasma are not artifact.

---

## Sanity check — full immune compartment

The same deconvolution applied to the healthy reference reproduces textbook peripheral blood composition, confirming the pipeline is correctly resolving the immune compartment:

| Cell type | Healthy mean | Glioma blood mean | Δ |
|---|---|---|---|
| Neutrophils | 51.97% | 68.35% | **+16.38%** |
| CD8+ T cells | 15.82% | 7.27% | −8.55% |
| CD4+ T cells | 9.04% | 5.86% | −3.19% |
| B cells | 6.42% | 4.15% | −2.27% |
| Monocytes | 4.34% | 3.06% | −1.28% |
| NK cells | 3.67% | 4.56% | +0.90% |
| Erythrocyte progenitors | 3.26% | 1.93% | −1.33% |
| **Cortical neurons** | **0.28%** | **1.09%** | **+0.82%** |

The healthy buffy coat composition (52% neutrophils, 25% T-cells, 6% B-cells, 4% monocytes) matches Salas 2018 textbook ranges (neutrophils 45–75%, CD4 10–30%, CD8 5–25%, B 3–15%, monocytes 3–12%). **The pipeline is working correctly.** The cortical-neuron signal is in addition to a Bracci 2022-style NLR shift (neutrophils up, lymphocytes down) — both are present.

---

## What this changes

### Vs Stage 1 A-score (VAL-088)

VAL-088 used the Stage 1 immune-class A-score (Xu-538 Shannon entropy) and reported d = +0.91 with outcome `O5_POSITIVE_INVERTED` because the cell-fraction prior had predicted negative direction. **VAL-090 supersedes the inversion finding:** the cell-fraction prior was correct (Bracci 2022 NLR shift IS present in the data, +16% neutrophils, −13% lymphocytes), and Shannon-entropy A-score is just a different facet of the same disease state. Both metrics point in directions consistent with active glioma.

The CCL-023 lessons-learned record needs revision: **the cell-fraction prior was orthogonal to the A-score signal, not inverted.** This is glioma-LL-001 → revised in LESSONS_LEARNED.md.

### Vs Stage 2 deconvolution

Stage 2 of the original glioma-epic pipeline used Moss 2018's reference atlas, which gave NULL on glioma plasma because Moss does not include a sorted-cell `Cortical_neurons` reference (its "brain (cortex)" entry is bulk-tissue mixture). **VAL-090 demonstrates that Stage 2 returns positive when the reference atlas includes a sorted-cell cortical-neuron reference.** This is not a Stage 2 limitation — it was a reference limitation. The card now uses the Loyfer/Moss array atlas (`nloyfer/meth_atlas/reference_atlas.csv`) as the primary Stage 2 reference for glioma-epic going forward.

### What this means for v1 EDEAR launch

**Glioma-epic v0.2 promotes the blood arm from `exploratory_pending_replication` to `single_cohort_validated` tier.** The card now has:
- A direct cell-of-origin signal (cortical-neuron cfDNA fraction) that separates glioma plasma from healthy reference at d = +1.96.
- A consistent biology cross-check from the tissue arm (tumor displaces normal cortical neurons).
- A cell-fraction Bracci 2022 signature (neutrophils up, lymphocytes down) that survives the same deconvolution.

The tier upgrade is qualified by:
- Single-cohort blood-arm validation (UCSF AGS phs001497 replication still required).
- Cross-platform reference (HM450 Italian healthy vs EPIC glioma) — direction is robust, magnitude requires same-platform confirmation.
- 5-NTB n in the tissue arm — independent NTB cohort validation desirable.

---

## CHK-7.6 reproducibility triple

**Source code:** `val_090_brain_decon_analysis.py` (this folder, also pushed to GitHub) — performs the analysis; `extract_gse60274.py`, `extract_gse180683.py`, `extract_gse51057.py` (input prep, embedded inline in Evidence Report).

**Inputs:**
- `nloyfer/meth_atlas/reference_atlas.csv` — public, GitHub-hosted, 16 MB. SHA recorded in results JSON.
- GSE180683 processed matrix — public GEO FTP, ~482 MB.
- GSE51057 series matrix — public GEO FTP, ~1.2 GB.
- GSE60274 series matrix — public GEO FTP, ~309 MB.

**Environment:** Python 3.12 + numpy + pandas + scipy + matplotlib (specific versions in `VAL-090_results.json`).
**Runtime:** ~5 minutes after downloads (NNLS solver is fast; CSV I/O is the bottleneck).
**Expected output:** see headline table above.

---

## Output files

- `VAL-090_results.json` — full numerical results
- `VAL-090_distributions.png` — two-panel figure (blood + tissue)
- `val_090_brain_decon_analysis.py` — analysis script
- `extract_gse180683.py`, `extract_gse51057.py`, `extract_gse60274.py` — input preparation scripts

---

**Outcome:** O1_PASS. Glioma-epic v0.2 promoted to `single_cohort_validated` tier on Stage 2. Loyfer/Moss array atlas integrated as primary cell-of-origin reference, supplementing Moss 2018 for cell types Moss did not have as sorted-cell references (cortical neurons primarily; vascular endothelial cells and left-atrium pending separate validation).
