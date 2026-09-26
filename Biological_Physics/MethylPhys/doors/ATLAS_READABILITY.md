# What this instrument can and cannot read — measured from the atlas, 2026-09-26

Five procedures failed this week and every one of them converged on the reference. This document stops
treating that as a recurring surprise and measures it, so that a procedure can be ruled out **before** it is
written rather than after it fails.

## 1. The atlas is an immune atlas with eight labels on it

| class | cell types | examples |
|---|---|---|
| **immune** | **51** | B, Baso, Bmem, Bnv, … |
| cycling | 19 | **Colon_epithelial_cells**, small_intestine, Bladder, Basal, BE, Antrum_undiff |
| secretory | 18 | Breast, Acinar, Chol, Endocrine, Antrum_diff, Corpus_diff |
| progenitor | 11 | CMP, GMP, MEP, MPP, L-MPP |
| terminal | 9 | Cortical_neurons, Glia, CM, Kera_diff, Left_atrium |
| stromal | 5 | adipocyte, endothelial, fibroblast, smooth_muscle |
| stem_adult | **1** | HSC |
| stem_pluri | **1** | stem_pluri |

Immune has 51 of the 115 cell types and by far the deepest source support. Every result that has ever worked
on this instrument is an immune result in whole blood. That is not a coincidence and it is not tuning.

**A pooled class cannot resolve an organ, by construction.** `secretory` pools breast, pancreas, liver,
thyroid and stomach; `cycling` pools colon, small intestine, bladder and skin basal cells. A reading on
`cycling` is not a reading on colon.

## 2. Three class pairs are collinear — the components are not individually identifiable

Correlation between the class reference columns, across the 22,548 loci where all eight classes are present:

| pair | r | |
|---|---|---|
| **stem_adult vs progenitor** | **+0.989** | the same language |
| **progenitor vs immune** | **+0.958** | |
| **cycling vs secretory** | **+0.955** | |
| stem_adult vs immune | +0.947 | |
| stem_pluri vs terminal | +0.817 | |

Condition number of the eight-class design: **46**.

**This is the cosmologist's situation exactly, and it is why the covariance is the highest-value unspent
item.** Two components correlated at r = 0.989 are individually badly determined while their *sum* is well
determined. The chain already knows this in one place — §108 pools progenitor and stem_adult into a joint
component on whole blood — but it knows it as a **hand-written special case**, not as a consequence of
carrying the covariance. With the off-diagonal terms retained, that pooling would fall out automatically for
every pair, at every specimen type, with honest uncertainties attached, instead of being a rule someone had
to notice and write down.

It is also the fix for the failure that closed PROC-PARTIAL-01: fitting composition with a diagonal
covariance is ordinary least squares, and the +0.067 β misfit that sank fidelity recovery is exactly what a
generalised least-squares fit with the real covariance is for.

## 3. A correction to PROC-TISSUE-01, found by this audit

PROC-TISSUE-01 reported the GSE131013 specimens as **23 % epithelial**, computed as `secretory + terminal`.
That was wrong: **colon epithelium is classified in `cycling`** in this atlas, not in `secretory`. Recomputed
per sample with every epithelium-containing class:

| group | secretory + terminal (as reported) | with `cycling` included |
|---|---|---|
| healthy mucosa | 0.230 | **0.446** |
| adjacent normal | 0.240 | **0.465** |
| tumour | 0.228 | **0.479** |

**The verdict does not change** — B6 required > 0.50 and 0.446 still fails it — and the scored class is still
immune, since immune (0.382) remains the largest single class. But the specimens are roughly **45 %**
epithelial rather than 23 %, and the outcome document has been corrected. The lesson is the one this
document exists for: *check which class actually contains the cell type you mean before writing a bar about
it.*

## 4. What is on the shelf, re-ranked by what this audit now shows

1. **The full cell-type covariance at one address.** Recoverable from the per-class MCMC archives; no new
   data. It converts a hand-written pooling rule into a general result, attaches real uncertainties to every
   fraction, and is the named fix for the PROC-PARTIAL-01 bias. **Highest value, and the measurement above is
   the argument for it.**
2. **Degeneracy analysis** — which composition solutions are genuinely distinguishable, rather than assuming
   the reported one is unique. This is no longer optional: the collinearity table is a degeneracy table, and
   a per-specimen version of it would have predicted PROC-TISSUE-01's failure *before* the download rather
   than after the analysis.
3. **The angular power spectrum of the residual sky** — partially spent. PROC-CLS-01 computed it and found
   real large-scale structure against a within-mask permutation null, but was not commissioned as a
   reference because two of its bars were arithmetically unreachable on 450K. The tool works; the
   pre-registration around it needs rewriting with platform-aware floors.
4. **Difference maps** — still blocked, and today confirmed the blockage is not solvable by searching harder:
   EPIC-Italy's public extract carries no subject identifier, so two draws from one person cannot be linked.
   This remains an acquisition problem, not a compute problem.

## 5. The caution, which outranks all four

The defect that nearly cost two real specimens this week had nothing cosmological about it: three
contaminated control addresses, and a mean where a median belonged. **The elegance of a borrowed method does
not make ordinary instrument work optional, and the analogy is most dangerous where it is most satisfying.**

Accordingly, each item above ships with the ordinary check it does *not* replace:

| borrowed method | the ordinary check it does not replace |
|---|---|
| covariance / GLS fit | residual inspection on real specimens — does the fit actually reconstruct the observed β |
| degeneracy analysis | control-address QC, before any solution is called degenerate |
| power spectrum | per-locus outlier and contamination screening on the residual that feeds it |
| difference maps | technical-replicate noise measured, not inferred from cross-sectional spread |

## 6. What follows for disease choice

**Read the disease the instrument can already read: one whose diseased cells *are* the immune compartment.**
Not a solid tumour reported on at second hand by immune bystanders — that is the configuration that failed
in blood (PROC-PARTIAL-01) and could not even be tested in tissue (PROC-TISSUE-01).

That points at haematological malignancy and autoimmune disease in whole blood, where the malignant or
dysregulated cells are immune-lineage, present at high fraction, on the platform and substrate the chain is
commissioned for — and where the reference support is 51 cell types deep rather than one.
