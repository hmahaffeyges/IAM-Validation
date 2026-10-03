# DEV-ATLAS-EPIC-01 — atlas v2 and NILC composition on EPIC whole blood (development, 2026-10-03)

**Why.** Chain v3 Stage A is an NNLS over 8 Salas EPIC groups. The atlas v2 solver was set aside on 2026-10-01 because it was said to under-read EPIC
neutrophils by ~0.05 and to split T cells into subtypes without an EPIC profile. The chain needs the atlas to learn which cells are in a whole blood and
at what fractions, and needs an independent second separation. This run measures both on EPIC specimens with known composition.

**What was run.** Atlas v2 solver (frozen settings) in three task variants — a: atlas as stored (array scale through the source terms); b: array-measured
cells only; c: a with merges from the atlas's own tests — plus e (array-measured circulating cells + the mixture-identifiability rule, added after run 2),
a constrained ILC (NILC) on the atlas templates, a NILC on the 8 EPIC templates, and the current NNLS8. 117 EPIC arrays: FACS whole bloods (GSE112618, 6),
one counted donor (GSE110530, 12), three Salas DNA-mixture series (GSE110554, GSE167998, GSE182379; 36), technical replicates (GSE250556, 63).
Neutrophil identity sites never enter a composition. NILC settings chosen on GSE110554 only.

**What it measured.**
- Neutrophils, FACS whole blood (n=6): NNLS8 -0.030; atlas a -0.013, b -0.007, c -0.012, e -0.011; NILC-c +0.003, NILC-e -0.005.
  On these arrays the atlas solver does **not** under-read neutrophils by 0.05; NNLS8 is the method outside 0.02.
- Neutrophils, the six blood-like DNA mixtures (63–75 %): every method reads low, NNLS8 -0.053, atlas -0.047 to -0.043,
  NILC -0.031 to -0.035. The NILC is linear and unconstrained and its fractions sum to 1.00–1.01 there, so no component is
  missing: the neutrophil template and the neutrophil DNA in these mixtures differ. That is a template / specimen question, not a solver question.
- T cells: the atlas's twin and cross-source rule merges nothing (WGBS-only T subtypes reach r <= 0.954 with any array cell). The WGBS-only subtypes take
  ~0 in EPIC whole blood. The T-cell error comes from the unsorted parent entries (`cd4 t cells`, `cd8 t cells`, `b cells`), which are mixtures of
  the sorted subsets (r 0.994–0.997 to a non-negative combination of them). CD8 T reads high: FACS +0.05 (atlas a), 12-cell mixtures +0.11.
- Variant c (mixture rule on all atlas blood cells) removed the parents and also the array-measured memory CD4 entry, keeping the WGBS-only T subtypes:
  T-cell error grew (FACS CD8 RMSE 0.070). Variant e (same rule, array cells only) removed only the three parents: CD4 T RMSE 0.024 worst set
  (atlas a 0.063), CD8 T 0.056 (atlas a 0.111).
- Non-blood atlas mass in healthy whole blood: 1–6 % (median 2.5 % FACS, 4.5 % GSE250556) for variants a–c; removed by construction in e.
- Repeatability (GSE250556 pooled replicates): neutrophil within-person SD 0.004–0.006 for all methods; unpooled 0.020–0.022 for all (draw-to-draw, not solver).
- Stage M: after the median tare the healthy neutrophil Met-A spread is 0.019–0.033 whatever the composition source; untared, GSE250556 reads 1.14–1.21
  (other laboratory). NILC-c gives the tightest tared FACS spread (SD 0.020) and NNLS8 0.026.

**What it teaches about the gauge.** On EPIC the atlas solver is a usable composition for neutrophils; the 0.05 under-read is a property of the blood-like
DNA mixtures, shared by every method. The atlas's T-cell trouble is the unsorted parent entries, not the WGBS subtypes; the twin rule cannot see a cell
that is a mixture of other cells, the mixture test can. No blood cell beyond neutrophils is within 0.02 on every truth set for every method; B cells and
basophils are within 0.02 for most atlas and NILC variants; NK and CD8 T read high on flow-counted bloods for every method (flow vs DNA, or template).

**Chain change that follows (proposed, not applied).** STAGE_A_PROPOSAL.patch: optional Stage A engine `atlas_e` (12 array-measured circulating atlas cells,
parents removed by the mixture rule) with NILC on the same templates recorded as the second opinion and the NNLS8 fractions kept as comparator. Before
use it needs a run on FACS-counted EPIC whole bloods not used here (variant e was formed after run 2 had been read).

Files: DEV_ATLAS_EPIC_01.zip (code/, results/, RESULTS.md, fig_est_vs_true.*, STAGE_A_PROPOSAL.patch).
