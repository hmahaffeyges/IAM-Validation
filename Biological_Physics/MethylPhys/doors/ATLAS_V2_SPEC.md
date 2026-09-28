# Atlas v2 — specification (PLAN item 20), written 2026-09-27 before any machine is rented

**What v1 is.** `IAMAtlasREBUILD.csv`: 483,092 loci × 114 cells (mean and posterior SD per cell per locus; 122 sd columns including
the eight pooled classes). Each locus's posterior is independent of every other; each class was run as its own job; a cell from a
source that measured only some loci carries **the grand mean** at every other locus. Of the 114 cells, **72 fall into eleven source families** that share an exact defined-locus count (one dataset each): 252 (9 cells), 333 (2), 2,464 (4), 2,543 (9), 2,545 (3), 2,781 (2), **6,105 (22)**, 380,467 (6), 482,419–482,421 (15); the other **42 cells each have a unique count** (42 distinct values) and are their own source or a per-cell subset of one. Only 24 cells are defined on more than half the array; 53 on less than 1 %.

**What that cost us (measured, this month).** The erythroblast hub (a 252-locus cell absorbing mass from every specimen); the
twins (one cell, two sources, solved apart); PROC-COV-01's reproducible 0.067 misfit with no term to hold it; 19 of 21 foreign
templates on 1.3 % of the array; the composition solver's ±3.5-point CD4/CD8/NK error, which is the fraction confound on the
Reading tab; the sky's −0.4 to −0.8 median-z offset off the identity loci. Every one of these is a property of *how the atlas
was assembled*, not of the physics — H_min, the identity loci and A = H(mean β)/H_min are untouched by anything below.

## The model (one joint fit)

For locus *i*, cell *c*, source *s* (a source is one published dataset on one platform):

    β_obs[i, c, s] ~ Normal( μ[i, c] + δ[i, s] ,  σ_obs[s]² + σ_loc[i]² )         where measured
    μ[i, c]        = m[i] + a[i, class(c)] + e[i, c]                                the cell's true level
    e[i, c]        ~ Normal( 0, τ[class(c)]² )                                      cell within class
    a[i, k]        ~ Normal( 0, ω² )                                                class within locus
    δ[i, s]        ~ Normal( d[s], ρ[s]² )   with  Σ_s d[s] = 0                     the SOURCE term (platform + pipeline + lab)
    m[i]           ~ Beta-logit prior centred on the 450K-wide mean

- **Locus effect** m[i] is shared by every cell: the probe's own behaviour.
- **Source effect** δ[i, s] is what PROC-COV-01 measured and v1 had no slot for. It is *estimated*, not left in the residual. Two
  entries of the same cell from two sources share μ[i, c] and differ by δ — twins become one cell by construction.
- **Imputation** is free: at a locus a cell's source never measured, μ[i, c] is drawn from m[i] + a[i, class] + e[i, c] with e's
  full prior width τ — a wide posterior, honestly wide. The 252-locus cell no longer knows 480k loci it never saw.
- **Covariance** across cells at each locus comes out of the joint posterior (e[i, ·] are correlated through a[i, k]); the
  composition solver gets Σ[i] instead of unit weights — the noise model the matched filter and PROC-UNMIX-01 lacked.
- **Nothing here reads a specimen, a person, or a laboratory of patients.** Reference methylomes only. H_min is not refitted.

Loci are conditionally independent given the shared hyperparameters (d, ρ, τ, ω, σ_obs). The sampler alternates a **batched step over
loci** (all 483k at once — the GPU step) with a small update of the shared terms (CPU). NumPyro/JAX, NUTS, 4 chains, 1,000 warm-up +
2,000 draws; the batch dimension is the locus.

## Inputs
The eleven source datasets as v1 used them (manifest: `atlas/IAMAtlasREBUILD_provenance.json`, `atlas/external_manifests/`), each on
the atlas scale via **its own pipeline map** where it is not already there; the celltype→class map; nothing else. Candidate new sources
(Loyfer 2023 WGBS; the 476-methylome purified-cell set; the July 2026 single-cell body atlas) enter **after** v2 is accepted, one at a
time, through the same script.


## Revision 2026-09-28 — inputs are samples, not panels
The author's v1 build vault (`atlas_vault_OLD.zip`) shows v1 pooled other groups' **reference tables**, most of them marker panels
(Caggiano 254 CpGs, Moss/Loyfer 6,105, EpiSCORE 2k–4k and partly imputed from RNA, UniLIFE 1,906, Salas 450). That is the origin of
the coverage families above, and it means v1's per-locus "SD" was fitted to one number per cell, not to donors. v2's inputs change:
**the samples behind each table, at every locus**, so the model estimates real between-donor variance, the source term and Σ.
The eleven-source list under "Inputs" is superseded by this list; acceptance tests A1–A9 are unchanged and A8 still decides.

| source | form | status 2026-09-28 |
|---|---|---|
| Moss 2018, GSE122126 | raw 450K/EPIC IDATs → our Stage 1 | **101/101 calibrated on AWS**; 23 of 28 sorted-cell arrays pass intake (`atlas/sources/moss2018_stage1_manifest.csv`) |
| Loyfer 2023, GSE186458 | hg19 WGBS beta + depth at array CpGs | **207 sorted samples read**; 895,713 of 895,827 probes mapped; 99 % covered, median depth 32 (`atlas/sources/loyfer2023_extract_report.json`) |
| Reinius 2012, Salas 2018/2022, UniLIFE's sorted immune sets | raw arrays → Stage 1 | to fetch (accessions to verify) |
| Caggiano's source WGBS (Roadmap/ENCODE tissues) | WGBS beta at array CpGs | to fetch |
| Tian 2023 brain (GSE215353), Zhou 2026 body | single-cell pseudobulk per type | to locate |
| EpiSCORE | imputed columns flagged; replaced where a measured source exists | — |

**Entry rule for a cell** (proposed 2026-09-28, the author's "as many cells as possible so long as they are represented well"):
≥ 2 independent donors; every array passes intake / every sequencing sample has ≥ 10 reads at ≥ 90 % of loci; not a twin of an
existing entry; reads 1.00 on its own profile. A cell failing any of these waits for data; it does not enter thin.

**First cross-platform measurement (same cell, array vs sequencing), `atlas/sources/crossplatform_moss_vs_loyfer.csv`:** 10 cell
types, 460k–845k shared CpGs each. r = 0.93–0.97; median |Δβ| 0.045–0.061; the fit WGBS = a + b·array has **b 1.03–1.09 and
a −0.04 to −0.07 for every cell type** — one transfer between platforms, not a per-cell difference. That is the source term δ the
model is built to estimate, measured before the model exists.


**Sources update 2026-09-28 (later).**
- Salas 2018 (GSE110554) + Salas 2022 (GSE167998): **117 sorted-blood EPIC arrays through our Stage 1, 0 below the intake line**
  (medians 0.96–0.99); 13 types incl. basophils, Treg, naive/memory CD4 and CD8, naive/memory B, eosinophils, neutrophils.
- Reinius 2012 (GSE35069): **no raw IDATs on GEO** (only an intensity table), so it cannot enter through Stage 1. Every Reinius cell
  type is covered by the Salas sets; its red/green data exist in the Bioconductor package FlowSorted.Blood.450k if a 450K second
  source is wanted later.
- Caggiano's placenta / heart / skeletal / mammary columns come from **bulk tissue** WGBS; tissues go to the tissue atlas, not the
  cell atlas. ENCODE holds ~70 released human tissue WGBS experiments across ~40 tissues.


- Tian 2023 brain (figshare 28438499, hg38 pseudobulk, 3 donors pooled per type): streamed and read at array CpGs via the hg38
  InfiniumAnnotation manifests. **Astrocytes 861,587 CpGs, median 50 reads, 99.1 % at ≥ 10 reads**; microglia, oligodendrocytes,
  OPC the same; VLMC median 25; pericytes median 11 (64.6 % at ≥ 10) and brain endothelium median 15 (85.9 %) are thin.
  **Mapping verified across genome builds:** Tian oligodendrocytes vs Loyfer oligodendrocytes r = 0.986 (same cell, two labs, two
  builds) against 0.876–0.911 for different cells; microglia is nearer Loyfer's macrophages (0.947) than monocytes (0.926).
  Pooled profiles carry no donor variance and enter, if the author accepts it, flagged "pooled, 3 donors".

**Test T1 — the cell atlas on known tissue (after A1–A9).** Each ENCODE tissue is deconvolved by v2 at array CpGs. Pass when
(a) the dominant cells are that organ's own (stomach → gastric epithelium; heart → cardiomyocyte, fibroblast, endothelium …), and
(b) every present cell reads inside NORMAL. The first measurement of the gauge outside blood. Bars fixed before any tissue is read.

**Tissue atlas (separate).** One profile per tissue, ENCODE WGBS + the 2026 Cell Reports Methods 450K compendium; answers "what
tissue is this / what shed into this specimen"; never enters the cell deconvolution. Biopsy scoring runs on the cell atlas and is
gated on T1.

## Acceptance tests — written now, run once, no bar moves afterwards
| | test | passes if |
|---|---|---|
| A1 | PROC-COV-01 re-run on v2: median misfit of the atlas against the 48 healthy panel arrays at the identity loci | **< 0.05** (v1: 0.089 raw, 0.067 after a hand-fitted constant) |
| A2 | twins: every v1 twin pair (r > 0.985, same class) is **one** μ with two δ | all pairs; no family/twin machinery needed in the solver |
| A3 | imputation is honest: for cells defined on < 5 % of loci, posterior SD at undefined loci | median **> 0.10**, never below the measured-locus SD |
| A4 | held-out prediction: mask 5 % of measured (locus, cell, source) cells at random; predict from the posterior | coverage of the 90 % interval 85–95 %; RMSE reported |
| A5 | constructed-truth composition with the new Σ: the PROC-UNMIX-01 mixes through the solver | CD4 / CD8 / NK fraction error **≤ 0.015** (v1: ±0.035) |
| A6 | constructed-truth A on minority cells (FRACTION_AND_A): a perfect specimen's cells at 5–10 % | NK / CD8 / CD4 read within **0.02 of 1.000** (v1: 0.946 / 0.961 / 1.031) |
| A7 | foreign detection floors (PROC-STAGE2D-03 re-run with v2 templates) on the same 732 arrays | per-template floor ≤ v1's; Breast and Bladder named at 5 % |
| A8 | the reading on the majority cells is unchanged: neutrophils / monocytes on the 48 panel arrays | median \|ΔA\| **< 0.005** against v1 — the physics did not move |
| A9 | convergence: R̂ < 1.01 on every shared term and on a random 10k loci; ESS > 400 per chain-set | — |

A1–A3 and A9 are properties of the atlas itself. A5–A8 are the ones that matter to the report; **A8 is the one that says the
instrument still reads the same thing** — it must pass or v2 is not an atlas of the same instrument.

## Dry run (before renting anything)
1,000 loci × all cells × all sources on this laptop or the M5 Ultra: model compiles, chains converge (A9 on those loci), per-locus
cost measured, memory per 10k-locus batch measured. **The full-run estimate is that cost × 483.** Not before.

## Compute
One A100/H100 80 GB (Lambda / RunPod / GCP, ~$2–4 per hour): 6–12 hours estimated for the full run, plus ~2 hours of post-processing
(imputation draws, Σ per locus, A1–A9). CPU-only 64–128 vCPU: 1–3 days. The M5 Ultra runs the same code through Metal, 3–5× slower
than an A100, unattended — the machine for adding cells later, not for the first build. The 13 GB data bundle and the eleven source
manifests are what go up with the job; the atlas that comes back is `IAMAtlasREBUILD_v2.csv` + per-locus Σ (`.npz`) + the acceptance
report, saved before anything is deleted from the rented machine.

## What does not change
H_min (G-002, frozen). The identity loci (v1.1; re-derived only if A8 fails). A = H(mean β)/H_min. Tiers. Healthy = 1.00.


## Amendments (dated; the original text above is not edited)
1. **2026-09-28 — A3 replaced (author, D9: no partial cells).** v2 imputes nothing: a (cell, locus) with no observation is NOT MEASURED,
   never a prior-only mean. A3 becomes: *every unmeasured pair is marked, none carries a value* (V3). The 252-locus families of v1 do
   not exist in v2 — only whole-array cells entered.
2. **2026-09-28 — identity loci, decision pending.** The text above keeps the v1.1 identity loci unless A8 fails. The class rule
   (CLASS_USE_INVENTORY.md) removes the shared class-keyed identity set at the switch-over. The author decides before V12 whether
   each cell gets its own identity loci on v2 (then A8 compares v1 on v1.1 loci with v2 on the new loci) or v1.1 stays until A8.
3. **2026-09-28 — A2 wording.** "same class" is dropped from the twin definition; twins are tested per sample for every close pair.
4. **2026-09-28 — tissue.** The "Tissue atlas (separate)" paragraph is superseded by D13: tissue profiles are test specimens for T1,
   never references. Compute: built on CPU (AWS c7a.32xlarge, 128 vCPU), not a GPU. Model as built: `../atlas/v2/README.md`.
