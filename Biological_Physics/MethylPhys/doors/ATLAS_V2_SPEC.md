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
