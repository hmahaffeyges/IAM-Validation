# CPG_CMB_vKISS — Lessons Learned (session 2026-06-29)
**Walther + Heath W. Mahaffey.** Append to the canonical CPG Lessons_Learned. These are the
hard-won lessons from building and combing the vKISS clinical report.

---
## L-1 · Cutting the display is not cutting the gate
When removing a behavior (the NILC AND-gate), removing the *text that describes it* is not the same
as removing the *logic that does it*. The AND-gate (`celltype_agreed`) was still filtering the census
and the departure ranking long after its collapsible was deleted — strangling the report to 1 cell.
**Rule:** when a feature is cut for KISS, grep for and remove the *gating logic*, then verify the cell
count / output actually changes. The fix took 1→45 cells, 0→67 CIs, strawman 3→41.

## L-2 · The disease matrix IS the comparison engine — and its safety depends on the origin map
The matrix (v1_13) is loaded, turned into per-cell signature vectors, and scored against the patient's
per-cell A-departures by **directional concordance** (`_concordance` → `route_B_concordance`). The
concordance compares **direction, not magnitude** (sign × disease-weight over cells the patient moved on),
which is exactly correct: the cohort supplies the *direction*, the A-score supplies the *magnitude*.
**The bug that frightened patients:** `disease_origin_cells.json` was MISSING, so the specificity
classifier had no cell-of-origin map and could not apply its tissue-origin rule — solid cancers
(lung/breast/rectal) matched on generic immune cells got labeled SPECIFIC and surfaced by name.
**Rule:** a solid-tissue cancer is SPECIFIC only when one of its own origin cells is actually present;
from whole blood, where the tissue is absent, such a match is NON_SPECIFIC_GENERIC ("the generic pattern
wearing the disease's name"). The origin map is what enforces this. Built it from the matrix's own
non-immune cells per disease; verify it loads at runtime.

## L-3 · Low atlas representation IS a noise source, and the MCMC posterior already flags it
Cells with thin atlas coverage (Microglia, macrophage, Kupffer, the lowercase aliases) have 10–18×
wider posterior sd than well-covered cells (~0.07 vs ~0.0044). Those are exactly the cells that read
wild on healthy blood. The per-cell 95% CI (propagated from the brightness posteriors) already encodes
this — a thin-reference cell comes back with a wide CI. **Rule:** lean into the CI; a wide interval is
the down-weight signal, not clutter. Two complementary noise sources: atlas-side (thin reference → wide
CI) and sample-side (cell absent → reads background, handled by the floor/presence gate).

## L-4 · Fold, don't cut (the AstroGenetics layer earns its place)
KISS does not mean delete the marvel. It means **lead with the lean clinical read** (verdict → cells +
CI → gauges → matches → refer) and **fold** the deep material below as collapsibles (Cosmic Methylome
Background, crown-jewel wall, straw man, machine-readable snapshot, the "How CPG works" explainer). The
reader reaches a decision without scrolling; the curious clinician and the reviewer expand the rest.

## L-5 · The line we never cross — physics measures, cohorts only point
Measuring a patient is physics and self-calibrating: **A = H(β)/H_min**, intrinsic to the patient,
transfers across platforms and populations. Learning a disease is empirical and cohort-derived, but it
enters **only as a direction** (a sign/unit vector), never as a baseline. The reference a patient is
scored against is ALWAYS derived (IAMAtlas, the informational floor, μ=1.0, the Mahaffey margin). The
moment a cohort's mean/SD becomes the yardstick, the line is crossed and the model works only in its
own cohort. This is the whole difference from GRAIL and the aging clocks.

## L-6 · Patient-facing language: A-score first, never frighten
Lead every read with the A-score against the reference gauge (H_min floor · ~1.00 mid healthy band ·
1.10 breach). "Resemblance" is the *shape* of a cell pattern — never a diagnosis, probability, or stage,
and never a bare cancer name as a headline. Gate the confirmation **before** presenting a concern, never
frighten-then-retract. Explain Mahalanobis, MCMC, posteriors as plain ideas; the jargon is for reviewers.

## L-7 · AD is architectural suppression toward H_min, not "bidirectional"
At the per-cell-type (A-score) level AD is uniformly suppressed (AIBL fan-out: 20 significant negative,
0 positive) — advanced aging of informational fidelity. The "bidirectional" per-CpG decomposition is the
*reason the pooled A-score cancels* (hence the sealed directional panel), NOT the architectural direction.
Lead with suppression; frame the panel as the fix for pooled cancellation. The directional composite is a
panel score, **not an A-score**.

## L-8 · One environment, full consistency (operational)
Walther runs everything — renders, runs the VALs, pushes to the repo. There is no separate "Heath's box."
Anything the report needs at runtime (healpy, the cpg→HEALPix mapping, the plates, the origin map, the
strawman assets, tier_breakpoints.json) must resolve **in this environment**, via `CPG_ENGINE_ROOT`.
"Renders on the production box" is never an acceptable hand-off — install it and render it here.

## L-9 · Calibration discipline (LESSON-DECONV-01, reaffirmed)
Cached raw β trips the input-scale guard (classes read below floor) because it lacks the per-sample noob
calibration the production IDAT path applies. A "healthy reads healthy" full report needs a noob-calibrated
whole-blood sample. The guard firing is correct behavior, not a builder bug.


### LESSON-SCALE-01 — Three β scales; the floor lives on one of them (discovered via PHASE 1, 2026-09-20; foreseen in VAL-003's April 2026 output)

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

**Lesson:** a within-pipeline statistic is blind to a scale offset by construction. The absolute gauge is the only instrument that sees it, and it saw it on the first absolute run. Never re-derive the floor to fit a pipeline; map the pipeline to the floor.

**Closed in code (2026-09-20):** Stage 1 stamps `meta['pipeline']`; `cpg_conductor.stage_1s_scale_map` + `stage_b_identity`. A lesson that lives only in a document is re-learned; this one now refuses to report an unmapped reading.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.
