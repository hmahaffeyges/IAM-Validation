# The completion sprint (spring 2026), scored 2026-09-19

The original plan is kept verbatim in `COMPLETION_SPRINT_original.txt`. Its own summary: "roughly 12–15 focused sessions … from C- to A."
Author's verdict, 2026-09-19: **"Too ambitious too quick. It ended up harming rather than helping. These should have been worked on long after the bones were trusted."**

| phase | deliverable | what happened |
|---|---|---|
| A1 | `cpg_null_runner.py`, nulls N1–N8 | **BUILT, live** (`MethylPhys/chain/CPG_Null_Runner/`). Every post-Atlas CPG-VAL carries `null_results.json`. The one phase built on trusted bones; still the best-engineered part of the chain. |
| A2 | synthetic patient generator | **BUILT** (`RETIRED/…/Synthetic_Patient_Generator/`; harness in `MethylPhys/chain/report_builders/`). N7 depends on it — to be restored beside the null runner. |
| A3 | Family A VALs through the nulls | **DONE** (CPG-VAL-001…007 all carry null results). |
| B1 | foreground registry | never built |
| B2 | NILC second deconvolver | **BUILT → CUT → REINSTATED.** Cut 2026-07-02 (c1be0c3) for collapsing on correlated blood mixtures and deleting correct calls; rerun as designed in September (PROC-NILC-01) and vindicated — the divergence was marking where the atlas does not separately determine the composition. Reinstated 2026-09-22 as `stage_2b_second_opinion`: class-level comparison against Walther, agreement bar L1 ≤ 0.10, reported as a flag and never as the composition. |
| B3/B4 | age / sex / smoking foreground modules | **BUILT → REFUSED** (SOP §104): the methylome's foreground is the patient's own biology — annotate, never subtract. In `RETIRED/…/IAM_Cellular_Age/`. |
| C1 | C(d) genomic-distance correlation, "acoustic peaks" | never built |
| C2 | bispectrum | never built |
| C3 | **banana degeneracy** mapper | never built ("I never got my banana degeneracy") |
| D1/D2 | sim-based and per-CpG covariance | never built |
| E1 | cellular age clock | **BUILT as a trained clock (v1)**, later judged wrong (not physics); replaced by the band inversion (v3), which does not yet calibrate. Not reportable. |
| E2/E3 | per-card likelihood + MCMC posteriors | never built; scoring remains threshold + age band |
| F1–F3 | chain audit, v2 report, v2 checklist | report iterated v2 → v10; checklist v2 not written; the audit is what the 2026-09 PROC series is doing now |

## The lesson (a Part II page)
Phase A worked because it tested a chain that already existed. Everything after it was built on bones not yet trusted: an atlas still going flat (see `MethylPhys/atlas/IAMAtlas_FLATNESS_LESSON.md`), a band compiled from nine pipelines, a gauge formula that changed four times between 2026-06-11 and 07-01. So B was built and torn out, and C–E had nothing to stand on. The right order was A → **trust the bones** (Stage 1 bit-identical; deconvolver conformant; anchors reproducing; the band rebuilt from one cohort through one pipeline — Phase 1 of the September plan) → then the correlation structure, the degeneracies, the Bayesian layer. The sprint had the right pieces in the wrong order. Its own sign-off question 5 named the risk — "the discipline doesn't hold when the first inconvenient result comes back" — and the discipline did hold for individual VALs (VAL-102 voided in four minutes, seal preserved); it did not hold for the sequence.

What the sprint still gets right, for later: C1 (two-point correlation per class with MASTER-style mode coupling), C3 (2D posterior shape for A-score pairs — the CIMP axis is the first), and E2/E3 (a per-card likelihood marginalised over composition and age, replacing thresholds) remain the correct next layer **once Phase 1 has re-derived the bands.** Not before.
