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

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Record/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

**Lesson:** a within-pipeline statistic is blind to a scale offset by construction. The absolute gauge is the only instrument that sees it, and it saw it on the first absolute run. Never re-derive the floor to fit a pipeline; map the pipeline to the floor.

**Closed in code (2026-09-20):** Stage 1 stamps `meta['pipeline']`; `cpg_conductor.stage_1s_scale_map` + `stage_b_identity`. A lesson that lives only in a document is re-learned; this one now refuses to report an unmapped reading.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Record/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Record/PROC_data/PROC-HMIN-BOOT-01/`.

**LAB ZERO — PANEL SPECIFICATION (PROC-PANEL-01 → PROC-PANEL-03, 2026-09-20; supersedes the '20–30 arrays' wording above).** A laboratory's zero is measured once on **40** healthy arrays of any age mix through the same Stage 1 and map: z = median[A − c(decade)] − 1, where c is the reference healthy age curve (`reference_age_curve_v1.json`; healthy immune A rises ≈0.045 from the teens to the eighties within a lab, while between-lab offsets are parallel). A patient reads A″ = A − c(decade) − z. Tested leave-one-lab-out on four labs: a band built on three holds 75–84 % of the fourth's healthy donors. In code: `MethylPhys/chain/lab_zero.py` — panels under 40 are refused and `lab_zero=UNSET` is not reportable. Record: `Record/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/`.

**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Record/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Record/PROC_data/PROC-HISTORY-01/`.

**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** `cpg_conductor.run_full` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on `iamatlas_gauge_identity_loci_v1_0.json`, on mapped β, minus c(decade) (`reference_age_curve_v1.json`), minus the laboratory zero (`lab_zero.py`), placed in `identity_band_v3.json` (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is `diagnostic_marker_union` — never the reported A. Stages 5 and 6 carry `pending_recalibration=True`. Test: `MethylPhys/kit/test_gauge_switch.py`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.

**STAGE 5 RE-BASED (PROC-MAHA-01, 2026-09-21; row 5 BUILT, not commissioned).** The departure now reads the identity gauge: z = (A″ − 1)/σ, σ = 0.0204 from `identity_band_v3`; on whole blood one banded axis, so the number is |z_immune| against 1.960 / 2.576; `bundle['mahalanobis']` carries the long keys the report builder reads plus the short aliases; UNSET → not reportable. The eight-class derived hull is `diagnostic_hull_marker_union`. **M2 failed as sealed:** Karolinska 9.8 % of healthy beyond p95 (bar 7 %). **Cause measured — the Sentrix chip:** per-chip median SD 0.020 there vs 0.012 elsewhere; chip-centring cuts every lab to 2–4 %. A laboratory constant cannot touch it; row 5b (chip term) is open and the acceptable false-alarm rate is the author's decision (PROC-MAHA-02). Record: `Record/PROC_data/PROC-MAHA-01/`.

**ROW 6 CLOSED — CELLULAR AGE NOT REPORTABLE AT SINGLE-ARRAY RESOLUTION (PROC-AGE-01, 2026-09-21).** Inverting the healthy immune identity-gauge curve (`reference_age_curve_v1`) for one array resolves age to ~50 years: the curve moves 0.47 mA/yr and the within-laboratory spread is 0.0235; leave-one-lab-out on 1,379 healthy donors, 15.9 % within ±10 yr (bar 80 %), Spearman 0.27; a healthy 58-year-old inverts to 23. **The population aging trajectory stands and is reproduced** (0.47 mA/yr, monotone by decade, four labs = CPG-VAL-015's slope on Hannum; it is now the reference age curve). What is below resolution is one person's position on it. Sign differs by surface: marker-union A falls with age, identity-gauge A rises (RECON D2). `stage_6_cellular_age` returns `reportable=False` with the resolution sentence, which the report prints in place of an age; the marker-union inversion is `diagnostic_cellular_age`. With this, **no reported number in the chain reads the marker-union statistic.** Record: `Record/PROC_data/PROC-AGE-01/`.

- **Row 4.6 — the patient's sky — COMMISSIONED (PROC-CMB-05, 2026-09-21, five seals; C2′ 4/4 on the restated bar [0.025, 0.08]).** `cpg_conductor.run_full` bundle key `patient_sky`: z = (β − Σ f_c μ_c − m_lab)/s_lab on the mapped β, class panels gated by measured presence floors (`Runtime Matrices/Patient_CMB/`), HEALPix NSIDE 128 genomic order. NOT AVAILABLE without the laboratory's residual scale (built from the same 40-array healthy panel as the lab zero). Calibration constant stated on every sky: healthy held-out tail 2.6–3.2 %, not 5 % (C2′ failed as sealed by ≤ 0.004; recorded). The retired formula read 61 % of a healthy genome as anomalous (C1) and is closed. Kit test `test_patient_sky.py`.

- **Row 7 — tiers — COMMISSIONED (PROC-TIER-01, 2026-09-21).** One tier function, `MethylPhys/chain/cpg_tiers.py`, reads `tier_breakpoints.json`; no tier word on a non-reportable gauge (§108 / UNMAPPED / lab_zero UNSET); A ≥ 1/H_min → AT_CEILING. Measured, not moved: under the July 1.01 onset 30 % of 1,379 healthy donors read ELEVATED on the identity gauge (1.07 admits 1; 1.10 none; healthy central 95 % = 0.954–1.041). PROC-TIER-02 set NORMAL to the healthy central 95 % → `tier_breakpoints.json` v1.4 [0.95, 1.04): 2.5 % of healthy read ELEVATED. Kit test `test_tiers.py`.

- **Row 8 — disease matching — REMOVED FROM THE CHAIN (author, 2026-09-21).** The signature matrix and cards come from the preliminary VAL record; the report shows cells detected, fractions, A per cell and class, placement and flags, and names no disease. The matrix is record-side (see `Disease Matrix/DISEASE_MATRIX/README_STATUS.md`). PROC-MATCH-01's fixes (fail-closed origin gate, firewall, surface = seal) stand. **Sealing rule:** we seal a built tool against a bar; building it is exploration with a working note, not a seal.

- **Row 4.5 — bidirectional detector — COMMISSIONED (PROC-BIDIR-01, 2026-09-21).** VAL-050/051 reproduce from the kit; engine == sealed formula (2e-16); 726 AIBL samples × 18 CpGs re-extracted from the raw GEO file match the sealed betas exactly. **Row 9 — the report — IN BUILD, unsealed:** `MethylPhys/chain/cpg_report_v3.py` renders the author's spec (cells, %, A per class with placement/tier, A per cell, departure + false-alarm rate, sky, flags; no condition named, no years; vocabulary guard); old `cpg_report_builder.py` is record-side.
