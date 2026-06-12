# RETIRED OFF-SCOPE VAL EXPLORATION

**Date retired:** 2026-06-06
**Reason for retirement:** Numbering scheme error + scope mismatch with the locked v1.0 immune card validation plan.

## What happened

Walther lost the pre-compaction session context and reverted to using the **pre-build sequence** numbering (`VAL-135` through `VAL-141`) when the **locked v1.0 immune card scope** uses the **CPG-VAL sequence** (`CPG-VAL-015` through `CPG-VAL-021`). The 3-digit `VAL-NNN` sequence belongs to the pre-build inventory (last sealed VAL-128, kidney-epic reserved 129–134). The post-build clinical engine VALs use `CPG-VAL-NNN`, currently sealed through CPG-VAL-014 (breast 001–007, AD 008–014).

Beyond the numbering error, the exploration also went off-scope. The pre-compaction lock was:
- CPG-VAL-015 — Aging trajectory immune cellular age on Hannum
- CPG-VAL-016 — Cross-disease universal alarm
- CPG-VAL-017 — Inflammaging quantum in pooled HC
- CPG-VAL-018 — HRT effect on female immune signal
- CPG-VAL-019 — Bidirectional immune direction discrimination
- CPG-VAL-020 — Hannum aging anchor reproduction (full chain)
- CPG-VAL-021 — Weight-loss inflammaging via GSE61450 bariatric pre/post

Walther instead ran exploratory tests on the GSE40279/GSE50660/AIBL cohorts using ad-hoc chain components (skipped Stage 2 Walther deconvolution; used top-200 class markers ad-hoc instead of canonical 115-cell markers; used cohort-internal Mahalanobis instead of the n=601 HC reference; substituted linear regression for the canonical Stage 6 cellular age inversion). The framing "fit-cohort-internal only" + the "v1.1 options" of statistical band-aids was statistics talking, not the physics-first framework.

## What's preserved in these retired directories

- `per_sample_*.csv` files for AIBL (726 samples), GSE50660 (464 samples), GSE40279 (656 samples) — REAL DATA that can feed into the locked-scope CPG-VAL-015 through CPG-VAL-021 work
- The runner scripts `val_135_run.py` + `val_136_141_runner.py` — code patterns useful to mine when building the proper runners, but with the caveats above (must replace ad-hoc class markers with canonical 115-cell markers; must add Walther + NILC deconvolution; must use the n=601 HC Mahalanobis reference; must use IAMCellularAge for proper Stage 6)

## Honest lessons (carried into MASTER_TRACKER §0.7)

1. After compaction, re-anchor to the locked scope before executing. Don't trust memory alone.
2. `VAL-NNN` vs `CPG-VAL-NNN` are TWO sequences. Pre-build uses the former, post-build clinical engine uses the latter.
3. "Run the full chain — EVERYTHING" means USE ALL CANONICAL MODULES (Walther deconvolver, NILC, 115-cell A-scoring against canonical markers, n=601 HC Mahalanobis, IAMCellularAge inversion against 80-cell baseline), not write a "minimum viable" replacement that bypasses them.
4. Empirical foreground subtraction is a SUPPORT tool. The architectural A-score H(β_mean)/H_min IS the physics. When the empirical layer doesn't transfer, run on raw β and let the architecture speak — don't pivot to "better regression."

## Where to go from here

The proper CPG-VAL-015 through CPG-VAL-021 buildout lives in `Biological_Physics/validation_runs/CPG_VAL_015_*` through `CPG_VAL_021_*` (directories to be created during the proper build) following the same 5-file pattern as CPG-VAL-001 through CPG-VAL-014.
