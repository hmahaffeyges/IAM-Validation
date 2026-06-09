# Session 2026-06-05 — L9 N7 chain-integrity test + canonicals + SOP

**Commit pushed:** `5a94ec7` on `hmahaffeyges/IAM-Validation` main
**Working tree:** clean, synced with origin

## Folder layout

```
N7_chain_integrity_session_2026_06_05/
├── cards_v3_1/                              # Both operational cards (updated)
│   ├── breast-epic_card_v3_1.json
│   └── ad-immune_card_v3_1.json
├── n7_orchestrators_and_results/            # Chain-level N7 test (once per chain version)
│   ├── n7_end_to_end_chain_recovery.py      # Main orchestrator (off-substrate + null)
│   ├── n7_panel_on_substrate.py             # Variant orchestrator (on-substrate injection)
│   ├── N7_OUTCOME.md                        # Full narrative + diagnostic findings
│   ├── n7_summary.json                      # v1 results (off-substrate + null)
│   └── n7_r3_within_cohort_results.json     # Within-cohort R3 across all 3 conditions
├── L9_null_runner_scripts/                  # Per-VAL nulls (always per VAL) + generator
│   ├── cpg_null_runner.py                   # N1, N2, N3, N4, N5, N6, N7, N8 — disease-specific
│   └── synthetic_patient_generator.py       # Used by N6 + N7 (chain-level test only)
└── canonicals/                              # The 3 canonicals + SOP
    ├── v7_CPG_IAMAtlas_Evidence_Report.html
    ├── v10_CPG_VAL_Inventory_Report.md
    ├── MASTER_TRACKER.md                    # Heath-only, never pushed
    └── CPG_Chain_of_Custody_SOP_v1_2.md     # Updated with N7 lessons
```

## When to run what

| Test | Cadence | Script |
|---|---|---|
| **N1** HC permutation | Every VAL (always) | `cpg_null_runner.py::run_N1` |
| **N2** age-strata permutation | Per VAL when ages exist in cohort | `cpg_null_runner.py::run_N2` |
| **N3** sex-strata permutation | Per VAL when sex covariate relevant | `cpg_null_runner.py::run_N3` |
| **N4** cohort-split replication | Per VAL when ≥2 cohorts | `cpg_null_runner.py::run_N4` |
| **N5** plate-position null | Per VAL when plate metadata present (rare from GEO) | `cpg_null_runner.py::run_N5` |
| **N6** injection-recovery | Per VAL when card has declared signal direction | `cpg_null_runner.py::run_N6` |
| **N7** end-to-end chain-recovery | **ONCE per chain version** (not per VAL) | `n7_end_to_end_chain_recovery.py` + `n7_panel_on_substrate.py` |
| **N8** look-elsewhere correction | Per VAL when scanning multiple features | `cpg_null_runner.py::run_N8` |

## N7 headline (this session)

- **R1 Walther class-fraction recovery: PASS at MAE = 0.0076–0.0093 across 8 classes × 3 conditions** (threshold 0.10)
- **R3 within-cohort Mahalanobis case-vs-HC:**
  - NULL = +3.07 (sampling-variance baseline from n_case=50 vs n_hc=200)
  - STRONG_OFF_SUBSTRATE = +5.10
  - STRONG_ON_SUBSTRATE = +10.24 (signal recovery 3.5× stronger when injection lands on cell-type marker substrate)
- Chain-modules wire end-to-end. Signal-on-substrate is the chain's natural detection condition.

## v0.2 carry-forward (next time N7 is run)

1. `synthetic_patient_generator.py` adds optional `restrict_panel_to_cpgs` parameter (default to cell-type marker substrate).
2. Within-cohort R3 uses matched arm sizes or k-fold cross-validation to remove the sampling-variance baseline.
3. `ChainRecoveryTester` extends from R1+R3 to the full R1–R8 recovery suite (per-class A-score, residual map, bimodality, PCA, chromosome isotropy, age dipole).

## Report-language correction (in same v6 → v7 bump)

Two unauthorized "What we are NOT claiming" paragraphs — added in the prior session's v5→v6 bump without your explicit approval, including one that incorrectly framed CPG as population-only with a multi-year-baseline requirement — REMOVED from v6 before bumping to v7. Per your 2026-06-05 directive: focus on what CPG does, not on disclaimers.

## SOP updates (so the next AI session has the context)

`CPG_Chain_of_Custody_SOP_v1_2.md` now contains:
- §87 (N7 description): new "2026-06-05 first-run experience" subsection with 5 lessons + recommended next-session protocol
- §89 (synthetic patient generator): new "v0.2 enhancements" block with 3 specific code changes
- §91 (invocation order): N7 cadence clarified — runs ONCE per chain version, NOT per VAL
