cd remote_jobs/predx && cat > PROC_PREDX_SLIDE_01_OUTCOME.md <<'EOF'
# PROC-PREDX-SLIDE-01 — outcome (2026-10-01). Pre-registration sha 434b2ed9da21be30.

Test set: 320 of the 329 GSE51057 arrays with ≥ 3 same-slide controls (170 controls, 144 breast cases; all women). Met-A, neutrophils, A_rel to
same-slide controls.

| prediction | result | verdict |
|---|---|---|
| S1 controls ≥ 80 % Normal | 158/170 = **92.9 %** | **PASS** |
| S2 breast > 8 y outside Normal more than controls | 2/23 vs 12/170, p = 0.52 | **FAIL** |
| S3 all breast cases | 20/144 vs 12/170, **p = 0.036** (median A_rel 1.011 vs 1.000) | **PASS** |

By lead time (breast, outside Normal): < 2 y 10.5 %, 2–5 y 17.6 %, 5–8 y 20.0 %, > 8 y 8.7 % (controls 7.1 %).

**Reading.**
1. The slide correction is what made Met-A usable on these arrays: healthy women read Normal 88–93 % in both sets, against 58 % before. The
   sex gap was a slide (batch) effect, not biology. Normalising to same-slide controls is an instrument tare, like PROC-TARE-01; it is a
   per-run reference, not a population model, but a reviewer will ask that it be stated plainly.
2. Breast cases read slightly above controls overall in both sets (median A_rel +0.011 to +0.014). The > 8-year signal of the first set did not
   replicate; the effect in the second set sits at 2–8 years. A small, consistent shift in the cases as a group is shown; a per-woman early
   warning is not (most cases read Normal).
3. These 329 arrays were used in earlier VALs, so this is a second look, not a naive replication.
EOF
echo ok