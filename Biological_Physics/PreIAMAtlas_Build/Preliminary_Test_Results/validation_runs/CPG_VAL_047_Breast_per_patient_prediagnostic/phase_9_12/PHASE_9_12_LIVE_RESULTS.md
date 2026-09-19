# Phase 9 + Phase 12 — LIVE RESULTS, 2026-04-23

Ran Phase 9 on GSE51057 (SHA-LOCKED to 828059...04d98bb0, bit-identical to your VAL047_prereg hash).
Ran Phase 12 on GSE51032 (3.15 GB).
Panel: Xu-538 (Kresovich-compatible), H_min(immune) = 0.838889, RNG seed 20260420.

Xu-6 coverage: 3/6 on both matrices (cg17301223, cg26203572, cg27091787 only; identical to VAL047_phase2 caveat).
Xu-538 coverage: 538/538 GSE51057, 538/538 GSE51032.

## Phase 9 — GSE51057 breast (n=146 C50 cases vs 177 controls)

Window          n    Xu-538 d         p  CI                       Xu-6 d         p
--------------------------------------------------------------------------------------------
0-2 yr         58      +0.088    0.5683  [-0.15,+0.43]           +0.137    0.3769
2-5 yr         34      +0.306    0.1066  [-0.08,+0.79]           +0.799    0.0001
5-10 yr        43      +0.712    0.0002  [+0.35,+1.18]           +0.537    0.0023
>10 yr         11      +1.783    0.0001  [+0.85,+2.99]           +1.083    0.0014
all_pre_dx    146      +0.452    0.0001  [+0.22,+0.64]           +0.477    0.0001

## Phase 12 — GSE51032 breast (n=224 C50 cases vs 424 controls)

Window          n    Xu-538 d         p  CI                       Xu-6 d         p
--------------------------------------------------------------------------------------------
0-2 yr         66      +0.268    0.0390  [+0.08,+0.57]           +0.275    0.0367
2-5 yr         50      +0.654    0.0001  [+0.35,+1.08]           +0.471    0.0020
5-10 yr        75      +0.921    0.0001  [+0.62,+1.26]           +0.442    0.0006
>10 yr         33      +1.363    0.0001  [+0.80,+1.91]           +0.366    0.0477
all_pre_dx    224      +0.707    0.0001  [+0.51,+0.87]           +0.352    0.0001

## Phase 12 — GSE51032 colorectal (n=76 C18/C19/C20 cases vs 424 controls)

Window          n    Xu-538 d         p  CI                       Xu-6 d         p
--------------------------------------------------------------------------------------------
0-2 yr          8      -0.473    0.1649  [-0.67,-0.21]           +0.326    0.3464
2-5 yr         27      -0.218    0.2691  [-0.44,+0.02]           +0.158    0.4247
5-10 yr        38      -0.319    0.0573  [-0.56,-0.15]           +0.164    0.3303
>10 yr          3  (insufficient n)
all_pre_dx     76      -0.326    0.0088  [-0.48,-0.17]           +0.177    0.1501

## Key pattern

Breast on Xu-538 immune: d GROWS monotonically with TtD window (+0.09 @ 0-2yr → +1.78 @ >10yr on GSE51057; +0.27 → +1.36 on GSE51032). Replication direction: same. Replication magnitude at >10yr: +1.78 vs +1.36 (close to HANDOFF README +1.85 / +1.34).

Colorectal on Xu-538 immune: d is NEGATIVE across all windows (-0.47 @ 0-2yr, -0.32 all-pre-dx, p=0.009). Inversion relative to breast is real on this panel with this H_min.

Xu-6 (3/6 coverage only) on CRC: SMALL POSITIVE (+0.18 all-pre-dx, p=0.15) — in the same direction as breast but not significant at n=76. The 3 available CpGs are breast-associated, so weak positive on CRC is consistent with 'borrowed panel' hypothesis.
