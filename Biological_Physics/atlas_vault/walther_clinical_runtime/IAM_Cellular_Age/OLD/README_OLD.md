# IAM_Cellular_Age/OLD — superseded artifacts

## age_clock_diagnostics_REJECTED_v1_horvath_style.json

Diagnostics from the **rejected v1 Horvath-style elastic-net regression clock** that the canonical Recipe §6.3 inversion (`iam_cellular_age_scoring.py`) supersedes. Kept for audit trail per SUPERSEDED.md. Was MAE 5.48 yr (target <5: FAIL by 0.48 yr) — not the reason for rejection; the reason was methodological (training-set-based clock, not physics inversion).

**NOT used at patient runtime.** Engine consumes only `iam_cellular_age_scoring.py` (Stage 6) for cellular age.
