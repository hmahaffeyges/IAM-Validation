# PROC-INTAKE-01 — pre-registration: the intake gate runs on the array's own numbers, and a deferred check never advances

**Written 2026-09-27, before any array is scored under these bars.** Follows FINDING_GSE125105_LOW_SIGNAL.md.

## The change (fixed)
1. Stage 1 ([`stage_1_idat_calibration.py`](../chain/stage_1_idat_calibration.py)) runs the decoder with per-probe detection (poobah, against this array's own negative
   controls) and the control probes kept, and returns: the beta vector, the detection-pass mask (p ≤ 0.05), the per-class
   control medians, and the fraction of probes at background.
2. `run_full` hands these to Stage 0.4 / 0.5 / 0.7. The gates are the SOP's, unchanged: detected fraction > 0.99 PASS,
   0.95–0.99 BORDERLINE, < 0.95 FAIL; call rate ≥ 0.98 PROCEED, 0.95–0.98 PROCEED_WITH_PENALTY, < 0.95 QUARANTINE.
3. Probes failing detection are **removed from the beta vector before any stage reads it** (no measurement at background).
4. `DEFERRED_PENDING_STAGE1_DECODER` on an IDAT input → `advance = False` (QUARANTINE_INTAKE_DEFERRED). On a betas-only input
   (no IDAT) intake cannot run; the bundle carries `intake_verified: False` and the Reading tab prints one line saying so.
   `--no-intake` is removed from the commissioning path.

## Bars
- **B1** each of the 12 GSE125105 panel arrays from PROC-SKY-01: call rate reported; those below 0.95 are QUARANTINED and
  produce no report (exit 2), those in 0.95–0.98 carry the PENALTY flag on the Reading tab.
- **B2** the gate is not refusing good arrays: of the first 100 GSE87571 IDAT pairs (sorted accession order) at least 95
  PROCEED. (An instrument check that the SOP threshold sits where the array's own noise says it should — not a definition of
  anything about people. If it fails, the threshold is reported against the observed call-rate distribution and NOT moved here.)
- **B3** masking does not move A on good arrays: on 12 GSE87571 arrays, per present cell, |A_masked − A_unmasked| < 0.002.
- **B4** a betas-only input renders with `intake_verified: False` and the one printed line; nothing else on the report changes.
- **B5** the negative control: a Stage 1 return with the mask deliberately withheld must QUARANTINE (deferred never advances).

## Decision rule
B1–B5 met → adopted; the Troubleshooting tab's "three ways of not giving you an answer" gains the intake line. B2 failed →
adopted anyway for B1/B3/B4/B5 but the call-rate thresholds are flagged UNCALIBRATED on the report (printed, not refused on),
exactly as the bisulfite row is today, and the distribution is recorded for the author's decision.
