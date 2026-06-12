# VAL-102 — VOIDED BEFORE EXECUTION

**Date voided:** 2026-04-28T20:35Z (within minutes of original seal at 2026-04-28T20:31:23Z)
**Voided by:** Claude (Walther) at Heath's direction "DO WHAT IS THE RIGHT THING TO DO"

## Original sealed state

- prereg.md SHA-256: `2b77ad9d3b69554a0658260756db0f08722e2be3fa96eb48aad9213974f4717c`
- Sealed at: 2026-04-28T20:31:23.247899+00:00
- RNG seed: 20260428
- Threshold change vs VAL-101: CHK-3.1 extreme >30% → >20% for TCGA HM450 sesame Level 3 platform

## Why VAL-102 was voided before execution

VAL-102 was sealed with a CHK-3.1 platform threshold (extreme >20%) that was derived from the very data VAL-102 was scheduled to interpret. The post-hoc verification on cached TCGA-COAD HM450 sesame Level 3 (the VAL-099 cohort) was run AFTER VAL-101 tripped at 26.6%. The threshold of >20% was chosen specifically because it sits below the observed 24-27% range on TCGA HM450 sesame Level 3 betas (VAL-099 + VAL-101) and above the noob-bg-corrected 3.9% from VAL-100 (which CCL-040 correctly defers).

This is not pre-registration. This is post-hoc threshold accommodation with a SHA stamp. The threshold was selected to accommodate the data VAL-102 was about to interpret. Sealing it with a hash does not make it methodologically pre-registered. The data was already known when the threshold was set.

The proper recovery path from a misspecified pre-locked threshold is NOT to immediately set a new threshold using the data that triggered the misspecification. The proper recovery path is:

1. **Set the platform threshold using a calibration cohort that is structurally separated from the test cohort.** For TCGA HM450 sesame Level 3, pick a small set of TCGA samples from a tissue NOT currently under test (e.g., TCGA-KIRC kidney or TCGA-PRAD prostate adjacent-normal samples, both of which have not been used in any active hcc-epic VAL). Measure the bimodality distribution on the calibration cohort. Set the threshold from THAT distribution. Seal it. Then apply it to TCGA-LIHC for a future VAL-XYZ.

2. **OR: Use the CCL-040 deferral pathway** that the cookbook already mandates for data-integrity issues. Defer to v0.2+ raw IDAT processing. Process the TCGA-LIHC IDATs through sesame from .idat input, verify bimodality at the standard pipeline output, and re-run the VAL with the reprocessed betas as input.

Both paths take longer than re-sealing VAL-102 against pre-known data. Both paths are honest. The first path is the proper extension of the cookbook to a new platform calibration. The second path follows the CCL-040 precedent already established for VAL-100.

## What stays in the cookbook record

- **VAL-101 stays at `O5_DATA_INTEGRITY_FLAG`.** Outcome unchanged. Pre-locked threshold tripped; cookbook discipline honors the trip. Biological readouts in VAL-101 results.json remain descriptive supplementary documentation only and do NOT propagate.
- **CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION** is logged as the lesson — the cookbook DOES need platform-specific CHK-3.1 thresholds going forward. The lesson stands. What is voided is the immediate-re-run-on-the-same-data attempt to rescue VAL-101's biology.
- **The biological-propagation pathway for VAL-101 readouts is OPEN.** Not closed forever. The path is: do a proper calibration VAL on a structurally-separate cohort, set a properly pre-registered platform threshold, then re-run the TCGA-LIHC test cohort under that pre-registered threshold. That's a multi-VAL workstream, not a same-day re-seal.

## Seal record (audit trail preserved)

The original sealed VAL-102 prereg.md (SHA `2b77ad9d3b69554a0658260756db0f08722e2be3fa96eb48aad9213974f4717c`) is preserved in this directory under `prereg_VOIDED.md`. The seal record is preserved in `PREREG_SEAL_VOIDED.txt`. No biological execution was performed against this sealed prereg. The void event is logged here.

The cookbook does not delete sealed records. It marks them and explains.
