# PROC-STAGE0-02 — Stage 0 intake, run retrospectively over the Uppsala cohort

**Pre-registered 2026-09-23, before any verdict was computed.** Commissioned by the author after it was
established that no path in the tree calls [`stage_0_intake.py`](../chain/stage_0_intake.py): every run in the
commissioning record, including the 268 arrays behind the sealed chip result, reached Stage 1 without passing a
single intake gate.

## The question

**Would Stage 0 have quarantined any of the arrays the sealed result rests on?** The chip term
(PROC-MAHA-03 deep arm) was measured on 268 arrays across 23 complete Sentrix chips from GSE87571. Intake was
never run on them. This procedure runs it now and reports the answer against bars fixed here.

## What runs

`stage_0_intake.py` steps 0.1 → 0.9 on all 732 GSE87571 IDAT pairs in `idats_gse87571/`, with each manifest
entry built from the cohort's own GEO metadata (accession, Sentrix barcode and position from the file name,
declared age and sex from the series matrix). The steps that execute on file evidence alone:

| step | SOP | what it decides |
|---|---|---|
| 0.1 | §11 | both channels present, readable header, array type from the probe count, truncation check |
| 0.2 | §12 | the manifest is complete; a missing required field is QUARANTINE_MANIFEST_INVALID |
| 0.3 | §13 | SHA-256 of both files, and re-transmission detection against earlier intakes of the same Sentrix ID |
| 0.6 | §16 | bead count ≥ 3 per probe (warn-only per the SOP) |
| 0.7b | | ≥ 80 % coverage of the reference CpGs, and the platform tag |
| 0.9 | §19 | PROCEED / PROCEED_WITH_PENALTY / QUARANTINE |

Steps 0.4 (control probes), 0.5 (detection p), 0.7 (call rate) and 0.8 (sex check) need Stage 1's decoded
intensities. That hand-off is the open action on register row 0. **They will report DEFERRED and a DEFERRED is
not counted as a pass** — the outcome will state how many gates actually ran.

## Bars, fixed before the run

1. **B1 — every pair is readable.** Any pair whose header cannot be parsed, or whose two channels disagree on
   the array type, is a QUARANTINE and is named individually in the outcome.
2. **B2 — platform.** All 732 must verify as 450K from the IDAT header. Any other verdict is reported, not
   corrected.
3. **B3 — the sealed set.** The 268 arrays are reconstructed by the rule the analysis used: every chip with
   ≥ 9 calibrated arrays in `results/percell/stage1_betamean_GSE87571.json`. The count of those 268 whose
   verdict is QUARANTINE is the headline of this procedure.
4. **B4 — the decision rule, fixed now.**
   - **0 quarantined** → the sealed chip result stands, and it gains intake evidence it did not have.
   - **1 to 13 (≤ 5 %)** → the deep arm is recomputed with the quarantined arrays removed, and the new ICC, F,
     permutation p and tail are reported beside the sealed ones **whether or not the verdict changes**.
   - **more than 13** → the deep arm is re-run on the surviving arrays and PROC-MAHA-03 is re-sealed.
5. **B5 — no threshold moves after the fact.** Every gate keeps the threshold already in `stage_0_intake.py`
   and the SOP. If a threshold looks wrong once verdicts are visible, that is a finding for a separate
   procedure, not an edit to this one.
6. **B6 — quarantine is reported per cause.** The outcome names which step produced each quarantine, so a
   reader can see whether the cause is the specimen, the file, or the metadata.

## What this procedure does not do

It does not wire intake into the chain. Whether `run_sample.py` should refuse a sample that fails a gate is a
change to what the instrument refuses, and the author commissions that separately.
