# PROC-STAGE0-04 — the bisulfite-conversion threshold, to be set by the author

**Pre-registered 2026-09-23.** PROC-STAGE0-02 measured bisulfite conversion on 731 healthy whole-blood arrays
and every one sits below the SOP's `BS_CONVERSION_MIN = 0.95`: median 0.7979, 5th percentile 0.7400, worst
0.6354. A gate that refuses every healthy specimen is not measuring the specimen.

## What is being decided

The number — and only the number. The metric is fixed and stated: for each matched BS Conversion I control pair
(C_i with U_i) the efficiency is C / (C + U) in the green channel, and the array's value is the median over the
six pairs. That construction is in
[`stage_0_1_qc_handoff.py`](../chain/stage_0_1_qc_handoff.py) and does not change here.

## The healthy distribution, published before the threshold is chosen

| quantity | value |
|---|---|
| arrays | 731 (GSE87571, healthy whole blood, four Sentrix-chip years) |
| median | 0.7979 |
| 5th / 95th percentile | 0.7400 / 0.8319 |
| minimum | 0.6354 |

## The rule this procedure follows

1. The threshold is set from **healthy** arrays only, and from more than one cohort before it is called
   commissioned. This cohort is the first; GSE42861, GSE111629 and GSE125105 are on disk and follow.
2. It is stated as a percentile of the healthy distribution, not as a round number: a gate should refuse the
   tail of what healthy tissue looks like, not a figure someone liked.
3. Whatever is chosen, `BS_THRESHOLD_CALIBRATED` is set True in the same commit, the SOP's §14 line is updated
   with the number and its provenance, and every prior reading keeps its `PROVISIONAL` status in the record.
4. Until then the gate reports and does not refuse, and no sample is passed that a calibrated gate would fail.

**The author decides the number.** An analyst may recommend; commissioning a gate that refuses a patient's
specimen is his call, and this document exists so the choice is made on the distribution rather than on a
default nobody measured.
