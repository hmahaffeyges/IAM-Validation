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


## Addendum 2026-10-20: the metric read three of its five pairs in the wrong channel

The construction above ("C / (C + U) in the green channel") is wrong for EPIC v1, and it is why every healthy array sat below 0.95. Of the five
BS Conversion I pairs, pairs 1 and 2 are green probes and pairs 3 to 5 red: on 61 healthy arrays (60 of GSE318173 and GSE235717 GSM7508429) the
C probes of pairs 3 to 5 read 203 to 287 in green and 14,220 to 16,139 in red. Read in green they sit at background on both probes and give
0.67 to 0.70, which pulls the median down. Read each in its own channel (the channel where the pair's C + U is brighter; it agrees with the
design on every pair of all 61 arrays), the healthy distribution is **median 0.970, range 0.927 to 0.978**; the metric as written gave median
0.738 (0.618 to 0.914). Exploratory, not sealed: `doors/data/PROC_STAGE0_04/bs_channel_explore.py`, GitHub runs 39535433627, 39536291479,
39537230609 (issue 32). The chain reads each pair in its own channel from 2026-10-20 (`stage_0_1_qc_handoff.py`; Chain Decoder finding F8).
The number above (median 0.7979 on 731 arrays) is the old metric's and is kept as recorded. The threshold is still the author's to set, by the
rule above, from the corrected distribution on more than one cohort; until then the gate prints and does not refuse.
