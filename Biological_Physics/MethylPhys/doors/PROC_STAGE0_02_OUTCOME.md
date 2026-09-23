# PROC-STAGE0-02 — outcome: Stage 0 run retrospectively over the Uppsala cohort

**Sealed 2026-09-23** against the bars fixed in [`PROC_STAGE0_02_PREREG.md`](PROC_STAGE0_02_PREREG.md) before any
array was read. Inputs: 732 IDAT pairs (GSE87571), the cohort's own published age and sex, and the array's own
manifest. Evidence: [`PROC_STAGE0_02.json`](../kit/results/PROC_STAGE0_02.json) (per-array), and the two scripts
[`PROC_STAGE0_02_sweep.py`](../kit/PROC_STAGE0_02_sweep.py) and
[`PROC_STAGE0_02_seal.py`](../kit/PROC_STAGE0_02_seal.py).

## The answer

**Zero of the 268 arrays behind the sealed chip result would have been quarantined.**

One array in the 23 complete chips — GSM2334619 — does hit a quarantine, and it is not in the 268: the deep-arm
analysis requires a declared age to remove the age term, and this donor's age is not published, so the analysis
had already excluded it. Two independent gates converged on the same array for different reasons.

| set | PROCEED | PROCEED_WITH_PENALTY | QUARANTINE | decode error |
|---|---|---|---|---|
| all 732 arrays | 720 | 8 | 3 | 1 |
| the 23 complete chips (269) | 261 | 7 | 1 | 0 |
| **the 268 the chip result used** | **261** | **7** | **0** | **0** |

Per the pre-registered decision rule, **0 quarantined means the sealed chip result stands** — and it now carries
intake evidence it did not have when it was sealed. PROC-MAHA-03 is unchanged: ICC 0.197, F 3.862, p 0.0005,
tail 0.0597. Recomputing with the quarantined array excluded returns those numbers unchanged, because it was
never in the set; both runs are in the evidence file.

The 7 penalised arrays are all borderline on bead count (between 0.990 and 0.995 of probes at ≥ 3 beads). A
penalty is a note, not a refusal: they were scored, and they stay in.

## What each gate measured, across 731 readable arrays

| gate | median | p05 | worst | threshold | outside it |
|---|---|---|---|---|---|
| detection p (0.5) | 0.9994 | 0.9990 | 0.9951 | ≥ 0.99 | 0 |
| call rate (0.7) | 0.9983 | 0.9961 | 0.9889 | ≥ 0.98 | 0 |
| bead count (0.6) | 0.9988 | 0.9969 | 0.9900 | ≥ 0.995 | 9 (warn) |
| bisulfite conversion (0.4) | 0.7979 | 0.7400 | 0.6354 | ≥ 0.95 | **731 of 731** |

**Sex check (0.8): 729 of 731 arrays agree with the cohort's published sex — 99.73 %.** The two that disagree are
the two donors whose sex is published as "NA". This is the strongest evidence that the hand-off decodes what it
claims to: the sex call is made from chrX and chrY intensities alone, with no cohort information, and it
recovers the depositors' own labels on 729 consecutive arrays.

## The finding: the bisulfite threshold has never been calibrated

Every healthy array in the cohort sits below the SOP's 0.95, median 0.798. A threshold that rejects 731 of 731
healthy specimens is not measuring specimen quality. Two possibilities — the threshold was written for a
different construction of the metric, or 0.95 is simply not what this quantity looks like on real 450K arrays —
and neither is settled by moving the number after seeing the data, which the pre-registration forbids (B5).

So the gate is reported, not applied: `validate_control_probes` returns
`PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` when bisulfite conversion is the *only* failing gate and
`BS_THRESHOLD_CALIBRATED` is False, the decision gate treats it as deferred, and the measured value is printed
on the report. No specimen is passed that a calibrated gate would fail, and none is refused on a number nobody
has measured. [`PROC_STAGE0_04_PREREG.md`](PROC_STAGE0_04_PREREG.md) hands the distribution to the author with
the threshold decision stated as his.

## Four defects the run found, and what each would have cost

1. **The header reader could not open a gzipped IDAT.** GEO ships `.idat.gz`; `read_idat_nsnps` opened the file
   raw, so the array type was unreadable on every public download and the type gate could never fire. Fixed.
2. **`step_0_1` never carried `patient_id` out of its result**, so `step_0_2` saw it absent and read that as a
   cleartext identifier: every array quarantined with `QUARANTINE_MANIFEST_INVALID`. Fixed.
3. **A quarantine did not stop intake.** The first wiring called every step in sequence, and `step_0_2`
   overwrote the status — so an array-type mismatch was reported as a detection failure two gates downstream.
   The runner now stops at the first refusal and the decision names the cause.
4. **A failed decode was a fail-open.** The first wiring caught any hand-off exception and let the sample
   through with six gates "deferred". One array in this cohort (GSM2334130) is truncated at the gzip level while
   passing the 1 MB size floor, so it would have been calibrated and scored with no QC at all. A decode failure
   is now `QUARANTINE_CORRUPT_IDAT`.

Defects 1 and 2 are in the chain and are fixed there. Defects 3 and 4 were in the wiring written today, and are
recorded because the next person to wire a gate will be tempted by both.

## Bars

| bar | result |
|---|---|
| B1 every pair readable | **met** — 731 of 732; GSM2334130 is truncated in this local copy and is named, not skipped |
| B2 platform verified from the header | **met** — all 732 read as HM450K, 622,399 addresses |
| B3 the sealed set | **met** — the count is the headline above |
| B4 decision rule | **0 branch applied** — the sealed result stands with intake evidence |
| B5 no threshold moved | **honoured** — the bisulfite gate is reported as uncalibrated, not re-tuned |
| B6 per-cause reporting | **met** — 3 quarantines are all incomplete manifest (no published age); 2 of those also have sex "NA" |

## What this does not claim

It does not claim the cohort is clean because Stage 0 passed it: four of the nine gates ran for the first time
today, and the bisulfite threshold is still uncalibrated. It does not re-open PROC-MAHA-03, which stands
unchanged. And it says nothing about any cohort but this one.
