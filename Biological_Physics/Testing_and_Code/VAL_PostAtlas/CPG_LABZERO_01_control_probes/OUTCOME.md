# OUTCOME — LAB-ZERO-01: predicting the per-lab offset from the array's control probes

**Run 2026-09-20.** 1,277 arrays (GSE87571 732, GSE42861 controls 335, GSE125105 controls 210), 850 Illumina control probes read from raw IDATs (`ctrl_worker.py`, 6 subprocess workers, 21 min), 33 features = log2 mean per Control_Type × channel + overall medians + G/R ratio. Target: mapped immune identity-loci A (1,173 gated samples). Ridge, **leave-one-cohort-out**, as sealed.

## Verdict: **P1 FAIL on one of three folds (Munich). Direction correct 3/3; magnitude within 0.002 for both Swedish labs.** Panel standard stands; control-probe route promoted to "promising — needs a fourth cohort".

| held out | observed offset (cohort medians) | predicted | \|err\| | P1 (≤ 0.010) | within-cohort r(pred, A) | sd ratio (P3 ≥ 0.8) |
|---|---|---|---|---|---|---|
| GSE42861 Karolinska | +0.0238 | +0.0214 | 0.0024 | ok | 0.53 | 0.85 |
| GSE87571 Uppsala | −0.0238 | −0.0260 | 0.0021 | ok | 0.65 | 0.79 |
| GSE125105 Munich | −0.0214 | −0.0714 | 0.0500 | **FAIL** | 0.62 | 1.10 |

(λ = 1; λ = 10 and 100 are worse on every fold. Offsets here are whole-cohort medians and differ slightly from the per-decade values in band_v2 OUTCOME: +0.018 / −0.030.)

## Reading
1. **The lab signature is in the control probes.** Features separating the cohorts (range/sd ≈ 2.6): NORM_T/A/C/G and BISULFITE CONVERSION II and SPECIFICITY I, red channel — the normalisation and conversion controls, exactly where a lab's chemistry would show. With a Swedish neighbour in training the held-out Swedish lab is predicted to 0.002. Munich, with no neighbour, gets the right sign and 3× the size: two training labs cannot span the feature space. **A fourth Stage-1 healthy cohort is the decisive test**, and per-sample fetching makes it minutes.
2. **Within a cohort, control probes explain r ≈ 0.6 of a healthy donor's mapped A.** About a third of the "healthy spread" is per-array technical variance the array records about itself. Correcting for it would NARROW the band — a larger prize than the offset, because it sharpens every real departure. This becomes its own procedure (LAB-ZERO-02: within-cohort correction, cross-validated, band width before/after).
3. **N-perm as sealed was uninformative** and is recorded as such: shuffling cohort labels removes the offsets, so the null error is trivially small (median 0.0016). A correct null for this question is a fourth held-out cohort, not a permutation.
4. Standard today: **per-lab healthy-control panel** (CLSI EP28), offset printed on every report. The control-probe zero is the upgrade path, not the current mechanism.

---
**SEALED** sha256 `60f6de88b474e7c8ca027ce647b34d8b46896b80964e78dd7a92aaaa9f933980` · 2026-09-20
