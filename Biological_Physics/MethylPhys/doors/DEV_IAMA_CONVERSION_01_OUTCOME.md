# DEV-IAMA-CONVERSION-01 — outcome (2026-10-10; development)

`doors/data/DEV_IAMA_CONVERSION_01/score_conversion_01.py`, rows `conversion_01_rows.csv` (Stage Q whole files, all 14 session 4 runs; c from
each run's lambda CHH record).

| donor, kit | c rep1 / rep2 | ε rep2 ÷ rep1 measured | predicted (c2/c1) | IAM-A rep2 ÷ rep1 raw | after ε/c |
|---|---|---|---|---|---|
| Sample2 TruSeq | 0.9902 / 0.9700 | 0.9591 | 0.9796 | 0.9690 | 0.9841 |
| Sample3 TruSeq | 0.9911 / 0.9738 | 0.9401 | 0.9826 | 0.9544 | 0.9672 |
| Sample4 TruSeq | 0.9908 / 0.9713 | 0.9508 | 0.9803 | 0.9627 | 0.9773 |
| Sample2 Swift | 0.9792 / 0.9794 | 0.9915 | 1.0002 | 0.9936 | 0.9934 |
| Sample3 Swift | 0.9795 / 0.9800 | 0.9848 | 1.0005 | 0.9886 | 0.9882 |
| Sample4 Swift | 0.9795 / 0.9799 | 0.9856 | 1.0004 | 0.9892 | 0.9889 |

**Read as written.** Dividing by c shrinks the repeat gap in 3 of 3 TruSeq donors (the direction predicted). It brings it within 0.02 in 1 of 3
(Sample2 0.016; Sample3 0.033, Sample4 0.023). Not met.

**What it shows.** The lower-conversion repeat libraries read 4–6 % lower ε; conversion accounts for about 2 %. The Swift repeats, with no
conversion difference, also read 1–1.5 % lower in 3 of 3, so these repeat libraries carry a library effect besides conversion. Whole-blood
repeat libraries here differ by 0.01–0.05 in IAM-A, wider than the neutrophil repeat spread (0.009). Two things are now on record:
conversion biases IAM-A in the direction and order the derivation gives, and the absolute 0.98 limit has no physical basis by itself.
The chain is unchanged: the correction is not adopted until a laboratory with a conversion spread and true repeats tests it.
