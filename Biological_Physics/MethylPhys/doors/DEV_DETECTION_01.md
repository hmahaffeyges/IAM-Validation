# DEV-DETECTION-01 - detection p-value: poobah against the Gaussian negative-control test (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Question (author decision E).** Which detection statistic removes probes at background better, measured the same way on every laboratory,
series, platform and instrument; the better one is used everywhere.

**The two statistics.**
- P (in use, Stage 1): methylprep poobah p per probe against this array's own out-of-band background; probe kept when p <= 0.05.
- G (Stage 0's statistic): p = 1 - Phi((M + U - mu_bg) / sigma_bg) per probe, mu_bg and sigma_bg the robust centre and spread of this array's
  negative controls (`stage_0_1_qc_handoff.decode_qc_inputs`, `stage_0_intake.compute_detection_p`); probe kept when p <= 0.01
  (`DETECTION_P_THRESHOLD`).
Both are computed on the same noob betas of the same array (Stage 1 run once without the mask).

**Arrays.** Every EPIC v1 and 450K array in the 56 series of DEV-BASE-CHAIN-01 whose IDAT pair can be read (EPIC v2 cannot be calibrated by Stage 1).
Strata: series (each series is one laboratory's submission), platform (EPIC v1, 450K), instrument (scanner named in the IDAT header run record:
iScan, NextSeq 550 or other text found; arrays with no scanner text are their own stratum).

**Quality (physics, per array and statistic).** The noise sites are 48,528 sites every purified blood group holds fixed (true state 0 or 1).
A probe read at background drifts toward beta 0.5 and raises the entropy there. Per array: N = mean H(beta) over the noise sites the statistic keeps
(the noise index); k = fraction of the 6,000 neutrophil identity sites kept (EPIC; on 450K the sites present on the array). Lower N with k not lost is
cleaner data.

**Decision rule (set now).** In each stratum with >= 5 arrays, statistic X is better when its median N is lower than the other's and its count of
arrays below 90 % identity-site coverage is not larger. The statistic better in more strata (ties counted for neither) is adopted for every array.
If both are better in the same number of strata, P stays. The strata where the other statistic is better are listed.
Read beside it, with no part in the rule: the within-person SD of tared A on the GSE250556 replicates and the number of other-laboratory purified
neutrophils in Normal, under each statistic.

**Wiring rule.** The chain's floor, profiles and noise gate were frozen under P. If G is adopted, the six floor arrays are re-read under G; when every
one of them moves by <= 0.001 in A, G becomes Stage 1's default; otherwise G is wired as `--detection gaussian` and the default waits for the author
(the frozen floor would need re-freezing under G).
