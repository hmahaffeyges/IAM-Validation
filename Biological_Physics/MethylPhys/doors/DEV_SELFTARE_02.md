# DEV-SELFTARE-02 - self-tare on type II fixed sites (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Why.** DEV-SELFTARE-01: the fixed sites of `noise_sites_EPIC_v1.json` are 99.7 % type I, the identity sites 97.9 % type II, and
the type II offsets rested on 148 probes. Author decision G: build the self-tare on type II fixed sites.

**Derivation.**
1. \derived Two-channel probe on one array: beta = b (1 - dU - dM) + dU, with dU, dM this array's offsets for the probe design (DEV-SELFTARE-01 step 1).
   beta is affine in the true state b, with slope (1 - dU - dM) and intercept dU.
2. \derived Two fixed-site sets of one design, one held low (true state s0) and one held high (true state s1) in every blood cell, give this array's
   anchors L = mean beta over the low set and U = mean beta over the high set. Two points fix an affine map. The map that puts this array on the
   reference arrays' scale is beta' = Lr + (beta - L) (Ur - Lr) / (U - L), with Lr, Ur the same anchors averaged over the six frozen reference arrays.
   It is exact under step 1 for any s0, s1, provided the fixed sites hold the same true state on every array. Nothing is fitted.
3. \conjecture The fixed sites hold the same true state on every array of healthy blood (the DEV-SELFTARE-01 step 2 assumption, weakened: the state
   need not be exactly 0 or 1).
4. Each design gets its own map (type I identity sites from type I anchors, type II from type II). Met-A is then formed from beta' as the chain forms it.
   The six reference arrays map close to themselves, so the frozen floor is unchanged.

**Type II fixed sites (rule set now).** EPIC type II probes (methylprep EPIC manifest), not a neutrophil identity site and not a composition marker;
in each purified group of GSE110554 (our Stage 1 betas) the group mean is <= 0.15 for every group (low set) or >= 0.85 for every group (high set),
group SD <= 0.02, and the largest difference between group means <= 0.03. The type I sets are the noise sites of the same state (DEV-NOISE-01).
If fewer than 1,000 type II sites pass, that is recorded and the run goes ahead.

**Readings.** (i) no tare; (ii) median tare; (iii) self-tare II; (iv) self-tare II then median tare. Sets: GSE250556 (63 replicate arrays, whole
blood, same-slide median tare as Stage T); other-laboratory purified healthy neutrophils GSE247193, GSE247195, GSE122244 (same-series median tare);
the six floor arrays (GSE110554).

**Bars (carried unchanged).** Replicates: within-person SD <= 0.020 and >= 95 % of readings in Normal (DEV_REPL_V3_01_PLAN). Other-laboratory purified
neutrophils: every array that reaches Stage 5 in Normal (DEV-BASE-CHAIN-01 b). Floor arrays: every one in Normal.

**Physical control DNA (decision G, second route).** Fully methylated and fully unmethylated control DNA on every slide (and a 50 % mix) measures
L and U on the slide itself, and the 50 % mix measures the channel-gain term the fixed sites cannot see (zero at b = 0 and b = 1). No public EPIC data
carry such controls; this route needs wet-lab runs and is written into the SOP as the next step if (iv) does not meet the bars.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 9bcbcf99 (the run of record), chain = this round's development code; Stage 1 betas from DEV-BASE-CHAIN-01. Runtime file built by the rule above:
`chain/Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json` (development, not frozen). Records: `data/DEV_SELFTARE_02/`.

**Type II fixed sites found** (\measured): 50,359 low and 166,379 high (the rule asked for >= 1,000), from 35 purified
GSE110554 arrays in six groups (B, CD4T, CD8T, MONO, NEU, NK). Reference anchors (six floor arrays): type I 0.0186 / 0.9820,
type II 0.0555 / 0.9486.

| reading | replicates median | within-person SD (bar <= 0.020) | replicates in Normal (bar >= 95 %) | other-lab neutrophils in Normal (bar: all) | r with N |
|---|---|---|---|---|---|
| no tare | 1.205 | 0.0356 | 0/63 | 3/49 (0.862-1.286) | 0.84 |
| median tare | 1.002 | 0.0369 | 48/63 | 43/49 (0.806-1.082) | 0.83 |
| self-tare II | 1.007 | 0.0069 | 60/63 | 33/49 (0.929-1.047) | 0.27 |
| self-tare II then median tare | 1.000 | 0.0164 | 62/63 | 49/49 (0.984-1.030) | 0.30 |

- \measured (iv) self-tare II then median tare: within-person SD **0.0164** (A 0.0119, B 0.0113, C 0.0232, D 0.0169); **62 of 63** in Normal
  (the one outside: GSM7981554, 1.0504); other-laboratory purified neutrophils **49 of 49** in Normal (0.984-1.030; GSE122244 4/4, GSE247193 21/21, GSE247195 24/24);
  floor arrays 6 of 6 in Normal (self-tared 0.996, 1.024, 1.004, 1.019, 0.992, 0.965; read against the other five 0.995, 1.029, 1.005, 1.024, 0.989, 0.957).
  Every bar of this note is met by reading (iv).
- \measured (iii) self-tare II alone: within-person SD 0.0069, 60/63 in Normal, but other-laboratory neutrophils 33/49
  (GSE247193 5/21, median 0.944): the self-tare alone does not carry another laboratory onto the reference scale for that series.
- \observed The link to the noise index N falls from r 0.83 (median tare) to 0.30 (iv).
- \observed Type II slope (Ur - Lr)/(U - L) on the replicates 1.036-1.082 (median 1.060): this laboratory's type II range is about 6 % narrower than the
  reference arrays'. GSE247193 1.054, GSE247195 1.011, GSE122244 1.010.
- \conjecture The remaining replicate spread (0.016) is the median tare's own reference noise: (iii) alone gives 0.007 on the same arrays.

**First run, recorded as run** (box job 5bd710b8): the title parser left the T cells out of the fixed-site rule (four of six groups), against the rule
above; the run of record re-ran with all six groups. Run 1 numbers: (iv) SD 0.0146, 63/63; other-lab 49/49; (iii) SD
0.0064. Run-1 records are kept beside the run of record (`run1_T_cells_left_out_*`).

**Wiring.** Behind `--dev-selftare-ii` (DEV-FLAGS-01). The median tare stays the reading. Next (author decision): make (iv) the Stage T reading
(it changes the frozen tare, so it needs the author); the physical control DNA route stays in the SOP as the check on the fixed-site assumption.
