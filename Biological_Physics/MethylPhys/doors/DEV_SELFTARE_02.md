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
