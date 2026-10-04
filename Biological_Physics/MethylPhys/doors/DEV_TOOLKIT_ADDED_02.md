# DEV-TOOLKIT-ADDED-02 - trace cell (3b), foreign cell (3c), surface brightness (11b) on each array's own noise; IAM-A versions (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Author decision K.** Each line is set on the array's own noise; no line comes from other arrays; beta scale = Stage 1 noob beta (no map);
a v3 per-site uncertainty.

**v3 per-site uncertainty.** On one array, for each probe design (I, II) and state (low, high): the SD of beta over the array's own fixed sites of
that design and state (noise sites for type I; the DEV-SELFTARE-02 type II sets), each site measured against its frozen reference value.
s_i = the value for site i's design, the low-state value where the reference value of site i is < 0.5 and the high-state value otherwise.

**3b trace cell (contamination of an isolated-neutrophil specimen).** At the 963 composition markers, residual r = beta - mu_NEU; for each other
blood group c, the weighted regression of r on d_c = mu_c - mu_NEU (weights 1/s_i^2) gives f_c and its standard error from this array's own fit
residual scatter. Called when f_c > 3 SE. Constructed test: other-laboratory purified neutrophil arrays (GSE247195) with the betas of another
laboratory's purified cell (GSE122244 monocyte, B and T arrays) added at f = 0, 1, 2, 5, 10 % (beta mixed linearly). Bars: at 5 % the spiked cell is
called on >= 95 % of constructed specimens; of the calls at 5 %, >= 95 % name the spiked cell. The f = 0 call rate is recorded (the true contamination
of the unspiked arrays is not known).

**3c foreign cell (non-blood material in whole blood).** The bucket holds one placenta series (GSE271697, 93 arrays). Template: the mean of the first
half of its arrays (GSM order); spike material: the other half (never in the template). Sites: the 963 markers plus up to 500 sites where the template differs from every blood group by >= 0.25. Fit:
non-negative least squares on [8 blood groups, template], weights 1/s_i^2; the template's standard error from this array's own fit residual scatter.
Called when f > 3 SE. Specimens: GSE250556 replicates (no foreign material) with placenta added at f = 0, 1, 2, 5, 10 %. Bars: at f = 0 <= 5 % called;
at 5 % >= 95 % called.

**11b surface brightness (interval on Met-A).** 200 draws beta_i + N(0, s_i^2), clipped to (0, 1), A recomputed each time; 95 % interval.
Bar: on GSE250556 pooled-replicate pairs of the same person, |A1 - A2| <= sqrt(w1^2 + w2^2) (w = half-widths) in >= 95 % of pairs.

**IAM-A versions (design).** A read's own noise line needs, per read, the bisulfite / enzymatic conversion failure (unconverted non-CpG C on that
read) and its base-call error (base qualities): the isolated copy errors those two alone would make on that read are the read's noise line, and
the copy error is read above it. A `.pat` file keeps neither (no sequence, no quality), so the IAM-A versions of 3b, 3c and 11b cannot be tested on
the single-molecule data available; they need BAM input. Written into the SOP as designed, not built.
