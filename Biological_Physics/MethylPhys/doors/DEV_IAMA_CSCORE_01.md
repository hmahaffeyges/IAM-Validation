# DEV-IAMA-CSCORE-01 - the IAM-A C-score (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Definition (author decision C).** For one site table (Stage Q), sites in genomic order (chromosome, CpG index) are cut into blocks of
1,000 consecutive sites. Per block b: opportunities o_b and isolated copy errors k_b. With the reading's own copy error eps (Stage Q),
C = sum_b (k_b - eps o_b)^2 / sum_b eps (1 - eps) o_b.
\derived If every opportunity errs independently at the same rate eps, k_b is a sum of independent Bernoulli trials, its variance is eps (1 - eps) o_b,
and C = 1 within sqrt(2 / n_blocks). C > 1 means the errors cluster in genomic order. Nothing is fitted; the reference 1 is derived, not taken from
other readings. One C per A: the pooled reading and each half (A, B) get their own C. Fewer than 10 blocks: no C, with the reason.

**Checks.** Constructed: (i) independent binomial errors at eps 0.035 over 50,000 sites x 20 opportunities: |C - 1| <= 4 sqrt(2 / n_blocks);
(ii) the same with the error rate tripled in one block of every ten: C > 1 + 4 sqrt(2 / n_blocks). Real (DEV-IAMA-REAL-01): C for each Loyfer
granulocyte file and its halves, recorded (no band set).
