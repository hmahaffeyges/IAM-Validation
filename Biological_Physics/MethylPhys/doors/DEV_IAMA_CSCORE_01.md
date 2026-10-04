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

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Constructed check run on the laptop with `stage_q_iam_a.cscore` (the same test is release check E9, run on the box):
- \measured (i) independent errors (eps 0.035, 50,000 sites x 20 opportunities, 50 blocks): **C = 0.823** (limit 1 +/- 0.80, within); half A 0.616.
- \measured (ii) error rate tripled in one block of every ten: **C = 447.5** (> 1.80, within the bar).
- \measured Real data (DEV-IAMA-REAL-01): whole files C 604-1,047, heads 11-17; halves C about half of the pooled C (they carry half the opportunities).
- \observed Healthy granulocytes do not hold the independent-error null: the copy-error rate changes along the genome. With 1,000-site blocks the C-score
  measures that regional spread as much as any clustering in one specimen. No band is set; the reading stays development.
