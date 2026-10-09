# DEV-METAA-450K-01 — commissioning Met-A on 450K purified neutrophils (written 2026-10-09, before any 450K array is read)

**Why.** Most disease data in S3 are 450K; commissioned Met-A refuses 450K (only 3,361 of its 6,000 frozen EPIC identity sites exist on
450K, below the 90 % gate). Scope of this note: **purified neutrophils on 450K** (whole blood follows once 450K composition is commissioned).

**Build (frozen before reading any test array).** Reference = GSE88824 healthy-control neutrophils (8 people, one laboratory). Identity
sites chosen by the frozen canon rule on those 8 only (SD ≤ 0.05; mean β 0.75–0.95 or 0.05–0.25; ≤ 3,000 per channel, smallest SD).
Floor = mean over the 8 of mean H(β) at the sites; per-site healthy H mean and shrunk spread (k = 10); clustering baseline at blocks
of 10, leave-one-out on the 8. Noise sites = the EPIC noise sites present on 450K; noise gate N_max re-set at the same quantile on the
8. Self-tare II anchors rebuilt by the EPIC rule on the 8. Every other step (median tare, ≥ 3 same-run references, refusals) unchanged.

**Bars (the EPIC v1 commissioning bars).**
1. Held-out reference precision: each of the 8 read with sites re-chosen on the other 7 — SD ≤ 0.020.
2. Other laboratories' healthy purified neutrophils (GSE124565 12, GSE65097 15, GSE35069 6 granulocytes, GSE318669 8 CD15),
   each tared against the other healthy ones of its own series: ≥ 95 % Normal (0.95–1.05).
3. Detection limit: constructed 2 % loss of pattern on each healthy held-out array reads outside Normal in ≥ 95 %.
**Disease readings (after 1–3 are met; one-sided, written now).** Lupus neutrophils (GSE65097, 15) and antiphospholipid-syndrome
neutrophils (GSE124565, 10), each tared against the healthy neutrophils of the same series: more read above Normal than the healthy
(Fisher one-sided p < 0.05), and the median A_rel is higher (Mann-Whitney one-sided). Lupus low-density granulocytes recorded only.

**Expectation added after simulation (DEV-SYNTH-LEVERS-01 §3), before reading.** Met-A detects a disease only if it shifts ≥ ~5 % of the
neutrophil identity sites by ≥ 10 % (power 0.89 at 15 vs 15). The published lupus-neutrophil signature (interferon genes, a few hundred
CpGs) would not do that, so a null in lupus neutrophils is the expected outcome and is not evidence against Met-A; lupus low-density
granulocytes (immature cells) are the likelier positive. A null here is recorded as "below what Met-A is built to see".
