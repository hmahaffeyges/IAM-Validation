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

---
## Results so far (2026-10-09; nothing above the line changed)
**Data change.** GSE65097 (lupus) and GSE35069 deposited no IDATs (processed intensity tables only), so Stage 1 cannot read them: the lupus
reading is dropped from this route. Second other laboratory: GSE224807 (sorted blood cells, smokers and non-smokers), 65 non-smoker CD15
neutrophils on 450K with IDATs (replaces GSE318669, whose 54 arrays read 894,182 probes — platform to be checked before use).
**Build.** 6,000 identity sites on the 8 GSE88824 control neutrophils (122 shared with the EPIC set); floor 0.32581. Self-tare II rebuilt by
the EPIC rule on GSE88824's six purified groups: I_low 51,884, I_high 20,166, II_low 43,329, II_high 45,320 fixed sites
(Runtime Matrices/Development/dev_metA_450K_neutrophils_v0.json, dev_selftare_450K_v0.json).
| bar | result | met |
|---|---|---|
| 1. held-out precision | raw SD 0.0336; after self-tare II **0.0177** (0.976–1.028) | **yes** (after self-tare, as EPIC) |
| 2a. other lab GSE124565, 12 healthy | untared A 0.943–0.980 (lab ~4 % low); tared A_rel 0.983–1.028 — **12/12 Normal** | yes (one lab) |
| 2b. other lab GSE224807, 65 healthy | downloading | pending |
| 3. detection limit | constructed 2 % loss: 12/12 outside Normal (A_rel 1.106–1.151); 1 %: 10/12 | **yes** |
Noise gate not yet rebuilt for 450K (applied in commissioning). APS patients not read until bar 2b is met.
