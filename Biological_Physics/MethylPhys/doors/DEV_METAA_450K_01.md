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

## Bar 2b — GSE224807, 64 healthy non-smoker CD15 neutrophils (2026-10-09)
64 of 65 arrays calibrated (one IDAT missing). **43 of 64 Normal (67 %) — bar (≥ 95 %) NOT MET.** Tared A_rel 2.5–97.5 % 0.936–1.119.
- Design: the 64 arrays sit on **47 slides** (30 alone, 17 in pairs), so no array has ≥ 3 same-slide healthy references and every tare fell
  back to the series median; the same-run tare cannot remove slide effects here.
- Coverage: 62 of 64 pass the 90 % site gate; excluding the two does not change the result (45 of 62).
- Noise gate built by the EPIC rule (N_max = top of the 8 reference arrays = 0.1622; 40,228 of 48,528 EPIC noise sites on 450K):
  withholds none of the 64 and does not track A_rel (r = 0.14). (It would withhold all 12 GSE124565 healthy arrays, N 0.168–0.179, which
  were read through the same-run tare as the EPIC rule allows.)
- Candidate causes, not tested: slide-to-slide contrast left after self-tare II; CD15 sort purity (CD15 also marks eosinophils).
**Consequence.** 450K Met-A stays development. Nothing was changed after seeing these numbers. The APS reading is not run (the note
requires bars 1–3 first). Next: test the two causes on this series (eosinophil-marker sites; pairs on one slide vs across slides).

## Bar 2b diagnosis (2026-10-09; development; nothing above changed)
**Cause: slide / processing batch, not a mixed-in cell type.** The 17 slides holding two of these neutrophil arrays hold two different
people (consecutive sample numbers, e.g. F186/F187, F102/F103). Their tared readings agree to a median |difference| of **0.013**; two
arrays from different slides differ by **0.051** (one-sided p < 0.001). A second cell type mixed in at varying amounts would differ
person by person, not slide by slide. One direction carries the spread: the first component of the identity-site residuals holds 17 %
of the variance and tracks the reading (r = 0.89), and it lowers the methylated identity sites while leaving the unmethylated ones
(a shift of the methylated channel, as a processing offset gives). The self-tare on invariant sites does not remove it.
**What it means for 450K.** Where same-slide healthy references exist (GSE124565), 12/12 read Normal; where they do not (this series,
1–2 neutrophil arrays per slide), the series-level tare cannot remove a slide offset. 450K Met-A stays development. Rule to test before
use: a 450K reading needs ≥ 3 same-cell-type healthy references on its own slide (the EPIC Stage T rule). Next: a third laboratory with
same-slide healthy neutrophils, scored by bars 1–3 as written; APS and the disease readings wait on it.

**Reproduce (2026-10-09):** `doors/data/DEV_METAA_450K_01/metaa_450k_01.py` rebuilds every 450K number above from pinned inputs
(`inputs_sha256.json`; sample sheets `samples_GSE*.csv`). Held-out after self-tare 0.0177 (0.976–1.028) reproduced exactly; with the
anchors also held out (stricter) 0.0170 (0.977–1.031), so bar 1 holds either way.
