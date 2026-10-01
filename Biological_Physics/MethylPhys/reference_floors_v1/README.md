# Met-A reference floors v1 — development record (2026-10-01)

Development build, not commissioned; no prediction is declared. Script: `refloor.py` (run on box 1, purified specimens through our Stage 1).

**What a reference floor is.** For one cell type on one platform: the Met-A quantity measured on purified healthy specimens of that cell type.
Classes are not used. Platforms: 450K, EPIC (Moss 2018 purified tissue cells; Salas 2018/2022 purified blood cells) and WGBS (Loyfer 2023,
projected onto array probe positions, coverage ≥ 10). Loci: the v2 identity loci of each cell.

**Held-out precision.** Every purified array, read against the other specimens of its cell type and platform: 105/105 in 0.95–1.05,
spread (SD) 0.014. Tightest: naive CD4 0.004, monocytes 0.008; widest: Tregs and colon epithelium 0.029.

**Sensitivity — the finding that shapes the method.** Each purified specimen was blurred toward β = 0.5 by a known amount and re-read
(`refloor_snr.csv`; shift = median A − 1; SNR = shift / held-out SD):

| data | form | held-out in Normal | SD | 5 % blur | 10 % blur | SNR at 10 % |
|---|---|---|---|---|---|---|
| arrays | H(mean β), all identity sites | 1.00 | 0.0145 | +0.021 | +0.039 | 2.7 |
| arrays | per-site mean H | 1.00 | 0.0145 | +0.021 | +0.040 | 2.8 |
| arrays | extreme sites (own β < 0.2 or > 0.8), per-site H | 1.00 | 0.0141 | +0.026 | +0.051 | 3.6 |
| WGBS | H(mean β), all identity sites | 0.90 | 0.035 | +0.014 | +0.030 | 0.9 |
| WGBS | per-site mean H | 0.84 | 0.040 | +0.071 | +0.124 | 3.1 |
| WGBS | extreme sites, per-site H | 0.64 | 0.081 | +0.242 | +0.427 | 5.3 |

1. On arrays, Met-A on the v2 identity loci reads healthy cells tightly but moves little when the pattern blurs: a 10 % blur still reads
   inside Normal for most specimens. The best array form is the extreme sites, but EPIC has few of them among the v2 loci (median 119).
2. On sequencing, the per-site and extreme-site forms see pattern loss far better, and their healthy spread is wider than ±0.05.
3. The two-channel split fails on arrays with the v2 loci: nearly every identity site sits above β 0.5 on arrays (unmethylated channel
   0–178 sites), so the v2 loci are not suited to a methylated/unmethylated reading on arrays.
4. Sequencing floors cannot stand in for array floors: array/WGBS floor ratio 0.96–1.84 across 21 cell types; one factor fits 6/21 to ±0.05.
   Tissue array floors need purified tissue arrays (only 9 tissue cell types exist here, most n = 1).

**Next build steps.** (a) Identity sites chosen per platform from purified arrays (stable across donors and at the extremes), so the array
form gains sensitivity; (b) state on every floor what A = 1.05 corresponds to in blur, per form and platform; (c) collect purified tissue
arrays for the tissue floors.

## v1.1 — identity sites chosen per platform from purified arrays (EPIC, 97 held-out specimens)
Sites for a cell are chosen without the specimen being read: across-donor SD ≤ 0.05, and own mean β in the window shown. Up to 3,000 per channel.
Same ruler for every form (author, 2026-10-01): Normal 0.95–1.05; the site window is tuned instead.

| sites | form | healthy in Normal | SD | 2 % blur | outside at 2 % | 5 % blur | outside at 5 % |
|---|---|---|---|---|---|---|---|
| v2 identity loci (v1) | per-site mean H | 1.00 | 0.015 | +0.008 | 0.00 | +0.021 | 0.00 |
| extreme (≥ 0.90 / ≤ 0.10) | per-site mean H | 0.66 | 0.072 | +0.331 | 0.99 | +0.798 | 1.00 |
| moderate (0.80–0.95 / 0.05–0.20) | per-site mean H | 0.85 | 0.046 | +0.099 | 0.93 | +0.238 | 1.00 |

- Taking each floor only from the same study does not narrow the spread (moderate per-site 0.047), so the spread is not study batch.
- On moderate sites most blood cells read healthy inside Normal (neutrophils, monocytes, NK, naive CD4/CD8, Tregs, memory CD4: SD 0.015–0.027).
  Wide: basophils 0.100, CD8 (Salas 2018, naive + memory mixed) 0.084, memory B 0.063, colon epithelium (Moss, n = 3) 0.151.
- The per-site form beats the two separate channels on arrays.
- Next: more sites per cell (cap 3,000 now), choose the window per platform so healthy SD ≤ 0.025, and state the blur at which A crosses 1.05
  so IAM-A is calibrated to the same point.
