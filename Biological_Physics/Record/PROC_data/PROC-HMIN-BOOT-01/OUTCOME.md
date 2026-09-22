# PROC-HMIN-BOOT-01 — the bootstrap cross-check of the eight methylation H_min values, run for the first time

**Run 2026-09-20.** Prompted by the author's request to cite the H_min calibration in Paper 1, and by checking the evidence report's links: `bootstrap_vs_mcmc_comparison.tsv` (commit 22749f0, removed from HEAD in 538667d) has **32 rows = 4 substrates (nucl, fuzz, wps, frag) × 8 classes — no methylation rows.** The record's sentence "all 40 values are MCMC posterior means with R-hat < 1.001 and bootstrap cross-validation agreement at 0.168 %" is therefore true of the 32 G-003b floors and **was never true of the eight G-002 methylation floors**, which had MCMC (R-hat < 1.001) and no bootstrap.

**What was run.** The G-002 reference database (37 published reference cell methylomes, 4–6 per class, Roadmap/ENCODE/Lister, FACS-sorted or microdissected) and the `bootstrap_h_min` function from `gape_bootstrap_comparison.py`, both taken from commit 22749f0; 10,000 resamples per class, seed 42, statistic = mean over cells of H(β) (the G-003b likelihood form); plus exact leave-one-out. Compared against the frozen values in `iamatlas_gauge_identity_loci_v1_0.json` at HEAD.

| class | n cells | frozen H_min | bootstrap mean | 95 % CI | LOO range | frozen in CI | rel diff |
|---|---|---|---|---|---|---|---|
| stem_pluri | 4 | 0.982200 | 0.982698 | [0.9786, 0.9864] | [0.9810, 0.9847] | yes | 0.051 % |
| stem_adult | 5 | 0.873700 | 0.873338 | [0.8608, 0.8868] | [0.8669, 0.8778] | yes | 0.041 % |
| progenitor | 4 | 0.852200 | 0.851908 | [0.8450, 0.8588] | [0.8485, 0.8554] | yes | 0.034 % |
| terminal | 5 | 0.772800 | 0.772456 | [0.7605, 0.7854] | [0.7668, 0.7765] | yes | 0.045 % |
| cycling | 5 | 0.856100 | 0.855534 | [0.8440, 0.8718] | [0.8474, 0.8589] | yes | 0.066 % |
| immune | 6 | 0.838889 | 0.838093 | [0.8170, 0.8589] | [0.8292, 0.8464] | yes | 0.095 % |
| secretory | 4 | 0.843300 | 0.842750 | [0.8304, 0.8601] | [0.8341, 0.8481] | yes | 0.065 % |
| stromal | 4 | 0.863000 | 0.862297 | [0.8488, 0.8794] | [0.8540, 0.8683] | yes | 0.081 % |

**Verdict: 8/8 frozen methylation floors inside the bootstrap 95 % CI; mean relative difference 0.060 %, max 0.095 %.** The methylation calibration is method-independent, as the record claimed — it just had not been shown until today. The frozen values are unchanged.

**Two consequences.** (1) The "all 40 values" sentence is corrected wherever it appears to "32 by G-003b bootstrap (0.168 %, 24/32 in CI); 8 methylation by PROC-HMIN-BOOT-01 (0.060 %, 8/8 in CI)". (2) The calibration code (`gape_mcmc_g002.py`, `gape_mcmc_g003b.py`, `gape_bootstrap_comparison.py`, the TSV) is **not at HEAD** — removed 2026-04-19 as the "commercial calibration layer" — but is in the public git history at 22749f0, so the evidence report's links are dead while the files remain retrievable by anyone. Disclosure decision for the author (see HANDOFF).

---
**SEALED** sha256 `15d35f686023e437ad72c3f2012dd7199d5fb7a64b858c9445e92d28936956ea` · 2026-09-20
