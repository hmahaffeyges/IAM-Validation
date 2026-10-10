# DEV-CSCORE-MOSS-01 — Moss 2018 in vitro mixes (GSE122126): what the chain reads (development; written 2026-10-10 before any mix array is read)

**Material.** 9 EPIC arrays of one donor's leukocyte DNA with 4–15 % hepatocyte, lung, colon or neuron DNA at known fractions (GEO descriptions
= paper Supplementary Data 1 Table 6; GEO mix_9..17 = paper Mix1..9). The same series holds the pure components on EPIC (leukocytes 1,
hepatocytes 2, lung 3, colon 3, neurons 2).

**Step 1, in silico from the pure arrays** (`data/DEV_CSCORE_MOSS_01/insilico_moss_01.py`, output `insilico_moss_01_output.txt`, rows
`insilico_moss_01_rows.csv`). Each mix is built in beta space from every combination of component replicates and read by
`conductor_v3.run_neutrophil(specimen="constructed DNA mixture")`. Component arrays lose up to 23 % of identity sites to Stage 1's background
line; a missing site takes the other replicates' mean, else the leukocyte value (no tissue difference assumed there, which understates the change).
An earlier build left those sites missing; its C readings (1.04–1.12) came from the gaps, not the tissue, and are not used.

**Result.** A moves by less than 0.01 (leukocytes alone 1.0035; mixes 0.9998–1.0072). C stays inside the healthy range 0.954–1.046 in 49 of 51
builds (Mix10 maximum 1.060). Tissue DNA at 4–15 % is not clustered change at the neutrophil identity sites, so this set **cannot test the
C-score's detection**. It tests specificity: neither instrument should call a change.

**Prediction for the real mix arrays, sealed now** (read by the same call, no tare: one leukocyte array only):
1. Met-A A of each of the 9 mixes within Normal 0.95–1.05 and within 0.03 of the leukocyte array read on the same series.
2. C of each mix within 0.90–1.10 (the healthy LOO range widened by its own width to allow for one-array noise).
Either failing means the instruments call a change where the known material has none at their sites.
