# OUTCOME — PROC-CMB-03: FAILED AS SEALED (C2′ 0/4); C4′ cap bar RETIRED by the author

**Run 2026-09-21, as sealed.** Floors: non-blood classes max(0.02, panel p99) → terminal 0.03, others 0.02; blood-lineage 0.02.

| lab | median frac \|z\| > 2 [range] | median z | C2′ |
|---|---|---|---|
| GSE87571 | 0.018 [0.005–0.150] | -0.006 | FAIL |
| GSE42861 | 0.015 [0.006–0.089] | -0.032 | FAIL |
| GSE111629 | 0.019 [0.004–0.102] | -0.012 | FAIL |
| GSE125105 | 0.022 [0.006–0.120] | -0.024 | FAIL |

Centred (median z within ±0.032 everywhere) but tails 1.5–2.2 %: **the scale is ≈ 18 % too large.** Cause, identifiable a priori: the shrinkage target was the β-bin **RMS** of per-CpG SDs, dominated by the few highly variable CpGs (SNP-adjacent, metastable), pulling every ordinary CpG's scale up. Gate rule 160/160; immune rendered 160/160; one healthy test array rendered a non-blood class (terminal ≥ 0.03). C5 PASS, C6 PASS. Cross-lab: GSE87571->GSE42861 0.061 / -0.17; GSE87571->GSE111629 0.050 / +0.04; GSE87571->GSE125105 0.148 / -0.17; GSE42861->GSE87571 0.039 / +0.18; GSE42861->GSE111629 0.062 / +0.32; GSE42861->GSE125105 0.104 / -0.10; GSE111629->GSE87571 0.057 / -0.05; GSE111629->GSE42861 0.115 / -0.29; GSE111629->GSE125105 0.157 / -0.23; GSE125105->GSE87571 0.063 / +0.14; GSE125105->GSE42861 0.054 / +0.03; GSE125105->GSE111629 0.054 / +0.24 — the zero is per-laboratory.

**Bar retired by the author (2026-09-21):** "≤ 3 test arrays render a non-blood class" assumed healthy blood never carries secretory, cycling, terminal, stromal or stem_pluri above the floor. The author: "We have detected secretory and cycling in whole blood and scored it fine, as long as it is over a percent we have been surprised on what actually shows up sometimes." A bar that penalises the instrument for rendering a class it found is a prior about biology, not a test of the gate. From CMB-04 the gate is judged on following its rule; what renders is REPORTED per class, with the rendered panel's healthy quietness reported beside it.

**Standing rule (author, same day):** no definitive statement about what the new chain can or cannot detect until it has been run on that question under seal. "Not yet tested", never "cannot".

---
**SEALED** sha256 `ab0797b09a191a2386d4f2e5749c03ac9fe54efc601b8c0260f807a03d7ca59b` · 2026-09-21
