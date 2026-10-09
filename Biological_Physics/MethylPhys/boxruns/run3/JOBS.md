# Box Run 3 — first tests on the commissioned Met-A (neutrophils, EPIC v1). Written 2026-10-09, before any array below was read.

**Scope rule.** Commissioned Met-A reads EPIC v1 (GPL21145) purified neutrophils or whole/peripheral blood with neutrophil fraction ≥ 0.20,
tared against ≥ 3 healthy references from the same run (slide, else series). A dataset is in scope only if it has such references.
Every array is read through `conductor_v3.run_neutrophil` exactly as commissioned (self-tare II, median tare, noise gate, detection limits).

## Inventory of S3 (2026-10-09, from the series matrices)
| in scope, read now | why |
|---|---|
| **GSE315366** — 64 peripheral bloods, clonal haematopoiesis (CH) TRUE 29 / FALSE 35, DNMT3A and TET2 mutation status | references: the 35 CH-negative bloods |
| **GSE315367** — acute leukaemia, peripheral blood at diagnosis / remission 1 / remission 2 / relapse (30 bloods; 16 marrows refused) | no healthy references in the run: read untared, within-person only (development reading, not a test) |

| not in scope now | what it would need |
|---|---|
| 450K sets (GSE42861, GSE51032, GSE125105, GSE111629, GSE51057, GSE87640, GSE136724, GSE55763, GSE40279, GSE157131 450K, …) | Met-A commissioned on 450K |
| EPIC v2 (GPL33022) sets | Met-A commissioned on EPIC v2 |
| bone marrow, CLL blood (lymphocyte-dominated), cell lines, sorted non-neutrophil cells | other cells' references (atlas v2 commissioning, STATUS 2b) |
| GSE157131 EPIC (946 leukocyte samples) | a healthy reference group inside the run (none labelled) |
| plasma cfDNA, tissue, stool, animals, fish | their own commissioning (STATUS section 4) |
| sequencing sets | IAM-A commissioning |

## Test 1 — clonal haematopoiesis (GSE315366)
**Question.** Does the commissioned Met-A read CH-positive bloods differently from CH-negative bloods of the same study?
**Prediction (one-sided):** tared Met-A A_rel of CH-positive bloods is higher than of CH-negative bloods (a mutant clone loses part of the
neutrophil pattern). **Checks:** (1) CH-negative bloods, each read against the other CH-negative bloods: ≥ 95 % Normal (the instrument
works in this laboratory). (2) Mann-Whitney one-sided, CH-positive > CH-negative, p < 0.05. (3) Share above Normal in each group, recorded.
By gene (DNMT3A, TET2): recorded, no prediction.
**Known limit, stated before reading:** commissioned detection limit in whole blood is a 5 % loss of pattern. CH clones are often a few
per cent of blood cells (variant allele fractions are not in the deposit), so a null result can mean the clones are below the detection
limit, not that Met-A cannot see them.

## Reading 2 — acute leukaemia, same person over time (GSE315367; development reading)
No healthy references in the run, so no state is reported. Recorded: untared A per array, and for each person diagnosis − remission and
relapse − remission. Arrays below the 0.20 neutrophil fraction (blast-rich bloods) are refused by intake and counted.

## Run
`run3.py` on the box (m7a.8xlarge, root disk), IDATs from S3 `downloads/A_blood_immune/<GSE>/`, results to `results/BOXRUN3/`, box shuts
down at the end. Results go in the LOG; STATUS updated.

---
## Results, Test 1 — GSE315366 (2026-10-09; nothing above the line changed)
64 peripheral bloods read, 0 errors, 0 refused (median neutrophil fraction 0.65, none withheld by the noise gate).
| check | bar | result | met |
|---|---|---|---|
| 1. CH-negative bloods Normal (tared on the other CH-negative bloods) | ≥ 95 % | **34 / 35 (97.1 %)**, median A_rel 1.001 | yes |
| 2. CH-positive A_rel > CH-negative, one-sided Mann-Whitney | p < 0.05 | medians 1.0013 vs 1.0011, **p = 0.41** | no |
| 3. share above Normal | recorded | CH-negative 1 / 35; CH-positive 0 / 29 (DNMT3A 0 / 21, TET2 0 / 12) | – |
\measured Commissioned Met-A reads this new laboratory's bloods Normal and does not separate clonal haematopoiesis carriers. As written
before reading, this is expected if the clones are below the whole-blood detection limit (a 5 % loss of pattern); the deposit has no
variant allele fractions, so which explanation holds cannot be decided from these data.
**C-score test (a), first new laboratory:** CH-negative bloods not above 1.152: **34 / 35 (97.1 %; bar ≥ 95 %) — met.** CH-positive
above the band: 0 / 29.
