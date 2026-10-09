# DEV-METAA-SENS-01 — does Met-A move when the neutrophil loses its pattern? (constructed positive control, development, 2026-10-09)

**DEVELOPMENT - not commissioned.** Checks written before any reading below the line.

**Why.** Every Met-A commissioning bar so far is a specificity bar (healthy reads Normal). A gauge that always printed 1.00 would pass them.
Job B found no in-scope real positive control: the treated and myeloid sets are cell lines, bone marrow or untared; the infection set's
disease and healthy arrays sit on different slides with different neutrophil fractions; GSE118144's patient and healthy neutrophils read the
same (0.999). The chain already defines a constructed change (conductor_v3: each identity site moves L % of the way toward β = 0.5, "loss
of pattern"). On the reference profile a 1 % loss moves A by +5.2 % (calculated). This check runs that change through the whole chain on
real healthy arrays, so the tare and self-tare II act on it as they would on a real change.

**Data.** Healthy arrays already read in job B: GSE247195 (24 isolated neutrophils, one laboratory, one run) and GSE110530 (12 whole bloods).
Stored betas: `results/DEV_BASE_CHAIN_01/betas/`.

**Construction.** One array at a time is the test array; the others of its series are its same-run references, read unchanged. At the
6,000 neutrophil identity sites (`blood_composition_EPIC_v1.json`), β' = β + L/100 · (0.5 − μ_NEU) for isolated neutrophils, and
β' = β + f · L/100 · (0.5 − μ_NEU) for whole blood, where f is that array's own neutrophil fraction from Stage A (the loss is in the
neutrophils only). L = 0, 0.5, 1, 2, 5. Then `conductor_v3.run_neutrophil` with the references.

**Checks.**
1. L = 0 reproduces job B's tared A for each array (within 0.0005).
2. Measured change in tared A per 1 % loss, against the chain's own `shift_per_1pct_loss` for the same array: median ratio within 0.9-1.1.
3. At L = 1 and above, the tared reading leaves Normal (above 1.05) on ≥ 95 % of test arrays in each series; at L = 0.5 the share is recorded.

---
## Results (2026-10-09, local; nothing above the line changed)

| | GSE247195, 24 isolated neutrophils | GSE110530, 12 whole bloods (median neutrophil fraction 0.52) |
|---|---|---|
| check 1: L = 0 against job B | untared A identical (max diff 0.0000); tared A identical with job B's own same-slide references | same |
| check 2: measured change per 1 % loss ÷ chain's `shift_per_1pct_loss` | 0.0519 ÷ 0.0520, median ratio **0.998** (0.966-1.006): met | 0.0223 ÷ 0.0214, median ratio **1.043** (1.029-1.051): met |
| median tared A at L = 0 / 0.5 / 1 / 2 / 5 % | 1.000 / 1.026 / 1.052 / 1.102 / 1.246 | 1.000 / 1.011 / 1.021 / 1.042 / 1.104 |
| share above 1.05 at L = 0.5 / 1 / 2 / 5 % | 0 / 0.875 / **1.000** / 1.000 | 0 / 0.083 / 0.417 / **1.000** |
| check 3 (≥ 95 % above 1.05 at L ≥ 1) | **not met at L = 1** (0.875); met at 2 and 5 | **not met at L = 1 or 2**; met at 5 |

**Reading.**
- \measured The chain responds to a known loss of the neutrophil pattern by the amount its own model says (check 2), through the self-tare
  and the median tare, on real arrays. Met-A is not a gauge that always prints 1.
- \measured Every healthy array leaves Normal at a 2 % loss in purified neutrophils and at a 5 % loss in whole blood (where only about half
  the cells carry the loss).
- Check 3 as written is not met. It was set badly: a 1 % loss moves A by 5.2 % on the reference profile (stated above the line), which puts
  the expected reading on the 1.05 edge, so about half-to-most arrays cross it by chance. The bar is left as written and recorded as not met;
  the measured detection limits above are the result. No bar is changed after the fact.
