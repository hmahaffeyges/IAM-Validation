# DEV-DIRECTION-02 - directional decomposition rebuilt physics-only (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Definition (author decision I).** For a cell's identity site i with reference value r_i (the cell's own floor pattern: isolated neutrophils
r = the purified neutrophil profile of `blood_composition_EPIC_v1.json`; whole blood r = the composition-matched expectation e_i of Stage M), the signed
move is d_i = |r_i - 0.5| - |beta_i - 0.5|. d_i > 0 is a move toward 0.5 (toward disorder), d_i < 0 a move away from 0.5 (toward over-order).
D = mean d_i over the identity sites measured. Tared like Met-A: D_rel = D - median(D of >= 3 same-run references); reference spread s = SD of the
references' D. A direction is read when |D_rel| > 2 s (the Stage T detection-limit form); otherwise "no direction". Nothing is fitted; no panel.

**Cell lines (treated series).** The cell's own floor is its untreated state: identity sites = sites the vehicle (DMSO) arrays of that cell line hold
(across-array SD <= 0.05, mean 0.75-0.95 or 0.05-0.25, up to 3,000 per state - the Met-A site rule), r_i = vehicle mean, references = the other vehicle
arrays of the same cell line (self excluded; 2 references where only 3 vehicle arrays exist - stated).

**Sets and bars.**
1. GSE250556 replicates (63, whole blood, same-slide references as Stage T): no direction expected. Bar: >= 95 % read "no direction".
2. GSE187291 (MV4-11 and HL-60; DMSO, decitabine, NTX-301 DNMT1 inhibitor; day 2; 3 replicates each): known push = loss of methylation, so methylated
   identity sites move toward 0.5. Bar: every decitabine and NTX-301 array reads "toward disorder"; every DMSO array reads "no direction".
3. Reported only: GSE165185 (azacitidine / decitabine, sensitive and resistant lines, no vehicle in the series: D against the series' own median).

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 5bd710b8. Records: `data/DEV_DIRECTION_02/`.

1. \measured GSE250556 replicates (63, same-slide references): **56 of 63** "no direction" (88.9 %; bar 95 %, outside the bar); 4 toward disorder, 3 toward over-order
   (A 2/0, B 0/1, C 0/2, D 2/0 by person).
2. \measured GSE187291: **every decitabine and NTX-301 array reads toward disorder (12 of 12)**: D_rel HL-60 decitabine 0.080-0.082,
   NTX-301 0.039-0.041; MV4-11 decitabine 0.064-0.068,
   NTX-301 0.021-0.022; the vehicle spread is ~1e-5. DMSO arrays: **4 of 6** "no direction" (bar: all);
   the two others sit 1e-5 from the median of the other two vehicles, against a spread of the same size.
3. \observed GSE165185 was not read: its four arrays carry no vehicle in the series record, and D against the series' own median needs >= 3 references.
- \observed The physics-only direction separates a known loss of methylation by four orders of magnitude over the vehicle spread. The vehicle arrays
  were part of their own reference (sites and r from all three), which makes their spread artificially small; that, not the treated arrays, is why
  two vehicle arrays are called.
- \openprob The replicate rate (88.9 %) sits outside the bar because 2 s of same-slide references is a tight line when only 3-5 references exist.

**Wiring.** Behind `--dev-direction` (reads D; D_rel when the reference table carries a D column). Not part of the reading.
