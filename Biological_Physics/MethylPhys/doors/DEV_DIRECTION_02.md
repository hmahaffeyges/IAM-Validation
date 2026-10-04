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
