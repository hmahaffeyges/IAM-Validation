# DEV-COMPOSITION-TRUTH-03 — outcome (2026-10-10; development)

**Read as written** (`doors/data/DEV_COMPOSITION_TRUTH_03/score_truth_03.py`, inputs sha256-pinned, committed e4d138c before reading;
per-blood rows `truth03_rows.csv`). 238 arrays calibrated by Stage 1, 0 errors.

| platform | reader | bloods | MAE vs truth | max error | within 0.05 | bias |
|---|---|---|---|---|---|---|
| 450K (GPL13534) | atlas_e | 30 | 0.0411 | 0.0689 | 23/30 | +0.0411 |
| 450K | Stage A | — | not read: 297 of 963 markers on 450K (867 required) | | | |
| EPIC+custom (GPL29753) | atlas_e | 4 | 0.0760 | 0.0864 | 0/4 | +0.0760 |
| EPIC+custom | Stage A | 4 | 0.0705 | 0.0938 | 0/4 | +0.0705 |

**Decision: undecided — the truth is not precise enough for the bar.** The rule written before reading was to record the reason first.
`diag_truth_03.py` gives three reasons:
1. These whole bloods are not the mixture of the six sorted templates the simulation assumed: the fit leaves a median RMS residual of 0.064,
   at the top of the simulated range. Cells not sorted, or sorted impurely, leave signal no template carries.
2. The sorted CD14 template carries a granulocyte signal of 0.175 at 635 granulocyte-specific sites (the lymphocyte sorts are ≤ 0.011).
   Granulocyte signal the fit gives to CD14 lowers the CD15 truth, which shows up as a positive bias in every reader.
3. The truth moves with the site choice: median −0.018 to +0.022, up to 0.048 for one blood, against a 0.02 bar. atlas_e − truth runs
   +0.010 to +0.053 over the same choices.
On EPIC, atlas_e and the shipped Stage A agree (+0.076 and +0.071, the same sign and size) and both differ from the truth. That points to the
truth, not either reader, on these 4 people.

This set therefore neither passes nor fails bars 1–2. A truth set for the composition step needs counted cells (FACS or a differential) or
mixtures of known amounts. None from a second laboratory was found on GEO or ArrayExpress (DEV-ATLAS-COMMISSION-01). Next: GSE133062 lavage
(counted differentials; tests atlas v2 lung templates, not atlas_e).
