# DEV-LOWFRAC-01 — neutrophil Met-A below 50 % neutrophils (development, 2026-10-01; after looking)

**Run.** 656 whole bloods (COVID-19 GSE179325 574, technical-replicate set GSE250556 64, FACS-counted GSE112618 6, known mixtures 12) through the chain's Stage 1
and Stage A, neutrophil Met-A computed at every fraction (no 50 % cut). Per array: noise index N, and the shift a known 2 % neutrophil blur would cause at
that array's own fraction. Healthy spread: the 101 NEGATIVE adults with a reading at any fraction, each divided by the expectation fitted on the other 100 from (fraction, N).

| neutrophil fraction | healthy n | healthy SD | healthy in Normal | 2 % blur shift | shift / SD |
|---|---|---|---|---|---|
| 0.40–0.50 | 18 | 0.024 | 18/18 | 0.033 | 1.4 |
| 0.50–0.60 | 16 | 0.022 | 15/16 | 0.040 | 1.8 |
| 0.60–0.70 | 22 | 0.024 | 21/22 | 0.050 | 2.1 |
| 0.70–1.00 | 40 | 0.020 | 40/40 | 0.064 | 3.2 |

(Below 0.40: 5 healthy arrays only.) Fraction recovery on the known mixtures: median error 0.034.

**What this shows.** Below 50 % the healthy reading stays as tight as above it; the noise does not blow up. What falls is the signal: a cell that is 40 % of the
specimen moves the reading half as much as one that is 80 %. So the 50 % cut is the wrong rule. The rule that follows from the physics is a detection limit per
specimen: report A when the known-change shift at this specimen's own fraction is at least twice the healthy spread (here ≥ 0.6 neutrophils for a 2 % change,
≥ 0.4 for a 4 % change). Every report states the smallest change it could have seen.

**Limits.** One lab for the healthy spread; the (fraction, N) expectation was fitted after looking and needs a held-out lab with ≥ 20 healthy references.
Simulated damage only.

**Basis of MIN_READ_FRACTION 0.20, computed (2026-10-10; `doors/data/DEV_LOWFRAC_01/shift_vs_fraction_01.py`, the chain's own reader and
profiles, no data).** The chain states that below 0.20 a 1 % loss shifts Met-A by under 0.01. Computed: the 1 % shift is 0.0073 at 0.20 and
reaches 0.01 at 0.27. The same computation reproduces the measured 2 % shifts above (0.031 at 0.40, 0.040 at 0.50; measured 0.033 at 0.40–0.50).
So 0.20 does not follow from its stated basis. Either the cut moves to 0.27 (its stated basis), or the fixed cut gives way to this note's own
rule, the per-specimen detection limit the chain already prints. Recorded for decision; the chain is unchanged.
