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
