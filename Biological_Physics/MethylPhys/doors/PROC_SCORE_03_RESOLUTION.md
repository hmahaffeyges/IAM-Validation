# PROC-SCORE-03 — per-specimen resolution in blood, in β units (2026-09-30)

24 real lab-made Salas mixtures (GSE110554/GSE167998), 190 cell readings, cells at 2–40 % (median ~10 %), v2 solver fractions. Error = the cell's
separated channel mean β minus the same cell's purified mean β on the same loci.

| channel | median |error| | 90th percentile |
|---|---|---|
| methylated identity sites | 0.076 | 0.213 |
| unmethylated identity sites | 0.135 | 0.291 |
| both together (blind form) | 0.040 | 0.109 |

Biological shifts on the same channels, in the three series (median β change against the cells' own standard):
- transformation, methylated channel: BJ HRAS −0.014; HBEC tumour vs matched control ~0.01
- senescence, unmethylated channel: IMR90 −0.086; BJ −0.003
- senescence, methylated channel: IMR90 −0.014

**Reading.** For a minor blood cell (~10 %) read from one specimen, the channel error is ~5× the transformation shift and about the size of the largest
senescence shift. Minor-cell A in whole blood is below resolution per specimen with the current solver. Error scales roughly as 1/fraction: a
dominant cell at ~60 % would carry ~0.013 on the methylated channel (estimate, not measured), about the size of the transformation shift. Pure,
sorted, cultured and tissue specimens are unaffected. The resolution target for minor cells is a channel error ≤ 0.005.
