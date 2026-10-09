# DEV-HORIZON-01 — is there a reader-independent edge of disorder? (conjecture test, simulated first, 2026-10-09)

**Source idea (large scale).** A horizon is a boundary past which information about what fell in cannot reach any outside observer.
**Cell question.** As loss rises, is there an edge past which a molecule carries no information about the state of its domain — the same
edge for every reader — or does information fade smoothly, with any "edge" set by the reader's own cutoff?
**Status.** CONJECTURE (cell-black-hole step, book saturation appendix). Simulation only; no data read.

**Model.** Two domains (normally methylated, normally unmethylated), 8 CpGs per molecule, 40,000 molecules, 60 divisions, loss raised by
s × (s = 1 healthy, 24 = both domains at β 0.5). Information = mutual information (bits) between a molecule's methylated count and its
domain (an optimal reader; 1 bit = fully informative). Compared with readers that keep molecules ≥ 60/70/80/90 % methylated.
Two holding mechanisms: independent sites, and cooperative holding (a site is held and restored better when its neighbours are
methylated; neighbour cooperativity of maintenance is established in the literature).

| s | independent: β | information | readable ≥ 80 % | cooperative: β | information | readable ≥ 60 % | ≥ 80 % |
|---|---|---|---|---|---|---|---|
| 1 | 0.96 | 1.00 | 0.96 | 0.99 | 1.00 | 1.00 | 1.00 |
| 8 | 0.75 | 0.78 | 0.37 | 0.89 | 0.99 | 0.99 | 0.79 |
| 12 | 0.67 | 0.47 | 0.19 | 0.81 | 0.91 | 0.95 | 0.52 |
| 16 | 0.60 | 0.20 | 0.11 | 0.69 | 0.64 | 0.80 | 0.22 |
| 18 | 0.57 | 0.11 | 0.08 | 0.62 | 0.38 | 0.63 | 0.09 |
| 20 | 0.55 | 0.05 | 0.06 | **0.50** | **0.002** | **0.011** | **0.000** |
Steady state reached (60 vs 240 divisions identical); no hysteresis (same β from methylated or unmethylated start).

**Findings.**
1. **Independent holding: no horizon.** Information fades smoothly; the readable share falls at a point set by the reader's cutoff.
   IAM-A's collapse in Simulation 2 is of this kind: an instrument edge.
2. **Cooperative holding: a reader-independent edge.** Information stays near 1 bit, then collapses within one step of loss (s 18 → 20:
   0.38 → 0.002 bits), at the same place for every reader, as the domain falls to β = 0.5 (maximal disorder, 1 bit of entropy per site).
   Past it the information is gone from the molecule itself, not hidden from one reader.
**Prediction (CONJECTURE, before any data).** If cells hold methylation cooperatively, tumour domains should be either near their healthy
state or at β ≈ 0.5, with few in between, and the information-per-molecule of a domain should drop sharply, not smoothly, against its
loss; the edge's position should not move with the reader's cutoff. If tumour domains fade smoothly, the cell has no horizon of this kind.
**Next (simulate first).** Turn this into a region-level readout on tumour WGBS with matched normal (distribution of domain β and
information against local loss), and its power, before choosing a data set.
