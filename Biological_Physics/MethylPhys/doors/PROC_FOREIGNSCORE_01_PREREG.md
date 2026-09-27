# PROC-FOREIGNSCORE-01 — pre-registration: the scoring floor for a detected foreign cell

**Written 2026-09-27 before any spike is scored.** Author, 2026-09-27: "If we can detect a cell and know its a cell we think it is…
we can score it at 1.9 percent according to early VAL's but I think we need to test that theory obviously." This is that test.

## Question
A foreign cell detected and named by Stage 2d (PROC-STAGE2D-03) at fraction f: from what f upward does its A, read on its own
identity loci, report the cell rather than the specimen it sits in? At f the identity loci carry (1−f) of the other cells' β
(FRACTION_AND_A.md).

## Construction (fixed)
Honest spikes: the cell's atlas profile at the loci where it is measured, host β elsewhere; 12 healthy Uppsala hosts not used in
PROC-STAGE2D-03's exploration or its 12 spike hosts; cells: Cortical_neurons, Glia (full coverage), Colon_epithelial_cells, Kidney,
Hepatocytes, Pancreatic_beta_cells (thin, detectable at 5 %); f = 0.02, 0.05, 0.10, 0.20, 0.50. Truth: the spiked profile is the
atlas mean, which reads A = 1.000 on its v1.1 identity loci by construction. Two readings per spike, both through the chain's statistic
A = H(mean β over the cell's identity loci) / H_min:
- **raw**: mean β over the specimen at the cell's identity loci (what the chain reads today when a cell is present);
- **inverted**: β_cell ≈ (β_specimen − (1 − f̂)·β_blood) / f̂ at the identity loci, with f̂ the joint-fit fraction from Stage 2d and
  β_blood the specimen's blood-only reconstruction from the composition solver's blood fractions and the atlas means. No population enters.

## Bars
- **B1** the raw reading's |A − 1.000| per f, median over hosts, is reported; the **raw floor** is the smallest f with median |ΔA| ≤ 0.02
  (less than half the NORMAL tolerance) and ≥ 90 % of spikes inside 0.05.
- **B2** the same for the inverted reading; the **inverted floor**.
- **B3** the floors are written to foreign_scoring_floor_v1.json (A_Scoring_Module, written by the outcome) per cell with the measurement; a
  detected foreign cell below its floor prints "fraction f̂, A not read below the scoring floor f_floor".
- **B4** at f = 0.50 both readings are inside 0.02 (sanity: the statistic reads the cell when the cell is half the specimen).
- **B5** no blood cell's A on the host moves by more than 0.005 under any spike ≤ 0.10 (spiking a foreign cell does not move the blood reading).

## Decision rule
Whatever floors are measured are adopted as measured — this procedure sets a constant, it does not pass or fail the chain. If no
f ≤ 0.20 meets B1 or B2, foreign cells print fraction only and the floor file says so. If the inverted reading beats raw by ≥ 2× in
floor, the inversion is wired in as the foreign-cell reading (a separate change, recorded).
