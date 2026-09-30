# PROC-V12-DIAG-01 — outcome (2026-09-29): the scatter follows the SOURCE, not the number of samples

Diagnostic only (PROC_V12_DIAG_01.md); nothing in the chain changed.

1. **More choosing samples does not tighten the reading.** Leave-one-out, fraction of left-out readings in NORMAL by number of
   choosing samples: 3 → 0.88, 4 → 0.87, 5 → 0.79, 6 → 0.71, 7 → 0.31, 8 → 0.28, 11 → 1.00, 12 → 0.77, 13 → 0.93. No trend. So the
   first explanation in the V12-02 outcome (selection noise from one choosing sample) is **not** the main cause.
2. **Donor-stable selection does not help** (L-stable 0.62 in NORMAL vs L-plain 0.67).
3. **The scatter follows the source of the left-out sample.** Fraction in NORMAL (L-plain): Salas2022 arrays 0.94 (53), GSE63409
   arrays 1.00 (4), Salas2018 arrays 0.57 (35), Loyfer2023 WGBS 0.56 (91), Moss2018 arrays 0.50 (20).
4. **Requiring every choosing sample inside the window fails for multi-source cells**: cortical neurons (Loyfer + Moss) keep 1 locus,
   hepatocyte 8, colon epithelium 52, so 40 readings are unreadable. Loci where a WGBS sample and an array sample of the same cell
   both sit at the floor are rare.

**Reading:** a single linear source term per source (a, b or d) puts each source on the atlas scale on average, but not locus by
locus near the floor. Where a cell's samples all come from one array source (neutrophils, eosinophils, naive B, memory CD4, GMP) its
identity set transfers to a new sample of that source inside NORMAL. Across sources it does not reliably. This is the same kind of
effect as the sky's offset off the identity loci (PROC-SKY-01): the scale correction is exact where it was fitted, not everywhere.

What a real specimen needs is transfer to **our Stage 1 array** — so the question for the author is how to choose identity loci for
cells whose samples are partly or wholly sequencing. Nothing is chosen until he rules.
