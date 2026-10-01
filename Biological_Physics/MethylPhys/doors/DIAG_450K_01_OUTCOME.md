# DIAG-450K-01 — why the v2 reader flagged an immune cell in every EPIC-Italy control (2026-10-01, development diagnosis)

GSE88824: 8 donors, 450K raw IDATs through our Stage 1 — whole blood (27 arrays incl. 19 extra), WBC, and purified neutrophils, monocytes, NK,
B, CD4 T, CD8 T per donor. Rules stated in the script before reading.

**D1 (the floor holds on 450K) — FAIL for three cells.** Purified arrays read against the current floors:

| purified cell | median A | in Normal |
|---|---|---|
| neutrophils | 0.932 | 0 / 8 |
| monocytes | 0.916 | 0 / 8 |
| NK cells | 0.904 | 0 / 8 |
| B cells | 0.988 | 7 / 8 |
| CD4 T (reads as naive CD4) | 0.996 | 6 / 8 |
| CD8 T (reads as naive CD8) | 0.992 | 6 / 8 |

The pure cells themselves sit 7–10 % below their floors on this platform. The defect is the floor's platform, not the specimen or the mixture.

**D2 (one person's dominant cell is read correctly from whole blood) — PASS.** Each donor's neutrophil read from whole blood (~40–59 %),
after subtracting the other cells, against the same donor's purified neutrophils: within ±0.05 in **7 of 8** donors (raw reading also 7 of 8).

**Fix tested: a 450K floor from purified 450K arrays, leave-one-donor-out** (a reference standard on the platform; no population of patients).
- Purified arrays read Normal in 8/8 (neutrophils, monocytes, NK), 6/6 (B).
- Whole-blood neutrophils read Normal in **7 / 8** donors (median 0.999).
- Minor cells in whole blood (monocytes 6 %, NK 4 %, B 9 %): 1–3 of 6–8 in Normal (medians 0.95–1.13) — below per-specimen resolution, as
  PROC-SCORE-03 found.

**Consequence for the chain.** (1) Floors must be platform-specific: built from purified arrays of that platform. (2) In whole blood, A is
printed for the dominant cell (neutrophils) only; minor cells get their fraction and no A. (3) The EPIC-Italy re-run uses both rules.
