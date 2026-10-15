# Whiteboard for coffee, 2026-10-15 (Marcel's overnight work)

Everything here is checked in code; numbers come from our own records (DEV-WRITER-01, PROC-CHANNEL-01 section 7, DEV-ARCH-01).
Nothing new was fetched. Status of each item is marked.

**News overnight: DEV-ERASE-01 is NOT MET.** All five primary units read: eraser removed over control 0.96, 0.79, 0.86, 0.89, 0.85
(4 NOT MET, 1 inconclusive). In every set the held error drops a little when the eraser is gone, never by half. So the eraser is real
but small where we count, which is what item 2 below says (item 2 was written after U1 and before U2-U5 were read; exploration, not a sealed test). The middle line is untouched: it is measured.

## 1. The black-hole route to the half does not work for the cell (dead end, found on paper)

- Euler's rule, checked: a black hole pays T S = 0.5 M c^2 (its energy grows as the square root of its bits); the cosmic horizon pays
  T S = 1.0 M_H c^2 (energy grows with its bits).
- A cell's marks are held by local machinery. Any system held locally, even with neighbors helping each other, has energy that grows in
  proportion to its size: degree 1 (checked: a chain of marks with neighbor bonds gives 1.05 at 10 marks, 1.0005 at 1,000, 1.0 at a million).
- So neighbor help cannot give the cell a black hole's half. The square-root growth is gravity's own property, not something a local system has.
- **Status: closed.** My proposal from last night was wrong, and it cost nothing to find out. The half, if it exists in the cell, has to come
  from somewhere else.

## 2. On fully held sites, the cell holds at the writer's own limit (the strongest thing tonight)

| | held error | holding energy |
|---|---|---|
| Writer alone, 1/(1 + D), two measured enzymes (D 47.4 to 80, median 63.7) | 0.0123 to 0.0207, median 0.0155 | 3.86 to 4.38 kT, median 4.15 |
| Cells, on the 26,800 sites every cell keeps fully methylated | 0.0136 to 0.0226, median 0.0165 | 3.77 to 4.28 kT, median 4.09 |
| Cells, genome-wide | 0.024 to 0.042, median 0.0314 | 3.13 to 3.71 kT, median 3.43 |

- Where a cell holds its pattern fully, its error sits inside the writer's measured range. Nothing extra is lost there.
- The "about twice the writer's error" that started the half-and-half idea is a genome-wide number. On fully held sites it is gone.
- This also explains U1 without any excuse: removing the eraser did not move the held error (ratio 0.961), because the sites we count
  sit mostly in fully held territory, where the reading is the writer's limit alone.
- **Status: candidate.** It rests on two enzyme measurements on short test DNA, and on summary numbers from the record (not per-cell values).

## 3. The middle may be a mix of two territories, not a per-site half

- Genome-wide held error = the writer's limit in fully held territory, plus extra loss in loosely held territory.
- The architecture line from last night, written out (exploration, threshold chosen by me, not physics):
  eps = 0.0222 + 0.0248 L, where L is the share held loosely. It explains about three quarters of the spread between cell types
  (residual SD 0.0019 against 0.0038).
- One loose end, stated plainly: the line at L = 0 reads 0.0222, not the writer's 0.0155. Either my 0.70 threshold misses some loose
  territory, or fully held territory is not purely the writer. A physics definition of "loose" is needed before the line can be derived.
- If this is right, 3.41 kT is not ln D - ln 2 by a law. It is the writer's 4.15 kT in held territory, diluted by the loose territory each
  cell's design carries. The near-match 3.46 against 3.41 would then be a coincidence of the average.
- **Status: candidate.** It replaces a premise we could not derive (r = 1) with two things we can measure from the sample itself.

## 4. What this does for the atlas-free instrument

- The writer's limit, 1/(1 + D), is a property of the enzyme, the same in every cell. Every sample's fully held sites should read it.
  That is a reference that needs no atlas.
- The loose share is read from the sample's own pattern. Together they give each sample its own expected reading.
- A sample that reads worse than the writer's limit on its own fully held sites is losing hold where a healthy cell cannot.
  That is the simplest possible instrument: one number per sample, against a constant of the enzyme.

## Questions for the whiteboard

1. Does "held territory reads the writer alone" match your picture? Where does the eraser live in it: only in the loose territory?
2. A fact for this one: the 0.70 threshold I used means a domain already at H = 0.88 bits per site, 88 percent of the ceiling
   (0.80 would be 0.72 bits, 0.90 would be 0.47 bits). So "loose" is territory near saturation, and the literature has long reported
   that these domains widen in tumors. What, in IAM's physics, is a loosely held domain? A region whose surface is close to saturation, or one where the writing is
   outrun? Your black hole picture may belong here, not at the single site.
3. If the middle is a mix of two territories, do we still want the half-and-half as the law, or does the law become
   "the writer's limit, diluted by the design"?

---

## Night shift, part 2 (written while the atlas diagnostics ran)

### The ceiling: GAPE Issue 002 and the Dennard wall say the same thing

- **GAPE 002 (pages 12-14), the saturation wall chart.** Every class and substrate had its own ceiling, A_max. The uncommitted designs
  (progenitor, adult stem) sat closest to the coin flip while healthy, so three of their five substrates saturated first. That is the same
  pattern as last night's architecture result, from different data and a different measure: red-cell progenitors carry the most loosely
  held territory (93 percent).
- **The methylation ceilings line up with the held error for the four classes we can match** (observation; four points, and both measure
  distance from the coin flip, so partly expected):

| class | held error (Loyfer) | holding | GAPE 002 methylation ceiling |
|---|---|---|---|
| terminal (neurons) | 0.024 | 3.71 kT | 1.294 |
| immune | 0.028 | 3.55 kT | 1.192 |
| secretory (epithelia) | 0.0315 | 3.43 kT | 1.186 |
| stromal | 0.034 | 3.35 kT | 1.159 |

  Caveat: GAPE 002 used the retired class floors; the ceiling pattern is kept as an observation, not as its A values.
- **The Dennard wall (book, "Landauer at the transistor gate"): "The Dennard wall is a Boltzmann wall."** A switch leaks as
  exp(-barrier/kT); the 60 mV/decade limit is (kT/q) ln 10. The cell's held error has the same form: eps = 1/(1 + e^E). On the backbone
  the barrier is the writer's discrimination, ln D = 4.15 kT.
- **Dark silicon is the chip's version of the loose share.** Once the power wall hit, chips could not run every transistor at once, and
  the design decides what share stays dark. Candidate translation (an idea, not a result): L is the genome's dark silicon, the share a
  design cannot afford to hold at nameplate. If so, L is set by the design's holding budget, which is your "design within thermodynamic
  limits".

### The nameplate: what physics fixes and what it does not

- **The numbers:** ln D = 4.15 kT (3.86 to 4.38 across the two measured enzymes) is 0.198 of the Mahaffey number (0.184 to 0.209).
  The genome-wide share is 0.163. In bits, the nameplate is 5.99 bits against M = 30.2 bits.
- **Derived:** a bound. The writer cannot discriminate by more than its drive, so ln D <= M. The nameplate sits well inside it.
- **Noticed, NOT claimed:** 5.99 bits is close to 6, and 0.198 is close to one fifth. Two enzyme measurements, spanning 5.6 to 6.3 bits;
  a near-round number with no reason behind it is exactly the 3.46-against-3.41 trap.
- **The book's own warning:** its coherence optimum (qubit chapter) has the form of an optimum, but where it sits "is set by the device
  constants, so it is a property of each device, not a universal number." The writer's nameplate may be the same kind of thing: like a
  transistor's threshold, a property of the device that physics bounds but does not fix.
- **Why that is still good news for the instrument:** a nameplate is measured once, on the enzyme in a test tube, and is the same for every
  person. That is an equipment rating, not an atlas. Deriving it would be the bonus, not the requirement.

### Questions for the whiteboard

1. Is the loose share the genome's dark silicon: the share a design cannot afford to hold at nameplate? What would set that budget?
2. Does the nameplate need to be derived, or is "measured once on the enzyme, the same for everyone" enough for the instrument?
3. The black-hole factor of 2: at the same radius a black hole is half as hot as the cosmic horizon, matching the held shares of one half
   and one. Is the cell's backbone (ratio 1 to the writer) the "horizon" reading and the genome-wide average the "black hole" reading?
