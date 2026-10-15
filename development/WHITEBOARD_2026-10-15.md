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

---

## Part 3, with coffee: the efficiency premise (author, 2026-10-15)

**The premise, in the author's words:** every cell design is the most efficient way to operate; "it only uses energy when it needs to."
The substrate (chromatin) was chosen long ago, with "billions of years of experience." Status: an ASSUMPTION, based on the pattern across
all four domains. Stated as such wherever it is used.

**What it gives:**
1. The design holds firm exactly what it uses, and lets the rest run loose. So the loose share L = 1 - U, where U is the share of the
   genome the design uses. Physics fixes the backbone end (the writer's nameplate); the loose end is not yet derived (see Open, below); the design fixes U.
2. Support already in hand: the egg holds methylation almost only in the genes it reads; loosely held domains sit on the lamina, in the
   inactive compartment; neurons (broad gene use) hold 73-80 percent firm; red-cell progenitors (shutting the genome down) hold 7 percent.
3. **Toward an instrument without an atlas:** a sample's own L can be read from the sample itself (the share of its genome held loosely).
   Its expected held error then follows from its own line, (1 - L) x backbone + L x loose, and the reading is how far it sits from its
   own line. No reference cells needed for the position, only the two ends.
4. **Cancer, the known fact this must survive:** tumors widen their loose domains. On this picture that alone moves a tumor along its
   own line, not off it. The cancer signal would be the backbone itself reading worse than the nameplate (step 4, tumor pairs).

**Open, and next on the board:**
- The backbone end is derived (the writer's limit, 1/(1 + D)). The loose end is NOT: measured cells top out near 0.042, far from a
  coin flip, so the loose end is not the ceiling itself. What holds the loose territory where it sits?
- Last night's line missed the backbone at L = 0 (0.0222 against 0.0155); the 0.70 cut for "loose" was mine, not physics.

## Part 4: the writer on overdrive, and what the meter can read (author and Marcel)

**Correction to part 3, item 4.** The cancer signal is not only the methylated backbone failing. The author's picture: cancer moves right
because the writer runs on overdrive, writing selected, non-random marks. On the gauge both directions move a held site toward the coin
flip: a held-methylated site that loses its mark, and a held-unmethylated site (an island) that gains one. The overdrive writer is the
island side. So the cancer signal is the held state failing on EITHER side, read against the same nameplate. The held error as we count it
today (a lost mark between two methylated neighbors) reads only the first side; the island side needs its mirror count (a gained mark
between two unmethylated neighbors).

**The efficiency premise gives cancer a definition:** a healthy design writes only where it uses the genome. A tumor spends writing where
the design holds empty (selected targets, not random). Cancer is inefficient writing.

**The loose end: the meter, not the cell (the author's Dennard question).** The held error counts only molecules whose neighbors are
held methylated. In loosely held territory few molecules qualify, so a cell's held error is an average weighted toward the territory the
meter can read. The 0.042 at the top of the table is a whole-cell average, not the level of the loose territory itself. Saturation of the
reading there is the instrument's read limit, as the author put it, not the ceiling.

**(a) or (b): the author leans to (a).** Holding the ability to write without writing is not the same kind of efficient; progenitors and
the egg stay unwritten until told what to become. So the loose territory is not held part-time; it is not held at all, and sits where the
substrate leaves it. Prediction of (a): read directly, the loose territory sits at the same level in every cell type.

## Part 5: the math sheet and two decisions (author, 2026-10-15)

Math sheet (status): ceiling H = 1 bit (derived); floor, Landauer (derived); holding E = ln[(1 - eps)/eps] (derived, two-state Boltzmann);
gauge reads H(eps) (definition); backbone eps_W = 1/(1 + D), E = ln D = 4.15 kT, H = 0.115 bits (derived from the writer, D measured on the
enzyme); genome-wide middle eps 0.0314, E 3.43 kT, H 0.201 bits (measured); drive M = 20.94, share held phi = E/M (backbone 0.198, middle
0.163); a cell's reading = (1 - L) x backbone + L x loose (bookkeeping); loose end OPEN.

**Pooling is not neutral.** H is concave, so the disorder of a pooled reading exceeds the average of its parts' disorder: +3.6 percent for
L 0.5 with loose eps 0.05, +9.0 percent with loose eps 0.10. Part of a cell's gauge position today is an averaging artifact.

**Decision 1 (author): one reading per instrument.** Ten scores per cell on a blood panel would overwhelm the reader. So each instrument
reads its territories separately on the gauge and combines them into ONE number by their shares: A = (1 - L) x A_backbone + L x A_loose.
That removes the pooling artifact and still gives one reading.

**Decision 2 (author): the loose end waits.** Row 9 is not used until the first tests show what they give us; they may change the question
across the whole IAM table, and may make the loose end easier to derive.

## Part 6: down to the bit (author's question, computed 2026-10-14 late evening Pacific)

**One CpG on one strand of one DNA copy = one bit of capacity** (methylated or not, as arrays and bisulfite read it). Counted from our own
hg19 sequence: 28,217,009 CpG sites per haploid genome (autosomes 26,752,702, X 1,246,401, Y 217,906).

| per cell | female | male |
|---|---|---|
| CpG sites (diploid) | 55,998,206 | 54,969,711 |
| methyl positions, both strands (bits of capacity) | 111,996,412 (14.0 MB) | 109,939,422 (13.7 MB) |
| held, at the genome-wide held error 0.0314 (1 - H per position) | 89.4 Mbit (11.2 MB) | 87.8 Mbit (11.0 MB) |
| held, if every position read at the writer limit 0.0155 | 99.1 Mbit (12.4 MB) | 97.3 Mbit (12.2 MB) |
| Landauer floor to write every position once, 310 K | 3.3e-13 J, about 3.7 million ATP | same within 2 percent |

One ATP (M = 20.94) carries 30.2 bits' worth of Landauer cost.

**The cosmic horizon** (Hubble horizon, bits = area / (4 Planck areas ln 2); the book's two H0; shape of the expansion taken as flat
matter plus Lambda, Omega_m = 0.3153, NOT yet IAM's own E(a)):

| | photon H0 67.16 | matter H0 72.26 |
|---|---|---|
| bits today | 3.29e122 | 2.84e122 |
| bit rate today | 6.8e104 bits per second | 6.3e104 bits per second |
| growth per year | 6.5e-11 of the total | 7.0e-11 |
| bits at recombination (z 1100) | 5.9e113 | 5.1e113 |

The bit rate peaked at z = 0.63 and is 0.68 of its peak today. Checked for four matter densities: the peak always sits exactly where the
expansion turns from slowing to speeding up (q = 0), so it is a property of the expansion shape, not a new result. In IAM's words: the
horizon's record grew fastest at the turn to acceleration. With IAM's own expansion curve the turn, and so the peak, sits where IAM's q = 0 is.

**A cell's bit rate: open.** A single sample gives the balance (held error = loss over loss plus restore), a ratio of two rates, not
the rates. Bits per second needs a time course of the same cells. Idea for the manifest, not a test.

## Part 7: the loose end and the island side, from data we hold (2026-10-15, early morning Pacific)

**The loose end, read off the line through 56 healthy cell types.** Held error against loose share (DEV-ARCH-01 table), robust line
(Theil-Sen), nothing tuned: at L = 0 (a cell with no loose territory) 0.0212 (bootstrap 95 % 0.019-0.024); at L = 1 0.0497 (0.046-0.053);
r 0.87. Leaving out any one cell class moves the firm end only within 0.019-0.024. So the loose territory reads ONE level in every cell
type, as the author predicted for option (a), about 0.05 as the meter sees it (only mostly-methylated molecules are counted, so this is
the meter's view of the loose territory, not yet its own level). The firm end 0.0212 equals an enzyme with D = 46, at the slow edge of the
published writer range (47.4-80), not its median.

**The island side against that firm end.** The island gain error (isolated gain between unmethylated neighbors, PROC-CHANNEL-01) has
median 0.0124. If the writer's accuracy is the D = 46 the line reads, every one of the 56 cells holds its islands BETTER than the writer
alone, by a factor 1 + r = 1.3 to 2.3 (median 1.73) in the odds. Two readings, one asymmetry: the 1s and the 0s differ by about 1.7 in odds.
 - Reading A: the writer sets the firm end; something extra holds the 0s, and the eraser (which cleans wrongly written marks) is the
   candidate. The eraser test found the eraser does not affect the 1s; this reading puts it on the 0s instead.
 - Reading B: the writer is the accurate one (D near 80) and sets the islands; something extra loses the 1s.

**Prediction, committed before the read (this push).** Read with PROC-CHANNEL-01's own counter (reads with at least 6 CpGs and at most
20 % methylated; an isolated methylated site between two unmethylated ones, per CpG) on the eraser-removed and control molecules already in
our S3 (DEV-ERASE-01 primary units U1-U5):
 - Reading A predicts the island gain error RISES with the eraser removed, toward the firm end: knockout over control about 1.7.
 - Reading B predicts it does NOT change (ratio near 1).
Read once, reported whatever it says. Known biology to keep in view: TET loss is known to raise methylation at island edges and enhancers,
so a rise is consistent with the literature; the number (about 1.7 against about 1) is what discriminates the two readings. Development use
of spent data; not a sealed test.
