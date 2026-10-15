# Ideas: bit rates, and the "Computer Science of Reality" paper (author, 2026-10-14 and 2026-10-15)

_Kept so nothing is lost. These are ideas, not results and not tests. The derivations come first (author); this paper is written after
the four-domain table (FOUR_DOMAINS.md) is as complete as we can make it. Computed numbers are in development/WHITEBOARD_2026-10-15.md part 6._

## Numbers we already have (computed, not quoted)

- One CpG on one strand of one DNA copy is one bit of capacity. hg19: 28,217,009 CpG sites per haploid genome. A diploid cell has about
  112 million methyl positions (about 14 MB of capacity) and holds about 89 million bits (about 11 MB) at the healthy held error.
- Held information per site is 1 - H(eps), less than one bit.
- Mahaffey numbers are in units of kT, not bits. One kT is 1/ln2 = 1.443 Landauer bits. So M = 20.94 kT per ATP is 30.2 Landauer bits of
  drive per ATP. The writer's holding 4.15 kT is about 6.0 bits; the genome-wide middle 3.43 kT is about 4.9 bits. The held share
  phi = 4.15/20.94 = 0.198 has no units, so it is the same in bits: about one fifth of each ATP's push stays in the held mark.
- Black hole: 2 ln2 kT per bit (2 in Landauer units), half of it held. Cosmic horizon: ln2 kT per bit (1 in Landauer units), all held.
- Hubble horizon: about 3 x 10^122 bits; grows about 7 x 10^104 bits per second today; about 6 x 10^113 bits at recombination. With the
  standard expansion shape the bit rate peaks exactly at the turn to acceleration (z 0.63): an identity of that shape, not a discovery.
  Not yet redone with IAM's own E(a).

## Questions the author asked, and the short answer so far

- **Does binary store only the 1s?** No. A computer holds both: a 0 is a definite state (an empty capacitor, a reset cell), not an absence.
  What costs energy (Landauer) is erasing, resetting a bit to a known state, whichever value it holds. Only some file formats list just
  the 1s (sparse or run-length storage). In the cell the same holds: an unmethylated island is a held 0. Our island-side result
  (2026-10-15) says the cell holds its 0s about as tightly as its 1s, near the writer's limit.
- **Is 20.94 bits?** It is kT. In bits it is 30.2 per ATP.
- **Could the horizon bit rate settle the Hubble tension?** By itself no: the bit rate is H(z) written in different units, so it carries
  the same information. What can settle it is IAM's own prediction that photon-based and matter-based measurements give different H0
  (67.16 and 72.26 in the book), checked against independent probes such as gravitational-wave sirens. The bit-rate form is a way to
  explain that split, era by era, in units anyone understands.
- **A cell's bit rate:** one sample gives the balance (loss over loss plus restore), not the rates. Rates need the same cells read over
  time (manifest PHY-044). No such data is in our S3 (checked 2026-10-15: a file-name search of downloads/, results/ and archive/ for time-course terms had one hit, results/BOXRUN1/A/A_bar_replicates.csv, a replicates table matched on 'repli', not a time course).

## The paper: "Computer Science of Reality" (working title)

Not that the universe is a computer or a simulation: how the computation happens in each of the four domains, side by side with the parts
of a modern computer, so a reader who knows computers can follow black holes, qubits, chips and cells. A bit is a bit is a bit.

Analogy list to work through (author's, in his order of excitement):
- black hole and cancer cell: a surface that has filled, writing outrunning the room to write;
- the genome and a black hole as two hard drives: the main PC and an external drive;
- the horizon as storage, its growth as a write rate in bits per second, compared with the speed of light and the time scale;
- gate error and leakage (the two ends of every gauge) as the error budget of a processor;
- RAM against long-term storage: short-term memory and language as different storage areas with different capacities;
- processing rates: cells, the brain, chips, qubits, the horizon, each in bits per second.

## Further ideas (speculative, for later)

- All of IAM converted to bits: a software model that runs on real bit rates and storage sizes (expansion, halos, stars, black holes,
  gravity as geometry), to get more out of the physics.
- Bit rates with a time course: the sliding timescale applied to living cells for accurate aging; time near a black hole.
- A per-cell bit rate as a meter sensitive enough to catch more than cancer.

## Added 2026-10-15 (author): the paper's title, storage by era, gravity and time (OTH-049, PHY-053, PHY-054)

**Title (OTH-049):** *Computer Science of Reality: The Bit-Rate of Cosmic Expansion from the Singularity Through Today, Black Hole Hard
Drives, and Cosmic Storage Space*: the computer geek edition, after the plain-language series, both sponsored on the page.

**Storage by era (PHY-053).** The horizon's bit count is a stock, so summing its bit rate from the start gives back today's count; the
new number is the count at each era (CMB, first stars, the turn to acceleration, today). The night sky as old bits: light from far stars
left them long ago, so we read old information arriving now; "stored on the horizon" is the IAM reading.

**Gravity, time and the bit rate (PHY-054, candidate).** The author: time is the one-way writing to the horizon, and gravity is
proportional to the bit rate. Known physics to set it against: clocks run slower deeper in gravity as seen from far (so a bit-rate
reading of time predicts a lower bit rate there, seen from far); the Tolman relation makes a surface deeper in gravity locally hotter by
the same factor, so the Landauer price per bit is higher locally and identical everywhere as seen from far. Prior work, credited for
exactly what it did: Jacobson 1995 derived Einstein's equations from horizon thermodynamics; Verlinde 2011 argued gravity is entropic.

**Storage by era, computed 2026-10-15 (PHY-053; standard expansion shape with Omega_m 0.3153, radiation 9.1e-5, not yet IAM's own E(a)):**

| era | z | bits on the horizon (H0 67.16 / 72.26) | share of today | horizon temperature |
|---|---|---|---|---|
| CMB released | 1090 | 6.1e113 / 5.3e113 | 2 in a billion | 6e-26 K |
| first stars | 20 | 1.1e119 / 9.7e118 | 3 in 10,000 | 1.4e-28 K |
| turn to acceleration | 0.63 | 1.6e122 / 1.4e122 | 49 percent | 4e-30 K |
| today | 0 | 3.3e122 / 2.8e122 | 1 | 2.6e-30 K |

Half of today's storage was written after the turn to acceleration. The shares depend only on the expansion's shape, not on H0, so the
two H0 values change every total by the same factor (0.864) and the storage history alone cannot choose between them.
