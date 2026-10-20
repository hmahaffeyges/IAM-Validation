# DEV-BLOOD-SENSITIVITY-01 — can whole blood on arrays carry a tissue-sized signal? (development, 2026-10-13; NOT sealed)

**The author's question.** If the atlas can say which cells are in a tube of blood, and one cell type is abundant, can the array
reading see a change in that cell type? Or is that the kind of group-level illusion the retired class era produced?

**Why it is not that illusion, and what the real limit is.** An array averages every cell in the tube before anyone reads it, so a
mixture's disorder already contains the mixing itself - two cell types that differ at a site give an intermediate value, which looks
like disorder. Knowing the fractions does not let you undo that at the molecule level. The chain handles it the only way an average
allows: it divides by the disorder the estimated mixture PREDICTS (`dev_comp_methods.met_a`, mean H(beta) / mean H(e), e = sum f_g mu_g),
which answers a one-directional question - is there more disorder than the mixture explains - rather than isolating one cell type.
That is an identifiability limit, not a fitted-class error: a within-molecule quantity cannot be recovered from a between-cell average.

**What was measured (`blood_sensitivity_01.py`, run twice; v1 kept as `*_v1_DILUTION_ONLY.*`).** Synthetic whole blood from the atlas
posteriors for six blood cell types, neutrophil sites chosen by a rule fixed before any planting, pattern loss planted in the
neutrophil component ONLY, read by the chain's own function with the TRUE fractions (the best case).

| neutrophil share | planted 1 % drop moves the reading by | planted 2 % |
|---|---|---|
| 0.40 | +2.64 % | +5.15 % |
| 0.55 | +3.79 % | +7.45 % |
| 0.70 | +5.13 % | +10.10 % |

(Computed in `blood_sensitivity_01_output.csv`, column `leverage` = median planted reading / median healthy reading - 1. An earlier version of this table was typed by hand from the printed medians and was wrong by 0.2 to 0.4 points at the 0.40 and 0.70 shares; the table is now generated from that column.)

**Dilution costs much less than expected.** A 1 % change confined to a 40 % component still moves the whole-blood reading by 2.6 %,
because the predicted-mixture denominator absorbs the mixing and what remains is close to the component's own change. The reading has
leverage.

**What this does NOT establish, stated plainly.** It does not give a detection limit. Both versions produced a healthy band of about
+-0.5 %, which is unrealistically tight: v1 gave every simulated person identical cell-type profiles, and v2's attempt to fix that
drew person-level profiles from the atlas posterior's `sd` - which is the uncertainty of the MEAN profile from a few donors, not the
biological spread between people, and at the low-noise sites the rule selects it is about 0.003. So the simulation measures leverage
and cannot measure the limit. Any claim of a detectable percentage from this note alone would be wrong.

**What it would take to finish.** Compute this same chain function on REAL healthy whole-blood arrays, measure the spread between
people, and combine it with the leverage measured here. Declared in advance: the real healthy whole bloods in hand (GSE184269,
GSE315366) were spent as the C-score band's held-out test, so reading them for a spread is development use and disqualifies them as a
held-out test for any blood reading later. A separate healthy whole-blood set should be reserved for that.

**In plain words.** Blood is a mixture, and an array mixes all the cells together before we see them, so some of what looks like
damage is really just the mixing. Our reading handles that by predicting how much disorder the mixture should have and asking whether
there is more. We tested on made-up blood whether that still works when the cell we care about is only part of the tube: it does, and
better than expected - a 1 % change in a cell type making up half the blood moves the reading by about 4 %. What we have NOT shown is
whether that is bigger than the ordinary differences between healthy people, because our made-up people were too alike. **What it
would take:** read real healthy blood samples with the same function and measure how much healthy people actually differ.

## Part 2, the real healthy spread, measured (2026-10-13): per person NO, per group YES

**Prior reading:** GSE184269 was read once as the C-score band's held-out test and is in `CANON/data_spent.csv`. This part is
DEVELOPMENT use - a descriptive spread with no bar and no verdict - and it disqualifies those arrays as a held-out test for any blood
reading later.

Read with the chain's own stages, unchanged (`blood_spread_01.py`; `conductor_v3.stage_a_composition` then
`conductor_v3.stage_m_blood`), on the 24 healthy whole bloods of that laboratory. 23 read; **1 withheld by the chain** for site
coverage (5,325 of 6,000 neutrophil sites, 5,400 required) - the chain refusing, not us dropping it.

| | |
|---|---|
| median reading (untared) | 0.9463 |
| spread between healthy people | SD 0.0601, **relative SD 6.36 %** |
| range / 5th-95th | 0.8674 to 1.1022 / 0.8863 to 1.0884 |
| 95th percentile above the median | **+15.02 %** |
| the chain's own leverage | **0.01978 in A per 1 % pattern loss** = 2.09 % of the median |
| neutrophil share in these arrays | median 0.469 (0.263 to 0.665) |
| entropy-ceiling flag | recorded on all 23; **none past the ceiling** |

**Per person: about a 7.2 % pattern loss is needed to clear the healthy 95th percentile** (7.9 % to clear the highest healthy array
read here). The tissue cancers we have read show changes of roughly 2 to 7 %. So a tissue-sized change in blood does NOT separate one
person from healthy on arrays - the spread between healthy people is simply larger than the signal.

**Per group it does, and comfortably.** A group comparison faces the spread divided by the square root of the number of people:

| group size | two standard errors | mean loss that separates the group from healthy |
|---|---|---|
| 20 | 2.84 % of the median | **1.36 %** |
| 50 | 1.80 % | **0.86 %** |
| 100 | 1.27 % | **0.61 %** |

**Correction to part 1.** The simulation's leverage (3.79 % per 1 % drop at a 0.55 neutrophil share) is about 1.8 times the chain's
own measured leverage (2.09 % of the median at a 0.469 share). The simulation used 3,000 atlas-selected sites and atlas posterior
profiles; the chain uses its own 6,000 neutrophil sites and the Salas purified EPIC profiles. **The chain's number is the one to
quote**, and part 1's leverage figures should be read as an approximation with different sites and profiles.

**One scope limit on this spread.** These readings are UNTARED, and that laboratory ran one array per slide, so a same-run tare could
not be formed - only a series-level tare, which removes a global offset and not a per-plate one. A properly tared reading may face a
smaller spread than 6.36 %, and measuring that needs a healthy set with several arrays per slide. Until that is measured, 7.2 % is the
honest per-person figure.

**In plain words.** We measured how much healthy people differ from one another on this blood reading, using real arrays and the same
code a researcher would run. They differ by about 6 % - which is larger than the changes we see in tumor tissue. So this reading
cannot tell one healthy person from one early-cancer person in a tube of blood: the ordinary variation between people is bigger than
the signal. But comparing GROUPS works well: with fifty people on each side, a change of under 1 % would show. That is a real result
and a real limit at once - it says group studies in blood are within reach now, and that per-person blood detection needs either a
tighter reading (the tare, or a self-taring contrast) or the molecule-by-molecule route, which is sequencing.

**Synthetic test:** DEV_BLOOD_SENSITIVITY_01/blood_sensitivity_01.py, DEV_BLOOD_SENSITIVITY_01/blood_sensitivity_01_summary.txt

**Not a milestone:** a measured property of the reading, with its detection limit not yet established.

## Correction, 2026-10-20: part 2 did not read whole blood the way the chain does

Part 2 (`blood_spread_01.py`) called the chain's composition and whole-blood stages directly and skipped Stage T step 1, self-tare II (probe-design
mapping), which `conductor_v3.run_neutrophil` applies before reading whole blood. So its spread is not the chain's reading, and the sentence above that
"a researcher pressing the button gets this" was wrong. The chain did read the same 23 healthy whole bloods, unchanged, in the C-score band test
(`CSCORE_COMMISSIONING/band_test_GSE184269_rows.csv`); the two readings of the same arrays correlate at only 0.34. Recomputed from the chain's own
reading by `blood_spread_02_chain.py` (output `blood_spread_02_chain_summary.txt`), arithmetic over committed chain outputs only:

| | part 2 (withdrawn) | the chain's reading |
|---|---|---|
| spread between healthy people (relative SD) | 6.36 % | **2.01 %** |
| 95th percentile above the median | +15.02 % | **+3.33 %** |
| leverage per 1 % pattern loss | 2.09 % | **2.12 %** (chain v4, one healthy whole blood) |
| loss that clears the healthy 95th percentile, one person | about 7.2 % | **about 1.57 %** |
| mean loss that separates a group of 50 | 0.86 % | **about 0.27 %** |

The leverage was right; the spread was not. Part 2's conclusion that one person cannot be read in whole blood is withdrawn: per-person detection in
whole blood is OPEN, to be decided by a sealed test across laboratories. These remain development numbers on arrays already spent (one laboratory,
one leverage array; no bar, no verdict).

