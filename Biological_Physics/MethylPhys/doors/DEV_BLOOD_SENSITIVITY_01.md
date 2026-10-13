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
| 0.40 | +2.8 % | +5.3 % |
| 0.55 | +3.8 % | +7.5 % |
| 0.70 | +4.8 % | +9.7 % |

**Dilution costs much less than expected.** A 1 % change confined to a 40 % component still moves the whole-blood reading by 2.8 %,
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

**Synthetic test:** DEV_BLOOD_SENSITIVITY_01/blood_sensitivity_01.py, DEV_BLOOD_SENSITIVITY_01/blood_sensitivity_01_summary.txt

**Not a milestone:** a measured property of the reading, with its detection limit not yet established.
