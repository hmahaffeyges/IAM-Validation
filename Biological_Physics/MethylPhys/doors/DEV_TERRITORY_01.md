# DEV-TERRITORY-01 — territory contrast: a self-taring reading, and a parameter-free prediction (design, 2026-10-13; NOT sealed)

**Why.** Every reading we have is vulnerable to the same thing: the spread between healthy people, plus plate and batch effects, plus
the cell mixture. Those all move a whole array together. A reading built as a RATIO between two kinds of site INSIDE one array
cancels whatever is common to both, because numerator and denominator see the same donor, the same plate and the same cell mixture.
The question is whether anything is left after that cancellation. The balance equation says there is, and says how much.

## The derivation (no free parameter)

A held site is lost with probability `f` over an interval and restored with probability `g`; the held error settles at

    eps = f / (f + g),        E_hold = kT ln(g / f)

Now let the cell's restore capacity fall by the same FRACTION everywhere - one energy pool, one shortfall, `g -> g(1 - d)`. Then

    eps' / eps = 1 / (1 - d (1 - eps))                    so      D eps  ~  eps (1 - eps) d

The absolute response is therefore `eps(1 - eps) d`, which is LARGER where the baseline error is larger (for eps < 0.5). So for two
site classes with measured healthy baselines `eps_fragile` and `eps_stable`, the ratio of their absolute responses to the SAME
shortfall is fixed by those baselines alone:

    D eps_fragile / D eps_stable  =  [eps_f (1 - eps_f)] / [eps_s (1 - eps_s)]

Nothing is fitted: the baselines are measured on healthy arrays, and the prediction is then a number. Note the direction carefully -
the RELATIVE rise is slightly larger at the stable (low-error) sites, `~ d(1 - eps)`, while the ABSOLUTE rise is larger at the
fragile sites. The reading must therefore be built on absolute disorder, not on each class's own relative change.

**So the reading is a DIFFERENCE, not a ratio, and the derivation forces that choice.** The contrast is `eps_fragile - eps_stable`,
whose response to a shortfall is also parameter-free:

    D (eps_f - eps_s)  ~  [eps_f (1 - eps_f) - eps_s (1 - eps_s)] d

A ratio `eps_f / eps_s` would be a worse instrument: because the relative rise is larger at the low-error sites, a shortfall moves
the ratio the other way and only weakly. The difference also cancels a plate-style additive offset exactly, which is the property the
whole design is for. (An earlier draft of this note described the reading as a ratio; that was wrong, found while writing the test.)

**What is assumed, and is therefore what gets tested.** That the shortfall is common to all sites - one pool of restore capacity
serving every territory. If instead each territory has its own supply, the measured response ratio will not match the predicted one.
That is the interesting outcome either way: it would say the cell budgets maintenance locally rather than globally.

## The site classes, fixed before any array is read

From the array MANIFEST, never from our own readings (which would be circular):

- **stable territory:** probes annotated as being in a CpG island, in a promoter.
- **fragile territory:** probes annotated as open sea (no island, no shore, no shelf).

Both classes are restricted to probes measured in every array of the set, and to the atlas cell types present, so composition enters
both classes alike. The classes are written into the committed script before any reading.

## Before any real array: the synthetic tests

1. **Does the predicted response ratio hold?** Simulate both territories at their measured healthy baselines, apply a common
   proportional shortfall, and compare the measured response ratio against `[eps_f(1-eps_f)] / [eps_s(1-eps_s)]`.
   **Bar, written now:** within 10 % of the prediction at every shortfall from 2 % to 20 %.
2. **Does the contrast really cancel a nuisance?** Apply, with NO shortfall: a plate-style offset (every probe shifted), a
   composition change (cell fractions moved), and measurement noise. **Bar, written now:** the contrast moves by less than a third of
   what a 2 % shortfall moves it, for each nuisance separately.
3. **Is it blind where it should be?** Apply a UNIFORM loss across both territories. **Bar:** the contrast does NOT move
   (this is the instrument's declared blind spot, and the test must confirm it rather than hide it).

A reading is taken on real arrays only after all three are met, and the real-array step is a separate sealed note.

## FALSIFIER (written before any data is read)

If, under a condition known to cost the cell maintenance energy, the measured response ratio between the two territories differs from
`[eps_f(1-eps_f)] / [eps_s(1-eps_s)]` by more than 25 % (equivalently, the contrast's slope departs from the predicted
`[eps_f(1-eps_f) - eps_s(1-eps_s)]` by more than 25 %), the balance account of the held error is wrong as applied to the cell.
**Scope: medium** - it would break our account of how maintenance is budgeted in a cell, not the law itself, unless the deviation also
violates the floor.

**In plain words.** Blood and tissue readings are all vulnerable to the same nuisance: people differ, plates differ, and the cell
mixture differs. But if we compare two kinds of place WITHIN one sample - the parts of the genome that are easy to maintain against
the parts that are hard - everything common to the whole sample cancels out. The physics then says something sharp: if the cell is
short of maintenance energy, the hard places should degrade more than the easy ones, by a factor we can calculate in advance from
healthy samples alone, with nothing fitted. If that factor is wrong, our account of how a cell budgets its upkeep is wrong, and we
would rather find that out than not. **What it would take:** three tests on made-up data (does the predicted factor hold, does the
comparison really cancel plate and mixture effects, and is it properly blind to a loss that hits both places equally), then one
reading of real arrays.

**Synthetic test:** DEV_TERRITORY_01/territory_synth_01.py, DEV_TERRITORY_01/territory_synth_01_summary.txt

## Outcome of the three tests, run once (2026-10-13): 1 and 3 MET, 2 NOT MET

`territory_synth_01.py`, outputs `territory_synth_01_output.csv` and `territory_synth_01_summary.txt`. Baselines stable 0.0120,
fragile 0.0520; 200 people per condition.

**Test 1, the parameter-free prediction: MET, and closely.** Predicted response ratio from the baselines alone: **4.158**.
Measured 4.171 / 4.199 / 4.126 / 4.122 at shortfalls of 2 / 5 / 10 / 20 % - within **1 %** of prediction at every shortfall, with
nothing fitted. The balance equation's `D eps ~ eps(1-eps) d` holds in the regime the instrument would work in.

**Test 3, the declared blind spot: MET.** A uniform absolute loss in both territories moves the contrast by 0.001 x10^-3, which is
nothing. The instrument is blind exactly where the design says it is.

**Test 2, nuisance cancellation: NOT MET**, and the two causes are real rather than coding errors. A 2 % shortfall moves the contrast
by 0.772 x10^-3; the bar asked every nuisance to move it by less than a third of that.

| nuisance | contrast moves | verdict |
|---|---|---|
| plate offset +0.02 | 0.006 x10^-3 | cancels, as designed |
| plate offset -0.02 | 8.121 x10^-3 | FAILS |
| composition +10 % | 3.999 x10^-3 | FAILS |
| composition -10 % | 4.005 x10^-3 | FAILS |
| measurement noise x3 | 1.022 x10^-3 | FAILS |

**Cause 1: the ceiling at beta = 1 makes an offset and noise one-sided.** A held site sits near beta = 1, so an offset or a noise
excursion that would push it above 1 is truncated. The positive offset cancels exactly (0.006 x10^-3, as the derivation says it
should); the NEGATIVE offset, which pushes toward the boundary, does not. The same truncation is why tripling the noise moves the
contrast at all: at nearly-fully-methylated sites, measurement error is effectively one-sided, and the stable territory - which sits
closest to the boundary - has the most of its apparent error contributed by that floor. This is a property of the measurement, not of
the simulation: real array beta values are bounded in exactly the same way.

**Cause 2: a difference cancels an ADDITIVE nuisance and cannot cancel a MULTIPLICATIVE one.** A composition change scales the
apparent error in both territories, so it scales their difference: the move is `comp x (eps_f - eps_s)` = 0.1 x 0.04 = 4 x10^-3,
which is what was measured, and about five times what a 2 % shortfall produces. No choice of site classes fixes this; the arithmetic
of a difference is the problem. The normalized form `(eps_f - eps_s) / (eps_f + eps_s)` cancels a multiplicative nuisance exactly and
reintroduces sensitivity to an additive one - so no single statistic cancels both.

**What this rules out and what it leaves.** The territory contrast is not usable as a stand-alone reading on raw array betas. What
survives is stronger than it sounds: test 1 established a parameter-free prediction that held to 1 %, so the quantity is real; the
problem is entirely in how the two nuisances enter. The next design (DEV-TERRITORY-02, not written) must therefore:
1. report BOTH the difference and the normalized difference, since one cancels each nuisance type, and require the predicted pattern
   in both - a two-number signature is harder to counterfeit than either number alone;
2. estimate the measurement floor from the data (the apparent error at sites known to be fully held) and subtract it before the
   contrast is formed, so the boundary effect is measured rather than inherited;
3. keep composition handled where it already is - by the mixture-predicted denominator the chain's reading uses - rather than hoping
   a difference removes it.

**In plain words.** The physics part worked, and better than expected: we predicted in advance, from healthy samples alone and with
nothing fitted, how much more the hard-to-maintain parts of the genome should degrade than the easy parts, and the simulation matched
that to within one percent. What failed was the practical claim. We hoped comparing two places inside one sample would cancel the
usual nuisances. It cancels one kind - an offset that adds the same amount everywhere - and cannot cancel the other kind, one that
scales everything proportionally, like a change in which cells are present. There is also a ceiling effect: these sites sit so close
to fully switched-on that noise can only push them one way, which inflates the quieter territory. **What it would take:** report two
versions of the comparison instead of one, measure the ceiling effect from the data rather than ignoring it, and keep handling the
cell mixture the way the main reading already does.

**Not a milestone:** a design whose physics bar was met and whose nuisance bar was not; nothing read on real arrays, nothing sealed.
