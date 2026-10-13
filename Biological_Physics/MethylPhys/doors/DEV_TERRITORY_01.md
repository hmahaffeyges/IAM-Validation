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

**Not a milestone:** a design with a derived prediction and three unwritten tests; nothing read, nothing sealed.
