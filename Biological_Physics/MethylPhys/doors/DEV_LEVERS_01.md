# DEV-LEVERS-01 — every lever on the methylation channel, with its caveat (audit, 2026-10-13; nothing read, no pass value written)

The quantum line (QAPE) enumerated every platform with its own decoherence channel and caveat before any platform was
scored. The cell has never had that audit: each lever was taken up one at a time. This note is that audit. **It is a
design note.** Nothing here is a result, no bar is written, and every row says plainly whether a *parameter-free*
prediction is possible, since that is the only kind of test a physicist cannot dismiss as fitting.

## The levers

| Lever | What it changes in the physics | Caveat | Parameter-free prediction possible? | Status |
|---|---|---|---|---|
| Temperature | kT sets the floor | mammals are all 36-39 C, so this needs birds (40-42), fish (ambient), hibernators (5-37), fever, or culture at 33/37/39 | **yes** - a derived ratio between two temperatures | designed, never run (`DEV_FLOOR_HEIGHT_02`) |
| Methyl supply (SAM, methionine, folate, B12) | the restore rate g, and so the balance directly | starvation also stops division, which only changes how fast the balance is reached; this is why methionine parked | partly - the direction is derived, the size needs the division rate | one mouse result met its committed window (`DEV_SAM_LEVER_01`); `DEV_METHIONINE_01` parked |
| Division rate | how fast the balance is reached, about 1/(f+g) intervals - not the balance itself (`CANON/foundation.json`, entry `balance`) | division rate travels with age, tissue and culture | **yes** - cells that never divide hold the balance by upkeep alone, which is a derived way | neuron rule sealed (`DEV_IAMA_NEURON_01`); passage series in hand (GSE268145) |
| Write-enzyme level | fewer write heads, more copy error | inhibitors are toxic and change division rate | no - dose response is empirical | decitabine gave no result at the decisive dose (`DEV_LINK_IAMA_METAA_02_OUTCOME`) |
| **Erase machinery (TET)** | the reverse channel, on its own | needs oxygen and alpha-ketoglutarate, so it is tangled with metabolism | **yes, if the three-channel split below is right** | **never tested** |
| **Oxygen and redox** | TET activity depends on O2; tumors are hypoxic | hypoxia changes much else at the same time | no | **never considered** |
| **IDH-mutant tumors** | 2-hydroxyglutarate blocks TET: a natural erase-channel knockout, in patients, in public data; mostly gliomas | the mutation shifts metabolism broadly | **yes** - a sign prediction: impairing only erasure must move the reading in a stated direction | **never considered** |
| **Spontaneous chemical damage** | 5-methylcytosine deaminates faster than cytosine, by a thermally activated rate | mimics loss of order with no enzyme involved | **yes, and it is the key to the temperature test**: the floor scales with kT while a chemical rate scales exponentially with it, so the two separate by their functional form | **never separated from enzymatic error** |
| 3D position in the nucleus | the regions that lose large blocks in cancer are the ones against the nuclear envelope, late-replicating and gene-poor | depends on other laboratories' assays | no, but it changes what the sky map measures | partly in the sky work (`DEV_SKY_02`) |
| **Imprinted regions** | two alleles in one cell, same temperature, different required state: a built-in control | few regions, patchy array coverage | no - a control, not a lever | **never used as a control** |

## Three channels, not one

The copy head reads the parental strand and writes the daughter: logically this is copying, which is cheap in principle.
The de novo head writes with no template: it *sets* a state. The erase channel resets a site to unmethylated, and
resetting to a known state is erasure in Landauer's sense - the operation that must dissipate. The floor may therefore
apply unequally to the three. **That is a testable claim and the strongest one in this note:** the erase direction
carries the cost, the copy direction need not.

Energy bookkeeping follows the same split. At the site, the write is driven by SAM, which is made at the cost of ATP's
whole triphosphate plus a methionine; the copy head does not hydrolyze ATP per site. Erasure excises the oxidized base,
nicks the strand and religates it, which costs ATP and transiently breaks DNA. The asymmetry is by mechanism, not by
direction.

## Corrected against the balance derivation (2026-10-13; the author caught this twice)

The lever table first said the cost was paid per copy [retired wording kept for the record], which assumes replication supplies the only operations. The
author's objection - a neuron never divides and its writing and erasing go on anyway - is already settled in the book, and
the settled answer is stronger than the per-operation patch written here earlier today. From the book's section *The held
state is a balance of restore and loss* (2026-10-11, DERIVED): over any interval, a held site is lost with probability `f`
and a lost one restored with probability `g`, so the held error settles where losses equal restores,

  **eps = f / (f + g)**, with **E_hold = kT ln(g/f)**.

**The interval need not be a division.** In a dividing cell much of the loss happens as the DNA is copied; in a cell that
has stopped dividing, loss and restore are upkeep - marks removed and written again in place. The equation is the same in
both. The held error is set by the *ratio* of restore to loss, not by how many copies have been made. What division sets is
only how fast the balance is reached: about `1/(f+g)` intervals.

So the division-rate row in the table above is wrong as written, and is corrected here: **division rate sets the time to
reach the balance, not the balance itself.**

**The prediction, from the book, and what would falsify it.** Once upkeep is at its balance, the held error does **not** rise
with age. A neuron series that shows no rise with age confirms the balance. A series that *does* rise says the ratio `g/f`
itself moved with age - which is a finding, not an accumulation, and it is then the ratio we have to explain. Either way the
neurons decide it, because they are the only cells in which the interval is upkeep alone.

**This re-reads the write-enzyme lever as one of the strongest, not one of the weakest.** Lowering the restore rate `g`
raises the held error by a derived amount with nothing fitted, whether or not the cell divides - so the old objection (that
inhibitors also change the division rate) loses most of its force, since division only sets the time to balance.

**Test 4 outcome (run once, 2026-10-13; `levers_synth_04.py`, output `levers_synth_04_output.csv`, summary
`levers_synth_04_summary.txt`). BAR MET.** With ages spanning 20-90 and the reader a plain slope of logit(eps) against age, a
drift in ln(g/f) across a lifetime is detectable at 5 % with 20 donors when donors differ by 5 %, and at 10 % with 20 donors
when they differ by 10 %; 10 donors reach 10 % and 20 % respectively. The reader's false-positive rate at no drift is 0.050
in every case, so it does not invent a drift. **What this licenses:** a brain series of about 20 donors can say whether the
balance holds with age to within 5-10 %, which is the prediction the foundation makes for cells that never divide. It says
nothing about whether the balance does hold; that is what reading the two laboratories answers.

## Two signatures that separate mechanism

The copy head is processive: when it fails it fails over a run of sites. Thermal damage is independent site to site.
So enzymatic error must be **clustered along the strand** and chemical damage must be **scattered**, which makes the
C-score a mechanism discriminator rather than a descriptive statistic. Second: a half-copied site is itself the copy
head's substrate, so an error propagates at the next division, and the large partially methylated regions in cancer grow
outward from existing ones. If that autocatalysis is real, the distribution of disorder is heavy-tailed and its growth
super-linear rather than a random walk.

## What this audit says to do, in order

1. **Two clocks, one temperature series.** Predict both forms before looking: the floor's near-linear dependence on kT
and the chemical rate's exponential dependence. A species or hibernation series separates them. Nothing in biology or
statistics explains that away.
2. **The erase channel alone,** through IDH-mutant tumors, where only the reverse channel is chemically blocked.
3. **Clustered versus scattered,** as a mechanism test rather than a description.

Each becomes its own door with its rule fixed in advance, or it is dropped.

## Synthetic tests of the levers (the author's rule: simulate first, one question at a time)

Each lever costs data we spend once, so each is first asked of simulated molecules: **is the effect detectable at the depth
and sample size we can actually get?** A synthetic result says nothing about the world, only about our instruments.

**Test 1, clustered versus scattered** (`data/DEV_LEVERS_01/levers_synth_01.py`, committed before it ran). Processive
copy-head failure (runs of consecutive sites) against independent chemical damage, at matched total error, read by the
chain's own clustering reading, 200 draws each. Bar written first: the two must not overlap across draws, with the
processive case above the healthy range, at a run length of 25 sites or fewer. Otherwise the clustering reading cannot
separate mechanism at this depth and this note says so.

**Test 1 outcome (run once, 2026-10-13; output `levers_synth_01_output.csv`, summary `levers_synth_01_summary.txt`).
BAR MET, with one boundary worth stating.** Scattered damage leaves the clustering reading at the healthy value at every
error amount tried, while processive runs raise it well outside the healthy range, and 35 of the 36 conditions show no
overlap at all across 200 draws each. The one condition that fails is the weakest: 200 sites moved in runs of 5 at 2
healthy units, where the processive case overlaps the top of the healthy range. So the reading separates mechanism as
soon as a failure runs over about ten consecutive sites, and below that it cannot.

**What this does and does not license.** It says the clustering reading can carry a mechanism argument on real arrays:
enzymatic error must clump and show up, chemical damage must not. It says nothing about which mechanism any real sample
has; that needs the levers themselves.

**Test 2, two clocks (planned).** An enzymatic floor scaling with kT against a damage rate scaling exponentially with it:
how many temperatures and donors per temperature are needed to tell a line from an exponential against real donor spread.

**Test 2 outcome (run once, 2026-10-13; `levers_synth_02_output.csv`, `levers_synth_02b_output.csv` and their summaries).**
The first version asked whether a *pure* floor can be told from *pure* thermal damage and separated them 100 of 100 in every
design, which is nearly worthless: over 277-313 K the damage term changes by tens of times while the floor term changes by
about 13 %. Test 2b asked the real question - both terms present, the activation energy fitted rather than given, and a
per-species offset, since each temperature in a cross-species series is a different animal. **18 of 108 designs detect
anything.** The smallest detectable thermal-damage share, by how much species differ for reasons unrelated to temperature:
3 % spread -> 10 % share (8 temperatures x 10 or 20 donors); 6 % spread -> 25 %; 10 % spread -> 25 %, and at that spread
only one design in the grid reaches it, 8 temperatures x 20 donors. Below those, the fitted share cannot be told from the
no-damage draws.

**What that says about the order of work.** A cross-species temperature series is a weak lever: it needs eight species'
worth of temperatures, twenty donors each, and species that differ by no more than a few percent for other reasons - which
is not what species are like. **Hibernation is the design to run**, because the same animal is warm and cold, so the species
offset is removed by construction rather than fitted. The fish series stays useful as a second line, not as the first.

**Test 3, three channels (planned).** Impair copy, de novo and erase separately and ask whether the gauge and the
clustering reading identify which one was hit. This is the prerequisite for reading IDH-mutant tumors as an erase-channel
knockout: no distinguishable signature in simulation means those patients cannot answer the question.

**Not a milestone:** an audit of levers and caveats; no data read and no pass value written.

**In plain words.** A law is only believable if you can change something in the physics and have it move the way the law
says, with nothing adjusted afterwards. This note lists every such handle we can reach - body temperature, the cell's supply
of methyl groups, how often it divides, the enzymes that write and erase, oxygen, tumors that block erasing, and plain
chemical damage - with what is wrong with each one. Three of them we had never considered. **What it would take.** Each
handle is first tried on simulated data, to find whether the effect is even visible at the amount of data we can get. Two
are done: the meters can tell clumped damage from scattered damage once a failure runs over about ten sites in a row, and
comparing different species' temperatures is a weak test while hibernating animals - the same animal warm and cold - is a
strong one. That changed which study we plan to run, before we spent a month on the weaker one.
