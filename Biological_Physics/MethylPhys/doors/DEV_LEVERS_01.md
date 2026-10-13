# DEV-LEVERS-01 — every lever on the methylation channel, with its caveat (audit, 2026-10-13; nothing read, no pass value written)

The quantum line (QAPE) enumerated every platform with its own decoherence channel and caveat before any platform was
scored. The cell has never had that audit: each lever was taken up one at a time. This note is that audit. **It is a
design note.** Nothing here is a result, no bar is written, and every row says plainly whether a *parameter-free*
prediction is possible, since that is the only kind of test a physicist cannot dismiss as fitting.

## The levers

| Lever | What it changes in the physics | Caveat | Parameter-free prediction possible? | Status |
|---|---|---|---|---|
| Temperature | kT sets the floor | mammals are all 36-39 C, so this needs birds (40-42), fish (ambient), hibernators (5-37), fever, or culture at 33/37/39 | **yes** - a derived ratio between two temperatures | designed, never run (`DEV_FLOOR_HEIGHT_02`) |
| Methyl supply (SAM, methionine, folate, B12) | the driving energy behind each write | starvation stops division, removing the per-copy term being measured; this is why methionine parked | partly - the direction is derived, the size needs the division rate | one mouse result met its committed window (`DEV_SAM_LEVER_01`); `DEV_METHIONINE_01` parked |
| Division rate | Landauer's cost is paid per copy | division rate travels with age, tissue and culture | **yes** - zero divisions must read differently in a derived way | neuron rule sealed (`DEV_IAMA_NEURON_01`); passage series in hand (GSE268145) |
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

**Test 3, three channels (planned).** Impair copy, de novo and erase separately and ask whether the gauge and the
clustering reading identify which one was hit. This is the prerequisite for reading IDH-mutant tumors as an erase-channel
knockout: no distinguishable signature in simulation means those patients cannot answer the question.

**Not a milestone:** an audit of levers and caveats; no data read and no pass value written.
