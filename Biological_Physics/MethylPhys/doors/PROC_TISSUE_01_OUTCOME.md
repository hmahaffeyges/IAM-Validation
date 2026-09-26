# PROC-TISSUE-01 — outcome: the gating bar failed. No field effect; adjacent normal reads BELOW healthy, and seven of eight classes move together.

**Sealed 2026-09-26** against the bars fixed in [`PROC_TISSUE_01_PREREG.md`](PROC_TISSUE_01_PREREG.md)
before the series was downloaded. GSE131013, 238 of 240 arrays scored (two have no column in the submitted
matrix), 450K, one laboratory: 48 healthy mucosae, 95 adjacent normal, 95 tumour, 91 patients contributing
both their own tumour and their own adjacent normal.
Evidence: [`PROC_TISSUE_01.json`](../kit/results/PROC_TISSUE_01.json) ·
[`PROC_TISSUE_01_scored.json`](../kit/results/PROC_TISSUE_01_scored.json) ·
scripts [`PROC_TISSUE_01_score.py`](../kit/PROC_TISSUE_01_score.py) ·
[`PROC_TISSUE_01_analyse.py`](../kit/PROC_TISSUE_01_analyse.py).

## The verdict

| bar | result | |
|---|---|---|
| **B6** the specimens are what they claim | epithelium-containing fraction **0.446 / 0.465 / 0.479** (corrected — see below) | **FAILED — gating** (bar > 0.50) |
| **B1** the ordering | healthy 1.1684, adjacent **1.1639**, tumour 1.1864 — the first rung goes the wrong way | **FAILED** |
| **B2** the field effect | **d = −0.520**, p = 0.0036 — adjacent normal reads *below* healthy | **FAILED (direction reversed)** |
| B3 the disease contrast | d = +1.209, p = 0.0002 | met — but see below |
| B4 a null that does not know the answer | median \|d\| = 0.1888, **p95 = 0.5781** | met (bar < 0.20) |
| B5 age is not the cause | 7-year gap; matched on sex, side and age: 36 pairs, **d = −0.464, p = 0.056** | matching required and run |
| B7 the instrument has not moved | max \|ΔA\| = 0.000e+00 on 318 published arrays | met |

**The pre-registered decision rule applies at B6: *"the composition finding is reported instead, and the
ordering is not."*** That rule was written before the data was seen and it governs here.

## What the specimens turned out to be

The series title says "colon cells". The deconvolution says otherwise.

**Corrected 2026-09-26 by the atlas readability audit.** This outcome first reported the epithelial
fraction as **0.23**, computed as `secretory + terminal`. That was wrong: in this atlas **colon
epithelium is classified in `cycling`**, not `secretory`. Recomputed per sample with every
epithelium-containing class, the fractions are **0.446 / 0.465 / 0.479** (healthy / adjacent / tumour) —
roughly 45 % epithelial, not 23 %. **B6 still fails**, since the bar was > 0.50, and the scored class is
still immune, since immune (0.382) remains the largest single class — so no verdict below changes. But
the specimens are much closer to the bar than first reported, and the stated reason was wrong.
See [`ATLAS_READABILITY.md`](ATLAS_READABILITY.md) §3 and
[`PROC_TISSUE_01_b6_recheck.json`](../kit/results/PROC_TISSUE_01_b6_recheck.json).

The median epithelium-containing fraction is **0.45** against an immune fraction of 0.34–0.38. These are **bulk mucosa**, not sorted
epithelium — and colonic lamina propria is genuinely lymphocyte-rich, so that is a plausible composition for
bulk tissue rather than a processing failure.

**This has a consequence the procedure must state plainly: the epithelial gauge was never tested.** The
pre-registered class-selection rule — *the class with the largest median fraction in the healthy group* —
selected **immune**, mechanically and correctly. So every number above is the immune compartment measured in
colon tissue, not colonic epithelium. The rule did what it said; what it could not do is conjure an
epithelial reading out of a specimen that is 23 % epithelium against a reference that represents colon poorly.

## The field effect was not observed, and the direction is the opposite of the hypothesis

Adjacent-normal mucosa reads **below** healthy mucosa (d = −0.520), where the pre-registration fixed the
direction as *above*. Three separate checks say this displacement should not be interpreted at all:

1. **It is inside the noise.** Splitting the 48 healthy mucosae at random gives \|d\| up to **0.578** at the 95th percentile. The observed 0.520 is smaller than that.
2. **It does not survive matching.** On 36 pairs matched for sex, colon side and age, d = −0.464 with **p = 0.056**.
3. **It is not specific to one class.** Adjacent-vs-healthy is negative in **seven of eight** architecture
   classes, from −0.49 to −0.63. The exception is `stem_pluri` (+0.112 adjacent, −1.134 tumour), which moves
   opposite to the other seven in both comparisons — worth naming rather than rounding away, though a single
   class inverting does not rescue the seven that move together. A biological field effect displaces the
   compartment that is changing; a substrate or batch difference displaces nearly everything at once.

Point 3 is the author's own documented artifact: **DISC-BLADDER-003** records that bulk atlases on *mucosal*
substrates inflate cross-tile A-scores from substrate-distribution mismatch alone. The same reasoning
disqualifies **B3's +1.209** from being read as a fidelity result: tumour-vs-healthy is also broad, moving
seven classes between +1.2 and +1.8, far above VAL-062's +0.724 reference on the same statistic. B3 is
recorded as met and is **not** interpreted.

## The one comparison that survives — and it was not a bar

Declared as an additional analysis, not as a pre-registered bar, because it was not written into the
pre-registration: the **within-patient** contrast. 91 patients each contribute a tumour and their own
adjacent normal — same person, same chip, same laboratory, same run — so every confound above cancels.

> **median Δ = +0.0211 · 90 % of patients positive · paired d = +1.131**

That is the procedure's only durable observation, and it must be stated for what it is: **the immune
compartment in tumour tissue reads higher than the immune compartment in that same patient's adjacent
normal.** It is not a statement about epithelial identity fidelity, and it does not depend on any of the
between-group comparisons that the class-spread check disqualified.

## What this changes

**The stool route is not licensed by this procedure.** The field effect was the result that would have
justified asking anyone to collect stool-derived colonocytes, and it was not observed — in a cohort where
the epithelial gauge could not be tested in the first place. Whether a field effect exists in *epithelium*
remains open; this says nothing about it either way.

**The binding limit is the reference again, for the fourth time today.** PROC-BAND-01, PROC-CLS-01 and
PROC-LABBAND-01 failed on thin reference panels; PROC-PARTIAL-01 failed on the reference's 0.067 misfit
against real blood; this procedure could not read epithelium because the atlas represents colon poorly
enough that a 23 %-epithelial specimen has no epithelial class to select. Every route forward passes through
the same object.

**Two design lessons, recorded because they cost a procedure:**

1. *"The class with the largest median fraction"* is not the same as *"the class this tissue is made of"* when the reference lacks that tissue. A class-selection rule needs a **floor** — if no candidate class exceeds some fraction, the procedure should stop rather than score whatever came top.
2. **Verify composition before writing the bars, not as a bar.** B6 was correctly placed as gating and correctly failed, but the check is cheap and could have been run on ten arrays before the pre-registration was written, which would have produced a *different and better* procedure rather than a failed one.
