# Which substrate can this instrument read? — the plan, decided by the physics rather than by preference

Written 2026-09-26, after PROC-PARTIAL-01 closed the question of scoring a non-blood class from whole blood.
The author's question: *if not whole blood, then what — tissue, plasma, stool? What exists that we can show?*

## The rule that decides every case

PROC-PARTIAL-01 did not merely fail; it measured **why**, and the reason generalises into a design rule:

> A class can be scored when the reference reconstructs the specimen at that class's identity loci to within
> roughly **f × (the class's own signal)**, where f is the fraction the class occupies. The atlas class means
> currently achieve **0.067 in β**. The 1/f amplification does the rest.

So the question "can we read disease in substrate X" reduces to a question with a number in it: **what
fraction does the class of interest occupy in X?**

| substrate | class of interest | its fraction | scoreable today |
|---|---|---|---|
| whole blood | **immune** | 0.90–0.97 | **yes** — commissioned |
| whole blood | epithelial / tumour | 0.01–0.05 | **no** — closed by PROC-PARTIAL-01 |
| **tumour tissue** | the tumour's own lineage | **0.5–0.9** | **yes, directly** |
| adjacent normal tissue | that tissue | 0.6–0.9 | **yes, directly** |
| plasma cfDNA | **immune** (the background) | 0.8–0.95 | plausible; needs a scale map for fragmented DNA |
| plasma cfDNA | tumour | 0.001–0.1 | no, except late-stage disease |
| stool | colonocyte | low and variable, bacterial-dominated | **no** |

**Stool — corrected 2026-09-26.** The first version of this document ruled stool out. That was an error, and
it came from answering about the wrong specimen: **bulk stool DNA** is bacterial-dominated with a small,
variable human fraction, and that is what the table row above describes. But stool is not interesting as bulk
DNA. It is interesting as the **non-invasive route to exfoliated colonocytes** — epithelial cells shed from
the entire colonic surface. Enrich those (immunocapture on an epithelial surface marker is the standard
approach) and the fraction inverts: the specimen becomes *dominated by the very epithelium the gauge needs to
read*, which is the regime where this instrument works rather than the one it fails in.

| stool, read two ways | colonocyte fraction | scoreable |
|---|---|---|
| bulk stool DNA | low, variable, bacterial-dominated | no |
| **enriched exfoliated colonocytes** | **high after capture** | **yes, in principle** |

That makes stool the *sampling* answer, not a substrate to dismiss — and it is the reason the tissue work
below matters, because it is what would license it.

## What has to be demonstrated — and it is not that Landauer holds in the methylome

**The author's correction, 2026-09-26: "I dont need to prove Landauer in the methylome, we need to show what
it means when we use Landauer Metrology."** That rules out the obvious first experiment. *Tumour reads
higher than normal* is a result every method in the field already has; producing it again on a physical
scale demonstrates the scale works, not that it is worth having.

What a Landauer measurement has that a marker panel does not:

| | a trained marker panel | a fidelity reading against a physical floor |
|---|---|---|
| what it returns | a **label** — like / unlike the training set | a **distance** — how far from the identity floor |
| what it needs | a cohort per disease, per platform | the floor, which is physics, and one scale map |
| comparability | to its own training set only | across laboratories, platforms and years |
| resolution | one answer | per architecture class — *which* compartment drifted |
| new disease | retrain | the same reading already covers it |

So the demonstrations that show *meaning* are the ones only a distance-measuring, class-resolved, absolutely
scaled instrument can produce:

1. **An ordering along a progression, on one scale, with nothing trained on any of it.** Healthy mucosa <
   adjacent normal < adenoma < carcinoma. A panel can separate two classes it was trained on; a distance
   should place all four in order without being shown the order. That is the claim worth making.
2. **Field effect — a reading at a distance from the lesion.** If *normal-looking* mucosa from a patient
   already sits above mucosa from someone without disease, then sampling the accessible surface tells you
   about a lesion elsewhere. **This is the result that licenses the stool route**, and without it, stool
   colonocytes are just a harder way to biopsy.
3. **Response to an intervention.** If the reading moves when something known to reduce risk is
   administered, it is a monitorable quantity rather than a classifier — and that is what "metrology" claims.

## Tier 1 — tissue. The decisive demonstration, and it is available now.

**Why tissue is the strongest case.** In a tumour biopsy the malignant lineage occupies most of the
specimen. There is no deconvolution step, no 1/f amplification, and no reliance on the reference being
accurate at the 0.015 level. The instrument reads what it was designed to read.

**And the design removes the blocker that has stopped every new laboratory: the laboratory zero.** A paired
tumour-versus-adjacent-normal comparison is *within one patient, one laboratory, one chip* — so the
laboratory zero, the age term and the donor all cancel exactly, the same way they cancelled in PROC-EPIC-01.
No new calibration panel is needed to run it.

**The physical claim being tested is the one the whole framework rests on:** a tumour has lost identity
fidelity relative to the tissue it came from, so its A-score should be higher than its own matched normal.
If that does not hold in tissue, where the measurement is cleanest, the framework has a problem; if it holds,
it holds without any of the caveats that attach to a blood result.

| candidate | n | platform | what it gives |
|---|---|---|---|
| **GSE131013** | 240 | 450K | **healthy · adjacent normal · tumour colon cells, in one series** — the ordering test and the field-effect test at once |
| **GSE48684** | 147 | 450K | normal · **adenoma** · carcinoma — the missing rung of the progression |
| GSE199057 | 229 | EPIC | epigenetic outliers in normal mucosa of CRC patients — field effect, independent platform |
| GSE132804 | 334 | EPIC/450K | normal colon and colorectal cancer *risk* |
| GSE142257 | 124 | EPIC | aspirin against age-related drift in healthy colon — an **intervention** series |

**GSE131013 is the one to run first**, because it carries all three groups in a single series and laboratory:
healthy colon cells from people without disease, adjacent normal from patients, and tumour. That is the
field-effect ladder with the laboratory held fixed, so no new calibration panel is needed and the comparison
is internal. **GSE48684 adds the adenoma rung**, which turns an ordering of three into an ordering of four.

Neither is scored by anything trained on them: the identity loci, the floors and the scale map are all
already commissioned, so the prediction — healthy lowest, then adjacent normal, then adenoma, then carcinoma —
is made by the instrument as it stands, and is fixed in the pre-registration before the first array is read.

**On the stool route.** A GEO search for exfoliated colonocytes or stool-derived human methylation returns
eight series, none of them an array cohort of captured colonocytes — the hits are biopsy, organoid and
microbiome work. So there is no public dataset to test the stool step on today: it needs either a
collaboration or samples collected for the purpose. That is not a reason to drop it. It is the reason to
establish the field effect in tissue first, since a field effect is exactly what makes a whole-surface
sample like stool informative, and it is the result that would justify asking anyone to collect.

## Tier 2 — the 2–8 year blood window. The result worth defending, and the data is scarce.

PROC-EPIC-01 found the colorectal immune signal loudest 2–8 years before diagnosis (d ≈ +0.72), which is the
window the author considers meaningful. Confirming it needs a **second prospective cohort with lead times**.

**A GEO search returns only seven prospective/prediagnostic methylation series, and none is a usable second
colorectal cohort.** That is a finding in itself: public prediagnostic methylation cohorts with cancer
follow-up are rare, which is precisely why EPIC-Italy is the one everybody re-uses. The realistic routes are
therefore not "search GEO harder":

1. **The EPIC-Italy arms not yet used** — 119 held-out cases of other cancer types are already scored and sitting in this repository, which tests whether the 2–8 year signal is colorectal-specific or general.
2. **Controlled-access cohorts** (dbGaP / EGA: NOWAC, MCCS, Sister Study) — an application, not a download.
3. **A collaboration**, which is what a published tissue result is *for*.

## Tier 3 — plasma cfDNA. Real, but it is an instrument change, not a cohort change.

The cfDNA background in plasma is immune-derived, so the **immune** A-score is in principle readable there —
and that is the same class that carries the colorectal signal, which makes it the natural non-invasive route.
Two things stand in the way, and both are engineering rather than physics:

- **Fragmented DNA needs its own scale map.** The chain's scale map is commissioned for intact genomic DNA on 450K/EPIC. cfDNA β distributions are shifted by fragmentation and by the cell-free preparation.
- **A healthy cfDNA panel** of ≥ 40 arrays is needed for a laboratory zero — unless the design is paired or case/control within one laboratory, in which case it cancels, as above.

The tumour fraction in plasma is 0.1–10 %, so **scoring the tumour class in plasma is the same closed
question as whole blood.** What is open is the immune reading, and the GAPE line already carries the
substrates for it (WPS, DELFI fragment size).

## What "definitive" would require, stated before any of it runs

A group difference is not a detection claim. To say this instrument *detects* disease:

1. a **threshold fixed in advance**, not chosen after seeing the separation;
2. **out-of-sample** performance — sensitivity and specificity on arrays not used to set the threshold;
3. a **stated operating point** (at X % specificity the sensitivity is Y), because a clinical claim is a point, not a curve;
4. a **null that does not know the answer** — label permutation and, for tissue, a matched-normal-vs-matched-normal comparison;
5. the **direction pre-specified**, as in every procedure since PROC-BAND-01.

PROC-EPIC-01 met none of these and did not claim to — it measured group separation. The tissue work is where
a real detection claim can first be made honestly, because the sample sizes and the effect sizes are both
large enough to support one.

## The order

1. **PROC-TISSUE-01 — the ordering, on GSE131013.** Healthy < adjacent normal < tumour on one absolute scale, with the direction pre-specified and nothing trained on the data. Tests the field effect and the ordering in a single series.
2. **PROC-TISSUE-02 — the adenoma rung, on GSE48684.** Does the reading place a pre-malignant lesion between normal and carcinoma without being told it exists?
3. **PROC-TISSUE-03 — the intervention, on GSE142257.** Does the reading move when drift is suppressed?
4. **PROC-EPIC-02** — the 119 held-out other-cancer cases already scored: is the 2-8 year immune signal colorectal-specific?
5. **The stool step**, once the field effect is established: captured colonocytes, which needs collected samples rather than a download.
