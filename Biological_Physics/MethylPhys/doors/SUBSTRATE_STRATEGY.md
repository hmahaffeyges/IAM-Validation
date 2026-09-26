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

**Stool is out, and it is worth saying why rather than leaving it open.** Stool DNA is overwhelmingly
bacterial; the human colonocyte fraction is small and varies with transit, diet and collection. The clinical
stool tests work because they ask a *yes/no* question about a handful of marker loci — which is a detection
problem, not a fidelity measurement. This instrument measures entropy across tens of thousands of loci
against a physical floor; it needs the class to dominate the specimen. Stool does not provide that.

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

| candidate | n | what it is |
|---|---|---|
| **GSE199057** | 229 | normal mucosa of colorectal cancer patients vs controls — the **field-effect** test |
| **GSE309002** | 32 | colorectal cancers vs adjacent normal colon — the **paired** test |
| GSE151732 | 256 | right vs left colon, epigenetic ageing — a within-tissue control surface |
| GSE233854 | 59 | colorectal tumours from MLH1 epimutation carriers |

**Run GSE309002 first** (paired, simplest, tests the core claim), then **GSE199057**, which is the more
interesting of the two: if *normal-looking* mucosa from a cancer patient already reads degraded, that is the
tissue-level analogue of the blood finding, and a much stronger statement than tumour-versus-normal, which
every method in the field can already do.

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

1. **PROC-TISSUE-01** — paired tumour vs adjacent normal, GSE309002. Tests the core physical claim where the instrument is strongest, and needs no new calibration.
2. **PROC-TISSUE-02** — field effect, GSE199057. Normal mucosa from cancer patients vs controls.
3. **PROC-EPIC-02** — the 119 held-out other-cancer cases already scored: is the 2–8 year immune signal colorectal-specific?
4. Then, and only with a tissue result in hand, the cohort applications and the cfDNA scale map.
