# PROC-BRAIN-01 — can the instrument find terminal-class cells in a liquid specimen from a CNS tumour patient?

**Pre-registered 2026-09-26, before any array was downloaded.**

## Provenance of the thresholds — they are not being invented now

The decision rule comes from the author's own **VAL-009, April 2026**, written before any of this data was
touched, and is used as written rather than re-tuned:

> *"The BBB prevents tumor-derived cfDNA from entering plasma until the BBB breaks down."*
> *"Step 3: IF terminal class A > 1.05 in plasma:"*

The presence gate comes from a measurement made today, before this cohort was chosen: across **845
EPIC-Italy whole-blood arrays from people with no CNS disease**, the terminal fraction has p99 = **0.0085**
and a maximum of **0.0312**, with only 7 of 845 exceeding 1 %.

## What is being tested, and what is not

**GSE292312** (n = 181, EPIC / GPL21145), *"Robust classification of pediatric brain tumors from cell-free
DNA methylomes"*.

> ### Amendment, 2026-09-26, before any array was scored
>
> **As first written this section described the whole series as cerebrospinal-fluid cfDNA. That is wrong.**
> Reading the series metadata gives `source_name`: **157 primary tumour tissue and 24 CSF** — so the CSF
> arm is **24 specimens, not 181**, and the original text overstated it roughly sevenfold.
>
> **No threshold and no bar changes.** The presence gate stays at 0.0312 and the A bar at 1.05, both fixed
> before this cohort was chosen. What changes is the declared arm sizes:
>
> - **B1 and B2 are the CSF arm, n = 24.** At n = 24 the B2 requirement that at least half of specimens
>   clear the gate means **12 or more**, and a proportion from 24 carries a 95 % interval of roughly
>   ±0.20 — so B2 can distinguish "most patients" from "a handful" but not much finer than that. This is
>   stated now rather than discovered in the outcome.
> - **The 157 primary tumour tissue arrays are a separate declared arm**, and they are the stronger
>   positive control: brain tumour *tissue* should be dominated by terminal-class material, so if terminal
>   cannot be found there it cannot be found anywhere, and B1's failure in CSF would be uninterpretable.
>   **The tissue arm is reported but is not a bar** — it was not pre-registered as one, and a tissue result
>   does not bear on whether a *liquid* specimen can be read.
> - **The series contains no healthy controls of either substrate.** The comparator therefore remains the
>   845-blood null as written, and the substrate difference between CSF cfDNA and whole-blood leukocyte DNA
>   is a limitation of that comparison, not a property of the finding. It is recorded here so the outcome
>   cannot claim otherwise.


**CSF is inside the barrier.** This procedure therefore tests whether the instrument can **find and score
brain-derived cells in a liquid specimen at all** — a necessary precondition. It does **not** test the
BBB-breakdown claim, which requires plasma. Stating this in advance so that a pass cannot later be read as
evidence for the plasma hypothesis.

## Bars

| | bar | met when |
|---|---|---|
| **B1** | terminal is **found** | median terminal fraction in the **24 CSF** specimens **> 0.0312**, the maximum observed in 845 non-CNS bloods — a threshold at which that cohort gives **0 of 845** false positives |
| **B2** | it is found in most patients, not a few | **≥ 50 %** of the 24 CSF specimens exceed the gate (**≥ 12 of 24**) |
| **B3** | the author's April rule holds | among specimens passing B1, median terminal **A > 1.05** |
| **B4** | it is not the absence artefact | in specimens **failing** the presence gate, terminal A is reported separately and is **not** used to support B3 — per PROC-CEIL-01, a class below its presence floor reads high because it is absent |
| **B5** | it is specific | terminal is elevated **more** than the median of the other six non-haematopoietic classes; if everything rises together it is substrate mismatch, per the DISC-BLADDER-003 signature that disqualified PROC-TISSUE-01 |
| **B6** | the instrument has not moved | immune A_mapped unchanged on the 318 published arrays |

**Direction fixed:** elevation. A terminal fraction at or below the blood null is a failure, not a finding.

**Which fit is reported, pre-specified:** both. The **cell-level solve is primary**, because
[`TWO_FIT_FINDING.md`](TWO_FIT_FINDING.md) measured today that the pooled solve misallocates ~8 points of
immune mass under a collinearity of r = 0.958–0.989, and terminal is a non-haematopoietic class whose
fraction is exactly what that degeneracy would disturb. The pooled value is reported alongside so the two can
be compared.

## What each outcome licenses

- **B1–B3 and B5 met:** the instrument finds and scores brain-derived cells in a liquid specimen. That licenses the plasma procedure, and nothing more.
- **B1 met, B5 failed:** substrate mismatch, not detection. Reported as such.
- **B1 failed:** terminal cannot be found even in CSF, where the fraction should be highest. That closes the liquid route for this class on this reference and is the more useful negative.

No clinical claim follows, no patient is involved, and nothing in the chain changes on the basis of this
procedure.
