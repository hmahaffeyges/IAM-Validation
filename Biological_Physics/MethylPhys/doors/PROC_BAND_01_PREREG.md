# PROC-BAND-01 — can progenitor and stem_adult carry a commissioned healthy band in whole blood?

**Pre-registered 2026-09-25, before any distribution was computed.** Enhancement A2, the first item on
[`ENHANCEMENTS.md`](ENHANCEMENTS.md): *"Converts the departure statistic from one banded axis into three, with
proper chi-square thresholds. That is what the Mahalanobis design was for."*

## The question

The departure statistic on every report today is `|z_immune|` — one axis — because **immune is the only class
with a commissioned band**. The Safeguards tab says so, and the Departure tab states the consequence in
words: *"With one commissioned class band, the departure is simply |z| of that class; as further class bands
are commissioned the distance becomes a true multi-axis Mahalanobis distance."* A one-axis distance cannot do
the thing the design was chosen for: notice a specimen that moves one component while another holds steady.

So: **do progenitor and stem_adult, measured on the same four healthy cohorts that commissioned the immune
band, support a band of their own?** If they do, the departure statistic becomes three axes with a measured
threshold. If they do not, they stay unbanded, their tier stays withheld, and this document says why.

## Amendment, 2026-09-25, before any array was read

Reading `stage_b_identity` rather than the enhancement description changed what this procedure can ask, so
the target is corrected here rather than quietly:

**The chain does not read progenitor and stem_adult separately on whole blood — it reads them jointly.** Its
own rule, in the function's docstring: *"s108 reporting rule: on whole blood only immune and the joint
haematopoietic-progenitor component are read."* The two classes' identity loci are pooled and scored against
progenitor's floor as one surface. So the achievable result is **two axes, not three**, and the cost estimate
I wrote in [`ENHANCEMENTS.md`](ENHANCEMENTS.md) ("converts the departure statistic from one banded axis into three") was written
from the description instead of the code. It is corrected there too.

**And the joint component has no reference layers.** In the live code the age term and the laboratory zero are
applied `if name == "immune"` only, so the joint component returns `A_mapped` and nothing absolute. A band
needs `A_abs = A_mapped - c(decade) - z_lab`, which means this procedure must **derive an age term and a
per-laboratory zero for the joint component** — neither of which exists — before a band can be measured at
all. That is real work, not the two to three hours the list claimed.

**How that is kept honest.** The age term and the laboratory zero are derived **inside each
leave-one-laboratory-out fold, from the three laboratories in that fold**, and the held-out laboratory's
coverage is measured against them. A curve and a band fitted on the same arrays that then judge the false-alarm
rate would be measuring themselves; this is the only design that avoids it with 318 arrays. B7 below is
restated accordingly.

## What enters, and what does not

| | |
|---|---|
| Arrays | **318 healthy whole-blood arrays**, four laboratories: GSE87571 (80), GSE42861 (78), GSE111629 (80), GSE125105 (80) |
| Betas | [`reference_data/`](../reference_data/) — the Stage 1 noob output published beside the constants, the same matrices the immune band was measured on. Nothing is re-calibrated |
| Loci | the class identity loci as frozen: progenitor 54,704 · stem_adult 46,907 · immune 42,134 (control) |
| Floors | as frozen: 0.8522 · 0.8737 · 0.838889. **Not refitted** |
| Scale map, age curve, laboratory zero | the existing commissioned values. **Not refitted** — a band fitted on top of references it also moved would be measuring itself |
| Ages | recovered from the four GEO series-matrix headers and **published as a per-array table with the outcome**, because the band cannot be reproduced without them |
| Scoring | the chain's own path (`stage_a_cells` → `stage_b_identity`), not a reimplementation. A band measured by different code from the reading is not a band for that reading |

## The bars, fixed now

**B1 — the component is present.** The joint haematopoietic-progenitor fraction must be **≥ 0.01 in ≥ 95 % of
the 318 arrays** - the same presence test the chain itself applies before it will score the surface at all. A
band measured where the component is not detectable is a band on noise.

**B2 — it reproduces across laboratories.** Leave-one-laboratory-out: build the band on three, measure what
fraction of the held-out laboratory's arrays fall inside the nominal 80 % interval. Every held-out fraction
must land in **[0.70, 0.90]**. This is the bar the immune band met — PROC-SWITCH-01 recorded
*"held 0.753–0.839 of a fourth lab, nominal 0.80"* — and it is applied unchanged so the new classes are held
to the standard the commissioned one passed.

**B3 — the false-alarm rate is honest.** With the adopted band, the fraction of healthy arrays beyond p95
must be **≤ 0.10 in every laboratory and ≤ 0.06 pooled**. The immune band's own per-laboratory rates run
0.044–0.098, so this permits no worse than the axis already in service.

**B4 — the axis adds something.** The joint component's A″ must have **|Pearson r| < 0.9 against immune A″**
across the 318 arrays. Two axes that move together are one axis with extra arithmetic, and the honest outcome
in that case is to keep reporting one.

**B5 — the multi-axis threshold is measured, not assumed.** The adopted p95 for the Mahalanobis distance over
k axes must be the **measured 95th percentile of the healthy distances**, with the chi-square table value
reported beside it. If they disagree by more than 15 % the measured value is used and the disagreement is
stated on the page.

**B6 — the commissioned reading does not move.** Immune A″ must be **identical to the sealed path to 1e-9 on
all 318 arrays**. An improvement that shifts the number already in service is rejected whatever it does for
the new classes.

**B7 — nothing is fitted on the arrays that judge it.** The joint component's age term, laboratory zero and
band are derived **inside each fold from that fold's three laboratories**, and the held-out laboratory is
scored against them; the leave-one-out fractions are the only evidence for B2. The pooled band reported at the
end is built once, on all four laboratories, and B3 is a property of that band rather than a search over
candidates. No band is chosen by comparing false-alarm rates.

## Decision rule

- **The joint component clears B1-B4:** the departure statistic becomes **two axes** with the measured
  threshold from B5, the joint component's tier stops being withheld, and the register row for the departure
  statistic is rewritten to say two.
- **It fails B2 or B3:** it stays unbanded, the statistic stays `|z_immune|`, and the failing number is
  published - that tells the next reader whether the obstacle is the component or the cohort count.
- **It fails B4:** the component is measurable but redundant. The band is published as a diagnostic and the
  reported statistic stays one axis, because two names for one measurement is how a reader is misled.
- **B6 fails:** the work is discarded entirely and the cause found before anything else is attempted.

Whether progenitor and stem_adult can be **separated** on whole blood is a different question, not asked here:
it would need the atlas's duplicate-label merge (roadmap item 2) and a reason the chain's own s108 rule should
be overturned. If the joint band succeeds, that separation becomes the next question rather than this one.

Nothing in this procedure reports on a patient. Its only product is whether one more axis can be trusted.

---

_Written before any array was read. Nothing about the 318-array distribution was known when the bars above
were chosen, and B2's interval is not a number invented for this procedure — it is the interval the immune
band was held to in PROC-SWITCH-01, applied unchanged._
