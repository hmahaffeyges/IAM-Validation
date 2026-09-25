# PROC-FOREIGN-01 — should the immune tier be withheld when a specimen carries material the gauge was not built for?

**Pre-registered 2026-09-25, before any mixture was scored.** Enhancement A4. Unlike A2, A3 and B2 this one
needs no new data: it is measured by mixing, the same construction PROC-SMALL-01 used, and it is a
**correctness fix on a number the chain prints today**.

## The question

The immune gauge is commissioned on **whole blood**. Its floor, its age curve, its laboratory zero and its
band were all measured on healthy blood, and a reading is placed against that band and given a tier word.
Nothing in the chain asks whether the specimen in front of it *is* whole blood.

A specimen carrying material the gauge was not built for — tumour tissue in a biopsy, a contaminated draw,
a tissue sample sent by mistake — still gets a tier. The composition step will report the foreign fraction
honestly, but the tier is printed beside it as though the reading were comparable to the healthy band. That
is the confound a tumour-bearing specimen presents, and it is the one case where a wrong tier is most
likely to be believed.

**So: at what level of foreign material does the immune tier become wrong, and can the chain see that level
coming before it prints one?**

## What enters

| | |
|---|---|
| Hosts | the **318 healthy whole-blood arrays** of PROC-BAND-01, four laboratories |
| Foreign material | the atlas's own class means for **stromal, secretory and terminal** — non-blood classes with their own identity loci |
| Mixture | `y = (1 − f) · y_host + f · μ_foreign` at **f = 0, 0.02, 0.05, 0.10, 0.20, 0.35**, exactly PROC-SMALL-01's construction |
| Scored by | the chain's own path: scale map → deconvolver → `stage_b_identity`, with the commissioned floor, curve and zero |
| Not used | no patient, no disease specimen, no recalibration |

**A limitation stated before the run:** an atlas class mean is smoother than real tissue, which has its own
donor variation. A mixture built from a class mean is therefore an *optimistic* foreign specimen — easier to
detect than the real thing. Any detector that fails here would fail worse in a clinic, so a failure is
conclusive and a pass is provisional. A real-tissue check on the EPIC adenoma in the test package is named
as the follow-up, not run here.

## The bars, fixed now

**B1 — foreign material really does corrupt the reading.** At some f ≤ 0.20, the median shift in immune A″
must exceed **2 σ_pooled** (σ_pooled = 0.02092, the band's own width). If foreign material at a fifth of the
specimen cannot move the reading by two healthy standard deviations, there is nothing to protect against and
the enhancement is withdrawn.

**B2 — the tier actually flips.** At that same f, **≥ 25 %** of hosts must change tier word against their
own f = 0 reading. A shifted number that never crosses a tier boundary is a smaller problem than this
procedure assumes.

**B3 — a detector exists, and it is one the chain already computes.** Using only quantities the chain
produces today (the composition fractions and the identity-gauge outputs — **no new statistic invented for
this procedure**), a rule must reach **sensitivity ≥ 0.90** at the smallest f that clears B1 and B2, at a
threshold set to give **≤ 0.05 false-positive rate on the 318 unspiked healthy arrays**. The threshold is
set on the healthy arrays alone, before any spiked array is scored against it.

**B4 — it fires before the tier misreads, not after.** At **every** f where ≥ 10 % of hosts flip tier, the
detector must fire on **≥ 90 %** of those hosts. A guard that triggers only once the specimen is obviously
foreign is not a guard.

**B5 — the guard is silent on healthy specimens.** On the 318 unspiked arrays the detector must fire on
**≤ 5 %**, and — the bar that matters for the clinic — **no tier that the chain prints today for a healthy
array may be withheld** beyond that rate. A guard that withholds tiers from healthy people has replaced one
error with a worse one.

**B6 — the reading itself does not move.** Immune A″ on the 318 unspiked arrays must be identical to
PROC-BAND-01's published values, **max |ΔA″| = 0**, verified by recomputation and reported as a number. This
procedure adds a refusal, not an adjustment.

## Decision rule

- **B1–B6 met:** the withholding rule is adopted, `stage_b_identity` gains the guard, the report states
  *why* a tier was withheld, and the register row for placement is rewritten.
- **B1 or B2 fails:** foreign material does not corrupt the tier at these levels; the enhancement is
  withdrawn and the measured insensitivity is published — it would mean the gauge is more robust than
  assumed, which is worth knowing.
- **B3 or B4 fails:** the confound is real and the chain cannot currently see it. **Nothing is adopted**, and
  the outcome states plainly that a tier on a specimen of unknown composition is not defensible — which is a
  caveat the manual must then carry.
- **B5 fails:** refused outright. Withholding tiers from healthy people is worse than the problem.
