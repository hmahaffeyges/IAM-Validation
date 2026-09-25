# PROC-LABBAND-01 — should each laboratory be judged against its own width?

**Pre-registered 2026-09-25, before any width was computed.** Enhancement A3, and the item both of today's
failures pointed at: PROC-BAND-01 failed leave-one-laboratory-out, and PROC-CLS-01 failed it *in both
directions* — one laboratory under-covered, another over-covered — which is the signature of a width that is
right on average and wrong everywhere.

## The question

Every reading is placed against **one pooled band**: `identity_band_v3`, µ = 1.000 with σ from the pooled
p10–p90 over four zeroed cohorts. The laboratory zero already centres each cohort at 1.000, so the pooled
band is correct in its *centre* by construction. What it cannot be correct in is its **width** — and the band
file's own records show the consequence: healthy arrays beyond p95 run **4.41 % at UCLA, 5.61 % at Uppsala,
6.47 % at Munich and 9.84 % at Karolinska** against a nominal 5 %.

That spread is why a healthy control can read ELEVATED. **Does each laboratory need its own σ?**

This changes a number the chain reports — the tier — so it is the strictest kind of change, and the bars
below include one that has nothing to do with false alarms: a wider band hides real departures, and an
improvement that buys calibration with sensitivity is not an improvement.

## What enters

| | |
|---|---|
| Readings | the **318 immune A″ values** measured and published by PROC-BAND-01, four laboratories |
| Zeros, curve, floors | the commissioned values. **Nothing is refitted** — only the width is in question |
| What is *not* used | no new data, no new calibration, and no disease specimen |

**A limitation stated up front:** these are 78–80 arrays per laboratory, while the band file's own tails were
measured on the full cohorts (204–659 arrays). A width from 80 arrays is itself uncertain, which is exactly
what B2 tests, and if B2 fails the answer is that this needs the full cohorts rather than that laboratories
are alike.

## The bars, fixed now

**B1 — the widths really do differ.** Two things must both hold: the ratio of the largest to the smallest
per-laboratory σ must exceed **1.20**, and a **label-permutation test** (5,000 shuffles of the laboratory
labels across the 318 readings, statistic = that same ratio) must give **p < 0.01**. If widths are
indistinguishable, the pooled band is already right and there is nothing to adopt.

**B2 — a width measured on 80 arrays is stable.** Split-half within each laboratory, 200 random splits: the
median ratio σ(half A)/σ(half B) must lie in **[0.80, 1.25]** for every laboratory, and the 90 % spread of
that ratio must not exceed **[0.65, 1.55]**. A width that will not reproduce inside one cohort cannot be
trusted across cohorts.

**B3 — it fixes what it claims to fix, out of sample.** Fit σ on a random half of a laboratory, measure the
tail beyond |z| > 1.96 on the other half, 200 splits. The median out-of-sample tail must fall in
**[0.03, 0.07]** for **every** laboratory. The pooled band's own tails (0.044–0.098) are the comparison, and
they will be reported beside it.

**B4 — sensitivity is not traded away.** A departure of **+2σ_pooled** injected into each healthy reading
must still read outside the per-laboratory band in **≥ 80 % of arrays in every laboratory**. A laboratory
whose σ widens is exactly where this can fail, and if it does, the wider band is hiding real signal and the
change is refused.

**B5 — the reading itself does not move.** Immune A″ is unchanged by this procedure: **max |ΔA″| = 0 across
all 318 arrays**, verified by recomputation and not by assertion. Only the width against which A″ is placed
may change.

**B6 — a laboratory with no band of its own must still be servable.** The adopted rule must name what
happens for a laboratory that has not been characterised, and that fallback must be the pooled band with the
reading marked as placed on the pooled width. A change that silently refuses new laboratories is not
adoptable.

## Decision rule

- **B1–B6 all met:** per-laboratory widths are adopted, `identity_band_v3` gains a per-laboratory σ block,
  the report states which width placed the reading, and the register row for placement is rewritten.
- **B1 fails:** the laboratories are alike; the pooled band stands and the 4.4–9.8 % spread is then *not*
  a width effect and needs a different explanation — which is itself worth publishing.
- **B2 fails:** the effect may be real but 80 arrays cannot measure it. Published, not adopted, with the
  full-cohort requirement named.
- **B3 or B4 fails:** the change does not deliver, or it costs sensitivity. Refused, and the number that
  refused it is published.

Nothing here reports on a patient. Its only product is whether the width a reading is judged against should
depend on who measured it.
