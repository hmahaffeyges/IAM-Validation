# PROC-LABBAND-01 — outcome: NOT COMMISSIONED. At 80 arrays a laboratory's width cannot be told from noise, and using it makes the reading worse.

**Sealed 2026-09-25** against the bars in [`PROC_LABBAND_01_PREREG.md`](PROC_LABBAND_01_PREREG.md), fixed
before any width was computed. Input: the 318 immune A″ values PROC-BAND-01 published. Nothing recalibrated.
Evidence: [`PROC_LABBAND_01.json`](../kit/results/PROC_LABBAND_01.json) ·
script [`PROC_LABBAND_01.py`](../kit/PROC_LABBAND_01.py).

## The verdict

| bar | result | |
|---|---|---|
| **B1** widths differ (ratio > 1.20 **and** permutation p < 0.01) | ratio **1.399** — but **p = 0.123**, and the null's own median ratio is **1.245** | **NOT MET** |
| **B2** a width from 80 arrays is stable | split-half medians 0.984–1.001, all inside [0.80, 1.25] | MET |
| **B3** out-of-sample tail in [0.03, 0.07] | per-laboratory **0.075, 0.075, 0.103, 0.100** — every laboratory outside, and **worse than pooled on the same halves** for three of four | **NOT MET** |
| **B4** sensitivity ≥ 0.80 | 0.31–0.66 — **but the bar was arithmetically unreachable; see below** | not assessable as written |
| **B5** the reading does not move | not run — the procedure recomputes nothing, but see the note | **not assessed** |
| **B6** a laboratory without a band is servable | not reached, B1 having failed | — |

**Decision rule, applied:** B1 fails, so *"the laboratories are alike; the pooled band stands."* No
per-laboratory width is adopted. The pooled band remains the band.

## What the numbers say, and it is worth more than a pass would have been

**The observed spread of widths is ordinary.** The ratio of widest to narrowest is 1.399 — which sounds
substantial until you shuffle the laboratory labels 5,000 times and find that the *typical* ratio between
four groups of 80 drawn from one distribution is **1.245**, with 1 in 8 shuffles exceeding what was
observed. Four laboratories that are genuinely identical would look about this different at this sample
size.

**And acting on those widths makes the reading worse.** Fitting σ on half a cohort and measuring the tail on
the other half, the per-laboratory width gives tails of 0.075–0.103 where the pooled width on the *same
halves* gives 0.025–0.128 — better for three laboratories out of four. This is the same shape of result as
PROC-MAHA-03's chip term: a correction estimated from thin data contributes more error than the effect it
removes. A width from 40 arrays is noisy, and dividing by a noisy width is worse than dividing by a
well-measured average.

**None of this says the band file's 4.4–9.8 % spread is imaginary.** Those tails were measured on the
**full cohorts** — 204 to 659 arrays each — where a 1.4× ratio would not be ordinary at all. What this
procedure establishes is narrower and more useful: **the 80-array published panels cannot resolve the
question**, and anyone who tries on them will get a width that hurts.

## Two admissions

**B4's bar was unreachable by arithmetic, and that is my second such error today.** A departure of +2σ
injected against a |z| > 1.96 threshold can only be detected in about half of healthy readings — a reading
that starts below the median ends below the line. The expected rate is ≈ 0.52, so a bar of 0.80 could never
have been met by any band, good or bad. It is reported as **not assessable** rather than as a failure of the
change, and it is not moved. A correct sensitivity bar injects at a level where the *pooled* band already
achieves the target, and then asks whether the per-laboratory band keeps it — the comparison that is in the
evidence file and is informative: per-laboratory detection beat pooled for three laboratories (0.662 vs
0.650, 0.650 vs 0.525, 0.525 vs 0.512) and lost badly for Karolinska (0.308 vs 0.500), whose width is the
widest.

**B5 was not run, and is recorded as not assessed rather than passed.** This procedure reads a published
table and recomputes no reading, so A″ cannot have moved — but that is an argument, and PROC-CLS-01 has
already shown today what happens when an argument is written into a verdict column. It is marked
**not assessed**.

## The finding that spans three procedures

Three procedures today, three different questions, one blocker:

| procedure | what it needed | why it failed |
|---|---|---|
| PROC-BAND-01 | an age curve for the joint component | fitted on 234 arrays where immune's had 1,379; came out non-monotone |
| PROC-CLS-01 | a transferable spectrum reference | per-laboratory residual scales, each from a 40-array panel |
| PROC-LABBAND-01 | a per-laboratory width | 80 arrays cannot distinguish 1.4× from noise |

**The single thing that unblocks all three is the same: Stage 1 on the full cohorts.** 1,379 arrays across
the four laboratories, which is what PROC-PANEL-03 used for the immune curve and what the band file's own
per-laboratory tails were measured on. That output **was not preserved** — the Reference tab of every report
says so — and the 80-array panels published in `reference_data/` are what remains.

The cost is real and worth stating plainly: roughly 1,400 IDAT pairs, about 15 GB of download and a day of
calibration. Against that: it is the prerequisite for a second axis on the departure statistic, for a
transferable sky reference, and for settling whether laboratories need their own widths. Three items on the
enhancement list, one input. **That is a recommendation, and the decision is the author's.**
