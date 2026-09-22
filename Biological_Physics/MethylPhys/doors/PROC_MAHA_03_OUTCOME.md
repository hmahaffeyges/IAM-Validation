# PROC-MAHA-03 — outcome: the chip term is real where it matters most, and a control array per chip is the wrong protocol

**Run 2026-09-22 against the bars pre-registered the same day** (`PROC_MAHA_03_PREREG.md`). Row 5b closes
**NOT COMMISSIONED** by its own decision rule, and the run says something more useful than that sentence alone.

## What was measured

318 healthy arrays from the four commissioned cohorts, Stage 1 noob-calibrated from IDATs, on the commissioned
scale — `beta_roadmap = (beta − 0.0662)/1.0127` applied before H per `beta_scale_maps_v1.json`'s own rule — then
A″ = H(β̄)/0.838889 − c(decade) − z_lab. Sentrix chip from the IDAT filenames. **What validates the path:** all four cohort medians land on the healthy line - 1.0052, 1.0000, 0.9985, 0.9981 -
which is the check that the scale map, the age curve and each laboratory's zero are wired correctly.

**What does not:** the tails measured on these 80-array panels do NOT reproduce the published ones, and the
ordering inverts.

| cohort | tail on this panel (n=78-80) | published tail (full cohort, n=204-659) |
|---|---|---|
| GSE42861 Karolinska | 0.1154 | 0.0984 |
| GSE111629 UCLA | 0.0625 | 0.0441 |
| GSE87571 Uppsala | 0.0625 | 0.0561 |
| GSE125105 Munich | 0.0375 | 0.0647 |

Munich goes from second-highest published to lowest here, and UCLA and Munich differ from their published
values by about 40 per cent of them. A tail is the fraction beyond p95, so on 80 arrays it is roughly four
arrays: the binomial standard error at 0.05 is about 0.024, which is half the whole spread between these four
laboratories. All four differences are within about one standard error, so the panels are not inconsistent
with the published rates - but they cannot resolve them either, and nothing here should be read as the
measured tails confirming the published ones. Only Karolinska's excess is large enough to survive that noise,
which is the one place this run draws a conclusion from a tail.

**Deviation from the pre-registration, and it is an improvement:** the run uses Stage 1 betas from IDATs rather
than the GEO series matrices, because those matrices are header-only for these cohorts. IDAT filenames carry the
barcode, so **GSE42861 Karolinska is included** — the pre-registration had excluded it, and it turned out to be
the decisive cohort.

## B1 — the chip term. FAILS the bar as written; passes pooled

| cohort | arrays | chips | chips with ≥2 | ICC | sd between | sd within | F | p (2,000 shuffles) | measured tail |
|---|---|---|---|---|---|---|---|---|---|
| GSE42861 Karolinska | 78 | 38 | 24 | **0.409** | 0.0160 | 0.0193 | 2.843 | **0.0030** | 0.1154 |
| GSE87571 Uppsala | 80 | 42 | 28 | 0.193 | 0.0114 | 0.0232 | 1.565 | 0.0975 | 0.0625 |
| GSE125105 Munich | 80 | 47 | 26 | 0.153 | 0.0079 | 0.0185 | 1.411 | 0.1794 | 0.0375 |
| GSE111629 UCLA | 80 | 47 | 24 | 0.000 | 0.0000 | 0.0197 | 0.969 | 0.5272 | 0.0625 |

The bar required p < 0.01 in at least two cohorts. **One cohort clears it.** B1 fails as written.

Pooled with chip nested in cohort (cohort mean removed first), the term is there: **ICC 0.208, F 1.631,
p = 0.005** on 102 chips and 246 arrays. That test was *not* the pre-registered one and is reported as a
post-hoc observation, not as a pass.

**The one result worth carrying forward:** the cohort with the worst false-alarm tail is the cohort with the
strongest chip term. Karolinska is 0.098 published, 0.115 measured here, ICC 0.409, p 0.003 — while UCLA, whose
tail is lowest, has no measurable chip term at all (ICC 0.000). That is exactly the pattern the hypothesis
predicts, in the two cohorts that matter most for it.

## B2/B3 — the protocols. A control array per chip makes the reading WORSE

Held-out estimation only, per B5: the offset never sees the array being read.

| cohort | k=1 tail before → after (band σ) | after (re-derived σ) | k=2 |
|---|---|---|---|
| GSE111629 UCLA | 0.0625 → 0.0877 | 0.0877 | under 30 arrays, not assessable |
| GSE125105 Munich | 0.0375 → 0.1186 | 0.1017 | not assessable |
| GSE87571 Uppsala | 0.0625 → 0.2273 | 0.1061 | not assessable |
| GSE42861 Karolinska | 0.1154 → 0.1719 | **0.0625** | **0.0312** (32 arrays) |

**Protocol A — one control array per chip — is counterproductive, and the reason is arithmetic.** Correcting by a
single array subtracts that array's own within-chip error along with the chip offset, so it adds variance of
sd_within ≈ 0.019–0.023 while removing an offset of 0.000–0.016. It only pays where the chip term dominates the
within-chip noise, which of these four cohorts is true only in Karolinska.

**Protocol B needs k ≥ 2, and this data cannot measure it.** Only Karolinska has enough chips carrying three or
more arrays, and there k = 2 took the tail from 0.115 to **0.031** — under the 0.05 bar. One cohort is not four.

No protocol met the pre-registered bar in every cohort, so B2 and B3 both fail.

## B4 — the erasure test was ill-posed, and saying so is the result

Recovery measured 100.0 per cent at every k. That is arithmetic, not evidence: a held-out additive offset is
computed from other arrays and cannot contain the array's own departure, so the injected shift survives exactly.
The erasure risk the bar was written for belongs to the estimator **B5 forbids** — a chip median that includes the
array being read — and that estimator is not used. B4 is withdrawn as a test of anything; B5 is what protects
against erasure, and it is structural rather than measured.

## A discrepancy to resolve, stated rather than resolved

`identity_band_v3.json` carries a column `tail_p95_if_chip_centred` reading 0.0196–0.0413, which is where the
expectation of "chip-centring halves the tail" came from. **This run does not reproduce that improvement under
held-out estimation** — with k = 1 the tail rises in all four cohorts. A chip median computed *including* the array
being read would produce exactly that kind of apparent improvement, because each array is then partly centred on
itself. Whether that is what the column did is not recorded in the file, so it is not asserted here. Until it is
established, that column should not be read as what a bench protocol can deliver, and the manual has been
corrected accordingly.

## What was rejected, and why it is in this record

A second arm was designed on GSE87571's full 732 arrays across 62 chips (9–12 per chip) using the depositors'
supplementary beta matrices, which would have given B1 real power. **Its pre-registered verification failed:** the
matrices' column order does not correspond to the series-matrix sample order (r = 0.032 between the two sources
on the 80 arrays whose GSM is known, against 0.065 for deliberately shuffled columns; offset −0.41; within-chip
spread 0.44 against 0.02 on the commissioned scale). The arm was discarded unused. The check is the only reason a
plausible-looking ICC from mis-aligned columns is not in this document.

## Decision, per the rule fixed in advance

**Row 5b: NOT COMMISSIONED.** The chip term is not demonstrable across cohorts at this chip depth, no protocol
brought every laboratory's tail to ≤ 0.05, and the four laboratories' false-alarm rates stand as they are and
continue to be printed per laboratory beside every departure.

**What would settle it, with its cost.** Chip depth is the binding limit: the 80-array Stage 1 panels give a
median of 2 arrays per chip, where 8–12 are needed. The measurement that decides it is Stage 1 on **all 732
Uppsala IDATs** — 62 chips at 9–12 arrays each on the commissioned scale — roughly 2.5 GB of IDATs and a few
hours of calibration. That run would test B1 with power and measure k = 2, 3, 5 properly. Until it exists, the
honest statement is the one above: one cohort in four, and it is the worst one.

**Recommendation to the author, not a commissioning:** if a laboratory is to spend arrays on chip control, spend
them in pairs — two panel arrays per chip, never one — and only where the chip term has been measured for that
laboratory. A single control array per chip would make its readings worse than no chip correction at all.


---

# Deep-chip arm, 2026-09-22 — the chip term IS real at depth, and the correction still does not pay

The outcome above named the measurement that would settle row 5b: Stage 1 on the GSE87571 IDATs, where the chips
carry 9–12 arrays each instead of two. It has now been run on **23 complete chips, 268 arrays**, all on the
commissioned scale. The remaining 39 chips are a continuation, not a different experiment: the calibration caches
per array and resumes where it stopped.

## B1 — met, and the thin panel was simply underpowered

| | 80-array panel (28 chips, ~2 arrays each) | 23 deep chips (9–12 each) |
|---|---|---|
| chip ICC | 0.193 | **0.197** |
| between-chip SD | 0.0114 | 0.0100 |
| within-chip SD | 0.0232 | 0.0201 |
| F | 1.565 | **3.862** |
| p (2,000 shuffles) | 0.0975 | **0.0005** |

The effect size barely moved — 0.193 to 0.197 — while the p-value fell by more than two orders of magnitude.
That is what a power problem looks like when you fix it: the panel had the right answer and could not prove it.

**B1's bar is therefore met**: p < 0.01 in two cohorts (GSE42861 at shallow depth, GSE87571 at depth).

The tail also lands closer to the published value with depth — 0.0597 here against 0.0561 on the full cohort,
where the 80-array panel read 0.0625. The median sits at 0.9968.

## B2/B3 — still not met, and the k = 1 row shows why the bar needed its second clause

| k | tail on the band σ | tail on a σ re-derived from the corrected values | σ re-derived |
|---|---|---|---|
| uncorrected | 0.0597 | — | 0.02084 |
| 1 | 0.1306 | **0.0485** | **0.02816** |
| 2 | 0.0896 | 0.0560 | 0.02408 |
| 3 | 0.0597 | 0.0597 | 0.02033 |
| 5 | 0.0597 | 0.0709 | 0.01955 |

**The k = 1 row reads as a pass and is not one.** Correcting each array by a single held-out reference on its chip
makes the tail on the band σ more than twice as bad (0.0597 → 0.1306) while *inflating the spread by 38 per cent*
(0.0208 → 0.0282). Re-deriving σ from that inflated distribution divides the departures by a larger number, and
the tail duly falls under 0.05. Nothing improved: the yardstick grew with the noise.

That is a flaw in how B6 was written, and it is recorded rather than exploited. B6 required the bar to be met
against a re-derived σ, to stop a corrected reading being judged against a σ narrower than its own distribution.
It did not also require that the correction **not inflate** the spread. The complete rule, for any future
procedure of this kind:

> A correction may be credited only if the tail falls **and** the spread does not grow. A tail computed against a
> σ that the correction itself widened is not evidence of anything.

Under that rule no k passes. k = 3 and k = 5 hold the spread roughly constant and leave the tail at 0.0597 —
exactly where it started. The arithmetic is visible in the two SD columns of the B1 table: the offset being
removed (0.0100) is half the noise within a chip (0.0201), so an estimate of it from a handful of arrays
contributes about as much error as it subtracts.

## Decision — unchanged verdict, inverted reason

**Row 5b remains NOT COMMISSIONED.** But the reason is no longer that the chip term cannot be demonstrated: at
depth it is demonstrated decisively, p = 0.0005 in the largest cohort. The reason is now that **correcting for it
does not reduce the false-alarm tail**, at any panel size this data can test, once the spread is held honest.

The four laboratories' false-alarm rates stand as published and continue to be printed per laboratory beside
every departure. The bench recommendation from the first arm is withdrawn rather than strengthened: spending
arrays on chip control is not supported by this measurement at any k, and a single control array per chip is
actively harmful.
