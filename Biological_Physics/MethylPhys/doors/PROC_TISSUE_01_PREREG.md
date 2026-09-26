# PROC-TISSUE-01 — does the gauge place healthy, adjacent-normal and tumour colon in order, without being shown the order?

**Pre-registered 2026-09-26, before any array was downloaded.** Written on the author's framing, which sets
the bar for what counts: *we do not need to prove Landauer in the methylome; we need to show what it means to
use Landauer metrology.* "Tumour differs from normal" is not that. An instrument that returns a **distance**
rather than a label should place three states in order having been trained on none of them — and should read
something in tissue that still *looks* normal.

**Cohort:** GSE131013 — healthy, adjacent-normal and tumour colon cells, n = 240, Illumina 450K, one series
and one laboratory. Chosen because all three groups sit in the same series, so the laboratory constant, the
platform and the processing are held fixed and cancel in every comparison below.

## What is fixed before the data is read

**The class that will be scored is decided from the healthy group alone.** After intake, the deconvolver's
median class fraction is computed on the **healthy specimens only**, and the architecture class with the
largest median fraction is the class scored for every group. This is written down now so the class cannot be
chosen after seeing which one separates.

**The quantity is A_abs = H(mean β over that class's identity loci) / H_min**, the chain's own definition,
with no laboratory zero and no age curve subtracted. A zero is a per-laboratory constant and this is a
within-series comparison, so it cancels exactly — the same cancellation that made PROC-EPIC-01 runnable
without an EPIC-Italy zero. Tiers will be withheld by the chain, correctly, and are not used.

**The direction is fixed: healthy < adjacent normal < tumour.** Higher A means more entropy at the identity
loci relative to the physical floor, i.e. *less* fidelity. A reversal at any rung is a failure, not a finding.

**Groups come from the series metadata**, parsed and frozen to a file before a single A is computed.

## The bars

| | bar | met when |
|---|---|---|
| **B1** | **the ordering** | median A_abs strictly increases healthy → adjacent normal → tumour |
| **B2** | **the field effect** — the result that licenses a whole-surface sample | adjacent normal vs healthy: d ≥ 0.5 **and** permutation p < 0.01 (5,000 label shuffles) |
| **B3** | the disease contrast is real | tumour vs healthy: d ≥ 1.0 and p < 0.001 |
| **B4** | a null that does not know the answer | healthy split at random 2,000×: median \|d\| < 0.20 |
| **B5** | age is not the cause | if median group ages differ by > 5 years, B1 and B2 must also hold on an age-matched subset |
| **B6** | the specimens are what they claim | median epithelial-class fraction > 0.50 in the tissue groups; composition-guard withholding rate reported for every group |
| **B7** | the instrument has not moved | max \|ΔA_mapped\| = 0 on the **318 arrays published in `kit/results/PROC_BAND_01_arrays.json`** — a source checked today to contain per-array readings |

B7 names its evidence file explicitly because PROC-EPIC-01's equivalent bar named a sealed file that holds
exit codes and timings but no readings, and could not be run as written. Checking what a bar's named source
actually contains is now part of writing one.

## Known risks, stated in advance rather than discovered later

**The scale map was commissioned on blood, not tissue.** `stage_1s_scale_map` is the chain's own mapping and
has never been exercised on colon epithelium. This does not invalidate a within-series comparison — the same
map is applied identically to all three groups, so a constant offset cancels — but it does mean **no absolute
placement may be quoted from this procedure**, only differences between groups in the same series. If the map
behaves non-linearly across the β range the groups occupy, that is a confound this design cannot exclude, and
it will be reported as a limitation whatever the outcome.

**The identity bands are commissioned for immune, not for epithelial classes.** No band, tier or placement is
used here; only A_abs differences. Nothing in this procedure can license a per-patient reading.

**Specimen purity is assumed and must be verified.** The series describes "cells", which may mean sorted
epithelium or may mean bulk tissue. B6 tests it rather than trusting it, and if the epithelial fraction is
below 0.50 the specimens are bulk tissue and the procedure reports that instead of an ordering.

## Decision rule, fixed now

- **B1–B4 met:** the instrument orders three states it was not trained on, and reads a departure in tissue that still looks normal. That is the field-effect result, and it is what would justify asking anyone to collect stool-derived colonocytes — the whole-surface sample only means something if a surface away from the lesion carries signal. It licenses PROC-TISSUE-02 (the adenoma rung) and the collaboration ask. It does **not** license a clinical claim: no threshold is fixed here, no out-of-sample performance is measured, and a group separation is not a detection.
- **B1 met, B2 not:** the gauge separates tumour from healthy but reads nothing in normal-looking tissue. The ordering stands; the stool route does not follow from it, and should not be pursued on this evidence.
- **B2 met, B1 not** (a reversal at the tumour rung): reported as-is and investigated, not rescued. A non-monotone reading would say the quantity is not a simple distance in this tissue, which is worth knowing.
- **B4 fails:** nothing is reported from this cohort. An effect that appears between two halves of the same healthy group invalidates every comparison above it.
- **B6 fails:** the composition finding is reported instead, and the ordering is not.

No new data is generated, no patient is involved, and nothing here changes what the chain reports.
