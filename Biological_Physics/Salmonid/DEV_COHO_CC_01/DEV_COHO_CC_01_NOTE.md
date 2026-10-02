# DEV-COHO-CC-01 — does a per-read conversion filter make the copy-error reading a property of the fish?

**Written 2026-10-02, before any fish in this run is scored.** This is a development reading, not a pre-registered test and not a
commissioning result. Nothing below is a finding about hatchery or wild fish.

## Why
Three earlier fish readings failed for the same reason: the per-fish copy error tracked a library-quality term instead of the fish
(Methow steelhead: ε vs conversion failure ρ −0.58; Rimouski Atlantic salmon: ρ +0.38; brook charr: duplication and conversion).
This run removes conversion failure read by read, inside each fish, with no fit across fish: copy errors are counted only on molecules that
are fully converted (≥ 3 non-CpG calls, none unconverted; `extract_se.py`, columns `*_cc`).

## Data
Le Luyer et al. 2017, PRJNA389610: 39 coho salmon smolts, RRBS, hatchery and wild, both sexes. 8 M reads per fish, Bismark on Okis_V2,
first and last 3 aligned bases ignored. RRBS reads start at MspI sites, so duplicates cannot be removed by position; duplication is not addressed here.

## Quantities (per fish)
- ε_all, ε_cc: isolated unmethylated call between two methylated calls, over opportunities, on qualifying molecules (≥ 6 CpGs, ≥ 80 % methylated).
- Common sites: CpGs with ≥ 3 conversion-filtered opportunities in ≥ 90 % of fish; ε_cc on those sites only, so every fish is read on the same sites.
- Halves A/B: alternate molecules (no half map for this set).
- Conversion failure c (non-CpG unconverted / all non-CpG calls, all reads) and A/T mismatch rate (sequencing error), from the extract log.
- E = ln((1 − ε)/ε) in kT.

## Checks, fixed now
- **D1 (repeatable within a fish):** ICC(1) of ε_cc on common sites between halves A and B ≥ 0.9.
- **D2 (library term removed):** |Spearman ρ| between ε_cc (common sites) and conversion failure c < 0.3, and also < 0.3 against
  qualifying molecules (depth). Reported for ε_all as well, to show what the filter changed.
- Only if D1 and D2 both hold: hatchery vs wild and female vs male are described (medians, Mann–Whitney p, difference in kT). They are
  descriptions on 39 fish of one study, not tests of the instrument. If D1 or D2 fails, the groups are not described at all.
