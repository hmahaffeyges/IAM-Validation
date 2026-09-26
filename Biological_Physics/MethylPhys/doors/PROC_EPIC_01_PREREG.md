# PROC-EPIC-01 — does the commissioned chain see a pre-diagnostic immune signal in genuinely held-out EPIC-Italy blood?

**Pre-registered 2026-09-26, before any array was scored.** The first test of the commissioned chain against
disease data, on the cohort the author has held since the pre-atlas era — and on a split that did not exist
before today.

## Why this split, and what was wrong with the old one

VAL-047 Phase 9 reported an immune-class elevation in pre-diagnostic blood on GSE51057 (n = 329), loudest at
long lead times. Phase 12 then tested it on GSE51032 (n = 845), describing that as *"a SECOND cohort"* and an
*"independent breast cancer cohort"*.

**Measured today from the accessions rather than the labels: all 329 GSE51057 samples are inside the 845.**
The two are overlapping GEO series over the same EPIC-Italy participants, and Phase 12's script contains no
overlap exclusion. By case type:

| group | in the discovery 329 | held out | overlap |
|---|---|---|---|
| breast (C50) | 146 | **78** | 65 % |
| colorectal (C18–C20) | 4 | **72** | 5 % |
| controls | 177 | **247** | 42 % |

So the breast replication re-used two thirds of its own discovery cases; the colorectal arm was essentially
independent already. **This procedure uses only the held-out samples** — the 516 never seen by Phase 9.

**Nothing here revisits the old result.** That was measured on the pre-atlas marker-union surface, which
moves in the opposite direction with age from the identity gauge (RECON D2), and it stays frozen where it is.
This asks a new question of the new instrument.

## What enters

| | |
|---|---|
| Cases | **78 breast (C50)** and **72 colorectal (C18/C19/C20)**, all female, all held out |
| Controls | the **163 female** held-out controls. The 84 male controls are **excluded**: every case is female, so including them would confound sex with disease status |
| Lead times | breast median 7.40 y (0–2 y: 8 · 2–5: 16 · >5: 54 · >8: 36 · >10: 22); colorectal median 5.95 y (>5: 40) |
| Scored by | the commissioned chain as a report runs it — scale map, deconvolver, identity gauge, **composition guard active** |
| Platform | 450K buffy coat, from the series matrix already on the author's disk. **No download** |

**No laboratory zero exists for EPIC-Italy**, so the chain will withhold absolute placement and tiers, which
is correct and will be reported as such. It does not obstruct this test: the zero is a constant subtracted
from every array in a laboratory, so it **cancels exactly** in any case-versus-control comparison. The
quantity compared is therefore

&nbsp;&nbsp;&nbsp;&nbsp;`A' = A_mapped − c(age)`

using the **commissioned** age curve, so the comparison is age-corrected and the unknown zero drops out.
Cases and controls have similar median ages (51.8 / 54.1 against 54.2), but the correction is applied
regardless.

## The bars, fixed now

**B1 — the primary test.** Breast cases with lead time **> 10 years** (n = 22) against the 163 female
controls. Direction is **pre-specified as elevation**, because that is what Phase 9 claimed; a depression of
any size is a failure, not a finding. Requires **Cohen's d ≥ 0.5** and a one-sided permutation
**p < 0.01** over 5,000 shuffles of the case/control labels.

**B2 — the temporal shape.** Phase 9's claim was not merely "elevated" but "loud far from diagnosis, quiet
near it". Requires **d(> 8 y) − d(0–2 y) > 0**. If the effect is flat in lead time, the signal may be real
but the story about it is not.

**B3 — the cross-cancer arm.** Colorectal cases with lead > 5 years (n = 40) against the same 163 controls,
same direction, permutation **p < 0.05**. This is the arm that was already independent, and it is the one
that would say the immune class registers drift from a different tissue of origin.

**B4 — a negative control that must be quiet.** The 163 female controls split at random into two halves,
2,000 times: the median |d| between halves must be **< 0.20**. If healthy blood split at random produces
effects of the size we are claiming, the effect is batch or age structure and nothing else.

**B5 — the guard must not be confounded with disease.** The composition guard (PROC-FOREIGN-01) will withhold
tiers from some arrays. Its withholding rate in cases and in controls must differ by **less than 2×**. A
guard that fires preferentially on cases would silently remove the very specimens under test — and would also
be an interesting result in its own right, so it is reported either way.

**B6 — the instrument has not moved.** Immune A″ on the eleven commissioning arrays must be unchanged from
the values sealed in PROC-E2E-01, **max |ΔA″| = 0**, verified by recomputation.

## Decision rule

- **B1, B2, B4 met:** the commissioned chain reproduces a pre-diagnostic immune signal on data it has never
  seen, in a split with no overlap. That is a finding worth a paper section — and still not a clinical claim:
  n = 22 at the primary stratum, one cohort, one platform, no prospective validation.
- **B1 fails:** published as measured. The old signal was found on a different surface with overlapping
  samples; if the identity gauge does not see it in held-out blood, that is the single most important thing
  to know before any cohort work, and it is exactly why this is being run before new data is sought.
- **B3 met but B1 not:** the cross-cancer arm stands alone and the breast claim is withdrawn pending more
  cases.
- **B4 fails:** nothing is reported from this cohort at all until the cause is found. An effect that appears
  between two halves of the same healthy group invalidates every comparison above it.
- **B5 fails:** the guard's behaviour is reported and the primary analysis is repeated on guard-passing arrays
  only, with both results published.

No result here changes what the chain reports. Disease evidence for the commissioned chain remains Issue 004,
after sealed runs.
