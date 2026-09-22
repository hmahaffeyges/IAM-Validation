# PROC-MAHA-03 — the Sentrix-chip term, and which panel protocol recovers it

**Pre-registered 2026-09-22, before the run.** Row 5b of CHAIN_COMMISSIONING has been OPEN with its bar already
fixed by PROC-MAHA-02 — *a reference on the chip brings every laboratory's p95 tail to ≤ 0.05* — and with the
blocker stated as *"panel-protocol design decision precedes it"*. This procedure makes that decision a
measurement instead of a choice.

## The question

Every reading the chain reports carries the laboratory's own false-alarm rate beside it, and across the four
commissioned cohorts that rate spans **0.0441 to 0.0984 at p95** where 0.05 is nominal. The excess is attributed
to the Sentrix chip. So: **does a chip term exist in the healthy data, and what does a laboratory have to do at
bench level to remove it — a control array on every chip, or a healthy panel spread across chips?**

## Inputs

Three commissioned healthy whole-blood cohorts whose Sentrix barcode is recoverable from the GEO series matrix
(the per-sample supplementary IDAT filenames):

| cohort | arrays | chips | samples per chip (min/median/max) |
|---|---|---|---|
| GSE87571 Uppsala | 732 | 62 | 9 / 12 / 12 |
| GSE111629 UCLA | to be counted in the run | | |
| GSE125105 Munich | to be counted in the run | | |

**GSE42861 Karolinska is excluded, and it is the cohort with the worst tail (0.0984).** Its series matrix carries
no Sentrix barcode, so the chip cannot be recovered without the raw IDATs. That is a limitation of this
measurement and is to be stated in the outcome, not worked around.

Measurement path — the existing scoring path, nothing new: per array, β̄ over the 42,134 immune identity loci of
`iamatlas_gauge_identity_loci_v1_0.json`; A_mapped = H(β̄)/H_min with H_min = 0.838889; then
A″ = A_mapped − c(decade) − z_lab with c from `reference_age_curve_v1.json` and z_lab from
`identity_band_v3.json`. One axis (immune), so the p95 threshold is |z| > √χ²(0.95, 1) = 1.95996 with
z = (A″ − 1.000)/σ.

## Bars, fixed now

**B1 — the chip term is real.** The between-chip variance component of A″ exceeds its permutation null (1,000
shuffles of chip labels within cohort) at p < 0.01 in at least two of the three cohorts. The intraclass
correlation is reported whatever the outcome.

**B2 — protocol A, one control array per chip.** Correcting each array by a *single held-out* reference array's
A″ on its own chip brings tail_p95 ≤ 0.05 in every cohort.

**B3 — protocol B, a panel spread across chips.** The same with k held-out reference arrays per chip for
k = 2, 3, 5. The commissioned requirement is the smallest k that meets ≤ 0.05 in all three cohorts.

**B4 — the correction must not erase a real departure.** With a shift of +2σ injected into one array, the chip
correction under the chosen protocol must leave at least 80 per cent of it standing (corrected |z| ≥ 1.568).
A chip term that absorbs the signal it is meant to clean is worse than none.

**B5 — held-out estimation only, and no silent correction.** The chip offset is estimated from reference arrays
only, never from the array being read. A chip carrying fewer than k reference arrays reads UNSET for the chip
term rather than being corrected on thin evidence.

**B6 — the improvement must not be a rescaling.** Chip-centring narrows the healthy spread, so judging a
corrected reading against the *uncorrected* band's σ would lower the tail arithmetically and prove nothing. The
≤ 0.05 bar must be met with σ **re-derived from the corrected healthy distribution** (σ = (p90 − p10)/(2 × 1.2816)
on the corrected A″). Both tails are reported: against the current band σ = 0.02044, and against the re-derived σ.
Only the re-derived one counts.

## Decision rule, fixed now

- **B1 fails** → the chip term is not demonstrable on public data. Row 5b closes NOT COMMISSIONED with that
  statement, the false-alarm rates stand as they are, and the manual keeps printing them per laboratory.
- **B1 and B4 hold and B2 or B3 meets ≤ 0.05 on the re-derived σ** → row 5b is commissioned with the cheapest
  protocol that met the bar, written into the runbook as a bench requirement, and the chip term is wired into
  Stage 5 behind the same UNSET discipline as the laboratory zero.
- **B1 holds but no protocol meets the bar** → row 5b stays OPEN with the measured ceiling stated: this is how
  much of the excess tail the chip explains, and it is not enough.

Nothing is written into the chain before the outcome is sealed. The analyst recommends; only the author
commissions.
