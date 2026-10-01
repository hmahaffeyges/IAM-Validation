# PROC-PREDX-NEUT-01 — outcome (2026-10-01). Pre-registration: PROC_PREDX_NEUT_01_PREREG.md (sha 97fdf967972519ef).

845/845 EPIC-Italy arrays through our Stage 1; held out (not in GSE51057): 247 controls, 89 breast, 162 colorectal. Neutrophil fraction ≥ 0.30 in
all but one. Met-A, neutrophils only, on the 450K floor from purified 450K neutrophils (GSE88824).

| prediction | result | verdict |
|---|---|---|
| P1 ≥ 80 % of controls in Normal | 167/246 = **67.9 %** (median A 0.986) | **FAIL** |
| P2 breast > 8 y outside Normal more than controls (and within women) | 18/40 vs 79/246, p = 0.080; women only: 18/39 vs 68/162, p = 0.38 | **FAIL** |
| P3 colorectal > 8 y (and within each sex) | 9/46 vs 79/246, p = 0.97; F p = 0.81; M p = 0.51 | **FAIL** |

**What drives the reading: sex.** Among controls, women read 0.961 and men 1.007 (difference −0.045, p = 9e-10);
women in Normal 58%, men 87%. Age adds a smaller trend (ρ = 0.23). Breast cases (all women) read like
female controls at every lead time, so the pooled breast–control difference is a sex difference, not a pre-diagnostic signal.

**Consequences.**
1. The 450K floor fits men and not women: the floor needs its donors' sex stated, and a floor from purified female neutrophils is required
   (a reference standard, not a population). Identity loci on chrX are 0.2 % of the neutrophil set, so this is not simply X inactivation.
2. Any earlier breast-versus-mixed-sex-control comparison on Met-A carries this confound and must be re-read within women.
3. Comparing like with like (breast cases against female controls; colorectal cases against controls of the same sex), no pre-diagnostic
   neutrophil shift is seen at any lead time. No sex-specific floor was applied here; that is the next test.

## Follow-up (development diagnosis, same day): is the sex gap neutrophil biology?
Donor sex in GSE88824 inferred from chrY/chrX probes: L1, L2, L4 male; L3, L5–L8 female.
- Purified neutrophils: women 0.932, men 0.940, a gap of **0.008**. Sex-specific floors (F 0.78159, M 0.78816) move EPIC-Italy controls only to
  69.1 % Normal (women 0.964, men 1.002). The 0.045 gap in EPIC-Italy is **not** explained by neutrophil methylation differing by sex.
- Within each sex, A rises with the solved neutrophil fraction (ρ ≈ 0.3), so part of the spread is the separation step, not the cell.
- Remaining candidates: (a) centre/plate, since EPIC-Italy women and men were recruited largely at different centres; (b) composition residual
  (other cells' atlas profiles not matching these donors). Test next: read A against array plate (sentrix ID from the IDAT names) within sex, and
  re-read with the composition solver's residual carried as an uncertainty.
- Re-read after sex floors: breast > 8 y vs female controls p = 0.43; colorectal > 8 y p = 0.99. No pre-diagnostic signal.
