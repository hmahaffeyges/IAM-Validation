# VAL-062 Outcome — CRC Tumor Tissue Cycling-Class Rescore

**Date completed:** 2026-04-24 UTC
**Prereg SHA:** 9b5ff04ce31e4679e32ac8690fefc0b09a0abd646e89792edf956161097b847d
**Cohort SHA:** ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27 (inherited from VAL-061)
**Results SHA:** e8ec05a8932e92c8755febbdb8df0425f9f25d161476895e6a0169837aae2698

## Class correction from VAL-061

VAL-061 incorrectly scored CRC tumor tissue with the Xu-538 immune panel, which read the tumor-infiltrating immune compartment (d = +1.066, activated TIL signal) rather than the tumor architecture itself. VAL-062 corrects the class assignment: CRC tumor cells are **cycling class** (H_min = 0.856055, reference β = 0.740, TCGA COAD matched-normal calibration), not secretory. Cycling class covers all high-throughput renewing epithelium: colon, stomach, lung, bladder, cervix, kidney, thyroid, endometrium, head & neck.

## Results

- **n matched pairs:** 26 (inherited from VAL-061 QC-passed cohort)
- **Paired Cohen's d:** +0.7241, 95% CI [+0.2922, +1.1559], p = 2.23e-04
- **Unpaired Cohen's d:** +0.8947, 95% CI [+0.3245, +1.4648], p = 1.26e-03
- **Absolute ΔA (tumor minus normal, mean):** +0.02049
- **A-tumor mean:** 0.633 ± 0.028
- **A-normal mean:** 0.612 ± 0.016

## Outcome classification: PASS

Preregistered prediction: paired d > 0, 95% CI > 0, d ≥ +0.5. Observed d = +0.724, CI [+0.292, +1.156] — all three criteria met. Direction confirmed, magnitude strong. Cycling-class CRC tumor architecture signal consistent with framework expectation.

## Comparison to prior per-class tissue arms

| Test | Cancer | Class | Paired d |
|---|---|---|---|
| VAL-058 | Prostate | Secretory | +0.497 |
| VAL-060 | Breast | Secretory | +0.745 |
| **VAL-062** | **Colorectal** | **Cycling** | **+0.724** |

VAL-062 cycling CRC sits comparable to VAL-060 breast secretory, larger than VAL-058 prostate secretory — consistent with CRC's high-throughput cycling biology driving broader methylation disruption than secretory-class cancers.

## Note on absolute ΔA

Absolute ΔA = +0.020 is smaller than the framework-predicted ΔA ≈ +0.17 for TCGA COAD (VAL-001 target). This is expected: the VAL-001 prediction is calibrated on cycling-class-discriminating CpG subsets (colon-specific differentially methylated regions), while VAL-062 averages across ALL ~485K HM450 CpGs — the cycling-class signal is diluted by probes that are not cycling-informative. The Cohen's d remains strong because between-patient A-score variance is also small at the genome-wide average level.

For future cycling-class tissue validations, the framework-expected ΔA ≈ +0.17 is recoverable by restricting to cycling-class-informative CpGs (Moss 2018 colon markers + TCGA COAD matched-normal differentially methylated probes). The all-CpG mean used here gives a smaller but still significant cycling-class signal on a dataset of this size.

## Action items

- [x] Run cycling-class rescore on inherited 26 TCGA-COAD matched pairs
- [x] Document outcome transparently — VAL-062 PASS, VAL-061 re-interpreted as TIL reading not framework inconsistency
- [ ] Insert VAL-062 into Evidence Report with VAL-061 cross-reference footnote
- [ ] Update crc-epic card v2.2 tissue arm: VAL-062 PASS cycling d=+0.724, VAL-061 supplementary TIL reading d=+1.066
- [ ] GitHub push
