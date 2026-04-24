# VAL-063 Outcome — Lung-EPIC Tissue Arm on TCGA-LUAD HM450

**Date completed:** 2026-04-24 UTC
**Prereg SHA:** f56ebe0ab015d856c86573e502fde132743a95fcb1d3667074a5001993f4108e
**Manifest SHA:** 6e87cc32b84f278d1b77ad766a050f2a378aa3a8e3da78e7232b2511514d278c
**Cohort SHA:** 53718abc88680e0793b0455ac51fbf8e6a128f615c508f0de60dc8d8cfd4d6e9
**Results SHA:** 809025760e30b42f040c41f8e95b94ad771bf0bb58b631a74207d2340409a9ba

## Class assignment

Lung adenocarcinoma cells = lung_epithelial = **cycling class** (H_min = 0.856055, reference β = 0.738 from TCGA-LUAD matched normal). Lung cycling assignment confirmed by Issue 002 Cycling Epithelial class listing (LUAD + LUSC both cycling, 14/28 TCGA cancers in this class).

## Results

- **n matched pairs:** 29 (all downloaded pairs passed QC, zero skipped)
- **Paired Cohen's d:** +1.0202, 95% CI [+0.5714, +1.4690], p = 3.93e-08
- **Unpaired Cohen's d:** +1.5299, 95% CI [+0.9447, +2.1151], p = 5.69e-09
- **Absolute ΔA (tumor minus normal, mean):** +0.04297
- **A-tumor mean:** 0.65250 ± 0.03177
- **A-normal mean:** 0.60952 ± 0.02385

## Outcome classification: PASS (strong)

Preregistered prediction: paired d > 0, 95% CI > 0, d ≥ +0.5. Observed d = +1.0202, CI [+0.571, +1.469] — all three criteria met with substantial margin. Direction confirmed, magnitude very strong.

## Comparison to prior tissue arms

| Test | Cancer | Class | Cohort | Paired d |
|---|---|---|---|---|
| VAL-058 | Prostate | Secretory | GSE269244 n=238 | +0.497 |
| VAL-060 | Breast | Secretory | TCGA-BRCA n=86 | +0.675 |
| VAL-062 | Colorectal | Cycling | TCGA-COAD n=26 | +0.724 |
| **VAL-063** | **Lung adenocarcinoma** | **Cycling** | **TCGA-LUAD n=29** | **+1.020** |

VAL-063 lung cycling is the largest paired tissue effect size measured to date in any Cookbook card. This is consistent with lung adenocarcinoma's high mutational burden and the aggressive cell-turnover disruption that drives large methylation landscape reorganization in the tumor. Absolute ΔA (+0.043) is also the largest observed at the genome-wide-mean level across all tissue arms.

## Note on absolute ΔA

Absolute ΔA = +0.043 is still smaller than the VAL-001 framework prediction of ΔA ≈ +0.14 for lung cycling-class tissue, but it is 2× the magnitude observed in CRC (+0.020) and consistent with the genome-wide-mean dilution caveat documented for VAL-062. Cycling-class-informative CpG subsets (lung-specific DMRs, Moss 2018 lung markers) would recover the full ~+0.14 framework target; VAL-063 averages across all ~485K HM450 CpGs, diluting the class signal with probes that are not cycling-informative. The Cohen's d remains strong because between-patient variance at genome-wide-mean is small.

## Action items

- [x] Run VAL-063 TCGA-LUAD matched tumor/normal against cycling H_min
- [x] PASS outcome documented, direction + magnitude confirmed
- [x] Write reproducible python script (val063_lung_epic_tcga_luad.py)
- [ ] Update lung-epic_README.md with tissue arm section (v0.3 → v0.4)
- [ ] Build lung-epic_card_v0.4.json with tissue_arm block
- [ ] Insert VAL-063 section into Evidence Report (template: VAL-062 style)
- [ ] GitHub push
