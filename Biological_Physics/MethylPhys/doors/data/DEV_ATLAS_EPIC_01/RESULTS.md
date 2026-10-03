# DEV-ATLAS-EPIC-01 — atlas v2 and NILC composition on EPIC whole blood (development, 2026-10-03)

Development measurements. No pre-registration, no verdicts. Box `methylphys-cpu-01` (128 cores), jobs 3b65d4bc (fetch + Stage 1), c1c810e6 (run 1),
595c0094 (run 2), 9941dae1 (run 3, the numbers below). Atlas: `s3://…/atlas_v2/IAMAtlas_v2.parquet` (813,951 loci x 74 cells). Chain files untouched.

## Methods compared
- **NNLS8** — NNLS8 (chain v3 Stage A) (`NNLS8`)
- **Atl-a** — Atlas a (as stored) (`ATLAS_a`)
- **Atl-b** — Atlas b (array-measured cells) (`ATLAS_b`)
- **Atl-c** — Atlas c (a + identifiability merges) (`ATLAS_c`)
- **Atl-c-blood** — Atlas c, circulating cells (`ATLAS_c_blood`)
- **Atl-e** — Atlas e (array blood cells + merges) (`ATLAS_e`)
- **NILC-c** — NILC on atlas c-blood (S1, D0) (`NILC_S1_markers_atlas_D0`)
- **NILC-c-D1** — NILC on atlas c-blood (S1, D1 offset) (`NILC_S1_markers_atlas_D1`)
- **NILC-e** — NILC on atlas e (S2, D0) (`NILC_e_S2_range0.2_atlas_D0`)
- **NILC-e-D1** — NILC on atlas e (S2, D1 offset) (`NILC_e_S2_range0.2_atlas_D1`)
- **NILC-EPIC8** — NILC on EPIC8 templates (`NILC_EPIC8_atlas+tech_D0`)

Common rules: frozen deconv_v2 SOLVER settings (hybrid markers, margin 0.10, pair margin 0.15, sigma 0.02), weights 1/(sum f^2 v + sigma^2) with
v = posterior SD^2 + donor SD^2; whole-blood cell set drops bone-marrow progenitors and the Moss vascular endothelium (as in stage_a_composition_v2);
the 6,000 neutrophil identity sites are excluded from every composition site set (chain rule). Point estimates only (no bootstrap).
NILC: W = C^-1 A (A^T C^-1 A)^-1, unit response to the target template, zero to the others; C = diag(sum f^2 v + 0.02^2) (atlas covariance, no
cohort statistic); D1 also deprojects a constant beta offset. NILC site set and covariance were chosen on GSE110554 (MIX18) only, by mean RMSE over
NEU, MONO, B, NK, CD4T, CD8T (selection tables in results/), and reported on the other sets.

Truth / test sets: FACS = GSE112618 (6 healthy whole bloods, flow counts); LONG = GSE110530 (one donor, 12 arrays, 5 with counts);
MIX18 = GSE110554 (12 Salas DNA mixtures; MIX18_blood = the 6 with neutrophils 63–75 %); MIX22 = GSE167998 (12 Salas 2022 mixtures);
MIX12 = GSE182379 (12 twelve-cell DNA mixtures); REPL = GSE250556 (63 arrays, 4 people, pooled and unpooled replicates). All Stage 1 = chain Stage 1.

## 1. Merges for variant c — what the atlas's own tests said
- Twin / cross-source rule (twin_family_thresholds_v1.json: cross_source_r 0.98, twin_r 0.985 + < 30 separating CpGs): **no merge**. The WGBS-only
  T subtypes reach at most r = 0.954 with an array-measured cell at the solver markers.
| a | b | r_markers | a_array | b_array |
|---|---|---|---|---|
| cd4 t cells | t central memory cd4 | 0.954 | True | False |
| memory cd4 t cells | t central memory cd4 | 0.944 | True | False |
| effector memory cd8 t cells | t effector cell cd8 | 0.931 | True | False |
| memory cd4 t cells | t effector memory cd4 | 0.927 | True | False |
| cd8 t cells | t central memory cd4 | 0.927 | True | False |
- Mixture identifiability (new, atlas-only): a cell is removed when a non-negative sum-1 combination of the other blood cells reproduces its
  template at r >= twin_r (0.985). On the variant-a cell set it removed, in order, `cd4 t cells`, `b cells`, `cd8 t cells` and `memory cd4 t cells`
  (the last is an array-measured cell — the rule kept the WGBS-only T subtypes and dropped it; see Failures). First iteration:
| cell | r_mixture | rms | components |
|---|---|---|---|
| cd4 t cells | 0.997 | 0.022 | {"cd8 t cells": 0.094, "memory cd4 t cells": 0.292, "naive cd4 t cells": 0.367, "regulatory t cells": 0.038, "t central memory cd4": 0.201} |
| cd8 t cells | 0.995 | 0.026 | {"cd4 t cells": 0.18, "effector memory cd8 t cells": 0.289, "naive cd8 t cells": 0.329, "t effector cell cd8": 0.201} |
| b cells | 0.994 | 0.029 | {"memory b cells": 0.294, "naive b cells": 0.675} |
| memory cd4 t cells | 0.989 | 0.037 | {"cd4 t cells": 0.425, "effector memory cd8 t cells": 0.19, "regulatory t cells": 0.319, "t central memory cd4": 0.049} |
| naive cd8 t cells | 0.984 | 0.047 | {"cd8 t cells": 0.263, "naive cd4 t cells": 0.704, "nk cells": 0.022} |
| naive cd4 t cells | 0.983 | 0.047 | {"cd4 t cells": 0.325, "naive cd8 t cells": 0.652} |
- Same rule on the array-measured blood cells only (variant e): removed `b cells`, `cd4 t cells`, `cd8 t cells` — the unsorted parent entries — and
  nothing else. Variant e = 12 cells: basophils, effector memory cd8 t cells, eosinophils, memory b cells, memory cd4 t cells, monocytes, naive b cells, naive cd4 t cells, naive cd8 t cells, neutrophils, nk cells, regulatory t cells.
| cell | r_mixture | rms | components |
|---|---|---|---|
| b cells | 0.995 | 0.030 | {"memory b cells": 0.308, "naive b cells": 0.688} |
| cd4 t cells | 0.994 | 0.029 | {"cd8 t cells": 0.023, "memory cd4 t cells": 0.547, "naive cd4 t cells": 0.389, "regulatory t cells": 0.034} |
| cd8 t cells | 0.986 | 0.043 | {"cd4 t cells": 0.088, "effector memory cd8 t cells": 0.47, "memory cd4 t cells": 0.028, "naive cd8 t cells": 0.378, "nk cells": 0.025} |
| memory cd4 t cells | 0.972 | 0.065 | {"cd4 t cells": 0.503, "effector memory cd8 t cells": 0.166, "regulatory t cells": 0.319} |
| naive b cells | 0.964 | 0.096 | {"b cells": 0.914, "basophils": 0.029, "naive cd4 t cells": 0.031} |
- The atlas means of the Salas blood cells equal the Salas arrays' own means at the markers (neutrophils mean |diff| 0.0007; largest cd8 t cells
  0.013): the atlas fit does not move the EPIC blood templates. Table: results/raw_box_outputs/diag_atlas_vs_salas_arrays.csv.

## 2. Neutrophil fraction — bias (estimate − truth); target |bias| <= 0.02
| set | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FACS | -0.030 | -0.013 | -0.007 | -0.012 | -0.010 | -0.011 | 0.003 | 0.000 | -0.005 | -0.006 | -0.000 |
| MIX18_blood | -0.053 | -0.043 | -0.045 | -0.043 | -0.046 | -0.047 | -0.035 | -0.037 | -0.031 | -0.032 | -0.031 |
| MIX18 | -0.034 | -0.018 | -0.022 | -0.018 | -0.023 | -0.024 | -0.015 | -0.017 | -0.006 | -0.007 | -0.012 |
| MIX22 | -0.005 | 0.032 | 0.017 | 0.034 | 0.013 | 0.012 | 0.011 | 0.009 | 0.029 | 0.027 | 0.016 |
| MIX12 | -0.004 | 0.002 | -0.002 | 0.002 | -0.002 | -0.004 | -0.001 | -0.008 | 0.012 | 0.008 | 0.000 |

RMSE:
| set | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FACS | 0.034 | 0.019 | 0.016 | 0.019 | 0.017 | 0.018 | 0.016 | 0.016 | 0.015 | 0.016 | 0.013 |
| MIX18_blood | 0.053 | 0.043 | 0.046 | 0.043 | 0.046 | 0.047 | 0.036 | 0.037 | 0.032 | 0.032 | 0.032 |
| MIX18 | 0.039 | 0.031 | 0.033 | 0.031 | 0.033 | 0.034 | 0.026 | 0.027 | 0.026 | 0.026 | 0.023 |
| MIX22 | 0.007 | 0.033 | 0.018 | 0.034 | 0.014 | 0.013 | 0.012 | 0.010 | 0.029 | 0.027 | 0.017 |
| MIX12 | 0.019 | 0.011 | 0.012 | 0.012 | 0.014 | 0.014 | 0.013 | 0.015 | 0.016 | 0.014 | 0.014 |

## 3. All groups — RMSE
| level_0 | level_1 | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FACS | NEU | 0.034 | 0.019 | 0.016 | 0.019 | 0.017 | 0.018 | 0.016 | 0.016 | 0.015 | 0.016 | 0.013 |
| FACS | MONO | 0.011 | 0.007 | 0.008 | 0.006 | 0.006 | 0.006 | 0.007 | 0.007 | 0.008 | 0.009 | 0.006 |
| FACS | B | 0.009 | 0.015 | 0.010 | 0.017 | 0.015 | 0.014 | 0.017 | 0.012 | 0.023 | 0.019 | 0.006 |
| FACS | NK | 0.034 | 0.029 | 0.033 | 0.024 | 0.031 | 0.038 | 0.033 | 0.035 | 0.025 | 0.029 | 0.031 |
| FACS | CD4T | 0.027 | 0.025 | 0.029 | 0.041 | 0.024 | 0.014 | 0.029 | 0.033 | 0.015 | 0.016 | 0.031 |
| FACS | CD8T | 0.030 | 0.050 | 0.047 | 0.070 | 0.065 | 0.046 | 0.075 | 0.053 | 0.058 | 0.042 | 0.044 |
| MIX12 | NEU | 0.019 | 0.011 | 0.012 | 0.012 | 0.014 | 0.014 | 0.013 | 0.015 | 0.016 | 0.014 | 0.014 |
| MIX12 | MONO | 0.012 | 0.036 | 0.031 | 0.035 | 0.029 | 0.027 | 0.024 | 0.024 | 0.036 | 0.033 | 0.015 |
| MIX12 | B | 0.032 | 0.014 | 0.018 | 0.012 | 0.013 | 0.013 | 0.010 | 0.014 | 0.014 | 0.010 | 0.044 |
| MIX12 | NK | 0.018 | 0.013 | 0.014 | 0.012 | 0.022 | 0.025 | 0.026 | 0.030 | 0.022 | 0.013 | 0.025 |
| MIX12 | CD4T | 0.030 | 0.063 | 0.061 | 0.077 | 0.029 | 0.024 | 0.039 | 0.048 | 0.053 | 0.058 | 0.029 |
| MIX12 | CD8T | 0.033 | 0.111 | 0.067 | 0.116 | 0.064 | 0.056 | 0.080 | 0.034 | 0.132 | 0.098 | 0.017 |
| MIX12 | EOS | 0.015 | 0.052 | 0.043 | 0.053 | 0.036 | 0.034 | 0.031 | 0.032 | 0.033 | 0.034 | 0.015 |
| MIX12 | BASO | 0.031 | 0.012 | 0.016 | 0.013 | 0.019 | 0.019 | 0.021 | 0.016 | 0.014 | 0.012 | 0.039 |
| MIX22 | NEU | 0.007 | 0.033 | 0.018 | 0.034 | 0.014 | 0.013 | 0.012 | 0.010 | 0.029 | 0.027 | 0.017 |
| MIX22 | MONO | 0.005 | 0.020 | 0.008 | 0.022 | 0.009 | 0.009 | 0.006 | 0.006 | 0.017 | 0.016 | 0.008 |
| MIX22 | B | 0.010 | 0.009 | 0.005 | 0.017 | 0.009 | 0.009 | 0.012 | 0.009 | 0.012 | 0.009 | 0.028 |
| MIX22 | NK | 0.023 | 0.019 | 0.023 | 0.013 | 0.020 | 0.023 | 0.025 | 0.027 | 0.010 | 0.014 | 0.026 |
| MIX22 | CD4T | 0.039 | 0.010 | 0.014 | 0.045 | 0.017 | 0.013 | 0.032 | 0.035 | 0.018 | 0.020 | 0.032 |
| MIX22 | CD8T | 0.027 | 0.013 | 0.010 | 0.047 | 0.025 | 0.009 | 0.026 | 0.015 | 0.024 | 0.013 | 0.014 |
| MIX22 | EOS | 0.039 | 0.053 | 0.044 | 0.054 | 0.043 | 0.043 | 0.042 | 0.042 | 0.046 | 0.046 | 0.045 |
| MIX22 | BASO | 0.017 | 0.011 | 0.009 | 0.012 | 0.010 | 0.011 | 0.010 | 0.008 | 0.010 | 0.009 | 0.020 |
| MIX18 | NEU | 0.039 | 0.031 | 0.033 | 0.031 | 0.033 | 0.034 | 0.026 | 0.027 | 0.026 | 0.026 | 0.023 |
| MIX18 | MONO | 0.009 | 0.016 | 0.010 | 0.016 | 0.013 | 0.014 | 0.011 | 0.010 | 0.016 | 0.014 | 0.014 |
| MIX18 | B | 0.018 | 0.010 | 0.013 | 0.008 | 0.009 | 0.009 | 0.008 | 0.009 | 0.009 | 0.009 | 0.029 |
| MIX18 | NK | 0.026 | 0.021 | 0.021 | 0.017 | 0.020 | 0.024 | 0.020 | 0.022 | 0.014 | 0.017 | 0.021 |
| MIX18 | CD4T | 0.038 | 0.015 | 0.011 | 0.012 | 0.007 | 0.023 | 0.009 | 0.010 | 0.020 | 0.019 | 0.036 |
| MIX18 | CD8T | 0.030 | 0.016 | 0.016 | 0.018 | 0.016 | 0.019 | 0.017 | 0.008 | 0.012 | 0.018 | 0.013 |
| MIX18 | EOS | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.007 | 0.011 | 0.011 | 0.004 |
| MIX18 | BASO | 0.012 | 0.001 | 0.001 | 0.002 | 0.004 | 0.003 | 0.007 | 0.005 | 0.006 | 0.005 | 0.011 |

Bias on the whole-blood-like sets:
| level_0 | level_1 | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FACS | NEU | -0.030 | -0.013 | -0.007 | -0.012 | -0.010 | -0.011 | 0.003 | 0.000 | -0.005 | -0.006 | -0.000 |
| FACS | MONO | 0.010 | -0.001 | 0.001 | -0.001 | -0.001 | -0.001 | 0.002 | 0.002 | 0.004 | 0.006 | 0.004 |
| FACS | B | 0.007 | 0.014 | 0.009 | 0.016 | 0.014 | 0.013 | 0.016 | 0.011 | 0.022 | 0.018 | 0.001 |
| FACS | NK | 0.031 | 0.024 | 0.029 | 0.019 | 0.027 | 0.036 | 0.028 | 0.030 | 0.018 | 0.023 | 0.028 |
| FACS | CD4T | 0.012 | -0.018 | -0.025 | -0.040 | -0.022 | -0.006 | -0.026 | -0.031 | -0.001 | -0.003 | 0.009 |
| FACS | CD8T | 0.020 | 0.047 | 0.045 | 0.069 | 0.064 | 0.044 | 0.073 | 0.050 | 0.056 | 0.039 | 0.035 |
| MIX18_blood | NEU | -0.053 | -0.043 | -0.045 | -0.043 | -0.046 | -0.047 | -0.035 | -0.037 | -0.031 | -0.032 | -0.031 |
| MIX18_blood | MONO | 0.007 | -0.010 | -0.004 | -0.010 | -0.006 | -0.006 | -0.005 | -0.005 | -0.008 | -0.007 | 0.001 |
| MIX18_blood | B | -0.003 | 0.004 | 0.000 | 0.006 | 0.004 | 0.003 | 0.005 | 0.002 | 0.005 | 0.004 | -0.008 |
| MIX18_blood | NK | 0.013 | 0.019 | 0.019 | 0.016 | 0.017 | 0.023 | 0.019 | 0.020 | 0.017 | 0.019 | 0.009 |
| MIX18_blood | CD4T | 0.038 | 0.013 | 0.009 | -0.006 | 0.009 | 0.022 | 0.006 | 0.003 | 0.023 | 0.022 | 0.034 |
| MIX18_blood | CD8T | -0.018 | 0.002 | 0.003 | 0.021 | 0.019 | 0.003 | 0.021 | 0.006 | 0.003 | -0.003 | -0.006 |

Correlation with truth (mixture series with spread in every group):
| level_0 | level_1 | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIX12 | NEU | 0.976 | 0.996 | 0.995 | 0.997 | 0.995 | 0.994 | 0.995 | 0.996 | 0.998 | 0.997 | 0.990 |
| MIX12 | MONO | 0.954 | 0.987 | 0.984 | 0.987 | 0.987 | 0.986 | 0.986 | 0.987 | 0.988 | 0.989 | 0.934 |
| MIX12 | B | 0.960 | 0.983 | 0.984 | 0.983 | 0.980 | 0.977 | 0.978 | 0.980 | 0.980 | 0.980 | 0.949 |
| MIX12 | NK | 0.994 | 0.995 | 0.995 | 0.991 | 0.989 | 0.994 | 0.995 | 0.996 | 0.991 | 0.996 | 0.991 |
| MIX12 | CD4T | 0.992 | 0.991 | 0.996 | 0.992 | 0.995 | 0.997 | 0.994 | 0.992 | 0.995 | 0.993 | 0.989 |
| MIX12 | CD8T | 0.968 | 0.978 | 0.990 | 0.952 | 0.957 | 0.990 | 0.973 | 0.973 | 0.965 | 0.986 | 0.946 |
| MIX12 | EOS | 0.976 | 0.990 | 0.991 | 0.992 | 0.995 | 0.995 | 0.996 | 0.995 | 0.988 | 0.986 | 0.973 |
| MIX12 | BASO | 0.960 | 0.986 | 0.988 | 0.989 | 0.984 | 0.980 | 0.983 | 0.983 | 0.989 | 0.989 | 0.934 |
| MIX22 | NEU | 0.988 | 0.986 | 0.994 | 0.988 | 0.992 | 0.990 | 0.992 | 0.993 | 0.991 | 0.991 | 0.987 |
| MIX22 | MONO | 0.992 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.988 |
| MIX22 | B | 0.996 | 0.998 | 0.997 | 0.998 | 0.998 | 0.997 | 0.997 | 0.998 | 0.997 | 0.997 | 0.996 |
| MIX22 | NK | 0.966 | 0.992 | 0.988 | 0.974 | 0.969 | 0.988 | 0.987 | 0.986 | 0.992 | 0.991 | 0.979 |
| MIX22 | CD4T | 0.991 | 0.991 | 0.988 | 0.997 | 0.994 | 0.988 | 0.995 | 0.995 | 0.991 | 0.990 | 0.988 |
| MIX22 | CD8T | 0.857 | 0.938 | 0.977 | 0.869 | 0.873 | 0.979 | 0.944 | 0.916 | 0.985 | 0.978 | 0.912 |
| MIX22 | EOS | 0.996 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 0.995 |
| MIX22 | BASO | 0.991 | 0.998 | 0.998 | 0.999 | 0.999 | 0.997 | 0.999 | 0.999 | 0.999 | 0.999 | 0.987 |

## 4. Repeatability, GSE250556 (within-person SD of the fraction; pooled = same DNA pool)
| index | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled B | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 | 0.002 |
| pooled CD4T | 0.007 | 0.007 | 0.004 | 0.004 | 0.002 | 0.004 | 0.003 | 0.003 | 0.005 | 0.006 | 0.006 |
| pooled CD8T | 0.007 | 0.010 | 0.006 | 0.006 | 0.005 | 0.007 | 0.006 | 0.002 | 0.013 | 0.005 | 0.006 |
| pooled EOS | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |
| pooled MONO | 0.002 | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 |
| pooled NEU | 0.004 | 0.004 | 0.005 | 0.004 | 0.004 | 0.004 | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 |
| pooled NK | 0.003 | 0.004 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.005 | 0.002 | 0.002 |
| pooled NONBLOOD |  | 0.006 | 0.006 | 0.005 |  |  |  |  |  |  |  |
| unpooled B | 0.003 | 0.003 | 0.002 | 0.002 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.004 | 0.003 |
| unpooled CD4T | 0.014 | 0.013 | 0.010 | 0.006 | 0.007 | 0.010 | 0.008 | 0.008 | 0.010 | 0.011 | 0.012 |
| unpooled CD8T | 0.012 | 0.013 | 0.009 | 0.012 | 0.010 | 0.009 | 0.012 | 0.011 | 0.015 | 0.010 | 0.012 |
| unpooled EOS | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.002 | 0.002 | 0.003 |
| unpooled MONO | 0.003 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0.002 | 0.003 |
| unpooled NEU | 0.022 | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 | 0.020 | 0.020 | 0.020 |
| unpooled NK | 0.005 | 0.004 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.005 | 0.002 | 0.004 |
| unpooled NONBLOOD |  | 0.007 | 0.007 | 0.006 |  |  |  |  |  |  |  |

## 5. Agreement on healthy whole bloods (FACS + LONG + REPL, n = 81): mean difference vs NNLS8
| group | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.000 | 0.011 | 0.003 | 0.015 | 0.012 | 0.012 | 0.013 | 0.004 | 0.020 | 0.011 | -0.010 |
| CD4T | 0.000 | -0.021 | -0.050 | -0.071 | -0.049 | -0.023 | -0.063 | -0.070 | -0.027 | -0.031 | -0.016 |
| CD8T | 0.000 | 0.022 | 0.031 | 0.069 | 0.063 | 0.031 | 0.081 | 0.043 | 0.053 | 0.018 | 0.033 |
| MONO | 0.000 | -0.023 | -0.015 | -0.024 | -0.014 | -0.014 | -0.012 | -0.012 | -0.021 | -0.018 | -0.007 |
| NEU | 0.000 | 0.026 | 0.025 | 0.028 | 0.025 | 0.022 | 0.029 | 0.024 | 0.038 | 0.035 | 0.023 |
| NK | 0.000 | -0.025 | -0.017 | -0.033 | -0.016 | -0.005 | -0.021 | -0.018 | -0.038 | -0.027 | -0.014 |

SD of the difference vs NNLS8:
| group | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.000 | 0.003 | 0.003 | 0.004 | 0.004 | 0.004 | 0.003 | 0.004 | 0.005 | 0.004 | 0.003 |
| CD4T | 0.000 | 0.015 | 0.019 | 0.021 | 0.017 | 0.017 | 0.019 | 0.021 | 0.020 | 0.021 | 0.007 |
| CD8T | 0.000 | 0.020 | 0.019 | 0.025 | 0.023 | 0.023 | 0.024 | 0.015 | 0.030 | 0.020 | 0.010 |
| MONO | 0.000 | 0.007 | 0.006 | 0.007 | 0.005 | 0.004 | 0.005 | 0.005 | 0.009 | 0.008 | 0.002 |
| NEU | 0.000 | 0.007 | 0.007 | 0.008 | 0.007 | 0.006 | 0.006 | 0.006 | 0.011 | 0.010 | 0.005 |
| NK | 0.000 | 0.018 | 0.014 | 0.016 | 0.012 | 0.010 | 0.016 | 0.016 | 0.021 | 0.016 | 0.006 |

NILC sum of fractions (median; unconstrained, so a completeness check): | m | FACS | LONG | MIX12 | MIX18 | MIX22 | REPL |
|---|---|---|---|---|---|---|
| NILC-EPIC8 | 1.025 | 1.007 | 1.039 | 1.011 | 1.013 | 0.997 |
| NILC-c | 1.024 | 1.011 | 1.034 | 1.010 | 1.001 | 0.997 |
| NILC-c-D1 | 0.988 | 0.983 | 0.958 | 0.983 | 0.972 | 0.925 |
| NILC-e | 1.021 | 1.009 | 1.027 | 1.006 | 0.995 | 0.988 |
| NILC-e-D1 | 1.001 | 0.997 | 0.985 | 0.994 | 0.977 | 0.940 |

Non-blood atlas mass in whole blood (median), variants a/b/c: | method | FACS | LONG | MIX12 | MIX18 | MIX22 | REPL |
|---|---|---|---|---|---|---|
| ATLAS_a | 0.025 | 0.014 | 0.044 | 0.011 | 0.002 | 0.045 |
| ATLAS_b | 0.028 | 0.019 | 0.061 | 0.017 | 0.016 | 0.059 |
| ATLAS_c | 0.026 | 0.013 | 0.048 | 0.011 | 0.001 | 0.046 |

## 6. Stage M — neutrophil Met-A on healthy whole bloods, per composition source
e_i = sum_g f_g mu_g,i at the 6,000 neutrophil sites. Tare = median of the other arrays of the same series (Stage T rule, >= 3 references).
Profiles = EPIC8 (blood_composition_EPIC_v1 group profiles, fractions mapped to 8 groups):
| m | set | n | untared_median | untared_sd | untared_min | untared_max | tared_median | tared_sd | tared_min | tared_max | tared_in_normal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NNLS8 | FACS | 6 | 0.970 | 0.022 | 0.937 | 1.002 | 1.000 | 0.026 | 0.963 | 1.037 | 6/6 |
| Atl-a | FACS | 6 | 0.972 | 0.022 | 0.935 | 1.000 | 1.000 | 0.026 | 0.959 | 1.032 | 6/6 |
| Atl-b | FACS | 6 | 0.976 | 0.023 | 0.938 | 1.005 | 1.000 | 0.026 | 0.958 | 1.033 | 6/6 |
| Atl-c | FACS | 6 | 0.972 | 0.022 | 0.936 | 1.000 | 1.000 | 0.026 | 0.960 | 1.032 | 6/6 |
| Atl-c-blood | FACS | 6 | 0.967 | 0.022 | 0.935 | 0.999 | 1.000 | 0.024 | 0.965 | 1.035 | 6/6 |
| Atl-e | FACS | 6 | 0.968 | 0.022 | 0.934 | 1.000 | 1.000 | 0.025 | 0.964 | 1.035 | 6/6 |
| NILC-c | FACS | 6 | 0.963 | 0.019 | 0.932 | 0.989 | 1.000 | 0.020 | 0.967 | 1.027 | 6/6 |
| NILC-c-D1 | FACS | 6 | 0.972 | 0.021 | 0.935 | 0.995 | 1.000 | 0.022 | 0.962 | 1.025 | 6/6 |
| NILC-e | FACS | 6 | 0.963 | 0.021 | 0.933 | 0.992 | 1.000 | 0.023 | 0.967 | 1.033 | 6/6 |
| NILC-e-D1 | FACS | 6 | 0.967 | 0.021 | 0.934 | 0.994 | 1.000 | 0.023 | 0.964 | 1.029 | 6/6 |
| NILC-EPIC8 | FACS | 6 | 0.971 | 0.022 | 0.939 | 1.005 | 1.000 | 0.024 | 0.965 | 1.038 | 6/6 |
| NNLS8 | LONG | 12 | 0.975 | 0.029 | 0.931 | 1.028 | 1.000 | 0.032 | 0.952 | 1.057 | 10/12 |
| Atl-a | LONG | 12 | 0.977 | 0.027 | 0.935 | 1.026 | 1.000 | 0.030 | 0.955 | 1.053 | 11/12 |
| Atl-b | LONG | 12 | 0.979 | 0.028 | 0.937 | 1.032 | 1.000 | 0.031 | 0.955 | 1.056 | 10/12 |
| Atl-c | LONG | 12 | 0.978 | 0.028 | 0.937 | 1.029 | 1.000 | 0.031 | 0.955 | 1.054 | 10/12 |
| Atl-c-blood | LONG | 12 | 0.976 | 0.027 | 0.935 | 1.023 | 1.000 | 0.030 | 0.955 | 1.050 | 10/12 |
| Atl-e | LONG | 12 | 0.975 | 0.027 | 0.933 | 1.024 | 1.000 | 0.031 | 0.955 | 1.053 | 10/12 |
| NILC-c | LONG | 12 | 0.967 | 0.026 | 0.927 | 1.014 | 1.000 | 0.029 | 0.957 | 1.052 | 11/12 |
| NILC-c-D1 | LONG | 12 | 0.972 | 0.028 | 0.930 | 1.026 | 1.000 | 0.030 | 0.954 | 1.058 | 11/12 |
| NILC-e | LONG | 12 | 0.970 | 0.026 | 0.928 | 1.014 | 1.000 | 0.029 | 0.955 | 1.049 | 12/12 |
| NILC-e-D1 | LONG | 12 | 0.971 | 0.026 | 0.929 | 1.015 | 1.000 | 0.030 | 0.953 | 1.050 | 12/12 |
| NILC-EPIC8 | LONG | 12 | 0.976 | 0.028 | 0.933 | 1.027 | 1.000 | 0.031 | 0.953 | 1.055 | 10/12 |
| NNLS8 | REPL | 63 | 1.205 | 0.040 | 1.121 | 1.300 | 1.000 | 0.033 | 0.930 | 1.079 | 55/63 |
| Atl-a | REPL | 63 | 1.202 | 0.038 | 1.121 | 1.293 | 0.999 | 0.032 | 0.932 | 1.076 | 55/63 |
| Atl-b | REPL | 63 | 1.211 | 0.039 | 1.131 | 1.305 | 0.999 | 0.033 | 0.932 | 1.078 | 55/63 |
| Atl-c | REPL | 63 | 1.206 | 0.039 | 1.125 | 1.298 | 1.000 | 0.033 | 0.932 | 1.077 | 55/63 |
| Atl-c-blood | REPL | 63 | 1.204 | 0.038 | 1.124 | 1.295 | 1.000 | 0.032 | 0.933 | 1.076 | 55/63 |
| Atl-e | REPL | 63 | 1.203 | 0.038 | 1.123 | 1.295 | 1.000 | 0.032 | 0.933 | 1.076 | 55/63 |
| NILC-c | REPL | 63 | 1.187 | 0.036 | 1.109 | 1.270 | 1.000 | 0.030 | 0.935 | 1.070 | 56/63 |
| NILC-c-D1 | REPL | 63 | 1.198 | 0.038 | 1.115 | 1.286 | 1.000 | 0.032 | 0.931 | 1.074 | 55/63 |
| NILC-e | REPL | 63 | 1.191 | 0.036 | 1.113 | 1.274 | 0.999 | 0.031 | 0.933 | 1.069 | 56/63 |
| NILC-e-D1 | REPL | 63 | 1.197 | 0.038 | 1.116 | 1.283 | 1.000 | 0.032 | 0.932 | 1.073 | 55/63 |
| NILC-EPIC8 | REPL | 63 | 1.205 | 0.039 | 1.124 | 1.300 | 1.000 | 0.033 | 0.933 | 1.079 | 55/63 |

Profiles = the composition's own reference (atlas means for atlas and NILC-atlas methods):
| m | set | n | untared_median | untared_sd | untared_min | untared_max | tared_median | tared_sd | tared_min | tared_max | tared_in_normal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NNLS8 | FACS | 6 | 0.970 | 0.022 | 0.937 | 1.002 | 1.000 | 0.026 | 0.963 | 1.037 | 6/6 |
| Atl-a | FACS | 6 | 0.951 | 0.017 | 0.925 | 0.976 | 1.000 | 0.022 | 0.969 | 1.031 | 6/6 |
| Atl-b | FACS | 6 | 0.950 | 0.016 | 0.923 | 0.973 | 1.000 | 0.022 | 0.967 | 1.029 | 6/6 |
| Atl-c | FACS | 6 | 0.948 | 0.017 | 0.926 | 0.977 | 1.000 | 0.019 | 0.976 | 1.033 | 6/6 |
| Atl-c-blood | FACS | 6 | 0.958 | 0.019 | 0.936 | 0.992 | 1.000 | 0.022 | 0.975 | 1.038 | 6/6 |
| Atl-e | FACS | 6 | 0.958 | 0.019 | 0.933 | 0.991 | 1.000 | 0.022 | 0.972 | 1.037 | 6/6 |
| NILC-c | FACS | 6 | 0.945 | 0.018 | 0.923 | 0.975 | 1.000 | 0.024 | 0.972 | 1.037 | 6/6 |
| NILC-c-D1 | FACS | 6 | 0.952 | 0.020 | 0.926 | 0.982 | 1.000 | 0.027 | 0.966 | 1.039 | 6/6 |
| NILC-e | FACS | 6 | 0.947 | 0.020 | 0.930 | 0.987 | 1.000 | 0.024 | 0.979 | 1.045 | 6/6 |
| NILC-e-D1 | FACS | 6 | 0.950 | 0.021 | 0.931 | 0.989 | 1.000 | 0.023 | 0.979 | 1.043 | 6/6 |
| NILC-EPIC8 | FACS | 6 | 0.971 | 0.022 | 0.939 | 1.005 | 1.000 | 0.024 | 0.965 | 1.038 | 6/6 |
| NNLS8 | LONG | 12 | 0.975 | 0.029 | 0.931 | 1.028 | 1.000 | 0.032 | 0.952 | 1.057 | 10/12 |
| Atl-a | LONG | 12 | 0.954 | 0.023 | 0.912 | 0.990 | 1.000 | 0.027 | 0.953 | 1.041 | 12/12 |
| Atl-b | LONG | 12 | 0.952 | 0.025 | 0.906 | 0.987 | 1.000 | 0.030 | 0.949 | 1.041 | 11/12 |
| Atl-c | LONG | 12 | 0.953 | 0.025 | 0.910 | 0.993 | 1.000 | 0.027 | 0.954 | 1.043 | 12/12 |
| Atl-c-blood | LONG | 12 | 0.960 | 0.027 | 0.914 | 1.004 | 1.000 | 0.029 | 0.952 | 1.046 | 12/12 |
| Atl-e | LONG | 12 | 0.957 | 0.028 | 0.910 | 1.001 | 1.000 | 0.030 | 0.951 | 1.046 | 12/12 |
| NILC-c | LONG | 12 | 0.943 | 0.026 | 0.899 | 0.984 | 1.000 | 0.028 | 0.953 | 1.043 | 12/12 |
| NILC-c-D1 | LONG | 12 | 0.951 | 0.028 | 0.901 | 0.998 | 1.000 | 0.031 | 0.946 | 1.052 | 10/12 |
| NILC-e | LONG | 12 | 0.950 | 0.026 | 0.904 | 0.994 | 1.000 | 0.028 | 0.951 | 1.047 | 12/12 |
| NILC-e-D1 | LONG | 12 | 0.951 | 0.026 | 0.904 | 0.995 | 1.000 | 0.029 | 0.950 | 1.047 | 11/12 |
| NILC-EPIC8 | LONG | 12 | 0.976 | 0.028 | 0.933 | 1.027 | 1.000 | 0.031 | 0.953 | 1.055 | 10/12 |
| NNLS8 | REPL | 63 | 1.205 | 0.040 | 1.121 | 1.300 | 1.000 | 0.033 | 0.930 | 1.079 | 55/63 |
| Atl-a | REPL | 63 | 1.139 | 0.029 | 1.072 | 1.200 | 1.000 | 0.026 | 0.941 | 1.055 | 59/63 |
| Atl-b | REPL | 63 | 1.142 | 0.030 | 1.070 | 1.203 | 1.000 | 0.027 | 0.937 | 1.054 | 59/63 |
| Atl-c | REPL | 63 | 1.143 | 0.029 | 1.076 | 1.204 | 1.000 | 0.026 | 0.941 | 1.054 | 59/63 |
| Atl-c-blood | REPL | 63 | 1.176 | 0.033 | 1.105 | 1.248 | 1.001 | 0.029 | 0.939 | 1.062 | 58/63 |
| Atl-e | REPL | 63 | 1.173 | 0.034 | 1.102 | 1.245 | 1.000 | 0.029 | 0.938 | 1.062 | 56/63 |
| NILC-c | REPL | 63 | 1.147 | 0.032 | 1.079 | 1.213 | 0.998 | 0.028 | 0.939 | 1.057 | 59/63 |
| NILC-c-D1 | REPL | 63 | 1.161 | 0.034 | 1.085 | 1.233 | 1.000 | 0.030 | 0.935 | 1.063 | 56/63 |
| NILC-e | REPL | 63 | 1.163 | 0.034 | 1.078 | 1.238 | 1.000 | 0.030 | 0.926 | 1.064 | 58/63 |
| NILC-e-D1 | REPL | 63 | 1.167 | 0.036 | 1.079 | 1.250 | 0.999 | 0.031 | 0.924 | 1.071 | 57/63 |
| NILC-EPIC8 | REPL | 63 | 1.205 | 0.039 | 1.124 | 1.300 | 1.000 | 0.033 | 0.933 | 1.079 | 55/63 |

## 7. Which further cells resolve (worst RMSE over the truth sets that hold the cell)
| cell | NNLS8 | Atl-a | Atl-b | Atl-c | Atl-c-blood | Atl-e | NILC-c | NILC-c-D1 | NILC-e | NILC-e-D1 | NILC-EPIC8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MONO | 0.012 | 0.036 | 0.031 | 0.035 | 0.029 | 0.027 | 0.024 | 0.024 | 0.036 | 0.033 | 0.015 |
| B | 0.032 | 0.015 | 0.018 | 0.017 | 0.015 | 0.014 | 0.017 | 0.014 | 0.023 | 0.019 | 0.044 |
| NK | 0.034 | 0.029 | 0.033 | 0.024 | 0.031 | 0.038 | 0.033 | 0.035 | 0.025 | 0.029 | 0.031 |
| CD4T | 0.039 | 0.063 | 0.061 | 0.077 | 0.029 | 0.024 | 0.039 | 0.048 | 0.053 | 0.058 | 0.036 |
| CD8T | 0.033 | 0.111 | 0.067 | 0.116 | 0.065 | 0.056 | 0.080 | 0.053 | 0.132 | 0.098 | 0.044 |
| EOS | 0.039 | 0.053 | 0.044 | 0.054 | 0.043 | 0.043 | 0.042 | 0.042 | 0.046 | 0.046 | 0.045 |
| BASO | 0.031 | 0.012 | 0.016 | 0.013 | 0.019 | 0.019 | 0.021 | 0.016 | 0.014 | 0.012 | 0.039 |
| GRAN | 0.028 | 0.044 | 0.035 | 0.043 | 0.025 | 0.027 | 0.024 | 0.029 | 0.018 | 0.022 | 0.033 |
| CD4nv |  | 0.070 | 0.058 | 0.056 | 0.031 | 0.035 | 0.055 | 0.052 | 0.091 | 0.095 |  |
| CD4mem |  | 0.085 | 0.047 | 0.104 | 0.104 | 0.044 | 0.196 | 0.168 | 0.097 | 0.104 |  |
| Treg |  | 0.074 | 0.042 | 0.075 | 0.087 | 0.041 | 0.131 | 0.103 | 0.052 | 0.041 |  |
| CD8nv |  | 0.130 | 0.084 | 0.123 | 0.069 | 0.071 | 0.042 | 0.024 | 0.134 | 0.121 |  |
| CD8mem |  | 0.022 | 0.018 | 0.019 | 0.020 | 0.016 | 0.041 | 0.018 | 0.008 | 0.024 |  |
| Bnv |  | 0.058 | 0.069 | 0.055 | 0.030 | 0.033 | 0.026 | 0.021 | 0.041 | 0.035 |  |
| Bmem |  | 0.062 | 0.046 | 0.062 | 0.034 | 0.032 | 0.020 | 0.018 | 0.031 | 0.032 |  |
Sets per cell: MONO: FACS+MIX18+MIX22+MIX12; B: FACS+MIX18+MIX22+MIX12; NK: FACS+MIX18+MIX22+MIX12; CD4T: FACS+MIX18+MIX22+MIX12; CD8T: FACS+MIX18+MIX22+MIX12; EOS: MIX22+MIX12; BASO: MIX22+MIX12; GRAN: FACS+MIX22+MIX12; CD4nv: MIX22+MIX12; CD4mem: MIX22+MIX12; Treg: MIX22+MIX12; CD8nv: MIX12; CD8mem: MIX12; Bnv: MIX22+MIX12; Bmem: MIX22+MIX12. Cells within 0.02 worst-set RMSE (method, value):
- MONO: NNLS8 0.012, NILC-EPIC8 0.015
- B: Atl-e 0.014, NILC-c-D1 0.014, Atl-c-blood 0.015, Atl-a 0.015, Atl-c 0.017, NILC-c 0.017, Atl-b 0.018, NILC-e-D1 0.019
- BASO: NILC-e-D1 0.012, Atl-a 0.012, Atl-c 0.013, NILC-e 0.014, Atl-b 0.016, NILC-c-D1 0.016, Atl-e 0.019, Atl-c-blood 0.019
- GRAN: NILC-e 0.018
- CD8mem: NILC-e 0.008, Atl-e 0.016, NILC-c-D1 0.018, Atl-b 0.018, Atl-c 0.019
- Bmem: NILC-c-D1 0.018, NILC-c 0.020

All other (cell, method) pairs are above 0.02 on at least one set. NK (+0.02–0.03) and CD8 T (+0.02–0.08) read high on FACS for every method, NNLS8 included.
Eosinophils read low on MIX22 for every method (−0.035 to −0.05).

Figure: fig_est_vs_true.png / .pdf (estimated vs true, per cell and method).

## Caveats and scope (read with every number above)
- Small n: neutrophil bias on whole blood rests on 6 FACS arrays (one slide, one lab) and 6 blood-like mixtures; provisional.
- The Salas DNA mixtures (MIX18, MIX22) come from the same laboratory as the purified reference arrays and may share donors' DNA; GSE182379 also from that group. Only GSE112618 / GSE110530 / GSE250556 are whole bloods, and only GSE250556 is another laboratory (no truth there).
- Flow-cytometry counts are cell proportions; methylation fractions are DNA proportions. Differences of a few percent per cell are expected from that alone.
- Variant e and the circulating-cell set (Atl-c-blood) were defined after run-2 results were read on all sets: their numbers are development, not held-out.
- The earlier '-0.05 neutrophil' finding was not reproduced here. This run removed the 6,000 neutrophil identity sites from the atlas markers and used point estimates (n_boot 0); the original configuration with those sites in was not rerun, so whether that difference matters is untested.
- NILC covariance is the atlas template variance + a 0.02 floor (chosen configs). The '+tech' configs used technical noise measured on GSE250556 pooled replicates; for those configs the GSE250556 repeatability is partly in-sample. No covariance across specimens (cohort statistic) was used.
- Atlas posterior draws (cross-cell covariance) were not used; weights are per-cell posterior SD and donor SD only.
- GSE250556: 63 of 64 arrays (GSM7981500 has no Stage-1 beta in the DEV-SELFTARE-01 store). Of the 55 EPIC GEO series in S3 only those with known composition or replicates were used (5 series + GSE250556).
