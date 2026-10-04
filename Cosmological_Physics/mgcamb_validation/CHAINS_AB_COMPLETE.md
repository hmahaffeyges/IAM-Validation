# Level 1 Runs A and B completed to R−1 < 0.01 (2026-10-01)
Same MGCAMB/Cobaya/Planck 2018 setup (χ² reproduced exactly before restart); 4 MPI chains each from the learned covariance.
| run | final R−1 | σ8 | H0 | µ0 | χ²_min |
|---|---|---|---|---|---|
| iam_fixed_mu0 (A, µ0 = −0.13495) | 0.0089 | 0.8015 ± 0.0058 | 67.08 ± 0.53 | fixed | 1013.95 |
| iam_float_mu0 (B, µ0 free) | 0.0099 | 0.8158 ± 0.0156 | 67.14 ± 0.54 | +0.015 ± 0.156 | 1011.69 |
Posterior means after 30 % burn-in. Free µ0: the IAM prediction −0.135 lies 0.96σ away; zero lies 0.09σ away — Planck alone cannot separate them.
With these two closed, every Level 1 run (A–L) is below R−1 = 0.01. The Technical Reference counts 14 of 17 below 0.01; which third chain it counts above 0.01 is to be checked before the book states a total.
