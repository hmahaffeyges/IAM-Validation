# CALCULATED FORECAST - Fisher-matrix sensitivity of Euclid (+Planck CMB lensing, +DESI) to IAM mu(z)

mu(z) = 1 + A [mu_IAM(z) - 1]; A = 0 is GR, A = 1 is IAM. Significance = 1/sigma(A). Planck 2018 LCDM background.
Marginalised over Omega_m, Omega_b, h, n_s, sigma8 (GR-normalised; equivalent to A_s) and all nuisance parameters.
template_sigma_mu0 = sigma(mu0) for Euclid's mu = 1 + mu0 Omega_DE(z)/Omega_DE(0) from the same pipeline (GR fiducial); template_naive_significance = 0.136/sigma(mu0), i.e. the old template-matching estimate, shown for comparison only.
Scale-matched values (Albuquerque-settings rows only) multiply sigma(A) by (published template sigma / our template sigma).

| scenario | sigma_case | sigma_A_IAMfid | significance_IAMfid | sigma_A_GRfid | significance_GRfid | template_sigma_mu0 | template_naive_significance | sigma_A_IAMfid_scalematched | significance_scalematched |
|---|---|---|---|---|---|---|---|---|---|
| Euclid DR1 pessimistic (1900 deg2) | Sigma=1 fixed | 2.532 | 0.39 | 2.548 | 0.39 | 0.170 | 0.80 |  |  |
| Euclid DR1 pessimistic (1900 deg2) | Sigma0 free | 3.640 | 0.27 | 3.638 | 0.27 | 0.218 | 0.62 |  |  |
| Euclid DR1 optimistic (1900 deg2) | Sigma=1 fixed | 1.472 | 0.68 | 1.483 | 0.67 | 0.098 | 1.39 |  |  |
| Euclid DR1 optimistic (1900 deg2) | Sigma0 free | 1.754 | 0.57 | 1.755 | 0.57 | 0.115 | 1.18 |  |  |
| Euclid full pessimistic | Sigma=1 fixed | 0.901 | 1.11 | 0.907 | 1.10 | 0.060 | 2.25 |  |  |
| Euclid full pessimistic | Sigma0 free | 1.295 | 0.77 | 1.295 | 0.77 | 0.078 | 1.75 |  |  |
| Euclid full optimistic | Sigma=1 fixed | 0.524 | 1.91 | 0.528 | 1.89 | 0.035 | 3.91 |  |  |
| Euclid full optimistic | Sigma0 free | 0.624 | 1.60 | 0.625 | 1.60 | 0.041 | 3.31 |  |  |
| Euclid full pessimistic + Planck lensing | Sigma=1 fixed | 0.897 | 1.12 | 0.903 | 1.11 | 0.060 | 2.29 |  |  |
| Euclid full pessimistic + Planck lensing | Sigma0 free | 1.295 | 0.77 | 1.294 | 0.77 | 0.077 | 1.76 |  |  |
| Euclid full optimistic + Planck lensing | Sigma=1 fixed | 0.523 | 1.91 | 0.527 | 1.90 | 0.035 | 3.93 |  |  |
| Euclid full optimistic + Planck lensing | Sigma0 free | 0.624 | 1.60 | 0.624 | 1.60 | 0.041 | 3.32 |  |  |
| Euclid full pessimistic + Planck lensing + DESI | Sigma=1 fixed | 0.759 | 1.32 | 0.765 | 1.31 | 0.049 | 2.77 |  |  |
| Euclid full pessimistic + Planck lensing + DESI | Sigma0 free | 0.866 | 1.15 | 0.871 | 1.15 | 0.053 | 2.56 |  |  |
| Euclid full optimistic + Planck lensing + DESI | Sigma=1 fixed | 0.482 | 2.07 | 0.487 | 2.06 | 0.031 | 4.38 |  |  |
| Euclid full optimistic + Planck lensing + DESI | Sigma0 free | 0.550 | 1.82 | 0.552 | 1.81 | 0.035 | 3.88 |  |  |
| GCsp alone, pessimistic | Sigma=1 fixed | 4.733 | 0.21 | 4.756 | 0.21 | 0.229 | 0.59 |  |  |
| GCsp alone, optimistic | Sigma=1 fixed | 4.532 | 0.22 | 4.551 | 0.22 | 0.219 | 0.62 |  |  |
| 3x2pt alone, pessimistic (all 10 GCph bins) | Sigma=1 fixed | 1.032 | 0.97 | 1.042 | 0.96 | 0.073 | 1.87 |  |  |
| 3x2pt alone, pessimistic (all 10 GCph bins) | Sigma0 free | 1.658 | 0.60 | 1.653 | 0.61 | 0.117 | 1.17 |  |  |
| 3x2pt alone, optimistic | Sigma=1 fixed | 0.574 | 1.74 | 0.579 | 1.73 | 0.040 | 3.38 |  |  |
| 3x2pt alone, optimistic | Sigma0 free | 0.654 | 1.53 | 0.652 | 1.53 | 0.045 | 3.02 |  |  |
| Albuquerque-conservative: GCsp k<0.1/Mpc + 3x2pt k<0.25/Mpc | Sigma=1 fixed | 1.017 | 0.98 | 1.023 | 0.98 | 0.067 | 2.02 |  |  |
| Albuquerque-conservative: GCsp k<0.1/Mpc + 3x2pt k<0.25/Mpc | Sigma0 free | 2.052 | 0.49 | 2.049 | 0.49 | 0.119 | 1.14 | 4.015 | 0.25 |
| Albuquerque 3x2pt alone k<4/Mpc | Sigma=1 fixed | 0.606 | 1.65 | 0.611 | 1.64 | 0.042 | 3.21 |  |  |
| Albuquerque 3x2pt alone k<4/Mpc | Sigma0 free | 0.700 | 1.43 | 0.699 | 1.43 | 0.048 | 2.83 | 0.584 | 1.71 |
| Euclid full pessimistic, GCsp in IST:F h-unit convention | Sigma=1 fixed | 0.919 | 1.09 | 0.927 | 1.08 | 0.063 | 2.16 |  |  |
| Euclid full pessimistic, GCsp in IST:F h-unit convention | Sigma0 free | 1.316 | 0.76 | 1.316 | 0.76 | 0.080 | 1.70 |  |  |
| Euclid full optimistic, GCsp in IST:F h-unit convention | Sigma=1 fixed | 0.456 | 2.19 | 0.459 | 2.18 | 0.029 | 4.61 |  |  |
| Euclid full optimistic, GCsp in IST:F h-unit convention | Sigma0 free | 0.624 | 1.60 | 0.625 | 1.60 | 0.041 | 3.32 |  |  |
| Euclid full optimistic + Planck lensing + DESI (kmax 0.2h/Mpc) | Sigma=1 fixed | 0.392 | 2.55 | 0.396 | 2.52 | 0.025 | 5.55 |  |  |
| Euclid full optimistic + Planck lensing + DESI (kmax 0.2h/Mpc) | Sigma0 free | 0.413 | 2.42 | 0.416 | 2.40 | 0.026 | 5.29 |  |  |
| Euclid full optimistic + Planck lensing + DESI z<0.9 only (no volume overlap) | Sigma=1 fixed | 0.489 | 2.05 | 0.493 | 2.03 | 0.032 | 4.24 |  |  |
| Euclid full optimistic + Planck lensing + DESI z<0.9 only (no volume overlap) | Sigma0 free | 0.558 | 1.79 | 0.560 | 1.79 | 0.036 | 3.74 |  |  |
