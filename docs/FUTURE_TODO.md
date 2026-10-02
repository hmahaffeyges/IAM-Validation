# Future to-do (low priority)

Work that is useful but not crucial for IAM right now. Nothing here takes compute or time from the methylation report. Each item says what it settles, where its ready-to-run package is, and what changes when it lands.

## Cosmology chains (parked 2026-10-02, author: not crucial, lowest priority)

| # | Item | What it settles | Package | When it lands |
|---|---|---|---|---|
| C1 | Free-µ0 chains B, E, H, K re-run with the µ0 prior widened from [−0.5, +0.2] to [−0.8, +1.0] | Where the free-µ0 posterior turns over (today about 20 % piles against +0.2). The fixed-µ0 result does not depend on it. | `remote_jobs/mu0wide/` (configs, same-code check, S3 backup); 4 chains on one 128-core box | Update the free-µ0 table in `p2_late_time_growth.tex`, `p2_dualsector_chains.tex`, `CHAIN_EXTRACTION_FINAL.csv`; errata L2 |
| C2 | Level 2 Run D′ (IAM) and C′ (ΛCDM) with fσ8 = dσ8/d ln a (the density-field growth rate) | A valid Level 2 growth test: CAMB's fσ8 does not carry the modification, so Run D did not test IAM growth (errata P12) | `remote_jobs/l2rsd/` (likelihood `iam_rsd_growth.py`, both configs, setup with logged Planck install, checks gate the chains); about a day on one box | Replace the Run D hold box in `p2_dual_sector_perturbation.tex`; errata P7/P12; push chains to `camb_validation/chains/` |

Until then the book prints the free-µ0 median, 90 % bound and P(µ0 < −0.135) with the prior edge stated, and the Level 2 chapter prints the Planck-only result with Run D held.
