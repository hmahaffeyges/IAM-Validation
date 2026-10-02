# DUAL_SECTOR_PERTURBATION_CHECK — "Dual-Sector Perturbation Cosmology: A Modified CAMB Implementation with µ < 1, Σ = 1" (Level 2, 28 Feb 2026, 17 pp)
Read in full 2026-10-02 (all 595 extracted lines; first pass saw previews only, re-read completely before this version). Checked against `camb_validation/` (equations_iam_level2.f90, likelihood_rsd.py, getdist_scripts/rsd_apples_to_apples.py,
chains/*.input.yaml and the chain files, 30 % burn-in).

## Reproduced from the chain files
| | paper | chains |
|---|---|---|
| Run A H0, σ8 | 67.161 ± 0.467, 0.7998 | 67.161, 0.7998 |
| Run C H0, σ8 | 67.188, 0.8087 | 67.188, 0.8087 |
| best-fit χ² A / C, Δ | 10972.61 / 10972.07, +0.54 | same |
| posterior-mean χ² A / C, Δ | −0.01 | 10985.07 / 10985.08 |
| Run D σ8, H0 | 0.7995, — | 0.7995, 67.19 |
| H0(matter) = 67.161 √1.15765 | 72.26 | 72.26 |
| Eqs. 11–12 | −0.37σ, −0.75σ | ✓ |
Sampled parameters {ω_b, ω_c, θ_MC, τ, ln A_s, n_s} ✓.

## Corrected
1. **Code listing (§2.3) does not match the source.** Printed: `grho_0 = 3.0`, and "Hubble friction in the CDM and baryon velocity equations".
   Source (`equations_iam_level2.f90` l. 2244–2338): `iam_grho_extra = iam_beta*E(a)*a²*(grhob+grhoc+grhornomass+grhog+grhov)` (today's total density,
   so the units are right), `adotoa_matter = sqrt((grho + iam_grho_extra)/3)`; then in the synchronous-gauge **density** equations the metric source
   `z = (½ dgrho/k + etak)/adotoa` is divided by `adotoa_matter` instead: `clxcdot = −k z_matter`, `clxbdot = −k (z_matter + vb)`. (CDM has no velocity in
   CAMB's synchronous gauge.) The chapter prints the code as it is.
2. **Measured growth of the coded mechanism** (`scripts/l2growth.sh`, `growth_out.py`; CAMB 1.5.8 built from `equations_iam_level2.f90` with the switch on and
   off, Run A posterior means; outputs `chains/data/growth_on.json`, `growth_off.json`):
   | z | σ8 ratio, code | D ratio, closed-form µ (Eq. 5) | f from dlnσ8/dlna, on / off | f from CAMB fσ8/σ8, on / off |
   |---|---|---|---|---|
   | 0 | 0.9880 | 0.9922 | 0.498 / 0.524 | 0.542 / 0.529 |
   | 0.5 | 0.9962 | 0.9978 | 0.746 / 0.760 | 0.770 / 0.763 |
   | 1 | 0.9988 | 0.9994 | 0.870 / 0.875 | 0.881 / 0.879 |
   The coded mechanism suppresses growth somewhat more than Eq. 5 (σ8 −1.2 % vs −0.8 % at fixed parameters, z = 0); same sign and redshift dependence.
   **CAMB's fσ8 is inconsistent with the modified code.** The change acts on the CDM and baryon density equations; CAMB computes fσ8 from velocities,
   which the change does not touch. With the switch on, CAMB reports f = 0.542 at z = 0 where the density field grows at f = 0.498 (+8.8 %); with it off the
   two agree to the interpolation accuracy (1 %). Physically, matter is conserved, so the velocity follows the density: fσ8 = dσ8/d ln a. In that form IAM's
   fσ8 today is 6 % below ΛCDM at fixed parameters; CAMB's output says 1 % above.
   **Consequence:** Run D's growth likelihood (`likelihood_rsd.py`, `get_fsigma8`) did not see the IAM growth suppression. Runs A and C (Planck only) are
   unaffected: CMB lensing comes from the density. Fix: a growth likelihood that takes fσ8 = −(1+z) dσ8/dz from `get_sigma8_z`, Run D re-run, plus a
   matched ΛCDM + growth chain.
3. **Table 1 and Table 6 E(a) columns** are shifted: printed 0.6977 / 0.3679 / 0.1353 / 0.0498 at z = 0.2 / 0.5 / 1 / 2; E = e^−z = 0.8187 / 0.6065 / 0.3679 /
   0.1353. Table 1 µ at z = 0.2, 0.5: 0.905, 0.948 (printed 0.893, 0.942); H_m/H: 1.051, 1.027 (printed 1.058, 1.030). Table 6 (Run A posterior,
   Ω_m 0.3166, H0 67.161): H_photon 88.89 / 120.44 / 204.06 / 307.37 at z = 0.5 / 1 / 2 / 3 (printed 87.45 / 117.68 / 199.00 / 305.00); H_m 91.29 / 121.53 /
   204.29 / 307.43 (printed 89.73 / 118.72 / 199.22 / 305.06); ratio 1.0269 / 1.0090 / 1.0012 / 1.0002; E 0.6065 / 0.3679 / 0.1353 / 0.0498.
4. **Background runs:** H0 = 61.45 ± 0.42 and 61.52 ± 0.43 — 10.9σ from Planck 67.36 ± 0.54 (printed "6σ").
5. **Likelihood:** `planck_NPIPE_highl_CamSpec.TTTEEE` is the PR4 (NPIPE) CamSpec (Rosenberg, Gratton & Efstathiou 2022), not the Planck 2018 CamSpec of
   Efstathiou & Gratton 2021. Level 1 used plik-lite (PR3), so χ² values are not comparable between levels.
6. **RSD data (`likelihood_rsd.py`)**: 6dFGS z = 0.067 (Beutler 2012) and SDSS MGS z = 0.15 are not BOSS/eBOSS; diagonal errors only (the BOSS DR12 points
   are correlated).
7. **Table 4 (Run D) mixes statistics.** IAM side: chain-average χ² (RSD 6.42, CMB 10984.93); ΛCDM CMB: Run C chain-average (10985.08); ΛCDM RSD: χ² at
   Run C's posterior-mean parameters (3.34). Chain-average χ² exceeds the best point by ~12–13 in these chains; for the RSD term, Run D's average is 6.42,
   its value at the chain's best point 6.26, its minimum 5.84. So the +3.08 RSD penalty is overstated by up to ~0.6 and the total +2.92 has no
   consistent definition. A clean comparison needs a ΛCDM + RSD chain with the same likelihoods (option: run it), or both χ² at their own best points.
   "Validated by the Level 1 analysis (+1.34)": Level 1's "RSD" runs used BOSS DR12 consensus + DR16 BAO likelihoods, not these seven points.
   Superseded in substance by item 2: Run D's fσ8 came from CAMB velocities that do not carry the modification.
8. **"Δχ² below the 95 % exclusion threshold 3.84"**: a χ²₁ threshold does not apply to two models with the same number of parameters (as L3).
9. **§5.3** "the posterior returns β_m = 0.1583 ± 0.0033": β_m is fixed; what the chain shows is Planck Ω_m = 0.3166 ± 0.0065 in this run, so the fixed
   0.15765 is 0.2σ from Ω_m/2. Kept as that consistency statement.
10. **§2.3** "MGCAMB µ = 1 + µ0 Ω_DE": missing /Ω_Λ (as L1).
12. **§8.2 item 2** "σ8 suppression of 0.009 (1.5 %)": 0.009/0.809 = 1.1 % (the abstract's value); 1.5 is the shift in σ units (1.51σ).
13. **KiDS-1000 S8 = 0.759 ± 0.021 cited to Heymans et al. 2021**: 0.759 (+0.024/−0.021) is the cosmic-shear result (Asgari et al. 2021); Heymans et al. 2021
   (3×2pt) report 0.766 (+0.020/−0.014). To trace against both sources before printing.
14. **§8.5** Euclid "∼3.4σ" = 0.135/0.04 ✓; "Σ ≠ 1 at > 10⁻⁴ falsifies" — no survey reaches 10⁻⁴; state the forecast precision instead.
11. References: Frusciante et al. title (as L6); "DESI 2025, JCAP 2025(02), 021" to trace (arXiv 2411.12022).
