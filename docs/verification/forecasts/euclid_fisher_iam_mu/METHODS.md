# METHODS: Fisher forecast of Euclid sensitivity to IAM's own mu(z)

**Status: CALCULATED FORECAST.** These are Fisher-matrix numbers from survey specifications, not measurements. No data were fitted and nothing was tuned toward a result. The IAM model was used exactly as specified, with no free parameter.

## 1. Result

IAM's growth modification is parametrised as mu(z) = 1 + A [mu_IAM(z) - 1], so A = 0 is GR and A = 1 is IAM. The table gives sigma(A) with an IAM fiducial; the significance of IAM versus GR is 1/sigma(A). Numbers with a GR fiducial agree to within 1 %. Full table: `out/results.md`.

| Scenario | Sigma = 1 fixed: sigma(A) / significance | Sigma0 free: sigma(A) / significance |
|---|---|---|
| Euclid DR1, pessimistic (1900 deg2) | 2.53 / 0.39 sigma | 3.64 / 0.27 sigma |
| Euclid DR1, optimistic (1900 deg2) | 1.47 / 0.68 sigma | 1.75 / 0.57 sigma |
| Euclid full, pessimistic | 0.90 / 1.11 sigma | 1.30 / 0.77 sigma |
| Euclid full, optimistic | 0.52 / 1.91 sigma | 0.62 / 1.60 sigma |
| Euclid full pess. + Planck lensing | 0.90 / 1.12 sigma | 1.29 / 0.77 sigma |
| Euclid full opt. + Planck lensing | 0.52 / 1.91 sigma | 0.62 / 1.60 sigma |
| Euclid full pess. + Planck lensing + DESI | 0.76 / 1.32 sigma | 0.87 / 1.15 sigma |
| Euclid full opt. + Planck lensing + DESI | 0.48 / 2.07 sigma | 0.55 / 1.82 sigma |

**Why the signal is weak.** IAM's deviation is concentrated at z < 0.5. There the f sigma8 deficit is 4.25 % at z = 0 and 1.35 % at z = 0.5, but only 0.1-0.3 % across Euclid's spectroscopic range, z = 0.9-1.8. In the photometric 3x2pt spectra, IAM lowers C_ell by only 0.3-1.1 % (figure, panel b), and that change is largely degenerate with sigma8. Planck CMB lensing adds almost nothing, because its kernel peaks at z ~ 2, where mu_IAM is close to 1. DESI's low-z f sigma8 points are what add information.

**Comparison with the old template estimate.** Earlier estimates treated IAM as Euclid's template mu = 1 + mu0 Omega_DE(z)/Omega_DE(0) with mu0 = -0.136, giving a significance of 0.136/sigma(mu0). That overstates the significance by about a factor of 2: 4.4 sigma against the 2.1 sigma above for the best combination (column `template_naive_significance`). The template keeps a deviation out to z ~ 1, where Euclid has most of its volume; IAM's mu(z) does not.

**Scale-matched rows.** These rows apply the factor (published template sigma / our template sigma) for identical settings:
- Albuquerque et al. conservative cuts (GCsp k < 0.1/Mpc + 3x2pt k < 0.25/Mpc, Sigma0 free): sigma(A) = 2.05 raw and 4.0 scale-matched (factor 1.96), i.e. 0.49 sigma raw and 0.25 sigma scale-matched.
- 3x2pt alone with k < 4/Mpc: 0.70 raw and 0.58 scale-matched (factor 0.83), i.e. 1.43 sigma raw and 1.71 sigma scale-matched.

Which is more reliable? For the k < 4/Mpc setting the raw and scale-matched values agree to 20 %, so both are usable. For the conservative setting we could not reproduce Albuquerque et al.'s 3x2pt data cut (Sect. 4, test 2). The scale-matched value inherits their cut definition and is the more conservative number. The raw value follows the cut formula as written in their paper.

## 2. Model (fixed, as specified)

- **Background.** Flat LCDM with Planck 2018 parameters: Omega_m 0.3153, h 0.6736, Omega_b h^2 0.02237, n_s 0.9649, ln(1e10 A_s) 3.044, tau 0.0544. One massive neutrino with 0.06 eV is held fixed, as in IST:F and Albuquerque et al. This gives sigma8 = 0.8112 (GR).
- **IAM growth function.** mu_IAM(a) = H2/(H2 + beta_m E(a)), with H2 = Omega_m a^-3 + Omega_L, E(a) = exp(1 - 1/a) and beta_m = Omega_m/2. This gives mu = 0.864, 0.948 and 0.982 at z = 0, 0.5 and 1. When Omega_m is varied for the derivatives, mu_IAM follows the varied Omega_m (beta_m = Omega_m/2), as the model defines it. mu is scale-independent.
- **Linear growth.** Solved from the ODE in ln a on the same LCDM background without radiation, starting from D = a at a = 1e-3. This means IAM and LCDM have the same primordial amplitude. The MG linear spectrum is P_MG(k,z) = P_GR,CAMB(k,z) [D_MG(z)/D_GR(z)]^2, and f_MG is taken from the same ODE.
- **Growth reproduction check (reproduced).** D(z=0) is -0.777 % relative to LCDM. The f sigma8 deficit is 4.251 %, 2.168 %, 1.349 % and 0.409 % at z = 0, 0.3, 0.5 and 1. The target values were -0.78 %, 4.25 %, 2.17 %, 1.35 % and 0.41 % (repo `docs/verification/scripts/verify_euclid_template.py`).
- **Lensing potential.** Sigma = 1 at all z in the main case. In the "Sigma0 free" case, Sigma(z) = 1 + Sigma0 Omega_DE(z)/Omega_DE(0), with fiducial Sigma0 = 0. This is the same functional form Albuquerque et al. use. Sigma multiplies every lensing kernel (Euclid shear, galaxy-galaxy lensing and CMB lensing).
- **sigma8 parameter.** sigma8 is the GR-normalised amplitude: the sigma8 the linear field would have today in GR. It is a reparametrisation of A_s, so marginal errors on A are unchanged by the choice.

## 3. Probes and survey specifications

### Euclid GCsp (Blanchard et al. 2020, IST:F, Sect. 3.2, Eq. 87, Table 3)

- **Redshift bins:** 4 bins, [0.9,1.1], [1.1,1.3], [1.3,1.5] and [1.5,1.8]. Inputs per bin were dN/dOmega dz = 1815.0, 1701.5, 1410.0 and 940.97 deg^-2, and b = 1.46, 1.61, 1.75 and 1.90. Number densities were recomputed for each fiducial; for IST:F's fiducial they reproduce n = 6.86e-4 h^3 Mpc^-3 and V = 7.94 Gpc^3 h^-3 in bin 1.
- **Sky and redshift errors:** 15,000 deg^2 and sigma_z = 0.001(1+z).
- **Power-spectrum model:** Alcock-Paczynski effect, Kaiser term with Lorentzian fingers-of-God, de-wiggled BAO (no-wiggle spectrum = Eisenstein & Hu 1998 no-wiggle times the Gaussian-smoothed ratio to CAMB, smoothing width 0.5 in ln k) and redshift-error damping. Shot-noise nuisance P_s is fiducial 0 and free in each bin. Fiducial sigma_v = sigma_p comes from Eq. 81.
- **Scale cuts:**
  - Pessimistic: k < 0.25 h/Mpc, with sigma_p and sigma_v free as two global amplitude factors.
  - Optimistic: k < 0.30 h/Mpc, with sigma_p and sigma_v fixed.
  - Albuquerque setting: k < 0.1/Mpc, with sigma_p and sigma_v free in each bin.
- **Fisher matrix:** computed directly in the final parameters. This is equivalent to the IST:F projection from {D_A, H, f sigma8, b sigma8, P_s}.
- **Units of k (main run in physical units).** All main-run computations use physical units (1/Mpc), with the reference cosmology fixed at the fiducial. IST:F effectively tabulates the model spectrum in h/Mpc of the model's own h. Reproducing that convention (option `hunits=True`, rows "IST:F h-unit convention") recovers their pessimistic GCsp errors within 6 %. In physical units the h error is 10 times larger. That extra h information comes from the unit convention, so we do not use it in the main numbers. The convention changes IAM's sigma(A) by 2-13 % (sensitivity rows).

### Euclid photometric 3x2pt (IST:F Sect. 3.3-3.4, Table 4-5, Eqs. 112-136)

- **Source sample.** n(z) is proportional to (z/z0)^2 exp[-(z/z0)^1.5] with z0 = 0.9/sqrt(2). There are 10 equipopulated bins with edges {0.001, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50}.
- **Photo-z model.** IST:F Table 5: c_b = 1, z_b = 0, sigma_b = 0.05, c_o = 1, z_o = 0.1, sigma_o = 0.05, f_out = 0.1.
- **Density and noise.** n_gal = 30 arcmin^-2 (3 per bin) and sigma_eps = 0.30. The shape noise is sigma_eps^2/n_i, as in IST:F Eq. 116.
- **Galaxy bias.** b_i = sqrt(1 + z_i,centre), free in each bin (10 parameters).
- **Intrinsic alignments.** The eNLA model with A_IA = 1.72, eta_IA = -0.41 and beta_IA = 2.17 (all free) and C_IA = 0.0134. The <L>/L* table is IST:F's `scaledmeanlum-E2Sa.dat`, taken from the public fishermathica repository (S. Casas, commit 955aa96). The 1/D(z) in the IA term uses the model's own growth, normalised to D(0) = 1.
- **Power spectrum and binning.** Limber approximation with the HALOFIT (Takahashi et al. 2012) nonlinear spectrum. There are 100 logarithmically spaced ell bins between 10 and ell_max.
- **Covariance and Fisher matrix.** Gaussian covariance. The Fisher matrix uses the full data vector (210 unique spectra per ell for 20 fields) with element-wise scale cuts.
- **Scale cuts:**
  - Pessimistic: WL to ell 1500; GCph and XC to ell 750. When combined with GCsp, GCph uses only the 5 bins below z = 0.9, as IST:F prescribes.
  - Optimistic: WL to ell 5000; GCph and XC to ell 3000.
  - Albuquerque setting: 60 bins from 10 to 5000, elements with ell > 3000 removed, and ell_max^ij = k_max min(r(z_i), r(z_j)).

### Planck CMB lensing

- **Data and fiducial.** Limber C_L^kk for 8 <= L <= 400, the range of the Planck 2018 conservative likelihood, with f_sky = 0.67. The spectrum uses the same MG spectrum and Sigma.
- **High-z extension.** Above z = 20 we use the z = 20 spectrum scaled by matter-era growth.
- **Noise (approximation).** Reconstruction noise is modelled as white, N_kk = 3.88e-7. That level is chosen so the total S/N equals Planck 2018's 40 sigma detection (Planck 2018 VIII abstract). This is not the real Planck N_L.
- **Covariance.** Gaussian, with no cross-covariance with Euclid. The surveys overlap on the sky, so the combined result is slightly optimistic.

### DESI

- **Data.** Independent Gaussian errors on f sigma8(z) from DESI Collaboration 2016 (arXiv:1611.00036), Table 2.3 (ELG+LRG+QSO, 14,000 deg^2, z = 0.65-1.85) and Table 2.5 (BGS, z = 0.05-0.45). We used the kmax = 0.1 h/Mpc column, which the DESI document calls the conservative choice; kmax = 0.2 h/Mpc is a sensitivity row.
- **Volume overlap with Euclid.** DESI bins at z > 0.9 overlap Euclid GCsp in volume. Treating them as independent is optimistic. With DESI restricted to z < 0.9, the best-case significance changes only from 2.07 to 2.05.

### Euclid DR1

We assumed 1900 deg^2, ESA's estimate for DR1, the first year of data (ESA "Euclid Data Release DR1: update", 15 June 2026, cosmos.esa.int/web/euclid/dr1-timeline). This is 0.127 of the 15,000 deg^2 used in IST:F. DR1 is modelled by scaling the Euclid Fisher matrices by area only, at full depth and completeness. **This is an assumption.** Real DR1 spectroscopic completeness and photo-z calibration will be worse, so the DR1 numbers are upper bounds on significance.

### Numerics

- **CAMB runs:** CAMB 2.0.4 from pip gives the linear delta_tot spectrum, with k up to 60/Mpc and power-law extrapolation to 500/Mpc.
- **Derivatives:** 4-point central stencil, with 1 % relative steps for the cosmological parameters. Absolute steps were 0.05 for A, 0.02 for mu0, Sigma0 and eta_IA, and 100 Mpc^3 for P_s.
- **Step-size test:** sigma(A) changes by < 0.5 % for A-steps of 0.025-0.1 and cosmological steps of 0.5 % versus 1 % (`out/step_test.json`).
- **Our HALOFIT implementation:** agrees with CAMB's Takahashi HALOFIT to within 0.77 % for k = 0.01-10/Mpc at z = 0-2.
- **Compute:** stage A (51 CAMB runs) and stage B (919 derivative tasks, 59 Fisher matrices) ran on the 128-core host (`out/forecast.log`). Stage C (`analysis.py`) runs locally.

## 4. Validation (full table: `out/validation.md`)

1. **IST:F flat LCDM (Blanchard et al. 2020, Table 9).** Ratios are ours/published, for Omega_m, Omega_b, h, n_s and sigma8.
   - WL alone, pessimistic: 0.96, 0.97, 0.93, 0.90, 0.99.
   - WL alone, optimistic: 1.05, 1.06, 0.95, 0.96, 1.05.
   - WL+GCph+XC: 0.95-1.04 in both settings.
   - GCsp pessimistic in the IST:F h-unit convention: 1.04, 1.02, 1.03, 0.94, 0.97.
   - GCsp optimistic in the IST:F h-unit convention: Omega_m, n_s and sigma8 at 0.98, 0.88 and 0.73, but Omega_b and h at 1.66 and 2.45 (**not reproduced**).
   - GCsp in physical units: Omega_m and sigma8 within 14 %, but h 8-10 times weaker, Omega_b 1.2-1.8 times weaker and n_s 0.8-2.2 times the published error (see the units of k item in Sect. 3).

   Within the 10-20 % target, we met it for WL and 3x2pt on all parameters. For GCsp we met it for Omega_m and sigma8 (and, with the h-unit convention, for all parameters in the pessimistic setting).
2. **Euclid template mu0 (Albuquerque et al. 2025, Table 5, PMG-1, LCDM fiducial).** HALOFIT on the growth-rescaled linear spectrum roughly corresponds to their "US" case; they use ReACT.
   - GCsp alone (k < 0.1/Mpc, Sigma fixed): 53.6 %, published 53.0 %.
   - 3x2pt alone, k < 4/Mpc: 4.8 %, published ~4 %. Sigma0: 1.5 %, published ~1 %.
   - 3x2pt alone, conservative cut (k < 0.25/Mpc): 24.0 %, published 169.4 %. **Not reproduced.**
   - GCsp + 3x2pt, conservative: 11.9 %, published 23.3 %. Sigma0: 2.5 %, published 2.6 %.

   **Diagnosis of the conservative mismatch.** Their text says the k = 0.05/Mpc cut leaves 807 of 12,600 data-vector elements. The formula as written leaves 4891 elements in our implementation. Rescaling the cut so that the count is 807 gives 3x2pt 174 % and a combination of 24.9 %, close to the published 169.4 % and 23.3 %. However, the cosmological-parameter errors then do not match theirs. Their exact cut is therefore unknown to us, which is why the conservative IAM number is reported both raw and scale-matched.

## 5. What is approximate (read before quoting)

- **Nonlinear MG modelling.** On nonlinear scales the MG modelling is approximate: HALOFIT is applied to the growth-rescaled linear spectrum, with no screening and no MG halo-model reaction. It is not calibrated on IAM simulations. The optimistic 3x2pt numbers (ell up to 5000, and ell up to 3000 for galaxy clustering with linear bias) depend on this.
- **Neutrinos and growth.** The growth ODE ignores radiation and neutrino scale dependence, as the IAM chain script does. Only the ratio D_MG/D_GR is applied to CAMB's spectrum.
- **Systematics left out.** Covariances are Gaussian with no super-sample covariance and no non-Gaussian terms. There are no baryonic feedback, photo-z or shear-calibration nuisances; the photo-z parameters are held fixed, as in IST:F. These omissions make the results optimistic.
- **CMB lensing noise and DESI likelihood.** The Planck lensing noise is a calibrated white level, not the real N_L. DESI is used only through its published f sigma8 projections, which assume known shape and geometry.
- **Probe combination.** Probes are combined by adding Fisher matrices, with no cross-covariances (Euclid-Planck, Euclid-DESI).
- **Parameter set.** No Planck primary-CMB prior is included, and w0 and wa are fixed at -1 and 0.
- **Sigma0 parametrisation.** In the Sigma0-free case, Sigma uses the Omega_DE template form; IAM does not predict a Sigma deviation.
- **Fisher approximation.** sigma(A) of order 1 means the Gaussian Fisher approximation is only indicative for DR1.

## 6. Sources

- Euclid Collaboration: Blanchard et al. 2020, A&A 642, A191, doi:10.1051/0004-6361/202038071 (arXiv:1910.09273).
- Euclid Collaboration: Albuquerque et al. 2025, arXiv:2506.03008, doi:10.48550/arXiv.2506.03008.
- DESI Collaboration 2016, arXiv:1611.00036, doi:10.48550/arXiv.1611.00036 (Tables 2.3, 2.5).
- Planck Collaboration 2020, VIII Gravitational lensing, A&A 641, A8, doi:10.1051/0004-6361/201833886.
- Planck Collaboration 2020, VI Cosmological parameters, A&A 641, A6, doi:10.1051/0004-6361/201833910.
- Takahashi et al. 2012, ApJ 761, 152, doi:10.1088/0004-637X/761/2/152.
- Eisenstein & Hu 1998, ApJ 496, 605, doi:10.1086/305424.
- Lewis, Challinor & Lasenby 2000 (CAMB), ApJ 538, 473, doi:10.1086/309179.
- ESA, Euclid DR1 timeline update, 15 June 2026, https://www.cosmos.esa.int/web/euclid/dr1-timeline.
- IAM model and growth check: github.com/hmahaffeyges/IAM-Validation, docs/verification/scripts/verify_euclid_template.py.

## 7. Figure

`iam_signal_forecast.png` / `.pdf`:

- **Panel (a).** The IAM f sigma8 deficit relative to LCDM (same primordial amplitude). Shaded bands show A = 1 +/- sigma(A) for the pessimistic and optimistic Euclid + Planck lensing + DESI combinations, with Sigma fixed. DESI's published fractional f sigma8 errors (kmax 0.1 h/Mpc) and Euclid GCsp per-bin f sigma8 errors from this Fisher are drawn on the curve. The GCsp per-bin errors use the pessimistic setting with cosmology fixed and b, P_s, sigma_p and sigma_v marginalised.
- **Panel (b).** C_ell(IAM)/C_ell(LCDM) - 1 for one shear, one galaxy-galaxy lensing and one clustering spectrum under optimistic cuts. Shading shows the Gaussian 1-sigma error per coarse ell band, combining the sub-bins by inverse variance. The ratio is at fixed cosmology and bias, so it shows the raw signal before marginalisation.
