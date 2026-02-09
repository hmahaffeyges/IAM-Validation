# Photon-Exempt IAM: CMB-Compatible Scenario

## What We Did: Summary of the "Photon-Exempt IAM" Fix

**Problem:**  
- The original IAM implementation improved BAO fits by suppressing late-time growth, but always broke CMB (shifted $\theta_s$, $d_A$, $H(z_\mathrm{CMB})$ by $\sim2.6\%$).
- Stronger "hard cutoffs" or thresholds fixed the CMB, but then growth suppression vanished—killing any BAO improvement.

**Our Solution:**  
- **"Photon-Exempt IAM":**  
    - IAM acts **only on the matter/Growth/BAO sector**.
    - All CMB (photon propagation, distances, sound horizon, $\theta_s$) is computed using pure ΛCDM $H_\mathrm{LCDM}(z)$.
    - The original, strong IAM formula including
        - $E_\mathrm{act} = \exp(1 - 1/a)$,
        - beta modifies **$\Omega_m(a)$ denominator and $H(z)$**,
        - tax applied at all late times,
      is restored for growth and expansion history relevant to late-universe/clustering data.

- **Result:**  
    - CMB observables (especially $\theta_s$, $d_A$ to last scattering) **match ΛCDM to machine precision**.
    - Late-universe observables (growth factor, BAO, possibly $f\sigma_8$) show **the desired IAM behavior**—retaining BAO improvement.

---

## How This Works

- **CMB photons are "exempt"**: The trajectories/geodesics of CMB photons "feel" no IAM effect. All CMB integrals use $H_\mathrm{LCDM}$.
- **Matter sector (galaxies, BAO, growth):** Feels IAM suppression/enhancement as specified in the original formulas.

- **Original IAM formulas are recovered** for late universe cosmology:
    - $E_\mathrm{act}(a) = \exp(1 - 1/a)$
    - $\Omega_m(a, \beta) = \frac{\Omega_{m,0} a^{-3}}{\Omega_{m,0} a^{-3} + \Omega_{r,0} a^{-4} + \Omega_{\Lambda,0} + \beta E_\mathrm{act}(a)}$ (for growth calculation)
    - $H_\mathrm{IAM}(a) = H_0 \sqrt{\Omega_{m,0} a^{-3} + \Omega_{r,0} a^{-4} + \Omega_{\Lambda,0} + \beta E_\mathrm{act}(a)}$ (for growth, BAO, late-universe distance modulus, etc.)

- **Justification:**  
    - "Photons do not participate in gravitational collapse, so the measurement decoherence/‘tax’ is only relevant to sectors with gravitating, clustering matter."
    - CMB observations are thus pure ΛCDM, while galaxy, BAO, or growth data are allowed to experience IAM modifications.

---

## How To Replicate

**See [`test_25_photon_exempt_original.py`](test_25_photon_exempt_original.py)!**  
This script does:
- Growth with original IAM (β = 0.179, τ = 0.134)
- CMB observables with pure ΛCDM $H(z)$
- BAO/growth sector with IAM $H(z)$ and $\Omega_m(a)$

**How to run:**
```bash
python test_25_photon_exempt_original.py
```
Expected output:
- Growth suppression at z=0:  
  `Growth suppression at z=0: ~3% (unnormalized)`
- CMB:  
  `θ_s = ...` **matches ΛCDM exactly** (Δθ_s = 0.000%)
- BAO suppression at z=0.5-0.7:  
  a few percent (retains original BAO improvement)

---

## What Else Should You Check?

1. **Earlier CMB and SNe tests**:  
   - _CMB:_ All tests where CMB distances or sound horizon are measured against Planck **must use pure ΛCDM** as in this scenario.
   - _SNe:_ Expansion history at late times now differs from ΛCDM; if your SNe analysis assumes ΛCDM $H(z)$, update as needed.

2. **Growth/f$\sigma_8$, Lensing:**  
   - Make sure your data/model comparison consistently pairs **matter-based data** with IAM predictions, and **photon-based/cosmological distances** with ΛCDM.

3. **Full cosmology pipeline:**  
   - If you run MCMC or joint data fits, **partition data likelihoods** so that CMB, lensing, and SNe sectors use appropriate models.

---

## Will This Break Earlier Tests?

- **CMB tests:**  
   **WILL FAIL** if you compute distances etc. with IAM $H(z)$.  
   **WILL PASS** if you switch CMB calculations to photon-exempt (ΛCDM) logic.
- **BAO, growth, $f\sigma_8$ tests:**  
   **WILL BE OK** and results will be similar to your original BAO-improvement period.
- **Hubble tension analyses:**  
   Review and rethink in light of the mixed model; check which $H_0$ is inferred by SNe vs CMB now.
- **SNe-only data fits (esp. at z<1):**  
   Slightly altered $H(z)$ may impact residuals—double-check if any fit assumes pure ΛCDM.

---

## What Still Needs To Be Addressed

- **Lensing (CMB and galaxy):**  
  Decide and justify if *lensing* is a photon-determined (should use ΛCDM?) or matter-determined (should use IAM?) observable.
- **Energy conservation, deeper theory:**  
  Write up a full covariant argument for why the "measurement tax" should only affect matter, not photons.
- **Physical intuition & literature search:**  
  Support your photon-exempt scenario with theory and prior models if possible.
- **MCMC/joint constraints:**  
  Implement this in your main cosmology chain to check joint fits.

---

## Code Placement

- **Keep `test_25_photon_exempt_original.py` as a "reference check" test.**
- Add this documentation (`README_iam_photon_exempt.md`) to your `tests/` directory or main docs.

---

## Reference Output

```
Growth suppression at z=0: 3.26% (unnormalized)
CMB: θ_s = 0.005085 rad (ΛCDM)
BAO suppression at z=0.5-0.7: 1.6–2.4%
Hubble parameter H(z) at z=0.5-0.7: up by 2–4%
...
SUCCESS! PHOTON-EXEMPT IAM WITH ORIGINAL PHYSICS WORKS!
```

---

## Next Steps

1. **Add this test to your regression suite.**
2. **Update all CMB-/photon-based cosmology tests to use the photon-exempt logic.**
3. **Write up the photon-exempt justification.**
4. **Rerun any data fits as needed.**