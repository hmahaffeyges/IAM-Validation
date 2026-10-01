# Cosmological constant derivation — recomputation (2026-10-01)

Source: docs/papers/latex/iam_cosmological_constant/iam_cosmological_constant.tex. Parameters from the 18th chain
(mgcamb_validation/chains/iam_baryon_test.1.txt, BBN prior on Ω_b h² removed, 30 % burn-in, 14,957 weighted rows).

## The 18th chain (CMB only, no BBN prior)
| | posterior |
|---|---|
| Ω_b h² | 0.02232 ± 0.00014 |
| H0 | 67.04 ± 0.53 |
| Ω_m | 0.3197 ± 0.0075 |
| Ω_Λ | 0.6802 ± 0.0075 |
| Ω_b/Ω_m | 0.1554 ± 0.0019 |
| η = 273.9×10⁻¹⁰ Ω_b h² | (6.113 ± 0.037)×10⁻¹⁰ — observed 6.137 |

Both the baryon asymmetry and the CC relation are read from this one posterior; they share parameters and are not
independent tests.

**Settings, from the chain's own yaml (iam_baryon_test.updated.yaml, MGCAMB/Cobaya 3.6, CAMB 1.5.2):**
- Ω_b h²: flat prior 0.010–0.040 — **no BBN prior**. Data: Planck 2018 low-ℓ TT, low-ℓ EE, high-ℓ plik-lite TTTEEE, lensing. No
  BAO, no SNe, no H0 prior, no deuterium or helium abundance data.
- μ0 fixed at −0.13495 (IAM); H0, τ, ln A, n_s, Ω_c h² free; Rminus1_stop 0.01.
- **Helium:** YHe is not a sampled or fixed parameter, so CAMB sets it from Ω_b h² with its built-in BBN consistency relation
  (standard practice in Planck analyses). Nuclear physics therefore enters only through the helium fraction used to compute the
  CMB damping tail, not as a constraint on Ω_b h². To remove even that, re-run with YHe fixed (e.g. 0.245) or free.

## What holds
ρ_Λ/ρ_vac = (3Ω_Λ/8π)(l_P/l_H)² identically (critical density in Planck units). The paper's formula therefore reduces to
**Ω_b/Ω_m = (3/16)√Ω_Λ**. On the 18th chain: ratio 1.0049 ± 0.0068; difference 0.0008 ± 0.0011 (0.7σ). Holds at the present
epoch only (Ω_Λ(a) changes, Ω_b/Ω_m does not).

## What does not
1. The 10⁻¹²³ cancels on both sides; it is H0 in Planck units, an input.
2. 2/π: with A_eff = 2π l_H², l_P²/(A_eff/4π) = 2(l_P/l_H)², not (2/π)(l_P/l_H)².
3. History integral, eq. (integral) as written, evaluated from a_EW = 2.3×10⁻¹⁵: coefficient 3×10³⁰ against the required 0.523;
   the integrand grows as a⁻³ at early times, so the earliest epoch dominates.
4. Weighting the same quantity by the activation function's growth dℰ(a) gives O(1) coefficients that depend on the form
   chosen: ∫(H0/H)² dℰ = 0.470, ∫(H0/H) dℰ = 0.641, ∫dℰ = 1, ∫(H/H0) dℰ = 1.98, ∫(H/H0)² dℰ = 5.80. None is selected by a
   derivation; picking the closest (0.470, 10 % low) after seeing the target would be a fit.

## Status for the book
Present the relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ as an observed present-epoch relation (0.5 %, 0.7σ on the CMB-only chain), with the
derivation open: the 2/π factor and the history-weighting both need a derivation fixed before comparison.

## The baryon paper (Baryon_Asymmetry_as_a_Derived_Quantity…pdf), read in full and checked
1. **The chain does not test IAM's constraint.** Eq. (3)/(5) is not in the chain; the chain measures Ω_b h² from the CMB acoustic
   peaks, as any Planck fit does. The ΛCDM baseline chains in the repo (no μ0, flat Ω_b h² prior 0.020–0.025) return the same
   value: lcdm_baseline 0.02234 ± 0.00014 → η = 6.120; planck_bao 6.139; planck_pantheon 6.119; planck_rsd 6.136 (×10⁻¹⁰).
   The 18th chain (IAM, prior 0.010–0.040) gives 0.02232 ± 0.00014 → 6.113. Widening the prior and fixing μ0 change nothing.
2. **The "standard configuration" had no BBN prior either.** N(0.02242, 0.00014) is Cobaya's `ref` (starting-point) distribution,
   not a prior; the prior in every Level 1 yaml is flat 0.020–0.025. Planck's own Ω_b h² is a CMB-only measurement.
3. **CMB–BBN agreement on η is established concordance** (WMAP, Planck), not new; the paper's Table 1 lists it (6.136 ± 0.038).
4. **The analytic numbers do not reproduce.** Inverting eq. (3) with Planck values gives Ω_b h² = 0.01839, η = 5.04×10⁻¹⁰
   (paper: 6.079). Inverting eq. (5) gives 0.02222, η = 6.09×10⁻¹⁰ (paper: 6.115).
5. **What survives:** eq. (5) inverted gives η = 6.09×10⁻¹⁰ against the CMB 6.11–6.14 (0.4–0.8 %). This is the same single
   relation, Ω_b/Ω_m ≈ (3/16)√Ω_Λ, read in a different variable. The baryon asymmetry and the cosmological constant are one
   observed relation, not two confirmations; its derivation is open (2/π, history weighting).
