# VIRIAL_CHECK — two numbers in the Virial papers, checked before the book (2026-10-01)

Read in full: *Virial Efficiency and Effective Nonlinear Exponent* (25 Feb 2026) and *The Thermodynamic Identity Governing the Virial Theorem* (PRL version, 18 Mar 2026).

**1. "17 MCMC chains return β_m = 0.1583 ± 0.0033" (PRL Table I and text).** β_m is fixed in every chain; no chain samples it. The number
equals Ω_m/2 from the Level 2 Run A chain: Ω_m = 0.3166 ± 0.0065 → 0.1583 ± 0.0032 (CHAIN_EXTRACTION_FINAL.csv). The ΛCDM chain gives the same,
0.1581. So it is Planck's matter density halved. It is not a measurement of β_m, and it is the same in ΛCDM.
**For the book:** β_m = Ω_m/2 = 0.1575 is the virial prediction and enters the chains fixed. What the chains test is whether the
growth modification that follows from it is consistent with Planck (Δχ² +0.5 to +1.7) and what free µ0 returns (consistent with both 0 and −0.135).
Do not print 0.1583 ± 0.0033 as a confirmation.

**2. "Six N-body studies measure 2K/|U| = 0.815 ± 0.025."** In those studies the virial ratio 2T/|U| of halos is at or above 1. Neto et al. (2007)
call a halo relaxed when 2T/|U| < 1.35 (also Duffy et al. 2008 and MUSIC-2). The values tabulated in the paper (0.76–0.90) are not the published 2K/|U|.
If they are |U|/2K, the reciprocal, the definition in the paper is inverted and the argument η_vir = 1/(2 f_coll) has to be restated for that quantity.
Each table entry has to be traced to a figure or table in its source before it can be printed. The n_eff table (3.22 ± 0.44) needs the same tracing.
**For the book:** chapter 2.1 prints the virial partition ⟨K⟩ = ½|⟨V⟩| and β_m = Ω_m/2 as the prediction, with the Landauer identity
(first law, second law, Landauer, 1/r equilibrium) from the PRL paper. The N-body "confirmation" stays out until each value is traced.

**3. Also flagged in the PRL table:** "Electron m_e, 6.6 ppm". This is the Particle group's result; it goes in only as audited there.
"Atoms T/|V| 1.0000": exact for converged Hartree–Fock by construction (the virial theorem holds for any variationally optimised wavefunction under
scaling). It is a property of the method, not evidence for the identity; print it that way.

---
## Papers 2–4, read in full (Virial Partition Across Wide Domains, 25 Feb; DM and DE as Virial Partners, Mar; Gravitational Decoherence and the Virial Partition, Mar)

**Reproduced.** µ0 = 1/(1+β_m) = 0.864; µ(z = 0.5) = 0.948 (5.2 % suppression); H0_matter = 67.36·√1.15765 = 72.48 (Planck) or
67.161·√1.15765 = 72.26 (Level 2 chain); E(z = 10) = 4.5 × 10⁻⁵; E(∞) = e; ΛCDM deceleration transition z_t = 0.632.
The R(a) column of the virial-ratio table (399, 20, 6, 2) is correct.

**To fix before the book.**
1. *Virial Partners* Table 1, E(a) column: 0.050 / 0.182 / 0.274 are wrong. E(a) = e^(−z) = 0.135 / 0.497 / 0.741. The R column was computed with the correct values.
2. R(1) = Ω_m/β_m = 2 holds by construction (β_m = Ω_m/2, E(1) = 1); the paper says so itself. It cannot be offered as evidence, and it does not
   bear on the coincidence problem, which concerns Ω_m against Ω_Λ: R compares Ω_m a⁻³ with β_m E(a). Print the ratio only as a definition.
3. *Virial Partners* §4.2: w_info = −4/3 and z_t(IAM) = 0.718 need a modified background. The model as run modifies the matter perturbations only
   ("no background quantities are modified", *Wide Domains* §1), and the chains that did modify the background (Level 2b) were excluded (H0 ≈ 61.5).
   With the background unmodified, z_t = 0.632 for both models. Leave z_t out unless a matter-sector q(z) is defined and derived.
4. *Wide Domains* Table 6: SNe Ia are counted as photon-sector probes and SH0ES (Cepheid-calibrated SNe Ia) as matter-sector. By the author's
   ruling, supernovae sit on the matter ruler; correct the census. H0LiCOW time delays measure photon paths, so assign them by the same rule.
5. *Wide Domains* §4.6: the χ² totals (36.16 vs 97.35, Δχ² +61.2) are dominated by assigning H0 measurements to sectors by hand (+49.6).
   That is a reclassification, not a fit; print it as the consequence of the sector assignment, not as a goodness-of-fit result.
6. *Gravitational Decoherence* §5.3 and abstract: "best-fit improvement Δχ² = +0.54". The positive sign means IAM's χ² is higher; the book says
   "consistent with Planck, Δχ² = +0.54". The same paper says all 17 chains converged to R − 1 < 0.01; at the paper's date Runs A and B were at
   0.023 and 0.020. Today all 18 are ≤ 0.010.
7. *Gravitational Decoherence* §4.3: M_min uses "Kim & Peter 2021" for the halo occupation drop; that arXiv number is a paper on SIDM cluster
   mergers, so the citation is wrong. σ_crit and missing satellites are out of the book by ruling.
8. "Three-channel decomposition" of β_m (temporal / geometric / radiative, *Wide Domains* §3.3): no derivation in any paper read; flag as speculative.
10. **The exponent n (top-down).** With İ ∝ ρ_m D^n H ∝ a^(n − 9/2) in matter domination, the value depends on what the record is accumulated over:
    per unit scale factor (∫ İ da ∝ a^(n − 7/2)) gives **n = 5/2**, matching the bottom-up Press–Schechter value; per d ln a, as the integral is printed
    in the papers (∫ a^(n − 9/2 − 1) da), gives 7/2; per unit time (∫ İ dt) gives 2. The book uses the da form, where both routes meet at 5/2, and
    the printed integral is corrected to ∫ a^(n − 9/2) da. The choice of measure is stated as part of the derivation (sympy check in this folder).
9. The dark-matter/dark-energy identification (*Virial Partners*) is interpretation; it belongs in Part 5 with its question stated as a question.

**What 2.1 prints:** the virial theorem for 1/r potentials; the Landauer identity ⟨K⟩ = Q = TΔS (PRL paper); β_m = Ω_m/2 as the prediction;
E(a) = exp(1 − 1/a) with its properties; µ(a) and its values; the matter-sector H0. The N-body numbers wait until they are traced to source.
