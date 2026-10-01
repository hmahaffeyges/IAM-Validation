# VIRIAL_CHECK — the five Virial papers, read in full (2026-10-01)

Read start to finish, oldest first: *Virial Efficiency and Effective Nonlinear Exponent* (25 Feb 2026, 7 pp), *The Virial Partition Across Wide Range of Physical
Scales* (25 Feb, 13 pp), *Dark Matter and Dark Energy as Virial Partners* (Mar, 12 pp), *Gravitational Decoherence, the Virial Partition, and the
Emergence of Classical Structure* (Mar, 12 pp), *The Thermodynamic Identity Governing the Virial Theorem* (PRL version, 18 Mar, 3 pp).
Every number below was recomputed (Planck 2018, Ω_m = 0.3153 unless stated).

## What stands — the core of chapter 2.1
- **The 1/2 is the virial theorem**, tested across every domain where 1/r binding is measured (*Wide Domains* Table 1; PRL Table I):
  20 atoms and 10 molecules (Hartree–Fock, T/|V| = 1 exact), equipartition, the Sun (~10 %), the Chandrasekhar limit (1.44 M⊙), galaxy clusters
  (~20 %). The factor is fixed by the degree of the potential (Euler), which Gauss's law fixes in three dimensions. Not a fit.
- **The Landauer identity** (PRL): first law Q = −E_f; second law ΔS = Q/T; Landauer E_L = TΔS = Q (T cancels); 1/r equilibrium −E_f = ⟨K⟩.
  So ⟨K⟩ = Q = TΔS = E_L = ½|⟨V⟩|. The paper's own remark is the right statement: this identifies K, it does not derive the virial theorem.
- **β_m = Ω_m/2** = 0.1577 (0.1575 for Ω_m = 0.315). Prediction, enters every chain fixed.
- **E(a) = exp(1 − 1/a) = e^(−z)**: E(z = 10) = 4.5 × 10⁻⁵, E(1) = 1, dE/da > 0, E(∞) = e. 10/50/90 % at z = 2.30/0.69/0.11.
- **µ(a)**: µ0 = 0.864 (13.6 %); µ(0.3) = 0.922; µ(0.5) = 0.948 (5.2 %). Σ = 1 (photons, dτ = 0).
- **H0_matter** = 67.36·√1.1577 = 72.48 (Planck value) or 67.16·√1.1577 = 72.26 (Level 2 chain). σ8 0.813 → 0.800.
- **Matter–dark-energy equality**, Ω_m a⁻³ = Ω_Λ + β_m E(a): z = 0.361 (ΛCDM 0.295) ✓. **Ω_m^growth = Ω_m µ(z)**: 0.2952 is the value at
  z = 0.40; at z = 0.5 it is 0.2990 (the LRG1 entry 0.299 at z = 0.51 is right).
- **BH Landauer fraction** ln 2/2 = 34.7 % ✓. σ_crit from Eq. 9 with M_min = 10^8.4 M⊙: 3.9 km/s ✓ (topic out of the book by ruling).

## The N-body row (*Virial Efficiency*; PRL Table I row 8)
The paper defines η_vir = 1/(2 f_coll) = 0.81 so that f_coll·η_vir = 1/2, then compares it with the published halo ratio 2K/|U|.
*Wide Domains* §7 states the status correctly: η and f_coll are "representative literature values; their product ≈ 1/2 is required by the virial
theorem". So β_m = Ω_m/2 does not rest on this row — it rests on the theorem and on the cross-domain table above.
Source check (NBODY_TRACE.md, arXiv full texts): the six simulation papers report 2T/|U| of about 1.05–1.4 (Bett, Neto, Power, Klypin; Neto's relaxed
cut is < 1.35); 0.76–0.90 match the reciprocal |U|/2T. Bryan & Norman report f_σ/f_T, and Ludlow only a cut. The n_eff values are not reported as
such in those papers (computed from their mass functions they are 2.5–3.8 only at 10^13.4–10^14.2 M⊙). Tinker 2008 reports no f_coll;
F = 0.62 needs halos down to ~10^8 M⊙.
**For the book:** print the cross-domain table without an N-body efficiency row; the halo row reads "clusters, K/|U| ≈ 1/2 to ~20 %".
If the efficiency argument is kept, it is restated with |U|/2K and its own definition (Part 2 appendix), not as six-study confirmation.

## What changes before the book
1. **β_m = 0.1583 ± 0.0033 "returned by 17 chains"** (PRL, *Virial Partners* §2, *Grav. Decoherence* §2.1): β_m is fixed in every chain; 0.1583
   is Ω_m/2 of the Level 2 Run A posterior (0.3166/2); the ΛCDM chain gives 0.1581. Print instead: Planck's Ω_m gives β_m = 0.1577, which the chains
   use fixed, and the modified growth is consistent with Planck (Δχ² +0.5 to +1.7).
2. **Exponent n = 7/2.** Theory paper Eqs. 36–39 give S ∝ a^(n − 9/2); n − 9/2 = −1 ⟹ 7/2 (the papers print 5/2). Full-ΛCDM integration and the Theory paper's Table 2 agree. Changed in the book with the author's approval (2026-10-02). E(a) is independent of n. The bottom-up 'Press–Schechter D^(5/2)' statement in other papers is not yet read line by line and is not printed.
3. *Virial Partners* Table 1, E(a) column: 0.135 / 0.497 / 0.741 (the printed 0.050 / 0.182 / 0.274 are wrong; R column 399 / 20 / 6 / 2 is right).
4. R(1) = Ω_m/(β_m E(1)) = 2 holds by construction (the paper says so). Print it as the definition it is. The coincidence-problem reading is
   interpretation (Part 5).
5. "Informational pressure 23 % of total dark energy today": β_m/(Ω_Λ + β_m) = 18.7 %; 23 % is β_m/Ω_Λ. At z = 0.3/0.7/1.5: 14.6/10.3/4.9 %.
6. w_info = −4/3 and z_t(IAM) = 0.718 need a modified background; the model modifies matter perturbations only ("no background quantities are
   modified", *Wide Domains* §1) and the background runs (Level 2b) were excluded. With the background unmodified, z_t = 0.632. Out until a
   matter-sector q(z) is defined.
7. *Wide Domains* Table 6 puts SNe Ia in the photon sector; by ruling supernovae sit on the matter ruler. Table 4 puts H0LiCOW (time delays,
   photon paths) in the matter sector. Redo the census by the worldline rule.
8. *Wide Domains* §4.6: Δχ² +61.2 is dominated (+49.6) by assigning H0 measurements to sectors; print as the consequence of the sector assignment.
9. Δχ² sign wording (*Grav. Decoherence* §5.3, "best-fit improvement +0.54"): IAM's χ² is higher by 0.54; print "consistent with Planck".
   "All 17 converged to R − 1 < 0.01": true today for all 18; at the paper's date Runs A and B were at 0.023 and 0.020.
10. Euclid σ(µ0): *Wide Domains* says DR1 ±0.08; *Virial Partners* and *Grav. Decoherence* say ±0.04 at DR1. The ±0.04 is the final-survey forecast.
11. Citation: "Kim & Peter 2021" (arXiv 2106.05984) is a paper on SIDM cluster mergers, not halo occupation (topic out by ruling).
12. Speculative, flag for discussion: the three-channel split of β_m (*Wide Domains* §3.3, no derivation given); DM/DE as the two halves and
    "what is spacetime made of" (*Virial Partners*); arrow-of-time and England/Rovelli/Smolin sections (*Grav. Decoherence* §6) → Part 5.
