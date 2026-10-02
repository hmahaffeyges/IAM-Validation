# VIRIAL_CHECK — the five Virial papers

**Read status.** PDF text 277 / 641 / 674 / 560 / 222 = 2,374 lines. Earlier versions of this file said "re-read in full in 100-line chunks"; several of
those chunks were previews only, so that claim was false. All five were read in full 2026-10-02 in 50-line chunks, ledger complete with no gaps
(`docs/book/read_ledgers/LEDGER_G3_Virial.md`). Items 19–29 are from that read.

Read start to finish, oldest first: *Virial Efficiency and Effective Nonlinear Exponent* (25 Feb 2026, 7 pp), *The Virial Partition Across Wide Range of Physical
Scales* (25 Feb, 13 pp), *Dark Matter and Dark Energy as Virial Partners* (Mar, 12 pp), *Gravitational Decoherence, the Virial Partition, and the
Emergence of Classical Structure* (Mar, 12 pp), *The Thermodynamic Identity Governing the Virial Theorem* (PRL version, 18 Mar, 3 pp).
Every number below was recomputed (Planck 2018, Ω_m = 0.3153 unless stated).

## What stands — the core of chapter 2.1
- **The 1/2 is the virial theorem**, tested across every domain where 1/r binding is measured (*Wide Domains* Table 1; PRL Table I):
  20 atoms and 10 molecules (Hartree–Fock, T/|V| = 1 exact), equipartition, the Sun (Kelvin–Helmholtz, 24 Myr), the Chandrasekhar limit (1.456 M⊙ for µ_e = 2), galaxy clusters
  (~20 %). The factor is fixed by the degree of the potential (Euler), which Gauss's law fixes in three dimensions. Not a fit.
- **The Landauer identity** (PRL): first law Q = −E_f; second law ΔS ≥ Q/T in the surroundings; Landauer E_L = TΔS_min = Q; 1/r equilibrium −E_f = ⟨K⟩.
  ⟨K⟩ = Q is mechanics (first law + Euler); the PRL's "temperature cancels" (with T the system's temperature) is an identity and carries no physics.
  So ⟨K⟩ = Q = TΔS = E_L = ½|⟨V⟩|. The paper's own remark is the right statement: this identifies K, it does not derive the virial theorem.
- **β_m = Ω_m/2** = 0.1577 (0.1575 for Ω_m = 0.315). Prediction, enters every chain fixed.
- **E(a) = exp(1 − 1/a) = e^(−z)**: E(z = 10) = 4.5 × 10⁻⁵, E(1) = 1, dE/da > 0, E(∞) = e. 10/50/90 % at z = 2.30/0.69/0.11.
- **µ(a)**: µ0 = 0.864 (13.6 %); µ(0.3) = 0.922; µ(0.5) = 0.948 (5.2 %). Σ = 1 (photons, dτ = 0).
- **H0_matter** = 67.36·√1.1577 = 72.48 (Planck value) or 67.16·√1.1577 = 72.26 (Level 2 chain). σ8 0.813 → 0.800.
- **Matter–dark-energy equality**, Ω_m a⁻³ = Ω_Λ + β_m E(a): z = 0.361 (ΛCDM 0.295) ✓. **Ω_m^growth = Ω_m µ(z)**: 0.2952 is the value at
  z = 0.40; at z = 0.5 it is 0.2990 (the LRG1 entry 0.299 at z = 0.51 is right).
- **BH Landauer energy**: ½Mc² (Smarr), not (ln 2/2)Mc² = 34.7 % (errata V17; the earlier ✓ here was wrong). σ_crit from Eq. 9 with M_min = 10^8.4 M⊙: 3.9 km/s ✓ (topic out of the book by ruling).

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

## Added on the complete re-read (2026-10-02)
13. **PRL Step 1 scope.** Q = −E_f is the energy released when a system binds from rest at infinity; it holds where the binding energy is radiated (atom
    formation: 13.6 eV photon = ⟨K⟩; contracting gas clouds, stars). Collisionless dark-matter halos relax without radiating (violent relaxation conserves
    energy), so for them the identity is about the information written in relaxation, not emitted heat. The chapter states this scope.
14. **DESI Ω_m^growth comparison** (*Virial Partners* §4.1, Fig. 1): Ω_m µ(z) = 0.2990 at z = 0.5 (0.2952 is the z = 0.40 value). DESI DR1 FS+BAO
    0.2962 ± 0.0095 is 0.3σ from 0.2990 (paper: "0.02σ" against 0.2952).
15. **σ8 = 0.802 ± 0.020 "2025 joint KiDS-Legacy + DES Y3 + DESI"** (*Wide Domains* Test 1, Table 3; *Virial Partners* Table 2): source value to trace
    (Stölzner et al. 2025 report S8). The chapter prints the Level 2 S8 = 0.822 ± 0.011 against KiDS-Legacy S8 = 0.815 (+0.016/−0.021).
16. **Euclid dates**: "DR1 October 2026" in papers 2–4. Full DR1 with clustering and weak-lensing products is expected mid-2027; σ(µ0) ≈ 0.04 is the
    full-survey forecast (≈ 3.4σ).
17. *Wide Domains* Table 2, Run B "0.006 ± 0.156, Δχ² −1.90 (AIC-penalised)": the free-µ0 posterior reaches the +0.2 prior edge (LATE_TIME_GROWTH_CHECK #9);
    not printed.
18. *Wide Domains* §5 M–σ summary belongs to the black-hole group (G6) and is carried with that paper.

## Added on the complete read in 50-line chunks (2026-10-02)
19. **µ formula without H0²**: *Virial Partners* Eq. 6 and *Grav. Decoherence* Eq. 4 (as IAM_Law Eq. 33, Theory Eq. 58).
20. **"Suppression" = 1 − µ, not growth.** *Virial Partners* Eq. 9 ("fσ8 suppressed 7.9 % at z = 0.295 … 0.6 % at z = 1.491") and *Grav. Decoherence*
    §6.5 ("13.6 % growth suppression … confirmed") quote 1 − µ(z). The fσ8 change at fixed early amplitude is 4.2 % (z = 0), 2.2 % (0.3), 1.3 % (0.5),
    0.4 % (1.0) for µ·G, within 0.3 % for the friction form; σ8 drops ~1 %. No measurement has confirmed it.
21. **DESI Ω_m 0.2962** (*Virial Partners* §4.1, Table 2) is the DR1 full-shape + BAO ΛCDM value, dominated by BAO geometry, which IAM leaves unchanged.
    It is not a growth-only Ω_m, so "lands on the IAM curve at 0.02σ" (or 0.3σ, item 14) is not a test. Needs a growth-only fit with an IAM template.
22. **DESI DR2 phantom crossing** (*Virial Partners* §4.3): the w0w_a preference comes from BAO + supernova + CMB distances, with no growth data, so the
    "distances + suppressed growth" mechanism does not apply. The dual-sector reading is the two-ruler test (`TWO_RULER_DESI_TEST.md`).
23. **Wide Domains Table 4** says 7 H0 points and lists 6; χ² (IAM 5.57, ΛCDM 55.20) does not reproduce from the listed values (IAM 0.88; ΛCDM held at
    67.36: 51.2; a fitted single H0 = 68.9 gives 37.2). Δχ² +49.6 / +61.2 not printed (item 8).
24. **Wide Domains σ8**: ΛCDM 0.8139 is 0.59σ from 0.802 ± 0.020 (printed 0.45σ). Table 9 "±0.22 → 0.85σ" is 0.62σ; Run B's σ is 0.156.
25. **KiDS-Legacy reading** (*Wide Domains* §4.2): weak-lensing S8 measures matter clustering, which IAM suppresses; it does not "converge to the unmodified
    photon sector". 0.815 lies between Planck ΛCDM (0.832) and the Level 2 chain (0.822) and does not separate them.
26. **Three-way cluster test / M_lens vs M_dyn** (*Wide Domains* §6.2): with the Poisson equation unmodified (Level 2, Theory §11), hydrostatic and lensing masses
    of a relaxed cluster agree; the predicted suppression of R1, R3 holds only in the Level 1 µ·G form. Implementation-dependent; held (item 12).
27. **Equipartition** (*Wide Domains* Table 1, §2, conclusions): k_BT/2 per quadratic degree of freedom is not the 1/r virial ½ (a quadratic potential gives
    ⟨K⟩ = ⟨V⟩). Removed from the table. "Landauer kT/2 per decoherence event" → k_BT ln 2 per bit. "Coulomb V ∝ +1/r": the electron–nucleus potential is
    attractive, −1/r.
28. **Grav. Decoherence Eq. 11** writes the µ·G form but calls it "an additional Hubble friction term"; state which implementation. §4 σ_crit:
    M_min = 4Ω_m σ³/(GH) prefactor not derived, ~100× normalisation offset admitted; out with Mechanism B (MISSING_SATELLITES_CHECK).
29. **Virial Efficiency Table 1** also misattributes methods: Power, Knebe & Knollmann 2012 is a cosmological N-body study, not GIMIC/OWLS; Bett 2007 (spin and
    shape), Ludlow 2010 and Bryan & Norman 1998 do not tabulate a mass-weighted 2K/|U| (NBODY_TRACE). The falsification thresholds of its Table 4 rest on those
    values. Acknowledgments in *Grav. Decoherence* name correspondents (errata N1).
