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
