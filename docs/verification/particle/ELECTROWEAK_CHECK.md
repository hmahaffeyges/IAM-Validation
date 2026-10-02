# ELECTROWEAK_CHECK — "Electroweak Symmetry Breaking and the Matter Sector" (Mar 2026, revised Oct 2026, 7 pp)
Read in full 2026-10-02 (PDF text 207 lines, ledger complete). Carried: `p2_particle_masses.tex` §"When the matter sector begins".
## Reproduced
E(z=10) = 4.5e-5, E(z=30) = 9.4e-14, E(z=1) = 0.37; Ω_b/Ω_m = 15.6 %; β_m = 0.15765; Cornell form and string tension; m_γ bound; η = 6.12e-10.
## Corrections
1. Eq. 2: Ω_dm/2 = 0.1330 (paper 0.1332); 0.0247 + 0.1330 = 0.1577.
2. §8 test 3 "β_m/Ω_m = 1/2 holds for any revision of Ω_m": true by definition; not a test. Removed.
3. §2 "the kinetic half is the heat released ... the Landauer cost": the book states IAM's Law first, with the virial half setting how much bound matter writes.
## Book chapter (2026-10-02)
Full treatment: `docs/book/part2/p2_22_electroweak.tex` (\label{ch:electroweak}); the summary section "When the matter sector begins" in
`part2/p2_15_particle_masses.tex` stays. Numbers: `docs/verification/scripts/verify_entanglement_electroweak.py` (W1–W12).
## Found while writing the chapter (in PAPER_ERRATA.md as EW3-EW5, 2026-10-02)
4. Abstract and Table 1 "T ≈ 100 GeV, t ≈ 10⁻¹¹ s" vs §4.1 "160 GeV": the Standard Model crossover is at T_c = 159.5 ± 1.5 GeV (D'Onofrio &
   Rummukainen 2016); t = 0.301 g*^(-1/2) m_P/T² = 9.2 × 10⁻¹² s for g* = 106.75; a_EW = 4.9 × 10⁻¹⁶, ln E = −2.0 × 10¹⁵.
   Book: `part5/p5_03_time.tex` line 24 prints "t ≈ 10⁻¹² s"; should be ≈ 10⁻¹¹ s (9 × 10⁻¹² s).
5. §4.1 "the Higgs expectation value vanishes": in a crossover there is no sharp order parameter; "negligible above the crossover".
6. §7 vacuum selection: points of the Higgs vacuum manifold are related by local gauge transformations and cannot be selected spontaneously
   (Elitzur 1975); no gauge-invariant quantity records a choice. Open question narrowed to: is any physical outcome selected at the electroweak epoch.
7. §8 test 2 "5.4σ with DESI Year 5": unsourced (same as SP5, EN3); not carried. §4.3 "σ(Σ0) ≈ 0.02 [Euclid Collaboration 2020]": attribution
   not checked; not carried (the book states the lensing test without a σ).
8. Table 1: nucleosynthesis "∼ 3 min, 0.1 MeV": t(0.1 MeV) = 132 s, t(0.07 MeV) = 269 s; recombination age 3.7 × 10⁵ yr at z* = 1089.9 (Planck 2018);
   first haloes z = 30 at 98.7 Myr. The grand-unification row is not carried.
