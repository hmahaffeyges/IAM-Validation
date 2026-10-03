# MANIFEST — line-for-line carriage: Lensing Dynamics, Three-Way Cluster Mass, Missing Satellites (2026-10-03)

Base: sparse clone of the public repo at HEAD e37aabb (later than 41f7646 / 12d8fb1). Nothing pushed or committed.

## Files delivered
| File | What |
|---|---|
| docs/book/part2/p2_17_lensing_dynamics.tex | Lensing mass and dynamical mass (311 lines; was 89) |
| docs/book/part2/p2_18_three_way_clusters.tex | Three cluster masses (265 lines; was 85) |
| docs/book/part2/p2_19_missing_satellites.tex | Missing satellites: the closure condition and two mechanisms (324 lines; was 85) |
| docs/book/bib_clusters_satellites.bib | 25 CrossRef-verified entries + 1 arXiv-only (Merloni2012) |
| docs/book/figscripts/fig_p2_satellites_closure.py | NEW: fig_sat_closure (µ(z); four-regime closure picture with census) |
| docs/book/figscripts/fig_p2_satellites.py | EDITED: fig_sat_mechanisms panel (b) now shows the corrected (6/π) dynamical-time form and the Nadler et al. 2020 bound |
| docs/book/figures/part2/fig_sat_closure.{pdf,png}, fig_sat_mechanisms.{pdf,png}, fig_sat_census.{pdf,png} | regenerated |
| docs/book/figures/part2/fig_lensdyn_forms, fig_lensdyn_test, fig_threeway_estimators, fig_threeway_slope | unchanged, included for completeness |
| docs/verification/scripts/verify_cluster_mass_satellites.py (+ _output.txt) | every \calc / \derived number in the three chapters (sections A–I) |

**main.tex:** chapter order unchanged (lines 38–40 already input p2_17, p2_18, p2_19). One line to change for the bibliography:
`\bibliography{iam,bib_clusters_satellites}` (line 115).

## Reading ledger (PDF text via pypdfium2, page markers included)
| Paper | Lines (this extraction) | Ledger PAPER_LINE_COUNTS.md | Read | LaTeX checked |
|---|---|---|---|---|
| IAM_Lensing_Dynamics_Paper | 316 | 316 | 1–316 in 50-line chunks, no truncation | IAM_Lensing_Dynamics_Paper.tex (497 lines), all 8 equations |
| 3Way_Mass_Discrepancy_in_Galaxy_Clusters | 304 | 304 | 1–304 in 50-line chunks | IAM_ThreeWay_Cluster_Paper.tex (498 lines), all 15 equations |
| Missing_Satellites | 578 | 574 | 1–578 in ≤50-line chunks | iam_missing_satellites.tex (533 lines), all 12 equations |
Also read in full: PAPER_ERRATA.md rows LD1–LD5, TW1–TW6, M2–M8, V11; LENSING_DYNAMICS_CHECK.md (20 lines), THREE_WAY_CLUSTER_CHECK.md (19),
MISSING_SATELLITES_CHECK.md (37); coverage wave-2 headers and ERRATUM/FLAG comments (19_, 20_, 24_ .tex). Sources traced for numbers:
Planck 2015 XXIV (arXiv 1502.01597) Table 2 and Sect. 5; Smith et al. 2016 (LoCuSS, arXiv 1511.01919) Sects. 2–3.

## Corrections applied (corrected form only in the chapters)
- LD1/TW1: the ratio 1/µ holds only in the Level 1 (G_eff) form; stated with that form throughout; Level 2 gives 1. New supporting derivation: in the
  Level 1 form with Σ = 1, Φ/Ψ = (2−µ)/µ = 1.315 today — an effective slip, so "unmodified Einstein equations" and "ratio 1/µ" cannot both hold.
- LD2: f(R) Σ = 1, 1 ≤ µ ≤ 4/3; normal DGP µ > 1, Σ = 1; self-accelerating DGP µ < 1, Σ = 1 (ghost); IAM not unique in its signs.
- LD3: Planck SZ tension reduced (σ8 −1.57 % L1, −1.10 % L2), not resolved; 1−b = 0.58 ± 0.04 from Planck 2015 XXIV.
- LD4: CCCP/WtG traced: Planck 2015 XXIV Table 2 priors 1−b = 0.780 ± 0.092 and 0.688 ± 0.072 (→ 1.28 ± 0.15, 1.45 ± 0.15), replacing 1.20 ± 0.12
  and 1.31 ± 0.11; LoCuSS β_X = 0.95 ± 0.05 and β_P redshift split added; Herbonnet 2020 1−b = 0.84 ± 0.04 ± 0.05 (abstract) added.
- LD5/TW6/M4/M6: 18 chains; β_m fixed in every chain (posterior 0.1583 = Ω_m/2 on the posterior Ω_m), chains test not verify.
- TW2: M_SZ/M_hydro ≈ 1 holds by Y–M calibration — not a test. The value 0.99 ± 0.04 (Bulbul 2024) was not traced and is not printed.
- TW3: the four "observed" binned ratios are withheld until each bin is traced (values in the table below; recomputed deviations).
- TW4: σ(dR/dz) = 0.03 / 0.008 and 6σ / 22σ not derived; the arithmetic is shown and labelled \openprob; required precision computed instead (1.5 % per bin).
- TW5: §9.2 cut.
- M2: Eq. 11 gives 10^8.44 at 4 km/s; the 100× offset is not carried.
- M3/M8: Mechanism B as stated rejected by the census (25 of 54 below 4 km/s; 16 below 3.37, 22 below 3.65); infall-σ re-test open.
- M5: ΔD/D = −0.78 % (form i), −0.67 % (ii), −1.87 % (iii).
- M7 and author ruling 2026-10-03: no σ²/σ⁴ family, no cusp-core, no M–σ, no small-scale/timestamp material.
- V11: Kim & Peter 2021 replaced by Nadler et al. 2020 (ApJ 893, 48): faintest satellites in halos with M_peak < 3.2 × 10^8 M⊙ (95 %).
- LG10/T10: µ written with β_m E(a) H0². Canon β_m = 0.15765 (tables: 1.158 / 15.8 % at z = 0; the papers print 1.157 / 15.7 % with 0.1575).
- Check file (MS): the two-channel wording ("potential half to curvature, kinetic half Landauer cost") replaced by the book's statement of IAM's Law
  (Part 1, ch:virial_law); the premise "cosmic horizon thermodynamically inadequate" labelled not derived.
- Euclid: only as in sec:lt_euclid (template-equivalent µ0 ≈ −0.07, 1.8σ at σ(µ0) = 0.04; DR1 mid-2027). "3.4σ", "5.4σ with DESI Y5" not carried.

## New findings for the author (proposed errata rows; not written into PAPER_ERRATA.md, which I do not own)
| Paper | Where | Printed | Correct | Evidence |
|---|---|---|---|---|
| Missing_Satellites | §4.2 Eq. 10 | t_dyn ≈ √(π/6) GM/σ³ (ρ ∝ σ⁶/G³M²) | with the paper's own ρ = 3M/4πr_v³, r_v = GM/2σ²: ρ = 6σ⁶/(πG³M²), t_dyn = (π/6) GM/σ³; t_dyn = 1/H0 gives M = (6/π)σ³/(GH0) = 10^8.63 M⊙ at 4 km/s (not √(6/π), 10^8.48, as in MISSING_SATELLITES_CHECK.md) | verify_cluster_mass_satellites.py G (sympy) |
| Missing_Satellites | §4.2 | Ω_m + Ω_Λ = 1.0002 | 1.0000 for Ω_m 0.3153, Ω_Λ 0.6847 | verify G |
| Missing_Satellites | refs | Read, Pontzen, Walker, Steger 2006, MNRAS 367, 387 | no such paper found; the printed DOI resolves to a different paper; dropped | CrossRef |
| Missing_Satellites | refs | Benson et al. 2002 "II", DOI …05387.x | paper II is …05388.x (MNRAS 333, 177) | CrossRef |
| Lensing_Dynamics | refs | Pizzuti et al. 2017, JCAP 04, 023 | JCAP 07 (2017) 023, doi 10.1088/1475-7516/2017/07/023 | CrossRef |
| Lensing_Dynamics, 3Way | §4.2 / Table 2 | CCCP 1.20 ± 0.12, WtG 1.31 ± 0.11; four binned "observed" ratios | Planck 2015 XXIV Table 2 values (above); binned values untraced | Planck 2015 XXIV |
| 3Way | §5 | flattening/turnover at z ≈ 0.5–1.0 | R × C_NT turns over at z = 1.40 | verify H |
| p5_05c_virial_decoherence.tex:120 (not mine) | — | t_dyn = 1/√(Gρ) = √(4π/3) GM/σ³ with σ² = GM/R | differs from Ch. satellites Eq. ms_tdyn2 by the virial definitions; owner to align | — |

Other notes for owners of shared files: appendices/app_E_formulas.tex entries 102–104 cite section titles that changed ("The three-way ratio in the
Level 1 form"; "Mechanism A: growth suppression from µ < 1"; "Mechanism B: the dispersal condition"); entry 104's σ_crit now has its own label
eq:ms_scrit. appendices/app_I_provenance.tex should add fig:sat_closure (figures/part2/fig_sat_closure.pdf, figscripts/fig_p2_satellites_closure.py).
Labels kept for external references: ch:lensdyn, sec:ld_hold, eq:ld_ratio, ch:threeway, eq:tw_R, ch:satellites, eq:ms_ps, eq:ms_mmin,
fig:sat_mechanisms, fig:sat_census, fig:lensdyn_forms, fig:lensdyn_test, fig:threeway_estimators, fig:threeway_slope.

## Withheld values (TW3), for the author's trace
| Bin | printed observed | printed deviation | recomputed (R×C_NT − obs)/σ |
|---|---|---|---|
| 0.1<z<0.2 | 1.28 ± 0.15 | +0.4σ | +0.44 |
| 0.2<z<0.3 | 1.22 ± 0.12 | +0.9σ | +0.86 |
| 0.3<z<0.5 | 1.25 ± 0.10 | +0.5σ | +0.47 |
| 0.5<z<0.8 | 1.18 ± 0.13 | +0.7σ | +0.68 |
Attributed jointly to Sereno 2017, Medezinski 2018, Herbonnet 2020, Grandis 2024; the printed Sereno and Grandis references did not match CrossRef records
(Sereno PSZ2LenS is MNRAS 472, 1946; no CrossRef hit for Grandis 2024 A&A 687 A178).

## Exclusions (complete list)
1. LD acknowledgements and self-citations; TW acknowledgements and self-citations (stand-alone rule).
2. TW §9.2 (AGN feedback; SMBH as local encoding surface; M–σ reference) — TW5.
3. TW Table 2 observed column and deviations — TW3, pending trace (values above).
4. TW Eq. 12 value M_SZ/M_hydro = 0.99 ± 0.04 — TW2 (not a test; value untraced).
5. MS σ³/σ²/σ⁴ family, M–σ and related comparisons in §4.3, §4.4, §5, §7 — M7 and author ruling 2026-10-03.
6. MS §4.4 "raw 10^6.4, ~100× offset" — M2.
7. MS "5.4σ with Euclid + DESI Y5" — unsourced (SP5).
8. MS acknowledgements, data availability, timestamped-prediction statements and figure footers — stand-alone rule; author ruling 2026-10-03.
9. Numbers not traced and therefore not printed: Planck SZ "S8 ≈ 0.78"; LSST "~200,000 clusters" and "µ_V ≈ 32 mag arcsec⁻²" (stated as order 10^5 / "very low surface brightness").

## Citations checked
25 new entries resolved by DOI against api.crossref.org (title, journal, volume, pages, year); Merloni2012 arXiv-only. Keys reused from iam.bib
(verified present): Planck2015SZ, Planck2018VI, Hoekstra2015, vonderLinden2014, Nelson2014, ShiKomatsu2014, PogosianSilvestri2016, Amendola2008,
Koyama2007, Fang2008, Bulbul2024, Neto2007, Ivezic2019, Jacobson1995, CaiKim2005, Landauer1961, PressSchechter1974, Klypin1999, Moore1999,
Pace2025LVDB, Kirby2013Segue2, Stolzner2025, Riess2022, Torrado2021, Zhao2009MGCAMB, Wang2023MGCAMB.

## Static checks (all three chapters)
Braces balanced; environments matched; every \ref/\eqref resolves against the current tree; no duplicate labels across the book; every \cite in
iam.bib or bib_clusters_satellites.bib; every figure exists; all floats [htbp]. Not compiled (no TeX in sandbox).

## Carriage tables (every equation, step, table, figure and quantitative claim)

### IAM_Lensing_Dynamics_Paper.pdf (25 Feb 2026) → part2/p2_17_lensing_dynamics.tex

| # | Paper location | Content | Book location | Verdict (source) | Label |
|---|---|---|---|---|---|
| LD1 | Abstract | dual-sector realisation; mu<1, Sigma=1 at linear order | p2_17_lensing_dynamics.tex:16 | CARRIED | summary |
| LD2 | Abstract | linearised Einstein equations unmodified, no anisotropic stress (Level 2 form) | p2_17_lensing_dynamics.tex:139 | CARRIED-CORRECTED (LD1: holds only in the Level 2 form, where the ratio is 1) | derived |
| LD3 | Abstract | weak lensing affected only through delta_m growth; Weyl relation preserved | p2_17_lensing_dynamics.tex:142 | CARRIED | derived |
| LD4 | Abstract | S8 implications; modest growth suppression; no extra parameters | p2_17_lensing_dynamics.tex:269 | CARRIED | calc |
| LD5 | S1 p1 | lensing masses exceed dynamical masses 10-30 % | p2_17_lensing_dynamics.tex:27 | CARRIED-CORRECTED (values traced: 5-45 % by sample, Table tab:ld_published) | observed |
| LD6 | S1 Eq.1 | M_true = M_X/(1-b) | p2_17_lensing_dynamics.tex:32 | CARRIED | observed |
| LD7 | S1 | b 0.1-0.4 (Nagai, Rasia, Biffi) | p2_17_lensing_dynamics.tex:34 | CARRIED | observed |
| LD8 | S1 | simulations b 0.1-0.15; Planck SZ needs b~0.4 | p2_17_lensing_dynamics.tex:37 | CARRIED-CORRECTED (Planck 2015 XXIV: 1-b = 0.58 +/- 0.04) | observed |
| LD9 | S1 | bias shows no clear dependence on cluster properties | p2_17_lensing_dynamics.tex:39 | CARRIED-CORRECTED (LoCuSS reanalysis finds a redshift dependence) | observed |
| LD10 | S1 | explanation may not be purely astrophysical | p2_17_lensing_dynamics.tex:41 | CARRIED | interp |
| LD11 | S1 | IAM gives discrepancy with correct magnitude and distinctive z-dependence | p2_17_lensing_dynamics.tex:43 | CARRIED-CORRECTED (magnitude: gravitational part 7-10 % at sample z) | interp |
| LD12 | S1 | 15 chains, Delta chi2 +0.54, sigma8 0.800 | p2_17_lensing_dynamics.tex:46 | CARRIED-CORRECTED (LD5: 18 chains; Level 2 value 0.7998) | fitted |
| LD13 | S2.1 Eq.2-3 | Poisson and lensing equations with mu, Sigma | p2_17_lensing_dynamics.tex:54 | CARRIED | derived (definitions) |
| LD14 | S2.2 Eq.4 | mu(a) = H^2/(H^2 + beta_m E), Sigma = 1 | p2_17_lensing_dynamics.tex:64 | CARRIED-CORRECTED (LG10/T10: beta_m E H0^2; beta_m = 0.15765 canon) | interp / prediction |
| LD15 | S2.2 | Sigma = 1 from null geodesics, no decoherence for light | p2_17_lensing_dynamics.tex:69 | CARRIED | interp |
| LD16 | S2.3 Eq.5 | M_dyn ~ mu M_true | p2_17_lensing_dynamics.tex:79 | CARRIED (derivation step added; Level 1 form) | derived |
| LD17 | S2.3 Eq.6 | M_lens ~ Sigma M_true | p2_17_lensing_dynamics.tex:84 | CARRIED | derived |
| LD18 | S2.3 Eq.7 | M_lens/M_dyn = Sigma/mu = 1/mu | p2_17_lensing_dynamics.tex:88 | CARRIED (Level 1 form) | derived prediction |
| LD19 | Table 1 | mu, ratio, excess at z 0..3 | p2_17_lensing_dynamics.tex:95 | CARRIED-CORRECTED (canon beta_m: 1.158/15.8 % at z=0, paper 1.157/15.7) | calc |
| LD20 | S2.4 Eq.8 | explicit form 1 + beta_m exp(1-1/a)/(Om a^-3 + OL) | p2_17_lensing_dynamics.tex:114 | CARRIED | derived |
| LD21 | new | slopes dR/dz -0.31, -0.18, -0.12, -0.04 | p2_17_lensing_dynamics.tex:116 | ADDED (verify C) | calc |
| LD22 | new | gravitational slip Phi/Psi=(2-mu)/mu in Level 1 form | p2_17_lensing_dynamics.tex:129 | ADDED (resolves LD1 contradiction) | derived |
| LD23 | S2.3/S7 (LD1) | form of the term: ratio needs mu in Poisson; Level 2 gives 1 | p2_17_lensing_dynamics.tex:136 | CARRIED-CORRECTED (LD1 hold) | openprob |
| LD24 | S3.1 | hydrostatic bias constant 1/(1-b) ~ 1.2 for b=0.17; distinguishing test | p2_17_lensing_dynamics.tex:149 | CARRIED | calc prediction |
| LD25 | S3.2 | MOND a0 = 1.2e-10; acceleration-dependent; distinguishing test | p2_17_lensing_dynamics.tex:156 | CARRIED | prediction |
| LD26 | S3.3 | f(R), nDGP mu>1; mu<1 models have Sigma!=1; IAM unique | p2_17_lensing_dynamics.tex:161 | CARRIED-CORRECTED (LD2: f(R) Sigma=1, 1<=mu<=4/3; sDGP mu<1, Sigma=1; not unique) | observed |
| LD27 | S3.3 | decoherence asymmetry timelike/null worldlines | p2_17_lensing_dynamics.tex:165 | CARRIED | interp |
| LD28 | Table 2 | signatures of competing models | p2_17_lensing_dynamics.tex:170 | CARRIED-CORRECTED (LD2; Level 2 row added) | observed / calc |
| LD29 | S4.1 | Planck SZ: S8~0.78 or b~0.4; IAM resolves naturally; residual b 0.1-0.15 | p2_17_lensing_dynamics.tex:183 | CARRIED-CORRECTED (LD3: reduced, not resolved; 1-b = 0.58 +/- 0.04 from source; S8~0.78 not traced, not printed) | observed calc openprob |
| LD30 | S4.2 | CCCP M_lens/M_X = 1.20 +/- 0.12 at z~0.3; IAM 1.085 | p2_17_lensing_dynamics.tex:201 | CARRIED-CORRECTED (LD4: traced to Planck 2015 XXIV Table 2: 1-b = 0.780 +/- 0.092 -> 1.28 +/- 0.15) | observed |
| LD31 | S4.3 | WtG 1.31 +/- 0.11 at z~0.25; IAM 1.095 | p2_17_lensing_dynamics.tex:200 | CARRIED-CORRECTED (LD4: 1-b = 0.688 +/- 0.072 -> 1.45 +/- 0.15) | observed |
| LD32 | new | LoCuSS beta_X, beta_P z-split; Herbonnet 2020 | p2_17_lensing_dynamics.tex:204 | ADDED (traced sources; reported with offsets) | observed calc |
| LD33 | S4.4 | discrepancy pattern; no uniform z-binned test yet | p2_17_lensing_dynamics.tex:219 | CARRIED-CORRECTED (magnitude statement corrected) | openprob |
| LD34 | S5.1 + Table 3 | Euclid ~1e5 clusters to z~2; Table 3 bins | p2_17_lensing_dynamics.tex:231 | CARRIED | calc prediction |
| LD35 | S5.1 | flat/acceleration-dependent challenges; declining = mu(z) measurement | p2_17_lensing_dynamics.tex:225 | CARRIED | prediction |
| LD36 | S5.2 | Rubin LSST ~200,000 clusters percent-level | p2_17_lensing_dynamics.tex:241 | CARRIED-CORRECTED (count stated as order 1e5; LSST science book replaced by Ivezic 2019) | prediction |
| LD37 | S5.3 | eROSITA ~100,000 clusters, cross-match 0<z<1.5 | p2_17_lensing_dynamics.tex:245 | CARRIED | prediction |
| LD38 | S5.4 | combined test design, 5 steps, BIC/AIC | p2_17_lensing_dynamics.tex:250 | CARRIED (+ 2.7 % per bin, verify H) | calc |
| LD39 | S6 | connection: H0 72.26/67.16, sigma8 0.811->0.800, fsigma8 | p2_17_lensing_dynamics.tex:266 | CARRIED-CORRECTED (Level 2 values; fsigma8 4.25/2.17/1.35/0.41 %; Level 1/2 condition added) | calc openprob |
| LD40 | S7.1 | systematics; ratio cancels; z-dependence discriminating | p2_17_lensing_dynamics.tex:276 | CARRIED | interp |
| LD41 | S7.2 | prior work Terukina, Wilcox, Pizzuti; mu(a) 1 -> 0.864 untested | p2_17_lensing_dynamics.tex:283 | CARRIED-CORRECTED (Pizzuti 2017 is JCAP 07 (2017) 023) | observed |
| LD42 | S7.3 | falsifiability: no z-dep, increasing, z=0 differs >3 sigma, follows Eq.8 | p2_17_lensing_dynamics.tex:287 | CARRIED-CORRECTED (Level 1 form; non-thermal part modelled) | prediction |
| LD43 | S7 para | connection to dual-sector framework (Sigma=1 from construction) | p2_17_lensing_dynamics.tex:296 | CARRIED | derived |
| LD44 | S8 | conclusion | p2_17_lensing_dynamics.tex:15 | CARRIED (as Summary) | summary |
| LD45 | Ack., refs | software acknowledgements; self-citations | — | EXCLUDED (stand-alone book rule) | - |

### 3Way_Mass_Discrepancy_in_Galaxy_Clusters.pdf (25 Feb 2026) → part2/p2_18_three_way_clusters.tex

| # | Paper location | Content | Book location | Verdict (source) | Label |
|---|---|---|---|---|---|
| TW1 | Abstract | merging systems; Level 2: lensing-gas offsets follow LCDM collisionless dynamics; differences via growth history | p2_18_three_way_clusters.tex:19 | CARRIED | summary |
| TW2 | S1 | three estimators; sector assignment; GR convergence | p2_18_three_way_clusters.tex:23 | CARRIED | interp |
| TW3 | S1 | hydrostatic bias 15-30 %, Nagai 2007, Nelson 2014 | p2_18_three_way_clusters.tex:37 | CARRIED-CORRECTED (range traced: 20-45 % Planck priors, 5 % LoCuSS) | observed |
| TW4 | S1 | gravitational part complementary; redshift slope smoking gun | p2_18_three_way_clusters.tex:41 | CARRIED | conjecture |
| TW5 | S2 Eq.1 | adot_matter = sqrt((rho_tot + beta_m E rho0)/3) | p2_18_three_way_clusters.tex:50 | CARRIED (identified with eq:l2_Hm) | derived (units) |
| TW6 | S2 Eq.2 | E(a) = exp(1 - 1/a) | p2_18_three_way_clusters.tex:55 | CARRIED | - |
| TW7 | S2 Eq.3 | beta_m = Omega_m/2 = 0.1575 | p2_18_three_way_clusters.tex:60 | CARRIED-CORRECTED (canon 0.15765) | prediction |
| TW8 | S2 | Sigma = 1 null geodesics | p2_18_three_way_clusters.tex:62 | CARRIED | interp |
| TW9 | S2 Eq.4 | mu(a); verified against 15 chains; mu0 = 0.8639, 13.6 % | p2_18_three_way_clusters.tex:68 | CARRIED-CORRECTED (TW6: 18 chains fix beta_m, test not verify; 0.8638, 13.62 % coupling not growth) | interp calc |
| TW10 | S3.1 Eq.5 | M_hydro = mu M_true | p2_18_three_way_clusters.tex:81 | CARRIED (derivation step added; Level 1 form, TW1) | derived |
| TW11 | S3.1 Eq.6 | M_SZ ~ mu M_true | p2_18_three_way_clusters.tex:87 | CARRIED | derived |
| TW12 | S3.1 Eq.7 | M_lens = M_true | p2_18_three_way_clusters.tex:92 | CARRIED | derived |
| TW13 | S3.2 Eq.8 | R = 1/mu | p2_18_three_way_clusters.tex:98 | CARRIED | derived |
| TW14 | S3.2 Eq.9 | M_SZ/M_hydro ~ 1 | p2_18_three_way_clusters.tex:103 | CARRIED-CORRECTED (TW2: holds by calibration) | calc |
| TW15 | S3.2 Eq.10 | ordering M_lens > M_SZ ~ M_hydro; pattern distinguishes | p2_18_three_way_clusters.tex:107 | CARRIED-CORRECTED (TW2: only the slope distinguishes) | calc |
| TW16 | S4 | R 1.158 -> 1.002; eROSITA z 0.2-0.4 R 1.07-1.11 | p2_18_three_way_clusters.tex:113 | CARRIED | calc |
| TW17 | Table 1 | z, a, mu, R | p2_18_three_way_clusters.tex:120 | CARRIED-CORRECTED (canon beta_m: mu 0.8638 at z=0) | calc |
| TW18 | S5 | dR/dz = -0.18 at z=0.3; NT +0.02..+0.04 | p2_18_three_way_clusters.tex:140 | CARRIED | calc observed |
| TW19 | S5 Eq.11 | b_hydro = b_NT + b_IAM | p2_18_three_way_clusters.tex:152 | CARRIED-CORRECTED (exact product form; additive to first order, cross term 0.014) | derived |
| TW20 | S5 | flattening/turnover at z~0.5-1.0 | p2_18_three_way_clusters.tex:160 | CARRIED-CORRECTED (turnover of R x C_NT at z = 1.40) | calc |
| TW21 | S5 | ~180 clusters, sigma(dR/dz) 0.03, ~6 sigma | p2_18_three_way_clusters.tex:180 | CARRIED-CORRECTED (TW4: not derived; arithmetic shown; no significance claimed) | openprob |
| TW22 | S6.1 | C_NT = 1 + 0.20(1+z)^0.2 (Nelson 2014) | p2_18_three_way_clusters.tex:157 | CARRIED-CORRECTED (form assumed, not fitted to simulations) | openprob |
| TW23 | Table 2 | IAM only / IAM+NT columns | p2_18_three_way_clusters.tex:166 | CARRIED | calc |
| TW24 | Table 2 | observed ratios 1.28/1.22/1.25/1.18 and deviations +0.4/+0.9/+0.5/+0.7 sigma | — | EXCLUDED pending trace (TW3: per-bin sources not given; recomputed deviations +0.44/+0.86/+0.47/+0.68 in verify D) | - |
| TW25 | S6.1 | consistent at < 1 sigma in all four bins | p2_18_three_way_clusters.tex:187 | CARRIED-CORRECTED (TW3; replaced by traced comparison) | calc observed |
| TW26 | S6.2 Eq.12 | M_SZ/M_hydro = 0.99 +/- 0.04 (Bulbul 2024); first test passes | p2_18_three_way_clusters.tex:197 | CARRIED-CORRECTED (TW2: not a test; value not traced, not printed) | calibrated |
| TW27 | Eq.13-15 | three conditions | p2_18_three_way_clusters.tex:203 | CARRIED-CORRECTED (TW2, TW4) | - |
| TW28 | S7 | systematics: HSE 5-10 %, /m/<0.02, projection, selection; sign of slope | p2_18_three_way_clusters.tex:208 | CARRIED (selection item: trend possible, LoCuSS) | interp |
| TW29 | S8 | eRASS:4 + Euclid ~5000 clusters, 22 sigma | p2_18_three_way_clusters.tex:222 | CARRIED-CORRECTED (TW4: 22 sigma not reproduced; Euclid only as sec:lt_euclid) | prediction |
| TW30 | S8 | convergent evidence from three probes | p2_18_three_way_clusters.tex:226 | CARRIED | prediction |
| TW31 | S9.1 | connection; SZ unity already confirmed | p2_18_three_way_clusters.tex:229 | CARRIED-CORRECTED (TW2; Level 1/2 condition) | openprob |
| TW32 | S9.2 | AGN feedback tuning; SMBH as local encoding surface; M-sigma paper | — | EXCLUDED (TW5: speculation; M-sigma abandoned; cut) | - |
| TW33 | S10 | conclusion items 1-5; all tests need no new observations | p2_18_three_way_clusters.tex:235 | CARRIED-CORRECTED (items 1-3, 5 per TW2-TW4; "no new observations" withdrawn: the slope test needs a uniform binned sample) | prediction |
| TW34 | Ack., refs | acknowledgements; self-citations | — | EXCLUDED (stand-alone book rule) | - |

### Missing_Satellites.pdf (March 2026) → part2/p2_19_missing_satellites.tex

| # | Paper location | Content | Book location | Verdict (source) | Label |
|---|---|---|---|---|---|
| MS1 | Abstract | factor ~8: ~500 subhalos vs ~60 satellites | p2_19_missing_satellites.tex:17 | CARRIED | observed |
| MS2 | Abstract | two mechanisms from one condition, no free parameters | p2_19_missing_satellites.tex:18 | CARRIED | conjecture |
| MS3 | Abstract | Mechanism A: mu0 = 0.864, 13.6 %; 17 chains; Delta chi2 +0.54 | p2_19_missing_satellites.tex:19 | CARRIED-CORRECTED (M4/M6: 18 chains; growth -0.78 % stated) | calc |
| MS4 | Abstract | Mechanism B: sigma_crit ~ 4 km/s; M_min ~ sigma^3 | p2_19_missing_satellites.tex:22 | CARRIED-CORRECTED (M8: rejected by census) | conjecture observed |
| MS5 | Abstract | sigma^3 distinct from sigma^2 and sigma^4 relations | — | EXCLUDED (M7; author ruling 2026-10-03) | - |
| MS6 | Abstract | ~100x normalisation offset | — | EXCLUDED (M2: no offset in Eq. 11; the open problem carried as the prefactor) | - |
| MS7 | Abstract | Euclid DR1 Oct 2026, sigma(mu0) 0.04, 3.4 sigma | p2_19_missing_satellites.tex:245 | CARRIED-CORRECTED (M4/M6: DR1 mid-2027; sensitivity only as sec:lt_euclid) | prediction |
| MS8 | Abstract | Rubin census below 4 km/s | p2_19_missing_satellites.tex:252 | CARRIED | prediction |
| MS9 | S1 | Klypin, Moore; 500 subhalos vc>10; ~60 satellites; Koposov, Tollerud | p2_19_missing_satellites.tex:27 | CARRIED (+ LVDB 68) | observed |
| MS10 | S1 | reionisation, stripping, feedback; act on baryons | p2_19_missing_satellites.tex:32 | CARRIED-CORRECTED (Read et al. 2006 reference unmatched by CrossRef, dropped; Benson 2002 is paper II) | observed |
| MS11 | S1 | IAM origin: same process as dark energy; acts on DM | p2_19_missing_satellites.tex:36 | CARRIED | conjecture |
| MS12 | S1 | roadmap | p2_19_missing_satellites.tex:38 | CARRIED | - |
| MS13 | S2 Eq.1 | 2K + V = 0, K = /V//2 | p2_19_missing_satellites.tex:47 | CARRIED | derived |
| MS14 | S2 | potential half to curvature, kinetic half Landauer cost | p2_19_missing_satellites.tex:51 | CARRIED-CORRECTED (check file: book statement of IAM's Law replaces the two-channel wording) | prediction |
| MS15 | S2 | closure condition: both halves close simultaneously | p2_19_missing_satellites.tex:60 | CARRIED | conjecture |
| MS16 | S2 Eq.2 | beta_m = Omega_m/2 = 0.1575; posterior 0.1583 +/- 0.0033 at 0.2 sigma | p2_19_missing_satellites.tex:55 | CARRIED-CORRECTED (M6: fixed in chains; posterior is Omega_m/2 on Omega_m) | prediction calc |
| MS17 | S3.1 Eq.3 | E(a); E->0, dE/da>0, ledger accumulates | p2_19_missing_satellites.tex:70 | CARRIED | interp |
| MS18 | S3.1 | photons dtau=0 | p2_19_missing_satellites.tex:74 | CARRIED | interp |
| MS19 | S3.1 Eq.4 | mu(a) | p2_19_missing_satellites.tex:78 | CARRIED-CORRECTED (LG10: H0^2) | interp |
| MS20 | S3.1 Eq.5 | mu(0) = 1/(1+beta_m) = 0.864; mu0 = -0.136; LCDM at z>~2 | p2_19_missing_satellites.tex:83 | CARRIED | calc |
| MS21 | S3.1 Eq.6 | growth equation with mu | p2_19_missing_satellites.tex:90 | CARRIED-CORRECTED (Omega_m(a) H^2 written out; form (i)) | derived |
| MS22 | S3.2 Eq.7 | Delta D/D = -0.074 | p2_19_missing_satellites.tex:99 | CARRIED-CORRECTED (M5: -0.78 % form i; -0.67 % ii; -1.87 % iii) | calc |
| MS23 | S3.2 | Press-Schechter exp(-dc^2/2 sigma^2); 1e7-1e9 sensitive | p2_19_missing_satellites.tex:109 | CARRIED-CORRECTED (derived Delta ln n = (nu^2-1) eps; nu 0.24-0.35 computed; change < 1 %) | derived calc |
| MS24 | S3.3 | 17 chains converged R-1<0.01, no discarded runs | p2_19_missing_satellites.tex:123 | CARRIED-CORRECTED (M4: 18 chains, all <= 0.010 today) | measured |
| MS25 | Table 1 | beta_m, sigma8, H0 photon/matter, Delta chi2 | p2_19_missing_satellites.tex:128 | CARRIED-CORRECTED (M6; sigma8 comparison: Stolzner 2025 joint, 0.1 sigma) | fitted calc |
| MS26 | S3.3 | sigma8 suppression = reduced small-scale power at satellite scale | p2_19_missing_satellites.tex:139 | CARRIED | interp |
| MS27 | S4.1 | two surfaces; cosmic horizon cold T_H 2.66e-30 K; inadequate on collapse times | p2_19_missing_satellites.tex:148 | CARRIED-CORRECTED (T_H = 2.65e-30 K; premise not derived, check file) | calc conjecture openprob |
| MS28 | S4.1 Eq.8 | S_info >= A/4 l_P^2 (BH forms) | p2_19_missing_satellites.tex:158 | CARRIED (one nat per 4 l_P^2) | conjecture |
| MS29 | S4.1 | halo disperses where condition fails | p2_19_missing_satellites.tex:160 | CARRIED | conjecture |
| MS30 | S4.2 Eq.9 | t_dyn = sqrt(pi/(6 G rho)) | p2_19_missing_satellites.tex:166 | CARRIED | definition |
| MS31 | S4.2 | rho ~ sigma^6/(G^3 M^2) | p2_19_missing_satellites.tex:171 | CARRIED-CORRECTED (full prefactor 6/pi) | derived |
| MS32 | S4.2 Eq.10 | t_dyn ~ sqrt(pi/6) GM/sigma^3 | p2_19_missing_satellites.tex:175 | CARRIED-CORRECTED (NEW: (pi/6) GM/sigma^3 with the stated rho; proposed errata row) | derived |
| MS33 | S4.2 | t_dyn = 1/H; H(z=0)=H0 sqrt(Om+OL), 1.0002 | p2_19_missing_satellites.tex:181 | CARRIED-CORRECTED (Om+OL = 1.0000; M = (6/pi) sigma^3/(G H0) = 10^8.63 at 4 km/s) | derived calc |
| MS34 | S4.2 Eq.11 | M_min = 4 Om sigma^3/(G H0); Om 0.3153, H0 67.36 | p2_19_missing_satellites.tex:185 | CARRIED (10^8.44 at 4 km/s; prefactor not derived) | calc openprob |
| MS35 | S4.3 Eq.12 | sigma_crit; M_min = 10^8.4 from Kim & Peter 2021 | p2_19_missing_satellites.tex:194 | CARRIED-CORRECTED (V11: source replaced by Nadler et al. 2020, M_peak < 3.2e8 at 95 %; sigma_crit 3.4-4.2) | observed calc |
| MS36 | S4.3 | halos below disperse, not missing | p2_19_missing_satellites.tex:200 | CARRIED | conjecture |
| MS37 | S4.3 | sigma^3 testable; distinct from sigma^2 and sigma^4; same identity | p2_19_missing_satellites.tex:202 | CARRIED-CORRECTED (M7/author ruling: family not carried) | prediction |
| MS38 | S4.4 | 100x offset; same as 130x elsewhere; one open problem | p2_19_missing_satellites.tex:205 | CARRIED-CORRECTED (M2: no offset in Eq.; local-to-global criterion kept as open problem; comparison with another prediction not carried, author ruling) | calc openprob |
| MS39 | Fig.1 | (a) mu(z); (b) M_min(sigma) with floor | p2_19_missing_satellites.tex:214 | CARRIED-CORRECTED (redrawn: fig_sat_closure(a) + fig_sat_mechanisms(b); Euclid band and timestamp dropped) | calc |
| MS40 | S5 | unified picture: A continuous, B threshold | p2_19_missing_satellites.tex:217 | CARRIED | conjecture |
| MS41 | Fig.2 | four halo regimes | p2_19_missing_satellites.tex:227 | CARRIED-CORRECTED (redrawn as fig_sat_closure(b) and Table tab:ms_regimes; census overlay) | conjecture |
| MS42 | S5 | differs from baryonic explanations; not exclusive; only proposals from same framework as Lambda and M-sigma | p2_19_missing_satellites.tex:236 | CARRIED-CORRECTED (M7: uniqueness and M-sigma not carried) | interp |
| MS43 | S6 | Mechanism A: Euclid 3.4 sigma; DESI Y5 5.4 sigma; mu0>0.90 at 2 sigma tension | p2_19_missing_satellites.tex:248 | CARRIED-CORRECTED (Euclid as sec:lt_euclid; 5.4 sigma unsourced (SP5), not carried) | prediction |
| MS44 | S6 | Mechanism B: Rubin census to mu_V ~ 32; criterion; sigma^3 for dSphs | p2_19_missing_satellites.tex:253 | CARRIED (32 mag/arcsec^2 not traced to Ivezic 2019; not printed) | prediction |
| MS45 | S6 | cross-check; beta_m confirmed 0.2 sigma | p2_19_missing_satellites.tex:259 | CARRIED-CORRECTED (M6) | prediction |
| MS46 | new (M8, check) | census: 25 of 54 below 4 km/s; tides | p2_19_missing_satellites.tex:263 | ADDED (check file) | observed openprob |
| MS47 | S7 | discussion: thermodynamic answer | p2_19_missing_satellites.tex:292 | CARRIED | interp conjecture |
| MS48 | S7 | open problem: local-to-global criterion; calculable | p2_19_missing_satellites.tex:303 | CARRIED | conjecture openprob |
| MS49 | S7 | sigma^3/sigma^2/sigma^4 coherent family | — | EXCLUDED (M7; author ruling 2026-10-03) | - |
| MS50 | Ack., Data avail. | timestamped predictions; repository archive | — | EXCLUDED (stand-alone book rule; author ruling 2026-10-03) | - |
