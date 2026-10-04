# MANIFEST: Late-time growth (Level 1, MGCAMB) and the Boltzmann-code mechanism (Level 2, CAMB): line-for-line carriage, 2026-10-03

Clone: HEAD e37aabb (later than 12d8fb1 and 41f7646). Nothing was pushed or committed.

## Files delivered (paths relative to the repository root)
- `docs/book/part2/p2_07_late_time_growth.tex`: Chapter `ch:latetime`, 424 lines, about 3,840 words of prose (before: 211 lines, about 2,110).
- `docs/book/part2/p2_06_dual_sector_perturbation.tex`: Chapter `ch:level2`, 527 lines, about 4,130 words (before: 193 lines, about 1,280).
- `docs/book/figscripts/fig_p2_latetime_level2.py` (new, on `_bookstyle.py` / `_chains.py`). It draws fig_mu_profile, fig_posterior_comparison, fig_fsigma8,
  fig_deltachi2_final and fig_mu0_posterior_final (these five had no generating script before) and the new fig_l2_triangle, fig_l2_sigma8,
  fig_l2_h0split and fig_l2_background. fig_sector_rates and fig_param_shifts come from the existing `fig_p2_level2.py` (re-run, unchanged).
- `docs/book/figures/part2/*.pdf/.png` for the 11 figures above.
- `docs/verification/scripts/verify_late_time_level2.py` + `_output.txt`: sympy algebra, every number recomputed from the chains (30 % burn-in,
  weighted) and from `chains/data/growth_on|off.json`. Result: 31 of 31 checks pass.
- `docs/book/bib_latetime_level2.bib`: 2 new CrossRef-verified entries (Beutler2011BAO, Ross2015MGS), and DOI fields to merge into four
  iam.bib entries that have none (Riess2022, DESY3, HSCY3, Heymans2021).
- **main.tex**: no change. The order stays `\input{part2/p2_07_late_time_growth}` then `\input{part2/p2_06_dual_sector_perturbation}`.
- **Bibliography**: add `bib_latetime_level2` to the `\bibliography{...}` list, or merge the two entries into iam.bib.

## Reading ledger
| source | lines | read | note |
|---|---|---|---|
| docs/papers/Late_Time_Growth_Suppression...pdf (pypdfium2, `=== PAGE` markers) | 567 (ledger 567) | 1-567 in 50-line chunks, none truncated | 4,564 words |
| docs/papers/Dual_Sector_Perturbation_Cosmology_CAMB.pdf | 595 (ledger 595) | 1-595 in 50-line chunks, none truncated | 4,420 words |
| coverage/wave2/03_Dual_Sector_Perturbation_CAMB_Level2.tex (author's LaTeX) | 904 | 1-904 in 50-line chunks | ERRATUM/FLAG comments applied |
| docs/papers/latex/iam_mu_sigma_paper/iam_mu_sigma_paper.tex | 591 | equation and figure environments extracted by script | equations match the PDF (Eq. 1 without H0^2 = LG10) |
| PAPER_ERRATA.md rows LG1-LG12, P1-P16; LATE_TIME_GROWTH_CHECK.md (52 lines); DUAL_SECTOR_PERTURBATION_CHECK.md (63 lines) | -- | in full | all applied |
| previous p2_07 (211 lines) and p2_06 (193 lines) | -- | in full | every label kept |
| Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv, CHAIN_PAIRS_FINAL.csv; Cosmological_Physics/camb_validation/likelihood_rsd.py, getdist_scripts/rsd_apples_to_apples.py, prepare_level2b.sh, equations_iam_level2.f90 (IAM lines and dtauda); chain input.yaml headers and priors; verify_euclid_template.py + output | -- | in full (f90: the IAM blocks and dtauda only) | |

## Late-Time Growth Suppression (Level 1): every item
| paper location | content | book location | verdict | status label |
|---|---|---|---|---|
| p1 abstract | model: mu<1, Sigma=1, E(a), mu(0)=0.865, 13.5 % | p2_07_late_time_growth.tex:14 | CARRIED-CORRECTED (check #8: 0.864, 13.6 %) | \calc |
| p1 abstract | twelve MCMC, Planck TT TE EE+lensing, RSD, BAO, Pantheon+ | p2_07_late_time_growth.tex:16 | CARRIED | -- |
| p1 abstract | Planck+RSD Delta chi2 = +1.34 | p2_07_late_time_growth.tex:21 | CARRIED-CORRECTED (LG2: +0.56) | \measured |
| p1 abstract | free mu0 = +0.024 +/- 0.123, 1.3 sigma | p2_07_late_time_growth.tex:22 | CARRIED-CORRECTED (LG2, check #9: median/quantiles) | \measured |
| p1 abstract | sigma8 0.813 -> 0.800 | p2_07_late_time_growth.tex:24 | CARRIED | \measured |
| p1 abstract | BAO and SN unaffected | p2_07_late_time_growth.tex:24 | CARRIED-CORRECTED (LG9 SN wording) | -- |
| p1 abstract | Euclid and DESI forecasts | p2_07_late_time_growth.tex:25 | CARRIED-CORRECTED (Euclid rule: sec:lt_euclid) | -- |
| p1 abstract | data public | p2_07_late_time_growth.tex:26 | CARRIED | -- |
| §1 p1-2 | LCDM fits; S8 definition; 2-3 sigma lensing deficit | p2_07_late_time_growth.tex:30 | CARRIED + KiDS-Legacy added | \observed |
| §1 p2 | mu-Sigma framework, DES/KiDS/Euclid | p2_07_late_time_growth.tex:38 | CARRIED | -- |
| §1 Eq.1 | mu = H2/(H2+beta E) | p2_07_late_time_growth.tex:47 | CARRIED-CORRECTED (LG10: H0^2) | \derived (via ch:theory) |
| §1 Eq.2 | E(a)=exp(1-1/a) | p2_07_late_time_growth.tex:51 | CARRIED | \derived |
| §1 Eq.3 | mu(0)=1/(1+beta)=0.865, 13.5 % | p2_07_late_time_growth.tex:56 | CARRIED-CORRECTED (0.864, 13.6 %) | \calc |
| §1 p2 | E(a) from decoherence rate + horizon thermo; 1/a^2 surface density; ln E = 1-1/a | p2_07_late_time_growth.tex:68 | CARRIED (step written out) | \derived |
| §1 p2 | beta=Omega_m/2 from virial; rational form from Cai-Kim first law | p2_07_late_time_growth.tex:72 | CARRIED | \derived / \measured |
| §1 p2 | minimal single-parameter realisation | p2_07_late_time_growth.tex:74 | CARRIED | \interp |
| §1 p2 | MGCAMB / CAMB | p2_07_late_time_growth.tex:77 | CARRIED | -- |
| §1 p2 | mu<1 less studied; f(R), DGP, scalar-tensor predict mu>1 | p2_07_late_time_growth.tex:80 | CARRIED-CORRECTED (LG12: SA-DGP) | \observed |
| §1 p2 | organisation | p2_07_late_time_growth.tex:87 | CARRIED (book wording) | -- |
| Fig.1 | mu(z) exact vs MGCAMB; E(a) | p2_07_late_time_growth.tex:91 | CARRIED, redrawn (fig_p2_latetime_level2.py) | \calc |
| §2.1 Eq.4-5 | modified Poisson and lensing equations | p2_07_late_time_growth.tex:100 | CARRIED | -- |
| §2.1 | Sigma=1; lensing <0.5 %; ISW within cosmic variance | p2_07_late_time_growth.tex:108 | CARRIED (Limber 0.05-0.3 % per check) | \calc |
| §2.2 Eq.6 | mu = 1 + mu0 Omega_DE(a) | p2_07_late_time_growth.tex:117 | CARRIED-CORRECTED (LG1: /Omega_L) | \derived |
| §2.2 | mu0 = -0.135, Sigma0=0; 1-2.5 % deviation | p2_07_late_time_growth.tex:122 | CARRIED (+ 2.8 % max, table added) | \calc |
| §2.2 | notation convention | p2_07_late_time_growth.tex:143 | CARRIED | -- |
| §2.2 | MGCAMB flags | p2_07_late_time_growth.tex:146 | CARRIED | -- |
| §3.1 | Planck 2018 likelihoods | p2_07_late_time_growth.tex:151 | CARRIED-CORRECTED (check #12: variants per combination) | -- |
| §3.1 | RSD BOSS DR12 + eBOSS DR16 | p2_07_late_time_growth.tex:157 | CARRIED-CORRECTED (LG11) | -- |
| §3.1 | BAO DR12, eBOSS DR16 LRG/QSO/Ly-alpha; sector assignment | p2_07_late_time_growth.tex:162 | CARRIED-CORRECTED (NEW: chains use 6dF 2011 + DR7 MGS + DR12 BAO) | -- |
| §3.1 | Pantheon+ photon-sector | p2_07_late_time_growth.tex:169 | CARRIED-CORRECTED (LG9) | -- |
| §3.2 | three configurations; prior [-0.5,0.2] | p2_07_late_time_growth.tex:177 | CARRIED (+ reference/proposal) | -- |
| §3.2 p5 | four combinations; sampled params; Cobaya R-1<0.01; 20,000 samples; same infrastructure | p2_07_late_time_growth.tex:181 | CARRIED-CORRECTED (NEW: rows per chain Table tab:lt_chains; '20,000 accepted after burn-in' not met by all) | \measured |
| §4.1 + Table 1 | Planck only; Delta chi2 +1.43; p=0.23; free mu0 0.006+/-0.156, 0.9 sigma | p2_07_late_time_growth.tex:216 | CARRIED-CORRECTED (LG2 final chains, LG3 likelihood ratio, check #9) | \measured |
| §4.2 + Table 2 | Planck+RSD; +1.34; sigma8 0.8001 vs 0.8131; free +0.024 +/-0.123 1.3 sigma | p2_07_late_time_growth.tex:243 | CARRIED-CORRECTED (LG2, check #9) | \measured |
| Fig.2 | Planck+RSD posteriors | p2_07_late_time_growth.tex:259 | CARRIED, redrawn from final chains | \measured |
| Fig.3 | f sigma8(z) vs SDSS BOSS/eBOSS | p2_07_late_time_growth.tex:263 | CARRIED-CORRECTED (LG7: 6dFGS, MGS named), redrawn | \calc / \observed |
| §4.3 + Table 3 | Planck+BAO +2.32; free +0.002 +/- 0.158 | p2_07_late_time_growth.tex:277 | CARRIED-CORRECTED (LG2: +1.73) | \measured |
| §4.4 + Table 4 | Planck+Pantheon+ +1.58; free -0.005 +/- 0.162 | p2_07_late_time_growth.tex:199 | CARRIED (Delta chi2 unchanged; params final) | \measured |
| Fig.4 | Delta chi2 bars with 3.84 threshold | p2_07_late_time_growth.tex:308 | CARRIED-CORRECTED (LG3: threshold replaced by likelihood ratio), redrawn | \measured |
| §4.5 + Table 5 | summary Delta chi2, sigma8 | p2_07_late_time_growth.tex:299 | CARRIED-CORRECTED (LG2) | \measured |
| Fig.5 | mu0 posteriors as Gaussians | p2_07_late_time_growth.tex:330 | CARRIED-CORRECTED (LG8: actual posteriors), redrawn | \measured |
| free-mu0 Delta chi2 -1.90/-3.99/+0.19/-0.60 | free-chain chi2_min differences | p2_07_late_time_growth.tex:205 | EXCLUDED as test (check #2: depends on sampling depth); reason stated in text | -- |
| §5.1 | zero-parameter prediction survives; <1 sigma; not guaranteed a priori | p2_07_late_time_growth.tex:337 | CARRIED-CORRECTED (LG3/check #13: likelihood ratio) | \interp |
| §5.1 | stable; RSD does not increase Delta chi2 | p2_07_late_time_growth.tex:343 | CARRIED | -- |
| §5.2 | sigma8 shift 1.6 %; zero-parameter; direction not input | p2_07_late_time_growth.tex:346 | CARRIED | \prediction |
| §5.2 | KiDS-1000 0.759, DES Y3 0.776; joint analysis needed | p2_07_late_time_growth.tex:352 | CARRIED (+ Heymans 0.766, KiDS-Legacy) | \observed / \openprob |
| §5.3 | mu<1 Sigma=1 less studied; f(R), DGP, Horndeski | p2_07_late_time_growth.tex:358 | CARRIED-CORRECTED (LG12) | -- |
| §5.3 | DES Y3 -0.4+/-0.4; Andrade 0.02+/-0.19; DESI 0.11 | p2_07_late_time_growth.tex:364 | CARRIED-CORRECTED (LG4, LG5) | \observed |
| §5.4 + Table 6 | Fisher sigma(mu0): Planck+RSD 0.12/1.1; Euclid pess 0.06/2.2, opt 0.04/3.4; DESI Y5 0.10/1.4; Euclid+DESI 0.025/5.4 | p2_07_late_time_growth.tex:369 | CARRIED-CORRECTED: Planck+RSD row (0.125, 1.1); Euclid rows replaced by sec:lt_euclid (author Euclid rule; verify_euclid_template.py); DESI row recomputed for IAM mu(z): 0.75 sigma | \calc / \observed / \prediction |
| §5.4 p10 | Euclid alone ~3 sigma; combined 0.025 | p2_07_late_time_growth.tex:373 | EXCLUDED (author rule: Euclid only as sec:lt_euclid); replaced | -- |
| §5.5 items 1-5 | limitations | p2_07_late_time_growth.tex:388 | CARRIED (+ item 6 prior edge) | -- |
| §6 items 1-5 | conclusions | p2_07_late_time_growth.tex:405 | CARRIED-CORRECTED (LG2, Euclid rule) | -- |
| §6 p11 | less explored region; testable; derivation elsewhere | p2_07_late_time_growth.tex:418 | CARRIED | -- |
| Acknowledgments | thanks to code developers | -- | EXCLUDED (stand-alone book rule: no author voice); codes cited in text | -- |
| Data availability | repo URL | p2_07_late_time_growth.tex:422 | CARRIED (repo paths) | -- |
| References | Frusciante title; Andrade PRD; DES Y3 mu0 | -- | CORRECTED in bib use (LG5, LG6) -- iam.bib keys Andrade2024 (MNRAS), Frusciante2025 | -- |

## Dual-Sector Perturbation Cosmology (Level 2): every item
| paper location | content | book location | verdict | status label |
|---|---|---|---|---|
| abstract p1 | CAMB implementation; beta_m=0.15765; three changes; zero parameters | p2_06_dual_sector_perturbation.tex:12 | CARRIED | \derived / \measured |
| abstract p1 | three chains; CamSpec; Delta chi2 -0.01/+0.54; below 3.84 | p2_06_dual_sector_perturbation.tex:18 | CARRIED-CORRECTED (P5 NPIPE; P8 likelihood ratio) | \measured |
| abstract p1 | sigma8 0.8087 -> 0.7998 (1.1 %) | p2_06_dual_sector_perturbation.tex:21 | CARRIED | \measured |
| abstract p1 | H0 photon 67.16, matter 72.26 | p2_06_dual_sector_perturbation.tex:24 | CARRIED (+ +/-0.50) | \calc |
| abstract p1 | background runs H0 ~61.5 -> perturbation level | p2_06_dual_sector_perturbation.tex:26 | CARRIED-CORRECTED (NEW finding: coded term E/a^2, rate today 66.1, chi2 equal; P4 superseded in substance) | \measured |
| §1 | Hubble tension 67.4 vs 73.04, 5 sigma; TRGB, lensing | p2_06_dual_sector_perturbation.tex:35 | CARRIED-CORRECTED (4.9 sigma computed) + TRGB 2025 | \observed / \calc |
| §1 | S8: KiDS 0.759+/-0.021 (Heymans), DES 0.776, HSC 0.776+/-0.032 | p2_06_dual_sector_perturbation.tex:43 | CARRIED-CORRECTED (P14: Asgari 0.759 / Heymans 0.766; HSC 0.769 per Li 2023) | \observed |
| §1 | early photon vs late matter; question | p2_06_dual_sector_perturbation.tex:50 | CARRIED | -- |
| §1 | companion paper +1.34 to +2.32; MGCAMB | p2_06_dual_sector_perturbation.tex:53 | CARRIED-CORRECTED (LG2: +0.56 to +1.73) | -- |
| §1 | beyond parametric; adotoa_matter | p2_06_dual_sector_perturbation.tex:58 | CARRIED | -- |
| §1 | H split sqrt(1+beta_m) | p2_06_dual_sector_perturbation.tex:64 | CARRIED | \derived |
| §1 | organisation | p2_06_dual_sector_perturbation.tex:67 | CARRIED | -- |
| §2.1 Eq.1 | Friedmann | p2_06_dual_sector_perturbation.tex:78 | CARRIED | -- |
| §2.1 Eq.2 | H_m^2 = H^2 + beta E H0^2 | p2_06_dual_sector_perturbation.tex:82 | CARRIED | \derived |
| §2.1 Eq.3 | E(a) | p2_06_dual_sector_perturbation.tex:83 | CARRIED (given once, eq:lt_activation) | \derived |
| §2.1 Eq.4 | beta_m = 0.3153/2 | p2_06_dual_sector_perturbation.tex:86 | CARRIED | \derived / \measured |
| §2.1 | photons unmodified; complete specification | p2_06_dual_sector_perturbation.tex:91 | CARRIED | -- |
| §2.2 Eq.5-6 | mu = H2/(H2+beta E); Sigma=1 | p2_06_dual_sector_perturbation.tex:103 | CARRIED-CORRECTED (H0^2 as LG10; P2: closed form vs coded) + mapping step | \derived |
| §2.2 Eq.7 | mu(0)=0.864, 13.6 % | p2_06_dual_sector_perturbation.tex:109 | CARRIED | \calc |
| Table 1 | mu(a), H_m/H at z | p2_06_dual_sector_perturbation.tex:117 | CARRIED-CORRECTED (P3; also z=3,5 rows: E 0.0498, 0.0067) | \calc |
| Fig.1 | mu(z), E(a) | p2_06_dual_sector_perturbation.tex:112 | CARRIED (same figure as ch:latetime Fig.; given once) | \calc |
| §2.3 Mod.1-3 | code listing grho_0=3, velocity friction | p2_06_dual_sector_perturbation.tex:129 | CARRIED-CORRECTED (P1: source printed as is) | -- |
| §2.3 | all other routines unmodified | p2_06_dual_sector_perturbation.tex:174 | CARRIED | -- |
| §2.3 | MGCAMB approx 1-2.5 %; removes approximation | p2_06_dual_sector_perturbation.tex:196 | CARRIED-CORRECTED (P10 /Omega_L; P2) | \calc |
| check P2 | measured growth of coded mechanism | p2_06_dual_sector_perturbation.tex:187 | ADDED (correction P2/P12) | \measured |
| §2.4 | parameter count bullets; six sampled | p2_06_dual_sector_perturbation.tex:200 | CARRIED | -- |
| §3.1 | CamSpec (Efstathiou & Gratton); lowl; lensing | p2_06_dual_sector_perturbation.tex:211 | CARRIED-CORRECTED (P5) | -- |
| §3.1 | RSD seven points BOSS/eBOSS | p2_06_dual_sector_perturbation.tex:217 | CARRIED-CORRECTED (P6) + data table | \observed |
| §3.2 | Cobaya, R-1, 20,000 samples, LCDM switch off | p2_06_dual_sector_perturbation.tex:235 | CARRIED | -- |
| §3.3 Table 2 | chain configurations A, C, D | p2_06_dual_sector_perturbation.tex:243 | CARRIED (+ A_b, D_b, final R-1) | \measured |
| §3.3 | A and C identical except switch | p2_06_dual_sector_perturbation.tex:252 | CARRIED | -- |
| §4 table | parameter accounting | p2_06_dual_sector_perturbation.tex:255 | CARRIED (status column re-labelled per author rules) | -- |
| §4 | no tuning; public | p2_06_dual_sector_perturbation.tex:267 | CARRIED | -- |
| §5.1 Table 3 | Planck A vs C | p2_06_dual_sector_perturbation.tex:277 | CARRIED-CORRECTED (chain recomputation: S8 -0.78, omega_b -0.07, lnAs +0.09; Omega_m row added) | \measured |
| §5.1 | sigma8 -0.009; Hubble friction | p2_06_dual_sector_perturbation.tex:294 | CARRIED-CORRECTED (P1: metric source, not friction) | \measured |
| §5.1 | growth diagnostic dlnAs +0.10, dOm +0.06 | p2_06_dual_sector_perturbation.tex:299 | CARRIED-CORRECTED (+0.09, +0.05) | \measured |
| Fig.2 | triangle H0 sigma8 Omega_m | p2_06_dual_sector_perturbation.tex:303 | CARRIED, redrawn | \measured |
| Fig.3 | sigma8 1D with KiDS/DES bands | p2_06_dual_sector_perturbation.tex:308 | CARRIED, redrawn (approximate survey bands not drawn: no source) | \measured |
| Fig.4 | parameter shifts | p2_06_dual_sector_perturbation.tex:313 | CARRIED (fig_p2_level2.py) | \measured |
| §5.2 + Table 4 | apples-to-apples RSD +3.08, total +2.92, validated by L1 +1.34, below 3.84 | p2_06_dual_sector_perturbation.tex:325 | EXCLUDED: Table 4 numbers (P7, P12 confirmed: Run D fsigma8 from untouched velocities); text replaced by the correction | -- |
| §5.2 | z=0.85 point pulls 1.40 sigma | p2_06_dual_sector_perturbation.tex:332 | CARRIED-CORRECTED (1.39 sigma recomputed) | \calc |
| Fig.5 | chi2 summary incl. +2.92 | -- | EXCLUDED (P7, P8, P12); Planck values carried in tab:l2_planck and tab:l2_summary | -- |
| §5.2 p11 | Run D params within 0.06 sigma; sigma8 0.7995; dlnAs +0.08, dOm 0.00 | p2_06_dual_sector_perturbation.tex:321 | CARRIED | \measured |
| §5.3 | beta_m posterior 0.1583 +/- 0.0033, 0.2 sigma | p2_06_dual_sector_perturbation.tex:335 | CARRIED-CORRECTED (P9) | \measured |
| §6 Eq.8 | H_m/H_gamma ratio | p2_06_dual_sector_perturbation.tex:344 | CARRIED | \derived |
| §6 Eq.9 | sqrt(1.15765)=1.0759 | p2_06_dual_sector_perturbation.tex:348 | CARRIED | \derived |
| §6.1 steps 1-4, Eq.10 | H0 matter 72.26 +/- 0.50 | p2_06_dual_sector_perturbation.tex:362 | CARRIED | \calc |
| §6.1 Eq.11-12 | -0.37 sigma, -0.75 sigma | p2_06_dual_sector_perturbation.tex:367 | CARRIED | \calc |
| Table 5 | Hubble comparison | p2_06_dual_sector_perturbation.tex:374 | CARRIED | \calc |
| Fig.6 | H0 split | p2_06_dual_sector_perturbation.tex:381 | CARRIED, redrawn | \measured |
| §6.1 p11 | CMB measures photon, ladder matter | p2_06_dual_sector_perturbation.tex:386 | CARRIED | \interp |
| §6.2 + Table 6 | H by redshift | p2_06_dual_sector_perturbation.tex:398 | CARRIED-CORRECTED (P3; z=5 row 558.14, not 579.00) | \calc |
| §6.2 | 0.1 % at z=2, 0.02 % at z=3 | p2_06_dual_sector_perturbation.tex:393 | CARRIED | \calc |
| §7 Eq.13 | background Friedmann with beta E | p2_06_dual_sector_perturbation.tex:420 | CARRIED | -- |
| §7 | H0 ~61.5, 6 sigma, theta_s shift, strongly excluded by CMB | p2_06_dual_sector_perturbation.tex:433 | CARRIED-CORRECTED (P4 10.9 sigma; NEW: coded term E/a^2 (Eq. l2_bg_coded), chi2 10970.77 < 10972.07, rate today 66.12; 'excluded by CMB' not supported) | \measured / \calc |
| §7 p13 | perturbation-level interpretation | p2_06_dual_sector_perturbation.tex:445 | CARRIED | \interp |
| Fig.7 | background vs perturbation diagnostic | p2_06_dual_sector_perturbation.tex:449 | CARRIED-CORRECTED, redrawn | \measured |
| §8.1 Table 7 | summary | p2_06_dual_sector_perturbation.tex:459 | CARRIED-CORRECTED (+2.92 row excluded: P7/P12; background row corrected) | \measured |
| §8.2 items 1-4 | what it demonstrates | p2_06_dual_sector_perturbation.tex:472 | CARRIED-CORRECTED (P13 1.1 %; item 4 per new finding) | -- |
| §8.3 | does not solve Hubble tension; definitive test | p2_06_dual_sector_perturbation.tex:483 | CARRIED-CORRECTED ('definitive' -> 'can separate') | -- |
| §8.4 | relation to Level 1; sigma8 0.800-0.801 | p2_06_dual_sector_perturbation.tex:489 | CARRIED-CORRECTED (0.800-0.802; 1.6 vs 1.1 %) | \measured / \interp |
| §8.5 items 1-4 | falsifiability; Euclid 3.4 sigma; Sigma>1e-4 | p2_06_dual_sector_perturbation.tex:496 | CARRIED-CORRECTED (Euclid rule; P15) | \prediction / \openprob |
| §9 | conclusions | p2_06_dual_sector_perturbation.tex:509 | CARRIED-CORRECTED | -- |
| Acknowledgments | thanks | -- | EXCLUDED (stand-alone rule) | -- |
| Data availability | repo | p2_06_dual_sector_perturbation.tex:524 | CARRIED | -- |
| References | Wang 2023 MGCAMB v2; Frusciante; Efstathiou | -- | CORRECTED (P16 -> Wang2023MGCAMB JCAP 08 038; P11; P5 -> Rosenberg2022) | -- |

## Exclusions (for the author)
1. Level 2 Table 4 (Run D apples-to-apples: RSD +3.08, total +2.92, "validated by Level 1 +1.34"), Fig. 5 (chi2 summary with +2.92) and the
   Table 7 row "Delta chi2 (Planck + RSD) +2.92". Confirmed errata P7 and P12: Run D's fsigma8 came from CAMB velocities, which the change
   does not touch, so Run D did not test the IAM growth. The chapter says why, carries what Run D does show, and lists the rerun as an open problem.
2. Level 1 Table 6 Euclid rows (pessimistic 0.06 / 2.2 sigma, optimistic 0.04 / 3.4 sigma, Euclid + DESI 0.025 / 5.4 sigma) and the line "Euclid
   alone ... at the ~3 sigma level". Author rule: Euclid sensitivity only as in sec:lt_euclid / verify_euclid_template.py. The forecast script
   hand-enters those sigma values; they are not computed. Replaced by sec:lt_euclid. The DESI Y5 row is recomputed with the IAM mu(z) (0.75 sigma,
   at the script's extrapolated errors).
3. Level 1 free-mu0 chi2_min differences (-1.90, -3.99, +0.19, -0.60): check #2. The chapter states the reason.
4. The "95 % exclusion threshold 3.84" framing and "p = 0.23" (LG3, P8): replaced by the likelihood ratio.
5. Acknowledgments of both papers (stand-alone rule). The codes are cited where used.
6. Level 2 Fig. 3 "approximate sigma8 ranges from KiDS-1000 and DES Y3" shaded bands: no source given for those ranges. The figure is redrawn without them.

## NEW findings: not in PAPER_ERRATA, written correctly in the chapters, need errata rows and the author's decision
1. **Level 2b background chains coded a different term (affects P4 and about 12 other book locations).** In `dtauda`, `grhoa2 = 8 pi G rho a^4`
   (CAMB comment). `prepare_level2b.sh` adds `0.15765*E(a)*a*a*grho0`. The added H^2 is therefore beta_m E(a) H0^2 / a^2, not beta_m E(a) H0^2 (Eq. 13):
   equal at a = 1, four times larger at a = 0.5. Omega_Lambda was not reduced, so the expansion rate today is H0 sqrt(1+beta_m). Results:
   - lowest chi2 10970.77 (A_b) against 10972.07 (LCDM Run C). The CMB fits it as well, so the paper's "strongly excluded by the CMB power
     spectrum / shifts theta_s" is not supported.
   - sampled H0 = 61.45 +/- 0.42 is a parameter. The expansion rate today is 66.12 +/- 0.45 (2.3 sigma below Run C), and the matter fraction is
     0.378/1.15765 = 0.326.
   - So "10.9 sigma from Planck" (P4) compares a parameter, not the present expansion rate.

   The chapter states this and lists a background chain with Eq. 13 as written as open. Other files that still say "61.5, excluded" and need the
   lead's attention: p1_02_iams_law.tex:553, p2_02_virial.tex:282, p2_03_theory.tex:404/770/1039, p2_04_dualsector_chains.tex:90,
   p2_05_dual_sector_note.tex:54, p2_08_s8_trend.tex:47, p2_10_dual_sector_validation.tex:91, p5_05_gravdec.tex:40, p5_07_predictions.tex:257,
   p5_11_status_all.tex:44, app_G_predictions_register.tex:36/98/99 (COS-005, COS-240, COS-241), app_C3_derivations.tex:149.
2. **Level 1 "Planck + BAO" data.** The chains use `bao.sixdf_2011_bao`, `bao.sdss_dr7_mgs` and `bao.sdss_dr12_consensus_bao`, not the
   "eBOSS DR16 LRG, QSO and Lyman-alpha" the paper (§3.1) states. Corrected; cites Beutler2011BAO and Ross2015MGS.
3. **"Minimum of 20,000 accepted samples after burn-in" (Level 1 §3.2)** is not met by every chain. Rows kept after 30 % burn-in: Planck + BAO
   fixed 6,552; Planck LCDM 12,544; and so on. Table tab:lt_chains gives the true counts and final R-1. The claim is not printed.
4. **Level 2 Table 1 z = 3, 5 rows and Table 6 z = 5 row** are also shifted. P3 covers z <= 2 / z <= 3. Correct: E = 0.0498 and 0.0067 at
   z = 3 and 5; mu(2) = 0.9977, mu(3) = 0.9996; H(z = 5) = 558.14 (printed 579.00). The Table 6 values recomputed at full chain precision differ
   from the check file in the last digit: H_m(1) 121.52 (check 121.53), H(2) 204.05 (204.06), H(3) 307.36 / 307.42 (307.37 / 307.43).
5. **HSC Y3 S8** printed 0.776 +/- 0.032. Li et al. 2023 (cosmic shear) give 0.769 +0.031 -0.034, as in the previous chapter text. Kept.
6. **Level 2 shifts from the chains** (settled values): S8 -0.78 sigma (printed -0.73), omega_b -0.07 (-0.08), ln As +0.09 (+0.10), Omega_m +0.05 (+0.06).
7. **z = 0.85 pull**: 1.39 sigma (printed 1.40), against LCDM fsigma8 0.448 at the Run A parameters (switch off).
8. **Level 1 Table 1 to 4 parameter values** are all updated to the final chain files (LG2 covered Delta chi2 and free mu0; the parameters change
   with them, e.g. Planck LCDM H0 67.19 -> 67.14, sigma8 0.8139 -> 0.8143).

## Wave-2 FLAG comments
- Stand-alone wording "We / this work": rewritten in third person throughout.
- Rule "DR1" (RSD lines): the DR12/DR16 data names, with 6dFGS/MGS made explicit (P6). Euclid DR1 is not mentioned.
- Rule "potential" (parameter table): rewritten as "virial theorem for the 1/r interaction".
- Rule "3.4 sigma": replaced by sec:lt_euclid.

## Figure overlap check
The `_bookstyle.overlaps` check is 0 for 9 of the 11 figures. Two flags remain, and neither is visible on the rendered page: fig_posterior_comparison has two pairs of off-axis tick labels from neighbouring panels, outside the drawn ranges; fig_l2_triangle has the figure legend over the bounding box of the switched-off upper-right panels. Both were inspected on the rendered PNG.

## Static checks (no TeX in the sandbox)
Braces balance. begin/end are balanced. Every float is [htbp]. The Fortran listings are copied verbatim (dedented) from equations_iam_level2.f90 l. 55-56, 2245-2252 and 2324-2337. No label is duplicated anywhere in the book. Every \ref / \eqref in both files
resolves against the current tree. All 11 \includegraphics files exist. Every \cite key is in iam.bib or bib_latetime_level2.bib. Abbott2017Siren
was mis-cased in a first draft; that is fixed. All 24 labels that other files reference (ch:latetime, ch:level2, eq:lt_mu, eq:lt_mgcamb, eq:l2_Hm,
eq:l2_mu, tab:lt_*, fig:lt_*, fig:param_shifts, fig:sector_rates, sec:lt_euclid) are kept. sec:lt_euclid is now a proper \subsection label.

## Citations checked on CrossRef: 43
37 iam.bib DOIs were resolved (titles match; Abbott2017Siren included). 4 DOIs were found for entries without one (Riess2022, DESY3, HSCY3, Heymans2021). 2 entries are new (Beutler2011BAO, Ross2015MGS). The S8 / H0 values for Asgari 2021
(0.759), Heymans 2021 (0.766) and Riess 2022 (73.04 +/- 1.04) are confirmed against the CrossRef abstracts. The HSC Y3 and DES Y3 abstracts are
not on CrossRef; those values are kept from the previous chapter.
