# MANIFEST — dual-sector chapters (Part 2): line-for-line carriage

Owner files: `part2/p2_10_dual_sector_validation.tex` (ch:dsvalidation), `part2/p2_05_dual_sector_note.tex` (ch:dsnote),
`part2/p2_04_dualsector_chains.tex` (ch:dual). Repository HEAD at clone: e37aabb (later than 12d8fb1 / 41f7646). Nothing pushed.

## Files in this delivery
| File | What it is |
|---|---|
| `docs/book/part2/p2_10_dual_sector_validation.tex` | rewritten: 103 → 605 lines; the Dual-Sector Validation paper carried sentence by sentence in the author's wording (wave-2 source), in order, corrected |
| `docs/book/part2/p2_05_dual_sector_note.tex` | rewritten: 89 → 242 lines; the Dual Sector Note carried sentence by sentence in the author's wording (wave-2 source), in order, corrected |
| `docs/book/part2/p2_04_dualsector_chains.tex` | extended: 113 → 192 lines; existing text kept; chain configuration, priors, likelihoods, per-chain record, Level 2/2b table added |
| `docs/book/figscripts/fig_p2_dual_sector.py` | new figure script on `_bookstyle.py` (six figures) |
| `docs/book/figures/part2/fig_dsv_three_tests.pdf/.png` | paper Fig. 1 and Fig. 3, redrawn from the Pantheon+ release |
| `docs/book/figures/part2/fig_dsv_hubble.pdf/.png` | paper Fig. 2, redrawn |
| `docs/book/figures/part2/fig_dsv_systematics.pdf/.png` | paper Fig. 4, redrawn with recomputed values |
| `docs/book/figures/part2/fig_dsv_schematic.pdf/.png` | paper Fig. 5, redrawn with corrected values |
| `docs/book/figures/part2/fig_dsnote_probes.pdf/.png` | Note Figs. 1(b,d,e) and 2(b,d,g), redrawn with chain values |
| `docs/book/figures/part2/fig_dsnote_growth.pdf/.png` | Note Figs. 1(c), 2(c,e), redrawn with chain values |
| `docs/verification/scripts/verify_dual_sector_chapters.py` + `_output.txt` + `_data.json` | every number of the three chapters recomputed (sympy algebra, Pantheon+ diagonal and full covariance, chain table) |
| `docs/book/bib_dualsector.bib` | five new sources, CrossRef-verified |
| `MANIFEST.md` | this file |

## main.tex
No new chapter files; the order in `main.tex` is unchanged (p2_04 line 24, p2_05 line 27, p2_10 line 30).
One line must change for the new bibliography file (lead to apply):
```
\bibliography{iam,bib_dualsector}
```
(replacing `\bibliography{iam}` at `main.tex` line 115).

## Reading ledger
Text extracted with pypdfium2 (page text, one end-of-page marker line per page). Line counts match `docs/book/PAPER_LINE_COUNTS.md` exactly:
Dual_Sector_Validation_Paper.pdf 1,239 lines (12 pages); IAM_Dual_Sector_Note.pdf 573 lines (9 pages). Every line read in chunks of 50 lines with
`read_file`; no chunk came back truncated. LaTeX read for the exact equations: `Dual_Sector_Validation_Paper.tex` (equation blocks l. 59–143,
286–304, 493–498) and `IAM_Dual_Sector_Note.tex` (l. 78–85, 174–176, 184–186, 197–202). Also read in full: PAPER_ERRATA.md rows 1–100 and every
later row naming these papers (N1, L22, X1–X6, SX1–SX10), `chains/DUAL_SECTOR_VALIDATION_CHECK.md` (69 lines), `chains/DUAL_SECTOR_NOTE_CHECK.md`
(38), `scripts/verify_dual_sector_validation.py` (46) + output (15), `verify_beta_gamma_output.txt` (8), the three owned chapters before editing
(103, 113, 89 lines), `p2_07` Section sec:lt_euclid, the 14 Level 1 `*.input.yaml`, `CHAIN_EXTRACTION_FINAL.csv` (19), `CHAIN_PAIRS_FINAL.csv` (5),
`CHAINS_AB_COMPLETE.md` (8).

### Dual_Sector_Validation_Paper.pdf (1,239 lines)
| chunk | lines | read |
|---|---|---|
| 1 | 1-50 | yes (read_file, 2026-10-03; no truncated chunk) |
| 2 | 51-100 | yes (read_file, 2026-10-03; no truncated chunk) |
| 3 | 101-150 | yes (read_file, 2026-10-03; no truncated chunk) |
| 4 | 151-200 | yes (read_file, 2026-10-03; no truncated chunk) |
| 5 | 201-250 | yes (read_file, 2026-10-03; no truncated chunk) |
| 6 | 251-300 | yes (read_file, 2026-10-03; no truncated chunk) |
| 7 | 301-350 | yes (read_file, 2026-10-03; no truncated chunk) |
| 8 | 351-400 | yes (read_file, 2026-10-03; no truncated chunk) |
| 9 | 401-450 | yes (read_file, 2026-10-03; no truncated chunk) |
| 10 | 451-500 | yes (read_file, 2026-10-03; no truncated chunk) |
| 11 | 501-550 | yes (read_file, 2026-10-03; no truncated chunk) |
| 12 | 551-600 | yes (read_file, 2026-10-03; no truncated chunk) |
| 13 | 601-650 | yes (read_file, 2026-10-03; no truncated chunk) |
| 14 | 651-700 | yes (read_file, 2026-10-03; no truncated chunk) |
| 15 | 701-750 | yes (read_file, 2026-10-03; no truncated chunk) |
| 16 | 751-800 | yes (read_file, 2026-10-03; no truncated chunk) |
| 17 | 801-850 | yes (read_file, 2026-10-03; no truncated chunk) |
| 18 | 851-900 | yes (read_file, 2026-10-03; no truncated chunk) |
| 19 | 901-950 | yes (read_file, 2026-10-03; no truncated chunk) |
| 20 | 951-1000 | yes (read_file, 2026-10-03; no truncated chunk) |
| 21 | 1001-1050 | yes (read_file, 2026-10-03; no truncated chunk) |
| 22 | 1051-1100 | yes (read_file, 2026-10-03; no truncated chunk) |
| 23 | 1101-1150 | yes (read_file, 2026-10-03; no truncated chunk) |
| 24 | 1151-1200 | yes (read_file, 2026-10-03; no truncated chunk) |
| 25 | 1201-1239 | yes (read_file, 2026-10-03; no truncated chunk) |

### IAM_Dual_Sector_Note.pdf (573 lines)
| chunk | lines | read |
|---|---|---|
| 1 | 1-50 | yes (read_file, 2026-10-03; no truncated chunk) |
| 2 | 51-100 | yes (read_file, 2026-10-03; no truncated chunk) |
| 3 | 101-150 | yes (read_file, 2026-10-03; no truncated chunk) |
| 4 | 151-200 | yes (read_file, 2026-10-03; no truncated chunk) |
| 5 | 201-250 | yes (read_file, 2026-10-03; no truncated chunk) |
| 6 | 251-300 | yes (read_file, 2026-10-03; no truncated chunk) |
| 7 | 301-350 | yes (read_file, 2026-10-03; no truncated chunk) |
| 8 | 351-400 | yes (read_file, 2026-10-03; no truncated chunk) |
| 9 | 401-450 | yes (read_file, 2026-10-03; no truncated chunk) |
| 10 | 451-500 | yes (read_file, 2026-10-03; no truncated chunk) |
| 11 | 501-550 | yes (read_file, 2026-10-03; no truncated chunk) |
| 12 | 551-573 | yes (read_file, 2026-10-03; no truncated chunk) |

## Wave-2 source (author's own words)
`docs/book/coverage/wave2/02_Dual_Sector_Validation.tex` (549 lines) and `04_IAM_Dual_Sector_Note.tex` (407 lines) read in full in 50-line chunks.
Both chapters were re-set from these files: the author's sentences carried in order, changed only where an erratum, check file or book rule requires.
Header items resolved: Validation Eqs. 2–4, 6–12 now in p2_10 (Eqs. `eq:dsv_bg`, `eq:dsv_bm`, `eq:dsv_H0g`, `eq:dsv_mbcorr`, `eq:dsv_mu`, `eq:dsv_dL`,
`eq:dsv_Hz`, `eq:dsv_chi2`, `eq:dsv_priors`, `eq:dsv_H0local`, `eq:dsv_shape`); tables at LaTeX l.172/201/230/257/359 → `tab:dsv_testA/B/C`,
`tab:dsv_compare`, `tab:dsv_bins`; Table VI → `tab:dsv_observables`; Figs. 1–5 redrawn (`fig:dsv_three_tests`, `fig:dsv_hubble`, `fig:dsv_three_tests`(b),
`fig:dsv_systematics`, `fig:dsv_schematic`). Note Eqs. 1–2, 5–6 → `eq:dsn_timelike`, `eq:dsn_null`, `eq:dsn_mu`, `eq:dsn_sigma`; Table at l.233 →
`tab:dsn_evidence`; Figs. → `fig:dsnote_probes`, `fig:dsnote_growth`. FLAG comments: 15 first-person / 'this work' flags made impersonal; 'potential'
(l.376) replaced by 'may reflect'; '3.4σ' (l.511) replaced by the sec:lt_euclid statement; Note 'duration' (l.59), 'actual' (l.99), 'DR1' (l.223) handled
(physics-terms rule; S10). PDF-only sentences listed at the ends of the wave-2 files are figure text, carried as redrawn figure content. Cite keys
Amendola2020 → Amendola2018 (the 2020 LRR citation does not exist; CrossRef), PlanckCollaboration2020 → Planck2018VI; IAMManuscript2025,
IAMCompendium2025 (the author's manuscripts) replaced by chapter references.

## Corrections applied (sources)
Validation paper: errata D1–D14 (D1 is marked 'author' in the errata; the author-approved D12 carries the same conclusions and is applied), plus
DUAL_SECTOR_VALIDATION_CHECK items 1–16. Note: errata S1–S10, DUAL_SECTOR_NOTE_CHECK items 1–12. Cross-paper: D7/X4 (β_γ bound), T10/S9 (H0² in µ),
T13/LG12 (f(R), DGP), T27 (δφ = 0 at linear order), V1 (β_m fixed), P4 (10.9σ), W1 (wording), settled values (1 − µ = 13.62 % is not growth
suppression; fσ8 deficits 4.25/2.17/1.35/0.41 %). New corrections found here (recomputed, evidence in `verify_dual_sector_chapters_output.txt`):
1. Eq. 7 prints both `/10 pc` and `+25`; with d_L in Mpc (as the code does) the `+25` is right, with 10 pc it is not. Book: both forms written correctly.
2. Eq. 6 `m_b^corr = m_b − M`: the release's `m_b_corr` is already the corrected apparent magnitude; the book writes µ_obs = m_b^corr − M.
3. 'Nelder–Mead … ensuring global convergence': the simplex stops off the minimum (Test C: 722.04 at H0 = 60 from a start at 70; Powell on Test B
   722.56, on Test A 723.26). Book: analytic offset + grid for every number.
4. Table V / Fig. 4 (D5 'recompute'): common-offset fit (full covariance) gives β = 0.000 [−0.005, +0.003], 0.000 [−0.013, +0.010], +0.070
   [+0.022, +0.117] (1.5σ, not 3.1σ); own-offset fits scatter (−0.29 to +0.67). Both in the book table.
5. §VI.C 'Δβ < 0.002 over Ω_m 0.308–0.322': recomputed ±0.015 (full covariance).
6. §VI.D 'optimisers identical': not reproduced (item 3).
7. 1701 'SNe' → 1701 light curves of 1550 SNe (Scolnic et al. 2022).
8. Reference [4] 'Living Rev. Rel. 23, 2 (2020)' does not exist as cited; Amendola et al., LRR 21, 2 (2018) (CrossRef).
9. Note §1: the geodesic law as an independent postulate — Einstein, Infeld and Hoffmann (1938) derived the motion from the field equations; added.
10. Diagonal vs full covariance at Ω_m = 0.315: best β −0.11 (diagonal) vs −0.035 (full); the book states the difference and uses the full value.

## Exclusions and items for the author
- N45: Note Fig. 2(h) and summary box — χ² by probe, 65 points, Δχ² = 79.8, '8.9σ' — EXCLUDED (errata S6, confirmed; pre-chain compilation).
- N42/N46: S8 'physical Ω_m = 0.753' — not carried (S6).
- N2, N57: author, email, Zenodo DOI — excluded (stand-alone rule).
- V87, V89: pointers to the author's own manuscripts ([6], [7], 'IAM–CAMB Technical Note') replaced by chapter references (stand-alone rule).
- V88: acknowledgments of the Validation paper (data credit, software, discussion with an AI assistant) — front matter; not placed in chapter text.
- Not redrawn: Note Fig. 1(a) CMB TT panel (the photon bound is shown by fig:beta_gamma); Fig. 1(f) CMB lensing (corrected number given in
  text); the 32-point cosmic-chronometer data of Fig. 2(b) and the fσ8 data points of Figs. 1(c), 2(c) (the compilations are not in the repository;
  curves recomputed, data not plotted).
- Interpretive passages carried as \interp/\conjecture: the Einstein boundary (N4, N8, N9), the Λ feedback loop and coincidence (N13, N15), the
  Hubble-tension reading (V6, V73, V85), 'are supernovae special' (V65).
- The open problem of how a distance built from light inherits the matter normalisation (p2_04 \openprob) is referenced from p2_10, not resolved.
- Remaining bib entries without a DOI field in iam.bib (pre-existing, not owned): Abazajian2016, DESI2016, DESY3, HSCY3, Einstein1915, Riess2022
  (Riess2022 DOI 10.3847/2041-8213/ac5c5b confirmed on CrossRef; the lead may add it).

## Verdict counts
ADDED: 8, CARRIED: 63, CARRIED-CORRECTED: 81, EXCLUDED: 2, NOT CARRIED: 1

## Item table (every equation, derivation step, table, figure and quantitative claim)
Book locations are file:line in this delivery. Status labels are those printed at the item.

| # | Paper location | Content | Book location | Verdict | Correction source | Label |
|---|---|---|---|---|---|---|
| V1 | title (l.1-2) | title 'SNe Validate Matter-Sector H0 Normalization with LCDM Geometric Consistency' | p2_10_dual_sector_validation.tex:7 | CARRIED-CORRECTED | D12 (author approved title) | - |
| V2 | abstract (l.7-10) | late-time expansion couples differently to photons and matter; SNe hosted in galaxies should probe matter sector | p2_10_dual_sector_validation.tex:11 | CARRIED | - | \interp (prose) |
| V3 | abstract (l.10-12) | complete Pantheon+ (1588 SNe, 0.01<z<2.26), three analyses A/B/C | p2_10_dual_sector_validation.tex:12 | CARRIED | - | \observed |
| V4 | abstract (l.12-15) | SNe reject photon-sector H0 (beta -> -0.30), accept matter-sector H0 = 73.04 (beta ~ 0) | p2_10_dual_sector_validation.tex:14 | CARRIED-CORRECTED | D1, D2, D12; DUAL_SECTOR_VALIDATION_CHECK #1-3 | \derived |
| V5 | abstract (l.15-17) | SNe distances keep LCDM geometry (beta_distance ~ 0); coupling affects growth not geometry | p2_10_dual_sector_validation.tex:16 | CARRIED | - | \observed |
| V6 | abstract (l.17-19) | dual sector data-driven; Planck and SH0ES both measure correctly | p2_10_dual_sector_validation.tex:19 | CARRIED-CORRECTED | D12 (reading, not demonstration) | \interp |
| V7 | abstract (l.20-23) | mu(a)=H^2/[H^2+beta_m E(a)] < 1, Sigma = 1 | p2_10_dual_sector_validation.tex:21 | CARRIED-CORRECTED | S9/T10 (H0^2) | \derived |
| V8 | abstract (l.23-25) | testable with CAMB/CLASS and DES, Euclid, CMB-S4; reproducible in under 2 minutes | p2_10_dual_sector_validation.tex:22 | CARRIED | - | - |
| V9 | sec. I (l.27-34) | Hubble tension >5 sigma: 67.4 +- 0.5 vs 73.04 +- 1.04; EDE [3], MG [4], interacting sectors [5] | p2_10_dual_sector_validation.tex:27 | CARRIED-CORRECTED | ref [4] LRR 23, 2 (2020) does not exist as cited: Amendola et al. LRR 21, 2 (2018), CrossRef 10.1007/s41114-017-0010-3 | \observed |
| V10 | sec. I (l.35-38) | beta_gamma, beta_m sector couplings; beta_m = Omega_m/2 = 0.15765 derived, zero free parameters | p2_10_dual_sector_validation.tex:33 | CARRIED | - | \prediction |
| V11 | sec. I (l.38-46) | 15 converged chains; dchi2 +0.54; sigma8 0.809 -> 0.800; H0 67.16 (0.37 sigma), 72.26 (0.75 sigma) | p2_10_dual_sector_validation.tex:36 | CARRIED | CHAIN_EXTRACTION_FINAL.csv (recomputed) | \measured |
| V12 | sec. I (l.47-51) | beta_gamma < 1.4e-6 (95 %, MCMC); ratio < 8.5e-6; 100,000x weaker | p2_10_dual_sector_validation.tex:39 | CARRIED-CORRECTED | D7, D10 (0.0039; 0.025; 40x) | \calc |
| V13 | sec. I (l.52-61) | is sector separation ad hoc? SNe as the test; 0.01<z<2.3; which history | p2_10_dual_sector_validation.tex:42 | CARRIED | - | - |
| V14 | sec. I (l.62-77) | Tests A, B, C definitions and expectations | p2_10_dual_sector_validation.tex:49 | CARRIED | - | - |
| V15 | sec. I (l.78-81) | 'results unambiguously demonstrate SNe reject photon sector ... empirical requirement' | p2_10_dual_sector_validation.tex:55 | CARRIED-CORRECTED | D1, D12 | \derived |
| V16 | sec. II.A Eq. 1 (l.84-94) | H^2(a)=H0^2[Om a^-3 + OL + beta E(a)]; background LCDM, modification in perturbations | p2_10_dual_sector_validation.tex:64 | CARRIED | - | \derived (E: app:der:activation) |
| V17 | sec. II.A (l.95-97) | E(a)=exp(1-1/a), E->0 early, E(1)=1 | p2_10_dual_sector_validation.tex:67 | CARRIED | sympy, verify_dual_sector_chapters.py s.0 | \derived |
| V18 | sec. II.A Eq. 2 (l.99-101) | beta_gamma < 1.4e-6 (95 % CL, MCMC) | p2_10_dual_sector_validation.tex:71 | CARRIED-CORRECTED | D7 | \calc |
| V19 | sec. II.A Eq. 3 (l.102-104) | beta_m = Omega_m/2 = 0.15765 | p2_10_dual_sector_validation.tex:72 | CARRIED | Matter-sector list drops 'BAO' (D6) | \prediction |
| V20 | sec. II.A (l.105) | Level 2 Om = 0.3166 +- 0.0065 -> beta_m = 0.1583 +- 0.0033, self-consistent 0.2 sigma | p2_10_dual_sector_validation.tex:75 | CARRIED-CORRECTED | V1/S3 (consistency, not recovery); sd recomputed 0.0032 | \calc |
| V21 | sec. II.A Eq. 4 (l.106-109) | H0(photon) = 67.16 (Level 2 posterior) | p2_10_dual_sector_validation.tex:80 | CARRIED | CSV | \measured |
| V22 | sec. II.A Eq. 5 (l.110-113) | H0(matter) = H0 sqrt(1+beta_m) = 67.161 x 1.0759 = 72.26 | p2_10_dual_sector_validation.tex:81 | CARRIED | derivation at a=1 added (sympy) | \calc / \derived |
| V23 | sec. II.B (l.115-118) | SNe measure normalisation (ladder) and geometry (Hubble-diagram shape) | p2_10_dual_sector_validation.tex:87 | CARRIED | - | - |
| V24 | sec. II.B Prediction 1 (l.119-122) | matter-sector normalisation from Cepheids, TRGB, H0 ~ 73 | p2_10_dual_sector_validation.tex:90 | CARRIED-CORRECTED | D13 (predicted value 72.26) | \prediction |
| V25 | sec. II.B Prediction 2 (l.123-127) | LCDM geometry (beta_distance ~ 0): effect on growth (Om dilution, f sigma8) | p2_10_dual_sector_validation.tex:91 | CARRIED | - | \prediction |
| V26 | sec. II.C (l.128-138) | Scenarios A (photon), B (matter: beta~0 or ~0.16), C (mixed, H0~70) | p2_10_dual_sector_validation.tex:99 | CARRIED | - | - |
| V27 | sec. III.A (l.141-142) | 1701 spectroscopically confirmed SNe Ia, 0.001<z<2.26 | p2_10_dual_sector_validation.tex:108 | CARRIED-CORRECTED | own check: 1701 light curves of 1550 SNe (Scolnic 2022 abstract) | \observed |
| V28 | sec. III.A (l.142-145) | z<0.01 excluded; 1588 SNe; median sigma_mb = 0.21 mag | p2_10_dual_sector_validation.tex:110 | CARRIED | verify s.2: 1588, 0.212 | \observed |
| V29 | sec. III.A Eq. 6 (l.146-149) | m_b^corr = m_b - M | p2_10_dual_sector_validation.tex:115 | CARRIED-CORRECTED | own check: the release's m_b_corr is the apparent magnitude; distance modulus = m_b^corr - M | - |
| V30 | sec. III.B Eq. 7 (l.151-155) | mu(z) = M + 5 log10[d_L/10 pc] + 25 | p2_10_dual_sector_validation.tex:122 | CARRIED-CORRECTED | own check: +25 applies with d_L in Mpc (the code uses Mpc) | - |
| V31 | sec. III.B Eq. 8 (l.157-164) | d_L = (1+z) int c dz'/H(z';beta) | p2_10_dual_sector_validation.tex:126 | CARRIED | - | - |
| V32 | sec. III.B Eq. 9 (l.166-174) | H(z;beta) = H0 sqrt(Om(1+z)^3 + OL + beta E(1/(1+z))) | p2_10_dual_sector_validation.tex:130 | CARRIED | - | - |
| V33 | sec. III.B (l.175-176) | Planck 2020 baseline Om = 0.315, OL = 0.685 | p2_10_dual_sector_validation.tex:132 | CARRIED | Planck 2018 results (A&A 641, 2020) | - |
| V34 | sec. III.C Eq. 10 (l.178-193) | chi^2 sum over 1588 + chi^2_prior | p2_10_dual_sector_validation.tex:138 | CARRIED | - | - |
| V35 | sec. III.C (l.196-214) | priors: Test A (H0-67.4)/0.5, Test B (H0-73.04)/1.04, Test C none | p2_10_dual_sector_validation.tex:142 | CARRIED | - | - |
| V36 | sec. III.C (l.215-218) | ranges Om [0.20,0.40], H0 [60,75], beta [-0.30,0.30], M [-20,-18] | p2_10_dual_sector_validation.tex:145 | CARRIED | - | - |
| V37 | sec. III.C (l.219-221) | Nelder-Mead, 5000 iterations, adaptive step 'ensuring global convergence' | p2_10_dual_sector_validation.tex:146 | CARRIED-CORRECTED | own check (verify s.2): a simplex is not global; analytic offset + grid added | - |
| V38 | new (from D1/D3) | H0-M degeneracy and analytic minimum over M | p2_10_dual_sector_validation.tex:155 | ADDED (correction derivation) | D1, D3; sympy s.0 | \derived |
| V39 | sec. IV.A Table I (l.227-239) | Test A: Om 0.2049, H0 67.40, beta -0.3000, M -19.79, chi2 721.12, chi2/dof 0.455, eff H0 56.39 | p2_10_dual_sector_validation.tex:187 | CARRIED | reproduced exactly (verify s.2) | \calc |
| V40 | sec. IV.A (l.240-248) | beta at boundary 'attempting to reduce H0'; unphysical 56.4, >6 sigma; 'categorically reject' | p2_10_dual_sector_validation.tex:200 | CARRIED-CORRECTED | D1; check #1-2 (Om-beta valley; H0 sqrt(1+beta) not physical) | \calc |
| V41 | sec. IV.B Table II (l.253-265) | Test B: Om 0.3736, beta -0.0005, M -19.24, chi2 723.16, 0.457, eff 73.03 | p2_10_dual_sector_validation.tex:213 | CARRIED-CORRECTED | D2 (0.2049, -0.30, -19.62, 721.12); 723.16 kept as the beta~0 point of the valley | \calc |
| V42 | sec. IV.B (l.266-270) | beta ~ 0 consistent with LCDM; validates Predictions 1 and 2 | p2_10_dual_sector_validation.tex:225 | CARRIED-CORRECTED | D1, D2 | \calc |
| V43 | sec. IV.C Table III (l.274-287) | Test C: Om 0.3645, H0 60.00 (boundary), beta -0.0161, M -19.68, chi2 723.04, eff 59.52 | p2_10_dual_sector_validation.tex:236 | CARRIED-CORRECTED | check #3: flat profile (721.12 at every H0) | \calc |
| V44 | sec. IV.C (l.288-300) | boundary-seeking diagnostic; 'SNe require matter-sector H0 normalisation' | p2_10_dual_sector_validation.tex:246 | CARRIED-CORRECTED | check #3 | \calc |
| V45 | sec. IV.D (l.301-305) | three analyses converge on rejection of photon sector | p2_10_dual_sector_validation.tex:264 | CARRIED-CORRECTED | D1, D12 | \derived |
| V46 | Fig. 1 (l.314-367) | three-panel chi2 vs beta (A, B), chi2 vs H0 (C) | p2_10_dual_sector_validation.tex:174 | CARRIED-CORRECTED | redrawn from the Pantheon+ release (fig_p2_dual_sector.py); H0 panel = fig_sn_h0_flat(b) | \calc |
| V47 | Table IV (l.368-376) | comparative results A/B/C with interpretations | p2_10_dual_sector_validation.tex:255 | CARRIED-CORRECTED | D1, D2, check #3 | \calc |
| V48 | Fig. 2 (l.377-415) | Hubble diagram and residuals, LCDM (Planck) vs IAM (SH0ES normalisation) | p2_10_dual_sector_validation.tex:274 | CARRIED-CORRECTED | redrawn; one curve (the LCDM shape is the same for any H0 with M free) | \observed |
| V49 | sec. V.A Eq. 11 (l.416-423) | H0(local) = H0(base) sqrt(1+beta_m) ~ 73; overall calibration effect | p2_10_dual_sector_validation.tex:289 | CARRIED-CORRECTED | D13 (72.26) | \prediction |
| V50 | sec. V.A Eq. 12 (l.424-431) | d_L proportional to (1+z) int dz'/H | p2_10_dual_sector_validation.tex:295 | CARRIED | - | - |
| V51 | sec. V.A (l.432-436) | geometric modification < 1 % for z < 2, subdominant | p2_10_dual_sector_validation.tex:297 | CARRIED-CORRECTED | D9 (-6.5 %/-2.3 %; shape +2.3 %, +3.6 %, +4.8 %; recomputed) | \calc |
| V52 | sec. V.B Eq. 13 (l.437-443) | Omega_m(a;beta) dilution < Omega_m(a;0) | p2_10_dual_sector_validation.tex:318 | CARRIED | - | \derived |
| V53 | new | Omega_m(a;beta)/Omega_m(a;0) = mu(a) | p2_10_dual_sector_validation.tex:322 | ADDED (derivation) | sympy s.0 | \derived |
| V54 | sec. V.B (l.444-447) | sigma8 0.800 vs 0.809; f sigma8 SDSS/BOSS/eBOSS probe growth | p2_10_dual_sector_validation.tex:325 | CARRIED | D11: 'consistent with beta_m' -> growth tests in ch:latetime/ch:level2 | \measured |
| V55 | sec. V.B (l.448-451) | SNe measure photon distances, Om dilution geometrically small, beta_distance ~0 despite beta_growth 0.157 | p2_10_dual_sector_validation.tex:329 | CARRIED-CORRECTED | D9, LG9 (background unmodified is the reason) | \interp |
| V56 | sec. V.C (l.454-466) | three tests could have falsified separation (three bullets); all converge | p2_10_dual_sector_validation.tex:333 | CARRIED-CORRECTED | D1 | \interp |
| V57 | Fig. 3 (l.467-499) | beta_m vs H0 contours, best fit (73.04, 0), growth beta_m = 0.157 (RSD) | p2_10_dual_sector_validation.tex:178 | CARRIED-CORRECTED | redrawn; bands horizontal (D1); beta_m fixed line, +-0.029 removed (D11) | \calc |
| V58 | sec. VI.A (l.501-509) | beta and M not degenerate; distinct minima | p2_10_dual_sector_validation.tex:344 | CARRIED-CORRECTED | D3; recomputed valley | \calc |
| V59 | sec. VI.B, Table V (l.510-530) | bins 1094/419/75: beta -0.007+-0.003, -0.004+-0.005, +0.037+-0.012; H0 73.10/73.02/73.06; high-z 3.1 sigma | p2_10_dual_sector_validation.tex:366 | CARRIED-CORRECTED | D5 (one set, recomputed: common offset 0.000, 0.000, +0.070+-0.048 = 1.5 sigma; own-offset values also given) | \calc |
| V60 | Fig. 4 (l.531-604) | panels A bins (892/486/210), B Om sensitivity (<0.002), C sample size, D summary incl. optimizers | p2_10_dual_sector_validation.tex:381 | CARRIED-CORRECTED | D5; redrawn with recomputed values; panel D text replaced by tab:dsv_bins and text | \calc |
| V61 | sec. VI.C (l.605-608) | Om 0.315 +- 0.007: Delta beta < 0.002 | p2_10_dual_sector_validation.tex:389 | CARRIED-CORRECTED | recomputed: +-0.015 (full cov.) | \calc |
| V62 | sec. VI.D (l.611-614); Fig. 4D | Powell, L-BFGS-B identical: -0.0005/-0.0008/-0.0006 | p2_10_dual_sector_validation.tex:393 | CARRIED-CORRECTED | recomputed: Powell stops at 722.56/723.26; L-BFGS-B and NM reach 721.12 | \calc |
| V63 | Fig. 4C | stable across 100-1588 SNe; not driven by outliers | p2_10_dual_sector_validation.tex:397 | CARRIED | recomputed | \calc |
| V64 | sec. VII.A (l.617-627) | systematics cannot explain (1) directional rejection, (2) three-test consistency, (3) SH0ES 73.04 | p2_10_dual_sector_validation.tex:402 | CARRIED-CORRECTED | D1 withdraws (1), (2); (3) carried | \observed |
| V65 | sec. VII.B (l.628-635) | SNe special? (1) local calibration, (2) galaxy hosting, (3) SH0ES classification | p2_10_dual_sector_validation.tex:408 | CARRIED | open problem pointer (ch:dual) | \interp |
| V66 | sec. VII.C (l.636-639) | MG predicts distance deviations; affects all matter equally | p2_10_dual_sector_validation.tex:418 | CARRIED-CORRECTED | D14, T13, LG12 | \observed |
| V67 | sec. VII.D (l.640-649) | EDE (1) not sector-dependent, (2) worsens S8 while IAM improves it, (3) no SN prediction | p2_10_dual_sector_validation.tex:426 | CARRIED-CORRECTED | D14 (S8 0.822 vs 0.830, 0.8 sigma) | \measured |
| V68 | sec. VIII.A (l.651-656) | uniform beta -> catastrophic 36 sigma acoustic-scale tension; ratio < 8.5e-6 from data | p2_10_dual_sector_validation.tex:435 | CARRIED-CORRECTED | D8 (fixed parameters; 34/36 sigma for 0.18, 30 sigma for beta_m), D7 | \calc |
| V69 | sec. VIII.A (l.657-660) | SNe analysis unambiguously selects matter sector; empirically required | p2_10_dual_sector_validation.tex:443 | CARRIED-CORRECTED | D1, D12 | \calc |
| V70 | Table VI (l.661-672) | observables by sector (CMB theta_s <1e-6; Planck 67.4; f sigma8 0.157; BAO matter 72.5; SH0ES; SNe 73.04; SNe geometry) | p2_10_dual_sector_validation.tex:449 | CARRIED-CORRECTED | D6 (BAO photon paths), D7, chain values | mixed (column) |
| V71 | sec. VIII.B (l.673-676) | measurements partition into photon and matter sectors | p2_10_dual_sector_validation.tex:463 | CARRIED | values corrected (67.16, 72.26) | \interp |
| V72 | Fig. 5 (l.679-718) | dual-sector schematic: photon, matter, SNe, informational actualization boxes | p2_10_dual_sector_validation.tex:466 | CARRIED-CORRECTED | redrawn; D7, D11, S6 values; mechanism box in physics terms (W1) | - |
| V73 | sec. VIII.C (l.719-730) | Hubble tension reframed: both measure correctly, different sectors | p2_10_dual_sector_validation.tex:474 | CARRIED | 'independently validated by SNe' corrected (D1) | \interp |
| V74 | sec. VIII.D (l.731-735) | CMB-S4 beta_gamma < 1e-7 | p2_10_dual_sector_validation.tex:485 | CARRIED-CORRECTED | D13 | \prediction |
| V75 | sec. VIII.D (l.736-738) | Euclid/LSST S8 = 0.78 +- 0.01 | p2_10_dual_sector_validation.tex:487 | CARRIED-CORRECTED | D4 (0.822) | \prediction |
| V76 | sec. VIII.D (l.739-742) | DESI Y5 f sigma8 ~1 %; sigma8 0.800 confirmed by Level 2 | p2_10_dual_sector_validation.tex:489 | CARRIED-CORRECTED | D13; DESI2016 precision | \prediction |
| V77 | sec. VIII.D (l.743-746) | standard sirens H0 ~ 73 | p2_10_dual_sector_validation.tex:492 | CARRIED-CORRECTED | D13, T14 (72.26; GW170817 70.0-75.5) | \prediction |
| V78 | sec. VIII.E Eq. 14 (l.749-761) | k^2 Phi = -4 pi G mu rho delta a^2; k^2(Phi+Psi) = -8 pi G Sigma rho delta a^2 | p2_10_dual_sector_validation.tex:505 | CARRIED | - | - |
| V79 | sec. VIII.E Eq. 15 (l.762-768) | mu(a) = H^2/(H^2 + beta_m E), Sigma = 1 | p2_10_dual_sector_validation.tex:509 | CARRIED-CORRECTED | S9/T10 (H0^2) | \derived |
| V80 | sec. VIII.E (l.769-773) | mu(0)=0.864 '(13.6 % growth suppression)'; mu(1)=0.982; ->1 for z>~3; Sigma=1 keeps theta_s; beta_gamma<1.4e-6 | p2_10_dual_sector_validation.tex:511 | CARRIED-CORRECTED | settled: 1-mu is not growth suppression (f sigma8 -4.25 %); D7 | \calc |
| V81 | sec. VIII.E (l.774-780) | first consequence: 12 L1 + 3 L2 chains, dchi2 +0.54 | p2_10_dual_sector_validation.tex:517 | CARRIED | - | \measured |
| V82 | sec. VIII.E (l.780-783) | Euclid sigma(mu0) ~ 0.04 detects mu0 = -0.136 at 3.4 sigma | p2_10_dual_sector_validation.tex:520 | CARRIED-CORRECTED | sec:lt_euclid / verify_euclid_template.py (1.8 sigma at 0.04; 0.3-7 sigma) | \prediction |
| V83 | sec. VIII.E (l.783-786) | mu<1 with Sigma=1 distinguishes IAM; generic MG mu != Sigma | p2_10_dual_sector_validation.tex:520 | CARRIED-CORRECTED | T13, S1 | \derived |
| V84 | sec. IX (l.788-811) | conclusions 1-5 | p2_10_dual_sector_validation.tex:525 | CARRIED-CORRECTED | D12 (conclusions: LCDM distances; beta on distances excluded) | mixed |
| V85 | sec. IX (l.812-819) | Hubble tension is sector-dependent expansion | p2_10_dual_sector_validation.tex:540 | CARRIED | D1 wording | \interp |
| V86 | sec. IX (l.820) | code public, reproducible in under 2 minutes | p2_10_dual_sector_validation.tex:544 | CARRIED | - | - |
| V87 | Note added (l.821-830) | 15 chains; dchi2 +0.54; sigma8 0.800; H0 matter 72.26; two background runs H0 ~61.5; see Technical Note | p2_10_dual_sector_validation.tex:546 | CARRIED-CORRECTED | D1 ('independently confirming' removed); pointer to the author's note replaced by Chapter ref (stand-alone rule) | \measured |
| V88 | Acknowledgments (l.831-836) | Pantheon+ data credit; NumPy/SciPy; discussion with an AI assistant; repository URL | p2_10_dual_sector_validation.tex:553 | NOT CARRIED (front matter) | data credit carried as citations; acknowledgments are front matter for the author to place | - |
| V89 | References (l.837-853) | [1]-[9] | p2_10_dual_sector_validation.tex:29 | CARRIED-CORRECTED | [6], [7] are the author's manuscripts (stand-alone rule: replaced by Chapter refs); [4] corrected (CrossRef) | - |
| V90 | Appendix A-C (l.856-1216) | complete Python code for Tests A, B, C | p2_10_dual_sector_validation.tex:565 | CARRIED (condensed) | one listing; same columns, cut, model, bounds, Nelder-Mead; d_L in Mpc with +25 | - |
| V91 | Appendix C (l.1186-1215) | interpretation printout (/H0-67.4/<1 -> photon; /H0-73.04/<1.5 -> matter) | p2_10_dual_sector_validation.tex:603 | CARRIED-CORRECTED | D1 (an H0 returned by the fit carries no sector information) | - |
| V92 | Appendix D (l.1217-1238) | data URL and file; Python >=3.8, NumPy >=1.18, SciPy >=1.5; 10-35 s per test, <2 min; no proprietary software | p2_10_dual_sector_validation.tex:557 | CARRIED | - | - |
| N1 | title (l.1-3) | On the Dual-Sector Structure of IAM: null geodesics, proper time, sector split in Einstein's equations | p2_05_dual_sector_note.tex:6 | CARRIED (retitled) | book chapter title | - |
| N2 | front (l.4-8) | author, address, email, Zenodo DOI, date | - | EXCLUDED | stand-alone rule (no author's papers or archive records) | - |
| N3 | front (l.9-13) | informal note; physical reasoning; no new claims beyond companion papers | p2_05_dual_sector_note.tex:8 | CARRIED | 'companion papers' -> other chapters of this Part | - |
| N4 | sec. 1 (l.14-18) | Einstein wrote two things defining the boundary; did not connect them | p2_05_dual_sector_note.tex:12 | CARRIED | - | \interp |
| N5 | sec. 1 (l.19-23) | geodesic equation an independent postulate; Einstein called it an imperfection | p2_05_dual_sector_note.tex:15 | CARRIED-CORRECTED | own historical check: Einstein-Infeld-Hoffmann 1938 derived the motion (CrossRef 10.2307/1968714) | \observed |
| N6 | sec. 1 Eqs. 1-2 (l.24-33) | timelike g dx dx = -1, dtau>0; null = 0, dtau = 0 | p2_05_dual_sector_note.tex:20 | CARRIED | signature and affine parameter stated | - |
| N7 | sec. 1 (l.36-39) | not approximation; causal structure; metric treats them differently | p2_05_dual_sector_note.tex:23 | CARRIED | - | - |
| N8 | sec. 1 (l.40-47) | E=mc^2: mass is energy crossed into timelike sector, proper time, duration, history; 1905 | p2_05_dual_sector_note.tex:27 | CARRIED-CORRECTED | DUAL_SECTOR_NOTE_CHECK #9 (physics terms) | \interp |
| N9 | sec. 1 (l.48-49) | Einstein found the same boundary twice | p2_05_dual_sector_note.tex:32 | CARRIED | - | \interp |
| N10 | sec. 1 (l.50-58) | core claim: processes needing dtau>0 inaccessible to photons; decoherence; S_info from matter only | p2_05_dual_sector_note.tex:34 | CARRIED | premise stated | \derived (given premise) |
| N11 | sec. 2 (l.60-64) | Lambda to hold universe static, 'greatest blunder'; property of geometry, sector-blind | p2_05_dual_sector_note.tex:43 | CARRIED-CORRECTED | wording: 'later abandoned it' (the 'blunder' phrase is second-hand) | - |
| N12 | sec. 2 (l.65-67) | vacuum baseline; quantum vacuum fluctuations map to Lambda; Lambda not wrong | p2_05_dual_sector_note.tex:47 | CARRIED | size pointer to ch:lambda | \interp |
| N13 | sec. 2 (l.68-74) | what Lambda cannot describe: decoherence events, records, horizon encoding, expansion, dilution feedback | p2_05_dual_sector_note.tex:50 | CARRIED | - | \conjecture |
| N14 | sec. 2 (l.77-80) | LCDM is null description applied uniformly; works because beta_m E smooth | p2_05_dual_sector_note.tex:56 | CARRIED | - | \interp |
| N15 | sec. 2 (l.80-85) | coincidence: dark energy dominates when structure peaks; IAM resolves 'trivially'; causal | p2_05_dual_sector_note.tex:58 | CARRIED | 'trivially' dropped | \conjecture |
| N16 | sec. 2 (l.86-92) | tools: Landauer 1961, Bekenstein-Hawking 1973, Jacobson 1995, Zurek 1980s; IAM follows them | p2_05_dual_sector_note.tex:63 | CARRIED | - | \interp |
| N17 | sec. 3 Eq. 3 (l.94-101) | -dE = T_H d(S_geo + S_info); S_geo = A_H/4G; Landauer k_B T_H ln2 per bit | p2_05_dual_sector_note.tex:72 | CARRIED-CORRECTED | S9 (units: k_B A_H/4 l_P^2; one nat per 4 l_P^2) | - |
| N18 | sec. 3 Eq. 4 (l.102-103) | S_info > 0 iff dtau > 0 | p2_05_dual_sector_note.tex:80 | CARRIED | - | - |
| N19 | sec. 3 (l.104-107) | photons: S_info = 0, standard first law, GR recovered; not special case | p2_05_dual_sector_note.tex:82 | CARRIED | - | \derived |
| N20 | sec. 3 Eq. 5 (l.108-113) | mu(a) = H^2/(H^2 + beta_m E(a)) < 1 | p2_05_dual_sector_note.tex:88 | CARRIED-CORRECTED | S9 (H0^2) | \derived |
| N21 | sec. 3 Eq. 6 (l.114) | Sigma(a) = 1 exact | p2_05_dual_sector_note.tex:89 | CARRIED | - | - |
| N22 | sec. 3 (l.115-116) | E(a) from integrating decoherence constraint; beta_m = 0.1577 from virial theorem | p2_05_dual_sector_note.tex:91 | CARRIED | 0.15765 | \derived |
| N23 | sec. 3 (l.119-123) | mu<1, Sigma=1 unique; f(R) mu>1 Sigma>1; DGP mu>1; Horndeski correlated; data beginning to prefer | p2_05_dual_sector_note.tex:95 | CARRIED-CORRECTED | S1, T13, LG12; 'data prefer' -> current data do not disfavour (ch:latetime) | \observed |
| N24 | sec. 4 (l.124-128) | four independent lines; data demanded the split | p2_05_dual_sector_note.tex:103 | CARRIED-CORRECTED | S4, D1 | - |
| N25 | Table 1 row 1 (l.161-176) | CMB: beta_gamma < 1.4e-6, ratio > 1e5, 100,000x; full Planck MCMC | p2_05_dual_sector_note.tex:111 | CARRIED-CORRECTED | S2, D7 (acoustic-scale fit; 0.0039; 0.025; 40x) | \calc |
| N26 | Table 1 row 2 (l.177-188) | Planck recovers beta_m without fitting, 0.2 sigma, strongest single result | p2_05_dual_sector_note.tex:114 | CARRIED-CORRECTED | S3, V1 | \measured |
| N27 | Table 1 row 3 (l.189-205) | 1,588 SNe select matter sector three ways | p2_05_dual_sector_note.tex:117 | CARRIED-CORRECTED | S4 | \observed |
| N28 | Table 1 row 4 (l.206-225) | DESI phantom crossing predicted artifact at z 0.33-0.43 (DR1&DR2 full-shape + BAO) | p2_05_dual_sector_note.tex:120 | CARRIED-CORRECTED | S5, S10 | \openprob |
| N29 | Table 1 caption (l.226-228) | all four consistent with beta_gamma~0, beta_m=Om/2, mu<1, Sigma=1 | p2_05_dual_sector_note.tex:106 | CARRIED-CORRECTED | row 4 open (S5) | - |
| N30 | sec. 4 bullets (l.129-142) | 17 chains; dchi2 +0.54 (below 3.84); sigma8 0.809->0.800 toward KiDS/DES/HSC; H0 67.16 +- 0.47, 72.26; background 61.5 | p2_05_dual_sector_note.tex:127 | CARRIED-CORRECTED | S7 (likelihood ratio 0.76); 10.9 sigma (P4) | \measured |
| N31 | sec. 5 Fig. 1 text (l.144-149) | (a) TT identical, beta_gamma<1e-6; (d) mu<1 to unity at high z; both predictions | p2_05_dual_sector_note.tex:151 | CARRIED-CORRECTED | D7; TT panel not redrawn (photon bound in fig:beta_gamma) | \calc |
| N32 | Fig. 1 panel (b) (l.251-267) | H(z) photon 67.4, matter 72.5 | p2_05_dual_sector_note.tex:159 | CARRIED-CORRECTED | S6 (67.16, 72.26); redrawn | \calc |
| N33 | Fig. 1 panel (c) (l.268-286) | f sigma8 vs SDSS/BOSS/eBOSS; chi2 0.8 (LCDM), 1.6 (IAM) | p2_05_dual_sector_note.tex:168 | CARRIED-CORRECTED | S6; redrawn as relative f sigma8 (perturbation form -4.25 %); data points and pre-chain chi2 labels not carried | \calc |
| N34 | Fig. 1 panel (d) (l.287-300) | mu(z=0)=0.864, Sigma=1 | p2_05_dual_sector_note.tex:162 | CARRIED | verified 0.8638 | \calc |
| N35 | Fig. 1 panel (e) (l.301-313) | matter density suppression -13.6 % at z=0 | p2_05_dual_sector_note.tex:164 | CARRIED | = mu-1 (sympy); labelled as matter fraction, not growth | \calc |
| N36 | Fig. 1 panel (f) (l.314-335) | CMB lensing reduced 2.0 %, helps A_L anomaly | p2_05_dual_sector_note.tex:155 | CARRIED-CORRECTED | S6 (0.05-0.3 %); not redrawn | \calc |
| N37 | Fig. 1 caption (l.338-345) | six-probe caption | p2_05_dual_sector_note.tex:160 | CARRIED-CORRECTED | S6 | - |
| N38 | Fig. 2 panel (a) (l.348-363) | H measurements by sector 67.4/72.5, sector gap | p2_04_dualsector_chains.tex:177 | CARRIED-CORRECTED | S6; carried by fig:two_hubble (chain values) | \measured |
| N39 | Fig. 2 panel (b) (l.364-385) | cosmic chronometers (32 points); gap 90 %/50 %/10 % at z 0.11/0.69/2.3 | p2_05_dual_sector_note.tex:163 | CARRIED-CORRECTED | E fractions and gaps carried; 32-point chronometer data not redrawn (compilation not in the repository) | \calc |
| N40 | Fig. 2 panel (c) (l.386-400) | growth rates of four models A LCDM, B background, C perturbation, D combined, with data | p2_05_dual_sector_note.tex:169 | CARRIED | recomputed; data points not redrawn | \calc |
| N41 | Fig. 2 panel (d) (l.401-421) | mu = 0.888 (z 0.11), 0.965 (0.69), 0.999 (2.3) | p2_05_dual_sector_note.tex:163 | CARRIED | verified | \calc |
| N42 | Fig. 2 panel (e) (l.422-431) | S8: LCDM 0.832, IAM (Planck Om) 0.810, IAM (physical Om) 0.753; DES/KiDS/HSC/Planck | p2_05_dual_sector_note.tex:172 | CARRIED-CORRECTED | S6 (chain 0.830/0.822; 0.753 not carried) | \measured |
| N43 | Fig. 2 panel (f) (l.432-457) | Pantheon+ Hubble diagram as photon sector, identical by construction | p2_10_dual_sector_validation.tex:274 | CARRIED-CORRECTED | S6 (SNe: LCDM distances with matter normalisation) | \observed |
| N44 | Fig. 2 panel (g) (l.458-474) | H_m/H - 1: 6.1 % (0.11), 1.8 % (0.69), 0.1 % (2.3) | p2_05_dual_sector_note.tex:164 | CARRIED | verified | \calc |
| N45 | Fig. 2 panel (h) + box (l.475-514) | chi2 by probe 69/12; 65 points, 6 probes; dchi2 79.8; 8.9 sigma | - | EXCLUDED | S6 (pre-chain compilation, reproduced by no chain); DUAL_SECTOR_NOTE_CHECK #6 | - |
| N46 | Fig. 2 summary box (l.491-512) | beta 0.15765; H0 67.36/72.48; mu0 0.8638; sigma8 0.7901 (2.6 %); S8 0.810, 0.753; 90/50/10 %; GR for z>5 | p2_05_dual_sector_note.tex:153 | CARRIED-CORRECTED | S6 (chain values); 0.753 not carried | \calc |
| N47 | Fig. 2 caption (g), sec. 5 (l.151-155, 518-519) | activation steep at z~0.3-0.5 where DESI phantom crossing appears as artifact | p2_05_dual_sector_note.tex:182 | CARRIED-CORRECTED | S5 | \openprob |
| N48 | sec. 6 (l.526-528) | Sigma != 1 detected by Euclid would challenge photon exemption | p2_05_dual_sector_note.tex:195 | CARRIED | - | - |
| N49 | sec. 6 (l.529-534) | beta_gamma above 1e-4; CMB-S4 sensitivity 1e-5 | p2_05_dual_sector_note.tex:198 | CARRIED-CORRECTED | S8, D13 (forecast open) | \openprob |
| N50 | sec. 6 (l.535-538) | beta_m/Om != 1/2; fitted beta_m does not shift | p2_05_dual_sector_note.tex:201 | CARRIED-CORRECTED | S8 | - |
| N51 | sec. 6 (l.539-542) | scale dependence; delta phi = 0 -> redshift only | p2_05_dual_sector_note.tex:204 | CARRIED-CORRECTED | T27 (linear perturbation theory) | - |
| N52 | sec. 6 (l.543-545) | Euclid, DESI Y5, CMB-S4 test; falsifiable | p2_05_dual_sector_note.tex:208 | CARRIED | - | - |
| N53 | sec. 7 (l.547-553) | summary items 1 (geodesics, 1916) and 2 (E=mc^2, 1905) | p2_05_dual_sector_note.tex:222 | CARRIED | 1915-1916 citations | \interp |
| N54 | sec. 7 (l.554-558) | finite proper time processes only in timelike sector | p2_05_dual_sector_note.tex:231 | CARRIED | - | \interp |
| N55 | sec. 7 (l.561-566) | empirical record: 17 chains, 1,588 SNe, Planck recovering beta_m, beta_gamma < 1.4e-6 | p2_05_dual_sector_note.tex:236 | CARRIED-CORRECTED | S2, S3, S4 | \measured |
| N56 | sec. 7 (l.567-568) | 'The philosophical argument explains why IAM is right. The numbers are the verdict.' | p2_05_dual_sector_note.tex:238 | CARRIED-CORRECTED | DUAL_SECTOR_NOTE_CHECK #9 (wording) | \interp |
| N57 | availability (l.569-571) | chains, CAMB source, scripts on GitHub; Zenodo DOI | p2_05_dual_sector_note.tex:241 | CARRIED | Zenodo record excluded (stand-alone rule) | - |
| C1 | chains/*.input.yaml | MGCAMB mu-Sigma setup (MG_flag 1, pure_MG_flag 2, musigma_par 1, GRtrans 0.001, sigma0 0), mu0 fixed/free, linear, lmax 2500 | p2_04_dualsector_chains.tex:65 | ADDED (from configs) | - | \measured |
| C2 | chains/*.input.yaml | flat priors and reference points | p2_04_dualsector_chains.tex:75 | ADDED (from configs) | - | \measured |
| C3 | chains/*.input.yaml | likelihood sets of the four data combinations; RSD = BOSS DR12 final consensus + eBOSS DR16 BAO | p2_04_dualsector_chains.tex:89 | ADDED (from configs) | X5, LG11 | \measured |
| C4 | CHAIN_EXTRACTION_FINAL.csv | per-chain files, samples, R-1, chi2_min (18 chains) | p2_04_dualsector_chains.tex:132 | ADDED (from outputs) | recomputed | \measured |
| C5 | CHAIN_EXTRACTION_FINAL.csv | Level 2/2b posteriors with S8 and Om/2 | p2_04_dualsector_chains.tex:158 | ADDED (from outputs) | recomputed | \measured |
| C6 | CHAIN_EXTRACTION_FINAL.csv | Om/2 = 0.1583 +- 0.0032 (0.2 sigma); 10.9 sigma (8.6 sigma both errors) for 2b | p2_04_dualsector_chains.tex:170 | ADDED (from outputs) | V1, P4 | \measured |
