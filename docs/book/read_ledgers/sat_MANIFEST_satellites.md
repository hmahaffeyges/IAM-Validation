# MANIFEST — p2_19 Missing satellites (revision, 2026-10-03, base HEAD e6c184e)

Author rulings applied: 2026-10-01 / 2026-10-03 (failed side predictions not carried: Mechanism B and the census), 2026-10-02 (only what is accurate or a live prediction). p2_17 and p2_18 untouched.

## Files
| File | What |
|---|---|
| docs/book/part2/p2_19_missing_satellites.tex | 191 lines (HEAD 85). Problem; virial coupling; growth suppression from µ<1 with derivation, three implementation sizes, Press–Schechter derivation and satellite-mass numbers; chain table; live tests; one open sentence. |
| docs/book/figscripts/fig_p2_satellites.py | rewritten: produces only fig_sat_mechanisms (three panels: µ(z); ΔD/D in forms i–iii; PS abundance change with satellite-halo ν range) |
| docs/book/figures/part2/fig_sat_mechanisms.{pdf,png} | regenerated (overlap check 0) |
| docs/verification/scripts/verify_cluster_mass_satellites.py (+ _output.txt) | shared with p2_17/p2_18 (sections A–E unchanged); section G reduced to the coupling and chain values (+ power −1.55 %, µ(z=2)); section H adds ΔlnN at satellite masses; former census and minimum-mass sections dropped |

Not in the zip and no longer used by the book: figures/part2/fig_sat_census.{pdf,png} (lead to delete or leave unreferenced).
Bibliography: all 20 keys cited by p2_19 are in iam.bib at e6c184e; no bib change.

## For the lead (files I do not own) — three references now dangle
- part5/p5_05c_virial_decoherence.tex:123 `\eqref{eq:ms_mmin}` (and its surrounding Mechanism B passage, l. ~100–130, carries the same failed prediction; owner to remove per the same ruling).
- appendices/app_E_formulas.tex:154 entry 104 (`eq:ms_mmin`, M_min / σ_crit) — remove; entry 103 (eq:ms_ps) stays, its section pointer is now 'Growth suppression from µ<1'.
- appendices/app_I_provenance.tex:91 `fig:sat_census` row — remove; row 90 (fig:sat_mechanisms) stays, script fig_p2_satellites.py.
- Also check for Mechanism B text outside p2_19: part5/p5_09_open.tex:41, part5/p5_03_time.tex (fig:satellite_context, Table tab:satellites_below), appendices/app_B_errata_physics.tex:48, app_F_glossary.tex, app_N_notation.tex:217–218 (all reference ch:satellites).

## Reading ledger
Missing_Satellites.pdf: 578 PDF-text lines with page markers (ledger 574), read 1–578 in ≤50-line chunks (2026-10-03); LaTeX iam_missing_satellites.tex (533 lines), 12 equations checked; errata M2–M8, V11; MISSING_SATELLITES_CHECK.md (37 lines).

## Verified numbers (verify_cluster_mass_satellites_output.txt)
µ(0) = 0.8638 (13.62 %); µ(z=2) = 0.9977, E = 0.135; ΔD/D today −0.78 / −0.67 / −1.87 % (forms i/ii/iii); linear power −1.55 % (i); σ8 −1.57 % (L1), −1.10 % (L2); fσ8 −4.25 / −2.17 / −1.35 / −0.41 % at z = 0 / 0.3 / 0.5 / 1; dlnn/dε = ν²−1 (sympy); σ_M = 6.96 / 5.87 / 4.83, ν = 0.242 / 0.287 / 0.349 at 1e7 / 1e8 / 1e9 M⊙ (EH no-wiggle, σ8 0.811) → ΔlnN = +0.73 / +0.71 / +0.68 %; β_m posterior 0.1583 ± 0.0033; H0 matter 72.26; offsets 0.37σ, 0.75σ; σ8 vs 0.802: 0.12σ.

## Carriage table (every equation, step, table, figure and quantitative claim carried)

| # | Paper location | Content | Book location | Verdict (source) | Label |
|---|---|---|---|---|---|
| MS1 | Abstract | factor ~8: ~500 subhalos vs ~60 satellites | p2_19_missing_satellites.tex:16 | CARRIED | observed |
| MS2 | Abstract | Mechanism A: mu0 = 0.864, 13.6 %; derived from Landauer cost on the horizon; mu -> 1 at high z | p2_19_missing_satellites.tex:19 | CARRIED-CORRECTED (canon 13.62 %; coupling, not growth) | calc |
| MS3 | Abstract | 17 chains, Delta chi2 = +0.54 | p2_19_missing_satellites.tex:128 | CARRIED-CORRECTED (M4/M6: 18) | measured fitted |
| MS4 | Abstract | Euclid DR1 Oct 2026, sigma(mu0) 0.04, 3.4 sigma | p2_19_missing_satellites.tex:156 | CARRIED-CORRECTED (M4/M6: complete DR1 mid-2027; sensitivity only as sec:lt_euclid) | prediction |
| MS5 | S1 | Klypin, Moore; 500 subhalos vc > 10 km/s; ~60 satellites; Koposov, Tollerud | p2_19_missing_satellites.tex:25 | CARRIED (+ LVDB 68) | observed |
| MS6 | S1 | reionisation, stripping, feedback act on baryons | p2_19_missing_satellites.tex:30 | CARRIED-CORRECTED (Benson 2002 is paper II; the printed Read et al. 2006 reference does not resolve, not cited) | observed |
| MS7 | S1 | IAM origin: same process as dark energy; acts on DM subhalos | p2_19_missing_satellites.tex:34 | CARRIED | conjecture |
| MS8 | S1 | roadmap | p2_19_missing_satellites.tex:36 | CARRIED (for the sections carried) | - |
| MS9 | S2 Eq.1 | 2K + V = 0, K = |V|/2 | p2_19_missing_satellites.tex:44 | CARRIED | derived |
| MS10 | S2 | exact theorem, H atoms to clusters | p2_19_missing_satellites.tex:45 | CARRIED | derived |
| MS11 | S2 | two channels: potential half to curvature, kinetic half Landauer cost to an encoding surface | p2_19_missing_satellites.tex:48 | CARRIED-CORRECTED (check file: the book's statement of IAM's Law) | prediction |
| MS12 | S2 Eq.2 | beta_m = Omega_m/2 = 0.1575; posterior 0.1583 +/- 0.0033, 0.2 sigma | p2_19_missing_satellites.tex:53 | CARRIED-CORRECTED (M6: fixed in chains; posterior = Omega_m/2 on posterior Omega_m; canon 0.15765) | prediction calc |
| MS13 | S3.1 Eq.3 | E(a); E->0, dE/da > 0; ledger accumulates | p2_19_missing_satellites.tex:63 | CARRIED | interp |
| MS14 | S3.1 | photons dtau = 0, matter sector only | p2_19_missing_satellites.tex:67 | CARRIED | interp |
| MS15 | S3.1 Eq.4 | mu(a) | p2_19_missing_satellites.tex:71 | CARRIED-CORRECTED (LG10: beta_m E H0^2) | interp |
| MS16 | S3.1 Eq.5 | mu(0) = 1/(1+beta_m) = 0.864; mu0 = -0.136; LCDM at z >~ 2 | p2_19_missing_satellites.tex:76 | CARRIED (+ E, mu at z = 2) | calc |
| MS17 | S3.1 Eq.6 | growth equation with mu | p2_19_missing_satellites.tex:84 | CARRIED-CORRECTED (Omega_m(a)H^2 written out; form (i) named; Level 2 form (ii)) | derived |
| MS18 | S3.1 | suppression of delta_m reduces low-mass halo abundance | p2_19_missing_satellites.tex:86 | CARRIED | - |
| MS19 | S3.2 Eq.7 | Delta D/D = -0.074 | p2_19_missing_satellites.tex:93 | CARRIED-CORRECTED (M5: -0.78 % (i), -0.67 % (ii), -1.87 % (iii)) | calc |
| MS20 | new | linear power -1.55 % scale-independent | p2_19_missing_satellites.tex:96 | ADDED (verify G) | calc |
| MS21 | S3.2 | Press-Schechter dn/dM ~ exp(-dc^2/2 sigma^2), dc 1.686 | p2_19_missing_satellites.tex:103 | CARRIED-CORRECTED (full PS form; derivation steps added) | derived |
| MS22 | S3.2 | 7.4 % lower D lowers sigma_M; exponential sensitive at 1e7-1e9 Msun; suppression independent of baryons | p2_19_missing_satellites.tex:109 | CARRIED-CORRECTED (Delta ln n = (nu^2-1) eps; nu = 0.24-0.35 at 1e7-1e9; +0.68 to +0.73 %; sub-per-cent) | derived calc |
| MS23 | S3.3 | 17 chains, R-1 < 0.01, MGCAMB + Cobaya, no discarded runs | p2_19_missing_satellites.tex:126 | CARRIED-CORRECTED (M4: 18 chains, all R-1 <= 0.010) | measured |
| MS24 | Table 1 | beta_m, sigma8, H0 photon/matter, Delta chi2 | p2_19_missing_satellites.tex:133 | CARRIED-CORRECTED (M6; sigma8 vs Stolzner 2025 joint analysis, 0.1 sigma) | fitted calc |
| MS25 | S3.3 | sigma8 0.7998 consistent with lensing at 0.1 sigma; lower sigma8 = less small-scale power | p2_19_missing_satellites.tex:143 | CARRIED | calc interp |
| MS26 | Fig.1(a) | mu(z), mu0 = 0.864, mu -> 1 at high z | p2_19_missing_satellites.tex:123 | CARRIED-CORRECTED (redrawn on _bookstyle: panel a; Euclid band and timestamp footer not drawn; panels b, c added: growth in three forms, PS abundance) | calc |
| MS27 | S5 | continuous suppression z~2 to today, one coupling; acts on DM, not baryons; not exclusive with baryonic processes | p2_19_missing_satellites.tex:147 | CARRIED (growth part) | interp |
| MS28 | S6 | Mechanism A: Euclid mu0 test; mu(z=0) > 0.90 at 2 sigma in tension | p2_19_missing_satellites.tex:155 | CARRIED-CORRECTED (Euclid as sec:lt_euclid; template translation stated) | prediction |
| MS29 | S6 | cross-check: correlated suppression of power at the same scales | p2_19_missing_satellites.tex:166 | CARRIED-CORRECTED (scale-independent linear deficit stated) | prediction |
| MS30 | new | fsigma8 deficits 4.25/2.17/1.35/0.41 %; sigma8 | p2_19_missing_satellites.tex:162 | ADDED (the growth mechanism's live test) | prediction |
| MS31 | S7 | baryonic framing vs whether DM halos form in the first place | p2_19_missing_satellites.tex:171 | CARRIED (growth part) | interp |
| MS32 | author note 2026-10-03 | one open sentence: floor at infall before tidal stripping | p2_19_missing_satellites.tex:176 | CARRIED (one sentence) | openprob |

## Not carried (for the author; nothing about these appears in the chapter)

| # | Paper location | Content | Ruling / errata |
|---|---|---|---|
| X1 | Abstract, S2, S4.1-4.4, S5, S6 (Mechanism B), S7, Figs 1(b), 2 | Mechanism B: closure/dispersal condition, local black-hole threshold Eq. 8, t_dyn Eqs. 9-10, M_min Eq. 11, sigma_crit Eq. 12, sigma^3 scaling, normalisation, regime picture, Rubin test | author rulings 2026-10-01 and 2026-10-03 (failed side prediction not carried); errata M2, M3, M8, V11; MISSING_SATELLITES_CHECK |
| X2 | S6 Mechanism A | "5.4 sigma with Euclid + DESI Y5" | unsourced (SP5); MISSING_SATELLITES_CHECK |
| X3 | Abstract, S4.3, S7 | sigma^3 / sigma^2 / sigma^4 family; M-sigma; cusp-core | M7; author ruling 2026-10-03 |
| X4 | S5 | "only proposals from the same framework as the cosmological constant and M-sigma" | M7 |
| X5 | Ack., Data avail., figure footers | timestamped predictions, archive statements | stand-alone book rule; author ruling 2026-10-03 |
| X6 | census (book addition of 2026-10-03 draft) | Local Volume Database census against the 4 km/s floor; fig_sat_census | author ruling 2026-10-03 |

## Proposed errata rows (from the 2026-10-03 audit; not written into PAPER_ERRATA.md)
| Where | Printed | Correct |
|---|---|---|
| MS §3.2 | exponential factor sensitive at 1e7–1e9 M⊙; suppression of satellite count | ν = 0.24–0.35 there: at fixed mass PS abundance changes by +0.7 % (ν<1); sub-per-cent either way |
| MS refs | Read, Pontzen, Walker, Steger 2006, MNRAS 367, 387 | does not resolve in CrossRef |
| MS refs | Benson et al. 2002 'II', DOI …05387.x | paper II is …05388.x (MNRAS 333, 177) |
| MS §4.2 (not carried) | t_dyn ≈ √(π/6) GM/σ³; Ω_m + Ω_Λ = 1.0002 | (π/6) GM/σ³ with the paper's own ρ; 1.0000 |

## Static checks
Braces balanced; environments matched; every \ref/\eqref in p2_19 resolves; no duplicate labels across the book; all 20 \cite keys in iam.bib; figure exists; floats [htbp]. Not compiled (no TeX in sandbox). Words checked absent from the chapter text: rejected, census, σ_crit, M_min, dispersal, vault, the quantum-processor report/the semiconductor report/the methylation report, 'the paper', cusp, M–σ. 'Mechanism' occurs only as 'the growth mechanism' (prose, l. 129, 132 and similar) and in the header comment quoting the paper's title (l. 1); neither 'Mechanism A' nor 'Mechanism B' appears.
