# MANIFEST — line-for-line carriage: Quantum Darwinism, Measurement Problem, Two Faces of Time (2026-10-03)

Clone: sparse checkout of IAM-Validation at e37aabb (docs/book, docs/verification, docs/papers, CANON). Nothing pushed.

## Files (merge rule: git merge-file against e37aabb; this child owns the three chapters)
| file | lines before → after | what changed |
|---|---|---|
| docs/book/part2/p2_14_quantum_records.tex | 96 → 375 | Quantum Darwinism carried in full (Zurek Eqs. 1–2, redundancy, cosmic frontier, Fig. 1 redrawn, virial Eqs. 3–5 law-first, four-step chain, Eqs. 6–8 bottom-up, exact Press–Schechter count, top-down Eqs. 11–13 corrected to 7/2, horizon Eq. 15, sector split Eq. 16, particle fixed point Eqs. 17–18, BH Eqs. 19–20 corrected, consequences Eqs. 21–22 + DESI-bin table + Fig. 4 redrawn, chains, Euclid pointer, open points, lab test). Gravitational-decoherence and entanglement content of the earlier version kept verbatim. |
| docs/book/part5/p5_04_measurement.tex | 86 → 258 | Measurement paper carried section by section (problem, two sectors, criterion, gravitational channel + Table 1 + Fig. 5, double slit, delayed choice + Fig. 2, eraser + Eq. 2 + temperature test, cat, Bell, Wigner's friend, Zeno, cosmology link, discussion, falsification, status). Title changed to "Measurement as the writing of a record" (MP4/MP7). Existing figures, table and checkbox kept. |
| docs/book/part5/p5_03_time.tex | 78 → 181 | Two Faces carried in full (two faces, three quantities, Eq. 1, Fig. 1 redrawn, Eqs. 2–6, Machian criterion, sector split, shape dynamics, Higgs open question). Electroweak remark, satellite-floor and dark-matter sections (Boundary paper) kept verbatim. |
| docs/book/bib_records_time.bib | new | 6 entries, CrossRef-verified 2026-10-03 (Zurek1991, FrauchigerRenner2018, Itano1990, Kim2000, Barbour2012 = arXiv:1105.0183 published version, BarbourKoslowskiMercati2014). If bib_coverage.bib is merged, keep one copy of BarbourKoslowskiMercati2014 (same DOI). main.tex must add `bib_records_time` to its \bibliography line. |
| docs/book/figscripts/fig_records_measurement_time.py | new | 6 figures on _bookstyle.py/_cosmo.py: part2/fig_qd_local_cosmic, fig_qd_exponent, fig_qd_mu; part5/fig_time_two_faces, fig_mp_record, fig_mp_tau_systems (PDF + PNG). |
| docs/verification/scripts/verify_records_measurement_time.py (+ _output.txt) | new | every equation and number above (sympy for the algebra). |
| docs/book/read_ledgers/LEDGER_G7_30/32/33 | updated | re-read 2026-10-03 in ≤ 50-line chunks. |

main.tex: no new chapter; placement unchanged (p2_14 in Part 2; p5_03, p5_04 in Part 5).

## Read ledger
| paper | PDF text lines (pypdfium2) | ledger PAPER_LINE_COUNTS.md | read |
|---|---|---|---|
| Quantum_Darwinism_at_Cosmological_Scales | 861 | 861 | 1–861 in 50-line chunks (last 61), 2026-10-03 |
| IAM_Measurement_Problem_Quantum | 388 | 388 | 1–388 in 50-line chunks, 2026-10-03 |
| The_Two_Faces_of_Time | 249 | 247 (+2: page-marker count) | 1–249 in 50-line chunks, 2026-10-03 |
The coverage-wave LaTeX (33_, 30_, 32_ in coverage/wave2) was used for its header (missing items, errata pointers, unresolved cite keys); the PDF text is the source of the wording.

## Corrections applied
QD1 (7/2; convergence withdrawn), QD2 (Diósi–Penrose attribution), QD3 (β_m fixed; 0.505 removed), QD4 (table via Part 1), QD5 (½Mc²), QD6 (18 chains, consistent, Euclid per sec:lt_euclid, DR1 mid-2027), QD7 (EM1; Koide and strong CP out), QD8 (law first), QD9 (names removed);
MP1 (spontaneous emission erasable), MP2 (1.3 km), MP3 (Zeno with dispersive readout), MP4 (Bennett: cost at reset/erasure; record on absorption), MP5 (F form underived), MP6 (environmental decoherence of the cat), MP7 (µ/Σ not state labels; 18 chains); GD3, GD4 (ramp and capacity assumed), EN8 (S_max = 2√(1+c²)), V17, V18, EM1, KO2;
TF1 (one growth form stated, both chain forms named), TF2 (predicted with β_m fixed), TF3 (nearest encoding surface; Sakharov dropped).

## New finding for the errata (author to confirm)
- Measurement Problem, Table 1, dust-grain row: printed E_G = 1.3e-28 J, τ_PD = 7.9e-7 s, τ_IAM = 5.3e8 s for m = 1e-12 kg; these need R ≈ 0.5 µm (density ≈ 1.9e6 kg m⁻³). With R = 5 µm (density 1.9e3, as for the other rows): E_G = 1.3e-29 J, τ_PD = 7.9e-6 s, τ_IAM(300 K) = 5.3e11 s (verify B7). The book prints the recomputed row.
- Measurement Problem §3.6: 2^-111604 ≈ 10^-33596 → with Q_L(300 K) = 0.01792 eV, N = 111,612 and 10^-33,599 (rounding).

## Items for the lead (files this child does not own)
- appendices/app_E_formulas.tex:240 heads the subsection "Measurement as sector crossing"; the chapter is now "Measurement as the writing of a record".
- main.tex \bibliography: add bib_records_time.

## Static checks (all three chapters)
Braces balanced; environments balanced; no label duplicated across the book (807 labels scanned + new); every \ref/\eqref resolves against the current tree; every \cite is in iam.bib or bib_records_time.bib; every \includegraphics and \input file exists. Forbidden-word scan clean (no 'the paper', the quantum-processor report/the semiconductor report/the methylation report/the cell-reading engine, 'superseded', correspondent names, 'this book', 17 chains, Oct 2026 Euclid). LaTeX not compiled (no TeX in sandbox).

## Carriage table — Quantum Darwinism at Cosmological Scales
| paper location | content | book location | verdict | status |
|---|---|---|---|---|
| p.1 abstract | Zurek's local framework; three questions; ultimate environment = cosmic horizon; ledger E(a) | part2/p2_14_quantum_records.tex:53 | CARRIED | \interp |
| p.1 abstract | virial theorem as connector; 'kinetic half is the decoherence energy' | part2/p2_14_quantum_records.tex:86 | CARRIED-CORRECTED (QD8 law first) | \derived/\interp |
| p.1-2 abstract | two-direction derivation 'both give n = 5/2' | part2/p2_14_quantum_records.tex:181 | CARRIED-CORRECTED (QD1, T1: 7/2; convergence withdrawn) | \derived/\calc |
| p.2 abstract | mu0 - 1 = -0.136; Euclid DR1 Oct 2026, 3.4 sigma | part2/p2_14_quantum_records.tex:312 | CARRIED-CORRECTED (QD6; Euclid only per sec:lt_euclid; DR1 mid-2027) | \prediction |
| §1 l.51-57 | foundational gap: can quantum physics be trusted at cosmological scale (named correspondent) | part2/p2_14_quantum_records.tex:14 | CARRIED-CORRECTED (QD9: name removed) | — |
| §1 l.58-74 | Zurek program; einselection; redundancy; local environment | part2/p2_14_quantum_records.tex:22 | CARRIED | \derived |
| §1 l.75-90 | horizon as ultimate environment; every virialization writes; Jacobson-Cai-Kim | part2/p2_14_quantum_records.tex:80 | CARRIED | \interp |
| §1 l.82-90 | virial 1/2 partition, 37 orders | part2/p2_14_quantum_records.tex:97 | CARRIED (given once in Part 1, tab:virial_domains) | \derived/\observed/\prediction |
| §1 l.91-98 | section roadmap | — | EXCLUDED (stand-alone book; no roadmap of a paper) | — |
| Eq. 1 | |Psi> = sum c_i |s_i>|e_i> | part2/p2_14_quantum_records.tex:34 | CARRIED | \derived |
| §2.1 text | pointer states commute with interaction Hamiltonian | part2/p2_14_quantum_records.tex:37 | CARRIED | \derived |
| Eq. 2 | tau_D ~ tau_R (lambda_th/dx)^2 | part2/p2_14_quantum_records.tex:41 | CARRIED (+ numerical example 1 g, 300 K, verify A1) | \derived/\calc |
| §2.2 | redundancy R_delta | part2/p2_14_quantum_records.tex:49 | CARRIED | \derived |
| §2.3 | three questions of the cosmic frontier | part2/p2_14_quantum_records.tex:58 | CARRIED | — |
| Fig. 1 | local vs cosmic architecture (incl. 'tau 1e-70 s', 'n = 5/2', 'Euclid Oct 2026', '10^88 events') | part2/p2_14_quantum_records.tex:64 | CARRIED-CORRECTED (redrawn; 2.5e-87 s, n = 7/2, Euclid line removed; '10^88 events' not sourced, dropped) | \calc/\interp |
| Eq. 3 | 2<K> = n<|V|> | part2/p2_14_quantum_records.tex:89 | CARRIED (sympy A3) | \derived |
| Eq. 4 | 2K = |V|, K = |V|/2 | part2/p2_14_quantum_records.tex:93 | CARRIED | \derived |
| §3.1 | potential half in curvature, kinetic half decoherence energy | part2/p2_14_quantum_records.tex:101 | CARRIED-CORRECTED (QD8) | \interp |
| Eq. 5 | beta_m = Omega_m/2 = 0.1575; posterior 0.1583 +- 0.0033, 0.2 sigma | part2/p2_14_quantum_records.tex:105 | CARRIED-CORRECTED (0.15765; QD3: fixed in every chain, posterior claim removed) | \derived |
| Table 1 | virial 1/2 across scales (incl. equipartition row, cosmological 0.3 %) | part2/p2_14_quantum_records.tex:97 | CARRIED-CORRECTED via Part 1 tab:virial_domains (QD4, V18, V31) | \observed/\prediction |
| Fig. 2 | virial ratio vs scale | part2/p2_14_quantum_records.tex:97 | CARRIED-CORRECTED via Part 1 fig:virial_domains (QD4) | \calc/\observed |
| §3.2 l.233-237 | same theorem governs hydrogen and coupling | part2/p2_14_quantum_records.tex:120 | CARRIED | \interp |
| §3.3 Steps 1-3 | four-step chain atomic/gravitational/informational | part2/p2_14_quantum_records.tex:112 | CARRIED | \derived/\interp |
| §3.3 Step 4 | f_coll 0.62 x eta_vir 0.815 = 0.505 'confirms beta_m' | part2/p2_14_quantum_records.tex:118 | EXCLUDED (QD3, V18: 0.815 = 1/(2 f_coll) is a definition; untraced) — replaced by beta_m = Omega_m/2 from steps 1-3 | \derived |
| §4 intro | answer to the foundational question; 'agreement is the proof' | part2/p2_14_quantum_records.tex:210 | CARRIED-CORRECTED (QD1) | \interp |
| Eq. 6 | tau_D ~ hbar R_vir/(G M^2) (Diosi-Penrose) | part2/p2_14_quantum_records.tex:128 | CARRIED-CORRECTED (QD2 attribution: Diosi-Penrose; environmental Joos-Zeh separately) | \derived |
| Eq. 7 | tau_D ~ 1e-70 s; 1e87 shorter than Hubble time | part2/p2_14_quantum_records.tex:132 | CARRIED-CORRECTED (2.5e-87 s, 1e104; verify A2, verify_quantum_darwinism) | \calc |
| §4.1 text | rate-limiting step is virialization at H^-1 | part2/p2_14_quantum_records.tex:136 | CARRIED | \derived |
| Eq. 8 | Idot = int dn/dM M/m_p H dlnF/dlna dM | part2/p2_14_quantum_records.tex:144 | CARRIED | \derived |
| §4.1 text | nu = delta_c/(sigma D), delta_c = 1.686 | part2/p2_14_quantum_records.tex:146 | CARRIED | \derived |
| Eq. 9 | Idot ~ rho f H D^2 int nu^2 e^{-nu^2/2} dnu | part2/p2_14_quantum_records.tex:155 | CARRIED-CORRECTED (integral = sqrt(pi/2) carried; the D^2 prefactor and <nu>_eff ~ D^-1/2 not derivable; exact result n_eff = nu_min^2 - 1, sympy A5; QD1) | \derived/\calc |
| Eq. 10 | Idot ~ rho D^{5/2} f H (bottom-up) | part2/p2_14_quantum_records.tex:151 | EXCLUDED (QD1: rests on asserted <nu>_eff ~ D^-1/2); replaced by the exact particle count and the energy-weighted table | — |
| Eq. 11 | S_total = S_geo + S_info | part2/p2_14_quantum_records.tex:169 | CARRIED | \interp |
| §4.2 scalings | H~a^-3/2, A_H~a^3, T_H~a^-3/2, D~a, dt | part2/p2_14_quantum_records.tex:146 | CARRIED | \derived |
| Eq. 12 | S_info ~ int ... ~ a^{n-9/2} | part2/p2_14_quantum_records.tex:178 | CARRIED-CORRECTED (integrand as printed gives a^{n-4}; correct integrand with 1/(T_H A_H) and dt = da/(aH) gives a^{n-9/2}; sympy A6) | \derived |
| Eq. 13 | n - 9/2 = -1 => n = 5/2 | part2/p2_14_quantum_records.tex:181 | CARRIED-CORRECTED (n = 7/2; QD1, T1) | \derived |
| §4.3, Eq. 14 | convergence; Idot ~ D^{5/2} | part2/p2_14_quantum_records.tex:144 | CARRIED-CORRECTED (D^{7/2}; convergence withdrawn: two directions meet within the running) | \derived/\calc |
| §4.3 l.462-468 | quantum piece local/instantaneous; cosmological classical/slow | part2/p2_14_quantum_records.tex:214 | CARRIED | \interp |
| Fig. 3 | two-direction schematic of n = 5/2 | part2/p2_14_quantum_records.tex:154 | CARRIED-CORRECTED (redrawn quantitatively: top-down slope vs n, bottom-up nu^2 - 1; plus energy-weighted fig:neff_bottomup) | \calc/\derived |
| §5.1 | Cai-Kim first law, Bousso bound, k T_H ln2 per bit; one entry per pointer state | part2/p2_14_quantum_records.tex:220 | CARRIED | \interp |
| Eq. 15 | E(a) = exp(1 - 1/a); E(0)=0 at electroweak transition; E(1)=1; E(inf)=e | part2/p2_14_quantum_records.tex:225 | CARRIED-CORRECTED (E exponentially small, not zero, at a_EW: remark in ch:time) | \derived |
| §5.1 l.551-554 | arrow of time in Zurek's language | part5/p5_03_time.tex:73 | CARRIED (given once in ch:time; pointer in p2_14) | \interp |
| Eq. 16 | mu = H^2/(H^2 + beta_m E), Sigma = 1 | part2/p2_14_quantum_records.tex:236 | CARRIED-CORRECTED (H0^2 factor as Eq. 21; MP7 conflation note) | \derived |
| §6 | fundamental particles as fixed points; electron most stable pointer state | part2/p2_14_quantum_records.tex:242 | CARRIED-CORRECTED (EM1: within 0.3 % set by H0; area law not Bekenstein saturation) | \conjecture/\interp |
| Eq. 17 | S = pi (m_P/m)^2 'saturates Bekenstein bound' | part2/p2_14_quantum_records.tex:247 | CARRIED-CORRECTED (area law on Compton sphere; Bekenstein gives 2 pi; ch:electronmass) | \conjecture |
| Eq. 18 | E_L = S k T_GH ln2 | part2/p2_14_quantum_records.tex:252 | CARRIED (+ computed: 4.6e-8 J = 5.6e5 m_e c^2; verify A8) | \calc |
| §6 l.597-601 | fixed point with alpha^{5/2} consistent with electron mass | part2/p2_14_quantum_records.tex:255 | CARRIED-CORRECTED (EM1) | \conjecture |
| §7.1 | black holes as most efficient record-keepers; saturate BH bound | part2/p2_14_quantum_records.tex:261 | CARRIED | \interp |
| §7.1 l.614-616 | BH formation when local rate saturates the bound | — | EXCLUDED (not derived and not stated in the black-hole chapters; no correction row: listed for author) | — |
| Eq. 19 | Gamma_BH = c^3/(1920 G M ln2) | part2/p2_14_quantum_records.tex:265 | CARRIED (152.6 bits/s; 4/3 entropy-rate note; verify A9) | \derived/\calc |
| §7.2 | two-horizon framework; hot local, cold global | part2/p2_14_quantum_records.tex:270 | CARRIED | \interp |
| Eq. 20 | E_Landauer = (ln2/2) M c^2 | part2/p2_14_quantum_records.tex:274 | CARRIED-CORRECTED (QD5, V17: T_BH S_BH = Mc^2/2, Smarr) | \derived |
| §8 intro | consequences follow with no free parameters | part2/p2_14_quantum_records.tex:288 | CARRIED-CORRECTED (beta_m fixed, TF2) | \derived |
| Eq. 21 | mu with H0^2 beta_m E; mu0 - 1 = -0.136 | part2/p2_14_quantum_records.tex:291 | CARRIED | \derived |
| §8.1 | sigma8 = 0.800 consistent with KiDS, DES, HSC | part2/p2_14_quantum_records.tex:299 | CARRIED | \prediction |
| Fig. 4 | mu(z) with 1-mu at DESI bins 7.9/5.1/3.3/2.1/0.9/0.6 %; DESI crossing lines | part2/p2_14_quantum_records.tex:285 | CARRIED-CORRECTED (redrawn; bin values tabulated, recomputed A10; 1-mu distinguished from f sigma8 deficit; DESI phantom-crossing redshifts not carried: unsourced) | \calc |
| Eq. 22 | H0(matter) = H0(Planck) sqrt(1+beta_m) = 72.26; photon 67.16 | part2/p2_14_quantum_records.tex:304 | CARRIED-CORRECTED (TF2: predicted with beta_m fixed) | \calc |
| §8.3 | 17 chains R-1<0.01; Delta chi2 = +0.54 improvement; Euclid Oct 2026, sigma(mu0) 0.04, 3.4 sigma | part2/p2_14_quantum_records.tex:310 | CARRIED-CORRECTED (QD6: 18, <= 0.010, consistent not improvement; Euclid per sec:lt_euclid) | \measured/\prediction |
| §9.1 | Zurek and IAM do not resolve single outcome; cosmic accounting | part2/p2_14_quantum_records.tex:316 | CARRIED | \interp |
| §9.2 | arrow of time = accumulated receipt | part5/p5_03_time.tex:72 | CARRIED (also pointer in p2_14) | \interp |
| §9.3 | E_q ramp profile, peak eta 0.5, T^2; Penrose max at t=0; 5 sigma above 1e-13 kg at 10 mK | part2/p2_14_quantum_records.tex:359 | CARRIED-CORRECTED (GD3 ramp conjecture; discriminator per ch:gravdec) | \conjecture/\prediction |
| §9.4 (1) | Koide n >= 4 open question | — | EXCLUDED (QD7, KO2, KO5: resolved, at most three) | — |
| §9.4 (3) | strong CP and the thermodynamic identity | — | EXCLUDED (QD7: outside the book) | — |
| Acknowledgements | named correspondents; Jacobson foundation; code developers | — | EXCLUDED (QD9 names; stand-alone book) | — |
| Data availability | repository, DOI | — | EXCLUDED (stand-alone book) | — |

## Carriage table — The Measurement Problem (IAM_Measurement_Problem_Quantum)
| paper location | content | book location | verdict | status |
|---|---|---|---|---|
| abstract | measurement = irreversible interaction with Q >= k T ln2; no observer | part5/p5_04_measurement.tex:18 | CARRIED-CORRECTED (MP4) | \interp |
| abstract, §2.1 | 17 chains, Delta chi2 +0.54, sigma8 0.800, H0 72.26 | part5/p5_04_measurement.tex:27 | CARRIED-CORRECTED (MP7: 18 chains) | \measured |
| §1 | unitary vs collapse; measurement undefined; randomness | part5/p5_04_measurement.tex:9 | CARRIED | — |
| §1 l.45-55 | criterion closes gap; resolves every paradox by one mechanism | part5/p5_04_measurement.tex:19 | CARRIED-CORRECTED (MP4; scope stated) | \interp |
| §2.1 | E(a), beta_m, mu<1, Sigma=1 | part5/p5_04_measurement.tex:25 | CARRIED | \measured/\derived |
| §2.2 | Q_L = k T_D ln2; Q<Q_L reversible; Q>=Q_L irreversible; 'S transitions to mu<1 sector' | part5/p5_04_measurement.tex:50 | CARRIED-CORRECTED (MP4 Bennett reading; MP7 mu/Sigma not state labels) | \conjecture |
| Fig. 1 | sector-crossing schematic | part5/p5_04_measurement.tex:34 | CARRIED-CORRECTED (redrawn as 'where a record is written') | \interp |
| Eq. 1 | tau_IAM = hbar k^2 T^2 ln2 / E_G^3, 'no free parameters' | part5/p5_04_measurement.tex:71 | CARRIED-CORRECTED (GD4: capacity k T/E_G assumed) | \conjecture |
| §2.3 | photons E_G = 0, tau = inf; macroscopic below Planck time | part5/p5_04_measurement.tex:73 | CARRIED | \derived |
| §3.1 | double slit; Q ~ 2 eV >> Q_L 0.018 eV; which-path reversible/irreversible; retina 140, CCD 170 | part5/p5_04_measurement.tex:104 | CARRIED (Q/Q_L 112, 140, 167 recomputed B2) | \calc/\interp |
| §3.2 | delayed choice; 5 m, 17 ns; zero accumulated | part5/p5_04_measurement.tex:114 | CARRIED (16.7 ns, B6) | \calc |
| Fig. 2 | delayed-choice timeline | part5/p5_04_measurement.tex:34 | CARRIED (redrawn, panel b) | \calc |
| §3.3 | eraser: reversible -> coherence present; irreversible -> erasure fails | part5/p5_04_measurement.tex:122 | CARRIED-CORRECTED (MP1, MP4: dissipated energy into a record, not carrier energy) | \interp |
| Eq. 2 | F = 1 - exp(1 - k T ln2/Q)/e | part5/p5_04_measurement.tex:141 | CARRIED (form underived, MP5/GD3; identity checked B3) | \conjecture |
| §3.3 l.155-158 | standard QM F = 1; reversible Q/Q_L<0.04 F~1; 'spontaneous emission' irreversible F<0.02 | part5/p5_04_measurement.tex:145 | CARRIED-CORRECTED (MP1: spontaneous emission is erasable; F<0.01 at Q/Q_L>100 for dissipated Q) | \calc |
| Fig. 3 | F vs Q/Q_L | part5/p5_04_measurement.tex:151 | CARRIED (existing fig_eraser) | \conjecture |
| §3.3 l.163-168 | atomic which-path test: erasure fails after emission at 10 ns | part5/p5_04_measurement.tex:130 | EXCLUDED as a prediction (MP1: contradicted by Blinov 2004, Moehring 2007, Hensen 2015); corrected statement carried | \observed |
| Fig. 4 | atomic timing test, Purcell 1-1000 | — | EXCLUDED (MP1) | — |
| §3.4 | cat E_G 7.1e-9 J, tau 3.7e-51 s, 1e7 below Planck time; 'decoheres by own gravity' | part5/p5_04_measurement.tex:179 | CARRIED-CORRECTED (3.5e-51 s, 1.6e7; MP6 environmental decoherence dominates) | \calc/\observed |
| Table 1 | gravitational times electron to human | part5/p5_04_measurement.tex:74 | CARRIED-CORRECTED (radii stated; dust-grain row recomputed with R = 5 um: E_G 1.3e-29 J, tau_PD 7.9e-6 s, tau_IAM 5.3e11 s — printed row used R ~ 0.5 um, inconsistent with 1e-12 kg; NEW finding for errata, see below) | \calc/\conjecture |
| Fig. 5 | tau_IAM vs tau_PD electron to human | part5/p5_04_measurement.tex:75 | CARRIED (redrawn) | \calc |
| §3.5 | Bell: photons Sigma=1, |S| = 2 sqrt2, not hidden variables, E(theta) = -cos theta | part5/p5_04_measurement.tex:188 | CARRIED | \derived |
| Eq. 3 | S(t) = 2 sqrt2 (1 - D), D = E_q/e; threshold 0.293 | part5/p5_04_measurement.tex:197 | CARRIED-CORRECTED (EN8: S_max = 2 sqrt(1+c^2); 0.586 fixed settings; 0.293 isotropic only) | \derived/\calc |
| §3.5 l.219-221 | Micius 1200 km vs matter 1.3 m (2019), ratio 1e6 | part5/p5_04_measurement.tex:202 | CARRIED-CORRECTED (MP2: 1.3 km, Hensen 2015; ratio ~900) | \observed |
| Fig. 6 | Bell correlation; S decay | part5/p5_04_measurement.tex:197 | CARRIED-CORRECTED via ch:entanglement fig:chsh_dephasing (EN8) | \derived |
| Fig. 7 | fidelity vs separation by mass; distance records | — | EXCLUDED (rests on Eq. 3 isotropic form and the 1.3 m record: EN8, MP2) | — |
| §3.6 | Wigner's friend; 1000 photons 2 eV; >1e5 bits; 2^-111604 = 1e-33596 | part5/p5_04_measurement.tex:206 | CARRIED (recomputed: 111,612, 10^-33,599; B10) | \calc/\interp |
| Fig. 8 | Wigner's friend schematic | part5/p5_04_measurement.tex:206 | CARRIED in text (schematic not redrawn; no quantitative content beyond the text) | — |
| §3.7 | Zeno requires irreversible measurement; dispersive readout gives none | part5/p5_04_measurement.tex:216 | CARRIED-CORRECTED (MP3: observed with dispersive readout, Slichter 2016) | \observed/\interp |
| Fig. 9 | Zeno survival vs Q/Q_L | — | EXCLUDED (MP3) | — |
| §4.1 | two-detector eraser (F<0.02 for fluorescence) | part5/p5_04_measurement.tex:246 | EXCLUDED as stated (MP1); falsifier (i) carried in corrected form | \prediction |
| §4.2 | atomic timing test | — | EXCLUDED (MP1) | — |
| §4.3 | temperature test: F 0.000/0.002/0.164 at 10 mK/4 K/300 K for 0.1 eV | part5/p5_04_measurement.tex:168 | CARRIED (0.0024 at 4 K, B4; rests on Eq. 2) | \calc/\prediction |
| Fig. 10 | F vs temperature at 0.1 eV | part5/p5_04_measurement.tex:171 | CARRIED (existing fig_eraser_temperature) | \conjecture |
| §5 | same process over 1e11 galaxies; E_q from 'same integral'; loop closure | part5/p5_04_measurement.tex:224 | CARRIED-CORRECTED (GD3) | \interp/\prediction |
| §6.1 | relation to decoherence theory and collapse models | part5/p5_04_measurement.tex:234 | CARRIED | \interp |
| §6.2 | does not change any performed experiment; 'diverge only for experiments not yet performed' | part5/p5_04_measurement.tex:240 | CARRIED-CORRECTED (MP1, MP3: experiments already performed agree with the corrected reading) | \observed |
| §6.3 | three falsification conditions | part5/p5_04_measurement.tex:246 | CARRIED-CORRECTED (conditions restated for the record-at-absorption reading) | \prediction |
| §7 | conclusion; 17 chains; measurement and acceleration same physics | part5/p5_04_measurement.tex:252 | CARRIED-CORRECTED (MP7) | \interp |
| Acknowledgments, refs | code developers; self-references | — | EXCLUDED (stand-alone book) | — |

## Carriage table — The Two Faces of Time
| paper location | content | book location | verdict | status |
|---|---|---|---|---|
| abstract | three quantities; photons d tau = 0; arrow of time; Hubble tension; shape dynamics | part5/p5_03_time.tex:14 | CARRIED | — |
| §1 | time is two-faced; Barbour 1982 first face; second face irreversibility | part5/p5_03_time.tex:7 | CARRIED | \interp |
| §2 intro | three roles, not terminological | part5/p5_03_time.tex:15 | CARRIED | — |
| §2.1 | coordinate time the map; Barbour's insight | part5/p5_03_time.tex:17 | CARRIED | \derived |
| Eq. 1 | tau = int sqrt(-g dx dx) | part5/p5_03_time.tex:25 | CARRIED (1/c made explicit) | \derived |
| §2.2 | proper time reversible; null d tau = 0 by definition | part5/p5_03_time.tex:28 | CARRIED | \derived |
| §2.3 | accumulated decoherence written 'on the cosmic horizon'; requirements 1-2 | part5/p5_03_time.tex:33 | CARRIED-CORRECTED (TF3: nearest encoding surface, horizon the largest) | — |
| §2.3 l.69-72 | EW T ~ 100 GeV, t ~ 1e-11 s; CP independence; E(z=10) = 4.5e-5 | part5/p5_03_time.tex:41 | CARRIED-CORRECTED (book values T_c 159.5 GeV, t ~ 9e-12 s, ch:electroweak; E(z=10) recomputed C2) | \calc |
| Fig. 1 | coordinate time, proper time, accumulated decoherence | part5/p5_03_time.tex:44 | CARRIED (redrawn) | \derived |
| Eq. 2 | E(a) = exp(1 - 1/a) | part5/p5_03_time.tex:61 | CARRIED | \derived |
| §3.1 bullets | E->0, E(1)=1, E->e, dE/da>0 | part5/p5_03_time.tex:68 | CARRIED (sympy C1) | \derived |
| §3.1 l.141-144 | monotonic from irreversibility; arrow of time thermodynamically necessary | part5/p5_03_time.tex:70 | CARRIED | \interp |
| §3.2, Eq. 3 | photons accumulate no decoherence: E_photon = 0 | part5/p5_03_time.tex:79 | CARRIED | \derived |
| §3.3 | Machian framework; Mach's criterion | part5/p5_03_time.tex:84 | CARRIED (Mach paraphrased, no quotation) | \interp |
| §4 | CMB H0 67.16 geometric; ladder 72.26 matter | part5/p5_03_time.tex:90 | CARRIED | \measured/\calc |
| §4 l.185-187 | two distinct quantities; 'derived from first principles with zero free parameters' | part5/p5_03_time.tex:99 | CARRIED-CORRECTED (TF2) | \prediction |
| Eq. 4 | beta_m = Omega_m/2 = 0.15765 | part5/p5_03_time.tex:103 | CARRIED | \derived |
| §5 | shape dynamics; IAM leaves background untouched | part5/p5_03_time.tex:108 | CARRIED | \interp |
| Eq. 5 | growth equation with mu in Poisson term, text 'friction' | part5/p5_03_time.tex:115 | CARRIED-CORRECTED (TF1: mu on source = Level 1; friction form = Level 2, both stated) | \derived |
| Eq. 6 | mu = H^2/(H^2 + beta_m E) < 1 | part5/p5_03_time.tex:115 | CARRIED-CORRECTED (H0^2 factor made explicit, as Eq. part2:eq:mu) | \derived |
| §5 l.215-218 | arena untouched; complementary questions | part5/p5_03_time.tex:121 | CARRIED | \interp |
| §6 | open question: Higgs vacuum selection as first record | part5/p5_03_time.tex:138 | CARRIED | \openprob |
| §7 | repository | — | EXCLUDED (stand-alone book) | — |
| Acknowledgments | debt to Jacobson, Cai-Kim, Barbour-Bertotti | — | EXCLUDED (stand-alone book; all three cited where used) | — |
| refs | Sakharov 1967 listed, uncited | — | EXCLUDED (TF3) | — |

## Exclusions (each with its correction row)
QD §1 roadmap, Acknowledgements, Data availability (stand-alone book; QD9); QD Step 4 f_coll×η_vir (QD3, V18); QD Eq. 10 D^(5/2) bottom-up (QD1); QD §7.1 'BH formation when the local rate saturates the bound' (not derived; no errata row — author to decide); QD §9.4 Koide (QD7, KO2/KO5) and strong CP (QD7); QD Fig. 4 DESI phantom-crossing redshifts (no source given; not carried pending a citation);
MP Fig. 4 and §4.2 atomic timing test, §4.1 version-B prediction, Fig. 9 Zeno (MP1, MP3); MP Fig. 7 (EN8, MP2); MP acknowledgments; TF §7 repository, acknowledgments, Sakharov reference (TF3).
