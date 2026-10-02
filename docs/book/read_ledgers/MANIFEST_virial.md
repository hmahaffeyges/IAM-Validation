# MANIFEST — virial and gravitational-decoherence chapters (2026-10-02, repo HEAD 41f7646)

## Delivered files
| File | Status | Lines | ~Words |
|---|---|---|---|
| `part1/p1_03_virial_law.tex` | rewritten (was 91 lines) | 176 | 2,433 |
| `part1/p1_04_virial_identity.tex` | **new** | 162 | 2,002 |
| `part2/p2_02_virial.tex` | rewritten (was 131 lines) | 342 | 4,407 |
| `part2/p2_02b_virial_tests.tex` | **new** | 154 | 2,034 |
| `part5/p5_05_gravdec.tex` | rewritten (was 78 lines, 722 words) | 278 | 3,476 |
| `part5/p5_05b_virial_partners.tex` | **new** | 157 | 2,285 |
| `part5/p5_05c_virial_decoherence.tex` | **new** | 195 | 2,984 |
| `bib_virial.bib` | **new** (28 CrossRef-verified articles, 3 books, 2 arXiv preprints) | | |
| `figscripts/fig_virial_book.py` | **new** (7 figures on `_bookstyle.py`) | | |
| `figures/part2/fig_virial_mu_E`, `fig_virial_halo_slope`, `fig_virial_h0_census`; `figures/part5/fig_virial_partition`, `fig_gravdec_scaling`, `fig_gravdec_lindblad`, `fig_gravdec_heating` (.pdf + .png) | **new** | | |
| `../verification/scripts/verify_virial_papers.py` + `_output.txt` | **new** (sympy + numerics, sections A–M) | | |

Chapter words now ≈ 19,600 (sum of the rows above, from `static_check_virial.py`) (was ≈ 3,600 in the three files), against ≈ 22,700 words in the seven papers (references, figure-axis text and
acknowledgments included in the paper count).

## main.tex lines (the lead merges; main.tex not edited)
```
\input{part1/p1_03_virial_law}
\input{part1/p1_04_virial_identity}          % new, after p1_03
...
\input{part2/p2_02_virial}
\input{part2/p2_02b_virial_tests}            % new, after p2_02
...
\input{part5/p5_05_gravdec}
\input{part5/p5_05b_virial_partners}         % new, after p5_05
\input{part5/p5_05c_virial_decoherence}      % new, after p5_05b
...
\bibliography{iam,bib_virial}                % add bib_virial
```

## Static checks (`work/static_check.py`, run on the proposed main.tex)
Braces balanced, environments matched, every `\ref`/`\eqref` resolves against the current tree (including `part:N` labels in main.tex), every `\cite`
in `iam.bib` or `bib_virial.bib`, every figure file exists, every float `[htbp]`, no duplicate label anywhere in the book. Banned-word scan (this book,
the paper, the source, the author, companion, the quantum-processor report/the semiconductor report/the methylation report, superseded, the cell-reading engine, the author's name): clean ("source term" in p2_02 is the physics term).
Labels kept for other chapters: `ch:virial_law`, `tab:virial_domains`, `fig:virial_domains`, `fig:binding_ledger` (moved to p1_04), `ch:virial`,
`part2:eq:mu_virial`, `fig:growth_vs_geometry`, `fig:record_history`, `ch:gravdec`, `eq:tauIAM`, `fig:decoherence_profiles`. The book was not compiled
(no TeX in the sandbox).

## Read ledger (pypdfium2 text of `docs/papers/<file>.pdf`, one `=== PAGE` line per page; read in chunks of ≤ 50 lines, no truncated chunk)
| # | Paper | Lines (ledger) | Read | LaTeX checked |
|---|---|---|---|---|
| 1 | The_Virial_Partition_from_Atoms_to_the_Horizon (not in PAPER_LINE_COUNTS.md) | 194 | 1–194 | `virial_partition_atoms_to_horizon.tex` (143 lines): all 6 equations match |
| 2 | PRL_Version_Thermodynamic_Identity_Governing_Virial_Theorem | 222 (222) | 1–222 | no LaTeX in `docs/papers/latex/` |
| 3 | Virial_Efficiency_and_Effective_Nonlinear_Exponent | 277 (277) | 1–277 | `IAM_Virial_Efficiency_neff_Paper.tex`: Eqs. 1–7 match |
| 4 | Virial_Partitian_Across_Wide_Domains | 641 (641) | 1–641 | `IAM_Virial_37_Orders_Paper.tex`: Eqs. 1–11 match |
| 5 | Dark_Matter_and_Dark_Energy_as_Virial_Partners | 674 (674) | 1–674 | `iam_virial_dark_sector.tex`: Eqs. 1–9 match |
| 6 | Gravitational_Decoherence_and_the_Virial_Partition | 560 (560) | 1–560 | `iam_decoherence_virial_partition.tex`: Eqs. 1–11 match |
| 7 | Gravitational_Decoherence_Quantum_Level | 478 (478; the brief said 456 — the ledger value 478 is what pypdfium2 gives) | 1–478 | `Gravitational_Decoherence.tex`: Eqs. 1–15 match; the LaTeX has two extra equations (master equation with $[X,[X,\rho]]$, $\Gamma\sim GM^2/\hbar R$) not in the PDF, not carried |

Files: P1a = `part1/p1_03_virial_law.tex`, P1b = `part1/p1_04_virial_identity.tex`, P2a = `part2/p2_02_virial.tex`, P2b = `part2/p2_02b_virial_tests.tex`,
P5a = `part5/p5_05_gravdec.tex`, P5b = `part5/p5_05b_virial_partners.tex`, P5c = `part5/p5_05c_virial_decoherence.tex`. Line numbers are of the delivered files.
"VS" = `verify_virial_papers.py` section. Status labels are the preamble macros used at that place.

---
## 1. The Virial Partition from Atoms to the Horizon (Oct 2026, 4 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | one half followed through every scale; thermodynamic content; Smarr; β_m = Ω_m/2, Δχ² = +0.54 | P1a:13–27 | CARRIED | — |
| §1 Eq. 1 | 2⟨K⟩ = k⟨V⟩ via Euler; k = −1: 2⟨K⟩+⟨V⟩ = 0, ⟨K⟩ = ½\|⟨V⟩\|, E = −⟨K⟩ | P1a:29–40 (eqs vl_euler, vl_virial) | CARRIED (sympy VS-A) | \derived |
| §1 | degree fixed by Gauss's law; half independent of N, masses, scale | P1a:39–40 | CARRIED | |
| §2 | hydrogen −13.6057 / 13.6057 / −27.2114 eV; photon = kinetic half | P1a:69–70 | CARRIED (VS-B) | \calc |
| §2 | 20 atoms, 10 molecules at HF limit, T/\|V\| = 1/2; variational wavefunction exact; not independent measurement | P1a:72–83 | CARRIED | \calc |
| §3 | Sun: 2K_th + U = 0; contracting star radiates half; n = 3 polytrope U = −5.69e41 J; \|U\|/2L = 24 Myr; age 4.6 Gyr | P1a:86–89 | CARRIED (23.6 Myr, VS-C) | \calc |
| §3 Eq. 2 | M_Ch = (ω₃⁰√(3π)/2)(ħc/G)^{3/2}/(μ_e m_u)² = 1.456 M⊙, ω₃⁰ = 2.01824 | P1a:91–97 (eq vl_chandra) | CARRIED (VS-C) | \derived |
| §3 | no white dwarf above it; SNe Ia | P1a:96–97 | CARRIED (+ Kepler 2007 1.33 M⊙ already in book) | \observed |
| §4 | Zwicky; cluster virial vs lensing/X-ray tens of percent | P1a:100–102 | CARRIED | \observed |
| §4 | halos 2T/\|U\| ≈ 1.1–1.3 (Bett, Neto, Power 1.15–1.25, Klypin); surface 1.02–1.17; relaxed < 1.35 | P1a:104–109 | CARRIED | \observed |
| §5 steps 1–4, Eq. 3 | first law, Eq. 1, second law, Landauer → ⟨K⟩ = Q = TΔS_min = E_L = ½\|⟨V⟩\| | P1b:47–79 | CARRIED once in full (with PRL) | \derived |
| §5 | step 2 is mechanics; identification; scope (radiating vs collisionless); Jacobson | P1b:81–85, 100–103, 143–148 | CARRIED | \interp |
| §6 Eq. 4 | S = k_Bc³A/4Għ = 4πGk_BM²/ħc; T_H = ħc³/8πGk_BM | P1a:111–115 | CARRIED (V20 form) | \derived |
| §6 Eq. 5 | N k_BT_H ln2 = T_H S = ½Mc² | P1a:117–119 | CARRIED (sympy VS-E) | \derived |
| §6 | 1 M⊙: N = 1.51e77, T_H = 6.17e-8 K; Sgr A*, M87 ratio 0.5000000000 | P1a:120–121 | CARRIED (VS-E) | \calc |
| §6 | Smarr Mc² = 2T_HS; Bardeen–Carter–Hawking; half of rest energy = cost of horizon information | P1a:121–124 | CARRIED | \interp |
| §6 | Kerr Mc² = 2T_HS + 2Ω_HJ; share √(1−χ²)/2: 0.433, 0.218, 0.032 | P1a:124–126 | CARRIED (VS-E) | \calc |
| §7 Eq. 6 | first-law extension; β_m = Ω_m/2 = 0.15765 | P1a:127–133 | CARRIED | \prediction |
| §7 | never sampled; three modified-CAMB chains Δχ² +0.54; σ8 0.809 → 0.800; μ(0) = 0.864, μ0 = −0.135 | P1a:134–138 | CARRIED-CORRECTED (μ0 = −0.136 = 1/(1+β_m) − 1, VS-G; Δχ² sign wording V9) | \measured \prediction |
| Table (p. 4) | eight rows | P1a:146–163 (tab:virial_domains, merged with PRL Table I and Wide Table 1) | CARRIED | |
| §8 "Summary" | empty heading in the PDF | — | nothing to carry | |

## 2. The Thermodynamic Identity Governing the Virial Theorem (PRL version, 18 Mar 2026, 3 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | identity; Q = −E_f; E_L = Q; −E_f = ⟨K⟩; K = Landauer cost; fundamental law proposal | P1b:10–19 | CARRIED; "fundamental law" carried as IAM's Law at the bound-state boundary (framing rule) | \interp |
| Intro Eq. 1 | 2⟨K⟩+⟨V⟩ = 0 | P1b:23–25 | CARRIED | |
| Intro | "atom (1e-10 m) to clusters (1e23 m) — 37 orders" | P1b:25–27; P1a:166–169 | CARRIED-CORRECTED: atom→cluster is 33 decades; 37 decades = 1e-11 m to 1e26 m (36.4 Bohr radius → c/H0, VS-B). **New finding, not in errata** | \calc |
| Intro | "154 years"; Clausius silent on transition; Lynden-Bell & Wood, Padmanabhan; not shown that virial = Landauer | P1b:29–38 | CARRIED-CORRECTED ("more than 150 years": 2026 − 1870 = 156) | |
| Setup | E_i = 0; E_f < 0 | P1b:47–49 | CARRIED | |
| Step 1 Eq. 2 | Q = E_i − E_f = −(⟨K⟩+⟨V⟩) > 0 | P1b:51–57 | CARRIED-CORRECTED (scope added, V13) | \derived |
| Step 2 Eq. 3 | ΔS = Q/T, "T of the bound system" | P1b:59–62 | CARRIED-CORRECTED: ΔS ≥ Q/T in the surroundings at their T (VIRIAL_CHECK "What stands") | \derived |
| Step 3 Eq. 4 | E_L = TΔS = Q; "temperature cancels exactly" | P1b:64–70 | CARRIED (+ note that the step is an identity once the bound is met, VIRIAL_CHECK) | \derived |
| Step 4 Eq. 5 | Euler degree −1 → −E_f = ⟨K⟩; ⟨K⟩ = Q = TΔS = E_L = ½\|⟨V⟩\| | P1b:72–79 | CARRIED (sympy VS-F) | \derived |
| Remark | not a derivation from thermodynamics alone; two expressions; "1/2 is the unique ratio at which both are satisfied" | P1b:81–85 | CARRIED-CORRECTED: the uniqueness sentence replaced by "the degree of the potential fixes it, the identity gives it meaning" (the thermodynamic steps hold for any Q; VIRIAL_CHECK: identifies K, does not derive the theorem) | \interp |
| Formal statement | ⟨K⟩ = Q = TΔS = E_L; heat encoded at k_BT ln2 per bit; iff | P1b:91–98 | CARRIED | \interp |
| Formal statement | "1/2 not a free choice; unique ratio"; scale invariance from Landauer | P1b:96–98, 153–156 | CARRIED-CORRECTED (as above) | |
| Table I row Electron | 6.6 ppm | P1a:155; P1b:122–126 | CARRIED-CORRECTED (EM1: within 0.3 % set by H0; one factor identified numerically) | \conjecture |
| Table I rows Atoms, Molecules | T/\|V\| = 1.0000 | P1a:157–158; P1b:128–131 | CARRIED-CORRECTED: the quantity is −T/E = 2T/\|V\| = 1; T/\|V\| = 1/2 (**new finding**, the Atoms paper prints the corrected form) | \calc |
| Table I row Equipartition | k_BT/2 exact | — | EXCLUDED (V29: quadratic potential, not the 1/r half; stated as such P1a:59–61) | |
| Table I row Solar | K/\|U\| ≈ 0.5, 10 % | P1a:160 | CARRIED (24 Myr) | \calc |
| Table I row Chandrasekhar | 1.44 M⊙ | P1a:159 | CARRIED-CORRECTED (V19: 1.456 M⊙) | \derived |
| Table I row Clusters | ≈ 0.5, 20 % | P1a:161 | CARRIED | \observed |
| Table I row N-body (6) | 2K/\|U\| 0.815 ± 0.025 | P1a:162; P1b:133–138 | CARRIED-CORRECTED (V2, V18: published 2T/\|U\| 1.1–1.3; 1.02–1.17 with surface term) | \observed |
| Table I row β_m | 0.1583 ± 0.0033, 17 chains | P1a:164; P1b:140–145 | CARRIED-CORRECTED (V1: β_m fixed; posterior Ω_m gives β_m/Ω_m = 0.498 ± 0.010) | \measured |
| Text after Table I | electron entry: ħ, H0, m_P, α, no fitted parameters | P1b:122–126 | CARRIED-CORRECTED (EM1) | \conjecture |
| Text | HF η = T/\|V\| = 1.0000000000, SD 0 | P1b:128–131 | CARRIED-CORRECTED (−T/E) | \calc |
| Text | N-body compilation, 25 years, η_vir = 1/(2f_coll) = 0.81, Ω_m f_coll η = 0.159 ± 0.010 agrees 1.0 % | P1b:133–138; P2a:67–81 | CARRIED-CORRECTED (V2, V18, T4: η_vir = 1/(2f_coll) is a definition; product is not a measurement) | |
| Text | 17 MCMC chains return β_m 0.1583 ± 0.0033 at 0.2σ; "sampler has no knowledge" | P1b:140–145 | CARRIED-CORRECTED (V1) | \measured |
| Jacobson section | δQ = TδS on Rindler horizons; extension to bound states | P1b:143–148 | CARRIED | \interp |
| Conclusion | four paragraphs | P1b:150–162 | CARRIED-CORRECTED (as Remark; "fundamental law" → IAM's Law) | \interp |
| Data/ack/email/refs [17]–[20] | repository, Zenodo, own papers | — | EXCLUDED (stand-alone rule: no self-citation) | |

## 3. Virial Efficiency and Effective Nonlinear Exponent (25 Feb 2026, 7 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract, §1 | two quantities flagged for N-body verification; "verification already exists"; six studies 0.815 ± 0.025; n_eff 3.22 ± 0.44; product 0.159 within 1 % | P2a:67–70, 128–132, 145–150 | CARRIED-CORRECTED: the request is carried; "confirmation exists" withdrawn (V2, V18, V31, T2, T4; NBODY_TRACE) | \openprob |
| Abstract "Interpretation of consistency" | studies done independently under ΛCDM | P2a:83–89 (implicitly: published values) | CARRIED in substance | |
| §1 | virial theorem exact for perfect equilibrium; halos not equilibrated; exponent analytic in matter domination; real universe radiation + Λ | P2a:68–69, 142–144 | CARRIED | |
| §2.1 Eq. 1 | β_m = Ω_m f_coll η_vir | P2a:71 (eq vc_decompose) | CARRIED | |
| §2.1 | f_coll = 0.62 ± 0.03 (Tinker 2008); η_vir definition; f_coll η_vir = ½ | P2a:72–75, 123–127 | CARRIED-CORRECTED: f_coll is an integral that depends on M_min (0.486 at 10^10.5, 0.62 needs 10^8.2; NBODY_TRACE) | \calc |
| §2.1 Eq. 2 | η_vir = 1/(2f_coll) = 0.81 | P2a:76–81 (eq vc_eta) | CARRIED-CORRECTED (stated as a definition, T4) | |
| §2.1 | observable: mass-weighted 2K/\|U\| within r_vir | P2a:83–84 | CARRIED | |
| §2.2 Eq. 3 | n_eff = 5/2 analytic | P2a:136–140 (eq vc_n) | CARRIED-CORRECTED (V3, T1: n = 7/2) | \derived |
| §2.2 Eq. 4 | n_eff ≈ 3.0–4.0 numerical | P2a:141–144 | CARRIED-CORRECTED (full ΛCDM integration best for D^{7/2}; bottom-up 7/2 at z ≈ 3–4, mean ≈ 4: ch:quantumrecords) | \calc |
| §2.2 | "Sheth–Tormen β = 1.009 at σ* = 1.2, n_eff ≈ 3.5"; observable = log slope of collapse rate vs D | P2a:145–150 | Observable CARRIED; ST claim EXCLUDED (T2; σ* = 1.2 not in ST99) | |
| §3.1 Table 1 | six studies, mass ranges, z, ⟨η_vir⟩, notes | P2a:89–101 (tab:vc_halo) | CARRIED-CORRECTED: replaced by what each study reports (V18, V31, NBODY_TRACE: Bryan & Norman a different quantity, Ludlow a cut only, Power not GIMIC/OWLS, Klypin MDPL) | \observed |
| §3.2 Eq. 5 | 0.815 ± 0.025 (midpoints 0.85 … 0.81) | — | EXCLUDED (V2, V18: not reported by the sources; tied to 1/(2f_coll)) | |
| §3.2 | spread 0.76–0.90 is physical; massive better virialized | P2a:83–89 (reciprocal 0.77–0.91 mass-dependent) | CARRIED-CORRECTED | \calc |
| §3.3 | three contributions: mergers/infall 10–15 %, rotation 3–5 %, non-thermal 1–3 %; Neto relaxed 0.87 / unrelaxed 0.72 | P2a:110–121 | CARRIED-CORRECTED: the three contributions carried in the author's words; percentages and the 0.87/0.72 values EXCLUDED (V18, NBODY_TRACE: not in sources) | \interp |
| §4.1 Table 2 | six n_eff values (PS+LC 2.5, ST 3.5, Jenkins 3.2, Reed 3.8, Tinker 3.0, Watson 3.3) | P2a:145–150; Fig. vc_halo(b) P2a:102–107 | CARRIED-CORRECTED: replaced by the computed d lnF/d lnD from the same six mass functions (NBODY_TRACE csv); printed values EXCLUDED (T2, NBODY_TRACE: not reported; 3.8 is Jenkins' fit exponent) | \calc |
| §4.2 Eq. 6 | 3.22 ± 0.44; 3.0–3.5 range; EPS lower bound, Reed upper | — | EXCLUDED (same) | |
| §5.1 Eq. 7, Table 3 | β_m^meas = 0.315 × 0.62 × 0.815 = 0.159 ± 0.010; ratio 1.010 ± 0.063 | — | EXCLUDED (T4: product of a definition, not a measurement; recomputed 0.1592 in VS-D) | |
| §5.2 | "not a fit; 155-year-old theorem; three inputs measured; confirmed by six groups" | P2a:128–132 | CARRIED-CORRECTED (first two sentences kept; "confirmed by six groups" withdrawn, T4) | |
| §6 | Gap 1 / Gap 2 "closed" | P2a:128–132, 145–150 | CARRIED-CORRECTED: both gaps stated as the open formal calculation and the bottom-up exponent | \openprob |
| §7 Table 4 | falsification thresholds (η_vir < 0.70 or > 0.95 etc.; "η decreases 5 % to z = 1") | P2b:117–133 | EXCLUDED as printed (V31: thresholds rest on the Table 1 values; "5 % to z = 1" not in Power 2012); the robust criterion carried: f_coll η_vir = ½ is not adjustable | |
| §8 | conclusions repeating §5–6 | P2a:128–132 | CARRIED-CORRECTED as §5.2 | |
| Refs | Bryan & Norman … Watson | bib_virial.bib (8 new keys) + iam.bib | CARRIED (DOI-verified) | |

## 4. The Virial Partition Across Wide Range of Physical Scales (25 Feb 2026, 13 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | β_m = Ω_m/2; single term in matter perturbations; no background change; 1/2 exact for 1/r; atoms, molecules, Chandrasekhar 1.44, clusters 20 %, cosmology "0.3 % via 17 chains" | P2a:19–26 | CARRIED-CORRECTED (V19 1.456; V1 β_m fixed, "0.3 %" withdrawn) | |
| Abstract | σ8 = 0.800 "confirmed at 0.1σ" by 2025 joint σ8 = 0.802 ± 0.020 | P2a:233–235 | CARRIED-CORRECTED (V15 traced in ch:sectortension: 0.802 +0.022/−0.018, Stölzner 2025; 0.12σ) | \observed |
| Abstract | H0_matter = 72.48 | P2a:244–246 | CARRIED (72.26 with the Level 2 H0; 72.48 with Planck) | \derived |
| Abstract | 22 points, χ² 36.16 vs 97.35 | — | EXCLUDED (V8, V25: dominated by sector assignment; Table 4 χ² not reproducible) — statement of why at P2b:48–50 | |
| Abstract | μ0 = 0.864 (μ0 − 1 = −0.136); Euclid DR1 Oct 2026 σ 0.08 | P2a:162–164; P2b:84–94 | CARRIED-CORRECTED (V10, V16; Euclid only as sec:lt_euclid) | \prediction |
| §1 | homogeneity paragraph; scale-independence test; central question | P1a:51–56 | CARRIED | |
| §1 | IAM description; single Hubble-friction term; background identical | P2a:33–36 | CARRIED-CORRECTED ("modified growth", implementation stated, V30) | |
| §1 | 1/2 a theorem from 1e-11 m to 1e26 m; 37 orders | P1a:24–27, 166–169 | CARRIED | |
| §1 | outline (§5 M–σ companion) | — | EXCLUDED (M–σ abandoned; G6) | |
| §2 Eq. 1 | ⟨T⟩ = (n/2)⟨\|V\|⟩ | P1a:42–45 (eq vl_n) | CARRIED (VS-A) | \derived |
| §2 | gravity −1/r and "Coulomb +1/r" | P1a:46–50 | CARRIED-CORRECTED (V29: electron–nucleus −1/r) | |
| §2 | "same 1/2 as Bohr and equipartition kT/2" | P1a:57–63 | CARRIED-CORRECTED (V29: equipartition is the quadratic case) | |
| §2.1 Table 1 | seven domains, scale, force, ratio, precision | P1a:146–163 | CARRIED-CORRECTED (merged; equipartition row EXCLUDED V29; 1.456 V19; cosmological row V1) | |
| §2.1 Fig. 1 | ratio vs log scale, "37 orders" | P1a:141–145 (fig:virial_domains, existing script fig_p1_virial.py) | CARRIED (redrawn earlier with halos and horizons; the equipartition and "17 chains 0.3 %" points not drawn) | |
| §2.2 Eq. 2 | η = −T/E = 1.0000000000 (SD = 0); H, Ne, Xe values; molecules identical; analytic | P1a:72–83 | CARRIED (VS-B) | \calc |
| §2.3 steps 1–3 | atomic, gravitational, informational | P1b:105–117 | CARRIED-CORRECTED (step 1 wording: half of \|U\| is kinetic) | \derived \interp |
| §2.3 step 4 Eq. 3 | β_m = Ω_m f_coll η = Ω_m/2; 0.62 × 0.815 = 0.505 | P1b:114–116; P2a:67–81 | CARRIED-CORRECTED (T4, V2) | |
| §3 Eq. 4 | E(a) = exp(1 − 1/a), from Cai–Kim; properties | P2a:152–160 | CARRIED | \derived |
| §3 Eq. 5 | μ(a) with H0² β_m E(a) | P2a:162–166 | CARRIED | \derived |
| §3 | μ(1) = 0.864; Σ = 1 because Δτ = 0 | P2a:166–171 | CARRIED | \derived |
| §3 Fig. 2 | μ(z), Euclid band ±0.08 | P2a:176–180 (fig:vc_muE) | CARRIED-CORRECTED (band removed: Euclid only as sec:lt_euclid) | \calc |
| §3.1 | σ8 0.800 vs 0.814; ln A_s 0.06σ, Ω_m 0.04σ | P2a:210–216 | CARRIED-CORRECTED (Level 2 shifts −1.51σ σ8, −0.78 S8, −0.07 ω_b, +0.09 ln A_s, +0.05 Ω_m) | \measured |
| §3.2 Eq. 6 | H0_matter = 67.36√1.15765 = 72.48 | P2a:242–246 | CARRIED | \derived |
| §3.2 | photon probes CMB; matter probes Cepheids, masers, SBF | P2a:246–252 | CARRIED-CORRECTED (worldline rule, V7) | |
| §3.3 | three-channel β split 0.076618 / 0.078825 / 0.002207 | P5b:95–106 | CARRIED as conjecture (V12 author; sum and β/2 checked VS-G) | \conjecture \openprob |
| §4.1 Table 2 | Runs A, B, C: μ0, σ8, Δχ² | P2a:217–232 (tab:vc_chains) | CARRIED-CORRECTED (chain CSV values; Run B EXCLUDED: posterior at the prior edge, VIRIAL_CHECK #17; Δχ² sign V9) | \measured |
| §4.1 | "statistically indifferent"; σ(μ0) 0.04 Euclid final; current Planck 0.22; Run B 0.9σ | P2a:230–232 | CARRIED-CORRECTED (V26: σ(μ0) = 0.125 Planck+RSD from ch:latetime; Euclid per sec:lt_euclid) | |
| §4.2 Table 3 | σ8 ΛCDM 0.8139 (0.45σ), IAM 0.8000 (0.09σ), joint 0.802 ± 0.020; 0.35σ closer | P2a:233–236; P2b:22–35 | CARRIED-CORRECTED (V26: recomputed 0.12σ / 0.29σ with the traced asymmetric error) | \calc |
| §4.2 | KiDS-1000 0.766; KiDS-Legacy 0.815 "converges to Σ = 1" | P2a:236–239 | CARRIED-CORRECTED (V27) | |
| §4.3 Table 4, Fig. 3 | six H0 probes, sectors, σ distances; χ² 5.57 vs 55.20 | P2a:253–281 (tab:vc_h0, fig:vc_h0) | CARRIED-CORRECTED (V7: H0LiCOW → photon ruler; recomputed χ² 13.0, single-H0 37.2, at 67.36 51.2, VS-J; V25) — **H0LiCOW 3.4σ from the photon rate: open** | \observed \calc \openprob |
| §4.3 | √(1+β_m) − 1 ≈ 7.5 % | P2a:251–252 | CARRIED-CORRECTED (7.6 %, VS-J) | \calc |
| §4.4 Fig. 4, Table 5 | fσ8 IAM vs ΛCDM against 10 RSD points; χ² 5.94 vs 6.04; "~3 % below z ≈ 0.5" | P2b:36–39 | CARRIED-CORRECTED: the 10-point list is not given and is not reproduced; carried with the DESI and SDSS χ² of ch:sectortension; figure not redrawn (data list absent) | \calc |
| §4.5 Table 6 | 26-probe census, 14 matter (12 of 14 anomalies), 12 photon | P2b:40–47 | CARRIED-CORRECTED: classification and the four-tension statement carried; counts EXCLUDED until redone by the worldline rule (V7, VIRIAL_CHECK #7) | \interp \openprob |
| §4.6, Table 7 | Δχ² +61.2; +49.6 H0; +11.46 S8 (5 pts); +0.10 fσ8 | P2b:48–50 | EXCLUDED (V8, V25) with the reason printed | |
| §5 Eq. 7–9, Table 8 | M–σ: E_Landauer = (ln2/2)Mc², σ⁴, f_geom = Ω_m/2π | — | EXCLUDED (V17: ln 2 counted twice, ½Mc² carried in P1a §vl_bh; M–σ line abandoned, BOOK_READING_TODO; VIRIAL_CHECK #18) | |
| §6.1 Eqs. 10–11 | E_G definition; E_G^IAM = Ω_m/f_IAM; > GR; "published all exceed GR" | P2b:52–68 | CARRIED-CORRECTED ("all exceed" withdrawn, ledger 10(f): Reyes 0.39 ± 0.06 vs GR 0.41; shift +3.6/+1.8/+1.1 % VS-H) | \derived \prediction |
| §6.2 | R1, R2, R3; R1, R3 suppressed by μ; R2 cancels; catalogues overlap | P2b:70–82 | CARRIED-CORRECTED (V28: Level 1 form only; Level 2 no suppression) | \prediction \openprob |
| §6.3 | Euclid VIS/NISP; μ/Σ = 0.864/1.000; 1.5e9 galaxies, 15,000 deg²; "not predicted by other models" | P2b:84–94 | CARRIED-CORRECTED (T13: sDGP μ<1, Σ=1 with a ghost) | \prediction |
| §6.3 Table 9 | DESI+CMB ±0.22 0.85σ; DR1 ±0.08 1.7σ; DR2 ±0.05 2.7σ; final ±0.04 3.4σ | P2b:89–94 | CARRIED-CORRECTED: replaced by sec:lt_euclid (0.3σ to ~7σ for the IAM μ(z); forecast open) per the Euclid rule; V10, V16, V26 | \openprob |
| §7 items 1–5 | falsification | P2b:117–129 | CARRIED-CORRECTED (item 5 limited to Level 1, V28; item 1 precision wording per sec:lt_euclid) | |
| §7 "What does not falsify" | η = 0.81, f_coll = 0.62 representative; product ≈ ½ required; 37 orders cannot reverse | P2b:130–133 | CARRIED-CORRECTED (no number assigned to η, f_coll) | |
| §8 | conclusions; interpretation paragraph; three comparisons; predictions on record | P2b:144–154; P5b:153–157 | CARRIED-CORRECTED ("kT/2 per decoherence event" → k_BT ln2 per bit, V29; M–σ clause EXCLUDED; Δχ² +61.2 EXCLUDED; Euclid date V16) | \interp \prediction |
| Data, ack, software | repository, collaborations, CAMB/MGCAMB/Cobaya | — | EXCLUDED (acknowledgments) | |

## 5. Dark Matter and Dark Energy as Virial Partners (Mar 2026, 12 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | DM = potential half, DE = kinetic half; R → 2 at a = 1; coincidence a consequence; the question | P5b:11–21 | CARRIED | \conjecture \interp |
| §1 | 95 %, 26 %, 69 %; ratio 0.38 near ½; a⁻³ vs Λ; anthropic; coincidence problem | P5b:23–33 | CARRIED (0.38 VS-G) | \calc |
| §1 | β_m from virial theorem; 1e-11 to 1e26 m, 37 orders; "0.3 % from 17 chains" | P5b:35–39 | CARRIED-CORRECTED (V1) | |
| §1 | question: "why is DM/DE today close to the virial ratio of 2?" | P5b:41–43 | CARRIED-CORRECTED: the ratio that is 2 is matter over the record term, R(1); DM/DE = 0.38 (**new finding**: internal inconsistency) | |
| §2 Eqs. 1–2 | ⟨T⟩ = (n/2)⟨\|V\|⟩; n = 1; Euler; theorem not fit | P1a:42–50, 62–65 | CARRIED once (Part 1) | \derived |
| §2 | two channels (geometric, informational); kinetic half drives decoherence | P2a:40–47 | CARRIED | \interp |
| §2 Eq. 3 | β_m = Ω_m/2 = 0.1577; not fitted; "posterior returns 0.1583 ± 0.0033 at 0.2σ, 17 chains" | P2a:48–53, 62–65 | CARRIED-CORRECTED (V1) | \prediction |
| §3 | potential half in curvature; kinetic half on the horizon; DM not a particle; Λ vacuum baseline; balance sheet | P5b:44–63 | CARRIED | \conjecture |
| §3.1 Eq. 4 | R(a) = Ω_m a⁻³/(β_m E) | P2a:310–313; P5b:65–68 | CARRIED | \calc |
| §3.1 Eq. 5 | R(1) = 2 | P2a:315–318; P5b:69–73 | CARRIED (stated as by construction, VIRIAL_CHECK #4) | \calc |
| §3.1 Table 1 | E(a) 0.050 / 0.182 / 0.274; R 399 / 20 / 6 / 2 | P2a:319–331 (tab:vc_R) | CARRIED-CORRECTED (V4: E = 0.135 / 0.497 / 0.741; R correct) | \calc |
| §3.1 | "today first cosmological virialisation" | P5b:69–73 | CARRIED with the definition stated | \interp |
| §3.2 | coincidence dissolved; E(1) = 1 from GH temperature; anthropic and physics same moment | P5b:75–81 | CARRIED | \interp |
| §4.1 Eq. 6 | Ω_m^growth = Ω_m μ; μ without H0² | P2a:288–293 | CARRIED-CORRECTED (V21: H0² restored) | \prediction |
| §4.1 | 13.6 % at z = 0, 5.2 % at z = 0.5 | P2a:293–295 | CARRIED | \calc |
| §4.1 | DESI DR1 FS+BAO 0.2962 ± 0.0095; Planck 0.3153 ± 0.0073; 6.1 % below; "consistent with 5.2 %" | P2a:297–303 | CARRIED-CORRECTED (V14: 0.2990 at z = 0.5, 0.3σ; V23: not a growth-only test) | \observed \calc |
| §4.1 | IAM-native χ² 3.618 vs 3.569 (six DESI bins) | — | EXCLUDED (V23: the comparison misidentifies the FS+BAO value; the per-bin fσ8 χ² 5.24 vs 4.51 from ch:sectortension carried at P2b:36–39) | |
| §4.1 Fig. 1 | Ω_m growth curve vs per-tracer DESI points; residuals; "0.02σ"; LRG1 0.255 ± 0.015, 14 %, "hardware systematic" | P2a:285–290 (fig:growth_vs_geometry, curve only); P2b:111–115 | CARRIED-CORRECTED (V14; per-tracer Ω_m points and LRG1 numbers not traced — not drawn; LRG1 carried with traced sources) | \calc \openprob |
| §4.1 | caveat: ΛCDM template; IAM-native template most important near-term analysis | P2a:300–303 | CARRIED | \openprob |
| §4.2 Eqs. 7–8 | w_info = −4/3; z_t 0.632 vs 0.718; H(z) compilations 0.698–0.740; Riess 2004 0.46 photon sector | P2a:339–342 | CARRIED-CORRECTED (V6: background unmodified gives 0.632, VS-G; IAM z_t needs a matter-sector q(z) — open; IAM value 0.718 and the compilation comparison EXCLUDED by V6; Riess 2004 "photon sector" EXCLUDED, V7 SNe on matter ruler) | \calc \openprob |
| §4.3 | DESI DR2 w0 > −1, wa < 0, z_cross 0.33–0.43; coincides with 7–8 % suppression; single-fluid artefact | P2b:103–110 | CARRIED-CORRECTED (V24: distance-only preference; two-ruler test; z_cross 0.35–0.50 from DR2 rows, verify_sector_tension.py; "7–8 % suppression" is 1−μ, V22) | \observed \openprob |
| §4.4 Table 2 | σ8, H0 m, H0 γ, Ω_m growth, z_t, z_cross, μ0 | P2b:16–35 (tab:vt_status) | CARRIED-CORRECTED (V15 traced, V14, V23, V6, V24) | |
| §5, Fig. 2 (9 panels) | R(a); energy budget; info share 23/17/10/5 %; q(z); chronometers (32 pts); sector gap; Ω_m discrepancy 13.6/7.8/3.4 %; rate of kinetic growth 6.4/2.9/0.6 %; key numbers z_eq 0.361 vs 0.295 | P5b:83–93 (fig:virial_partition, six panels) | CARRIED-CORRECTED: shares 18.7/14.6/10.3/4.9 % (V5); rate 7.3/2.9/0.6 % at z = 0.3/0.7/1.5 (recomputed E a²/6, VS-G; printed 6.4 % at z = 0.3 not reproduced); q(z) panel EXCLUDED (V6); chronometer panel not redrawn (32-point list not in the repo; open) | \calc |
| §5 | transition zone window | P2a:333–337; P5b:88–93 | CARRIED | |
| §6 | question the framework forces; "does not dilute"; spacetime made of deposits | P5b:108–130 | CARRIED; "does not dilute" carried with the note that in R(a) the half dilutes as a⁻³ (ledger 11(g)) | \conjecture \openprob |
| §7 | Euclid DR1 Oct 2026 σ 0.04; μ0 = −0.136, Σ0 = 0; "unique to IAM" | P2b:134–142 | CARRIED-CORRECTED (V10, V16, sec:lt_euclid; T13 uniqueness) | \interp |
| §7 Eq. 9 | fσ8 ramp; "7.9 % at z = 0.295 … 0.6 % at z = 1.491" | P2a:201–208 (eq vc_fs8); P2b:96–101 | CARRIED-CORRECTED (V22: those are 1−μ; fσ8 deficit 2.19 % … 0.13 %, VS-H) | \prediction |
| §7 LRG1 | 2.4σ; fibre incompleteness ~35 %; AP factor; removal restores ΛCDM; 5.2 % vs 19 % | P2b:111–115 | CARRIED-CORRECTED: qualitative statement and sources (Liu 2024, Ó Colgáin 2025) carried; 2.4σ, 35 %, 19 % not traced — left out pending trace (ledger 11(f)) | \observed \openprob |
| §7 Summary bullets | four outcomes | P2b:139–142 | CARRIED (LRG1 bullet in the LRG1 paragraph) | |
| §8 | conclusions | P5b:132–151 | CARRIED-CORRECTED ("does not dilute" sentence: see §6; Euclid date V16) | \conjecture \interp |
| Data, ack, refs | | bib (DESI, Planck, Cai–Kim etc. in iam.bib) | ack EXCLUDED | |

## 6. Gravitational Decoherence, the Virial Partition, and the Emergence of Classical Structure (Mar 2026, 12 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | two questions, one mechanism; BH vault; σ_crit ≈ 4 km/s; E(a) from electroweak to e; 17 chains Δχ² +0.54; Euclid DR1 Oct 2026 σ 0.04, 3.4σ | P5c:10–24 | CARRIED-CORRECTED (σ_crit → tested, rejected in ch:satellites; "beginning at the electroweak transition" → negligible before structure; V9, V10, V16) | \conjecture \interp |
| §1 | two faces of time; separately treated; IAM common origin; kinetic half Landauer cost; arrow of time; persistence; outline | P5c:26–45 | CARRIED | \conjecture |
| §2.1 Eq. 1 | 2K = \|V\|, K = ½\|V\| | P5c:47–50 | CARRIED (ref eq vl_virial) | \derived |
| §2.1 | potential half curvature; kinetic half decoherence energy; k_BT ln2 per bit | P5c:52–56; P2a:55–60 | CARRIED | \interp |
| §2.1 Eq. 2 | β_m = Ω_m/2 = 0.1575; posterior 0.1583 ± 0.0033 at 0.2σ | P5c:56–58 | CARRIED-CORRECTED (V1) | \prediction |
| §2.2 | verified 1e-10 m to 1e26 m, 37 orders; universality key point | P5c:60–64 | CARRIED | \interp |
| Fig. 1 (a) partition schematic; (b) vault and σ_crit; (c) E(a); (d) 1−μ and Euclid 3.4σ | | (c),(d): P2a:176–180 fig:vc_muE; (a),(b) schematics not redrawn (text in P5c:93–131) | CARRIED-CORRECTED ((d) as 1−μ with the fσ8 deficit; Euclid label per sec:lt_euclid) | \calc |
| §3.1 | coordinate, proper, accumulated decoherence; T_H = ħH/2πk_B; electroweak beginning; CP violation | P5c:66–76 | CARRIED (once in full in ch:time; this text keeps the distinct statements; T_H = 2.65e-30 K VS-K) | \interp |
| §3.2 Eq. 3 | E(a); derived not postulated; properties E(z=10) = 4.5e-5 | P5c:78–81; P2a:155–160 | CARRIED | \calc |
| §3.3 | arrow of time not statistical; ledger only accumulates | P5c:83–86 | CARRIED | \interp |
| §3.3 Eq. 4 | μ without H0²; Σ = 1; Bertschinger & Zukin | P5c:88–91 (ref eq mu_virial) | CARRIED-CORRECTED (V21) | |
| §4.1 | closure on both halves; cosmic horizon cold 2.66e-30 K; t_dyn ≪ 1/H needs local surface | P5c:94–101 | CARRIED (premise flagged not derived, ch:satellites) | \conjecture \openprob |
| §4.1 Eq. 5 | S_info/A ≥ 1/(4ℓ_P²) | P5c:103–108 (eq vd_sat) | CARRIED (units: one nat per 4ℓ_P²) | \conjecture |
| §4.1 Eq. 6 | Γ_BH = c³/(1920 GM ln2) bits/s | P5c:109–115 (eq vd_gamma) | CARRIED (sympy VS-E: P_H/(k T_H ln2)) | \derived |
| §4.1 | evaporation ~1e67 yr; permanent vault | P5c:114–115 | CARRIED (2.1e67 yr VS-E) | \calc |
| §4.2 Eq. 7 | t_dyn = 1/√(Gρ) ≈ √(π/6) GM/σ³ | P5c:117–121 | CARRIED-CORRECTED: with σ² = GM/R the prefactor is √(4π/3) (sympy VS-K) — **new finding** | \derived |
| §4.2 Eqs. 8–9 | M_min = σ³/GH with prefactor 4Ω_m; M_min = 4Ω_m σ³/(GH) | P5c:121–124 | CARRIED-CORRECTED (prefactor not derived, V30, ch:satellites eq ms_mmin) | |
| §4.3 Eq. 10 | σ_crit ≈ 4 km/s at 10^8.4 M⊙ (Kim & Peter 2021) | P5c:124–131 | CARRIED-CORRECTED (3.9 km/s, VS-K; citation EXCLUDED, V11; mechanism rejected by the census in ch:satellites; V30) | \calc \observed |
| §4.3 | σ³ distinct from σ² and σ⁴; ~100× offset; "never written into classical existence" | P5c:124–128; P5c:186–190 | CARRIED (offset: normalisation open, item 1 of §vd_open) | \conjecture |
| §5.1 Eq. 11 | growth equation "Hubble friction" with μ·G form; σ8 0.7998 ± 0.0058 consistent with KiDS, DES, HSC at 0.1σ | P2a:161–175, 210–239 | CARRIED-CORRECTED (V30 implementation stated; ledger 12(c): S8 comparisons) | \measured |
| §5.2 | H0 γ 67.16 ± 0.47 (0.37σ); H0 m 72.26 (0.75σ); two distinct quantities | P2a:240–282 | CARRIED | \derived \interp |
| §5.3 | 17 chains R−1 < 0.01; "best-fit improvement +0.54"; Euclid DR1 σ 0.04, 3.4σ | P5a:30–36; P2b:84–94 | CARRIED-CORRECTED (18 chains; V9; V10, V16) | \measured |
| §6.1 | time as accumulated cost; Smolin temporal naturalism; evolving laws not claimed; E(a) not imposed | P5c:133–145 | CARRIED | \interp \openprob |
| §6.2 | England dissipation-driven adaptation; closure condition as gravitational instance; "not an analogy"; σ_crit falsifiable | P5c:147–162 | CARRIED-CORRECTED (σ_crit consequence rejected, ch:satellites) | \conjecture \openprob |
| §6.3 | Rovelli thermal time; E(a) carrier, beginning, limit, consequences | P5c:164–177 | CARRIED-CORRECTED (beginning: formation of structure after the electroweak transition; "σ8 = 0.800 falsifiable by Euclid DR1" → testable by Euclid and DESI) | \interp \openprob |
| §6.4 | Smolin cosmological natural selection; complementary BH role | P5c:179–184 | CARRIED | \conjecture |
| §6.5 | normalisation of M_min; cusp–core same problem; Mechanism B; "13.6 % growth suppression derived and confirmed" | P5c:186–195 | CARRIED-CORRECTED (V22: 13.6 % is 1−μ; fσ8 4.25 %; not yet confirmed) | \openprob \prediction |
| Acknowledgments | named correspondents | — | EXCLUDED (rule: no names of private correspondents; errata N1) | |
| Refs | Barbour & Bertotti, Bertschinger & Zukin, Cai & Kim, Connes & Rovelli, England ×2, Gibbons & Hawking, Jacobson, Kim & Peter, Landauer, Rovelli ×3, Sakharov, Smolin ×3 | iam.bib + bib_virial.bib | CARRIED (Kim & Peter EXCLUDED, V11; Rovelli 2022 printed as Entropy 24, 1394 — CrossRef gives 24, 1022, doi 10.3390/e24081022: **new citation correction**) | |

## 7. Gravitational Decoherence from Dual-Sector Thermodynamics: Optomechanics and Quantum Computing (Feb 2026, 22 pp)
| Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|
| Abstract | semiclassical open-system framing; no collapse; environment; goals | P5a:12–18 | CARRIED | |
| §1.1 Eq. 1 | horizon chain; surface density ∝ 1/a²; E(a) | P5a:21–28 | CARRIED | \derived |
| §1.1 Eq. 2 | μ < 1, Σ = 1; β = 0.1575, μ0 = −0.135 | P5a:26–28 | CARRIED-CORRECTED (H0², μ0 = −0.136) | |
| §1.2 Eq. 3 | 17 chains (12 L1, 5 L2); R−1 < 0.01; Δχ² +0.54; σ8 0.800; H0 m 72.26 | P5a:30–36 | CARRIED-CORRECTED (GD5: 18 chains — 12 L1 + baryon test, 3 L2, 2 L2b; CSV) | \measured |
| §1.2 | σ8 0.8087 → 0.7998 consistent with KiDS-1000, DES Y3, HSC Y3; both endpoints within 1σ | P5a:33–36 | CARRIED-CORRECTED (S8 per survey in ch:sectortension) | |
| §1.3 | Level 2b: Runs A, D, samples 125,440/120,832, acceptance 46.8/46.3 %, R−1 0.0096/0.0068; H0 ≈ 61.5; structural confirmation | P5a:38–44 | CARRIED-CORRECTED (CSV: R−1 0.0100/0.0068, H0 61.45 ± 0.42 / 61.52 ± 0.43; sample counts and acceptance not in the CSV — left out; "wrong direction" → far from every measurement) | \measured |
| §1.4 | sector split; boundary at the superposition | P5a:46–48 | CARRIED | \conjecture |
| §2.1 Eq. 4 | τ_PD = ħ/E_G, E_G ≈ Gm²/R | P5a:50–58 | CARRIED | |
| §2.1 Eq. 5 | dN/dt = E_G²/(ħ k_BT ln2) | P5a:60–66 | CARRIED (postulated, GD4) | \conjecture |
| §2.2 | S_boundary = k_BT/E_G | P5a:65–67 | CARRIED (underived, GD4) | \conjecture |
| §2.2 Eq. 6 | ln E_q = ∫Γ/S dt = E_G³ t/(ħ k_B²T² ln2) | P5a:69–74 | CARRIED-CORRECTED (GD3: constant integrand → exponential; sympy VS-L) | \derived |
| §2.2 Eq. 7 | τ_IAM = ħ k_B²T² ln2 / E_G³ | P5a:71 (eq:tauIAM) | CARRIED | |
| §2.2 Eq. 8 | E_q(η) = exp(1 − 1/η) | P5a:76–82 | CARRIED-CORRECTED (assumption by analogy, GD3) | \conjecture |
| §2.2 Eqs. 9–10 | C_IAM = 1 − E_q/e; C_PD = exp(−t/τ_PD) | P5a:77–80 | CARRIED | |
| §2.2 Fig. 1 | profiles for 1e-12 kg at 10 mK, "τ_IAM ≈ 560 µs" | P5a:83–89 (fig:decoherence_profiles, existing fig_p5.py) | CARRIED-CORRECTED (GD1; book density 2200: 509 s, τ_PD 7.5 µs) | \calc |
| Fig. 2 | rate profiles, zero at t = 0, peak η ≈ 0.5 | P5a:93–96 (fig:gd_scaling b) | CARRIED (peak η = 1/2 sympy VS-L) | \conjecture \calc |
| §3.1 Fig. 3 | τ vs mass at five temperatures, ρ = 2000; "mesoscopic frontier 1e-15–1e-10 kg, ~100 µs" | P5a:93–104 (fig:gd_scaling a) | CARRIED-CORRECTED (ρ = 2200; frontier restated: τ < 1 s above 3.5e-12 kg, < 1 ms above 1.4e-11 kg at 10 mK; GD1) | \prediction \calc |
| §3.2 Fig. 4 | τ ∝ T²; ×4, ×16; PD no change | P5a:105–108; fig:decoherence_profiles b | CARRIED | \calc |
| §3.3 Fig. 5 | τ ∝ m⁻⁶ vs m^{−5/3}; crossover 2.3e-10 kg; Δα 4.33; 3–4 masses > 5σ | P5a:109–114 | CARRIED-CORRECTED (GD2: m⁻⁵, Δα 3.33; crossover 2.2e-10 kg at ρ 2200; σ(α) 0.058 VS-L) | \calc |
| §3.4 Eq. 11 | P_IAM = E_G³/(ħ k_BT); dn/dt = E_G³/(ħ² k_BT ω0); P ∝ 1/T | P5a:115–129 (eq gd_heating, fig:gd_heating) | CARRIED-CORRECTED: identified as k_BT ln2/τ_IAM (one bit per τ); the bit rate of Eq. 5 would give E_G²/ħ — **new finding, open** | \calc \openprob |
| §4.1 Eq. 12 | Γ(η) = η⁻² e^{1−1/η} | P5a:131–135 | CARRIED (follows from the ramp only) | \conjecture |
| §4.1 Eq. 13 | Lindblad with L ∝ x̂; "energy (phonon populations) conserved" | P5a:136–141 | CARRIED-CORRECTED: x̂ dephasing heats, d⟨n⟩/dη = Γ/2 (VS-M) — **new physics correction** | \derived |
| §4.2 Fig. 6 | purity, entropy, coherence, rate for \|α = 2⟩, RK4; rate peak "η ≈ 0.23" | P5a:143–154 (fig:gd_lindblad, recomputed) | CARRIED-CORRECTED (GD5: 0.23 is the purity-difference peak; rate peaks at 0.5) | \calc |
| §4.3 Fig. 7 | density-matrix snapshots at 0, 0.5τ, τ, 2τ, 5τ with purities | P5a:149–153 (purities at 0.5, 1, 2, 5 τ in text) | CARRIED-CORRECTED (snapshot matrices not redrawn; purities recomputed) | \calc |
| §4.4 Fig. 8 | ⟨n⟩ "decay"; dS/dη; ΔP peak η 0.23, 0.14 | P5a:155–158 | CARRIED-CORRECTED (ΔP 0.139 at η 0.225 reproduced; ⟨n⟩ rises) | \calc |
| §5.1 Eqs. 14–15, Fig. 9 | N = 50, σ_C 2 %; 3σ and 5σ at m > 3.7e-15 kg; > 100σ above 1e-12; 5σ above 1e-13 at 10 % | P5a:160–165 | CARRIED-CORRECTED: the design carried; thresholds EXCLUDED (GD1: rest on τ = 560 µs; with τ = 509 s the profile is measurable only near and above 1e-12 kg) | \calc |
| §5.2 Fig. 10 | temperature test 5σ with 20 %; mass test 1e-13–1e-10 kg | P5a:166–169 | CARRIED-CORRECTED (range 1e-12–1e-11 kg at 10 mK; 9.8σ at 20 %, VS-L) | \calc |
| §5.3 Fig. 11 | dn/dt vs mass, four temperatures, ω0 2π × 100 kHz; 1 phonon/s; above threshold at picograms | P5a:170–173; fig:gd_heating | CARRIED (2.8 /s at 1e-12 kg; 1 /s at 0.8e-12 kg, VS-L) | \prediction |
| §5.4 Fig. 12 | timeline 10×/3–5 yr from 1e-19 kg; 3σ 2028–35, 5σ 2030–37; converges with Euclid | P5a:174–179 | CARRIED-CORRECTED (GD1: recomputed 23–38 years to 3.5e-12 kg; heating the nearer test; figure not redrawn) | \calc |
| §6 Table 1 | IAM, PD, CSL m⁻¹, graviton emission T⁵, semiclassical T² m⁻² 1/T | P5a:180–200 (tab:gd_compare) | CARRIED-CORRECTED (IAM m⁻⁵, GD2; PD row carried; CSL, graviton-emission and semiclassical rows left out: no source given or found — listed as open) | \calc \openprob |
| §7.1 | irrelevant for qubits; masses; isolation vs reversibility paradigm; Moon | P5a:219–228 | CARRIED | \interp |
| §7.2 | five principles | P5a:230–241 | CARRIED (principles 1 and 4 as conjecture) | \interp \conjecture |
| §8 items 1–5 | falsification; 1e13–1e14 amu; α = 6; T²; exponential; τ exceeded | P5a:255–268 | CARRIED-CORRECTED (item 1: at 1.7e-14–1.7e-13 kg τ_IAM = 4e6–4e11 s, so the range is restated as m ≳ 3.5e-12 kg; α = 5; profile item limited to the ramp) | |
| §8 | quantum falsification does not falsify cosmology | P5a:266–268 | CARRIED | |
| §9 | same process over 1e11 galaxies, 13.8 Gyr; rate per event vs cumulative; same equation at every scale; close the loop | P5a:243–253 | CARRIED-CORRECTED (quantum E_q as conjecture; "cosmic acceleration" → growth of structure) | \conjecture |
| §10 | conclusion; four signatures; ΔP at 0.23; 5σ above 3.7e-15 kg; early 2030s; "The missing piece was always information." | P5a:270–278 | CARRIED-CORRECTED (three signatures + ramp; thresholds and dates EXCLUDED, GD1) | \prediction \conjecture |
| Ack, data, refs [1]–[2] | software, repository, own papers | — | EXCLUDED | |
| Refs [7]–[13] | Bérut, Ciampini, Rossi, Delić, Neumeier, Kaltenbaek, arXiv:2512.02838 | bib_virial.bib (5 new) + iam.bib | CARRIED; arXiv:2512.02838 not cited (not checked) | |

---
## Exclusions for the author (confirmed corrections only)
1. Six-study η_vir = 0.815 ± 0.025, Table 1 values and Neto 0.87/0.72 — V2, V18, V31 (Virial Efficiency; PRL Table I; Wide §2.3).
2. n_eff = 3.22 ± 0.44 and Table 2 values; ST "σ* = 1.2, β = 1.009" — T2, NBODY_TRACE.
3. β_m^meas = 0.159 ± 0.010, ratio 1.010 ± 0.063, Table 3 — T4.
4. Falsification thresholds of Virial Efficiency Table 4 and "η −5 % to z = 1" — V31.
5. Equipartition row (k_BT/2) — V29.
6. Wide Domains Run B (μ0 free 0.006 ± 0.156, Δχ² −1.90) — VIRIAL_CHECK #17 (prior edge).
7. Combined χ² tables (Wide Table 7, abstract 36.16/97.35, Table 4 χ² 5.57/55.20) — V8, V25; recomputed values printed instead.
8. Sector-census counts (14/12, 12 of 14) — V7 (census to be redone by the worldline rule).
9. M–σ section (Wide §5, Eqs. 7–9, Table 8) — V17, M–σ line abandoned.
10. Wide Table 9 / Euclid σ-ladder (DR1 1.7σ … final 3.4σ) and every "Euclid DR1 October 2026" — V10, V16, Euclid rule (sec:lt_euclid).
11. Virial Partners z_t(IAM) = 0.718, w_info = −4/3 and the H(z)-compilation comparison; Riess 2004 photon-sector — V6, V7.
12. Virial Partners IAM-native χ² 3.618/3.569 — V23.
13. Kim & Peter 2021 citation — V11.
14. Quantum-level detection thresholds (3.7e-15 kg), the 2028–2037 timeline, "mesoscopic frontier ~100 µs" — GD1 (restated with τ = 509 s).
15. Acknowledgments of all papers (named correspondents) and self-citations — rules.

## New findings on this read (not yet in PAPER_ERRATA.md; proposed rows)
| Paper | Where | Printed | Correct | Evidence |
|---|---|---|---|---|
| PRL | intro, Table I caption, conclusion | atom (1e-10 m) to clusters (1e23 m) = 37 orders | 33 decades atom→cluster; 37 decades is 1e-11 m → 1e26 m | VS-B |
| PRL | Table I atoms/molecules; text | T/\|V\| = 1.0000 | −T/E = 2T/\|V\| = 1 (T/\|V\| = ½) | VS-B; Wide Eq. 2 |
| PRL | Remark, formal statement, conclusion | "1/2 is the unique ratio at which mechanical and thermodynamic equilibrium are both satisfied" | steps 1–3 hold for any Q; the degree of the potential fixes ½ | VS-F; VIRIAL_CHECK "What stands" |
| Virial Partners | §1 | "why is DM/DE today close to the virial ratio of 2?" | DM/DE = 0.38; the ratio equal to 2 is R(1) = Ω_m/β_m | VS-G |
| Virial Partners | Fig. 2 panel (i) | 6.4 % (transition zone) | E a²/6 = 7.3 % at z = 0.3 | VS-G |
| Grav. Decoherence (virial) | §4.2 Eq. 7 | t_dyn ≈ √(π/6) GM/σ³ | √(4π/3) GM/σ³ for σ² = GM/R | VS-K (sympy) |
| Grav. Decoherence (virial) | refs | Rovelli, Entropy 24, 1394 (2022) | Entropy 24, 1022, doi 10.3390/e24081022 | CrossRef |
| Grav. Decoherence (quantum) | §4.1, Fig. 8 | x̂ dephasing conserves phonon number; ⟨n⟩ "decays" | ⟨n⟩ grows at Γ/2 per unit η | VS-M |
| Grav. Decoherence (quantum) | Eq. 11 vs Eq. 5 | P_IAM = E_G³/(ħk_BT) as Landauer cost per event | equals k_BT ln2/τ_IAM (one bit per τ); Eq. 5 × k_BT ln2 = E_G²/ħ — which is meant is open | VS-L |
| Grav. Decoherence (quantum) | §8 item 1 | falsified if nothing at 1e13–1e14 amu | τ_IAM = 4e6–4e11 s there (10 mK): no falsification reach | VS-L |
| Wide Domains | Fig. 3 label | SBF (Khetan+ 2021) 73.3 ± 2.5 | 73.3 ± 2.5 is Blakeslee et al. 2021 | CrossRef, value |

## Left undone / for the author
- Sector census of all 26 probes to be redone by the worldline rule (counts not printed).
- The lensing time-delay H0 (73.3 ± 1.8) sits 3.4σ above the photon-sector rate under the worldline rule: an open item of the dual-sector picture.
- LRG1 numbers (2.4σ, ~35 % fibre incompleteness, Ω_m 0.255 ± 0.015) not traced; carried qualitatively.
- Per-tracer DESI Ω_m points (Partners Fig. 1), the 10 RSD points (Wide Fig. 4) and the 32 cosmic-chronometer points (Partners Fig. 2e) are not listed in the papers or found in the repository; those figure panels are not redrawn.
- Quantum Table 1 rows for CSL, graviton emission and semiclassical gravity need sources before they can be printed.
- Schematic panels (Grav. Decoherence virial Fig. 1a–b; quantum Fig. 7 density matrices; Fig. 12 timeline) not redrawn.
- Three-channel split (V12) awaits the author's ruling; carried as \conjecture in P5b.
- The book was not compiled (no TeX); static checks only.
