# MANIFEST — p2_03_theory.tex: line-for-line derivation carry (2026-10-02)

Owner: docs/book/part2/p2_03_theory.tex. Repo cloned at HEAD 41f7646 (descendant of c39b251; sparse checkout of docs/book, docs/verification, CANON, docs/papers/latex/iam_theory_paper).
Nothing was committed or pushed.

## 1. Reading ledger
| file | lines | read | note |
|---|---|---|---|
| docs/papers/IAM_Theory_Paper.pdf (pypdfium2 text, one `=== PAGE n ===` marker per page) | **1916** (1882 text + 34 markers; matches PAPER_LINE_COUNTS.md) | 1–1916, 39 chunks of ≤ 50 lines, none truncated | version of record |
| docs/papers/latex/iam_theory_paper/iam_theory_paper.tex | 2417 | targeted: Eq. 83 (l.1607–1616), Eq. 84 (l.1625–1632) for exact equation form | 18 Mar LaTeX; used only for equation forms |
| docs/book/part2/p2_03_theory.tex (41f7646) | 319 | 1–319, chunks of ≤ 60 | kept: summary framing, §source, observed values, figures fig_exponent/fig_w_eff |
| docs/verification/theory/THEORY_CHECK.md | 59 | full | items 1–15 applied |
| docs/verification/theory/EXPONENT_LINE_BY_LINE.md | 32 | full | n = 7/2 |
| docs/verification/theory/BOTTOM_UP_EXPONENT.md | 31 | full | n_eff running, 7/2 at z ≈ 3–4 |
| docs/verification/theory/ENTROPIC_GRAVITY_NOTE_CHECK.md | 17 | full | §source kept; f(R)/DGP corrections |
| docs/verification/PAPER_ERRATA.md rows T1–T18 (l.12–30) and N1 | — | rows in full | all applied |
| docs/verification/scripts/verify_theory_paper_output.txt, verify_euclid_template_output.txt | 22, 5 | full | |
| part5/p5_01_interpretation.tex | 59 | full | shorter version of source §13 |
| prior C3 delivery (v229932f5_C3_derivations_delivery.zip): MANIFEST.md rows for p2_03 (l.56–82, 226–290, 446–480, 560–601) | 601 | the p2_03 rows | used: Cai–Kim r_A³ [D4], Jacobson f factor [D3], minisuperspace lapse note (A17), divergence of the integral from a = 0 (D15). app_C3 not in the tree, so no app:der refs |

## 2. Equation and step table
Paper location = PDF text line numbers. Book location = line in the delivered p2_03_theory.tex. Verdicts: CARRIED (identical in content, author's words), CARRIED-CORRECTED (source of correction cited), EXCLUDED (confirmed correction shows it wrong, or acknowledgments). "NEW" = correction found in this check, not yet an errata row (see §3).

| # | paper line | equation / step | book location | verdict | correction source / check | status label |
|---|---|---|---|---|---|---|
| A1 | 9-19 | Abstract: steps (i)-(iv); µ(0)=0.864, µ0=-0.136 from β=Ωm/2 | part2/p2_03_theory.tex:16 | CARRIED |  | \derived (in body) |
| A2 | 19-25 | Abstract: action S_info, φ=1-1/a, w_info=-1-1/(3a) | part2/p2_03_theory.tex:24 | CARRIED |  | \derived |
| A3 | 25-27 | Abstract: 'βm verified against N-body-calibrated halo mass functions to 0.3%' | — | EXCLUDED | T11 (β_m fixed; identity via η_vir definition) |  |
| A4 | 27-28 | Abstract: 'ST recovers the 1/a exponent to within 1%' | — | EXCLUDED | T2/T18, THEORY_CHECK #1; recomputed ST β=0.89 at σ*=1.2 (verify_theory_derivations.py §12) |  |
| A5 | 28-32 | Abstract: perturbations standard GR on IAM rate; µ=E²ΛCDM/E²IAM, Σ=1, zero parameters | part2/p2_03_theory.tex:27 | CARRIED | T18 (matter-sector rate) | \derived |
| A6 | 33-35 | Abstract: companion paper Δχ²=+1.34 (Level 1) | part2/p2_03_theory.tex:28 | CARRIED-CORRECTED | T16 (final extraction); stand-alone rule (no companion paper) | \measured (body) |
| A7 | 36-39 | Abstract: Level 2 Δχ²=+0.54; no new fields | part2/p2_03_theory.tex:29 | CARRIED |  | \measured (body) |
| S1.1 | 41-50 | §1 arrow of time, decoherence as origin of irreversibility | part2/p2_03_theory.tex:39 | CARRIED |  |  |
| S1.2 | 51-54 | §1 Jacobson / Cai–Kim generality | part2/p2_03_theory.tex:48 | CARRIED |  |  |
| S1.3 | 55-57 | §1 the identification; µ0=-0.136; tested in companion paper | part2/p2_03_theory.tex:52 | CARRIED-CORRECTED | stand-alone rule (chapters cited) |  |
| S1.4 | 58-65 | §1 organisation of sections | part2/p2_03_theory.tex:56 | CARRIED-CORRECTED | added §sec:source and §sec:th:interp pointer |  |
| S2.1 | 70-85 | §2.1 mechanism; R_H=c/H; holographic growth | part2/p2_03_theory.tex:69 | CARRIED |  |  |
| S2.2 | 88-96 | §2.2 two sources i_vac, i_struct → Λ and βE(a) | part2/p2_03_theory.tex:84 | CARRIED |  | \conjecture |
| S2.3 | 97-105 | §2.3 photons exempt; µ<1, Σ=1 | part2/p2_03_theory.tex:93 | CARRIED |  |  |
| S2.4 | 106-110 | §2.4 feedback; E(a)→e | part2/p2_03_theory.tex:99 | CARRIED |  | \interp |
| P1 | 115-118 | Eq.1 \|ψS⟩\|E0⟩ → Σ ci\|si⟩\|Ei⟩ | part2/p2_03_theory.tex:108 | CARRIED |  |  |
| P1a | 120-122 | inline: ρS off-diagonals ∝ ⟨Ej\|Ei⟩ (step made explicit) | part2/p2_03_theory.tex:110 | CARRIED | verify_theory_derivations.py §1 (toy 2-state) | \derived |
| P2 | 123-129 | Eq.2 ρS → Σ\|ci\|²\|si⟩⟨si\| | part2/p2_03_theory.tex:113 | CARRIED | verify_theory_derivations.py §1 | \derived |
| P3 | 137 | Eq.3 I = log2 N | part2/p2_03_theory.tex:120 | CARRIED | verify_theory_derivations.py §1 |  |
| P4 | 140 | Eq.4 ΔE = kT ln2; experiments Bérut, Jun | part2/p2_03_theory.tex:122 | CARRIED |  | \observed |
| P5 | 159-163 | Eq.5 Unruh T = ħκ/2πkB | part2/p2_03_theory.tex:141 | CARRIED | verify_theory_derivations.py §3 |  |
| P5a | 156-158 | inline: χ^a = -κλk^a | part2/p2_03_theory.tex:137 | CARRIED |  |  |
| P6 | 164-178 | Eq.6 δQ = ∫Tab χ^a dΣ^b = -κ∫λ Tab k^a k^b dλ dA | part2/p2_03_theory.tex:143 | CARRIED |  |  |
| P7 | 180-184 | Eq.7 δS = η δA = η∫θ dλ dA | part2/p2_03_theory.tex:147 | CARRIED |  |  |
| P8 | 188-200 | Eq.8 Raychaudhuri | part2/p2_03_theory.tex:151 | CARRIED |  |  |
| P8a | 201-204 | inline: θ ≈ -λ Rab k^a k^b | part2/p2_03_theory.tex:153 | CARRIED |  |  |
| P9 | 205-212 | Eq.9 δA = -∫λ Rab kk dλ dA | part2/p2_03_theory.tex:154 | CARRIED |  |  |
| P10 | 213-232 | Eq.10 Clausius relation | part2/p2_03_theory.tex:156 | CARRIED | verify_theory_derivations.py §2 | \derived |
| P10a | 233 | inline: κ cancels | part2/p2_03_theory.tex:157 | CARRIED | verify_theory_derivations.py §2 |  |
| P11 | 236-239 | Eq.11 Tab = ħη/2π Rab + f gab | part2/p2_03_theory.tex:159 | CARRIED-CORRECTED | NEW: f g_ab carries the factor ħη/2π (prior C3 audit [D3]; verify_theory_derivations.py §2) |  |
| P11a | 240 | inline: f = -R/2 + Λ from ∇aTab=0 + Bianchi | part2/p2_03_theory.tex:160 | CARRIED | verify_theory_derivations.py §2 |  |
| P12 | 241-248 | Eq.12 Einstein eq.; G = 1/(4ħη) | part2/p2_03_theory.tex:162 | CARRIED-CORRECTED | T9 (G = c³/4ħη) | \derived |
| P12a | 250-251 | inline: 'machine' remark | part2/p2_03_theory.tex:165 | CARRIED |  |  |
| S4.2A | 256-267 | §4.2(A) S∝A from causal accessibility (3 steps) | part2/p2_03_theory.tex:171 | CARRIED |  | \interp |
| P13 | 279-293 | Eq.13 Euclidean Rindler period 2π/κ | part2/p2_03_theory.tex:190 | CARRIED | verify_theory_derivations.py §3 (cone circumference) | \derived |
| P13a | 294-302 | inline: 8π = 2×4π (Gauss, trace) | part2/p2_03_theory.tex:195 | CARRIED |  |  |
| P14 | 303-313 | Eq.14 ħη/2π = c⁴/8πG | part2/p2_03_theory.tex:203 | CARRIED-CORRECTED | T9 (c³) |  |
| P15 | 316-334 | Eq.15 η = c⁴/(4ħG) = 1/4ℓP², 1/4 = 2π/8π | part2/p2_03_theory.tex:206 | CARRIED-CORRECTED | T9 (η = c³/(4ħG)); verify_theory_derivations.py §2 numeric 9.570e68 m⁻² | \derived |
| P16 | 338-364 | Eq.16 δAmin via κ=c²/ℓP; one bit per 4ℓP² | part2/p2_03_theory.tex:213 | CARRIED-CORRECTED | T9 (δAmin = 4ħG/c³ for any κ; one nat per 4ℓP², bit 4ln2 ℓP²); verify_theory_derivations.py §4 | \derived |
| P17 | 372-375 | Eq.17 r̃A = 1/H | part2/p2_03_theory.tex:221 | CARRIED |  |  |
| P17a | 378 | inline: AH = 4π/H² | part2/p2_03_theory.tex:222 | CARRIED |  |  |
| P18 | 380-386 | Eq.18 Sgeo = AH/4G = π/GH² | part2/p2_03_theory.tex:223 | CARRIED | verify_theory_derivations.py §5 |  |
| P19 | 387-390 | Eq.19 TH = H/2π | part2/p2_03_theory.tex:224 | CARRIED |  |  |
| P20 | 391-400 | Eq.20 Misner–Sharp E = r̃A/2G = 4πρ/3H³ | part2/p2_03_theory.tex:226 | CARRIED | verify_theory_derivations.py §5 |  |
| P21 | 401-405 | Eq.21 dSgeo = -2π/(GH³) dH | part2/p2_03_theory.tex:229 | CARRIED | verify_theory_derivations.py §5 |  |
| P22 | 406-411 | Eq.22 -dE = 4π r̃A² (ρ+P) H dt | part2/p2_03_theory.tex:232 | CARRIED-CORRECTED | NEW: Cai & Kim flux is 4π r̃A³(ρ+P)H dt; with r̃A² Eq.24 does not follow (prior C3 audit [D4]; verify_theory_derivations.py §5) |  |
| P23 | 412-421 | Eq.23 first law substituted | part2/p2_03_theory.tex:234 | CARRIED-CORRECTED | follows P22 correction |  |
| P24 | 423 | Eq.24 Ḣ = -4πG(ρ+P) | part2/p2_03_theory.tex:236 | CARRIED | verify_theory_derivations.py §5 | \derived |
| P25 | 424-430 | Eq.25 H² = 8πGρ/3, Λ integration constant | part2/p2_03_theory.tex:239 | CARRIED | verify_theory_derivations.py §5 (step 2HḢ written out) | \derived |
| P26 | 431-437 | Eq.26 ΔE ≥ kB TH ln2 = (H/2π) ln2 | part2/p2_03_theory.tex:244 | CARRIED | verify_theory_derivations.py §5 (today 2.5e-53 J, added) | \calc |
| S4.4 | 438-443 | §4.4 Stotal = Sgeo + Sinfo (key observation) | part2/p2_03_theory.tex:250 | CARRIED |  |  |
| S5.1 | 447-453 | §5.1 structure formation as decoherence; H⁻¹ | part2/p2_03_theory.tex:273 | CARRIED |  |  |
| P27 | 455-459 | Eq.27 Stotal = Sgeo + Sinfo | part2/p2_03_theory.tex:281 | CARRIED |  | \conjecture |
| P28 | 461-469 | Eq.28 İ ∝ ρm D^n f H | part2/p2_03_theory.tex:288 | CARRIED |  | \conjecture |
| P28a | 470-474 | inline: δc≈1.686; 'Press–Schechter gives n≈2.5–4' | part2/p2_03_theory.tex:290 | CARRIED-CORRECTED | T18 (PS n≈2.5–4 not derived; removed) |  |
| P29 | 475-484 | Eq.29 dSinfo/dt = İ/TH · 1/AH | part2/p2_03_theory.tex:296 | CARRIED |  | \conjecture |
| P29a | 485-491 | inline: 1/TH = 2π/H Landauer; 1/AH surface density | part2/p2_03_theory.tex:297 | CARRIED |  |  |
| P30 | 497 | Eq.30 2K + U = 0 | part2/p2_03_theory.tex:306 | CARRIED |  |  |
| P31 | 502-506 | Eq.31 β = Ωm/2; 0.1575 (Ωm 0.315) | part2/p2_03_theory.tex:310 | CARRIED-CORRECTED | canon Ωm 0.3153 → 0.15765 (CANON/iam_canon.json); same arithmetic | \derived, \calc |
| P32 | 510-512 | Eq.32 H ∝ a^-3/2 | part2/p2_03_theory.tex:318 | CARRIED |  |  |
| P33 | 513-517 | Eq.33 AH ∝ a³ | part2/p2_03_theory.tex:319 | CARRIED |  |  |
| P34 | 518-522 | Eq.34 TH ∝ a^-3/2 | part2/p2_03_theory.tex:320 | CARRIED |  |  |
| P35 | 523 | Eq.35 D ∝ a; f≈1, Ωm(a)≈1 | part2/p2_03_theory.tex:321 | CARRIED |  |  |
| P36 | 525-532 | Eq.36 dS/dln a ∝ ρm D^n f/(TH AH) | part2/p2_03_theory.tex:325 | CARRIED | EXPONENT_LINE_BY_LINE (H cancels) |  |
| P37 | 533-546 | Eq.37 = a^(n-9/2) | part2/p2_03_theory.tex:327 | CARRIED | verify_theory_derivations.py §6 |  |
| P38 | 547-551 | Eq.38 dS/da ∝ a^(n-11/2) | part2/p2_03_theory.tex:329 | CARRIED | verify_theory_derivations.py §6 |  |
| P39 | 552-562 | Eq.39 S ∝ a^(n-9/2)/(n-9/2) | part2/p2_03_theory.tex:331 | CARRIED-CORRECTED | integral written indefinite (∫ from 0 diverges for n≤9/2; verify_theory_derivations.py §12) | \derived |
| P40 | 566-570 | Eq.40 S ∝ -1/a + const | part2/p2_03_theory.tex:341 | CARRIED |  |  |
| P41 | 571-580 | Eq.41 n - 9/2 = -1 ⇒ n = 5/2 | part2/p2_03_theory.tex:343 | CARRIED-CORRECTED | T1 (n = 7/2); EXPONENT_LINE_BY_LINE, BOTTOM_UP_EXPONENT; verify_theory_derivations.py §6 | \derived |
| P41a | 581-585 | inline: 'full ΛCDM shifts to neff≈3–4 (Section 10), consistent with N-body measurements' | part2/p2_03_theory.tex:348 | CARRIED-CORRECTED | T2/T4 (N-body n_eff tables removed); numeric matter-era −1.02, Λ era −1.57 (verify_theory_derivations.py §6) | \calc |
| F-exp | — | Figure: exponent slope test (book addition, existing) | part2/p2_03_theory.tex:337 | CARRIED | verify_theory_paper.py / verify_theory_derivations.py §6 | \calc |
| P42 | 587-591 | Eq.42 Sinfo ∝ -1/a + C (with n=5/2) | part2/p2_03_theory.tex:353 | CARRIED-CORRECTED | T1 (with n = 7/2) |  |
| P43 | 592-596 | Eq.43 -dEinfo = TH dSinfo | part2/p2_03_theory.tex:356 | CARRIED |  |  |
| P43a | 597 | inline: φ̇ = H/a ⇒ dS/dt ∝ H/a | part2/p2_03_theory.tex:358 | CARRIED | verify_theory_derivations.py §9 (φ̇=H/a) |  |
| P44 | 598-601 | Eq.44 ρ̇info = ρinfo H/a | part2/p2_03_theory.tex:359 | CARRIED |  | \conjecture |
| P45 | 602-619 | Eq.45 ρinfo ∝ exp(∫H/a da/(aH)) = exp(-1/a) | part2/p2_03_theory.tex:362 | CARRIED-CORRECTED | printed 'da/H' → da/(aH) (dt = da/aH); result unchanged; verify_theory_derivations.py §7 |  |
| P46 | 620-626 | Eq.46 ΔH² ∝ exp(C - 1/a) | part2/p2_03_theory.tex:364 | CARRIED |  |  |
| P47 | 631-632 | Eq.47 E(1)=exp(C-1)=1 ⇒ C=1 | part2/p2_03_theory.tex:370 | CARRIED | verify_theory_derivations.py §7 |  |
| P48 | 634-640 | Eq.48 E(a) = exp(1 - 1/a) | part2/p2_03_theory.tex:372 | CARRIED | verify_theory_derivations.py §7 | \derived |
| P49 | 643-663 | Eq.49 D/AH ∝ a/a³ = 1/a²; ∫ → -1/a | part2/p2_03_theory.tex:377 | CARRIED | verify_theory_derivations.py §7 | \derived, \interp |
| P50 | 664-670 | Eq.50 E(z) = exp(-z) = e·e^-(1+z) | part2/p2_03_theory.tex:384 | CARRIED | verify_theory_derivations.py §7 (values E(z=10,2,1)) | \derived, \calc |
| S6.6 | 671-677 | §6.6 beyond matter domination; 'coefficients within 5–10%' | part2/p2_03_theory.tex:388 | CARRIED-CORRECTED | T17 → recomputed: D^{7/2} gives exp(0.93−1.02/a) (verify_theory_derivations.py §12) | \calc |
| P51 | 678-685 | Eq.51 -dE = T dSgeo + T dSinfo | part2/p2_03_theory.tex:398 | CARRIED |  |  |
| P51a | 686-687 | inline: E→0 as a→0, E(1)=1, E≈0 at recombination | part2/p2_03_theory.tex:401 | CARRIED | verify_theory_derivations.py §7 |  |
| P51b | 688-692 | inline: background application gives H0≈61.5 | part2/p2_03_theory.tex:404 | CARRIED-CORRECTED | values from the Level 2b chains (p2_04 l.88) | \measured |
| S7.1 | 697-709 | §7.1 Weyl-tensor argument; open problem | part2/p2_03_theory.tex:411 | CARRIED |  | \openprob |
| P52 | 710-711 | Eq.52 -dE = TH d(Sgeo + Sinfo) | part2/p2_03_theory.tex:423 | CARRIED |  |  |
| P53 | 714-722 | Eq.53 H² = 8πGρ/3 + Λ/3 + βE H0² | part2/p2_03_theory.tex:425 | CARRIED |  |  |
| P54 | 727-730 | Eq.54 ρinfo = 3H0²/(8πG) βE(a) | part2/p2_03_theory.tex:428 | CARRIED |  |  |
| P55 | 731-764 | Eq.55 substitution; 8πG/3 cancels | part2/p2_03_theory.tex:432 | CARRIED | verify_theory_derivations.py §8 | \derived |
| P55a | 767-773 | inline: Eq.53 formal output; perturbation-level friction 2H_IAM δ̇ | part2/p2_03_theory.tex:436 | CARRIED-CORRECTED | T18 (matter-sector rate) |  |
| D1 | 779-784 | Definition 1 decoherence eligibility (dτ>0) | part2/p2_03_theory.tex:448 | CARRIED |  | \conjecture |
| D1a | 785-789 | inline: massive vs null worldlines | part2/p2_03_theory.tex:455 | CARRIED |  |  |
| Pr1 | 790-794 | Proposition 1 dual-sector coupling | part2/p2_03_theory.tex:462 | CARRIED |  | \derived (given Def.) |
| Pr1a | 795-801 | inline: Einstein eqs universal; diffeomorphism invariance | part2/p2_03_theory.tex:469 | CARRIED |  | \interp |
| P56 | 804-807 | Eq.56 k²Φ = -4πGµa²ρδ | part2/p2_03_theory.tex:479 | CARRIED |  |  |
| P57 | 808-812 | Eq.57 k²(Φ+Ψ) = -8πGΣa²ρδ | part2/p2_03_theory.tex:480 | CARRIED |  |  |
| P58 | 815-819 | Eq.58 µ = H²ΛCDM/(H²ΛCDM + βE) | part2/p2_03_theory.tex:484 | CARRIED-CORRECTED | T10 (βE(a)H0²) |  |
| P59 | 820-824 | Eq.59 µ0 = -β/(1+β) = -0.136 | part2/p2_03_theory.tex:486 | CARRIED | verify_theory_derivations.py §8 (−0.1362) | \derived, \calc |
| P60 | 827-831 | Eq.60 Σ = 1 (exact); central prediction | part2/p2_03_theory.tex:488 | CARRIED |  | \derived, \prediction |
| P61 | 836-840 | Eq.61 φ ≡ ln E = 1 - 1/a | part2/p2_03_theory.tex:513 | CARRIED |  |  |
| P62 | 841-848 | Eq.62 φ̇ = H/a (constraint) | part2/p2_03_theory.tex:515 | CARRIED | verify_theory_derivations.py §9 | \derived |
| P63 | 850-853 | Eq.63 S = SEH + SΛ + Smatter + Sinfo | part2/p2_03_theory.tex:522 | CARRIED |  |  |
| P64 | 854-869 | Eq.64 Sinfo action | part2/p2_03_theory.tex:524 | CARRIED |  | \conjecture |
| P64a | — | added: FRW minisuperspace Lagrangian with lapse (constraint term lapse-free) | part2/p2_03_theory.tex:531 | CARRIED-CORRECTED | NEW step (prior C3 audit A17; verify_theory_derivations.py §9) |  |
| P-λ | 872-873 | Variation wrt λ → φ̇ = H/a | part2/p2_03_theory.tex:534 | CARRIED | verify_theory_derivations.py §9 |  |
| P-φ | 874-878 | Variation wrt φ → λ̇ = +(3H0²/8πG)βe^φ | part2/p2_03_theory.tex:537 | CARRIED-CORRECTED | NEW: d(a³λ)/dt = −a³ρinfo, i.e. λ̇+3Hλ = −ρinfo (sign and 3Hλ; verify_theory_derivations.py §9) | \derived |
| P65 | 881-892 | Eq.65 variation wrt g → H² = 8πG(ρm+ρr)/3 + Λ/3 + βe^φ H0² | part2/p2_03_theory.tex:544 | CARRIED | verify_theory_derivations.py §9 (lapse variation) | \derived |
| P66 | 894-900 | Eq.66 ρ̇info = ρinfo H/a | part2/p2_03_theory.tex:551 | CARRIED |  |  |
| P67 | 901-907 | Eq.67 winfo = -1 - 1/(3a); w(1) = -4/3 | part2/p2_03_theory.tex:553 | CARRIED | verify_theory_derivations.py §9 | \derived |
| P67a | 908-912 | inline: phantom origin; NEC | part2/p2_03_theory.tex:557 | CARRIED |  |  |
| P68 | 913-922 | Eq.68 w_eff^DE = (-ρΛ + w ρinfo)/(ρΛ+ρinfo) | part2/p2_03_theory.tex:567 | CARRIED |  |  |
| P68a | 923-934 | inline: w_eff(1)≈-1.06; w0≈-1.07, wa≈+0.04 | part2/p2_03_theory.tex:574 | CARRIED-CORRECTED | T3 (w0 = −1.062, wa = −Ωm²/[3(2−Ωm)²] = −0.012); verify_theory_derivations.py §10 | \derived, \calc |
| F-weff | — | Figure: w_info, w_eff and CPL forms (existing) | part2/p2_03_theory.tex:507 | CARRIED | verify_theory_paper.py §3 | \calc |
| S8.6 | 935-946 | §8.6 nature of the informational field | part2/p2_03_theory.tex:579 | CARRIED |  | \interp |
| P69 | 949-954 | Eq.69 ⟨T⟩ = -½⟨V⟩ | part2/p2_03_theory.tex:593 | CARRIED |  |  |
| P69a | 955-958 | inline: equal share between Sgeo and Sinfo | part2/p2_03_theory.tex:596 | CARRIED |  | \conjecture |
| P70 | 959-964 | Eq.70 ρinfo(1) = ρm/2 = Ωm ρcrit/2 | part2/p2_03_theory.tex:598 | CARRIED |  |  |
| P71 | 965-970 | Eq.71 βm = Ωm/2 = 0.1575 | part2/p2_03_theory.tex:600 | CARRIED-CORRECTED | canon 0.15765 (Ωm 0.3153) | \derived, \calc |
| P72 | 971-984 | Eq.72 fcoll ST 0.593, Tinker 0.646, 0.62±0.03 | part2/p2_03_theory.tex:608 | CARRIED-CORRECTED | NEW: recomputed 0.64 / 0.71 (EH no-wiggle, M>10⁶ M⊙; verify_theory_derivations.py §11); errata row proposed | \calc |
| P72a | 985-988 | inline: Ωm fcoll = 0.195 overshoots 'MCMC value' by 24%; virial 'matches to 0.3%' | part2/p2_03_theory.tex:610 | CARRIED-CORRECTED | T11 (no MCMC fit of βm; 0.3% removed); recomputed 0.20–0.22, +27–41% | \calc |
| P73 | 989-1000 | Eq.73 βm = Ωm fcoll ηvir; ηvir = 0.81 | part2/p2_03_theory.tex:614 | CARRIED-CORRECTED | T4 (ηvir ≡ 1/(2fcoll), a definition; 0.7–0.8 with recomputed fcoll) | \derived |
| P73a | 1001-1006 | inline: '0.158 vs 0.1575, a 0.3% difference ... empirical confirmation' | — | EXCLUDED | T11 (identity, not a confirmation) |  |
| T1 | 1013-1026 | Table 1: six-study ηvirial = 0.815±0.025 | — | EXCLUDED | T4 (cited studies report 2T/\|U\| ≥ 1: Neto 2007, Power 2012; carried as \observed instead) |  |
| P74 | 1027-1032 | Eq.74 βm^meas = 0.159±0.010 | — | EXCLUDED | T4 |  |
| S9.3b | 1033-1045 | §9.3 '19% deviation ... mergers 10–15% ...'; independence of cross-check | — | EXCLUDED | T4 (rests on Table 1) |  |
| S9.3c | — | published 2T/\|U\| (Neto 2007, Power 2012) (book addition, existing) | part2/p2_03_theory.tex:619 | CARRIED | THEORY_CHECK #3 | \observed |
| S9.4 | 1046-1049 | §9.4 'fitted βm should shift with Ωm priors' | part2/p2_03_theory.tex:625 | CARRIED-CORRECTED | T12 (βm fixed; test via free µ0 and growth data) | \prediction |
| S9.5 | 1050-1059 | §9.5 parameter count; '1/a confirmed to 1% by ST' | part2/p2_03_theory.tex:630 | CARRIED-CORRECTED | T2/T18 (ST 1% removed; 1.02 from D^{7/2} integration) |  |
| S9.6 | 1060-1069 | §9.6 independence from microscopic details | part2/p2_03_theory.tex:643 | CARRIED |  |  |
| S10 | 1070-1073 | §10 background Ωm 0.315, Ωr 9.1e-5, H0 67.4 | part2/p2_03_theory.tex:655 | CARRIED |  |  |
| P75 | 1076-1092 | Eq.75 I(a) = ∫ R/(TH AH) da' | part2/p2_03_theory.tex:662 | CARRIED-CORRECTED | T17 (per dt); R defined per Hubble time; divergence of the literal Eq.28 rate shown (verify_theory_derivations.py §12) |  |
| T2 | 1093-1113 | Table 2: α, β, r for D², D^{5/2}, D^{7/2}, D⁴, PS | part2/p2_03_theory.tex:675 | CARRIED-CORRECTED | NEW: recomputed with stated definition (verify_theory_derivations.py §12; printed values not reproduced); errata row proposed | \calc |
| T2a | 1110-1116 | inline: best D^{7/2} exp(0.95-1.05/a); n=5/2 76–87%; neff≈3.5 reasonable | part2/p2_03_theory.tex:698 | CARRIED-CORRECTED | recomputed exp(0.93−1.02/a); 'n=5/2 attributable to matter domination' removed (T1) | \calc, \interp |
| S10.3 | 1117-1121 | §10.3 α=1 near 3.5, β=1 near 3.3 | part2/p2_03_theory.tex:706 | CARRIED-CORRECTED | recomputed: α=1 at 3.77, β=1 at 3.42 | \calc |
| P76 | 1122-1142 | Eq.76 ST multiplicity f_ST(ν), A,q,p | part2/p2_03_theory.tex:716 | CARRIED |  |  |
| P77 | 1143-1157 | Eq.77 σ*=1.2: I ∝ exp(0.925 - 1.009/a), r=0.992 | part2/p2_03_theory.tex:687 | CARRIED-CORRECTED | NEW: not reproduced; recomputed 0.75/0.89 at σ*=1.2 (β=1 between σ* 1.0 and 1.2) (verify_theory_derivations.py §12); THEORY_CHECK #1 | \calc |
| S10.4b | 1158-1162 | §10.4 caveat: ST calibrated on ΛCDM; virial is primary | part2/p2_03_theory.tex:724 | CARRIED |  |  |
| T3 | 1163-1188 | §10.5 Table 3 neff (Jenkins, Reed, Tinker, Watson) 3.33±0.33; PS n≈2.5 | — | EXCLUDED | T4/T18 (NBODY_TRACE: not collapse-rate slopes; PS n not derived) |  |
| S10.5b | 1189-1192 | 15-test suite publicly available (tests/iam_derivation_tests.py) | part2/p2_03_theory.tex:729 | CARRIED-CORRECTED | file absent at HEAD; replaced by verify_theory_derivations.py |  |
| S11.1 | 1194-1200 | §11.1 perturbative status of δφ (first order) | part2/p2_03_theory.tex:734 | CARRIED |  |  |
| P78 | 1201-1213 | Eq.78 δφ = 0 'exactly, at all orders in linear PT'; formal δφ̇=0 | part2/p2_03_theory.tex:744 | CARRIED-CORRECTED | wording: 'in linear perturbation theory' (first order, per the source's own §11.1 caveat) | \derived |
| P79 | 1222-1225 | Eq.79 Poisson k²Ψ = -4πGa²ρδ | part2/p2_03_theory.tex:756 | CARRIED |  |  |
| P80 | 1226-1228 | Eq.80 Ψ = Φ | part2/p2_03_theory.tex:758 | CARRIED |  |  |
| P81 | 1233-1234 | Eq.81 δ̈ + 2H_IAM δ̇ - 4πGρδ = 0 | part2/p2_03_theory.tex:760 | CARRIED | verify_theory_derivations.py §13 (−0.67 %, −0.78 %) | \derived, \calc |
| P81a | 1235-1237 | inline: H_IAM > H_ΛCDM due to βE(a) | part2/p2_03_theory.tex:762 | CARRIED-CORRECTED | T18 (matter-sector rate; background unchanged) |  |
| P81b | 1238-1247 | inline: Level 2 CAMB; 61.5; Δχ²=+0.54; σ8 0.809→0.800; H0(matter) 72.26, 0.75σ | part2/p2_03_theory.tex:767 | CARRIED | p2_04/p2_06 chain values | \measured |
| P82 | 1248-1260 | Eq.82 µ = E²ΛCDM/E²IAM < 1, Σ=1; µ(0,0.5,1) = 0.864/0.948/0.982 | part2/p2_03_theory.tex:777 | CARRIED-CORRECTED | T18 wording ('effective matter density parameter seen by perturbations' for 'Ωm^IAM<Ωm^ΛCDM'); verify_theory_derivations.py §8 | \derived, \calc |
| P82a | 1261-1264 | inline: Σ=1 from δφ=0 | part2/p2_03_theory.tex:783 | CARRIED |  |  |
| S11.4 | 1265-1271 | §11.4 'unique region'; f(R) µ>1; Horndeski | part2/p2_03_theory.tex:787 | CARRIED-CORRECTED | T13 (f(R) Σ=1; sDGP µ<1, Σ=1 ghost; 'distinctive' among viable models) | \observed |
| F1 | 1280-1370 | Figure 1 (a) µ(z) with data; (b) µ0–Σ0 plane 'unique to IAM'; (c) E(a), µ(a) | part2/p2_03_theory.tex:493 | CARRIED-CORRECTED | redrawn by figscripts/fig_p2_theory_derivations.py; T13 placements; MGCAMB gap 2.8% at z≈0.64 (T18); survey error bars and 'Euclid 3.4σ band' not drawn (T15, Euclid rule) | \calc |
| S11.5 | 1272-1277 | §11.5 motivation (DESI Y5, Euclid ~1% fσ8) | part2/p2_03_theory.tex:796 | CARRIED |  |  |
| P83 | 1371-1395 | Eq.83 D̈2 + 2H D2' − (3/2)H0²Ωm D2 = −(7/2)(3/2)H0²Ωm D1²; ratios 1.014/1.052/1.075 | part2/p2_03_theory.tex:804 | CARRIED-CORRECTED | NEW: source −4πGρ̄D1² (factor 7/2 inconsistent with D2→−3/7 D1²; EdS check); ratios recomputed 0.989/0.995/0.997/0.999 (T5); verify_theory_derivations.py §13 | \derived, \calc |
| P84 | 1396-1424 | Eq.84 F2 kernel; bispectrum ratios 1.033/1.052/1.072, −1.3% at z=0 | part2/p2_03_theory.tex:810 | CARRIED-CORRECTED | T5 (same early amplitude: 0.974/0.989/0.993/0.998; same amplitude today 1.015/1.020/1.025) | \derived, \calc |
| S11.5a | 1427-1429 | inline: amplitude crossover near z≈0.2 'unique' | — | EXCLUDED | T5 (no crossover with one normalisation) |  |
| S11.5b | 1430-1438 | 1-loop: c2=−61/630, σ²NL≈0.30, −2.5% both, <0.07% | part2/p2_03_theory.tex:821 | CARRIED-CORRECTED | T18 (numbers not reproduced; argument kept) | \derived |
| S11.5c | 1439-1448 | k_nl 0.254 vs 0.257 at z=1; shifts 0.7–1.3% to smaller k | part2/p2_03_theory.tex:825 | CARRIED-CORRECTED | NEW: values are z=0 (0.251/0.255, IAM larger); z=1 0.759/0.760 (verify_theory_derivations.py §13) | \calc |
| S11.5d | 1449-1454 | §11.5 summary | part2/p2_03_theory.tex:831 | CARRIED-CORRECTED | T5 |  |
| S12 | 1455-1457 | §12 intro: companion paper, twelve MCMC analyses | part2/p2_03_theory.tex:837 | CARRIED-CORRECTED | stand-alone rule |  |
| S12.1 | 1458-1468 | §12.1 Δχ² +1.43/+1.34; Level 2 +0.54; free µ0 = 0.033±0.125 (1.3σ) | part2/p2_03_theory.tex:840 | CARRIED-CORRECTED | T16 (final extraction: +0.96 Planck, +0.56 Planck+RSD; free µ0 consistent with 0 and −0.136, p2_04) | \measured |
| S12.2 | 1469-1472 | §12.2 σ8 0.813→0.800 (1.6%) | part2/p2_03_theory.tex:847 | CARRIED-CORRECTED | T7 (per level) | \measured |
| S12.3 | 1475-1478 | §12.3 photon-sector observables; TT < 0.13% at ℓ>30 | part2/p2_03_theory.tex:851 | CARRIED | not re-verified here (needs chain spectra) | \calc |
| F2 | 1479-1513 | Figure 2 CMB TT IAM vs ΛCDM vs Planck; 'Δχ²=+0.75'; σ8 0.814→0.801 | — | EXCLUDED | not redrawn: spectra not in tree; T6 (+0.75 typed text). Text result kept in S12.3 |  |
| S12.4 | 1514-1517 | §12.4 continuity identity | part2/p2_03_theory.tex:859 | CARRIED | verify_theory_derivations.py §9 | \derived |
| P85 | 1518-1546 | Eq.85 ∇µT^µν_total = 0; Bianchi | part2/p2_03_theory.tex:865 | CARRIED |  | \interp |
| S12.5 | 1547-1551 | §12.5 DESI w0–wa direction | part2/p2_03_theory.tex:871 | CARRIED |  | \observed |
| P86 | 1552-1556 | Eq.86 w_eff(a); w_eff(0) = −1.062 | part2/p2_03_theory.tex:874 | CARRIED | verify_theory_derivations.py §10 | \calc, \prediction |
| P87 | 1557-1567 | Eq.87 H0 sirens = 67.4√1.1575 = 72.51 | part2/p2_03_theory.tex:881 | CARRIED-CORRECTED | photon-sector H0 67.16 → 72.26 (author rule); verify_theory_derivations.py §14 | \calc |
| P87a | 1568-1574 | inline: Level 2 72.26; GW170817 75.5 cited to Abbott 2017/Nicolaou 2023, 0.5σ | part2/p2_03_theory.tex:882 | CARRIED-CORRECTED | T14 (Palmese 2024; other analyses 68–70; consistent with both) | \observed |
| T4 | 1575-1590 | §12.7 Table 4 S/N: DESI DR1 0.22 (0.6σ), DESI Y5 1.7σ, Euclid 3.4σ, combined 4.5σ | part2/p2_03_theory.tex:886 | CARRIED-CORRECTED | T15 (DESI FS µ0 = 0.11 +0.45/−0.54); Euclid only per verify_euclid_template.py (0.3σ–7σ); DESI Y5 / combined rows removed (unsourced) | \observed, \calc |
| S13.1 | 1594-1614 | §13.1 nine-step causal chain | part2/p2_03_theory.tex:894 | CARRIED-CORRECTED | T18 (step 8 matter-sector rate; step 9 effective Ωm seen by perturbations); status split as in Part 5 | \conjecture |
| S13.2 | 1615-1623 | §13.2 gravity as cause and product; Verlinde, Padmanabhan | part2/p2_03_theory.tex:914 | CARRIED | 'triggers wave function collapse' → 'resolves superpositions' (physics terms rule) | \interp |
| S13.3a | 1626-1636 | §13.3 two regimes; collapse regime | part2/p2_03_theory.tex:931 | CARRIED |  |  |
| P88 | 1637-1644 | Eq.88 Sinfo(R)/A(R) ≥ 1/4ℓP² | part2/p2_03_theory.tex:935 | CARRIED |  | \conjecture |
| P89 | 1645-1662 | Eq.89 SBH/A at Rs = c³/4ħG = 1/4ℓP² (hoop) | part2/p2_03_theory.tex:938 | CARRIED | verify_theory_derivations.py §15 ('identically equivalent' → 'equivalent') | \derived, \interp |
| P90 | 1663-1673 | Eq.90 Meq = c³/4GH ≈ 2.3e22 M⊙ | part2/p2_03_theory.tex:945 | CARRIED | verify_theory_derivations.py §15 (2.324e22) | \derived, \calc |
| P90a | 1675-1679 | inline: Γ = c³/(1920 GM ln2) bits/s | part2/p2_03_theory.tex:950 | CARRIED | verify_theory_derivations.py §15 (P from black-body Hawking power written out) | \derived |
| S13.3b | 1680-1693 | encoding transition; E(a)/e fraction; high-z black holes | part2/p2_03_theory.tex:954 | CARRIED |  | \interp |
| S13.4 | 1694-1699 | §13.4 coincidence problem | part2/p2_03_theory.tex:965 | CARRIED |  | \conjecture |
| S13.5 | 1700-1716 | §13.5 arrow of time; E(z)=exp(−z); Penrose | part2/p2_03_theory.tex:39 | CARRIED-CORRECTED | T8 ('potential … actualized' → records, low-entropy state); Carroll & Chen 2004 not cited (no verified bib entry) | \interp |
| S14.1 | 1718-1727 | §14.1 three-step chain | part2/p2_03_theory.tex:990 | CARRIED-CORRECTED | Step 3 'for the matter sector' (T18) |  |
| S14.2 | 1730-1735 | §14.2 what is standard | part2/p2_03_theory.tex:1000 | CARRIED |  |  |
| S14.3 | 1736-1741 | §14.3 what is new | part2/p2_03_theory.tex:1008 | CARRIED |  | \conjecture |
| S14.4 | 1742-1748 | §14.4 relation to previous work; 'generically predict µ≥1' | part2/p2_03_theory.tex:1016 | CARRIED-CORRECTED | T13 (sDGP µ<1 noted); DES Y3 extensions citation dropped (no verified bib entry) |  |
| S14.5 | 1749-1772 | §14.5 assumptions 1–5 (MGCAMB 1–2.5%) | part2/p2_03_theory.tex:1026 | CARRIED-CORRECTED | T18 (MGCAMB within 2.8%); stand-alone rule (chains) |  |
| S15 | 1773-1799 | §15 predictions 1–6 | part2/p2_03_theory.tex:1045 | CARRIED-CORRECTED | T13 (item 2 f(R) Σ=1; 'unique' → among viable models; item 6 f(R)-type only); T12 (item 5); D7 bound 0.0039 added (item 4) | \prediction |
| S16 | 1800-1822 | §16 conclusion (0.3%, 1% ST claims; companion paper) | part2/p2_03_theory.tex:1070 | CARRIED-CORRECTED | T11, T2/T18 (claims removed); stand-alone rule |  |
| ACK | 1823-1829 | Acknowledgments (names a private correspondent) | — | EXCLUDED | N1 (no names of private correspondents); book has no per-chapter acknowledgments |  |
| DATA | 1830-1832 | Data availability | — | EXCLUDED | repository paths carried in the verification script reference |  |
| REF | 1833-1915 | References (47 entries) | — | CARRIED (cited) | every reference used in the chapter cited from iam.bib or bib_theoryder.bib; Jenkins/Reed/Watson/Bryan/Bett/Ludlow/Klypin not cited (tables excluded); Nicolaou 2023 replaced by Palmese 2024 (T14); DES 2023 extensions, Carroll & Chen, Einstein 1915, Peebles 1980 not needed here (§13 in Part 5) |  |
| SRC | — | §An entropy source, not a new entropy law (from the Entropic Gravity note; existing, kept) | part2/p2_03_theory.tex:253 | CARRIED | ENTROPIC_GRAVITY_NOTE_CHECK.md | \interp, \observed |

Totals: 191 rows — CARRIED 120, EXCLUDED 11, CARRIED-CORRECTED 59, CARRIED (cited) 1. Every numbered source equation (1–90), every table (1–4) and figure (1–2), and the inline steps are listed.

## 3. Corrections found in this check (not yet in PAPER_ERRATA.md — proposed rows)
| proposed | where | printed | correct | evidence |
|---|---|---|---|---|
| T19 | Eq. 22–23 | −dE = 4π r̃_A²(ρ+P)H dt | 4π r̃_A³(ρ+P)H dt (Cai & Kim 2005); r̃_A² gives Ḣ = −4πG(ρ+P)H | verify_theory_derivations.py §5 |
| T20 | Eq. 11 | T_ab = (ħη/2π)R_ab + f g_ab, f = −R/2+Λ | T_ab = (ħη/2π)(R_ab + f g_ab) | §2 |
| T21 | §8.3 | λ̇ = +(3H0²/8πG)βe^φ | d(a³λ)/dt = −a³ρ_info (λ̇ + 3Hλ = −ρ_info) | §9 |
| T22 | Eq. 83 | RHS −(7/2)(3/2)H0²Ω_m(a)D1²; "(3/2)H0²Ω_m(a)" | RHS −4πGρ̄D1², 4πGρ̄ = (3/2)H0²Ω_m a⁻³ (7/2 contradicts D2 → −3/7 D1²) | §13 (EdS sympy) |
| T23 | Eq. 72 | f_coll ST 0.593, Tinker 0.646, Ω_m f_coll 0.195 (+24 %) | recomputed 0.64 / 0.71 (EH no-wiggle, M > 10⁶ M⊙), 0.20–0.22 (+27–41 %), η_vir 0.79/0.71 | §11 (DISCREPANCY line) |
| T24 | Table 2, §10.3, Eq. 77 | α/β 0.76/0.87, 0.95/1.05, 1.04/1.15, PS 1.04/1.18; crossings 3.5/3.3; ST σ*=1.2 0.925/1.009 | with R = Ω_m(a)fD^n per Hubble time, accumulated per dt: 0.66/0.74, 0.93/1.02, 1.06/1.16; crossings 3.77/3.42; ST σ*=1.2 0.75/0.89. Literal Eq. 28 diverges from a = 0 | §12; T17 should be revised ("within 5 %" → constant 7 %, 1/a 2 %) |
| T25 | §11.5 | k_nl 0.254 (IAM) / 0.257 (ΛCDM) "at z = 1", IAM smaller | these are z = 0: 0.255 / 0.251 with the same early amplitude (IAM larger); z = 1: 0.760 / 0.759 | §13 |
| T26 | Eq. 45 | exp(∫ (H/a) da/H) | exp(∫ (H/a) da/(aH)) (dt = da/aH); result unchanged | §7 |
| T27 | §11.1 | δφ = 0 "exactly, at all orders in linear perturbation theory" | δφ = 0 in linear (first-order) perturbation theory | source's own §11.1 caveat |

## 4. Verification
- `docs/verification/scripts/verify_theory_derivations.py` → `verify_theory_derivations_output.txt` (131 lines): 45 PASS, 0 FAIL, 5 open DISCREPANCY lines (rows T23–T25 above and T5's D2 ratios). Symbolic: Jacobson coefficient, η, G, δA_min, Cai–Kim (both r̃_A powers), n = 7/2, E(a) ODE, C = 1, E(z), cancellation of 8πG/3, µ0, minisuperspace action with lapse (constraint term lapse-free; φ and λ Euler–Lagrange), w_info, continuity, w_eff(1), dw/da, D2 EdS coefficient, F2 limits, hoop ratio, Γ. Numerical: decoherence toy, exponent slopes, µ(z), MGCAMB gap 2.82 % at z = 0.64, CPL least squares, f_coll, accumulated-record fits, growth −0.67/−0.78 %, D2 and bispectrum ratios, k_nl, sirens 72.26, M_eq 2.324×10²² M⊙.
- Figures: `docs/book/figscripts/fig_p2_theory_derivations.py` (uses `_bookstyle.py`) → `figures/part2/fig_theory_mu_sigma.pdf/.png` (source Fig. 1 redrawn, T13 placements, no survey bars) and `figures/part2/fig_record_fit.pdf/.png` (new, Table 2 shape). Overlap check 0 for both. Existing fig_exponent, fig_w_eff kept (their scripts unchanged).
- Source Fig. 2 (CMB TT) not redrawn: the posterior-mean spectra are not in the tree; its result (< 0.13 % at ℓ > 30) is carried as text.

## 5. Static checks (delivered chapter)
1091 lines (was 319), 90 display equations, 116 labels, 53 \ref targets — all resolve against the built tree (main.tex order); no label duplicated anywhere in the built book; braces balanced; environments matched; $ parity even; all 4 \includegraphics files exist; 55 cite keys all present in iam.bib + bib_theoryder.bib. Every label of the 41f7646 chapter is kept (eq:Idot, eq:dSdt, eq:Sn, part2:eq:Ea, eq:HIAM, sec:var, sec:numerics, sec:source, fig:exponent, fig:w_eff, ch:theory). Chapter references: back to ch:iams_law, ch:blackholes; forward to ch:dual, ch:latetime, ch:level2, ch:dsvalidation, ch:theoryinterp (all worded as forward). Words absent: 'this paper', 'the paper', 'companion', author name, the quantum-processor report/the semiconductor report/the methylation report, the cell-reading engine, 'the author', 'superseded'. Not compiled (TeX bundle unavailable in the sandbox).

## 6. Bibliography
`docs/book/bib_theoryder.bib`: Bernardeau2002, Schlosshauer2007, Wald1984, AbbottDESY3_2022 (new keys; merge into iam.bib or add to the \bibliography line). CrossRef: all 36 cited keys that carry a DOI resolved (titles/years matched); 19 cited iam.bib entries carry no DOI field (Bekenstein1973, Jacobson1995, CaiKim2005, Hawking1975, Unruh1976, … — pre-existing, not changed here).

## 7. Merge notes for the lead
- Source §13 is now carried in full here (Section sec:th:interp, author's words, \interp/\conjecture); part5/p5_01_interpretation.tex carries a shorter version — reconcile.
- Jacobson → η (§th:jacobson, §th:eta) duplicates the Bekenstein-coefficient chapter by design (coordination note).
- β_m printed as canon 0.15765 (Ω_m 0.3153); numerical integrations keep the source background Ω_m 0.315, Ω_r 9.1×10⁻⁵, H0 67.4 (stated in the text).
- Free µ0: PAPER_ERRATA T16 says 0.039 ± 0.125 (1.4σ), p2_04's table gives medians +0.059/+0.064 with the posterior at the prior edge; the chapter says only 'consistent with both 0 and −0.136' and cites ch:dual. The two sources should be reconciled.
