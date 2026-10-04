# MANIFEST — Part 2: the cosmological constant and the baryon density (line-for-line carriage, 2026-10-03)

Base: sparse clone of IAM-Validation at e37aabb (later than 41f7646 / 12d8fb1). Nothing pushed.

## Files in this delivery
| File | Role |
|---|---|
| `docs/book/part2/p2_12_lambda.tex` | Ch. `ch:lambda` — CC paper §1–§6 (rewritten, owned) |
| `docs/book/part2/p2_12b_lambda_history.tex` | Ch. `ch:lambda_history` — CC paper §7–§10 + the shared history integral (new) |
| `docs/book/part2/p2_13_baryon.tex` | Ch. `ch:baryon` — Matter–Antimatter paper §1–§4, §7, §9, §10 (rewritten, owned) |
| `docs/book/part2/p2_13b_baryon_chain.tex` | Ch. `ch:baryon_chain` — Baryon-Asymmetry paper + MA §8 + 18th-chain record (new) |
| `docs/book/main.tex` | placement lines below |
| `docs/book/bib_lambda_baryon.bib` | 7 new references, all CrossRef-checked |
| `docs/book/figscripts/fig_p2_lambda_history.py` | new figure script (on `_bookstyle.py`) |
| `docs/book/figures/part2/fig_cc_history.pdf/.png` | new figure `fig:cc_history` |
| `docs/verification/scripts/verify_lambda_baryon_book.py` + `_output.txt` | every equation (sympy) and number of the four chapters |

Figures reused unchanged (already in the repo): `fig_cc_factors`, `fig_cc_relation` (in `ch:lambda`), `fig_eta`, `fig_baryon_posterior` (moved with their labels to `ch:baryon_chain`).

## main.tex
```
\input{part2/p2_12_lambda}
\input{part2/p2_12b_lambda_history}     % new, line 33
\input{part2/p2_13_baryon}
\input{part2/p2_13b_baryon_chain}       % new, line 35
...
\bibliography{iam,bib_lambda_baryon}    % line 117
```

## Reading ledger (pypdfium2 text, `=== PAGE` markers included; matches docs/book/PAPER_LINE_COUNTS.md exactly)
| Paper | Lines | Chunks read (≤ 50) | Status |
|---|---|---|---|
| Matter_Antimatter_Asymmetry_and_the_Information_Writing_Constraint.pdf | 438 | 1–50, 51–100, …, 401–438 (9 chunks) | complete |
| Baryon_Asymmetry_as_a_Derived_Quantity_CMB_Evidence_Without_BBN_Prior.pdf | 242 | 1–50 … 201–242 (5) | complete |
| The_Cosmological_Constant_as_Actualized_Vacuum_Energy.pdf | 842 | 1–50 … 801–842 (17) | complete |
| 18thChainBaryonAsymmetry.rtf | 10 raw RTF lines (ledger: 7 after RTF→text); the text is one paragraph in line 10 | read in full | complete |
| LaTeX `iam_cosmological_constant.tex` (1033), `iam_baryon_asymmetry.tex` (703), `Evidence_Baryon.tex` (382) | every equation environment extracted and compared with the PDF | complete for equations |
| Coverage wave-2 sources 14_/15_/17_ (606/331/955 lines) | headers, ERRATUM/FLAG lines | used as pointers |
| Checks: `PAPER_ERRATA.md` C1–C24, W1, T8; `CC_AND_BARYON_CHECK.md` 124 lines; `verify_cc_and_baryon.py` 71 + output 26; chain yaml 4192 B | read in full | — |

Word counts (non-comment text): p2_12_lambda.tex 4931, p2_12b_lambda_history.tex 2114, p2_13_baryon.tex 2521, p2_13b_baryon_chain.tex 2375. Before (HEAD e37aabb): p2_12_lambda.tex 1,875 and p2_13_baryon.tex 1,101 words (2,976); now 11,941 in four chapters (4.0-fold); the three papers have 11,761 words.

## Exclusions (for the author)
| Paper item | Reason (confirmed correction / author ruling) |
|---|---|
| CC §5.3, eq. 27: "holographic round trip 2/π × π/2 = 1", the π/2 in the lepton relation | C15 (not a holographic result; remove) |
| MA abstract ¶3, §5 (incl. eq. 5 in that role), §10 ¶3: the 10⁹ annihilation partners as the dark sector; "95 % memory, 5 % present" | C21; author 2026-10-02 ("keep anything out that is wild speculation") |
| MA §6: weak force as "force of becoming"; forces of "being"; EW breaking as "origin of duration"; CP violation as a structural necessity of E(a); CKM phase fixed by the loop | CC_AND_BARYON_CHECK #21; author 2026-10-02 exclusion list |
| MA §8.2 "IAM posteriors at < 0.1σ"; MA §10 duplicated paragraph | C23 |

No age-excess, axis-of-evil, cusp-core or small-scale material occurs in these papers or chapters.

## New findings for the errata (not yet in PAPER_ERRATA.md — the lead should add rows)
1. **C10 traced.** Cyburt et al. 2016 (RMP 88, 015004) Table IV gives η×10¹⁰: CMB-only 6.108 ± 0.060; BBN+D 6.180 ± 0.195; BBN+Yp+D 6.172 ± 0.195; CMB+BBN 6.098 ± 0.042 (text: Planck 2015 6.10 ± 0.04). The "Observed (BBN) 6.137 ± 0.017" of BA Table 1 / MA §8.2 / the record is **not in that source**. The "Observed (CMB) 6.136 ± 0.038" matches no Planck 2018 Table 2 entry either (TT,TE,EE+lowE+lensing Ω_bh² = 0.02237 ± 0.00015 → 6.127 ± 0.041; +BAO 0.02242 ± 0.00014 → 6.141 ± 0.038). The book now uses these printed values; "0.36 % agreement" is not carried.
2. **η 6.1155 vs 6.113.** The record's η uses 2.74 × 10⁻⁸; with 2.739 × 10⁻⁸ (Steigman 2006) the same Ω_bh² = 0.022320 gives 6.113. The 30 % burn-in is applied in both (record Ω_bh² 0.022319 = recomputed 0.022320). CC_AND_BARYON_CHECK.md line 69 ("the record's 6.1155 uses the full chain") is wrong — the cause is the conversion factor.
3. **Electroweak lower limit.** a_EW ≈ 2.3 × 10⁻¹⁵ (CC §8, MA §3) is T₀/T without the g_{*s} factor; with entropy conservation a(100 GeV) = 7.8 × 10⁻¹⁶ and a(159.5 GeV) = 4.9 × 10⁻¹⁶. The history-integral coefficient rises from 3.1 × 10³⁰ to 2.7 × 10³¹ / 6.9 × 10³¹.
4. **History integral form in the old chapter.** The previous `p2_12_lambda.tex` printed the integrand with l_H²/l_P² in the numerator; the paper (LaTeX and PDF) has it in the denominator, which is the form whose a⁻³ divergence the check computes. Fixed.
5. **Look-elsewhere count.** "2 of 540" (C17, from a review file not in the repo) could not be reproduced without its family. The book now carries its own defined count: coefficients q·π^k·(Ω_b/Ω_m)^i·Ω_Λ^j with q among the 23 distinct ratios of 1–6, k ∈ {−1,0,1}, i ∈ {0,1}, j ∈ {0,½,1}: 3 of 414 within 1 % (0.7 %), the expression among them (verify D). app_C3/ errata C17 still say 2 of 540.
6. **Units.** "N_max = A_H/l_P² bits" and "l_P² is the area per bit" corrected to one nat per 4l_P², one bit per 4 ln2 l_P² (horizon today 2.265 × 10¹²² nats = 3.268 × 10¹²² bits; QCD horizon 2.9 × 10⁷⁸ nats = 4.2 × 10⁷⁸ bits).
7. **"122 orders".** log₁₀(ρ_Λ/ρ_vac) = −122.95: "about 123 orders".
8. **Ω_mh².** 0.3153 × 0.6736² = 0.1431 (chapters previously wrote 0.1430; η values unchanged at 6.080 / 5.031).

## Citations
All 39 cite keys in the four chapters resolve: 32 in iam.bib, 7 new in bib_lambda_baryon.bib (Riess1998 10.1086/300499; Perlmutter1999 10.1086/307221; BoussoPolchinski2000 10.1088/1126-6708/2000/06/006; Caldwell1998 10.1103/PhysRevLett.80.1582; RubakovShaposhnikov1996 10.1070/PU1996v039n05ABEH000145; Davidson2008 10.1016/j.physrep.2008.06.002; Witten2001 10.1007/978-3-662-04587-9_3), each checked on CrossRef. Values quoted from Cyburt2016 (Table IV) and Planck2018VI (Table 2, eq. 24) were read in the full texts.

# Carriage tables

Verdicts: CARRIED / CARRIED-CORRECTED (source) / MISSING / EXCLUDED. No item is MISSING. Book locations are file:line in the delivered files.


## A. The Cosmological Constant as Actualized Vacuum Energy (842 PDF lines, read 1-842)

| # | Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|---|
| 1 | Abstract ¶1, l.13-19 | CC problem as category error; ρvac vs Λobs not same kind | p2_12_lambda.tex:11 | CARRIED | \interp |
| 2 | Abstract ¶2, l.20-25 | ρvac = reservoir of Planck states; Λ = accumulated Landauer cost | p2_12_lambda.tex:15 | CARRIED-CORRECTED (wording rule T8/W1: 'potential/actuality' -> states/records) | \interp |
| 3 | Abstract ¶3, l.26-34 | ratio = fraction of bits written × baryon fraction; only baryons write | p2_12_lambda.tex:20 | CARRIED | \conjecture |
| 4 | Abstract eq.(1), l.35-47 | (2/π)(lP/lH)² Ωb/Ωm | p2_12_lambda.tex:157 | CARRIED (as Eq. base) | \calc |
| 5 | Abstract l.50-54 | 2/π from static patch subtending 2π sr, 'geometrically exact' | p2_12_lambda.tex:302 | CARRIED-CORRECTED (C1, C14) | \openprob |
| 6 | Abstract l.55-60 | numbers 1.38e-123, factor 1.2 | p2_12_lambda.tex:177 | CARRIED | \calc |
| 7 | Abstract l.61-86 | √ΩΛ correction, 1.141e-123, '0.07 %', 'zero free parameters' | p2_12_lambda.tex:208 | CARRIED-CORRECTED (C16: +0.79 %, introduced to close 1.22) | \fitted |
| 8 | Abstract l.86-89 | history integral 'independent derivation'; Λ increases; 'w > −1, consistent with DESI' | p2_12_lambda.tex:322 | CARRIED-CORRECTED (C2, C18) | \openprob |
| 9 | Abstract l.89-95 | 18th chain η = 6.115±0.037, 0.36 %, 'BBN prior removed', 'independently of nuclear physics' | p2_13b_baryon_chain.tex:117 | CARRIED-CORRECTED (C3, C4, C8, C9, C10, C22; η 6.113 with 2.739e-8) | \measured |
| 10 | Abstract l.96-97 | three objections | p2_12_lambda.tex:324 | CARRIED | — |
| 11 | §1 l.104-106 | CC problem most significant (Weinberg, Martin) | p2_12_lambda.tex:31 | CARRIED | — |
| 12 | §1 eq.(2) l.107-114 | ρvac ~ E_P⁴/(ħc)³ ≈ 4.6e113 | p2_12_lambda.tex:35 | CARRIED (4.633e113; E_P 1.956e9 J) | \calc |
| 13 | §1 eq.(3) l.115-124 | ρΛ = ΩΛ ρc ≈ 5.25e-10; ρc = 3H0²/8πG; Riess, Perlmutter | p2_12_lambda.tex:40 | CARRIED (ρc c² written explicitly) | \observed |
| 14 | §1 eq.(4) l.125-129 | ratio ≈ 1.14e-123, '122 orders' | p2_12_lambda.tex:44 | CARRIED-CORRECTED (recomputed: 1.133e-123, log10 −122.95, 'about 123 orders'; verify B2) | \calc |
| 15 | §1 l.130-134 | SUSY, landscape, quintessence, modified gravity; all reduce ρvac | p2_12_lambda.tex:48 | CARRIED | — |
| 16 | §1 l.135-145 | different approach: not too large; distinct quantities; 'follows ... no free parameters' | p2_12_lambda.tex:53 | CARRIED-CORRECTED (identity sets 10^-123; O(1) factor open) | \interp/\conjecture |
| 17 | §2.1 l.150-156 | principle 1: gravitational decoherence dominant | p2_12_lambda.tex:83 | CARRIED | \interp |
| 18 | §2.1 l.157-159 | principle 2: Landauer, T_H, E_bit = ħH0 ln2/2π | p2_12_lambda.tex:88 | CARRIED (+ numbers 2.655e-30 K, 2.541e-53 J) | \derived/\calc |
| 19 | §2.1 l.160-166 | principle 3: N_max = A_H/l_P² = 4π(c/H0)²/l_P² | p2_12_lambda.tex:95 | CARRIED-CORRECTED (C13: A_H/4l_P²; nats and bits per author rule) | \derived |
| 20 | §2.1 l.167-170 | single friction term; β_m derived, 'confirmed by 17 chains at 0.2σ' | p2_12_lambda.tex:101 | CARRIED-CORRECTED (C19) | \derived |
| 21 | §2.2 l.171-183 | ρvac complete substrate; l_P² minimum area per bit | p2_12_lambda.tex:109 | CARRIED-CORRECTED (one nat per 4l_P²) | \interp |
| 22 | §2.2 l.186-190 | Λobs accumulated cost since EW breaking | p2_12_lambda.tex:113 | CARRIED | \interp |
| 23 | §2.2 l.191-195 | not the same; equality needs E(a→∞)=e | p2_12_lambda.tex:120 | CARRIED (limit checked, sympy A8) | \calc/\interp |
| 24 | §2.2 l.196-199 | 10^122 is expected ratio | p2_12_lambda.tex:121 | CARRIED-CORRECTED (123; identity caveat) | \interp |
| 25 | §2.3 l.200-212 | dark matter = geometric half, already classical; only baryons write; Ωb/Ωm | p2_12_lambda.tex:126 | CARRIED-CORRECTED (wording T8) | \interp/\conjecture |
| 26 | §2.3 l.213-215 | 'not introduced to improve agreement ... before the CC calculation' | p2_12_lambda.tex:137 | CARRIED-CORRECTED (check #14: added when gap stood at 12) | \fitted |
| 27 | §3.1 eq.(5) l.220-241 | f_geo = l_P²/A_H = l_P²H0²/4πc² | p2_12_lambda.tex:148 | CARRIED (+ = (1/4π)(lP/lH)², C13) | \derived |
| 28 | §3.1 eq.(6) l.242-247 | f_b = Ωb/Ωm | p2_12_lambda.tex:152 | CARRIED | \conjecture |
| 29 | §3.1 eq.(7) l.248-262 | ratio = (2/π)(lP/lH)²Ωb/Ωm (no step from f_geo) | p2_12_lambda.tex:159 | CARRIED-CORRECTED (C13: missing step stated) | \openprob |
| 30 | §3.2 eqs.(8)-(12) l.263-271 | inputs l_P, H0 (2.184e-18), l_H 1.373e26, Ωb, Ωm | p2_12_lambda.tex:165 | CARRIED | \measured/\calc |
| 31 | §3.2 eq.(13) l.272-281 | (lP/lH)² = 1.387e-122 | p2_12_lambda.tex:173 | CARRIED | \calc |
| 32 | §3.2 eq.(14) l.284-290 | Ωb/Ωm = 0.1564 | p2_12_lambda.tex:173 | CARRIED | \calc |
| 33 | §3.2 eq.(15) l.291-302 | 1.38e-123 | p2_12_lambda.tex:177 | CARRIED (1.380e-123) | \calc |
| 34 | §3.2 eq.(16) l.303-313 | observed 5.25e-10/4.63e113 = 1.14e-123 | p2_12_lambda.tex:181 | CARRIED (1.133e-123) | \calc |
| 35 | §3.2 eq.(17) l.314-322 | ratio 1.22; '122 orders reduced to 1.2' | p2_12_lambda.tex:185 | CARRIED (1.218) with identity caveat | \calc |
| 36 | §3.3 l.323-330 | factor 1.2 from departure from de Sitter; 2/π exact in pure dS | p2_12_lambda.tex:191 | CARRIED-CORRECTED (2/π not derived) | — |
| 37 | §3.3 eq.(18) l.331-334 | T_H = ħH0/2πk_B | p2_12_lambda.tex:195 | CARRIED | \derived |
| 38 | §3.3 eq.(19) l.335-343 | T_dS = T_H √ΩΛ | p2_12_lambda.tex:197 | CARRIED | \derived |
| 39 | §3.3 eq.(20) l.346-349 | E_bit,eff = k_B T_dS ln2 | p2_12_lambda.tex:199 | CARRIED | \derived |
| 40 | §3.3 eq.(21) l.350-371 | corrected formula | p2_12_lambda.tex:203 | CARRIED | \fitted |
| 41 | §3.3 eqs.(22)-(23) l.372-386 | √0.6846 = 0.8274; 1.141e-123, 0.07 % | p2_12_lambda.tex:207 | CARRIED-CORRECTED (C16: 1.142e-123, +0.79 %) | \fitted |
| 42 | §3.3 l.387-395 | exponent 0.502; 'not fitted' | p2_12_lambda.tex:208 | CARRIED-CORRECTED (C16: 0.521; introduced to close 1.22) | \calc |
| 43 | §3.4 Table 1 l.396-409 | calculation table | p2_12_lambda.tex:221 | CARRIED-CORRECTED (2/π row 'Exact' -> not derived) | table |
| 44 | §4 l.412-416 | Planck cutoff; EW/QCD cutoffs break down | p2_12_lambda.tex:274 | CARRIED + quantified ((E_P/M)^4 = 2.2e68, 1.4e79; verify B8) | \calc |
| 45 | §4 l.417-441 | cutoff set by horizon encoding; lower cutoffs incomplete | p2_12_lambda.tex:283 | CARRIED-CORRECTED (nat/bit area) | \interp |
| 46 | §5 l.442-445 | 2/π needs motivation; de Sitter | p2_12_lambda.tex:292 | CARRIED | — |
| 47 | §5.1 l.446-460 | static patch subtends 2π sr; Rindler analogy | p2_12_lambda.tex:301 | CARRIED-CORRECTED (C14) | \derived |
| 48 | §5.2 eq.(24) l.461-473 | A_eff = 2π l_H² | p2_12_lambda.tex:308 | CARRIED (premise withdrawn, C14) | — |
| 49 | §5.2 eq.(25) l.474-492 | N_bits = πl_H²/2l_P² | p2_12_lambda.tex:310 | CARRIED (units: nats) | \derived |
| 50 | §5.2 eq.(26) l.493-508 | f_bit = l_P²/(A_eff/4π) = (2/π)(lP/lH)² | p2_12_lambda.tex:312 | CARRIED-CORRECTED (C1: = 2(lP/lH)²) | \derived/\openprob |
| 51 | §5.3 eq.(27) l.509-526 | holographic round trip 2/π × π/2 = 1; Koide π/2 | — | EXCLUDED (C15, confirmed: not a holographic result) | — |
| 52 | §5.4 l.527-539 | departure from exact de Sitter; history integral 'complementary derivation' | p2_12_lambda.tex:317 | CARRIED-CORRECTED (C2) | \openprob |
| 53 | §6 l.540-541 | three objections | p2_12_lambda.tex:324 | CARRIED | — |
| 54 | §6.1 l.542-564 | Planck cutoff arbitrary; response; conditional result | p2_12_lambda.tex:328 | CARRIED | \interp |
| 55 | §6.2 l.565-586 | Ωb/Ωm post hoc; response; 5-step order | p2_12_lambda.tex:342 | CARRIED-CORRECTED (check #14) | \fitted |
| 56 | §6.3 l.587-602 | 1.2 incomplete; response 0.07 % | p2_12_lambda.tex:357 | CARRIED-CORRECTED (C16) | \fitted |
| 57 | §6.3 l.603-610 | exponent 0.502 'indistinguishable', 'not tuned' | p2_12_lambda.tex:366 | CARRIED-CORRECTED (0.521) | \calc |
| 58 | §6.3 l.611-614 | 'vanishingly small' probability | p2_12_lambda.tex:368 | CARRIED-CORRECTED (C17; own count 3 of 414 replaces '2 of 540', verify D) | \calc |
| 59 | §6.3 l.615-617 | 10^122 reduced to 1.2, closed to 0.07 % | p2_12_lambda.tex:369 | CARRIED-CORRECTED | \openprob |
| 60 | §7 l.618-624 | Λ not constant; E(a) monotonic | p2_12b_lambda_history.tex:14 | CARRIED | \conjecture |
| 61 | §7 l.625-627 | dE/da|_1 = 1 | p2_12b_lambda_history.tex:21 | CARRIED | \derived |
| 62 | §7 eq.(28) l.629-632 | w = −1 + ε, ε>0; above −1 today | p2_12b_lambda_history.tex:26 | CARRIED-CORRECTED (C18: w = −1 − ε; + derivation Eqs. lh_w, lh_winfo) | \derived |
| 63 | §7 l.633-642 | DESI DR2 2.8-4.2σ, w>−1; not added for DESI | p2_12b_lambda_history.tex:34 | CARRIED-CORRECTED (C18; comparison open) | \observed/\openprob |
| 64 | §8 l.643-646 | full history from a_EW ≈ 2.3e-15, 100 GeV | p2_12b_lambda_history.tex:41 | CARRIED-CORRECTED (a(100 GeV) = 7.8e-16 with entropy conservation; new, verify G2) | \calc |
| 65 | §8 eq.(29) l.647-681 | history integral | p2_12b_lambda_history.tex:47 | CARRIED (LaTeX form; the old book line had l_H²/l_P² inverted — fixed) | \conjecture |
| 66 | §8 l.682-686 | epoch weighting narrative | p2_12b_lambda_history.tex:54 | CARRIED-CORRECTED (C2, #3, #17: grows as a^-3) | \derived |
| 67 | §8 l.687-695 | √ΩΛ from late time; normalization ongoing | p2_12b_lambda_history.tex:62 | CARRIED-CORRECTED (K = 3.1e30 vs 0.523) | \calc/\openprob |
| 68 | §9 l.696-703 | Weinberg, SUSY, landscape, unimodular | p2_12b_lambda_history.tex:103 | CARRIED | — |
| 69 | §9 l.704-707 | two differences; 'derived consequence' | p2_12b_lambda_history.tex:109 | CARRIED-CORRECTED (prefactors open) | \interp |
| 70 | §9 l.708-717 | Verlinde, Padmanabhan, Jacobson, Cai-Kim | p2_12b_lambda_history.tex:114 | CARRIED (wording: 'additional entropy term', not 'modification') | — |
| 71 | §10 l.718-726 | single distinction; ρvac; Λobs | p2_12b_lambda_history.tex:124 | CARRIED | \interp/\conjecture |
| 72 | §10 eq.(30) l.729-743 | baseline 1.38e-123 | p2_12b_lambda_history.tex:134 | CARRIED | \calc |
| 73 | §10 eq.(31) l.744-768 | corrected 1.141e-123, 0.07 % | p2_12b_lambda_history.tex:139 | CARRIED-CORRECTED (C16) | \fitted |
| 74 | §10 l.769-771 | secondary: w > −1, DESI | p2_12b_lambda_history.tex:145 | CARRIED-CORRECTED (C18) | \derived/\openprob |
| 75 | §10 l.772-781 | chain 'direct observational confirmation'; two expressions | p2_12b_lambda_history.tex:150 | CARRIED-CORRECTED (C5, C8, C22) | \observed |
| 76 | §10 l.782-789 | persisted 50 years; both correct as computed | p2_12b_lambda_history.tex:153 | CARRIED | \interp |
| 77 | §10 l.790-795 | 13.8 Gyr, 10^61 Planck lengths, 15.6 %, 68.5 %, 1.141e-123 | p2_12b_lambda_history.tex:157 | CARRIED-CORRECTED (8.5e60 radius; 1.142e-123) | \calc |
| 78 | Data/Ack/Refs l.798-842 | repository; Jacobson; CAMB/MGCAMB/Cobaya; references | p2_13b_baryon_chain.tex:203 | CARRIED (paths per C11; MGCAMB JCAP 08, 038 per C24 = Wang2023MGCAMB) | — |

## B. Matter-Antimatter Asymmetry and the Information Writing Constraint (438 PDF lines, read 1-438)

| # | Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|---|
| 1 | Abstract l.13-17 | asymmetry free parameter; question | p2_13_baryon.tex:15 | CARRIED | — |
| 2 | Abstract l.18-24 | writing compatible with capacity; loop; fixed point | p2_13_baryon.tex:20 | CARRIED | \conjecture |
| 3 | Abstract l.25-29 | 10^9 annihilation partners = dark sector | — | EXCLUDED (C21; author 2026-10-02) | — |
| 4 | Abstract l.32-36 | no derivation; conditions; falsification test | p2_13_baryon.tex:24 | CARRIED | — |
| 5 | §1 eq.(1) l.40-46 | η ≈ 6.1e-10 | p2_13_baryon.tex:31 | CARRIED | \observed |
| 6 | §1 l.47-51 | BBN and CMB consistent; before nucleosynthesis | p2_13_baryon.tex:33 | CARRIED + traced values (Cyburt Table IV; Planck Table 2) | \observed |
| 7 | §1 l.52-56 | Sakharov; CKM insufficient; literature | p2_13_baryon.tex:39 | CARRIED | — |
| 8 | §1 l.57-61 | η free parameter | p2_13_baryon.tex:46 | CARRIED | — |
| 9 | §1 l.62-70 | writing budget question | p2_13_baryon.tex:51 | CARRIED | \conjecture |
| 10 | §2 l.71-85 | framework, three principles, S = A/4l_P² | p2_13_baryon.tex:58 | CARRIED (refers to Ch. lambda for details) | \derived/\interp |
| 11 | §2 l.86-88 | μ<1, Σ=1; β_m 'confirmed by 17 chains' | p2_13_baryon.tex:68 | CARRIED-CORRECTED (C19) | \derived |
| 12 | §2 l.89-92 | DM geometric half, DE kinetic half; 2K+V=0 | p2_13_baryon.tex:74 | CARRIED | \interp |
| 13 | §2 l.92-94 | virial ratio 'asymptotes to exactly 2.0'; coincidence 'dissolved' | p2_13_baryon.tex:75 | CARRIED-CORRECTED (C19: 2 by definition) | \derived/\interp |
| 14 | §3 eq.(2) l.95-131 | history integral | p2_12b_lambda_history.tex:47 | CARRIED once (shared with CC eq. 29) | \conjecture |
| 15 | §3 l.132-133 | a_EW ≈ 2.3e-15 'origin of duration' | p2_12b_lambda_history.tex:64 | CARRIED-CORRECTED (a recomputed; 'origin of duration' wording excluded, check #21) | \calc |
| 16 | §3 l.134-137 | integrand = writing rate | p2_13_baryon.tex:82 | CARRIED | \conjecture |
| 17 | §3 l.138-146 | upper limit | p2_13_baryon.tex:88 | CARRIED (nats) | \conjecture |
| 18 | §3 l.147-150 | lower limit | p2_13_baryon.tex:93 | CARRIED | \conjecture |
| 19 | §3 l.151-155 | Ωb/Ωm 'not a free input ... independently of agreement' | p2_13_baryon.tex:100 | CARRIED-CORRECTED (check #14) | \observed |
| 20 | §4 l.156-163, eq.(3) | Step 1; a_QCD 1.6e-12; n_b = η n_γ | p2_13_baryon.tex:110 | CARRIED-CORRECTED (C20: a_QCD 9.5e-13) | \derived |
| 21 | §4 l.164-169 | Step 2 | p2_13_baryon.tex:113 | CARRIED | \conjecture |
| 22 | §4 l.170-172 | Step 3 | p2_13_baryon.tex:117 | CARRIED | \conjecture |
| 23 | §4 l.173-177 | Step 4 | p2_13_baryon.tex:120 | CARRIED | \conjecture |
| 24 | §4 l.178-182 | Step 5 | p2_13_baryon.tex:124 | CARRIED + DM present at z≈1100 (check #20) | \conjecture/\observed |
| 25 | §4 eq.(4) l.183-185 | loop η→n_b→Ẇ→Λ→{ρdm,ρde}→H→η | p2_13_baryon.tex:133 | CARRIED (full form; old book omitted {ρdm,ρde}) | \conjecture |
| 26 | §4 l.186-192 | fixed point; Sakharov mechanism kept | p2_13_baryon.tex:23 | CARRIED | \conjecture |
| 27 | §5 l.193-226, eq.(5) | dark sector as encoded annihilation history; 19.4; 95 % memory | — | EXCLUDED (C21; check #20-21; author 2026-10-02). 19.39 recomputed in verify F5 only | — |
| 28 | §6 l.227-254 | weak force 'force of becoming'; EW breaking 'origin of duration'; CP as structural necessity; CKM phase from loop | — | EXCLUDED (check #21; author 2026-10-02 exclusion list) | — |
| 29 | §7 l.255-258 | loop is argument, not derivation | p2_13_baryon.tex:143 | CARRIED | — |
| 30 | §7 l.259-264 | Element 1, a_QCD 1.6e-12 | p2_13_baryon.tex:147 | CARRIED-CORRECTED (C20) | \openprob |
| 31 | §7 l.265-270 | Element 2, l_H 1e-3 pc, 1e40 bits | p2_13_baryon.tex:152 | CARRIED-CORRECTED (C20: 15.5 km = 5.0e-13 pc, 2.9e78 nats = 4.2e78 bits) | \calc/\openprob |
| 32 | §7 l.271-275 | Element 3 | p2_13_baryon.tex:158 | CARRIED | \openprob |
| 33 | §7 l.276-278 | research programme | p2_13_baryon.tex:163 | CARRIED | \openprob |
| 34 | §8.1 l.281-299 | chain setup, 'BBN prior removed', 'four times', 'reference values' | p2_13b_baryon_chain.tex:71 | CARRIED-CORRECTED (C4, C6, C7, C9) | \observed |
| 35 | §8.2 eqs.(6)-(7) l.300-311 | Ωbh² = 0.022319±0.000136; η = 6.115±0.037; 0.36 %; R−1; Gaussian | p2_13b_baryon_chain.tex:115 | CARRIED-CORRECTED (C3, C10; 0.022320, η 6.113) | \measured |
| 36 | §8.2 l.309 | 'IAM posteriors at < 0.1σ' | — | EXCLUDED (C23 stray line) | — |
| 37 | §8.3 l.312-329 | three independent frameworks | p2_13b_baryon_chain.tex:151 | CARRIED-CORRECTED (C8) | \observed |
| 38 | §8.3 l.330-331 | 'falsifiable in both directions'; 'stated in advance' | p2_13b_baryon_chain.tex:182 | CARRIED-CORRECTED (C22) | \openprob |
| 39 | §8.4 l.332-341 | √ΩΛ closes CC to 0.07 % and η from 5.07; 'one correction, two results' | p2_13b_baryon_chain.tex:189 | CARRIED-CORRECTED (C5, C16; 5.03) | \fitted |
| 40 | §9 l.342-352 | prior work: EW baryogenesis, leptogenesis, Affleck-Dine; no new mechanism | p2_13_baryon.tex:177 | CARRIED | \conjecture |
| 41 | §9 l.353-359 | Weinberg anthropic parallel; no observer selection | p2_13_baryon.tex:188 | CARRIED | \interp |
| 42 | §10 l.360-370 | loop 'confirmed numerically'; chain result; 'no nuclear physics'; 'stated in advance' | p2_13_baryon.tex:193 | CARRIED-CORRECTED (C5, C9, C22) | \measured/\openprob |
| 43 | §10 l.371-375 | 999,999,999 partners encoded as dark sector | — | EXCLUDED (C21) | — |
| 44 | §10 l.376-378 | CC and η one constraint; 'zero free parameters' | p2_13_baryon.tex:201 | CARRIED-CORRECTED (C5) | \observed/\openprob |
| 45 | §10 l.379-384 | same category as Λ and β_m | p2_13_baryon.tex:204 | CARRIED | \interp |
| 46 | §10 l.385-389 | is η free? 'value the universe had to have' | p2_13_baryon.tex:211 | CARRIED | \conjecture |
| 47 | §10 l.390-397 | duplicated paragraph | — | EXCLUDED (C23 duplicate) | — |
| 48 | Data/Ack/Refs l.398-438 | chain path, yaml, Jacobson, references | p2_13b_baryon_chain.tex:203 | CARRIED-CORRECTED (C11 paths; C24 MGCAMB ref) | — |

## C. The Baryon Asymmetry as a Derived Quantity (242 PDF lines, read 1-242)

| # | Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|---|
| 1 | Title/subtitle l.2-6 | 'IAM's Law selects η' | p2_13b_baryon_chain.tex:9 | CARRIED-CORRECTED (C8: chapter title states what the chain measures) | — |
| 2 | Abstract l.14-18 | η free; self-consistency constraint | p2_13b_baryon_chain.tex:12 | CARRIED | \conjecture |
| 3 | Abstract l.18-24 | test: 'BBN prior removed', 'four times wider', 0.36 %, 'no nuclear physics' | p2_13b_baryon_chain.tex:15 | CARRIED-CORRECTED (C4, C6, C9, C10) | \measured |
| 4 | Abstract l.24-27 | same correction closes both; two expressions | p2_13b_baryon_chain.tex:189 | CARRIED-CORRECTED (C5) | \fitted |
| 5 | §1 eq.(1) l.30-35 | η ≈ 6.137e-10 | p2_13b_baryon_chain.tex:21 | CARRIED-CORRECTED (6.137 untraced, C10 -> ≈6.1e-10) | \observed |
| 6 | §1 l.38-44 | precise; BBN and CMB agree; free parameter | p2_13b_baryon_chain.tex:22 | CARRIED | \observed |
| 7 | §1 l.45-48 | Λ analogous, '122 orders' | p2_13b_baryon_chain.tex:27 | CARRIED-CORRECTED (123) | — |
| 8 | §1 eq.(2) l.49-51 | ΔE_act = k_B T_H ln2 per bit | p2_13b_baryon_chain.tex:33 | CARRIED-CORRECTED (W1 wording) | \derived |
| 9 | §1 l.52-56 | law connects Λ and η | p2_13b_baryon_chain.tex:36 | CARRIED | \conjecture |
| 10 | §1 l.57-61 | prediction stated in advance | p2_13b_baryon_chain.tex:39 | CARRIED-CORRECTED (C12, C22) | — |
| 11 | §2 eq.(3) l.62-75 | Λ/ρvac = (2/π)(lP/lH)² Ωb/Ωm | p2_13b_baryon_chain.tex:45 | CARRIED | \openprob |
| 12 | §2 eq.(4) l.75-86 | T_dS/T_H = H_dS/H0 = √ΩΛ | p2_13b_baryon_chain.tex:49 | CARRIED | \derived |
| 13 | §2 eq.(5) l.87-107 | corrected = 1.141e-123, 0.07 %, zero free parameters | p2_13b_baryon_chain.tex:52 | CARRIED-CORRECTED (C16) | \fitted |
| 14 | §2 eq.(6) l.108-115 | η ≈ 2.74e-8 Ωbh² | p2_13b_baryon_chain.tex:56 | CARRIED (2.739e-8, Steigman 2006) | \derived |
| 15 | §2 l.116-120 | pre-test η = 6.079 (eq.3), 6.115 with √ΩΛ | p2_13b_baryon_chain.tex:58 | CARRIED-CORRECTED (C3, #24: 5.03 and 6.08) | \calc |
| 16 | §3.1 l.121-143 | setup; standard/test configurations; four times; no BBN | p2_13b_baryon_chain.tex:70 | CARRIED-CORRECTED (C4, C6, C7, C9) + yaml table | \observed |
| 17 | §3.2 eqs.(7)-(8) l.144-155 | R−1 0.009273, 22,400; Ωbh² 0.022319; η 6.115; 0.36 %; 6.079 | p2_13b_baryon_chain.tex:117 | CARRIED-CORRECTED (C3, C10) | \measured |
| 18 | §4 Table 1 l.160-167 | BBN 6.137±0.017; CMB 6.136±0.038; analytical 6.079; MCMC 6.115 | p2_13b_baryon_chain.tex:134 | CARRIED-CORRECTED (C10 traced: Cyburt Table IV and Planck Table 2 values; C3 analytic; C8) | table |
| 19 | §4 l.156-184 | three independent frameworks | p2_13b_baryon_chain.tex:151 | CARRIED-CORRECTED (C8) | \observed |
| 20 | §4 l.185-192 | CC and η common origin; one correction closes both | p2_13b_baryon_chain.tex:189 | CARRIED-CORRECTED (C5) | \fitted |
| 21 | §5 l.193-196 | falsifiable both directions; stated in advance | p2_13b_baryon_chain.tex:182 | CARRIED-CORRECTED (C22) | \openprob |
| 22 | §5 l.197-205 | √ΩΛ accounts for offset; history integral eq.19 independent derivation | p2_13b_baryon_chain.tex:191 | CARRIED-CORRECTED (C2) | \openprob |
| 23 | §6 l.206-218 | conclusions: η from CMB alone; not free; same problem | p2_13b_baryon_chain.tex:194 | CARRIED-CORRECTED (C4-C9, C22) | \measured/\observed |
| 24 | Data/Ack/Refs l.219-242 | repository, path, Jacobson, references | p2_13b_baryon_chain.tex:203 | CARRIED-CORRECTED (C11) | — |

## D. 18th-chain record (18thChainBaryonAsymmetry.rtf; 10 raw RTF lines, text in line 10, read in full)

| # | Paper location | Content | Book location | Verdict | Label |
|---|---|---|---|---|---|
| 1 | RTF l.10 (text) | ombh2 mean 0.022319473, std 0.000136384, 30 % burn-in | p2_13b_baryon_chain.tex:118 | CARRIED (recomputed 0.022320±0.000136) | \measured |
| 2 | RTF l.10 | η implied 6.1155e-10 | p2_13b_baryon_chain.tex:119 | CARRIED-CORRECTED (factor 2.74 vs 2.739 explains 6.1155 vs 6.113; check-file item 'full chain' is wrong, see Notes) | \calc |
| 3 | RTF l.10 | η_observed 6.137, agreement 0.36 % | p2_13b_baryon_chain.tex:134 | CARRIED-CORRECTED (C10: not in Cyburt 2016; traced comparators) | \measured |
| 4 | RTF l.10 | R−1 0.012617 (17,248), 0.011284 (17,472), 0.009664 (17,696), 0.009273 (17,920); bounds 0.063629; acceptance 0.749; 22,400 accepted; 'converged' | p2_13b_baryon_chain.tex:100 | CARRIED | \measured |
| 5 | RTF l.10 | progress lines 89,031-92,124 steps taken, 21,584-22,363 accepted, 2026-03-20 04:22-04:35 | p2_13b_baryon_chain.tex:98 | CARRIED (summarised: run window; start time from .progress file) | \observed |
| 6 | RTF l.10 header | file path am_planck_chains/iam_baryon_test.1.txt | p2_13b_baryon_chain.tex:203 | CARRIED-CORRECTED (C11) | — |

## Static checks (run 2026-10-03 on the delivered tree)
- Braces and environments balance in all four files; `$` count even; every float `[htbp]`.
- Labels unique across the book (non-retired .tex); every `\ref`/`\eqref` in the four files resolves; all four `\includegraphics` files exist.
- All labels that other files reference (ch:lambda, ch:baryon, eq:base, eq:corr, eq:ident, eq:rel, fig:cc_factors, fig:cc_relation, fig:eta, fig:baryon_posterior, tab:lambda_numbers, tab:baryon_chains) are still defined. `fig:eta`, `fig:baryon_posterior` and `tab:baryon_chains` now sit in `ch:baryon_chain`; `appendices/app_I_provenance.tex` pairs them with `ch:baryon` — the lead may want to change those three pairings to `ch:baryon_chain`.
- The book was not compiled (no TeX in the sandbox).

## Rules applied
Status label on every result; nothing fitted or measured called derived (the 2/π, √Ω_Λ and Ω_b/Ω_m factors and Eq. corr are \fitted/\openprob); no cohort, clinical, product or correspondent material; no paper/author self-reference; wording rule T8/W1 ("potential/actualized" → states, records, written); IAM's Law as the completion of the horizon entropy functional (Einstein equations untouched), not a rival to GR or ΛCDM; β_m = Ω_m/2 derived and fixed in every chain; author-approved CC sentence kept verbatim in ch:lambda_history; Euclid not mentioned.

## Reproduce
```
python docs/verification/scripts/verify_lambda_baryon_book.py      # from the repo root; writes _output.txt
python docs/book/figscripts/fig_p2_lambda_history.py
```
