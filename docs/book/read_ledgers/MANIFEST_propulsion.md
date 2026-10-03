# MANIFEST: Part 5, Chapter "Exploratory: propulsion and the conservation laws" (`ch:propulsion`)

Clone: `5f5997b` (sparse: docs/book, docs/verification, docs/papers, CANON, plus docs/RETIRED_2026-10/top_level for the propulsion PDF). Nothing pushed. No TeX compile.

## Key finding first
The propulsion PDF is **not a separate paper**. `Gravitational_Propulsion_and_IAM.pdf` is no longer in docs/papers (moved by commit 3aeec57 to
`docs/RETIRED_2026-10/top_level/Gravitational_Propulsion_and_IAM_duplicate.pdf`). Its title is "Gravitational Engineering and Interstellar Transit in the
Informational Actualization Model". pypdfium2 text of it and of `docs/papers/IAM_Gravitational_Engineering_Exploration.pdf` differ by **0 lines**
(268 lines each; SHA-256 f5939652... vs 248bbdc8..., errata GE5). Every line of it is already carried in `part5/p5_02_exploratory.tex`, with GE1-GE7 applied.
So the new chapter does not repeat that text. It links to it and adds only the three propulsion checks the task names: momentum conservation,
equivalence-principle (universality) limits, and energy conditions. Proposed errata GE8-GE11.

## Files
| file | status |
|---|---|
| `docs/book/part5/p5_02b_propulsion.tex` | NEW, 175 lines. `\chapter{Exploratory: propulsion and the conservation laws}\label{ch:propulsion}`. Opens with the 'written for fun' sentence. 7 numbered equations, 1 status table, no figure (the source has no figure). |
| `docs/verification/scripts/verify_propulsion.py` + `verify_propulsion_output.txt` | NEW. Every number and step in the chapter, plus a re-check of every number of the source (s.1). sympy + numpy/scipy. |
| `docs/book/bib_propulsion.bib` | NEW, 4 entries (Bondi1957, Olum1998, BartlettVanBuren1986, Singh2023), each DOI checked on CrossRef 2026-10-03. Merge into iam.bib (block 6). Existing keys used: Touboul2022, PfenningFord1997. |
| `docs/book/read_ledgers/MANIFEST_propulsion.md` | this file |

main.tex placement: new line **102** `\input{part5/p5_02b_propulsion}`, directly after line 101 `\input{part5/p5_02_exploratory}`.

## Read ledger
- Source: `docs/RETIRED_2026-10/top_level/Gravitational_Propulsion_and_IAM_duplicate.pdf`, 6 pages. Last line found first: line 268 (page number '6'); line 266-267 "Exploratory note. Zone I derivable ... Timestamped in repository."
  pypdfium2 text: **268 lines**. Ledger row 46 says 274 = 268 text lines + 6 `=== PAGE` markers (the ledger's stated counting rule). Reconciled.
- Chunks read in full, text shown, none truncated: 1-50, 51-100, 101-150, 151-200, 201-250, 251-268. Complete 1..268.
- `part5/p5_02_exploratory.tex` read in full first (361 lines; chunks 1-50, 51-100, 101-150, 151-200, 201-250, 251-300, 301-361; the last chunk was 61 lines, over the 50-line rule, shown in full),
  and its manifest `read_ledgers/MANIFEST_exploratory.md` (92 lines; chunks 1-32, 33-92, the second 60 lines, shown in full; long lines cut at 400-600 characters in display).
- Pfenning-Ford 1997 full text (arXiv gr-qc/9702026, 642 lines) read at Eqs. (22)-(31) and the summary to trace the numbers used.

## Carriage table (source line -> book file:line)
Line numbers of p5_02 are those at 5f5997b. "Own words": the source's sentences are carried in ch:exploratory; ch:propulsion carries none of them again.
| source lines | content | book | verdict |
|---|---|---|---|
| 1-4 | title | p5_02:1 (chapter title) | CARRIED in ch:exploratory |
| 5-8 | author, affiliation, e-mail, ORCID, 'Exploratory note - repository timestamp' | - | EXCLUDED (stand-alone book: no author line, no naming of own papers, no repository narrative) |
| 9-18 | status of document: two zones, boundary never blurred | p5_02:4-11; restated in p5_02b:4-10 | CARRIED in ch:exploratory |
| 19-21 | premise: decoherence on timelike worldlines, Landauer cost | p5_02:13-15 | CARRIED-CORRECTED in ch:exploratory (book frame: GR unchanged) |
| 22-25 | Eq. (1) Landauer cost at T_H | p5_02:16-22 (eq:ge_landauer) | CARRIED |
| 26-30 | virial halves; beta_m = Omega_m/2 | p5_02:24-27 | CARRIED |
| 30-36 | Eq. (2) E(a) | p5_02:27-32 (eq:ge_Ea) | CARRIED |
| 37-42 | Eq. (3) H^2_eff,m | p5_02:32-36 (eq:ge_Hm) | CARRIED |
| 43, 80, 131, 178, 233, 268 | page numbers | - | EXCLUDED (page furniture) |
| 44-46 | not new; single question | p5_02:37-38 | CARRIED |
| 47-50 | Zone I; I.1 steerable potential | p5_02:40-43 | CARRIED |
| 51-56 | Eq. (4) g = -grad Phi; definitional | p5_02:44-51 (eq:ge_g) | CARRIED |
| 57-67 | I.2 Eqs. (5)-(7): geodesic, f_felt = 0, tidal | p5_02:53-73 (eq:ge_eom, ge_felt, ge_tidal); checked in p5_02b:56-85 | CARRIED; NEW CHECK GE9 (universality, steep-focus tide) |
| 68-72 | Consequence (rigorous): no felt acceleration, 90-degree turn | p5_02:75-79; checked in p5_02b:56-85 | CARRIED; NEW CHECK GE9 |
| 73-79 | I.3 hover Eq. (8), abrupt departure; 'no equation exceeds Newtonian gravity plus the equivalence principle' | p5_02:81-91 (eq:ge_hover, GE6); checked in p5_02b:12-55 | CARRIED; NEW CHECK GE8 (momentum of a closed craft) |
| 81-90 | I.4 plumb-bob orientation, Eq. (9) | p5_02:93-101 (eq:ge_plumb) | CARRIED-CORRECTED (GE1) |
| 91-96 | tilt signature, no banking | p5_02:103-115 | EXCLUDED as a signature (GE1); correct orientation result carried |
| 97-108 | I.5 Eqs. (10)-(11), recession, D_H | p5_02:117-131 (eq:ge_vrec, ge_DH) | CARRIED |
| 109-113 | 'speed limit' not c; bounded by E -> e | p5_02:128-133 | CARRIED-CORRECTED (GE2) |
| 114-134 | I.6 Eqs. (12)-(13) proper time in a flat displaced region | p5_02:143-166 (eq:ge_alc, ge_tau, ge_dtau) | CARRIED |
| 135-140 | Consequence: no twin penalty; 'no new physics beyond the existence of such a region' | p5_02:168-184 (eq:ge_rho, GE3); extended in p5_02b:85-111 | CARRIED-CORRECTED (GE3); NEW CHECK GE11 (Olum theorem; total negative energy) |
| 141-160 | I.7 Pleiades 136 pc = 444 ly; Eqs. (14)-(15), xi = 2.3e4 | p5_02:193-206 (eq:ge_tone, ge_xi); used in p5_02b:94-111 | CARRIED; numbers re-checked verify_propulsion s.1 |
| 161-166 | not forbidden; negligible fraction of D_H; not the obstacle | p5_02:207-210 | CARRIED-CORRECTED (GE2, GE3) |
| 167-177 | transition to Zone II | p5_02:220-226 | CARRIED |
| 179-198 | II.1 Conjecture C1, Eq. (16), kernel G | p5_02:228-239 (eq:ge_kernel); used in p5_02b:31 | CARRIED; C1 between two bodies noted in p5_02b |
| 199-213 | II.2 premises P1, P2; tension | p5_02:241-256 | CARRIED |
| 214-220 | II.3 conditional prediction: does not source gravity in proportion; weighs less; weigh it | p5_02:258-272 (GE4 baseline); checked in p5_02b:113-139 | CARRIED; NEW CHECK GE10 (active vs passive mass; hover needs delta > 1) |
| 221-232 | II.4 steering Eq. (17) | p5_02:274-281 (eq:ge_drive); sources aboard: p5_02b:12-30 | CARRIED; NEW CHECK GE8 |
| 234-240 | three is the minimum; jointly necessary | p5_02:282-297 | CARRIED-CORRECTED (GE7) |
| 241-248 | II.5 what would close the gap | p5_02:308-314 | CARRIED |
| 249-262 | Summary Zone I, Zone II | p5_02:316-327; p5_02b summary 140-152 | CARRIED |
| 263-266 | closing; Zone I derivable / Zone II conjectural | p5_02:329-331 | CARRIED |
| 267 | 'Timestamped in repository' | - | EXCLUDED (stand-alone; no repository narrative) |

## Errata
Applied (inherited through ch:exploratory, same text): GE1, GE2, GE3, GE4, GE5, GE6, GE7. No row of PAPER_ERRATA.md names `Gravitational_Propulsion_and_IAM` except GE5.
Proposed new rows (block 2): **GE8** momentum conservation, **GE9** equivalence-principle / universality limit and steep-focus tides, **GE10** active vs passive mass and hover by weight reduction, **GE11** energy conditions (Olum theorem, total negative energy). PAPER_ERRATA.md lines starting '| ': 381 at HEAD -> 385 after block 2.

## Verification (docs/verification/scripts/verify_propulsion_output.txt)
Source numbers: 136 pc = 443.6 ly; c/H0 = 1.3968e10 ly (70), 1.4559e10 (67.16), 1.3532e10 (72.26); 7 d = 0.01916 yr; xi = 2.3125e4 (444/0.0192), 2.3145e4 (exact); Pleiades/D_H 3.05e-8, 3.28e-8.
Momentum: F_A + F_B = -(G'(u)+G'(-u)) = 0 for 1/|u|, Yukawa, Gaussian; odd-part kernel gives 2(2u^2-1)exp(-u^2) != 0; 40 random sources aboard: |sum F| 1e-13 (round-off).
Thrust 1/c = 3.336e-9 N/W; 1 t at 1 g 2.940e12 W, at 100 g 2.940e14 W. Bondi pair: both accelerate +Gm/r^2, momentum 0, KE 0.
Active/passive: F_net = G m_p1 m_p2 (m_a2/m_p2 - m_a1/m_p1)/r^2 (identity True). Hover: delta = (m_d + m_pay)/m_d; 2 for m_pay = m_d; passive mass of drive -m_pay.
Universality: |eta| <= 1e-4 (100 g, 0.01 g), 1e-3 (10 g), 1e-4 (1000 g, 0.1 g). Tide/acc = 2L/r; 20 g (100 m), 2 g (1 km), 0.01 g (200 km); source 5.877e23 kg = 0.0984 Earth masses.
Energy: angular integral 8 pi/3; coefficient -1/12; wall integral R^2/Delta + Delta/12; tanh check 26666.68 vs R^2 sigma/3 = 26666.67; 100 m, Delta = 100 v L_P: 6.943e62 kg (v=1), 1.607e67 kg (v = 2.3145e4); 3.47e20 Milky Ways (PF: 3e20); 1 m wall: 1.122e30 kg = 0.564 M_sun.

## Static checks (p5_02b_propulsion.tex)
Braces balanced (depth 0); 10 environments matched; `$` count even; 15 labels, all unique across docs/book (RETIRED excluded); 18 \ref/\eqref targets all resolve
(ch:exploratory, sec:ge_freefall, sec:ge_bubble, sec:ge_zone2, eq:ge_hover, eq:ge_drive, eq:ge_eom, eq:ge_tidal, eq:ge_rho, eq:ge_xi and own); 6 \cite keys present
(iam.bib + bib_propulsion.bib); no 'paper', 'note', 'actualization', author name, the quantum-processor report/the semiconductor report/the methylation report, population or cohort wording; status macro on every claim (derived 33, calc 12, conjecture 9, observed 5, interp 1).

## Insertion blocks (target | anchor line, copied exactly, occurs once at HEAD 5f5997b | position | text)

### 1. main.tex
- target: `docs/book/main.tex`
- anchor (line 101): `\input{part5/p5_02_exploratory}`
- position: AFTER
```latex
\input{part5/p5_02b_propulsion}
```

### 2. PAPER_ERRATA.md (rows GE8-GE11)
- target: `docs/verification/PAPER_ERRATA.md`
- anchor (line 388):
```
| GE7 | IAM_Gravitational_Engineering_Exploration | II.4 | three non-coplanar sources are the minimum to steer one focal node through a 3D volume | for an isotropic kernel n sources are mirror-symmetric about any plane containing them: three sources steer a single node only in their plane (off-plane foci have an equal mirror twin); a volume needs >= 4 non-coplanar sources or an anisotropic kernel; also a static Laplace kernel has no isolated focus (maximum principle), so the drive must oscillate | confirmed (verify s.9; fig_exploratory_steering) |
```
- position: AFTER
```
| GE8 | Gravitational Engineering note | I.3, II.4, Summary | hover and abrupt departure are gradient operations; no equation exceeds Newtonian gravity plus the equivalence principle | with the sources aboard, a craft's own static field exerts no net force on it (even kernel; momentum conservation); a hover or departure needs an external body to push on, emitted field momentum at P >= m a c (2.94e12 W per tonne at 1 g), or negative mass (Bondi 1957) | confirmed | `docs/book/part5/p5_02b_propulsion.tex` (verify_propulsion.py s.2-4) |
| GE9 | Gravitational Engineering note | I.2 | no felt acceleration regardless of the magnitude or direction of the gradient | needs a field that acts alike on all mass-energy: felt load m eta a, so |eta| <= eps g/a (1e-4 for 100 g felt below 0.01 g); premise P2 makes the field non-universal by construction; a steep focus adds the tide 2aL/r: 20 g across 10 m at 100 m from a 100 g focus, 0.01 g needs r >= 200 km and 5.9e23 kg | confirmed | `docs/book/part5/p5_02b_propulsion.tex` (verify_propulsion.py s.7-8) |
| GE10 | Gravitational Engineering note | II.3 | coherent fraction does not source gravity, so the sample weighs slightly less | conflates active mass (sourcing) and passive mass (weight); unequal m_a/m_p makes a rigid pair self-accelerate (third law fails); lunar ranging: equal for Al and Fe to 4e-12 (Bartlett-Van Buren 1986), 3.9e-14 (Singh et al. 2023); a self-contained craft hovering by weight reduction needs delta = 1 + m_pay/m_d > 1, negative passive mass, not 'slightly less' | confirmed | `docs/book/part5/p5_02b_propulsion.tex` (verify_propulsion.py s.5-6) |
| GE11 | Gravitational Engineering note | I.6-I.7 | flat displaced region carrying the craft at xi = 2.3e4 | any superluminal travel (Olum 1998 definition, generic condition) needs WEC violation, not only Alcubierre's metric; total E = -(v^2/12)(R^2/Delta + Delta/12); at the quantum-inequality wall (Delta <= 1e2 v L_P) a 100 m bubble needs 6.9e62 kg at c (Pfenning-Ford: 6.2e62 kg), 1.6e67 kg at 2.3e4 c; a 1 m wall needs 0.56 solar masses | confirmed | `docs/book/part5/p5_02b_propulsion.tex` (verify_propulsion.py s.9) |
```

### 3. Book errata appendix
- target: `docs/book/appendices/app_B_errata_physics.tex`
- anchor (line 31): `Gravitational Engineering & displaced flat region needs no new physics & needs negative energy density (Ch.~\ref{ch:exploratory}) \\`
- position: AFTER
```latex
Gravitational Engineering & hover and departure need only a shaped gradient & a craft's own field exerts no net force; needs an external body, radiated momentum ($P\ge mac$) or negative mass (Ch.~\ref{ch:propulsion}) \\
Gravitational Engineering & no felt acceleration regardless of magnitude & only for a universal field, $|\eta|\le\varepsilon g/a$; a steep focus adds the tide $2aL/r$ (Ch.~\ref{ch:propulsion}) \\
Gravitational Engineering & material does not source gravity, so it weighs slightly less & active and passive mass conflated; hover by weighing less needs negative passive mass (Ch.~\ref{ch:propulsion}) \\
Gravitational Engineering & flat displaced region at $\xi=2.3\times10^4$ & every superluminal displacement needs negative energy; $1.6\times10^{67}$\,kg at that rate (Ch.~\ref{ch:propulsion}) \\
```

### 4. Status-of-all table
- target: `docs/book/part5/p5_11_status_all.tex`
- anchor (line 107): `Equivalence of inertial and gravitational mass & $\eta\lesssim10^{-15}$ for ordinary materials & \observed & Ch.~\ref{ch:exploratory} \\`
- position: AFTER
```latex
Equality of active and passive gravitational mass & Al and Fe equal to $3.9\times10^{-14}$ (lunar ranging) & \observed & Ch.~\ref{ch:propulsion} \\
Thrust of a closed craft & $F\le P/c$; $2.94\times10^{12}$\,W per tonne at $1\,g$ & \derived & Ch.~\ref{ch:propulsion} \\
Negative energy of a 100\,m superluminal bubble & $6.9\times10^{62}$\,kg at $c$; $1.6\times10^{67}$\,kg at $2.3\times10^4c$ & \calc & Ch.~\ref{ch:propulsion} \\
```

### 5. Read ledger rows 46, 47 and total
- target: `docs/book/PAPER_LINE_COUNTS.md`
- anchor (line 57): `| 46 | Gravitational_Propulsion_and_IAM | 6 | 274 | — | — | not confirmed |  |` — REPLACE with
```
| 46 | Gravitational_Propulsion_and_IAM | 6 | 274 | — | 1–274 (2026-10-03; 268 text lines + 6 page markers, chunks 1-50, 51-100, 101-150, 151-200, 201-250, 251-268) | **complete** | same text as row 47 (pypdfium2 diff 0 lines); file now at docs/RETIRED_2026-10/top_level/Gravitational_Propulsion_and_IAM_duplicate.pdf (GE5); carried in ch:exploratory, checks in ch:propulsion |
```
- anchor (line 58): `| 47 | IAM_Gravitational_Engineering_Exploration | 6 | 274 | — | — | not confirmed |  |` — REPLACE with
```
| 47 | IAM_Gravitational_Engineering_Exploration | 6 | 274 | — | 1–274 (268 text lines + 6 page markers; read in full for ch:exploratory and again 2026-10-03) | **complete** | carried in part5/p5_02_exploratory.tex |
```
- anchor (line 60): `Total to read: 30,815 lines across 46 papers. Complete: 21,809 lines (36 papers).` — REPLACE with (recompute if another delivery changed the total first: +548 lines, +2 papers)
```
Total to read: 30,815 lines across 46 papers. Complete: 22,357 lines (38 papers).
```

### 6. Bibliography
- target: `docs/book/iam.bib`
- anchor (line 2883): `@article{EverettRoman1997,`
- position: BEFORE (blank line after the block)
```bibtex
@article{Bondi1957,
  author={H. Bondi}, title={Negative mass in general relativity},
  journal={Reviews of Modern Physics}, volume={29}, pages={423--428}, year={1957}, doi={10.1103/RevModPhys.29.423}
}

@article{Olum1998,
  author={K. D. Olum}, title={Superluminal travel requires negative energies},
  journal={Physical Review Letters}, volume={81}, pages={3567--3570}, year={1998}, doi={10.1103/PhysRevLett.81.3567}
}

@article{BartlettVanBuren1986,
  author={D. F. Bartlett and D. {Van Buren}}, title={Equivalence of active and passive gravitational mass using the moon},
  journal={Physical Review Letters}, volume={57}, pages={21--24}, year={1986}, doi={10.1103/PhysRevLett.57.21}
}

@article{Singh2023,
  author={V. V. Singh and J. M{\"u}ller and L. Biskupek and E. Hackmann and C. L{\"a}mmerzahl},
  title={Equivalence of active and passive gravitational mass tested with lunar laser ranging},
  journal={Physical Review Letters}, volume={131}, pages={021401}, year={2023}, doi={10.1103/PhysRevLett.131.021401}
}
```

## Open items
1. The task describes the propulsion PDF as a separate paper to carry line for line. It is the same text as the engineering note (diff 0 lines), so a line-for-line
   carriage would duplicate ch:exploratory. I carried nothing twice; the chapter adds the three checks only. If the lead prefers, ch:propulsion can instead be merged
   into ch:exploratory as a final section (no label changes needed).
2. Pfenning-Ford's own numbers are internally a little inconsistent: Eq. (29) gives 6.2e70 v_b L_Planck (= 1.35e66 v_b g with m_Planck) next to 6.2e65 v_b g; their formula
   (28) with Delta = 1e2 v_b L_P gives 6.9e62 v_b kg, near their grams figure. Their '1 m wall ~ a quarter of a solar mass' is 0.56 M_sun from the same formula. The chapter prints
   the recomputed values and quotes only their 6.2e62 kg and 3e20 galaxy figures. Not an erratum of the source text.
3. The Pfenning-Ford wall bound (their Eq. 23) is derived under v_b t0/rho << 1; at v_b = 2.3e4 it is used "at face value" (stated in the chapter).
4. Olum's theorem needs his definition of superluminal travel and the generic condition (both stated). A 2021 claim of positive-energy warp solitons exists in the literature and
   is disputed; not cited (not DOI-checked here).
5. `app_I_provenance.tex`: no new figure, no row needed.
