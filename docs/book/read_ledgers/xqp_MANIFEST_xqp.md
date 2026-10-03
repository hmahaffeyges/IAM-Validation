# MANIFEST — quasiparticle floor carriage (Chapter `ch:xqp`, `docs/book/part3/p3_02_xqp.tex`), 2026-10-03

## Files delivered
| file | role |
|---|---|
| `docs/book/part3/p3_02_xqp.tex` | the chapter (346 lines, 4,368 words; was 99 lines). Placement unchanged: `main.tex` line 48 `\input{part3/p3_02_xqp}` |
| `docs/book/bib_xqp.bib` | 8 new entries (Burnett2014, Diamond2022PRXQ, Lenander2011, Bal2024, Catelani2011PRB, Greytak1964, Yelton2024, Kittel2005); 7 DOIs checked on CrossRef 2026-10-03, Kittel is a book. Add `\bibliography{iam,bib_xqp}` (or merge into iam.bib) |
| `docs/book/figscripts/fig_p3_xqp_book.py` | draws the three new figures on `_bookstyle.py` |
| `docs/book/figures/part3/fig_xqp_ladder.{pdf,png}`, `fig_xqp_london.{pdf,png}`, `fig_xqp_tau.{pdf,png}` | new figures (existing `fig_xqp_thermal`, `fig_xqp_sites` from `fig_p3.py` kept, unchanged) |
| `docs/verification/scripts/verify_xqp_book.py` + `_output.txt` | every equation (sympy) and number of the chapter recomputed |
| `docs/book/read_ledgers/MANIFEST_xqp.md` | this file |

## Reading ledger
- PDF `docs/papers/IAM_Xqp_Mahaffey.pdf`, pypdfium2 text with `=== PAGE` markers: **854 lines** (ledger `PAPER_LINE_COUNTS.md` row 36: 854). Read 1–854 in 17 chunks of ≤ 50 lines; no chunk truncated.
- No LaTeX source exists in `docs/papers/latex/`. Author's words also taken from `docs/book/coverage/wave2/36_Xqp.tex` (161 lines, read 1–161).
- Corrections read in full: `PAPER_ERRATA.md` rows XQ1–XQ8; `particle/XQP_CHECK.md` (18 lines); `particle/XQP_REFEREE_NOTE.md` (47 lines); `scripts/verify_xqp.py` (17) + output (12).
- Cited sources opened to check the claims they carry (see "New findings"): Kurter et al. 2022 (full text), De Dominicis et al. arXiv:2405.18355 (full text), Ristè et al. 2013 (full text), Burnett et al. 2014 (full text), Bal et al. 2024 (abstract), Sundelin/Andersson arXiv:2602.01945, Lisenfeld arXiv:2511.05365, Anthony-Petersen arXiv:2208.02790, Connolly arXiv:2302.12330, Yelton arXiv:2402.15471, Diamond arXiv:2204.07458, Catelani arXiv:1106.0829 (abstracts).

## New findings in this carriage (corrections not yet in PAPER_ERRATA; proposed rows XQ9–XQ16)
| id | paper location | printed | correct (source) | applied in chapter |
|---|---|---|---|---|
| XQ9 | Intro l.45–52; Conclusion l.790 | underground site reduces "muon flux by a factor of thirty"; "identical T1" | muon interactions reduced by six orders of magnitude (1.4 km rock); "similar average T1 ≈ 80 µs"; same study found a significant excess of radiation-induced events above ground (De Dominicis, arXiv:2405.18355, text and Table I) | l.38–44 |
| XQ10 | Intro l.63–65; Table I row 8 | background QPT "largely independent of T1 and capacitor pad geometry"; pinholes "dominate QPT at millikelvin" | QPT rate sensitive to capacitor material and geometry, scales with capacitor area in some designs; reduced-gap sites are a model for an anomalous T-dependence below 100 mK in some devices (Kurter 2022 abstract and text) | l.48–52, Table `tab:xqp_obs` |
| XQ11 | l.148–151, 347–350, 707–709; Table I | Ristè 2013 "observed no change in x_qp as T_fridge was varied below 150 mK" | parity-switching rates rise with T over 20–170 mK, much weaker than thermal at low T; T1 insensitive to T until 150 mK; n_qp = 0.04 µm⁻³ at 20 mK (Ristè 2013 text) | l.172–176; Table row 5 |
| XQ12 | Eq. (10), Eq. (11), l.396 | τ_φ = ħ/Δ ≈ 3.6 fs; ratio ≈ 1e-10; "∼fs" | 3.6 ps; 1.2e-7 at 30 µs (verify_xqp_book.py §4) | Eqs. `eq:xqp_tauphi`, `eq:xqp_sep` |
| XQ13 | Eq. (7) | x_th ≈ 2 √(2πΔ/k_BT) e^(−Δ/k_BT) | √(2πk_BT/Δ) e^(−Δ/k_BT) (BCS integral 4ν₀ΔK₁, sympy; same form as primer and fig_p3.py) | Eq. `eq:xqp_xth` |
| XQ14 | Eq. (9) | δφ ~ g/ω_q ~ 1e-3 | 2e-4 – 2e-3 for g/2π = 1–10 MHz at 5 GHz | Eq. `eq:xqp_kick` |
| XQ15 | l.91–94, 488–489; Fig. 2 | Burnett 2014 "established 1/f noise persisting to timescales ~1 µs to >10³ µs"; "published range 1–100 µs" | Burnett measured 1/f noise down to 0.1 Hz (interacting TLS, switching times over many decades); 1–100 µs is a working range, not a Burnett result | l.71–76; Table `tab:xqp_inputs` |
| XQ16 | Numerical evaluation l.486–487 | Δ = 182 µeV as "median from Kurter: 183–193 µeV" | 182 µeV is BCS 1.764 k_BT_c at T_c = 1.20 K; Kurter design medians 183–193 µeV | Table `tab:xqp_inputs` |
| XQ17 | Eq. (8) and §bath | T_gap = 2.11 K presented as a bath temperature | stated as an energy scale: Al is normal above T_c = 1.20 K, so no part of the film is at 2.11 K (conjecture kept) | l.84–89, Eq. `eq:xqp_ladder` |
| XQ18 | l.490–491 | τ_qp "published range 100–200 µs" (Lenander; Serniak) | range not found in the cited abstracts; carried as a working value measured per device | Table `tab:xqp_inputs` |
| XQ19 | Table I row 4 | Ref. [11] shows "Poissonian" background statistics | Ref. [11] shows individual events uncorrelated between two co-housed qubits, bursts correlated about once per minute | l.53–54; Table row 4 |
Also: the London integral (Eq. 4–5) gives πλ³ only with the exponential envelope normalised at the site; the full e^(−r/λ)/r profile normalised at a core radius r₀ gives 2πλr₀² (stated in the chapter, l.127–131).

## Item table (every equation, derivation step, table, figure and quantitative claim)
Paper location = PDF text line(s). Book location = `p3_02_xqp.tex` line. Status label as printed in the book.
| # | paper location | content | book location | verdict | status |
|---|---|---|---|---|---|
| 1 | Abstract l.8–11; Intro l.38–44 | floor x_qp ~ 1e-7 persists two decades after IR, radioactivity, cosmic rays reduced | 25–35 | CARRIED | observed |
| 2 | Abstract l.11–14 | proposal: floor endogenous, cost of coherence against TLS bath | 9–14, 59–64 | CARRIED | conjecture |
| 3 | Abstract l.14–17 | refresh at τ_TLS⁻¹ paying k_B T_gap ln2; T_gap = 2.11 K | 77–89, 200 | CARRIED | conjecture |
| 4 | Abstract l.18–20 | V_eff = πλ_L³ from London Green's function, λ_L = 50 nm | 109–135 | CARRIED-CORRECTED (XQ2: restoration volume, not the QP volume) | derived |
| 5 | Abstract l.21–26 | x_min = 6.5e-8, factor 1.5, exact at τ_TLS ≈ 20 µs, no adjustable parameters | 218–226, 243–271 (corrected model) | EXCLUDED as printed (XQ6, XQ7) → replaced by Eq. `eq:xqp`, N to be measured | derived/calc |
| 6 | Abstract l.26 | consistent with nine observations | 274–292 Table `tab:xqp_obs` | CARRIED-CORRECTED (XQ8, XQ9–XQ11, XQ19) | observed |
| 7 | Abstract l.27–30 | invariant x n_cp τ_TLS πλ³/τ_qp = ln 2 | 298 Eq. `eq:xqp_invariant` (= 2 with N, V) | CARRIED-CORRECTED (XQ7) | prediction |
| 8 | Abstract l.31–33 | endogenous T1 ceiling not reduced by shielding alone | 321–327 | CARRIED | conjecture |
| 9 | Intro l.35–41 | QPs are broken pairs limiting coherence; x = n_qp/n_cp | 23–28 | CARRIED-CORRECTED (n_cp definition, XQ6) | observed |
| 10 | Intro l.45–52 | Gran Sasso test; rules out cosmic radiation as dominant | 38–44 | CARRIED-CORRECTED (XQ8, XQ9) | observed |
| 11 | Intro l.53–62 | Connolly: non-eq density with eq energy distribution; endogenous source preferred | 45–49 | CARRIED-CORRECTED (XQ8) | observed |
| 12 | Intro l.63–65 | Kurter: QPT independent of geometry, sensitive to barrier pinholes Δ₁ ≈ 0.1Δ₀ | 50–54 | CARRIED-CORRECTED (XQ10) | observed |
| 13 | Intro l.66–68 | observations consistent with a source at the barrier; the proposal | 57–58 | CARRIED | conjecture |
| 14 | Mechanism l.70–76 | three central assumptions (i)–(iii) | 59–64 | CARRIED | conjecture |
| 15 | l.78–88 | junction maintains coherence continuously; control-system analogy; cost structural | 65–70 | CARRIED | interp |
| 16 | l.89–94 | τ_TLS = 30 µs, Burnett range | 71–76; Table `tab:xqp_inputs` | CARRIED-CORRECTED (XQ15) | working value |
| 17 | l.95–104 | irreversibility: restoring the phase discards which-TLS information; Landauer k_B T_H ln2; T_H = T_gap | 77–83 | CARRIED | conjecture |
| 18 | l.105–109 | Kurter pinholes Δ₁ 5–30 µeV ≈ 10 % of Δ₀ ≈ 185 µeV; λ_L ≈ 50 nm | 50–54; 126 | CARRIED-CORRECTED (XQ10) | observed |
| 19 | l.110–124 | V_eff per site; λ_L vs ξ₀ ≈ 1.6 µm; ξ₀ volume ~1e4× larger; Prediction 3 tests λ³ scaling | 132–140 | CARRIED-CORRECTED (factor printed as 1e4 here, 3e4 in l.288: book gives (32)³ ≈ 3×10⁴; λ³ scaling dropped, XQ5) | calc/interp |
| 20 | l.125–130, Eq. (1) | T_CMB > T_gap > T_fridge | 90–96 Eq. `eq:xqp_ladder` | CARRIED-CORRECTED (T_c added, XQ17; CMB does not reach chip) | calc |
| 21 | l.131–143 | condensate decoupled from fridge bath; e^(−Δ/kT) ∼ 10⁻⁶³⁰ | 158–160 | CARRIED-CORRECTED (XQ3: 10⁻⁶¹) | calc |
| 22 | l.144–154 | prediction: x independent of T_fridge; consistent with Ristè | 172–176 | CARRIED-CORRECTED (XQ4, XQ11) | prediction |
| 23 | l.155–158 | foundations derive the assumptions from BCS, London, Landauer–Bennett | 104–108 | CARRIED-CORRECTED (derivations make assumptions precise, not established) | interp |
| 24 | l.161–173, Eq. (2) | London equation ∇²A = A/λ² | 109–112 Eq. `eq:xqp_london` | CARRIED | derived |
| 25 | l.222–230, Eq. (3) | Green's function A(r) = μ₀J/(4π) e^(−r/λ)/r | 113–116 Eq. `eq:xqp_green` | CARRIED (checked symbolically) | derived |
| 26 | l.231–233 | ∇φ = (2e/ħ)A; δφ ∼ e^(−r/λ) | 117–118 | CARRIED | derived |
| 27 | l.234–257, Eq. (4) | V_eff = ∫|δφ/δφ(0)|² d³r = 4π∫e^(−2r/λ)r²dr | 120–122 Eq. `eq:xqp_veff_int` | CARRIED (assumptions stated) | derived |
| 28 | l.258–280, Eq. (5) | u = 2r/λ; Γ(3); = πλ³ exact | 123–126 Eq. `eq:xqp_veff`; Fig. `fig:xqp_london` | CARRIED | derived |
| 29 | l.281–286 | ξ₀ amplitude vs λ_L phase length; erasure is phase, so λ_L | 132–136 | CARRIED | interp |
| 30 | l.287–289 | ξ₀ would give x ∼ 1e-11 | — | EXCLUDED (value rests on the withdrawn per-site formula, XQ6/XQ7); the volume ratio kept | — |
| 31 | l.290–292 | κ = λ/ξ₀ ≈ 0.03 | 137–138 | CARRIED | calc |
| 32 | l.293–305, Eq. (6) | BCS DOS N(E) = N₀E/√(E²−Δ²) | 150–152 Eq. `eq:xqp_bcsdos` | CARRIED | derived |
| 33 | l.306–317, Eq. (7) | thermal fraction | 153–157 Eq. `eq:xqp_xth` | CARRIED-CORRECTED (XQ13) | derived |
| 34 | l.317–326 | Bath A: e^(−141) ∼ 10⁻⁶¹ | 158–160 | CARRIED | calc |
| 35 | l.327–332, Eq. (8) | Bath B; T_gap = Δ/k_B = 2.11 K | 161–165 Eq. `eq:xqp_tgap` | CARRIED-CORRECTED (XQ17) | conjecture |
| 36 | l.333–335 | at T_gap thermal fraction of order unity | 165–166 | CARRIED (0.92 from asymptotic form, flagged out of range) | calc |
| 37 | l.336–337 | each erasure deposits Δ ln2, generating ln2 quasiparticles | — | EXCLUDED (XQ1) → yield 2 per broken pair, l.207–210 | — |
| 38 | l.337–343 | T_H = T_gap unique self-consistent choice; no free parameter | 166–171 | CARRIED | conjecture |
| 39 | l.344–350 | sharp parameter-free prediction; T_fridge models struggle | 172–176; 305–306 | CARRIED-CORRECTED (XQ4, XQ11) | prediction |
| 40 | l.351–359 | joint state φ and TLS; coupling g/2π 1–10 MHz vs 5 GHz | 179–182 | CARRIED | observed |
| 41 | l.360–368, Eq. (9) | δφ ∼ g/ω_q ∼ 1e-3 | 183 Eq. `eq:xqp_kick` | CARRIED-CORRECTED (XQ14) | calc |
| 42 | l.369–375 | perturbative; δφ does not enter the bound | 184–186 | CARRIED | — |
| 43 | l.376–380, Eq. (10) | τ_φ ∼ ħ/Δ ≈ 3.6 fs | 187 Eq. `eq:xqp_tauphi` | CARRIED-CORRECTED (XQ12) | calc |
| 44 | l.381–390, Eq. (11) | τ_φ/τ_TLS ≈ 1e-10 | 188–190 Eq. `eq:xqp_sep` | CARRIED-CORRECTED (XQ12) | calc |
| 45 | l.391–400 | independent events; Poissonian statistics (Ref. 11); Markov/Lindblad | 190–195 | CARRIED-CORRECTED (XQ19) | interp |
| 46 | l.401–408 | restoration discards one bit about TLS state | 196–199 | CARRIED | conjecture |
| 47 | l.409–416, Eq. (12) | E_erase = k_B T_gap ln2 = Δ ln2 | 200–203 Eq. `eq:xqp_erase` (126 µeV) | CARRIED | conjecture |
| 48 | l.417–427, Eq. (13) | N_QP = E/Δ = ln2; origin of ln2 | — | EXCLUDED (XQ1); Eq. `eq:xqp_erase` labelled a minimum heat, not a yield | — |
| 49 | l.428–434, Eq. (14) | E_L = k_B T_gap ln2 = Δ ln2 | 200 (same as Eq. 12, given once) | CARRIED | conjecture |
| 50 | l.435–448, Eq. (15) | Γ_qp = ln2/τ_TLS | 224–225 (Γ = Y/τ_TLS, Y = 2) | CARRIED-CORRECTED (XQ1) | derived |
| 51 | l.449–460, Eq. (16) | n_qp = ln2 τ_qp/(τ_TLS πλ³) | 211–213, 218–222 | CARRIED-CORRECTED (XQ2: island volume V, N sites) | derived |
| 52 | l.463–473, Eq. (17) | x_min = ln2 τ_qp/(n_cp τ_TLS πλ³) | 222 Eq. `eq:xqp` | CARRIED-CORRECTED (XQ6, XQ7) | derived |
| 53 | l.474–483, Eq. (18) | invariant = ln2; all five measurable on one device | 298 Eq. `eq:xqp_invariant` | CARRIED-CORRECTED (XQ7: = 2, quantities x, τ_qp, τ_TLS, N, V) | prediction |
| 54 | l.484–496 | parameter list (Δ, τ_TLS, τ_qp, n_cp = n_e/2, λ_L) | Table `tab:xqp_inputs` l.229–241 | CARRIED-CORRECTED (XQ6, XQ15, XQ16, XQ18) | calc/working |
| 55 | l.497–508, Eq. (19) | numerical x = 6.5e-8 | 243–259, Table `tab:xqp_sites` | EXCLUDED as printed (XQ6, XQ7) → sites table | calc |
| 56 | l.509–517 | factor 1.5; exact agreement at τ_TLS ≈ 20 µs; sensitivity to τ_TLS testable | 259–262, Fig. `fig:xqp_tau` | CARRIED-CORRECTED (sensitivity carried as N vs τ_TLS scaling) | calc |
| 57 | Fig. 1 (l.176–221) | temperature structure + junction inset + formula | Fig. `fig:xqp_ladder` (redrawn) | CARRIED-CORRECTED (formula 6.5e-8 removed; T_c added; volume = restoration region) | calc/observed |
| 58 | Fig. 2 (l.518–600) | x_min vs τ_TLS; floor band; τ_TLS band; 19.5 µs; T1 0.32 ms inset; invariant | Fig. `fig:xqp_tau` (redrawn, corrected model) + l.315–319 (T1) | CARRIED-CORRECTED (XQ6, XQ7, XQ15) | derived/calc |
| 59 | l.601–604 | Table I introduction; several previously unexplained | 274–276 | CARRIED-CORRECTED ("previously unexplained" not carried: each row has a standard reading, XQ8) | — |
| 60 | l.606–626, Eqs. (20)–(21) | invariant numerically 6.94e-5 s m³ / 100 µs = 0.694 | 299–300 (corrected example = 2.000) | CARRIED-CORRECTED (XQ7) | calc |
| 61 | Table I (l.630–688), 9 rows | observations, references, explanation, status | Table `tab:xqp_obs` 9 rows | CARRIED-CORRECTED (XQ8–XQ11, XQ19; quantitative row → N needed) | observed |
| 62 | l.689–702 | Prediction 1 same-device; x·τ_TLS = const | 295–304 | CARRIED-CORRECTED (x τ_TLS V/N = const) | prediction |
| 63 | l.703–709 | Prediction 2 T independence | 305–306 | CARRIED-CORRECTED (XQ4) | prediction |
| 64 | l.710–717 | Prediction 3 material ratio ∝ λ³ n_cp | 307–312 Eq. `eq:xqp_ratio` | CARRIED-CORRECTED (XQ5) | prediction |
| 65 | l.718–729, Eq. (22) | Γ₁ = α x ω_q, α ≈ 1; T1 ≈ 0.3 ms | 314–318 (+ Catelani form Eq. `eq:xqp_gamma1`, 0.24 ms) | CARRIED | calc |
| 66 | l.730–745 | ceiling endogenous; only material changes lower it (τ_TLS, λ³/n_cp) | 321–327 | CARRIED-CORRECTED (λ³ dropped, XQ5; N added) | conjecture |
| 67 | l.746–751 | shielding/underground/gap engineering will not improve T1 beyond the ceiling | 324–326 | CARRIED | conjecture |
| 68 | l.752–754 | best Al/AlOx, Nb/AlOx T1 ∼ 300 µs [20] within a factor of a few | 318–320 | CARRIED-CORRECTED (Bal 2024: Ta-capped Nb, median > 0.3 ms, max 0.6 ms; requires x ≤ 4e-8) | observed/calc |
| 69 | l.755–761 | one mechanism for background x_qp, TLS–QP coupling, equilibrium distribution | 329–333 | CARRIED-CORRECTED (XQ8: last two consistent with other sources too) | interp |
| 70 | l.762–773 | framework, not a proof; identifications derived and self-consistent; Prediction 1 decides | 333–337 | CARRIED-CORRECTED (erasure and bath are conjectures) | interp |
| 71 | Conclusion l.774–795, Eq. (23) | summary; formula 6.5e-8; nine observations; three predictions; Gran Sasso factor thirty; maintenance cost | 339–346 (Status) | CARRIED-CORRECTED (XQ6, XQ7, XQ9) | — |
| 72 | l.796–800 | acknowledgement; provisional patent applications | — | EXCLUDED (author rule: no patent or company material; acknowledgements are front/back matter) | — |
| 73 | l.801–854 | references [1]–[21] | bibliography: all 21 cited or mapped (Kittel → Kittel2005; Andersson [11] → Sundelin2026, same arXiv:2602.01945) | CARRIED | — |
| R1 | REFEREE_NOTE §1 | n_cp = 2ν₀Δ; 22,600× | 25–32, 214–216 | CARRIED | calc |
| R2 | REFEREE_NOTE §2 | Δ ln2 < 2Δ; yield 2 | 207–210 | CARRIED | — |
| R3 | REFEREE_NOTE §3 | diffusion 250–800 µm; island average | 211–213 | CARRIED | calc |
| R4 | REFEREE_NOTE model + table | x = 2Nτ_qp/(τ_TLS n_cp V); 60/600/6,000; 1.7e-9/-10/-11 | 222, Table `tab:xqp_sites`, Fig. `fig:xqp_sites` | CARRIED | derived/calc |
| R5 | REFEREE_NOTE "what stands" | idea compatible not established; test sharper; not parameter-free; Landauer sets rate and minimum heat, 2Δ sets yield | 55–58, 243–262, 294–304, 339–346 | CARRIED | conjecture |
| R6 | XQP_CHECK reproduced | T_gap 2.112 K; T1 0.32 ms; V_eff exact | 164, 125, 315–318 | CARRIED | calc |

## Exclusions (for the author)
1. Eq. (13) and every "ln 2 quasiparticles per event" (XQ1).
2. The printed x_min = 6.5e-8, Eqs. (17), (19), (23), the factor 1.5, "exact agreement at τ_TLS ≈ 20 µs / 19.5 µs", the invariant value ln 2 and its numerical check 0.694 (XQ6, XQ7). The corrected forms are carried.
3. "Using ξ₀ instead would give x ∼ 1e-11" (rests on the withdrawn per-site formula).
4. Acknowledgement and provisional patent numbers (author rule).
Nothing else is excluded; old values appear only in this manifest (for app_B), never in the chapter.

## Notes for the lead
- `appendices/app_I_provenance.tex` l.257 refers to "p3_02_xqp.tex, line 72" (the sites table); it is now at l.249–255 (`tab:xqp_sites`).
- `appendices/app_E_formulas.tex` entries 112–114 point to sections "The anomaly", "From events to quasiparticles", "The test": all three titles kept.
- New figures need rows in `app_I_provenance.tex` (`fig:xqp_ladder`, `fig:xqp_london`, `fig:xqp_tau` → `figscripts/fig_p3_xqp_book.py`).
- `iam.bib` entry `Kurter2022` has no DOI: 10.1038/s41534-022-00542-2 (CrossRef-checked). `DeDominicis2026` is the arXiv:2405.18355 record.
- Proposed errata rows XQ9–XQ19 above for `PAPER_ERRATA.md` and `appendices/app_B_errata_physics.tex`.
- Static checks run against HEAD e37aabb: braces balanced, environments balanced, 40 labels all unique across the book, every \ref resolves, 31 cite keys all in iam.bib or bib_xqp.bib, 5 figures exist, all floats [htbp]. Not compiled (no TeX in sandbox).
