# MANIFEST — line-for-line carriage of the entropic-gravity note

**Source (version of record):** `docs/papers/A_Note_on_Entropic_Gravity__Saridakis_.pdf` (March 2026, 10 pp). pypdfium2 text with one `=== PAGE` marker per page:
**317 lines** (ledger `docs/book/PAPER_LINE_COUNTS.md` row 5: 317 — match). Read 1–317 in 50-line chunks (1–50, 51–100, 101–150, 151–200, 201–250,
251–300, 301–317), no chunk truncated. LaTeX base `docs/papers/latex/IAM_Saridakis_Bridge/IAM_Saridakis_Bridge.tex` (558 lines) used through the
mechanically prepared `coverage/wave2/05_A_Note_on_Entropic_Gravity.tex` (366 lines, read 1–366 in 50-line chunks) for the exact equations; the one
PDF-only sentence (Tsallis entropy) is merged.

**Other files read for consistency (ranges used):** `PAPER_ERRATA.md` rows EG1–EG8 and N1; `verification/theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` (17 lines, all);
`verification/theory/THEORY_CHECK.md` (all); `part2/p2_03_theory.tex` l.248–287, 365–503, 589–654; `part2/p2_04_dualsector_chains.tex` l.45–99;
`part2/p2_07_late_time_growth.tex` l.172–211; `part2/p2_17_lensing_dynamics.tex` (89 lines, all); `part2/p2_02b_virial_tests.tex` l.30–40, 103–112;
`part2/p2_20_wz_far_future.tex` l.36–44; `verification/scripts/verify_euclid_template_output.txt` (all); `verification/scripts/verify_theory_paper.py` l.1–75.

## Files delivered
| File | Purpose |
|---|---|
| `docs/book/part2/p2_03a_entropic_gravity.tex` | NEW chapter `ch:entropicgravity` (320 lines) |
| `docs/book/figscripts/fig_p2_entropic_gravity.py` | draws `figures/part2/fig_eg_forms.pdf/.png` on `_bookstyle.py` (overlap check 0) |
| `docs/book/figures/part2/fig_eg_forms.pdf`, `.png` | Fig. `fig:eg_forms` |
| `docs/verification/scripts/verify_entropic_gravity.py` + `_output.txt` | every equation (sympy) and number of the chapter |
| `docs/book/bib_entropic_gravity.bib` | 4 new entries, CrossRef-verified |
| `docs/book/coverage/MANIFEST_entropic_gravity.md` | this file |

## main.tex lines (for the lead)
```
\input{part2/p2_03_theory}
\input{part2/p2_03a_entropic_gravity}     % NEW, directly after the theory chapter (main.tex l.23 -> new l.24)
```
and the bibliography line (main.tex l.115): `\bibliography{iam,bib_entropic_gravity}` — or merge the 4 entries into `iam.bib` (keys checked: none already in iam.bib).

## Equation, step, table, figure and claim table
Paper location = PDF text line (page). Book location = `p2_03a_entropic_gravity.tex` line unless stated.

| Paper loc. | Content | Book loc. | Verdict | Status |
|---|---|---|---|---|
| l.12–23 (p1) Abstract | IAM extends Jacobson–Cai–Kim; source not functional; single friction term; β_m = Ω_m/2 from virial; E(a) from decoherence integral | 8–17 (opening), carried in body | CARRIED (abstract content distributed; no separate abstract in a book chapter) | — |
| l.25–32 §1 | Friedmann eqs as thermodynamic equations of state from the first law on the horizon | 20–22 | CARRIED | \interp |
| l.33–36 | A_H = 4π/H², T_H = H/2π, S_BH = A_H/4 | 24–26 | CARRIED (G restored: A_H/4G; A_H/4 in Planck units) | — |
| l.37–39 Eq. (1) | first law dE = T_H dS_BH − W dV recovers Friedmann exactly | 27–43: four-step derivation Eqs. eg_flux, eg_dS, eg_firstlaw, eg_Hdot, eg_friedmann; unified form dE = T_h dS + W dV with T_h = κ/2π l.41–43 | CARRIED-CORRECTED (found in this check, sympy section 1: the printed combination T_H = H/2π with −W dV does not return Ḣ = −4πG(ρ+P); the Cai–Kim flux form −dE = T_H dS does, and so does the unified form with +W dV and the moving-horizon temperature κ/2π, Akbar & Cai 2007). **Author/lead: `p2_03_theory.tex` l.255 (sec:source) prints the same “dE = T_H dS − W dV”; not edited (not my file).** | \derived |
| l.39 | field equations are thermodynamic identities [Jacobson] | 45 | CARRIED (author's own summary wording “equations of state”, l.255–256) | — |
| l.40–42 | shared by IAM, Barrow, Tsallis, Kaniadakis | 47–49 | CARRIED (+ Kaniadakis cosmology cited, Lymperis 2021) | — |
| l.44–45 §2 | two natural ways beyond ΛCDM | 53 | CARRIED | — |
| l.46–51 §2.1 | Barrow S = (A/A0)^{1+Δ/2}, Δ fractal | 56–59, Eq. eg_barrow | CARRIED (+ Barrow 2020 original cited beside Saridakis 2020) | — |
| l.51–53 | Tsallis S = γA^δ, δ long-range correlations (PDF-only sentence) | 59–62, Eq. eg_tsallis | CARRIED (+ Tsallis–Cirto 2013) | — |
| l.53–55 | modification to geometric term; Δ, δ constrained by data | 62–64 | CARRIED (+ limits Δ = 0, δ = 1 return the area law) | — |
| l.56–58 | well-motivated; Planck-scale horizon | 65–67 | CARRIED | \interp |
| l.60–67 §2.2 Eq. (2) | S_total = S_BH + S_info, from gravitational decoherence | 69–72, Eq. eg_stotal | CARRIED | \conjecture |
| l.68–69 | decoherence via gravitational gradients, branching | 74–75 | CARRIED (+ Zurek 2003, listed in the references) | \interp |
| l.69–71 | k_B ln2 per bit; Bérut 2012, Jun 2014 | 75–76 | CARRIED | \observed |
| l.71–72 | encoded on horizon at T_H | 76–79, Eq. eg_bitcost (bit cost ħH ln2/2π, as eq:th:bitcost) | CARRIED | \conjecture |
| l.73–75 | S_info is a source term, not a modification | 81–82 | CARRIED | — |
| l.77–78 §3 | not taxonomic; observational consequences | 86–87 | CARRIED | — |
| l.79–81 | Barrow/Tsallis alter Friedmann; Δ/δ affect H(z) and growth | 88–90 | CARRIED | — |
| l.82–85 | DESI DR2: compatible, neither resolves H0, both disfavoured by information criteria [Luciano 2025] | 90–93 | CARRIED-CORRECTED (published abstract, CrossRef 10.1088/1475-7516/2025/09/013: best-fit Δ negative; Δ = 0 within 2σ for 3 of 4 data sets; one combination H0 = 72.2 ± 0.9, “may potentially alleviate” the tension; AIC and Bayes evidence “slightly” favour ΛCDM). Not an errata row: **author to confirm**; consistent with `p2_03_theory.tex` l.260–261. | \observed |
| l.86–87 | S_info enters perturbations only, as extra friction | 95–96 | CARRIED | — |
| l.88–98 Eq. (3) | δ'' + (2 + β_mE)Hδ' − (3/2)Ω_mH²δ = 0, cosmic time | 97 Eq. eg_growth (cosmic time, 4πGρ̄_m = (3/2)Ω_m(a)H² written out); 98–100 Eq. eg_growthN (e-fold form, derivation of the change of variable) | CARRIED-CORRECTED (EG1: the form stated explicitly; Ω_m(a) not Ω_m0 with H²) | \derived |
| (EG1) | third implementation; numbers differ by form | 104–113 + Fig. eg_forms (112–119) | MISSING → WRITTEN: friction form −1.64 % in D today; Level 1 μG −0.78 %; Level 2 2H_IAM −0.67 %; friction coefficients 2.158 vs 2.152 H0 (verify section 5) | \calc |
| l.98–99 | background ΛCDM; photon sector unaffected | 101–102 | CARRIED | — |
| l.100–106 | timelike vs null worldlines; Σ = 1 from geometry | 121–125 | CARRIED (wording “lensing-potential modification” → “change to the lensing response”: rule flag in prepared file) | \conjecture (eligibility), \derived |
| l.107–108 | μ < 1 (suppressed growth) from the friction term | 125–131 | CARRIED + clarified: 1 − μ = 13.62 % is the coupling deficit, not growth; growth −0.7 to −0.8 % in D, fσ8 −4.25/−2.17/−1.35/−0.41 % at z = 0/0.3/0.5/1 (verify_euclid_template_output) | \derived \calc |
| l.110 §4 | β_m not free, from virial theorem | 133 | CARRIED | — |
| l.111–115 Eq. (4) | 2⟨T⟩+⟨V⟩ = 0 ⇒ ⟨T⟩ = ½|⟨V⟩| | 134–139, Eq. eg_virial, with the derivation (virial G, Euler's theorem, k = −1) | CARRIED (derivation added) | \derived |
| l.116 | exactly half in kinetic channel | 140 | CARRIED | \derived |
| l.116–119 | kinetic channel = decoherence channel | 140–142 | CARRIED | \conjecture |
| l.119–120 | Ω_m/2, β_m = 0.1575 | 142–145, Eq. eg_beta | CARRIED-CORRECTED (book canon Ω_m = 0.3153: β_m = 0.15765) | \derived \calc |
| l.121–125 | six N-body groups, ⟨η⟩ = 0.815 ± 0.025, β_m = 0.159 ± 0.010, 1.1 % agreement | 147–150 | CARRIED-CORRECTED (EG2: the cited studies report 2T/|U| ≈ 1.15–1.25; the measurement claim and its six citations are not carried; the corrected fact cites Neto 2007, Power 2012) | \observed |
| l.125–126 | ½ partition from hydrogen to clusters, 37 orders | 150–153 (→ ch:virial_law, sec:vi_evidence) | CARRIED | — |
| l.127–129 | Planck posterior recovers β_m to 0.2σ; “most compelling result” | 146–150 | CARRIED-CORRECTED (EG3: β_m is fixed in every chain; the test is free μ0 and future growth) | \prediction |
| l.131–134 §5 | E(a) from decoherence integral weighted by T_H, A_H | 157–161 (compact steps, full derivation sec:th:activation) | CARRIED (derivation given once in full in ch:theory; steps restated) | \derived |
| l.135–140 Eq. (5) | E(a) = exp(1 − 1/a) | 162, Eq. eg_Ea | CARRIED | \derived |
| l.144–148 | two boundary conditions E(1) = 1, E → e | 165–171 | CARRIED (+ within exp(C − k/a): C = k = 1, sympy section 4; n = 7/2 gives k = 1 independently; full-ΛCDM fit exp(0.93 − 1.02/a)) | \derived \calc |
| l.149 | E(1)/e = 1/e ≈ 36.8 % | 174 | CARRIED | \calc |
| l.149–150 | today at the inflection, production at its peak | 174–177 | CARRIED-CORRECTED (EG4: per e-fold only; per unit time peak z = 1.26; per unit a at a = 1/2) | \calc |
| l.150–152 | six analyses n_eff = 3.22 ± 0.44, range 3.0–3.5 | 158–160 | CARRIED-CORRECTED (EG4 / THEORY_CHECK #1, #3: the exponent is n = 7/2; the six-analysis value has no traceable source and is not printed) | \derived |
| l.153–156 §6 | Table 1 introduced; where new physics enters | 180–182 | CARRIED | — |
| l.157–191 Table 1 | 8 rows: foundation, modification, free parameters, background, perturbations, μ–Σ, H0 tension, AIC | 184–204, Table eg_compare | CARRIED-CORRECTED: H0 row → Luciano 2025 abstract (Barrow/Tsallis) and the two sector rates 67.16 ± 0.47 / 72.26 (EG6); AIC row → “ΛCDM slightly favoured” (Luciano 2025) and ΔAIC = Δχ²_min = +0.54 to +1.73. μ–Σ row for the deformed laws carried as written, labelled \interp in the caption — **author to confirm** (not an errata row; in Barrow/Tsallis cosmology the value depends on how the deformation enters the perturbations). | \interp |
| l.195–199 | AIC paragraph | 207–212 | CARRIED (+ ΔAIC = Δχ² + 2Δk written out, IAM values) | \calc |
| l.200–204 §7 | why the exponent; Δ, δ measured | 214–217 | CARRIED | — |
| l.205–209 | incomplete accounting of sources; BH entropy may be exact | 218–221 | CARRIED | \conjecture |
| l.210–212 | open question; narrower point | 223–226 | CARRIED | \openprob \interp |
| l.215–218 §8.1 | 17 chains (12 + 3 + 2), R − 1 < 0.01 | 230–235 | CARRIED-CORRECTED (EG5: 18 chains = 12 + 3 + 2 + 1 baryon; Level 2b drives H0 to 61.45/61.52; R − 1 ≤ 0.010, sec:chains) | \measured |
| l.222–232 Table 2 | 6 rows μ0, Σ0, σ8, H0^matter, Δχ², β_m | 237–254, Table eg_numbers | CARRIED-CORRECTED (EG6 + trace): μ0 data 0.04 ± 0.22 (DESI DR1 FS+BAO+CMB+DES Y3, DESI 2024 VII abstract) and 0.02 ± 0.19 (ACT+WMAP+SDSS+SN, Andrade 2024 abstract) — the printed “0.05 ± 0.22” does not appear in either source; Σ0 data 0.044 ± 0.047 (DESI 2024 VII) and 0.021 ± 0.068 (Andrade 2024) — the printed “0.008 ± 0.045” is not found; σ8 IAM 0.7998 ± 0.0058 Level 2 chain (\measured, not derived) vs 0.802 +0.022/−0.018 (Stölzner 2025, as in p2_02/p2_09); H0^matter 72.26 (EG6); Δχ² +0.54 L2, +0.56…+1.73 L1, “18 chains”; β_m row: N-body column removed (EG2), “fixed in every chain”. | as in table |
| l.234–238 §8.2 item 1 | Euclid σ(μ0) ∼ 0.04; uniqueness; f(R) μ>1, Σ>1; DGP μ>1; Horndeski Σ ≠ 1 | 258–265 | CARRIED-CORRECTED: EG7 (f(R) Σ = 1; sDGP μ < 1, Σ = 1 with ghost; “unique” → signature among viable models; Koyama 2007, Fang 2008); Euclid sensitivity only as sec:lt_euclid (template-equivalent μ0 ≈ −0.07; 0.3σ / 1.8σ / ~7σ) | \calc \prediction |
| l.239–243 item 2 | DESI DR2 fσ8 consistent without w < −1; phantom crossing at z ∼ 0.5 as growth–geometry tension | 266–271 | CARRIED-CORRECTED (EG6: DESI DR1 full shape, χ² 5.24 vs 4.51; DR2 preference from distances alone, sec:vt_phantom; test = two-ruler, ch:sectortension) | \calc \conjecture \openprob |
| l.244–247 item 3 | M_lens/M_dyn = 1/μ_eff(z) > 1, → 1 at high z; distinct from constant hydrostatic bias; eROSITA, Planck SZ, WL | 272–277 | CARRIED + values 1.158/1.105/1.055/1.018/1.002; Level 1 form only, Level 2 gives 1 (sec:ld_hold) | \calc \prediction \openprob |
| l.248 | Euclid μ consistent with zero at σ ∼ 0.04 rules IAM out | 278–281 | CARRIED-CORRECTED (sec:lt_euclid: at 0.04 the separation is 1.8σ; exclusion at ~7σ needs the one-per-cent case) | \prediction |
| l.248–250 | “The author is not aware of any…model predicting μ<1, Σ=1 from first principles” | 259–262 | CARRIED-CORRECTED (EG7 / THEORY_CHECK #10; stand-alone rule: no “the author”) | — |
| l.255–256 §9 | shared foundation | 284–285 | CARRIED | — |
| l.257–261 | IAM's contribution; β_m recovered by N-body and Planck to 0.2σ | 287–290 | CARRIED-CORRECTED (EG2, EG3) | — |
| l.262–266 | Δχ² +0.54; σ8 0.800; H0 endpoints within 1σ; no AIC penalty | 291–294 | CARRIED (+ −0.37σ, −0.75σ, 0.12σ; verify sections 3, 7) | — |
| l.267–270 | Euclid test; repository | 296–298 | CARRIED-CORRECTED (Euclid as sec:lt_euclid; “Zenodo DOI and GitHub repository listed on the title page” → “in the repository”, stand-alone rule) | — |
| l.271–278 | Acknowledgements naming a researcher | — | EXCLUDED (EG8 / N1 names rule); his papers are cited: Saridakis 2020 (BHDE), Saridakis et al. 2018 (THDE), Luciano et al. 2025, Lymperis et al. 2021 | — |
| l.282–316 | References (17) | bib | Cited where used: Bérut, Cai–Kim, Jacobson, Jun, Luciano, Neto, Planck 2018 VI, Power, Riess, Saridakis 2018, Saridakis 2020, Stölzner, Zurek. **Not cited** (only supported the withdrawn N-body η claim, EG2): Bett 2007, Bryan & Norman 1998, Klypin 2016, Ludlow 2010. | — |
| — | Status table | 298–320 | NEW (book convention) | — |

Equations: paper 5 numbered (1)–(5): all carried (Eqs. 1 and 3 corrected). Tables: 2 of 2 reproduced (corrected). Figures: paper has none; 1 new figure (EG1).

## Exclusions (for the author)
1. Acknowledgements naming a researcher (EG8, N1).
2. Six-group N-body ⟨η⟩ = 0.815 ± 0.025 ⇒ β_m = 0.159 ± 0.010 and its citations (EG2).
3. “Planck posterior recovers β_m to 0.2σ … single most compelling result” (EG3).
4. n_eff = 3.22 ± 0.44 from six analyses (EG4; THEORY_CHECK #3).
No speculative section was dropped: §7 is carried in full as \conjecture / \openprob / \interp; the phantom-crossing interpretation is carried as \conjecture.

## Corrections not in PAPER_ERRATA (flag for the author / lead)
- **Eq. 1 sign/temperature** (sympy, verify section 1). Same text in `p2_03_theory.tex` l.255 — lead to decide there.
- **Luciano et al. 2025 summary** corrected to the published abstract (H0 72.2 ± 0.9 in one combination; ΛCDM “slightly” favoured).
- **Table 2 data values** μ0 0.05 ± 0.22 and Σ0 0.008 ± 0.045 not found in DESI 2024 VII or Andrade 2024; replaced by the abstracts' values (EG6 said “trace”).
- `iam.bib` entry `Riess2022` has no DOI (10.3847/2041-8213/ac5c5b) — not edited.

## Static checks (2026-10-03, against HEAD 1808034)
Braces balanced; environments matched; 22 labels, none duplicated anywhere in docs/book (coverage/ excluded); 39 distinct \ref/\eqref targets all resolve;
22 cite keys all in iam.bib or bib_entropic_gravity.bib; 1 figure exists; floats [htbp]; no “paper / the author / this book / the quantum-processor report / the semiconductor report / the methylation report / the cell-reading engine”.
Not compiled (no TeX in sandbox).

## Citations checked (CrossRef, DOI → title/journal/volume/pages/year/authors)
New: Luciano2025, Saridakis2018tsallis, Saridakis2020barrow, Lymperis2021 (+ Bryan1998 checked, not used). Existing iam.bib DOIs re-checked: Barrow2020,
TsallisCirto2013, CaiKim2005, Jacobson1995, Zurek2003, Berut2012, Jun2014, Planck2018VI, Stolzner2025, DESI2024VII, Andrade2024, Neto2007, Power2012,
PogosianSilvestri2016, Koyama2007, Fang2008, AkbarCai2007. Data quoted from the abstracts of Luciano2025, DESI2024VII, Andrade2024 (CrossRef abstract field).
