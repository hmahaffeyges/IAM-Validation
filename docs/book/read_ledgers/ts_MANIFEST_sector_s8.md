# MANIFEST — line-for-line carriage of "Dark Energy or Sector Tension?" and "The Redshift-Dependent S8 Trend" (2026-10-03)

Base: repo HEAD 12d8fb1. Nothing pushed or committed.

## Files delivered
| file | status | main.tex |
|---|---|---|
| `docs/book/part2/p2_08_s8_trend.tex` | rewritten, 69 -> 207 lines (`ch:s8trend`) | unchanged (l.28) |
| `docs/book/part2/p2_09_sector_tension.tex` | rewritten, 87 -> 305 lines (`ch:sectortension`): DE paper §1–4 | unchanged (l.29) |
| `docs/book/part2/p2_09b_phantom_crossing.tex` | NEW, 162 lines (`ch:phantom`): DE paper §5–7 | **add** `\input{part2/p2_09b_phantom_crossing}` right after `\input{part2/p2_09_sector_tension}` (done in the delivered main.tex) |
| `docs/book/bib_sector_s8.bib` | NEW: DESI2024VI, HuSawicki2007, BelliniSawicki2014 (DOIs checked on CrossRef) | **add** `bib_sector_s8` to `\bibliography{iam,...}` (l.116; not done here) |
| `docs/book/figscripts/fig_p2_sector_s8.py` | NEW, on `_bookstyle.py` + `_cosmo.py` | — |
| `docs/book/figures/part2/fig_mu_evolution.{pdf,png}`, `fig_sector_phantom.{pdf,png}`, `fig_s8_trend_iam.{pdf,png}` | NEW | — |
| `docs/verification/scripts/verify_sector_tension.py` + `_output.txt` | extended (sections 6–14), rerun | — |
| `docs/verification/scripts/verify_s8_trend.py` + `_output.txt` | extended (sections 7–9), rerun | — |

## Reading ledger
| source | lines | read |
|---|---|---|
| `docs/papers/Dark_Energy_or_Sector_Tension.pdf` (pypdfium2 text) | 934 (ledger 934) | 1–934, chunks ≤150 lines shown in full |
| `docs/papers/The_Redshift_Dependent_S_8_Trend_in_the_Context_of_IAM.pdf` | 323 (ledger 323) | 1–323 in full |
| `coverage/wave2/23_Dark_Energy_or_Sector_Tension.tex` (author LaTeX, flags) | 1124 | 1–1124 in full (331–640 re-read after a truncated preview) |
| `coverage/wave2/22_S8_Trend.tex` | 361 | 1–361 in full |
| PAPER_ERRATA rows SX1–SX10, ST1–ST7, S1–S10, T13, L22; SECTOR_TENSION_CHECK.md (27); S8_TREND_CHECK.md (22); TWO_RULER_DESI_TEST.md (13) | — | in full |
| existing p2_08 (69), p2_09 (87); p2_07 l.8–16, 143–201; p2_06 l.100–150 | — | in full |
| primary sources for values: DESI 2024 V (arXiv:2411.12021) §7.1, Table 9, Fig. 14, App. A (A.1–A.12), Table 11; eBOSS DR16 cosmology Table III; KiDS-Legacy joint (Stölzner) abstract; growth-index (Nguyen) abstract; trend paper (MNRAS 528 L20) pp. 1–3; DESI 2024 VI abstract | — | the parts used, in full |

## New findings for the author (not in PAPER_ERRATA yet)
1. **SX9 traced.** Table 3's six DESI values (0.377, 0.514, 0.484, 0.422, 0.377, 0.435 ± 0.094 … 0.045) are exactly DESI 2024 V Appendix A `fσ_s8` (0.377174 … 0.434858; σ = √C33). They are fσ_s8 (template σ_s8), which DESI uses for inference; DESI's fσ8 (Fig. 14) is "for visualization only". The book carries them labelled fσ_s8 with that caveat.
2. **Fig. 2 centre panel / §5.3 "DESI DR1 DM/rd measurements (Table 11)": Table 11 of DESI 2024 V is the FIDUCIAL (Planck-ΛCDM template) distances**, so "points sit on the prediction curve with no offset" compares ΛCDM with itself. Redrawn with the measured ShapeFit-only D_V/r_d of Appendix A (LRG2 low, as DESI reports).
3. **The S8 trend is RSD-based.** MNRAS 528 L20 infers S8 from fσ8(z) with an Ωm prior and z_min cuts (not from weak lensing). Book adds the correct counterpart: the same z_min analysis on the term's own growth gives S8 0.8231 → 0.8306 (rise 0.3σ of the data error). \calc
4. **γCDM is not an approximation to the term at low z** (paper §7): γ = 0.633 lowers fσ8 by 11.4 % today vs the term's 4.25 %; they differ at all z < 1.
5. **Citation fixes:** Ó Colgáin & Sheikh-Jabbari MNRAS Lett. 542 L24 is doi 10.1093/mnrasl/slaf042 (paper prints slaf055, which is another article); arXiv:2504.04417 is now Phys. Dark Univ. 52, 102268; DESI DR1 BAO cosmology is DESI 2024 VI, doi 10.1088/1475-7516/2025/02/021 (paper prints 2025/07/021, another article); Freedman TRGB doi 10.3847/1538-4357/ac7c74 in the paper is the Astropy paper (book uses Freedman2025 key); KiDS-1000 S8 = 0.766 is Heymans 2021 (3×2pt), Asgari 2021 cosmic shear is 0.759; eBOSS LRG is not de Mattia (ELG); HSC Y3 0.776 ± 0.032 is Dalal (power spectra), Li is 0.769; arXiv:2402.04767 is a review (Universe 10, 305), not the ΛsCDM model paper; the quote attributed to the trend paper ("at best, a simple and effective approximation") is not in its text — not carried.
6. DES Y3 area ≈ 4100 deg² (paper 5000 = full DES footprint). E(a) < 0.1 for z > 2.30 (paper Fig. 3: 2.2).
7. ODE σ8: the reduced equation lowers σ8 by 0.78 % vs Level 2's 1.10 % (0.8024 vs 0.7998); the paper's σ8^ODE = 0.8048 is not reproduced.
8. Level 2 shifts: ln10¹⁰A_s +0.09σ, Ωm +0.05σ, S8 −0.78σ (paper: logA +0.10σ, Ωm +0.06σ / 0.06σ, S8 0.73σ).
9. Cross-file: `appendices/app_E_formulas.tex` l.16 lists ch:sectortension as having no displayed equation — it now has Eqs. st_entropy, st_hm, st_mu, st_ode, st_sigma8, st_S8 (and ch:phantom has ph_zcross). `appendices/app_I_provenance.tex` l.55 row `fig:w0wa` → file is now `fig_sector_phantom.pdf` (panel c); a `\label{fig:w0wa}` alias is kept on that figure so the reference resolves. app_I l.231–232 cite "p2_09 line 13/40": those lines moved (DR2 table now l.57–67; DESI χ² sentence l.243). `fig_w0wa.pdf` no longer included anywhere.
10. The DESI-χ² values 4.51/5.24 and SDSS 6.19/6.95 (cited by p2_02b l.38) are kept in p2_09 l.243 as they stood; they were NOT re-run here (their fσ8 conversion is from DESI Table 9 ratios). The new diagonal χ² from Appendix A is 3.84/3.81 (DESI) and 6.61/6.53 (legacy).

## Carriage table — "Dark Energy or Sector Tension?" (DE). PDF line numbers from det.txt.
| paper location | content | book location | verdict | label |
|---|---|---|---|---|
| Abstract ¶1 (l.12–19) | DESI w0>−1, wa<0, crossing; question; perturbation-only term | p2_09:15–20 | CARRIED-CORRECTED (crossing 0.35–0.50, SX1) | — |
| Abstract ¶2 (l.20–24) | μ<1, Σ=1; decoherence origin; βm = 0.1577 "recovered at 0.2σ" | p2_09:18–20 | CARRIED-CORRECTED (βm 0.15765 fixed, posterior consistent; SX5) | — |
| Abstract ¶3 (l.25–30) | sector split; single fluid must infer w≠−1 | p2_09:22–25 | CARRIED-CORRECTED (lensing amplitude follows δm, SX4) | — |
| Abstract ¶4 (l.31–37) | data confronted | p2_09:26–29 | CARRIED-CORRECTED (KiDS values/cites, ST7) | — |
| Abstract ¶5 (l.38–48) | Δχ² +0.54, 17 chains; σ8 0.1σ; crossing ↔ 7–8 %; Euclid 3.4σ; DESI Y5 | p2_09:29–35 | CARRIED-CORRECTED (18 chains SX8; 1−μ not growth SX2; Euclid per sec:lt_euclid; DR2 not produced SX3) | \fitted \calc |
| Keywords (l.49–50) | keywords | — | EXCLUDED (front matter, not chapter content) | — |
| §1 ¶1 (l.82–87) | ΛCDM success; two tensions persist | p2_09:37–39 | CARRIED | \interp |
| §1 ¶2 (l.88–96) | H0 67.4±0.5 vs 73.04±1.04, 4–5σ; TRGB, time delays | p2_09:41–45 | CARRIED-CORRECTED (4.9σ computed; TRGB 70.39±1.94) | \observed \calc |
| §1 ¶3 (l.97–104) | S8 tension: KiDS-1000 0.766, DES 0.776, HSC 0.776, Planck 0.832; KiDS-Legacy σ8 0.802 | p2_09:46–52; Table st_s8 | CARRIED-CORRECTED (0.766 = Heymans 3×2pt; Asgari 0.759; HSC Dalal; KiDS-Legacy 0.815; joint S8 0.814, σ8 0.802 with Pantheon+; ST7) | \observed |
| §1 ¶4 (l.105–115) | DR1 2.6σ; DR2 2.8–4.2σ; crossing; canonical scalar | p2_09:53–72, Table st_dr2 | CARRIED-CORRECTED (DR1: 2.6σ CMB, 2.5/3.5/3.9σ with SNe; DR2 rows SX1) | \observed \interp |
| §1 ¶5 (l.116–122) | two consistency analyses; no mechanism | p2_09:68–72 | CARRIED (cited by journal/arXiv only, SX10) | \observed |
| §1 ¶6 + list (l.123–139) | three predictions from one βm | p2_09:74–85 | CARRIED-CORRECTED (2: 13.6 % is 1−μ, growth deficits; 3: conjecture, tested in ch:phantom) | \prediction \conjecture |
| §1 ¶7 (l.140–147) | organisation of paper | — | EXCLUDED (paper navigation; replaced by chapter structure) | — |
| §2.1 (l.149–172) | Jacobson, Cai–Kim, S = A/4G; structure formation → information; Landauer; holography; Eq. 1; perturbation-only | p2_09:88–105, Eq. st_entropy | CARRIED-CORRECTED (units A/4ℓ_P², S9) | \interp |
| §2.2 (l.175–209) | worldline argument; Eq. 2 H̃m²; Eqs. 3–4 μ, Σ; photon observables | p2_09:106–122, Eqs. st_hm, st_mu | CARRIED-CORRECTED (βmE(a)H0², S9/T10; lensing amplitude SX4) | \interp \derived |
| §2.3 (l.210–227) | βm from 1:1 virial partition, Eq. 5; E(a) origin; "recovered to 1 % by Sheth–Tormen" | p2_09:123–128, Eq. s8_beta | CARRIED; ST claim EXCLUDED (ST5, SECTOR_TENSION_CHECK #11) | \interp |
| §2.4 Eq. 6 (l.234–243) | μ(0)=1/1.1577=0.864; "13.6 % growth suppression"; 7.9/3.3/0.6 % | p2_08:61–74 (derivation, Eq. s8_mu0); p2_09:129–133, Table st_mu | CARRIED-CORRECTED (coupling deficit, growth deficit column, SX2) | \derived \calc |
| §2.4 ¶ (l.244–247) | crossing redshifts fall where suppression 8→5 % | p2_09b:43–48 | CARRIED-CORRECTED (z_cross 0.35–0.50; 1−μ 5–7 %, growth 1.4–1.9 %; weight withdrawn by SX3) | \calc \conjecture |
| Table 1 (l.248–258) | μ at six tracers | Table st_mu (p2_09:134) | CARRIED (+ growth, σ8(z), E_G columns) | \calc |
| Fig. 1 (l.261–331) | μ(z), 1−μ, crossing lines | fig:mu_evolution (p2_09:144), redrawn | CARRIED-CORRECTED (crossing lines from corrected DR2; growth deficit added) | \calc |
| §2.5 ¶1–2 (l.333–342) | 17 chains; L1 12 chains, Δχ² ≤ +2.32, σ8 shift −0.013±0.001 | p2_09:151–158 | CARRIED-CORRECTED (18; +0.56 to +1.73 with likelihood ratio, SX8) | \fitted |
| §2.5 ¶3 (l.343–348) | L2 Run A vs C; σ8; logA +0.10σ, Ωm +0.06σ; 0.3σ threshold | p2_09:159–162 | CARRIED-CORRECTED (+0.09σ, +0.05σ from the chain table) | \fitted |
| §2.5 ¶4 (l.349–352) | βm hardcoded; Ωm/2 = 0.1583 ± 0.0033 at 0.2σ "a posteriori confirmation" | p2_09:163–165 | CARRIED-CORRECTED (consistency check, SX5) | \calc |
| Table 2 (l.354–380) | parameters and L2 results | Table st_params (p2_09:166) | CARRIED-CORRECTED (βm 0.15765; Ωm +0.05σ; S8 −0.78σ; 18 chains; μ0 free → tab:lt_free values; "IAM at 0.9σ" dropped) | \derived \calc \fitted |
| §3.1 (l.382–395) | DESI ShapeFit-only App. A; legacy 6dF, MGS, BOSS, eBOSS | p2_09:190–198 | CARRIED-CORRECTED (fσ_s8 and DESI caveat; eBOSS cites → Alam2021 Table III) | \observed |
| §3.2 Eq. 7 (l.396–425) | growth ODE, ICs, DOP853, fσ8 normalisation | p2_09:199–216, Eq. st_ode, with three-step derivation | CARRIED (derivation added) | \derived |
| §3.2 ¶ (l.426–433) | ODE simplified; σ8^ODE 0.8048 (+0.6 %) | p2_09:217–220 | CARRIED-CORRECTED (0.78 % vs 1.10 %; 0.8024; finding 7) | \calc |
| §3.3 ¶ (l.434–440) | pulls −0.87…+1.42 / −1.02…+1.34; 8–13 %; ≲2 % | p2_09:240–244 | CARRIED-CORRECTED (recomputed −0.88…+1.40 / −1.01…+1.36; χ²; 2.5/1.7 %) | \calc |
| §3.3 ¶ (l.441–447) | redshift dependence, not constant rescaling; DESI Y5 | p2_09:246–250 | CARRIED | \prediction |
| Table 3 (l.450–469) | 13 rows obs, σ, IAM, ΛCDM, pulls | Table st_fsig8 (p2_09:223) | CARRIED-CORRECTED (recomputed; DESI values = fσ_s8, SX9 traced) | \observed \calc |
| §4.1 list (l.471–493) | KiDS-1000, KiDS-Legacy, DES Y3, HSC Y3, Planck lensing | p2_09:252–262 | CARRIED-CORRECTED (ST7; areas; "lensing photon-sector, no S8 needed" → σ8Ωm^0.25 comparison) | \observed |
| §4.2 Eqs. 8–9 (l.496–513) | σ8, S8 L2; "at Planck Ωm"; logA/Ωm shifts | p2_09:263–275, Eqs. st_sigma8, st_S8 | CARRIED-CORRECTED (S8 at chain Ωm; shifts) | \fitted \calc |
| §4.3 (l.514–530) | 0.1σ; surveys below; between Planck and surveys; larger shift needs μ0 or βγ < 8.5e-6; Σ distinction; CMB lensing no change | p2_09:276–305 | CARRIED-CORRECTED (βγ/βm < 0.025, SX6/S2; CMB lensing −0.08 %, SX4) | \calc \measured \prediction |
| Table 4 (l.544–568) | S8 compilation, tensions | Table st_s8 (p2_09:281) | CARRIED-CORRECTED (values/cites ST7; offsets recomputed) | \observed \calc |
| §5.1 (l.532–579) | sector lists; single-fluid conflict → phantom crossing | p2_09b:10–28 | CARRIED-CORRECTED as \conjecture (SNe matter ruler, author ruling; κ amplitude → matter) | \conjecture |
| §5.2 Eq. 10 (l.580–593) | a_cross, z_cross; three DR2 values | p2_09b:30–42, Eq. ph_zcross, Table ph_cross | CARRIED-CORRECTED (four DR2 rows, SX1; sympy) | \derived \observed \calc |
| §5.2 ¶ (l.595–599) | "not a numerical coincidence" | p2_09b:46–48 | CARRIED as \conjecture with the SX3 qualifier | \conjecture |
| Table 5 (l.600–608) | DR2 fits, z_cross, "IAM supp. at z_cross" | Table ph_cross (p2_09b:38) | CARRIED-CORRECTED (SX1, SX2) | \observed \calc |
| §5.3 (l.609–626) | Fig. 2 description; qualitative; full likelihood left for future | p2_09b:52–76 | CARRIED-CORRECTED (panel b measured data, finding 2; three deciding facts; mock) | \calc \observed \prediction |
| Fig. 2 (l.629–728) | fσ8, DM/rd, w0–wa | fig:sector_phantom (p2_09b:55), redrawn | CARRIED-CORRECTED (App. A data; corrected DR2 with errors; mock) | \calc \observed |
| §5.4 Scale independence (l.732–738) | time-only friction; test in DESI full shape | p2_09b:79–83 | CARRIED | \prediction |
| §5.4 Σ = 1 (l.739–750) | lensing, CMB lensing, E_G identical; fσ8/(Σσ8) | p2_09b:84–90 | CARRIED-CORRECTED (lensing follows δm; E_G +1.8 %/+3.6 %; SX4) | \calc \prediction |
| §5.4 Hubble split (l.751–762) | 72.26 (0.75σ), 67.16 (0.37σ); single βm | p2_09b:91–93 | CARRIED-CORRECTED ("phantom crossing" removed from the list of consequences, SX3) | \calc \interp |
| §6.1 (l.764–778) | consistency analyses; IAM provides mechanism; tension grows; DR1→DR2 features present | p2_09b:95–100 | CARRIED-CORRECTED (observations carried; mechanism claim for distance-only DR2 fits EXCLUDED by SX3, L22) | \observed \interp |
| §6.2 ¶1–2 (l.779–786) | f(R) μ>1 Σ>1; Horndeski; DGP μ>1; IDE | p2_09b:101–108 | CARRIED-CORRECTED (f(R) Σ=1, μ 1–4/3; sDGP μ<1 Σ=1 with ghost; S1, SX7, T13) | \observed \interp |
| §6.2 ¶3 (l.787–792) | μ0 = 0.033 ± 0.125, 1.3σ; Euclid 0.04, 3.4σ "decisive" | p2_09b:109–114 | CARRIED-CORRECTED (tab:lt_free median/bound; Euclid per sec:lt_euclid) | \fitted \calc |
| §6.2 ¶4 (l.793–798) | "not achievable in any published framework" | p2_09b:107–108 | CARRIED-CORRECTED (SX7: kept as 'follows from the worldline argument, no ghost') | \interp |
| §6.3 (l.799–825) | four limitations | p2_09b:115–131 | CARRIED-CORRECTED (17→18 chains; ODE numbers; Ωm reading as \interp; fσ_s8 template) | \openprob \interp |
| §7 (l.826–856) | five findings | p2_09b:132–149 | CARRIED-CORRECTED (pulls; S8 offsets; crossing not produced; βm fixed) | \calc \fitted \prediction |
| §7 predictions (l.857–870) | Euclid; DESI Y5 shape 7.9 %→0.6 % | p2_09b:150–157 | CARRIED-CORRECTED (Euclid per sec:lt_euclid; growth deficits 3.1→0.1 %, 5.0σ at 1 %/bin; "designed to approach" without a forecast number) | \prediction \calc |
| Acknowledgements (l.871–876) | collaborations, codes, repository | p2_09b:162 (repository pointer only) | EXCLUDED (acknowledgement, stand-alone rule); code/chain pointer kept | — |
| References (l.879–933) | — | cites mapped to iam.bib / bib_sector_s8.bib | CARRIED-CORRECTED (finding 5) | — |

## Carriage table — "The Redshift-Dependent S8 Trend" (S8). PDF line numbers from s8.txt.
| paper location | content | book location | verdict | label |
|---|---|---|---|---|
| Title / subtitle (l.2–5) | addressed to named authors | ch:s8trend title | EXCLUDED (ST6; book title applies) | — |
| Abstract (l.11–28) | trend; E(a) shape; βm virial "confirmed by chains"; Σ = 1; aim | p2_08:13–21 | CARRIED-CORRECTED (ST2 size; ST3 fixed; "joint investigation" wording dropped) | — |
| §1 (l.29–43) | trend carefully argued; not offset but trend; ΛCDM constant S8; MG rescalings | p2_08:23–41 | CARRIED-CORRECTED (trend is from fσ8 with Ωm prior and z_min cuts, finding 3; details from the source: 20 points, 1.6σ, 66 points 2.8σ, jump model) | \observed \interp |
| §1 ¶3 (l.40–43) | purpose of the note; "your datasets" | — | EXCLUDED (correspondence wording, stand-alone rule) | — |
| §2 (l.44–48) | Jacobson + Landauer; matter perturbations only | p2_08:43–47 | CARRIED | — |
| Eq. 1 (l.49–54) | μ(a) | Eq. s8_mu (p2_08:49) | CARRIED-CORRECTED (H0², T10/S9) | — |
| Eq. 2 (l.55–63) | E(a) | Eq. s8_Ea (p2_08:53) | CARRIED | — |
| Eq. 3 (l.64–71) | βm = Ωm/2; Σ = 1 | Eq. s8_beta (p2_08:57) | CARRIED (0.15765) | — |
| Eq. 4 (l.72–78) | μ0 = −βm/(Ωm+ΩΛ+βm) ≈ −0.1349 | Eq. s8_mu0 (p2_08:70), four-step derivation | CARRIED-CORRECTED (−0.1362 exact form; −0.135 is MGCAMB amplitude; sympy) | \derived |
| §2 ¶ (l.79–82) | E(a) key object, shape | p2_08:76–80 | CARRIED (+ E<0.1 for z>2.30) | \calc |
| §3 Eq. 5 (l.85–99) | S8_inferred = S8 × μ; 0.719; 0.818 at z = 2 | p2_08:81–103 | CARRIED-CORRECTED (ST1: lensing D ratio 0.8255; growth-rate ratio 0.7966; table) | \calc |
| §3 ¶ (l.100–103) | figures and "no parameter adjusted" | p2_08:76–80, 105–121 | CARRIED | — |
| Fig. 1 (l.104–129) | trend data (Table 1 of the source) | fig:s8_trend_iam (b) | CARRIED-CORRECTED: the source's binned S8 values are not tabulated in it (only plotted); replaced by the same z_min analysis on the term's growth (finding 3). Source data NOT redrawn | \calc |
| Fig. 2 (l.140–173) | S8 × μ overlay; 0.702/0.818 | fig:s8_trend_iam (b), fig:s8_inferred (a) | CARRIED-CORRECTED (ST1) | \calc |
| Fig. 3 (l.193–225) | E(a); "E<10 % for z>2.2"; ST 1 % | fig:s8_trend_iam (a) | CARRIED-CORRECTED (2.30; ST5 removed) | \calc |
| §4 (l.130–137, 174–179) | perturbation only; 61.5; Δχ² +0.54; 17 chains return βm at 0.2σ "most important result" | p2_08:126–135 | CARRIED-CORRECTED (ST3; 18 chains; consistency) | \fitted \calc |
| §5 (l.180–190) | BAO unchanged; cluster lensing mass unchanged; M_lens/M_dyn = 1/μ | p2_08:137–146 | CARRIED-CORRECTED (values 1.16/1.06/1.02; open inside clusters) | \prediction \calc \openprob |
| §6 (l.226–233) | horizon-scale cost → k-independent; f(R) scale-dependent; flat P ratio | p2_08:148–152 | CARRIED | \interp \prediction |
| §7 ¶1 (l.234–239) | ΛsCDM, IDE, MG families | p2_08:154–157 | CARRIED-CORRECTED (2402.04767 is a review; cited by journal) | \observed |
| §7 growth index (l.240–258) | γ 0.633, 3.7σ, 4.2σ; quote "at best…"; IAM supplies interpretation; diverges at z≳1 | p2_08:159–166 | CARRIED-CORRECTED (quote not found in source → not carried; γ_eff and fσ8 comparison, finding 4) | \observed \interp \calc |
| §7 list (l.259–277) | BAO, ISW 10–30 %, cluster ratio, μ–Σ, scale | p2_08:168–188 | CARRIED-CORRECTED (ISW 3.4 %/~3 %, ST4; ΛCDM ratio = 1) | \prediction \calc |
| §7 ¶ (l.278–281) | joint analysis; "we would welcome feedback" | p2_08:186–188 | CARRIED; feedback sentence EXCLUDED (correspondence) | \prediction |
| §8 (l.282–291) | Euclid 3.4σ; falsified/detection; DESI Y5 1 % | p2_08:189–200 | CARRIED-CORRECTED (sec:lt_euclid; growth deficits; 5.0σ at 1 %/bin) | \prediction \calc |
| §9 (l.292–301) | summary; 17 chains; repository | p2_08:202–207 | CARRIED-CORRECTED (18 chains; size statement) | \calc |
| Acknowledgements (l.302–305) | thanks a named cosmologist | — | EXCLUDED (ST6, no private names) | — |
| References (l.306–322) | — | cites mapped; self-citations removed | CARRIED-CORRECTED | — |

## Exclusions list (for the author)
- DE §2.3 and S8 Fig. 3: "E(a) recovered to ~1 % by Sheth–Tormen" — ST5 / SECTOR_TENSION_CHECK #11.
- DE §6.1 claim that the DR1→DR2 shifts are the predicted sector-mixing features, and §5.1/§7 "phantom crossing specifically predicted" — SX3 (distance-only DR2 fits), L22.
- DE §6.2 "μ<1, Σ=1 not achievable in any published framework" in its absolute form — SX7.
- S8 §7 quoted phrase attributed to the trend paper — not found in that paper's text.
- Front matter, keywords, paper-navigation paragraph, acknowledgements, "we would welcome feedback", "your datasets" — stand-alone and naming rules.

## Static checks (run on the delivered tree)
Braces balanced (0) in all three files; environments nest; every `\label` unique across the book (RETIRED_drafts and coverage excluded); every `\ref`/`\eqref` resolves; every `\cite` key is in iam.bib or bib_sector_s8.bib; every `\includegraphics` file exists; no "the paper", "the author", the quantum-processor report/the semiconductor report/the methylation report/the cell-reading engine, "superseded", 3.4σ/1.7σ, `[h]`/`[H]` floats, "we".
Not compiled (no TeX in sandbox).
