# MANIFEST — line-for-line carriage of two papers into Part 2

Chapters (owned): `docs/book/part2/p2_11_dark_energy.tex` (ch:darkenergy, 265 lines), `docs/book/part2/p2_20_wz_far_future.tex`
(ch:wzfuture, 229 lines), `docs/book/part2/p2_16_survey_predictions.tex` (ch:surveys, 275 lines). Clone: HEAD 12d8fb1. Nothing pushed.

## main.tex placement
No change. The three chapters keep their existing lines: `main.tex:31 \input{part2/p2_11_dark_energy}`,
`main.tex:37 \input{part2/p2_16_survey_predictions}`, `main.tex:41 \input{part2/p2_20_wz_far_future}`. No new chapter file was needed:
ch:darkenergy carries Sections 1–4 of the far-future paper, ch:wzfuture Sections 5–8, ch:surveys the whole survey paper.

## Source of the author's words
The second pass (2026-10-03) rebuilt the prose from the author's own text in `docs/book/coverage/wave2/18_wz_far_future.tex` (369 lines) and
`21_Survey_Predictions.tex` (495 lines), both read in full (50-line chunks), with the PDFs (version of record) read in full as well. His
sentences are carried verbatim except where an erratum, a check-file correction or a book rule requires a change. Handling of the coverage
flags: "we / this paper / this work" → impersonal or "this chapter"; "actualized / informational potential" (metaphysical wording) → "written /
informational record"; "gravitational potential" (the metric potential, physics) kept; "DR1" kept only where the cited result is DR1 (DESI 2024
full shape); "3.4σ" and the other Fisher significances not carried (SP5); figures redrawn on _bookstyle.py. New in the second pass: the
effective equation of state of the vacuum-like total, w_eff = −1 − (1/3a)·ρ_info/(ρ_Λ+ρ_info) (−1.062 today, minimum −1.063 at z = 0.19,
−1.015 at z = 3), which is the corrected reading of the survey paper's §6.4 (script section C9).

## Read ledger (pypdfium2 text of docs/papers/<file>.pdf)
| Paper | PDF lines (this extraction) | Ledger PAPER_LINE_COUNTS.md | Chunks read (≤ 50 lines, each shown in full) |
|---|---|---|---|
| wz_far_future.pdf | 609 | 609 | 1–50, 51–100, 101–150, 151–200, 201–250, 251–300, 301–350, 351–400, 401–450, 451–500, 501–550, 551–609 (59 lines in one view, shown in full, no truncation) |
| IAM_Survey_Predictions_Paper.pdf | 736 | 736 | 1–50, 51–100, …, 651–700 (50-line chunks), 701–736 |
LaTeX (docs/papers/latex/): IAM_wz_FarFuture_Paper.tex 498 lines, IAM_Survey_Predictions_Paper.tex 514 lines; checked for the exact form of
Eqs. 1, 5, 6 (survey) and 14, 15, 16 (far future); they match the PDF.

## Corrections applied (sources)
docs/verification/PAPER_ERRATA.md rows WZ1–WZ7, SP1–SP10, ST4, ST7, T14, S10, V5, V24; check files
docs/verification/observations/WZ_FAR_FUTURE_CHECK.md (#1–#7), SURVEY_PREDICTIONS_CHECK.md (#1–#11), theory/THEORY_CHECK.md (#10, #11).
Corrections made in this carriage and flagged for the author (no errata row yet; each recomputed in the script):
1. H_inf = H0 sqrt(Omega_L) = 0.8275 H0 (the paper printed 0.831): 55.57 for photon H0 67.16, 55.78 for 67.4.
2. No-Big-Rip proof: "Big Rip requires dw/da < 0" replaced by the bounded-density criterion (constant w < −1 also rips); full proof added
   (rho_info(a)/rho_info(1) = E(a) recovered from w_info; H_m ≤ 1.195 H0 for a ≥ 1; a(t) at most exponential).
3. Siren forecast "~50 BNS by ~2030 → ~1 %" replaced by the sourced forecast: 2 % from ~50 binary-neutron-star sirens (Chen, Fishbach & Holz
   2018, DOI 10.1038/s41586-018-0606-0); 3.5σ separation of 72.26 from 67.16 at 2 %, 7.1σ at 1 %.
4. "Planck lensing consistent with Σ = 1 to ~10 %" replaced by the sourced value: Planck 2018 VIII detects lensing at 40σ (~2.5 % amplitude).
5. KiDS σ8 0.76 ± 0.02: sourced to KiDS-1000 3×2pt (Heymans et al. 2021), 0.76 +0.025/−0.020; 0.800 is 1.6σ above it with the upper error
   (SP10 gave 2σ with a symmetric ±0.02).
6. Survey conclusion "w0wa extensions predict Σ ≠ 1 when µ ≠ 1": background w0–wa models keep µ = Σ = 1; models in which µ and Σ move together
   have Σ ≠ 1 when µ ≠ 1.
7. ISW Eq. 5: given in the Pogosian–Silvestri metric convention, ΔT/T = ∫(Φ̇+Ψ̇)dτ, with the paper's −2∫∂Φ/∂η form stated for the opposite sign
   convention; the source follows from Σ = 1 (Φ+Ψ ∝ D/a).
8. f(R) "~10 % ISW suppression": sign carried, magnitude not quoted (depends on the f(R) amplitude; no source given).
9. Two-ruler table in ch:darkenergy now leads with the Level 2 photon-sector H0 = 67.16 (rule: photon 67.16, matter 72.26); Planck 2018 base
   values (67.40/72.52/55.78/71.12) are stated beside it. New figure fig_two_rulers_future_l2.pdf replaces fig_two_rulers_future.pdf under the same
   label fig:two_rulers_future.
10. ρ_info/ρ_Λ at z = 3 printed as 1.15 % (0.011458); the errata row WZ4 rounds it to 1.2 %, the earlier chapter text to 1.1 %.

## Moved labels (all labels kept; no external \ref broken)
eq:wz_cpl moved from p2_20 to p2_11 (Section sec:de_cpl). part2:eq:Hm, eq:wz_norip, eq:sp_mu, eq:sp_isw, eq:sp_siren, sec:sp_falsify and all
figure labels keep their names. New labels: eq:de_sinfo, eq:de_rhoinfo, eq:de_cont, eq:de_dlnE, eq:de_winfo, eq:de_wlim, eq:de_w1,
eq:de_Elimits, eq:de_friedmann, eq:de_Hinf, eq:de_weff, eq:de_maturity_today, eq:de_rate_a, eq:de_rate_lna, eq:de_af, eq:de_rate_t,
fig:wz_eos_maturity_a, sec:de_*; eq:wz_rho_from_w, eq:wz_desi1–3, sec:wz_*; eq:sp_poisson, eq:sp_mu0, eq:sp_zone, eq:sp_iswT, eq:sp_iswsrc,
fig:survey_transition, fig:survey_isw, sec:sp_*.

## Table A. Dark Energy Evolution and the Far Future (wz_far_future.pdf, 609 lines)

| Paper location | Content | Book location (file:line) | Verdict (correction source) | Status label |
|---|---|---|---|---|
| Abstract | LCDM w = -1 permanent; eternal expansion and heat death | part2/p2_11_dark_energy.tex:21 | CARRIED | \observed/text |
| Abstract | Validated through 17 converged chains, Delta chi^2 = +0.54, sigma8 = 0.800 | part2/p2_16_survey_predictions.tex:215 | CARRIED-CORRECTED (WZ7: 18 chains; Delta chi^2 +0.54 is carried in part2/p2_02_virial.tex:65) | \measured |
| Abstract | w_info(a) = -1 - 1/(3a), no free parameter | part2/p2_11_dark_energy.tex:71 | CARRIED-CORRECTED (WZ6: from rho_info ~ E and energy conservation) | \derived |
| Abstract | always mildly phantom, asymptotes to -1, opposite of a runaway | part2/p2_11_dark_energy.tex:82 | CARRIED | \derived |
| Abstract | E(a) saturates at e; bounds the term; no Big Rip | part2/p2_20_wz_far_future.tex:74 | CARRIED | \derived |
| Abstract | asymptotic Hubble parameter | part2/p2_11_dark_energy.tex:141 | CARRIED-CORRECTED (photon H0 67.16; sqrt(0.6847) = 0.8275) | \calc |
| Abstract | map to w0-wa plane for DESI DR2 | part2/p2_11_dark_energy.tex:119 | CARRIED-CORRECTED (WZ1, WZ3) | \derived |
| Abstract | today at E(1)/e = 1/e = 36.8 %, inflection point, maximum creativity | part2/p2_11_dark_energy.tex:198 | CARRIED-CORRECTED (WZ2: inflection in ln a; fastest per e-fold) | \derived |
| Abstract | 50 % in ~5.7 Gyr, 99 % in ~79.5 Gyr | part2/p2_11_dark_energy.tex:221 | CARRIED | \calc |
| Abstract | three arguments against heat death | part2/p2_20_wz_far_future.tex:34 | CARRIED (interpretation) | \interp |
| Abstract | falsifiable by DESI Y5, Rubin-LSST, Roman | part2/p2_20_wz_far_future.tex:211 | CARRIED-CORRECTED (WZ4: Roman high-z w test removed) | \prediction |
| Sec. 1 | acceleration established [1-3]; cause unexplained | part2/p2_11_dark_energy.tex:18 | CARRIED | \observed |
| Sec. 1 | fine-tuning and coincidence problems | part2/p2_11_dark_energy.tex:20 | CARRIED | text |
| Sec. 1 | heat death: no saturation, no completion | part2/p2_11_dark_energy.tex:23 | CARRIED | text |
| Sec. 1 | DESI DR1/DR2 prefer dynamical DE at 2.8-4.2 sigma; w0 > -1, wa < 0 | part2/p2_11_dark_energy.tex:26 | CARRIED-CORRECTED (S10/V24: distance-only preference) | \observed |
| Sec. 1 | Jacobson-Cai-Kim procedure + S_info from gravitational decoherence; perturbation-level; mu<1, Sigma=1 | part2/p2_11_dark_energy.tex:32 | CARRIED | text |
| Sec. 1 | section roadmap | part2/p2_11_dark_energy.tex:12 | CARRIED (as the chapter place paragraphs) | text |
| Sec. 2.1 Eq. 1 | S_info(a) = S0 E(a) | part2/p2_11_dark_energy.tex:50 | CARRIED | text |
| Sec. 2.1 | E(a) from horizon thermodynamics; S0 fixed by virial theorem, beta_m = Omega_m/2 | part2/p2_11_dark_energy.tex:54 | CARRIED (virial partition, ch:virial_law) | \prediction |
| Sec. 2.1 | scalar field on the encoding surface gives w_info | part2/p2_11_dark_energy.tex:63 | CARRIED-CORRECTED (WZ6) | \derived |
| Sec. 2.1 Eq. 2 | w_info(a) = -1 - 1/(3a) | part2/p2_11_dark_energy.tex:71 | CARRIED-CORRECTED (WZ6; derivation steps eq:de_cont, eq:de_dlnE) | \derived |
| Sec. 2.1 | -1 vacuum-like pressure; -1/(3a) from derivative of E | part2/p2_11_dark_energy.tex:72 | CARRIED | text |
| Sec. 2.2 | always phantom; consistent with mild phantom preferred by DESI DR2 | part2/p2_11_dark_energy.tex:82 | CARRIED-CORRECTED (WZ3: DESI prefers w0 > -1 on the light ruler; phrase removed) | \derived |
| Sec. 2.2 Eq. 3 | lim w_info = -1 | part2/p2_11_dark_energy.tex:85 | CARRIED | \derived |
| Sec. 2.2 | phantom behaviour transient feature of structure formation | part2/p2_11_dark_energy.tex:86 | CARRIED | \derived |
| Sec. 2.2 Eq. 4 | w_info(1) = -4/3 | part2/p2_11_dark_energy.tex:90 | CARRIED | \derived |
| Sec. 2.2 | No Big Rip: dw/da > 0, approaches -1 from below | part2/p2_11_dark_energy.tex:93 | CARRIED | \derived |
| Sec. 2.2 | self-limiting feedback: structure formation slows | part2/p2_11_dark_energy.tex:95 | CARRIED (labelled interpretation) | \interp |
| Fig. 1 | w_info(a) with LCDM and today | part2/p2_11_dark_energy.tex:105 | CARRIED (redrawn, panel a) | \derived |
| Sec. 2.3 | CPL form w0 + wa(1-a) [13,14] | part2/p2_11_dark_energy.tex:116 | CARRIED | text |
| Sec. 2.3 Eqs. 5-6 | w0 = -4/3, wa = -1/3 | part2/p2_11_dark_energy.tex:119 | CARRIED | \derived |
| Sec. 2.3 | "small positive wa" | part2/p2_11_dark_energy.tex:120 | CARRIED-CORRECTED (WZ1: wa negative) | \derived |
| Sec. 2.3 | partial but imperfect match to DESI DR2 | part2/p2_20_wz_far_future.tex:111 | CARRIED-CORRECTED (WZ3) | \calc |
| Sec. 3.1 Eqs. 7-9 | E(0)=0, E(1)=1, E(oo)=e | part2/p2_11_dark_energy.tex:128 | CARRIED | \derived |
| Sec. 3.1 | ceiling not imposed; asymptotic approach | part2/p2_11_dark_energy.tex:129 | CARRIED | \derived |
| Sec. 3.2 Eq. 10 | background Friedmann equation unchanged | part2/p2_11_dark_energy.tex:138 | CARRIED | text |
| Sec. 3.2 Eq. 11 | H_inf = H0 sqrt(OL) ~ H0 x 0.831 ~ 56 | part2/p2_11_dark_energy.tex:141 | CARRIED-CORRECTED (this carriage: sqrt(0.6847) = 0.8275; 55.57 for photon H0 67.16, 55.78 for 67.4) | \calc |
| Sec. 3.2 | expansion does not halt; informational contribution saturates; final state | part2/p2_11_dark_energy.tex:145 | CARRIED | \calc |
| Sec. 3.2 | not heat death: informational completion that LCDM cannot describe | part2/p2_11_dark_energy.tex:150 | CARRIED (interpretation; WZ check: interpretation) | \interp |
| Sec. 3.3 Eq. 12 | lim w_eff = -1; phantom phase transient from z ~ 2 | part2/p2_11_dark_energy.tex:158 | CARRIED | \derived |
| Sec. 4.1 | maturity fraction E/e from 0 to 1 | part2/p2_11_dark_energy.tex:193 | CARRIED | text |
| Sec. 4.1 Eq. 13 | E(1)/e = 1/e = 0.36788 | part2/p2_11_dark_energy.tex:198 | CARRIED | \derived |
| Sec. 4.1 | not a coincidence of units; inflection point of E | part2/p2_11_dark_energy.tex:199 | CARRIED-CORRECTED (WZ2: inflection in ln a) | \derived |
| Sec. 4.1 Eq. 14 | d(E/e)/da = 1/(e a^2), max at a = 1 where it equals 1/e | part2/p2_11_dark_energy.tex:201 | CARRIED-CORRECTED (WZ2: E/(e a^2); max 4/e^2 at a = 1/2; 1/e at a = 1; per e-fold max at a = 1, eq:de_rate_lna) | \derived |
| Sec. 4.1 | maturing faster than at any other epoch | part2/p2_11_dark_energy.tex:205 | CARRIED-CORRECTED (WZ2: per e-fold only) | \derived |
| Sec. 4.2 Eq. 15 | a(f) = -1/ln f | part2/p2_11_dark_energy.tex:210 | CARRIED | \derived |
| Sec. 4.2 | ages from Friedmann equation, H0 67.4, Om 0.315, OL 0.685 | part2/p2_11_dark_energy.tex:212 | CARRIED (Level 2 background differences stated) | \calc |
| Table 1 | maturity milestones 1 % ... 99 % (10 rows) | part2/p2_11_dark_energy.tex:222 | CARRIED (all rows reproduce) | \calc |
| Sec. 4.3 Eq. 16 | d(E/e)/dt at a=1 = H0/e ~ 2.54 %/Gyr | part2/p2_11_dark_energy.tex:229 | CARRIED | \calc |
| Sec. 4.3 | rate currently at its all-time peak | part2/p2_11_dark_energy.tex:230 | CARRIED-CORRECTED (WZ2: per Gyr peak z = 1.26, 3.38 %/Gyr) | \calc |
| Sec. 4.3 | sigmoidal; 13.8 Gyr to 36.8 %, +5.7 to 50 %, +~32 to 90 %, +~41 to 99 % | part2/p2_11_dark_energy.tex:236 | CARRIED (32.6, 41.2 Gyr) | \calc |
| Fig. 2 | maturity fraction and rate against a | part2/p2_11_dark_energy.tex:105 | CARRIED-CORRECTED (WZ2; redrawn panels b, c; cosmic-time version fig:maturity) | \derived |
| Sec. 5 intro | standard heat death picture | part2/p2_20_wz_far_future.tex:35 | CARRIED | \interp |
| Sec. 5.1 | Argument 1: bounded informational pressure; informational epoch ends | part2/p2_20_wz_far_future.tex:40 | CARRIED | \interp |
| Sec. 5.1 | "DESI DR2 already hints at this ... iam predicts exactly this" | — | EXCLUDED (WZ3: not supported; DESI prefers w0 > -1 on the light ruler) | — |
| Sec. 5.2 | Argument 2: horizon at T_GH, equilibrium not dissolution | part2/p2_20_wz_far_future.tex:49 | CARRIED | \interp |
| Sec. 5.3 | Argument 3: S = A/4 l_P^2 holds the record; maximally encoded, not disordered | part2/p2_20_wz_far_future.tex:58 | CARRIED | \interp |
| Sec. 5.4 Eq. 17 | dw/da = 1/(3a^2) > 0; Big Rip excluded | part2/p2_20_wz_far_future.tex:74 | CARRIED; proof completed (eq:wz_rho_from_w, bounded H_m) | \derived |
| Sec. 5.4 | "Big Rip requires dw/da < 0" | part2/p2_20_wz_far_future.tex:84 | CARRIED-CORRECTED (this carriage: constant w < -1 also rips; boundedness of rho is the criterion, as Sec. 2.2 states) | \derived |
| Sec. 6.1 Eqs. 18-19 | DESI values -0.827/-0.75, -0.752/-1.05 | part2/p2_20_wz_far_future.tex:96 | CARRIED-CORRECTED (WZ3: DR2 values, three supernova samples) | \observed |
| Sec. 6.1 | 2.8-4.2 sigma preference | part2/p2_20_wz_far_future.tex:100 | CARRIED | \observed |
| Sec. 6.2 Eq. 20 | (w0, wa) = (-4/3, -1/3) | part2/p2_20_wz_far_future.tex:105 | CARRIED | \derived |
| Fig. 3 | IAM point vs DESI DR2 contours | part2/p2_20_wz_far_future.tex:114 | CARRIED-CORRECTED (WZ1, WZ3; DR2 fits, 1 sigma bars) | \observed/\calc |
| Sec. 6.2 | outside DESI 2 sigma contours; same qualitative character (positive wa) | part2/p2_20_wz_far_future.tex:106 | CARRIED-CORRECTED (WZ1: wa negative; WZ3) | \calc |
| Sec. 6.3 | CPL is a Taylor expansion; w_info nonlinear; inaccurate at a << 1, a >> 1 | part2/p2_20_wz_far_future.tex:118 | CARRIED (quantified by a new table) | \calc |
| Sec. 6.3 | DESI measures BAO distances; direct likelihood deferred | part2/p2_20_wz_far_future.tex:133 | CARRIED-CORRECTED (WZ3: photon-sector likelihood is LCDM; matter-ruler comparison open) | \derived/\openprob |
| Sec. 6.4 item 1 | w(z) = -1 - (1+z)/3; DESI Y5 ~1 % w; w > -1 disfavours | part2/p2_20_wz_far_future.tex:140 | CARRIED-CORRECTED (WZ3: matter ruler; WZ7: Snowmass ~1 % not carried) | \prediction |
| Sec. 6.4 item 2 | strongly phantom early w testable by Roman high-z supernovae | part2/p2_20_wz_far_future.tex:147 | CARRIED-CORRECTED (WZ4: rho_info 1.15 % of rho_L at z = 3; not observable) | \calc |
| Sec. 6.4 item 3 | no Big Rip signature | part2/p2_20_wz_far_future.tex:150 | CARRIED | \prediction |
| Sec. 6.4 item 4 | w(z) and growth from the same E(a); unique cross-check | part2/p2_20_wz_far_future.tex:152 | CARRIED | \prediction |
| Sec. 7.1 | phantom scalar fields: negative kinetic term, energy conditions, Big Rip; IAM bounded | part2/p2_20_wz_far_future.tex:167 | CARRIED | \derived |
| Sec. 7.1 | not constant, not CPL, not quintessence; a prediction, not a model | part2/p2_20_wz_far_future.tex:174 | CARRIED | \prediction |
| Sec. 7.2 | E(1) = 1 follows from the Planck-epoch reference | part2/p2_20_wz_far_future.tex:178 | CARRIED-CORRECTED (WZ5: normalisation at a = 1) | \derived |
| Sec. 7.2 | d(E/e)/da maximised at the inflection; era of fastest growth | part2/p2_20_wz_far_future.tex:183 | CARRIED-CORRECTED (WZ2: per e-fold) | \derived/\interp |
| Sec. 7.2 | observers appear when most information is produced | — | EXCLUDED (WZ5: anthropic remark removed) | — |
| Sec. 7.2 | whether the coincidence is significant is beyond scope | part2/p2_20_wz_far_future.tex:180 | CARRIED-CORRECTED (WZ5: it is the normalisation) | \derived |
| Sec. 8 | conclusion summary | part2/p2_20_wz_far_future.tex:197 | CARRIED | text |
| Sec. 8 | unified origin of S8 tension and DESI dark-energy evolution | part2/p2_20_wz_far_future.tex:213 | CARRIED-CORRECTED (WZ3: DESI evolution is a light-ruler result; growth and two-ruler parts kept) | \prediction |
| Sec. 8 | three tests: DESI Y5 fs8 ~1 %, Rubin-LSST, Roman | part2/p2_20_wz_far_future.tex:211 | CARRIED-CORRECTED (WZ4 Roman removed; ~1 % unsourced, not carried) | \prediction |
| Acknowledgments | software and archive | — | NOT CONTENT (back matter; repository is cited in every chapter header) | — |
| Refs [5]-[7], [16] | own papers; Snowmass 2013 | — | NOT CITED (stand-alone rule; WZ7 for [16]) | — |

## Table B. Falsifiable Predictions for Euclid, DESI and Next-Generation Surveys (IAM_Survey_Predictions_Paper.pdf, 736 lines)

| Paper location | Content | Book location (file:line) | Verdict (correction source) | Status label |
|---|---|---|---|---|
| Abstract (1) | Fisher: Euclid 2.2/3.4 sigma; Euclid + DESI Y5 5.4 sigma | part2/p2_16_survey_predictions.tex:170 | EXCLUDED (SP5); replaced by sourced Euclid statement of sec:lt_euclid | \observed/\calc |
| Abstract (2) | A_ISW = 1.134; opposite sign to f(R) | part2/p2_16_survey_predictions.tex:135 | CARRIED-CORRECTED (SP3/ST4: ~1.03; sign stands) | \calc |
| Abstract (3) | tomographic 1.2-1.9 sigma; beta_m 0.1617 +- 0.0867 | part2/p2_16_survey_predictions.tex:103 | EXCLUDED (SP7: no code); shape test carried as prediction/open problem | \prediction/\openprob |
| Abstract (4) | transition zone 0.06-1.12; peak z ~ 0.05 | part2/p2_16_survey_predictions.tex:88 | CARRIED-CORRECTED (SP4: |dmu/dz| largest at z = 0) | \calc |
| Abstract | beta_m = Omega_m/2 = 0.1575; no free parameters | part2/p2_16_survey_predictions.tex:35 | CARRIED-CORRECTED (Omega_m 0.3153 -> 0.15765) | \prediction |
| Abstract | predictions explicit; comparison unambiguous | part2/p2_16_survey_predictions.tex:22 | CARRIED | text |
| Sec. 1 | single Hubble friction term; dual-sector mu < 1, Sigma = 1 | part2/p2_16_survey_predictions.tex:25 | CARRIED | text |
| Sec. 1 | mapping onto mu-Sigma (Pogosian & Silvestri 2016) | part2/p2_16_survey_predictions.tex:29 | CARRIED (Poisson equations added) | text |
| Sec. 1 Eq. 1 | mu(a) = H^2/(H^2 + beta_m E), Sigma = 1 | part2/p2_16_survey_predictions.tex:33 | CARRIED | \prediction |
| Sec. 1 | 15 chains, Delta chi^2 +0.54, sigma8 0.800 | part2/p2_16_survey_predictions.tex:215 | CARRIED-CORRECTED (SP10: 18 chains) | \measured |
| Sec. 1 | forecasts not guarantees; any outcome informative | part2/p2_16_survey_predictions.tex:20 | CARRIED | text |
| Sec. 2.1 | mu0 = -0.135 | part2/p2_16_survey_predictions.tex:36 | CARRIED-CORRECTED (exact -0.136; MGCAMB -0.13495 stated) | \prediction |
| Sec. 2.1 | IAM Planck MCMC mu0 = +0.006 +- 0.156, 0.9 sigma | part2/p2_16_survey_predictions.tex:166 | CARRIED-CORRECTED (SP6: chain medians and one-sided bounds, as ch:latetime) | \fitted |
| Table 1 | current mu0 constraints (5 rows) | part2/p2_16_survey_predictions.tex:163 | CARRIED-CORRECTED (SP6: traced values DES Y3+ext, DESI DR1, ACT; untraced Planck +0.000 +- 0.200 and DES Y3 -0.40 +- 0.40 rows not printed) | \observed |
| Sec. 2.2 Eqs. 2-4 | sigma(mu0) 0.060/0.040/0.025 -> 2.2/3.4/5.4 sigma | — | EXCLUDED (SP5: unsourced forecast) | — |
| Fig. 1(a) | current and projected mu0 constraints | part2/p2_05_dual_sector_note.tex fig:mu0_constraints | CARRIED (current, traced) in ch:dsnote; projected points EXCLUDED (SP5) | \observed |
| Fig. 1(b) | f sigma8(z), IAM vs LCDM, DESI Y5 errors | part2/p2_16_survey_predictions.tex:69 | CARRIED as deficits (fig:survey_ramp, fig:survey_transition e); DESI Y5 error bars EXCLUDED (SP5) | \calc |
| Fig. 1(c), Table 2, Sec. 2.3 | detection timeline 0.9 ... 7.5 sigma | part2/p2_16_survey_predictions.tex:180 | EXCLUDED (SP5); Euclid DR1 mid-2027 carried | \openprob |
| Fig. 1(d) | mu(z) exact vs MGCAMB with Euclid bins | part2/p2_16_survey_predictions.tex:94 | CARRIED (panel b; bins EXCLUDED, SP5) | \calc |
| Sec. 3.1 Eq. 5 | Delta T/T = -2 int dPhi/deta | part2/p2_16_survey_predictions.tex:121 | CARRIED (general form with the metric convention; the paper form given) | \derived |
| Sec. 3.1 | mu < 1 -> potentials decay faster; Sigma = 1 leaves geodesics | part2/p2_16_survey_predictions.tex:125 | CARRIED (source derived from Sigma = 1) | \derived |
| Sec. 3.2 Eq. 6 | A_ISW = 1.134 (+13.4 %) | part2/p2_16_survey_predictions.tex:135 | CARRIED-CORRECTED (SP3: ~1.03) | \calc |
| Sec. 3.2 | f(R) mu > 1 -> A < 1, ~10 % suppression; sign discriminant | part2/p2_16_survey_predictions.tex:137 | CARRIED (sign; magnitude not quoted: model-dependent, no source) | \observed |
| Sec. 3.2 | enhancement concentrated at z < 1 | part2/p2_16_survey_predictions.tex:131 | CARRIED (source ratios by redshift) | \calc |
| Sec. 3.2 | current ISW ~4 sigma, 20-30 % errors; not a near-term discriminator | part2/p2_16_survey_predictions.tex:141 | CARRIED | \observed |
| Fig. 2(a)-(b) | ISW kernel and ratio (10-30 %) | part2/p2_16_survey_predictions.tex:144 | CARRIED-CORRECTED (SP3/ST4: ratio <= 1.035) | \calc |
| Fig. 2(c) | potential decay -5.3 % at z = 0.5 | part2/p2_16_survey_predictions.tex:128 | CARRIED-CORRECTED (SP2 logic: potential follows D; -0.22 %) | \calc |
| Fig. 2(d) | C_l^Tg; DESI BGS/LRG A = 1.092/1.115/1.165 | part2/p2_16_survey_predictions.tex:133 | CARRIED-CORRECTED (SP3: 1.035/1.034/1.031; MGCAMB form 1.054/1.064/1.072) | \calc |
| Fig. 2(e) | amplitude vs z; Planck +-0.25, Euclid+DESI +-0.10 bands | part2/p2_16_survey_predictions.tex:141 | CARRIED (curve, ~25 % current error); Euclid+DESI band EXCLUDED (SP5) | \calc |
| Fig. 2(f) | model comparison IAM 1.134, LCDM 1, f(R) 0.90 | part2/p2_16_survey_predictions.tex:138 | CARRIED-CORRECTED (SP3; f(R) value not quoted) | \calc |
| Sec. 4.1, Table 3 | mu(z) at z_eff 0.1-2.0 and Delta mu/mu | part2/p2_16_survey_predictions.tex:44 | CARRIED (mu and 1 - mu columns) | \calc |
| Table 3 | current sigma(mu), Euclid sigma(mu) columns | — | EXCLUDED (SP5: unsourced forecast; Euclid only as sec:lt_euclid) | — |
| Sec. 4.2 | ten-bin reconstruction 1.2-1.9 sigma; 1000 mocks, beta_m recovery | — | EXCLUDED (SP7: no code, not reproduced) | — |
| Sec. 4.2 | tests IAM shape vs constant mu or power law; beta_m testable from shape | part2/p2_16_survey_predictions.tex:109 | CARRIED | \prediction |
| Fig. 3(a) | mu(z) models (IAM, MGCAMB, constant) with Euclid bands | part2/p2_16_survey_predictions.tex:107 | CARRIED (curves; bands EXCLUDED, SP5) | \calc |
| Fig. 3(b) | shape difference IAM exact vs MGCAMB | part2/p2_16_survey_predictions.tex:106 | CARRIED (panel f; "below Euclid threshold" EXCLUDED, SP5) | \calc |
| Fig. 3(c)-(f) | per-bin detection, mock, chi^2, beta_m histogram | — | EXCLUDED (SP7) | — |
| Sec. 5.1 | E(a) rises 0 -> e; transition zone definition | part2/p2_16_survey_predictions.tex:74 | CARRIED | text |
| Sec. 5.1 Eq. 7 | 0.06 <~ z <~ 1.12 | part2/p2_16_survey_predictions.tex:88 | CARRIED | \calc |
| Sec. 5.1 | midpoint z ~ 0.37 in DESI BGS and Euclid photometric ranges | part2/p2_16_survey_predictions.tex:89 | CARRIED | \calc |
| Sec. 5.1 | |dmu/dz| peaks at z ~ 0.05 | part2/p2_16_survey_predictions.tex:90 | CARRIED-CORRECTED (SP4) | \calc |
| Table 4 | activation milestones 1 %-99 % (9 rows) | part2/p2_16_survey_predictions.tex:84 | CARRIED (all rows reproduce) | \calc |
| Table 5 | Delta mu, Delta D/D, Delta f sigma8, Delta Phi/Phi | part2/p2_16_survey_predictions.tex:53 | CARRIED-CORRECTED (SP1, SP2) | \calc |
| Fig. 4(a) | E(a) milestones 2.30/0.69/0.11, inflection z = 1 | part2/p2_16_survey_predictions.tex:94 | CARRIED | \calc |
| Fig. 4(b) | mu(z): IAM, MGCAMB, f(R), nDGP; IAM only model with mu < 1 | part2/p2_16_survey_predictions.tex:249 | CARRIED-CORRECTED (THEORY_CHECK #10: among viable models; sDGP mu<1 has a ghost; f(R)/nDGP shown as the mu > 1 side) | \derived |
| Fig. 4(c) | normalised turn-on profiles | part2/p2_16_survey_predictions.tex:97 | CARRIED | \calc |
| Fig. 4(d) | |dmu/dz| peak z = 0.05 | part2/p2_16_survey_predictions.tex:98 | CARRIED-CORRECTED (SP4) | \calc |
| Fig. 4(e) | observable deviations | part2/p2_16_survey_predictions.tex:98 | CARRIED-CORRECTED (SP1, SP2) | \calc |
| Fig. 4(f) | S/N map (Euclid-like) | — | EXCLUDED (SP5: unsourced errors) | — |
| Sec. 6.1 Eq. 8 | H0 sirens = 67.161 sqrt(1.1575) = 72.26 | part2/p2_16_survey_predictions.tex:190 | CARRIED-CORRECTED (67.16 sqrt(1.15765)) | \calc |
| Sec. 6.1 | GW170817 70.0 +12/-8; updated 75.5 +- 5.4 (Nicolaou 2023) | part2/p2_16_survey_predictions.tex:192 | CARRIED-CORRECTED (T14: 75.46 is the 2024 afterglow analysis; 68.9 added) | \observed |
| Sec. 6.1 | ~50 BNS by ~2030 -> ~1 % | part2/p2_16_survey_predictions.tex:195 | CARRIED-CORRECTED (this carriage: sourced 2 % from ~50 BNS, Chen et al. 2018; 3.5 sigma) | \observed/\calc |
| Sec. 6.2 | Sum m_nu < 0.07-0.08 eV | — | EXCLUDED (SP8: no calculation) | — |
| Sec. 6.3 | C_phiphi identical to LCDM | part2/p2_16_survey_predictions.tex:155 | CARRIED-CORRECTED (SP9: -0.08 %) | \calc |
| Sec. 6.3 | CMB-S4 deviation challenges Sigma = 1; Planck consistent to ~10 % | part2/p2_16_survey_predictions.tex:156 | CARRIED-CORRECTED (this carriage: sourced Planck 2018 VIII 40 sigma, ~2.5 %) | \observed/\prediction |
| Sec. 6.4 | w_eff < -1 recently, -> -1 at high z; consistent with DESI hints | part2/p2_16_survey_predictions.tex:203 | CARRIED-CORRECTED (SURVEY check #10, WZ3: true for the vacuum-like total, w_eff -1.062 today, min -1.063 at z 0.19, -1.015 at z 3, on the matter ruler; "consistent with DESI hints" removed) | \calc |
| Table 6 | scorecard (10 rows) | part2/p2_16_survey_predictions.tex:213 | CARRIED-CORRECTED (SP10: S8 0.822; KiDS-1000 sigma8 0.76 +0.025/-0.020 is 1.6 sigma; H_photon 67.16; sirens per T14; ISW ~1.03, "+15 +- 30 %" unsourced not printed; Sigma 1.0 +- 0.1 unsourced not printed; E_G row added) | \calc/\observed |
| Table 6 | Sum m_nu row | — | EXCLUDED (SP8) | — |
| Sec. 7 | consistent with every prediction; none excluded; Euclid+DESI decisive | part2/p2_16_survey_predictions.tex:232 | CARRIED | \observed |
| Sec. 8 | parameter-free: beta_m from virial theorem, E(a) from horizon thermodynamics | part2/p2_16_survey_predictions.tex:35 | CARRIED | \prediction |
| Sec. 8 | unique signature vs LCDM, f(R), nDGP, w0wa (Sigma != 1 when mu != 1) | part2/p2_16_survey_predictions.tex:249 | CARRIED-CORRECTED (THEORY_CHECK #10; this carriage: w0wa background models keep mu = Sigma = 1) | \derived |
| Sec. 8 | ISW sign discriminant | part2/p2_16_survey_predictions.tex:257 | CARRIED | \derived |
| Sec. 8 | Euclid + DESI Y5 5.4 sigma; full combination 7.5 sigma | — | EXCLUDED (SP5) | — |
| Sec. 8 | deviations equally informative; surveys will decide | part2/p2_16_survey_predictions.tex:260 | CARRIED | text |
| Acknowledgments | software | — | NOT CONTENT (back matter) | — |
| References | Euclid 2020, Nicolaou 2023, Zucca 2019, Planck 2016/2020, own papers | — | Replaced by traced keys (Albuquerque2025/Frusciante2025 per sec:lt_euclid; Palmese2024; Wang2023MGCAMB in ch:latetime; Planck2015ISW; Planck2018VIII); own papers not cited (stand-alone rule) | — |

Item counts (verdict prefix): Table A 81 items; Table B 66 items.

## Exclusions for the author (each withdrawn by a confirmed correction)
| Item | Errata row |
|---|---|
| "DESI DR2 already hints at this … iam predicts exactly this" (wz §5.1) | WZ3 |
| Anthropic remark: observers appear when most information is produced (wz §7.2) | WZ5 |
| Roman high-z supernova test of the strongly phantom early w (wz §6.4 item 2, §8) — replaced by the corrected statement that it is unobservable | WZ4 |
| DESI Year 5 "~1 % w at multiple redshifts" (Snowmass 2013 citation) | WZ7 |
| Fisher significances 2.2/3.4/5.4σ, Eqs. 2–4; Table 2 timeline; Fig. 1(c); 5.4σ and 7.5σ in the conclusion | SP5 |
| Table 3 per-bin σ(µ) columns (current, Euclid); Fig. 1(b) DESI Y5 error bars; Fig. 2(e) Euclid+DESI band; Fig. 3(a),(b) Euclid bands; Fig. 4(f) S/N map | SP5 (same unsourced forecast; Euclid only as sec:lt_euclid) |
| Tomographic mock: 1.2–1.9σ, β_m = 0.1617 ± 0.0867, Fig. 3(c)–(f), Fig. 3(d) β_m = 0.0588 | SP7 |
| Neutrino-mass bound Σm_ν < 0.07–0.08 eV (§6.2, Table 6 row) | SP8 |
| Table 1 Planck row (+0.000 ± 0.200) and DES Y3 row (−0.40 ± 0.40): untraced | SP6 |
| Table 6 data entries "Σ ~ 1.0 ± 0.1" and "ISW +15 ± 30 %": unsourced | SP10 |

## Flags for the lead
- iam.bib entries Heymans2021 and Riess2022 have no doi field. CrossRef confirms 10.1051/0004-6361/202039063 (Heymans 2021) and
  10.3847/2041-8213/ac5c5b (Riess 2022); add them when merging iam.bib.
- `bib_darkenergy_surveys.bib` (3 new entries, DOIs CrossRef-verified) must be merged into iam.bib or added to `\bibliography{iam,bib_darkenergy_surveys}`.
- SURVEY_PREDICTIONS_CHECK.md notes an author placement rule (prediction papers to the predictions appendix). This task names p2_16 as the
  owned file, so the carriage is in ch:surveys; the chapter can be moved as a whole if that rule is applied.
- fig_maturity.pdf, fig_wz_history.pdf, fig_wz_cpl_clocks.pdf, fig_survey_ramp.pdf and fig_survey_precision.pdf are unchanged (made by
  fig_p2_sn_future.py, fig_p2_wz_history.py, fig_p2_survey_tests.py); fig_two_rulers_future.pdf is no longer referenced by these chapters.

## Verification
`docs/verification/scripts/verify_dark_energy_far_future_surveys_book.py` (+ `_output.txt`, 126 lines): sympy for every algebraic step
(continuity residual 0; dlnE/dlna = 1/a; CPL; limits of E; inflections at a = 1/2 and in ln a at a = 1; d(E/e)/da = e^{-1/a}/a^2, max 4/e^2 at
a = 1/2; a(f); H_m^2(1) = H0^2(1+β_m); µ0 = −β/(1+β); d(D/a)/dτ = HD(f−1)); numerics for the two-ruler rates (both backgrounds), maturity
table (both), rate peaks, ρ_info weights, DESI offsets and crossings, CPL error table, µ(z) table, MGCAMB difference (0.0266 at z = 0.68; 2.84 %
relative at z = 0.65), milestones, |dµ/dz|, E milestones, ISW source and per-sample amplitudes, potential change, E_G, sirens, KiDS, Planck lensing.
Static checks on the three chapters: braces balanced, environments matched, no duplicate labels across the book, every \ref resolves against the
current tree, every \cite resolves in iam.bib or bib_darkenergy_surveys.bib (35 keys), all figures exist, every float [htbp]. Compilation not run
(TeX bundle unavailable in the sandbox).
