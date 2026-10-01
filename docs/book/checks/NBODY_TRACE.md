# NBODY_TRACE — the two N-body tables in the virial papers, traced to their sources (2026-10-01)

Scope: Table 1 (claimed "mass-weighted virial efficiency ⟨η_vir⟩ = 2K/|U| within r_vir") and Table 2 (claimed "n_eff = log slope of halo collapse rate vs linear growth factor"), plus f_coll = 0.62 ± 0.03.
Method: arXiv full text of each source downloaded and searched (text plus figure pages read by eye). Page numbers are **arXiv preprint pages**, not journal pages.
Press & Schechter 1974 and Lacey & Cole 1993 are pre-arXiv and were **not fetched**; that row relies on the standard PS formula.

## Bottom line
1. **None of the six Table 1 sources reports a halo-population 2K/|U| between 0.72 and 0.90.** Every source that measures the ratio (Bett, Neto, Power, Klypin) finds 2T/|U| **above 1** (about 1.05–1.4, rising with mass). The tabulated numbers line up with the **reciprocal |U|/2T** as read from their figures and fits, but only approximately. None of the sources tabulates |U|/2T.
2. Bryan & Norman 1998 reports a **different quantity**: f_σ and f_T, the velocity-dispersion and temperature normalisations of the virial scaling relations. Ludlow et al. 2010 reports **no virial-ratio values**, only a selection cut. Several simulation and mass-range attributions are also wrong (Power 2012 is not GIMIC/OWLS; Ludlow 2010 is not Millennium-II 10^10–10^12).
3. **No Table 2 source reports a "log slope of collapse rate vs growth factor".** The numbers 2.5–3.8 do not appear as such a quantity. The 3.8 is the exponent in the Jenkins 2001 fitting function (Reed 2003 only quotes it). "n_eff" in Jenkins and Reed is the **effective power-spectrum slope** (≈ −1.4 to −1.7), which is a different quantity with the opposite sign. ST99's parameters are a = 0.707 and p = 0.3; "σ* = 1.2" does not appear.
4. The slope can be computed: d ln F(>M)/d ln D = f(σ)/F(>M). It depends strongly on mass. At z = 0 with Planck 2018 it runs from about 0.3 (10^8 h⁻¹M⊙) to about 3–4 (10^14 h⁻¹M⊙). The claimed values are reached only at **cluster masses**, M ≈ 10^13.4–10^14.2 h⁻¹M⊙ (ν ≈ 1.4–2.0). So the table is not a universal exponent.
5. **f_coll:** Tinker 2008 does not report a collapsed fraction. Integrating its Δ=200m z=0 fit with Planck 2018 gives F(>10^10.5) = 0.486 at the lower edge of the calibrated range, F(>10^11) = 0.447 and F(>10^12) = 0.348. Reaching **0.62 requires M_min ≈ 10^8.2 h⁻¹M⊙**, 2.3 dex below Tinker's calibrated range. The ±0.03 maps to M_min = 10^7.5–10^8.8. The fit is also not normalised: F keeps growing as M_min → 0 (0.76 at 10^4).

## Table 1 — virial ratio

| Source (arXiv) | Paper's value | What the source reports (definition, sample) | Where | Verdict |
|---|---|---|---|---|
| Bryan & Norman 1998, ApJ 495, 80 (astro-ph/9710107) | 0.80–0.90; relaxed ~0.87, merging ~0.75 | No 2K/\|U\|. Reports **f_σ = 0.82–0.89** (σ/σ_vir, DM velocity-dispersion normalisation versus a singular isothermal sphere) and **f_T = 0.75–0.79** (T/T_vir) for CDM270/CHDM512/OCDM256/CHDM256; adopts f_T = 0.77. Eulerian hydro simulations of ~25 clusters per model; non-ΛCDM models (SCDM/CHDM/OCDM). No relaxed/merging split. | Table 2, p8; p16; p20 | **Different quantity**. Relaxed/merging values are not in the source. |
| Bett et al. 2007, MNRAS 376, 215 (astro-ph/0608607) | 0.77–0.83 | Instantaneous "virial ratio" written as **2T/U + 1** (U < 0; zero means virialised). The quasi-equilibrium cut is \|2T/U+1\| ≤ Q = 0.5, i.e. 0.5 ≤ 2T/\|U\| ≤ 1.5. Fig. 4 ridge sits at 2T/U+1 ≈ −0.2 to −0.3, i.e. **2T/\|U\| ≈ 1.2–1.3** (read by eye). Millennium, TREE haloes, N_p > 300 (M ≳ 2.6×10^11 h⁻¹M⊙) up to ~10^15. No tabulated mean. | §3.2.3 eq. (12), p6; Fig. 4 and text, p7 | **Reciprocal approximately matches** a figure read-off (1/1.2–1/1.3 = 0.83–0.77). Not tabulated. Mass range in the paper (10^11–10^13) is not the source's. |
| Neto et al. 2007, MNRAS 381, 1450 (0706.2919) | 0.79–0.87; relaxed 0.87, unrelaxed 0.72 | **2T/\|U\|**, T and U of particles within r_vir, no surface term. "Median 2T/\|U\| is slightly greater than unity." Relaxation cut **2T/\|U\| < 1.35**. Fig. 2 (bottom-left): log10(2T/\|U\|) ≈ 0.05–0.10, i.e. ≈ 1.12–1.26. Millennium, lower mass limit ~10^12 h⁻¹M⊙. | §2.3.1(iii), p4; §2.3.2 and Fig. 2, p5 | **Reciprocal approximately matches** (0.79–0.89) a figure read-off. The relaxed/unrelaxed values 0.87/0.72 are not in the source; 0.74 = 1/1.35 is the cut, not a measurement. |
| Ludlow et al. 2010, MNRAS 406, 137 (1001.2310) | 0.76–0.82 | Uses **2K/\|Φ\| < 1.3** only as a relaxation cut; no distribution or mean reported. Sample: 21 resimulated haloes (6 Aquarius + 15 others), ~10^12 to a few ×10^14 h⁻¹M⊙. Not Millennium-II 10^10–10^12. | §2.2, p2; p4 | **Not in source.** Sample is misattributed. |
| Power, Knebe & Knollmann 2012, MNRAS 419, 1576 (1109.2671) | 0.76–0.85; "η decreases ~5% z=0→1"; GIMIC/OWLS | **η = 2T/\|W\|** within r_vir; also η′ = (2T − E_s)/\|W\| with surface pressure. Fits: ⟨log10 η⟩ = 0.05 + 0.016 log10 M12; median log10 η = 0.04 + 0.019 log10 M12. **η ≈ 1.15 (10^12) to 1.25 (10^15)**, "systematically greater than unity". η′ distribution centred on **≈ 0.9**. Own GADGET-2 ΛCDM boxes L20–L500, haloes at **z = 0 only**. No GIMIC/OWLS and no z = 1 trend. | eq. (1), p2; eqs. (11)–(12), p6–7; eqs. (17)–(18) and text, p9 | **Reciprocal partly matches** (1/1.25–1/1.15 = 0.80–0.87; the paper's 0.76 is below that range). The redshift claim and the simulation attribution are **not in the source**. η′ ≈ 0.9 is a different, surface-corrected quantity. |
| Klypin et al. 2016, MNRAS 457, 4340 (1411.4001) | 0.78–0.84 | **2K/\|W\| − 1**. Fig. 6 (MDPL, z=0, all haloes, uncorrected): ≈ 0.1 at 10^12 rising to ≈ 0.4 at 10^15, i.e. **2K/\|W\| ≈ 1.1–1.4**. Surface-pressure correction is ≈ 0.1–0.2, giving corrected 2K/\|W\| ≈ 1.02–1.17. Relaxed selection uses 2K/\|W\| < 1.5. Figures are MDPL, not Bolshoi-Planck. | §3, p3; Figs. 5–6 and eqs. (7)–(10), p5–6 | **Reciprocal falls inside the uncorrected range** (0.71–0.91; 0.78–0.84 corresponds to M ≈ 10^14–10^14.7). No reported value of 0.78–0.84. |

The paper's 0.815 ± 0.025 equals 1/(2 × 0.62) = 0.806 within its error. The number therefore appears tied to f_coll through η_vir = 1/(2 f_coll), not measured independently.

## Table 2 — "n_eff"

| Source | Paper's value | In source? | What the number corresponds to | Computed d ln F(>M)/d ln D reaches the claimed value at (Planck18, z=0) | Verdict |
|---|---|---|---|---|---|
| Press & Schechter 1974 + Lacey & Cole 1993 (not fetched) | 2.5 | Not checked (pre-arXiv). The PS formula has no such constant. | — | M = 10^13.4 h⁻¹M⊙ (ν = 1.37) | **Not in source (provisional; texts not read)**. Mass-dependent. |
| Sheth & Tormen 1999 (astro-ph/9901122) | 3.5 (σ* = 1.2) | No. Parameters are a = 0.707, p = 0.3 (eq. 10, p3); "σ* = 1.2" does not appear. | — | 10^14.13 (ν = 1.93) | **Not in source** |
| Jenkins et al. 2001 (astro-ph/0005260) | 3.2 | No. Fit f = 0.315 exp(−\|ln σ⁻¹ + 0.61\|^**3.8**) (eq. 9, p9). Its "n_eff" = 6 d ln σ⁻¹/d ln M − 3 is the power-spectrum slope, −1.39 to −1.70 (eq. 8, p7). | 3.8 is this paper's fitting exponent | 10^14.02 (ν = 1.83) | **Not in source** (n_eff there is a different quantity) |
| Reed et al. 2003 (astro-ph/0301270) | 3.8 | No. 3.8 appears only as the **quoted Jenkins exponent** (eq. 6, p3). Own fit f_ST × exp[−0.7/(σ cosh(2σ)^5)] (eq. 9, p5). n_eff = power-spectrum slope (eq. 8, p5). | Jenkins exponent, misattributed | 10^14.18 (ν = 1.99) | **Not in source / different quantity** |
| Tinker et al. 2008 (0803.2706) | 3.0 | No. Δ=200: A = 0.186, a = 1.47, b = 2.57, c = 1.19 (Table 2, p9). Low-mass log slope of the mass function is ≈ −1.85 (p8), a different quantity. | — | 10^13.90 (ν = 1.73) | **Not in source** |
| Watson et al. 2013 (1212.0095) | 3.3 | No. FOF fit A = 0.282, α = 2.163, β = 1.406, γ = 1.210, valid 1.8×10^12–7×10^15 h⁻¹M⊙ (eq. 12, p8). "3.3" is only a section number. | — | 10^14.10 (ν = 1.91) | **Not in source** |

The six claimed values average 3.22 with a standard deviation of 0.44 (the paper's 3.22 ± 0.44).

**How the slope was computed.** For a multiplicity function f(σ) with x = ln σ⁻¹, the collapsed fraction above M is F(>M) = ∫_{x(M)}^∞ f dx. Linear growth sends σ → Dσ, so at fixed M, d ln F(>M)/d ln D = f(σ_M)/F(>M). The table below gives this at selected masses, z = 0, Planck 2018 (colossus `planck18`, σ8 = 0.810), δc = 1.686. Parameters are the published ones; Tinker uses Δ = 200m and Watson uses FOF. Values outside each fit's calibrated range are extrapolations.

| M_min (h⁻¹M⊙) | PS | ST | Jenkins | Reed | Tinker | Watson |
|---|---|---|---|---|---|---|
| 10^10 | 0.50 | 0.53 | 0.48 | 0.54 | 0.52 | 0.59 |
| 10^11 | 0.71 | 0.70 | 0.71 | 0.71 | 0.67 | 0.71 |
| 10^12 | 1.10 | 1.01 | 0.96 | 1.02 | 0.96 | 0.95 |
| 10^13 | 1.92 | 1.63 | 1.49 | 1.65 | 1.59 | 1.49 |
| 10^14 | 4.01 | 3.17 | 3.14 | 3.25 | 3.27 | 3.02 |

An alternative reading, d ln(dF/d ln D)/d ln D = d ln f/dx, gives values between −2.3 and +0.9 for every fit inside its calibrated range. It never reaches 2.5–3.8 there. Only Jenkins, extrapolated below its validity limit (ln σ⁻¹ < −1.2, M ≲ 3×10^10), exceeds 3. Full grid: NBODY_TRACE_massfunction_slopes.csv.

## f_coll = 0.62 ± 0.03 ("Tinker 2008")

Tinker 2008 gives no collapsed-fraction value (text search). Integrating its z=0 Δ=200m f(σ) with Planck 2018:

| M_min (h⁻¹M⊙) | 10^8 | 10^9 | 10^10 | 10^10.5 (calibration edge) | 10^11 | 10^12 | 10^13 |
|---|---|---|---|---|---|---|---|
| F(>M_min) | 0.629 | 0.580 | 0.521 | 0.486 | 0.447 | 0.348 | 0.220 |

F = 0.62 is reached at M_min ≈ 10^8.2 h⁻¹M⊙, and 0.59–0.65 at 10^8.8–10^7.5. That is 1.7–3 dex below the ~10^10.5–10^15.5 range Tinker calibrated. The Tinker f(σ) tends to a constant A at low mass, so F(>M) does not converge (0.70 at 10^6, 0.76 at 10^4). For comparison, the normalised ST and PS functions give 0.47 and 0.65 above 10^10.

## What can honestly be printed
- The published N-body results put the halo virial ratio **2T/|U| at or above 1**, typically 1.05–1.4 and rising with mass. Neto 2007 reports a median slightly above unity. Power 2012 gives η ≈ 1.15 at 10^12 and ≈ 1.25 at 10^15. Klypin 2016 Fig. 6 shows 2K/|W| ≈ 1.1–1.4 uncorrected and ≈ 1.0–1.2 after the surface-pressure correction. These can be cited with the page and figure references above.
- "Six N-body studies measure 2K/|U| = 0.815 ± 0.025" **cannot be printed**. The sources do not report it, two of them do not measure the ratio at all, and the agreement with reciprocals is only approximate and taken from figures. If the argument is restated for |U|/2T, it must say that this quantity is ≈ 0.7–0.95 and mass-dependent. It must also cite figure read-offs and fits (Neto Fig. 2, Bett Fig. 4, Power eqs. 17–18, Klypin Fig. 6), not tabulated "efficiencies". Relaxed/unrelaxed sub-values and the "~5% z=0→1" statement must be dropped.
- **n_eff = 3.22 ± 0.44 cannot be printed as a literature value.** No source defines or reports it, and the components are mis-sourced: 3.8 is the Jenkins fitting exponent, and "n_eff" in Jenkins and Reed is the power-spectrum slope. What can be printed is that d ln F(>M)/d ln D = f/F computed from the published fits is strongly mass-dependent (~0.5 at 10^10, ~1 at 10^12, ~3 at 10^14 h⁻¹M⊙, Planck 2018). A value near 3 holds only for cluster-mass haloes.
- **f_coll = 0.62 must not be attributed to Tinker 2008.** It is a computed integral that needs a stated M_min of ~10^8 h⁻¹M⊙, well below Tinker's calibrated range. Within that range, F(>10^10.5) ≈ 0.49. Because η_vir = 1/(2 f_coll) inherits this M_min choice, the 0.815 also depends on it.
