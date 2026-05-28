# GAPE Validation Record — Master Index

**Last updated: 2026-05-27.** This is the complete, ordered validation record for the GAPE biological-physics framework (the engine behind EDEAR). It supersedes and absorbs the earlier cascade-only README, whose full text is preserved verbatim further down under **"Original cascade README (VAL-037 → VAL-054), preserved."** Nothing has been deleted.

Everything here is the **biology program**. It contains no IAM cosmology content — the Level 1 / Level 2 MCMC chains, the baryon test, Koide, and the virial confirmation live in the separate physics program and are intentionally excluded.

> **How to read this.** Each VAL has a pre-registration, a SHA-256 seal stamped before any β-value access (VAL-050 onward), a reproducible analysis script, and a SHA-locked results JSON in this directory. Outcome labels: **O1** = passed pre-locked criterion · **O2** = partial / differentiating · **O3** = null · **O4** = data-integrity or expected-null · **O5** = unexpected / baseline-dominated · **O6** = panel/transferability flag. A null or negative result that confirms a pre-registered prediction (or a prior finding) is a pass of the scientific method, and is marked as such.

Some single VAL numbers bundle multiple sub-tests: the drift cascade is **35 of 39 pre-specified predictions**, G-003b is **32 posteriors**, the 5×6 evidence matrix is **30 confirmed cells**, and VAL-049 is **12 cohort tests**. The individual-test count therefore runs well past the row count below.

## Validation families at a glance

| Family | Coverage | Rows |
|---|---|---|
| 1 · MCMC atlas calibration | GAPE H_min floors — G-002 (17 chains, methylation) + G-003b (5 chains, 32 posteriors) + bootstrap cross-check | 3 |
| 2 · Methylation | VAL-001 → VAL-013 — cancer signal, field effect, aging, pre-cancer, cross-species | 13 |
| 3 · Multimodal (5 substrates) | VAL-014 → VAL-033 — methylation + nucleosome occupancy + fuzziness + WPS + fragment size | 20 |
| 4 · Multi-class drift cascade | VAL-037 → VAL-046 — 35/39 pre-specified predictions | 10 |
| 5 · Cross-population T-series | VAL-049 (T1 → T15) — panel transfer across 4 populations + pipelines | 15 |
| 6 · EDEAR disease cards | VAL-047 → VAL-128 — per-card validations across 12 disease cards | 76 |
| 7 · IAMAtlas v0.1 build (LIVE) | Production MCMC, 8 architecture-class H_min chains, running since ~May 6, 2026 | 8 |

## Family 1 — GAPE H_min atlas-calibration MCMC (G-series)

These are GAPE atlas-calibration chains (not IAM cosmology). G-002 calibrates the 8 architecture-class methylation floors; G-003b calibrates the 4 non-methylation substrates across all 8 classes (32 posteriors); a 320,000-resample bootstrap independently cross-checks every value. H_min values themselves are part of the proprietary calibration layer (available under NDA).

| Study | What it calibrated | MCMC result | Source | Status |
|---|---|---|---|---|
| G-002 | H_min for 8 architecture classes (methylation) | 17 chains · R-hat < 1.001 · 800,000 samples · 8 class floors converged · H_min_methyl(cycling)=0.856055 ± 0.000312 | NIH Roadmap Epigenomics | CONVERGED — values proprietary |
| G-003b | H_min for 4 non-methylation substrates × 8 classes | 5 chains × 32 walkers × 5,500 steps · R-hat max 1.0009 · 800,000 samples/substrate · 42.1s · 32 posteriors converged | ENCODE / GSE71378 / GSE149268 | CONVERGED — 32 H_min values (8 classes × 4 substrates) |
| G-003b bootstrap cross-check | Non-parametric validation of all 32 H_min values | 10,000 bootstrap resamples × 32 = 320,000 total · MCMC vs bootstrap agree (8 informative disagreements documented) | Same reference cell DBs | CONFIRMED — independent method agreement |

## Family 2 — Methylation validations (VAL-001 → VAL-013)

| VAL | What it tested | Result | Source | Status |
|---|---|---|---|---|
| VAL-001 | Cancer signal, 6 types | 6/6 confirmed | TCGA GDC Portal | PASS |
| VAL-002 | Bulk blood null test (Health ABC, n=20) | Null as predicted; class-stratified best d=0.303 (secretory) p=0.68 | GSE130748 · Luo 2019 | PASS (predicted null) — confirms bulk-blood dilution |
| VAL-003 | Cancer field effect, 28 types, 4,092 matched pairs | 28/28; p=1.32e-15; +20.2% adjacent-normal elevation | TCGA Pan-Cancer Atlas | PASS |
| VAL-004 | OSK rejuvenation (Yamanaka factors), 7/7 predictions | 63.8% aging ΔA reversed (RGC); 84.8% (SH-SY5Y) | Lu 2020 · GSE147436 | PASS |
| VAL-005 | Longitudinal entropy trajectory (n=17 pilot) | directional signal below threshold (best d=-0.303 p=0.68) | GSE130748 · Health ABC 5-yr | UNDERPOWERED — awaits larger cohort |
| VAL-006 | Aging trajectory (n=656); normal aging does NOT reach A=1.05 | r=0.9999 p=6.1e-12; A=1.05 extrapolates to ~-1,075 yr | Hannum 2013 · GSE40279 | PASS |
| VAL-007 | Tissue-specific cfDNA signal, 9/9 P1 | mean ΔA=+0.177; 104,297× bulk-blood improvement | Moss 2018 · GSE122126 | PASS |
| VAL-008 | Specimen matrix, 19 cancer types | 19/19 FLOOR BREACH; mean \|ΔA\|=0.167 (SARC 0.132 → LGG 0.301) | TCGA + Moss 2018 | PASS |
| VAL-009 | Pre-cancer window, cervical (WID-CIN, n=2,254) | 3/5 strict; A=1.015 CIN2, A=1.100 invasive | Widschwendter 2021 | PARTIAL PASS (3/5) |
| VAL-010 | HCC combined score S = fraction × ΔA (novel) | Cirrhosis S=0.072 vs early-HCC S=0.583; 8.03× separation | TCGA-LIHC + Moss 2018 | PASS (AFP cannot discriminate these) |
| VAL-011 | Pre-cancer window, endometrial (n=306) | Monotonic progression; 1/4 strict | Widschwendter 2017 | PARTIAL — tissue-dependent threshold |
| VAL-012 | D+Q senolytic (global-mean proxy) | GAPE ΔA=-0.00079 (decrease) vs Hannum/Horvath/PhenoAge all increase | Lee 2024 | PASS (direction) — only GAPE moves correctly |
| VAL-013 | Cross-species: canine, 3/3 | H_min diff=0.004 across 70M yr; r(dog_age,A)=0.9273; osteosarcoma ΔA=+0.131 vs human +0.136 | Wang 2020 (n=104) · Azambuja · Angstadt | PASS |

## Family 3 — Five-substrate multimodal validations (VAL-014 → VAL-033)

Methylation, nucleosome occupancy, nucleosome fuzziness, windowed protection score (WPS), and fragment-size entropy across field-effect, aging, pre-cancer, and tissue-specific axes. *Note: a few cascade-era effect sizes differ between the GAPE Evidence Report and the Biological_Physics README; both values are shown where they differ, pending reconciliation to a single source of truth.*

| VAL | What it tested | Result | Source | Status |
|---|---|---|---|---|
| VAL-014 | MESA theory — why combining substrates works | inter-substrate r=0.54; d_combined/d_single=1.15× (vs 2.0× if independent → same floor); ceiling AUC=1.000 | Li 2024 · Zenodo:6812876 | PASS |
| VAL-015 | Four Mahaffey values derivation | all four G-003b MCMC confirmed; R-hat<1.001; 42s runtime | ENCODE · Snyder · Cristiano | PASS (MCMC) |
| VAL-016 | Nucleosome occupancy — breast cancer (n=139) | ΔA_nucl FLOOR BREACH (report +0.265 / README +0.55) | Doebley 2022 | PASS [report/README ΔA differ] |
| VAL-017 | Fuzziness — prostate cancer grading (n=26 PDX) | ΔA_fuzz +0.311 (README +0.32); monotonic ARPC→NEPC | Esfahani 2022 | PASS |
| VAL-018 | WPS — 15 tissue types | 15/15; ΔA_WPS +0.501 (README +0.53); 8 yr before MESA | Snyder 2016 · GSE71378 | PASS |
| VAL-019 | Fragment size — 7 cancer types (n=208) | 7/7; ΔA_frag +0.373 (README +0.37); AUC=0.940 | Cristiano 2019 · GSE149268 | PASS |
| VAL-020 | Five-substrate convergence | 5/5 direction; r_inter=0.54; ceiling AUC=1.000 | Combined | PASS |
| VAL-021 | Nucleosome occupancy field effect (22 types) | 22/22; p=3.6e-14; mean ΔA field +0.102 (README +0.218); TGCT inversion | Corces 2018 | PASS [report/README ΔA differ] |
| VAL-022 | Fuzziness field effect (22 types) | 22/22; p=6.9e-12; mean ΔA +0.081 (README +0.084); TGCT inversion | Corces 2018 + Esfahani 2022 | PASS |
| VAL-023 | WPS field effect (22 types) | 22/22; p=9.1e-12; mean ΔA +0.160 (README +0.174); TGCT inversion | Snyder 2016 + Corces 2018 | PASS |
| VAL-024 | Fragment size field effect (22 types) | 22/22; p=9.8e-11; mean ΔA +0.099 (README +0.102); TGCT inversion | Cristiano 2019 + Mathios 2022 | PASS |
| VAL-025 | Nucleosome occupancy aging (human + 104 canine) | r=0.9998 human, r=0.986 canine; slope 24.9× (README 53.6×) methylation | Wang 2020 + Pal 2016 | PASS [report/README slope differ] |
| VAL-026 | Fuzziness aging (human + canine) | r=0.9995 human, r=0.982 canine; slope 20.2× (README 21.0×) | Bochkis 2014 + Ucar 2017 | PASS |
| VAL-027 | WPS aging (human + canine) | r=0.9990 human, r=0.983 canine; slope 37.7× (README 40.9×) | Snyder 2016 + Mouliere 2018 | PASS |
| VAL-028 | Fragment size aging (human + canine) | r=0.9962 human, r=0.993 canine; slope 24.0× (README 24.5×) | Mathios 2022 | PASS |
| VAL-029 | Nucleosome occupancy — tissue-specific cfDNA | FLOOR BREACH; AUC=0.89 (Griffin ER); bulk plasma buried | Doebley 2022 | PASS |
| VAL-030 | Fuzziness pre-cancer window | Monotonic dysplasia gradient; A=1.01-1.05 zone | Esfahani 2022 + Bochkis 2014 | PASS |
| VAL-031 | WPS pre-cancer + field effect | Adjacent-normal WPS depletion (8 yr pre-MESA) | Snyder 2016 Fig 5 | PASS |
| VAL-032 | Fragment size early detection | Pre-dx signal 2 yr before dx; Stage I→IV monotonic | Mathios 2022 | PASS |
| VAL-033 | Complete 5×6 evidence matrix | all 5 substrates MCMC-confirmed; 30/30 cells confirmed | All sources | PASS |

## Family 4 — Multi-class drift cascade (VAL-037 → VAL-046)

Overall: **35 of 39 pre-specified predictions confirmed (89.7%).** The one complete failure (VAL-038) confirms the framework's own prior negative finding (VAL-002): bulk plasma requires deconvolution and does not track tissue-architectural ΔA directly. Full per-script detail and sources are in the preserved original README below.

| VAL | What it tested | Result | Source | Status |
|---|---|---|---|---|
| VAL-037 | Cross-class field effect (24 TCGA types, n=1,109 STN) | 3/4; mean ΔA_field +0.036; 22.9% of tumor signal; 24/24 directionally correct; p<1e-10 | TCGA PanCanAtlas · Roadmap · Moss | PASS (3/4) |
| VAL-038 | Plasma cfDNA pan-cancer correlation (Zeng 2026, n=1,294, 14 types) | 1/3; HONEST NEGATIVE; Spearman ρ=-0.02; confirms VAL-002 | Zeng 2026 Nat Cancer | HONEST NEGATIVE (confirms prior null) |
| VAL-039 | Spatial field-effect gradient (6 distance-annotated cancers) | 4/4; 6/6 monotonic T→N→F→H; far ≥5-10cm still +0.025 | Kadota · Teschendorff · Shen · Damaschke · Villanueva · Kang | PASS |
| VAL-040 | Alzheimer's multi-class peripheral drift (7 class combos) | 4/4; 4 classes elevated (terminal/immune/secretory/stromal); 7/7 severity gradient | De Jager · Shireby · Nabais · Lunnon | PASS |
| VAL-041 | Tissue-of-origin deconvolution localization (10 types) | 4/4; 10/10 top-1 correct; mean max ΔA +0.174 | Moss 2018 · Liu 2020 | PASS — Stage 2 anchor |
| VAL-042 | Monotonic pre-cancer progression (5 systems) | 4/4; 5/5 monotonic; 4/5 FLOOR BREACH; MARGINAL in 5/5 | Widschwendter · Jammula · Jerónimo · Luo · Yoshizato | PASS |
| VAL-043 | Cross-species replication (5 canine cancers, n=104 Labradors) | 4/4; mean cross-species diff 0.010; canine aging r=0.9995 | Wang 2020 · Pal · Beck · Decker · Hendricks | PASS |
| VAL-044 | Post-treatment reserve depletion (5 trials) | 4/4; 5/5 responder vs non-responder separable; CR → A≈1.00 | Ceccarelli · Parikh · Stover · Ley · Cabel | PASS |
| VAL-045 | Inversion specificity (seminoma vs 5 TGCT histologies) | 2/4; seminoma inversion confirmed (A=0.755); 2.1× divergence distinguishes seminoma | Shen 2018 · Killian · TCGA-TGCT | PARTIAL (2/4) |
| VAL-046 | Systemic multi-class pre-dx signature (7 cohort-cancer combos) — capstone | 4/4; 9/9 endpoints ΔA≥0.008; 3 classes; detectable 2-5 yr pre-dx; mean ΔA +0.014 | Kresovich · Hillary · Horvath · Hou | PASS — capstone |

## Family 5 — Cross-population T-series (VAL-049, T1 → T15)

Xu-538 immune panel + 19-CpG secretory panel run through the identical EPIC-Italy pipeline against non-EPIC-Italy cohorts at frozen H_min and frozen panel. Manifest + SHAs in `cross_population/CROSS_POPULATION_MANIFEST.json`. T4 (NOWAC), T6 (Sister Study extension), T7 (Framingham) remain dbGaP-gated.

| T | Cohort | Result | Design | Status |
|---|---|---|---|---|
| T1 | GSE40279 Hannum US healthy (n=656, 450K) | healthy distribution; mean A=0.493 flat across decades | baseline | PASS (calibration baseline) |
| T2 | GSE104942 Australian HBOC (n=191, 450K) | d=+0.291 p=0.056 [+0.03,+0.60]; WPB-only subset d=+0.664 p=0.048 | constitutional/familial | PASS (weak positive) |
| T3 | GSE148663 Uruguayan sporadic (n=32, 450K) | d=-0.373 p=0.33; post-dx design, outlier-driven | post-diagnostic | OUTLIER (design-incomparable) |
| T5 | GSE89093 TwinsUK MZ (paired) | pooled d=+0.16; 0-2yr window positive | paired pre-dx breast | PASS |
| T8 | NHANES 1999-2002 | HR per SD = 1.58 (Cox on published clocks) | framework-premise (orthogonal) | PASS |
| T9 | GSE283951 Polish | d=+0.285 | pre-diagnostic breast | PASS |
| T10 | GSE37965 Heyn UK (paired EpiTwin) | d=+0.177 | paired pre-dx breast | PASS |
| T11 | GSE243529 Singapore | d=+0.120 | at-diagnosis Chinese breast | PASS |
| T12 | GSE314261 St Jude | attempted, not interpretable | specificity attempt | NOT INTERPRETABLE |
| T13 | GSE51057 (secretory) | d=+0.83 at >10yr | multi-class EPIC-Italy | PASS |
| T14 | GSE51032 (secretory) | d=+0.28 pre-dx; CRC divergence | multi-class + CRC | PASS |
| T15 | NHANES 1999-2002 | DunedinPoAm HR=2.14 (blinded prospective) | framework-premise (orthogonal) | PASS |
| T4 | NOWAC Norwegian pre-dx | not deposited / dbGaP-gated | blocked-access | NOT RUN (gated) |
| T6 | Sister Study extension | dbGaP-gated | blocked-access | NOT RUN (gated) |
| T7 | Framingham | dbGaP-gated | blocked-access | NOT RUN (gated) |

## Family 6 — EDEAR per-card disease validations (VAL-047 → VAL-128)

VAL-050 onward are individually pre-registered and SHA-256 sealed before any β-value access. Seal hash prefix or git commit shown where recorded. Each card follows the AD-sprint template: panel selection → holdout recovery → cross-platform → confound bound → honest tier labeling. VALs referenced without a standalone directory (VAL-055, 078–080, 083–087, 114–116) are deferred/biobank-gated/calibration items and are marked as such.

| VAL | Date | Card / Program | What it tested | Cohort / Source (n) | Result | Outcome | Sealed? | Lesson / Note |
|---|---|---|---|---|---|---|---|---|
| VAL-047 | 2026-04-21/23 | breast-epic / crc-epic | Blood pre-dx per-patient (Phase 9 + Phase 12 + Tightening v2) | GSE51057 + GSE51032 EPIC-Italy (n=1,174) | Breast secretory d=+1.85 (Ph9), CRC inverted; pooled +0.45→+0.71, +1.36→+1.78 at >10yr | O1 PASS (paradigm) | SHA-locked matrices | Immune signal monotonically stronger further pre-dx; the gold-standard blood-arm design |
| VAL-048 | 2026-04-20 | crc-epic | Framework-derived 650-CpG cycling panel on CRC pre-dx | GSE51032 (n=845) | Phase 1 too-strict (clean fail, logged); Phase 2 empirical threshold → null | Null (logged) | Prereg v1+v2 | Null motivated directional scoring (→ VAL-051) |
| VAL-049 | 2026-04-21 | cross-population | Cross-population series T1-T15 (see T-series tab) | 12 cohorts, 4 populations | 5/5 informative pre-dx cohorts directionally positive (binomial p=0.031) | Directional PASS | Frozen panel + H_min | Signal transfers across US/AU/UK/PL/CN-SG populations + pipelines |
| VAL-050 | 2026-04-23 | ad-immune | AIBL AD pooled-entropy A-score | AIBL EPIC (n=128 AD/376 HC train; 33/95 holdout) | d=+0.077 p=0.32 AUC=0.51; 7/18 CpGs FDR<0.05 bidirectional | O3 NULL | Hash-sealed | Pooled entropy cancels bidirectional signal — the canonical failure case |
| VAL-051 | 2026-04-23 | ad-immune | AD directional 7-CpG panel (Rule A), sealed holdout recovery | AIBL holdout (n=33 AD/95 HC) | d=+0.624 p=0.0013 AUC=0.677; Male +0.51, Female +0.71 | O1 FULL RECOVERY | Hash-sealed | Directional-Score Principle: A_dir is a first-class GAPE metric |
| VAL-052 | 2026-04-23 | ad-immune | AddNeuroMed cross-platform AD replication | AddNeuroMed GSE144858 (450K, n=93 AD/96 HC) | raw d=+0.33; age-adj d=+0.12; age R²=26% | MIXED (O1 + O3-borderline) | Hash-sealed | Clinical output must be age-adjusted Z (Alpha-Omega §E.5) |
| VAL-053 | 2026-04-23 | ad-immune | Sex-specific vs unified AD panel | AIBL splits (F n=72/217; M n=56/159) | Panel-F worse; Panel-M under-selects; Jaccard 0.10 | O4 unified wins | Hash-sealed | Deploy unified panel, not sex-split |
| VAL-054 | 2026-04-23 | ad-immune | Age-confounding bound on AIBL holdout | AIBL HC-internal permutation | 54b: p=0.003 signal exceeds within-HC variance; 54a flagged non-test | STRONG (54b) | Pre-registered | AD signal exceeds HC noise AND has age-tracked component |
| VAL-055 | (extension) | ad-immune | AD extension (ADNI / direct-access AIBL age metadata) | referenced in evidence report | strengthens with ADNI replication or direct AIBL age metadata | Planned/extension | — | Biobank-access dependent; not yet run |
| VAL-056 | 2026-04-24 | lung-epic | Lung multi-anchor (Kadota field + Moss plasma + TCGA-LUAD/LUSC) | Kadota 2014 + Moss 2018 + TCGA | field effect + top-1/top-2 ratio 60.87×; FLOOR_BREACH | PASS (multi-anchor) | Sealed | Landscape survey is itself a VAL; template for new cards |
| VAL-057 | 2026-04-24 | ad-immune | AD-directional specificity vs FTD and PSP/CBD (5 analyses) | GSE53740 (n=15 AD) | pooled null d=+0.013; male AD d=+0.415; PSP/CBD 5/7 directions preserved; +2.306 SD HC offset | External null + sex recovery | Sealed | Cross-cohort batch-offset warning; tauopathy co-detection |
| VAL-058 | 2026-04-24 | prostate-epic | First own tumor-vs-adjacent-normal tissue arm (secretory) | GSE269244 (n=238 AA men, EPIC; 118 paired) | paired d=+0.497 [+0.314,+0.681] p=1e-07 | O1 PASS | Sealed | Built tier directly from tissue; no prostate blood cohort available |
| VAL-059 | 2026-04-24 | hcc-epic | Cross-cohort HCC: ccfDNA vs whole-blood | GSE298812 ccfDNA (n=245) + GSE281691 (n=481) | ccfDNA d=+0.634 p=0.002 monotonic; whole-blood d=-0.156 NULL | PASS (ccfDNA only) | Sealed | Substrate decides detection: Xu-538+ccfDNA works, +leukocyte doesn't; HIV-HCC confound |
| VAL-060 | 2026-04-24 | breast-epic | Tissue arm TCGA-BRCA (secretory) | TCGA-BRCA HM450 (86 pairs) | paired d=+0.675 [+0.448,+0.902] p=4.4e-09 | O1 PASS | SHAs logged | First retroactive per-card tissue re-validation (CCL-011); OOD transfer argument |
| VAL-061 | 2026-04-24 | crc-epic | CRC tumor TIL immune-compartment (Xu-538 in tumor) | TCGA-COAD HM450 (n=26 paired) | paired d=+1.066 [+0.585,+1.547] p<1e-05 | O2 PASS (strong) | SHA def8a690 | TIL immune positive inside tumor; 1 of 3 CRC compartments |
| VAL-062 | 2026-04-24 | crc-epic | CRC tumor cycling-class rescore | TCGA-COAD HM450 (n=26 paired) | paired d=+0.724 [+0.292,+1.156] p=2.2e-04 | O1 PASS | SHA e8ec05a8 | Cycling-class anchor for CRC tissue arm |
| VAL-063 | 2026-04-24 | lung-epic | Lung tissue arm (cycling) | TCGA-LUAD HM450 (n=29) | paired d=+1.020 [+0.571,+1.469] p=3.9e-08; ever-smoker +1.283; non-smoker n=2 underpowered | O1 PASS (strong) | SHA 80902576 | Largest paired tissue effect to date; smoking-stratified per CCL-009 |
| VAL-064 | 2026-04-24 | hcc-epic | HCC tissue arm, etiology-stratified (secretory) | TCGA-LIHC HM450 (46 pairs) | pooled d=+0.498 p=7.4e-04; non-viral +0.664; viral +0.023 NULL | PASS (non-viral) | SHA 6ce8b346 | Viral adjacent-normal field defect blunts paired contrast (Villanueva 2015) |
| VAL-065 | 2026-04-25 | prostate-epic | Urine arm specimen comparison | GSE119260 Brikun 2018 (n=4 advanced) | urine vs benign paired d=-2.39 (wrong direction) | O5 UNEXPECTED | Sealed f1d1a997 | n=4 too small + advanced-skewed; need larger mixed-stage urine cohort (CCL-026) |
| VAL-066 | 2026-04-25 | pancreatic-epic | PDAC tissue arm cohort #1 | TCGA-PAAD HM450 (n=5 paired) | d=+1.182 p=8.2e-03; 46.9% positive-direction (bidirectional) | O5 UNEXPECTED | Sealed 69420620 | 2nd bidirectional-cancellation disease after AD |
| VAL-067 | 2026-04-25 | pancreatic-epic | PDAC large unpaired tissue case-control | GSE49149 (n=167 tumor/29 normal) | d=+0.249 Welch p=0.22; 50.4% positive — cleanest bidirectional evidence | O3 TISSUE NULL (large cohort) | Sealed f0de98bd | Training set for VAL-069 directional panel |
| VAL-068 | 2026-04-25 | pancreatic-epic | PDAC multi-substrate | GSE74071 (n=7 paired; juice/CAFs) | paired d=-0.310; juice d=-0.720; 52.9% positive; PH64 outlier | O3 TUMOR NULL | Sealed 50c0c7e8 | Direction opposite VAL-066; confirms cohort heterogeneity |
| VAL-069 | 2026-04-25 | pancreatic-epic | PDAC directional Xu-538 fallback (324-CpG, z-scored) | TCGA-PAAD + GSE74071 holdouts | TCGA d=+1.511 p=6.4e-05 (all 7 positive); GSE74071 d=+0.222 partial (PH64) | O2 PARTIAL RECOVERY | Sealed e31de916 | A_dir recommended primary Stage 1 metric for PDAC (CCL-027) |
| VAL-070 | 2026-04-25 | cervical-epic | Not run — landscape error caught at runtime | — | — | NOT RUN (note) | Note logged | Documented not-run, not silent |
| VAL-071 | 2026-04-25 | cervical-epic | Cervical landscape survey | 11 candidate cohorts (HM450+EPIC; tissue/LBC/plasma) | breadth mapped per CCL-029 | Landscape | — | Survey VAL preceding the cervical sprint |
| VAL-072 | 2026-04-25 | cervical-epic | Cervical tissue exploratory TCGA-CESC | TCGA-CESC HM450 (n=3 paired) | paired d=+1.26 CI straddles zero; bidirectional signal | O3 TISSUE NULL (exploratory n=3) | Sealed 5a72e1ec | Entire public TCGA-CESC matched-pair pool is n=3 |
| VAL-073 | 2026-04-25 | cervical-epic | Cervical tissue progression anchor | GSE99511 Verlaat Amsterdam (n=68) | Normal vs CIN3 d=+0.7253 p=0.004; monotonic Normal<CIN3<SCC | O1 TISSUE ANCHOR | Sealed f4f637c3 | Positive anchor — later shown to be cohort outlier |
| VAL-074 | 2026-04-25 | cervical-epic | Cervical tissue replication (HPV-neg normals) | GSE46306 Farkas Stockholm (n=43) | Normal vs CIN3 d=-0.61; not monotonic | O5 NEGATIVE DIRECTION | Sealed e39a4e50 | HPV-neg normals sit at depressed baseline (CCL-019 flip) |
| VAL-075 | 2026-04-25 | cervical-epic | GSE38266 attempted | — | cohort is HNSCC (head/neck), NOT cervical | EXCLUDED (runtime catch) | — | cerv-LL-008; correct catch at sample-title verification |
| VAL-076 | 2026-04-25 | cervical-epic | LBC primary pathway #1 | GSE143752 El-Zein Quebec (n=186, EPIC) | Healthy vs all-lesion d=-0.114 flat | O6 UNEXPECTED (panel transferability) | Sealed e0a949cb | Xu-538 buffy-coat trained; LBC is different cell mixture |
| VAL-077 | 2026-04-25 | cervical-epic | LBC primary pathway #2 (largest LBC) | GSE287994 Bowden London (n=247, EPIC) | d=-0.029 flat | O6 UNEXPECTED → DATA INTEGRITY | Sealed 4fc18252 | File is residual M-values not raw β; defer v0.2+ raw IDAT (CCL-040) |
| VAL-081 | 2026-04-25 | cervical-epic | Cancer-only SCC at large n | GSE68339 Lando Oslo (n=270) | tumors d=-0.43 vs VAL-073 normals; only 6.7% above p95 | O5 NEGATIVE DIRECTION | Sealed a86547ce | 2/3 tissue cohorts read tumors at/below anchor → tier exploratory_with_cohort_heterogeneity |
| VAL-082 | 2026-04-25 | heme-epic | AML myeloid arm at blood level | GSE62298 Glass (n=68) vs EPIC-Italy HC (n=115) | ΔA=+0.1039; d=+3.71 [+3.23,+4.20] p≈0; 98.5% above HC p95 | O1 PASS | Sealed a8c37145 | STRONGEST single-cohort effect in the Cookbook; disease cells ARE the panel cells |
| VAL-083 | (v0.2 queue) | heme-epic | EnviroGenomarkers pre-dx CLL (n=347; 28 future-CLL 2-15.7 yr pre-dx) | EPIC-Italy + NSHDS biobanks | signal exists in published analysis; data biobank-gated | Priority v0.2 (gated) | — | Formal data-access application required (heme-LL-011) |
| VAL-084 | (v0.2 queue) | heme-epic | MARLIN myeloid cross-cohort replication | Capper 2025 (n=2,540 acute leukemia) | reference identified | Queued v0.2 | — | Myeloid arm cross-cohort replication target |
| VAL-085 | (v0.2 queue) | heme-epic | CHIP→AML serial trajectory (G-2026-P010) | roadmap item | — | Queued v0.2 | — | Serial trajectory monitoring |
| VAL-086 | (v0.2 queue) | heme-epic | ICI immunotherapy response (G-2026-P011) | roadmap item | — | Queued v0.2 | — | Treatment-response arm |
| VAL-087 | (v0.2 queue) | heme-epic | Heme v0.2 queue placeholder | validation_runs notes | — | Queued v0.2 | — | Documented placeholder |
| VAL-088 | 2026-04-25 | glioma-epic | Glioma Stage 1 immune A-score in blood | GSE180683 Salas/Wiencke (n=76) vs EPIC-Italy HC | d=+0.91 [+0.61,+1.22]; pre-surg naïve +0.94; LGG>GBM | O1 PASS (revised from O5 inverted) | Sealed | Cell-fraction & A-score directions orthogonal not inverted (revised post VAL-090) |
| VAL-089 | 2026-04-25 | glioma-epic | Glioma tumor tissue arm | GSE60274 Lai (n=72 GBM + 5 NTB + 4 spheres) | GBM primary d=+0.24 (wide CI); recurrent +1.17; spheres -1.81 | O2 PARTIAL (variance high) | Sealed | High A-score = heterogeneity marker NOT tumorness — key biology cross-check |
| VAL-090 | 2026-04-25 | glioma-epic | Cortical-neuron cfDNA in glioma plasma (Loyfer array atlas) | GSE180683 + GSE51057 + GSE60274 + nloyfer/meth_atlas | glioma 1.092% vs HC 0.276%; d=+1.96 [+1.62,+2.31]; pre-surg +1.97 | O1 PASS | Sealed d00c7cee | Brain cfDNA IS detectable on standard EPIC with right atlas; killed '4% floor' assumption (CHK-7.7) |
| VAL-091 | 2026-04-26 | ad-immune | AD Stage 2 neuro deconvolution (layered atlas) | AIBL + AddNeuroMed + GIFT | AIBL d=-0.026; AddNeuroMed d=-0.083; GIFT +0.96 outlier; PSP/CBD -0.51 | O4 AD NEURO NULL | Sealed 56c7cac9 | Confirms Stage 2 NULL for AD; glioma-vs-AD differential tile; AddNeuroMed routing artifact |
| VAL-092 | 2026-04-26 | framework / run-everything | Stage 2 A_terminal on cortical-neuron CpGs (run-everything demo) | GSE51057+180683+60274+153712+144858+53740 | AIBL AD d=-0.228; PSP d=-0.433 (replicable BELOW_NORMAL); glioma blood cross-cohort +0.987 (caveated) | O1 DRIFT DISCRIMINATOR | Sealed 7249e964 | First run-everything architecture demo; within-cohort vs cross-cohort asymmetry flagged |
| VAL-093 | 2026-04-26 | breast-epic | Stage 2 25-tile per-class A-score (Loyfer array) | GSE51057 + GSE51032 | ≥3/4 secretory tiles \|d\|≥0.3; breast not uniquely largest | O2 SECRETORY DISTRIBUTED | Sealed | Distributed-then-localized two-component temporal pattern |
| VAL-094 | 2026-04-26 | breast-epic | EpiSCORE BreastRef sub-cell-type resolution | GSE51057 + GSE51032 (all TTD windows) | 7 sub-types behave as one signal (within 0.10-0.16) | O2 DISTRIBUTED (resolution-collapse) | Sealed 501fafad | Test EpiSCORE refs for resolution-collapse before adding per-sub-cell-type layer |
| VAL-095 | 2026-04-26 | breast-epic | UniLIFE 19-cell vs Salas 6-cell Stage 3 head-to-head | GSE51057 + GSE51032 | aTreg >10yr d=+1.26/+0.79; aBnv 0-2yr +0.44/+0.49 (CIs exclude 0) | O1 RESOLUTION GAIN | Sealed 5f74259d | UniLIFE added as parallel adult-specific Stage 3 overlay; Salas stays production |
| VAL-096 | 2026-04-26 | breast-epic | TTD-window stratification on Loyfer 25-tile | GSE51057 + GSE51032 (re-slice of VAL-093) | breast tile +0.43/+0.49 at 0-2yr only; immune inversion near-dx | O1 (partial) + O4 amplifies near-dx | Sealed 01247146 | Two-component temporal model; localization happens near diagnosis (CCL-035 candidate) |
| VAL-097 | 2026-04-28 | lung-epic | Never-smoker LUAD 25-tile cross-cohort | GSE256092 Korean NSLA (n=141) vs TCGA-LUAD adj-normal (n=29) | Lung_cells d=-0.27 (2nd-weakest); 11/25 tiles >3 SD; 22/25 same direction | O5 BASELINE DOMINATED | Sealed 9a1bd45e | Honest override of auto-O2 (CHK-4.8); SWAN-vs-sesame + population + composition variance; CCL-036 |
| VAL-098 | 2026-04-28 | crc-epic | Early-onset rectal subsite tissue arm | TCGA-READ HM450 (n=7 paired) | cycling d=+0.612 [+0.227,+1.882]; Colon tile -2.501; Rectum-NOS +0.750 | O1 CYCLING RECTAL CONFIRMED | Sealed 57d830d6 | Extends CRC cycling anchor to rectal subsite; CCL-039 marker-tile fidelity (CHK-4.11) |
| VAL-099 | 2026-04-28 | crc-epic | TCGA-COAD age-stratified re-analysis | TCGA-COAD HM450 (n=26 paired, cached) | d=+0.7241 reproduces VAL-062 byte-for-byte; under-50 ΔA=+0.0357 (descriptive) | O1 AGE-STRATIFIED CONFIRMED | Sealed 8e4ee02c | Reproducibility proof; 3rd CCL-039 confirmation; Colon tile d=-1.603 |
| VAL-100 | 2026-04-28 | crc-epic | Under-50 buffy-coat polyp Stage 1 (EPIC v2.0) | GSE282666 Kumar (n=51, GPL33022) | extreme 3.9%/middle 6.8% FAILS raw-β; +15.13 anchor-SD off; d=+0.236 descriptive | O5 DATA INTEGRITY FLAG | Sealed 4017913d | First EPIC v2.0 VAL; supplementary = noob-bg-corrected, not raw β; defer v0.2+ (CCL-040) |
| VAL-101 | 2026-04-28 | hcc-epic | HCC 25-tile etiology stratification (Marcus-analog stratum) | TCGA-LIHC HM450 (n=46, same as VAL-064) | extreme 26.6%/middle 9.1% trips CHK-3.1; Hepatocytes tile d=-1.521 (descriptive) | O5 DATA INTEGRITY FLAG | Sealed fa366bf0 | Clean biology does NOT justify post-hoc threshold relaxation; card stays v0.2 |
| VAL-102 | 2026-04-28 | hcc-epic | Re-seal attempt under platform-tuned threshold | TCGA-LIHC HM450 | — | VOIDED before execution | Voided 2b77ad9d | Post-hoc threshold from tripped data = circular; voided in 4 min. CCL-041 born |
| VAL-106 | 2026-04-28 | calibration | TCGA HM450 sesame CHK-3.1A baseline | TCGA-KIRC+PRAD adj-normal (n=210) | full-genome f_extreme ~55.87%; KIRC vs PRAD p=1.55e-05 | O3 CALIBRATION DEGENERATE → anchor | Sealed 0330a3c6 | Standing healthy-substrate calibration cohort; triggered CHK-3.1A/B split (CCL-042) |
| VAL-107 | 2026-04-28 | cardio-epic (calib) | CHK-3.1B threshold on 8,100-CpG marker subset | TCGA HM450 subset (n=210) | subset f_extreme ~63%; threshold extreme≥55%, middle≤8.5% | O2 PLATFORM DIVERGENCE | Sealed b58ce4db | Unblocks cardio-epic CHK-3.1B reference threshold |
| VAL-108 | 2026-04-28 | cardio-epic | Ischemic stroke 3-subtype (TOAST) | GSE69138 (n=404→383, whole blood) | every d<0.5; max \|d\|=0.167 | O3 3-SUBTYPE UNDIFFERENTIATED | Sealed 6f40ebd9 | Biology-correct null; post-stroke inflammation homogenizes immune methylation |
| VAL-109 | 2026-04-28 | cardio-epic | PAH cultured pulmonary endothelial cells | GSE84395 (n=39→37 PEC) | Vascular tile control vs hPAH d=+0.79; Stage 1 +0.65 | O2 VASCULAR TILE DIFFERENTIATING | Sealed f6450b4c | Pure cell type → pure cell signal; hPAH > iPAH (BMPR2 germline) |
| VAL-110 | 2026-04-28 | cardio-epic | Ascending aorta dissection / BAV+dilation | GSE84274 (n=24 tissue) | Stage 1 immune normal vs BAV d=+1.08; Vascular tile NOT differentiating (\|d\|≤0.15) | O2 AORTIC TILE DIFFERENTIATING | Sealed 1041738c | Substrate-cell match matters (LL-CARDIO-001); aorta is SMC/fibroblast-dominated |
| VAL-111 | 2026-04-29 | cardio-epic | EpiSCORE HeartRef Stage 2 atlas integration | GSE69138+84395+84274 (652 samples) | all 5 cardiac tiles ~0.46-0.51 regardless of substrate; max discrimination 0.0152 vs 0.10 threshold | O3 TISSUE FLOOR DOMINATED | Sealed 172c6ae2 | First negative atlas-integration result; gene-promoter ≠ tile-coverage atlas (CHK-5.11) |
| VAL-112 | 2026-04-29 | cardio-epic | Run-everything multi-atlas calibration + rescoring | TCGA n=210 calib; 3 cardio cohorts (31,948 readings) | PAH convergent across 3 atlases (Caggiano heart +1.42); stroke convergent null max \|d\|=0.19 | Calibration sealed | Sealed (commit pushed) | Corrects run-everything violation; deduped atlas (CCL-047); substrate-norm required (CCL-048) |
| VAL-113 | 2026-04-29 | cardio-epic | Caggiano CelFiE TIM array-bridged calibration | TCGA n=210 + 3 cardio cohorts | 254-CpG / 19-tile atlas calibrated and scored | Calibration sealed | Sealed (commit pushed) | Third atlas in cardio run-everything stack |
| VAL-114 | (cardio Wave-1) | cardio-epic | Wave-1 calibration on Hannum GSE40279 (n=656) | healthy aging blood | cohort-substrate precheck baked in | Wave-1 calibration | — | Wave-1 protocol calibration |
| VAL-115 | (cardio Wave-1) | cardio-epic | Wave-1 promotion-path calibration | Caggiano TIM immune subset (1,906 CpGs × 19 tiles); Salas IDOL | promotion-path reference | Wave-1 calibration | — | Wave-1 protocol calibration |
| VAL-116 | (cardio Wave-1) | cardio-epic | Wave-1 calibration close-out | Wave-1 protocol | — | Wave-1 calibration | — | Wave-1 protocol calibration |
| VAL-117 | 2026-04-30 | prostate-epic | EpiSCORE ProstateRef Phase B calibration | TCGA HM450 adj-normal (n=210) | all CHK gates clear; LE tile mean 0.4254 sd 0.0041 q5 0.4190 | O1 (calibration sealed) | Sealed 40ce175 | ProstateRef does NOT collapse to floor — DISC-PROSTATE-001 (cell-type distinctness) |
| VAL-118 | 2026-04-30 | prostate-epic | Phase C run-everything multi-atlas rescoring | GSE269244 (n=238; 118 paired) | LE tile d=-0.767; other 5 tiles +0.48 to +1.31; Salas Mono +0.771; Xu-538 reproduces VAL-058 | O1 + O2 (LE_NEGATIVE) + O4 | Sealed edf6229→58ecd16 | 5-vs-1 direction split = prostate signature; CHK-2.7 magnitude-\|d\| rule (DISC-PROSTATE-002) |
| VAL-119 | 2026-05-01 | bladder-epic | EpiSCORE BladderRef Phase B calibration | TCGA-KIRC+PRAD adj-normal (n=210) | max within-cohort tile range 0.0694 (Epi tightest 0.0410); 4 cell types clean | O1 CALIBRATION SEALED | Sealed 404eed3 | DISC-BLADDER-001: distinctness not count drives gene-promoter atlas fitness |
| VAL-120 | 2026-05-01 | bladder-epic | Stage 1 Xu-538 on bladder | TCGA-BLCA (n=440; 21 paired) | panel coverage 51.1% pass; diagnostic paired d=+1.8977 p=3.14e-08 (not sealed) | O4 DATA INTEGRITY (panel coverage) | Sealed 404eed3 | Xu-538 transferability cohort-specific (DISC-BLADDER-004 → CHK-2.17); floor amended (DISC-BLADDER-002 → CHK-2.16) |
| VAL-121 | 2026-05-01 | bladder-epic | Stage 2 multi-atlas direction check | TCGA-BLCA (n=21 paired) | Loyfer Bladder +1.91 vs BladderRef Epi -1.46 (same pairs); all 14 Loyfer non-bladder tiles +2.34 to +2.92 | O2 DIRECTION AMBIGUOUS | Sealed 404eed3 | Bulk-WGBS inflates on mucosal substrate → gene-promoter atlas must be primary (DISC-BLADDER-003 → CHK-2.18) |
| VAL-122 | 2026-05-01 | bladder-epic | Stage 3 immune fine-tune | TCGA-BLCA (n=21 paired) | 6/6 Salas IDOL tiles +0.49 to +1.24 (broad infiltration) | O1 IMMUNE DIFFERENTIATING | Sealed 404eed3 | Broad TIL+TAM+MDSC pattern characteristic of muscle-invasive bladder cancer |
| VAL-123 | 2026-05-02 | gastric-esophageal-epic | BoccellatoStomachRef HM450 Phase B calibration | TCGA adj-normal (n=210) | calibration sealed; 738,115 CpGs × 6 tiles | O1 BOCCELLATO CALIBRATION SEALED | Sealed d7c26f6 | Frozen atlas artifact; SWAN+ChAMP-filtered gastric mucosoid reference |
| VAL-124 | 2026-05-02 | gastric-esophageal-epic | EpiSCORE EsoRef bridged Phase B calibration | TCGA adj-normal (n=210) | cross-tile separation 0.0990 (largest of any EpiSCORE bridge); 2,464 CpGs × 8 esophageal cell types | O1 CALIBRATION SEALED | Sealed d7c26f6 | Added mid-sprint for esophageal cell-of-origin coverage |
| VAL-125 | 2026-05-02 | gastric-esophageal-epic | EpiSCORE OEref bridged Phase B calibration (cross-card arm) | TCGA adj-normal (n=210) | 4/9 strict floors cleared; cross-tile separation 0.0407; 5,396 CpGs × 9 oral cell types | O2 PARTIAL FLOORS | Sealed d7c26f6 | Partial-floor result documented honestly; all 9 q5 sealed for production |
| VAL-126 | 2026-05-02 | gastric-esophageal-epic | module_1 STAD Phase C run-everything (8 atlases) | TCGA-STAD (n=395 tumor + 2 paired normal) | f_extreme 0.4399 = 5.02 anchor-SD shift (tier-3); within-cohort MSI d=+4.03 ≈ EBV +3.85 > CIN +3.30 > GS +2.89 | O5 SUBSTRATE BASELINE T3 + O1 WITHIN-COHORT SIGNAL | Sealed d7c26f6 | DISC-GE-001: subtype hierarchy robust under substrate shift; H. pylori + Lauren stratified |
| VAL-127 | 2026-05-02 | gastric-esophageal-epic | module_2 ESCA Phase C run-everything (8 atlases) | TCGA-ESCA (n=185: 96 ESCC + 89 EAC) | ESCC +2.64 / EAC +3.70; d_ESCC-EAC=-1.06 p=1.5e-11; Barrett+ d=+4.50 vs Barrett- +2.81 (+1.69) | O1 SUBTYPE DISCRIMINATION + O5 T3 + O1 BARRETT | Sealed d7c26f6 | DISC-GE-002 first >1 d-unit histological discrimination; DISC-GE-004 Barrett amplification |
| VAL-128 | 2026-05-02 | gastric-esophageal-epic | module_3 Crohn's-pathway blood (8 atlases) | GSE87650 (n=240 sorted + wh-blood companion) | max \|d_CD-HC\|=1.72 (UniLIFE aCD8Tnv); Stage 1 cycling panel does NOT detect IBD (\|d\|<0.5) | O1 CROHNS LANGUAGE SUPPORTED + O5 MIXTURE-ATTENUATION REVERSAL | Sealed d7c26f6 | DISC-GE-005 (FOUNDATIONAL) mixture-attenuation reversal; DISC-GE-006 cycling panel IBD null |

## Family 7 — IAMAtlas v0.1 production MCMC build (LIVE)

The architecture-class H_min atlas under active construction — chains running around the clock since ~May 6, 2026. 5 of 8 classes converged clean; cycling running; progenitor and immune queued (immune is the largest class at 4.99M rows and dominates the back end). Two blocking gates precede v1: the **stromal re-run** with label harmonization (17 atlas labels collapsed to 9 canonical cell types — fixes an identifiability degeneracy, not a sampler-geometry problem), and the **Gasparoni terminal-class addition** (GSE66351, 957,638 rows) + terminal re-run. Production artifacts live in `iamatlas_production_data/` (git LFS). Status below reflects the 2026-05-08 roadmap and updates as chains complete.

| Architecture class | Chain | MCMC status | Convergence | Note |
|---|---|---|---|---|
| terminal | Per-class H_min production chain | R-hat 1.01 · ESS 1493 · 0 div · Pearson 0.799 · n=30,895 | CLEAN ✓ | Gasparoni GSE66351 (957,638 rows) staged to deepen → ~988K rows; terminal re-run is GATE 3 |
| secretory | Per-class H_min production chain | R-hat 1.02 · ESS 1023 · 0 div · Pearson 0.791 · n=1,211,597 | CLEAN ✓ | Breast/liver/pancreas/prostate |
| stem_pluri | Per-class H_min production chain | R-hat 1.01 · ESS 501 · 2 div · Pearson 0.919 · n=482,421 | CLEAN ✓ | ESC/iPSC; no re-run needed for v1 |
| stem_adult | Per-class H_min production chain | R-hat 1.01 · ESS 723 · 0 div · Pearson 0.904 · n=482,421 | CLEAN ✓ | HSC/NSC |
| stromal | Per-class H_min production chain | R-hat 3.67 · ESS 4 · 6 div · Pearson 0.897 · n=96,733 | FAILED → rerun staged (GATE 1) | 17 atlas labels = identifiability degeneracy; harmonized to 9 canonical cell types for rerun |
| cycling | Per-class H_min production chain | in flight (~batch 20/77; ~28 hr remaining as of 2026-05-08) | RUNNING ⏳ | Gut/skin/lung/bladder |
| progenitor | Per-class H_min production chain | queued (~50 hr after cycling) | QUEUED | GMP/CMP/NPC |
| immune | Per-class H_min production chain | queued (~100+ hr after progenitor; 4.99M rows, largest class) | QUEUED | T/B/NK; dominates back end of timeline; relaunched at batch_size 1500 per recent session |

---

## Original cascade README (VAL-037 → VAL-054), preserved

*The following is the previous version of this file, retained verbatim. It carries the per-script inventory, run instructions, primary-source citations, and the Directional-Score Principle write-up. Where it and the master index above differ in scope, the master index is the more complete record; where they differ in detail, both are kept.*

### GAPE Validation Runs — Multi-Class Drift Cascade

This directory contains VAL-037 through VAL-047 validation scripts and their JSON results.
The 80-cell healthy baseline reference is part of the proprietary calibration layer
(available under NDA — email hmahaffeyges@gmail.com).

#### Cascade Summary (April 18, 2026)

**Overall: 35 of 39 pre-specified predictions confirmed (89.7%)**

| ID | Title | Predictions Passed |
|---|---|---|
| VAL-037 | Cross-class field effect (24 TCGA types) | 3/4 |
| VAL-038 | Plasma cfDNA pan-cancer correlation (Zeng 2026) | 1/3 (honest negative) |
| VAL-039 | Spatial field effect gradient (6 cancers) | **4/4** |
| VAL-040 | Alzheimer's multi-class peripheral drift | **4/4** |
| VAL-041 | Tissue-of-origin deconvolution localization | **4/4** |
| VAL-042 | Pre-cancer monotonic progression (5 systems) | **4/4** |
| VAL-043 | Cross-species cancer replication (canine) | **4/4** |
| VAL-044 | Post-treatment reserve depletion (5 trials) | **4/4** |
| VAL-045 | Inversion detection specificity (seminoma) | 2/4 |
| VAL-046 | Systemic multi-class pre-diagnostic signature | **4/4** |

#### Running the scripts

```bash
pip install numpy scipy
cd validation_runs
python3 VAL_037_field_effect_cross_class.py
### ... each runs in ~30 seconds
```

All scripts use published primary-source β values. No proprietary data.
Each implements pre-specified predictions with explicit pass/fail outputs.

#### Files

- `VAL_037` through `VAL_046` — cascade validation scripts + JSON results
- `VAL_047_*` — external validation on real per-patient 450K β values (GSE51057, GSE51032, GSE69914) — first individual-sample-level validation
- `CASCADE_SUMMARY.py` — aggregates all cascade results
- `CASCADE_SUMMARY.json` — summary JSON

#### VAL-047 script inventory

- `VAL_047_real_analysis.py` — first-pass: bulk immune mean β on GSE51057 (null result, Cohen's d=0.08, confirms Xu 2019 KS p=0.5 finding)
- `VAL_047_extended_v2.py` — architectural variance, multi-class signature, directional drift, top-N CV analyses
- `VAL_047_options_1_2.py` — headline script: Xu-2019 6-CpG directional score + time-to-dx stratification. CV Cohen's d = +0.605 ± 0.190 on breast pre-dx.
- `VAL_047_replication.py` — GSE51032 replication (n=845): breast CV d = +0.379, colorectal CV d = +0.835
- `VAL_047_option3.py` — GSE69914 tissue-level validation: monotonic healthy → adjacent → tumor, AUC 0.70

#### Primary sources (by validation)

- **VAL-037/VAL-039**: TCGA PanCanAtlas + Roadmap Epigenomics + Moss 2018
- **VAL-038**: Zeng 2026 Nature Cancer (doi:10.1038/s43018-026-01116-3)
- **VAL-040**: De Jager 2014 + Shireby 2022 + Nabais 2021 + Lunnon 2014
- **VAL-041**: Moss 2018 + Liu 2020 Ann Oncol
- **VAL-042**: Widschwendter 2021 + Jammula 2020 + Jerónimo 2008 + Luo 2014 + Yoshizato 2020
- **VAL-043**: Wang 2020 Cell Reports + Pal 2016 + Beck 2020 + Decker 2015 + Hendricks 2018
- **VAL-044**: Ceccarelli 2016 + Parikh 2019 + Stover 2018 + Ley 2010 + Cabel 2018
- **VAL-045**: Shen 2018 + Killian 2016 + TCGA TGCT 2018
- **VAL-046**: Kresovich 2019 + Hillary 2020 + Horvath 2014 + Hou 2012 + Horvath 2015
- **VAL-047**: Xu 2020 JNCI (doi:10.1093/jnci/djz065) + Kresovich 2022 Mol Onc (doi:10.1002/1878-0261.13087) + Teschendorff 2016 Nat Commun (doi:10.1038/ncomms10478) + Demetriou 2013 (GSE51057 primary) + Zhao 2020 BMC Cancer (doi:10.1186/s12885-020-07194-5)

Full citation list available under NDA — email hmahaffeyges@gmail.com for access.

---

#### April 2026 extensions — VAL-048 through VAL-051

The cascade above (VAL-037 through VAL-047) was deposited in April 2026 as a single snapshot. Subsequent validations extend the framework to:

##### VAL-048 — Framework-derived cycling CpG panel on colorectal pre-dx
Location: `cross_population/` (committed 2026-04-22)
Status: Honest v1 prereg failure documented; v2 prereg with empirically-justified threshold logged as null result on 650-CpG framework-derived cycling panel against GSE51032 (n=845). The null motivated the move to directional scoring (see VAL-051).

##### VAL-049 — Cross-population T1–T15 series
Location: `cross_population/CROSS_POPULATION_MANIFEST.json` (committed 2026-04-22)
Status: 12 of 15 cohorts run; 5 of 5 informative pre-diagnostic breast cohorts directionally positive (one-sided binomial p=0.031 excluding T3 Uruguayan post-dx). T4 NOWAC, T6 Sister Study, T7 Framingham blocked on dbGaP.

##### VAL-050 — AIBL Alzheimer's cross-sectional
Location: `val_050_aibl/`
Status: Pre-registered, hash-sealed. **OUTCOME 3 — NULL** on pooled-β entropy A-score (d=+0.077, p=0.32, AUC=0.51). 7 of 18 panel CpGs individually FDR<0.05 with bidirectional pattern (4 down, 3 up in AD). Motivates directional scoring.

##### VAL-051 — AD-directional immune panel, holdout recovery
Location: `val_051_ad_directional/`
Status: Pre-registered, hash-sealed. **OUTCOME 1 — FULL RECOVERY** on sealed AIBL holdout (d=+0.624, p=0.0013, AUC=0.677). Pooled-entropy null (d=+0.056) holds on same holdout, confirming the recovery is metric-specific. Both sexes significant (Male d=+0.51, Female d=+0.71). Cross-platform replication on AddNeuroMed GSE144858 pending (VAL-052).

##### The Directional-Score Principle

VAL-051 introduces a first-class GAPE metric: the **directional A-score** (A_dir = mean of direction × z-score across a disease-directional panel). This is complementary to, not a replacement for, the pooled-β entropy A-score.

- **Entropy A-score** works when a disease shifts the entire panel uniformly (e.g. breast secretory in VAL-047 Phase 9: d=+1.85).
- **Directional A-score** works when a disease shifts different CpGs within a class in competing directions (e.g. AD immune: 4 CpGs down + 3 up, pooled mean cancels).

For each new disease, the panel selection is outcome-blind on a training split, the sign of Δβ on training assigns direction, and the scoring on holdout uses training-derived directions and standardization. VAL-050 + VAL-051 on the same cohort with the same CpGs is the definitive side-by-side demonstration: pooled entropy d=+0.08 (null) vs directional d=+0.62 (recovered).

##### VAL-052 — AddNeuroMed cross-platform AD replication
Location: `val_052_addneuromed/`
Status: Pre-registered, hash-sealed. **MIXED outcome** — raw d=+0.33 (OUTCOME 1 cross-platform replication) AND age-corrected d=+0.12 (OUTCOME 3-borderline). Age explains R²=26% of A_dir variance on AddNeuroMed. Clinical deployment requires age-adjusted Z-score (Alpha-Omega §E.5), not raw A_dir.

##### VAL-053 — Sex-specific AD panel selection
Location: `val_053_sex_panels/`
Status: Pre-registered, hash-sealed. **OUTCOME 4 — unified panel wins.** Panel-F (10 CpGs) slightly worse than unified on female holdout; Panel-M (1 CpG) under-selects and fails coverage gate. Jaccard 0.10 between F and M panels. EDEAR deploys unified panel.

##### VAL-054 — Age-confounding bound on AIBL
Location: `val_054_age_bound/`
Status: VAL-054a (cellular-age regression) honestly flagged as non-test (80-cell baseline incompatible with panel-subset β). VAL-054b (HC-internal permutation bound) **STRONG** — p=0.003 that observed AD signal exceeds any within-HC variance source collectively. Complements VAL-052 age regression on AddNeuroMed; together they establish: AD signal exceeds HC-internal noise AND has an age-tracked component that requires §E.5 adjustment.

##### AD Sprint — Consolidated

VAL-050 through VAL-054 together establish the AD-Immune directional test for EDEAR:

- **Panel:** 7 CpGs (cg16867657, cg25809905, cg22454769, cg09809672, cg26614073, cg00431549, cg02228185), unified across sexes, directional weighting
- **Validation tier:** `cross_platform_validated` (AIBL EPIC internal holdout + AddNeuroMed 450K external)
- **Performance:** raw AUC 0.60-0.68; age-adjusted ~0.55-0.60
- **Primary clinical output:** age-adjusted Z-score per Alpha-Omega §E.5
- **Secondary output:** raw A_dir to capture accelerated-aging-in-AD component
- **Known limitations:** d ~0.12 after age adjustment; deployment is cohort screening and serial trajectory monitoring, not single-shot diagnosis
- **Pending:** ADNI, Framingham (dbGaP-blocked); AIBL direct-access age metadata

This is the template every future disease card in the Cookbook follows: panel selection + holdout recovery + cross-platform + age-confound bound + honest tier labeling.
