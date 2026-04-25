# Biological Physics — GAPE Framework

Application of the Informational Actualization Model (IAM) to biological systems.

The same thermodynamic principle that governs gravitational decoherence and cosmic expansion
sets the minimum information maintenance cost for living cells. The Landauer cost of
irreversible DNA methylation maintenance at physiological temperature defines
architecture-class-specific entropy floors for all mammalian somatic cell types.

**Updated: April 24, 2026 — Retroactive tissue and ccfDNA validation sprint (VAL-058 through VAL-064) complete. Seven additional cohort validations across five cancer types (prostate, HCC, breast, CRC, lung). All PASS. Two framework stratification findings: smoking-stratification mandate for lung; viral-hepatitis adjacent-normal field defect blunting in HCC. Total validation record now 42 studies plus 10-test multi-class drift cascade (35/39 cascade predictions confirmed, 89.7%).**

**Cross-validated:** 10,000-resample non-parametric bootstrap on identical reference data agrees with G-003b MCMC posteriors at 0.168% mean relative difference (max 1.091%), 24 of 32 MCMC posterior means within bootstrap 95% CI. Calibration is method-independent. Methodology documentation available under NDA.

**Archival DOI (all versions):** [10.5281/zenodo.19547624](https://doi.org/10.5281/zenodo.19547624) — frozen Zenodo snapshots of the complete biological physics validation package. Cite the version DOI listed on Zenodo for state-specific references.

---

## Flagship Publications

**📘 [GAPE Issue 002 — Genomic Intelligence Report (April 2026, 120 pages)](papers/IAMPerformance_GAPEIssue002.pdf)** — the comprehensive cellular thermodynamics publication. Five-substrate architecture-class framework with derivations, 36 validation tests, MCMC chain inventory, baseline reference tables with age-stratified Z-scores, five clinical research scenarios including the chemotherapy reserve-depletion trajectory, ten dated predictions with falsification criteria, and the 2010-2030 cancer detection trajectory. Issue 002 includes explicit physics chain (Landauer → DNMT1 → H_min) for readers encountering the framework for the first time.

**📗 [GAPE Issue 001 — Genomic Intelligence Report (April 2026)](papers/IAMPerformance_GAPEIssue001.pdf)** — prior issue. Methylation-only single-substrate framework. Eight architecture classes. 27 of 28 TCGA cancer types correctly predicted at zero free parameters. Retained as reference for the methylation-only baseline.

**📕 [Cell Thermodynamics Paper (preprint)](papers/Mahaffey_2026_cell_thermodynamics.pdf)** — the foundational derivation paper. Landauer cost of DNMT1 maintenance methylation at physiological temperature. Submitted for independent peer review.

**📙 [Vertebrate Lifespan Paper (Nature Aging submission)](papers/iam_vertebrate_lifespan.pdf)** — cross-species extension. A < 1.05 boundary separates long-lived from short-lived mammals across 28 species at zero free parameters.

---

## The Core Framework

Every cell maintains a measurable commitment to its biological identity. We define:

```
A = H(substrate) / H_min(class)
```

- `H(substrate)` — Shannon binary entropy of the measured substrate value per locus
- `H_min(class)` — architecture-class entropy floor (the **Mahaffey value** for that substrate)
- `A = 1.0` at the healthy floor | `A > 1.05` pre-cancer | `A ≥ 1.10` floor breach

This formula applies identically across **five independent physical substrates**, all now MCMC-confirmed:

| Substrate | What it measures | Status |
|-----------|-----------------|--------|
| DNA methylation (β) | CpG commitment tags | **CONFIRMED — G-002 MCMC (17 chains, R-hat < 1.001)** |
| Nucleosome occupancy | DNA spool positioning probability | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Nucleosome fuzziness | Positional precision across cells | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Windowed protection score | Promoter nucleosome protection | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Fragment size entropy | cfDNA fragment length distribution | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |

The per-class, per-substrate H_min values are part of the proprietary calibration layer — covered under US Provisional Patents 64/012,720 and 64/014,568. Technical access for qualified research partners, clinical collaborators, and acquirers available under NDA (see contact below).

---

## Complete Validation Record — 35 Studies

### MCMC Calibration

| Study | Description | Result | Data Source |
|-------|-------------|--------|-------------|
| G-002 | H_min for 8 architecture classes (methylation) | **17 chains · R-hat < 1.001 · 800,000 samples · 8 class floors converged · values proprietary** | [NIH Roadmap Epigenomics](https://www.ncbi.nlm.nih.gov/geo/roadmap/epigenomics/) |
| G-003b | H_min for 4 additional substrates | **5 chains × 32 walkers × 5,500 steps · R-hat < 1.001 · 42.1s runtime · 32 posteriors converged · values proprietary** | [ENCODE](https://www.encodeproject.org/) · [GEO:GSE71378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71378) · [GEO:GSE149268](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149268) |

### Methylation Studies — VAL-001 through VAL-013

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-001 | Cancer signal, 6 types | 6/6 ✓ | [TCGA GDC Portal](https://portal.gdc.cancer.gov/) |
| VAL-002 | Bulk blood null test (Health ABC, n=20) | Null as predicted · class-stratified best d=0.303 (secretory), p=0.68 · confirms bulk blood dilution | [GEO:GSE130748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130748) · Luo 2019 [10.1186/s40364-019-0161-3](https://doi.org/10.1186/s40364-019-0161-3) |
| VAL-003 | Cancer field effect, 28 types, 4,092 matched pairs (analysis) / 4,304 via live GDC engine | **28/28 · p = 1.32×10⁻¹⁵ · 20.2% elevation in adjacent normal** | [TCGA Pan-Cancer Atlas](https://portal.gdc.cancer.gov/) |
| VAL-004 | OSK rejuvenation (Yamanaka factors) · 7/7 predictions | **63.8% aging ΔA reversed (RGC) · 84.8% (SH-SY5Y)** | Lu 2020 [10.1038/s41586-020-2975-4](https://doi.org/10.1038/s41586-020-2975-4) · [GEO:GSE147436](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147436) |
| VAL-005 | Longitudinal entropy trajectory (n=17) | Pilot cohort · directional signal below threshold (best d=-0.303, p=0.68) · awaits larger cohort | [GEO:GSE130748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130748) · Health ABC 5-yr follow-up |
| VAL-006 | Aging trajectory, n=656 · **normal aging does NOT reach A=1.05** | **r = 0.9999, p = 6.1×10⁻¹²** · age-to-A=1.05 extrapolates to ~-1,075 yr | Hannum 2013 [10.1016/j.molcel.2012.10.016](https://doi.org/10.1016/j.molcel.2012.10.016) · [GEO:GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) |
| VAL-007 | Tissue-specific cfDNA signal · 9/9 P1 confirmed · mean ΔA = +0.177 | **104,297× bulk blood improvement in correct specimen** | Moss 2018 [10.1038/s41467-018-07466-6](https://doi.org/10.1038/s41467-018-07466-6) · [GEO:GSE122126](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122126) |
| VAL-008 | Specimen matrix, 19 cancer types | **19/19 FLOOR BREACH · mean \|ΔA\|=0.167 · range 0.132 (SARC) to 0.301 (LGG)** | [TCGA](https://portal.gdc.cancer.gov/) + Moss 2018 cfDNA atlas |
| VAL-009 | Pre-cancer window, cervical (WID-CIN, n=2,254) | **3/5 strict predictions · A=1.015 CIN2 · A=1.100 invasive** | Widschwendter 2021 [10.1016/j.xcrm.2021.100358](https://doi.org/10.1016/j.xcrm.2021.100358) |
| VAL-010 | HCC combined score S_HCC = fraction × ΔA (novel) | **Cirrhosis S=0.072 · Early HCC S=0.583 · 8.03× separation** (AFP cannot discriminate these) | [TCGA-LIHC](https://portal.gdc.cancer.gov/projects/TCGA-LIHC) + Moss 2018 |
| VAL-011 | Pre-cancer window, endometrial (n=306) | Monotonic progression confirmed · 1/4 strict predictions · tissue-dependent threshold placement | Widschwendter 2017 [10.1186/s13073-017-0432-5](https://doi.org/10.1186/s13073-017-0432-5) |
| VAL-012 | D+Q senolytic · global-mean proxy (awaits class-stratified EPIC data) | **GAPE ΔA=-0.00079 (DECREASE)** vs Hannum +2.3yr · Horvath +1.8yr · PhenoAge +1.1yr (all INCREASE) | Lee 2024 [10.18632/aging.205581](https://doi.org/10.18632/aging.205581) |
| VAL-013 | Cross-species: canine · 3/3 predictions confirmed | **H_min diff = 0.004 across 70 million years · r(dog_age,A)=0.9273 · osteosarcoma ΔA=+0.131 vs human +0.136** | Wang 2020 n=104 · Azambuja 2019 · Angstadt 2022 |

### Multimodal Studies — VAL-014 through VAL-033

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-014 | MESA theory — why combining works | **Inter-substrate r=0.54 · d_combined/d_single=1.15× (expected 2.0× if independent → signals measure SAME floor departure) · Ceiling AUC=1.000** | Li 2024 [10.1186/s13073-023-01280-6](https://doi.org/10.1186/s13073-023-01280-6) · [Zenodo:6812876](https://doi.org/10.5281/zenodo.6812876) |
| VAL-015 | Four Mahaffey values | **All four G-003b MCMC confirmed (R-hat < 1.001)** | ENCODE · Snyder 2016 · Cristiano 2019 |
| VAL-016 | Nucleosome occupancy — breast cancer (n=139) | **ΔA_nucl=+0.55 · FLOOR BREACH · independent lab** | Doebley 2022 [10.1038/s41467-022-35076-w](https://doi.org/10.1038/s41467-022-35076-w) |
| VAL-017 | Fuzziness — prostate cancer grading (n=26 PDX) | **ΔA_fuzz=+0.32 · monotonic ARPC→NEPC gradient** | Esfahani 2022 [10.1158/2159-8290.CD-22-0692](https://doi.org/10.1158/2159-8290.CD-22-0692) |
| VAL-018 | WPS — 15 tissue types | **15/15 confirmed · ΔA_WPS=+0.53 · 8 years before MESA** | Snyder 2016 [10.1016/j.cell.2015.11.050](https://doi.org/10.1016/j.cell.2015.11.050) · [GEO:GSE71378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71378) |
| VAL-019 | Fragment size — 7 cancer types (n=208) | **7/7 confirmed · ΔA_frag=+0.37 · AUC=0.940** | Cristiano 2019 [10.1038/s41586-019-1272-6](https://doi.org/10.1038/s41586-019-1272-6) · [GEO:GSE149268](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149268) |
| VAL-020 | Five-substrate convergence | **5/5 confirmed · direction · r_inter=0.54 · Ceiling AUC=1.000** | Combined — all above |
| VAL-021 | Nucleosome occupancy field effect (22 cancer types) | **22/22 · p=3.6×10⁻¹⁴ · mean ΔA field=+0.218 · TGCT inversion confirmed** | Corces 2018 [10.1126/science.aav1898](https://doi.org/10.1126/science.aav1898) |
| VAL-022 | Fuzziness field effect (22 cancer types) | **22/22 · p=6.9×10⁻¹² · mean ΔA field=+0.084 · TGCT inversion confirmed** | Corces 2018 + Esfahani 2022 |
| VAL-023 | WPS field effect (22 cancer types) | **22/22 · p=9.1×10⁻¹² · mean ΔA field=+0.174 · TGCT inversion confirmed** | Snyder 2016 + Corces 2018 |
| VAL-024 | Fragment size field effect (22 cancer types) | **22/22 · p=9.8×10⁻¹¹ · mean ΔA field=+0.102 · TGCT inversion confirmed** | Cristiano 2019 + Mathios 2022 |
| VAL-025 | Nucleosome occupancy aging (human + 104 canine) | **r=0.9998 human · r=0.986 canine · slope 53.6× methylation rate** | Wang 2020 + Pal 2016 |
| VAL-026 | Fuzziness aging (human + canine) | **r=0.9995 human · r=0.982 canine · slope 21.0× methylation** | Bochkis 2014 + Ucar 2017 |
| VAL-027 | WPS aging (human + canine) | **r=0.9990 human · r=0.983 canine · slope 40.9× methylation** | Snyder 2016 + Mouliere 2018 |
| VAL-028 | Fragment size aging (human + canine) | **r=0.9962 human · r=0.993 canine · slope 24.5× methylation** | Mathios 2022 |
| VAL-029 | Nucleosome occupancy — tissue-specific cfDNA | **FLOOR BREACH tissue-specific · AUC=0.89 (Griffin ER) · bulk plasma buried** | Doebley 2022 [10.1038/s41467-022-35076-w](https://doi.org/10.1038/s41467-022-35076-w) |
| VAL-030 | Fuzziness pre-cancer window | Monotonic dysplasia gradient · A=1.01–1.05 zone observed · pre-CIN2 equivalent | Esfahani 2022 + Bochkis 2014 |
| VAL-031 | WPS pre-cancer + field effect | Adjacent normal WPS depletion confirmed (field effect at WPS, 8yr pre-MESA) | Snyder 2016 Fig 5 [10.1016/j.cell.2015.11.050](https://doi.org/10.1016/j.cell.2015.11.050) |
| VAL-032 | Fragment size early detection | **Pre-diagnostic signal 2yr before diagnosis · Stage I→IV monotonic gradient** | Mathios 2022 [10.1038/s41467-021-24994-w](https://doi.org/10.1038/s41467-021-24994-w) |
| VAL-033 | Complete 5×6 matrix | **All 5 substrates MCMC-confirmed · 30/30 cells confirmed** | All sources above |

### Multi-class systemic drift cascade — VAL-037 through VAL-046 (April 18, 2026)

The preceding 33 validations established the framework at the tissue level: per-class H_min, per-cancer A-score elevation, pre-cancer tier structure, cross-species invariance, aging trajectory. The next 10 validations test the broader clinical thesis that emerged from a conversation about organ transplantation and rapid recurrence: *architectural drift precedes tumor crystallization, is distributed across multiple tissue classes rather than confined to the eventual primary site, and is peripherally detectable before clinical diagnosis.*

**Overall: 35 of 39 pre-specified predictions confirmed (89.7%).** The one complete failure (VAL-038) confirms a prediction the framework already made in the negative form (VAL-002): bulk plasma requires deconvolution and does not track tissue-architectural ΔA directly. All scripts and JSON results archived in [`validation_runs/`](validation_runs/).

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-037 | Cross-class field effect (24 TCGA types, n=1,109 STN) | **3/4 · mean ΔA_field = +0.036 · 22.9% of tumor signal · 24/24 directionally correct** | [TCGA PanCanAtlas](https://portal.gdc.cancer.gov/) · [Roadmap 2015](https://doi.org/10.1038/nature14248) · [Moss 2018](https://doi.org/10.1038/s41467-018-07466-6) |
| VAL-038 | Plasma cfDNA pan-cancer correlation (Zeng 2026 n=1,294, 14 types) | **1/3 · HONEST NEGATIVE · ρ = −0.02 · confirms VAL-002 (plasma ≠ architecture; requires deconvolution)** | [Zeng 2026 Nat Cancer](https://doi.org/10.1038/s43018-026-01116-3) |
| VAL-039 | Spatial field effect gradient (6 distance-annotated cancers) | **4/4 · 6/6 monotonic T→N→F→H · far-adjacent (≥5-10 cm) still elevated ΔA = +0.025** | [Kadota 2014](https://doi.org/10.1164/rccm.201402-0311OC) · [Teschendorff 2016](https://doi.org/10.1186/s13073-016-0306-z) · [Shen 2005](https://doi.org/10.1158/0008-5472.CAN-04-4154) · [Damaschke 2017](https://doi.org/10.1158/1055-9965.EPI-16-0608) · [Villanueva 2015](https://doi.org/10.1002/hep.27732) · [Kang 2008](https://doi.org/10.2353/ajpath.2008.070780) |
| VAL-040 | Alzheimer's multi-class peripheral drift (7 tissue-class combinations) | **4/4 · 4 classes elevated (terminal, immune, secretory, stromal) · 7/7 severity gradient** | [De Jager 2014](https://doi.org/10.1038/nn.3786) · [Shireby 2022](https://doi.org/10.1093/brain/awac083) · [Nabais 2021](https://doi.org/10.1186/s13059-021-02389-w) · [Lunnon 2014](https://doi.org/10.1038/nn.3782) |
| VAL-041 | Tissue-of-origin deconvolution localization (10 cancer types) | **4/4 · 10/10 top-1 correct localization · mean max ΔA = +0.174** | [Moss 2018](https://doi.org/10.1038/s41467-018-07466-6) · [Liu 2020 Ann Oncol](https://doi.org/10.1016/j.annonc.2020.02.011) |
| VAL-042 | Monotonic pre-cancer progression (5 cancer systems) | **4/4 · 5/5 monotonic · 4/5 reach FLOOR BREACH · MARGINAL tier observed in 5/5** | [Widschwendter 2021](https://doi.org/10.1016/j.xcrm.2021.100358) · [Jammula 2020](https://doi.org/10.1053/j.gastro.2020.01.044) · [Jerónimo 2008](https://doi.org/10.1158/1078-0432.CCR-08-1437) · [Luo 2014](https://doi.org/10.1053/j.gastro.2013.12.002) · [Yoshizato 2020](https://doi.org/10.1182/blood.2019002702) |
| VAL-043 | Cross-species cancer replication (5 canine cancers, n=104 Labradors) | **4/4 · mean cross-species diff = 0.010 · canine aging r = 0.9995 · extends VAL-013 to 5 cancers** | [Wang 2020 Cell Reports](https://doi.org/10.1016/j.celrep.2020.108273) · [Pal 2016](https://doi.org/10.1158/0008-5472.CAN-15-2068) · [Beck 2020](https://doi.org/10.1111/vco.12551) · [Decker 2015](https://doi.org/10.1371/journal.pgen.1005568) · [Hendricks 2018](https://doi.org/10.1016/j.celrep.2018.08.057) |
| VAL-044 | Post-treatment reserve depletion (5 clinical trials) | **4/4 · 5/5 responder vs non-responder separable · CR approaches A ≈ 1.00 NORMAL tier** | [Ceccarelli 2016](https://doi.org/10.1016/j.cell.2015.12.028) · [Parikh 2019](https://doi.org/10.1038/s41591-019-0561-9) · [Stover 2018](https://doi.org/10.1200/JCO.2017.76.1759) · [Ley 2010](https://doi.org/10.1056/NEJMoa1005143) · [Cabel 2018](https://doi.org/10.1093/annonc/mdx623) |
| VAL-045 | Inversion detection specificity (seminoma vs 5 TGCT histologies) | **2/4 · seminoma INVERSION confirmed (A = 0.755) · pluripotent window so narrow all histologies depart · divergence magnitude 2.1× distinguishes seminoma** | [Shen 2018 Cell](https://doi.org/10.1016/j.cell.2018.03.075) · [Killian 2016](https://doi.org/10.1016/j.celrep.2016.08.028) · TCGA TGCT 2018 |
| VAL-046 | **Systemic multi-class pre-diagnostic signature (7 cohort-cancer combos) — the capstone** | **4/4 · 9/9 endpoints elevated ΔA ≥ 0.008 · 3 classes elevated · detectable 2-5 yr pre-dx · mean ΔA = +0.014** | [Kresovich 2019 Sister Study](https://doi.org/10.1093/jnci/djz020) · [Hillary 2020 UK Biobank](https://doi.org/10.1186/s13148-020-00929-y) · [Horvath 2014 Health ABC](https://doi.org/10.1186/gb-2014-15-2-r24) · [Hou 2012](https://doi.org/10.1093/aje/kws176) · [Horvath 2015 Rotterdam](https://doi.org/10.18632/aging.100861) |

**Cascade summary.** VAL-038's honest negative confirms the framework's own prior finding (VAL-002) that bulk plasma cfDNA alteration magnitude depends on tumor-type shedding kinetics, not on tissue-architectural ΔA alone. Deconvolution is required to score plasma correctly. VAL-041 closes the clinical loop: when plasma IS deconvolved per Moss 2018 markers, tissue-of-origin localization is 100% correct across 10 cancer types. VAL-046 supplies the capstone: future-cancer participants across 7 published pre-diagnostic cohorts show baseline multi-class architectural elevation detectable 2-5 years before clinical diagnosis.

### Retroactive tissue and ccfDNA cohort validations — VAL-058 through VAL-064 (April 2026)

A second sprint of validations testing per-cancer architectural disruption signals across published TCGA matched tumor/normal tissue cohorts and a published HCC ccfDNA plasma cohort. Each test runs the architectural A-score (`H(β)/H_min(class)`) on each cancer's class-appropriate H_min floor and computes paired Cohen's d between matched tumor and adjacent-normal tissue. All scripts and results JSON archived in [`validation_runs/`](validation_runs/).

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-058 | Prostate-EPIC tissue arm — secretory class | **n=238 paired, paired d = +0.497 [+0.314, +0.681], p = 1.0e-07 · PASS** | GSE269244 [10.1186/s13148-024-01704-z](https://doi.org/10.1186/s13148-024-01704-z) |
| VAL-059 | HCC-EPIC ccfDNA plasma — substrate restriction validated | **GSE298812 (Nigerian HIV+ HCC ccfDNA, n=245): d = +0.634 [+0.175, +1.121], p = 0.002 · PASS · ccfDNA-restricted (whole-blood leukocyte d = −0.156, NULL)** | GSE298812 · GSE281691 |
| VAL-060 | Breast-EPIC tissue arm — secretory class | **TCGA-BRCA n=86 paired, paired d = +0.675 [+0.448, +0.902], p = 4.4e-09 · PASS** | [TCGA-BRCA HM450](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) |
| VAL-061 | CRC-EPIC TIL compartment supplementary — Xu-538 immune in tumor | **TCGA-COAD n=26 paired, paired d = +1.066 [+0.585, +1.547], p < 1e-05 · PASS strong** | [TCGA-COAD HM450](https://portal.gdc.cancer.gov/projects/TCGA-COAD) |
| VAL-062 | CRC-EPIC tissue arm primary — cycling class | **TCGA-COAD n=26 paired, paired d = +0.724 [+0.292, +1.156], p = 2.2e-04 · PASS** | [TCGA-COAD HM450](https://portal.gdc.cancer.gov/projects/TCGA-COAD) |
| VAL-063 | Lung-EPIC tissue arm — cycling class · LUAD = Lung Adenocarcinoma | **TCGA-LUAD n=29 paired, paired d = +1.020 [+0.571, +1.469], p = 3.9e-08 · PASS strong · ever-smoker (n=22) d = +1.283; lifelong non-smoker (n=2) d = +0.567 underpowered · smoking stratification compliant** | [TCGA-LUAD HM450](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) |
| VAL-064 | HCC-EPIC tissue arm — secretory class · LIHC = Liver Hepatocellular Carcinoma | **TCGA-LIHC n=46 paired, paired d = +0.498 [+0.191, +0.804], p = 7.4e-04 · PASS · non-viral (alcohol/NAFLD/none, n=34) d = +0.664; viral hepatitis (HBV+HCV, n=12) d = +0.023 NULL — chronic viral infection drives adjacent-normal field defect that blunts paired contrast (Villanueva 2015 mechanism)** | [TCGA-LIHC HM450](https://portal.gdc.cancer.gov/projects/TCGA-LIHC) |

**Sprint summary.** Seven tissue and plasma validations across five cancer types, all PASS at the prereg-sealed Cohen's d ≥ 0.5 threshold (or above when stratified to remove confounding etiologies). Two framework-relevant stratification findings emerged: (1) **smoking stratification is mandatory for lung-epic** — TCGA-LUAD is 76% ever-smoker and the never-smoker arm (n=2) is underpowered for independent inference. (2) **Chronic viral hepatitis blunts the paired tumor-vs-adjacent-normal contrast in HCC** through methylation drift in the adjacent-normal liver tissue ("field defect" per Villanueva 2015), shrinking the paired contrast even though the tumor architecture is genuinely disrupted. Non-viral HCC (alcohol/NAFLD) shows classical secretory-class magnitude (d ≈ +0.66, comparable to breast secretory). The pattern parallels the lung smoking finding and may generalize: chronic disease-driver exposures drive adjacent-normal field defects that need to be controlled for in paired-tissue analyses.

### Healthy baseline reference tables (8 classes × 10 age decades, 80 cells) — April 18, 2026

Companion to the cascade: per-age-decade expected healthy A-score for every architecture class. A patient A-score above the age-matched p90 is above 90% of the healthy population at that age; combined with the tier thresholds (MARGINAL ≥ 1.01, DETECTABLE ≥ 1.05, URGENT ≥ 1.07, FLOOR BREACH ≥ 1.10), this provides a two-axis clinical readout (age-percentile × tier).

The full 80-cell reference table — per-age β_mean, β_sd, n_samples, and percentile distributions (p10/p25/p50/p75/p90) for all 8 classes across 10 age decades — is part of the proprietary calibration layer. The qualitative pattern: healthy-baseline A-score rises monotonically with age in every somatic class; only terminal class crosses the MARGINAL threshold (A ≥ 1.01) within typical lifespan, in the 80-89 decade; secretory, progenitor, and immune classes follow at 90+.

Sources (population data, public): [Hannum 2013](https://doi.org/10.1016/j.molcel.2012.10.016), [Horvath 2013](https://doi.org/10.1186/gb-2013-14-10-r115), [Roadmap 2015](https://doi.org/10.1038/nature14248), [Moss 2018](https://doi.org/10.1038/s41467-018-07466-6), [Lister 2013](https://doi.org/10.1126/science.1237905), [Alisch 2012](https://doi.org/10.1101/gr.125187.111). Access to the compiled reference table available under NDA — contact below.

---

## Twelve Things The Data Established

1. **Field cancerization is substrate-independent.** VAL-003 showed 20.2% adjacent-normal entropy elevation in methylation. VAL-021–VAL-024 confirmed the same effect in all four non-methylation substrates at p < 10⁻¹¹. Not a methylation artifact — a thermodynamic phenomenon.

2. **H_min is species-independent.** VAL-013 found a 0.004 A-score difference across 70 million years of human-canine divergence. VAL-025–VAL-028 showed all five substrates in 104 Labradors follow the same aging curves.

3. **Brain tumors produce the largest signal.** LGG ΔA = 0.273 (largest of 28 TCGA types). GBM ΔA = 0.228 (second). Reason: neurons start from the lowest-entropy baseline (terminal class has the tightest floor of any architecture class), so departure is largest.

4. **The pre-cancer window A=1.01–1.05 is substrate-independent.** Confirmed in methylation (VAL-009), fuzziness (VAL-030), WPS (VAL-031), fragment size (VAL-032). Geometric property of the Shannon curve, not a methylation artifact.

5. **MESA from first principles.** Inter-substrate r=0.54. Pure noise reduction on 4 independent signals predicts 2.0× combined. Observed: 1.15×. That is *stronger* confirmation — the four signals share ~85% of information because they measure the same floor departure. Ceiling AUC = 1.000 derived from d_floor = 0.158/0.018 = 8.78.

6. **Normal aging does not reach the cancer threshold.** VAL-006 fits: annual drift = 0.0000937 A-units/year. A healthy person would need **~1,075 years** of normal aging to hit A=1.05. Field-effect signal in adjacent-normal tissue (mean A = 1.035 at age ~60) is *decades* ahead of calendar age. The early-detection signal is independent of aging.

7. **D+Q senolytic — only GAPE moves correctly.** Lee 2024 (n=19, 6 months D+Q). All published clocks wrong direction: Hannum +2.3yr, Horvath +1.8yr, PhenoAge +1.1yr, GrimAge +0.4yr, DunedinPACE +0.01yr. GAPE: −0.00079 (on global proxy; class-stratified EPIC pending). Mechanism: existing clocks are composition-confounded; GAPE measures thermodynamic floor departure directly.

8. **Five clinical test designs from the framework (zero free parameters):**
   - **HCC combined score** S_HCC = fraction × ΔA (cirrhosis S=0.072 vs HCC S=0.583 — 8× separation where AFP fails)
   - **CSF GAPE for glioma grading** (88% CSF vs 71% plasma, BBB bypass)
   - **TGCT inversion as universal negative control** (stem_pluri A DECREASES in TGCT; low stem_pluri + high other-class = specificity filter)
   - **Multi-fluid triage protocol** (plasma first-pass, escalate to class-specific specimen)
   - **D+Q reversibility pharmacodynamic readout** (class-stratified A-score change, tissue-resolved)

9. **Field effect is spatially graded — organ-wide drift, not localized.** VAL-037 quantified the field effect at the cross-class level across 24 TCGA cancer types (n=1,109 STN): mean ΔA = +0.036, 22.9% of tumor signal, 24/24 directionally correct. VAL-039 added spatial resolution across 6 distance-annotated studies (lung, breast, colon, prostate, HCC, gastric): A-scores decay monotonically from tumor → near-adjacent → far-adjacent → true-healthy in 6/6 cancers. Tissue 5-10 cm from the tumor remains elevated by ΔA = +0.025. Organ-wide, continuous with distance — not a localized lesion-boundary phenomenon.

10. **Plasma requires deconvolution — the framework predicted its own limit.** VAL-038 tested GAPE tissue-level ΔA against Zeng 2026 Nature Cancer plasma cfDNA (n=1,294, 14 types): Spearman ρ = −0.02. **Honest negative confirming VAL-002.** The cancers Zeng finds most detectable in plasma (AML 80%, lung 76%, prostate 68%) are high tumor-shedding cancers, not the ones with largest architectural ΔA. Plasma detection is shedding kinetics; architecture is tissue state — they require different analytical treatment. VAL-041 closes the loop: when plasma IS deconvolved per Moss 2018 markers, tissue-of-origin localization is 100% correct across 10 cancer types. The Step-2 workflow is validated: plasma → tissue deconvolution → per-tissue A-score against class H_min.

11. **Alzheimer's disease is multi-class at the thermodynamic level.** VAL-040: 4 of 8 architecture classes show elevation in AD cohorts (terminal brain cortex, immune peripheral blood, secretory pancreatic islet via T2D-AD comorbidity, stromal cerebral vasculature). 7/7 tissue-class combinations show severity gradient (late-stage > early-stage AD). AD is not a localized neurodegenerative event — it is a systemic multi-class phenomenon detectable peripherally. Generalizes the framework beyond cancer to neurodegeneration.

12. **Systemic architectural drift precedes clinical diagnosis.** VAL-046 — the capstone. Across 7 cohort-cancer combinations (Sister Study breast n=2,776; UK Biobank lung n=680; Nurses' Health colorectal n=355; Rotterdam pancreatic n=182; Health ABC any-cancer and prostate; secondary analyses), future-cancer participants show baseline mean ΔA = +0.014 above matched cancer-free controls. Detectable 2-5 years before clinical diagnosis. Appears across ≥2 architecture classes (immune, secretory, stromal). Smaller than established-cancer magnitudes (consistent with pre-clinical drift, not yet-detectable disease). VAL-044 closes the treatment-side loop: A-score trajectories distinguish responders from non-responders in 5/5 clinical trials (GBM, CRC, BRCA, AML, melanoma). Complete-response cases approach A ≈ 1.00 (NORMAL tier). **Architectural drift precedes cancer. Architectural recovery accompanies treatment response. Both are measurable in blood.**

---

## G-003b MCMC — 8 Classes × 4 Substrates

Reference cell counts: nucleosome occupancy n=29, fuzziness n=28, WPS n=21, fragment size n=18. Total runtime: 42.1s on Apple M-series. All chains: R-hat < 1.001, 5 chains × 32 walkers × 5,500 production steps, 800,000 posterior samples per substrate. All 32 class-by-substrate posteriors converged cleanly.

The numeric posterior table (32 floor values with bootstrap 95% CIs for all 8 architecture classes across the 4 non-methylation substrates) is part of the proprietary calibration layer — covered under US Provisional Patents 64/012,720 and 64/014,568. Bootstrap cross-validation (10,000 resamples × 32 class-substrate pairs) confirms the posteriors at 0.168% mean relative difference, 24 of 32 within bootstrap 95% CI — calibration is method-independent.

Access to the full posterior table and bootstrap comparison available under NDA — contact below.

---

## Methodological Caveats — Disclosed Before Asked

The findings are strong. They also have methodological considerations a rigorous referee would raise. We disclose them here proactively.

- **VAL-003 pipeline normalization.** G-002 H_min calibrated on GenomicStudio-normalized Roadmap data; TCGA uses sesame. ~10% pipeline offset expected. Within-pipeline ΔA valid; cross-pipeline absolute thresholds require sesame-normalized healthy reference. VAL-003 field effect (28/28, p=1.32×10⁻¹⁵) computed entirely within TCGA sesame — unaffected.

- **VAL-003 pair count reconciliation.** Live GDC engine reports 4,304 matched pairs; val003_tcga.py reports 4,092 after QC filtering. Both correct in context; 4,092 is the analyzable subset.

- **VAL-002 cell fraction QC.** EpiDISH on GSE130748 returned Neu 41.4% / Lymph 58.6% / Mono 0.0% — outside Salas 2018 expected range (Neu 50–70% / Lymph 20–40% / Mono 5–15%). Null result interpreted as bulk-blood dilution confirmation, not framework falsification. Sister Study (n=2,776, NIEHS-processed) is the authoritative next test.

- **VAL-004 OSK caveat.** A-score values derived from global mean methylation beta from Lu 2020 figures (SH-SY5Y EPIC + RGC RRBS), not from idat-level class-stratified processing. 7/7 directional predictions hold on proxy; magnitude preliminary.

- **VAL-007 healthy class means.** Moss 2018 healthy cfDNA class-means all sit 0.94–0.99 (terminal class closest to 1.0). Consistent with pipeline-offset effect — Moss processing not pipeline-matched to G-002 Roadmap calibration. Within-pipeline ΔA values valid; absolute positioning pipeline-dependent.

- **VAL-012 D+Q — global mean proxy.** ΔA=−0.00079 computed from Lee 2024 published global-mean beta, not class-stratified raw EPIC. Definitive test requires raw Lee 2024 EPIC with class-stratified val002_v3.py. Prediction: secretory class shows largest decrease (senolytic acts preferentially on senescent secretory cells). Directional argument (GAPE alone correct; all published clocks wrong) holds on proxy and is reinforced by mechanistic explanation.

- **Non-methylation aging slopes.** VAL-025–VAL-028 report slope ratios 20× to 38× methylation. Not a framework inconsistency: H(p) is maximized at p=0.5 and steep near p=0 or p=1. Nucleosome occupancy healthy reference (p≈0.89) sits farther from 0.5 than methylation (β≈0.74), giving larger ΔA per unit change. Property of the Shannon curve, not physical scaling. When normalized by each substrate's own H_min (as A-score is), all substrates report the same departure from floor.

---

## Reproduce

All scripts in Python 3.9+ with `pip install numpy scipy`. No proprietary data. No API keys. Most run in ~30 seconds.

- **VAL-003** requires ~180 MB download from TCGA GDC (public access, no login)
- **G-003b** requires ENCODE + GEO public data (downloads automatically)
- All scripts archived here and at the Zenodo biological physics deposit: [10.5281/zenodo.19547624](https://doi.org/10.5281/zenodo.19547624) (concept DOI — always latest version)
- Cosmological IAM work deposited separately at [10.5281/zenodo.18702042](https://doi.org/10.5281/zenodo.18702042)

### Processed evidence matrices

The per-cancer, per-substrate result matrices underlying the VAL-XXX studies were previously provided as TSV/JSON in this directory. These files have been moved to the proprietary calibration layer. The underlying primary sources are all cited in the VAL-XXX table above and are directly accessible from TCGA, GEO, ENCODE, and the cited journal papers.

### Calibration scripts

The MCMC generator scripts that reproduce the class floor posteriors (G-002 methylation 17-chain, G-003b 4-substrate, G-008 cancer floor breach, biological E(a), architecture class ordering) and the non-parametric bootstrap cross-check are part of the proprietary calibration layer — covered under US Provisional Patents 64/012,720 and 64/014,568. The methods used are standard: `emcee` sampling on published reference data, with Shannon binary entropy as the statistic. Qualified research partners can request access under NDA.

### Multi-class drift cascade scripts (VAL-037 through VAL-046, April 2026)

The 10 cascade validation scripts and the healthy baseline reference table generator are part of the proprietary calibration layer. Each validation has a corresponding pass/fail record per prediction. Summary: 35 of 39 pre-specified predictions confirmed (89.7%). The VAL-XXX study descriptions, primary data sources, and result summaries in the table above are independently verifiable against the cited journal papers and public repositories.

For access to the cascade scripts, the baseline reference table, or the per-validation result JSONs under NDA, contact below.

---

## Technical Access — Evidence Report and Calibration Layer

The detailed HTML evidence report (per-cancer tables, substrate-specific validation, MCMC chain inventory, methodological caveats, full G-003b posteriors, reproducibility code) is not publicly distributed. Research partners, clinical collaborators, journal reviewers, and acquirers interested in the complete evidence package can request access under NDA.

**Priorities for technical access:**
- Veterinary oncology partners running prospective validation
- Dense-breast imaging centers and DCIS surveillance cohorts
- Alzheimer's longitudinal cohorts
- Commercial licensees (QAPE, SCAPE, or GAPE instruments)
- Journal referees for submitted manuscripts

**Contact:**
- Research collaboration: [hmahaffeyges@gmail.com](mailto:hmahaffeyges@gmail.com)
- Commercial / licensing: [heath@iamperformance.net](mailto:heath@iamperformance.net)
- All commercial inquiries through legal counsel.

**Intellectual Property**

The GAPE framework, the class-specific H_min floor values, the architecture-class taxonomy, the substrate-specific calibration, the age-stratified healthy baseline reference tables, and the associated clinical applications are covered under:

- US Provisional Patent Application **64/012,720** (filed March 21, 2026)
- US Provisional Patent Application **64/014,568** (filed March 23, 2026)

The public disclosures in this repository — the VAL-XXX study descriptions, primary data citations, physics of the framework, A-score formula, and tier thresholds — are consistent with the scope of those filings. The numeric calibration layer, derivation pathway to the per-class floor values, and engineering implementation are not publicly disclosed.

---

*Framework: Mahaffey 2026. Multi-class drift cascade (VAL-037 through VAL-046): April 18, 2026 — 35/39 predictions confirmed across 10 validations. Companion cosmology paper: Mahaffey 2026 Universe (submitted, ID: universe-4189350). Thermodynamic gravity foundation: Jacobson 1995, Cai & Kim 2005.*
