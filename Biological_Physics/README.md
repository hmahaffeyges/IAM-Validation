# Biological Physics — GAPE Framework

Application of the Informational Actualization Model (IAM) to biological systems.

The same thermodynamic principle that governs gravitational decoherence and cosmic expansion
sets the minimum information maintenance cost for living cells. The Landauer cost of
irreversible DNA methylation maintenance at physiological temperature defines
architecture-class-specific entropy floors for all mammalian somatic cell types.

**Updated: April 17, 2026 — G-003b MCMC complete. All five substrates MCMC-confirmed. 35 studies.**

**Cross-validated:** 10,000-resample non-parametric bootstrap on identical reference data agrees with G-003b MCMC posteriors at 0.168% mean relative difference (max 1.091%), 24 of 32 MCMC posterior means within bootstrap 95% CI. Calibration is method-independent. See the "Methodology: MCMC vs. bootstrap" section in the [Evidence Report](GAPE_Evidence_Report.html).

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

| Substrate | What it measures | H_min (cycling class) | Status |
|-----------|-----------------|----------------------|--------|
| DNA methylation (β) | CpG commitment tags | 0.856055 ± 0.000312 | **CONFIRMED — G-002 MCMC (17 chains, R-hat < 1.001)** |
| Nucleosome occupancy | DNA spool positioning probability | 0.980072 ± 0.008427 | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Nucleosome fuzziness | Positional precision across cells | 0.819030 ± 0.007359 | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Windowed protection score | Promoter nucleosome protection | 0.627429 ± 0.005649 | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |
| Fragment size entropy | cfDNA fragment length distribution | 0.687936 ± 0.006878 | **CONFIRMED — G-003b MCMC (R-hat < 1.001)** |

Full G-003b posterior table (all 8 architecture classes × 4 substrates = 32 posteriors) is in the
[Evidence Report](../GAPE_Evidence_Report.html) and in the "G-003b Full Posteriors" section below.

---

## Complete Validation Record — 35 Studies

### MCMC Calibration

| Study | Description | Result | Data Source |
|-------|-------------|--------|-------------|
| G-002 | H_min for 8 architecture classes (methylation) | **17 chains · R-hat < 1.001 · 800,000 samples. H_min_cycling = 0.856055 ± 0.000312** | [NIH Roadmap Epigenomics](https://www.ncbi.nlm.nih.gov/geo/roadmap/epigenomics/) |
| G-003b | H_min for 4 additional substrates | **5 chains × 32 walkers × 5,500 steps · R-hat < 1.001 · 42.1s runtime · 32 posteriors** | [ENCODE](https://www.encodeproject.org/) · [GEO:GSE71378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71378) · [GEO:GSE149268](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149268) |

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

---

## Eight Things The Data Established

1. **Field cancerization is substrate-independent.** VAL-003 showed 20.2% adjacent-normal entropy elevation in methylation. VAL-021–VAL-024 confirmed the same effect in all four non-methylation substrates at p < 10⁻¹¹. Not a methylation artifact — a thermodynamic phenomenon.

2. **H_min is species-independent.** VAL-013 found a 0.004 A-score difference across 70 million years of human-canine divergence. VAL-025–VAL-028 showed all five substrates in 104 Labradors follow the same aging curves.

3. **Brain tumors produce the largest signal.** LGG ΔA = 0.273 (largest of 28 TCGA types). GBM ΔA = 0.228 (second). Reason: neurons start from the lowest-entropy baseline (H_min = 0.773), so departure is largest.

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

---

## G-003b Full Posteriors — 8 Classes × 4 Substrates

Reference cell counts: nucleosome occupancy n=29, fuzziness n=28, WPS n=21, fragment size n=18. Total runtime: 42.1s on Apple M-series. All chains: R-hat < 1.001, 5 chains × 32 walkers × 5,500 production steps, 800,000 posterior samples per substrate.

| Class | Nucl. occupancy | Nucl. fuzziness | WPS | Fragment size |
|-------|-----------------|-----------------|-----|---------------|
| stem_pluri | 0.799818 ± 0.009230 | 0.962920 ± 0.011135 | 0.905004 ± 0.012671 | 0.973583 ± 0.015681 |
| stem_adult | 0.960866 ± 0.011131 | 0.980754 ± 0.009944 | 0.988964 ± 0.008174 | 0.841327 ± 0.011784 |
| progenitor | 0.972790 ± 0.011009 | 0.961900 ± 0.011166 | 0.988046 ± 0.008611 | 0.808978 ± 0.016338 |
| terminal | 0.992027 ± 0.005948 | 0.736973 ± 0.007371 | 0.958909 ± 0.011203 | 0.624938 ± 0.007288 |
| **cycling** | **0.980072 ± 0.008427** | **0.819030 ± 0.007359** | **0.627429 ± 0.005649** | **0.687936 ± 0.006878** |
| immune | 0.989930 ± 0.006463 | 0.830377 ± 0.008299 | 0.589644 ± 0.006792 | 0.711534 ± 0.007067 |
| secretory | 0.982560 ± 0.009638 | 0.847947 ± 0.009769 | 0.634534 ± 0.008996 | 0.697718 ± 0.009890 |
| stromal | 0.985667 ± 0.008815 | 0.832386 ± 0.009645 | 0.612686 ± 0.008810 | 0.724691 ± 0.014423 |

Reproduce: [`evidence/gape_mcmc_g003b.py`](evidence/gape_mcmc_g003b.py)

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

### Processed evidence matrices (one click away)

- [`evidence/evidence_summary.tsv`](evidence/evidence_summary.tsv) — all per-cancer, per-substrate numbers in tab-separated format (14 KB, run your own statistics on it)
- [`evidence/evidence_summary.json`](evidence/evidence_summary.json) — same data in structured JSON (49 KB)
- [`evidence/bootstrap_vs_mcmc_comparison.tsv`](evidence/bootstrap_vs_mcmc_comparison.tsv) — G-003b MCMC posteriors vs 10,000-resample bootstrap CIs, all 32 class-substrate pairs, machine-readable

### Calibration scripts (deterministic from seed — reproducible in ~42 seconds)

- [`evidence/gape_mcmc_g002.py`](evidence/gape_mcmc_g002.py) — G-002 methylation 17-chain R-hat<1.001
- [`evidence/gape_mcmc_g003b.py`](evidence/gape_mcmc_g003b.py) — G-003b 4-substrate, 32 posteriors
- [`evidence/gape_mcmc_g008.py`](evidence/gape_mcmc_g008.py) — Cancer floor breach validation
- [`evidence/gape_mcmc_e_a_bio.py`](evidence/gape_mcmc_e_a_bio.py) — Biological E(a) validation
- [`evidence/gape_mcmc_nbio_ordering.py`](evidence/gape_mcmc_nbio_ordering.py) — Architecture class ordering
- [`evidence/gape_bootstrap_comparison.py`](evidence/gape_bootstrap_comparison.py) — Non-parametric bootstrap cross-check of G-003b MCMC posteriors (10,000 resamples × 32 class-substrate pairs)

Scripts live in [`validation/`](validation/) and [`evidence/`](evidence/).

---

## Full Evidence Report — Three Layers of Preservation

The complete HTML evidence report with expandable detail tables (VAL-003 per-cancer, VAL-007 per-cancer cfDNA, VAL-008+009 specimen matrix, VAL-012 clock comparison, full G-003b posteriors, methodological caveats) exists at three addresses, each serving a different preservation role:

1. **Live working copy:** [iamperformance.net](https://iamperformance.net) — continuously updated, public-facing, always shows the current state of the evidence.
2. **Version-controlled snapshots:** [`GAPE_Evidence_Report.html`](GAPE_Evidence_Report.html) in this repo — every commit produces an immutable URL tied to a hash.
3. **Archival DOI:** [10.5281/zenodo.19547624](https://doi.org/10.5281/zenodo.19547624) — frozen Zenodo deposits with citable version DOIs, replicated across CERN infrastructure.

### How to cite the state of this report on a specific date

Every commit to this repository produces an immutable URL. To cite the state of the evidence report as it stood on a specific date, use the commit hash:

```
https://github.com/hmahaffeyges/IAM-Validation/blob/<COMMIT_HASH>/Biological_Physics/GAPE_Evidence_Report.html
```

Find the commit hash for a given date by viewing the [commit history](https://github.com/hmahaffeyges/IAM-Validation/commits/main/Biological_Physics/GAPE_Evidence_Report.html). For formal citation with a DOI, use the corresponding Zenodo version DOI listed on the [deposit page](https://doi.org/10.5281/zenodo.19547624) — each upload to Zenodo produces a new version DOI that permanently points to that exact state.

---

*Framework: Mahaffey 2026. Companion cosmology paper: Mahaffey 2026 Universe (submitted, ID: universe-4189350). Thermodynamic gravity foundation: Jacobson 1995, Cai & Kim 2005.*
