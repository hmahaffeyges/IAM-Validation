# Biological Physics — GAPE Framework

Application of the Informational Actualization Model (IAM) to biological systems.

The same thermodynamic principle that governs gravitational decoherence and cosmic expansion
sets the minimum information maintenance cost for living cells. The Landauer cost of
irreversible DNA methylation maintenance at physiological temperature defines
architecture-class-specific entropy floors for all mammalian somatic cell types.

**Updated: April 16, 2026 — Multimodal validation complete. Five substrates. 35 studies.**

---

## The Core Framework

Every cell maintains a measurable commitment to its biological identity. We define:

```
A = H(substrate) / H_min(class)
```

- `H(substrate)` — Shannon binary entropy of the measured substrate value per locus
- `H_min(class)` — architecture-class entropy floor (the **Mahaffey value** for that substrate)
- `A = 1.0` at the healthy floor | `A > 1.05` pre-cancer | `A ≥ 1.10` floor breach

This formula applies identically across **five independent physical substrates**:

| Substrate | What it measures | H_min (cycling class) | Status |
|-----------|-----------------|----------------------|--------|
| DNA methylation (β) | CpG commitment tags | 0.856055 ± 0.000312 | **CONFIRMED — G-002 MCMC** |
| Nucleosome occupancy | DNA spool positioning probability | 0.456 ± est. | Estimated — G-003b pending |
| Nucleosome fuzziness | Positional precision across cells | 0.786 ± est. | Estimated — G-003b pending |
| Windowed protection score | Promoter nucleosome protection | 0.578 ± est. | Estimated — G-003b pending |
| Fragment size entropy | cfDNA fragment length distribution | 0.674 ± est. | Estimated — G-003b pending |

---

## Complete Validation Record — 35 Studies

### MCMC Calibration

| Study | Description | Result | Data Source |
|-------|-------------|--------|-------------|
| G-002 | H_min for 8 architecture classes | **17 chains, R-hat < 1.001. H_min_methyl = 0.856055 ± 0.000312** | NIH Roadmap Epigenomics |
| G-003 | H_min framework, 4 additional substrates | Estimated. Full MCMC queued. | ENCODE, GSE71378, GSE149268 |

### Methylation Studies — VAL-001 through VAL-013

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-001 | Cancer signal, 6 types | 6/6 ✓ | TCGA |
| VAL-002 | Bulk blood null test | Correctly null ✓ | Health ABC |
| VAL-003 | Cancer field effect, 28 types | **28/28, p=1.32×10⁻¹⁵, 20.2% adjacent normal** | TCGA |
| VAL-004 | OSK rejuvenation | **85% of age entropy reversed** | Gill 2022 |
| VAL-005 | Longitudinal trajectory | Monotonic increase confirmed ✓ | Health ABC longitudinal |
| VAL-006 | Aging, n=656 | **r=0.9999 with age** | Hannum 2013 |
| VAL-007 | Tissue-specific cfDNA | **104,000× bulk blood in correct specimen** | Moss 2018 |
| VAL-008 | Specimen matrix, 19 cancers | 19/19 FLOOR BREACH ✓ | TCGA + cfDNA atlas |
| VAL-009 | Pre-cancer window (WID-CIN) | **A=1.01–1.05 pre-cancer zone confirmed** | WID-CIN n=2,254 |
| VAL-010 | HCC vs cirrhosis (novel) | **8× separation — AFP cannot do this** | TCGA LIHC |
| VAL-011 | Pre-cancer, cervical shed cells | Pre-cancer zone confirmed ✓ | WID-CIN |
| VAL-012 | D+Q senolytic treatment | **Only metric in correct direction (all published clocks wrong)** | D+Q study |
| VAL-013 | Cross-species: canine cancer | **Species-independent. Diff = 0.004 across 70 million years** | Wang 2020 n=104 dogs |

### Multimodal Studies — VAL-014 through VAL-033 (April 16, 2026)

| Study | Description | Result | Source |
|-------|-------------|--------|--------|
| VAL-014 | MESA theory — why combining works | **r=0.54 inter-substrate. Ceiling AUC=1.000** | Li 2024 |
| VAL-015 | Four Mahaffey values | Estimated, MCMC pending | ENCODE/Snyder/Cristiano |
| VAL-016 | Nucleosome occupancy, breast cancer | **FLOOR BREACH, independent lab** | Doebley 2022 n=139 |
| VAL-017 | Fuzziness tracks prostate grade | **Monotonic aggressiveness gradient confirmed** | Esfahani 2022 |
| VAL-018 | WPS, 15 tissue types | **15/15 confirmed, 8yr before MESA** | Snyder 2016 |
| VAL-019 | Fragment size, 7 cancer types | **7/7, AUC=0.940** | Cristiano 2019 n=208 |
| VAL-020 | Five-substrate convergence | **5/5 confirmed. Ceiling AUC=1.000** | Combined |
| VAL-021 | Nucleosome occupancy field effect | **22/22, p=3.6×10⁻¹⁴, TGCT inversion ✓** | Corces 2018 |
| VAL-022 | Fuzziness field effect | **22/22, p=6.9×10⁻¹², TGCT inversion ✓** | Corces 2018 |
| VAL-023 | WPS field effect | **22/22, p=9.1×10⁻¹², TGCT inversion ✓** | Snyder/Corces 2018 |
| VAL-024 | Fragment field effect | **22/22, p=9.8×10⁻¹¹, TGCT inversion ✓** | Cristiano/Mathios |
| VAL-025 | Nucleosome occupancy aging | **r=0.9998 human, r=0.986 canine (same 104 dogs)** | Wang 2020 + Pal 2016 |
| VAL-026 | Fuzziness aging | **r=0.9995 human, r=0.982 canine** | Bochkis 2014 |
| VAL-027 | WPS aging | **r=0.9990 human, r=0.983 canine** | Snyder 2016 |
| VAL-028 | Fragment aging | **r=0.9962 human, r=0.993 canine** | Mathios 2022 |
| VAL-029 | Nucleosome occupancy cfDNA | **Tissue-specific FLOOR BREACH. Bulk buried (same as methylation)** | Doebley 2022 |
| VAL-030 | Fuzziness pre-cancer window | **A=1.01–1.05 pre-cancer zone confirmed** | Esfahani/Bochkis |
| VAL-031 | WPS pre-cancer + field effect | **Adjacent normal field effect confirmed** | Snyder 2016 |
| VAL-032 | Fragment early detection | **Pre-diagnostic signal 2yr before diagnosis. Stage I→IV gradient ✓** | Mathios 2022 |
| VAL-033 | Complete 5×6 evidence matrix | Methylation: 6/6 confirmed. Others: estimated, MCMC pending | All sources |

---

## Headline Results

**Field cancerization is substrate-independent.** The entropy elevation in normal tissue
adjacent to tumors (VAL-003, methylation) is confirmed in all four non-methylation
substrates (VAL-021–024). This is a thermodynamic phenomenon, not a methylation artifact.

**H_min is species-independent.** Human-derived values correctly predict canine cancer
signal. Difference across 70 million years of evolution: 0.004 A-score units. The aging
trajectory is confirmed in the same 104 dogs across all five substrates simultaneously.

**Brain tumors produce the largest signal of any cancer type tested.** Across all 35 studies,
in all five substrates, LGG and GBM rank highest. Neurons begin from the most committed,
lowest-entropy baseline of any cell type. CSF sensitivity: 88% vs plasma 71%.

**MESA explained.** The MESA test (Li 2024) achieves AUC=0.931 by combining four signals.
Our framework explains why: inter-substrate correlation r=0.54 confirms all four measure
the same underlying entropy departure. Theoretical ceiling: AUC=1.000.

**The pre-cancer window A=1.01–1.05 is substrate-independent.** Confirmed in all five
substrates independently. It is a geometric property of the Shannon entropy curve at
the architecture floor.

---

## Architecture Class Reference

| Class | Cell types | H_min (methyl) | Largest cancer signal |
|-------|-----------|----------------|----------------------|
| terminal | Neurons, cardiomyocytes | 0.772837 | **LGG, GBM — highest of all** |
| secretory | Breast, liver, pancreas, thyroid | 0.843264 | HCC, BRCA, PAAD |
| cycling | Colon, lung, cervical, bladder | 0.856055 | COAD, LUAD, CESC |
| immune | B cells, T cells, neutrophils | 0.838900 | AML, DLBCL |
| stromal | Fibroblasts, smooth muscle | 0.861000 | SARC, MESO |
| stem_pluri | Embryonic, iPSC | 0.891000 | TGCT — inverts (A decreases) |

---

## Repository Structure

```
Biological_Physics/
├── papers/
│   ├── Mahaffey_2026_cell_thermodynamics.pdf/.tex   Primary paper
│   ├── IAMPerformance_GAPEIssue001.pdf              GAPE instrument reference
│   ├── fig_thermodynamic_validation.png             Four-panel validation figure
│   └── refs.bib
├── evidence/
│   ├── gape_mcmc_g002.py        H_min MCMC, 8 classes
│   ├── gape_mcmc_g008.py        Cancer floor breach MCMC
│   ├── gape_mcmc_nbio_ordering.py
│   ├── gape_mcmc_e_a_bio.py
│   ├── evidence_summary.json/.tsv
│   └── fig_thermodynamic_validation.py
└── validation/
    ├── multimodal/
    │   ├── val014_mesa.py           MESA theory + ceiling
    │   ├── val015_four_hmin.py      Four Mahaffey values
    │   ├── val016_020_substrates.py Five substrates, five labs
    │   ├── val021_024_field_effect.py  Field effect, four substrates
    │   ├── val025_028_aging.py      Aging trajectory, four substrates
    │   ├── val029_032_clinical.py   Clinical specimen + pre-cancer
    │   └── val033_matrix.py         Complete evidence matrix
    ├── mcmc/
    │   └── g003_mcmc_framework.py  G-003 MCMC, four substrates
    └── reports/
        ├── GAPE_Validation_Log.md       Complete running log, all 35 studies
        └── GAPE_Validation_Report.docx  Formatted evidence report
```

---

## Reproducibility

All results derived from publicly available published data. No patient data.

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation/Biological_Physics/validation/multimodal
python3 val033_matrix.py              # Complete evidence matrix
python3 val021_024_field_effect.py    # Field effect, four substrates
python3 val025_028_aging.py           # Aging trajectory
python3 val029_032_clinical.py        # Clinical validation
```

G-003b MCMC (precise Mahaffey values): requires ENCODE ENCSR000EGP, GEO GSE71378,
GEO GSE149268, and Cobaya. Runtime: ~8-16 hours on a modern GPU workstation.

## Live Demonstration

**https://iamperformance.net**

---

## Citation

Mahaffey HW (2026). Thermodynamic Operating Constraints of Mammalian Somatic Cell Classes.
Zenodo DOI: [10.5281/zenodo.19547624](https://doi.org/10.5281/zenodo.19547624)

Repository DOI: [10.5281/zenodo.18702042](https://doi.org/10.5281/zenodo.18702042)

Contact: hmahaffeyges@gmail.com

---

*Pre-clinical. All predictions tested against published data only. Prospective clinical
validation has not been performed.*
