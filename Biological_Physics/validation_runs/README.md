# GAPE Validation Runs — Multi-Class Drift Cascade

This directory contains VAL-037 through VAL-046 validation scripts, JSON results,
and the 80-cell healthy baseline reference.

## Cascade Summary (April 18, 2026)

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

## Running the scripts

```bash
pip install numpy scipy
cd validation_runs
python3 VAL_037_field_effect_cross_class.py
# ... each runs in ~30 seconds
```

All scripts use published primary-source β values. No proprietary data.
Each implements pre-specified predictions with explicit pass/fail outputs.

## Files

- `VAL_037` through `VAL_046` — cascade validation scripts + JSON results
- `VAL_047_*` — external validation on real per-patient 450K β values (GSE51057, GSE51032, GSE69914) — first individual-sample-level validation
- `HEALTHY_BASELINES.py` — 80-cell healthy reference (8 classes × 10 age decades)
- `HEALTHY_BASELINES.json` — clinical-ready JSON for demo integration
- `CASCADE_SUMMARY.py` — aggregates all cascade results
- `CASCADE_SUMMARY.json` — summary JSON

## VAL-047 script inventory

- `VAL_047_real_analysis.py` — first-pass: bulk immune mean β on GSE51057 (null result, Cohen's d=0.08, confirms Xu 2019 KS p=0.5 finding)
- `VAL_047_extended_v2.py` — architectural variance, multi-class signature, directional drift, top-N CV analyses
- `VAL_047_options_1_2.py` — headline script: Xu-2019 6-CpG directional score + time-to-dx stratification. CV Cohen's d = +0.605 ± 0.190 on breast pre-dx.
- `VAL_047_replication.py` — GSE51032 replication (n=845): breast CV d = +0.379, colorectal CV d = +0.835
- `VAL_047_option3.py` — GSE69914 tissue-level validation: monotonic healthy → adjacent → tumor, AUC 0.70

## Primary sources (by validation)

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

Full citation list is in the GAPE_Evidence_Report.html references [22]-[58].
