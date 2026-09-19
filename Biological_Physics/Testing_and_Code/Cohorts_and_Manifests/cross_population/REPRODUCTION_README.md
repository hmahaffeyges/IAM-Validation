# EDEAR / GAPE Cross-Population Validation — Reproduction Bundle

**Session date:** 2026-04-22
**Bundle owner:** Heath W. Mahaffey (hmahaffeyges@gmail.com)
**Repository:** https://github.com/hmahaffeyges/IAM-Validation

This bundle contains everything needed to reproduce the cross-population results
reported in the EDEAR Breast/Colorectal Evidence Report v1.1 and the GAPE
Evidence Report VAL-049 section.

---

## Contents

```
evidence_folder/
├── CROSS_POPULATION_MANIFEST.json          # v1.1 consolidated manifest
├── CROSS_POPULATION_MANIFEST_v1.0.json.bak # prior manifest (archived)
├── REPRODUCTION_README.md                  # this file
├── scripts/
│   ├── xu538_breast_panel.json             # the 538-CpG panel (SHA ada6729605...)
│   ├── T1_GSE40279_baseline.py             # T1: healthy aging reference
│   ├── T2_GSE104942_aussie_breast.py       # T2: Australian HBOC
│   ├── T3_GSE148663_uruguay_breast.py      # T3: Uruguayan post-dx
│   ├── T5_GSE89093_twins_cancer.py         # T5: TwinsUK MZ paired
│   ├── T8_NHANES_cox.py                    # T8: NHANES Cox regression
│   ├── T9_GSE283951_polish_prediagnostic.py # T9: Polish pre-dx
│   ├── T10_GSE37965_heyn_twins.py          # T10: Heyn UK EpiTwin
│   ├── T11_GSE243529_chinese_atdx.py       # T11: Singapore Chinese
│   ├── T12_GSE314261_stjude_grep.py        # T12a: St Jude (attempted, not interp.)
│   ├── T12_GSE314261_stjude_stream.py      # T12b: St Jude partial stream
│   ├── T13_GSE51057_secretory_class.py     # T13: secretory on GSE51057
│   ├── T14_GSE51032_secretory_class.py     # T14: secretory on GSE51032
│   ├── T15_NHANES_blinded_prospective.py   # T15: NHANES blinded-cohort flag
│   ├── VAL047_tightening_fresh.py          # Tightening v2 (headline pipeline)
│   └── VAL047_tightening_v2_patch.py       # Tightening v2 patch helper
└── results/
    ├── T1_GSE40279/                        # one JSON + per-sample CSVs per test
    ├── T2_GSE104942/
    ├── T3_GSE148663/
    ├── T5_GSE89093/
    ├── T8_NHANES/
    ├── T9_GSE283951/
    ├── T10_GSE37965/
    ├── T11_GSE243529/
    ├── T12_GSE314261/
    ├── T13_secretory_GSE51057/
    ├── T14_secretory_GSE51032/
    └── T15_NHANES/                         # T15 results + run log
```

Total bundle size: scripts ~300 KB, results ~13 MB.

---

## Environment

```
Python >= 3.9
numpy (any modern version, tested on 2.4.3)
pandas (any modern version, tested on 3.0.1)
scipy (optional; used for chi-squared and log-rank tests; scripts fall back
       to approximations if scipy is not installed)
```

No GPU, no cluster, no API keys, no institutional credentials. Each test
runs in 30 seconds to 2 minutes on a laptop, except T12 (St Jude) which
requires a multi-hour compute window and is documented as
attempted-not-interpretable.

---

## Calibration constants (frozen 2026-04-06)

These constants are required to reproduce the A-scores. They go with the
panels, and the reproduction uses them as inputs without needing the
derivation:

```
Xu-538 panel           -> constant 0.838889
19-CpG secretory panel -> constant 0.843264
```

No value in this bundle has been tuned on any reported cohort. Any
re-run with different constants will produce different A-scores; only
the constants above reproduce the reported numbers bit-exactly.

---

## Input data downloads (not included in the bundle — too large)

Each script expects a GEO series matrix or NHANES file as its input.
Download each from the URL in the manifest. SHAs in the manifest verify
the file you downloaded is the one we analyzed.

GEO cohort downloads (all free, no login):
- GSE40279  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
- GSE51057  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51057
- GSE51032  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51032
- GSE104942 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104942
- GSE148663 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148663
- GSE89093  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89093
- GSE283951 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283951
- GSE37965  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37965
- GSE243529 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243529
- GSE314261 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314261

NHANES 1999-2002 downloads (all free, no login):
- dnmepi.sas7bdat              https://wwwn.cdc.gov/nchs/nhanes/dnam/
- DEMO.xpt (1999-2000)         https://wwwn.cdc.gov/nchs/nhanes/1999-2000/DEMO.XPT
- DEMO_B.xpt (2001-2002)       https://wwwn.cdc.gov/nchs/nhanes/2001-2002/DEMO_B.XPT
- NHANES_1999_2000_MORT_2019_PUBLIC.dat  CDC NDI Linked Mortality Files page
- NHANES_2001_2002_MORT_2019_PUBLIC.dat  CDC NDI Linked Mortality Files page

---

## How to reproduce any test

Each T*.py script is standalone. The general pattern:

```bash
# example: reproduce T14 secretory-class on GSE51032
python3 scripts/T14_GSE51032_secretory_class.py \
  --matrix /path/to/GSE51032_series_matrix.txt.gz \
  --output_dir results_myrun/T14_secretory_GSE51032/
```

Each script prints its parameters, reads its input, computes A-scores per
sample, runs the case-vs-control Cohen's d with permutation p-values, and
writes a JSON results file. The JSON will match the one in
`results/T14_secretory_GSE51032/` bit-exactly if you have the same input
and the same numpy version.

### T15 NHANES command (the new blinded-cohort test)

```bash
mkdir -p results_myrun/T15_NHANES
python3 scripts/T15_NHANES_blinded_prospective.py \
  --dnmepi  /path/to/dnmepi.sas7bdat \
  --demo_a  /path/to/DEMO.xpt \
  --demo_b  /path/to/DEMO_B.xpt \
  --lmf_a   /path/to/NHANES_1999_2000_MORT_2019_PUBLIC.dat \
  --lmf_b   /path/to/NHANES_2001_2002_MORT_2019_PUBLIC.dat \
  --output_dir results_myrun/T15_NHANES/
```

Runtime: approximately 60 seconds on a laptop. Random seed baked in: 20260420.

---

## Hash verification

Every result JSON has a SHA-256 recorded in the manifest. After running a
test, verify:

```bash
sha256sum results_myrun/T14_secretory_GSE51032/GSE51032_secretory_analysis.json
# Expected: 3fa2c12bc27bdc44578217b4c97eb709e09906a4a1352bfd24762f6ad241ba3d
```

A non-identical SHA means one of:
1. A different version of the input series matrix (GEO sometimes updates)
2. A different numpy version doing subtly different floating-point reductions
3. A substantive change to the script
4. You found an error we have not yet caught (please email)

---

## Figures

The 8 publication figures are produced by `figures.py` (bundled separately
outside this evidence folder). It reads the JSONs in `results/` and the
Tightening v2 numbers and renders all 8 figures as PNG and SVG.

```bash
# Edit paths in figures.py (or set env vars) and run
python3 figures.py
```

Environment variables:
- `EDEAR_RESULTS_DIR`  default `/home/claude/CrossPopValidation/results`
- `EDEAR_FIGURES_DIR`  default `/home/claude/edear_work/figures`

---

## What this bundle does NOT contain, and why

Per the disclosure architecture stated in EDEAR §3.5:

Not in this bundle:
- The framework's architectural-class taxonomy (what the 8 classes are,
  why that specific number, where the boundaries are)
- The class-assignment rule (how a new CpG/tissue/gene is assigned to a
  class)
- The calibration code (G-002 and G-003b MCMC likelihoods that produced
  the specific constants 0.838889 and 0.843264)
- The first-principles derivation of the H_min values (Jacobson-Landauer-
  virial-entropy chain; covered under USPTO provisional patents 64/012,720
  and 64/014,568 filed March 21 and March 23, 2026)

In this bundle:
- The two calibrated numerical constants
- The two CpG panels (Xu-538 and 19-CpG secretory)
- The scoring formula
- Every per-cohort test script
- Every per-cohort result JSON with SHA verification

This is the same three-layer disclosure architecture used by Grail
(Galleri), Foundation Medicine, and Natera (Signatera): clinical and
analytical validity are public and reproducible; panel derivation and
classifier internals are proprietary.

---

## Contact

hmahaffeyges@gmail.com

If you find an error in any script or any number, please email rather than
publishing a correction. We correct errors as soon as we are aware of
them, and acknowledgment traces back to the timestamped GitHub repository.
