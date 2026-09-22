# CPG_CMB — Run Manifest & Cold-Start README
**Goal:** a future AI (or a fresh session) can clone this, decompress the atlas, and run real IDAT
files immediately. Every path below maps to a key in `cpg_conductor.py : DEFAULT_CONFIG`. The
engine root is set once via the env var **`CPG_ENGINE_ROOT`** (and `CPG_ROOT`, kept equal).

---
## 0 · One-time setup (cold start)
**One command does all of it** — finds/installs the right Python, installs every missing package at a
working version, decompresses the atlas, and checks the HEALPix mapping. Safe to re-run:
```bash
./bootstrap.sh            # core chain
./bootstrap.sh --idat     # also install methylprep for raw IDAT decode
```
The manual equivalent (what bootstrap.sh automates):
```bash
# 1. set the engine root (the folder that holds cpg_conductor.py)
export CPG_ENGINE_ROOT="/path/to/CPG_CMB_v5"
export CPG_ROOT="$CPG_ENGINE_ROOT"

# 2. decompress the atlas (ships compressed; ~605 MB .xz -> ~578 MB .csv)
xz -dk "$CPG_ENGINE_ROOT/MethylPhys/atlas/IAMAtlasREBUILD.csv (historical path).xz"   # -k keeps the .xz

# 3. python deps
pip install numpy pandas scipy healpy pillow reportlab --break-system-packages
#   for IDAT decode (Stage 1) only:
pip install methylprep --break-system-packages      # + the EPIC/HM450 manifest it pulls

# 4. (optional) regenerate the cpg->HEALPix mapping if the .npy is absent:
python "MethylPhys/atlas/healpix_mapping/generate_cpg_healpix_mapping.py"
```

## 1 · Entry points
```python
# A) from raw IDAT (production): Stage 1 noob calibration, then the chain
from stage_1_idat_calibration import calibrate_idat_to_beta
import cpg_conductor as C

beta, meta = calibrate_idat_to_beta(grn_path, red_path)      # one array, a Series indexed by cg-id
bundle = C.run_full(beta.dropna().to_dict(),
                    "../atlas/IAMAtlasREBUILD.csv",            # decompress the .xz first
                    cfg={"age": 72, "pipeline": "stage1_noob_450K",
                         "lab_zero": -0.0117, "lab": "GSE87571"})

# B) the report
import sys; sys.path.insert(0, "MethylPhys_Interface")
import build_methylphys as B
B.build(bundle, "report.html", "GSM2333901")
```
Or from the command line, which does both:
```bash
python "MethylPhys_Interface/run_sample.py" --grn <Grn.idat.gz> --red <Red.idat.gz> \
       --age 72 --lab GSE87571 --lab-zero -0.0117 --out report.html --id GSM2333901
```
Without `--lab-zero` the laboratory zero reads UNSET and the placement, tier and departure are withheld;
without `--age` the absolute reading is withheld. A_mapped is still printed in both cases.

## 2 · The dependency tree (every file the chain reads, by role)

### Core engine (Python)
| File | Role |
|---|---|
| [`cpg_conductor.py`](cpg_conductor.py) | the chain: all stages in call order, the entry point |
| `Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py` | A = H(β)/H_min per cell |
| `Walther_iam_deconvolver/walther_iam_deconvolver.py` | composition/presence (gates no call) |
| [`cpg_gauge.py`](cpg_gauge.py) | reference bar gauge + star gauge renderer |

### The atlas (decompress first)
| File | Role |
|---|---|
| `MethylPhys/atlas/IAMAtlasREBUILD.csv.xz` → `.csv` | 115 cell types × 483,092 CpGs (class + per-cell mean/sd/ci) |
| `MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json` | cell→class membership |
| `MethylPhys/atlas/IAMAtlasREBUILD_provenance.json` | **single source of truth for frozen H_min** (2026-04-06) |
| `MethylPhys/atlas/iamatlas_class_archives/*.tar.xz` | per-class brightness (mean/sd/ci_lo/ci_hi) → CI + reliability + CMB |

### Disease comparison (the cellular comparison engine)
| File | Role |
|---|---|
| `Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_13.csv` | per-disease per-cell signatures |
| `Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json` | 115 atlas cells → matrix columns |
| `Disease Matrix/DISEASE_MATRIX/disease_origin_cells.json` | **cell-of-origin specificity gate** (NEW — built 2026-06-29; without it solid cancers get named off immune cells) |

### Runtime matrices / priors / panels
| File | Role |
|---|---|
| `Runtime Matrices/Tier_breakpoints/tier_breakpoints.json` | gauge breakpoints (1.07 Warburg, 1.10 breach) |
| `Runtime Matrices/Literature_anchors_Report building/literature_anchors.json` | published anchors |
| `Runtime Matrices/Mahalanobis_healthy_reference/…_v1_0_derived.json` (+ scorer) | derived-hull verdict (Layer 0) |
| `Runtime Matrices/Directional Panel/…` | AD sealed 7-CpG directional panel |
| `Runtime Matrices/Celltype_Marker/…` · `Collinearity_Groups/…` | markers (NILC collinearity = deferred/shelf) |

### Report assets (Stage 9)
| File | Role |
|---|---|
| `MethylPhys/chain/Crown Jewel and Patient Strawman/strawman_data_v2.json` · [`render_patient_wall.py`](report_builders/render_patient_wall.py) · [`render_strawman_v2.py`](report_builders/render_strawman_v2.py) | patient straw man |
| `Crown Jewel and Patient Strawman/IAM_Disease_Wall_CROWN_JEWEL_v3.html` | crown-jewel wall (v3) |
| `MethylPhys/atlas/healpix_mapping/iamatlas_cpg_to_healpix_nside128.npy` (+ provenance) | CMB pixel map |
| `CPG_AstroGenetics_explainer_section.html` | "How CPG works" explainer |
| `A1_reference_gauge.png` · `star_gauge.png` | gauge images (cpg_gauge regenerates from tier_breakpoints if absent) |

### Deferred (NOT in the lean run — shelf)
`NILC Deconvolver/*` · the agreement AND-gate · the fraction-presence gate · collinearity groups.

## 3 · Python dependencies
`numpy`, `pandas`, `scipy`, `healpy` (CMB render), `pillow` (plate thumbnails), `reportlab` (PDF).
IDAT decode only: `methylprep` (+ array manifest). Standard library: `csv`, `json`, `importlib`, `pathlib`.

## 4 · Provenance discipline (do not violate)
- **No external atlas/reference/matrix, ever.** The IAMAtlas is physics-derived (G-002 floor, MCMC posteriors).
  CpG genomic order is hg19 coordinate only (a property of the genome), used solely for the CMB sky-map ordering.
- H_min is frozen (provenance JSON); Stage 4 refuses to run if runtime H_min disagrees.
- DERIVED-IAMAtlas-only: a cohort supplies a disease **direction**, never a baseline.

## 5 · What still needs to be added to a perfectly cold clone
- [ ] the `.npy` HEALPix mapping if not shipped (regenerate via the script above)
- [ ] `methylprep` + array manifest for IDAT decode (Stage 1)
- [ ] confirm `iamatlas_class_archives/*.tar.xz` are present (CI + reliability + CMB depend on them)
- [ ] [`disease_origin_cells.json`](Disease%20Matrix/DISEASE_MATRIX/disease_origin_cells.json) present in `Disease Matrix/DISEASE_MATRIX/` (patient-safety gate)
