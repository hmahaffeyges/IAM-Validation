# CPG / GAPE Issue 003 — Reproduction Kit

**See also `COMPONENT_MAP.md`** — what lives in the repo, what lives in the kit, what lives only in the author's folder, and the repo commits this kit implies.

**Purpose.** Everything used to produce GAPE Issue 003 and to verify the measurement core of the CPG chain,
in one directory, so that any person or any AI can run it without being taught the project first.
Every script prints its own input / operation / expected / observed / verdict block.

**Provenance.** Engine and runtime files are from `github.com/hmahaffeyges/IAM-Validation` at commit
`66f37fe3aa7a5302d77cbe418aba15fe4a11b471` (2026-07-03), except where RULING M1b (below) adopts the author's
corrected copy. `CHECKSUMS.sha256` covers every file in this kit; verify with `sha256sum -c CHECKSUMS.sha256`.

---

## 0. Read this first — the four rules that took a day to learn

1. **The physics measures; cohorts only point.** Never compute a group statistic (Cohen's d, AUC, Mann-Whitney)
   as the primary result. Each sample is read as an absolute A against the fixed reference (H_min, the age band).
   Cohorts enter only to establish the *direction* a disease moves a class.
2. **Two surfaces, two aggregations, fixed by their references (RULING A3).**
   - Class **gauge** (8 classes, identity loci): `A = H(mean β) / H_min(class)`. H_min is itself H of a mean β; the age band is compiled this way.
   - **Separation** (115 cell types, discriminative markers): `A = mean_i H(β_i) / H_min(class)`. Markers are bimodal; averaging β first is wrong here (SOP v1.4.0 §105). The sealed anchors are this statistic.
   - Never read a statistic against a band compiled the other way.
3. **Presence before score.** A class absent from the substrate (deconvolved fraction < `DETECT_FLOOR` = 0.01) is not scored.
   Its identity loci in a sample that contains none of its DNA read as a uniform offset, not as a finding.
   The gauge in `cpg_kit.gauge_A` refuses absent classes and bimodal panels.
4. **Substrate decides what can be read.** Whole blood carries immune architecture only (epithelial fraction ≈ 0 in
   healthy donors, by biology). Plasma cfDNA carries shed tissue. Bulk tissue is a mixture and inflates `H(mean β)`.
   Every H_min is per (class, substrate): the 40-cell table is in `engine/cpg_gauge_engine.py::H_MIN_TABLE`.

---

## 1. Environments

| env | purpose | pins |
|---|---|---|
| `python` (default) | everything except Stage 1 | numpy ≥ 1.26, pandas ≥ 2, scipy, reportlab (for the Issue build), pypdfium2 (page checks) |
| `methylprep` | Stage 1 only (`PROC_CAL_01.py`) | **python 3.11, methylprep==1.7.1, numpy==1.26.4, pandas==1.5.3, pytz, python-dateutil** — methylprep calls `DataFrame.append`, removed in pandas 2 |

Stage 1 also needs: `HOME` pointing at a writable directory (methylprep writes `$HOME/.methylprep_manifest_files/`), and
network access to `https://array-manifest-files.s3.amazonaws.com/` for the Illumina manifests on first use
(`HumanMethylation450k_15017482_v3.csv.gz`, `HumanMethylationEPIC_manifest_v2.csv.gz`). Offline: place both files in that directory.

---

## 2. Large inputs (not shipped) — put them in `data/` or point `CPG_KIT_DATA` at them

| file | source | notes |
|---|---|---|
| `IAMAtlasREBUILD.csv` | repo `Biological_Physics/AstroGenetics/CPG_KISS_Commercial_Engine/IAM_Atlas/IAMAtlasREBUILD.csv.xz` → `xz -d` | 605 MB, 483,092 rows, build 2026-05-28 |
| `betas_cache.pkl` | `10_TEST_DATA.zip` (author) | 140 MB; Stage-1 output for the 11 test samples |
| `idats/*_Grn.idat.gz, *_Red.idat.gz` | `10_TEST_DATA.zip` | 11 pairs: 7 × 450K whole blood, 4 × EPIC colorectal tissue |
| `GSE51032_series_matrix.txt.gz` | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51032/matrix/` | 3.0 GB |
| `GSE51057_series_matrix.txt.gz` | same path, GSE51057 | 1.2 GB |
| `GSE122126-GPL21145_series_matrix.txt.gz` | `.../GSE122nnn/GSE122126/matrix/` | plasma cfDNA + in-vitro mixes (Moss 2018) |

---

## 3. Procedures — run in this order; each is independent

| script | env | what it proves | expected | observed 2026-09-19 |
|---|---|---|---|---|
| `PROC_CAL_01.py` | methylprep | raw IDAT → β reproduces the project's Stage-1 cache | bit-identical | **11/11, r=1.000000, max diff 0.000000** |
| `PROC_DECON_01.py` | python | deconvolver reproduces `TEST_DATA_MANIFEST.md`; whole blood reads epithelial ≈ 0; gauge read with presence gate and age band | MAE ≤ 0.001; epi < 0.02 | **MAE 0.0004 / 0.0002 / 0.0002; WB epi 0.000–0.011** |
| `PROC_ANCHOR_01.py` | python | sealed 115-cell anchors reproduce from raw GEO | r ≥ 0.9999 | **GSE51032 r=1.00000 (112/115); GSE51057 r=1.00000 (115/115)** |
| `PROC_FORMULA_01.py` | python | the measurement behind RULING A3 | see docstring | WB immune Spearman +1.000, offset +0.029; tissue +0.24; band = H(mean β) 80/80 |
| `PROC_PLASMA_MIX_01.py` | python | deconvolver vs real known mixtures (Moss Table 6) | per-tissue r ≥ 0.9 | **terminal PASS (r=0.945); secretory FAIL (0.19); cycling FAIL (0.10)** |

Results land in `results/PROC_*.json`. `results/VAL_INDEX.{csv,json}` is the mechanical index of all 103 VAL identifiers in the repo (Issue 003 Appendix V).

`PROC_SKIES_01` (not a script yet): Issue 003 Fig. 5A-1 'Four skies' was produced by the repo's own `cpg_patient_cmb.py` on GSM1051533 + `IAMAtlasREBUILD.csv` immune_mean/immune_sd + CAMB Planck-2018 → `healpy.synfast`; needs `healpy`, `camb`, and `HOME`/`XDG_CONFIG_HOME` pointed at a writable dir (astropy config). A PROC that prints FAIL is a finding, not an error — record it.

---

## 4. Rulings recorded in this kit (Issue 003 §1.5)

- **A3** — one aggregation per surface (rule 2 above). `SOP v1.4.0 §105` is amended to scope its "never H(β_mean)" to marker panels.
- **M1b** — `runtime/iamatlas_celltype_markers_v0_2.json` is the **chrX-removed** file (131 chrX markers dropped 2026-06-11 for
  derived sex-invariance). The repo HEAD copy is kept as `..._REPO_HEAD_prechrX.json` because the 2026-05-29 seal (`anchors_v1/`)
  was made with it. `anchors_v2/` are the re-sealed values under the canonical file: 32/115 cells shift, max 0.059 (Mela),
  r = 0.9996 to v1. **Author action:** commit the chrX-removed file to the repo; mark `anchors_v1` SUPERSEDED.

---

## 5. Building Issue 003

```
cd issue003_build
CPG_TRIAL=../runtime python build_gape_issue003.py IAMPerformance_GAPEIssue003_DRAFT.pdf
```
`data003.py` holds every number printed in the document; change a value there and rebuild. `gape002_lib.py` is the Issue 002
script with its `build()` cut into page functions — every 002 primitive, card and section reused verbatim.
Note: `data003.py` reads `handoff/*.json` relative to its parent for the live-run tables; in this kit those live in `results/` —
set `CPG_HANDOFF=../results` or copy them.

---

## 6. What is still open (do not assume it is settled)

- **Healthy whole blood reads below the age band** (43M z −2.07, 58M −4.03, 67F −3.49) on the project's own Stage-1 betas. Not a
  Stage-1 artifact (PROC-CAL-01). Leading hypothesis: the band was compiled on a different IDAT→β pipeline (Xu-538 relativity, 0.38–0.62).
  Test: locate the pipeline the band was compiled from, or re-derive the band from ≥30 public healthy blood samples through Stage 1.
- **Deconvolver tissue-of-origin.** Passes on synthetic linear mixes and on the terminal class; does not recover hepatocyte or colon
  spikes into their classes; routes shed epithelium to gastric references. Atlas work, not solver work (LESSON-DECONV-01).
- **Two presence floors** (1% conductor, 3% adjudicator) and **two tier vocabularies** in the corpus — see Issue 003 RECON D2, T3.
- **Stage 3 is not wired by decision** (SOP §104 foreground firewall) — do not add age/sex/smoking subtraction.
- Recipe §6.3 (vault) still states the pre-§105 ruling; the author records the correction there.

---

*Nothing in this kit is clinical validation. Public retrospective data, small n, no prospective testing.*

### 6a. The below-band mechanism, quantified (added after the guard test)
The immune identity loci are selected where the atlas immune reference sits at β ≈ 0.73 (`H_min_beta` 0.7318, band ±0.05).
In all seven Stage-1 whole-blood samples those same loci read at **β_mean 0.786–0.822, i.e. +0.054 to +0.090 above the atlas
reference** — outside the ±0.05 selection band in 7 of 7 (smallest shift +0.0544). Higher β at these loci means lower entropy, hence A ≈ 0.81–0.89 against
a band centred near 0.95. So "below band" is a reference-β offset between the atlas's source pipelines and noob Stage-1 output,
not a Stage-1 error and not a property of the donors. The test that closes it: re-derive `H_min_beta` and the age band from ≥30
public healthy whole-blood IDATs run through Stage 1 (`PROC_CAL_01.py`), then re-read the seven.
