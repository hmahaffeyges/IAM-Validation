# CPG CMB v2 — Engine snapshot for a future build/test session

This folder is a self-contained snapshot of the Cellular Performance Gauge (CPG) clinical
methylation chain, current as of this push. It is here so a future AI can run the chain,
reproduce the tests, or fix/extend it without re-deriving the architecture.

## What CPG does
From one patient methylation IDAT (450K or EPIC), the chain deconvolves the sample into the
8 architectural classes and 115 cell types, computes per-cell and per-class A-scores
(architectural state vs the derived healthy floor of 1.0), matches against the disease
signature matrix (Route B concordance + Mode 2 cell-of-origin presence), tiers the signal,
and writes one clinician-facing report. A second "confirmation" chain runs only when a flag
fires (it never mutates the primary): a derived-Mahalanobis global-departure adjudicator,
plus literature-anchor and residual-map qualifiers, appended as one integrated verdict.
From the second draw on, a per-cell trajectory tracks change against the patient's own prior.

## How to run
1. Python 3.11 with: methylprep 1.7.1, numpy, pandas (<2 for native methylprep), scipy,
   scikit-learn, matplotlib. (`conda create -n cpg python=3.11` then pip install.)
   - NOTE: in a Python 3.12 container, methylprep 1.7.1 needs the pandas-compat shim at
     `TEST_DATA/harness/pdshim.py` (it restores DataFrame.append). On 3.11 no shim is needed.
2. Per-patient run: `python run_batch.py --patients /path/to/patients` where each patient is
   `patients/<ID>/<YYYY-MM-DD>/` containing the IDAT pair + `questionnaire.json`.
   Single visit: `python walther_clinical.py --folder <visit_folder> --out <visit_folder>`.
3. The conductor auto-resolves the engine root by locating `IAM_Atlas/` and auto-decompresses
   `IAM_Atlas/IAMAtlasREBUILD.csv.xz` on first run.

## Atlas
The runtime atlas `IAM_Atlas/IAMAtlasREBUILD.csv.xz` is included here. The per-class brightness
archives are in `IAM_Atlas/iamatlas_class_archives/`. These are identical to CPG_CMB_v1 and to
`Biological_Physics/atlas_vault/`. The large decompressed CSV is NOT shipped (regenerated on run).

## What's current in v2 (vs v1)
- Mahalanobis adjudicator presence gate: a class counts only if genuinely present (abundance
  >= 3%) AND outside the NORMAL band [0.95, 1.04). Stops suppressed non-substrate classes
  (stem_pluri, terminal in blood) from inflating the distance. (stage_5_second_chain.py)
- Per-cell trajectory: per-cell deltas (not class scores) + rotation-toward-signature, led by
  deconvolver-resolved cells; bulk pseudo-cells excluded. (walther_clinical.py `_compute_trajectory`)
- Report: trajectory section, two-deconvolver explainer, Mahalanobis callout. (cpg_report_builder.py)
- Flowchart updated to match (flowchart_v4.html).

## Tests / reproduction
See `TEST_DATA/TEST_DATA_MANIFEST.md` for every sample (GEO accession, array, substrate) and the
result it established. Harness scripts are in `TEST_DATA/harness/`; demo reports in
`TEST_DATA/reports/`. Raw IDATs are public (URLs in the manifest); the calibrated betas cache
(`betas_cache.pkl`, ~100MB) is regenerable from them and is not shipped here.

## Key validated facts
- Whole blood DNA is ~96% immune (the immune class = all 51 leukocyte types incl. granulocytes);
  secretory/epithelial reads ~0 in healthy blood — correct, not a limitation.
- The chain reads secretory/cycling when epithelial DNA is present: CRC carcinoma (EPIC) stage 1
  secretory 12.9%, stage 4 secretory 24.8% (rising with stage). EPIC arrays are supported.
- Substrate matters: whole blood = immune-architecture / field-effect readout; plasma cfDNA =
  direct shed-tumour readout; tissue = positive control. A report must state its substrate.

## Standing rules (carried)
Source-doc before concluding; no fabrication; referee language ("consistent with", never
"confirms/validates/proves"); DERIVED-IAMAtlas-only (no cohort pooling); surgical edits with
before/after; set up tests fully then await go.
