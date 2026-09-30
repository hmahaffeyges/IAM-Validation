# Atlas v2 — after the build (2026-09-29 to 2026-09-30)

What was run on the finished atlas, the scripts that ran it and what each measured. Everything here reads atlas v2 only; nothing reads a
v1 file, name, marker panel or scale map. The pre-registrations and outcomes are in `doors/` (PROC_V5_HELDOUT_*, PROC_V12_IDENTITY_*,
PROC_V12_DIAG_01*, PROC_DECONV_V2_01_*).

| step | scripts | result |
|---|---|---|
| V5 held-out posterior check | [`v5_heldout.py`](scripts/v5_heldout.py), `run_v5.sh` | PASS: 92.68 % of 342,716 held-out observations inside the 90 % interval |
| V12 identity loci | [`v12_crossfit.py`](scripts/v12_crossfit.py), [`v12b.py`](scripts/v12b.py), [`v12c.py`](scripts/v12c.py), [`v12_diag.py`](scripts/v12_diag.py), [`v12_prod.py`](scripts/v12_prod.py) | two failures recorded; production build array-first with a per-locus sequencing correction: 74 of 74 cells, 434–58,938 loci (`runtime/iamatlas_v2_identity_loci_v1_1.json`, `records/v12_prod_build.csv`). Held-out transfer: array→array 87.7 % of readings NORMAL (28/29 cell medians); corrected sequencing→array 73.1 % (14/17). Bar 95 % not yet met |
| v2 composition solver | [`deconv_v2.py`](scripts/deconv_v2.py), [`sweep.py`](scripts/sweep.py), [`sweep2.py`](scripts/sweep2.py), [`check3.py`](scripts/check3.py), [`check4.py`](scripts/check4.py), [`prof_diag.py`](scripts/prof_diag.py), [`test_mixtures.py`](scripts/test_mixtures.py) | first pre-registered run failed CD4/CD8 (0.024/0.031); development sweep chose hybrid markers; with the Moss vascular-endothelium profile excluded (it reads as a blend of other endothelia, adipocytes and blood) the 24 known mixtures give CD4 0.0121, CD8 0.0124, NK 0.0092 mean absolute error, non-blood median 0.0038, no non-blood cell present (`records/check4_vascular_endothelium_in_out.log`) |
| v2 reader, PREVIEW | [`methylphys_v2.py`](scripts/methylphys_v2.py), [`run01.py`](scripts/run01.py), [`run02.py`](scripts/run02.py), [`bm_test.py`](scripts/bm_test.py), [`bm_where.py`](scripts/bm_where.py), [`affine_test.py`](scripts/affine_test.py) | bone-marrow progenitors absorbed healthy-blood mass until removed from the circulating-blood candidate set (`SPECIMEN_DROP`); run 02 on 24 GSE87571 arrays and 44 sorted GSE63409 AML arrays (`records/v2_chain_run02_readings.json`). Uncommissioned: not evidence |

Fixed 2026-09-30: `methylphys_v2.py` printed 'BELOW NORMAL' and had no tier above ELEVATED; it now uses the v1.5 scheme
(SUPPRESSED / NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED / BREACH). run 02's stored tier words predate the fix; its A values are unchanged.
