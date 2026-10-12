# DEV-TISSUE-LIVER-01 — Met-A on liver tumors against the same patient's normal liver (development; design, 2026-10-12; NOT sealed)

**Why.** Liver first among the remaining cancers (author, 2026-10-12). The design of DEV-TISSUE-BREAST-01: each patient's tumor against that
patient's own normal tissue processed on the same plate.

**Data.** TCGA-LIHC open-access Illumina 450K IDATs from the NCI GDC (`data/DEV_TISSUE_LIVER_01/list_tcga_lihc.py`, metadata only, output
`tcga_lihc_450k_files.csv` and `layout_lihc_output.txt`): 50 patients with one primary tumor and one normal liver sample, all 50 on the same plate.
**Prior reading:** TCGA-LIHC was analyzed earlier by the retired class-gauge method only (old VAL; CANON/data_spent.csv), a different quantity;
this measurement has not been made on these arrays.
The four GEO liver sets were set aside for now: their slide IDs are only inside the array files, and finding them would mean opening tumor files;
GSE157341 shows slides, but only 5 of its 24 pairs share one.

**Inputs** (`fetch_tcga_lihc.py`, `calib_tcga_lihc.py`, from DEV-TISSUE-BREAST-01): normal liver only until `sealed_liver_tissue.json` is committed.

**Before any seal (LESSONS T15, from DEV-TISSUE-LUNG-01):** (1) measure the normals' between-person spread and report it against breast, colon and
lung; (2) synthetic tests that plant changes down to 1 % and the sizes real tumors have shown; (3) a rule that uses every pair, tested beside the
count rule, the one with more power at small changes sealed. Normal liver next to a tumor is often scarred (cirrhosis), so its spread may be wide.
**Not a milestone:** design (development).
