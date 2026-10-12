# DEV-TISSUE-LUNG-04 — Met-A on lung adenocarcinoma against the same patient's normal lung (development; NOT SUITABLE, 2026-10-12; tumors unread)

**Why.** Lung 1 (GSE94785) read NO FINGERPRINT; lung 2 (GSE83842) and lung 3 (TCGA-LUSC) were not suitable by their synthetic tests (tumors
unread). This applies the lung 3 v3 design unchanged (measured lung stroma and every atlas lung cell type in the panel and direction test) to a
different tumor type with its own normals.

**Data.** TCGA-LUAD open-access Illumina 450K IDATs from the NCI GDC (`data/DEV_TISSUE_LUNG_04/list_tcga_luad.py`, metadata only, outputs
`tcga_luad_450k_files.csv`, `layout_luad_output.txt`): 23 patients with lung adenocarcinoma and their own normal lung on the same plate.
**Prior reading:** TCGA-LUAD may have been analyzed earlier by the retired class method (CANON/data_spent.csv); the quantity tested here has not
been measured on these arrays by any version.
Normal arrays only until `sealed_luad_tissue.json` is committed; the tumor lock is inherited from the lung 3 fetch script.

**Synthetic tests:** `data/DEV_TISSUE_LUNG_04/luad_synthetic.py` (the lung 3 v3 design, run once on the normals). If it fails any test, LUAD is
not suitable and its tumors stay unread.

**Test data:** TCGA-LUAD

**Synthetic tests, run once, 2026-10-12** (`data/DEV_TISSUE_LUNG_04/luad_synthetic.py` at 8ad3bfc; outputs `luad_synthetic_output.csv`,
`luad_direction_output.csv`, `luad_synthetic_summary.txt`). Composition, planted 5 % and power PASSED, and the every-pair rule was chosen. The
direction test FAILED: a normal lung blended with lung endothelium (60-90 %) raises R and fakes the chosen rule. **LUAD is not suitable; its
tumors stay unread.**
**Not a milestone:** not suitable (development).

