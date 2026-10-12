# DEV-TISSUE-LUNG-04 — Met-A on lung adenocarcinoma against the same patient's normal lung (development; design, 2026-10-12; NOT sealed)

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
**Not a milestone:** design (development).
