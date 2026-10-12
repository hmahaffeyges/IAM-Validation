# DEV-TISSUE-LUNG-03 — Met-A on squamous lung tumors against the same patient's normal lung (development; design, 2026-10-12; NOT sealed)

**Why.** Lung 1 (GSE94785, 28 pairs) read NO FINGERPRINT with most tumors leaning up; lung 2 (GSE83842, 12 pairs) was not suitable: normal lungs
differ so much between people that 12 pairs cannot decide. This set has many more pairs.

**Data.** TCGA-LUSC open-access Illumina 450K IDATs from the NCI GDC (`data/DEV_TISSUE_LUNG_03/list_tcga_lusc.py`, metadata only, outputs
`tcga_lusc_450k_files.csv`, `layout_lusc_output.txt`): 40 patients with lung squamous cell carcinoma and their own normal lung, all on the same plate.
**Prior reading:** TCGA-LUSC was analyzed earlier by the retired old VAL / class method only (CANON/data_spent.csv), a different quantity; this
measurement has not been made on these arrays.

**Reading:** as DEV-TISSUE-LUNG-01 and -02 (both lung epithelia agree with every panel cell at the blind sites). **Inputs:** `fetch_tcga_lusc.py`
(normal arrays only until `sealed_lusc_tissue.json` is committed), `calib_tcga_lusc.py`.
**Before any seal:** the lung 2 synthetic design (LESSONS T15 and the direction test) on the 42 normals; if a planted 5 % change or power fails,
the set is not sealed and the tumors stay unread.
**Not a milestone:** design (development).
