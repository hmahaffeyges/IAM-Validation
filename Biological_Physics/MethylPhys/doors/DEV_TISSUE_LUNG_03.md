# DEV-TISSUE-LUNG-03 — Met-A on squamous lung tumors against the same patient's normal lung (development; NOT SUITABLE, 2026-10-12; tumors unread)

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

**Synthetic tests v1 on the 42 real normal lungs, 2026-10-12** (`data/DEV_TISSUE_LUNG_03/lusc_synthetic_v1_FAILED.py`, outputs
`lusc_synthetic_v1_FAILED_*`, `lusc_direction_v1_FAILED_output.csv`). These normals differ little between people (stand-in SD 0.0230), so the pair
bar is 1.0383 and the test is very sensitive: a change the size of lung 1's lean is detected in every draw. Composition, planted 5 % and power
PASSED. **Direction (test 7) FAILED:** blending 30-60 % stroma into a normal lung pushes the count rule to a false call; a tumor with more stroma
than its normal could read FINGERPRINT from composition alone. **Not sealed.**
**v2, committed before it is run:** the same tests with blind sites tightened to 0.03 (the DEV-TISSUE-LIVER-01 v2 change), run once. If v2 fails
any test, the set is not suitable and the 40 tumors stay unread.

**Synthetic tests v2, run once, 2026-10-12** (`data/DEV_TISSUE_LUNG_03/lusc_synthetic.py`, outputs `lusc_synthetic_output.csv`,
`lusc_direction_output.csv`, `lusc_synthetic_summary.txt`): planted 5 % PASSED; composition FAILED (an immune blend crosses the bar); power and
direction FAILED (neither rule is free of composition). **Decision, by the rule written before running: not suitable; not sealed; the 40
tumors stay unread** and remain available to a design that can hold lung composition still.

**Why, and what it needs.** The atlas holds no lung fibroblasts or lung smooth muscle (lung tests use colon and heart fibroblasts as stand-ins), so
the blind sites cannot be checked against the stroma actually in lung; with normals this alike, that gap decides. A lung design needs purified
healthy lung stromal cells in the atlas (DEV-ATLAS-COMMISSION-01, use U2) or composition measured per sample.

**Synthetic tests v3, run once, 2026-10-12** (`data/DEV_TISSUE_LUNG_03/lusc_synthetic.py` at ef71030, outputs `lusc_synthetic_output.csv`,
`lusc_direction_output.csv`, `lusc_synthetic_summary.txt`; v2 kept as `_v2_FAILED_*`). Panel and direction test now include measured lung stroma
(GSE146376 airway smooth muscle, GSE107226 lung fibroblasts; both cultured, so the direction test covers cultured stroma) and every lung cell
type in the atlas. Planted 5 % PASSED; composition, power and direction FAILED: an immune blend (with lung macrophages) moves R by about 4 %
against a pair bar near 5 % at these tightly matched normals. **LUSC is not suitable for a fingerprint call; its 40 tumors stay unread.** Next:
TCGA-LUAD with the v3 design (DEV-TISSUE-LUNG-04).
**Not a milestone:** not suitable (development).

**Test data:** TCGA-LUSC
