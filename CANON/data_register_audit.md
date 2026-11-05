# Data register audit (series named in committed sample lists or run scripts, missing from `CANON/data_spent.csv`)

_Mechanical check: every GSE id in a sample list, input list or run script under `Biological_Physics/MethylPhys/doors/data/`, against the
spent-data register. A series is added to the register only when a committed record shows it was read (the v2.1 build's IN samples, the v2.1
held-out test's verdicts, a scored output). Until then it is listed here, not in the register._

**Added to the register:** GSE350403 (partly spent by the showcase, its samples named) and 27 IAM-Atlas v2.1 series confirmed read by its build or its held-out test.

**Still to check, one by one** (named in a list or script; whether any sample was read is not yet confirmed from a record):

| series | named in |
|---|---|
| GSE102970 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/u1_samples_v21.csv` |
| GSE111445 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE114753 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv` |
| GSE126017 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/u1_samples_v21.csv` |
| GSE127824 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE134217 | `DEV_ATLAS_V21_01/geo_GSE122126_samples.txt` |
| GSE135205 | `PROC_TUMOUR_01/RECOVERED_CELLS/10-01T2204_aa048b1a_ok.sh` |
| GSE137716 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE146093 | `DEV_METAA_ATLAS_CURVE_01/bt_confirmation_samples.csv` |
| GSE146552 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE174666 | `DEV_METAA_ATLAS_CURVE_01/bt_samelab_samples.csv` |
| GSE184116 | `DEV_ERASE_01/graded_samples.csv` |
| GSE184500 | `DEV_METAA_ATLAS_CURVE_01/bt_confirmation_samples.csv` |
| GSE192451 | `DEV_METAA_ATLAS_CURVE_01/bt_confirmation_samples.csv` |
| GSE269608 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv` |
| GSE273074 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv` |
| GSE281305 | `SHOWCASE_V4/showcase_v4.sh`, `SHOWCASE_V4/showcase_v4_run2.sh` |
| GSE329728 | `PROC_TUMOUR_01/RECOVERED_CELLS/10-01T2204_aa048b1a_ok.sh`, `PROC_TUMOUR_01/RECOVERED_CELLS/10-02T1016_a754aeba_ok.sh` |
| GSE63409 | `DEV_ATLAS_V21_01/final_samples_v21.csv`, `PROC_HMIN_REFIT_01/hminA_samples.csv` |
| GSE68456 | `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE82084 | `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE83842 | `DEV_TISSUE_LUNG_02/samples_lung_02.py` |
| GSE94785 | `DEV_TISSUE_LUNG_01/samples_lung_01.py` |
