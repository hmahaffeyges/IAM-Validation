# Data register audit (series named in committed sample lists or run scripts, missing from `CANON/data_spent.csv`)

_Mechanical check: every GSE id in a sample list, input list or run script under `Biological_Physics/MethylPhys/doors/data/`, against the
spent-data register. A series is added to the register only when a committed record shows it was read (the v2.1 build's IN samples, the v2.1
held-out test's verdicts, a scored output). Until then it is listed here, not in the register._

**Added to the register:** GSE350403 (partly spent by the showcase, its samples named) 27 IAM-Atlas v2.1 series confirmed read by its build or its held-out test; then, one by one, six more confirmed from their records (GSE137716 and GSE63409, v2.1 final build sources; GSE281305, the showcase mixtures; GSE94785, the lung test; GSE135205 and GSE329728, the DNMT1 tests).

**Held, unread** (named in a committed list for a test not yet run; `DEV_METAA_ATLAS_CURVE_01.md`: the confirmation is not read): the B and T confirmation arrays GSE146093, GSE184500 and GSE192451, and GSE174666 (same laboratory, kept apart).

**Listed or screened by label only, not read** (an entry in an inventory, roster or candidate list; no record reads a value):

| series | named in |
|---|---|
| GSE102970 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/u1_samples_v21.csv` |
| GSE111445 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE114753 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv` |
| GSE126017 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/u1_samples_v21.csv` |
| GSE127824 | `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE134217 | `DEV_ATLAS_V21_01/geo_GSE122126_samples.txt` |
| GSE146552 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE184116 | `DEV_ERASE_01/graded_samples.csv` |
| GSE269608 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv` |
| GSE273074 | `DEV_ATLAS_V21_01/atlas_v21_build_samples.csv`, `DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv` |
| GSE68456 | `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE82084 | `DEV_ATLAS_V21_01/build_samples_v21.py` |
| GSE83842 | `DEV_TISSUE_LUNG_02/samples_lung_02.py` |
