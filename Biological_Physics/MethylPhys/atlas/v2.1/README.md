# IAM-Atlas v2.1: compact copy

The fitted atlas (2026-10-16): 79 cell types, 813,951 array sites, from the Bayesian fit of record `../../doors/DEV_ATLAS_V21_01.md`.
**Not yet commissioned:** the use tests on held-out arrays are next. Until then, v2 (`../v2/`) is the reference in use.

| File | What it holds |
|---|---|
| `sites.parquet` | every site id, in the row order of the tables below |
| `mu_part1-3.parquet` | best estimate (posterior mean) per site and cell; columns are the cells, in the source atlases' exact names |
| `mu_lo_part1-3.parquet`, `mu_hi_part1-3.parquet` | the 2.5th and 97.5th percentiles (the 95 % range of the cell's average) |
| `donor_sd_part1-3.parquet` | the fit's person-to-person spread of each cell at each site (posterior mean of the donor SD); a new person's cell varies this much around the average |
| `compact_manifest.json` | each file's size and SHA-256 |

Values are stored as round(value x 10000) in unsigned 16-bit integers: value = stored / 10000 (exact to 4 decimals); **65535 = not measured**.
The 79 cells are split into three files in alphabetical order so every file stays under GitHub's 100 MB limit; join the parts side by side.

Checked 2026-10-16: every file matches its checksum, and on three cells at every site the copy differs from the full atlas by at most 0.00005
(the rounding). Made by `../../doors/data/DEV_ATLAS_V21_01/export_compact_v21.py` (rounding only). The full atlas file (2.4 GB, with donor
spread, R-hat and ESS per site) and the 100 posterior draws per block (24 GB) are in S3 (`results/ATLAS_V21/assembly/`, `results/ATLAS_V21/fit/blocks/`)
and go to Zenodo with the release.

```python
import pandas as pd
sites = pd.read_parquet("sites.parquet").cpg
mu = pd.concat([pd.read_parquet(f"mu_part{n}.parquet") for n in (1, 2, 3)], axis=1)
mu.index = sites; mu = mu.where(mu != 65535) / 10000
```
