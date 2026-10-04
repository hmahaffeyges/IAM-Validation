# IAM-Atlas v2 — how it was built, and how to rebuild it exactly

Atlas v2 is a per-cell reference: for each of **74 cells** at each array CpG, the cell's mean methylation on **our own Stage 1 array
scale**, its posterior SD and 95 % interval, its spread between the separate purified samples of that cell (the per-address noise), the number of samples behind it, 20 posterior draws, and the
per-locus prior. **No class enters the fit** — a cell's class only names the floor its A is divided by. Why each choice was made:
[DECISIONS.md](DECISIONS.md). What went wrong on the way: [LESSONS.md](LESSONS.md). The conversation: COMMUNICATION.md.

## Contents
| folder | what |
|---|---|
| `scripts/` | every step, numbered in run order (below). `stage_1_idat_calibration.py` is the chain's own Stage 1, copied so array sources are calibrated exactly as a patient array is. |
| `scripts/not_used/` | steps that ran and were rejected, kept as record (no IDATs; wrong genome build; partial coverage). |
| `scripts/exploratory/` | class-structure tests; not part of the atlas. |
| `inputs/` | the exact files the scripts read: roster, sample list, manifests, twin rule, source terms, probe map. |
| `records/` | every result: extraction reports, roster, twin tests, dry runs, stage A (not converged), source terms, bridges, stage-B timing block. |
| `environment/` | exact package versions of the two environments on the build machine. |

## Machine
AWS EC2 `c7a.32xlarge` (128 vCPU, 246 GB), Ubuntu 22.04.5, Python 3.11.16. Two virtualenvs:
`~/env` (`environment/box_env_requirements.txt`: methylprep 1.7.1, pandas 1.5.3 — Stage 1, extraction, roster, source terms) and
`~/mcmc` (`environment/box_mcmc_requirements.txt`: jax/jaxlib 0.10.2, numpyro 0.22.0 — the MCMC). Data root `/home/ubuntu/data`.
Stage-B timing: 0.83 s per locus per 4-chain process; ~814k loci; 32 processes on 128 cores.

## Rebuild, in order
| step | script | reads | writes / result |
|---|---|---|---|
| 01 | `01a_box_setup_and_test.sh`, `01b_init_hg19_index.sh` | — | environments; wgbstools hg19 CpG index (28,217,448 sites) |
| 02 | [`02_fetch_sources.py`](scripts/02_fetch_sources.py) | `inputs/loyfer_beta_list.json` | Loyfer 2023 (GSE186458) per-sample `.beta` files — `records/02_fetch_summary.json` |
| 03 | [`03_loyfer_extract.py`](scripts/03_loyfer_extract.py) | `manifest.csv` (see `inputs/ARRAY_MANIFEST_IS_IN_REPO.txt`) | Loyfer beta and depth at every array CpG; `probe_map_summary.csv` (= `inputs/probe_map_summary.csv.gz`) |
| 04 | [`04_moss_stage1.py`](scripts/04_moss_stage1.py) | `inputs/moss_manifest.csv` | Moss 2018 sorted-cell arrays through Stage 1 |
| 05 | [`05_salas_blood_stage1.py`](scripts/05_salas_blood_stage1.py) | — | Salas 2018 (GSE110554) and 2022 (GSE167998) sorted blood through Stage 1 — `inputs/salas_blood_manifest.csv` |
| 06 | [`06_tian_extract.py`](scripts/06_tian_extract.py) | — | Tian 2023 (GSE215353) brain non-neuron pseudobulk at array CpGs |
| 07 | [`07_encode_stem_extract.py`](scripts/07_encode_stem_extract.py) | — | ENCODE H1 / HUES64 WGBS at array CpGs |
| 08 | [`08_gse63409_hsc_stage1.py`](scripts/08_gse63409_hsc_stage1.py) | — | GSE63409 sorted bone-marrow HSC and progenitors through Stage 1 (AML arrays kept aside, never in the atlas) — `inputs/hsc_manifest.csv` |
| 09 | [`09_roster.py`](scripts/09_roster.py) | the above, `inputs/twin_family_thresholds_v1.json` | candidate cells, sample QC — `records/09_roster_summary.json` |
| 10 | [`10_twin_test_sample_level.py`](scripts/10_twin_test_sample_level.py) | `records/twins.csv` | sample-level twin test — `records/10_*`; roster final: `inputs/roster.csv`, `inputs/roster_samples.csv` |
| 11 | [`11_dryrun_and_stageA.py`](scripts/11_dryrun_and_stageA.py) | roster | smoke tests and stage A — `records/11*` (stage A did **not** converge; its source terms are not used) |
| 12 | [`12_source_terms.py`](scripts/12_source_terms.py) | roster | closed-form source terms + Loyfer bridge — `records/12*` |
| 13 | [`13_encode_bridge_gse116754.py`](scripts/13_encode_bridge_gse116754.py) | ENCODE extract | ENCODE term measured via GSE116754 — `records/13b_*`; all terms: `inputs/source_terms_v1.json` |
| 14–15 | `15_stageB_run_all.sh` → [`14_stageB_block.py`](scripts/14_stageB_block.py) | `inputs/{roster,roster_samples,hsc_manifest}.csv`, `inputs/source_terms_v1.json` | 700 blocks → `atlas_v2/blocks_v2/block_NNNNN{.parquet,_draws.npz,_prior.parquet}`; `stageB_summary.json` with the distinctness gate |

Random seeds are fixed (`PRNGKey(1000 + block)`), so a rebuild on the same package versions reproduces the posterior draws, not only
their summaries. A finished block is skipped on restart.

## Where the fitted atlas lives
The blocks (several GB) are not in git. After the run they go to S3 and a Zenodo deposit; [`ATLAS_V2_OUTPUT_MANIFEST.json`](postbuild/records/ATLAS_V2_OUTPUT_MANIFEST.json) (added when
the run finishes) lists every block with its SHA-256.

## Acceptance before use
The atlas is not used by the chain until it passes the distinctness gate and the acceptance tests A1–A9 in
[`../../doors/ATLAS_V2_SPEC.md`](../../doors/ATLAS_V2_SPEC.md). Status flags carried on readings: ASSUMED (GSE63409 array offset),
pooled (Tian brain cells).

## Related records elsewhere in the repo
`doors/ATLAS_V2_SPEC.md` (model, acceptance tests), `doors/ATLAS_SOURCES_SURVEY.md` (every source considered),
`doors/CLASS_HISTORY.md` and `doors/CLASS_ASSIGNMENT_RULE_DRAFT.md` (the eight classes), `doors/CLASS_USE_INVENTORY.md`,
`doors/PROC_OUTSPAN_01_PREREG.md` (the clean map built from this atlas), `doors/PROC_PARTIALCOV_01_PREREG.md` (not adopted, D9).
