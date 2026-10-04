# Chain v3 development round 2 - report (DEVELOPMENT - not commissioned)

Date 2026-10-04. Repository `hmahaffeyges/IAM-Validation`, base `main` at `185f609`; five local commits on top (not pushed):
`8ef7d24` check notes written before data, `8f64db3` one source line in a note (placenta series named before data), `7c33552` chain code,
`74507ed` outcomes and documents, and a fifth commit adding the release-check result, the rebuilt manual PDF and two reworded docstrings. Box: ssh:methylphys-cpu-01. Every check was written in a dated `doors/DEV_*.md` note before its data were read;
each outcome sits under the line in that note. Status labels as in the notes. Neutrophils are the only cell read.

| item | stage | what was tried (note) | result | wired / flag / out | next |
|---|---|---|---|---|---|
| F age and sex optional | 0 | `--sex`, `--age` optional, `NOT_DECLARED` (DEV-INTAKE-02) | 1,569 arrays that stopped in round 1 re-run: 1,569/1,569 end to end, 0 crashes, 0 stops on age or sex; sex not declared on 278 | **wired** | - |
| L blood specimens only | 0 | `ACCEPTED_SPECIMENS`, `SPECIMEN_REFUSED` naming the specimen (DEV-INTAKE-02) | 955 refused (PBMC 235, sorted T 189, cell line 134, placenta 93, monocytes 86, B 76, marrow 71, unspecified 43, other 28), each with report and bundle, no Met-A; 613 blood specimens read | **wired** | each refused specimen enters when it has its own reference |
| B hashed ids | 0 / 13 | sha256 (32 hex) in bundle and ledger; report keeps the typed id (DEV-INTAKE-02) | typed id in 0 of 1,837 bundles, 0 of 1,837 ledgers; in 1,837/1,837 report titles | **wired** | - |
| A noise coverage < 90 % | 9 | gauge state withheld with the reason in plain words (DEV-INTAKE-02) | no real array below 90 % (lowest 48,127 of 48,528 sites); constructed array: release check E8 | **wired** | - |
| M EPIC v2 | 0 / 1 | refused by default; SeSAMe read behind `--dev-epic-v2` (DEV-EPIC-V2-01) | 72/72 v2 calibrated; identity sites median 5,391 of 6,000 (below 5,400, so no A by rule); v2 minus v1 at shared sites mean -0.034, SD 0.054; SeSAMe minus methylprep on v1 -0.026 | **flag**; v2 stays refused | v2 floor needs purified v2 neutrophils; v2 replicates; nonlinear dye-bias step |
| E detection | 1 | poobah against the Gaussian negative-control test, 55 strata (DEV-DETECTION-01) | poobah better in 31, Gaussian in 0, neither in 24 (poobah lower N but more arrays below 90 % identity coverage) | **poobah stays** | author: poobah takes 367 EPIC arrays below 90 % identity coverage (Gaussian 2) |
| Enlarged purified-neutrophil set | 5 / 8 | 19 healthy purified neutrophils lost at intake in round 1, now read and tared (DEV-INTAKE-02 check 6) | GSE167998 6/6 Normal; GSE118144 8/13; other laboratories 56/68 (round 1 42/49) | running (unchanged) | read them with self-tare II then median tare |
| G self-tare on type II fixed sites | 8 | array's own low and high fixed-site anchors map each probe design onto the reference scale, then median tare (DEV-SELFTARE-02) | replicate within-person SD 0.0164, 62/63 Normal; other laboratories 49/49; floor 6/6 | **flag** `--dev-selftare-ii` | author: make it the Stage T reading; physical control DNA next |
| H composition truth | 3 / 4 | another laboratory's mixtures; none on EPIC for adults, GSE77797 (450K) read (DEV-COMPOSITION-TRUTH-02) | atlas_e RMSE within 0.02 for 5 of 6 types, granulocytes 0.041; NILC-e granulocytes 0.061 | **flag** | wet-lab adult EPIC mixture set |
| I direction, physics only | 10 | signed move of each identity site toward or away from beta 0.5, tared, read against 2 x the reference spread (DEV-DIRECTION-02) | treated (decitabine, NTX-301) 12/12 toward disorder; replicates 56/63 no direction (bar 95 %); vehicle 4/6 | **flag** | GSE165185 (reported-only set) not read yet |
| J sky, block-shuffle null | 11 / 12 | within-chromosome block shuffle; look-elsewhere by simulation (DEV-SKY-02) | median power ratio band 1 1.84, bands 2-6 1.03-1.15 (bar 0.9-1.1); look-elsewhere 91 % (bar 8.4 %) | **flag** `--dev-sky` | the large-scale (band 1) residual is the open item |
| K 3b / 3c / 11b on own noise | 3b, 3c, 11b | per-site SD from the array's own noise (DEV-TOOLKIT-ADDED-02) | 3b: 82.5 % called at 5 % (bar 95 %), 0/360 unspiked; 3c: 100 % called at f = 0 (bar <= 5 %); 11b: 3.8 % of repeat pairs inside the interval (bar 95 %) | **flag** | 3c needs a same-run zero; IAM-A versions designed in the SOP, not built |
| C IAM-A C-score | 7 / Q | blocks of 1,000 sites in genomic order (DEV-IAMA-CSCORE-01) | constructed independent C 0.823 (limit +-0.80, 50 blocks); clustered 447.5; real whole files 604-1,047 | **wired** (printed, band not set) | band from same-person repeat files |
| IAM-A on real files | 7 / Q | three Loyfer granulocyte `.pat` files end to end (DEV-IAMA-REAL-01) | 3/3 end to end; 60 MB heads reproduce the floor file exactly; whole files 1.0394, 1.0632, 1.0344 (2/3 Normal); halves within 0.0005 | running | other laboratory's single-molecule files |
| D new-cell rule | 5 | three tests in the SOP, applied to monocytes and B cells (DEV-NEWCELL-01) | monocytes 11/13, SD 0.032; B 16/22, SD 0.037; neither meets the rule | out (B behind `--dev-percell-b`) | - |
| N development flags | all | one flag per stage, labelled, reading unchanged (DEV-FLAGS-01) | 63/63 readings identical with every flag on; every block labelled | **wired** | out list in SOP |

Release check (fresh public clone at `185f609` + the first four commits, `doors/` included): **17 of 17 checks PASS** (F1, F1b, S1-S4, E1-E10, M1), box commit `edc6ae6`; `kit/results/release_check.json`. Operations manual PDF rebuilt (`manual/MethylPhys_CPG_Operations_Manual.pdf`).

## Disclosed fixes and reruns
- Self-tare run 1 left the T cells out (title parser); fixed and rerun (job 9bcbcf99 is the run of record); run 1 kept as `run1_T_cells_left_out_*`.
- IAM-A real: first run read the hg38 files; the floor file is hg19; rerun on hg19 (hg38 kept as `hg38_build/`).
- 11b: NaN in the brightness draw fixed and rerun.
- EPIC v2: the SeSAMe dye-bias step QCDPB failed on the box (preprocessCore threads); the linear dye-bias step was used and is named in the note.
- EPIC v2 pairs were read below the 90 % rule; labelled development in the note.

## Not done
- EPIC v2 floor and v2 replicate test (no public purified v2 neutrophils; no v2 replicates).
- Single-molecule check on another laboratory's files.
- IAM-A versions of 3b, 3c and 11b: designed in the SOP, not built.
- Physical control DNA (wet lab) and an adult EPIC mixture set (wet lab).
- `sop/MethylPhys_CPG_SOP_v3_full.md`: a round-2 note added at the top; its line numbers are from the earlier build and were not re-proofed.
