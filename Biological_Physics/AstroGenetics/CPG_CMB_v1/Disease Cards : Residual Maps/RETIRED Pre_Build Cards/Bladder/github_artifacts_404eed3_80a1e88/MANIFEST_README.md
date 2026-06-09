# GitHub artifacts package — bladder-epic v0.1 sprint
**Generated:** 2026-05-01
**Source repository:** https://github.com/hmahaffeyges/IAM-Validation
**Commits captured:**
- `404eed3` (2026-05-01) — bladder-epic v0.1 sprint: 55 files (4 atlas vault BladderRef + 4 VAL directories + unified Phase C runner + post-pass + Biological_Physics/README.md update + atlas_vault/INVENTORY.json update)
- `80a1e88` (2026-05-01) — atlas_vault README backfill: 1 file (logs all four EpiSCORE per-tissue bridges: HeartRef + BreastRef + ProstateRef + BladderRef)

**Total: 56 files, 3.4 MB on disk**

---

## How to use this archive

The directory structure mirrors the repo exactly. To restore from this archive (e.g., if GitHub becomes inaccessible or the repo is corrupted):

```bash
# Option 1: drop on top of a fresh clone (or a worktree directory)
unzip github_artifacts_404eed3_80a1e88.zip
# This creates ./Biological_Physics/... matching the repo layout exactly.

# Option 2: restore into an existing local clone
cd /path/to/IAM-Validation
unzip -o /path/to/github_artifacts_404eed3_80a1e88.zip
# -o = overwrite existing files
```

Every file in this archive is byte-identical to what lives at commit `404eed3` or `80a1e88` on `origin/main`.

---

## What's in the archive — by directory

### Biological_Physics/README.md
Public per-sprint summary. Updated this round with the bladder-epic v0.1 paragraph.

### Biological_Physics/atlas_vault/
- `INVENTORY.json` — atlas vault inventory (90 → 94 entries; the four new entries are the BladderRef bridge script, bridged CSV, source Entrez matrix, and per-atlas README).
- `README.md` — atlas vault public README (242 lines after the `80a1e88` backfill; logs all four EpiSCORE per-tissue bridges with calibration anchors, statuses, atlas family fitness rule, mucosal-cohort rule).
- `stage2_cell_of_origin/episcore_bladderref/` — the four-file BladderRef atlas package:
  - `README.md` — per-atlas README (source citation, license, methodology, calibration anchor, SHA-256)
  - `bridge_bladderref_to_array.py` — bridge engineering script (Entrez gene IDs → 450K array CpGs via probeInfo450k.lv)
  - `episcore_bladderref_cpg_bridged.csv` — **the production-ready CpG-resolved atlas** (2,696 unique 450K CpGs × 4 cell types). Atlas SHA-256: `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`. This is what EDEAR Stage 2 loads.
  - `episcore_bladderref_entrez_matrix.csv` — original Zhu/Teschendorff Entrez-indexed matrix (163 EIDs × 4 cell types) for provenance.

### Biological_Physics/validation_runs/
- `unified_phaseC_runner.py` — single-pass Phase C runner that produces per-sample tables for VAL-120/121/122 simultaneously (270.7 sec for n=440 cohort).
- `postpass_amended.py` — post-pass that re-evaluates all three VALs against the amended CHK-3.1A floor.

### Biological_Physics/validation_runs/VAL-119_bladderref_calibrate/
EpiSCORE BladderRef Phase B calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210. Sealed `O1_BLADDERREF_CALIBRATION_SEALED`.
- `val119_bladderref_calibrate.py` — calibration script
- `prereg.md` + `PREREG_SEAL.txt` — pre-registration (sealed BEFORE β observed)
- `prereg_amendment.md` + `PREREG_AMENDMENT_SEAL.txt` — atlas NaN-serialization correction (β not yet observed at amendment seal)
- `outcome.md` — sealed outcome with full disclosure
- `VAL-119_calibration_results.json` — headline numbers (per-tile healthy floor distributions, q5/q95, sd, mean)
- `VAL-119_per_sample_calibration.csv` — per-sample A-scores for n=210 calibration cohort

### Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538/
Stage 1 Xu-538 immune red flag on TCGA-BLCA n=440. Sealed `O4_STAGE1_DATA_INTEGRITY_FAILURE` (panel cohort-substrate coverage gate fired). Diagnostic d_paired = +1.8977 (n=21, p=3.14×10⁻⁸) reported as diagnostic-not-sealed.
- `val120_bladder_stage1_xu538.py` — scoring script
- `prereg.md` + `PREREG_SEAL.txt` — pre-registration sealed BEFORE β
- `prereg_amendment_002.md` + `PREREG_AMENDMENT_002_SEAL.txt` — CHK-3.1A tissue-class floor amendment (β observed before amendment, full CCL-041 second-best disclosure)
- `outcome.md` — sealed outcome
- `EXECUTION_NOTE.md` — runtime notes
- `VAL-120_results.json` — headline numbers
- `VAL-120_stratified_results.json` — per-subgroup breakdowns
- `VAL-120_per_sample.csv` — per-sample A_immune scores (n=440)
- `VAL-120_paired_pairs.json` — n=21 paired tumor-vs-adjacent-normal pair structure
- `cohort_manifest.json` — TCGA-BLCA file_id list (use with GDC API)
- `clinical_metadata.json` — de-identified clinical metadata per sample

### Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas/
Stage 2 multi-atlas cell-of-origin scoring on TCGA-BLCA n=440. Sealed `O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS`. Loyfer Bladder POSITIVE +1.91 vs BladderRef Epi NEGATIVE −1.46 on same n=21 paired pairs (DISC-BLADDER-003 substrate-distribution mismatch).
- Same structure as VAL-120 plus:
- `VAL-121_per_sample_per_atlas.csv` — per-sample A-scores per atlas per tile
- `VAL_121_unified_per_sample.csv` — unified table containing all 73 tile A-scores (Loyfer 25 + BladderRef 4 + Caggiano 19 + Salas 6 + UniLIFE 19) plus QC fields. **This is the workhorse output of the unified Phase C runner.**
- `VAL-121_cross_tile_sanity.json` — CHK-3.2 cross-tile sanity check (the 14 Loyfer non-bladder solid-tissue tiles all firing POSITIVE +2.34 to +2.92, confirming substrate-distribution mismatch)

### Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune/
Stage 3 immune fine-tune on TCGA-BLCA n=440. Sealed `O1_STAGE_3_IMMUNE_DIFFERENTIATING`. All 6/6 Salas IDOL tiles fire POSITIVE; broad multi-lineage infiltration consistent with mixed TIL+TAM+MDSC of MIBC.
- Same structure as VAL-120/121.

---

## Reproducibility (CHK-7.6 reproducibility triple)

Every VAL directory has the four required reproducibility components:
1. **Inline source code** — the `valNNN_*.py` script
2. **Inputs** — `cohort_manifest.json` lists the GDC file_ids; the BladderRef atlas is at `atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv` with SHA-256 in `INVENTORY.json`
3. **Environment** — Python 3.12.3, numpy 2.4.4, pandas 2.x, scipy 1.17.1, pyreadr 0.5.6
4. **Expected headline output** — `VAL-NNN_results.json` is the bit-for-bit comparison target

Reproduction recipe:
```bash
# 1. Acquire TCGA-BLCA n=440 cohort
#    Use cohort_manifest.json file_ids to query GDC API:
#    https://api.gdc.cancer.gov/data/{file_id}

# 2. Acquire TCGA-KIRC + TCGA-PRAD calibration cohort (n=210)
#    Same VAL-106 cohort; manifest at VAL-106 (separate from this archive)

# 3. Run unified Phase C runner
python3 Biological_Physics/validation_runs/unified_phaseC_runner.py

# 4. Run post-pass against amended CHK-3.1A floor
python3 Biological_Physics/validation_runs/postpass_amended.py

# 5. Verify outputs against sealed results JSON
diff VAL-120_results.json Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538/VAL-120_results.json
diff VAL-121_results.json Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas/VAL-121_results.json
diff VAL-122_results.json Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune/VAL-122_results.json
```

---

## Sealed prereg + amendment SHA-256 chain

| VAL | Prereg SHA-256 | Amendment 002 SHA-256 |
|---|---|---|
| VAL-119 | `04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a` | `c3015ca3ba25f6c13f4f93fec85edea8506f64472657d03b59ed9ccda8355787` (atlas NaN serialization; β not yet observed at amendment seal) |
| VAL-120 | `6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996` | `93cd2171b131977f3bbd6e76d57df6cf291ae7d5ce2d297d5bd9bd656444c31d` (CHK-3.1A tissue-class floor amendment 002; β observed before amendment) |
| VAL-121 | `eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962` | `7f4b3148949060d6f0b8c27a5b55161c06a848d9b00d1e765ddcb182b3d0ec30` |
| VAL-122 | `2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855` | `db3f6563533ab625326acd42aab7a8028313a898bfec833c756f7be85f00df29` |

---

*End of manifest.*
