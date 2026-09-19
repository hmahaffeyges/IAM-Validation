# RETIRED — kept for the record, not used by the current chain

| folder | what it was | why retired |
|---|---|---|
| `PreBuild/Phase1_PreBuild_Cards/` | the Phase-1 disease cards built before the Atlas (729 files) | superseded by the Atlas-based cards in `CPG_Engine/Disease Cards : Residual Maps/` |
| `PreBuild/chain_of_custody/`, `PreBuild/iamatlas_production_data/`, `PreBuild/scripts/`, `PreBuild/evidence/`, `PreBuild/validation_reports/` | the pre-build machinery and early evidence reports | the Atlas rebuild (2026-05-28) replaced them |
| `PreBuild/README_PreIAMAtlas_Build.md`, `README_Preliminary_Test_Results.md` | the READMEs of the folders that held all of this | the folders no longer exist; the VAL runs moved to `Testing_and_Code/` |
| `PostBuild_atlas_vault_snapshot_2026-06/` | the `atlas_vault` snapshot from the 2026-06-12 reorganization: an older deconvolver, runtime-matrix copies, SOP v1.3, the pre-fix marker file, and `IAMAtlas_v0_1/` whose `.csv.xz` is a 134-byte placeholder | the engine copy (2026-06-29, "one canonical copy") is canonical; the real atlas is `IAM_Atlas/`. The HEALPix mapping, external manifests, build scripts and plates were moved out of here to their live homes before retirement |
| `NILC_Deconvolver_cut_from_chain_2026-07-02/` | the cross-method NILC deconvolver | cut from Stage 2 by commit c1be0c3 ("Walther-alone per the flowchart") |

Nothing here should be imported by current code. If a path in an old document points into `PreIAMAtlas_Build/…` or `atlas_vault/…`, this is where it now resolves.
