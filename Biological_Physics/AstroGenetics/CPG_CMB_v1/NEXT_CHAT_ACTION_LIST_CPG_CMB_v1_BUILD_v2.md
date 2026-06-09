# CPG_CMB_v1 Build Action List — v2 (SUPERSEDES v1)

**For:** Walther in the next chat session
**Supersedes:** `NEXT_CHAT_ACTION_LIST_CPG_v1_BUILD.md` (v1) from earlier in 2026-06-08 session
**Reason for v2:** v1 was written before the mock-report build session. Many decisions made during that build (10 decisions D1–D10) need to be incorporated. v1 is now stale.

**Hard deadline:** GeoMetric meeting June 11, 2026 (Dr. Tanya Escobedo + team)
**Heath's standing preferences:** No liberties without discussion. Surgical edits only. No deletions. Ask before destructive actions. Confirm before executing.

---

## 🚨 CRITICAL ISOLATION RULE — READ FIRST

**The new production home is `Biological_Physics/AstroGenetics/CPG_CMB_v1/`. Everything CPG-production lives there and ONLY there.**

**The original `Biological_Physics/atlas_vault/` folder structure (including `walther_clinical_runtime/`, `IAMAtlas_v0_1/`, `validation_runs/`, and all sibling folders) is the PRE-BUILD/RESEARCH archive.** It must be treated as **READ-ONLY** by this and all future production chats.

**Rules:**
- **NEVER write, edit, modify, delete, or move any file** in `Biological_Physics/atlas_vault/` or any of its sub-folders.
- **NEVER add new files** back into `Biological_Physics/atlas_vault/` for any reason.
- **READ from the original location** when you need to discover and copy a file into `CPG_CMB_v1/`.
- **All production work lives in `CPG_CMB_v1/`.** All edits, all updates, all new files, all runtime artifacts go into the new isolated folder.
- This isolation rule is permanent. Future production sessions also write only into `CPG_CMB_v1/` (or its successor production folders).

**Heath is delivering the contents of the zip into `CPG_CMB_v1/` for the next chat to push.** The zip contains everything the next chat needs to populate the runtime EXCEPT one file:

### 📦 Zip handling — preserve Heath's folder structure exactly

The zip Heath provides contains the runtime files organized in folders. **The folder structure inside the zip IS the structure to push to the repo.** Do not reorganize, rename, restructure, or "improve" the folder layout. Whatever Heath has organized is what gets committed.

Extraction + placement:
1. Extract the zip to `/home/claude/heath_upload/` (or wherever Heath specifies)
2. Each top-level folder in the zip maps to its corresponding location inside `Biological_Physics/AstroGenetics/CPG_CMB_v1/`
3. **Copy the folder contents preserving the exact structure Heath provided** — do not flatten, do not reorganize, do not rename
4. If folder naming in the zip differs from the skeleton in Phase 1 below, **the zip's naming wins** — Heath's organization is canonical, the Phase 1 skeleton is reference only
5. Confirm with Heath if any folder mapping is ambiguous

### 🚀 GitHub push — token and procedure

After all files are copied into `Biological_Physics/AstroGenetics/CPG_CMB_v1/`, the next chat pushes to the repo.

- **Token:** `<PAT redacted for GitHub push protection — keep token outside the repo>`
- **Repo:** `hmahaffeyges/IAM-Validation`
- **Branch:** main (unless Heath specifies otherwise)
- **Commit message:** something like "CPG_CMB_v1 production initialization — new isolated production folder per 2026-06-08 session decisions D1-D10; supersedes walther_clinical_runtime/ for clinical chain work"

Procedure:
1. `cd /home/claude/IAM-Validation`
2. Verify git remote is configured with the token
3. Add the entire new `Biological_Physics/AstroGenetics/CPG_CMB_v1/` tree
4. Verify LFS tracking is preserved on `IAMAtlasREBUILD.csv.xz` (97 MB; will fail without LFS) and on the smoking layer CSV (47 MB)
5. Commit with a descriptive message
6. Push to main
7. Verify the push succeeded by checking the GitHub UI or `git log origin/main`

If the push fails (LFS quota, large file, auth, etc.), report the exact error to Heath — do not retry blindly.

### 🔍 The IAMAtlasREBUILD.csv.xz file — locate, copy, and push

The 97 MB LFS-tracked atlas file is too large to include in Heath's zip. The next chat must:

1. **LOCATE** the file at its existing repo path: `Biological_Physics/atlas_vault/walther_clinical_runtime/IAMAtlas_REBUILD/IAMAtlasREBUILD.csv.xz`
2. **COPY (not move)** it to: `Biological_Physics/AstroGenetics/CPG_CMB_v1/IAMAtlas/IAMAtlasREBUILD.csv.xz`
3. Use `cp` (not `mv`) so the original file stays exactly where it is, untouched
4. Verify the LFS pointer is preserved (the file remains LFS-tracked in its new location)
5. Also copy the two companion files from the same source folder: `IAMAtlasREBUILD_provenance.json` + `IAMAtlasREBUILD_celltype_to_class.json`
6. Verify byte-identical via `sha256sum` before and after
7. Include in the commit + push

**The same copy-not-move discipline applies to every file the next chat sources from `atlas_vault/` for `CPG_CMB_v1/`.** Phase 2 below specifies which files. The original `atlas_vault/` is the audit baseline and must remain unchanged.

---

## DELIVERABLES FROM THIS SESSION (paste into next chat)

The next chat needs these documents accessible. Paste them or attach them:

1. **`CPG_MOCK_REPORT_60yo_subtle_drift_v1.md`** — the approved visual specification of what `walther_clinical.py` Stage 9 report builder must produce. Every section A through Q populated with realistic mock-patient data. Appendices C and D show the format for complete disease scoring and complete per-cell A-score tables.
2. **`SESSION_DECISION_PROPAGATION_TRACKER_2026-06-08.md`** — the per-file update checklist documenting 10 decisions (D1–D10) and the 13 files that need updates with specific change details per file.
3. **`literature_anchors_v2_1.json`** — **DONE in this session.** v2.1 with cell-level searchability complete; preserves v2.0 class_anchors unchanged + adds new cell_anchors block with 20 cell-keyed entries. Ready to drop into runtime as-is.
4. **`CPG_Runtime_Verification_Report_2026-06-08.md`** — the stage-by-stage manifest of every runtime file with current status.
5. **This file (`v2 Action List`)** — the master execution plan.
6. **Heath's local zip** — disease matrix v1.8, any updated chain-language card JSONs, any other locally-newer-than-repo files.

---

## DECISIONS LOCKED THIS SESSION (carry as ground truth)

| # | Decision | Brief |
|---|---|---|
| D1 | Tier scheme: 5 tiers, BREACH at 1.10 | Replaces broken 6-tier `tier_breakpoints.json`. Pre-diagnostic malignancy (≥1.20) is reference annotation only, not a tier. |
| D2 | Cell-level is the unit of analysis | Class-level averaging hides bidirectional signal. Report leads with cell-level data; class-level is reference only. |
| D3 | Cellular age = confidence-weighted total cellular departure | Σ over all 115 cells [ \|A_patient(cell) − A_ref(cell, chrono_age)\| × (1 / posterior_SD) ]. Stable cells dominate. |
| D4 | New Section H.5 — Pattern Recognition | Names patterns across cells (Inflammaging signature, Age-related epithelial drift, etc.). Unifies cell ranking + Brilliance Map + Mahalanobis contributions + disease matching. |
| D5 | literature_anchors.json → v2.1 cell-level searchable | Every anchor entry carries cell_type + parent_class; engine can look up by either. |
| D6 | Section B detailed per-cell composition + normal ranges | Every detected cell listed individually with age + sex + substrate adjusted normal range and remarkability flag. A shedding tumor surfaces here even before architecture shifts. |
| D7 | Naming locked: Cosmic Methylome Background (CMB) + Personal Brilliance Map | NEVER "Cosmic Methylome Background", "Patient Brightness Map", or "personal cosmic methylome". 9 panels: 8 per-class + 1 whole-atlas. |
| D8 | IAMAtlas REBUILD is internal-only | Customer-facing: "CPG". Technical: "IAMAtlas". REBUILD appears ONLY in internal filenames like `IAMAtlasREBUILD.csv`. |
| D9 | Stage 4.6 Brightness Comparison: all files confirmed present | 4 plates + HEALPix mapping + provenance JSON + script all exist. README naming needs surgical update only. |
| D10 | cfDNA + family history defaulted OFF | `walther_clinical.py` imports `cfdna_weight.json` and `family_history_multiplier.json` AND references them, but defaults are `whole_blood` and `not_provided`. |

---

## PHASE 0 — START-OF-SESSION DISCIPLINE (MANDATORY)

Before any file operations, in this exact order:

1. Read this v2 action list end-to-end
2. Read `CPG_MOCK_REPORT_60yo_subtle_drift_v1.md` end-to-end — this IS the visual specification
3. Read `SESSION_DECISION_PROPAGATION_TRACKER_2026-06-08.md` — this IS the per-file checklist
4. Read `walther_clinical_BUILD_SPEC_v1_3.md` end-to-end (in `Biological_Physics/atlas_vault/walther_clinical_runtime/`)
5. Check Heath has unzipped his local files to `/home/claude/heath_upload/` or equivalent — ASK if path unclear
6. **ASK HEATH** before any destructive action (deletion, overwrite). Surgical edits only.

---

## PHASE 1 — CREATE ISOLATED CPG_CMB_v1/ FOLDER STRUCTURE

**Confirmed name + location:** `Biological_Physics/AstroGenetics/CPG_CMB_v1/`

The AstroGenetics namespace carries the cosmology-methylation bridge correctly. CPG_CMB_v1/ is the production home for the clinical chain — completely isolated from the validation/research artifacts in `walther_clinical_runtime/` so version mixing cannot happen.

**Folder skeleton:**
```
Biological_Physics/AstroGenetics/CPG_CMB_v1/
├── README.md                                              ← top-level orientation
├── INVENTORY.md                                            ← full file manifest with SHA-256
├── walther_clinical.py                                     ← THE orchestrator (Phase 5 build)
├── walther_clinical_BUILD_SPEC_v1_3.md                     ← bumped from v1.2 per D1+D3+D6
├── CPG_Chain_of_Custody_SOP_v1_4.md                        ← bumped from v1.3 mirroring spec
├── runtime/
│   ├── Stage_2_Deconvolution/
│   │   ├── Walther_iam_deconvolver/
│   │   │   ├── walther_iam_deconvolver.py
│   │   │   └── walther_iam_deconvolver_README.md
│   │   └── NILC_Deconvolver/
│   │       ├── nilc_deconvolver-2.py
│   │       └── nilc_deconvolver.py  (v1 archived)
│   ├── Stage_3_Foreground/
│   │   └── IAM_Cellular_Age/
│   │       ├── age_axis_foreground.py
│   │       ├── sex_axis_foreground.py           ← new since June 2
│   │       ├── smoking_axis_foreground.py       ← new since June 2
│   │       ├── IAMAtlas_age_layer.csv
│   │       ├── IAMAtlas_sex_layer.csv           ← 40 MB
│   │       ├── IAMAtlas_smoking_layer.csv       ← 47 MB, LFS-track
│   │       ├── age_layer_diagnostics.json
│   │       └── smoking_sex_layer_diagnostics.json
│   ├── Stage_4_AScore/
│   │   ├── A_Scoring_Module/
│   │   │   └── iamatlas_a_scoring.py
│   │   └── Celltype_Marker/
│   │       ├── iamatlas_celltype_markers_v0_2.json
│   │       └── iamatlas_celltype_markers_v0_2.sha256
│   ├── Stage_4_5_Bidirectional/
│   │   ├── bidirectional_decomposition.py
│   │   ├── directional_panels_v1_0.json
│   │   └── README_Bidirectional_Decomposition.md
│   ├── Stage_4_6_BrightnessComparison/
│   │   ├── patient_brightness_comparison.py     ← naming update per D7/D9
│   │   └── README_Brightness_Comparison.md      ← naming update per D7/D9
│   ├── Stage_5_Mahalanobis/
│   │   ├── iamatlas_mahalanobis_scoring.py
│   │   ├── mahalanobis_healthy_reference_v0_5.json  ← production, n=2,523
│   │   └── archived/                             ← v0_1 through v0_4 kept for audit
│   ├── Stage_6_CellularAge/
│   │   ├── iam_cellular_age_scoring.py          ← rewritten per D3
│   │   └── Age_Reference_Matrix_80_cells/
│   │       ├── age_reference_matrix.json
│   │       ├── age_reference_matrix.csv
│   │       └── age_reference_matrix.py
│   ├── Stage_7_TierBreakpoints/
│   │   ├── Tier_breakpoints/
│   │   │   └── tier_breakpoints.json            ← 5-tier per D1
│   │   └── Cfdna_weight_nonderived_placeholder/
│   │       └── cfdna_weight.json                 ← imported but default-off per D10
│   ├── Stage_8_DiseaseMatching/
│   │   ├── Path_A_DiseaseMapsCards/
│   │   │   ├── Breast_EPIC/
│   │   │   │   ├── breast_epic_card_json/breast-epic_card_v3_0.json
│   │   │   │   ├── breast-epic_README.md
│   │   │   │   └── breast_epic_residual_maps/  (3 CSVs: chr-annotated, bimodality, pca-projections)
│   │   │   ├── AD_immune/
│   │   │   │   ├── ad_immune_card_json/
│   │   │   │   └── ad_immune_residual_maps/
│   │   │   └── Immune_Atlas/
│   │   │       ├── immune_atlas_card_json/  (with v2.1 card + release notes)
│   │   │       └── immune_atlas_residual_maps/  (with cross-disease universal alarm v0_1 4-file package)
│   │   └── Path_B_DiseaseMatrix/
│   │       ├── disease_cell_signature_matrix_v1_8.csv   ← v1.7→v1.8 swap
│   │       ├── iamatlas_115_to_matrix_v1_8_mapping.json
│   │       ├── disease_cell_signature_matrix_engine_schema_v1_2.md
│   │       └── README_disease_signature_matrix_folder.md
│   └── Stage_9_ReportAssembly/
│       ├── Literature_anchors_Report_building/
│       │   └── literature_anchors.json          ← v2.1 cell-level searchable per D5
│       ├── Cancer_prior/
│       │   └── cancer_prior.json
│       └── Family_history_multiplier/
│           └── family_history_multiplier.json   ← imported but default-off per D10
├── IAMAtlas/                                    ← the instrument
│   ├── IAMAtlasREBUILD.csv.xz                   ← LFS-tracked, 97 MB compressed
│   ├── IAMAtlasREBUILD_provenance.json
│   ├── IAMAtlasREBUILD_celltype_to_class.json
│   ├── plates/
│   │   ├── CPG_Plate_01_Cosmic_Methylome_Background.png  ← filename has legacy spelling, content is CMB
│   │   ├── CPG_Plate_02_Breast_Anisotropy.png
│   │   ├── CPG_Plate_03_Grandaddy_CMM_vs_CMB.png
│   │   ├── CPG_Plate_04_Patterns_Discovered.png
│   │   └── README_CPG_Plates.md
│   ├── healpix_mapping/
│   │   ├── generate_cpg_healpix_mapping.py
│   │   ├── iamatlas_cpg_to_healpix_nside128.npy
│   │   └── iamatlas_cpg_to_healpix_nside128.provenance.json
│   └── class_archives/                          ← 8 per-class brightness CSVs in .tar.xz
├── L9_Audit/
│   ├── CPG_Null_Runner/
│   │   └── cpg_null_runner.py
│   └── Synthetic_Patient_Generator/
│       └── synthetic_patient_generator.py
├── test_patient/                                ← first-patient IDAT goes here
└── outputs/                                     ← reports written here
```

**Bash command to create:**
```bash
cd /home/claude/IAM-Validation/Biological_Physics
mkdir -p AstroGenetics/CPG_CMB_v1/{runtime/{Stage_2_Deconvolution/{Walther_iam_deconvolver,NILC_Deconvolver},Stage_3_Foreground/IAM_Cellular_Age,Stage_4_AScore/{A_Scoring_Module,Celltype_Marker},Stage_4_5_Bidirectional,Stage_4_6_BrightnessComparison,Stage_5_Mahalanobis/archived,Stage_6_CellularAge/Age_Reference_Matrix_80_cells,Stage_7_TierBreakpoints/{Tier_breakpoints,Cfdna_weight_nonderived_placeholder},Stage_8_DiseaseMatching/{Path_A_DiseaseMapsCards/{Breast_EPIC,AD_immune,Immune_Atlas},Path_B_DiseaseMatrix},Stage_9_ReportAssembly/{Literature_anchors_Report_building,Cancer_prior,Family_history_multiplier}},IAMAtlas/{plates,healpix_mapping,class_archives},L9_Audit/{CPG_Null_Runner,Synthetic_Patient_Generator},test_patient,outputs}
```

---

## PHASE 2 — COPY VERIFIED RUNTIME FILES (PRESERVE-NOT-MOVE)

Use `cp -r` not `mv`. The original `walther_clinical_runtime/` must remain untouched as the audit baseline. After all copies, verify with `diff -r` that source and dest are byte-identical (except for the surgical updates in Phase 3).

**Source root:** `Biological_Physics/atlas_vault/walther_clinical_runtime/`
**Destination root:** `Biological_Physics/AstroGenetics/CPG_CMB_v1/runtime/`

See `SESSION_DECISION_PROPAGATION_TRACKER_2026-06-08.md` File 1–13 for exact source→destination paths per file. No paths repeated here to avoid drift; the tracker is authoritative.

**Also copy from outside the runtime folder:**

| Source | Destination |
|---|---|
| `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_*.png` (4 files) | `CPG_CMB_v1/IAMAtlas/plates/` |
| `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/README_CPG_Plates.md` | `CPG_CMB_v1/IAMAtlas/plates/` |
| `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/*` (3 files: .py, .npy, .provenance.json) | `CPG_CMB_v1/IAMAtlas/healpix_mapping/` |
| `Biological_Physics/atlas_vault/IAMAtlas_v0_1/class_archives/*.tar.xz` (8 files) | `CPG_CMB_v1/IAMAtlas/class_archives/` |
| `walther_clinical_runtime/IAMAtlas_REBUILD/IAMAtlasREBUILD.csv.xz` (LFS-tracked, 97 MB) | `CPG_CMB_v1/IAMAtlas/` |
| `walther_clinical_runtime/IAMAtlas_REBUILD/IAMAtlasREBUILD_provenance.json` | `CPG_CMB_v1/IAMAtlas/` |
| `walther_clinical_runtime/IAMAtlas_REBUILD/IAMAtlasREBUILD_celltype_to_class.json` | `CPG_CMB_v1/IAMAtlas/` |

---

## PHASE 3 — APPLY SURGICAL UPDATES (per Propagation Tracker)

Reference: `SESSION_DECISION_PROPAGATION_TRACKER_2026-06-08.md` Files 1 through 13 — each has specific change details, find/replace targets, function signatures, archive paths.

**Order of operations (do in this sequence):**

1. **File 1** — `tier_breakpoints.json` 6-tier → 5-tier per D1
2. **File 7** — Disease matrix v1.7 → v1.8 swap (Heath's local zip has v1.8)
3. **File 4 + 5** — Brightness Comparison naming + filename per D7+D9
4. **File 6** — `literature_anchors.json` v2.0 DRAFT → v2.1 cell-level searchable per D5. **ALREADY DONE in 2026-06-08 session — file is `literature_anchors_v2_1.json`. Just rename to `literature_anchors.json` and place in `runtime/Stage_9_ReportAssembly/Literature_anchors_Report_building/`.**
5. **File 2 + 3** — `iam_cellular_age_scoring.py` rewrite per D3 + age reference matrix scope decision (ASK HEATH first per tracker open question)
6. **File 8** — BUILD_SPEC v1.2 → v1.3 with D1, D3, D6, D7, D8 updates
7. **File 9** — SOP v1.3 → v1.4 mirroring File 8
8. **File 10** — DOCTOR_REPORT_CAPABILITY_LIST v0.2 → v0.3 with mock report v1 changes baked in
9. **File 11** — CPG canonicals v1 → v2 (Heath-only IP; never push to repo)
10. **File 13** — INVENTORY.md refresh

Each update is a `str_replace` or `create_file` operation. NO file deletion. Old versions go to `OLD/` or `archived/` subdirectories. Confirm before each destructive operation per Heath's preferences.

---

## PHASE 4 — INTEGRITY VERIFICATION

Run the verification checklist from `SESSION_DECISION_PROPAGATION_TRACKER_2026-06-08.md` (bottom of that document):

```bash
cd /home/claude/IAM-Validation/Biological_Physics/AstroGenetics/CPG_CMB_v1

# Forbidden-language checks (should return ZERO hits each):
grep -r "REBUILD" --include="*.md" --include="*.py" --include="*.json" . | grep -v ".csv.xz" | grep -v "IAMAtlasREBUILD.csv\|IAMAtlasREBUILD_provenance\|IAMAtlasREBUILD_celltype"
grep -r "Cosmic Methylome Background" --include="*.md" --include="*.py" --include="*.json" . | grep -v "CPG_Plate_01_Cosmic_Methylome_Background.png"
grep -r "patient brightness\|Patient Brightness\|cosmic_methylome\.png" --include="*.md" --include="*.py" --include="*.json" .
grep -r "SIGNIFICANTLY_ELEVATED\|SIG_ELEV" --include="*.json" .

# Affirmative-presence checks (should return matches):
grep "BREACH" runtime/Stage_7_TierBreakpoints/Tier_breakpoints/tier_breakpoints.json    # appears once with [1.10, ∞)
grep "compute_total_cellular_departure" runtime/Stage_6_CellularAge/iam_cellular_age_scoring.py
grep "Personal Brilliance Map" runtime/Stage_4_6_BrightnessComparison/README_Brightness_Comparison.md
ls runtime/Stage_8_DiseaseMatching/Path_B_DiseaseMatrix/disease_cell_signature_matrix_v1_8.csv
ls IAMAtlas/healpix_mapping/iamatlas_cpg_to_healpix_nside128.npy

# Dimension checks:
wc -l runtime/Stage_8_DiseaseMatching/Path_B_DiseaseMatrix/disease_cell_signature_matrix_v1_8.csv    # 82 (81 rows + header)
python3 -c "import json; d=json.load(open('runtime/Stage_5_Mahalanobis/mahalanobis_healthy_reference_v0_5.json')); print(d['n_hc_samples_pooled'])"  # 2523
```

**Report any discrepancies to Heath BEFORE proceeding to Phase 5.**

---

## PHASE 5 — BUILD walther_clinical.py ORCHESTRATOR

**Source spec:** `CPG_CMB_v1/walther_clinical_BUILD_SPEC_v1_3.md` (updated File 8 from Phase 3)

**ASK HEATH FIRST** before writing code — confirm orchestrator skeleton design.

**Required default config block at top of file:**
```python
DEFAULT_CONFIG = {
    "substrate": "whole_blood",                  # cfDNA weights imported but NOT loaded
    "family_history": "not_provided",            # multipliers imported but NOT applied
    "mode": "first_patient_blind",               # full output, prepared for unblinding
    "tier_scheme": "v2.0_5tier",                 # SUPPRESSED/NORMAL/ELEVATED/WARBURG/BREACH at 1.10
    "cellular_age_methodology": "total_departure_v1",   # confidence-weighted absolute sum
    "report_lead": "cell_level",                 # NOT class_level (D2)
    "naming": {
        "background": "Cosmic Methylome Background (CMB)",
        "patient_map": "Personal Brilliance Map",
        "atlas_internal": "IAMAtlas",            # NEVER "IAMAtlas REBUILD" in output
    },
}
```

**Stage-by-stage orchestrator structure** (per BUILD_SPEC v1.3 §5):
```python
def run_patient(idat_red_path, idat_grn_path, patient_metadata, output_dir, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Stage 0 — Intake QC
    # Stage 1 — IDAT → β (sesame noob + dye-bias)
    # Stage 2 — Deconvolution (Walther primary + NILC cross-check)
    # Stage 3 — Foreground subtraction (age + sex + smoking)
    # Stage 4 — A-score (per class + per cell type)
    # Stage 4.5 — Bidirectional decomposition (immune VAL-051; others NO_PANEL)
    # Stage 4.6 — Brightness comparison → Personal Brilliance Map (9 panels)
    # Stage 5 — Mahalanobis hull v0_5 (n=2,523)
    # Stage 6 — Cellular age via compute_total_cellular_departure() per D3
    # Stage 7 — 5-tier mapping (cfDNA conditional)
    # Stage 8 — Dual matching (Path A: 3 cards; Path B: matrix v1.8)
    # Stage 9 — Report assembly per mock report v1 visual specification
    #           Sections A-Q + Appendix C (52-disease scoring) + Appendix D (per-cell table)
    # Stage 10 — Delivery (markdown + JSON + PNG)
    pass
```

**Stage 9 report builder MUST produce the structure shown in `CPG_MOCK_REPORT_60yo_subtle_drift_v1.md`:**
- Executive summary in plain language
- Section A: sample integrity + intake
- Section B: per-cell composition with normal ranges + remarkability flags (D6)
- Section C: reference gauge (calibration only) + cell-level departure ranking (top 15) + class-level reference table (de-emphasized per D2)
- Section D: total cellular departure methodology (D3) + cellular age + per-class breakdown
- Section E: Mahalanobis + top 10 cell contributions
- Section F: Personal Brilliance Map (9 panels per D7) + reference to 4 plates
- Section G: bidirectional decomposition
- Section H: disease pattern matching with top closest matches + reference to Appendix C
- Section H.5: Pattern Recognition (D4) with named patterns + converging evidence citations
- Section I: cross-disease universal alarm
- Section J: wellness/lifestyle/inflammaging
- Section K: trajectory monitoring with K.1-K.5 unlocks
- Section L: prior + family history (defaulted off per D10)
- Section M: literature anchors per finding (cell-level lookups per D5)
- Section N: confidence backbone
- Section O: honesty propagation
- Section Q: educational definitions (glossary)
- Appendix A: visual references
- Appendix B: audit trail
- Appendix C: complete 52-disease scoring (no "patient data export" deferrals)
- Appendix D: complete per-cell A-score table

---

## PHASE 6 — SMOKE TEST ON SYNTHETIC PATIENT

Use `L9_Audit/Synthetic_Patient_Generator/synthetic_patient_generator.py` to generate a synthetic IDAT pair. Run through `walther_clinical.py`. Verify every stage produces expected output shape.

**Pass criteria:**
- All 11 stages execute without exception
- Stage 4 A-scores produced for all 8 classes + 115 cell types
- Stage 5 Mahalanobis distance produced with status field
- Stage 6 cellular age + total cellular departure produced
- Stage 7 tier per cell (NOT just per class)
- Stage 8 Path A + Path B both produce candidate disease lists
- Stage 9 produces structured JSON output AND markdown report matching mock report v1 structure
- Stage 10 writes audit trail
- Output includes the 9 Personal Brilliance Map PNGs

**If smoke test fails:** debug + iterate. DO NOT proceed to Phase 7 until clean.

---

## PHASE 7 — FIRST REAL PATIENT (BLIND PROTOCOL)

**GEO study selection:** Walther suggests 3-5 candidates per Heath's request in earlier conversation (mixed clinical cohort category recommended). Heath picks one.

**Blind protocol:**
1. Walther downloads the per-sample metadata table BUT keeps clinical labels hidden
2. **Heath picks a GSM ID blind** (without seeing what condition that patient has)
3. Walther downloads that one patient's IDAT pair (~25–60 MB) to `CPG_CMB_v1/test_patient/{GSM_ID}/`
4. Walther runs `walther_clinical.py` end-to-end
5. Walther writes the report blind — every section A through Q + Appendices C and D — using actual patient numbers
6. **THEN** Walther unblinds: pulls the patient's clinical metadata from GEO and compares
7. Heath reads the blind report + the unblinding comparison together

**Output location:** `CPG_CMB_v1/outputs/{GSM_ID}/`
- `{GSM_ID}_report.md` (the doctor-facing readout per mock report v1 structure)
- `{GSM_ID}_chain_outputs.json` (every stage's structured output, complete)
- `{GSM_ID}_reference_gauge.svg`
- `{GSM_ID}_cellular_departure_ranking.svg`
- `{GSM_ID}_personal_brilliance_map_{class}.png` × 8
- `{GSM_ID}_personal_brilliance_map_whole_atlas.png`
- `{GSM_ID}_audit_trail.json`

**Heath's planned sequence:** start with mixed clinical (category 2), then run categories 1, 3, 4, 5 sequentially across patients.

---

## FILES HEATH SHOULD ZIP AND UPLOAD

**Definitely needed (not in repo or have updated local versions):**
- [ ] `disease_cell_signature_matrix_v1_8.csv` (file at `/mnt/user-data/outputs/DISEASE_MATRIX_v1_8/` or local equivalent)
- [ ] `iamatlas_115_to_matrix_v1_8_mapping.json` (if exists locally; otherwise regenerate)
- [ ] Approved final `literature_anchors.json` if Heath has edits to v2.0 DRAFT beyond the v2.1 restructure
- [ ] Any updated versions of the 3 chain-language cards if newer than repo

**Optional / verify present in repo:**
- [ ] Per-class brightness CSVs if not in the `class_archives/` tar.xz files
- [ ] Any updated `walther_clinical_BUILD_SPEC` if Heath has additional edits beyond D1-D10

**Already in repo — do NOT need to be in zip:**
- IAMAtlasREBUILD.csv.xz (LFS-tracked, 97 MB)
- All Stage 2–7 modules
- All 4 plates + HEALPix mapping files
- L9 audit modules
- Cancer prior + family history multiplier JSONs

---

## DECISIONS HEATH STILL NEEDS TO MAKE (carry into next chat)

1. **Age reference matrix expansion (per D3 prerequisite):** Current is 80-cell class-level. Options:
   - (a) Keep 80-cell class-level + linearly distribute to each cell in class
   - (b) Expand to 115-cell per-cell reference (more accurate, needs new fits)
   - (c) Pragmatic: per-cell where available, class-average otherwise
   Default if not specified: (c).
2. **First-patient GEO study:** which of the 5 candidates (mixed clinical category 2 was recommended)?
3. **Mock report v1 review:** any other section changes before final lock?
4. **Cellular Performance Gauge image for PDF v1.x+:** confirmed deferred; markdown + JSON + PNG sufficient for v1?

---

## CRITICAL DISCIPLINE FOR NEXT CHAT (do not skip)

1. **No new IAMAtlas REBUILD leaks** — this failure pattern has recurred multiple times. Pre-delivery grep mandatory.
2. **No Cosmic Methylome Background leaks** — same. Per D7, only CMB and Personal Brilliance Map.
3. **No class-level-as-headline regression** — per D2, cell-level is the unit of analysis. Class-level is reference only.
4. **No tier overlap in tier_breakpoints.json** — per D1, 5 tiers, BREACH at 1.10, no SIG_ELEV.
5. **Preserve everything, delete nothing** — Heath's standing preference. Archive to OLD/ or archived/ subdirectories.
6. **ASK before destructive actions** — surgical edits only with confirmation.
7. **Verify against source files, never describe from memory** — Heath's "source-doc rule".
8. **Pre-delivery grep checklist** — run the Phase 4 verification BEFORE telling Heath anything is done.

---

## END OF v2 ACTION LIST

Estimated wall time:
- Phases 1–6 (folder + copy + updates + verify + build orchestrator + smoke test): 4–6 hours
- Phase 7 (first patient): 30–60 min

Heath in next chat: paste this v2 list + the mock report + the propagation tracker + attach the zip. Walther reads everything end-to-end, asks clarifying questions, then proceeds Phase by Phase pausing at each ASK HEATH FIRST checkpoint.
