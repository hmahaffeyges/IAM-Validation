# Session Decision Propagation Tracker — 2026-06-08

**Purpose:** Every design decision made in this session (mock report build) that must propagate to runtime files, build specs, SOPs, canonicals, and the `walther_clinical.py` to be built. Organized by file with specific changes per file. Reference this when executing Phase 3 of `NEXT_CHAT_ACTION_LIST_CPG_v1_BUILD.md`.

**Session date:** 2026-06-08
**Mock report version that captures these decisions:** `CPG_MOCK_REPORT_60yo_subtle_drift_v1.md` (post-edit, IAMAtlas REBUILD removed + Section B expanded + MOCK NUMBERS warning added)

---

## 🛑 BLOCKING DECISION — MUST RESOLVE BEFORE PHASE 5 ORCHESTRATOR BUILD

### B-DEC-1: Age architecture — where in the chain does age-adjustment happen?

**This is a blocking decision. The next chat must NOT build `walther_clinical.py` Stage 6 (or any downstream stage that uses age-adjusted ranges) until Heath has chosen.**

**Why this is blocking:** The chain currently has two separate age-handling subsystems:
- `IAM_Cellular_Age/age_axis_foreground.py` + `IAMAtlas_age_layer.csv` (Stage 3 foreground subtraction — removes age axis from β at the genomic level, customer-age-geared, analogous to what other epigenetic-age companies do)
- `Age_Reference_Matrix_80_cells/age_reference_matrix.{json,csv,py}` (Stage 6 cellular age reference — per-class age-adjusted baselines, cell-geared)

These two systems handle age differently. The age architecture decision determines:
- WHERE in the chain age-adjustment happens
- WHETHER post-Stage-3 A-scores are already age-adjusted or still absolute
- HOW per-cell normal ranges in Section B are computed for the patient's chronological age
- WHETHER Section D produces an absolute cellular age (e.g., "62.2") or an age delta (e.g., "+2.2")
- WHETHER a 20-year-old and a 60-year-old reading the same raw β values get the same A-scores (alarming for the 20yo, typical for the 60yo) or different age-adjusted A-scores (both reading "normal-for-age" relative to their own baseline)

**Required reading sequence for the next chat — IN THIS ORDER, before making the recommendation:**

1. **Read every file in `Biological_Physics/atlas_vault/walther_clinical_runtime/Age_Reference_Matrix_80_cells/`:**
   - `age_reference_matrix.json`
   - `age_reference_matrix.csv`
   - `age_reference_matrix.py`
   - Any README in that folder
   - Goal: understand how the 80-cell reference matrix encodes age-adjusted baselines (8 classes × 10 decadal bins). This is the cell-geared aging subsystem.

2. **Read every file in `Biological_Physics/atlas_vault/walther_clinical_runtime/IAM_Cellular_Age/`:**
   - `age_axis_foreground.py` (the Stage 3 age-axis foreground subtraction)
   - `sex_axis_foreground.py`
   - `smoking_axis_foreground.py`
   - `IAMAtlas_age_layer.csv` (the per-CpG age fits)
   - `IAMAtlas_sex_layer.csv`
   - `IAMAtlas_smoking_layer.csv`
   - `age_layer_diagnostics.json` (per-CpG fit diagnostics)
   - `smoking_sex_layer_diagnostics.json`
   - `iam_cellular_age_scoring.py` (the Stage 6 cellular age scoring module)
   - Goal: understand the customer-age-geared subsystem — how the age axis is fit at the per-CpG level (analogous to Horvath/Hannum-style clocks) and how it's subtracted in Stage 3.

3. **Read `Biological_Physics/atlas_vault/walther_clinical_runtime/walther_clinical_BUILD_SPEC_v1_3.md` end-to-end** with specific attention to Stage 3, Stage 6, and the data flow between them.

4. **THEN make a recommendation to Heath** about which option to lock in:

   **Option A — Age removed early, cellular age = acceleration only:** Stage 3 removes age fully → Stage 4 A-scores are above/below typical-for-age → Stage 6 cellular age = how much above/below typical (an age-delta, not an absolute age). Pro: clean separation, Stage 3 does heavy lifting once. Con: the report can't show "biological age 64.8" — only "age delta +4.8".

   **Option B — Age preserved, cellular age = absolute:** Stage 3 does NOT remove age → Stage 4 A-scores are absolute departures from young-adult baseline → Stage 6 inverts the cumulative departure to estimate absolute biological age. Pro: report can show "biological age 64.8 vs chronological 60". Con: every per-cell normal range has to be re-anchored at every stage; competes with established clocks (Horvath/Hannum) on their own terms.

   **Option C — Hybrid (most defensible):** Stage 3 removes the chronological-time methylation drift (the universal aging signal that drives Horvath/Hannum clocks) but leaves the cell-type-specific aging signatures intact → Stage 4 A-scores read residual cell-type-specific departures → Stage 6 cellular age = absolute, computed by combining the chronological baseline + the post-Stage-3 cell-specific residuals. Pro: each stage has a clear job; both "biological age 64.8" AND "age delta +4.8" are computable; per-cell normal ranges in Section B can be computed at each patient's specific age; clear bridge between the two existing subsystems. Con: most complex; needs careful specification of which parts of the age signal Stage 3 removes vs preserves.

5. **The recommendation should be informed by what the existing files actually do** (not by what the mock report assumed). The mock report Section D used a placeholder methodology that may or may not match what the real architecture should be.

6. **Heath chooses.** Only after Heath chooses does the next chat proceed to Phase 3 surgical updates to `iam_cellular_age_scoring.py` and the BUILD_SPEC, and to Phase 5 orchestrator build.

**This decision must be made BEFORE:**
- Any edit to `iam_cellular_age_scoring.py` (File 2 below)
- Any decision on the age reference matrix scope (File 3 below — currently listed under "Options" but contingent on B-DEC-1)
- Any change to Section D methodology in BUILD_SPEC v1.3 (File 8 below)
- Any Section D content in the doctor report capability list v0.3 (File 10 below)
- The Stage 6 implementation in `walther_clinical.py` (File 12 below)

Effectively, B-DEC-1 is upstream of half the work in this tracker.

---

## DECISIONS MADE THIS SESSION (the canonical source of truth for the propagation work)

### D1. Tier breakpoints: 6-tier → 5-tier, breach at 1.10

**Decision:** Production locks to 5 tiers with breach at 1.10. The 6-tier `tier_breakpoints.json` (with SIG_ELEV [1.10, 1.12) and BREACH [1.10, ?) overlapping) is broken and must be replaced.

**Production tier scheme:**
- SUPPRESSED [< 0.95]
- NORMAL [0.95 – 1.04]
- ELEVATED [1.04 – 1.07]
- WARBURG_TRANSITION [1.07 – 1.10] — dashed line in visuals
- BREACH [≥ 1.10] — solid red line in visuals

**Pre-diagnostic active malignancy magnitude (≥ 1.20):** annotation/reference only, not its own tier.

### D2. Class-level A-score gauge is wrong — cell-level is the unit of analysis

**Decision:** The class-average A-score hides the signal (bidirectional cancellation across cells within a class). The unit of analysis is **individual cells**, not class averages. The report (and `walther_clinical.py` Stage 9 report builder) must lead with cell-level data; class-level summaries are reference only, clearly de-emphasized.

This matches what the disease matrix v1.7/v1.8 already does (123 cell-type columns, per-cell match-magnitude, sign-aligned, weighted by √n).

### D3. Cellular age = confidence-weighted absolute sum of per-cell departures

**Decision:** Cellular age is computed as:

```
Total_Cellular_Departure = Σ over all 115 cells [ |A_patient(cell) − A_ref(cell, chrono_age)| × (1 / posterior_SD(cell)) ]
```

Confidence-weighted so stable cells (tight posterior SD) dominate the sum. Absolute value avoids bidirectional cancellation. The total maps via Stage 6 calibration to a cellular age estimate. Per-class cellular ages remain in the report as a reference table only.

**Required input:** age-adjusted reference values per cell at the patient's chronological age. Current `age_reference_matrix.json` provides 80-cell baseline (8 classes × 10 decadal bins). This may need expansion to per-cell granularity (115 cells × age bins) — confirm scope before building.

### D4. Pattern Recognition section (new Section H.5) added to report

**Decision:** A new Pattern Recognition section unifies the visual pattern-recognition tools (cell ranking + Personal Brilliance Maps + Mahalanobis top contributions + disease matching) into named patterns. The report names patterns explicitly: Inflammaging signature, Age-related epithelial drift, Cross-class aging-associated tissue stress, etc.

Each named pattern cites the four visualization sources as converging evidence.

### D5. literature_anchors.json → v2.1 with cell-level searchability

**Decision:** Restructure `literature_anchors.json` from class-grouped (v2.0) to cell-level searchable (v2.1). Every anchor entry carries both `cell_type` and `parent_class` fields. The engine can look up by either. Pre-grouped class lists kept for back-compatibility with current Stage 9 engine.

### D6. Section B expanded with detailed per-cell composition + normal ranges + remarkability flags

**Decision:** Section B of the doctor report must list every detected cell (not just class totals like "18 of 29"), give the normal range per cell (age + sex + substrate adjusted), and explicitly flag any cell outside its normal range. Composition is its own diagnostic signal — a shedding tumor would surface here even before its methylation architecture shifts (Section C).

### D7. Naming: Cosmic Methylome Background + Personal Brilliance Map

**Decision (confirmed multiple times):**
- "Cosmic Methylome Background (CMB)" — never "Cosmic Methylome Background", never "C-Methylane-B"
- "Personal Brilliance Map" — never "Patient Brightness Map" or "personal cosmic methylome"
- Output filename: `{patient_id}_personal_brilliance_map.png` (NOT `_cosmic_methylome.png`)
- 8 per-class panels + 1 whole-atlas panel = 9 panels total

### D8. IAMAtlas REBUILD is internal-only — never customer-facing

**Decision (recurring failure pattern called out again this session):**
- Customer-facing: "CPG" (the product)
- Technical/research context: "IAMAtlas" (the instrument)
- "IAMAtlas REBUILD" appears ONLY in internal filenames like `IAMAtlasREBUILD.csv`
- Never in doctor reports, patient communications, customer-facing docs, or any output the customer sees

### D9. Stage 4.6 Brightness Comparison: file references confirmed

**Decision:** All required files for Stage 4.6 confirmed present in the repo:
- 4 plates at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/` (CPG_Plate_01 through 04 + README)
- HEALPix mapping at `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/`:
  - `generate_cpg_healpix_mapping.py`
  - `iamatlas_cpg_to_healpix_nside128.npy`
  - `iamatlas_cpg_to_healpix_nside128.provenance.json`
- `patient_brightness_comparison.py` script present
- 8 per-class brightness CSVs live in class_archives `.tar.xz` files

The "pending repo addition" note in the Brightness Comparison README is stale — files are present. README needs that note removed.

### D10. cfDNA + family history defaulted OFF for first patient

**Decision:** `walther_clinical.py` must import `cfdna_weight.json` and `family_history_multiplier.json` AND reference them in the chain, but the default switches are:
- `substrate = "whole_blood"` (cfDNA weights NOT loaded)
- `family_history = "not_provided"` (multipliers NOT applied)
- `mode = "first_patient_blind"` (full output, prepared for unblinding)

Files present + code path exists + defaults are off.

---

## RUNTIME FILE UPDATES REQUIRED (the actual work to be done in next chat)

### File 1: `Tier_breakpoints/tier_breakpoints.json`

**Action:** Replace 6-tier scheme with 5-tier scheme per D1.

| Tier | Range | Customer label | Engine label |
|---|---|---|---|
| SUPPRESSED | [0.0, 0.95) | Suppressed | SUPPRESSED |
| NORMAL | [0.95, 1.04) | Normal | NORMAL |
| ELEVATED | [1.04, 1.07) | Elevated | ELEVATED |
| WARBURG_TRANSITION | [1.07, 1.10) | Warburg Transition | WARBURG_TRANSITION |
| BREACH | [1.10, ∞) | Breach | BREACH |

Plus annotation block (not a tier): `pre_diagnostic_active_malignancy_magnitude_annotation: {threshold: 1.20, label: "Pre-diagnostic active malignancy magnitude", purpose: "Reference annotation for A-scores ≥ 1.20; not a tier"}`

Archive existing 6-tier to `OLD/tier_breakpoints_v1_2_6tier_BROKEN.json`. Tag new version as `v2.0` with date 2026-06-08 and supersession reason "6-tier scheme had SIG_ELEV [1.10, 1.12) and BREACH [1.10, ?) overlapping at 1.10; per Heath 2026-06-08, breach is at 1.10 and 5-tier scheme is canonical."

### File 2: `IAM_Cellular_Age/iam_cellular_age_scoring.py`

**Action:** Implement confidence-weighted total cellular departure methodology per D3.

New top-level function signature:
```python
def compute_total_cellular_departure(
    per_cell_A_scores: Dict[str, float],         # 115 cells
    per_cell_posterior_SD: Dict[str, float],     # 115 cells
    chronological_age: float,
    age_reference_matrix: AgeReferenceMatrix,    # provides A_ref(cell, age) per cell
) -> CellularAgeResult:
    """
    Total cellular departure = Σ over all 115 cells of:
        |A_patient(cell) − A_ref(cell, chrono_age)| × (1 / posterior_SD(cell))
    
    Returns: total_departure, cellular_age_estimate, per_class_age_breakdown,
             age_delta, inflammaging_quantum, dominant_driver_cells
    """
```

**Prerequisite:** `age_reference_matrix` must provide age-adjusted A_ref per cell type. Current matrix is 80-cell (8 classes × 10 decadal bins). May need expansion to 115-cell × age bins — confirm with Heath before scoping expansion.

Archive the old class-average-based `IAMCellularAge` class to `OLD/` with note about supersession.

### File 3: `Age_Reference_Matrix_80_cells/age_reference_matrix.{json,csv,py}`

**Action:** Confirm scope. Current is 80 entries (8 classes × 10 decadal bins, ages 4–95). The new cellular-age methodology may need per-cell age references (115 cells × ~9 bins ≈ 1,035 entries).

**Options for next chat to discuss with Heath:**
- (a) Keep 80-cell class-level reference + linearly distribute to each cell in class (simple, may lose precision)
- (b) Expand to 115-cell per-cell reference (more accurate, requires new fits from atlas source data)
- (c) Use existing 80-cell as fallback; per-cell where available, class-average otherwise (pragmatic, incremental)

**Default recommendation if Heath doesn't specify:** Option (c) — pragmatic, doesn't block first patient.

### File 4: `Brightness_Comparison/README_Brightness_Comparison.md`

**Action (surgical edits only — preserve all other content):**

| Find | Replace |
|---|---|
| "Cosmic Methylome Background" | "Cosmic Methylome Background (CMB)" |
| "personal Cosmic Methylome Background" | "Personal Brilliance Map" |
| Output filename: `{patient_id}_cosmic_methylome.png` | `{patient_id}_personal_brilliance_map.png` |
| "pending repo addition" note (HEALPix file) | Remove — file exists at `IAMAtlas_v0_1/healpix_mapping/iamatlas_cpg_to_healpix_nside128.npy` |

### File 5: `Brightness_Comparison/patient_brightness_comparison.py`

**Action:** Update output filename string from `_cosmic_methylome.png` to `_personal_brilliance_map.png`. Surgical edit. Test it doesn't break the rest of the module.

### File 6: `Literature_anchors_Report_building/literature_anchors.json`

**Action:** Build v2.1 from v2.0 DRAFT per D5. Add `cell_type` and `parent_class` fields to every anchor entry. Engine can look up by either. Keep `class_anchors` pre-grouped block for back-compatibility. Add new top-level `cell_anchors` block with anchors indexed by cell_type. Reuse all existing content from v2.0 DRAFT.

### File 7: `DISEASE_MATRIX/disease_cell_signature_matrix_v1_7.csv` → v1.8 swap

**Action:** Already pending. v1.8 file at `/mnt/user-data/outputs/DISEASE_MATRIX_v1_8/disease_cell_signature_matrix_v1_8.csv`. Copy to runtime, archive v1.7 to OLD/, regenerate or update `iamatlas_115_to_matrix_v1_8_mapping.json`, update folder README version log.

### File 8: `walther_clinical_BUILD_SPEC_v1_3.md`

**Action:** Edit to v1.3. Sections to update:

- **§5 Stage 6 (Cellular age):** rewrite per D3 — confidence-weighted total cellular departure across all 115 cells; per-class breakdown reference only
- **§5 Stage 7 (Tier breakpoints):** update to 5-tier scheme per D1; remove SIG_ELEV; add pre-diagnostic active malignancy annotation at 1.20
- **§5 Stage 9 (Report assembly):** expand to specify cell-level lead structure + new H.5 Pattern Recognition section per D4 + Section B detailed per-cell composition per D6 + Brilliance Map naming per D7
- **§14 Naming LOCKED block:** confirm CMB / Personal Brilliance Map (D7); confirm IAMAtlas REBUILD never customer-facing (D8); add 5-tier scheme to the locked terminology

### File 9: `CPG_Chain_of_Custody_SOP_v1_3.md`

**Action:** Edit to v1.4. Same updates as File 8 plus:

- Stage 7 SOP section updated to 5-tier
- Stage 6 SOP section updated to total cellular departure methodology
- Stage 4.6 SOP section: confirm naming uses CMB + Personal Brilliance Map
- Stage 8 SOP section: confirm Path B v1.8 disease matrix

### File 10: `DOCTOR_REPORT_CAPABILITY_LIST_v0_2.md` → v0.3

**Action:** Edit to v0.3:

- **Section C.3:** update tier ranges to 5-tier
- **Section D:** rewrite cellular age methodology per D3
- **Section F:** confirm CMB + Personal Brilliance Map naming per D7
- **NEW Section B detail:** per-cell composition tables with normal ranges (this session adds this)
- **NEW Section H.5:** Pattern Recognition section (this session adds this)
- **Section N:** confirm Mahalanobis hull v0_5 n=2,523 (not v0_4 n=2,481)
- **Section Q.3:** glossary updates for 5-tier + Cellular age methodology + Personal Brilliance Map definitions

### File 11: CPG canonicals (Heath-only IP — NEVER push to repo)

**Action:** Bump versions per the canonical update rule. Files in `cpg_canonicals/`:

- `v2_CPG_AI_Primer.md` (from v1) — incorporate 5-tier, cell-level analysis principle, total cellular departure methodology, Pattern Recognition concept, CMB + Personal Brilliance Map naming
- `v2_CPG_Pipeline_Walkthrough.md` (from v1) — same updates
- `v2_CPG_Pipeline.svg` (from v1) — visual diagram updates if structure changes
- `v2_CPG_Recipe.md` (from v1) — Stage 6 + Stage 7 methodology updates
- `v2_CPG_Roadmap.md` (from v1) — note v1.3 BUILD_SPEC and v1.4 SOP
- `v2_CPG_VAL_Test_Checklist.md` (from v1) — checks for the new methodology
- `v2_CPG_Lessons_Learned.md` (from v1) — append today's lessons (especially the IAMAtlas REBUILD recurrence + class-level vs cell-level discipline)
- `v2_CPG_Capability_Translator.md` (from v1) — confirm customer-facing language

**Old versions:** archive as `RETIRED_v1_CPG_*.md`. Do NOT delete.

### File 12: `walther_clinical.py` — NEW FILE, to be built in next chat

**Action:** Build per `walther_clinical_BUILD_SPEC_v1_3.md` (updated File 8 above) incorporating all decisions D1–D10.

Default config block at top of file:
```python
DEFAULT_CONFIG = {
    "substrate": "whole_blood",          # cfDNA weights NOT loaded by default
    "family_history": "not_provided",    # multipliers NOT applied by default
    "mode": "first_patient_blind",       # full output for unblinding
    "tier_scheme": "v2.0_5tier",         # 5-tier with BREACH at 1.10
    "cellular_age_methodology": "total_departure_v1",  # confidence-weighted absolute sum
    "report_lead": "cell_level",         # NOT class_level
    "naming": {
        "background": "Cosmic Methylome Background (CMB)",
        "patient_map": "Personal Brilliance Map",
        "atlas_internal": "IAMAtlas",    # NEVER "IAMAtlas REBUILD" in output
    },
}
```

Stage 9 report builder must:
- Lead Section B with per-cell composition tables + normal ranges + remarkability flags
- Lead Section C with cell-level departure ranking (top 15 of 115); class table is reference-only
- Section D use confidence-weighted total cellular departure
- New Section H.5 Pattern Recognition that identifies and names patterns
- Section F generate the 9-panel Personal Brilliance Map (8 per-class + 1 whole-atlas)
- Section Q glossary appendix
- Output: markdown + JSON + PNG (no PDF in v1; cellular performance gauge image is the v1.x+ PDF addition)

### File 13: `INVENTORY.md` (runtime folder)

**Action:** Refresh after all above updates. Document everything that has changed since 2026-06-02. Use same format as existing.

---

## VERIFICATION CHECKLIST (run after all updates)

- [ ] grep all customer-facing docs for "REBUILD" — should return zero hits
- [ ] grep all customer-facing docs for "Cosmic Methylome Background" — should return zero hits
- [ ] grep all customer-facing docs for "Patient Brightness Map" / "patient brightness" — should return zero hits
- [ ] grep `tier_breakpoints.json` for SIG_ELEV or SIGNIFICANTLY_ELEVATED — should return zero hits
- [ ] grep `tier_breakpoints.json` for BREACH — should appear once with range `[1.10, ∞)`
- [ ] verify `disease_cell_signature_matrix_v1_8.csv` is in runtime folder (not v1.7)
- [ ] verify `literature_anchors.json` is v2.1 with cell-level searchability
- [ ] verify `IAM_Cellular_Age/iam_cellular_age_scoring.py` has `compute_total_cellular_departure` function
- [ ] verify `Brightness_Comparison/patient_brightness_comparison.py` outputs `_personal_brilliance_map.png`
- [ ] verify `Brightness_Comparison/README` uses "Cosmic Methylome Background" + "Personal Brilliance Map" terminology throughout
- [ ] verify `walther_clinical_BUILD_SPEC` is v1.3 with all updates
- [ ] verify `CPG_Chain_of_Custody_SOP` is v1.4 with all updates
- [ ] verify `DOCTOR_REPORT_CAPABILITY_LIST` is v0.3 with all updates
- [ ] verify all 7 CPG canonicals bumped to v2
- [ ] verify `walther_clinical.py` exists with DEFAULT_CONFIG block matching D1–D10
- [ ] smoke test `walther_clinical.py` on a synthetic patient — all stages execute without exception
- [ ] before running on real first patient: verify INVENTORY.md reflects current state

---

## OPEN QUESTIONS FOR HEATH (carry into next chat)

1. **Age architecture (B-DEC-1):** the blocking decision at the top of this document. Resolves the age-reference-matrix scope question as a downstream consequence.
2. **First-patient GEO study:** which of the 5 candidates Walther suggested (mixed clinical cohort recommended)?
3. **Mock report v1 review:** any other section changes before v1 → v2 lock?
4. **Cellular Performance Gauge image for PDF v1.x+:** confirmed deferred to PDF version; markdown + JSON + PNG sufficient for v1?

---

*This tracker is the authoritative source of truth for what changed in the 2026-06-08 mock-report-build session. Reference it whenever propagating changes to the runtime or building `walther_clinical.py`. Once all File 1–13 updates are complete and the verification checklist passes, archive this tracker to `cpg_canonicals/RETIRED_session_trackers/` for audit.*
