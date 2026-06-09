# VAL-12X+1 (provisional ID) — Phase C Run-Everything on TCGA-STAD (Gastric Adenocarcinoma)

**Prereg version:** v1.0-DRAFT (awaiting Heath sign-off + VAL-12X seal before this seal)
**Date drafted:** 2026-05-02
**Card:** gastric+esophageal-epic v0.1 sprint, Phase C gastric arm
**Prereg type:** Phase C disease cohort scoring (run-everything regime)
**Depends on:** VAL-12X (BoccellatoStomachRef calibration) sealed first

---

## 1. VAL identification

- Provisional VAL ID: **VAL-12X+1** (sequential)
- Cohort: **TCGA-STAD** (Stomach Adenocarcinoma, n≈395 tumor + n≈27 paired/unpaired adjacent-normal HM450 sesame Level 3 + HM27 mix)
- Comparison strategy: **Welch tumor-vs-pooled-normals** (paired-d structurally underpowered at n=2 paired HM450; documented limitation)
- Stratifications mandatory per CCL-025 + TCGA-STAD molecular subtype literature: **H. pylori serology**, **EBV status**, **MSI status**, **CIN/GS subtype**, **Lauren classification (intestinal/diffuse)**, **sex** per CCL-002

## 2. Hypothesis (pre-locked, BIDIRECTIONAL per CHK-2.7 + Heath reminder)

**Stage 1 hypothesis (Xu-538 architectural drift):** TCGA-STAD tumor samples show **|Cohen's d|** ≥ 0.5 vs pooled normals on Stage 1 Xu-538 pooled-entropy A-score. Direction expected POSITIVE (tumor > healthy) based on cycling-class precedent (TCGA-COAD VAL-062 d=+0.7241 paired; TCGA-LIHC non-viral VAL-064 d=+0.664 paired secretory-class). Null hypothesis: |d| < 0.2.

**Stage 2 hypothesis (cell-of-origin tile pattern):** under run-everything regime, all available Stage 2 atlases (Boccellato + Loyfer 25-tile + Caggiano TIM) score on every IDAT. Per CHK-2.7 + CCL-039, expected pattern in tumor-vs-adjacent-normal-paired comparisons:
- Cell-of-origin tile (gastric/Upper_GI/stomach) reads NEGATIVE direction (de-differentiation degrades cell-of-origin tile fidelity)
- Other tissue tiles (Bladder, Hepatocytes, Lung_cells, Pancreatic_beta) read POSITIVE direction (homogenization toward generic-tumor methylation)

**Stage 3 hypothesis (immune sub-composition):** Salas IDOL + UniLIFE 19-cell head-to-head; per CCL-021 + heme-LL-005 differential, expect Stage 3 lineage-specific shifts in tumor vs control (T-cell exhaustion signature, NK shift, regulatory T-cell expansion). Magnitude direction not pre-locked (BIDIRECTIONAL).

**Bidirectionality declaration:** Per Heath's reminder (always assume bidirectional behavior in tissue and immune response), all outcome thresholds use magnitude-based |d| with explicit direction labels. No outcome pre-locks "positive d" or "negative d" alone.

## 3. Pre-locked decision criteria (CHK-2.1 + CHK-2.7 + CHK-4.11)

### Stage 1 outcomes

- **O1_STAGE1_PASS**: Xu-538 |d_unpaired| ≥ 0.5, lower CI bound bound away from 0 in the observed-d direction. Direction label STAGE1_POSITIVE if d>0, STAGE1_NEGATIVE if d<0.
- **O2_STAGE1_PARTIAL**: 0.2 ≤ |d_unpaired| < 0.5
- **O3_STAGE1_NULL**: |d_unpaired| < 0.2
- **O5_STAGE1_NEGATIVE_UNEXPECTED**: |d_unpaired| ≥ 0.5 in the unexpected direction (i.e. d_negative if positive expected, vice versa)

### Stage 2 outcomes (per atlas, per tile)

For each Stage 2 atlas tile, the |d_unpaired| Welch effect size is computed and one of these outcome labels assigned:
- `{tile}_DIFFERENTIATING_POSITIVE`: |d| ≥ 0.5, d > 0 (tumor methylation drifts away from healthy at this tile, e.g. Bladder/Lung tiles)
- `{tile}_DIFFERENTIATING_NEGATIVE`: |d| ≥ 0.5, d < 0 (cell-of-origin de-differentiation; e.g. Boccellato Antrum_diff tile NEGATIVE expected for STAD tumor)
- `{tile}_PARTIAL`: 0.2 ≤ |d| < 0.5
- `{tile}_NULL`: |d| < 0.2

Pre-registered cell-of-origin tiles for STAD (where direction is biologically expected NEGATIVE per CCL-039):
- BoccellatoStomachRef.Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff
- Loyfer 25-tile.Upper_GI, .Colon_epithelial_cells (related)
- Moss 25-tile.stomach (Moss-only entry per CHK-2.6)

Pre-registered "homogenization positive" tiles (where positive direction is expected per CCL-039):
- Loyfer 25-tile.Bladder, .Lung_cells, .Hepatocytes, .Pancreatic_beta
- These should read POSITIVE direction in tumor-vs-normal STAD per CCL-039 colorectal precedent

### Stage 3 outcomes
For each Stage 3 atlas (Salas IDOL 6-cell, UniLIFE 19-cell):
- `{cell_type}_DIFFERENTIATING`: |d_unpaired| ≥ 0.5 with explicit direction label
- `{cell_type}_PARTIAL`: 0.2 ≤ |d| < 0.5
- `{cell_type}_NULL`: |d| < 0.2

### Multi-disease detection patterns (run-everything mandate)
Per CHK-3.2 + 2026-04-26 elevation rule, the prereg explicitly enumerates the multi-disease anomaly patterns this VAL is designed to surface:
1. **STAD specifically** — cell-of-origin tile NEGATIVE + Stage 1 POSITIVE + Stage 2 homogenization tiles POSITIVE
2. **STAD with H. pylori chronic infection (subset)** — adjacent-normal field defect predicted to BLUNT paired-d per CCL-025 anchor 3 (would be third confirming data point promoting CCL-025 to formal framework principle)
3. **STAD with EBV+ subtype** — high-CIMP epigenotype predicted to enhance positive Stage 1 d above pooled
4. **Stage 3 immune profile differential** — NOT routed into a "STAD-specific" claim; reported as patient-level immune microenvironment characterization

## 4. Pre-locked stratifications (CHK-2.2)

| Stratum | Source | Expected n (tumor) |
|---------|--------|-------------------|
| H. pylori positive | TCGA-STAD risk_factors / serology | partial coverage; ~30-50 |
| H. pylori negative | TCGA-STAD risk_factors / serology | partial coverage; ~50-100 |
| H. pylori unknown | TCGA-STAD | majority |
| EBV+ molecular subtype | TCGA molecular subtype call | ~30 (8% of STAD) |
| MSI subtype | TCGA molecular subtype call | ~80 (20%) |
| CIN subtype | TCGA molecular subtype call | ~200 (50%) |
| GS subtype | TCGA molecular subtype call | ~80 (20%) |
| Lauren intestinal | TCGA clinical | ~250 |
| Lauren diffuse | TCGA clinical | ~120 |
| Sex (M/F) | TCGA clinical | M~270 / F~125 |

## 5. CHK-3.1A and CHK-3.1B substrate gates (pre-locked)

- **CHK-3.1A**: TCGA HM450 sesame Level 3 substrate threshold — f_extreme ≥ 50.5%, f_middle ≤ 9.0% (full-genome). Sample-level fail-fast.
- **CHK-3.1B per-sample coverage gates**: ≥80% per atlas (per CHK-2.8 substrate floor for TCGA HM450K sesame Level 3). Run separately for each atlas's CpG list (Xu-538, Boccellato, Loyfer, Caggiano, Salas, UniLIFE).
- **CHK-3.1C**: PASS (verified at atlas build time for all atlases in run-everything stack)

## 6. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING per bladder-LL precedent)

Sample 5-10 random TCGA-STAD HM450 β files. For each, compute per-sample Xu-538 panel coverage. **If mean coverage <90% OR q5 <80%, prereg seal is HALTED** and either:
(a) re-derive Xu-538 panel from EPIC v1.0 → TCGA HM450 bridged probe-list (panel-coverage repair pathway per CHK-3.1B)
(b) declare gastric-epic Xu-538 transferability not-yet-established and route to a STAD-specific panel derivation

The bladder VAL-120 had this gate fail at 51.1% pass rate at 90% threshold (CHK-2.17 catch). The same gate applies here.

## 7. CHK-3.2 cross-cohort baseline check (mandatory)

For each Stage 1/2/3 panel and tile:
- Compute TCGA-STAD adjacent-normal-only mean A-score
- Compare to VAL-106 anchor (TCGA-KIRC+PRAD adjacent-normal mean A-score on same panel/tile)
- Report difference in anchor-SD units
- Flag tier per Stage 3 elevation rule:
  - <1 SD: report only
  - 1-3 SD: `baseline_mismatch_flag: true`, downgrade cross-cohort comparison
  - ≥3 SD: invalidate cross-cohort absolute comparisons; within-cohort only

## 8. Methodology

### Run-everything atlas stack (signed off 2026-04-26)
1. **Stage 1**: Xu-538 panel pooled-entropy A-score
2. **Stage 2**: Layered Moss+Loyfer 25-tile + BoccellatoStomachRef v1 (6 tiles) + Caggiano CelFiE TIM (19 tiles)
3. **Stage 3**: Salas IDOL 6-cell + UniLIFE Guo 2025 19-cell

Total: 1 Stage 1 panel × 25 + 6 + 19 = 50 Stage 2 tiles × 25 Stage 3 cell types per IDAT.

### Per-stratum analysis (run on each pre-registered stratum)
- Welch unpaired Cohen's d ± 95% CI
- Welch p-value
- Group means + SDs
- Saturation flag check per CHK-3.5 (immune ceiling 1.1921, secretory 1.1859)

### CCL-025 chronic-driver field-defect verification
Subgroup paired-d on the n=2 paired HM450 normal pairs is structurally uninformative (n=2). Instead:
- **Compare adjacent-normal A-score across H. pylori+ vs H. pylori- subgroups** in the unpaired pooled normal subset
- If H. pylori+ adjacent-normal A-score is elevated above H. pylori- by ≥0.02 (the VAL-064 threshold), this is the **third confirming data point for CCL-025** alongside lung-smoking and HCC-viral-hepatitis, promoting CCL-025 to formal framework principle

## 9. Specimen pathway compliance (CHK-2.4)

Specimen: **bulk tumor / adjacent-normal tissue** (not blood, not ccfDNA). Tissue substrate is Xu-538-validated per cycling-class TCGA precedents (VAL-058 prostate, VAL-060 breast, VAL-062 colorectal, VAL-063 lung, VAL-064 hepatocellular). No new transferability caveat needed.

## 10. CHK-7.6 reproducibility triple

- **Source code:** `val12X+1_stad_run_everything.py` (~250 lines, standard NumPy + pandas + scipy + skimage if needed for visualizations)
- **Inputs:**
  - TCGA-STAD HM450 sesame Level 3 cohort matrix (download via `tcgabiolinks` or GDC API; manifest sealed)
  - TCGA-STAD clinical metadata for stratification (sealed manifest)
  - All 6 atlas matrices (Xu-538, BoccellatoStomachRef, Loyfer 25, Caggiano TIM, Salas IDOL, UniLIFE) — SHA-sealed in INVENTORY.json
- **Environment:** Python 3.x, NumPy, pandas, scipy.stats. Expected runtime 30-60 min, ~10 GB memory peak.
- **Expected output:** `val12X+1_stad_results.json` with per-stage / per-stratum / per-atlas / per-tile A-scores + Cohen's d + CHK-3.2 cross-cohort baseline check + CCL-025 H. pylori field-defect test result

---

## Awaiting sequence

1. ✅ VAL-12X (Boccellato calibration) prereg drafted
2. ⏳ VAL-12X (Boccellato calibration) sealed + executed → per-tile thresholds locked
3. ⏳ VAL-12X+1 (this prereg) sealed
4. ⏳ CHK-2.17 pre-flight Xu-538 coverage check on TCGA-STAD HM450 5-10 sample subset
5. ⏳ Heath sign-off on outcome thresholds (O1-O5 above) + stratification table + multi-disease detection pattern enumeration
6. ⏳ Execute VAL-12X+1
