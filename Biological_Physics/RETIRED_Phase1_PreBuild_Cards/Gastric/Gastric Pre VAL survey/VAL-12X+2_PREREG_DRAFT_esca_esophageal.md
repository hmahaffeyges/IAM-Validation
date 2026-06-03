# VAL-12X+2 (provisional ID) — Phase C Run-Everything on TCGA-ESCA (Esophageal Cancer, ESCC + EAC stratified)

**Prereg version:** v1.0-DRAFT
**Date drafted:** 2026-05-02
**Card:** gastric+esophageal-epic v0.1 sprint, Phase C esophageal arm (run side-by-side with gastric per Heath sign-off)
**Prereg type:** Phase C disease cohort scoring (run-everything regime)
**Depends on:** VAL-12X (BoccellatoStomachRef calibration) sealed first

---

## 1. VAL identification

- Provisional VAL ID: **VAL-12X+2** (sequential after STAD VAL)
- Cohort: **TCGA-ESCA** (Esophageal Carcinoma, n≈185 mixed ESCC + EAC + paired adjacent-normal HM450 sesame Level 3)
- Comparison strategy: **Welch tumor-vs-pooled-normals**, SUBTYPE-STRATIFIED (ESCC vs EAC are different cells of origin per Hao/Berman 2023 Genome Biology)
- Stratifications mandatory: **histology subtype (ESCC vs EAC vs adenosquamous)**, **smoking status (CCL-009 + ESCC mandate)**, **alcohol consumption status**, **sex**, **stage**

## 2. Hypothesis (pre-locked, BIDIRECTIONAL per CHK-2.7 + Heath reminder)

**Critical biology distinction** (Hao/Berman 2023, Genome Biology):
- **ESCC** arises from squamous epithelial cells → cell-of-origin tile = squamous (Loyfer Skin_keratinocyte + Head_and_neck_larynx)
- **EAC** arises from columnar cells → cell-of-origin tile = gastric/intestinal (Boccellato + Loyfer Upper_GI/stomach/small intestine)
- **Pooled ESCA analysis is biologically uninterpretable** because the two subtypes have orthogonal cell-of-origin signatures

**Stage 1 hypothesis (Xu-538 architectural drift):** Both subtypes show |d_unpaired| ≥ 0.5 vs pooled normals on Stage 1 Xu-538. Direction expected POSITIVE (cycling-class tumor signature).

**Stage 2 hypothesis (subtype-stratified cell-of-origin):**
- ESCC subtype: Loyfer Skin_keratinocyte tile reads NEGATIVE direction (squamous de-differentiation); BoccellatoStomachRef tiles read NULL or POSITIVE-homogenization direction
- EAC subtype: BoccellatoStomachRef tiles + Loyfer Upper_GI tile read NEGATIVE direction (columnar de-differentiation); Loyfer Skin_keratinocyte reads NULL or POSITIVE-homogenization

This subtype-stratified pattern, IF observed, is a **first-of-kind multi-atlas cell-of-origin discrimination demonstration** — a single Phase C VAL distinguishes ESCC from EAC on methylation alone using two complementary atlases.

**Bidirectionality declaration:** Per Heath's reminder, all outcome thresholds use magnitude-based |d| with explicit direction labels. No outcome pre-locks direction alone.

## 3. Pre-locked decision criteria (CHK-2.1 + CHK-2.7 + CHK-4.11)

### Stage 1 outcomes (per subtype)

For each of {ESCC, EAC, adenosquamous, all_pooled}:
- **O1_STAGE1_PASS**: |d_unpaired| ≥ 0.5, lower CI bound away from 0 in observed direction
- **O2_STAGE1_PARTIAL**: 0.2 ≤ |d_unpaired| < 0.5
- **O3_STAGE1_NULL**: |d_unpaired| < 0.2

### Stage 2 outcomes (per atlas, per tile, per subtype)

For each tile, the subtype-stratified |d_unpaired| is computed. The **subtype-discrimination flag** fires if:
- For ESCC subtype: Loyfer Skin_keratinocyte |d| ≥ 0.5 NEGATIVE direction AND BoccellatoStomachRef tiles |d| < 0.3 in same subtype
- For EAC subtype: BoccellatoStomachRef tiles |d| ≥ 0.5 NEGATIVE direction AND Loyfer Skin_keratinocyte |d| < 0.3 in same subtype

If both conditions fire (independently per subtype), the **first-of-kind multi-atlas subtype-discrimination demonstration** is recorded as primary VAL finding; if neither fires, the run-everything stack still produces full A-score outputs but the subtype-discrimination claim is not made.

### Stage 3 outcomes
Same as VAL-12X+1 STAD: per-cell-type immune sub-composition with magnitude-based |d| and explicit direction labels.

## 4. Pre-locked stratifications (CHK-2.2)

| Stratum | Source | Expected n |
|---------|--------|-------------|
| ESCC | TCGA histology call | ~95 (51%) |
| EAC | TCGA histology call | ~85 (46%) |
| adenosquamous | TCGA histology call | ~5 (3%) |
| Smoking ever | TCGA exposures | ~120 |
| Smoking never | TCGA exposures | ~30 |
| Smoking unknown | TCGA exposures | ~35 |
| Alcohol consumer | TCGA exposures | ~80 |
| Sex (M/F) | TCGA clinical | M~145 / F~40 |
| Stage I/II vs III/IV | TCGA clinical | Roughly equal split |

## 5. CHK-3.1A and CHK-3.1B substrate gates

Same as VAL-12X+1 STAD per identical TCGA HM450 sesame Level 3 substrate.

## 6. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING)

Sample 5-10 random TCGA-ESCA HM450 β files. For each, compute per-sample Xu-538 + Loyfer + Boccellato + Caggiano + Salas + UniLIFE coverage. If any atlas mean coverage <90% OR q5 <80%, halt prereg seal and route to repair pathway.

## 7. CHK-3.2 cross-cohort baseline check

For each panel and tile:
- Compute TCGA-ESCA adjacent-normal-only mean A-score
- Compare to VAL-106 anchor (TCGA-KIRC+PRAD)
- Compare to VAL-12X+1 STAD adjacent-normal-only mean A-score (CROSS-DISEASE comparison; if STAD and ESCA adjacent-normals match closely, this validates the Boccellato atlas as a foregut-endoderm-class detector at run-everything scale)
- Flag tiers per Stage 3 elevation rule

## 8. Multi-disease detection patterns enumerated

Per CHK-3.2 + run-everything mandate, this VAL is designed to surface:
1. **ESCC subtype identification** via Stage 2 Loyfer Skin_keratinocyte NEGATIVE pattern + Stage 1 POSITIVE
2. **EAC subtype identification** via Stage 2 Boccellato+Upper_GI NEGATIVE pattern + Stage 1 POSITIVE
3. **Smoking-stratified ESCC** confirming CCL-009 + CCL-025 chronic-driver field-defect at the squamous lineage
4. **Subtype-discrimination on methylation alone** as a first-of-kind cookbook demonstration
5. **Cross-organ comparison with VAL-12X+1 STAD** at run-everything Boccellato scoring (foregut-endoderm class verification)

## 9. CCL-025 chronic-driver field-defect verification (smoking + alcohol)

Same approach as VAL-12X+1 — compare adjacent-normal A-score across smoking-positive vs smoking-negative subgroups (ESCC arm specifically). If smoking-positive adjacent-normal A-score is elevated above smoking-negative by ≥0.02, this would be a fourth CCL-025 confirming data point alongside lung VAL-063, HCC VAL-064, and (predicted) STAD VAL-12X+1 H. pylori. Three independent confirmations promotes CCL-025 to formal framework principle.

## 10. Specimen pathway compliance (CHK-2.4)

Specimen: bulk tumor / adjacent-normal tissue (not blood, not ccfDNA). Tissue substrate is Xu-538-validated.

## 11. CHK-7.6 reproducibility triple

- **Source code:** `val12X+2_esca_run_everything.py` (~280 lines including subtype stratification + multi-atlas A-score loop)
- **Inputs:** TCGA-ESCA HM450 sesame Level 3 cohort matrix; TCGA-ESCA clinical metadata; 6 atlas matrices (SHA-sealed)
- **Environment:** Python 3.x, NumPy, pandas, scipy.stats. ~30-60 min runtime, ~10 GB memory peak.
- **Expected output:** `val12X+2_esca_results.json`

---

## Awaiting sequence

1. ✅ Atlas built + Phase 0 cohort survey done
2. ⏳ VAL-12X (Boccellato calibration) sealed first
3. ⏳ VAL-12X+1 STAD prereg sealed
4. ⏳ This prereg sealed alongside (gastric AND esophageal side-by-side per Heath sign-off)
5. ⏳ Pre-flight CHK-2.17 on TCGA-ESCA
6. ⏳ Heath sign-off on outcome thresholds + subtype-discrimination claim language
7. ⏳ Execute together (Heath's "side-by-side" mandate)

---

## Cross-VAL synthesis after both Phase C complete

The combined STAD+ESCA outcome data informs the Phase D card decision:
- **If Boccellato tiles + Loyfer Upper_GI tile + Loyfer skin_keratinocyte all behave coherently per subtype** → single combined gastric+esophageal-epic v0.1 card (atlas readers cleanly distinguish subtype on the same atlas stack)
- **If subtype-discrimination patterns are weaker than expected** → separate gastric-epic v0.1 + esophageal-epic v0.1 cards
- **Either way:** Crohn's-disease-IBD pathway language documented in known_limitations of both, with hcc-epic v0.3.1 amendment also adding the same language
