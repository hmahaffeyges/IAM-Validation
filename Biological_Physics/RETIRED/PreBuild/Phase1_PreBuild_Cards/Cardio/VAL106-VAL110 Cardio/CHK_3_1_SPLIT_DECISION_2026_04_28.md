# CHK-3.1 Split Convention — Policy Decision Locked 2026-04-28

**Decision date:** 2026-04-28  
**Decided by:** Heath W. Mahaffey  
**Phase status:** PHASE 1 active (apply forward to cardio testing); PHASE 2 pending (cookbook-wide retroactive update); PHASE 3 pending (per-card retroactive review).

---

## The decision

CHK-3.1 is split into two distinct named checks, each with its own threshold convention. Both must pass for a sample to clear data-integrity gating.

### CHK-3.1A — Full-genome bimodality

**Question answered:** Is the upstream IDAT-to-β pipeline producing real raw-β output with intact bimodal distribution structure?

**Measurement:** Compute f_extreme (β<0.10 or >0.90) and f_middle (0.40-0.60) on **every valid β value in the input file**, no subsetting. Quality threshold: ≥400,000 valid β values per sample (cookbook standard).

**Threshold:** Single threshold per measurement substrate, established by calibration VAL on structurally-separated healthy adjacent-normal cohorts. Once calibrated, the threshold is reused indefinitely for that substrate.

**Examples of substrate categories that need separate calibration:**
- Raw EPIC v1.0 / EPIC v2.0 β (un-normalized, GenomeStudio AVG_Beta or sesame raw output)
- TCGA HM450K sesame Level 3 β (standard TCGA pipeline with dye bias correction)
- minfi noob-bg-corrected EPIC v2 β (the VAL-100 GSE282666 substrate — likely fails)
- Future substrates: WGBS-derived array projection, ddRBS, nanopore-derived methylation

**Catches:** Processed-output deferrals (CCL-040 pattern), wrong-substrate IDAT pipelines, normalization-residual artifacts that homogenize the genome globally.

### CHK-3.1B — Card-specific marker subset bimodality

**Question answered:** Are the actual CpG subsets the framework will score on this cohort intact, with viable bimodality on the panel-specific markers?

**Measurement:** Compute f_extreme and f_middle on **the union of all CpGs the card's scoring will use** — Stage 1 panel ∪ Stage 2 atlas markers ∪ Stage 3 atlas markers, as applicable per card. Each card has its own CHK-3.1B threshold derived from the same calibration cohorts as CHK-3.1A but computed on that card's specific union.

**Threshold:** Per-card, computed on the same calibration cohorts (TCGA-KIRC + TCGA-PRAD adjacent-normal for HM450K substrates). Recomputed when a card adds a new atlas or updates a marker panel. Stored in the card's universal_pipeline_acknowledgment block.

**Examples of subsets per card:**
- breast-epic: Xu-538 immune ∪ Loyfer 25-tile breast ∪ EpiSCORE BreastRef ∪ UniLIFE 19-cell ∪ Salas 6-cell
- ad-immune: VAL-051 7-CpG Rule A ∪ Xu-538 immune (older variants used UniLIFE only)
- hcc-epic: Xu-538 immune ∪ Loyfer Hepatocytes tile ∪ Loyfer 25-tile etiology stratification subset
- cardio-epic (NEW): Xu-538 immune ∪ Loyfer Vascular_endothelial_cells + Left_atrium tiles ∪ EpiSCORE HeartRef CM/EC/FB/MP/SMC ∪ Caggiano CelFiE heart_meth + endothelial_meth ∪ UniLIFE ∪ Salas

**Catches:** Probe-list lift-over dropouts (450K → EPIC v1 → EPIC v2 panel coverage gaps), ancestry-specific failed probes that hit panel CpGs disproportionately, atlas-specific marker concentration in regions affected by a localized artifact.

### The conjunction rule

A sample passes CHK-3.1 iff (CHK-3.1A passes) AND (CHK-3.1B passes). Either failure routes to the appropriate deferral pathway:
- CHK-3.1A fail → CCL-040 reprocessing pathway (raw IDAT re-extraction, alternative pipeline)
- CHK-3.1B fail → panel-coverage repair pathway (probe-list verification, alternative atlas with overlap, defer card to next version)

---

## Why this is right for EDEAR specifically

**Production deployment architecture (CCL-037).** Customer IDATs run through a single calibrated pipeline. CHK-3.1A is computed once per customer (substrate gate). CHK-3.1B is computed per disease card (panel-coverage gate). A customer with substrate-clean data but partial panel coverage gets cards their data supports, not an all-or-nothing report failure. This is a meaningful UX advantage over conflated CHK-3.1.

**Future-proofing across atlas additions.** New atlases (Konigsberg cardiac, Jacques 17-tissue ageing, MARLIN leukemia, Tanaka neural cfDNA, future WGBS-derived array references, EPIC v3 if released) each bring new markers. Under conflated CHK-3.1, every atlas integration triggers cookbook-wide rethreshold. Under split convention, CHK-3.1A is substrate-stable (set once per substrate, reused for every card); CHK-3.1B recomputes per-card-per-cohort on demand. The split absorbs new atlas additions cleanly.

**Auditability for referees / FDA / future regulatory inspection.** Each VAL outcome becomes traceable to which CHK gate produced the result. The cookbook can answer "why did VAL-XYZ fail" with a precise answer, not a conflated answer.

---

## Phase 1 — Apply forward to cardio-epic

Cardio-epic is the first card built under the split convention. From this point forward in the cardio testing chain:

- **VAL-106** (calibration VAL on TCGA-KIRC + TCGA-PRAD): outcome currently O3 under conflated convention. Re-classify under split convention as the **CHK-3.1A calibration anchor for TCGA HM450K sesame Level 3 substrate**. The data on disk is sufficient — no re-download needed.
- **VAL-107** (new): CHK-3.1B calibration for cardio-epic specifically. Run the same TCGA-KIRC + TCGA-PRAD cohort through the cardio-epic marker subset to establish the cardio-epic-specific CHK-3.1B threshold for HM450K substrate.
- **VAL-108, VAL-109, VAL-110** (cardio-epic disease VALs on GSE69138, GSE84395, GSE84274): each prereg seals with both CHK-3.1A and CHK-3.1B thresholds locked from VAL-106 and VAL-107 calibration outcomes.

---

## Phase 2 — Cookbook-wide convention update (PENDING completion of Phase 1)

After cardio-epic card and README are sealed, update these six cookbook documents to incorporate the CHK-3.1A/B split:

| Document | What changes |
|---|---|
| **TESTING_CHECKLIST.md** | CHK-3.1 section split into CHK-3.1A (full-genome) and CHK-3.1B (card-specific marker subset). Platform threshold table extended with both columns. CCL-040/041 cross-references updated. |
| **EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md** | Part 16 updated with CHK-3.1A/B split. New Part 17 documenting Phase 1 cardio rollout + Phase 2 cookbook-wide retroactive reclassification. |
| **LESSONS_LEARNED.md** | CCL-040 reclassified as CHK-3.1A failure mode (processed-output deferral). CCL-041 reclassified as CHK-3.1B platform-calibration lesson. New CCL-042 documenting the CHK-3.1 split decision and Phase 1/2/3 rollout. |
| **README_MASTER.md** | New v2.4 amended line documenting the CHK-3.1 split, with pointer to CCL-042. |
| **GAPE_Evidence_Report_UPDATED.html** | VAL-100, VAL-101 entries get retroactive split-classification footnotes (no change to sealed outcomes; classification footnote only). |
| **GAPE_Reproduction_Paper_v1.md** | Section on CHK-3.1 updated with the split convention. This is the primary public-facing methodology document — must reflect the corrected convention. |

---

## Phase 3 — Per-card retroactive review (PENDING completion of Phase 2)

After Phase 2 cookbook documents are updated, walk every existing card and update its `universal_pipeline_acknowledgment` block to include both CHK-3.1A platform thresholds (substrate-keyed table) and the card-specific CHK-3.1B threshold for that card's marker union.

Cards in scope:
- breast-epic v2.3 → v2.4 (CHK-3.1B threshold for Breast subset on HM450K + EPIC v1 substrates)
- lung-epic v0.2 → v0.3 (CHK-3.1B for lung subset)
- ad-immune (CHK-3.1B for VAL-051 7-CpG panel)
- hcc-epic v0.3 → v0.4 (CHK-3.1B for HCC subset; VAL-101 retroactive classification per Phase 2)
- crc-epic v2.4 → v2.5 (CHK-3.1B for CRC subset; VAL-098/099/100 retroactive classification)
- kidney-epic, cervical-epic (CHK-3.1B per card)
- cardio-epic v0.2 (already built under split convention in Phase 1; no retroactive update needed)

Each card update is an additive documentation update that does not change sealed validation tiers.

---

## What does NOT change

- **Sealed VAL outcomes do not unseal.** VAL-100 stays at O5_DATA_INTEGRITY_FLAG. VAL-101 stays at O5_DATA_INTEGRITY_FLAG. VAL-097 stays at O5_BASELINE_DOMINATED. The seals honor themselves.
- **The retroactive classification is documentation-only.** It explains which CHK gate would have produced each existing outcome under the split convention. It does not retroactively change outcomes.
- **EDEAR commercial deployment is unaffected** per CCL-037. Deployment runs on a single calibrated pipeline that's structurally insulated from public-data CHK-3.1 reclassification.

---

## Decision rationale (for the cookbook record)

This decision was driven by VAL-106 outcome O3, which revealed that the cookbook's CHK-3.1 had been conflating two distinct data-integrity questions: upstream pipeline integrity (substrate-level) vs panel-specific damage (subset-level). The conflation manifested as a methodological inconsistency between threshold values derived from CpG-subset measurements (VAL-101 26.6%, VAL-099 24.4%, VAL-077 12%) and threshold values that would be appropriate for full-genome measurements (VAL-106 calibration shows healthy adjacent-normal HM450K reads ~55-56% extreme on the full ~408K-CpG distribution).

The split is defensible for 3+ years because the two questions don't change as the cookbook scales: substrate gating is stable per-substrate, panel-coverage gating recomputes per-card-per-cohort. New atlases extend the framework without destabilizing existing thresholds.

---

**This decision is locked as of 2026-04-28. PHASE 1 active immediately. PHASE 2 + PHASE 3 pending completion of cardio testing.**
