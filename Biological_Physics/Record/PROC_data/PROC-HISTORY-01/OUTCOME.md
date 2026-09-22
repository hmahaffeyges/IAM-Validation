# PROC-HISTORY-01 — the complete validation history, counted from the record (2026-09-21)

**Why.** The author read the "103 indexed pre-registered validations (81 pre-Atlas, 15 post-Atlas, 7 retired)" sentence in Paper 1 and said: "No, we have 128 VAL's plus VAL-049 includes 15 of them in our T Series (T1 → T15). Not to mention the full G series before VAL001 ever started. THEN post IAM-Atlas build we actually tested cpg_val_001-022, not 15." He was right on every count.

**What was wrong, and why.** `VAL_INDEX.csv` (built 2026-09-19 by walking repository paths) had two defects. (1) It keyed on the bare number, so pre-Atlas **VAL-001** and post-Atlas **CPG-VAL-001** collapsed into one row — 22 collisions. (2) It counted only identifiers with a folder at HEAD; 51 pre-Atlas VALs whose records live in the RETIRED evidence report and the author's Zenodo deposit (CC-BY-4.0, 10.5281/zenodo.19633499) have no folder and were not counted. Seven post-Atlas AD folders (CPG-VAL-008..014) had also been mis-filed under `VAL_PreAtlas/` during the 2026-09-19 reorganisation; moved to `VAL_PostAtlas/`.

**The record, counted from `RETIRED_VAL_inventory_report.md` (compiled line-by-line from the April evidence report and README, each outcome cross-checked against `validation_runs/`) and the post-Atlas v10 evidence report, plus repository folders:**

| series | when | count | executed with a recorded outcome | notes |
|---|---|---|---|---|
| **G-series** (H_min calibration, before VAL-001) | Apr 2026 | G-002 (17 chains, 8 methylation floors), G-003b (32 floors), bootstrap of the 32; also G-008, E_A,bio, n_bio ordering | 3 sealed | code restored to `Hmin_Calibration/`; methylation bootstrap run 2026-09-20 (PROC-HMIN-BOOT-01) |
| **VAL-001 → VAL-128** (pre-Atlas) | Apr–May 2026 | **119 distinct identifiers** (numbering gaps 34–36, 78–80, 103–105) in six families: methylation 001–013; five-substrate 014–033; drift cascade 037–046 (35/39 predictions); EDEAR disease cards 047–128 across 12 cards | **107**; 12 not run / queued / dbGaP-gated / excluded at runtime / voided (VAL-102, 4 min) | VAL-050 onward individually pre-registered and SHA-256 sealed before β access |
| **T1 → T15** (VAL-049 cross-population) | Apr 2026 | 15 | 12 (T4, T6, T7 dbGaP-gated) | US / AU / UY / UK / PL / CN-SG populations, frozen panel + frozen H_min |
| **CPG-VAL-001 → CPG-VAL-022** (post-Atlas, IAMAtlas REBUILD) | 29 May – 7 Jun 2026 | **22 slots** | **21** (021 deferred, cohort acquisition) | breast 001–007 (2 RESTATED), AD 008–014 (three cohorts, AD/FTD/PSP-CBD direction discrimination), immune/aging 015–020 & 022 (Hannum full chain 020); L9 null suite N1–N8 per VAL; PREREGs marked RETROSPECTIVE by the author |
| **Mahalanobis HC hull** v0_1 → v0_5 | 6 Jun 2026 | 5 versions | n_HC 601 → 2,523; 8 cohorts; 4 populations incl. **Han Chinese GSE141682 n=42 (first Asian)** | fixed d ≥ 2.0 found invalid in 112-D and replaced by percentile-of-HC; anchor d fell honestly as the hull broadened |
| **L9 N7** chain integrity | 5 Jun 2026 | 1 | synthetic truth through Walther → A-scoring → Mahalanobis | R1 MAE < 1 %; the September PROC-N7-01 rerun is what found the gauge reading the wrong loci |
| **September 2026 procedures** (rebuild from scratch) | 18–21 Sep | PROC-CAL/DECON/ANCHOR/FORMULA/N7/NILC/SEP/CHAIN/STAGE0/WB-IMMUNE/HMIN-BOOT/PANEL-01..03, PHASE 1/1c, band_v2, LAB-ZERO-01/02, CPG-NEW-001 | all sealed before data | this document's own record |

**Corrected sentence for Paper 1 and Issue 003:** *Between April and June 2026 the author ran a G-series calibration, 119 pre-Atlas validations (107 executed; 12 not run for stated reasons) including a 15-cohort cross-population series, and 22 post-Atlas validations (21 executed) on the rebuilt atlas, followed by a five-version healthy-hull expansion to 2,523 controls across four populations and a chain-integrity test with synthetic truth. Records: RETIRED evidence report and inventory; v10 evidence report; Zenodo 10.5281/zenodo.19633499; `Record/VAL_INDEX.csv` (175 rows, unique keys by series).*

**What none of them could see (unchanged):** every one was a within-pipeline comparison or a reading against a control centroid; the pipeline-scale offset and the age curve were found in September by absolute reading. The April caveats tab had predicted the scale offset.

**Lesson (RUNBOOK):** an index built by walking the tree counts folders, not the record. When the record and the tree disagree, the record wins and the index is rebuilt from it with unique keys per series.

---
**SEALED** sha256 `e76b57c960f3a23ea710abe125fd712f6ceadc498b5c92869c94cf6e8c1f25d9` · 2026-09-21
