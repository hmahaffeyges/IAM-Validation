# OUTCOME — PROC-BIDIR-01: Stage 4.5 bidirectional detector. B1–B5 PASS — **row 4.5 COMMISSIONED** (`row_4_5_commissioned: True`, the sealed script's verdict).

**Run 2026-09-21.** A built tool (VAL-050 / VAL-051 scripts, `bidirectional_decomposition.py`, `directional_panels_v1_0.json`) tested against sealed bars — a seal, under the sealing rule.

| bar | result | verdict |
|---|---|---|
| B1 seal integrity | 15/15 sealed SHA-256 match at HEAD (VAL-051's inputs restored beside its script; hashes identical to VAL-050's) | PASS |
| B2 VAL-050 rerun | pooled-entropy d = +0.0768 (sealed +0.0768), AUC 0.5123 | PASS |
| B3 VAL-051 rerun | Rule A holdout d = +0.6237, AUC 0.6769, null comparator d = +0.0562; 33 AD / 20 MCI / 95 HC | PASS |
| B4 engine = seal | `score_directional_composite` on the runtime immune panel vs the 148 sealed per-sample A_dir_A: max diff 2.2e-16; runtime panel identical to sealed Rule A | PASS |
| B5 raw GEO | `GSE153712_normalized_average_betas.txt.gz` (5.13 GB, samples-as-rows): 726/726 samples, 18/18 IMM CpGs, 13,068 values, **max diff 0.00** against `aibl_imm_betas.json` | PASS |

**Disclosed and recorded.** The record's "pooled" d for CPG-VAL-019 is on the pooled β mean, not the pooled A (recomputed from its per-sample CSV; both stated). The row's specification had conflated VAL-050's 18-CpG IMM panel with VAL-051's 7-CpG Rule A panel; they are distinct and both reproduce. The B5 extractor first assumed GEO series-matrix orientation; the supplementary file is the transpose and the kit reads it as such.

**What is and is not said.** The detector reproduces its sealed record on AIBL. Whether it detects anything on the commissioned chain in a new cohort is *not yet tested* (detection rule). Kit: `PROC_BIDIR_01.py` (needs the 5.1 GB file in `CPG_KIT_DATA`).

---
**SEALED** sha256 `a23270c2d7a2bd1861adf91e52e74500784318026888567ed03a774d9bf406d6` · 2026-09-21
