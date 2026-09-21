# OUTCOME — PROC-MATCH-01: Stage 8 disease matching. M1, M2, M4 PASS; M3 reported; M5 prediction correct — **row 8 stays OPEN** with its gate named.

**Run 2026-09-21.**

| bar | result | verdict |
|---|---|---|
| M1 origin gate fail-closed | `disease_origin_cells.json` missing → Stage 8 status NOT AVAILABLE, 0 candidates (was: silently `{}`); present → OK | PASS (fixed in `walther_clinical.py`) |
| M2 surface = seal | 115 cells: conductor per-cell A = mean_i H(β_i)/H_min over v0_2 markers to 4e-16 — the formula that reproduced the sealed anchors | PASS |
| M3 matrix integrity | v1_13: 80 rows, 53 diseases, 129 cell columns, 9 substrates; sha 53896622c1b4; **52 columns (cardiomyocytes, astrocytes, breast_ductal, brain_pooled …) are not reachable from the 115-cell atlas mapping** — tissue signatures the whole-blood atlas cannot score | REPORTED |
| M4 substrate firewall | whole_blood patient: 12 signatures scored, none from plasma/tissue; plasma_cfDNA patient: 0 whole-blood signatures | PASS |
| **M5 report** | 11 cached arrays: PRESENT cells entering the departure profile **0–5** (median 3); route-B candidates none. Healthy per-cell A on this surface: median 0.436, class H_min 0.77–0.98, so 104 of 115 cells are `below_floor` and excluded | prediction "< 10" **correct** |

**Why row 8 is not commissionable yet.** Stage 8's patient departure is (A_cell − 1.0) over cells above their class H_min — a definition that presumes healthy per-cell A ≈ 1. On the commissioned separation surface healthy per-cell A sits near 0.44–0.52 (PROC-ANCHOR-01: 0.520 median on the foundation cohort). The detector therefore sees almost nothing on any sample and cannot be said to match or fail to match. **Gate for row 8:** re-derive the per-cell departure reference on the separation surface from the four-lab healthy panels (a per-cell healthy level and spread, laboratory-zeroed like everything else), re-express the signature matrix against it, then re-seal. Until then `run_full` carries Stage 8 as `diagnostic_disease_matching` with `reportable=False` and `row_status: OPEN`. In the detection rule's words: disease matching on the commissioned chain is **not yet tested**.

Kit test `test_disease_matching_gate.py` (M1, M4). Per-array M5 record: `m5_cached_arrays.json`.

---
**SEALED** sha256 `a83e5ddaf3956445f26c462969c78f3609394770fff5687ac6e9a3fecd353ce7` · 2026-09-21
