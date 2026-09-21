# PREREG — PROC-MATCH-01: Stage 8 disease matching (CHAIN_COMMISSIONING row 8)

**Sealed 2026-09-21 before any run.** Read before sealing, disclosed: `walther_clinical.stage_8_dual_matching` loads `disease_origin_cells.json` inside `try/except Exception: origin_map = {}` — a missing or corrupt origin file silently degrades the specificity rule (fails OPEN). Its patient departure profile is (A_cell − 1.0) over PRESENT cells on the per-cell separation surface, where PROC-ANCHOR-01 measured healthy per-cell A median ≈ 0.52 with the class H_min floor ≈ 0.84.

**Bars.**
- **M1 origin gate fail-CLOSED.** With `disease_origin_cells.json` absent or unparsable, Stage 8 returns `available=False` with a reason and scores NO candidates; with the file present it runs. Kit test.
- **M2 surface = seal.** The per-cell A entering Stage 8 (conductor Stage A `cells[*].A`) equals mean_i H(β_i)/H_min over the v0_2 markers (the formula that reproduced the sealed anchors at r = 1.00000) on GSM2333901 to 1e-9 for every cell with markers present.
- **M3 matrix integrity.** `disease_cell_signature_matrix_v1_13.csv` parses (81 rows, 53 diseases); every non-metadata column is a key of `iamatlas_115_to_matrix_v0_2_mapping.json` or a known matrix column; SHA-256 recorded.
- **M4 substrate firewall.** A whole-blood patient is scored against zero `plasma_cfDNA` / tissue signatures; a `plasma_cfDNA` patient against zero whole-blood signatures.
- **M5 REPORT (detection rule — no bar).** The 11 cached arrays through Stage 8: number of PRESENT cells entering the departure profile, top route-B cosine and disease, whether any concern fires. **Analyst's sealed prediction:** on healthy whole blood the departure profile admits **< 10 cells** because (A − 1.0) with a below-H_min gate excludes nearly every cell whose healthy A ≈ 0.52 — i.e. Stage 8's departure is referenced to a level (per-cell A ≈ 1 healthy) that the commissioned separation surface does not sit at. If so, the row is NOT commissionable on M1–M4 alone: the departure reference must be re-derived on the separation surface (healthy per-cell level from the four-lab panels) before any disease matching is declared, and that is recorded as the gate.
- Row 8 commissioned only if M1–M4 pass AND M5 shows the departure profile is populated on healthy arrays (≥ 30 present cells) — otherwise row 8 stays OPEN with the re-referencing as its gate.

---
**SEALED** sha256 `75f5a1a27273b452a0c654daf5ee69acbbb3ae09214b09dfd8f75936d0ae59af` · 2026-09-21
