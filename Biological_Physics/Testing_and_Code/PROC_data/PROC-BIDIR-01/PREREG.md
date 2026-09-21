# PREREG — PROC-BIDIR-01: Stage 4.5 bidirectional detector — VAL-050 / VAL-051 / CPG-VAL-019 reproduced from the kit (CHAIN_COMMISSIONING row 4.5)

**Sealed 2026-09-21 before any script is rerun.** One check was performed before sealing and is disclosed: CPG-VAL-019's statistics were recomputed from its own per-sample CSV (d_up +0.4939, d_down −0.5153, d_signed +0.5991, 7/7 concordance — all match) as an internal-consistency read of the record; it is not a bar below.

**Bars.**
- **B1 integrity** — every file named in VAL_050_SEAL.txt and VAL_051_SEAL.txt hashes to its sealed SHA-256 at HEAD.
- **B2 VAL-050 rerun** — `run_val_050.py` on the sealed `aibl_imm_betas.json` returns pooled-entropy Cohen's d within ±0.002 of +0.077, AUC within ±0.005 of 0.512, OUTCOME 3 NULL.
- **B3 VAL-051 rerun** — `val051_analyze.py` on the sealed inputs and split map returns Rule A holdout d within ±0.002 of +0.6237, AUC within ±0.005 of 0.6769, null-comparator pooled-entropy d within ±0.002 of +0.0562, holdout counts 33 AD / 95 HC.
- **B4 engine = seal** — `CPG_Engine/Runtime Matrices/Directional Panel/bidirectional_decomposition.a_dir_score` applied to the sealed 7-CpG Rule A panel returns, for every AIBL holdout sample, the same A_dir as `val051_analyze.py` (max |diff| < 1e-9).
- **B5 raw GEO** — the 18 IMM CpGs re-extracted for the 726 AIBL samples from `GSE153712_normalized_average_betas.txt.gz` (GEO supplementary, 5.1 GB) match `aibl_imm_betas.json` at max |diff| < 1e-4; B2 and B3 rerun on the re-extracted betas within the same tolerances.

**Row 4.5 passes** if B1–B5 pass. **Record note regardless of outcome:** the row's spec "pooled d ≈ +0.08 → directional d ≈ +0.62 on the same AIBL samples" conflates VAL-050 (18-CpG IMM panel, pooled entropy, all 726) with VAL-051 (7-CpG Rule A, holdout 128); and CPG-VAL-019's "d_pooled" is d on the pooled β-mean, not on the pooled A-score. Both are stated in the outcome.

---
**SEALED** sha256 `e1bde910501c76cf064e3b5a0077741c45ce87d73819e283c9f9db8dd9d268db` · 2026-09-21
