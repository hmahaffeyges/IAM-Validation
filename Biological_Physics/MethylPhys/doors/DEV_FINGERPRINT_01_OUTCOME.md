# DEV-FINGERPRINT-01 — outcome (2026-10-10T21:52Z; development)

Read by the sealed scorer, unchanged (`data/DEV_FINGERPRINT_01/score_fingerprint_01.py`, sealed f4fce17, sha256 2f6fdde13dafa344...; planted
test passed first, cafa6d6). Rows `data/DEV_FINGERPRINT_01/fingerprint_01_rows.csv`; scorer output `fingerprint_01_output.txt`.
LNCaP (prostate cancer line, 5 WGBS runs, 2 EPIC arrays) against PrEC (normal prostate epithelium, 4 runs, 2 arrays), GSE86833, one laboratory.

| quantity | PrEC | LNCaP | read |
|---|---|---|---|
| copy error ε (Stage Q, runs pooled) | 0.04797 | 0.05237 | IAM-A_rel 1.0673 |
| share of ≥ 6-call molecules Stage Q can read, LNCaP ÷ PrEC | — | — | 0.618 |
| Met-A, mean H(β) on 5,531 prostate identity sites | 0.38224, 0.38299 | 0.46717, 0.47137 | Met-A_rel 1.2265 |
| excess run loss per run | +0.0000 to +0.0043 | +0.1105 to +0.1130 | every LNCaP run above the largest PrEC run |

**Arm choice (sealed): share read 0.618 < 0.70, so Arm B decides.** Arm B: **FINGERPRINT** (every LNCaP run's excess run loss, +0.1105 to
+0.1130, exceeds the largest PrEC run's, +0.0043). Arm A, recorded: curve B at IAM-A_rel 1.0673 is 1.1672 (inside its table, no extension);
Met-A_rel 1.2265 lies +0.0593 above it, z 2.37 > 1.645: also FINGERPRINT.

**What it means, as far as this set reaches.** In LNCaP, Met-A moved further than its copy error allows (Arm A), and the extra movement is
methylation lost in runs along molecules, not scattered single errors (Arm B: territory molecules fully held 0.52 vs 0.88). The two
instruments together separate "the copier got worse" from "blocks of the pattern were lost". Run-type loss in cancer genomes is known
(partially methylated domains); here it is read as a fingerprint against the IAM-A ↔ Met-A relation, by a rule sealed before reading.
**Limits.** One cell line against one primary culture from one laboratory; LNCaP is a long-cultured line, so culture as well as cancer may
contribute. 38 % fewer of LNCaP's molecules qualify for Stage Q, so IAM-A_rel here reads only the molecules still held and understates the
copier's loss: the reason Arm B, not Arm A, decides.
