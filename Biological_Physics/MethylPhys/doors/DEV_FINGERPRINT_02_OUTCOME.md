# DEV-FINGERPRINT-02 — outcome (scored 2026-10-11 by rule 02b; development)

Each pair scored by `data/DEV_FINGERPRINT_02/score_fingerprint_02b.py score <pair>`, which refused unless the pair's seal was committed unchanged and
its own sha256 matched the seal. Table written by `outcome_02.py` from the committed `scored_<pair>.json`. Pairs are added as they are scored.

| pair | cancer vs normal | Arm B excess per cancer file | bar | min excess / donor excess | verdict (Arm B) | Arm A, reported only | readable share, cancer / normal |
|---|---|---|---|---|---|---|---|
| liver | HepG2 vs hepatocyte | 0.0113 to 0.0166 | 0.0211 | 0.53 | **no fingerprint** | IAM-A_rel 0.9614, Met-A_rel 1.0371, curve B 1.0000, z 0.26 (no fingerprint) | share read 3.34 |
| lung_alveolar | A549 vs epithelial cell of alveolus of lung | 0.0067 to 0.0068 | 0.0211 | 0.32 | **no fingerprint** | IAM-A_rel 0.7667, Met-A_rel 1.0487, curve B 1.0000, z 0.24 (no fingerprint) | share read 3.29 |
| lung_bronchial | A549 vs bronchial epithelial cell | 0.0087 to 0.0092 | 0.0211 | 0.41 | **no fingerprint** | IAM-A_rel 0.8404, Met-A_rel 1.3219, curve B 1.0000, z 3.84 (FINGERPRINT) | share read 2.77 |


**Reading.** Arm B, the arm that decided DEV-FINGERPRINT-01 (LNCaP against PrEC, excess +0.1105 to +0.1130), reads no fingerprint in the pairs
above: their cancer excess lies inside the range two healthy donors reach against each other. The run-loss meter detects about 0.023 of
molecules losing every methylated call (planted tests), so these cancer lines show at most that much run loss beyond donor variation.
Arm A is not a verdict (donor null: healthy donors sit up to 0.19 off curve B). The cancer lines read more fully methylated multi-CpG molecules
than their normals (readable share above 2.7 times), recorded without interpretation.
Not read: the breast pairs (MCF-7 held for the author's decision on the breast_mcf10a seal).
