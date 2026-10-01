# DEV-NOISE-01 — array noise index (development, 2026-10-01; follows the PROC-NEUT-TEST-01 T2 failure)

**Index.** N = mean H(β) at **48,528 EPIC sites** that every Salas purified blood group holds fixed: each group's mean β is ≤ 0.03 (40,882 sites) or ≥ 0.97
(7,646 sites), with every group's SD ≤ 0.02. The neutrophil identity sites are excluded. Biology holds these sites still, so their entropy is the array's own
noise (chemistry plus scanner). Read from each array by itself; no reference population. Frozen as noise_sites_EPIC_v1.json.

| arrays | n | N median (range) | Met-A median |
|---|---|---|---|
| Salas floor neutrophils | 12 | 0.1285 (0.1223–0.1489) | 1.001 |
| second lab, 54 y man | 24 | 0.1504 (0.1067–0.1655) | 1.069 |
| second lab, 30 y man | 24 | 0.1656 (0.1365–0.2434) | 1.192 |

- Within the second lab, N follows Met-A: rho 0.79 (30 y) and 0.83 (54 y); 0.76 together (n = 45).
- N does not explain everything: some second-lab arrays with N at the floor's level still read about 1.07. That residual offset is the reason the tare is also needed.

**Rule proposed for the chain (to be applied after the pre-registered battery is scored, then tested on independent arrays).**
1. Stage 0 reports N for every array.
2. If N is above the floor arrays' range (> 0.149), the gauge state is withheld unless the array is tared against same-run references.
3. Every reading, isolated or whole blood, is tared against same-run healthy references when they exist. The untared reading is printed as a number only.
