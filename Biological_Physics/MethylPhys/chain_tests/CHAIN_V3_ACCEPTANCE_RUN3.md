# Chain v3 end to end, run 3 (2026-10-01) — IDAT pair in, report out, through run_sample.py --engine v3 only

Stages exercised on every specimen: Stage 0 intake (manifest, hash, controls, detection, call rate, sex check, gate) → Stage 1 calibration
→ Stage A composition (EPIC blood NNLS) → Stage M Met-A → Stage MC C-score → Stage T same-batch tare (second pass) → HTML report + bundle.
22/22 specimens processed; 0 failures. Mixtures run with intake skipped (lab-made DNA blends have no donor custody record); recorded.

| group | n | intake | untared Met-A | tared A_rel | Normal |
|---|---|---|---|---|---|
| healthy purified neutrophils (isolated; own floor) | 6 | 6 PROCEED | 0.994–1.006 | — | 6/6 (these arrays are IN the floor: not independent) |
| known mixtures, ≥ 50 % neutrophils | 6 | skipped | 0.943–0.968 | 0.987–1.021 | 6/6 tared |
| AML second-remission blood, another lab (GSE315367) | 10 | 2 PROCEED, 8 PENALTY (2 borderline call rate) | 1.073–1.115 (5) | 0.986–1.032 | 5/5 tared; 5 withheld (neutrophils 6–47 %, not dominant) |

**What this shows.** The chain runs from raw IDATs to a report without hand steps. Untared whole-blood A carries a lab/composition offset
(−0.04 in the Salas lab, +0.09 in the AML lab); the same-batch tare removes it. The dominant-cell rule withholds A where neutrophils are a minority.
**Not shown yet.** Independence for isolated neutrophils (needs held-out arrays; the 12-donor leave-one-out gave 12/12 earlier);
real healthy whole bloods; repeat pairs; any disease. C-score: healthy isolated 0.69–1.21, whole blood 0.78–1.49; band not set
(whole-blood residual includes composition error).
