# Finding 2026-09-27 — GSE125105 (Munich) arrays are low-signal, and the intake gate that should refuse them never fires

**Why it was looked at.** The author: one laboratory fails every procedure while three pass — "once is a result, twice is
interesting, every time is screaming that something may be off." GSE125105 was the outlier in PROC-TARE-01 (most compressed
T_scale 0.890, largest T_offset +0.043), carried the largest of the retired laboratory zeros (−0.035), and in PROC-SKY-01 was
the laboratory on which the on-array noise model failed (3 of 12 in range; 12/12, 11/12, 10/12 elsewhere).

**What was measured** (three arrays per laboratory through Stage 1 with the control probes kept; kit/results/FINDING_GSE125105_controls.csv (archived privately)):

| median | GSE87571 | GSE42861 | GSE111629 | **GSE125105** |
|---|---|---|---|---|
| non-polymorphic control G / R (raw signal, methylation-independent) | 8,582 / 14,427 | 5,959 / 10,773 | 4,823 / 7,252 | **1,291 / 2,268** |
| bisulfite-conversion II, R | 32,559 | 20,159 | 16,590 | **7,966** |
| hybridisation, G | 27,410 | 14,726 | 20,392 | **12,494** |
| negative controls G / R (background) | 311 / 370 | 165 / 270 | 215 / 280 | 175 / 275 |
| probes at background, poobah p > 0.05 | 0.9 % | 1.5 % | 5.4 % | **12.5 % (10.8–17.2)** |
| SNP homozygous-cluster SD (β = 0 / 1), PROC-SKY-01 | 0.017 / 0.016 | 0.025 / 0.025 | 0.027 / 0.048 | **0.069 / 0.072** |
| chip decoded → scanned (IDAT header) | 3 months | — | 1 month | **8 months** |

Same background, one-sixth the signal. Every anomaly this laboratory has shown follows from that: wide SNP clusters, compressed
dynamic range, the largest zero, an on-array noise term (built from its SNP probes) that over-predicts its cg scatter.

**Why the chain let it through.** [`stage_0_intake.py`](../chain/stage_0_intake.py) carries the SOP §15/§17 gates (detected fraction; call rate ≥ 0.98
PROCEED, 0.95–0.98 PENALTY, < 0.95 QUARANTINE). Munich at 0.875 is a QUARANTINE by the SOP as written. But Stage 1 never hands
Stage 0 the per-probe detection numbers, the check is recorded `DEFERRED_PENDING_STAGE1_DECODER`, and the code sets
`advance = True`. A deferred check was acting as a pass — the exact thing the standing rule forbids. Commissioning runs also
use `--no-intake`.

**What it is not.** Not the physics, not the atlas, not the pipeline map. An address whose signal sits at background carries no
information about β; averaging it in injects ≈ 0.5. This is an instrument-quality fact about specific arrays, measured on the
array itself.

**Decisions.**
1. PROC-INTAKE-01 (pre-registered separately): Stage 1 returns the per-probe detection mask and control summary; Stage 0.5/0.7
   run on real numbers; failed probes are masked before any mean; a deferred check never advances an IDAT input.
2. GSE125105 stays a commissioning laboratory for its own detection-panel line only until PROC-INTAKE-01 has run on its 80
   panel arrays; anything cross-laboratory that used it is re-read afterwards (the retired band and zero already are gone).
3. PROC-SKY-01 B2 is **not** re-scored. Its outcome stands as sealed; this finding is recorded beside it as the probable cause of
   the fourth laboratory's miss, to be tested by re-running the same script once the gate exists.
