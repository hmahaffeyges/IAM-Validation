# PROC-REPL-V3-01 — technical replicates on chain v3 with the median tare (pre-registered 2026-10-03, before any array is read)

**Why.** Part 4 (serial reading) reports 63 of 64 replicate readings with a within-person SD of 0.008. That run (DEV-CHAIN-V3-RUN3) used the
fitted noise-corrected tare, which was removed from the chain (DEV_TARE_02_OUTCOME.md). The result has to be measured again on the chain as it stands.

**Data.** GSE250556 (EPIC whole-blood technical replicates, 64 arrays), raw IDATs from GEO. No other series.

**Run.** Chain v3 end to end, as in RUN3: IDAT -> intake -> calibration -> composition -> Met-A -> Stage T median tare
(A_rel = A / median of >= 3 same-run healthy references, same slide else same batch, the array itself excluded; nothing fitted) -> noise gate
(noise_gate_EPIC_v1.json) -> report. `run_sample.py --engine v3`, two passes (pass 2 with `--slide-ref-table`). Code and frozen inputs at the
commit recorded in the outcome file. Nothing is tuned after reading.

**Measured.** Number read end to end; number tared; tared A_rel per array; within-person SD (replicates of the same person); SD over all;
number in Normal (0.95-1.05); number withheld by the noise gate.

**Bars (set now).**
1. The chain reads at least 62 of 64 arrays end to end; every array not read has a stated reason.
2. Within-person SD of tared A_rel <= 0.020 (the purified-array floor precision already measured).
3. At least 95 % of tared readings in Normal.

All three met: the median tare is carried in Part 4 with these numbers. Any missed: the numbers are reported as measured, the chapter states
them, and the cause is investigated in a dated development note before anything changes. No bar is moved after reading.
