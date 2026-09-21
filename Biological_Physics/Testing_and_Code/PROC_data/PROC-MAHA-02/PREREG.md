# PREREG — PROC-MAHA-02: Stage 5 commissioned with the laboratory's false-alarm rate on the report; row 5b (chip term) opened

**Sealed 2026-09-21 after PROC-MAHA-01 was written up and before this was run.** Author's decision: option 1 + 5b ("always do what you surmise to be the best option").
**Design.** The departure statistic is unchanged from MAHA-01. Each laboratory's record (`identity_band_v3.json` `_meta.cohorts`, and any future lab-zero record from `lab_zero.py`) carries its empirical healthy tail at p95 and p99 on the immune axis, measured on the arrays that set its zero. The departure output and the report print it: "at this laboratory, N of 100 healthy read beyond p95 on this axis (chip-driven; row 5b)". Where a laboratory's tail is unknown (a 40-panel cannot estimate a 5 % tail), the report prints the four-lab range 4–10 % and says so.
**Tests (fixed).**
- **M6 — the rate is in the record.** `identity_band_v3.json` carries `tail_p95`, `tail_p99`, `n` for each of the four laboratories, equal to the MAHA-01 values (UCLA 0.044/0.005, Munich 0.065/0.010, Karolinska 0.098/0.051, Uppsala 0.056/0.015) to 3 dp.
- **M7 — the rate reaches the report.** `run_full` for a Karolinska array carries `departure.lab_false_alarm_p95 == 0.098`, and `_departure_section` renders a sentence containing "beyond p95" and the laboratory's rate. For an array whose lab has no measured tail, the sentence carries "4–10 %".
- **M8 — MAHA-01 M1, M3, M5 unchanged.** Re-run: 5/5, 40/40, 7/7.
- **Row 5b opened** in CHAIN_COMMISSIONING with its bar fixed now: a chip reference (control array per chip, or the panel spread across chips) brings every laboratory's p95 tail to ≤ 0.05 — the value chip-centring achieved in MAHA-01 (2–4 %) says the bar is reachable.
**Outcome.** M6–M8 pass → row 5 COMMISSIONED (whole blood, immune axis, false-alarm rate stated); 5b OPEN with its bar. Written as found.

---
**SEALED** sha256 `34951095d6816046c0b9aff0a1b1613dd080db2135101afed46007ed1344d34d` · 2026-09-21
