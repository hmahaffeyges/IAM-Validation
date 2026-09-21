# OUTCOME — PROC-MAHA-02: row 5 commissioned; row 5b opened

**Run 2026-09-21.** Author's decision: option 1 + 5b.
| test | bar | result | verdict |
|---|---|---|---|
| M6 rate in the record | four labs' p95/p99 tails in identity_band_v3 to 3 dp | UCLA 0.044/0.005 · Munich 0.065/0.010 · Karolinska 0.098/0.051 · Uppsala 0.056/0.015, with n and chip_median_sd | **PASS** |
| M7 rate reaches the report | Karolinska array: `lab_false_alarm_p95` = 0.098; sentence rendered; unknown lab → four-lab range | "At this laboratory 10 of 100 healthy donors read beyond p95 on this axis (5 of 100 beyond p99); the excess over 5 is the chip term (row 5b)." Unknown lab: "…across four commissioned laboratories 4–10 of 100…" | **PASS** (first check compared a literal 0.098 to the stored 0.0984 — the prereg says 3 dp; re-checked as sealed) |
| M8 MAHA-01 M1/M3/M5 unchanged | 5/5, 40/40, 7/7 | 5/5, 40/40, 7/7 | **PASS** |

**ROW 5 COMMISSIONED** — whole blood, immune axis, the laboratory's false-alarm rate stated on every report. `run_full(cfg={"lab": <cohort key>})` looks the rate up; without it the four-lab range is printed.
**ROW 5b OPENED — the chip term.** Bar fixed now: a reference on the chip (a control array per chip, or the 40-panel spread across the laboratory's chips) brings every laboratory's p95 tail to ≤ 0.05. MAHA-01's chip-centring (2–4 %) shows the bar is reachable. Design decision on the panel protocol precedes it.

---
**SEALED** sha256 `8dc3f540ea891e397718bf82a8f46037831989d880c38e5a3e3b58639fa9bc35` · 2026-09-21
