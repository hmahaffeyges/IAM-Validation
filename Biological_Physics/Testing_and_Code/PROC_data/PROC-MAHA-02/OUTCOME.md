# OUTCOME — PROC-MAHA-02: row 5 commissioned; row 5b opened

**Run 2026-09-21.** Author's decision: option 1 + 5b.
| test | bar | result | verdict |
|---|---|---|---|
| M6 rate in the record | four labs' p95/p99 tails in identity_band_v3 to 3 dp | UCLA 0.044/0.005 · Munich 0.065/0.010 · Karolinska 0.098/0.051 · Uppsala 0.056/0.015, with n and chip_median_sd | **PASS** |
| M7 rate reaches the report | Karolinska array: `departure.lab_false_alarm_p95 == 0.098` (exact, as sealed); sentence rendered; unknown lab → four-lab range | stored value **0.0984** (the MAHA-01 measurement to 4 dp); the sealed literal 0.098 was the analyst's 3-dp transcription of it. Sentence renders: "At this laboratory 10 of 100 healthy donors read beyond p95…"; unknown lab: "…4–10 of 100…" | **FAIL as sealed** on the exact literal; the rendered sentence and the unknown-lab fallback pass |
| M8 MAHA-01 M1/M3/M5 unchanged | 5/5, 40/40, 7/7 | 5/5, 40/40, 7/7 | **PASS** |

**Correction to the first write-up (auditor, 2026-09-21):** this OUTCOME first recorded M7 as PASS "because the prereg says 3 dp". It does not — the 3-dp qualifier is M6's; M7 as sealed is an exact equality against a literal the analyst had rounded when writing the seal. M7 is therefore FAIL as sealed, and the earlier sentence was a post-hoc loosening of a failed bar with the seal cited as authority, which the protocol forbids. Recorded here and in FALSIFICATION.

**ROW 5 COMMISSIONED on M6 and M8, with M7 recorded as FAIL as sealed** — the mechanism M7 tests (the rate reaching the report) is demonstrated by the rendered sentences, and a re-seal of the exact-literal condition would test nothing further; the record shows the failure rather than a rewritten bar. The author may overrule the commissioning. `run_full(cfg={"lab": <cohort key>})` looks the rate up; without it the four-lab range is printed.
**ROW 5b OPENED — the chip term.** Bar fixed now: a reference on the chip (a control array per chip, or the 40-panel spread across the laboratory's chips) brings every laboratory's p95 tail to ≤ 0.05. MAHA-01's chip-centring (2–4 %) shows the bar is reachable. Design decision on the panel protocol precedes it.

---
**SEALED (corrected write-up)** sha256 `c66f52bac2f1f28dae9913e97fccadca210cbed99bec56cc44b2e8867f48e273` · 2026-09-21 — supersedes the first seal of this file (git history)
