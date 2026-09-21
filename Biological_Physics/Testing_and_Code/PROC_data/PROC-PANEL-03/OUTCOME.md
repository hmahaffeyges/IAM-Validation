# OUTCOME — PROC-PANEL-03: the age-referenced lab zero — **LAB ZERO COMMISSIONED**
**Design.** z_L = median_i [A_i − c(dec_i)] − 1 over a 40-array healthy panel, where c(dec) is the reference healthy age curve (per-decade median about the grand median) built on the *other* laboratories only (leave-one-out; the held-out lab never contributes to c). A″ = A − c(dec) − z_L.
| test | bar | result | verdict |
|---|---|---|---|
| P2″ k=40, LOO c | ≥95 % within ±0.010 | UCLA 97.0 · Munich 99.8 · Karolinska 98.4 · **Uppsala 97.6** (medians 0.9999–1.0004) | **PASS** |
| P4″ single-decade 40-panel vs full-cohort zero | ≥95 % within 0.010 | UCLA(70s) 96.0 · Munich(50s) 99.9 · Karolinska(50s) 100 · Uppsala(40s) 99.5 | **PASS — a panel does not need age matching once it is read against the curve** |
| P3″ LOO age-referenced band (p10–p90, width ≈0.053) | ≥0.70 all four | **UCLA 0.823 · Munich 0.839 · Karolinska 0.767 · Uppsala 0.753** (nominal 0.80) | **PASS** |
**Reference age curve (Uppsala held out, as an example):** 20s −0.008, 30s −0.008, 40s −0.002, 50s +0.001, 60s +0.003, 70s +0.006, 80s +0.003. The production curve `reference_age_curve_v1.json` is built on all four labs.
**Verdict: LAB ZERO COMMISSIONED** as 40 healthy arrays per laboratory, any age mix, read against the reference age curve; offset printed on every report. Supersedes the "20–30 arrays" wording of LAB-ZERO-02 (CLSI EP28's 20 is the verification minimum; the measured within-lab SD of 0.019–0.025 requires 40 for SD(z) ≤ 0.005). CHAIN_COMMISSIONING row B opens: the gauge switch.
**What the panel does NOT fix:** the Sentrix-chip term within a laboratory (Karolinska p 8e-16, Uppsala 2e-19) — a constant cannot; it is the next residual and is inside the band width.

---
**SEALED** sha256 `1d832b79c1708af7733a0f0cea1ee1e0c05eedb3242e3655838cea0c31e7f8b4` · 2026-09-20
