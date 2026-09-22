# PREREG — PROC-TIER-01: Stage 7 tiers on the commissioned gauge (CHAIN_COMMISSIONING row 7)

**Sealed 2026-09-21 before any measurement.** Disclosed pre-read: `cpg_report_builder.py` carries two hand-coded tier functions (line ~427: normal ≥ 0.95, elevated > 1.04; line ~952: NORMAL < 1.04) while `tier_breakpoints.json` v1.3 (July note) puts NORMAL→ELEVATED at 1.01 — three definitions of one thing. That is T1's subject, not a result.

**Bars.**
- **T1 one definition.** Every tier assignment in the engine and report builder reads `tier_breakpoints.json`; grep finds zero hard-coded breakpoint literals in tier functions after the fix; a kit assertion test exercises every boundary (both sides, ±1e-6) against the JSON.
- **T2 healthy occupancy — MEASURE, DO NOT MOVE.** On the 1,379 healthy identity-gauge readings (A″ = A_mapped − c(decade) − z_lab, four labs), report the fraction in each tier under v1.3 breakpoints, per lab. Prediction (analyst, before measuring): with NORMAL→ELEVATED at 1.01 and band p90 = 1.0248, **≥ 25 % of healthy donors will read ELEVATED**; ≥ 1.07 (Warburg) and ≥ 1.10 (BREACH) will be ≪ 1 %. No breakpoint is changed by this procedure: the 1.07 and 1.10 lines are the framework's physics claims and the 1.01 onset is the author's July decision; the number is reported to the author for a ruling.
- **T3 no tier without a reportable gauge.** For every class the identity stage marks `reportable=False` (§108: no band for the component on that specimen; UNMAPPED scale; lab_zero UNSET) the bundle carries `tier=None` and the report prints no tier word. Test on the 7 cached whole-blood arrays: stem_adult / progenitor-alone / stromal etc. carry no tier; immune carries one.
- **T4 ceiling.** A_max = 1/H_min per class is honoured: any A above it is reported as `AT CEILING` with the ceiling value, never as a number above it; assertion test with a synthetic β that would exceed it.
- Row 7 commissioned if T1, T3, T4 pass; T2 is a report with a prediction scored.

---
**SEALED** sha256 `33ad0bafbb09fd7e9b1d89156599139cbbcd00ebcd573b687a5b3c31868803fd` · 2026-09-21
