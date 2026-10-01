# OUTCOME — PROC-TIER-01: Stage 7 tiers on the commissioned gauge. T1, T3, T4 PASS — row 7 COMMISSIONED. T2 measured; prediction scored correct; ruling requested.

**Run 2026-09-21.**

| bar | result | verdict |
|---|---|---|
| T1 one definition | Before: three tier definitions (report builder ×2 with NORMAL ending at 1.04, colour map a fourth; JSON v1.3 at 1.01). After: `CPG_Engine/cpg_tiers.py` reads `tier_breakpoints.json`; identity stage, both report-builder sites and the colour map call it; 0 literal breakpoints remain; every boundary ±1e-6 agrees | PASS |
| T3 no tier without a reportable gauge | identity components with `reportable=False` (§108 no band, UNMAPPED, lab_zero UNSET) carry `tier=None`; on GSM2333901 only immune carries a tier; with lab_zero UNSET every tier is None | PASS |
| T4 ceiling | A ≥ 1/H_min → `AT_CEILING` with the ceiling value (immune 1/0.8389 = 1.1921); just below → BREACH | PASS |
| **T2 healthy occupancy (measured, nothing moved)** | 1,379 healthy donors, four labs, A″ on the identity gauge under v1.3 breakpoints: **SUPPRESSED 2.2 %, NORMAL 67.5 %, ELEVATED 30.2 %, ≥ 1.07 one donor (0.07 %), ≥ 1.10 none.** Per lab ELEVATED 25.9–31.4 % — the same in every laboratory. Analyst's sealed prediction "≥ 25 % ELEVATED": **correct**. Central 95 % of healthy A″ = [0.954, 1.041]. | REPORTED |

**Reading of T2.** The two physics lines are clean of healthy people: the Warburg line (1.07) admits 1 in 1,379 and the breach line (1.10) none. The NORMAL→ELEVATED onset at 1.01 was moved from 1.04 in July on the marker-union statistic; on the commissioned identity gauge it sits well inside the healthy band, so under it 30 % of healthy donors receive the word "Elevated" and the context label "Rising trajectory / pre-diagnostic". The pre-July band (0.95–1.04) is, to three decimals, the central 95 % of healthy on the commissioned gauge. **This procedure changes no breakpoint**: the onset is the author's decision, and the number is handed to him. Options: (a) restore 1.04 (healthy central 95 %); (b) set the onset at the band p90 (1.025) so ELEVATED begins where the healthy band ends; (c) keep 1.01 and accept that ELEVATED includes 30 % of healthy people, with that stated on every report.

**In code.** `cpg_tiers.py`; `stage_b_identity` records `tier`/`tier_note`; `cpg_report_builder._tier` and the gauge colour map read it; kit test `test_tiers.py`.

---
**SEALED** sha256 `d0416b81d8d1b99134a754fe72e53987f4270134e17360386d88584f042416fe` · 2026-09-21
