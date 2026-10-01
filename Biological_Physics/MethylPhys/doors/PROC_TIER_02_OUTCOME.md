# OUTCOME — PROC-TIER-02: NORMAL set to the commissioned healthy population. U1, U2, U3 PASS.

**Run 2026-09-21.** `tier_breakpoints.json` → **v1.4**: NORMAL [0.95, 1.04), ELEVATED [1.04, 1.07); Warburg 1.07 and breach 1.10 unchanged; `_meta` carries the supersession note and the measured occupancy. No code changed; `cpg_tiers` read it.

**U2 healthy occupancy, 1,379 donors, four labs:** SUPPRESSED 2.18 %, NORMAL 95.21 %, ELEVATED 2.54 %, ≥ 1.07 0.07 % (one donor), ≥ 1.10 none. ELEVATED+ 2.61 % — sealed prediction 2.0–3.0 %: **correct**. Per lab ELEVATED 1.6–5.0 % (Munich highest at 5.0 %; Karolinska carries the SUPPRESSED tail, 5.7 %).

**U3:** `test_tiers.py` PASS — after fixing the test itself: it had typed the 1.01 onset as a literal, exactly the defect T1 forbids in engine code; it now reads every boundary from the JSON.

**Decision record.** The July 1.01 onset was set on the marker-union statistic (retired by PROC-N7-01); on the identity gauge it labelled 30 % of healthy donors "Elevated / rising trajectory / pre-diagnostic". NORMAL is now the healthy central 95 %, the reference-interval convention. A donor can be ABOVE_BAND (p90 of the band) and NORMAL (inside the 95 %): both words are defined, both print, the band is the finer statement. Author may overrule.

---
**SEALED** sha256 `23def5090be7a3d0320579b5c27c6b3ef25af0f14860cea08fa384107bc835df` · 2026-09-21
