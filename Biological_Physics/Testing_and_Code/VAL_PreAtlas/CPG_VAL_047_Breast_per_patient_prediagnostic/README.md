# CPG-VAL-047 — Blinded per-patient validation of the GAPE A-score on pre-diagnostic breast arrays

**Pre-registered:** 2026-04-20T04:30:00Z (`VAL047_prereg.json`, GAPE 5.0, H_min from G-003b MCMC)
**Cohorts:** GSE51057 (SHA-locked 828059…98bb0), GSE51032 · **Panels:** Xu-6 (3/6 covered on 450K) and Xu-538
**Status in Issue 002 §8.1:** "what VAL-047 validated" — this directory restores the record that the repository lacked until 2026-09-19.

| folder | contents |
|---|---|
| `VAL047_prereg.json` | sealed pre-registration, input SHAs, decision rules |
| `phase_1-8_json/` | phases 1–8 result files (ranking, replication, options 1–3, sensitivity, comparison, TTD stratification, Kresovich-100), `kresovich_100_cpgs.json` |
| `phase_9_12/` | Phase 9 (GSE51057, 146 C50 vs 177 HC) and Phase 12 (GSE51032, 224 vs 424): `PHASE_9_12_LIVE_RESULTS.md`, per-window results JSON, run logs |

Headline (Phase 9, Xu-538, GSE51057): d = +0.088 (0–2 yr), +0.306 (2–5), +0.712 (5–10), **+1.783 (>10 yr, n=11)**, all pre-dx +0.452 (p = 0.0001).

Already in the repository, unchanged (identical checksums): `../VAL_047_option3_results.json`, `../VAL_047_replication_results.json`; scripts `../VAL_047_*.py`, `../cross_population/scripts/VAL047_tightening_*.py`.
Restored from the author's archive 2026-09-19; no file was modified.
