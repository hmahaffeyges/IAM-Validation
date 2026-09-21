# PROC-RECORD-02 — VAL-025 to VAL-028 (four-substrate aging trajectory) reclassified: modeled prediction, not measurement

**2026-09-21.** Prompted by the author supplying the VAL-025..028 output after PROC-AGE-01 and asking whether the age closure was premature.

**What the script does** (`val025_028_aging.py`, Zenodo 10.5281/zenodo.19633499, byte copy here). `HUMAN_AGES` and `CANINE_AGES` are literal tables: eight (age, n, μ_nucl, μ_fuzz, μ_WPS, μ_frag) rows for humans and six for dogs, typed from the described direction of published aging effects ("Sources: Wang 2020 syntenic data + published aging literature"). Each μ is then scored by a Monte Carlo draw of 10,000 values at a fixed SD and A = mean H / H_min. **No per-sample substrate measurement enters.** Hannum 2013 (GSE40279) is a 450K methylation array cohort and Wang 2020 is RRBS methylation in 104 Labradors; neither carries nucleosome occupancy, fuzziness, WPS or fragment-size data. The reported r = 0.9998 (human) and 0.986 (canine) are correlations between age and a smooth monotone table, i.e. properties of the table.

**The record disagreed with itself.** Issue 002's build (April) lists VAL-025..028 as **"(modeled) — Prediction filed."** DETAILED_VALIDATION_RECORD.md, README_validation_runs_original.md and VAL_INDEX.csv list them as **PASS** with r = 0.9998. Issue 002's reading is correct.

**Reclassified.** VAL_INDEX status → `MODELED PREDICTION (literature-typed age table; no per-sample substrate data; Issue 002: "prediction filed")`; DETAILED_VALIDATION_RECORD rows and the "substrate-independent aging" summary sentence amended in place; the "Non-methylation aging slopes 20–38× methylation" note amended (those ratios are ratios between typed tables).

**What this does NOT touch.** VAL-006 / CPG-VAL-015 / PROC-AGE-01 — the *methylation* aging trajectory — are per-sample measurements (Hannum n = 656; four labs n = 1,379) and stand (PROC-AGE-01 corrected write-up). VAL-013's dog methylation r = 0.927 is on the real Wang 2020 RRBS data (to be checked separately: PROC-RECORD-03 candidate). The four-substrate aging *prediction* stands as a prediction: it is testable on plasma cfDNA fragmentomics cohorts with donor ages (Snyder 2016; Mouliere 2018; Mathios 2022 — mostly controlled access) and is filed as a Future Goal with that requirement.

**Lesson (RUNBOOK).** A PASS row must name what was measured on which samples. A Monte Carlo around a typed table is a model, and the record must say "modeled" in the status column, not only in a footnote elsewhere.

---
**SEALED** sha256 `48c4584c69165c8b835e45ab90b865c0b2ed056b9e0363fa0e3f3a3b474805aa` · 2026-09-21
