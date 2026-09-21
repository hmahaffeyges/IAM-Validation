# Row 9 — the report — WORKING NOTE (exploration; not sealed — sealing rule 2026-09-21)

**2026-09-21, first end-to-end run.** `CPG_Engine/cpg_report_v3.py` written to the author's specification (Issue 003 p4): cells detected + percentages; A per class on the three-layer reference with band placement and tier; A per cell (separation surface, no band); Stage 5 departure with the laboratory's false-alarm sentence; the sky; flags; a scope paragraph. It names no condition and gives no age in years; a vocabulary guard refuses to write the file if any forbidden word appears — it caught the author's own scope sentence twice before the wording was clean.

The old `cpg_report_builder.py` (Stage 8 concordance, disease cards, straw-man wall, cellular age; 125 disease/age references) is record-side and is not called by the chain.

**11 cached arrays → 11 reports (152 s).** Three Uppsala arrays (commissioned lab zero −0.0117, sky scale): immune A″ 0.9771 / 0.9951 / 1.0236, all IN_BAND, NORMAL, sky available (2.1–3.2 % beyond |z| = 2). Eight arrays from laboratories with no 40-array panel: gauge, departure and sky all print NOT REPORTABLE with the reason; no number is invented.

**Caught by reading the first report against the commissioning table (the point of row 9):**
1. Stage 2 stores class fractions in PERCENT while per-cell fractions are in [0,1]; the first render printed "immune 8340.0 %". Fixed; the conductor should store one unit — open item.
2. My footnote asserted "healthy cells read ~0.4–0.6 on this surface"; the cells on these arrays read 0.77–1.12. The 0.52 figure was the foundation-cohort median over 115 cells; it does not describe a single array's placed cells. Footnote now states only that the per-cell A carries no band.
3. `haematopoietic_progenitor` prints NOT REPORTABLE (no band) on every blood array — correct today, and a reminder that the second blood axis (RECON B4) has no band yet.

**Not yet:** a legal/vocabulary gate as a kit test; the report on a cfDNA sample; the report read by the author. Row 9 is sealed when the generator runs on the 11 arrays with no line contradicting the commissioning table and the author has read one.
