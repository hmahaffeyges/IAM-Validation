# PROC-V12-IDENTITY-02 — outcome (2026-09-29): B3 PASS, B1 FAIL, B2 FAIL

Run against PROC_V12_IDENTITY_02_PREREG.md as written. Nothing moved after the run.

- **B3 PASS.** All 74 admitted cells build a production set from the v2 posterior: 11,478–120,966 loci (median 52,437).
  Self-read on the fitted mean 0.987–1.001 (circular; printed for comparison only).
- **B1 FAIL.** 71.0 % of the 310 held-out sample readings fall in NORMAL (bar ≥ 95 %). 5 readings had < 100 loci and are unreadable.
- **B2 FAIL.** 64 of 70 cross-fittable cells have their median held-out reading in NORMAL; six do not: memory B cells 0.900,
  MEP 0.915, T central memory CD4 0.940, colon macrophages 0.945, kidney glomerular endothelium 0.947, lung alveolar
  endothelium 0.949. Every one of the six is **below** NORMAL, none above.
- Not testable (one pooled sample): astrocytes, microglia, OPC, vascular leptomeningeal cells.
- By source, fraction of held-out readings in NORMAL: ENCODE 1.00 (2), Tian 1.00 (1), GSE63409 0.83 (18), Salas2022 0.84 (56),
  Salas2018 0.71 (35), Moss2018 0.68 (22), Loyfer2023 0.65 (176).
- Cells with many samples read tightly: neutrophils 12/12 in NORMAL (median 0.997), eosinophils 4/4, HSC 3/3, memory CD4 4/4.
  The widest: CD8 T cells 0.884–1.160 (2/9 in NORMAL), cortical neurons 0.852–1.076, smooth muscle 0.914–1.120.

## What this is, and what it is not
v1's identity loci were never tested this way: v1.1's only check was each cell's own atlas mean read on loci chosen from that same
mean, which reads ~1.00 by construction. So this is the **first held-out measurement of how well an identity set transfers to a
new sample of the same cell**, not a comparison in which v2 did worse than v1.

Two explanations fit the numbers and the outcome does not choose between them:
1. **Selection noise.** In this test, a cell with 2–3 samples chooses its loci from ONE sample. Loci enter the window partly by that
   sample's noise; methylation is skewed high, so more enter from above than below, and a new sample's mean at those loci sits above
   b*. That gives H below H_min, so A reads below 1. All six B2 failures are below. Production sets are chosen from the posterior
   over all samples, so if this is the cause, production transfers better than this test shows.
2. **Real donor-to-donor spread** of a sorted cell at the floor loci, larger than ±5 %.
The diagnostic PROC-V12-DIAG-01 (written before it runs) separates the two.
