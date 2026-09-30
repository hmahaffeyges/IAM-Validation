# PROC-V12-IDENTITY — outcome of the first build (2026-09-29): B3 FAILED; the rule, not the atlas

Run against PROC_V12_IDENTITY_PREREG.md as written. **B3 failed: 30 of 74 cells built ≥ 100 identity loci; 44 built fewer (most 0).**
B1/B2 were not run, because they read the sets B3 builds.

**Cause: the v2 tightening I wrote into the rule cannot be met by the cells it was meant to protect.** It required the whole 95 %
posterior interval of μ to lie inside a 0.10-wide window (b* ± 0.05). For the 40 cells measured on 2–5 Loyfer WGBS samples the
median interval width is 0.105–0.174 — wider than the window — so almost no locus can qualify however well the cell sits at its
floor. Their point means sit in the window at 37,000–121,000 loci each (estimated from every 7th block). The four Tian pooled cells
(astrocytes, microglia, OPC, vascular leptomeningeal) have one pooled sample, so n_obs ≥ 2 excluded every locus.
The 30 that built are the array-measured and multi-source cells (interval width 0.016–0.098).

This is recorded as measured; the rule is not edited. The replacement is PROC-V12-IDENTITY-02, written before its build.
Numbers: `v12_identity_build.csv` (per cell n_loci, class, sources) in this procedure's record folder.
