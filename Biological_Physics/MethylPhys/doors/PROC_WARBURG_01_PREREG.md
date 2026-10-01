# PROC-WARBURG-01 — pre-registration (written 2026-09-30, before any methylation array of this set was read)

**Question.** The gauge places the Warburg line at A = 1.07: the point at which the cell has switched toward aerobic glycolysis. Does a measured
switch in the cell's energy source appear on the gauge at that value?

**Data.** Cellular Lifespan Study (Sturm, Picard et al.; Sci Data 2022). Primary human fibroblasts, 6 healthy donors + 3 SURF1, grown across their
lifespan with time-matched controls. Methylation: GSE179847, 479 EPIC arrays, raw IDATs through our Stage 1. Metabolism: FigShare 18441998,
Seahorse-derived ATP production from glycolysis (ATPglyc) and OxPhos (ATPox) on the same cultures. 340 arrays have a Seahorse measurement of the
same line, treatment and study part within 10 days grown.

**Reading (fixed now).** Per array: identity loci = CpGs stable across its standard (max − min β ≤ 0.05). Standard = the same donor line, untreated
control at 21 % O2, same study part, the 3 control arrays nearest in days grown (the array itself excluded). A_meth = H(mean β on loci the standard
holds methylated) / the same on the standard arrays; A_unmeth likewise on unmethylated loci; per-locus A as before.
Metabolic axis: glycolytic ATP fraction f_g = ATPglyc / (ATPglyc + ATPox); Δf_g = array − its time-matched control.

**Predictions (from the gauge as written).**
- P1. Arrays whose energy source switched toward glycolysis (Δf_g ≥ +0.25) read A_meth ≥ 1.07 in ≥ 80 % of arrays.
- P2. Arrays without a switch (|Δf_g| < 0.10) read A_meth in Normal (0.95–1.05) in ≥ 80 % of arrays.
- P3. Across all arrays A_meth rises with Δf_g: Spearman ρ > 0, p < 0.01 by permutation within donor line.

**How each outcome will be read.** P1 + P2 pass: a metabolic switch reads at the Warburg line on the methylated channel. P2 passes, P1 fails: the
switch alone does not reach 1.07, so the ~1.08 seen at transformation (BJ HRAS, HBEC) is not explained by metabolism. P2 fails: the time-matched
standard is not stable in this system and the test is uninformative. The line is not moved by this test; any change is a separate decision.

**Known limits, stated now.** Oligomycin and the mitochondrial-substrate blockers force the switch pharmacologically in normal cells; that is a
metabolic switch without transformation. Seahorse and methylation come from the same culture on nearby days, not the same flask on the same day.
Control cultures' own glycolytic fraction drifts with age in culture (0.13–0.91).
