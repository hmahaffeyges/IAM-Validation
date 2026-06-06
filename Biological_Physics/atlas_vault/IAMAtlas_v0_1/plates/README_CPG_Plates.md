# CPG Plates 1–4 — Framework Visualization Reference

**Date pushed to repo:** 2026-06-06
**Author:** Heath W. Mahaffey
**Conventions:** HEALPix NSIDE=128 (npix=196,608), Mollweide projection, CpGs ordered by chromosome × position (chr1 → chrX), genomic-order pixel assignment.

These four plates are the **canonical visualization references** for the CPG framework. They are the visual analog of the IAMAtlas data: where the brightness CSVs and the IAMAtlasREBUILD.csv are the underlying numerical matrices, the Plates are how those matrices look projected onto the celestial sphere using Planck visualization conventions.

## The four plates

**Plate 1 — `CPG_Plate_01_Cosmic_Microwave_Methylome.png`**
Eight Mollweide panels — one per architectural class (stem_pluri, stem_adult, stromal, progenitor, cycling, secretory, terminal, immune). Each panel shows the per-CpG posterior mean β across 481,966 CpGs from the IAMAtlas REBUILD MCMC. The stromal panel's "galactic mask" (4.93% MCMC coverage) is the methylome's declared known-unknown.

**The healthy reference per class.** Patient runtime consults this data (via the per-class brightness CSVs) at Stage 4.6.

**Plate 2 — `CPG_Plate_02_Breast_Anisotropy.png`**
Full-sky scatter of 1,392 concordant breast pre-diagnostic CpGs, signed Cohen's d, sized by |d|, cyan-blue hypomethylated vs orange-yellow hypermethylated. 5.4:1 hypomethylation dominance. Lower panel: chr6 zoom showing MHC region enrichment.

The cross-genome anisotropy field-effect signature for breast pre-dx — used at Stage 8 Route A per-card residual map matching.

**Plate 3 — `CPG_Plate_03_Grandaddy_CMM_vs_CMB.png`**
Side-by-side methylome vs CMB at matched Mollweide projection, matched colormap, matched pixelization. Top: full-sky overview. Bottom: zoom on small-scale anisotropy texture.

Makes the CMB↔methylome analogy visually irrefutable. The right panel (CMB realization) was produced via `healpy.synfast()` from Planck's ΛCDM C_ℓ spectrum — same generative discipline as the synthetic patient generator at Stage 4.6 / L9 N7.

**Plate 4 — `CPG_Plate_04_Patterns_Discovered.png`**
Six findings the spherical methylome projection makes visible:
- **A** — Class-Difference Map (IMMUNE class posterior minus TERMINAL class posterior per CpG)
- **B** — chr16+chr17 Cold-Patch Zones (the VAL-006 deconvolver-explained-away anomaly; chr16/chr17 systematically carry less concordant residual than expected)
- **C** — Concordant Signal Density (1,392 breast pre-dx CpGs per HEALPix pixel)
- **D** — Differentiation Gradient (STEM_PLURI minus TERMINAL posterior — which regions get re-methylated during differentiation)
- **E** — MCMC Coverage Map (how many of the 8 classes have converged posterior per pixel; reveals stromal galactic mask)
- **F** — Breast Pre-Diagnostic Anisotropy (1,392 concordant CpGs colored by signed Cohen's d on the methylome sphere)

## Patient runtime consumption

At Stage 4.6 (per-class healthy brightness comparison + patient Mollweide projection), the engine produces a per-patient version of Plate 1:

1. For each of 8 architectural classes, compute per-CpG z-score: z[i] = (β_patient[i] − μ_class[i]) / σ_class[i]
2. Project the 8 z-vectors onto the same HEALPix NSIDE=128 Mollweide grid as Plate 1
3. Generate `patient_id_cosmic_methylome.png` — 8-panel personal CMM where red = significantly hypermethylated departures, blue = significantly hypomethylated departures, neutral = within healthy variance

The customer's personal CMM ships in the report as the visualization endpoint of their immune-class A-score + immune cellular age + Mahalanobis distance — the same data, mapped to a sphere.

## Plate generation conventions

All plates use:
- **HEALPix NSIDE=128** (npix = 12 × 128² = 196,608 pixels)
- **Mollweide projection** (equal-area, full-sky)
- **CpG-to-pixel mapping:** CpGs ordered by chromosome (chr1 → chr22 → chrX → chrY), then by MAPINFO within chromosome. Sequential assignment to HEALPix pixels in genomic order.
- **Colormap:** β posterior mean uses a diverging cyan-orange palette centered at 0.5 (β=0 → cyan, β=1 → orange). Z-score departure uses diverging blue-red palette centered at 0 (z<0 → blue, z>0 → red).
- **Multiple CpGs per pixel:** averaged (per-pixel mean of CpG values that fall in that pixel).

The `patient_brightness_comparison.py` module at `walther_clinical_runtime/Brightness_Comparison/` mirrors these conventions exactly so per-patient projections sit on the same grid as the reference Plates.
