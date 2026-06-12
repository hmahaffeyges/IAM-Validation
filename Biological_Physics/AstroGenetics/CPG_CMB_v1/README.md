# CPG_CMB_v1 — Cellular Performance Gauge (current production chain)

This is the current, production IAMAtlas-post-build chain. It reads a patient's DNA-methylation array and reports, per cell type, how far the methylation pattern has departed from its derived architectural floor — the **Cosmic Methylome Background (CMB)** and the per-cell **A-score**. Everything in this folder is the live chain; the pre-build development phase lives separately under `../../PreIAMAtlas_Build/`.

## The score is derived, not comparative

For a representative methylation value `v` in architecture class `c`:

```
A = H(v) / H_min(c)
```

where `H` is binary Shannon entropy (bits) and `H_min(c)` is the derived architectural floor for the class — the minimum entropy a healthy cell of that class holds while keeping its identity. The eight `H_min` values are MCMC posteriors, frozen 2026-04-06, and are the single source of truth in `IAM_Atlas/IAMAtlasREBUILD_provenance.json` (the chain reads them from there and refuses to run on mismatch). `A ≈ 1` means a cell sits at its floor (healthy); `A` rising toward the class ceiling `1/H_min` means the pattern is walking toward the coin-flip state (departure).

This is a departure from a *derived* reference scale, not a statistical distance to a population. The A-score scores each class's most-methylated loci — never the discriminative deconvolution markers (those are for component separation only). No cohort is pooled and no reference panel is regressed against.

## No foregrounds subtracted

The chain subtracts no age / sex / smoking / batch foreground. That change is part of the cellular departure the score measures, not contamination in front of it. Intake facts (age, sex, smoking, pregnancy, active treatment) are carried as **report annotations** for the clinician and are never operands in the score, the tier, or the departure. This is the one stage where the CMB analogy is deliberately not followed (see the chain-of-custody SOP, foreground-firewall section).

## Built on the CMB data-processing pipeline

The chain is constructed stage for stage on the Planck-style cosmic-microwave-background pipeline. The mapping is literal:

| CMB / cosmology method | Use in this chain |
|---|---|
| L1 raw detector timestream | IDAT raw intensities (probe ≈ bolometer) |
| L2 calibration | β-value calibration per CpG |
| L3 all-sky map | the per-CpG β matrix |
| Component separation (Commander / NILC) | the deconvolver (cell-type fractions) |
| Foreground templates / galactic dust | age / sex / smoking — **annotated, not subtracted** (see above) |
| Galactic mask | the stromal MCMC coverage gap |
| Power spectrum decomposition | per-class entropy `H` |
| Dimensionless cosmological parameter | the A-score |
| Critical-density reference scale (ρ_crit) | the architectural floor `H_min` |
| Mollweide projection + HEALPix pixelization | the patient's all-sky departure maps (NSIDE = 128) |
| Null-test suite (jackknife, half-mission, scan-direction, injection/recovery) | the sealing null suite |
| MCMC posteriors with R̂ convergence | how the atlas and the `H_min` floors were estimated |
| "Confirm, not validate" referee discipline | the language standard used throughout |

Acoustic-peak, bispectrum, and lensing-style analyses are on the roadmap and are **not** part of the current chain.

## Folder map

- **`IAM_Atlas/`** — the derived reference atlas (`IAMAtlasREBUILD.csv.xz`), its provenance (`IAMAtlasREBUILD_provenance.json`, the frozen `H_min`), the cell-type→class map, and the per-class brightness/MCMC archives. The flatness-lesson note documents how the atlas was built and verified.
- **`Walther_Clinical Python Script/`** — the clinical orchestrator (`walther_clinical.py`) and its build spec.
- **`Runtime Matrices/`** — A-scoring loci and module, deconvolution markers, tier breakpoints, bidirectional decomposition, brightness/Mollweide rendering.
- **`CPG Chain of Custody SOP/`** — the operator SOP (stage-by-stage chain of custody, failure modes, the CMB↔methylome glossary, the derived-method and foreground firewalls).
- **`Disease Cards : Residual Maps/`**, **`Disease Matrix/`** — per-disease matching cards and the disease-signature matrix.
- **`CPG_Report_Generator/`** — the patient report builder.

## Status

Preliminary. The pipeline has been run end-to-end on real patient methylation arrays; results are described as consistent with the framework's predictions, not as proven. Prospective patient validation is the next step.
