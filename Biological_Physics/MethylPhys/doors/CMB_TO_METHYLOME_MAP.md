# CMB → Methylome: the translation map, scored

**Written by the author before the chain was built** (the 32-section CMB-postdoc spec walked module by module). Kept verbatim; a `2026-09-19 status` column is added so the map becomes the scorecard of the translation rather than a wish list. Legend as in the original: ✓ have · ⟳ roadmap · ➕ should add · ◐ translates with reinterpretation · ✗ does not translate.

**Scorecard:** 79 rows. The two rows where the chain went *against* the map are the two most instructive: a second deconvolver (row 20, cut) and de-aging (row 47, refused). Both are Part II chapters: *where the CMB analogy breaks, and why that is the finding.*


## Section I — Foundations: what the CMB field actually is

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 1 | Spherical harmonic decomposition T(n̂) = Σ a_ℓm Y_ℓm, including E/B polarization | The genome isn't a sphere, but it's a structured manifold (chromosomes × position × 3D chromatin TADs). Harmonic decomposition becomes (a) per-chromosome Fourier along position, (b) per-TAD modal decomposition, (c) the 115-cell-type basis itself acts as a "feature-space harmonic basis" | ◐ | Need to write this up — it's the single biggest conceptual move in the rulebook. The 115 cell types are our spherical harmonics. A patient's β profile gets decomposed onto that basis via Walther IAM Deconvolver. | UNWRITTEN as a chapter — but the chain does exactly this: the deconvolver projects a patient's β onto the 115-cell-type basis (Stage 2) and the class gauge reads eight coefficients of that projection. The 'harmonic basis' is the Atlas. Row 1 is the conceptual frame for Part II's Atlas chapters. |
| 2 | Monopole, dipole, quadrupole, low-ℓ culture | Monopole = mean genome-wide β. Dipole = age axis (CPG-VAL-007) + sex axis. Quadrupole+ = cellular-composition axes (TODO 1.5 PCA). Low-ℓ anomalies = global-genome-level patterns (CIMP+, X-inactivation status, ploidy) | ✓ | Implicit today, need to make explicit. |  |
| 3 | Cosmic variance — finite-mode irreducible noise | Cellular variance — finite number of independent cells contributing to a methylation reading. A pure plasma sample only carries ~10,000 cell-equivalents of cfDNA; you cannot reduce the variance below the cell-count floor. Same structural limit as 2ℓ+1 modes. | ➕ | Important to name. Defines fundamental limits on small samples. | NOT BUILT — cfDNA cell-equivalent variance floor never modelled |

## Section II — Experimental / instrument layer

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 4 | Frequency coverage (multi-band foreground separation) | Multi-substrate methylation (5mC, 5hmC, 5fC, 5caC), plus orthogonal modalities (RNA-seq, ATAC-seq, fragmentomics). CPG v0.1 is single-frequency (5mC); the architecture must declare the multi-substrate slot even though we're not multi-frequency yet. | ⟳ | Tier 5 of EDEAR Physics Roadmap — 5-substrate multi-assay (already named in the L3 lab plan). |  |
| 5 | Beam modeling | Probe response function — each 450K/EPIC probe has a sequence context, hybridization efficiency, and adjacent-CpG cross-reactivity. Probe design errors are the methylome's beam errors. | ➕ | Not yet explicit in CPG. Should add. |  |
| 6 | Bandpass calibration | Bisulfite conversion efficiency + probe-type normalization (Type I vs Type II). Same physics: non-delta-function response in the chemistry domain leaks signal between channels. | ◐ | Standard preprocessing handles it but we don't TREAT it as bandpass calibration with nuisance parameters. Should formalize. |  |
| 7 | Gain calibration / drift | Batch effect + plate position + run date drift. Lab-to-lab and run-to-run gain drift is the methylome version of detector gain drift. | ✓ | Standard pre-processing (BMIQ, funnorm, ComBat) — but we should declare it L2 explicitly. |  |
| 8 | Polarization-angle calibration | Strand-orientation calibration. Methylation has a forward-strand vs reverse-strand asymmetry; bisulfite sequencing can mis-assign. The polarization-rotation analog is real but minor at the array level. | ◐ | Mostly a sequencing-era concern; arrays largely solved it. Document but low-priority. |  |
| 9 | Scanning strategy | Cohort sampling design. How patients are sampled across age/sex/site/batch determines what coverage you have and what nulls you can run. Pre-build cohorts were chosen for us; future CPG-acquired cohorts must be designed with "scanning strategy" rigor. | ➕ | Important for prospective cohort acquisition. |  |

## Section III — Time-ordered data to maps

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 10 | TOD analysis | IDAT intensities + control probe traces + bead-pool QC. Before β values exist, there are raw intensities + control signals = our "timestream." | ✓ | Standard tooling handles it. |  |
| 11 | De-glitching + data cuts | Probe flagging, sample QC dropouts, detection-p-value masking. Failed probes, low-quality samples, contamination flags. | ✓ | Standard. Declare explicitly. |  |
| 12 | Noise modeling (1/f, atmospheric, correlated detector noise) | Probe-position-on-array correlated noise, plate-edge effects, scanner drift within run. The methylome has its own non-white noise — and CPG hasn't modeled it explicitly. | ➕ | Significant gap. Should add a noise-covariance module. | NOT BUILT |
| 13 | Destriping / max-likelihood map-making | Quantile normalization / funnorm / BMIQ. All raw-intensity-to-β methods are inverse-problem solvers in this same family. | ✓ | Standard tooling. Map preprocessing. |  |
| 14 | Transfer functions (pipeline filters the true sky) | The deconvolver itself filters disease signal — it explains away anything that looks like cell composition. We saw this in CPG-VAL-006: chr16/chr17 depleted because their CpGs got "explained away." | ➕ | This is exactly what we already discovered empirically and didn't have a name for. Transfer-function thinking is critical — every step of the chain alters the signal in modeled ways. | NAMED in the lessons (CCL-039, glioma-LL-002: the deconvolver 'explains away'); no module |

## Section IV — Pixelization & spherical computation

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 15 | HEALPix pixelization | CpG manifest as the pixelization — 450K → 850K → EPIC v2 1.6M = increasingly fine pixelization. Each CpG is a pixel. Cell-type space is also pixelized (8 classes → 115 cells → finer with each MCMC refinement). | ✓ | Implicit. Make explicit. |  |
| 16 | Masks + apodization | CpG masks — X-inactivation, imprinting, repeat regions, low-confidence probes, SNP-overlapping probes. Mask apodization = down-weighting near suspect regions rather than hard-cutting. | ◐ | Hard masking common; soft apodization unusual. Could be a CPG innovation. |  |
| 17 | Cut-sky mode coupling, pseudo-Cℓ, MASTER | Cut-genome mode coupling. When some CpGs are masked, the per-chromosome correlation function (TODO 2.1) is biased — same MASTER algorithm applies. | ➕ | Critical for TODO 2.1 to be rigorous. |  |

## Section V — Foregrounds and component separation (the biggest section)

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 18 | Galactic foreground modeling (sync, dust, free-free, AME, CO, etc.) | Cellular foreground stack: cell composition, age, sex, ancestry, batch, smoking, BMI, medication exposure, plate position. These are our synchrotron+dust+free-free+AME. | ◐ | We handle cell composition (Walther) + age (CPG-VAL-007). We do NOT yet handle sex, ancestry, batch, smoking explicitly. Big gap. | PARTIAL — composition (Walther) and age (band) handled; sex/smoking/batch layers BUILT and deliberately NOT wired (SOP §104) |
| 19 | Extragalactic foregrounds (CIB, SZ, point sources) | High-resolution disease-overlapping foregrounds:subclinical inflammation, comorbidities, pharmacological methylation effects (e.g., 5-azacytidine), recent acute illness, vaccination response. These are the methylome's "point sources." | ➕ | Almost completely unmodeled. Big gap. | NOT BUILT |
| 20 | Component-separation methods (Commander, NILC, SMICA, SEVEM, GNILC) | Walther IAM Deconvolver = one method. We need to build at least one INDEPENDENT method (parametric Bayesian vs blind/ILC-style) so we can compare, the way Planck compared Commander/NILC/SMICA/SEVEM. | ⟳ | Critical addition. No single-method CMB result was ever trusted. Need at least 2 deconvolvers. | REVERSED — NILC built as the second method, then CUT 2026-07-02 (c1be0c3): it collapsed on correlated blood mixtures and deleted correct calls. Single-method today, by decision. The Planck principle this row cites is the one the chain knowingly does not follow. |
| 21 | Foreground residual marginalization in likelihood | Nuisance parameter for residual cell-composition error, residual age error, residual batch error. Right now we point-estimate these. Should marginalize. | ➕ | Big upgrade. Required for clean L7 likelihood. | NOT BUILT — point estimates |

## Section VI — Power-spectrum estimation

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 22 | Auto vs cross-spectra (split maps, half-mission × half-mission) | Split-cohort cross-correlations. GSE51057 × GSE51032 cross is the methylome's "half-mission × half-mission." We already used this in CPG-VAL-003 (concordance gate). | ✓ | Mark explicit. Could go further with within-cohort odd/even sample splits. | BUILT — GSE51032 × GSE51057 is the foundation-cohort anchor; reproduced 2026-09-19 at r = 1.00000 |
| 23 | Pseudo-Cℓ on masked sky | Per-chromosome C(d) with masked CpGs handled correctly. Required for TODO 2.1. | ⟳ | Build with MASTER-style mode-coupling correction. |  |
| 24 | Quadratic max-likelihood at low ℓ | Per-TAD or per-chromosome-arm maximum-likelihood spectrum estimation. For low-mode-count regimes. | ➕ | Niche but worth declaring. |  |
| 25 | Bandpowers | Genomic distance bandpowers — group C(d) into log-spaced d-bins with proper covariance. | ⟳ | Part of TODO 2.1. |  |
| 26 | Beam/calibration nuisance treatment | Probe-response + bisulfite efficiency nuisance treatment. | ➕ | Add to L7 likelihood. |  |

## Section VII — Likelihoods and covariances

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 27 | Low-ℓ likelihoods (non-Gaussian, pixel-space, Gibbs sampling) | Low-frequency disease signal likelihoods — when there are very few discriminating CpGs (rare cancers, early pre-dx), the distribution of A-score isn't Gaussian. Need exact-likelihood treatment. | ➕ | Future. |  |
| 28 | High-ℓ likelihoods (Gaussian bandpower with detailed covariance) | Mahalanobis hyper-volume (CPG-VAL-002) is the seed.Need to extend to per-card formal likelihoods. | ⟳ | This is L7 of the chain. |  |
| 29 | Covariance matrices | A-score covariance (have via Ledoit-Wolf in CPG-VAL-002). Need: per-CpG covariance, cross-cohort covariance, nuisance-parameter covariance. | ⟳ | Tier 2-3 work. |  |
| 30 | End-to-end simulations | Synthetic patient generation: known cell-type fractions + known disease signal + simulated batch + run through entire chain. Validate that the chain recovers the truth. | ➕ | The single biggest missing piece in CPG today. Cannot defend the chain without it. | BUILT — synthetic_patient_harness.py + null N7 |
| 31 | Null tests | HC permutation, age permutation, sex permutation, cohort splits, plate-position null, batch null, synthetic injection-recovery. Some scattered today, no unified suite. | ⟳ | L9 of the chain — unified null suite is on the engine-completion sprint. | BUILT — cpg_null_runner.py N1–N8 (found in RETIRED/ and restored to MethylPhys/chain/ 2026-09-19) |
| 32 | PTEs and goodness-of-fit + look-elsewhere | Mandatory. Especially for rare biological anomalies. | ➕ | Add to outcome.md template. |  |

## Section VIII — Theory prediction engines

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 33 | Boltzmann solvers (CAMB/CLASS) | The IAM physics framework + H_min derivation IS our Boltzmann solver. It predicts the architectural floor against which everything else is measured. | ✓ | Already in place — the Recipe is the methylome's CAMB. Vault content. | BUILT — the Recipe / H_min derivation (vault); 40 floors in cpg_gauge_engine.py |
| 34 | Recombination physics (RECFAST/HyRec) | Cellular differentiation thermodynamics — the per-class H_min values are frozen at differentiation, analogous to acoustic peaks frozen at recombination. | ✓ | Implicit in IAM. |  |
| 35 | Line-of-sight integration | Lineage tracing through cell development. A methylation pattern in a mature cell reflects integrated history of decisions along the developmental trajectory. | ◐ | Translatable but speculative. |  |
| 36 | Acoustic-peak phenomenology | The pattern cheat-sheet for methylation features. (See Section XVII below.) This is the rulebook's most teachable section. | ➕ | Build a CPG version of the cheat-sheet. |  |

## Section IX — Parameter inference & degeneracies

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 37 | MCMC + posterior geometry | MCMC for per-card posterior tier (replacing threshold-based scoring). Already familiar from IAMAtlas build. | ⟳ | Engine-completion work. | NOT BUILT — MCMC used for the Atlas and the floors; scoring is threshold + age band |
| 38 | Banana degeneracies | CIMP-axis degeneracy is visible in Image 2. Other expected ones: age-vs-class-drift, immune-fraction-vs-disease-signal, sex-vs-X-inactivation. | ⟳ | TODO 2.3 banana degeneracy is the first concrete one. |  |
| 39 | Priors | Disease prior (cancer_prior runtime matrix), family-history prior(family_history_multiplier), age prior. Already implemented but should be treated as proper Bayesian priors with volume awareness. | ◐ | Have, but informal. Upgrade. |  |
| 40 | Profile likelihood vs marginalized posterior | Critical distinction. Profile likelihood = "where is the best-fit cell-type fraction" vs marginalized = "where is the volume in fraction space." They can disagree when nuisance dimensions are high. | ➕ | Add to L7. |  |
| 41 | Multimodal distributions | Disease subtypes are multimodal posteriors. HER2+ vs Basal vs Luminal-A breast cancer don't sit on one mode. Posterior clustering matters. | ◐ | Recognized but not formal. |  |
| 42 | Nested sampling and Bayesian evidence | Model comparison: does this patient look more like 'breast pre-dx' or 'pancreatic pre-dx' or 'inflammatory baseline'? Bayesian evidence is the right tool. | ➕ | Future engine layer. |  |
| 43 | Tension metrics | Cross-card tension when two cards both fire on one patient. Need proper tension language, not just "both elevated." | ➕ | Operational rule for Stage 6 reports. |  |

## Section X — Lensing: the CMB as a backlight

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 44 | Lensing smoothing of acoustic peaks | Late-life methylation drift smooths early-life methylation patterns. Aging is the methylome's lensing — it smears the original developmental signal. | ◐ | Beautiful analogy. Worth a paper. | UNWRITTEN — the AGE BAND is the operational form of 'aging as lensing'; the analogy itself appears nowhere in the corpus and is the cleanest way to explain the band to a physicist |
| 45 | Four-point lensing reconstruction | CpG-CpG-CpG-CpG four-point correlations — coordinated methylation changes across 4 loci would reveal coordinated regulatory programs. Bispectrum (TODO 2.2) is 3-point; this would be 4-point. | ➕ | Tier 2+ research. |  |
| 46 | Lensing power spectrum | Aging "drift power spectrum" — measure how aging redistributes power across methylation features. The IAM cellular age clock (TODO 2.4) is the seed. | ⟳ | TODO 2.4 lives here. |  |
| 47 | Delensing | De-aging: subtract the age component to recover the disease component. CPG-VAL-007 dipole subtraction is the first instance. | ✓ | Have seed. Generalize. | REVERSED — de-aging (age_axis_foreground.py) BUILT, then §104 ruled annotate-don't-subtract. The chain does the opposite of delensing, on purpose: the methylome's foreground is the patient's own biology. The one place the analogy is deliberately broken. |
| 48 | Cross-correlations with external probes | Cross-correlate CPG output with RNA-seq, proteomics, imaging. | ⟳ | Future multi-omics. |  |

## Section XI — Polarization special topics

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 49 | E/B separation | Methylated vs hydroxymethylated separation — the methylome's literal polarization. 5mC vs 5hmC is an E/B-like spin decomposition. | ◐ | Substrate-level; requires oxBS-seq (5-substrate Tier 5). |  |
| 50 | B-mode taxonomy | 5hmC contamination taxonomy — what shows up in a "5hmC signal" can be true 5hmC, residual unconverted 5mC, sequencing error, or biological switch. | ➕ | Substrate-dependent. |  |
| 51 | Reionization bump (low-ℓ EE) | Developmental methylation reprogramming bump — large-scale early-life methylation events leave a "reionization-like" signature in the lowest-frequency modes. | ◐ | Speculative analogy. |  |
| 52 | Cosmic birefringence (EB/TB correlations) | Methylation-state rotation during disease (5mC → 5hmC → 5fC → 5caC oxidation cascade is literally a rotation in methylation chemistry-space). Would produce equivalent cross-correlation signatures. | ◐ | Genuinely novel research direction. |  |

## Section XII — Non-Gaussianity / higher-order statistics

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 53 | Bispectrum estimators (local, equilateral, orthogonal, etc.) | TODO 2.2 bispectrum on highest-A-score class markers. Different "shapes" of bispectrum probe different biological coupling structures (local = pairwise regulatory networks; equilateral = shared upstream regulator; orthogonal = anti-correlated regulation). | ⟳ | On the roadmap. The shape taxonomy will need its own translation. |  |
| 54 | Trispectrum (4-point) | Coordinated 4-CpG methylation events — chromatin remodeling complexes, multi-enhancer hubs. | ➕ | Tier 2+. |  |
| 55 | Minkowski functionals / topology | Genome-wide methylation topology — how methylation is spatially organized within the nucleus (A/B compartments, TADs, LADs). | ➕ | Connects to Hi-C data — multi-omics cross. |  |
| 56 | Phase statistics | CpG phase structure — same amplitude, different ordering = different biology. Phase information is what the bispectrum sees that the power spectrum misses. | ➕ | Tier 2+. |  |

## Section XIII — Isotropy, anomalies, preferred axes

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 57 | Statistical isotropy tests (bipolar harmonics, dipole modulation, hemispherical asymmetry, local variance maps, multipole vectors) | CPG-VAL-006 chromosome isotropy is the seed. Extend to bipolar (paternal vs maternal allele asymmetry), dipole modulation, hemispherical asymmetry (genomic half-asymmetry). | ✓→⟳ | Have seed; extend. |  |
| 58 | Parity asymmetry | Strand-bias parity tests + even/odd-chromosome parity tests. | ➕ | Niche but specifically testable. |  |
| 59 | Alignment tests (axis of evil, multipole vectors) | Chromosomal arm alignment — do disease-signal-rich chromosomes share an axis (e.g., chr6 MHC + chr2 immune-related = immune-axis alignment)? | ◐ | Need to be careful with look-elsewhere. |  |
| 60 | Topology searches (matched circles) | Methylation "matched-pair" searchesbetween distant loci with co-regulated patterns. | ◐ | Speculative. |  |
| 61 | Look-elsewhere discipline | MANDATORY. Pre-register the test before running it. We've been informal about this. | ➕ | Make absolute rule in v1_VAL_Test_Checklist. | BUILT — N8 in the null runner; VAL-006's chr6 signal died under it |

## Section XIV — Secondary anisotropies

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 62 | Integrated Sachs-Wolfe effect | Late-life methylation drift integrated along developmental trajectory.Same structure: late-time potential decay → large-angle imprint. | ◐ | Speculative analogy worth exploring. |  |
| 63 | Rees-Sciama | Nonlinear corrections. Niche. | ✗ | Skip. |  |
| 64 | Thermal SZ (hot electron Comptonization) | Inflammation-associated methylation shifts — chronic inflammation reshapes the methylome the way hot cluster gas reshapes the CMB. | ◐ | Real biological analog. |  |
| 65 | Kinetic SZ (bulk velocities) | Clonal hematopoiesis bulk-velocity-like signal — large clones with shared methylation drift create a kSZ-like signature. | ◐ | Specific to CHIP/AML cards. |  |
| 66 | Patchy reionization | Patchy senescence / cellular replacement in aging tissues. | ◐ | Aging-related. |  |
| 67 | Cosmic infrared background | Background methylation noise from healthy tissue turnover. | ➕ | Worth modeling. |  |
| 68 | Rayleigh scattering | Frequency-dependent scattering. | ✗ | Skip. |  |
| 69 | Spectral distortions (μ, y-type) | Methylation spectral distortions: systematic deviations from the IAM-predicted information spectrum at specific scales = signatures of disease-specific energy injection into the methylome. | ◐ | Genuinely novel research direction. |  |

## Section XV — Data combination & external probes

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 70 | BAO combination | Multi-substrate combination (RNA-seq, fragmentomics, ATAC-seq, proteomics) as the methylome's BAO — breaks degeneracies CMB-only / methylation-only cannot resolve. | ⟳ | Tier 5. |  |
| 71 | Supernovae (distance ladder) | Cellular age clock as the distance ladder — calibrate biological age via known-age reference cohorts. | ⟳ | TODO 2.4. |  |
| 72 | BBN cross-check | Mendelian inheritance + germline methylation cross-checks — early-life patterns set bounds on later patterns. | ◐ | Niche but principled. |  |
| 73 | Large-scale structure | Cross-cohort population structure. UK Biobank scale eventually. | ⟳ | Far future. |  |
| 74 | Joint likelihoods (handle shared covariance) | Joint likelihood across cards for one patient — handle covariance between immune-class drift detected by breast-epic AND immune-epic. | ➕ | Stage 5 architecture. |  |

## Section XVI — Modern computational methods

| # | CMB module | CPG analog | status (author) | notes (author) | **2026-09-19 status** |
|---|---|---|---|---|---|
| 75 | Public compressed likelihoods | Per-card compressed-likelihood release. ACT-DR6 style. Future. | ➕ | Once cards stabilize. |  |
| 76 | Emulators | Neural-network emulator for IAMAtlas reconstruction at very large CpG coverage. Speed optimization. | ➕ | Tier 4+. |  |
| 77 | Simulation-based inference | Synthetic-patient SBI — when L7 likelihood is intractable. | ⟳ | Pairs with end-to-end sims. |  |
| 78 | Blinding | Mandatory for confirmation cards. Apply pipeline before seeing case labels. Unblind only after seal. | ➕ | Add to v1_VAL_Test_Checklist. | PARTIAL — PREREG sealed before data opens; no formal blinding step |
| 79 | Reproducibility discipline | Already absolute — the reproducibility quadruple is our version. | ✓ | We have this. | BUILT — bit-identical Stage 1, deconvolver MAE 0.0004, anchors r = 1.00000; the PROC kit |

---
*The original text (the 'Cosmological Multiomics, Informational Biophysics, or Holographic Epigenetics' walk-through) is the author's; only the last column is new.*