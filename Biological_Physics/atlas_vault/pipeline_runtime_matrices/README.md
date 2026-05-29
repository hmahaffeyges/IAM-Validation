# EDEAR Pipeline Runtime Matrices

**Extracted from:** `GAPE_WEB_v13.py` on 2026-05-28
**Purpose:** Standalone, citable, version-controllable copies of the lookup tables the EDEAR runtime pipeline consumes between IDAT-in and report-out.

These are the clinical interpretive context that wraps IAMAtlas's biological output. **IAMAtlas tells the engine what the biology is doing; these matrices help translate that biology into clinically actionable patient context.** A report that combines biology with risk context is dramatically more useful than one that just reports A-scores.

---

## The 9 files in this folder

| File | Pipeline stage | Purpose |
|---|---|---|
| `iamatlas_celltype_markers_v0_1.json` | Stage 2 | **NEW 2026-05-29.** Per-cell-type one-vs-rest marker CpGs (top-100 per cell type, 115 cell types). Consumed by `iamatlas_a_scoring.py` to compute per-cell-type A-scores in addition to the 8 class A-scores. Unlocks the sub-class signal that VAL-095/096 surfaced (e.g. Basophils d=+1.577 at >10yr breast pre-dx in GSE51057, BE breast-epithelial d=+1.281). SHA-256: `a56576cd5a7b2219d22d9a7a6efccd141a43c6d5fe4f5eb1d81e7375e1061ddc`. |
| `iamatlas_a_scoring.py` | Stage 2 | **NEW 2026-05-29.** Scoring module with `score_per_class()` and `score_per_celltype()` functions. Returns A-score + coverage + confidence + status per class/cell type. **Sibling to the Walther IAM Deconvolver, not a replacement**: deconvolver does NNLS fractions; this module does entropy-at-markers A-scores. Different math, different CpG sets, different failure modes — kept separate for independent testability. |
| `age_reference_matrix.json` / `.csv` / `.py` | Stage 3 | 80-cell age × class lookup for cellular-age computation and age-matched percentile ranking — the headline product feature |
| `tier_breakpoints.json` | Stage 4 | A-score thresholds for engine tier calls (1.05 / 1.07 / 1.10) + engine→customer language collapse |
| `cfdna_weight.json` | Stage 4 | Healthy-blood cfDNA tissue-of-origin weights (Snyder 2016 + Moss 2018); used when substrate is plasma cfDNA to compute expected-vs-observed pan-tissue context |
| `literature_anchors.json` | Stage 6 | Published reference A-scores per class — positions patient's reading against healthy / disease / cancer cohort anchors |
| `cancer_prior.json` | Stage 6 | US lifetime cancer incidence per class — Bayesian risk-prior context for elevated readings |
| `family_history_multiplier.json` | Stage 6 | First-degree-relative RR per class — turns "keep watching" into "go to a doctor" when family history is present |

The 8 methyl H_min values that the Stage 2 A-score formula divides by are not duplicated here — they live in IAMAtlas's `IAMAtlasREBUILD_provenance.json` under the `h_min_values_frozen_2026_04_06` key. One source of truth.

The age matrix is mirrored as `.csv` for inspection and `.py` for drop-in import with `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()` helpers (linear interpolation in age, since the matrix is decadal).

---

## Where each plugs into the IDAT-to-report flow

### Stage 2 — Per-class + per-cell-type A-scores

`iamatlas_a_scoring.py` runs Stage 2. Two scoring surfaces, both computing A = mean(H(β) / H_min(class)) at marker CpGs:

- **`score_per_class(beta_dict, class_markers, H_min_by_class)`** — 8 architecture-class A-scores. The headline product surface. H_min comes from the frozen IAM-framework MCMC posteriors (Jacobson→virial→Landauer chain, R-hat < 1.001): terminal 0.7728, immune 0.838889, secretory 0.843264, cycling 0.856055, progenitor 0.852216, stromal 0.862950, stem_adult 0.873718, stem_pluri 0.982166.
- **`score_per_celltype(beta_dict, celltype_markers, celltype_to_class, H_min_by_class)`** — 115 cell-type A-scores at finer resolution. Each cell type's H_min looked up via its class membership. **New 2026-05-29**: this surface unlocks signal that the 8-class output cannot resolve. VAL-095 (UniLIFE 19-cell head-to-head): aTreg at >10yr breast pre-dx d=+1.26/+0.79. VAL-096 (Loyfer 25-tile per-tile): pancreatic + kidney + erythrocyte-progenitor elevation at >10yr. Both findings now natively in the 8-class + 115-cell-type output of one pipeline run, without separate UniLIFE/Loyfer/Salas/EpiSCORE deconvolution stacks. The cell-type readout also surfaces sub-cellular signals neither v1 atlas had (Basophils d=+1.577 in GSE51057, BE breast-epithelial d=+1.281 at >10yr).

`iamatlas_celltype_markers_v0_1.json` is the runtime artifact loaded once per session. 115 cell types × top-100 one-vs-rest marker CpGs each. Selection logic: `|target_celltype_mean − mean(other_114_celltypes_means)|`, top-N by absolute score, computed from IAMAtlas REBUILD per-cell-type posterior means. Adapted from val_093.py's `identify_marker_cpgs` (per-Loyfer-tile) to per-IAMAtlas-cell-type. SHA-256 sealed.

Both functions return per-result dicts with `A`, `n_markers_expected`, `n_markers_matched`, `coverage`, `confidence`, `status`. Status codes: `OK`, `INSUFFICIENT_MARKERS` (< 20 matched), `NO_MARKER_OVERLAP`. Confidence = `coverage × max(0, 1 − dispersion/0.20)` where dispersion = stdev of per-CpG A-scores. Bounded [0, 1], not a calibrated probability — recalibrate against labelled data when available.

**The Walther IAM Deconvolver does not change.** Its job stays exactly what it is: NNLS fractions + diagnostics on the cell-type marker pool. The scoring module is a sibling, computing entropy-at-markers A-scores on a different CpG pool (per-cell-type one-vs-rest, not the deconvolver's between-cell-type-variance Tier 2 pool). Different math, different CpG sets, different failure modes — kept independent.

### Stage 3 — Demographic adjustment + cellular age

`age_reference_matrix.json` runs the show. The matrix has 80 cells: 8 classes × 10 decadal age bins from 4 to 95, each cell carrying `(age_midpoint, A_mean, A_sd, β_mean, β_sd, n_samples, A_p10, A_p25, A_p50, A_p75, A_p90, source_citation)`.

For each of 8 classes, at the patient's age:
- Look up age-matched A_mean, A_sd, and 5 percentiles (linear interpolation across decade midpoints).
- Compute patient's percentile within the age-matched distribution.
- Compute cellular age by inverting "what age would produce this A_mean" → per-class cellular age + overall cellular age.

**Sources cited in the matrix:** Hannum 2013, Horvath 2013, Roadmap Epigenomics 2015, Moss 2018, Lister 2013, Alisch 2012, Adelman 2019, De Jager 2014 / Shireby 2022, Jaiswal 2014 (CHIP-neg). These are *data citations* — where the age-decade reference β values were originally measured. They are not runtime atlas queries.

### Stage 4 — Floor-departure detection + cfDNA pan-tissue context

`tier_breakpoints.json` runs the tier calls:

| Engine tier | A-score range | Customer-facing label |
|---|---|---|
| BELOW_NORMAL | A significantly below age-matched expectation | SUPPRESSED |
| NORMAL | A ≤ 1.05 | NORMAL |
| MARGINAL | 1.05 < A ≤ 1.07 | ELEVATED |
| DETECTABLE | 1.07 < A ≤ 1.10 | SIGNIFICANTLY_ELEVATED |
| URGENT | 1.10 < A | SIGNIFICANTLY_ELEVATED |
| FLOOR_BREACH | A > class ceiling | SIGNIFICANTLY_ELEVATED |

`cfdna_weight.json` activates when substrate is plasma cfDNA. The matrix encodes expected per-class tissue-of-origin shedding for healthy plasma: immune 0.70, cycling 0.12, secretory 0.08, stromal 0.04, stem_adult 0.03, progenitor 0.02, terminal 0.005, stem_pluri 0.005. The engine compares the deconvolver's observed per-class fractions to these expected weights and flags departures: e.g. cortical-neuron fraction at 1% when healthy expectation is 0.5% is a +1% departure surfaced in the report as pan-tissue context.

### Stage 6 — Report assembly: clinical interpretive layers

Three Stage 6 matrices wrap the biology in clinical context:

**`literature_anchors.json`** — published A-score anchors per class. Lets the report say "your A_terminal of 1.05 is in the range of published low-AD-neuropathology cohorts (De Jager 2014, A=1.043) rather than glioblastoma (A=1.256, Ceccarelli 2016)" — turning a number into a meaningful position relative to known biology.

**`cancer_prior.json`** + **`family_history_multiplier.json`** — Bayesian risk context. Combined:
```
posterior_context_class = baseline_prior × age_factor × sex_factor × fh_factor × match_magnitude
```
Same A-score reading means different things at different priors. A secretory-class elevation in a 45-year-old female whose mother had breast cancer is a different actionable report than the same A-score in a 45-year-old male with no family history. The matrices encode the priors; the report builder combines them with the biology.

Sex risk-prior adjustment lives inline in the engine code, not as a separate matrix (only 4 numbers: female-secretory 1.4×, male-secretory 1.2×, plus smaller per-class adjustments).

---

## Conditional consumption

Several matrices activate only when their input data is present:
- `cfdna_weight.json` activates only when substrate is plasma cfDNA. If substrate is tissue biopsy or buffy-coat DNA, the engine skips this step.
- `family_history_multiplier.json` activates only when the intake form supplied family history. No family history → no FH adjustment.
- `literature_anchors.json` activates only when the report includes clinician-facing context (configurable per delivery channel).

If the input isn't present, the engine passes by — no error, no degraded output, just no enrichment from that particular layer. **Zero-risk-to-include pattern: every matrix can ship in the production folder ahead of the data that activates it.**

---

## Where smoking lives (and why it's not here)

Smoking is **not** a pipeline-wide matrix — it's a **per-card covariate**. The lung-epic card JSON keys its thresholds by smoking status (never / former / current). Same pattern for viral hepatitis inside the HCC card, age inside the breast card, etc.

Cards own their own covariate logic because each disease responds differently to each covariate. A single global "smoking adjustment matrix" would lose that per-disease specificity — the framework's covariate handling is intentionally distributed into the cards where the disease-specific biology lives.

---

## Provenance and audit trail

Each JSON has a `_meta` block at the top recording the original constant name in `GAPE_WEB_v13.py`, the purpose statement, and the extraction date.

If a matrix is updated, the change should happen in `GAPE_WEB_v13.py` first, then re-extracted here. **These files are READ-ONLY mirrors of the engine source — they exist for inspection, citation, and version control, not for direct editing.**

The age matrix and tier breakpoints are the two most-cited artifacts when a clinician (or a regulator) asks "what's the reference distribution you compare a patient against?" Keeping them separately citable is the difference between "look at line 364 of this 10,000-line Python file" and "look at this 80-row matrix with a citation header."
