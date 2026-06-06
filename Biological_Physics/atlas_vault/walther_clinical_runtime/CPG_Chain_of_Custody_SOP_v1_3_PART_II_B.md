# CPG Chain-of-Custody SOP — Part II-B (Stages 5 through 7) (v1.2 — walkthrough aligned)

> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


**Continues from Part II-A (§11-§46). This document contains §47-§64 — Stages 5 (Mahalanobis), 6 (cellular age inversion), 7 (tier breakpoints).**

---

# Stage 5 — Multi-D departure (Mahalanobis hyper-volume)

Stage 5 produces the **single headline number** every EDEAR report carries: the Mahalanobis distance of the patient's 115-cell-type A-score vector from the pooled healthy-cohort centroid, weighted by the covariance structure of IAM's own feature space. This is the methylome's joint-posterior-banana measurement — the multi-dimensional analog of CMB cosmology's joint cosmological-parameter ellipsoid (Ωm vs ΩΛ vs σ_8 etc.).

The cleanest framing: the per-class A-score (Stage 4) answers "how far is each class from H_min." The Mahalanobis distance answers "how far is the patient from healthy in the full 115-D A-score manifold." Both are IAM measurements at different granularities. The Mahalanobis distance gives clinicians one calibrated number — and the top-10 axis decomposition makes that number explainable.

---

## §47. Step 5.1 — Patient 115-cell-type A-score vector assembly

**What this step does.** Takes the patient's 115 per-cell-type A-scores from Stage 4 (§44) and assembles them into a single 115-element feature vector ready for Mahalanobis distance computation. Handles per-cell-type imputation when QC flagged some cell types as `INSUFFICIENT_MARKERS`.

**Inputs.** Per-sample per-cell-type A-scores + status codes from <A-score output — emitted by `iamatlas_a_scoring.py` (real module per inventory)>.

**Atlas reference.** Indirect — the 115 cell-type ordering is fixed by the atlas's celltype_to_class mapping.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — class `MahalanobisHealthyHull`, method `_assemble_patient_vector(per_celltype_a)`.

**The math.** Order the 115 cell types in canonical sequence (defined in `mahalanobis_healthy_reference_v0_1.json`). For each cell type:
- If status `OK`: use the A-score as-is.
- If status `MARGINAL_COVERAGE`: use the A-score but flag for downgraded confidence at output.
- If status `INSUFFICIENT_MARKERS` or `NO_MARKER_OVERLAP`: impute using the cohort-pooled-HC mean for that cell type (from the Mahalanobis reference centroid).

The imputation count is tracked separately. Patients with > 5 imputations get a `PARTIAL_DATA` status at output.

**CMB equivalent.** **Cosmological parameter vector assembly before joint-posterior evaluation.** Planck's cosmological inference operates on a joint vector of parameters (Ωm, Ωb, h, n_s, σ_8, τ, etc.). Before evaluating the joint posterior or computing tension with other surveys, the parameter vector is assembled in canonical order with missing parameters imputed from priors. The methylome's 115-element A-score vector is the methylome's parameter vector for joint-posterior evaluation.

**How the methylome differs in implementation.** 115 dimensions vs ~7 cosmological parameters. The high dimensionality is why Mahalanobis (which accounts for covariance) is necessary — naive 115-D Euclidean distance would treat correlated features as independent and inflate apparent distance.

**How it's the same in principle.** Joint-posterior evaluation requires a single feature vector in a defined coordinate system. Methylome and CMB cosmology share that architectural requirement.

**Outputs.** Per-patient 115-element A-score vector + per-cell-type imputed-flag mask + `n_features_imputed` count.

**Decision points.**
- ≤ 5 imputations → proceed with status `OK` to §48.
- 6-15 imputations → proceed with status `PARTIAL_DATA` flag.
- > 15 imputations → flag patient as `INSUFFICIENT_DATA` for Mahalanobis; do not return Mahalanobis distance; downstream report omits the headline number with explanation.

**Failure modes.**
- Cohort-wide high imputation rate (> 10% of patients > 5 imputations) → indicates substrate-atlas mismatch; flag cohort.

**Canonical cross-references.**
- Recipe §6.4 (Mahalanobis specification).
- Roadmap §3.13 (multi-dimensional hyper-volume).

**CPG Plate references.** None at this granularity.

**Chain-link assignment.** L6 (covariance modeling enters) + L8 (parameter inference).

---

## §48. Step 5.2 — HC centroid load (`mahalanobis_healthy_reference_v0_1.json`)

**What this step does.** Loads the pre-computed healthy-control reference object — the centroid (mean) and inverse-covariance matrix in 115-cell-type A-score space — into memory. This is the **calibrated healthy reference** the patient's vector will be measured against. Loaded once per session at engine startup; pinned in memory thereafter.

**Inputs.** `mahalanobis_healthy_reference_v0_1.json` (sha256: `fae063012ff7542a56ae4f91a494bad087d714f944911d6ff289113014a95b2b`).

**Atlas reference.** The reference object is built from IAMAtlas-A-scored healthy controls (n_hc = 601 from GSE51057 + GSE51032). **The reference itself IS the atlas's HC posture in 115-cell-type A-score space** — it is not an external comparison atlas.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull.load_reference()`.

**The math.** The reference carries:
- `hc_centroid`: 115-element mean vector (pooled-HC mean A-score per cell type).
- `covariance`: 115×115 covariance matrix (with Ledoit-Wolf shrinkage, shrinkage = 0.0088).
- `inverse_covariance`: pre-inverted 115×115 matrix for fast per-patient distance computation.
- `cell_type_order`: canonical 115-element ordering matching §47.
- `n_hc`: 601 (calibration sample size).
- `build_provenance`: cohort identifiers, build date, atlas version.

Loaded with SHA-256 verification. If hash mismatches the pinned value, halt and investigate.

**CMB equivalent.** **Cosmological-parameter covariance matrix from chain analysis.** Planck publishes the cosmological-parameter posterior chain along with a derived covariance matrix used for tension calculations with other surveys. The methylome's HC reference object is structurally identical: a centroid (mean parameter vector) + covariance matrix derived from the calibration set.

**How the methylome differs in implementation.** Methylome covariance is over IAM-A-score features; CMB covariance is over cosmological parameters. Different feature spaces; same operational structure.

**How it's the same in principle.** A multi-dimensional reference requires a centroid AND a covariance to make distance computation physically meaningful. Both modalities pin these as fundamental reference objects.

**Outputs.** In-memory `MahalanobisHealthyHull` object accessible to all subsequent per-patient Mahalanobis calculations.

**Decision points.**
- Reference hash matches → proceed.
- Hash mismatches → HARD HALT.

**Failure modes.**
- File missing or corrupted → halt; canonical reference must be present.

**Canonical cross-references.**
- Recipe §6.4.
- Runtime Matrices README §"Stage 2.5" (Mahalanobis specification).

**CPG Plate references.** None.

**Chain-link assignment.** L6 (covariance loaded).

---

## §49. Step 5.3 — Inverse-covariance distance computation

**What this step does.** Computes the **Mahalanobis distance** of the patient's 115-element A-score vector from the HC centroid in the inverse-covariance-weighted metric. This is the single headline number.

**Inputs.** Patient vector from §47 + HC reference object from §48.

**Atlas reference.** Indirect.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull.score(patient_vector)`.

**The math.** Per patient:
```
δ = patient_vector - hc_centroid               (115-element residual)
mahalanobis_distance = sqrt( δᵀ · Σ⁻¹ · δ )      (scalar)
```
where `Σ⁻¹` is the inverse-covariance matrix from §48.

**Why this is the right metric:** Naive Euclidean distance treats all 115 dimensions as independent and identically scaled. In reality, cell-type A-scores are highly correlated (immune-class cell types correlate with each other; secretory-class cell types correlate with each other; etc.). Euclidean would over-count a patient who departed along a correlated direction. Mahalanobis re-scales the 115-D space so that all directions are equivalently weighted accounting for the covariance, producing a statistically interpretable distance.

**Typical ranges (from EPIC-Italy validation):**
- HC: distance ~ 6-12.
- Cases (breast pre-dx >10yr): distance ~ 10-20.
- Cohen's d on breast pre-dx vs HC: +1.871 (GSE51057), +2.088 (GSE51032). Beats Xu-538 by +0.752 on GSE51032 without being breast-trained.

**CMB equivalent.** **Joint-posterior tension distance between Planck and an external survey.** When Planck and an external survey (e.g., DES, SDSS-Y3) both report cosmological parameters, the tension between them is computed as the Mahalanobis distance in the joint parameter space using the combined covariance. The methylome's distance from the HC reference is the analog: a single number summarizing multi-dimensional departure under the appropriate covariance metric.

**How the methylome differs in implementation.** Single-patient vs cohort-survey-vs-cohort-survey. Different sample sizes feeding the metric; same metric.

**How it's the same in principle.** Mahalanobis distance is the canonical multi-dimensional departure metric in both cosmology and biology. The math doesn't care which substrate it's computed on.

**Outputs.** Per-patient scalar `mahalanobis_distance` value.

**Decision points.**
- Distance computed → proceed to §50 for axis decomposition.
- Numerical failure (rare, e.g., non-positive-definite covariance from extreme imputation) → flag with `MAHALANOBIS_NUMERIC_FAIL`; report omits the distance with note.

**Failure modes.**
- Negative δᵀ · Σ⁻¹ · δ (mathematically impossible for valid Σ⁻¹) → indicates corrupted reference; halt.

**Canonical cross-references.**
- Recipe §6.4.
- VAL-002 (Mahalanobis hyper-volume sealed, d=+1.876/+2.097).

**CPG Plate references.** Plate 2 (Breast Pre-Diagnostic Anisotropy) — the 1,392 concordant CpGs that compose the pre-dx signature; the Mahalanobis distance captures their joint departure in 115-cell-type A-score space.

**Chain-link assignment.** L8 (parameter inference / posterior — partial).

---

## §50. Step 5.4 — Top-10 axis contribution decomposition

**What this step does.** Decomposes the patient's Mahalanobis distance into per-cell-type contributions. **A single distance number is uninterpretable without knowing which directions in the 115-D space drove it.** Step 5.4 produces the top-10 cell types whose departures most contributed to the patient's distance.

**Inputs.** Patient vector + HC centroid + inverse-covariance + computed Mahalanobis distance from §49.

**Atlas reference.** Indirect.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull._decompose_axes(patient_vector)`.

**The math.** For each cell type `j`:
```
z_shift[j] = (patient_vector[j] - hc_centroid[j]) / sqrt(diag(Σ)[j])
contribution[j] = z_shift[j] × Σ⁻¹·δ [j]
```
Sort by `|contribution|`, take top 10. Each top-axis entry reports:
- Cell type name.
- Patient's A-score for that cell type.
- HC centroid's A-score for that cell type.
- Z-shift (how many SD departed).
- Sign of departure (+ = elevated; − = suppressed).
- Per-axis contribution to the total Mahalanobis distance.

**Why this matters clinically:** The customer report can say "your Mahalanobis distance is 14.2, with the biggest contributions from basophils (+2.1 z, elevated), plasma cells (+1.8 z, elevated), and microglia (+1.6 z, elevated) — your immune compartments are most departed from healthy in your sample." That's a statistically interpretable, mechanistically explainable headline.

**CMB equivalent.** **Decomposition of cosmological-tension into per-parameter directions.** When Planck and DES disagree, the disagreement can be decomposed into directions in parameter space: "the tension is along the S_8 axis, not the H_0 axis." The methylome's top-10 axis decomposition is the same operation: tell the reader which axes in the high-D feature space drove the headline number.

**How the methylome differs in implementation.** 115 cell types as axes; ~7 cosmological parameters as axes. Methylome decomposition is naturally a top-K reporting because of the higher dimensionality.

**How it's the same in principle.** A single distance number is half a measurement. The full measurement requires knowing the direction. The decomposition is the direction.

**Outputs.** Per-patient `top10_axis_contributions` list (each entry as above).

**Decision points.** None — pure decomposition.

**Failure modes.**
- All top contributions from imputed cell types → flag report as "axis decomposition unreliable due to high imputation rate."

**Canonical cross-references.**
- Recipe §6.4.
- Roadmap §3.15 (parameter dependencies / cross-class correlations as axis decomposition).

**CPG Plate references.** Plate 4 Panel A (Class-Difference Map) shows which class-direction axes dominate the methylome — the per-class equivalent of the per-cell-type axis decomposition.

**Chain-link assignment.** L8.

---

## §51. Step 5.5 — Stage 5 output: Mahalanobis distance + per-axis explainability

**What this step does.** Consolidates Stage 5 outputs into the per-cohort artifact.

**Inputs.** Outputs of §49 + §50.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Published joint-posterior tension table with per-axis decomposition.

**How the methylome differs in implementation.** Per-patient row; same content per row as a per-survey-pair row.

**How it's the same in principle.** Multi-D measurement summarized as scalar + decomposition.

**Outputs.** <Mahalanobis output — emitted by `iamatlas_mahalanobis_scoring.py` (real module per inventory)> carrying:
- `mahalanobis_distance` per patient
- `n_features_used` per patient
- `n_features_imputed` per patient
- `status` (OK / PARTIAL_DATA / INSUFFICIENT_DATA / MAHALANOBIS_NUMERIC_FAIL)
- `top10_axis_contributions` per patient (list of dicts)

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 6 (§52 — cellular age inversion).

**Failure modes.** None at consolidation.

**Canonical cross-references.**
- Recipe §6.4.
- Roadmap §3.13.
- VAL-002.

**CPG Plate references.** Plate 2 (Breast Pre-Diagnostic Anisotropy).

**Chain-link assignment.** L6 + L8 (close).

---

# Stage 6 — Cellular age inversion

Stage 6 produces the framework's other headline product feature: **the per-class cellular age** — the chronological age at which a typical healthy person has the same A-score the patient has, computed independently per architectural class. **Eight per-class cellular ages, never collapsed by default.** This is the canonical Recipe §6.3 inversion: no training set, no regression coefficients, no comparison to other frameworks. The 80-cell age reference matrix IS the calibrated instrument; the v4 scoring module inverts it.

The patient's per-class A-score (Stage 4) is the input. The 80-cell baseline (age × class) is the calibration curve. The output is the per-class age at which baseline A_mean crosses the patient's A. When the patient's A is outside the baseline range, the saturation flag carries that information forward — it is not a bug, it is a measurement.

---

## §52. Step 6.1 — Per-class A-score input (from Stage 4)

**What this step does.** Receives the patient's 8 per-class A-scores from Stage 4 (§43). These are the inputs to the inversion.

**Inputs.** Stage 4 output (<A-score output — emitted by `iamatlas_a_scoring.py` (real module per inventory)>).

**Atlas reference.** Indirect.

**Files invoked.** `iam_cellular_age_scoring.py` — class `IAMCellularAge`, method `score_patient(per_class_a)`.

**The math.** Loading only.

**CMB equivalent.** **Loading per-survey cosmological-parameter posteriors before tension analysis.** A trivial loading step that connects two computational stages.

**How the methylome differs in implementation.** Trivial.

**How it's the same in principle.** Trivial.

**Outputs.** In-memory per-patient 8-element A vector ready for inversion.

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (likelihood — partial; the cellular age clock lives here).

---

## §53. Step 6.2 — Age reference matrix load (`age_reference_matrix.json`)

**What this step does.** Loads the 80-cell age × class baseline reference matrix into memory. This is the calibrated instrument the per-class A-score is inverted against.

**Inputs.** `age_reference_matrix.json` (or `.csv` / `.py` — three formats, same data).

**Atlas reference.** **THIS IS the age slice of IAMAtlas.** The 80 cells (8 classes × 10 decadal age bins from 4 to 95) each carry `(age_midpoint, A_mean, A_sd, β_mean, β_sd, n_samples, A_p10, A_p25, A_p50, A_p75, A_p90, source_citation)` measured from the IAMAtlas reference cohort.

**Files invoked.** `iam_cellular_age_scoring.py` — `IAMCellularAge.load_reference()`. Or directly via `age_reference_matrix.py`'s helpers `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()`.

**The math.** Loading only. SHA-256 verification.

**Source citations in the matrix (data references — NOT runtime atlas queries):** Hannum 2013, Horvath 2013, Roadmap Epigenomics 2015, Moss 2018, Lister 2013, Alisch 2012, Adelman 2019, De Jager 2014 / Shireby 2022, Jaiswal 2014 (CHIP-neg). **These citations document where the original per-decade β values were measured.** They are NOT external atlases consulted at runtime — they are the historical data sources that contributed to the IAMAtlas reference build's age stratification.

**CMB equivalent.** **Loading the redshift-evolution calibration tables.** When inferring cosmological parameters from supernovae or BAO at different redshifts, the framework requires per-redshift calibration curves (luminosity-distance relation, sound-horizon scale). Loading these tables is the methylome-equivalent of loading the per-age calibration curves.

**How the methylome differs in implementation.** Methylome reference is decadal age bins (4-95 years); CMB redshift is continuous-but-binned. Both are pre-computed calibration tables.

**How it's the same in principle.** A measurement at the patient's age is interpreted against the reference's expectation at that age.

**Outputs.** In-memory age reference matrix.

**Decision points.** SHA matches → proceed. Mismatch → HALT.

**Failure modes.** File missing or corrupted → halt.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §54. Step 6.3 — Per-class A inversion against the 80-cell baseline curve

**What this step does.** For each architectural class, finds the chronological age at which the baseline A_mean curve crosses the patient's A-score for that class. **This is the canonical Recipe §6.3 inversion — the framework operation that turns an A-score into a cellular age.**

**Inputs.** Patient per-class A from §52 + age reference matrix from §53.

**Atlas reference.** Via the age reference matrix (atlas's age slice).

**Files invoked.** `iam_cellular_age_scoring.py` — method `_invert_per_class(a_class, class_name)`.

**The math.** Per class `c`:

1. Construct the baseline A_mean curve: 10 points `(age_midpoint[k], A_mean[c, k])` for k = 0..9 (decadal bins from 4 to 95).
2. Find the age `t*` such that `A_mean(c, t*) = A_patient[c]` via linear interpolation between adjacent decade midpoints.
3. **Saturation handling** (this is critical):
   - If `A_patient[c]` > max(A_mean[c, :]) → the patient is ABOVE the entire baseline curve → status `SAT_HIGH`. Report cellular age as ">95" with the saturation flag.
   - If `A_patient[c]` < min(A_mean[c, :]) → the patient is BELOW the entire baseline curve → status `SAT_LOW`. Report cellular age as "<4" with the saturation flag.
   - Otherwise → status `OK`. Cellular age is the interpolated `t*`.

**Implementation note:** The baseline A_mean curve is class-specific. Some classes (e.g., immune) show a monotonic increase with age (older = more departure from H_min); some classes (e.g., stem_pluri) show a flat profile (stem cell architectures change less with age). The inversion respects class-specific monotonicity.

**CMB equivalent.** **Inverting the angular-diameter-distance relation to find redshift.** Given a measurement of the angular-diameter distance to an object, invert the cosmological-model distance-redshift relation to find the redshift at which a standard-model object would produce that distance. **The methylome's age inversion is structurally identical: given a measurement of A-score, invert the standard-class A-vs-age curve to find the age at which a healthy individual would produce that A.**

**How the methylome differs in implementation.** Cellular age inversion is per-class (8 independent inversions); cosmological distance-redshift inversion is one-shot. Both use the same algorithmic structure (find the parameter value that matches the observation against a calibrated curve).

**How it's the same in principle.** The framework measurement is converted into a physical age by inverting a calibrated reference curve. The math doesn't care that one is cellular biology and the other is cosmological distance.

**Outputs.** Per-class cellular age (8 values) + per-class status (8 codes: OK / SAT_HIGH / SAT_LOW / INSUFFICIENT_MARKERS).

**Decision points.**
- All 8 classes status `OK` → proceed.
- Some classes `SAT_HIGH` / `SAT_LOW` → proceed, carrying saturation flags forward; this is data signal, not bug.
- Some classes `INSUFFICIENT_MARKERS` (inherited from Stage 4) → those classes return cellular age `NaN` with status `INSUFFICIENT`.

**Failure modes.**
- Baseline matrix corrupted (non-monotonic A_mean curve where biology demands monotonic) → halt; investigate atlas build.

**Canonical cross-references.**
- Recipe §6.3 (the canonical inversion algorithm).

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §55. Step 6.4 — Saturation handling (SAT_HIGH / SAT_LOW / OK / INSUFFICIENT_CPGS)

**What this step does.** Documents the patient's saturation pattern across classes. The saturation pattern itself is a measurement.

**Inputs.** Per-class status codes from §54.

**Atlas reference.** None.

**Files invoked.** `iam_cellular_age_scoring.py` — method `_analyze_saturation(per_class_status)`.

**The math.** Count and report:
- `n_sat_high` = number of classes with status SAT_HIGH.
- `n_sat_low` = number of classes with status SAT_LOW.
- `n_ok` = number with status OK.
- `n_insufficient` = number with status INSUFFICIENT.

Patient `saturation_signature` = a labeled 8-class vector (e.g., "SAT_HIGH on terminal, immune, secretory; OK on cycling, progenitor; SAT_LOW on stromal, stem_adult, stem_pluri"). This is itself a structural readout of the patient's deviation from the IAMAtlas-calibration range.

**Observed pattern on EPIC-Italy validation (1,174 patients):** 100% of patients saturate on at least one class. 7 of 8 classes saturate for most patients. Only cycling is in-range for ~half the cohort. **This is direct data about the EPIC-Italy cohort's posture against the IAMAtlas calibration — it is NOT a bug.** The cohort and the calibration sit in different regions of A-score space; saturation makes that posture explicit.

**CMB equivalent.** **Flagging when a survey's parameter posteriors lie outside Planck's calibration range.** Some external surveys (e.g., DES-Y3 σ_8) report values outside the joint Planck posterior region. The disagreement is flagged, not suppressed — the framework reports the survey's value and notes the calibration-range mismatch. The methylome's saturation flag is the same: report what we measured, note that it's outside the calibration baseline.

**How the methylome differs in implementation.** Saturation is per-class (8 flags) vs per-parameter (≤7 parameters). Same operational principle.

**How it's the same in principle.** Out-of-calibration values are recorded with their flags, never silently clipped or imputed.

**Outputs.** Per-patient `saturation_signature` + `n_sat_high` / `n_sat_low` / `n_ok` / `n_insufficient` counts.

**Decision points.** None — pure reporting.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §56. Step 6.5 — Eight per-class cellular ages — never collapsed by default

**What this step does.** Records the eight per-class cellular ages as **the framework's headline cellular age output**. **They are never collapsed to a single number by default.** Each class carries its own age because each architecture's relationship to age is biologically distinct.

**Inputs.** Per-class cellular age + status from §54.

**Atlas reference.** None.

**Files invoked.** Trivial recording.

**The math.** None.

**Why never collapse:** A patient with "cellular age 47" hides whether the 47 came from an OK reading across all classes, or whether terminal-class reads 35 and immune-class reads 65. The latter is a clinical signal (immune compartment ageing faster than terminal compartment); collapsing destroys it. Customer reports show all eight ages with class labels; an optional summary line reports a `n_samples`-weighted mean across non-saturated classes for the brief-summary box, but the eight per-class ages are the primary deliverable.

**CMB equivalent.** **Per-parameter cosmological measurement, never collapsed into a single "cosmology number."** Planck reports Ωm, ΩΛ, h, σ_8, etc. each as its own number. There is no "the cosmological parameter is 0.42" — that would be meaningless. Same here: there is no single cellular age. There are eight per-class cellular ages.

**How the methylome differs in implementation.** Eight values vs ~7 cosmological parameters. Same structural decision: do not collapse.

**How it's the same in principle.** Multi-parameter framework measurements are reported as vectors. Collapsing destroys signal.

**Outputs.** Per-patient 8-element `per_class_cellular_age` vector + per-patient 8-element `per_class_status` vector + optional `summary_cellular_age` (weighted mean over non-saturated classes).

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §57. Step 6.6 — Percentile rank at patient's chronological age

**What this step does.** For each class, compute the patient's **percentile within the age-matched healthy distribution**. This is the readout most directly interpretable by a clinician: "your immune cellular age is at the 78th percentile for your chronological age."

**Inputs.** Per-class A from Stage 4 + patient's chronological age + age reference matrix percentiles (A_p10, A_p25, A_p50, A_p75, A_p90).

**Atlas reference.** Via age reference matrix.

**Files invoked.** `age_reference_matrix.py` — helper `age_ref_percentiles(class, age)` returns the 5-percentile vector for the patient's age.

**The math.** Per class `c`:
1. Linearly interpolate the 5 percentile points (A_p10, A_p25, A_p50, A_p75, A_p90) at the patient's chronological age between adjacent decade midpoints.
2. Determine which inter-percentile band the patient's A falls into.
3. Linear interpolation within the band gives a refined percentile estimate.

Example: patient A = 1.08 for immune class at age 52. At age 52, immune A_p50 = 1.04, A_p75 = 1.09. Patient is at ~73rd percentile.

**CMB equivalent.** **Within-survey-cohort percentile of a derived parameter.** When a survey reports a measurement of S_8 = 0.776, the survey can report "this is at the 23rd percentile of the Planck-fit S_8 posterior, given the systematic uncertainties." Same operation: position a measurement against a calibrated distribution.

**How the methylome differs in implementation.** Methylome percentile is per-class against an age-stratified HC distribution. Same principle.

**How it's the same in principle.** Percentile-against-calibrated-distribution is the operational way to communicate where a single measurement sits relative to expected.

**Outputs.** Per-patient per-class `percentile_rank` (8 values). Reported in the customer's report at Stage 9.

**Decision points.** None.

**Failure modes.**
- Patient chronological age outside reference range (4-95) → flag with `AGE_OUT_OF_RANGE`; percentile reported as boundary.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §58. Step 6.7 — Stage 6 output: per-class cellular age vector + optional summary

**What this step does.** Consolidates Stage 6 outputs into the per-cohort artifact.

**Inputs.** Outputs of §54 + §55 + §56 + §57.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Published per-survey cosmological-parameter table with percentile context.

**How the methylome differs in implementation.** Per-patient vs per-survey. Same content structure.

**How it's the same in principle.** Multi-parameter measurement output with context.

**Outputs.** <cellular-age output — emitted by `iam_cellular_age_scoring.py` (real module per inventory)> carrying per patient:
- 8 per-class cellular ages.
- 8 per-class status codes (OK / SAT_HIGH / SAT_LOW / INSUFFICIENT).
- 8 per-class percentile ranks at chronological age.
- `saturation_signature` (string description).
- `summary_cellular_age` (optional weighted mean, OK classes only).
- Patient's chronological age (for cross-reference).

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 7 (§59 — tier breakpoints).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (close).

---

# Stage 7 — Tier breakpoint detection

Stage 7 turns the framework's continuous measurements (A-scores, cellular ages) into the **discrete tier calls** the customer report and the cards consume. Engine-tier vocabulary is internal (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH); customer-tier vocabulary is collapsed (NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED). The cfDNA branch activates when substrate is plasma. **Stage 7 does no new physics — it applies pre-specified breakpoints and language collapse to outputs already produced upstream.**

---

## §59. Step 7.1 — Per-class A-score tier call (`tier_breakpoints.json v1.2`)

**What this step does.** For each architectural class, maps the patient's A-score to an engine tier using the 6-tier physics-derived breakpoints in `tier_breakpoints.json v1.2`. The breakpoints are universal (same across all classes); per-class structural ceilings (1/H_min) cap the highest reachable tier per class.

**Inputs.** Per-class A from Stage 4 (§46) + per-class A 95% CI propagated from MCMC posteriors + patient intake covariates (for the override modes) + `tier_breakpoints.json v1.2` + (optional) Stage 4.5 directional composite from §46.5 if FLAG_BIDIRECTIONAL is set.

**Atlas reference.** Indirect: structural ceiling per class is `1 / H_min(class)` from the frozen MCMC posteriors (G-003b freeze 2026-04-06). Tier breakpoints (1.07 Warburg + 1.10 breach) are physics-defined inflection points, not statistical percentiles.

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/Tier_breakpoints/tier_breakpoints.json` (v1.2). Engine consumes the JSON via a small helper. v0 4-tier statistical-percentile predecessor archived in `Tier_breakpoints/OLD/tier_breakpoints_v0_4tier_statistical.json`.

**The math.** Per class `c`, using the 6-tier system v1.2:

| Engine tier | Condition | Customer label | Physics meaning |
|---|---|---|---|
| SUPPRESSED | A[c] < 0.95 | Suppressed | Below baseline — context-dependent (treatment/transplant/immunosuppression) |
| NORMAL | 0.95 ≤ A[c] < 1.04 | Normal | Within healthy sampling variance |
| ELEVATED | 1.04 ≤ A[c] < 1.07 | Elevated | Recoverable drift; intervention window |
| WARBURG_TRANSITION | 1.07 ≤ A[c] < 1.10 | Warburg Transition | **1.07 Warburg line** — intervention character changes from "add fuel" to "restrict and rebuild" |
| SIGNIFICANTLY_ELEVATED | 1.10 ≤ A[c] < 1.12 | Significantly Elevated | Structural-fidelity breach territory; trajectory direction is primary read |
| BREACH | A[c] ≥ 1.10 sustained OR A[c] ≥ 1.12 single-timepoint | Breach | **1.10 architectural-fidelity breach line** — prompt for clinical workup; NOT a diagnosis |

Per-class structural ceiling (`structural_ceiling_by_class` in tier_breakpoints.json v1.2): if a class's `1/H_min` is below 1.10, the BREACH tier is structurally unreachable for that class. stem_pluri (ceiling 1.0181) is structurally blind for BREACH; SIGNIFICANTLY_ELEVATED is the practical ceiling. Runtime saturation margin: 0.005 below the ceiling, engine emits SATURATED flag.

**Override modes (v1.2):** Per the patient intake covariates routed at BUILD_SPEC v1.2 §4.5, the standard 6-tier output is replaced with mode-specific interpretation when triggered:
- **EXPECTED_SUPPRESSION** (current_immunosuppression / transplant_status) — SUPPRESSED reading interpreted as therapeutically expected
- **TRAJECTORY_WATCH** (autoimmune / chronic inflammatory / HIV+ treated) — ELEVATED floor shifted upward to 1.10; trajectory is primary
- **TREATMENT_RESPONSE** (current_cancer_in_treatment) — trajectory framing across treatment timepoints
- **CONTEXT_PREGNANCY** / **POSTPARTUM** — physiological immune shift framing
- **CONTEXT_HRT_BASELINE** — HRT-stratified baseline (CPG-VAL-018)
- **CONTEXT_WEIGHT_LOSS_INTERVENTION** (GLP-1 / bariatric) — expected anti-inflammatory trajectory (CPG-VAL-021)

**Smoking-bin interim mitigation (v1.2; retires when `IAMAtlas_smoking_layer.csv` fit at v1.3):** ELEVATED floor shifted by smoking_bin: current=1.10 / former_0_5y=1.08 / former_5_15y=1.07 / former_15plus_y=1.05 / never=1.04.

**Bidirectional pattern handoff (v1.2):** When Stage 4.5 (§46.5) sets `FLAG_BIDIRECTIONAL = True` for a class, this step uses the directional composite (signed) rather than the pooled A-score to drive tier reporting. Mapping per `bidirectional_pattern_handoff.directional_composite_tier_mapping`:
- |a_dir| < 0.40 → NORMAL
- 0.40 ≤ |a_dir| < 0.80 → ELEVATED
- 0.80 ≤ |a_dir| < 1.20 → WARBURG_TRANSITION
- 1.20 ≤ |a_dir| < 1.60 → SIGNIFICANTLY_ELEVATED
- |a_dir| ≥ 1.60 → BREACH-ANALOG

**Tier confidence propagation (v1.2):** Tier confidence is the probability of A falling in each tier under the MCMC-propagated posterior distribution. When |P(primary_tier) − P(second_max_tier)| < 0.20, engine emits BORDERLINE_TIER flag and customer report says "your reading straddles the {tier_A}/{tier_B} boundary."

**CMB equivalent.** Significance tier of a cosmological measurement (3σ / 5σ thresholds), but with physics-defined inflection points rather than statistical percentiles. The 1.07 Warburg line is analogous to a phase-transition threshold in cosmology — the same intervention has different effects above and below the line.

**How the methylome differs in implementation.** The 1.07 Warburg line + 1.10 breach line are framework-internal physics inflection points, not statistical percentiles relative to a null. The 6-tier system also propagates CI-based tier confidence forward (BORDERLINE_TIER) and supports covariate-conditional override modes — neither has a direct CMB analog.

**How it's the same in principle.** Continuous → discrete via pre-specified thresholds + forward propagation of measurement uncertainty into the tier confidence.

**Outputs.** Per-patient per-class:
- Primary engine tier (one of 6)
- Customer-facing label (matching engine label v1.2)
- BORDERLINE_TIER flag (when tier-boundary straddler detected)
- Override mode in effect (when covariate-triggered)
- Customer paragraph (rendered from tier × override-mode lookup)

**Decision points.** Override modes activate via covariate-trigger lookup; bidirectional handoff activates when Stage 4.5 set FLAG_BIDIRECTIONAL; smoking-bin floor-shift activates pre-Stage-3-foreground-fit.

**Failure modes.** None at this step (pure mapping). Downstream Stage 8 evaluates multi-class breach + override-mode compatibility.

**Canonical cross-references.** BUILD_SPEC v1.2 §5 Stage 7 + §3.4 (tier_breakpoints v1.2 schema). Recipe §7 (tier specification).

**CPG Plate references.** None directly.

**Chain-link assignment.** L8 (parameter inference — discrete tier readout with forward CI propagation).

---

## §60. Step 7.2 — Per-cell-type A-score tier call

**What this step does.** Same operation as §59, but for the 115 per-cell-type A-scores. Cell-type tier calls feed Stage 8 card matching.

**Inputs.** Per-cell-type A from Stage 4 (§46) + cell-type-specific breakpoints (inherit from parent class breakpoints, with optional cell-type-specific overrides in `tier_breakpoints.json`).

**Atlas reference.** Indirect.

**Files invoked.** Same as §59.

**The math.** Identical to §59 with the cell type's parent class breakpoints (or overrides where applicable).

**CMB equivalent.** Per-sub-component tier readouts. Same operational principle.

**How the methylome differs in implementation.** 115 cell types vs 8 classes — finer granularity, same operation.

**How it's the same in principle.** Same as §59.

**Outputs.** Per-patient per-cell-type engine tier (115 values).

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---

## §61. Step 7.3 — cfDNA branch (when substrate is plasma) — `cfdna_weight.json`

**What this step does.** **Activates only when the patient's substrate is plasma cfDNA.** When activated, compares the patient's per-class fractions to the expected healthy-blood cfDNA tissue-of-origin weights (Snyder 2016 + Moss 2018) and flags departures as pan-tissue context. **This is the only Stage 7 step that conditionally activates.**

**Inputs.** Per-class fractions from Stage 2 (§34) + `cfdna_weight.json` + patient `substrate` from manifest.

**Atlas reference.** None directly. The cfDNA weights are an external literature-derived reference (Snyder 2016, Moss 2018), used at Stage 7 as a clinical-context overlay — not as a framework calibration.

**Files invoked.** `cfdna_weight.json`. Per-class scoring module consumes the JSON.

**The math.** Healthy-plasma cfDNA per-class expected weights:
- immune: 0.70
- cycling: 0.12
- secretory: 0.08
- stromal: 0.04
- stem_adult: 0.03
- progenitor: 0.02
- terminal: 0.005
- stem_pluri: 0.005

Per class `c`:
```
departure_c = observed_fraction_c - expected_weight_c
```
Positive departure = elevated representation of that tissue-of-origin in plasma; negative = suppressed.

**Notable pattern:** Terminal-class (especially neurons) elevation in plasma cfDNA is a marker for tissue damage or turnover (e.g., neurological insult). Cycling-class elevation can indicate increased cellular turnover (regenerative tissues or proliferative diseases).

**CMB equivalent.** **Per-channel foreground subtraction with channel-specific templates.** When Planck observes the same sky pixel across multiple frequencies, each frequency has a different expected foreground contribution. The departures from expected (after foreground subtraction) are the signal. The methylome's cfDNA departures from expected weights are the substrate-specific equivalent.

**How the methylome differs in implementation.** Activates conditionally on substrate (plasma only); tissue and buffy substrates skip this step. Frequency-channel foreground subtraction is not conditional in CMB (all frequencies always contribute).

**How it's the same in principle.** Substrate-specific systematic baseline subtracted to surface the true departure signal.

**Outputs.** Per-class `cfdna_departure` value when substrate is plasma. Flagged as pan-tissue context in the report.

**Decision points.**
- Substrate = plasma → execute step.
- Substrate = tissue or buffy → skip step; report has no cfDNA section.

**Failure modes.**
- Substrate field missing or unknown → flag and skip cfDNA step.

**Canonical cross-references.** Recipe §7 (cfDNA branch). Runtime Matrices README "Stage 4 cfDNA".

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---

## §62. Step 7.4 — FLOOR_BREACH detection

**What this step does.** Cross-class detection of patients whose A-scores exceed expected ranges in **multiple** classes simultaneously. A single FLOOR_BREACH in one class is informative; multiple FLOOR_BREACH-es across uncorrelated classes indicates a global pathology (or massive batch effect, or substrate misidentification).

**Inputs.** Per-class engine tiers from §59.

**Atlas reference.** None.

**Files invoked.** Tier aggregation logic.

**The math.** Count classes with engine tier `FLOOR_BREACH`. Aggregation:
- 0 FLOOR_BREACH → no global flag.
- 1 FLOOR_BREACH → standard single-class breach; flag in report.
- 2-3 FLOOR_BREACH-es → flag `MULTI_CLASS_BREACH`; manual review at Stage 9.
- ≥ 4 FLOOR_BREACH-es → flag `GLOBAL_BREACH`; hold patient at Stage 9; do not auto-deliver report until reviewed.

**CMB equivalent.** **Multi-parameter tension detection.** When multiple cosmological parameters jointly disagree with Planck (e.g., DES S_8 + H_0 + Ωm all 2σ low), the joint tension is reported even when no single parameter is 5σ off. Same logic: multiple simultaneous departures get aggregated into a higher-significance flag.

**How the methylome differs in implementation.** Cell-class FLOOR_BREACH count vs parameter-tension count. Same aggregation principle.

**How it's the same in principle.** Joint failures across multiple framework dimensions warrant higher-significance flagging than any single failure.

**Outputs.** Per-patient `global_breach_flag ∈ {none, multi_class, global}`.

**Decision points.**
- `none` or `multi_class` → proceed.
- `global` → HOLD report; route to manual review.

**Failure modes.** Global breach is itself a finding, not a failure.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---


**Step 7.4b — Bidirectional flag for BELOW_NORMAL + documented-suppression-pattern coincidence.** A class scoring `BELOW_NORMAL` (A-score significantly below age-matched HC expectation) is informative on its own, but becomes a Stage-8-relevant flag when the patient's Stage 5 / Stage 8 outputs ALSO match a card or matrix pattern that includes documented suppression signature for that same class. Two named instances make this concrete:

- **AD-immune B-vs-T-lineage divergence** (AD card, when built): if patient shows immune class A-score elevated AND B-lineage cell types DOWN while T-lineage cell types UP (or vice versa), the bidirectional pattern is itself the signature — neither direction alone fires. Card returns BIDIRECTIONAL_PATTERN tier.
- **Long-window pre-diagnostic breast secretory homogenization** (breast-epic card): if patient shows secretory class A-score below age-matched HC AND breast-epic residual map's hypomethylation pattern hits, the homogenization is the signature. Card weights the BELOW_NORMAL secretory as POSITIVE evidence rather than treating it as "uninteresting." Anchored by VAL-047 Phase 6 Deep Audit GSE51057 (10yr+ d=−1.226, p=3×10⁻⁴).

The orchestrator's Stage 7 emits the `bidirectional_flag` field for every class scoring BELOW_NORMAL. Stage 8 consumes the flag when evaluating cards and matrix candidates that document suppression directions. Suppression is signal — the SOP forbids silently dropping it as "low priority."

## §63. Step 7.5 — Engine-to-customer language mapping

**What this step does.** Translates engine-tier vocabulary into customer-facing vocabulary. Customer reports never see engine-internal labels like FLOOR_BREACH or URGENT.

**Inputs.** Per-class engine tiers from §59 + per-cell-type engine tiers from §60.

**Atlas reference.** None.

**Files invoked.** `tier_breakpoints.json` — its `customer_label_map` section.

**The math.** Per engine tier:

| Engine tier | Customer label |
|---|---|
| BELOW_NORMAL | SUPPRESSED |
| NORMAL | NORMAL |
| MARGINAL | ELEVATED |
| DETECTABLE | SIGNIFICANTLY_ELEVATED |
| URGENT | SIGNIFICANTLY_ELEVATED |
| FLOOR_BREACH | SIGNIFICANTLY_ELEVATED |

The customer label DOES NOT distinguish DETECTABLE from URGENT from FLOOR_BREACH. **This is intentional.** All three engine tiers indicate the patient has departed significantly from baseline; the engine-tier distinction is for internal triage and for the chain audit trail. The customer label collapse exists to avoid implying clinical-grade discrimination between "detectable" and "urgent" that the framework's confidence intervals don't yet support.

**CMB equivalent.** **Mapping internal cosmological-tier vocabulary to public-language descriptors.** When Planck reports "3.5σ tension with DES S_8," internal language is precise; public-press-release language is "in tension with low-redshift surveys." The collapse exists to avoid implying false precision.

**How the methylome differs in implementation.** Per-class label per patient vs per-parameter label per release. Same operational structure.

**How it's the same in principle.** Engine-internal precision is collapsed into customer-facing labels that match what the framework's confidence supports.

**Outputs.** Per-patient per-class customer label + per-patient per-cell-type customer label.

**Decision points.** None — pure mapping.

**Failure modes.** None.

**Canonical cross-references.** Recipe §7. Part I §10 (Stage 9 legal boundary).

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

## §64. Step 7.6 — Stage 7 output: per-class tier vector + customer labels

**What this step does.** Consolidates Stage 7 outputs into the per-cohort artifact.

**Inputs.** Outputs of §59 + §60 + §61 + §62 + §63.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Per-survey tier-readout table.

**How the methylome differs in implementation.** Per-patient row. Same content structure per row.

**How it's the same in principle.** Discrete tier output with context.

**Outputs.** <tier vector — emitted internally by `GAPE_WEB_v13.py`> carrying per patient:
- 8 per-class engine tiers + 8 customer labels.
- 115 per-cell-type engine tiers + 115 customer labels.
- `cfdna_departures` per class (if applicable).
- `global_breach_flag`.

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 8 (card matching — covered in Part II-C).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

**End of Part II-B (Stages 5 through 7, §47-§64). Part II-C continues with Stages 8 (card matching), 9 (report assembly), 10 (delivery). Part III follows with chain-integrity scaffolding (L9 null suite). Part IV with failure modes. Part V with reference tables.**

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*
