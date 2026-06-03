# CPG Chain-of-Custody SOP — Part II-C (Stages 8 through 10) (v1.2 — walkthrough aligned)

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


**Continues from Part II-B (§47-§64). This document contains §65-§79 — Stages 8 (card-level pattern matching), 9 (report assembly), 10 (delivery).**

---

# Stage 8 — Card-level pattern matching

Stage 8 is where the engine stops measuring and starts interpreting. Stages 0 through 7 produced a tier vector — eight per-class engine tiers, 115 per-cell-type engine tiers, optional cfDNA departures, a global floor-breach flag. Stage 8 takes that tier vector and asks a different kind of question: **does this pattern match a known disease signature?**

The answer is **never** a diagnosis. The answer is a card-specific tier call — "the pattern matches the breast-epic long-pre-dx signature with confidence X" — which the report layer in Stage 9 then wraps in legally-permissible language. Card matching is the inflection point between the physics (which is the same for everyone) and the cards (which encode specific disease-state hypotheses against which patterns are tested).

The disease signature matrix v1.5 (77 rows × 131 columns, 354 populated signature cells) is the lookup table. Every row is a disease × time-range × substrate × severity-class combination. Every column is one of 131 immune sub-cell-types and tissue-of-origin cells. Every populated cell is an expected effect size range (e.g., `+0.81/+1.26` meaning "Cohen's d expected somewhere in [+0.81, +1.26]"). The matrix is the empirical posterior of every CPG-VAL ever sealed.

The chain-of-custody discipline here is strict: a card-specific tier call is a **conjunction of conditions**. Multiple per-class tiers, multiple per-cell-type departures, optional cfDNA flags, all need to satisfy the card's declared pattern at the card's declared confidence threshold. No single signal carries a card on its own. **The card never fires on one tile.** That rule is in `breast-epic_card_v2_3.json` and in every other production card by construction.

---

## §65. Step 8.1 — Disease signature matrix v1.5 lookup

> **CRITICAL — Stage 8 runs TWO parallel matching paths, NOT sequential.** Both consume the same Stage 4 + Stage 5 + Stage 7 outputs; both produce verdicts; both flow into Stage 9 report assembly. They are complementary, not redundant. (Walkthrough §4 Stage 5.)
>
> **Path A — Per-card matching** (this section + §66–§69). Each card in `DISEASE_MAPS_CARDS/{card_id}/` is loaded, eligibility-gated against patient metadata, evaluated against its panel + residual map + bimodality map + PCA projections + covariate thresholds, and returned with verdict (FIRED / NOT_FIRED / NOT_ELIGIBLE) + tier + contributing pattern + `educational_page_url`.
>
> **Path B — Disease signature matrix lookup** (this step §65 + §67's matching). The 77 × 131 matrix is scored row-by-row against the patient's per-cell-type A-score profile via the schema's `compute_match_magnitude()` (Mahalanobis-style sign-aligned product weighted by √n — NOT raw dot product). Top-3 candidates returned with tier (from `compute_customer_tier(match × phase × severity × evidence)`) + `organ_pages_to_link`.
>
> **Worked example of why dual matching catches what cards alone miss.** A patient with `regulatory_T_cells +1.2 + erythroid_progenitor +0.8 + pancreatic_beta_cells +1.0 + multi-organ distributed elevation` returns from Path A: NO single card fires above ELEVATED (no card uses that exact combination). From Path B: `breast_cancer / long_pre_dx` is the strongest matrix match — the >10yr distributed pre-diagnostic signature with 7 cells contributing. Without the matrix, the report says "everything looks slightly off, nothing fires" and misses the pattern. With the matrix, the report says "this combination of cellular drift most resembles the documented pattern for [X] at [phase]." Both paths report; neither overrides the other; Stage 9 surfaces both.


**What this step does.** For each card that applies to the patient's substrate (e.g., buffy coat DNA → breast-epic, lung-epic, prostate-epic, cardio-epic, AD-immune, MS-immune, Parkinson-immune, CRC-immune-inv all apply; plasma cfDNA → adds hcc-cfdna, pancreatic-cfdna), pull the relevant disease signature rows from the v1.5 matrix and prepare them for pattern matching.

**Inputs.**
- Patient substrate (from Stage 0 manifest).
- Stage 7 output: <tier vector — emitted internally by `GAPE_WEB_v13.py`>.
- The disease signature matrix CSV.

**Atlas reference.** None directly. The disease signature matrix is **derived from** IAMAtlas posteriors (every populated cell traces back to a CPG-VAL that ran the chain from L1 through L9), but Stage 8.1 reads the matrix directly, not the atlas.

**Files invoked.**
- Module: `<card-matching logic inside `GAPE_WEB_v13.py`>`
- Lookup table: `disease_cell_signature_matrix_v1_5.csv` (77 rows × 131 columns, 354 populated cells, SHA-256 hashed at load).
- Card registry: `<card registry — currently embedded in `GAPE_WEB_v13.py`>` — maps each card to the disease_id rows it pulls.

**The math.** None at this step — it's a SQL-like lookup. For a card with `disease_id=breast_cancer` and substrate `whole_blood_buffy_coat`, pull all rows where `disease_id == 'breast_cancer' AND substrate == 'whole_blood_buffy_coat'`. The result for breast-epic is four rows: `long_pre_dx` (>10y), `mid_pre_dx` (5-10y), `mid_late_pre_dx` (2-5y), `near_dx` (within 2y) — each row containing the expected per-cell-type Cohen's d ranges for that phase.

**CMB equivalent.** The **template bank** in matched-filter searches — LIGO's compact-binary template bank, Planck's primordial-non-Gaussianity template comparison, the cosmological-parameter grid evaluation. A measurement is compared against a set of pre-declared signal templates, and the template that best matches (subject to declared thresholds) is the candidate hypothesis. The templates themselves are not derived from this measurement; they were laid down in advance by independent calibration work.

**How the methylome differs in implementation.** The templates are time-phased — long_pre_dx vs near_dx are different templates of the same disease's evolution, not different diseases. CMB templates are typically time-static (the universe doesn't change between two CMB observations); methylome templates encode disease progression.

**How it's the same in principle.** Both are template banks. Both are declared in advance. Both are tested by overlap with measurement, not by training on the measurement.

**Outputs.** A per-patient, per-card candidate-template dictionary: `{card_id: [phase_template_1, phase_template_2, ...]}`. Each phase template carries the expected per-cell-type effect-size ranges in v1.5 format.

Stored in-memory for Stage 8.2 (residual map application). Not persisted as a separate file.

**Decision points.**
- If no cards apply to the substrate (which should never happen in production — every supported substrate has at least one card by the engine deployment rule), the patient's Stage 8 output is empty. The chain proceeds to Stage 9 with a "no applicable cards" flag.
- If the matrix SHA-256 at load doesn't match the registered version, the engine halts and refuses to process. No silent degradation.

**Failure modes.**
- **Matrix version mismatch.** The card registry pins a specific matrix version (v1.5 SHA). A mismatch means someone updated the matrix without updating the registry. Hard halt.
- **Substrate-card mismatch.** A card declared for plasma cfDNA cannot match a buffy coat sample. Caught at registry lookup — the patient simply doesn't get that card. Not a failure; an absence.
- **Empty card registry.** Indicates engine deployment misconfiguration. Hard halt.

**Canonical cross-references.** Recipe §8 (disease-state matching). Roadmap §10.2.6 Phase F (re-audit through complete chain). Disease matrix README at `disease_signature_matrix_README.md`.

**CPG Plate references.** None at this step.

**Chain-link assignment.** L7 (likelihood prep) + L8 (parameter inference prep).

---

## §66. Step 8.2 — Per-card residual map application

**What this step does.** For cards that ship with a **per-card residual map** (breast-epic has one at `breast_epic_residual_map_v0_1.csv`, 1,392 concordant CpGs with signed Cohen's d derived from CPG-VAL-003), apply the residual map to the patient's foreground-cleaned β matrix from Stage 3.

The residual map is a card's empirical per-CpG signal-direction blueprint at the disease's relevant phase. Applying it produces a **patient-level signal-overlap score** — how much of the card's expected residual pattern is present in this patient's β profile.

**Inputs.**
- Patient cleaned β matrix from Stage 3 output (the substrate after foreground subtraction).
- Card's residual map CSV (one file per card; not all cards have one).
- Card metadata declaring overlap method (default: Pearson correlation between patient's per-CpG departure and the residual map's signed Cohen's d).

**Atlas reference.** None directly. The residual map was built by running the full chain on a sealed VAL cohort (e.g., CPG-VAL-003 for breast-epic) and storing the resulting per-CpG signal direction. Atlas was consulted when the residual map was made; not when applying it.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Per-card residual maps live at `<per-card residual map — placeholder dir at `pipeline_runtime_matrices/card_residual_maps/`, populated as VALs lock>`. Currently in production: `breast_epic_residual_map_v0_1.csv`. Other cards pending residual map derivation per Phase F backlog.

**The math.** Given the patient's per-CpG departure vector δ_patient (length N_CpGs_in_map) and the card's signed Cohen's d vector d_card (same length, same CpG order):

> overlap_score = ρ(δ_patient, d_card)  
> where ρ is Pearson correlation, with confidence interval via Fisher z-transform.

A patient whose disease pattern matches the card's residual map will have overlap_score significantly positive (typically ρ > 0.10 with N=1,392 is p < 0.001). A patient unrelated to the card's disease will have overlap_score consistent with zero.

A patient with a **reversed** signal (signal exists but pointing the opposite direction — possible when a different disease drives similar CpGs in the opposite direction) will have overlap_score significantly negative. The sign of overlap_score is itself diagnostic.

**CMB equivalent.** The **matched-filter inner product** — `<data | template>` weighted by inverse noise covariance — that LIGO, EHT, and CMB template searches all compute as their primary statistic. The Pearson form here is a noise-flat approximation; a proper matched filter would weight each CpG by its inverse per-CpG variance from L6. Phase D delivers that upgrade.

**How the methylome differs in implementation.** The "template" is per-CpG signed Cohen's d, not a continuous frequency-domain waveform. The noise weighting is currently approximated as flat (each CpG weighted equally) because per-CpG covariance hasn't been delivered by Phase D yet. Phase D upgrade replaces ρ with the full whitened inner product.

**How it's the same in principle.** Both are inner products of a measurement against a declared template, both produce a signed and magnitude-bounded scalar, both can be evaluated for significance against null.

**Outputs.** Per-card overlap score with 95% CI. Signed. Stored in the patient's Stage 8 working dictionary.

**Decision points.**
- If the card has no residual map, Step 8.2 is skipped for that card and pattern matching at Step 8.3 proceeds without the residual-overlap channel.
- If overlap_score has the wrong sign for the disease state being matched, the card's phase template is downgraded — the engine has detected a signal but in the opposite direction. Reported as flag, not as match.

**Failure modes.**
- **CpG coverage mismatch.** Residual map declares 1,392 CpGs; patient may have <90% coverage due to EPIC vs 450K mismatch or QC failures. Default policy: require ≥80% CpG coverage of the residual map; below that, residual-overlap channel is marked INSUFFICIENT and matching falls back to per-class-tier-only matching at 8.3.
- **All-zero patient departure.** A patient whose foreground-cleaned β matrix is essentially flat (e.g., no decoherence above noise floor) will have undefined overlap_score (zero numerator and zero denominator in the Pearson form). Caught with an explicit zero-variance check; reported as NO_SIGNAL.

**Canonical cross-references.** Recipe §8.2. VAL-003 outcome (residual map source). breast-epic card README at `breast-epic_README.md`.

**CPG Plate references.** **Plate 2** (Breast Pre-Diagnostic Anisotropy) shows the residual map for breast-epic rendered as a sky map. Every dot on Plate 2 is one CpG with its signed Cohen's d coloring; the patient overlap calculation here is, conceptually, "how aligned is this patient's per-CpG signal sky with the breast-epic anisotropy sky." **Plate 4 Panel F** shows the same data with the cyan-shifted (hypomethylated) field effect made explicit.

**Chain-link assignment.** L7 (likelihood, partial — residual-overlap is the per-card likelihood proxy until full Bayesian likelihood ships in Phase E).

---

## §67. Step 8.3 — Multi-class pattern matching

**What this step does.** Combine the per-class and per-cell-type tier vector from Stage 7 with the residual-overlap scores from Step 8.2 to produce a card-level tier call. The matching rule is encoded in each card's JSON; the engine reads the rule and applies it mechanically — it does not invent new matching rules.

**Inputs.**
- Stage 7 tier output for this patient.
- Step 8.1 candidate phase templates per card.
- Step 8.2 residual-overlap scores per card (where available).
- Card JSON: `<per-card definition — currently embedded in `GAPE_WEB_v13.py`>` — declares the matching rule.

**Atlas reference.** None directly.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Cards. Currently in production: breast-epic v2.3, lung-epic, prostate-epic, hcc-epic, heme-epic, cardio-epic, AD-immune, MS-immune, Parkinson-immune, CRC-immune-inv, CRC-secretory, cervical-secretory, LGG/GBM-terminal, pancreatic, aging-baseline.

**The math.** Each card's matching rule is a Boolean expression over:
- Per-class engine tiers (e.g., `immune_tier >= DETECTABLE`)
- Per-cell-type engine tiers (e.g., `Baso_tier >= MARGINAL AND breast_BE_tier >= MARGINAL`)
- Per-cell-type effect-size ranges from v1.5 matrix (e.g., `Baso_A_score within [1.01, 1.58]`)
- Residual-overlap thresholds (e.g., `breast_epic_residual_overlap > 0.10 AND CI_lower > 0`)
- Phase-template disjunctions (e.g., `MATCHES(long_pre_dx) OR MATCHES(mid_pre_dx)`)

For each candidate phase template, evaluate the rule. The phase whose rule evaluates TRUE with highest residual-overlap CI lower bound is the card's reported phase. Confidence is the residual-overlap CI lower bound (or, when no residual map exists, the minimum of the per-class tier confidences).

**Worked breast-epic example, long_pre_dx phase, hypothetical patient:**
- Stage 7 tiers: immune_engine=DETECTABLE, secretory_engine=NORMAL, stem_pluri_engine=NORMAL, stromal_engine=MARGINAL, ...
- Per-cell-type tiers: Baso_engine=DETECTABLE (A=1.42), Plasma_engine=DETECTABLE (A=1.18), breast_BE_engine=MARGINAL (A=1.08), microglia_engine=DETECTABLE (A=1.21).
- Step 8.2 residual-overlap: ρ=0.143, 95% CI [0.092, 0.193], p<10⁻⁵.
- v1.5 long_pre_dx row says: Baso d∈[+1.01,+1.58], Plasma d∈[+0.81,+1.26], breast_BE d∈[+0.61,+1.28], microglia d∈[+0.71,+1.30], immune_pooled d=+1.78.
- Patient's per-cell-type A-scores translated to Cohen's d via the IAMAtlas covariance: Baso d≈+1.30 (in range), Plasma d≈+0.95 (in range), breast_BE d≈+0.74 (in range), microglia d≈+1.05 (in range).
- Boolean rule: `immune_tier >= DETECTABLE AND >=3 of (Baso, Plasma, breast_BE, microglia, NeuMa, Mela, neurons_pooled, smooth_muscle) in expected range AND residual_overlap_CI_lower > 0`.
- Evaluation: TRUE. Card fires for long_pre_dx phase with confidence 0.092 (the CI lower bound).

**CMB equivalent.** The **template-likelihood evaluation** in cosmological-parameter inference. Each candidate parameter set evaluates the likelihood of the data given that template; the maximum-likelihood template (subject to prior) becomes the report. The chain-of-custody discipline here is that the Boolean rule is **declared in the card before the patient is run** — the equivalent of pre-registering the template grid before the data arrives.

**How the methylome differs in implementation.** The Boolean rule is currently threshold-based rather than likelihood-based. Phase E delivers the upgrade to full Bayesian likelihood evaluation per card.

**How it's the same in principle.** Both ask: given a measurement and a set of pre-declared templates, which template best matches? Both protect against post-hoc template invention by pre-registering the template grid.

**Outputs.**
- Per-card phase verdict: MATCH_long_pre_dx, MATCH_mid_pre_dx, ..., NO_MATCH, INSUFFICIENT, NEGATIVE_OVERLAP.
- Per-card confidence (lower bound of residual-overlap CI; or minimum of per-class tier confidences when no map).
- Per-card matched-element trace: which Boolean components were TRUE, which were FALSE.

Stored in-memory; flows to §68.

**Decision points.**
- Each card produces independently. A patient can match breast-epic long_pre_dx AND lung-epic mid_pre_dx — the engine reports both. Decision combining is the customer's physician's job, not the engine's.
- If a card produces both a MATCH and a NEGATIVE_OVERLAP for different phases, the card fires with the MATCH phase but adds a contradictory-overlap flag for Stage 9 review.

**Failure modes.**
- **Card rule syntax error.** Caught at engine startup when card registry loads — a malformed Boolean expression triggers hard halt.
- **Phase verdict ambiguity.** Multiple phases of the same card pass the rule with overlapping confidence intervals. Default policy: report the highest-confidence phase; add an ambiguous-phase flag.
- **Residual-overlap channel missing AND per-class tiers all NORMAL.** The card produces NO_MATCH. Not a failure; an absence of pattern.

**Canonical cross-references.** Recipe §8.3. Each card's JSON contains its own matching rule (`breast-epic_card_v2_3.json` §`matching_rule`).

**CPG Plate references.** None at this step (the matching logic itself doesn't have a sky-projection visual).

**Chain-link assignment.** L7 + L8 (partial).

---

## §68. Step 8.4 — Card-specific covariate adjustment

**What this step does.** Some cards declare covariates that need to be applied **inside the card** rather than as engine-wide foregrounds. Smoking history is the prototypical example: lung-epic adjusts for smoking pack-years as a within-card covariate because smoking is part of the disease's causal pathway, not an external nuisance. Subtracting smoking as a global foreground would erase real lung-disease signal.

The card's JSON declares which covariates are within-card. Engine reads the declaration and applies the adjustment locally before finalizing the card's tier call.

**Inputs.**
- Card's matched verdict from §67.
- Patient metadata from Stage 0 manifest (smoking history, BMI, prior treatment, etc. — whatever the card requires).
- Card JSON covariate declaration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Per-card covariate transfer functions live in each card's JSON (`within_card_covariates` block).

**The math.** Each within-card covariate is a linear adjustment to the card's confidence value. For lung-epic with smoking:

> adjusted_confidence = raw_confidence + smoking_penalty(pack_years)

where `smoking_penalty` is monotonically negative (heavy-smoker patients have **lower** card-firing confidence because the residual-overlap with non-smoker-derived signature templates is partially explained by smoking). For cardio-epic with BMI:

> adjusted_confidence = raw_confidence + bmi_adjustment(bmi)

where `bmi_adjustment` is non-monotonic (both under- and over-BMI penalize signature match).

The transfer functions are **per-card declared** in the JSON. The engine does not invent them. A new card adding a new covariate adjustment must declare it in the JSON before deployment.

**CMB equivalent.** The **nuisance-parameter marginalization** in cosmological likelihood evaluation, where some parameters (galactic-foreground amplitudes per frequency, calibration offsets, beam-window-function uncertainties) are nuisance parameters integrated over before reporting cosmological parameters. The within-card covariates are nuisance parameters whose effect on the card's confidence must be integrated before reporting the patient's card-firing confidence.

**How the methylome differs in implementation.** Linear adjustment in the current engine; full Bayesian marginalization in Phase E.

**How it's the same in principle.** Both protect the headline result from being driven by an unmodeled covariate.

**Outputs.** Adjusted per-card confidence values. The card's MATCH/NO_MATCH verdict from §67 is **never** changed by §68 — only the confidence is adjusted. A card that fires with adjusted confidence ≤ 0 is reported as "fired below threshold" — the pattern is consistent with the card's template but explained by the within-card covariate.

**Decision points.**
- If patient metadata required for the covariate adjustment is missing (e.g., smoking history not provided for a lung-epic match), the card's confidence is flagged as UNADJUSTED. The card still reports; the report includes an explicit "covariate-uninformative" disclaimer.

**Failure modes.**
- **Missing metadata.** Handled as UNADJUSTED flag (above).
- **Covariate transfer function out of declared range.** A patient with 200 pack-years (impossible) or BMI 60 (extreme) falls outside the card's declared covariate range. Engine clamps to the range boundary and adds a clipped-covariate flag.

**Canonical cross-references.** Recipe §8.4. Each card's `within_card_covariates` JSON block. lung-epic and cardio-epic card READMEs document the per-card transfer functions.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (likelihood, partial — within-card nuisance handling).

---

## §69. Step 8.5 — Stage 8 output: card verdict package

**What this step does.** Consolidate Steps 8.1-8.4 into a Stage 8 output file. One file per cohort run, hashed.

**Inputs.** All Stage 8 intermediate outputs for all patients.

**Atlas reference.** None.

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Per-mission template-match catalog.

**How the methylome differs in implementation.** Per-patient row.

**How it's the same in principle.** Discrete card verdict with confidence and trace.

**Outputs.** <card verdict package — emitted internally by `GAPE_WEB_v13.py`> carrying per patient:
- All applicable cards' verdicts (MATCH_{phase} / NO_MATCH / NEGATIVE_OVERLAP / INSUFFICIENT).
- All applicable cards' raw and covariate-adjusted confidences.
- All applicable cards' matched-element traces.
- Card residual-overlap scores per card.

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 9 (report assembly).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §8.5.

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

# Stage 9 — Report assembly (the legal boundary layer)

Stage 9 is where the chain of custody ends and the customer's understanding begins. Stages 0 through 8 produced an exhaustive, internally-consistent record of how a patient's IDAT became a card verdict. Stage 9 translates that record into language a non-clinician can act on — without ever telling them anything CPG is not legally permitted to tell them.

The boundary discipline (§10 of Part I) is the entire substance of Stage 9. Every word that goes into the rendered report is checked against:
1. **The Stage 6 engine-to-customer language collapse** (NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED — never FLOOR_BREACH, never URGENT).
2. **The literature anchors** (we say "in studies, people with your reading had outcome X" — never "you have outcome X").
3. **The cancer prior + family history multiplier** (base-rate context, never deterministic risk).
4. **The CANNOT_SAY list** (no diagnoses, no actions, no future-states, no recommendations).

Stage 9 has seven steps (§70-§76). They run **in order**: the language collapse first, then literature, then prior context, then family history, then sex adjustment, then rendering, then the final legal-boundary check. The last step is a gate: a report that fails the legal-boundary check at §76 does not ship from §77.

---

## §70. Step 9.1 — Customer-facing language collapse

**What this step does.** Apply the Stage 6 engine-to-customer language mapping (introduced at §63 in Part II-B) to every engine tier in the patient's Stage 7 output. After this step, no engine vocabulary survives into the report.

**Inputs.** Stage 7 per-class and per-cell-type tier vector for this patient.

**Atlas reference.** None.

**Files invoked.**
- Module: `<engine-to-customer language collapse inside `GAPE_WEB_v13.py`>`
- Lookup: `tier_breakpoints.json` (engine-to-customer mapping block, same file used at §63).

**The math.** Pure lookup.

Engine → Customer:
- NORMAL → NORMAL
- MARGINAL → NORMAL (slight margin variation)
- DETECTABLE → ELEVATED
- URGENT → SIGNIFICANTLY_ELEVATED
- FLOOR_BREACH → SIGNIFICANTLY_ELEVATED (a customer never sees the breach language)

**CMB equivalent.** The **mission-public release pass**: internal-precision results (microKelvin per HEALPix pixel, with full covariance) get collapsed into the public-release temperature map (Mollweide, smoothed, with confidence band, with the survey mission's narrative wrapper). The internal precision survives in the science release; the public release is human-readable.

**How the methylome differs in implementation.** Per-patient collapse. Same content structure per patient.

**How it's the same in principle.** Internal precision is preserved in the audit trail (Stage 7 output); customer report is the legally-permissible projection of it.

**Outputs.** Patient's tier vector with engine vocabulary replaced by customer vocabulary. Stored in-memory for §71-§76.

**Decision points.** All Stage 9 downstream steps operate on the collapsed vector, not the engine vector.

**Failure modes.**
- **Mapping table missing or wrong version.** Caught by SHA-256 at load. Hard halt.
- **Engine tier outside declared mapping range.** Should never happen if Stage 7 is correct, but caught as INVALID_TIER → report fails legal boundary check at §76 and does not ship.

**Canonical cross-references.** Recipe §9.1. §63 (Step 7.5 mapping definition) in Part II-B.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer).

---

## §71. Step 9.2 — Literature anchors (the reporting-layer translator)

**What this step does.** For each card that fired in Stage 8, attach the literature anchor that gives the patient's reading clinical context — phrased as "in published studies, patients with [your reading range] had outcome [X]" rather than "you have outcome X."

The literature anchor is **not measurement.** It is a reporting-layer translator. The measurement is the A-score and the card verdict. The literature anchor is a legally-permissible language wrapper that lets a customer understand what the measurement means in clinical terms used by their physician.

This is **the only place in the chain** where information from external (non-IAMAtlas, non-IAM-derived) sources enters the customer-facing output. The discipline: literature anchors come from peer-reviewed publications cited in `literature_anchors.json`, with explicit DOI and effect-size mapping. Never paraphrased from secondary sources. Never invented.

**Inputs.**
- Stage 8 card verdict package from §69.
- Patient's per-class A-scores from Stage 4.
- Patient's cellular ages from Stage 6.
- Patient's Mahalanobis distance from Stage 5.

**Atlas reference.** None directly. (The literature anchor table was built once, off-engine, by vetting each citation against the framework's measurement scale. The vetting consulted IAMAtlas; the runtime lookup does not.)

**Files invoked.**
- Module: `<anchor selection inside `GAPE_WEB_v13.py`>`
- Lookup: `literature_anchors.json`. Schema per anchor: `{card_id, phase, reading_range, citation_doi, effect_summary, language_template}`.

**The math.** Lookup keyed by `(card_id, phase, reading_range_bucket)`. The reading_range_bucket is the patient's measurement discretized into the literature anchor's defined buckets (e.g., for breast-epic long_pre_dx with Mahalanobis d ∈ [+1.5, +2.5], the anchor cites the Xu 2020 JNCI cohort outcome at that effect range).

**CMB equivalent.** The **cosmological-significance language** in survey-mission public releases — Planck's "Our results are consistent with the six-parameter ΛCDM model" wrapper around the underlying TT/TE/EE/BB likelihood. The wrapper is what the public understands; the underlying likelihood is what the science is. Both must agree, but they are not the same.

**How the methylome differs in implementation.** Per-patient anchor selection (different patients in the same card get different anchors based on their reading bucket). Public-release language wrappers don't change per reader.

**How it's the same in principle.** A legally and ethically permissible language wrapper around a precise measurement.

**Outputs.** Per-card literature anchor blocks with full citation, effect summary, and the customer-facing language template the renderer (§75) will use.

Stored in-memory; flows to §72.

**Decision points.**
- If a card fires but no literature anchor exists for the patient's reading bucket, the card is reported with a "limited published context" disclaimer rather than fabricated language. **Never invent.**
- If the literature anchor's effect summary conflicts with the patient's measurement direction (rare — would happen if a card mis-fires in opposite sign), the conflict is logged for review and the report uses a conservative "elevated reading; specific clinical interpretation requires physician review" wrapper.

**Failure modes.**
- **Anchor citation 404.** All DOIs in `literature_anchors.json` are checked at engine deployment time. A dead DOI fails deployment, not patient runtime. (Engine refuses to deploy until anchors are fully resolvable.)
- **Anchor table version mismatch.** SHA-256 hashed. Hard halt on mismatch.
- **No anchor at all for a fired card.** Card was deployed without anchor support. Deployment-time check should catch this. If somehow it survives to runtime, the card reports with the "limited published context" disclaimer.

**Canonical cross-references.** Recipe §9.2. The literature_anchors.json schema doc. Per-card READMEs each list their primary citations.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer).

---

## §72. Step 9.3 — Cancer prior context

**What this step does.** Attach population-base-rate context for every card that fires. The cancer prior is the published lifetime incidence (US SEER + age-stratified) for the disease the card targets. It is presented as base-rate context — "US lifetime incidence of breast cancer is approximately 13% in women; your reading places you at the [P] percentile of women your age" — not as a personalized risk number.

**Inputs.**
- Stage 8 fired cards.
- Patient sex, age from Stage 0 manifest.

**Atlas reference.** None.

**Files invoked.**
- Module: `<prior lookup inside `GAPE_WEB_v13.py`>`
- Lookup: `cancer_prior.json`. Schema per prior: `{disease_id, sex, age_decade, lifetime_incidence, citation}`.

**The math.** Lookup keyed by `(card_disease_id, patient_sex, patient_age_decade)`. Returns the base-rate language block to be wrapped by the report renderer.

**CMB equivalent.** The **prior π(θ)** in Bayesian likelihood evaluation. A measurement combined with a prior produces a posterior; reporting the measurement without the prior is misleading because it implies the measurement IS the posterior. The cancer prior is the population prior on disease state, not on the measurement.

**How the methylome differs in implementation.** Per-patient prior because the prior is age/sex-conditioned. CMB priors are typically uniform or simple-Gaussian and don't vary across observations.

**How it's the same in principle.** Both attach base-rate context to a measurement.

**Outputs.** Per-card cancer-prior block with citation. Flows to §73.

**Decision points.**
- If patient sex or age is missing, prior is reported as US-population-overall (less precise but legally available).
- If the card's disease has no published prior at the precision needed (rare — most malignancies have SEER coverage), the card's report block omits the prior context and adds a "population context unavailable" line.

**Failure modes.**
- **Prior table version mismatch.** SHA-256 hashed. Hard halt.
- **Missing patient metadata.** Falls back to overall-population prior; flagged.

**Canonical cross-references.** Recipe §9.3.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §73. Step 9.4 — Family history multiplier

**What this step does.** Apply the family-history multiplier to the cancer prior when the patient's Stage 0 manifest declares first-degree relative history of the disease.

The multiplier itself is a published, peer-reviewed effect size (e.g., for breast cancer: first-degree relative ≈ 1.7× lifetime incidence; bilateral first-degree ≈ 2.5×). The multiplier IS the language; CPG does not modify the multiplier per patient. **The multiplier is applied to the prior, not to the measurement.**

**Inputs.**
- §72 prior block.
- Patient family history declaration from Stage 0 manifest.

**Atlas reference.** None.

**Files invoked.**
- Module: `<multiplier lookup inside `GAPE_WEB_v13.py`>`
- Lookup: `family_history_multiplier.json`. Schema per multiplier: `{disease_id, relative_relationship, multiplier, citation}`.

**The math.** `adjusted_prior = base_prior × family_history_multiplier(disease, relationship)`. Discrete multiplier per relationship category (first-degree single, first-degree multiple, second-degree, etc.). Multipliers are published values, not engine-derived.

**CMB equivalent.** The **prior factorization** in hierarchical Bayesian inference — splitting the prior into a population-level component and an individual-level conditioning component.

**How the methylome differs in implementation.** Per-patient conditioning.

**How it's the same in principle.** Both refine a prior with declared individual information before combining with measurement.

**Outputs.** Adjusted-prior block: `adjusted_prior = base_prior × multiplier`, with both the base rate and the multiplier displayed transparently in the report (the report shows both numbers).

**Decision points.**
- If the patient declares no family history, multiplier defaults to 1.0 and the prior block flows through unchanged.
- If the patient declines to share family history, the report says so explicitly: "Family history not reported; population-base context only."

**Failure modes.**
- **Multiplier table version mismatch.** SHA-256 hashed. Hard halt.
- **Patient declared family history of a disease not in the multiplier table.** Multiplier defaults to 1.0; flagged.

**Canonical cross-references.** Recipe §9.4.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---


**Step 9.4b — Risk-context formula (the integration step).** The cancer prior (§72), age factor, sex factor, family history multiplier (§73), and Stage 8 match magnitude combine into a single per-class context-posterior used by the renderer (§75):

```
posterior_context_class = baseline_prior          (from cancer_prior.json)
                        × age_factor              (US incidence by decade)
                        × sex_factor              (e.g., secretory: female 1.4×, male 1.2×)
                        × fh_factor               (from family_history_multiplier.json IF present; else 1.0)
                        × match_magnitude         (from Stage 8 matrix match for this class)
```

**Framing rule — strict (walkthrough §4 Stage 6).** The doctor report and any future patient report use the phrasing "your reading combined with your risk context suggests…" — NEVER "you have a high probability of…". This is risk-context integration, not diagnosis. The legal-boundary gate (§76) catches probability-statement violations.

**Conditional consumption.** When `patient_metadata.family_history` is absent or empty, `fh_factor = 1.0` and the formula falls back to overall-population context. The audit trail records that family history was not supplied; the doctor report's risk-context section notes the same in the Quality block (§75).

**Differentiation matters.** The formula is what differentiates "your secretory class shows A=1.06 and your mother had no cancer history — keep watching and adjust lifestyle" from "your secretory class shows A=1.06 and your mother died of breast cancer at 52 — discuss accelerated screening with your clinician now." Same A-score, two different risk contexts, two different report framings. This is why Stage 9 is not a pure rendering layer — it's a context-integration layer that the rendering layer (§75) then translates.

## §74. Step 9.5 — Sex-specific risk adjustment

**What this step does.** Some cards are sex-conditioned (breast-epic primarily female; prostate-epic male-only; cervical-secretory female-only). The sex adjustment ensures cards do not report on the wrong sex with population-level context that doesn't apply.

**Inputs.**
- §73 adjusted prior block.
- Patient sex from Stage 0.
- Card sex-condition declaration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<sex conditioning inside `GAPE_WEB_v13.py`>`
- Card JSON sex-condition block.

**The math.** Conditional. If `patient_sex ∈ card.applicable_sex`, card flows through. Else, card is dropped from the report (the engine had a verdict, but the card doesn't report on this patient's sex). For partial-sex cards (e.g., male breast cancer is rare but real), the multiplier shifts to the appropriate male-sex prior at §72-§73 and the card reports with explicit minority-sex context.

**CMB equivalent.** None directly — CMB has no sex-conditioning. The closest analog is **survey-area conditioning**: a polarization measurement reported only over the survey footprint, not extrapolated to the whole sky.

**How the methylome differs in implementation.** Per-patient conditioning is part of the report-assembly logic, not the measurement.

**How it's the same in principle.** Reports only on regions where the measurement framework applies.

**Outputs.** Filtered card set with sex-appropriate context. Flows to §75.

**Decision points.**
- Card excluded by sex: card drops from report; logged in audit trail.
- Card retained with minority-sex context: card reports; explicit minority-sex framing in language.

**Failure modes.**
- **Card sex-condition declaration ambiguous.** Caught at card deployment time, not patient runtime.

**Canonical cross-references.** Recipe §9.5.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §75. Step 9.6 — Report rendering pass

**What this step does.** Compose the final report from the per-card blocks assembled by §70-§74. The renderer is **template-driven** — each card has a declared report template; the renderer fills slots in the template with the patient's specific values. The renderer does not invent language. Every customer-facing word is either a template literal or a slot value from one of the upstream steps.

**Inputs.** All per-card blocks from §70-§74.

**Atlas reference.** None.

**Files invoked.**
- Module: `<report rendering inside `GAPE_WEB_v13.py`>`
- Per-card templates: <inside `GAPE_WEB_v13.py` — no standalone module file at present writing> (Markdown with slot syntax).
- Global template: `<global report skeleton — inside `GAPE_WEB_v13.py`>` (the report skeleton — patient header, summary, per-card sections, disclaimers).

**The math.** None — string composition via Mustache-style templating.

**CMB equivalent.** The **publication LaTeX rendering** of cosmological-parameter results — a typeset paper-grade document assembled from a parameter table and a citation database, with no manual prose interpolated at rendering time.

**How the methylome differs in implementation.** Per-patient rendering with per-patient slot values.

**How it's the same in principle.** Mechanical composition of declared template + declared values. No interpretive room at this step.

**Outputs.** A rendered Markdown report. Per-patient file at `<draft report — emitted internally by `GAPE_WEB_v13.py`>`. Status: DRAFT until §76 clears it for shipping.

**Decision points.** Report DRAFT flows to §76 legal boundary check.

**Failure modes.**
- **Missing slot value.** Caught by the renderer — any unfilled slot triggers REPORT_INCOMPLETE; report does not advance to §76.
- **Template syntax error.** Caught at card deployment time.

**Canonical cross-references.** Recipe §9.6.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §76. Step 9.7 — Legal boundary check (the gate)

**What this step does.** The final check before a report ships. The DRAFT report from §75 is scanned for any language that violates the CANNOT_SAY list. A report that contains forbidden language **does not ship**. The patient's run is flagged for manual review; the engine does not silently degrade.

**Inputs.** DRAFT report from §75. The CANNOT_SAY list.

**Atlas reference.** None.

**Files invoked.**
- Module: `<legal-boundary gate inside `GAPE_WEB_v13.py`>`
- The CANNOT_SAY list: `<legal-boundary CANNOT_SAY list — TBD; currently enforced inline within `GAPE_WEB_v13.py`>`.

**The CANNOT_SAY list (current as of v1):**
- Diagnostic statements: "You have [disease]", "You are at risk of [disease]", any verb conjugation suggesting determinism.
- Action statements: "You should [action]", "You must [action]", any recommendation involving medication, dosage, procedure, lifestyle change beyond "discuss with your physician."
- Future-state statements: "You will [outcome]", any statement about future probabilities phrased deterministically.
- Treatment statements: any reference to specific drugs, dosages, or procedures.
- Prevention statements: any claim about preventing a disease.

**The CAN_SAY list (by design):**
- Measurement statements: "Your immune class A-score is [value]."
- Percentile statements: "This places you at the [percentile] of the healthy reference population at your age."
- Literature anchor statements: "In published studies (citation), patients with A-scores in this range had outcome [X]."
- Base-rate statements: "US lifetime incidence of [condition] is approximately [N]%."
- Family history conditioning: "Family history of [condition] in a first-degree relative raises the population base rate to approximately [N × multiplier]%."
- Recommendation to discuss with physician: "Consider discussing this reading with your physician."

**The math.** Regex + keyword scanning. Each forbidden phrase or pattern in CANNOT_SAY triggers a violation flag. A report with ≥1 flag halts. A report with 0 flags clears.

**CMB equivalent.** The **collaboration internal review** before public release of a cosmological-parameter measurement. A Planck paper does not ship from the analyst to arxiv directly — it passes through collaboration-wide internal review where the language and the conclusions are checked against the discipline's standards. The CANNOT_SAY check is the CPG version.

**How the methylome differs in implementation.** Automated check rather than human review (because it runs per-patient at scale). Periodic manual sampling of cleared reports is part of QA discipline.

**How it's the same in principle.** Both prevent a measurement from being communicated in a way that misrepresents what was measured.

**Outputs.** Cleared report: `<cleared report — emitted internally by `GAPE_WEB_v13.py`>`. Or HALT_FOR_REVIEW flag with detailed violation report.

**Decision points.**
- 0 violations → CLEARED. Report flows to Stage 10.
- ≥1 violations → HALT_FOR_REVIEW. Report does not ship until a human reviewer fixes the source (template, literature anchor, or covariate language) and re-renders.

**Failure modes.**
- **False positive (cleared CAN_SAY language flagged).** Adjust the CANNOT_SAY regex; do not weaken the discipline.
- **False negative (forbidden language not flagged).** Add the missed pattern to CANNOT_SAY immediately; re-scan all already-cleared reports from the current cohort run.

**Canonical cross-references.** Recipe §9.7. **§10 of Part I** (the foundational legal boundary discussion).

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer — the gate).

---

# Stage 10 — Delivery

Stage 10 is the smallest stage. The report has been cleared by §76; the engine now packages it, routes it, and closes the audit trail. Three steps.

---

## §77. Step 10.1 — Report packaging

**What this step does.** Wrap the cleared Markdown report into the delivery format the customer or partner expects. Default is PDF via the LaTeX→PDF rendering pipeline. Some lab partners receive HTML.

**Inputs.** Cleared report from §76.

**Atlas reference.** None.

**Files invoked.**
- Module: `<delivery packaging inside `GAPE_WEB_v13.py` — or `web.commercial.py` orchestrator when built>`
- Templates: `<delivery templates — TBD>` (LaTeX, HTML).

**The math.** None.

**CMB equivalent.** Public-release file packaging (FITS, HEALPix WCS, with documented file format).

**Outputs.** `<final report — emitted internally by `GAPE_WEB_v13.py`>` with embedded checksums and engine version metadata.

**Decision points.** Cohort proceeds to §78.

**Failure modes.** Rendering pipeline failure → flagged; cleared report retained as Markdown for manual repackaging.

**Canonical cross-references.** Recipe §10.1.

**CPG Plate references.** None.

**Chain-link assignment.** L1 (audit-trail close).

---

## §78. Step 10.2 — Delivery channel routing

**What this step does.** Route the packaged report to the configured delivery channel (customer portal, lab partner API, secure email, etc.). The channel is per-customer or per-partner; the engine reads the delivery configuration and sends.

**Inputs.** Packaged report from §77. Customer/partner delivery configuration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<channel routing inside `GAPE_WEB_v13.py` — or `web.commercial.py` orchestrator when built>`
- Configuration: `<per-customer delivery configuration — TBD>`.

**The math.** None.

**CMB equivalent.** Survey data-release distribution (Planck Legacy Archive, NASA's HEASARC).

**Outputs.** Delivery confirmation per recipient; delivery failures retried with bounded retry policy.

**Decision points.**
- Delivery success → §79.
- Delivery failure after retry exhaustion → flagged; report queued for manual handling.

**Failure modes.**
- Channel authentication failure → bounded retry; manual handoff after exhaustion.
- Channel throughput limit hit → queue with rate-limit backoff.

**Canonical cross-references.** Recipe §10.2.

**CPG Plate references.** None.

**Chain-link assignment.** L1.

---

## §79. Step 10.3 — Audit trail capture (the chain closes)

**What this step does.** The final step. Every output from every stage for this patient — Stage 0 manifest hash, Stage 1-3 β matrix hashes, Stage 4 A-scores, Stage 5 Mahalanobis, Stage 6 cellular ages, Stage 7 tiers, Stage 8 card verdicts, Stage 9 cleared report hash, Stage 10 delivery confirmation — is captured into an audit-trail record.

The audit trail is the answer to "show me how this number was produced." A question about any patient's reading should be answerable in full by reading the audit trail back through the chain.

**Inputs.** Every prior stage's hashed output for this patient.

**Atlas reference.** The atlas SHA-256 fingerprint used at every stage that consulted the atlas is recorded.

**Files invoked.**
- Module: `<audit-trail capture — TBD when `web.commercial.py` orchestrator is designed>`
- Audit log: `<audit log — TBD per orchestrator design>`.

**The math.** None — pure logging.

**CMB equivalent.** The **observation log** of a mission, plus the **likelihood pipeline record** of the analysis, plus the **publication-time methods section** — collapsed into a single per-observation audit record.

**How the methylome differs in implementation.** Per-patient audit. Each patient gets a full chain trace.

**How it's the same in principle.** Both record everything needed to reproduce a measurement after the fact.

**Outputs.** Per-patient audit JSON with:
- Stage 0-10 input/output hashes.
- IAMAtlas SHA-256 fingerprint at every stage that consulted it.
- Engine version + module versions at every stage.
- Decision-point flags from every stage.
- Failure-mode flags from every stage.
- L9 null-suite results (if the patient was part of a sealed VAL cohort).
- Final cleared report hash.

The audit log is **append-only**. A patient's audit record is written once and never modified. Re-runs produce new run_id's with their own audit records.

**Decision points.** The chain ends here. Audit trail is the closing record.

**Failure modes.** Audit capture failure is the only Stage 10 failure that triggers a hard halt — a report cannot ship without its audit record. If the audit log write fails, the report is held at §78.

**Canonical cross-references.** Recipe §10.3. **§3 of Part I** (the chain-link-to-stage mapping that this audit closes).

**CPG Plate references.** None directly — but the audit trail of a sealed VAL becomes the data behind any future plate produced from that VAL.

**Chain-link assignment.** L1 closes the loop. The chain is complete.

---

**End of Part II-C (Stages 8 through 10, §65-§79).**

**End of Part II — the complete operational chain from IDAT-on-server to audit-trail-closed.**

Part III follows with the chain-integrity scaffolding (L9 null suite, synthetic patients, VAL sealing protocol — §80-§91). Part IV with failure modes and decision trees (§92-§96). Part V with reference tables (§97-§102).

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*
