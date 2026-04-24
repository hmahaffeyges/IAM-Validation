# EDEAR Cookbook — Master Lessons Learned Catalog

**Generated:** 2026-04-24
**Maintainer:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Purpose:** Cross-card catalog of every quirk, failure mode, and methodological lesson encountered during EDEAR validation and deployment. Each disease card also contains its own per-card `lessons_learned` section; this document aggregates across cards so patterns become visible.

---

## Why this document exists

The EDEAR Cookbook is a set of disease cards that share a universal pipeline (Stage 1 Xu-538 immune + Stage 2 Moss NNLS + Stage 3 EpiDISH). During validation, real data reveals behaviors the universal pipeline did not anticipate. A previously-correct v2.0 card becomes v2.1 because a validation run exposed a quirk that needs to be embedded in the card's operational spec.

This catalog records those quirks at the cross-card level so:

1. A new analyst can scan for patterns that may apply to the card they are building.
2. Collaborators and potential acquirers can see legitimate testing discipline — every surprise documented, every failure mode acknowledged, no retrospective smoothing.
3. Future pre-registrations can lock the analyses the lesson reveals were missing.

---

## Cross-card lessons (apply to every card)

### CCL-001 — The Directional-Score Principle (discovered via VAL-050/VAL-051)

**Source:** ad-immune validation, April 2026
**Lesson:** Pooled-entropy A-score fails when a disease drives per-CpG drift bidirectionally within the same panel — some CpGs up, some down. H(β) is symmetric around β=0.5, so signed Δβ contributions cancel at the pooled-mean level.

**Evidence:** VAL-050 AIBL AD d=+0.077 null (10/18 positive Δβ, 8/18 negative). VAL-051 same cohort same 7 CpGs after direction weighting d=+0.624.

**Embedded rule:** Every card reports BOTH pooled-entropy A-score AND a directional A_dir score. Uniform-direction diseases (breast, CRC) use pooled as primary. Bidirectional diseases (AD, likely autoimmune) use directional as primary. Stage 1 evaluates both on every IDAT so no disease signal is silently zeroed.

**Cards affected:** all. ad-immune uses A_dir primary; breast-epic, crc-epic, lung-epic use A_pooled primary.

---

### CCL-002 — Sex stratification is mandatory (discovered via VAL-051/VAL-057)

**Source:** ad-immune validation
**Lesson:** AD signal magnitude differs substantially by sex (AIBL female d=+0.71 vs male d=+0.51). When cohort sex composition differs from training cohort, pooled-sex analysis can produce a null that masks a real per-sex signal.

**Evidence:** VAL-057 GSE53740 pooled d=+0.013. Sex-stratified: male d=+0.415 (n=7, replicates AIBL male d=+0.512); female d=−0.131 (n=7, fails to replicate AIBL female d=+0.705). The pooled null came from opposing sex contributions averaging to zero.

**Embedded rule:** Every card's Stage 1 report MUST include patient sex. Cards with documented sex-differential signal (ad-immune at v2.1) apply sex-specific calibration. Cards without documented sex-differential still record sex as a mandatory covariate pending future sex-stratified validation.

**Cards affected:** all; implemented explicitly in ad-immune v2.1.

---

### CCL-003 — Pre-registration must lock stratifications, not just the primary test (discovered via VAL-057)

**Source:** ad-immune VAL-057 pre-registration failure
**Lesson:** A pre-registration that locks only a pooled primary test is insufficient. The original VAL-057 pre-reg sealed 2026-04-24 05:44 UTC did not pre-register sex stratification (despite VAL-051 having reported it), per-CpG directional preservation (despite the 7-CpG panel having 2:5 direction split), or 80-cell age anchor (despite the Cookbook containing the baseline). All three omissions were added post-hoc after the pooled null, which is honest to disclose but less referee-defensible than full pre-registration.

**Embedded rule:** Every future validation pre-registration must lock:

1. Primary pooled test with decision matrix.
2. Sex-stratified tests per disease group.
3. Per-CpG directional preservation check.
4. 80-cell age anchor (or equivalent) cohort-offset check.
5. Cohort batch-offset sanity check.

Each with its own pre-specified decision rule. All five must be sealed with SHA before data access.

**Cards affected:** all future validations; codified in ad-immune lesson ad-LL-005.

---

### CCL-004 — The 80-cell baseline is not universal across preprocessing pipelines (discovered via VAL-057)

**Source:** ad-immune VAL-057 Analysis 4
**Lesson:** The Cookbook's 80-cell age-decade immune baseline was derived from cohorts with standard preprocessing (primarily minfi/sesame). Cohorts with different preprocessing (e.g. Ferrari 2014 GSE53740 with ComBat + quantile normalization) can show cohort-level batch offsets of +2 SD or more vs the 80-cell baseline. This is a systematic offset from preprocessing differences, not a biological finding.

**Evidence:** GSE53740 HC mean A_age_z = +2.306 SD above 80-cell baseline. Every GSE53740 sample — including healthy controls — reads as architecturally elevated.

**Embedded rule:** The 80-cell baseline is directly applicable on AIBL-equivalent and AddNeuroMed-equivalent preprocessing. On any other cohort, deployment requires either (a) re-anchoring to within-cohort HC for tier thresholds, or (b) running a normalization bridge to the 80-cell scale before applying Cookbook tier thresholds.

**Cards affected:** all. Universal-reference block now includes this warning in every card.

---

### CCL-005 — Panel specificity is not disease specificity (discovered via VAL-057)

**Source:** ad-immune VAL-057 Analysis 3
**Lesson:** A panel derived to discriminate disease X from healthy may also fire on disease Y if Y shares mechanistic features with X. The frozen AD Rule A 7-CpG panel preserved 5/7 direction signs on GSE53740 PSP/CBD samples vs only 4/7 on GSE53740 AD. This does not mean the panel doesn't detect AD; it means the panel's AIBL-derived direction pattern captures tauopathy-associated drift at least as well as AD-specific drift.

**Embedded rule:** Every card must be tested against adjacent non-target diseases before claiming disease-specificity. "Cross-platform validated on 2 AD cohorts" is insufficient to claim "AD-specific" — an FTD/PSP head-to-head is required. Future cards should include a pre-registered specificity-arm test alongside the primary disease validation.

**Cards affected:** ad-immune (explicit caveat added). Other cards flagged as needing future specificity testing against adjacent diseases.

---

### CCL-006 — Disease direction is not universal across cancer types (discovered via VAL-047 Phase 12)

**Source:** crc-epic validation
**Lesson:** Breast and CRC produce opposite Stage 1 A_immune signatures on the same Xu-538 panel: breast d=+0.65 (positive, immune activation consistent with immune escape) vs CRC d=−0.33 (negative, lower entropy consistent with Treg/immune-suppression dominance). Same panel, same population, opposite direction.

**Evidence:** VAL-047 Phase 9 GSE51057 (breast-only). VAL-047 Phase 12 GSE51032 (breast + CRC). Both published post-hoc analyses by Zhao 2020 corroborate the CRC signal (different methodology, consistent direction).

**Embedded rule:** Never assume an immune A_score panel has uniform direction across diseases. Every card specifies `expected_direction` (POSITIVE or NEGATIVE), and decision logic in README_MASTER handles both directions explicitly.

**Cards affected:** breast-epic (POSITIVE), crc-epic (NEGATIVE), ad-immune (POSITIVE via directional), lung-epic (POSITIVE per VAL-046 cohort anchor).

---

### CCL-007 — Near-diagnosis signal is not always stronger than long pre-diagnosis (discovered via VAL-047)

**Source:** breast-epic + crc-epic
**Lesson:** Breast signal attenuates from d=+0.71 (5-10yr pre-dx) to d=+0.37 (0-2yr pre-dx) — immune escape as the tumor crystallizes reduces the architectural signature near diagnosis. CRC signal INTENSIFIES from d=−0.33 (all pre-dx) to d=−0.47 (0-2yr pre-dx) — immune suppression persists and deepens.

**Embedded rule:** Tier thresholds should be calibrated to the disease's characteristic pre-dx window, not assumed to be monotonic. Breast EDEAR prefers 2-5yr+ trajectories; CRC EDEAR is valid at near-dx too.

**Cards affected:** breast-epic, crc-epic.

---

### CCL-008 — Adjacent-normal is not architecturally healthy (discovered via VAL-037/VAL-039/VAL-056)

**Source:** lung-epic, earlier cancer cards
**Lesson:** TCGA "solid tissue normal" (STN) adjacent to a tumor is architecturally elevated above true-healthy reference. VAL-037 quantified this across 24 TCGA types: mean ΔA_adjacent-normal = +0.036 (22.9% of tumor signal). VAL-039 showed the gradient extends with distance: Kadota 2014 lung near-2cm ΔA=+0.052, far-5cm ΔA=+0.017. Field effect extends past surgical resection margins.

**Embedded rule:** Never use adjacent-normal as a healthy reference. Use Moss 2018 healthy-donor tissue β or 80-cell reference (with cohort-normalization caveat per CCL-004).

**Cards affected:** all. Universal-reference block uses Moss 2018 Table S1 healthy reference β.

---

### CCL-009 — Smoking stratification is mandatory for lung (discovered via VAL-056 + Hong 2019 + Baglietto 2017)

**Source:** lung-epic validation
**Lesson:** Lung cancer methylation signature is smoking-stratified at the per-CpG level. Current smokers have elevated immune A-score from smoking-driven F2RL3/AHRR hypomethylation independent of cancer. TCGA-LUAD adjacent-normal ΔA=+0.030 reflects ~80% smoker prevalence in the cohort, not a pure field effect. Hong 2019 found cg12169243 (DPH6) and cg25429010 (IMP3) reach genome-wide significance in current smokers only. Baglietto 2017 established 5-10yr decay kinetics for smoking CpGs post-cessation.

**Embedded rule:** lung-epic uses four smoking strata (never / former ≥10yr / former <5yr / current) with per-stratum deployment rule. Current smokers require mandatory smoking-adjustment sentence in every report + tightened Stage 2 rule (top-1/top-2 ≥ 2× vs generic DETECTABLE).

**Cards affected:** lung-epic explicitly; other cancer cards should audit their cohort smoker prevalence.

---

### CCL-010 — Substrate-specific panel transferability (discovered via VAL-059)

**Source:** hcc-epic validation
**Lesson:** A panel validated on one blood-derived substrate does not automatically transfer to other blood-derived substrates. VAL-059 demonstrated this starkly for HCC: Xu-538 on ccfDNA plasma produced d = +0.634 on GSE298812 (HCC vs HCC-free controls in an HIV+ Nigerian cohort), but Xu-538 on whole-blood leukocyte DNA produced d = −0.156 on GSE281691 (metabolic HCC vs metabolic-liver-disease controls) — opposite direction, null magnitude. Two different substrates, same panel, completely different outcomes.

The mechanism is physical: whole-blood leukocyte DNA reflects the blood immune compartment directly (stable baseline of leukocyte methylation plus disease-associated immune drift). Plasma ccfDNA contains substantial tumor-shed and tissue-of-origin-shed DNA on top of the leukocyte-turnover baseline. A panel trained on breast immune drift may catch hepatocyte-derived ccfDNA for reasons (architectural drift signature) unrelated to its original training objective. The same panel applied to pure leukocyte DNA lacks the hepatocyte contribution.

**Embedded rule:** every card specifies its validated substrate(s). When expanding a card to a second substrate, treat it as a separate validation — not a replication. If substrates disagree, document the divergence and restrict the card to the validated substrate(s).

**Cards affected:** hcc-epic (ccfDNA plasma validated, whole-blood leukocyte NULL). All future cards that deploy across multiple specimen types.

---

### CCL-011 — First-card-specific tissue re-validation is the new minimum standard (discovered via VAL-058, first retroactive instance VAL-060)

**Source:** prostate-epic validation (VAL-058, first Cookbook card with its own tissue re-validation run — no public blood cohort available, so tissue was the primary arm) and breast-epic retroactive upgrade (VAL-060, first retroactive tissue arm added to a card that was already blood-validated).

**Lesson:** The first four Cookbook cards (breast-epic, crc-epic, ad-immune, lung-epic) originally relied on Moss 2018 atlas tissue reference β values for their Stage 2 expectations — they quoted Moss's numbers without re-running tissue case-control on their own panel. Two runs changed the standard. VAL-058 (prostate-epic) went tissue-first because no per-patient blood cohort existed: ran Xu-538 directly on GSE269244 n=238 tumor vs adjacent-normal African-American prostate tissue and demonstrated paired d = +0.497 on 118 matched patient pairs. VAL-060 (breast-epic) added the tissue arm retroactively to an already-blood-validated card: ran Xu-538 on TCGA-BRCA HM450 n=186 matched tumor-normal (86 complete pairs after score-validity filtering) and demonstrated paired d = +0.676, unpaired d = +0.745 [+0.451, +1.075] — larger effect than prostate, consistent with Xu-538 being the panel originally selected for breast cancer. Both runs confirm the Xu-538 panel produces tissue-level separation on the disease in question; the magnitude hierarchy (disease-of-panel-origin > cross-applied-disease) is itself a framework consistency check.

**Embedded rule (prospective, in force):** every new card runs tissue re-validation on a publicly available tumor-vs-normal or tumor-vs-tissue-matched-healthy cohort as part of its primary validation package. TCGA is the natural first stop for most common cancers; alternative cohorts used when TCGA is thin or missing the disease. The tissue arm gives each card its own tissue effect size on its own panel (not just Moss's atlas), documents whether the panel transfers meaningfully to that disease tissue, and establishes the foundation for per-patient blood claims. Tissue validation is necessary but not sufficient for per-patient blood tier — it is the ground truth against which blood signals are interpreted.

**Embedded rule (retroactive, in progress):** breast-epic completed as VAL-060 (2026-04-24). CRC-epic next as VAL-061 on TCGA-COAD. AD as VAL-062 on ROSMAP or BDR cortex cohort. Lung as VAL-063, extending VAL-056 Part 2's TCGA-LUAD/LUSC tumor-vs-adjacent-normal work into a properly pre-registered per-card tissue arm entry. Each retroactive run upgrades the card from `cross_platform_validated_two_cohorts` (or equivalent primary tier) to the same tier PLUS a `tissue_arm_validated` modifier flag (see master README tier definitions).

**Cards affected:** prostate-epic (original tissue-first build), breast-epic v2.2 (first retroactive tissue arm). All future cards will include tissue arm as a standard element. Retroactive upgrade queue: crc-epic → ad-immune → lung-epic.

**Observed pattern worth tracking:** breast tissue d = +0.676 > prostate tissue d = +0.497 on the same panel, same H_min, same pipeline. The Xu-538 panel was originally selected for breast cancer (Sister Study). Effect-size ordering across the two tissue runs so far is disease-of-panel-origin > cross-applied-disease. Prediction for CRC (VAL-061): because CRC is an inversion-direction disease (negative d on blood per VAL-047 Phase 12, d = −0.33), CRC tissue should also show negative direction if the framework is consistent. Positive tissue direction for CRC would be a framework inconsistency to investigate. This is a falsifiable prediction locked in before the run.

---

### CCL-012 — Public Moss vs proprietary H_min calibration boundary (discovered via VAL-058/VAL-059 amendment process)

**Source:** VAL-058 and VAL-059 amendment process
**Lesson:** Moss 2018 marker CpG list and reference matrix R are PUBLIC — published in Moss 2018 Supplementary Table S4 and mirrored on GitHub at `nloyfer/meth_atlas`. The NNLS deconvolution itself uses `scipy.optimize.nnls` which is open-source. What IS proprietary is the H_min calibration layer (G-003b MCMC posteriors per architecture class per substrate, covered under US Provisional Patents 64/012,720 and 64/014,568). The VAL-058 and VAL-059 amendments originally removed "Moss per-CpG reference metrics" from the pre-reg citing NDA-gating — that was over-cautious. A correctly-scoped public-facing script CAN use public Moss markers with scipy NNLS to produce per-tissue β estimates; the results are honest deconvolution output. The script cannot include H_min values for converting per-tissue β to architecture-class A-scores — that step is the proprietary calibration.

**Embedded rule:** two-layer design separates the two. Layer 1 (public, may appear in GitHub scripts): use public Moss S4 markers + `scipy.optimize.nnls` to produce per-tissue β. Layer 2 (vault-only, never in public scripts): apply H_min per architecture class to produce A-scores. Future VAL-058b / VAL-059b / VAL-060+ scripts may include Layer 1 metrics for Stage 2 validation.

**Cards affected:** all. Retroactive option: extend VAL-058 and VAL-059 with b-variant runs producing per-tissue β as independent metrics.

---

### CCL-013 — "Panel-specific direction" vs "genome-wide direction" are not the same (discovered via VAL-058)

**Source:** prostate-epic validation
**Lesson:** Berglund 2024 published GSE269244 reporting "an overall trend of hypermethylation in prostate tumors" (~5,139 differentially methylated CpGs at q<0.01, |Δβ|>0.2, genome-wide). VAL-058 on the same cohort restricted to Xu-538 panel CpGs found 217/481 hypermethylated in tumor (45.1%) vs 264/481 hypomethylated (54.9%) — panel-specific trend is slightly hypomethylation-predominant, opposite to the published overall trend. This is not a contradiction. The Xu-538 panel was trained on Sister Study breast cancer immune drift, not on the genome-wide Berglund trend. The panel-specific direction reflects which subset of CpGs are in Xu-538 AND differentially methylated in prostate tumor. The two directions (genome-wide vs panel-specific) answer different questions.

**Embedded rule:** when comparing panel behavior to a published disease signature, always check whether the published signature was defined on the same CpG subset or a different one. Published "overall" directions are not valid baselines for panel-specific directional analysis. Report both when relevant.

**Cards affected:** all cards reporting per-CpG direction preservation. Explicit caveat added to prostate-epic interpretation notes.

---

## Per-card lesson catalogs

Each card's `lessons_learned` section in the card JSON contains the disease-specific version of these plus disease-unique quirks. See:

- `breast-epic/breast-epic_card_v2.2.json` → `lessons_learned` (4 lessons; v2.2 adds breast-LL-004 from VAL-060)
- `crc-epic/crc-epic_card_v2.1.json` → `lessons_learned` (3 lessons)
- `ad-immune/ad-immune_card_v2.1.json` → `lessons_learned` (5 lessons)
- `lung-epic/lung-epic_card_v0.3.json` → `lessons_learned` (5 lessons)
- `prostate-epic/prostate-epic_card_v0.1.json` → `lessons_learned` (5 lessons)
- `hcc-epic/hcc-epic_card_v0.1.json` → `lessons_learned` (5 lessons)

---

## Process lessons

### PL-001 — Post-hoc analyses are valid but cost referee credibility

Post-hoc analyses done openly with explicit labeling (as in VAL-057 Analyses 2-5) are scientifically valid and essential for honest record-keeping. But a referee trained on pre-registration discipline will weigh them less than pre-registered analyses. The gap between "pre-registered" and "post-hoc" is credibility, not validity.

**Rule:** Pre-register everything that could plausibly matter. The cost of an over-specified pre-reg is near zero. The cost of an under-specified pre-reg is reputational.

### PL-002 — The honest record is the best defense

When a primary test comes back null and post-hoc analyses reveal qualification (VAL-057 male recovery), the honest approach is to present BOTH the pre-registered null AND the post-hoc recovery side-by-side with explicit labeling. This is more credible than either (a) hiding the post-hoc analyses to avoid "p-hacking" accusations or (b) re-running as a new "pre-registration" that cherry-picks the sex-split.

**Rule:** Every report, every card, every validation entry shows the full analysis including failures. Collaborators reading the record can then evaluate our judgment.

### PL-003 — Universal inline over DRY

The Cookbook deliberately duplicates universal-reference constants (H_min values, 80-cell baseline, Moss 2018 healthy β) into every card rather than cross-referencing a single source. This is anti-DRY on purpose: a card is self-contained, loadable with GAPE_WEB_v13.py alone, no hidden dependencies.

**Cost:** If the 80-cell baseline or an H_min constant updates, every card needs re-generation from the updated `universal_reference_block.py`.

**Benefit:** A partner running just one card can never fail because they didn't load README_MASTER, and our universal constants are visibly locked in every card JSON.

**Rule:** When constants change, re-run `update_all_cards_v2.1.py` to regenerate all card JSONs. Bump card versions uniformly. Record the change in LESSONS_LEARNED.md.

---

### CCL-019 — A-score direction depends on (class, compartment) pair, not disease alone (discovered via VAL-061)

When I ran VAL-061 to test CRC tumor tissue, I preregistered "tumor should go negative" because VAL-047 had established that peripheral blood immune-panel direction for CRC is negative (d = −0.33). I conflated two different compartments of the same immune class reading the same disease. The tumor came back d = +1.066 and I initially flagged it as framework inconsistency. Heath correctly identified the mix-up: peripheral blood immune cells read negative because the circulating immune compartment is suppressed/exhausted in response to disease presence, while tumor-infiltrating immune cells read positive because they are activated and expanded inside the tumor bed. Same immune class, same panel, opposite signs — because the compartment is different.

**The generalization:** specifying (class, compartment) as a pair is required for any directional prediction. "Immune class" alone is insufficient. "Immune class in peripheral blood" vs "Immune class in tumor-infiltrating" produce opposite-sign A-scores for the same disease.

**Preregistration rule going forward:** every directional prediction must name the compartment explicitly. Never assume blood and tissue readings of the same class will show the same direction for the same disease.

---

### CCL-020 — Panel choice vs class choice vs specimen choice are three independent dimensions (discovered via VAL-061 → VAL-062)

VAL-061 used the Xu-538 immune panel on CRC tumor tissue and measured the tumor-infiltrating immune compartment (d = +1.066). VAL-062 used the correct cycling-class scoring with all available CpGs on the same CRC tumor tissue and measured the tumor architecture (d = +0.724). Same tissue, same samples, same 26 patients. Two completely different readings. The difference was the scoring class and the CpG subset used, not the specimen.

**The three independent dimensions:**
1. **Specimen** — what physical sample are we running (blood, tissue biopsy, CSF, urine, stool, cervical mucosa)?
2. **Class** — what H_min are we scoring against (immune 0.839, cycling 0.856, secretory 0.843, terminal 0.773, stromal 0.863, etc.)?
3. **Panel** — what CpG subset are we scoring across (Xu-538 for immune signature; Moss 2018 colon markers for cycling-class colon; directional panels for specific diseases)?

A correct EDEAR call requires all three to match the clinical question. Mis-matching any one produces a technically valid but clinically meaningless reading. This was the VAL-061 failure: correct specimen (CRC tumor tissue), wrong panel-class combination (Xu-538 immune panel scored against immune H_min — measures the immune infiltrate, not the tumor).

**Rule for future cards:** every new card must explicitly document (specimen × class × panel) for every test arm. No single-dimension specifications.

---

### CCL-021 — The 4% cfDNA detection floor creates structured Stage-1-positive / Stage-2-null patterns that are disease-family-specific (discovered via immune-atlas authoring)

The per-class cfDNA contribution to healthy plasma:
- immune 70%, cycling 12%, secretory 8%, stromal 4%, stem_adult 3%, progenitor 2%, terminal 0.5%, stem_pluri 0.5%

The 4% detection floor (documented in GAPE_WEB_v13.py: "Everything below 4% cfDNA should be treated as exploratory only") excludes specific classes from reliable Stage 2 blood deconvolution. This is not a framework failure — it is honest physics. What it does create is **structured patterns of Stage-1-positive Stage-2-null results** that are disease-family-specific:

1. **Terminal class hidden by specimen** — brain cancer, advanced neurodegeneration. Plasma fails; CSF succeeds.
2. **Hematologic/immune-compartment disease** — AML, DLBCL, CLL, MM, thymoma. Stage 2 returns null because the immune class IS the diseased tissue. Stage 3 EpiDISH is the discriminator.
3. **Cardiovascular/systemic inflammation** — atherosclerosis, CHD, systemic inflammatory disease. Stage 2 returns null because the disease is systemic vascular inflammation, not localized.
4. **Unexplained early drift** — too early in any disease progression for Stage 2 resolution. Trajectory watch is the diagnostic.

**Rule:** Stage-1-positive Stage-2-null is not a failure case. It is a specific clinical signal that routes to one of four well-defined diagnostic pathways (documented in immune-atlas Pathways 1-4). Future cards for non-solid-organ disease (heme-epic, cardio-epic, future autoimmune cards) fire into these pathways by design, not as afterthoughts.

---

### CCL-022 — Single-timepoint EDEAR is a flag; serial-trajectory EDEAR is a diagnostic (discovered via framework trajectory predictions + Pudas 2023 AD negative result)

The Pudas 2023 AD study showed epigenetic age acceleration does not predict AD at single timepoints up to 16 years pre-onset. Yet the framework's trajectory predictions (G-2026-P001 through G-2026-P015, ten trajectory-based predictions across all eight classes) are explicitly serial-sampling-based. Both facts are consistent: single-timepoint methylation has limited predictive power for pre-clinical disease, while trajectory-of-methylation is diagnostic.

**Implications for Cookbook deployment:**
- Every card benefits from trajectory analysis; some cards (prostate indolent-vs-aggressive, CHIP→AML, ICI response, cardiovascular chronic disease) can only achieve their full diagnostic value with serial sampling.
- Subscription deployment model is not a commercial preference — it is the validation path for half the framework's dated predictions.
- Rate of change (dA/dt) and acceleration (d²A/dt²) are first-class diagnostic outputs, not afterthoughts.
- Trajectory-matches-disease-signature alerts (cosine similarity of current drift pattern vs known disease progression trajectories) become critical once the subscription cohort accumulates serial data.

**Rule:** Every new card must document its trajectory criticality (CRITICAL / HIGH / MODERATE / POST-DX ONLY / REFERENCE ONLY) and its trajectory-specific clinical actions. Cards that are CRITICAL or HIGH for trajectory cannot be deployed with single-timepoint data alone — they require the subscription model.

---

### CCL-023 — Direction of peripheral immune-class A-score may encode the disease's immune-modulatory phenotype and temporal stage (open hypothesis, two anchoring data points)

**The hypothesis (open).** Peripheral immune-class A-score direction may not just encode disease presence — it may encode (a) the disease's intrinsic immune-modulatory phenotype (activation vs suppression) and/or (b) the temporal stage of immune dysregulation (early-phase activation → late-phase suppression). Two cancers anchor opposite ends of this hypothesis at different stages:

- **CRC (validated, VAL-047):** pre-diagnostic peripheral blood immune-class direction is **negative** (d = -0.33 pooled across pre-dx cohorts, 5-10yr pre-dx window). Direction: suppression-shifted from healthy baseline. CRC drives peripheral immune suppression detectable a decade before clinical diagnosis.
- **AD (validated, VAL-051/052/Nabais 2021 n=3,424):** peripheral blood immune class direction is **positive** (d = +0.62 directional, smaller pooled). Direction: activation-shifted, neuroinflammation-pattern.
- **Breast / Lung / Prostate / HCC (validated):** all positive d at pre-diagnostic 2-10yr window. Activation-shifted phase.

**Glioma — supportive evidence at cell-fraction level, hypothesis-direction match:** Bracci/Wiencke 2022 (J Neuro-Oncol, n=139 pre-surgery glioma + 454 controls, EPIC array, dexamethasone-adjusted) reports significantly lower CD4, CD8, B-cell, NK, monocyte fractions AND significantly higher neutrophils in glioma vs controls (all p < 0.001). This shifted-toward-suppression composition would produce a **negative immune-class A-score** if scored through GAPE Stage 1 — same direction as CRC, opposite of AD/breast/lung/prostate. The β-value-level test cannot be performed on this cohort today because UCSF AGS data is controlled-access (multi-month application). The cell-fraction result is the strongest published support for the hypothesis. An earlier internal curiosity test on an active-chemo glioma cohort showed inverse direction but was confounded by chemotherapy effects and is NOT cited as evidence.

**The bigger framework implication.** The NLR (neutrophil-to-lymphocyte ratio) literature catalogs cancer-associated peripheral immune shifts at the time of diagnosis or post-treatment. EDEAR's framework operates 5-10 years pre-diagnostically. This means: A-score direction may encode the **temporal stage** of immune dysregulation, not just the disease. Some cancers (breast, lung, prostate, HCC) appear to show early-phase peripheral immune activation that later transitions to clinical-era suppression. Other cancers (CRC, possibly glioma) appear to show persistent peripheral suppression starting at the pre-diagnostic phase. If true, this is a major framework refinement: direction encodes (disease, temporal stage) pair, where temporal stage may be characterized by the trajectory rather than a single timepoint.

**Implications for card design.** This CCL is the empirical basis for the immune-atlas card's differential-diagnosis engine. It justifies treating direction (positive vs negative) as a first-order discriminator across diseases:

1. Stage 1 immune A-score POSITIVE → suspect cancers in the activation-phase set (breast, lung, prostate, HCC, pancreatic, AD-like neuroinflammation)
2. Stage 1 immune A-score NEGATIVE → suspect cancers in the suppression-phase set (CRC, possibly glioma, possibly advanced disease, possibly hematologic with suppressive subtype)
3. Stage 2 deconvolution provides the tissue-specific confirmation
4. Stage 3 EpiDISH cell composition provides the further differentiation (lineage-specific shifts for hematologic, monocyte-FOXP3 for cardiovascular, etc.)

**The "running disease-specific medical chart" use case.** The card structure with lessons-learned and future testing ideas is well-suited to capturing both the empirical A-score signature per disease AND the temporal-stage trajectory hypothesis. As validation data accumulates from prospective cohorts (G-2026-P003 prostate, P010 CHIP→AML, P011 ICI response, etc.), each card's "disease-specific medical chart" gains both single-timepoint signatures AND trajectory shapes.

**Strategic capture — beyond detection.** The framework may have utility beyond detection: it phenotypes cancers by their immune-modulation mechanism, with potential applications in (a) treatment selection (immune-restoration vs response-modulation approaches), (b) immunotherapy response prediction (G-2026-P011 trajectory test), (c) decade-pre-diagnostic mechanistic time-lapse for cancer biology research. This is a future-paper-scale finding, not a card-update-scale finding.

**Validation status.** Hypothesis open. Two validated anchoring data points (CRC negative, AD positive). One literature-supported additional point (glioma cell-fraction shift consistent with negative direction, Bracci 2022). Magnitudes and direction-stability across temporal phases require pre-diagnostic prospective cohort validation.

**Action items recorded.**
- [ ] UCSF AGS data application for direct β-value testing of Bracci 2022 cohort through GAPE pipeline (multi-month timeline)
- [ ] Cross-cancer NLR-vs-A-score comparison study using cohorts where both metrics are computable
- [ ] Temporal-stage trajectory study for any cancer with serial pre-diagnostic blood available
- [ ] Future paper scoping: "Peripheral immune-class methylation signature as decade-pre-diagnostic phenotyping of cancer immunomodulation"

---

### CCL-024 — Glioma-EPIC pathway design notes (consolidated from session 2026-04-24)

These detection pathways for brain pathology should be preserved as glioma-epic card design begins. They are anchored in published literature, not in our own validation runs.

**Pathway 1 — cfMeDIP-seq enrichment overcomes the 4% plasma cfDNA detection floor under active disease.** Healthy-baseline plasma is 0.5% terminal-class cfDNA, below the Moss 4% floor for solid-organ deconvolution. Under active aggressive brain disease (glioma, GBM, advanced neurodegeneration), the cfDNA dynamics shift — cfMeDIP-seq enrichment recovers AUC 0.99 [0.96-1.00] in published cohorts (Nassiri 2020). Lubotzky et al. detected brain-cell-type-specific cfDNA in plasma of 27/29, 25/29, 29/29 patients (neuron, oligodendrocyte, astrocyte) with brain metastases. **Plasma is exploratory for healthy-baseline screening but recoverable for active-disease detection with the right enrichment chemistry.**

**Pathway 2 — Lymphatic concentration via deep cervical lymph nodes.** The glymphatic system (Iliff/Nedergaard 2012) and meningeal lymphatic vessels (Louveau 2015 / Aspelund 2015) drain CSF and brain interstitial fluid to deep cervical lymph nodes, then to systemic circulation. Deep cervical lymph node aspirate or fluid sampling could in principle deliver brain-derived cfDNA at concentrations significantly higher than peripheral plasma. This is real anatomy. No published large-cohort methylation study exists for this approach. Candidate "tier 2" specimen above blood, below LP. Open framework prediction.

**Pathway 3 — Multi-specimen tier system for terminal-class detection.** LP-CSF (gold standard, invasive) → ventricular shunt sampling (when shunt exists) → Ommaya reservoir (chemotherapy-delivery devices in CNS lymphoma and pediatric brain tumors) → cisterna magna sampling → cervical lymphatic drainage → focused-ultrasound-disrupted-BBB plasma + cfMeDIP-seq → standard plasma + cfMeDIP-seq. Each tier carries its own validation requirement and cfDNA recovery profile.

**Pathway 4 — Microglial and brain-immune-trafficking signature in peripheral blood.** Microglia (resident brain macrophages, embryonic yolk-sac origin, decades-long turnover) have distinct methylation signatures from peripheral immune cells. Brain pathology drives monocyte trafficking from blood into CNS, monocyte-to-TAM conversion in glioma, and microglial activation states (DAM in AD, distinct activation in glioma). **A peripheral blood directional panel built from microglial markers (TMEM119, P2RY12, TREM2) + brain-immune-trafficking markers (CCL2, CCR2, monocyte-to-TAM signature) may discriminate brain pathology from solid-organ cancer at peripheral blood level.**

**Pathway 5 — Direction-as-discriminator (CCL-023 application to brain pathology).** AD shows positive peripheral immune-class A-score direction. Glioma's published cell-fraction signature (Bracci 2022) is consistent with negative direction. **If the direction principle holds, AD vs glioma discrimination starts at the sign of Stage 1 itself**, before any tissue-specific testing. This combined with Pathway 4 (microglial signature) gives glioma-epic two independent peripheral-blood discriminators, even before cfMeDIP-seq or CSF.

**Validation candidates for glioma-epic build (when ready).**
- Bracci/Wiencke 2022 cohort (UCSF AGS) — primary validation target via dbGaP-equivalent application. n=139 pre-surgery glioma + 454 controls, EPIC array.
- Nassiri 2020 cfMeDIP-seq glioma cohort — primary cfMeDIP-seq validation if methodology is portable to our framework.
- GSE180683 (76 glioma patients EPIC blood, mixed treatment stages) — supplementary, requires careful stratification by treatment status.
- TCGA-GBM and TCGA-LGG tissue methylation — already characterized in framework (LGG ΔA = +0.239, GBM ΔA = +0.217, "physics is loud").

**Build readiness:** glioma-epic should NOT be built until at least one of Bracci 2022 or Nassiri 2020 data is accessible. Building a card without per-patient data on the right specimen at the right timepoint is premature.

---

## How this document evolves

Every new validation run that produces a lesson adds an entry. Every new card adds its per-card lessons catalog. When a lesson spans cards, it becomes a cross-card entry (CCL-###). When it's process, it becomes a process entry (PL-###). The catalog grows monotonically; prior entries are never removed, only superseded (with a pointer to the superseding entry).

This file is Cookbook-vault: not pushed to GitHub, distributed under NDA to partners.
