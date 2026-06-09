# CPG Doctor-Report Capability List v0.2

**Date:** 2026-06-07
**Purpose:** What the IAMAtlas chain CAN output for a single patient that could be included in the doctor-report builder. Each item: what comes out, which chain stage produces it, and the confidence the IAMAtlas posterior backbone provides.
**Use:** Heath review → format/delivery decision → feeds the "report builder" section of BUILD_SPEC v1.2.1 → v1.2.2.

---

# THE PRIMARY POSITIONING — CPG is the cellular health and fitness tracker of the future

**CPG is not fundamentally a diagnostic test for a specific disease.** Single-sample disease detection is real and validated — the chain will detect any architectural departure magnitude large enough to cause concern, including the strong field-effect signatures of active cancers (VAL-001 6/6 cancers; VAL-003 28-cancer field effect at p = 1.32e-15; VAL-008 19/19 FLOOR BREACH; VAL-047 breast 10yr+ pre-dx d = +1.78). But by the time a single-sample readout shows a concerning magnitude, the architectural drift has already accumulated over years. **Cross-sectional detection is the downstream consequence, not the primary value proposition.**

**Where CPG actually shines is in serial monitoring for patients who care about tracking their entire body as a whole.** These are the wellness-engaged patients — the ones who see a naturopath, the ones who care about long-term cellular health, the ones who want a real molecular-level signal on how they're aging, how lifestyle changes are landing, whether inflammaging is accumulating, whether their immune system is drifting toward neurodegen patterns a decade before clinical presentation. **CPG is the cellular health and fitness tracker of the future** — a Fitbit-class commitment to ongoing measurement, but at the level of how every cell type in the IAMAtlas is maintaining its identity over time.

**For the wellness patient (the naturopath's patient), the model is:**
- **First sample = baseline** (everything in this document under "single-sample" is what they get day one)
- **Every subsequent sample compounds the value** (Section K — trajectory monitoring; each new sample transforms the analytical frame from population-baseline to patient-as-own-baseline)
- **Over years, the chain becomes a cellular-fitness chronograph** — forecasting trajectory, detecting molecular-level intervention response, surfacing drift toward neurodegen / cancer / aging / inflammation signatures more than a decade out, running personalized active surveillance protocols built on this patient's own molecular signature
- **Every patient who joins the CPG family contributes back** — anonymized aggregate data refines the CMB itself, sharpens the disease pattern catalog, narrows the MCMC posteriors on every cell type, teaches the chain what each disease looks like more precisely. The system gets better the more patients commit to it. The cellular fitness tracker improves through collective learning the same way the Apple Watch improved through aggregate user data — but at the molecular level, on the only base layer that matters for cellular identity: the methylome.

**Two patient archetypes, two primary value propositions:**

| Patient archetype | Primary value | Sampling cadence |
|---|---|---|
| **Wellness patient (naturopath's primary patient)** — health-engaged, whole-body monitoring focus, no specific acute concern | Serial cellular-fitness tracking; pre-symptomatic detection >10 years out; intervention-response monitoring; aging trajectory awareness; collective learning contributor | Annual minimum; quarterly for high-engagement |
| **Diagnostic patient** — specific concern, family history, active symptoms, BRCA/Lynch carrier, post-treatment survivor | Single-sample diagnostic readout (full Section A–O capabilities); then serial follow-up if abnormal findings or high-risk profile | Single sample → trajectory if findings or risk factors warrant |

The same chain produces both readouts. The doctor report builder needs to know which archetype the patient sits in so the presentation emphasizes the right framing — wellness tracking for the naturopath patient, diagnostic interpretation for the high-concern patient. Both modes leverage the full chain capability; the framing differs.

---

**One foundation point on confidence:** the IAMAtlas posterior backbone (483,093 CpGs × 8 classes × {mean, sd, ci_lo, ci_hi} = 15.4M posterior values) is universal — it does not depend on which disease card the patient's pattern matches against. Every output below carries a 95% CI by default, computed by propagating patient β-measurement uncertainty through the atlas posterior + Mahalanobis hull covariance. This means **posterior probability + CI on disease detection is computable for every disease entry in matrix v1.8** — all 81 phase rows / 52 unique diseases. Not deferred. Built in.

---

## A. Sample integrity and intake context

| Item | What the chain outputs |
|---|---|
| A.1 | Patient identifier (de-identified) + collection date + sample tier (1 / 2 / 3 / R) |
| A.2 | Chronological age + sex + 24-covariate intake context (ancestry, BMI, smoking, alcohol, sleep, stress, exercise, nutrition, hormonal state, HPV, prior cancer, prior chemo/radiation, recent infection/vaccination, chronic infections, current meds, family cancer history, family neurodegenerative history) |
| A.3 | Lab provenance (collection timestamp, bisulfite conversion rate, detection-p failure count, predicted-sex check) |
| A.4 | Overall data quality verdict: PASS / PASS_WITH_FLAGS / FAIL_HARD with explanation |
| A.5 | Sample suitability per disease card the chain ran (marker coverage ≥80% gate, platform tag, cross-method deconvolution gate L1 ≤ 0.15 / p95 ≤ 0.20) |

## B. Cellular composition (the methylation-resolution blood / tissue cell map)

The IAMAtlas has **115 cell types across 8 architecture classes**. The chain outputs fraction + 95% CI for every one of them; downstream sections then compute A-score, cellular age, 6-tier verdict, and posterior + CI per cell. The customer-facing report can drill from top-level lineage pages down to the underlying atlas cells.

| Item | What the chain outputs | Confidence |
|---|---|---|
| B.1 | **Cell-type fraction for all 115 cells in the IAMAtlas** with 95% CI per cell | Per-cell fraction CI from MCMC backbone |
| B.2 | **51 immune cell subdivisions** — the primary blood-traceable readout for every patient. Customer-facing presentation organizes these into **19 lineage pages** (B-cell lineage / T-cell lineage including naive/memory/regulatory subdivisions / NK + granulocyte lineage / monocyte + macrophage lineage / dendritic + plasma + nRBC) with click-to-expand drill-down into the underlying atlas cells per page | Per-cell CI; per-page roll-up CI |
| B.3 | **64 non-immune cell types** across the other 7 architecture classes (terminal / secretory / progenitor / cycling / stromal / stem_adult / stem_pluri) — the Walther IAM Deconvolver identifies cell-of-origin for these from ANY methylation input (not cfDNA-dependent); fractions returned for every cell with 95% CI regardless of substrate. In whole-blood input, non-immune fractions are typically small but quantified; substrate (whole blood, cfDNA, tissue biopsy, etc.) affects relative prominence of the signal, not the deconvolver's ability to identify the source cell | Per-cell CI |
| B.4 | **Age-stratified reference range overlay** per cell — patient's value vs age-matched HC distribution from n=2,481 healthy hull (sex-stratified where relevant). Applies to every cell with HC reference coverage | Hull v0_5 percentile CI per cell |
| B.5 | **Tissue-resolved cell-of-origin extensions** (when separate tissue atlases are integrated alongside the IAMAtlas core 115): 25-tissue base atlas extension (hepatocyte, cardiomyocyte, etc.), heart-tissue extension (right atrium / LV / coronary), neural-tile extension (cortical / dopaminergic / spinal motor / astrocytes / Schwann / microglia — atlas integration v0.3 queue). These extensions return fractions from any methylation input via the deconvolver; cfDNA substrate AMPLIFIES tissue signals because tissues shed DNA into plasma, but tissue cell-of-origin is identifiable from whole-blood input as well, just at lower magnitude | Per-tissue CI |
| B.6 | **Per-cell drill-down table** — the doctor can navigate from the 19 customer-facing immune lineage pages down to the underlying 51 immune subdivisions, and from the 8 architecture-class summaries down to all 115 cells | Per-cell CI surfaced at any drill level |

## C. Architectural state (the IAM physics readout)

| Item | What the chain outputs | Confidence |
|---|---|---|
| C.1 | **A-score per architecture class** (8 classes: terminal, immune, secretory, progenitor, cycling, stromal, stem_adult, stem_pluri) with 95% CI | A-score CI from H_min posterior + patient β noise |
| C.2 | **A-score per cell type** (115 cells; 51 immune subdivisions) with 95% CI | Per-cell A-score CI |
| C.3 | **6-tier verdict per class** (SUPPRESSED / NORMAL / ELEVATED / WARBURG_TRANSITION / SIGNIFICANTLY_ELEVATED / BREACH) — physics-derived breakpoints (1.07 Warburg line, 1.10 architectural-fidelity breach line) | CI-bracketed DEFINITE or CI-spanning BORDERLINE |
| C.4 | **6-tier verdict per cell type** (drill-down table for all 115 cells) | CI-bracketed |
| C.5 | **Top-10 most-architecturally-departed cells** with effect size against HC reference | Per-cell CI on departure |
| C.6 | **Cells in BREACH** flagged with red severity bar | CI-bracketed |
| C.7 | **Tier covariate overrides applied** (when smoking-bin / sex / age / chemo-history etc. shift the tier interpretation per `tier_breakpoints.json v1.2`) | Documented in output |
| C.8 | **Smoking-bin stratification** for terminal / secretory / immune classes — current smokers have shifted floor expectation; former-smoker reversibility signature read | CI per smoking-bin |

## D. Cellular aging

| Item | What the chain outputs | Confidence |
|---|---|---|
| D.1 | **Immune cellular age** + **immune age delta** = cellular age − chronological (the inflammaging quantum) | 95% CI on age + delta |
| D.2 | **Per-class cellular age** (8 classes) + **per-cell-type cellular age** (51 immune cells) | Per-class / per-cell CI |
| D.3 | **Age percentile within HC cohort** — "patient sits at the Xth percentile for chronological age band" | Percentile CI |
| D.4 | **Cross-substrate concordance** on cellular age (methylation vs fragmentomics vs WPS) — discordance is itself diagnostic | Cross-substrate CI agreement gate |
| D.5 | **OSK-direction reference** — context label showing whether patient's age-delta direction maps to youthful or aged architecture (Yamanaka-OSK reference: lower A = youthful) | Reference context |

## E. Universal architectural departure (Mahalanobis)

| Item | What the chain outputs | Confidence |
|---|---|---|
| E.1 | **Mahalanobis distance** in the 112-feature space against n=2,481 HC hull v0_5 (Ledoit-Wolf shrinkage 0.00875); for reference, multivariate-normal expected median is √112 ≈ 10.58 | CI from hull covariance posterior |
| E.2 | **Hull percentile** — patient's d as percentile against n=2,481 HC distribution | Percentile CI |
| E.3 | **Trigger status** — Route A activated (yes/no) at p95=13.62 default or p99=18.59 strict per orchestrator config | CI-bracketed trigger |
| E.4 | **Top-10 contributing cell-types** to the Mahalanobis distance — which cells are pulling the patient away from healthy | Per-cell contribution CI |
| E.5 | **Sign of contribution** per top-cell — disease-typical direction vs opposite (suppression vs elevation) | CI on sign |
| E.6 | **Platform confidence qualifier** — 450K vs EPIC tag with attenuation note when applicable | Documented in output |

## F. Personal Brilliance Map — Scored against the Cosmic Methylome Background: the genetic base layer mirroring the Cosmic Microwave Background

| Item | What the chain outputs |
|---|---|
| F.1 | **8-panel Mollweide projection** — one CMB panel per architecture class, HEALPix NSIDE=128 (196,608 pixels, 483,093 CpGs mapped at 100% atlas coverage) |
| F.2 | **Personal Brilliance Map** — per-pixel z-score of patient β against the frozen CMB; bright spots = local departures from the genetic base layer |
| F.3 | **Color-coded by departure direction** — red elevated, blue suppressed |
| F.4 | **Chromosomal genomic-order panel** — linear-genomic alternative to Mollweide for doctors who prefer linear context |
| F.5 | **Class-specific anisotropy zoom** — when a class shows BREACH tier, drill into that class's CMB panel |
| F.6 | **Patient-vs-self overlay** (serial samples) — same Mollweide rendered for two timepoints with delta panel |
| F.7 | **Whole-genome Personal Brilliance Map vs whole CMB** — single Mollweide rendering the patient's full-atlas departure against the complete frozen Cosmic Methylome Background (all 483,093 CpGs, all 8 classes combined into one base layer) — gives the single-image "everything at once" view alongside the 8 per-class panels |

## G. Bidirectional decomposition (the pooled-mute / directional-loud signal)

| Item | What the chain outputs |
|---|---|
| G.1 | **FLAG_BIDIRECTIONAL per class** — fires when pooled A is mute (\|A−1\| < 0.05) AND directional composite is loud (\|composite\| > 0.40) |
| G.2 | **Directional composite for immune class** (v1.0 Rule A 7-CpG panel: 2 positive + 5 negative CpGs) with 95% CI |
| G.3 | **Other 7 classes** declared NO_PANEL when no v1.0 directional panel is sealed for that class (honest declaration; pooled A is the only output) |
| G.4 | **Tier handoff** — when FLAG_BIDIRECTIONAL fires, the customer-facing tier is driven by directional magnitude not pooled A |

## H. Disease pattern matching — every disease in the catalog scored against THIS patient

**What this section actually does:** the chain compares the patient's 115-cell architectural pattern against every disease entry in the catalog (matrix v1.8 — 52 unique diseases, 81 disease/phase rows). For each disease, it returns: how strongly the patient's pattern matches the disease's documented signature, how confident the chain is in that match, which cells contributed most, and the probability the patient carries that disease's architectural signature with a confidence interval.

**The framing is "per-disease detection probability with confidence," NOT "we always pick 3 diseases."** For a typical healthy wellness patient, the expected output is "no disease patterns detected above clinical threshold" — and that absence is itself a clinically meaningful result the doctor reports. For a patient with architectural drift, the diseases whose signatures the patient actually matches get surfaced for review.

| Item | What the chain outputs | Confidence |
|---|---|---|
| H.1 | **Per-disease match magnitude** — for every one of the 81 disease/phase rows in the catalog, how strongly the patient's 115-cell pattern matches the documented signature (Pearson ρ between patient departure and disease signature pattern) | 95% CI on match (Fisher z-transform) |
| H.2 | **Per-disease detection probability** with CI — the posterior probability the patient carries the disease's architectural signature, computed from IAMAtlas posterior + patient β noise + healthy reference distribution + the disease's matrix entry — all uncertainty propagated forward | 95% CI on posterior |
| H.3 | **Flagged-disease surfacing** — diseases whose detection probability exceeds a clinical threshold (configurable per disease and per phase; default = lower bound of 95% CI > 0 on match) get surfaced in the report. **For a healthy patient with no architectural drift, the expected and most common output is "no disease patterns detected above clinical threshold" — and that explicit no-detection result is itself the clinical readout.** When diseases ARE flagged, they appear ordered by clinical priority (detection probability × confidence × disease severity), not arbitrarily capped at any fixed number | Per-disease CI surfaced |
| H.4 | **Per-disease cell-pattern explainer** — for each flagged disease, which cells in this patient contributed most to the match, with the immune-perspective explainer from the immune lens (1-2 sentences per disease) | Per-cell contribution CI |
| H.5 | **Disease phase estimate** — for diseases with phase-stratified entries (breast 7 phases, colorectal 7, HCC 4, pancreatic 3, prostate 2, lung 2, AD/FTD/PSP-CBD 3 each, etc.), the chain returns which phase the patient's pattern most resembles, with CI | Per-phase CI |
| H.6 | **Disease severity class** per flagged disease (from matrix v1.8 `disease_severity_class` column) | CI-bracketed |
| H.7 | **Bidirectional grouping** — diseases that move in opposite directions across cells are SEPARATED in the output (not averaged), so the doctor sees both directional patterns when both are present | Per-direction CI |
| H.8 | **Full per-disease table available on request** — the doctor can drill down to see this patient's match magnitude + CI + detection probability + CI for EVERY one of the 81 disease/phase rows the chain scored, not just the flagged ones. Useful for differential diagnosis, second-opinion review, and active surveillance over time | Per-disease CI shown for non-flagged entries too |
| H.9 | **Empty-matrix-cell honesty** — disease/cell pairs without documented signature shown as "—" with "research-in-progress" footnote (not zero) | Documented in output |

### Disease coverage from matrix v1.8 (the chain computes H.1–H.9 for each — 81 phase rows / 52 unique diseases verified)

**Solid cancers (14 diseases / 36 phase rows):** breast (7 phases), colorectal (7), HCC (4), pancreatic (3), lung (2), cervical CIN-stratified (2), esophageal EAC (2), prostate (2), glioma GBM (2), glioma LGG (1), gastric (1), esophageal ESCC (1), bladder (1), kidney (1)

**Hematologic malignancies (10 diseases / 13 phase rows):** AML (2), CLL (2), multiple myeloma (2), B-ALL (1), CML (1), T-ALL (1), DLBCL (1), MDS (1), MPN (1), thymoma (1)

**Neurodegeneration (6 diseases / 10 phase rows):** Alzheimer's (3 phases), frontotemporal dementia (2), PSP/CBD tauopathies (2), Parkinson's (1), ALS (1), multiple sclerosis (1)

**Cardiovascular (3 / 3):** PAH, aortic dissection BAV, ischemic stroke

**Autoimmune / inflammatory (5 / 5):** Crohn's, ulcerative colitis, rheumatoid arthritis, lupus SLE, psoriasis — note IBD shows informative null at Stage 1 (the null itself is the readout per VAL-128)

**Acute response context (5 / 5):** recent vaccination, recent viral infection, recent bacterial infection, active allergies, pregnancy

**Chronic infection context (4 / 4):** chronic HIV, chronic hepatitis B/C, chronic CMV, chronic EBV

**Mental health (2 / 2):** schizophrenia, major depression

**Baseline state context (2 / 2):** normal aging, inflammaging

**Treatment-state context (1 / 1):** active chemotherapy

**Total verified: 52 unique diseases / 81 phase rows. The chain computes H.1–H.9 — including posterior + CI on disease detection — for every one.**

**Trajectory-essential vs trajectory-additive disease coverage** (the doctor must understand this distinction):

- **Trajectory-essential diseases** (single-sample detection moderate-to-weak; clinical-grade detection requires serial samples and patient-as-own-baseline per Section K): All 10 phase rows of neurodegeneration (AD 3, FTD 2, PSP/CBD 2, Parkinson's, ALS, MS) — AD single-sample AUC = 0.68 / 0.60 cross-platform; pooled NULL; bidirectional drift cancels out at single sample. Pre-cancer windows at longest lead time (10yr+ pre-dx; the signal is weakest farthest from clinical presentation, exactly when intervention has most value). Chronic inflammatory conditions without acute presentation. Schizophrenia, major depression, ALS (slow progression, subtle architectural drift). Treatment response monitoring and recurrence surveillance (require pre/post or baseline-by-design).
- **Trajectory-additive diseases** (single-sample detection strong; serial samples extend and refine): Active solid cancers with field-effect signatures (breast pre-dx 10yr+ d > 1.3, HCC vs cirrhosis 8.03× separation, gastric d = +3.34, esophageal EAC d = +3.70, pan-cancer 19-cancer specimen matrix all 19/19 FLOOR BREACH). Hematologic malignancies (lymphoid-vs-myeloid panel split is decisive at first sample). Acute response context (vaccination, infection, allergies — these are explicitly acute; baseline shift detected single-sample).
- **Sampling-cadence recommendation per category** is in the report (K.4.5 active surveillance mode): trajectory-essential disease risk = annual minimum for general neurodegen risk; quarterly for active surveillance of BRCA carriers, Lynch syndrome carriers, post-treatment cancer survivors, family-history-high-risk patients. Trajectory-additive disease risk = single-sample sufficient for first read; serial samples extend value.

## I. Cross-disease universal alarm (the v0_1 channel)

| Item | What the chain outputs | Confidence |
|---|---|---|
| I.1 | **Universal alarm channel** firing status — Pearson ρ of patient per-CpG departure against the cross-disease universal alarm v0_1 residual map | Fisher z 95% CI on ρ |
| I.2 | **Cross-disease concordance sub-channel** ρ on the 17 same-direction CpGs (shared aging/inflammation drift across diseases) | CI on ρ |
| I.3 | **Bidirectional universal alarm sub-channel** ρ on the 12 opposing-direction CpGs (the universal alarm signature at per-CpG resolution) | CI on ρ |
| I.4 | **Bimodality cross-fire** — patient's bimodality at the cross-disease bimodality channels (1,592 double-disease gain + 26 double-disease loss + 850 opposing-bimodality) | Per-channel CI |

## J. Wellness / lifestyle / inflammaging context

When intake covariates are supplied, the chain contextualizes the architectural readouts with:

| Item | What the chain outputs |
|---|---|
| J.1 | **Aging & inflammaging** — age delta (D.1) + inflammaging burden score + deviation-from-trajectory CI + aging-direction-vs-OSK reference |
| J.2 | **Smoking signature read** — active current vs former vs never; post-cessation residual signature flag for former smokers |
| J.3 | **Other lifestyle covariate contributions** — alcohol, sleep, stress, exercise, nutrition (when supplied) |
| J.4 | **Acute response context** — recent vaccination flag (≤30d) with confounder note; recent infection / allergies / surgery |
| J.5 | **Life stages and hormonal** — menarche signature reference, menopausal status overlay, pregnancy flag, puberty flag, hormone therapy context |
| J.6 | **Chronic conditions affecting baseline** — chronic HIV / HBV / HCV / CMV / EBV residual signatures |
| J.7 | **Treatment context** — active chemotherapy footprint; prior chemotherapy / radiation persistent footprint |
| J.8 | **Environmental exposures** — when supplied (occupational, residential) cross-referenced against documented signatures |
| J.9 | **Homeostasis quality indicators** — per-class CI tightness + cell-fraction coherence + overall homeostatic stability composite |

## K. Trajectory monitoring (serial sampling) — what unlocks the more tests the patient completes

**Foundation:** each new sample doesn't just add a data point — it transforms the analytical frame. The chain progressively learns the patient's own stable architecture, narrows every CI, detects trends and accelerations, forecasts trajectories, and (for high-risk patients) enables active surveillance protocols. The compound value: after enough samples, **the patient becomes their own reference** — future readings get measured against THIS patient's stable baseline, not the population HC hull. Detection sensitivity dramatically improves; what was within population normal may be outside the patient's personal normal.

**Critical clinical framing — some diseases are trajectory-essential, not optional:** the chain's single-sample sensitivity varies dramatically across diseases. Active solid cancers with field-effect signatures often produce strong single-sample readouts (breast pre-dx 10yr+ d = +1.78 / +1.36 per VAL-047; HCC vs cirrhosis 8.03× separation per VAL-010; gastric d = +3.34 per VAL-126). **Neurodegenerative diseases do not.** AD single-sample detection from the VAL-051 7-CpG Rule A panel is AUC = 0.68 on AIBL holdout (d = +0.62); cross-platform attenuates to AUC = 0.60 (VAL-052 AddNeuroMed, d = +0.33); pooled A-score is NULL (VAL-050, d = +0.08, because AD drift is bidirectional and pooled metrics cancel out). **For AD specifically, and for the other neurodegenerative cards (FTD, PSP/CBD, Parkinson's, ALS, MS), trajectory analysis is the diagnostic modality, not optional supplementary data.** Single samples flag possibilities; serial samples with patient-as-own-baseline (K.3.1) plus drift-cascade detection (K.3.3) plus bidirectional-pattern emergence over time are what produce clinical-grade detection. Pre-cancer windows (the 5–10+ year pre-diagnostic phase before clinical cancer) similarly benefit dramatically from trajectory because the architectural signal is weakest farthest from clinical presentation — and that is exactly the window where the earliest intervention has the most value. The report explicitly recommends sampling cadence based on which trajectory category a patient sits in: trajectory-essential disease risk (annual minimum for neurodegen risk; quarterly for active surveillance of BRCA / Lynch / post-treatment survivors); trajectory-additive disease risk (single-sample sufficient for first read; serial samples extend value).

### K.1 — What unlocks at sample 2 (first follow-up)

| Item | What the chain outputs | Confidence |
|---|---|---|
| K.1.1 | **Per-class slope** (ΔA per unit time) for all 8 architecture classes | Slope CI from 2-point posterior |
| K.1.2 | **Per-cell slope** for all 115 atlas cells (not just 51 immune — the full atlas trajectory) | Per-cell slope CI |
| K.1.3 | **Cellular age trajectory** — per-class (8) + per-cell (51 immune cells minimum, all 115 if relevant) cellular age slope; the "are you aging faster or slower than the clock" signal | Slope CI per class/cell |
| K.1.4 | **Cellular age delta change** — Δ(cellular age delta) between samples; if chronological clock advances by Δt and cellular age advances by 0.5Δt, "aging slower"; by 2Δt, "faster"; if Δ(cellular age delta) is negative, the patient is getting BIOLOGICALLY YOUNGER between samples | Delta CI |
| K.1.5 | **6-tier verdict transitions** — which cells / which classes crossed tier boundaries between samples (NORMAL→ELEVATED, ELEVATED→WARBURG_TRANSITION, etc.); per-cell and per-class transition table | CI-bracketed transition status |
| K.1.6 | **Mahalanobis trajectory** — is the patient moving toward HC hull center or away from it? At what rate? Direction + magnitude | Trajectory CI |
| K.1.7 | **Personal Brilliance Map delta** — patient-vs-self Mollweide overlay; same Brilliance Map at two timepoints scored against the unchanging CMB; bright spots = pixels that changed brightness over time | Per-pixel delta CI |
| K.1.8 | **Whole-atlas Brilliance Map delta** — F.7 view at two timepoints with the difference panel | Per-pixel delta CI |
| K.1.9 | **Bidirectional flag stability** — is FLAG_BIDIRECTIONAL firing consistently across samples? Are directional composites drifting? | Per-class CI on composite drift |
| K.1.10 | **Bayesian posterior update** — sample 2 narrows the posterior on sample 1; the narrowing CI is itself a clinical metric showing the chain "learning" the patient | Posterior-tightening fraction per metric |

### K.2 — What unlocks at samples 3–5 (early longitudinal series)

| Item | What the chain outputs | Confidence |
|---|---|---|
| K.2.1 | **Trajectory significance** — statistical significance of trends, not just point estimates ("your immune class A-score is increasing at 0.012/year, p < 0.01"; per-metric trend p-value) | Per-trend p-value + slope CI |
| K.2.2 | **Trajectory flags per class** — UP_SIGNIFICANT / DOWN_SIGNIFICANT / STABLE / VOLATILE / CONVERGING_TO_BREACH | CI-bracketed flag |
| K.2.3 | **Disease-card serial readout** — for diseases flagged at any prior sample, current match magnitude vs prior; regression vs progression magnitude with CI per disease | Per-disease serial CI |
| K.2.4 | **Disease pattern matching evolution** — flagged diseases at sample 1 tracked across the series; new diseases crossing threshold to flagged, previously-flagged diseases falling back below threshold; the per-disease trajectory itself becomes a clinical signal | Per-disease trajectory CI |
| K.2.5 | **Cross-disease universal alarm stability** — is the alarm channel firing consistently across samples? Strengthening? Weakening? Both sub-channels (cross-disease concordance + bidirectional universal alarm) tracked | Per-channel stability CI |
| K.2.6 | **Inflammaging burden trajectory** — over the sample series, is the inflammaging burden score increasing, stable, or decreasing? | Slope CI |
| K.2.7 | **Within-patient variance estimation** — how variable is THIS patient day-to-day, week-to-week, month-to-month? Calibrates future single-sample readouts against this patient's own measurement scatter | Per-metric variance estimate |
| K.2.8 | **Per-cell stability profile** — chain learns which cells are stable for THIS patient and which are volatile; volatility map is itself a clinical readout | Per-cell stability index with CI |

### K.3 — What unlocks at samples 6+ (stable patient baseline established)

| Item | What the chain outputs | Confidence |
|---|---|---|
| K.3.1 | **Patient-as-own-baseline (the major unlock)** — reference range shifts from population HC hull to THIS patient's own stable architecture; future deviations measured against the patient's personal normal, not the population's normal. Detection sensitivity dramatically improves | Patient-specific reference CI |
| K.3.2 | **Personal reference range overlay** added to every C, D, E, F metric — the doctor sees both the population HC range and the patient's own historic range; deviations from personal baseline often surface before population-baseline detection would | Overlay CI from N-sample posterior |
| K.3.3 | **Drift cascade detection** — VAL-037 → VAL-046 demonstrated multi-class drift cascade signatures (35/39 pre-specified predictions confirmed). With serial samples on a single patient, the chain detects whether the patient is FOLLOWING one of these documented cascade patterns (e.g., immune drift → secretory drift → terminal drift → BREACH) | Per-cascade match CI |
| K.3.4 | **Acceleration / deceleration** (second derivative) — is the rate of change itself changing? A class that was stable then began drifting; a metric that was drifting but is now decelerating | Per-metric acceleration CI |
| K.3.5 | **Smoking cessation reversibility tracking** (for former smokers) — over serial samples, is the patient's immune architecture actually reverting toward never-smoker baseline? At what rate? When would it complete? | Reversion trajectory CI |
| K.3.6 | **OSK-direction tracking** — lifestyle changes (exercise, nutrition, sleep, stress reduction) — over serial samples, is the patient moving toward youthful architecture (Yamanaka-OSK direction = lower A-scores) or aged direction? | Direction trajectory CI |
| K.3.7 | **Aging trajectory vs peer cohort** — is the patient aging faster than peers (same age band)? Slower? At what cellular level specifically? | Per-class peer-relative slope CI |

### K.4 — What unlocks at long-term surveillance (years of data; high-risk patients)

| Item | What the chain outputs | Confidence |
|---|---|---|
| K.4.1 | **Forecasting** — given current trajectory, when would the patient cross the BREACH tier on this class? When would a disease concordance pattern be expected to emerge? Per-metric time-to-event prediction with CI | Forecast CI |
| K.4.2 | **Treatment response monitoring** (for patients undergoing treatment) — pre-treatment baseline vs post-treatment serial samples; cellular response detected before clinical response; on-treatment vs off-treatment trajectory comparison | Pre/post CI per metric |
| K.4.3 | **Recurrence early warning** (for cancer survivors) — post-treatment baseline established; drift from that baseline = architectural recurrence signal, often detected before clinical recurrence presents | Drift-from-survivorship-baseline CI |
| K.4.4 | **Pre/post intervention tracking** — when a patient changes diet, starts a supplement, alters exercise pattern, etc., trajectory comparison shows whether the intervention is producing cellular response. The doctor and patient see molecular-level confirmation or absence | Intervention response CI |
| K.4.5 | **Active surveillance mode** — for BRCA carriers, Lynch syndrome carriers, post-treatment cancer survivors, family-history-high-risk patients, etc. — quarterly or biannual sampling becomes a true active surveillance protocol with progressively tighter detection; the chain knows the patient's own architecture so any departure triggers earlier alerts than population-baseline screening would | Per-finding active-surveillance CI |
| K.4.6 | **Risk modeling per disease** — trajectory-based posterior probability of disease emergence within N months / years per disease in matrix v1.8 (all 52 diseases / 81 phase rows, not just the 3 IAMAtlas chain-language cards) | Time-windowed risk CI |
| K.4.7 | **Per-CpG stability profile** for THIS patient — 483,093 CpGs each have a trajectory; the chain learns which CpGs are stable for this patient vs which are drifting; the volatility CpG set is itself a clinical signature | Per-CpG stability index |
| K.4.8 | **Patient-specific Brilliance Map evolution sequence** — animated / sequenced Mollweide rendering showing the patient's Brilliance Map evolving across all samples in the series; longitudinal CMB drift visualization | Per-frame CI |
| K.4.9 | **Posterior tightening evidence accumulation** — explicit visualization of how the 95% CI on every metric narrows with each additional sample; patient sees evidence accumulating and confidence growing | CI-width trajectory per metric |
| K.4.10 | **Trajectory-informed decision points** — when the chain has high-confidence trajectory data, it can surface decision-support framings: "intervention X applied between samples 4 and 5 produced a measurable cellular response in 6 of 8 classes"; "current trajectory predicts BREACH at class Y within 18 months — consider intervention" | Decision-support CI |

---

**The compound message to the patient:** the more tests you complete, the more the chain knows about YOU specifically. First test: you're compared to a population of 2,481 healthy people. By sample 6: you're compared to YOUR own stable baseline, which is far more sensitive. By long-term surveillance: the chain forecasts your trajectory, detects molecular intervention response, gives recurrence early warning, and runs active surveillance protocols built on YOUR personal molecular signature.

## K.5 — The collective learning network — what every patient contributes back to the chain

CPG is designed so each patient who commits to the cellular health and fitness tracking model contributes back to the chain itself. Anonymized aggregate data — with appropriate consent and privacy controls — refines the universal infrastructure that every other patient depends on. The system gets better the more patients join the CPG family.

| Item | What contributing back unlocks | Cadence |
|---|---|---|
| K.5.1 | **CMB refinement** — the Cosmic Methylome Background base layer itself is a frozen reference at any given moment; as more healthy patient samples accumulate, the CMB calibration tightens (the per-CpG posterior {mean, sd, ci_lo, ci_hi} narrows) and the Mahalanobis HC hull broadens beyond the current n = 2,481 baseline. Every patient who contributes a sample to the HC pool sharpens the reference everyone else is scored against | Quarterly atlas refresh |
| K.5.2 | **Disease pattern catalog refinement** — disease matrix v1.8 has 81 phase rows / 52 unique diseases; as more patients with known clinical outcomes contribute their architectural patterns, the matrix entries become more precise and new phase rows can be added (e.g., earlier pre-dx windows, more granular subtypes) | Quarterly matrix release |
| K.5.3 | **New disease pattern discovery** — patterns the chain hasn't catalogued yet emerge from large-N patient data; novel architectural signatures get identified and added to the disease matrix, expanding what the chain can detect for everyone | Per-discovery |
| K.5.4 | **MCMC posterior tightening** — H_min calibrations and per-CpG posteriors are recomputed periodically with larger atlas data; CI narrows for everyone | Annual atlas rebuild |
| K.5.5 | **Drift cascade catalog expansion** — VAL-037 → VAL-046 established 35/39 multi-class drift cascade signatures; as more longitudinal patient data accumulates, additional cascade patterns get characterized and added to the K.3.3 detection set | Per-cascade-validation |
| K.5.6 | **Trajectory-essential disease characterization improves** — the AD trajectory pattern, the FTD trajectory pattern, the PSP/CBD compaction trajectory, etc. — each becomes more precisely characterized as more confirmed-diagnosis patients contribute serial samples back. AUC improves over time for trajectory-essential disease detection | Per-cohort enrollment |
| K.5.7 | **Demographic stratification refinement** — sex, age, ancestry, smoking, hormonal context stratifiers in the Mahalanobis hull and per-class reference distributions become more granular as the contributing patient population diversifies | Quarterly |
| K.5.8 | **Lifestyle-intervention response catalog** — the smoking-cessation reversibility signature (CPG-VAL-022), the OSK-direction lifestyle tracking, the weight-loss signature (CPG-VAL-021 deferred) — each grows a real-world response distribution as more patients contribute pre/post-intervention serial samples. The chain learns what a specific intervention typically produces at the cellular level, providing increasingly informative pre-intervention forecasts | Per-intervention-validation |
| K.5.9 | **Personalized active surveillance protocol refinement** — for BRCA carriers, Lynch carriers, family-history-high-risk patients, etc., the active surveillance protocols themselves improve as enrolled cohorts contribute long-term trajectory data; cadence recommendations, alert thresholds, and forecasting horizons get tuned to real outcomes | Annual protocol review |

**The compound model:** every patient who commits to ongoing CPG sampling is both BENEFICIARY (their own cellular fitness tracking improves with each sample) AND CONTRIBUTOR (their anonymized data improves the chain for every patient who comes after). This is the same network effect that made consumer fitness trackers improve over time — but at the molecular base layer, on the genetic methylome that determines cellular identity.

## L. Prior + family history integration

| Item | What the chain outputs |
|---|---|
| L.1 | **Per-disease prior probability** — population baseline incidence × intake covariates (age, sex, ancestry, BMI, smoking, prior cancers, family history) |
| L.2 | **Family history multiplier** — when 1st-degree relative had a disease at age X, multiplier applied to that disease's prior |
| L.3 | **Posterior** — Bayesian combination of prior × patient's chain-derived match magnitude likelihood → per-disease posterior with CI |
| L.4 | **Triage ordering** — disease posteriors ranked, with top-N surfaced to the doctor |
| L.5 | **No-priors mode** — when intake covariates absent, report falls back to match magnitude alone (labeled) |

## M. Literature anchors per finding

| Item | What the chain outputs |
|---|---|
| M.1 | **Per-finding literature citation** — each disease-card finding shows the literature anchor used in matrix v1.8 (`evidence_anchors` column) with PMID/DOI |
| M.2 | **Per-CpG annotation** — when a specific CpG is surfaced (e.g. cg01006587 in the 12-CpG opposing subset), chromosome + nearest-gene + GeneCards link |
| M.3 | **Mechanism statement per disease** — 1-2 sentence mechanism citation from matrix v1.8 `mechanism` column |
| M.4 | **Auditable anchor trail** — back-office Stage 10 audit trail with every anchor consulted (not exposed to doctor by default; available on request) |

## N. The confidence backbone — every number above has it

Every metric in sections A–M carries a 95% CI by default, computed by forward-propagating patient β-measurement uncertainty through the IAMAtlas posterior backbone (483,093 CpGs × 8 classes × {mean, sd, ci_lo, ci_hi} = 15.4M posterior values) + Mahalanobis hull v0_5 covariance posterior (Ledoit-Wolf shrinkage 0.00875 on n=2,481 HC).

| Item | What the chain outputs |
|---|---|
| N.1 | **95% CI default** on every number — never a point estimate alone |
| N.2 | **Multi-level CI on request** — 68% (screening), 99% (high-confidence triage), 99.9% (sentinel / outlier review) |
| N.3 | **CI-bracketed tier confidence** — when patient's A-score CI brackets one tier, tier is DEFINITE; when CI spans a breakpoint, tier is BORDERLINE with both tiers shown |
| N.4 | **Posterior + CI on every disease in the matrix** — not just the 3 cards in the IAMAtlas chain language; the universal posterior backbone enables disease-detection posterior + CI for all 81 phase rows |
| N.5 | **Bidirectional cancellation warning** — when pooled A is mute but directional is loud, both shown with annotation |
| N.6 | **Honest "we don't know"** — when patient β noise is high or marker coverage is incomplete, the CI widens visibly rather than the point estimate moving |
| N.7 | **Triage by confidence** — when multiple diseases are flagged, they are ordered by `detection probability × 1/CI_width × disease severity` so the most-confident-and-most-actionable finding surfaces first (not by raw match magnitude alone, which would prioritize loud-but-uncertain findings over clear-and-confident ones) |
| N.8 | **Discordance flagging** — when methylation cellular age CI and fragmentomics cellular age CI overlap, "concordant"; when not, "DISCORDANT — investigate" |

## O. Confidence + caveats (honesty propagation)

| Item | What the chain outputs |
|---|---|
| O.1 | **Platform tag** — 450K vs EPIC; attenuation note where documented |
| O.2 | **Single-cohort findings** explicitly labeled when underlying evidence is single-cohort |
| O.3 | **Marker-coverage flag** — when a card's required marker coverage drops below threshold for this sample, that card's verdict is suppressed: "INSUFFICIENT MARKER COVERAGE FOR [CARD]" rather than a degraded readout |
| O.4 | **Run-everything transparency** — every disease card the chain ran is enumerated (NOT_FIRED entries shown explicitly) |
| O.5 | **Empty-matrix-cell honesty** — disease/cell pairs without documented signature shown as "—" with footnote |
| O.6 | **Disease coverage statement** — explicit list of which matrix rows have full-card readout vs matrix-only Stage 8 Route B readout |

## Q. Educational definitions — every novel concept defined for the doctor and patient

**Foundation:** the doctor opening this report has never seen a CPG output before. The patient has never seen one in their life. The report educates as it informs — every novel concept gets a brief inline definition the first time it appears, with deeper explanation available in a glossary appendix. Visualizations (diagrams, example A-score curves, the cosmology→methylome CMB mirror, etc.) are generated programmatically by Python code embedded in the report-generation script (Stage 10), so the PDF the doctor receives carries the educational figures inline rather than referring out. This is the same generation pattern as the Personal Brilliance Map and the per-class panels — the report builder script renders every figure in the same run.

### Q.1 — Foundation (methylation basics)

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.1.1 | **Methylation** | A chemical tag (a methyl group) attached to DNA at specific places that affects how genes are read by the cell. Methylation patterns are how the same DNA sequence produces different cell types — a liver cell and a brain cell have identical DNA but different methylation patterns. | Schematic of a methyl group attached to DNA |
| Q.1.2 | **CpG (CpG site)** | A specific DNA location where methylation can happen — a Cytosine nucleotide followed immediately by a Guanine. The human genome has about 28 million CpG sites; the IAMAtlas measures 483,093 of them | Schematic of a CpG dinucleotide |
| Q.1.3 | **β value** | The measurement returned for each CpG — the fraction of the patient's cells (from 0 to 1) that carry methylation at that specific location. β = 0 means no cells methylated there; β = 1 means all cells methylated there; intermediate values reflect mixed populations | Example histogram of β values across atlas |
| Q.1.4 | **Methylation patterns and cellular identity** | The pattern of methylation across many CpGs together is what tells the chain which cell type a piece of DNA came from. The IAMAtlas is the reference map of these patterns. | Example heatmap snippet |

### Q.2 — Cellular concepts

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.2.1 | **IAMAtlas** | The reference map at the heart of CPG. Contains the methylation signatures of 115 different cell types across 8 architecture classes — the chain compares the patient's methylation against this reference to figure out (a) which cells are in the sample and (b) how well each cell type is maintaining its identity | Tree diagram showing 8 classes → 115 cells |
| Q.2.2 | **Architecture class** | One of 8 groups the chain organizes cells into based on their physical/informational role: terminal (mature differentiated cells), immune, secretory, progenitor, cycling, stromal, stem (adult), stem (pluripotent). Each class has its own architectural floor (H_min, see Q.3.2) | 8-class diagram with example cells per class |
| Q.2.3 | **Cell type** | A specific cell within a class (e.g., naive CD4 T cell, memory B cell, hepatocyte, cardiomyocyte). The IAMAtlas has 115 cell types total; 51 of them are immune cell subdivisions | Hierarchy: class → cell type |
| Q.2.4 | **"Cell maintaining its identity"** | The central CPG question. Every cell type has a minimum amount of information it needs to carry in its methylation pattern to actually BE that cell type and do its job. When a cell drifts away from that minimum (architectural floor), it's losing its identity — becoming more disordered, less specialized. This drift is what CPG measures and is the early signal of cellular health decline | Schematic of cells on/above/at the architectural floor |
| Q.2.5 | **Cell-of-origin** | For any methylation signal the chain detects in the sample, which cell type it came from. The Walther IAM Deconvolver identifies this for all 115 atlas cells from any methylation input (whole blood, plasma cfDNA, or tissue biopsy) | Schematic of deconvolution: mixed sample → per-cell fractions |
| Q.2.6 | **Cell fraction** | The estimated percentage of the patient's sample that came from each specific cell type. Returned with 95% confidence interval per cell | Example pie chart for a healthy blood sample |

### Q.3 — Architecture / IAM physics

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.3.1 | **A-score** | The CENTRAL CPG measurement — a number that tells the doctor how well a cell or cell-class is maintaining its identity. A = 1.0 means the cell is sitting AT the minimum informational floor required to be that cell type. A > 1.0 means the cell has drifted ABOVE the floor (more disorder than the minimum required — informational reserve is being consumed). A < 1.0 means the cell is BELOW the floor (either a real biological suppression like in certain neurodegenerative patterns, or a sample-quality artifact the chain flags) | Reference A-score curve from a healthy patient + an actual example IAMAtlas-anchored reading the patient can compare against |
| Q.3.2 | **H_min (architectural floor)** | The minimum informational content a cell type needs to maintain its identity. Derived from physics, not statistics — calibrated by the chain's MCMC machinery. The 8 class floors are: terminal 0.77, immune 0.84, secretory 0.84, progenitor 0.85, cycling 0.86, stromal 0.86, adult stem 0.87, pluripotent stem 0.98 | Bar chart of the 8 H_min floors |
| Q.3.3 | **6-tier verdict** | The clinical interpretation of an A-score. Six tiers from physics-defined breakpoints, NOT statistical percentiles: SUPPRESSED (A < 0.95 — below baseline, context-dependent); NORMAL (0.95 ≤ A < 1.04 — healthy range); ELEVATED (1.04 ≤ A < 1.07 — recoverable drift, intervention window); WARBURG_TRANSITION (boundary line at A = 1.07 — the metabolic-strategy-change line, not a band); SIGNIFICANTLY_ELEVATED (1.07 ≤ A < 1.10 — past the Warburg line, below breach); BREACH (A ≥ 1.10 — architectural fidelity lost; senescence ≈ 1.24–1.27 / malignancy ≈ 1.28–1.32) | 6-tier color-coded chart with each line drawn |
| Q.3.4 | **Warburg transition line (A = 1.07)** | A physics-defined inflection point where it becomes thermodynamically favorable for a cell to switch its metabolism into the Warburg-effect mode (the metabolic shift first described by Otto Warburg in cancer cells). Crossing this line is a meaningful clinical marker — not because we picked it statistically, but because the physics says it's where the metabolic reprogramming becomes favorable | Annotation on A-score axis |
| Q.3.5 | **Architectural fidelity breach line (A = 1.10)** | The physics-defined line above which a cell has lost meaningful architectural fidelity — meaningful disease-state architecture. Clinically significant. | Annotation on A-score axis |
| Q.3.6 | **Bidirectional decomposition (pooled vs directional)** | Some diseases (notably Alzheimer's) move different methylation sites in OPPOSITE directions — some up, some down. If you average these together (a "pooled" score), they cancel out and the chain reads NORMAL even when the disease is present. The bidirectional decomposition separates the up-moving sites from the down-moving sites and scores them independently, catching diseases that pooled scoring would miss | Schematic showing two CpGs moving in opposite directions averaging to zero vs separate directional readouts |

### Q.4 — Cellular aging

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.4.1 | **Cellular age vs chronological age** | Chronological age = years since birth. Cellular age = how old your cells look at the methylation level. These can differ — a 42-year-old whose immune cells look 47 has accelerated cellular aging | Two-clock diagram |
| Q.4.2 | **Immune age delta (the "inflammaging quantum")** | The difference between immune cellular age and chronological age. Positive delta = aging-accelerated; negative delta = younger-than-chronological. The headline aging-burden number the doctor and patient track over time | Example delta with reference range |
| Q.4.3 | **Inflammaging** | The slow accumulation of low-grade systemic inflammation that comes with aging, distinct from acute inflammatory response. CPG measures this directly at the cellular-architecture level (not via blood biomarkers) | Schematic of inflammaging trajectory across ages |
| Q.4.4 | **OSK direction** | Yamanaka's OSK reprogramming factors move cells toward youthful methylation states (validated in CPG by VAL-004: 63.8% / 84.8% aging reversal in two cell types). The chain tracks whether lifestyle changes are moving the patient toward this youthful direction or away from it | Arrow diagram youthful ← OSK direction → aged |

### Q.5 — Statistical concepts (with the astrophysics connection)

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.5.1 | **MCMC (Markov chain Monte Carlo)** | The mathematical machinery underneath every CPG number's confidence. The chain runs simulations that explore all the possible values each measurement could take given the patient's actual data — and the spread of those simulations is what gives the 95% confidence interval. The IAMAtlas posteriors come from 800,000 such samples per cell with rigorous convergence checks (R-hat < 1.001) | Example convergence trace plot |
| Q.5.2 | **Posterior probability** | The probability of something being true GIVEN the data the chain has seen. In CPG: the posterior on "patient carries this disease's signature" is the answer to "how likely is it, given what we measured, that this patient's pattern matches the documented disease pattern?" Always reported with a confidence interval | Example posterior distribution curve |
| Q.5.3 | **Confidence interval (CI)** | The range of values that the true measurement is likely to fall within. 95% CI means: if we repeated the measurement many times, 95% of those measurements would land inside this range. CPG defaults to 95%; 99% available for high-confidence triage; 99.9% for sentinel-event review | Example point estimate + CI bar |
| Q.5.4 | **Mahalanobis distance (the astrophysics connection)** | A way to measure how far one point is from a cloud of other points, accounting for the fact that some directions in the cloud are stretched more than others. Originally developed by P.C. Mahalanobis in 1936; widely used in astrophysics to identify outlier stars and galaxies that don't fit the local cluster's natural shape. CPG uses it because cells don't live in simple spherical neighborhoods — each cell type has its own natural variability pattern, and a patient's "departure from healthy" must respect those natural shapes. A patient might look ordinary on each cell type individually but be an outlier in the MULTIDIMENSIONAL space of all 115 cells together — which is exactly what Mahalanobis catches | Diagram: 2D cloud of points showing why Euclidean distance fails and Mahalanobis succeeds at finding the outlier |
| Q.5.5 | **Multidimensional space** | If you measure 115 cells per patient, each patient becomes a point in 115-dimensional space — one coordinate per cell. You can't draw this directly, but the math works the same as 2D or 3D. Mahalanobis distance lets us measure how far the patient sits from the healthy cloud in this 115-D space | Conceptual diagram with reduction to 2D for visualization |
| Q.5.6 | **Healthy hull (n = 2,481 reference)** | The current CPG reference cloud — measurements from 2,481 healthy people across multiple cohorts. New patients get compared against this cloud's center and shape. The hull grows as more patients contribute (see K.5.1) | Schematic of the hull |
| Q.5.7 | **Percentile** | The fraction of healthy people in the reference cloud whose measurement is below the patient's. 84th percentile = 84% of healthy people score lower (i.e., closer to the cloud center) than the patient. Standard clinical statistical convention | Example percentile bar with patient marker |

### Q.6 — Visualization (CMB and Personal Brilliance Map)

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.6.1 | **Cosmic Methylome Background (CMB)** | The unchanging methylation reference layer the chain scores every patient against — the same for everyone. Just as cosmology's Cosmic Microwave Background is the universal radiation field from the Big Bang that every observation is measured against, the Cosmic Methylome Background is the genetic base layer that every patient's methylation is measured against. Same acronym (CMB), same mathematical projection (Mollweide), same underlying physics framework — applied for the first time to genetics. | Side-by-side: cosmology CMB Mollweide image + CPG CMB Mollweide image |
| Q.6.2 | **Personal Brilliance Map** | THIS patient's methylation departures from the CMB, rendered as bright spots on the same Mollweide projection. Each patient has their own Brilliance Map; bright spots = local departures from the unchanging genetic base layer; color codes direction (red = elevated, blue = suppressed). Different patients shine differently | Example Personal Brilliance Map (healthy vs drift patterns) |
| Q.6.3 | **Mollweide projection** | The map projection used for both the CMB and the Personal Brilliance Map. Same projection used by NASA and ESA to display the cosmology CMB across the whole sky. Flattens a 3D surface onto a 2D oval while preserving area | Side-by-side: a globe vs the Mollweide flat-map view |
| Q.6.4 | **HEALPix (NSIDE = 128, 196,608 pixels)** | The pixelization scheme used by both the cosmology CMB community and CPG. Divides the projection surface into equal-area pixels so every region of the map gets fair representation. The patient's 483,093 CpGs are mapped onto these pixels | Schematic of HEALPix grid |
| Q.6.5 | **8 per-class panels + 1 whole-atlas panel** | The Brilliance Map is rendered as 8 separate views (one per architecture class — terminal, immune, secretory, etc.) so the doctor can see which class is shining where, plus one whole-atlas view that aggregates everything into a single "everything at once" image | Example 9-panel layout |
| Q.6.6 | **Anisotropy zoom** | When a class is in BREACH tier, the report can drill into that class's CMB panel to show the specific regions of the methylome that are shining most | Example zoom on a BREACH region |
| Q.6.7 | **Patient-vs-self overlay (serial)** | Same patient's Brilliance Map rendered at two timepoints with the difference shown — letting the doctor and patient see how the patient's brightness pattern has evolved | Example two-timepoint overlay |

### Q.7 — Disease detection

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.7.1 | **Disease matrix** | The chain's catalog of documented disease signatures — what each disease "looks like" in terms of which cells are affected and how. Currently 52 unique diseases / 81 disease-phase rows. Grows over time as more patients contribute their patterns back (K.5.2) | Schematic table of the matrix |
| Q.7.2 | **Detection probability per disease** | The posterior probability the patient carries a specific disease's architectural signature, with confidence interval. For a healthy patient with no architectural drift, the expected output for almost every disease is "below clinical threshold" — and that absence is itself a clinically meaningful result | Example bar chart of detection probabilities across diseases |
| Q.7.3 | **Field effect** | When a disease's signature appears not just in the diseased tissue itself but also in the surrounding apparently-normal tissue. CPG validates this across 28 cancer types (VAL-003, p = 1.32e-15) — meaning the chain can detect cancer-pattern drift even in sampled tissue that hasn't yet become cancerous | Schematic of field effect spread |
| Q.7.4 | **Pre-diagnostic window** | The years before a disease becomes clinically detectable when the methylation signature is already emerging. For breast cancer, the immune-class signature is detectable 10+ years before clinical diagnosis (VAL-047 d = +1.78). For AD, the trajectory signal is detectable decades before clinical symptoms | Timeline diagram showing pre-dx window |
| Q.7.5 | **Cross-disease universal alarm** | A specific set of methylation sites that move in coordinated patterns across multiple different diseases — a generalized "something is wrong" signature that fires before any specific disease pattern emerges | Schematic of the alarm channel |
| Q.7.6 | **Trajectory-essential vs trajectory-additive** | A clinical distinction the chain makes explicit. Trajectory-essential diseases (Alzheimer's, FTD, PSP/CBD, Parkinson's, ALS, MS, the 10+ year pre-cancer windows) have moderate-to-weak single-sample signal — they NEED serial samples for clinical detection. Trajectory-additive diseases (active solid cancers with field-effect signatures) have strong single-sample signal — serial samples extend value but aren't required for detection | Comparison diagram showing single-sample vs trajectory signal strength |

### Q.8 — Trajectory (serial sampling)

| Item | Concept | Brief inline definition draft | Visual aid |
|---|---|---|---|
| Q.8.1 | **Baseline (population vs patient)** | At the patient's first sample, "normal" is defined by the population reference (the n = 2,481 healthy hull). After enough samples on the same patient, "normal" can be defined by THIS patient's own stable architecture — which is far more sensitive for detecting drift specific to this patient | Two-line graph: population range vs patient personal range |
| Q.8.2 | **Patient-as-own-baseline (the major Tier 3 unlock)** | The chain's shift from comparing the patient to a population to comparing the patient to themselves. Available at ~sample 6+ when enough data has accumulated to characterize the patient's stable architecture | Schematic of the shift |
| Q.8.3 | **Drift cascade** | A documented pattern where one cell class begins drifting first, then a second class follows, then a third — a cascade signature characteristic of certain disease trajectories. Validated by VAL-037 → VAL-046 (35/39 pre-specified predictions confirmed). Serial samples let the chain detect whether a patient is in a documented cascade | Cascade timeline diagram |
| Q.8.4 | **Active surveillance** | A monitoring protocol for high-risk patients (BRCA carriers, Lynch syndrome carriers, post-treatment cancer survivors, family-history-high-risk patients). Quarterly or biannual sampling with progressively tighter detection thresholds tuned to the patient's own baseline | Cadence diagram |
| Q.8.5 | **Forecasting** | Once enough trajectory data exists, the chain can extrapolate: "given current drift rate, you would cross the BREACH line on immune class in ~3 years if rate continues unchanged." Per-metric time-to-event predictions with CI | Forecast trajectory plot |
| Q.8.6 | **Network learning** | Every patient who commits to ongoing CPG sampling contributes their anonymized data back to the chain itself — refining the CMB, expanding the disease catalog, narrowing posteriors on every cell type. The cellular fitness tracker improves the more patients commit | Schematic of contribute-back loop |

### Q.9 — Wellness / context

| Item | Concept | Brief inline definition draft |
|---|---|---|
| Q.9.1 | **Inflammaging** (defined in Q.4.3 — referenced here in wellness context) | The slow accumulation of low-grade systemic inflammation that comes with aging |
| Q.9.2 | **Smoking cessation reversibility signature** | For former smokers, the chain detects whether the immune architecture is still showing residual smoking signature or has reverted toward never-smoker baseline. Validated by CPG-VAL-022 |
| Q.9.3 | **Menarche immune signature** | Documented hormonal-context immune signature used to stratify female patients in the chain's reference distribution. Validated by CPG-VAL-018 |
| Q.9.4 | **Lifestyle stratification** | Patient intake covariates (smoking, alcohol, sleep, stress, exercise, nutrition, hormonal state) used to interpret architectural readouts in context — same A-score means different things for different patient contexts |

### Q.10 — Generation infrastructure

| Item | What the chain outputs |
|---|---|
| Q.10.1 | **Inline figures via embedded Python** — the report-generation script (Stage 10) carries the Python code that renders every figure in the report: the Personal Brilliance Map, the per-class panels, the 6-tier verdict chips, the example A-score curve from a CPG comparison patient, the Mahalanobis 2D conceptual diagram, the cosmology-CMB ↔ CPG-CMB side-by-side, every educational schematic. The doctor receives a single PDF with all figures rendered inline; no external references; reproducible from the same script if the lab needs to regenerate |
| Q.10.2 | **Glossary appendix** — the full definitions from Q.1–Q.9 also live in a glossary appendix at the end of the report so the doctor and patient can navigate to any concept without scrolling back to the inline first-appearance definition |
| Q.10.3 | **Per-section reading-level option** — the report supports two reading modes: standard clinical (default; written for the doctor) and patient (written for the patient with the doctor's context). Same data, different framing depth per concept |

---

# WHAT THE DOCTOR LEARNS FROM A SINGLE REPORT (the baseline)

For the **wellness patient seeing a naturopath**, this is the first sample — the cellular fitness tracker baseline. Everything below is what they get day one; the trajectory analytics in Section K + the collective learning in K.5 are what their commitment to the model compounds into over time.

For the **diagnostic patient** with a specific concern, this is the full readout — a comprehensive molecular-level look at the patient's cellular state at this moment in time.

1. WHO is in the blood draw — 115 cells total in the IAMAtlas (51 immune subdivisions organized into 19 customer-facing lineage pages + 64 cells across 7 non-immune classes), all with cell-of-origin identified by the deconvolver from any methylation input (substrate affects relative prominence, not detectability) — all with CI
2. HOW each cell is maintaining its identity (A-score per cell + class + 6-tier verdict) with CI
3. HOW OLD the patient looks at the cellular level (immune age + delta + per-class) with CI
4. HOW FAR from healthy in multidimensional space (Mahalanobis + top contributing cells) with CI
5. WHAT the per-class Personal Brilliance Map shows (scored against the CMB base layer) — 8 per-class panels + 1 whole-atlas panel
6. WHETHER directional patterns are present that pooled scores would miss
7. **WHICH diseases the chain detected at clinical threshold** — every one of the 52 diseases / 81 phase rows is scored; diseases that match the patient's pattern strongly enough to cross the clinical threshold are flagged with detection probability + CI; **for a healthy patient with no architectural drift, the typical and expected output is "no disease patterns detected above clinical threshold" — that absence is itself the clinical readout**
8. WHETHER the cross-disease universal alarm fires
9. WHAT lifestyle / inflammaging / hormonal / chronic / acute context is active
10. HOW family history + prior cancers shift the priors
11. WHERE the literature backing each finding lives
12. WHAT is NOT KNOWN (honest gaps surfaced)
13. HOW CONFIDENT we are in every single number (CI on every metric, computable for every disease because the IAMAtlas posterior backbone is universal)
14. **WHETHER the patient's situation is trajectory-essential or trajectory-additive** — a critical clinical distinction the chain makes explicit per finding: some diseases (neurodegeneration like AD, FTD, PSP/CBD, Parkinson's, ALS, MS; pre-cancer windows at longest lead time; treatment response; recurrence surveillance) produce moderate-to-weak single-sample signal and the trajectory is the diagnostic modality, not optional. AD single-sample AUC is 0.68 (VAL-051) / 0.60 cross-platform (VAL-052); pooled is NULL because AD drift is bidirectional. The single-report value for these patients is the BASELINE — serial sampling is where the clinical answer lives. For trajectory-additive diseases (active solid cancers with strong field-effect signatures), single-sample is often decisive; serial samples extend value.
15. **AND the report explains itself as it informs.** Every novel concept (methylation, CpG, cell class, A-score, 6-tier verdict, Mahalanobis distance with its astrophysics connection, multidimensional space, posterior probability, CI, Cosmic Methylome Background, Personal Brilliance Map, bidirectional decomposition, inflammaging, OSK direction, etc. — ~40 concepts in Section Q) gets a brief inline definition the first time it appears, with an example or diagram rendered inline by the report-generation script's embedded Python. The doctor and patient learn the framework AS they read the report. Full glossary appendix at the end for navigation.

At **Tier 2** (+ fragmentomics): adds cross-substrate cellular age concordance gate.
At **Tier 3** (+ serial samples): adds full trajectory monitoring (Section K) — progressively unlocks rate-of-change at sample 2, trend significance at samples 3–5, **patient-as-own-baseline** at samples 6+ (the major unlock where reference range shifts from population to patient-specific), and at long-term surveillance: forecasting, treatment response monitoring, recurrence early warning, drift cascade detection, active surveillance protocols, and intervention-response tracking. The compound message: each new sample doesn't just add data — the chain learns more about THIS patient specifically, and detection sensitivity dramatically improves over time.

---

# DECISIONS FOR HEATH BEFORE THE REPORT BUILDER SECTION GETS DRAFTED INTO BUILD_SPEC

1. **Report format** — clinical one-pager (top findings only), full clinical report (8–12 pages), portal with drill-down, EHR-integrated JSON, or some combination?
2. **Headline visual** — 8-panel Personal Brilliance Map per class (F.1) + whole-atlas Brilliance Map (F.7), per-class tier verdict chips (C.3), flagged-disease detection summary (H.3) — for healthy patients this will most often read as "no disease patterns detected above threshold," for patients with drift this surfaces actual flagged diseases — or all three on the first page?
3. **Default CI level** — 95% standard. Surface 99% for borderline calls automatically, or doctor-toggled?
4. **Posterior shape rendering** — full marginal density curves in doctor portal, or summary statistics only?
5. **Audit trail visibility** — "click for provenance" link per finding, or back-office only?
6. **Disease section sort** — by posterior magnitude, by `posterior × confidence`, by clinical urgency, or doctor-selectable?
7. **Wellness section depth** — full Section J (9 categories) or trimmed to what intake supplied?
8. **Family history integration** — include L.3 posterior combination, or hold until intake gathers full family history?
9. **Section O caveats** — fully exhaustive or trimmed to most-relevant-to-this-patient?
10. **What is the headline call?** — does the report lead with the most confident finding, the highest posterior finding, the highest-severity finding, or doctor-selectable framing?
11. **Trajectory presentation strategy** — for patients with multiple samples, three sub-decisions:
    - (a) Does the report explicitly surface "you've completed N samples; here's what unlocks at sample N+1" guidance to motivate continued sampling, or is the trajectory analytics just rendered without that framing?
    - (b) When patient-as-own-baseline (K.3.1) becomes available at sample 6+, does the report switch the primary reference range silently, or does it explicitly show "previously compared to population HC; now compared to YOUR personal baseline" with both side-by-side for context?
    - (c) For active surveillance patients (BRCA carriers, post-treatment survivors), does the chain emit dedicated active-surveillance reports with quarterly forecasting, or is that the same report format with additional trajectory sections?
12. **Patient archetype framing (the cellular fitness tracker positioning)** — the report builder needs to handle two distinct framings:
    - (a) For **wellness patients** (the naturopath's primary patient, health-engaged, whole-body monitoring focus): does the report lead with the cellular-fitness-tracker framing ("here is your cellular baseline; here is what serial samples will unlock"), surface the K.5 collective learning value proposition, and de-emphasize the disease-list section in favor of trajectory + lifestyle context + intervention response?
    - (b) For **diagnostic patients** (specific concern, family history, active symptoms, BRCA/Lynch carrier, post-treatment survivor): does the report lead with disease pattern matching + posterior + CI, with trajectory framing as supporting context for follow-up cadence recommendation?
    - The same chain produces both readouts; the question is whether the report builder takes a `patient_archetype` flag from intake and presents accordingly, or whether the doctor selects framing at delivery time
13. **Collective learning consent + privacy framing** — does the report explicitly invite the patient into the K.5 contribution model (anonymized aggregate data improving the chain for all patients) with separate opt-in, or is that handled outside the report at intake-consent time? How is the patient told that their committing to ongoing sampling improves the system for everyone who comes after?
14. **Educational definitions strategy (Section Q)** — three sub-decisions:
    - (a) Inline definitions at first appearance + glossary appendix (current proposal), or glossary appendix only with hover/click definitions in the digital version?
    - (b) For the foundational concepts (methylation, CpG, β value, cell class) — assume the doctor knows these and only define them for the patient view, or always include in both views (the doctor may not have seen them in this clinical framing before)?
    - (c) For the astrophysics-connection definitions (Mahalanobis, multidimensional space, CMB mirror) — lead with the technical explanation and add the astrophysics analogy for color, or lead with the cosmology analogy as the intuition-building hook and follow with technical detail? The CMB mirror is one of the most distinctive things about CPG — the question is how prominent to make it in the educational layer

Once decisions land, the report builder section in BUILD_SPEC v1.2.1 → v1.2.2 gets drafted from the approved subset.
