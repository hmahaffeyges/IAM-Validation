# Early-Onset Rectal Subsection — Phase 1 Cohort Survey

**Survey date:** 2026-04-28
**Operational stance:** LL-PUBLIC-TIER (public Tier 1 GEO/GDC only, no biobank applications, no outreach, no preprint-first)
**Card scope (per Heath sign-off 2026-04-28):** Subsection within `crc-epic` (not standalone `rectal-epic` card) — atlas resolution does not support rectum-vs-colon cell-of-origin disambiguation. Subsection adds early-onset (under-50) age-stratified clinical-action routing on top of the existing crc-epic biology layer. Future promotion to standalone card pending atlas-resolution evolution.

---

## 1. Headline finding

**Adequate Tier 1 public data exists to validate the early-onset rectal/CRC stratum within crc-epic, BUT the data does NOT support a Hispanic-stratum biology claim.** TCGA-READ has 21 patients in the 30-49 age range; only 1 self-reports as Hispanic. TCGA-COAD has 62 patients age 30-49; only 4 self-report as Hispanic. The Hispanic stratum is biobank-gated in the cohorts that would actually answer that question (GSE284325 has 16 Hispanic + African American EOCRC patients but is WGBS, not array, requiring methods translation work).

The honest scope for Phase 2 is therefore: validate that the cycling-class architectural drift signal documented in VAL-061/VAL-062 (TCGA-COAD pooled cohort, predominantly older patients) holds in the under-50 stratum; document the Hispanic-stratification angle as an unmet-evidence-gap with the appropriate biobank-gated cohorts logged in `crc-epic/future_when_support_arrives.md`; route clinical action by age × ethnicity × family history at the action-matrix layer, not at the methylation-detection layer.

---

## 2. Tier 1 cohorts surveyed

### 2.1 TCGA-READ (Rectum Adenocarcinoma) — primary anchor candidate

- **Source:** NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}`. No dbGaP, no application required.
- **Platform:** Illumina HumanMethylation450 (HM450, GPL13534), sesame level3 betas
- **Sample composition (queried 2026-04-28):**
  - 98 unique cases with HM450 methylation files
  - 98 Primary Tumor samples + 7 Solid Tissue Normal + 1 Recurrent Tumor (= 106 total HM450 files)
  - 7 paired tumor/adjacent-normal pairs available (the rectal counterpart to VAL-062's 26 COAD pairs)
- **Age stratification (n=132 case-diagnosis events with age data):**
  - 30s: 3 patients
  - 40s: 18 patients
  - 50s: 30 patients
  - 60s: 33 patients
  - 70s: 42 patients
  - 80s: 6 patients
  - **Age 30-49 stratum: 21 patients** — the early-onset rectal target stratum
- **Ethnicity (case-level self-report):**
  - "not hispanic or latino": 77
  - "not reported": 28
  - "hispanic or latino": 1
  - **Hispanic stratum: structurally underpowered (n=1).** TCGA-READ does not support a Hispanic-vs-non-Hispanic statistical comparison at any age stratum.
- **Race:**
  - white: 77
  - not reported: 23
  - black or african american: 5
  - asian: 1
- **Caveats pre-locked:** TCGA-READ is predominantly Western, predominantly white, predominantly older. Smoking-status metadata is incomplete in the diagnosis records (need to query separately if smoking-stratification is desired). Anatomic origin includes "Colon", "Rectosigmoid junction", "Rectum", and others — the project label is "Rectum Adenocarcinoma" but the primary_site field shows the cohort spans rectosigmoid junction and includes some colon-labeled cases. Subsite stratification at the patient level is required to confirm which of the 98 cases are pure-rectal vs rectosigmoid.
- **Manifest:** Will need to construct a `READ_matched_manifest.json` analogous to `LUAD_matched_manifest.json` and `LUAD_pairs.json`. The 7 paired tumor/normal pairs are the candidate VAL-098 cohort (within-cohort paired comparison, satisfies CHK-3.8 condition 1 — no cross-cohort calibration problem).
- **Status:** **Primary candidate for VAL-098 within-cohort paired tumor-vs-adjacent-normal analysis. Subsetted by age decile to surface the under-50 stratum reading specifically.** Same methodology as VAL-062 on COAD; this is the rectal-subsite extension.

### 2.2 TCGA-COAD age-stratified re-analysis — secondary anchor

- **Source:** Already in cookbook record via VAL-061/VAL-062. No new data acquisition needed.
- **Platform:** HM450 (GPL13534), sesame level3
- **Sample composition (queried 2026-04-28):**
  - 295 unique cases with HM450 methylation files
  - 312 Primary Tumor + 38 Solid Tissue Normal + 1 Recurrent Tumor + 1 Metastatic
  - 26 paired tumor/normal pairs already used in VAL-062 (n=26)
- **Age stratification:**
  - 30s: 13 patients
  - 40s: 49 patients
  - 50s: 86 patients
  - 60s: 106 patients
  - 70s: 127 patients
  - 80s: 78 patients
  - **Age 30-49 stratum: 62 patients** — substantially larger than TCGA-READ's 21
- **Ethnicity:**
  - "not hispanic or latino": 304
  - "not reported": 43
  - "hispanic or latino": 4
  - "Unknown": 1
  - **Hispanic stratum at age 30-49: 3 cases.** Direction-only evidence at best.
- **Anatomic subsite distribution (top 8):** Ascending colon 87, Sigmoid colon 82, Cecum 73, Colon-NOS 66, Descending colon 16, Liver 14, Transverse colon 13, Not-Reported 96. **Sigmoid colon (n=82) is the anatomic neighbor of rectum and where rectum-vs-colon disambiguation is least clear.**
- **Status:** Re-analyze VAL-061/VAL-062 stratified by age decile + anatomic subsite to surface (a) whether cycling-class architectural drift signal magnitude differs in the under-50 stratum vs older, and (b) whether sigmoid-vs-other-colon shows any methylation distinction (the anatomic-neighbor question for atlas-resolution discussion). **This is a re-slicing of existing VAL-061/VAL-062 per-sample CSVs, not a new VAL.** Logged as VAL-099 candidate.

### 2.3 GSE282666 — Biological Age Acceleration and Colonic Polyps in Persons Under Age 50

- **Source:** GEO public, FTP-accessible. No restrictions.
- **Citation:** No PubMed ID yet (not yet indexed); deposit is Tier 1 public.
- **Platform:** Illumina MethylationEPIC (GPL33022 — newer EPIC version), buffy coat peripheral blood
- **Sample composition:** 51 patients, all under age 50, all undergoing colonoscopy
- **Sample characteristics:** "tissue: Buffy Coat" — Stage 1 immune blood pathway, NOT tissue
- **Phenotype:** Pre-neoplastic polyps (PNP) — tubular adenomas and sessile serrated adenomas — vs no-polyp at colonoscopy
- **Supplementary files:**
  - `GSE282666_Betas.csv.gz` — preprocessed beta matrix (ready for direct loading, no IDAT processing required)
  - `GSE282666_RAW.tar` — IDATs available for re-processing if needed
  - `GSE282666_epigeneticage_default_predictions.xlsx` — pre-computed epigenetic clock predictions
- **What this cohort tests:** The Stage 1 immune A-score signal (Xu-538 panel, immune H_min 0.838889) on EPIC buffy coat from under-50 patients with colonic polyps. Whether the existing crc-epic Stage 1 immune signal direction (negative d, suppressed circulating response per VAL-047) holds in the under-50 stratum, and whether polyp-positive patients show measurable A-score departure from polyp-negative patients in the same age range.
- **Caveats pre-locked:**
  - Polyps are pre-neoplastic, not invasive cancer — this is even earlier in the disease trajectory than VAL-047 pre-diagnostic (which was within 10 years of clinical diagnosis). The signal direction could be the same as VAL-047 or could be different (polyp biology may not yet have triggered the systemic immune suppression that overt CRC does).
  - Sample size n=51 is modest; subgroup analysis (polyp+/polyp- by polyp type) further reduces stratum sizes. Pre-locked underpower for any stratum n<10.
  - GPL33022 is newer EPIC — confirm CpG coverage on Xu-538 panel (538/538 expected on EPIC v1 and v2; verify before scoring).
  - No anatomic location of polyp metadata in the basic series matrix — would need to check the supplementary clinical file or the publication for polyp location (rectum vs colon vs both).
- **Status:** **Direct candidate for VAL-100 — Stage 1 immune A-score on under-50 buffy coat with polyps, EPIC platform.** This is independent evidence of whether the crc-epic Stage 1 signal extends to the early-onset polyp biology layer. Same Xu-538 panel as VAL-047. Same H_min. No cross-cohort calibration problem because the cohort is internally controlled (polyp+ vs polyp- in the same cohort, same pipeline).

### 2.4 GSE284325 — DNA Methylation in EOCRC Hispanic + African American — DEFER per LL-PUBLIC-TIER methods constraint

- **Source:** GEO public, but **WGBS (whole-genome bisulfite sequencing), NOT array methylation.**
- **Citation:** PubMed 39844333
- **Platform:** GPL24676 — Illumina NovaSeq 6000
- **Sample composition:** 16 EOCRC tissue samples, Hispanic + African American patients
- **Why this would be relevant:** The most-on-target Hispanic-stratification cohort in public data, specifically focused on early-onset CRC in underrepresented populations.
- **Why we cannot use it directly at v1:**
  - WGBS produces continuous methylation coverage at base-pair resolution; our framework is calibrated on array CpG sites (Loyfer 25-tile atlas built on EPIC array). Translating WGBS to array-CpG positions requires platform conversion that the cookbook has not yet implemented. Caggiano CelFiE TIM matrix is the closest comparable WGBS-region atlas; it is on disk but is in queue for engineering integration (per atlas vault README, Caggiano is "Queue 1, NOT yet in production. WGBS-region-based, integration requires mapping WGBS regions to array CpG positions per platform").
  - Sample size n=16 is small even for direct in-cohort analysis.
  - Cross-cohort comparison to TCGA-COAD/READ would trigger CHK-3.8 violation (different platform, different population, different preprocessing) and would land on O5_BASELINE_DOMINATED per the VAL-097 lesson.
- **Status:** **Logged in `crc-epic/future_when_support_arrives.md` for v0.3 promotion when the WGBS-to-array projection engineering work is complete (post-Caggiano integration).** Not pursued at v1 of the early-onset rectal subsection. The Hispanic-stratification biology question stays open and explicitly documented.

### 2.5 GSE288652 — Colon adenoma low-grade vs high-grade vs adenocarcinoma — secondary candidate

- **Source:** GEO public, FTP-accessible.
- **Citation:** PubMed 41291100
- **Platform:** Methylation array (platform ID not in initial query — verify before use)
- **Sample composition:** 32 samples covering non-tumor colon tissue → low-grade adenoma → high-grade adenoma → adenocarcinoma
- **What this cohort tests:** The methylation trajectory across the adenoma-to-carcinoma progression. This was already noted as relevant in the existing crc-epic v2.2 record (VAL-039 / Kadota 2014 was the lung adenocarcinoma distance-annotated field effect equivalent). For the early-onset rectal subsection, GSE288652 is a candidate for documenting whether the early-onset architectural signal is detectable at the adenoma stage (pre-invasive) vs only at adenocarcinoma stage. Anatomic site coverage (colon vs rectum vs sigmoid) needs verification before commit.
- **Status:** Secondary candidate for VAL-101, logged for Phase 2 if VAL-098/099/100 leave gaps. Defer to confirm anatomic and age coverage before committing.

### 2.6 GSE220160 — Colorectal cancer and polyp methylation — secondary candidate

- **Source:** GEO public.
- **Sample composition:** 16 samples (CRC + polyp + control). Small but on-platform.
- **Status:** Logged as a small-n cross-cohort replication target if the Stage 1 GSE282666 result is positive. Lower priority than GSE288652.

### 2.7 Cohorts NOT pursued (per LL-PUBLIC-TIER)

The following candidate cohorts were identified during the survey but are biobank-gated, dbGaP-restricted, or otherwise outside Tier 1 public access. Logged in `crc-epic/future_when_support_arrives.md`; not pursued.

- **Memorial Sloan Kettering EOCRC methylation cohort** (referenced in EOCRC literature) — institutional access only.
- **City of Hope EOCRC Latino cohort** (Latino-stratified, the most directly on-target for the Hispanic question) — biobank-gated, institutional collaboration required.
- **Dana-Farber EOCRC Project** — biobank-gated.
- **EnviroGenomarkers EOCRC subset** — same biobank-gating pattern as the existing crc-epic VAL-046 UK Biobank reference and the heme-LL-011 Italian biobank gating pattern.
- **NCI Cohort Consortium EOCRC pooled methylation** — application-required.

These are documented for completeness. **No outreach. No application submission.**

---

## 3. The honest validation slate for Phase 2 (subsection build)

Three VALs deliver the early-onset rectal subsection at Tier 1 public:

### VAL-098 — TCGA-READ within-cohort paired tumor-vs-adjacent-normal cycling-class architectural drift, age-stratified

- **Cohort:** TCGA-READ HM450 7 paired tumor/adjacent-normal pairs (the rectal counterpart to VAL-062's 26 COAD pairs)
- **Method:** Identical to VAL-062 — paired Cohen's d on cycling-class A-score, H_min 0.856055, all valid HM450 CpGs. Bootstrap 10000 paired-d CI. Within-cohort paired (CHK-3.8 condition 1 satisfied — no cross-cohort calibration problem). RNG seed 20260428.
- **Pre-locked stratification:** Age decile (30s, 40s, 50s, 60s+); anatomic subsite (rectum vs rectosigmoid junction vs colon-labeled); n<5 sub-strata pre-locked as descriptive-only per CHK-2.7. Smoking status if metadata available, otherwise documented as honest CHK-2.7 caveat.
- **Pre-locked decision criteria:** O1_PASS (paired d ≥ +0.5 across pooled cohort, direction consistent with VAL-062 COAD result d=+0.724), O2_DIRECTION_CONSISTENT_AGE_INDEPENDENT (paired d positive across all age strata even if magnitude varies), O3_AGE_STRATIFIED_DIVERGENCE (under-50 stratum shows different d magnitude than 50+, flag for biological interpretation), O4_NULL (paired d <+0.5 with CI crossing zero), O5_BASELINE_DOMINATED (does not apply — within-cohort paired comparison structurally avoids this), O6_DATA_INTEGRITY (β distribution health check failure, manifest mismatch, etc.).
- **Pre-locked CHK-4.10:** ≥3 tiles >3 anchor-SD AND ≥80% same-direction = baseline-dominated. **Does not apply to single-tile cycling-class scoring — placeholder only.**
- **Expected outcome direction (sign-locked before β access):** Paired d > 0, magnitude comparable to VAL-062 COAD (+0.724) or larger (rectal cancer has higher mutational burden than colon). Direction-only confidence high; magnitude calibration uncertain.
- **Reproducibility triple:** TCGA-READ manifest (will be constructed from the GDC query above as `READ_matched_manifest.json`); GDC public API for IDAT/sesame-level3 download; Loyfer atlas for Stage 2 layered scoring; Python 3 + numpy + pandas + scipy (same environment as VAL-062).
- **Runtime estimate:** ~5 minutes total — 7 pairs is a small download; A-score computation is trivial at this n.

### VAL-099 — TCGA-COAD age-stratified re-analysis of VAL-061/VAL-062 per-sample data

- **Cohort:** Existing TCGA-COAD VAL-062 per-sample CSV (n=26 paired pairs already in cookbook record). No new data acquisition.
- **Method:** Re-slice VAL-062's per-sample paired d by age decile (30s, 40s, 50s, 60s+) and anatomic subsite (sigmoid colon vs ascending vs descending vs cecum vs other). Bootstrap age-stratified d CI per stratum.
- **Pre-locked underpower note:** TCGA-COAD VAL-062 cohort n=26 is small for age-stratified analysis — the under-50 stratum likely contains 5-8 patients. Per CHK-2.7, n<5 strata are descriptive-only; n=5-10 strata are reported with explicit "underpowered, direction-only" framing.
- **Why this matters:** Tests whether the existing crc-epic cycling-class tissue arm signal (paired d=+0.724) is age-stratification-stable, OR whether under-50 cases show a different magnitude. If signal is age-independent, the early-onset subsection is a clinical-action layer only (no methylation distinction). If signal differs by age, the subsection has biology-detection content.
- **No new prereg, no new data download — this is a CSV re-analysis on existing sealed data.** Reanalysis prereg is light: "applying age + subsite stratification to existing VAL-062 per-sample CSV with bootstrap CIs per stratum."

### VAL-100 — GSE282666 Stage 1 immune A-score on under-50 buffy coat with colonic polyps

- **Cohort:** GSE282666, n=51 EPIC buffy coat under-50 patients with colonoscopy-confirmed polyp status (PNP+ vs PNP-).
- **Method:** Stage 1 immune A-score on the Xu-538 panel (538 CpGs × immune H_min 0.838889, byte-match to existing crc-epic and breast-epic Stage 1 panels). Within-cohort case-vs-control on PNP+ vs PNP-. Bootstrap 10000 d CI.
- **Pre-locked stratification:** Polyp type (tubular adenoma vs sessile serrated adenoma vs no polyp); polyp location if metadata available (rectum vs colon vs both); sex.
- **Pre-locked decision criteria:** O1_DIRECTION_CONSISTENT_WITH_VAL_047 (PNP+ vs PNP- d ≤ −0.3, direction consistent with the negative d that VAL-047 documented for CRC pre-dx blood — the immune-suppression-of-circulating-response signal); O2_DIRECTION_INVERTED (d ≥ +0.3, signal direction inverts at the polyp/under-50 stratum vs invasive-CRC stratum, would be a finding worth investigating); O3_NULL (|d| < 0.3, polyp biology does not yet trigger the systemic Stage 1 signal); O5_BASELINE_DOMINATED (does not apply — within-cohort comparison); O6_DATA_INTEGRITY.
- **Pre-locked CHK-3.2 expectation:** Within-cohort comparison; CHK-3.2 still required for sanity but no cross-cohort baseline issue expected. CHK-3.8 condition 1 satisfied.
- **Expected outcome direction:** Honestly uncertain. VAL-047 documented d=-0.33 for invasive CRC pre-dx blood. Polyp biology is earlier in the disease trajectory and may not yet have produced the systemic immune-response shift. Direction lock pre-set: uncertain. The VAL is genuinely investigative, not confirmatory.
- **Why this matters:** Tests whether the crc-epic Stage 1 signal extends backward in disease trajectory (from invasive CRC pre-dx, where VAL-047 documented it, to pre-neoplastic polyps where GSE282666 lives). Direct evidence for the early-onset detection-window question.
- **Runtime estimate:** Direct β-matrix download + Stage 1 scoring + bootstrap = ~10 minutes.

### Phase 2 deliverable timeline

VAL-098 + VAL-099 + VAL-100 form the minimum viable Phase 2 evidence base for the early-onset rectal subsection. All three are within-cohort comparisons (no cross-cohort calibration problem), all three are Tier 1 public data, all three use existing canonical Loyfer atlas + Xu-538 panel (no new methodology). Combined runtime ~30 minutes of compute; combined prereg-and-write ~3-4 hours of careful authoring.

---

## 4. Subsection v0.1 design (high-level scope before commit)

The subsection lives within `crc-epic_card_v2.4.json` (additive update from v2.3) under a new top-level key `early_onset_rectal_subsection`. Mirrors the schema of existing top-level blocks (`stage_1_immune_flag`, `stage_2_localization`, `tissue_arm`, etc.).

**Block contents (proposed schema for Phase 2):**

```
early_onset_rectal_subsection: {
  scope: "Clinical-action routing layer for under-50 patients (especially Hispanic, especially with family history of CRC <50) with elevated cycling-class colorectal A-score on the existing crc-epic Stage 2 layer. Subsection does NOT claim rectum-vs-colon biology-detection at v1 — atlas resolution does not support that distinction. Subsection adds age × ethnicity × family-history routing on top of the existing crc-epic methylation signal.",
  
  motivating_epidemiology: { ... published rising-incidence statistics with proper citations ... },
  
  atlas_resolution_constraint: "v1 Loyfer atlas (production Stage 2) has Colon_epithelial_cells tile but no rectum-specific tile. Methylation signal cannot disambiguate rectum vs sigmoid vs ascending colon at v1. Future atlases (single-nucleus methylation, 223-cell-type WGBS, GTEx-derived sigmoid-vs-rectum split) will enable subsite-specific tile separation; subsection promotes to standalone rectal-epic card at that point.",
  
  validation_evidence: {
    val_098_tcga_read_paired: { ... },
    val_099_tcga_coad_age_stratified: { ... },
    val_100_gse282666_under_50_polyp: { ... },
  },
  
  age_stratified_clinical_action_matrix: {
    under_30: "...",
    age_30_to_49_no_family_history: "...",
    age_30_to_49_with_family_history_of_crc_under_50: "...",
    age_30_to_49_hispanic_self_reported: "Per ACG 2021 EOCRC screening guidance; ethnicity-aware routing to colonoscopy with rectal exam emphasis",
    age_50_plus: "Standard crc-epic clinical action matrix (existing card content unchanged)",
  },
  
  hispanic_stratification_status: "Direct biology evidence not available at Tier 1 (TCGA-READ n=1 Hispanic, TCGA-COAD n=4 Hispanic). The on-target Hispanic-stratification cohort (GSE284325 EOCRC Hispanic + African American n=16) is WGBS — requires post-Caggiano-integration methods translation. Hispanic stratum logged as future_when_support_arrives.md entry. Clinical-action routing applies ethnicity-aware screening per published ACG / NCCN EOCRC guidance, NOT based on Hispanic-specific methylation evidence (which we cannot claim from the available Tier 1 data).",
  
  commercial_deployment_unaffected_by_subsection_evidence_gaps: "Per CCL-037: EDEAR commercial deployment is single-pipeline patient-vs-internal-reference and is unaffected by Tier 1 cohort coverage gaps for any subgroup. The early-onset rectal subsection adds clinical-action routing logic, not deployment-architecture changes.",
  
  next_validation_steps: [...],
  known_limitations: [...],
}
```

---

## 5. Hispanic-stratification framing (CRITICAL — get this right)

The honest version of the Hispanic-stratification claim, given what the data supports:

**What we CAN say:**
- "Published epidemiology documents disproportionately rising incidence of early-onset colorectal cancer in Hispanic populations in the western United States, particularly in age strata 30-49." (with proper citations)
- "The crc-epic clinical_action_matrix routes patients with elevated cycling-class colorectal A-score by age, family history, and self-reported ethnicity according to published ACG 2021 and NCCN EOCRC screening guidance."
- "EDEAR's methylation signal is the same architectural drift across all populations; what differs across populations is the prior probability of disease in age strata and the clinical-action pathway it triggers."

**What we CANNOT say at v1:**
- "EDEAR detects Hispanic-specific colorectal methylation signatures." (We have no Hispanic-stratified array methylation data at Tier 1 sufficient for that claim.)
- "EDEAR is calibrated for Hispanic patients." (Calibration is calibrated against the public Loyfer/Moss atlases, which are not population-stratified at v1.)
- "EDEAR is sensitive to early-onset rectal cancer specifically." (The methylation signal localizes to colorectal epithelium; rectum vs sigmoid is at v1 a clinical-pathway distinction, not a methylation-detection distinction.)

The framing in the subsection is therefore: **EDEAR is not population-specific; the clinical action pathway IS population-aware.** EDEAR fires the same red flag for all patients with elevated cycling-class colorectal architectural drift. The clinical-action routing — which patients get an immediate colonoscopy with rectal exam emphasis vs which get standard screening interval — uses age + ethnicity + family history as ROUTING covariates, not as DETECTION covariates.

This framing matches the LL-PUBLIC-TIER stance ("EDEAR is health-and-wellness early-detection, not regulated diagnostic") and avoids overclaiming on data we do not have.

---

## 6. Phase 2 commit

If Heath signs off on this Phase 1 survey, Phase 2 work proceeds as:

1. **VAL-098 prereg** sealed before any TCGA-READ β access. Construct READ_matched_manifest.json from GDC API. Run paired analysis. Write outcome.md.
2. **VAL-099 prereg** sealed before TCGA-COAD CSV re-slicing. Run age + subsite stratified bootstrap. Write outcome.md.
3. **VAL-100 prereg** sealed before GSE282666 β access. Pull beta matrix. Run Stage 1 immune A-score. Write outcome.md.
4. **crc-epic v2.4** card update — additive only, adds `early_onset_rectal_subsection` block + commercial_deployment_unaffected_by_validation_limitations block (per Heath sign-off the latter is universal cookbook addition); does NOT modify any existing v2.3 evidence or claim.
5. **GitHub push** of all three VAL packages + crc-epic v2.4 card update + Biological_Physics/README.md update with VAL-098/099/100 rows.
6. **Heath-only delivery** of updated 5 reference docs (README_MASTER, LESSONS_LEARNED, TESTING_CHECKLIST, EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2, crc-epic_README) following the canonical 7-files-update-every-card workflow per memory #11.

Combined effort: 3 VALs × per-card-workflow protocol = approximately 1 long working session per VAL plus the card update; ~3-4 sessions total.

---

## 7. Phase 1 sign-off check for Heath

Three things to confirm before I move to Phase 2:

1. **Subsection-in-crc-epic confirmed.** The atlas vault check returned zero rectum-distinct cell types across every reference matrix; rectum-vs-colon disambiguation is not biologically achievable at v1. Subsection within crc-epic is the honest scope. **(Heath confirmed 2026-04-28.)**

2. **The Hispanic-stratification framing is "EDEAR is not population-specific; the clinical action pathway IS population-aware."** This aligns with what the data supports without overclaiming and without underclaiming. The biobank-gated cohorts that would actually answer the Hispanic-specific biology question are logged in `future_when_support_arrives.md` per LL-PUBLIC-TIER. Confirm this framing is what you want — or redirect.

3. **The three-VAL slate (VAL-098 TCGA-READ paired, VAL-099 TCGA-COAD age-stratified re-analysis, VAL-100 GSE282666 under-50 polyp Stage 1) is the Phase 2 commit.** Each VAL is within-cohort, each is Tier 1 public, each uses canonical existing methodology (no new atlas, no new panel). Confirm or redirect on the slate.

If 1 + 2 + 3 are confirmed, I draft VAL-098 prereg next turn.

---

**End of Phase 1 cohort survey.** Tier 1 public data inventory complete. No biobank applications proposed. No outreach proposed. No preprint-first proposed. Hispanic-specific biology cohorts logged as future-when-support-arrives. EDEAR commercial deployment unaffected.
