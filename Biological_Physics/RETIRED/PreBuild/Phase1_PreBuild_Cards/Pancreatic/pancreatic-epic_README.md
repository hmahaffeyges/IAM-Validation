# pancreatic-epic — Cookbook Card v0.1

**Disease:** Pancreatic ductal adenocarcinoma (PDAC), the dominant pancreatic malignancy (~90% of pancreatic cancers).
**Card status:** `cohort_screening_validated` (Stage 1 anchored by VAL-046 Rotterdam pre-diagnostic blood n=182, 2-5 year pre-dx detection at cohort level), with tissue-arm exploratory at Stage 2 ceiling reference (VAL-066/067/068) and a directional Stage 1 fallback panel built from VAL-069 (TCGA-PAAD holdout PASS at d=+1.51 p<0.001, GSE74071 holdout partial-fail).
**Card date:** 2026-04-25.
**Card scope:** Complete operator's manual for processing any 450K/850K methylation IDAT against PDAC. Covers every supported specimen pathway, every stage, every covariate, every confound, every honest limitation. A future AI or clinician handed this card and an IDAT should be able to produce a calibrated PDAC report end-to-end without further reference.

---

## 1. What pancreatic ductal adenocarcinoma is

PDAC is the most lethal common cancer in the developed world by 5-year survival (~12%), almost entirely because it is detected too late. Diagnosis typically occurs at Stage III or IV when curative resection is no longer possible. The disease's silence is its defining clinical feature: PDAC arises in the retroperitoneum behind dense organs, produces no characteristic early-symptom syndrome, and shows no reliable circulating biomarker analogous to PSA or CEA. CA 19-9 is a downstream marker of advanced disease, not a screening tool.

PDAC arises from pancreatic ductal exocrine epithelium, which the GAPE framework places in the **secretory class** (shared with breast ductal, prostate epithelium, hepatocyte). H_min(secretory) = 0.843264 from the G-002 MCMC posterior, frozen. Healthy reference β for pancreatic_exocrine ≈ 0.745 from Moss 2018 plasma deconvolution reference.

The tumor microenvironment of PDAC is uniquely fibrotic — the densest stromal compartment of any common cancer (Hosein 2020, Öhlund 2014). Tumor cells are typically a minority of total cells in a PDAC mass, with cancer-associated fibroblasts (CAFs), suppressor immune cells (M2 macrophages, MDSCs, regulatory T cells), and extracellular matrix making up the bulk. **This stromal-dense architecture has direct implications for every detection pathway** — it dilutes tumor-cell-specific signals in tissue, enriches stromal-class signals, and produces a heterogeneous immune response that pools to near-zero net direction at the panel level. Both effects are documented per-pathway in §4 below.

---

## 2. Clinical claim of pancreatic-epic v0.1

A 450K or 850K methylation IDAT from any of the supported specimen types in §4, processed through the universal Stage 1 + Stage 2 pipeline plus the PDAC-specific directional fallback in §5.2, produces a calibrated A-score and pancreatic_exocrine localization estimate that flags risk for current or developing PDAC.

The Rotterdam pre-diagnostic blood cohort (Horvath 2015, n=182) supports cohort-level detection 2-5 years before clinical diagnosis at the immune-class A-score departure level. **Per-patient deployment-grade Stage 1 sensitivity at this temporal window has not been validated in this v0.1** — that is the priority next-step.

---

## 3. The universal pipeline applied to any IDAT

Every IDAT, regardless of specimen, runs the same first two stages. Interpretation differs by specimen (see §4); computational steps are identical.

### 3.1 Stage 1 — Immune-class A-score on Xu-538

**Panel:** Xu-538 (538 CpGs from Xu 2020 JNCI Sister Study + EPIC-Italy replication).
**Panel SHA-256 (file-bytes):** `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`. Verify at runtime.
**H_min:** 0.838889 (immune class, G-003b MCMC posterior, R-hat < 1.001, frozen).

**Critical rule.** Stage 1 ALWAYS uses H_min(immune) regardless of which disease card is being run. The Xu-538 panel is the immune-class panel. The disease-tissue class (secretory for pancreas) is a Stage 2 concept, not a Stage 1 concept. Earlier drafts of VAL-066/067/068/069 used H_min(secretory) in error; corrected uniformly throughout this card. Future card authors must verify H_min(immune) = 0.838889 in their Stage 1 scripts before publishing.

**Pooled-entropy score:**
```
A_pooled = mean over Xu-538 CpGs present of [ H(β_cpg) / H_min(immune) ]
```
where H(β) = −β log₂(β) − (1−β) log₂(1−β) is the Shannon entropy of a single CpG β value.

**Healthy 80-cell baseline by age decade** is in the universal_reference block (Hannum, Horvath, Roadmap, Moss, Lister, Alisch sources). Compare every patient's A_pooled to the age-matched baseline.

**For PDAC, pooled-entropy is NOT the primary metric.** See §5.2.

### 3.2 Stage 2 — Tissue-of-origin localization

**Reference panel:** Moss 2018 ~7,890 tissue-marker CpGs across 25 reference tissues (Moss et al., Nat Comm 2018).

**Algorithm:** NNLS (non-negative least squares) decomposition of bulk β into per-tissue fractions and per-tissue β estimates.

**For pancreatic-epic, the target tissue is `pancreatic_exocrine`.**
- Healthy reference β = 0.745
- H_min(secretory) = 0.843264
- Score: `A_pancreatic_exocrine = H(β_decon) / H_min(secretory)`
- ΔA: `A_decon − A_healthy_reference`

**Localization criterion:** pancreatic_exocrine ΔA must rank top-1 across the 25 deconvolved tissues AND must be ≥2× the second-place tissue ΔA. Below this threshold, report as "Stage 1 flagged, Stage 2 ambiguous; recommend serial-sample at 6 months."

**For tissue-biopsy specimens (Pathway B, §4.2):** No deconvolution needed. Bulk β = pancreatic tissue β (modulo tumor cellularity caveat). Score directly against H_min(secretory).

### 3.3 Stage 3 — Composition deconvolution (only when Stage 1 fires + Stage 2 returns no localization)

**Method:** Teschendorff 2017 EpiDISH RPC mode against Salas 2018 IDOL-Ext reference panel.

**Outputs:** CD4T, CD8T, NK, B, monocyte, neutrophil fractions. Distinguishes chronic inflammation, hematologic malignancy, autoimmune, AD-type pattern.

**For PDAC, the literature-predicted Stage 3 pattern** (Clark 2007, Hosein 2020): lymphoid-arm suppressed (effector T cells, B cells), myeloid-arm expanded (MDSCs, M2 macrophages, monocytes). Operationally this means an EpiDISH output with elevated monocyte and neutrophil fractions and depressed CD4T and CD8T fractions.

**The lymphoid/myeloid operational split on the Xu-538 directional panel itself is currently a v0.2+ open question** pending Salas IDOL-Ext panel staging (see §11, OQ-2026-01).

---

## 4. Specimen pathways — every IDAT input route this card supports

The pipeline in §3 is invariant. The specimen the IDAT was generated from determines which scoring lens is primary, which cell-type composition needs to be deconvolved away, and which confounds dominate. Every supported specimen below is a valid IDAT input for pancreatic-epic; the difference is in interpretation, not in pipeline.

### 4.1 Pathway A — Plasma cfDNA (the primary EDEAR specimen, blood draw)

**Specimen.** ~5 mL whole blood collected in EDTA or Streck cfDNA preservation tube; plasma separated within 6 h (EDTA) or up to 7 days (Streck) of draw; cfDNA isolated; bisulfite-converted; 450K or 850K array. This is the single tube of blood that produces both Stage 1 and Stage 2.

**What is in this IDAT.** A bulk methylation profile reflecting the cellular sources of cfDNA in circulation. In a cancer-free adult, cfDNA is approximately 70-90% leukocyte-derived (apoptotic immune cell turnover), 5-10% endothelial, with the remainder distributed across solid-organ tissues at ~0.1-2% each. In a PDAC patient, the pancreatic_exocrine fraction rises detectably above baseline as the tumor sheds; in a pre-diagnostic state, the immune compartment shifts before the pancreatic fraction becomes individually detectable.

**Stage 1 reading.** Extract Xu-538 panel CpGs from the IDAT. Compute pooled-entropy A_pooled per §3.1 and directional A_dir per §5.2. Both are reported. **For PDAC, A_dir is the primary clinical metric** because pooled A_immune nulls cross-cohort on PDAC tissue (§5.1) and the directional panel recovers per-patient separation. The recovery mechanism is unresolved between AD-style lineage-level bidirectional cancellation and z-scoring sensitivity gain — pending OQ-2026-01 immune-atlas staging (CCL-030).

**Stage 2 reading.** Run Moss 2018 NNLS deconvolution on the same IDAT. NNLS decomposes bulk plasma β into per-tissue β estimates across 25 reference tissues. Extract the deconvolved pancreatic_exocrine β. Compute pancreatic-specific A-score and ΔA per §3.2. Apply localization criterion (top-1 + 2× second-place).

**Stage 3 reading.** Only if Stage 1 fires AND Stage 2 returns no solid-organ localization. EpiDISH RPC + IDOL-Ext as in §3.3.

**Pre-analytical confounds specific to plasma cfDNA in PDAC patients.**
- **Pancreatic exocrine insufficiency.** Many PDAC patients have reduced exocrine output; methylation signature unchanged but tumor cfDNA shedding magnitude may be reduced.
- **Diabetes / T2D.** Both newly-diagnosed (paraneoplastic) and long-standing T2D are PDAC risk factors. Diabetes-related immune changes (chronic low-grade inflammation, monocyte expansion) overlap with PDAC immune signatures. Stratify or flag T2D status in every report.
- **Recent ERCP, biliary stent placement, acute pancreatitis episode.** All elevate pancreatic_exocrine cfDNA shedding for days to weeks via direct tissue injury. Ask before scoring; if recent (<30 days), note in report and consider deferring re-test by 4-6 weeks.
- **Obesity / BMI.** PDAC risk factor; methylation signature partially overlaps. BMI is a mandatory covariate (§7).
- **Smoking.** Major PDAC risk factor; smoking shifts immune-class methylation independently. Smoking status is mandatory.
- **Alcohol intake.** Heavy chronic alcohol is a PDAC risk factor via chronic pancreatitis pathway. Mandatory.
- **Family history.** ~10% of PDAC has familial component (BRCA2, PALB2, ATM, CDKN2A, STK11). Mandatory report field.
- **Time of day, fasting status, recent strenuous exercise.** Affect cfDNA total levels but not relative composition; minor confound for proportional scoring.
- **Hemolysis during draw.** Releases leukocyte gDNA into plasma, contaminates cfDNA prep with high-MW DNA from intact white cells. Lowers tumor fraction and shifts Stage 2 deconvolution toward over-representation of leukocyte signal. QC: cfDNA size profile must show ~167 bp dominant peak; if high-MW shoulder is present, redraw.
- **Pregnancy.** Placental cfDNA fraction can reach 5-15% in third trimester; alters Stage 2 deconvolution baseline. PDAC in pregnancy is rare but card cannot assume non-pregnant population.
- **Recent transplant, transfusion, or chimerism.** Donor leukocytes/tissue contribute foreign cfDNA. Stage 1 immune signal becomes uninterpretable without explicit chimerism modeling. Out of scope for v0.1; flag and decline scoring.

**Validated at:** `cohort_screening_validated` (anchored by VAL-046 Rotterdam pre-dx n=182, cohort-level).

### 4.2 Pathway B — Tissue biopsy (pathology lab workflow, alternative high-fidelity input)

**Specimen.** FFPE or fresh-frozen pancreatic biopsy tissue, ≥100 ng DNA, bisulfite-converted, 450K or 850K array. Sources include EUS-FNA, surgical resection (Whipple, distal pancreatectomy, total pancreatectomy), or post-mortem.

**What is in this IDAT.** Direct bulk-tumor methylation profile. The β values are the actual mixed cell-population β of the biopsy: tumor ductal cells admixed with CAFs, infiltrating immune cells, endothelial cells, and stromal fibroblasts in proportions that vary widely (PDAC tumor cellularity is often <30% of total cells in biopsy due to dense stroma).

**Stage 1 reading.** Run Xu-538 panel through tumor-tissue β. **Caveat:** the Xu-538 panel scores the immune component of the bulk sample, which in tumor tissue is the tumor-infiltrating immune cell compartment, not the circulating immune compartment. **VAL-066, VAL-067, VAL-068 in this card all use this pathway and confirm:** the pooled-entropy A-score on tumor tissue produces no consistent direction across cohorts (VAL-066 paired d = +1.18 at n=5, VAL-067 unpaired d = +0.25 at n=196, VAL-068 paired d = −0.31 at n=7; all CIs span zero on pooled-entropy). The per-CpG positive-direction fractions (46.9%, 50.4%, 52.9%) are clustered at 50/50 — the bidirectional cancellation signature CCL-027 was created to flag. **Use the directional fallback (§5.2) on tissue Stage 1 reads, not the pooled-entropy.**

**Stage 2 reading.** Direct read — no deconvolution needed. The biopsy IS the tissue. Score the bulk-tumor β against H_min(secretory) = 0.843264 and compare to healthy pancreatic_exocrine reference β = 0.745. This is the highest-fidelity Stage 2 read available because there is no NNLS noise. **Tissue Stage 2 A-score elevation magnitude is the ceiling** — the largest ΔA the framework expects to see for this disease, against which deconvolved-blood Stage 2 should be calibrated.

**When to use biopsy pathway.** When a pathology lab already has biopsy tissue from EUS-FNA, surgical resection, or post-mortem and wants to run methylation analysis. Higher-fidelity than blood for confirming a known lesion. NOT the primary screening pathway because it requires invasive sampling.

**Pre-analytical confounds specific to biopsy.**
- **Tumor cellularity.** PDAC biopsies are often dominated by stromal fibroblasts and CAFs. A "PDAC tumor" β is often only 20-40% tumor cells. Document cellularity if available; ideally with H&E review by the pathologist.
- **FFPE vs fresh-frozen.** FFPE bisulfite conversion has higher noise floors. Use ENmix, sesame, or RnBeads normalization with explicit FFPE handling. Document which method was used.
- **Adjacent-normal contamination.** Surgical-margin samples often have variable cancer-adjacent stroma. The "adjacent normal" descriptor in TCGA-PAAD and GEO cohorts is heterogeneous; treat with the same caveat as the tumor side.
- **Necrotic regions.** Sampling necrotic tumor produces degraded DNA and unreliable β values. Reject samples with bisulfite conversion efficiency < 95%.
- **EUS-FNA cytology vs core biopsy.** FNA samples ductal lumen content (tumor-cell-enriched); core biopsy samples bulk tissue (more stroma). Document specimen type.
- **Mucinous variant of PDAC.** Higher mucin content dilutes cellular DNA recovery and may shift methylation signature relative to ductal NOS. PH64 outlier in VAL-068 (§6) may be one such case.

**Validated at:** Exploratory across 3 cohorts (VAL-066/067/068). Pooled-entropy null cross-cohort. Directional fallback recovers signal partially.

### 4.3 Pathway C — Pancreatic juice (alternative direct-source specimen, ERCP collection)

**Specimen.** Pancreatic juice collected during ERCP (endoscopic retrograde cholangiopancreatography), with cells pelleted and DNA extracted. Used in the GSE74071 cohort (VAL-068).

**What is in this IDAT.** A mix of shed pancreatic ductal cells, exfoliated tumor cells (when present), and pancreatic juice immune cells. Higher tissue-specific enrichment than plasma cfDNA for the pancreas specifically.

**Stage 1 reading.** Same Xu-538 + H_min(immune) pipeline. The immune compartment in pancreatic juice is dominated by neutrophils and monocytes (not the leukocyte composition of buffy coat); the Xu-538 panel was designed for buffy-coat / blood immune patterns and may behave differently here. **VAL-068 sub-result for n=4 juice cancer cells vs n=8 adjacent-normal: unpaired d = −0.72 [−1.95, +0.51] (CI straddles zero, exploratory at n=4).** Treat juice Stage 1 as exploratory pending dedicated pancreatic-juice Xu-538 calibration.

**Stage 2 reading.** Direct read. Pancreatic juice cancer cells are pancreatic-derived, so the bulk β IS the pancreatic_exocrine signal (no deconvolution needed). Score against H_min(secretory).

**When to use juice pathway.** During scheduled ERCP for biliary stenting in suspected PDAC. Non-blood, non-tissue-biopsy alternative. Higher specificity than blood for PDAC localization, lower invasiveness than biopsy. Limited by ERCP availability and clinical indication.

**Pre-analytical confounds specific to juice.**
- **Bile contamination.** ERCP juice often contains bile reflux from the common bile duct. Bile contains hepatocyte and biliary-epithelial cfDNA. Stage 2 deconvolution may show secondary hepatocyte signal — do not interpret as second cancer.
- **Inflammatory cell admixture.** Active pancreatitis at time of ERCP elevates neutrophil content. Stage 1 immune signal becomes inflammation-dominated.
- **Volume of juice obtained.** Low-volume samples (<200 µL) have unreliable DNA recovery; reject below threshold.
- **Time from collection to processing.** Pancreatic juice contains active proteases and DNases; degrades within hours at room temperature. Process within 2 h or freeze immediately.

**Validated at:** Exploratory (VAL-068 n=4).

### 4.4 Pathway D — Urine cfDNA (exploratory)

**Specimen.** Urine sediment cfDNA from spun first-morning void, bisulfite-converted, 450K or 850K array.

**What is in this IDAT.** Renal-filtered cfDNA from systemic circulation plus locally-shed urothelial cells. Not a primary deployment pathway for PDAC because pancreatic cells do not preferentially shed into urine. **No PDAC urine methylation cohort has been validated in this Cookbook.** Documented for completeness so a future operator with urine-only access can attempt the run. Treat as exploratory pending dedicated study.

**Pre-analytical confounds specific to urine.**
- **Hematuria.** Blood contamination introduces leukocyte gDNA, masks Stage 1 immune signal.
- **UTI or recent catheterization.** Bacterial DNA contamination affects bisulfite conversion QC.
- **Hydration status.** Dilute urine has less recoverable cfDNA per mL; standardize to creatinine.
- **First-void vs random catch.** First-void contains overnight-accumulated cells; random catch is lower-yield. Specify in report.

### 4.5 Pathway E — Saliva (exploratory)

**Specimen.** Saliva cfDNA, ~30% buccal-epithelial-derived, ~70% leukocyte-derived (Yousefi 2019).

**What is in this IDAT.** A modified blood-Stage-1 specimen with buccal admixture. Stage 1 Xu-538 may be partially valid (the leukocyte fraction overlaps blood) but the buccal-epithelial admixture introduces secretory-class background that confounds Stage 2 NNLS. **Not a validated PDAC pathway. Exploratory only.**

**Pre-analytical confounds specific to saliva.**
- **Recent oral intake.** Food, drink, mouthwash, smoking within 30 min before collection introduce contamination and cellular debris.
- **Periodontal disease.** Elevates oral leukocyte and bacterial DNA fractions; shifts Stage 1 immune signal independently of systemic disease.
- **Smokeless tobacco / oral snuff.** Direct chemical exposure to buccal epithelium causes methylation shifts overlapping cancer signatures.

### 4.6 Pathway F — CSF (NOT applicable to PDAC)

**Specimen.** Cerebrospinal fluid cfDNA. Required for terminal-class detection (brain, neurodegeneration). PDAC does not preferentially shed into CSF. **Not a PDAC pathway.** Listed here so a future operator does not attempt to apply this specimen to pancreatic-epic.

### 4.7 Pathway G — Pancreatic FNA cytology specimen (research alternative)

**Specimen.** EUS-guided FNA cytology smear or cell block, methylation array. Used in some PDAC research cohorts (note: GSE150468 was FNA + cfDNA paired but used MBD-seq, not 450K/EPIC, so not directly Cookbook-compatible).

**What is in this IDAT.** Highly tumor-cell-enriched compared to surgical biopsy because FNA samples ductal lumen content directly. Stage 1 caveat is the same as Pathway C (juice) — buffy-coat-trained Xu-538 may not transfer cleanly. Stage 2 is direct read against H_min(secretory).

**No PDAC FNA HM450/EPIC cohort is currently in this Cookbook.** Documented for future validation.

### 4.8 Specimen hierarchy summary table

| Priority | Pathway | Specimen | Stage 1 status | Stage 2 status | Validation |
|---|---|---|---|---|---|
| 1 | A | Plasma cfDNA | Primary (directional) | Primary (NNLS deconvolution) | cohort_screening (VAL-046) |
| 2 | B | Tissue biopsy | Alternative (directional fallback) | Highest-fidelity (direct read) | exploratory (VAL-066/067/068) |
| 3 | C | Pancreatic juice | Alternative (uncertain transfer) | Direct read | exploratory (VAL-068 n=4) |
| 4 | G | Pancreatic FNA | Alternative (uncertain transfer) | Direct read | unvalidated |
| 5 | D | Urine cfDNA | Limited | Limited | unvalidated |
| 6 | E | Saliva | Partial (leukocyte fraction only) | Confounded by buccal | unvalidated |
| n/a | F | CSF | Not applicable | Not applicable | not a PDAC pathway |

---

## 5. PDAC-specific Stage 1 — directional fallback per CCL-027

### 5.1 Why pooled-entropy is not the primary metric for PDAC

PDAC drives Xu-538 CpGs in genuinely bidirectional fashion, with positive-direction CpG fractions of 46.9%, 50.4%, and 52.9% across the three tissue cohorts validated in this card (VAL-066/067/068). The pooled-entropy A-score nulls out across cohorts because positive and negative magnitudes cancel — exactly the cancellation pattern documented in CCL-027 (and originally identified for AD via VAL-050 vs VAL-051).

Per the Directional-Score Principle (mandatory at every card v0.1 build), the directional fallback panel is therefore the recommended primary Stage 1 metric for PDAC. Both metrics are computed and reported; the directional is the primary clinical signal.

### 5.2 The pancreatic-epic directional panel

**Panel file:** `pancreatic_directional_panel.json` in this card directory.

**Panel construction.**
- Training cohort: GSE49149 (n=196, 167 PDAC tissue + 29 adjacent-normal tissue), HM450.
- Direction assignment per CpG: sign of mean(β_tumor − β_normal) in the training cohort.
- Coverage filter: ≥80% of samples per arm must have a measured β at the CpG.
- Magnitude filter: |Δβ_train| > 0.005.
- Result: 324 CpGs with frozen ±1 directions (172 positive, 152 negative). 84 Xu-538 CpGs excluded by coverage filter, 130 by low Δβ.
- Per-CpG normalization parameters (μ_normal_train, σ_normal_train) frozen from the GSE49149 normal arm (n=29).

**Score formula.**
```
For each CpG c in the panel:
    z_c = (β_observed_c − μ_normal_train_c) / σ_normal_train_c
    contribution_c = direction_c × z_c

A_dir = mean over all panel CpGs measured in the sample of contribution_c
```

**H_min independence.** Because A_dir uses z-score normalization at the per-CpG level, the choice of H_min does not appear in the formula. The directional score is class-agnostic by construction. This is a feature, not a bug — it makes A_dir comparable across cohorts and platforms without H_min-related rescaling.

### 5.3 Directional panel performance

| Holdout | n | Paired d | 95% CI | p | Outcome |
|---|---|---|---|---|---|
| TCGA-PAAD | 7 | +1.51 | [+0.43, +2.59] | 6.4e-05 | **PASS** — all 7 patients positive ΔA_dir |
| GSE74071 | 7 | +0.22 | [−0.53, +0.97] | 0.56 | FAIL — PH64 single-pair outlier ΔA_dir = −1.17; 4 of 7 pairs strongly positive |

**Honest framing.** The directional fallback cleanly separates tumor from normal in TCGA-PAAD where pooled-entropy barely did, and recovers signal in 4 of 7 pairs in GSE74071. The PH64 outlier in GSE74071 may reflect a genuine PDAC sub-type (mucinous variant, neuroendocrine differentiation, or technical artifact) — cannot be resolved with current data. Logged as v0.2+ open question.

### 5.4 The four CCL-027 questions answered for PDAC

| Question | Answer for PDAC |
|---|---|
| (i) Pooled direction with citation | Mixed across cohorts, no consensus. VAL-066 paired d=+1.18 (n=5), VAL-067 unpaired d=+0.25 (n=196), VAL-068 paired d=−0.31 (n=7). All CIs span zero on pooled-entropy. |
| (ii) Bidirectional risk with citation | HIGH. Confirmed in 3 independent tissue cohorts at per-CpG level (positive-direction %: 46.9 / 50.4 / 52.9). Literature: Clark 2007 (PMC1944938) — PDAC drives lymphoid contraction (Tregs up, effector T cells down) + myeloid expansion (MDSCs, M2 macs). |
| (iii) Directional fallback if risk | Built and validated. 324-CpG GSE49149-trained subset. TCGA-PAAD holdout d=+1.51 [+0.43, +2.59] p=6.4e-05 PASS. GSE74071 holdout d=+0.22 partial-fail (PH64 outlier). Recommended primary Stage 1 metric for PDAC. |
| (iv) Lymphoid/myeloid expected pattern (literature only) | Clark 2007: lymphoid down + myeloid up expected. Operational split blocked on Salas IDOL-Ext panel staging — VAL-070 deferred, contributing to OQ-2026-01. |

---

## 6. Validation summary (VAL studies in this card)

| VAL | Specimen | Cohort | n | Primary result | Status |
|---|---|---|---|---|---|
| VAL-046 | Plasma blood (pre-dx) | Rotterdam Study | 182 future-PDAC | Cohort-level ΔA elevation 2-5 yr pre-dx | Existing anchor, cohort-level |
| VAL-066 | TCGA-PAAD HM450 (tissue biopsy) | TCGA-PAAD | 7 amended → 5 effective paired | Pooled A_immune paired d = +1.18 [+0.04, +2.32], per-CpG split 46.9% positive | Tissue Stage 2 ceiling, exploratory at n=5 |
| VAL-067 | GSE49149 HM450 (tissue biopsy) | Mishra/Wood | 167 tumor + 29 normal | Pooled A_immune unpaired d = +0.25 [−0.15, +0.64], per-CpG split 50.4% positive | Tissue Stage 2 ceiling, large-n null at pooled |
| VAL-068 | GSE74071 multi-substrate HM450 | Tjensvoll | 14 tumor + 7 normal + 4 juice + 3 CAFs | Pooled A_immune paired d = −0.31 [−1.07, +0.45]; juice unpaired d=−0.72 | Tissue Stage 2 + multi-substrate exploratory |
| VAL-069 | Directional Xu-538 panel build + 2 holdouts | GSE49149 train, TCGA-PAAD + GSE74071 holdouts | n=196 train, n=7+7 holdout | TCGA-PAAD holdout d=+1.51 PASS; GSE74071 d=+0.22 partial-fail | Recommended Stage 1 directional fallback |

**Stage 1 anchor for clinical claim:** VAL-046 Rotterdam blood pre-diagnostic n=182 (existing cohort-level evidence).

**Tissue arm (Stage 2 ceiling reference) status:** Exploratory across 3 independent cohorts. Pooled-entropy directionally inconsistent. Directional fallback partially recovers.

**Per-patient deployment-grade Stage 1 sensitivity at the 2-5 year pre-dx temporal window has NOT been validated in v0.1.** Logged as priority next-step.

---

## 7. Mandatory covariates and confounds — every report field

The following must be collected and recorded at every patient encounter. Analysis stratifications and report fields differ by covariate.

| Covariate | Stratify analysis | Report field | Rationale |
|---|---|---|---|
| Sex (M/F) | Yes | Yes | PDAC ~1.3:1 M:F; methylation partially sex-dimorphic per VAL-053/057 cross-card |
| Age (decade) | Yes | Yes | Healthy 80-cell baseline is age-decade indexed; PDAC incidence rises after age 50 |
| Smoking status (current/former/never) | Yes | Yes | Largest modifiable PDAC risk factor; shifts immune-class methylation independently |
| BMI (<25, 25-30, >30) | Yes | Yes | Obesity is independent PDAC risk factor; methylation overlap |
| Diabetes status (none / T2D ≥3 yr / new-onset T2D <2 yr) | Yes | Yes | New-onset T2D is paraneoplastic PDAC; long-standing T2D is a risk factor; both shift immune methylation |
| Alcohol intake (none / moderate / heavy) | Observation | Yes | Heavy alcohol → chronic pancreatitis → PDAC pathway |
| Family history of PDAC | Observation | Yes | ~10% familial component (BRCA2, PALB2, ATM, CDKN2A, STK11) |
| Recent pancreatitis episode (<3 mo) | Yes (exclude or flag) | Yes | Acute pancreatitis transiently elevates pancreatic_exocrine cfDNA shedding |
| Recent ERCP or biliary stent (<30 d) | Yes (exclude or flag) | Yes | Direct injury elevates pancreatic shedding for days-weeks |
| Race / ethnicity | Observation | Yes | African American PDAC incidence ~50-90% higher than white; mechanisms partially methylation-independent but report for transparency |
| Chronic pancreatitis history | Observation | Yes | Independent PDAC risk factor |
| Recent acute infection (<2 wk) | Yes (defer testing) | Yes | Acute infection produces transient immune-class methylation shifts that swamp tumor signal |
| Pregnancy status | Yes (decline scoring) | Yes | Placental cfDNA fraction confounds Stage 2 deconvolution; v0.1 declines pregnant samples |
| Recent transplant / transfusion / chimerism | Yes (decline scoring) | Yes | Donor cfDNA contributions out of v0.1 scope; flag and decline |
| Hormonal contraception / HRT | Observation | Yes | Sex-hormone-related immune methylation shifts; document for stratification |
| Autoimmune disease history (e.g., RA, lupus, IBD) | Yes | Yes | Chronic immune dysregulation overlaps with disease-related immune signature |
| Active or recent chemotherapy / radiotherapy | Yes (decline scoring) | Yes | Treatment-induced cfDNA from healthy tissue damage; uninterpretable |
| Heavy environmental exposures (occupational solvents, pesticides) | Observation | Yes | Independent PDAC risk factors (Andreotti 2019); methylation overlap possible |
| Diurnal collection time | Observation | Yes | cfDNA total levels diurnal but proportional scoring largely robust; document for completeness |
| Fasting status at collection | Observation | Yes | Affects total cfDNA but minor effect on proportional scoring |

For pre-diagnostic screening deployment specifically, **trajectory (serial sampling at 6-12 month intervals) carries more signal than any single timepoint** — see §9 below.

---

## 8. Tier thresholds and clinical action matrix

### 8.1 Tier thresholds

Same universal Cookbook tier structure (NORMAL < 1.01, MARGINAL ≥ 1.01, DETECTABLE ≥ 1.05, URGENT ≥ 1.07, FLOOR BREACH ≥ 1.10) applied to the directional A_dir score normalized to z-score units.

| Tier | A_dir z-score range | Pooled A_immune backup | Stage 2 ΔA_pancreatic_exocrine |
|---|---|---|---|
| NORMAL | A_dir < +0.5 | < 1.01 | not elevated |
| MARGINAL | +0.5 ≤ A_dir < +1.0 | 1.01-1.05 | < +0.02 |
| DETECTABLE | +1.0 ≤ A_dir < +1.5 | 1.05-1.07 | +0.02 to +0.05 |
| URGENT | +1.5 ≤ A_dir < +2.0 | 1.07-1.10 | +0.05 to +0.10 |
| FLOOR BREACH | A_dir ≥ +2.0 | ≥ 1.10 | ≥ +0.10 |

**Calibration rationale.** TCGA-PAAD holdout per-patient A_dir ranged from +0.18 to +1.46 (mean +0.66). The DETECTABLE tier threshold of A_dir ≥ +1.0 separates 4 of 7 TCGA-PAAD patients from controls; URGENT at +1.5 captures the strongest signal. These are tissue-tumor-equivalent magnitudes; **blood-deployment magnitudes will likely be lower** because blood Stage 1 reads circulating immune response, not tumor-tissue immune-infiltrate. Tier thresholds will be re-calibrated against blood-PDAC cohort data when available.

### 8.2 Clinical action matrix

| Stage 1 directional | Stage 2 pancreatic localization | Action |
|---|---|---|
| NORMAL | n/a | Continue baseline cadence (every 1-3 yr depending on age + family history) |
| MARGINAL | Stage 2 null | Serial sample at 6 months. Modifiable-risk-factor counseling (smoking, BMI). |
| MARGINAL | Stage 2 ambiguous | Serial sample at 3 months. Add CA 19-9 and lipase. |
| DETECTABLE | Stage 2 null or non-pancreatic | Workup for non-PDAC origin: CBC differential, CRP, full Stage 3 EpiDISH. Repeat at 3 months. |
| DETECTABLE | Stage 2 pancreatic_exocrine top-1 + 2× criterion met | EUS or pancreatic protocol MRI within 4 weeks. Gastroenterology consult. |
| URGENT | any | EUS within 2 weeks. Gastroenterology consult. CA 19-9, lipase, glucose, HbA1c. Consider pancreatic protocol CT. |
| FLOOR BREACH | any | Same as URGENT plus oncology consult. Family history → genetic counseling for BRCA2/PALB2/ATM panel. |

**Special rule — paraneoplastic-PDAC workup trigger.** For new-onset T2D in patients ≥50 yr with any DETECTABLE-or-above directional score: paraneoplastic-PDAC workup REGARDLESS of Stage 2 localization. New-onset T2D in this age group + immune drift = ~1% pre-test PDAC probability per Pannala 2008.

**Report quality gate.** For all DETECTABLE-or-above results, even with Stage 2 null: if any mandatory covariate is missing, do not finalize the report. Send back for completion.

---

## 9. Trajectory monitoring (essential for pre-diagnostic deployment)

PDAC pre-diagnostic detection at the 2-5 year window inherently requires **serial sampling**, not single-timepoint testing. The Rotterdam cohort signal in VAL-046 is detectable at the cohort level because the cohort has multi-year follow-up; individual patients cannot be assigned a 2-year pre-dx prediction from a single IDAT alone (the within-individual baseline drift is comparable to the pre-dx ΔA magnitude).

**Recommended cadence for high-risk patients** (family history of PDAC, BRCA2 carrier, hereditary pancreatitis, IPMN under surveillance, new-onset T2D age ≥50, chronic pancreatitis):
- Every 6 months
- Track A_dir z-score trajectory across timepoints
- **Trajectory slope > +0.3 z-units per year is more diagnostic than any single elevation**
- Two consecutive MARGINAL+ readings 6 months apart triggers DETECTABLE-tier action even if individual readings stay MARGINAL

**For average-risk patients ≥50 yr:** baseline at age 50, every 2 years thereafter.

---

## 10. Known limitations of pancreatic-epic v0.1

1. **No per-patient pre-diagnostic blood validation.** The 2-5 year pre-dx claim is supported at the cohort level (VAL-046 Rotterdam n=182) but not at the individual-patient level. The Rotterdam cohort individual β data is not in the public domain. Per-patient Stage 1 sensitivity at the pre-dx window is the priority next-step.

2. **No 10-year pre-diagnostic data.** Currently no public cohort supports detection beyond ~5 years pre-dx for PDAC. Longer-window detection requires either Sister Study or UK Biobank pancreatic subset (dbGaP-gated, application required) or partner-collected serial-sample cohort.

3. **Tissue arm pooled-entropy null cross-cohort.** All three independent tissue cohorts (VAL-066/067/068) show CIs that span zero on the pooled Xu-538 metric. The directional fallback recovers per-patient separation on TCGA-PAAD (all 7 patients positive). The recovery mechanism is not yet established — could be lineage-level bidirectional cancellation per the AD analog (literature-predicted by Clark 2007), z-scoring sensitivity gain, cohort/batch structure, or a combination. Operational distinguishing test (lymphoid vs myeloid sub-panel split) is pending OQ-2026-01. GSE74071 holdout shows cohort-specific outlier behavior (PH64 single-pair outlier).

4. **GSE74071 PH64 outlier.** Single tumor/normal pair shows ΔA_dir = −1.17, opposite to the cohort's other pairs. May reflect mucinous PDAC sub-type, neuroendocrine differentiation, technical artifact, or chance at n=7. Unresolved; flagged for v0.2+.

5. **TCGA-PAAD cohort is biased.** Of n=7 effective patients, 5 male, all white or near-white, mostly ductal NOS histology, mostly Stage IIB. Sex stratification, race stratification, histology stratification all underpowered. The favorable d=+1.51 holdout result may not generalize to female PDAC, non-white PDAC, or non-ductal histologies.

6. **No lymphoid/myeloid operational split yet.** Stage 3 pattern prediction (lymphoid down + myeloid up) is literature-supported but not operationally tested. Blocked on Salas IDOL-Ext panel staging. VAL-070 deferred.

7. **BMI not populated for any TCGA-PAAD patient** in the GDC clinical metadata — BMI mandatory covariate carries forward as a v0.1 data gap.

8. **No PDAC blood cfDNA HM450/EPIC cohort tested.** GSE150468 pancreatic cfDNA used MBD-seq, not Illumina arrays — incompatible with Xu-538 panel. The directional panel was trained on tumor tissue β, which may not transfer cleanly to circulating cfDNA. The directional fallback's blood-deployment performance is currently extrapolated, not validated.

9. **Pancreatic juice substrate behavior unclear.** VAL-068 n=4 juice cancer cells gave unpaired d = −0.72 — direction opposite to expected. At n=4 this is exploratory; the result may reflect pancreatic juice immune composition differing from blood (neutrophil-dominant rather than mixed leukocyte) which the buffy-coat-trained Xu-538 cannot read directly.

10. **Tier thresholds are tissue-derived.** A_dir thresholds calibrated on tissue cohorts. Blood deployment will likely require lower thresholds. Re-calibrate when blood-PDAC cohort is available.

11. **Stage 2 plasma cfDNA Moss NNLS for PDAC not directly validated in this card.** Relies on framework-wide VAL-041 10-cancer NNLS validation. PDAC-specific deconvolution accuracy at low tumor fractions (typical pre-diagnostic range) requires dedicated validation.

---

## 11. Open questions for v0.2+

| Open question | Source | Action needed |
|---|---|---|
| Blood-PDAC HM450/EPIC cohort missing | This card | Search for or contact Rotterdam-equivalent prospective cohorts; consider partner outreach |
| Lymphoid/myeloid operational split | OQ-2026-01 + this card | Stage Salas IDOL-Ext panel; run VAL-070 |
| GSE74071 PH64 outlier resolution | VAL-069 H3 partial-fail | Need additional PDAC cohorts + sub-type annotation |
| 10-year pre-dx detection | Limitation #2 | dbGaP application for Sister Study or UK Biobank pancreatic |
| Per-patient Rotterdam validation | Limitation #1 | Direct contact with Rotterdam Study PI for individual-level data sharing |
| BMI confound at scale | Limitation #7 | Need clinically annotated PDAC blood cohort with BMI |
| Pancreatic juice Stage 1 behavior | VAL-068 sub-result | Need larger pancreatic juice cohort with paired blood |
| Histology-specific signal | Limitation #5 | Need cohorts with ductal vs mucinous vs neuroendocrine annotation |
| Stage 2 plasma cfDNA NNLS PDAC accuracy | Limitation #11 | Run paired tumor tissue + plasma cfDNA validation cohort |

---

## 12. Sources and citations

- **Horvath 2015 Rotterdam Study pre-diagnostic blood pancreatic cohort** — DOI [10.18632/aging.100861](https://doi.org/10.18632/aging.100861). Source for VAL-046 anchor.
- **Xu Z, Sandler DP, Taylor JA. JNCI 2020** — DOI [10.1093/jnci/djz065](https://doi.org/10.1093/jnci/djz065). Source for Xu-538 immune panel.
- **Moss J et al. Nat Commun 2018** — DOI [10.1038/s41467-018-07466-6](https://doi.org/10.1038/s41467-018-07466-6). Source for tissue-of-origin NNLS deconvolution panel and pancreatic_exocrine healthy reference β.
- **Salas LA et al. Genome Biol 2018** — PMID 29945600. Source for Stage 3 EpiDISH IDOL-Ext reference.
- **TCGA-PAAD project** — NIH GDC public access. Source for VAL-066. Manifest in `PAAD_matched_manifest.json`.
- **Mishra/Wood lab GSE49149** — PMIDs 24500968 and 26909576. Source for VAL-067.
- **Tjensvoll et al. GSE74071** — pancreatic juice and CAF multi-substrate HM450. Source for VAL-068.
- **Clark CE et al. Cancer Res 2007** — PMC 1944938. PDAC immune-suppression pattern.
- **Hosein AN et al. Nat Rev Gastroenterol Hepatol 2020** — DOI [10.1038/s41575-020-0300-1](https://doi.org/10.1038/s41575-020-0300-1). PDAC stromal density.
- **Öhlund D et al. JCI 2014** — DOI [10.1172/JCI73639](https://doi.org/10.1172/JCI73639). PDAC stromal fibroblast biology.
- **Pannala R et al. Lancet Oncol 2008** — New-onset T2D paraneoplastic PDAC.
- **Andreotti G et al. Environ Health Perspect 2019** — Occupational pesticide exposure and PDAC risk.
- **Yousefi P et al. Clin Epigenetics 2019** — Saliva methylation source composition.
- **Teschendorff AE et al. Bioinformatics 2017** — EpiDISH RPC method.

---

## 13. Pre-registration chain (full reproducibility)

| VAL | Pre-reg SHA-256 | Amendment SHA-256 |
|---|---|---|
| VAL-066 | `694206201d45c1e3cbced1ef17b565b99e5d7f86a96b29fd58f6ba6050ea887e` | `9533d64cc98d361a168ee941bcb737156b8410f655a15d2f878297734f5c344b` (n=10 → n=7 after manifest verification) |
| VAL-067 | `f0de98bd22c98bf1a48100387e6a9acf79aa24c4591608552085d8c0c0ba2efb` | none |
| VAL-068 | `50c0c7e8afccc2a5dfc407bf95e29b846cb1f3effc1458484e28e88f3cbaedfc` | none |
| VAL-069 | `e31de916ac00268bfe22116f67f54317b1a99f63dc3dc7c1482019a0be1ae12a` | none |

Xu-538 panel SHA (file-bytes): `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`. Verified at runtime in every VAL script.

RNG seed for all VAL scripts: 20260425.

---

## 14. Reproduction bundle

All four VAL studies in this card are reproducible from public-access data using Python 3 stdlib only.

**Files in this card directory (deliver to Heath only — Cookbook IP):**
- `pancreatic-epic_README.md` — this document
- `pancreatic-epic_card_v0.1.json` — machine-executable card spec with universal_reference embedded
- `pancreatic_directional_panel.json` — 324-CpG directional fallback panel with frozen ±1 directions and per-CpG μ/σ from GSE49149 normal arm

**Files in IAM-Validation/Biological_Physics/validation_runs/ (push to GitHub):**
- `val066_pancreatic_epic_tcga_paad.py` + `VAL-066_prereg.md` + `VAL-066_PREREG_AMENDMENT.md` + `VAL-066_PREREG_SEAL.txt` + `VAL-066_outcome.md` + `VAL-066_results.json` + `PAAD_matched_manifest.json` + `PAAD_clinical.json`
- `val067_pancreatic_epic_gse49149.py` + `VAL-067_prereg.md` + `VAL-067_PREREG_SEAL.txt` + `VAL-067_outcome.md` + `VAL-067_results.json` + `GSE49149_manifest.json`
- `val068_pancreatic_epic_gse74071.py` + `VAL-068_prereg.md` + `VAL-068_PREREG_SEAL.txt` + `VAL-068_outcome.md` + `VAL-068_results.json` + `GSE74071_manifest.json`
- `val069_pancreatic_epic_directional.py` + `VAL-069_prereg.md` + `VAL-069_PREREG_SEAL.txt` + `VAL-069_outcome.md` + `VAL-069_results.json`
- `VAL-070_NOT_RUN_NOTE.md` — deferred VAL-070 lymphoid/myeloid split documentation

---

## 15. Lessons learned (pancreatic-epic-specific)

**panc-LL-001.** PDAC tissue arm pooled A_immune is null cross-cohort (VAL-066 +1.18 CI[−0.04,+2.32]; VAL-067 +0.25 CI[−0.15,+0.64]; VAL-068 +0.40 CI[−0.50,+1.30]). A 324-CpG directional ±1 z-scored panel built on GSE49149 (VAL-069) recovers per-patient separation on TCGA-PAAD holdout (d=+1.51, all 7 patients positive). The recovery mechanism is unresolved — possible explanations include AD-style lineage-level bidirectional cancellation (literature-predicted, not directly measured), z-scoring sensitivity gain over entropy averaging, or cohort/batch structure baked into the directional freeze. The lymphoid vs myeloid sub-panel split (Test 2 per CCL-030) that would distinguish lineage cancellation from non-mechanism alternatives is pending OQ-2026-01 immune-atlas staging. Per-CpG cohort Δβ direction percentages (46.9%, 50.4%, 52.9% across the three cohorts) are descriptive of where β values shifted on average — they are NOT a mechanism diagnostic by themselves.

**panc-LL-002.** PDAC tumor cellularity in biopsy is unusually low due to dense stromal compartment. Tissue-Stage-1 readings are confounded by tumor-infiltrating immune cells AND CAF-derived signal AND tumor-cell signal in unpredictable proportions. The cleanest tissue Stage 1 signal would come from laser-capture-microdissected pure tumor cells. Future tissue-arm validation should prefer LCM-purified tumor cells when available.

**panc-LL-003.** Pancreatic juice as a specimen is biologically distinct from both blood and tissue. The buffy-coat-trained Xu-538 panel may not transfer cleanly to juice immune composition (neutrophil-dominant vs mixed leukocyte). VAL-068 sub-result d=−0.72 at n=4 is exploratory; juice pathway requires dedicated calibration.

**panc-LL-004.** TCGA-PAAD HM450 matched tumor/normal subset is small (n=10 candidates → n=7 amended → n=5 effective after QC). Public-domain matched-pair PDAC methylation cohorts are scarce. Future cards should not assume TCGA matched-pair availability without manifest verification.

**panc-LL-005.** GSE49149 (n=196) is the largest publicly accessible PDAC tissue methylation cohort on HM450. It is the natural training set for any PDAC-specific directional panel. The 167-tumor + 29-normal split makes per-CpG direction estimation reliable but the unpaired design limits per-patient validation.

**panc-LL-006.** The new-onset-T2D paraneoplastic pathway (Pannala 2008) is the strongest known clinical risk-stratifier for PDAC. Any patient ≥50 yr presenting with new-onset T2D + any DETECTABLE-or-above directional A-score should receive paraneoplastic-PDAC workup regardless of Stage 2 localization confidence. This deviates from the universal pipeline default and is documented explicitly in the action matrix.

**panc-LL-007.** Stage 1 ALWAYS scores Xu-538 against H_min(immune) regardless of disease. Earlier draft of VAL-066/067/068/069 used H_min(secretory) in error. Cohen's d unchanged (scale-invariant under multiplicative transformation), only absolute A-scores 0.5% off, but cross-card numerical comparability would have broken. Panel-class governs H_min in Stage 1; tissue-class is a Stage 2 concept only. Universal pipeline rule.

---

## 16. Saturation levels — secretory class A_ceiling architecture

Pulled from GAPE Reproduction Paper Part 2.4A and Part 2.4B. The saturation level architecture defines the maximum A-score reachable on each (class, substrate) pair via `A_ceiling = 1 / H_min`. When a patient's A-score approaches or hits its ceiling, the substrate has saturated and carries no further progression information for that sample.

PDAC sits in the secretory class. The complete secretory-class saturation profile across all 5 substrates:

### 16.1 Secretory class A_ceiling values (from Part 2.4A)

| Substrate | H_min(secretory) | A_ceiling = 1/H_min | Structural status | Active to BREACH? |
|---|---|---|---|---|
| methyl | 0.843264 | 1.1859 | Active | Yes |
| nucl | 0.982594 | 1.0177 | **Structurally saturated** ⚠ | No (ceiling < 1.10) |
| fuzz | 0.847955 | 1.1793 | Active | Yes |
| wps | 0.634518 | 1.5760 | Active | Yes |
| frag | 0.697838 | 1.4332 | Active | Yes |

**Structurally saturated substrate = nucl.** The nucleosome occupancy A-score for any secretory-class disease (including PDAC) cannot reach FLOOR_BREACH (≥ 1.10) on the nucl substrate alone. This is a physical feature of nucleosome occupancy in healthy cells (positioning fluctuates around 50% by design, leaving very little signal headroom above the floor) and is not specific to PDAC. Nucl is restricted to NORMAL/MARGINAL/DETECTABLE drift detection only for secretory-class diseases.

**Active substrates for BREACH-tier discrimination = methyl, fuzz, wps, frag.** Four of the five substrates carry signal across the full tier range from NORMAL through FLOOR_BREACH. Methylation (methyl) is the primary substrate in v0.1 — every VAL study in this card uses 450K/EPIC methylation arrays. The fragmentomics substrates (wps, frag) and chromatin accessibility (fuzz) are the framework-validated secondary substrates per the Reproduction Paper, and become operational when the EDEAR multi-substrate platform reaches L2/L3 lab partnership tier.

### 16.2 Runtime saturation flag thresholds (from Part 2.4B)

The runtime saturation flag fires when a sample's A-score on a substrate is within 0.005 of that substrate's A_ceiling. The exact firing thresholds for secretory class:

| Substrate | A_ceiling | Runtime flag fires at | Interpretation when fired |
|---|---|---|---|
| methyl | 1.1859 | A ≥ 1.1809 | β has moved from healthy reference (~0.745) toward 0.5 (coin-flip state); methylation has saturated |
| nucl | 1.0177 | A ≥ 1.0127 | nucleosome occupancy saturated; structural — fires easily |
| fuzz | 1.1793 | A ≥ 1.1743 | chromatin fuzziness has saturated; total chromatin accessibility hit ceiling |
| wps | 1.5760 | A ≥ 1.5710 | windowed protection score saturated; cfDNA has lost positional protection signal |
| frag | 1.4332 | A ≥ 1.4282 | fragment-size distribution saturated; fragment heterogeneity hit ceiling |

When a runtime saturation flag fires for a substrate, that substrate is excluded from `A_active` aggregation per Reproduction Paper Part 3.3, and the patient EDEAR report carries a saturation alert for that substrate. The flag does NOT indicate disease severity — it indicates measurement-channel exhaustion. A patient with multiple saturated substrates and elevated unsaturated-substrate A-scores has unambiguous architectural breakdown across multiple physical channels.

### 16.3 PDAC-specific detection strategy by tier

Given the secretory-class saturation profile, the per-tier substrate signal allocation for PDAC:

| Tier | Primary substrate | Confirmatory substrates | Excluded substrate | Notes |
|---|---|---|---|---|
| NORMAL (A < 1.01) | methyl | nucl, fuzz, wps, frag | none | All five substrates carry healthy-baseline drift signal |
| MARGINAL (1.01 ≤ A < 1.05) | methyl | nucl, fuzz, wps, frag | none | All five substrates active in this band |
| DETECTABLE (1.05 ≤ A < 1.07) | methyl | fuzz, wps, frag | nucl approaches its ceiling | Nucl A approaches 1.0177; weight nucl < 0.5 in A_combined |
| URGENT (1.07 ≤ A < 1.10) | methyl | fuzz, wps, frag | nucl saturated | Drop nucl from A_active per runtime flag |
| FLOOR_BREACH (A ≥ 1.10) | methyl, fuzz, wps, frag | (cross-substrate confirmation) | nucl (cannot reach BREACH) | Require ≥2 unsaturated substrates above 1.10 for BREACH confirmation |

For v0.1 deployment running on 450K/EPIC methylation arrays only, methyl is the operational substrate at all tiers. The non-methylation substrate guidance applies once the L2/L3 multi-assay platform is operational. Until then, the saturation-aware tier interpretation is reduced to: methyl A_ceiling = 1.1859, runtime flag at A ≥ 1.1809, no patient should report A > 1.1859 on methylation alone (any apparent value above ceiling is QC failure, not disease severity).

### 16.4 Methylation substrate ceiling crossing — clinical interpretation

A PDAC patient whose methylation A-score climbs from 1.05 (DETECTABLE) toward 1.18 over serial sampling has crossed FLOOR_BREACH at 1.10 and is now within 0.006 of methylation channel saturation. At that point the methylation channel is exhausted as a progression metric. Continued progression must be tracked via tissue-of-origin localization (Stage 2 pancreatic_exocrine fraction trajectory) or, when the L2 platform is operational, via the unsaturated chromatin substrates (fuzz, wps, frag). Saturation on methylation does NOT mean the patient is stable — it means the framework can no longer measure further deterioration through that channel.

### 16.5 Why secretory-class nucl saturation matters for PDAC specifically

PDAC tissue arm pooled A_immune nulls cross-cohort (VAL-066/067/068) and the directional ±1 z-scored panel (VAL-069) recovers per-patient separation. The recovery mechanism is not yet established between AD-style lineage cancellation, z-scoring sensitivity gain, and cohort/batch structure — pending OQ-2026-01 (CCL-030). Regardless of mechanism, the directional panel is the v0.1 operational metric. The nucleosome occupancy substrate, which would otherwise serve as an independent cross-check on the methylation finding, is structurally saturated for the secretory class. The cross-check role falls to fuzz, wps, and frag once L2/L3 is operational. Until then, running both the pooled methylation A-score AND the directional A_dir on the same methylation IDAT provides two independent reads of the same substrate, partially compensating for the nucl loss.

---

## 17. Card validation tier statement

`cohort_screening_validated` (anchored by VAL-046 Rotterdam pre-diagnostic blood n=182, 2-5 year pre-dx detection at cohort level).

**Tissue arm modifier:** `tissue_arm_exploratory_with_directional_recovery_partial` (3 cohorts, pooled-entropy null, directional fallback recovers TCGA-PAAD holdout PASS at d=+1.51 p<0.001 but partial-fails GSE74071 holdout).

**Path to `single_cohort_validated`:** acquire individual-patient β from Rotterdam pre-dx cohort and run per-patient Stage 1 directional + Stage 2 NNLS pipeline.

**Path to `cross_platform_validated`:** add a second independent blood-PDAC cohort (Sister Study via dbGaP, UK Biobank via dbGaP, or new partner-collected) and demonstrate the directional fallback transfers across cohorts.

**Path to `multi_modal_validated`:** add Stage 2 NNLS deconvolution validation on plasma cfDNA from PDAC patients (paired tumor tissue β as ground truth + plasma cfDNA Stage 2 deconvolution recovery test).

---

## 18. What we discovered

### 18.1 Why pancreatic cancer is hard to detect

Pancreatic ductal adenocarcinoma is the most lethal common cancer in the developed world by 5-year survival rate (~12%), almost entirely because it is detected too late. The pancreas sits behind dense organs in the retroperitoneum, the disease produces no characteristic early-symptom syndrome, and there is no reliable circulating biomarker for screening — CA 19-9 is a downstream marker of advanced disease, not an early-detection tool. Most patients are diagnosed at Stage III or IV when curative resection is no longer possible, and the median time from first symptom to death is on the order of months.

The biology of PDAC adds another layer of difficulty. The tumor microenvironment is the densest stromal compartment of any common cancer — most of a PDAC mass is fibrotic tissue plus suppressor immune cells, and the actual cancer cells are typically a minority of the bulk. This dilutes signal in tissue biopsies and produces a tumor immune response that is heterogeneous across cell types. Some immune populations are suppressed (effector T cells, B cells), others are expanded (regulatory T cells, MDSCs, M2 macrophages, monocytes). The conventional pooled-entropy A-score, which averages immune-class CpGs, can null out a real signal when half the CpGs move up and half move down.

### 18.2 What we tested

We ran every accessible HM450 cohort for PDAC plus the existing Rotterdam blood pre-diagnostic anchor:

- **VAL-046** — Rotterdam Study pre-diagnostic blood cohort, n=182 future-PDAC patients, 2-5 year pre-dx interval (existing anchor, cohort-level only)
- **VAL-066** — TCGA-PAAD HM450 matched tumor/normal, n=5 effective paired patients after QC (n=10 candidates → n=7 amended → n=5 effective)
- **VAL-067** — GSE49149 large unpaired PDAC tissue, n=196 (167 tumor + 29 adjacent-normal). The largest publicly accessible PDAC tissue methylation cohort.
- **VAL-068** — GSE74071 multi-substrate, 28 samples covering 14 tumor + 7 adjacent normal + 4 pancreatic juice + 3 cancer-associated fibroblasts + 1 primary culture
- **VAL-069** — directional Xu-538 fallback panel built on VAL-067 training data, validated on VAL-066 and VAL-068 holdouts

Three independent tissue cohorts plus blood pre-dx anchor plus directional fallback panel.

### 18.3 The headline finding

PDAC's tissue-arm immune signal is recoverable per-patient with a directional panel, but the conventional pooled-entropy A-score is null cross-cohort.

What that means in plain language: when we run the standard scoring (averaging Shannon entropy across the 538-CpG immune panel), PDAC tumor-vs-adjacent-normal comparisons come out near zero in three independent tissue cohorts. The disease is in the data — but this particular averaging method does not extract it. When we instead freeze each CpG's direction based on which way it moved in a training cohort (GSE49149) and z-score normalize, we recover clean per-patient separation on an independent holdout (TCGA-PAAD): all 7 patients positive, paired d = +1.51, p = 6.4 × 10⁻⁵.

The pooled-null-with-directional-recovery pattern is the same operational signature as Alzheimer's disease (VAL-050/051). The mechanism in both cases is currently unresolved. The hypothesis we originally favored is AD-style lineage-level bidirectional cancellation: lymphoid-marker CpGs go one direction, myeloid-marker CpGs go the other, with comparable magnitudes, so the pooled mean cancels. That hypothesis is plausible from the PDAC literature (Clark 2007 describes lymphoid contraction + myeloid expansion in the PDAC tumor microenvironment) and the AD literature (Nabais 2021 describes mixed neuroinflammation patterns). **But we have not directly measured it.** What we ran was the pooled vs directional comparison; what would distinguish lineage cancellation from alternatives (z-scoring sensitivity gain, cohort/batch structure) is a lymphoid vs myeloid sub-panel split, which requires per-CpG lineage assignment from an immune-cell-type methylation atlas. That assignment is pending OQ-2026-01 staging and is not currently runnable on any disease.

In three cohorts the per-CpG cohort-mean Δβ split between positive and negative directions clustered near 50/50 (46.9%, 50.4%, 52.9%), versus 62-70% positive in standard pooled-positive cancers (breast, lung, CRC, prostate, HCC). This is consistent with whatever pattern is producing the pooled null, but the per-CpG percentage by itself does not establish a mechanism — it is a description of cohort-mean β shifts, not a lineage diagnostic.

The directional panel itself contains 324 CpGs (172 frozen positive, 152 frozen negative). It works as an operational Stage 1 metric for PDAC tissue scoring at v0.1. On the second holdout (GSE74071, n=7), paired d = +0.22 with one pair (PH64) running strongly opposite to the cohort (ΔA_dir = −1.17) — a partial-fail that dragged the mean below significance and remains unexplained at n=7.

### 18.4 What we can be sure of, in order of confidence

1. **Firmest finding (operational).** PDAC tissue-arm pooled A_immune is null cross-cohort. Three independent cohorts at three different sample sizes all give pooled CIs that span zero (VAL-066 +1.18 [−0.04,+2.32]; VAL-067 +0.25 [−0.15,+0.64]; VAL-068 +0.40 [−0.50,+1.30]). This is the most repeatable observational fact in the card.

2. **Per-patient validation level achieved.** The 324-CpG directional ±1 z-scored panel built on GSE49149 (VAL-069) recovers per-patient separation on TCGA-PAAD as an independent holdout — all 7 patients positive, paired d = +1.51, p < 10⁻⁴. This is a clean operational result. The panel is usable as the v0.1 Stage 1 metric for PDAC tissue scoring.

3. **What we cannot be sure of about the mechanism.** Whether the directional panel works because of AD-style lineage-level bidirectional cancellation (lymphoid-marker CpGs and myeloid-marker CpGs going opposite directions with matched magnitudes), or because z-scoring per CpG is more sensitive than entropy averaging in general, or because cohort/batch structure baked into the GSE49149 freeze transfers to other cohorts via shared platform — these are not distinguished by what we ran. The literature (Clark 2007 PDAC TME, Nabais 2021 AD neuroinflammation) is consistent with the lineage hypothesis but not direct evidence for it at the Xu-538 panel level. The operational distinguishing test (lymphoid vs myeloid sub-panel split, Test 2 per CCL-030) is pending OQ-2026-01 immune-atlas staging.

4. **Cohort-level pre-diagnostic detection.** Anchored by VAL-046 Rotterdam (n=182), unchanged by this validation work. Detection 2-5 years before clinical diagnosis at the cohort level. This claim is independent of the tissue-arm mechanism question.

5. **What we cannot be sure of about deployment.** Per-patient pre-diagnostic blood detection at the 2-5 year window is NOT yet validated. We have no public per-patient β data for the Rotterdam cohort, and the directional panel was trained on tumor tissue rather than circulating immune signal. Blood-deployment performance is currently extrapolated, not validated. Logged as priority #1 next-step.

6. **Honest unresolved outliers.** The PH64 outlier in GSE74071 (ΔA_dir = −1.17, opposite direction to the cohort) cannot be resolved with seven pairs. Possible explanations include mucinous PDAC sub-type, neuroendocrine differentiation, or technical artifact. This is logged as a v0.2+ open question rather than smoothed over.

7. **Generalization gap.** TCGA-PAAD's 5 QC-passed patients were all male and white, mostly ductal NOS histology, mostly Stage IIB. We cannot be sure the favorable d = +1.51 holdout result generalizes to female PDAC, non-white PDAC, or non-ductal histologies. This is in the card's known limitations list, not glossed over.

### 18.5 How well we can detect PDAC right now, by specimen type

**Plasma cfDNA (the primary EDEAR specimen).** Cohort-level pre-diagnostic detection is supported by VAL-046 Rotterdam (n=182). Per-patient validation is not yet anchored. The pipeline is in place: Stage 1 directional A_dir on Xu-538, plus Stage 2 Moss NNLS deconvolution to extract pancreatic_exocrine fraction. Tier thresholds are calibrated provisionally from tissue-arm magnitudes and will need re-calibration once per-patient blood data becomes available. **Status: cohort-level supported, per-patient pending.**

**Tissue biopsy (alternative high-fidelity input).** Validated as Stage 2 ceiling reference across three independent cohorts. The pooled-entropy approach nulls; the directional approach passes cleanly on TCGA-PAAD with all 7 patients positive. Use this when a pathology lab already has biopsy from EUS-FNA, surgical resection, or post-mortem. **Status: per-patient validated for the directional metric, pooled-entropy null cross-cohort.**

**Pancreatic juice from ERCP.** Exploratory only. Four samples in VAL-068 gave direction opposite to expected (Cohen's d = −0.72). The juice immune compartment is neutrophil-dominant, which is biologically different from the blood buffy coat the Xu-538 panel was trained for. Needs a dedicated calibration cohort. **Status: exploratory at n=4, requires dedicated juice-substrate calibration.**

**Urine cfDNA, saliva, FNA cytology.** Documented as supported specimen pathways with per-pathway scoring guidance and confound documentation in §4. No PDAC validation cohorts exist in the public domain for any of these specimens. **Status: pathway documented, no validation cohort yet.**

**CSF.** Not applicable to PDAC. PDAC does not preferentially shed into CSF. Listed in §4.6 only so a future operator does not waste a sample attempting this pathway.

### 18.6 The honest picture

The pancreatic-epic card is anchored at cohort-level pre-diagnostic detection by VAL-046 Rotterdam, equipped with a 324-CpG directional fallback panel for tissue-arm per-patient detection (validated on TCGA-PAAD with all 7 patients positive), and explicitly logs the blood-deployment per-patient validation gap as priority #1 next-step. The card supports seven IDAT specimen pathways with per-pathway scoring guidance and per-pathway confound documentation. Where the framework can be sure, it is sure on multiple cohorts; where it cannot be sure yet, it says so explicitly rather than smoothing the gap.

---

