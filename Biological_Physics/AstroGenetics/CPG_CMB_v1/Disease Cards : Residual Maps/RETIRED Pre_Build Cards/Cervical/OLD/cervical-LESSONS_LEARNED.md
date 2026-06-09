# cervical-epic — Lessons Learned (build session 2026-04-25)

These are the lessons specific to the cervical-epic v0.1 build. They feed into the master Cookbook `LESSONS_LEARNED.md` and the per-card pre-VAL `TESTING_CHECKLIST.md`. Heath's primary correction in this session was: **biology common sense IS a check on the math**. Numbers that contradict well-characterized biology require diagnostic investigation BEFORE conclusions. Walther repeatedly defaulted to "the data says X, therefore X" and Heath repeatedly had to redirect to "the data probably has a problem, find it."

---

## cerv-LL-008 — Landscape survey errors must be caught at the landscape stage, not at runtime

VAL-075 was originally planned as HPV-stratified cervical cancer using GSE38266. At runtime sample-title inspection, the cohort turned out to be HNSCC (head and neck squamous cell carcinoma — HPV-driven oropharyngeal cancer), NOT cervical. Walther only caught this when the parsed Sample_title strings showed "HNSCC" prefix. If the script had skipped sample-title validation and gone straight to scoring, an entire HNSCC cohort would have been published as cervical-epic validation evidence. **Required check: every landscape-survey entry must have at least one Sample_title verified against the survey claim BEFORE the cohort is run.**

## cerv-LL-009 — Supplementary β files are NOT necessarily β values

VAL-077 (GSE287994 Bowden 2025) supplementary file `GSE287994_ewas_betas_2.txt.gz` (1.7 GB) was NOT raw β. It was batch-corrected/residualized M-values. The "_2" suffix in the filename should have been a red flag — published EWAS supplementary tables are nearly always model output (residuals, corrected M, normalized values), NOT raw arrays. The smoking gun in the data: mean β across 538 panel CpGs ≈ 0.5, which is biologically incompatible with raw methylation data (real β is bimodal at 0.05/0.95, never flat at 0.5). **Required check: before running ANY scoring on a supplementary β file, verify the β distribution shape matches a known-good raw β reference (bimodal, ~30%+ at extremes <0.1 or >0.9, <10% in [0.4, 0.6]). If the distribution is flat near 0.5, the file is residuals/processed values, not raw β.**

Walther's mistake: ran the M→β conversion (β = 2^M / (1+2^M)) on residual M-values and treated the output as biology. The conversion is mathematically correct for raw M-values; it produces garbage for residuals because residual M ≈ 0 maps to β ≈ 0.5 across the panel.

## cerv-LL-010 — Healthy reference baseline shifts across cohorts are diagnostic, not invisible

VAL-073 healthy A = 0.681; VAL-074 healthy A = 0.621. Same panel, same platform (HM450), same disease. A 0.06 A-unit shift in the HEALTHY baseline between cohorts is substantial — about 1.5 SDs of either cohort's healthy SD. Walther reported the VAL-074 disease numbers without flagging this baseline mismatch. The mismatch is the most likely explanation for VAL-074's negative-direction CIN3 reading: GSE46306 normals are likely tumor-adjacent normal (cervical biopsy taken adjacent to a malignant lesion, carrying paracrine inflammatory contamination), while VAL-073 normals are population normal (women without CIN history). **Required check: every cohort run must report mean A and SD of the HEALTHY/CONTROL group. The first comparison after any new cohort is healthy-vs-healthy across cohorts (ANCHOR vs NEW). If the ANCHOR healthy mean A and the NEW healthy mean A differ by >1 SD, the cohorts have a baseline mismatch and the disease-vs-control numbers cannot be directly compared.**

## cerv-LL-011 — LBC is not buffy-coat. Specimen mixture matters more than platform

Xu-538 was selected from buffy-coat training data (Xu et al. 2020 Sister Study, blood samples). LBC samples are ~80% exfoliated cervical epithelium + ~10-20% mucosal-resident lymphocytes + variable mucus and inflammatory infiltrate. The cell mixture is fundamentally different from blood. Walther treated VAL-076's flat-across-CIN-grade reading as a "framework null finding" without recognizing that the panel may not transfer between cell mixtures. **Required check: any new specimen pathway (LBC, urine, saliva, anything not already validated for the panel) must have an explicit "panel transferability not yet established" caveat in the prereg before scoring. A null reading on a new specimen pathway is a transferability open question, not a framework finding.**

This is also the reason cervical-LBC-specific panels (FAM19A4/miR124-2, ZNF671, EPB41L3) exist as published clinical assays — those panels were specifically trained on cervical LBC samples. Xu-538 is NOT one of those panels. The cervical screening literature has been telling us this for a decade and Walther failed to weight it appropriately.

## cerv-LL-012 — Saturation flag check is mandatory before ANY null-finding outcome

Block 7 saturation architecture (Reproduction Paper Part 2.4B) has runtime flags at A_ceiling − 0.005. For the immune class: ceiling = 1.1921, flag fires at A ≥ 1.1871. VAL-077 mean A was 1.011 — under flag, but at 84.8% of ceiling. Walther did NOT run the saturation check before drafting outcomes. The ACTUAL relevant question for VAL-077 wasn't saturation (it wasn't saturated); it was data-format integrity (the file wasn't raw β). But a real saturation case would have been missed by the same omission. **Required check: every cohort run must include a saturation-flag report in the results JSON, even when the answer is "no saturation". Block 7 exists for exactly this scenario.**

## cerv-LL-013 — Per-CpG cohort-mean Δβ direction percentage is descriptive only (CCL-030 reaffirmed)

VAL-072 TCGA-CESC at n=3 paired produced a 47.9% per-CpG positive Δβ that initially LOOKED like bidirectional cancellation. VAL-073 at n=68 produced 37.3% — a 10-point swing at the same disease, just with proper sample size. This is noise, not biology. CCL-030 explicitly: per-CpG percentage is descriptive, not mechanism. Walther initially conflated the VAL-072 result with the AD bidirectional-cancellation pattern and Heath had to write CCL-030 to lock the terminology. **Required check: per-CpG Δβ direction percentages are NEVER cited as evidence of bidirectional cancellation. They are descriptive cohort-mean statistics. Bidirectional cancellation requires Test 1 pooled-null + Test 2 directional-pass (CCL-031), and Test 2 is currently blocked on OQ-2026-01.**

## cerv-LL-014 — Common sense biology is the first check, not the last

The cervical immunology literature is one of the most well-characterized in oncology: HPV-driven inflammation, T-cell infiltration, MHC-I downregulation by E7, Treg expansion, MDSC accumulation, M2-polarized TAMs. The qPCR FAM19A4/miR124-2 literature shows methylation detection in LBC up to 8 years pre-AIS/ADC clinical diagnosis. The PAX1/NREP-AS1 panel (Bowden 2025) achieved AUC 0.92 on the SAME GSE287994 cohort where Walther's Xu-538 scoring nulled. **The framework's null was a panel-transferability result, NOT a cervical-immune-signal result.** Heath had to halt the build to point this out: "common sense tells you it's not correct." Walther's mistake was to use the data as evidence against the biology, when the biology should have been used as evidence against the data. **Required check: before publishing any null-finding outcome, ask "is this consistent with the published clinical-grade panels for this disease?" If clinical-grade panels exist showing strong signal on the same cohort and the framework reads null, the framework's panel does not transfer — that is the finding, not "the disease has no signal."**

## cerv-LL-015 — Compaction amnesia is a pattern that causes repeat mistakes

Heath's exact words: "Every time you compact the chat you forget all this stuff and keep doing it." This is not a metaphor — it is a structural failure mode. Walther in this session repeated multiple errors that had already been corrected in earlier cards (per-CpG percentage conflation, treating null findings as biology before checking measurement, not running saturation checks). The Cookbook's lessons-learned file exists, but Walther did not read it before starting the cervical build. **Required protocol change: every new card build must START by reading (a) the master `LESSONS_LEARNED.md`, (b) the master `TESTING_CHECKLIST.md`, (c) all per-card lessons from the closest analog (e.g., for cervical-epic, the closest analogs are the LBC-pathway-having cards and the immune-class cards). This is non-negotiable. If Walther starts running cohort scripts without first reading these files, Heath should halt the session.**

## cerv-LL-016 — The diagnostic order is fixed: data integrity → biology → framework

When a cohort run produces a result that contradicts well-established biology, the diagnostic order must be:

1. **Data integrity check.** Is the file what I think it is? Verify against the source paper's methods. Check distribution shape against known-good reference. Verify panel coverage isn't drifting (HM450 → EPIC 850K loses ~100 of Xu-538). Verify QC pass count matches expected. Check saturation flags. Verify sample-group assignments by spot-checking titles against published cohort metadata.

2. **Biology consistency check.** Does the result fit the published clinical-grade panels for this disease? Does the result fit the established immunology literature? Does the result fit the cohort's own published findings? If the result contradicts all three, the framework reading is the suspect, not the biology.

3. **Framework finding (last, not first).** Only after data and biology are validated can a result be claimed as a framework-relevant finding. Even then, qualify: "the framework's universal Stage 1 panel does not transfer to [new specimen pathway]" is a TRANSFERABILITY finding, not a "the disease has no signal" finding.

Walther's mistake order in this session: ran cohort → got numbers → drafted "framework finding" outcomes → Heath halted. Correct order: run cohort → check data integrity → check biology consistency → THEN draft outcome.

---

## What we got right at the v0.1 build

Two things despite the mistakes:

1. The Block 1-20 expectations (Master README §17) demanded full-breadth validation before v0.1 publish. Single-cohort tissue-arm validation (VAL-073 alone) would have looked like a clean anchor and been published prematurely. The cohort-completeness rule (CCL-029) caught the heterogeneity even though the diagnostic interpretation went sideways for a while.

2. Heath's halt-and-redirect protocol works. Walther got off the rails twice in this session (the CCL-030/031 terminology drift and the cervical-epic-conclusion overclaim). Both were caught and corrected before publication. The system depends on Heath's biology common sense as the final check; that is fine but should not be the ONLY check. The testing checklist below distributes the checks across the script-writing stage, the runtime stage, and the outcome-drafting stage so Heath does not have to catch every mistake himself.

---

## Card consequences for v0.1

cervical-epic v0.1 publishes at `draft_pending_substrate_diagnostics` tier. The findings as currently understood:

- **VAL-073 tissue anchor preserved** — Normal vs CIN3 d = +0.73 [+0.22, +1.24], monotonic. This is the only validated framework reading for cervical.
- **VAL-074 reflects cohort-baseline mismatch** — most likely tumor-adjacent normal vs population normal in GSE46306. Re-interpretation requires reading the Farkas 2013 paper to confirm normal-cohort definition.
- **VAL-076 reflects panel transferability question** — Xu-538 was buffy-coat-trained, LBC is a different cell mixture. The flat-across-CIN reading is a panel-transferability finding, not a "no signal in cervical LBC" finding.
- **VAL-077 is not interpretable** — supplementary file is residual/corrected M-values, not raw data. Re-running requires raw IDATs from `GSE287994_RAW.tar` processed through minfi/sesame. v0.2+ engineering.
- **VAL-075 excluded** — landscape error, HNSCC not cervical.
- **VAL-078, VAL-079, VAL-081 deferred** until VAL-074/076/077 diagnostic resolution complete.

The path to v0.2+ is documented in §11 of the cervical-epic README and in `TESTING_CHECKLIST.md` for next-card builds.
