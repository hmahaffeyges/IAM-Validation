# psp-epic v0.1 — Progressive Supranuclear Palsy / Corticobasal Degeneration

**Tier:** `exploratory_pending_replication`
**Created:** 2026-04-26
**Stub card.** Captures the replicable PSP-specific signal that surfaced under run-everything architecture (CCL-033) without yet meeting `single_cohort_validated` criteria. Documented here so the signal is not buried inside ad-immune where it does not structurally fit.

---

## What this card captures

Three independent VAL studies on the same Munich GIFT GSE53740 cohort produce a consistent below-normal signature on cortical-neuron-discriminating CpGs that distinguishes PSP from AD, FTD, and HC:

| VAL | Method | PSP vs HC d | p | Direction |
|---|---|---|---|---|
| **VAL-057** | Stage 1 directional 7-CpG Rule A panel | PSP/CBD preserved 5/7 frozen directions vs 4/7 on AD samples in the same cohort | descriptive at small-n | preservation count > AD |
| **VAL-091** | Stage 2 cortical-neuron *fraction* via Loyfer-atlas NNLS deconvolution | **−0.51** | within-cohort | PSP/CBD reads BELOW HC |
| **VAL-092** | Stage 2 per-class A_terminal at top-100 cortical-neuron-discriminating CpGs vs H_min(terminal)=0.7728 | **−0.433** [−0.747, −0.098] | 0.010 | BELOW_NORMAL |

The signal is **PSP-specific, not generic tauopathy** — FTD vs HC d = +0.19 (VAL-091) and d = −0.004 (VAL-092) confirm FTD reads at HC baseline, ruling out a "any tauopathy" mechanism.

The signal has **the opposite sign from AD** — AD reads at HC baseline on cortical-neuron pathways (VAL-091 fraction d=−0.026 to −0.083 within-cohort, VAL-092 per-CpG drift d=−0.030 within-cohort AddNeuroMed). PSP is therefore not "AD plus more" or "AD-adjacent." Different signature, opposite direction.

---

## Working hypothesis (pending replication)

PSP's tau pathology produces architectural homogenization at cortical-neuron-discriminating positions detectable in peripheral plasma cfDNA at array resolution. The homogenization-not-elevation direction parallels:

- VAL-047 Phase 6 Deep Audit secretory-class variance reduction at >10yr breast pre-dx (d = −1.226)
- heme-epic v0.1 SUPPRESSED tier (post-chemo, post-transplant, immunocompromised, primary immunodeficiency)

**Three independent below-normal-as-signal cases now documented across the cookbook.** Below-normal is a category of mechanism, not a one-off curiosity. Run-everything architecture (CCL-033) is what surfaces these patterns; elevation-gated pipelines hide them.

---

## Honest weaknesses at v0.1

1. **Single-cohort evidence.** Only GIFT GSE53740, n=43 PSP / 1 CBD / 128 FTD / 193 HC.
2. **No pre-diagnostic data.** Detection at-diagnosis only.
3. **No cross-platform replication.** GIFT is 450K. EPIC PSP cohort needed for cross-platform confirmation.
4. **PSP-vs-AD-vs-FTD-vs-HC three-way clinical decision boundary not yet calibrated.** The card does not yet specify what tier-set thresholds clinicians use to call PSP vs AD vs FTD on a single IDAT. Future v0.2 task.
5. **Stage 1 Xu-538 pooled A_immune behavior on PSP not separately characterized.** PSP's Stage 1 immune A-score in GIFT is at HC baseline (which is why the PSP signal would not have been computed under elevation-gated Stage 2). Whether PSP shows a directional Stage 1 signature analogous to AD's Rule A panel is open — not yet tested.
6. **Cohort-batch-effect-flagged data.** The cohort had Ferrari 2014 ComBat preprocessing producing +2.306 SD HC offset vs the 80-cell baseline (VAL-057 noted this). The within-cohort case-vs-control comparison is fully valid (both arms suffer the same offset, the comparison is internally consistent). The cross-cohort A-score absolute comparison from this cohort to other cohorts is NOT interpretable until a second PSP cohort lands. CCL-034 within-cohort vs cross-cohort hierarchy applies.
7. **The card is a stub, not a full card.** No commercial.web.py decision tree yet, no full Stage 1/2/3 routing logic, no patient-facing report template. v0.2+ tasks.

---

## Why a stub card and not a full card

Under run-everything architecture, the PSP signal is computed for every IDAT regardless of disease-of-interest, so PSP can be detected as a Stage 2 anomaly pattern (cortical-neuron tile reads BELOW_NORMAL) without a fully-built card. The stub captures the replicable signal at the right tier so it (1) does not get buried inside ad-immune where it does not structurally fit, (2) anchors the priority replication cohort list, (3) demonstrates the run-everything architecture surfaces below-normal patterns. When a replication cohort lands, the stub gets built out into a full card.

---

## Priority replication cohorts

Promotion to `single_cohort_validated` requires at least one of these to replicate the BELOW_NORMAL signal at d ≤ −0.3 within-cohort on cortical-neuron Stage 2 tile.

| Cohort | Source | Access tier | Expected n |
|---|---|---|---|
| **PROGRESS-PSP biobank** | Boxer/Cure-PSP consortium | Tier 3 (biobank application) | ~150 PSP, multi-site |
| **Allen et al. Mayo PSP cohort** | Allen 2018 Mayo Clinic | Tier 1 (public, 450K) | n~40 per arm |
| **Tang 2014 PSP/MSA blood methylation** | Tang 2014 | Tier 1 (GEO/IDAT availability TBD) | n=68 PSP |

If at least one of these replicates within-cohort BELOW_NORMAL at d ≤ −0.3, promote to `single_cohort_validated` and build out the card to full v0.2 structure.

---

## What the card delivers at v0.1

- A documented BELOW_NORMAL pattern on cortical-neuron-discriminating CpGs.
- Replicable across two metrics on the same cohort (fraction VAL-091 + per-CpG drift VAL-092).
- Distinct from AD (null on same tile) and FTD (null on same tile) — confirms PSP-specific not generic tauopathy.
- Explicit replication-cohort priority list.
- Explicit `exploratory_pending_replication` tier label, not inflated to `single_cohort_validated`.

---

## Card-internal lessons

### psp-LL-001 — PSP signal would not have been computed under elevation-gated Stage 2 architecture

GIFT GSE53740 reads Stage 1 immune A-score at HC baseline for PSP samples — there is no Stage 1 elevation to trigger Stage 2 under conditional-gating architecture. The PSP-specific BELOW_NORMAL signal on the cortical-neuron Stage 2 tile (VAL-091 fraction d=−0.51, VAL-092 per-CpG drift d=−0.43) only became visible after the run-everything architecture sign-off (CCL-033) because run-everything computes Stage 2 every IDAT regardless of Stage 1 status. **Diseases whose primary signal is in the negative direction on the disease-of-interest tile cannot be discovered under elevation-gated pipelines.** PSP joins heme-epic SUPPRESSED and breast-epic >10yr secretory-class variance reduction as the third documented case.

### psp-LL-002 — PSP is not "AD plus more" — opposite direction on Stage 2 cortical-neuron tile vs AD's null

AD reads Stage 2 cortical-neuron at HC baseline on both fraction (VAL-091 AIBL d=−0.026, AddNeuroMed d=−0.083 within-cohort) and per-CpG drift (VAL-092 AddNeuroMed d=−0.030 within-cohort). PSP reads BELOW HC on both metrics (VAL-091 fraction d=−0.51, VAL-092 per-CpG drift d=−0.43). FTD reads at HC baseline on both metrics. **Three signatures, distinct, with PSP opposite-direction from the AD baseline-null on the same tile.** A same-card-as-AD framing for PSP would obscure both diseases. Cards group by disease-mechanism signature, not by disease-class adjacency. AD and PSP are both tauopathies but their methylation signatures on the same panel/tile are different in direction.

### psp-LL-003 — Single-cohort with cross-platform-batch-effect flagged is sufficient for `exploratory_pending_replication` tier when within-cohort effect replicates across two independent metrics

The PSP signal is currently single-cohort (GIFT GSE53740 only) and the cohort has a +2.306 SD HC offset vs the 80-cell baseline due to Ferrari 2014 ComBat preprocessing. Under CCL-034 cross-cohort baseline rules, the cross-cohort A-score absolute comparison from this cohort to other cohorts is not interpretable. However, within-cohort PSP-vs-HC comparison is fully valid (both arms suffer the same offset, the comparison is internally consistent), and the within-cohort effect replicates across two metrics: VAL-091 fraction and VAL-092 per-CpG drift, on the same cohort, run by independent VAL studies on different days with different code paths.

**Within-cohort replication across two independent metrics on a single cohort with cross-cohort baseline issues is sufficient to capture a finding at the `exploratory_pending_replication` tier.** It is NOT sufficient for `single_cohort_validated` — that requires either (a) clean cross-cohort baseline + primary outcome on case-vs-control, or (b) multi-cohort within-cohort replication. The right tier captures the signal at the right confidence level.

---

## Validation summary at v0.1

- **VAL-057** (2026-04-23, ad-immune): Specificity arm on GIFT GSE53740 — PSP/CBD preserved 5/7 frozen Rule A panel directions, AD preserved 4/7. Descriptive cohort-level finding at small AD n.
- **VAL-091** (2026-04-26, ad-immune v2.2): Stage 2 cortical-neuron fraction. PSP/CBD vs HC d = −0.51 within-cohort.
- **VAL-092** (2026-04-26, run-everything first demonstration): Stage 2 per-class A_terminal on cortical-neuron-discriminating CpGs vs H_min(terminal)=0.7728. PSP vs HC d = −0.433 [−0.747, −0.098] p = 0.010 within-cohort. Pre-registered (SHA `7249e964afbf…`) sealed before any β access.

---

## Cross-references

- Master pipeline architecture: `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` Part 12 (VAL-092 first demonstration of run-everything).
- Run-everything architecture absolute rule: README_MASTER §"ABSOLUTE RULE — Run-everything pipeline architecture (CCL-033)".
- Cross-cohort baseline mandatory rule: TESTING_CHECKLIST.md CHK-3.2 + LESSONS_LEARNED.md CCL-034.
- AD differential: ad-immune v2.2 Stage 2 differential-diagnosis tile (Stage 1 immune positive AND Stage 2 cortical-neuron > 0.5% triggers DIFFERENTIAL_DIAGNOSIS_REQUIRED, consistent with glioma not AD; PSP separated by direction).
- Below-normal-as-signal category: heme-epic SUPPRESSED tier; breast-epic VAL-047 Phase 6 >10yr secretory variance reduction; this card.

**End of psp-epic v0.1 stub. Promotion criteria: replication on at least one of PROGRESS-PSP / Mayo Allen / Tang 2014 cohorts at within-cohort d ≤ −0.3.**
