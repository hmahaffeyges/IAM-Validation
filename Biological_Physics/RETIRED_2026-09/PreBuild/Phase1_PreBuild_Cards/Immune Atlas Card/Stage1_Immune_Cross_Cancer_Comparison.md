# Stage 1 immune red-flag signature: bladder vs other cancers

## Cross-cancer comparison for immune-atlas card cross-reference

**Date:** 2026-05-01
**Source:** Sealed VAL data in /mnt/project/GAPE_Evidence_Report_UPDATED.html (primary source). All numbers traced to specific VAL outcomes.
**Purpose:** Articulate the unique Stage 1 immune signature per cancer, so the immune-atlas card can use this as the reference for "which organ is in danger when EDEAR fires Stage 1 with a particular profile."

---

## The framing

The Stage 1 immune A-score is universal — every IDAT runs through the same Xu-538 pooled-entropy panel against H_min(immune) = 0.8389. But the *magnitude* of the Stage 1 signal, *what it correlates with at Stage 2 and Stage 3*, and *how it behaves at different timepoints* differ systematically across cancers. The differences are diagnostic. They tell us what the immune system is doing in response to a specific tissue's commitment trajectory, which means they also tell us where the commitment is happening when read together.

The immune-atlas card's role is to translate "what does the Stage 1 signal mean given everything else we see" into actionable interpretation. This cross-cancer comparison fills in the row of that translation table for each disease.

---

## 1. Bladder cancer — the loudest immune-class red flag in the cookbook

**At-diagnosis Stage 1 signal:** d_paired = **+1.90** on n=21 paired tumor-vs-adjacent-normal patients (TCGA-BLCA HM450K), p = 3.14×10⁻⁸ (VAL-120 sealed 2026-05-01, diagnostic-not-sealed under O4 Xu-538 panel cohort-substrate coverage gate).

**What the signal means:** Bladder cancer presents to the immune class as a 4× louder red flag than prostate (d = +0.50, VAL-058) and 2.5× louder than colorectal at-diagnosis (d = +0.72, VAL-099 cycling-class). The signal is consistent with the disease's underlying biology — bladder cancer has heavy infiltration baked into how it works, which is why BCG immunotherapy has been standard of care for non-muscle-invasive bladder cancer for decades, and why PD-L1 checkpoint inhibitors received FDA approval for advanced urothelial carcinoma before most other solid tumors.

**Combined Stage 1 + Stage 2 + Stage 3 fingerprint:**
- Stage 1 immune: very loud (d = +1.90 paired)
- Stage 2 BladderRef Epi (gene-promoter, primary cell-of-origin reader on mucosal cohort): d_paired = **−1.46** (urothelial dedifferentiation)
- Stage 2 Loyfer Bladder bulk-WGBS tile: d_paired = +1.91 (substrate-distribution-mismatch — discard for cell-of-origin per CHK-2.18)
- Stage 3 Salas IDOL: ALL 6/6 tiles fire POSITIVE at d_paired range +0.49 to +1.24 — broad multi-lineage infiltration (TIL + TAM + tumor-associated neutrophils + tertiary lymphoid structures together)
- Lymphoid-vs-myeloid lineage skew: NONE — both lineages elevated at substantial magnitude

**Bladder-unique signature for the immune-atlas card:**
*Stage 1 immune A-score very loud (d > +1.5 at the cohort-level) + Stage 2 cell-of-origin gene-promoter atlas fires strongly NEGATIVE on a mucosal tissue + Stage 3 IDOL lineup fires broad-positive across all 6 cell types with no lineage skew = bladder cancer or another mucosal-tissue cancer with heavy mixed infiltration. Distinguish from other broad-positive Stage 3 patterns by the substrate (mucosal not solid parenchyma) and by the cell-of-origin tile direction (NEGATIVE on the gene-promoter atlas, not bulk-WGBS POSITIVE).*

---

## 2. Prostate cancer — the quiet-immune-loud-cell-of-origin signature

**At-diagnosis Stage 1 signal:** Cohen's d = +0.400 [+0.176, +0.624]; paired d = **+0.497** on GSE269244 EPIC 850K AA men n=118 paired (VAL-058 sealed).

**What the signal means:** Prostate cancer presents with a moderate Stage 1 immune signal — about 1/4 the bladder magnitude — but the cell-of-origin reading at Stage 2 is very strong. The diagnostic weight in prostate is on the cell-of-origin tile (luminal dedifferentiation), not on the immune red flag.

**Combined fingerprint:**
- Stage 1 immune: moderate (d = +0.50 paired)
- Stage 2 ProstateRef LE (gene-promoter, primary): d_paired = **−1.78** (luminal dedifferentiation; structurally identical to bladder Epi pattern but on solid parenchyma)
- Stage 2 ProstateRef BE/EC/Fib/Leu/SM (microenvironment): all POSITIVE consistent with CCL-039
- Stage 3 Salas IDOL Mono: d = +0.77; multi-lineage POSITIVE (broad-positive but quieter than bladder)

**Prostate-unique signature:**
*Stage 1 immune moderate (d in 0.4 to 0.6) + Stage 2 cell-of-origin gene-promoter LE tile fires strongly NEGATIVE + Stage 3 broad-positive but with smaller magnitude than bladder = prostate adenocarcinoma. The cell-of-origin commitment is the dominant signal; the immune red flag is a corroborator, not the headline.*

---

## 3. Breast cancer — the slow-Phase-1, decade-out immune signature

**Pre-diagnosis Stage 1 signal (VAL-047 + VAL-093/094/096, GSE51057 Phase 9 + GSE51032 Phase 12 EPIC-Italy):**
- >10 years before diagnosis: Stage 1 immune d = **+1.78** (Phase 9) — among the strongest pre-clinical Stage 1 signals in the cookbook
- 5-10 years before diagnosis: Stage 1 immune attenuates
- 2-5 years before diagnosis: continues attenuating
- 0-2 years before diagnosis: Stage 1 immune narrows; three immune-class tiles SIGN-FLIP to negative (monocytes EPIC d = +0.33 → −0.35 in GSE51057 and +0.00 → −0.40 in GSE51032; neutrophils drift toward negative; erythrocyte progenitors drop from d = +0.83/+0.48 to −0.14/−0.08)

**At-diagnosis Stage 1 signal (VAL-060, TCGA-BRCA HM450 paired n=86):** Cohen's d = +0.7453; paired d on tumor vs adjacent-normal d = +0.6755.

**What the signal means:** Breast cancer presents with the most extended Phase 1 window of any cancer in the cookbook. The Stage 1 immune signal is detectable a decade before clinical diagnosis at high magnitude (d = +1.78), then *narrows* and partially *inverts* as the trajectory approaches Phase 2 commitment. This is the inverse of what most clinicians would expect — that the Stage 1 signal would grow as the cancer approaches. The multi-organ distributed signature of Phase 1 quiets and partly inverts as the secretory tissue approaches its H_min floor and crystallizes; by the time the tumor exists, the systemic alarm has subsided.

**Concurrent Phase 1 multi-organ distributed signature (VAL-093 / VAL-096 sealed):**
- Pancreatic beta cells: d = +1.02 / +0.94 at >10yr (across Phase 9 and Phase 12 cohorts)
- Pancreatic acinar cells: d = +0.91 / +1.02
- Pancreatic duct cells: d = +0.99 / +0.70
- Kidney: d = +0.73 / +0.90
- Head/neck-larynx: d = +0.75 / +0.81
- Colon epithelial: d = +0.72 / +0.65
- Breast tile (the actual cancer-tile): d = +0.20 / +0.10 — near-null at >10yr

**Late-localizing breast tile signal (Phase 2 commitment approaching):**
- >10yr: d = +0.20 / +0.10
- 5-10yr: d = +0.05 / +0.19
- 2-5yr: d = +0.14 / +0.16
- **0-2yr: d = +0.43 / +0.49** — this is the breast tile finally rising as commitment approaches

**Breast-unique signature:**
*Stage 1 immune very loud at years-out timepoint + concurrent multi-organ distributed elevation (pancreas, kidney, colon, head/neck all d > +0.7) + breast tile near-null at years-out = breast Phase 1, 10+ years before clinical commitment. As trajectory approaches diagnosis, Stage 1 immune NARROWS and several immune sub-tiles SIGN-FLIP to negative; concurrently the breast tile rises monotonically through the windows. The narrowing of Stage 1 immune is not a sign of disease resolution — it is a sign of Phase 2 transition approaching. Serial-trajectory monitoring distinguishes the two.*

---

## 4. Colorectal cancer — at-diagnosis cycling-class commitment

**At-diagnosis Stage 1 signal (VAL-099, TCGA-COAD HM450 paired n=26):** paired d = **+0.539** for age 50+ stratum; pooled paired cycling-class d = +0.7241; sub-site Cecum d = +1.094, Colon NOS d = +1.702.

**What the signal means:** Colorectal cancer presents with moderate Stage 1 immune at diagnosis but a very strong cycling-class A-score signature (d = +0.72 pooled, with strong sub-site stratification). The cell-of-origin Colon_epithelial_cells tile reads paired d = −1.55 to −1.60 at diagnosis. Three independent paired cohorts (VAL-098 TCGA-READ, VAL-062 TCGA-COAD revisit, VAL-099 TCGA-COAD reproduction) confirm the pattern.

**Combined fingerprint:**
- Stage 1 immune: moderate (d ≈ +0.54 paired age-50+)
- Stage 2 cycling-class (full HM450): paired d = +0.72 — strong commitment in the cycling architecture class
- Stage 2 Loyfer Colon_epithelial_cells (bulk-WGBS, primary on mucosal cohort under DISC-BLADDER-003 retroactive flag): d_paired = −1.55 to −1.60 — cell-of-origin commitment confirmed in the same direction as bladder/prostate gene-promoter atlas readings

**Colorectal-unique signature:**
*Multi-organ Phase 1 signature in years-out cohorts (pancreas, liver, immune class drifting in concert) → at-diagnosis: Stage 1 immune moderate + cycling-class architecture loud + Colon_epithelial cell-of-origin tile fires strongly NEGATIVE. The cycling class signature is the architectural correlate; the cell-of-origin tile is the tissue commitment fingerprint. The Phase 1 compression relative to breast may reflect the higher H_min floor of cycling tissues (0.8561) compared to secretory (0.8433) — less reserve to consume, faster transition to commitment.*

---

## 5. AML — the special case where the immune class IS the cancer

**At-diagnosis Stage 1 signal (VAL-082, GSE62298 vs GSE51057 Italian healthy):** Cohen's d = **+3.71** [+3.23, +4.20], p < 1×10⁻⁵⁰. **The strongest single-cohort effect size measured in the cookbook to date.** 98.5% of AML samples score above the Italian healthy 95th percentile; 91.2% above the 99th percentile.

**What the signal means:** AML is the case where the framework's two-phase architecture compresses to a single direct-detection signal because the encoding surface that has saturated *is* the encoding surface the screening test reads. The Xu-538 panel was trained on whole-blood buffy coat (~50-75% neutrophils); AML cancer cells *are* the malignant myeloid lineage cells in the readout compartment. There is no Phase 1 → Phase 2 distinction in the same sense for AML — the cancer is the immune compartment.

**AML-unique signature:**
*Stage 1 immune extremely loud (d > +3) + Stage 2 cell-of-origin generally null (because the cell of origin is in the immune compartment, not a solid tissue) + Stage 3 immune lineage signature consistent with myeloid-lineage malignancy = hematologic malignancy in the readout compartment. The single-shot direct detection model works here in a way it does not work for solid-tumor cancers.*

---

## 6. Glioma — the terminal-class compression with unexpected plasma reach

**At-diagnosis Stage 1 signal (VAL-088, GSE180683 EPIC peripheral blood):** Cohen's d = **+0.91** [+0.61, +1.22] vs Italian healthy buffy coat. Pre-surgery treatment-naive subset (n=37) d = +0.94. **Pre-surgery LGG (n=12) d = +1.25 LARGER than pre-surgery GBM (n=25) d = +0.80** — the lower-grade tumor produces a louder Stage 1 immune signal than the higher-grade tumor.

**Concurrent Phase 2 fingerprint (VAL-090, plasma cfDNA via Loyfer/Moss array atlas):** Cortical-neuron cfDNA fraction = 1.092% glioma vs 0.276% healthy reference. Cohen's d = **+1.96** [+1.62, +2.31]. 89% of glioma samples cross the 0.5% threshold; 63% cross 1%; healthy reference only 7% cross 1%. **Two full standard deviations of separation.**

**What the signal means:** Glioma is the terminal-class case (cortical neurons, H_min = 0.7728 — the lowest H_min of any architecture class). The terminal-class architecture has the smallest informational reserve and the slowest replenishment. Phase 2 commitment in this class produces detectable cell-of-origin cfDNA in plasma despite the blood-brain barrier and despite the 4% cfDNA detection floor concern — because the right reference atlas (Loyfer/Moss with sorted-cell Cortical_neurons reference) reads brain-derived cfDNA fractions in the 1-2% range cleanly. **The LGG-louder-than-GBM ordering is consistent with the framework's prediction:** lower-grade tumors retain more architectural distinctness, so their cell-of-origin signature is read more cleanly; higher-grade tumors are already farther into post-commitment dedifferentiation.

**Glioma-unique signature:**
*Stage 1 immune moderate-to-loud (d ~ +0.9) + Stage 2 cortical-neuron cfDNA fraction ABOVE 0.5% in plasma + LGG > GBM ordering on both Stage 1 and Stage 2 = brain-tumor-class commitment. The terminal-class signature is the operational Phase 2 fingerprint here. Standard cancer screening misses glioma because there's no obvious blood biomarker; EDEAR catches it because the right Stage 2 atlas reads brain-derived cfDNA at array resolution.*

---

## Cross-cancer summary table — the Stage 1 immune signature differential

| Cancer | Stage 1 immune d | Stage 2 cell-of-origin tile | Stage 3 lineage pattern | Phase 1 timeline |
|---|---|---|---|---|
| **AML** | **+3.71 unpaired** | n/a (cancer is in readout compartment) | Myeloid lineage direct detection | Single-shot (no Phase 1 distinction) |
| **Bladder** | **+1.90 paired (at-dx)** | BladderRef Epi NEGATIVE −1.46 | All 6/6 IDOL POSITIVE, broad multi-lineage | At-diagnosis only sealed; Phase 1 pending |
| **Glioma** | **+0.91 unpaired (LGG +1.25, GBM +0.80)** | Cortical-neuron cfDNA d = +1.96 | n/a (terminal class) | Terminal-class compression |
| **Breast** | **+1.78 (>10yr) → +0.27 (0-2yr) → +0.75 (at-dx)** | Late-localizing breast tile +0.43/+0.49 at 0-2yr; concurrent immune-tile sign-flips | Pending Stage 3 multi-atlas | **Decade-scale; Stage 1 immune NARROWS as Phase 2 approaches** |
| **Colorectal** | +0.54 paired age-50+ (at-dx) | Loyfer Colon_epithelial NEGATIVE −1.60 | Pending Stage 3 multi-atlas | Compressed Phase 1; cycling-class |
| **Prostate** | +0.50 paired (at-dx) | ProstateRef LE NEGATIVE −1.78 | Mono +0.77, multi-lineage POSITIVE | At-diagnosis only sealed |

---

## Operational rule for the immune-atlas card

**When EDEAR fires Stage 1 immune at high magnitude, the immune-atlas card uses the following decision tree to interpret which organ is in danger:**

1. **Substrate is whole blood AND Stage 1 d > +3.0 AND Stage 3 myeloid-dominant?** → AML or other hematologic malignancy in the readout compartment.
2. **Substrate is plasma/cfDNA AND Stage 1 d > +0.8 AND Stage 2 cortical-neuron cfDNA > 0.5%?** → Glioma (terminal-class commitment with plasma reach). LGG > GBM ordering confirms.
3. **Substrate is tissue AND Stage 1 d > +1.5 AND Stage 2 gene-promoter cell-of-origin tile fires NEGATIVE on a mucosal-tissue atlas (BladderRef Epi or future LungRef/ColonRef Epi)?** → Mucosal-tissue cancer with broad multi-lineage Stage 3 infiltration. Bladder is the canonical case.
4. **Substrate is tissue AND Stage 1 d in +0.5 to +1.2 AND Stage 2 gene-promoter cell-of-origin tile fires NEGATIVE on a solid-parenchyma atlas (ProstateRef LE, BreastRef equivalent)?** → Solid-parenchyma cancer with localized commitment. Prostate is the canonical case.
5. **Substrate is whole blood AND Stage 1 d > +1.5 (years-out cohort) AND multi-organ Stage 2 fractional drift across pancreas/kidney/colon/head-neck (no single tile dominates) AND target-tissue-tile near-null?** → Breast Phase 1 signature (or analogous secretory-class Phase 1). Tissue of origin not yet identifiable by tile profile alone — needs serial-trajectory data to track which tile rises monotonically as Phase 2 approaches.
6. **Substrate is whole blood AND Stage 1 d in +0.3 to +0.7 AND multi-organ Stage 2 fractional drift?** → Phase 1 distributed signature. Tissue of origin not yet identifiable. Serial-trajectory monitoring required.
7. **Substrate is plasma/cfDNA AND Stage 1 d in +0.3 to +0.7 AND Stage 2 cell-of-origin tile NULL?** → AD trajectory (chronic non-commitment) OR Phase 1 cancer trajectory (commitment pending). Distinguish by serial-trajectory monitoring.
8. **At any timepoint: if Stage 1 immune-tile sub-components SIGN-FLIP from positive to negative (monocytes, neutrophils, erythrocyte progenitors all going from positive at years-out to negative near-diagnosis)?** → Phase 1 → Phase 2 transition signature. The body is reorganizing its informational accounting around the imminent tumor commitment. Watch the rising tissue tile to identify the committing organ.

The decision tree is the operational interpretation of what Stage 1 magnitude + Stage 2 + Stage 3 + substrate context tells the EDEAR clinician about which organ is most likely the source.

---

*End of cross-cancer Stage 1 immune signature comparison.*
