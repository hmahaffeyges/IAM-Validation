# VAL-051 Pre-Registration — AD-Directional Immune Panel, Selection + Replication

**Principal Investigator:** Heath W. Mahaffey
**Framework:** IAM / GAPE Informational Actualization Model
**Drafted:** 2026-04-23
**Seal target:** pre-analysis SHA-256 freeze of this file + input files + analysis scripts
**Motivating finding:** VAL-050 delivered a pre-registered null on pooled-β A-score from a breast-cancer-derived panel applied to AD. Per-CpG analysis revealed 7 of 18 CpGs individually significant at FDR < 0.05, with bidirectional pattern (4 down in AD, 3 up in AD). Pooled averaging cancels the signal; directional weighting recovers it.

---

## 1. Scope

Test whether an AD-directional immune-class panel, selected on a training split of AIBL under an outcome-blind selection rule, recovers per-patient discrimination of AD vs HC in:

**A.** A held-out 20% split of AIBL that was not used for panel selection (internal replication, cross-validated)

**B.** AddNeuroMed GSE144858 (450K platform, independent population, independent preprocessing — external cross-platform replication)

**C.** Sex-stratified sub-analyses on both cohorts

**Not in scope for VAL-051:**
- EpiDISH immune subcomposition (VAL-052)
- Time-to-onset stratification (requires AIBL direct access)
- Purpose-built AD panel from scratch (VAL-053)
- EWAS-scale CpG screen beyond the existing 18-CpG IMM panel

---

## 2. Frozen constants (inherited from VAL-050)

| Constant | Value | Source |
|---|---|---|
| H_min(immune, methyl) | 0.838889 | G-002 MCMC posterior |
| A-score formula | A = H(β) / H_min(immune) | GAPE_WEB_v13.py |
| H formula | H(β) = −β log₂β − (1−β) log₂(1−β) | Shannon binary entropy |
| Seed | 42 | all randomization |
| α | 0.05 one-sided for primary | locked before split |
| N permutation | 10,000 | locked |
| N bootstrap | 10,000 | locked |

**Starting panel:** IMM_CPGS_EPIC_18 from VAL-050 (18 CpGs).

---

## 3. Training / holdout split (HASH-SEALED BEFORE ANY OUTCOME ACCESS)

**Method:** Deterministic 80/20 stratified split of AIBL within disease-status × sex cells.

- Random seed 42
- Stratified on (disease status, sex) so training and holdout have matched case/control/MCI ratios and matched sex distributions
- Record GSM ID assignments in `val051_split_map.json` BEFORE looking at any β-value outcomes from the holdout

**Training:** ~80% of AIBL — used for panel selection AND panel-weight fitting
**Holdout:** ~20% of AIBL — UNTOUCHED until the frozen panel is scored on it (one-shot)

Expected n split:
- Training: ~129 AD, ~75 MCI, ~377 HC
- Holdout: ~32 AD, ~19 MCI, ~94 HC

**Once `val051_split_map.json` is hashed into VAL_051_SEAL.txt, the split is final. No re-splitting, no reshuffling, no re-examination.**

---

## 4. Panel selection rule (OUTCOME-BLIND ON HOLDOUT)

All selection occurs on TRAINING SET ONLY. Holdout is not touched.

**Rule A (primary — directional panel):**
For each of the 18 starting CpGs, on training data only:
- Compute Δβ = mean(β_AD) − mean(β_HC)
- Compute p_two-sided Mann-Whitney
- Apply BH-FDR correction across 18 tests

**Selection criterion:**
- CpG INCLUDED if: `|Δβ| > 0.015` **AND** `q_FDR < 0.10` in training set
- Direction assigned: +1 if Δβ > 0 (up-in-AD), −1 if Δβ < 0 (down-in-AD)

**Rationale for thresholds:**
- `|Δβ| > 0.015` is above typical EPIC technical noise (±0.015-0.020)
- `q_FDR < 0.10` is slightly looser than 0.05 because we're selecting candidates for a composite score, not making per-CpG claims

**Rule B (secondary — VAL-050 panel as-is, directional-weighted):**
For direct comparison with VAL-050's pooled-null, re-score the SAME 18 CpGs using their training-set directions (no CpG dropping, just directional weighting).

---

## 5. Scoring formula

**Directional A-score per sample** (used for primary test on holdout):

```
A_dir = (1 / n_panel) * Σᵢ [ dir_i * (β_i − β̄_HC_train) / σ_HC_train ]
```

Where:
- `dir_i ∈ {+1, −1}` is the direction assigned on the training split
- `β̄_HC_train`, `σ_HC_train` are the training-set HC mean and SD for CpG i
- Standardization uses HC-train only, not combined, to avoid leakage

**Why not entropy A-score for the directional test?**
Entropy is directionally blind (H(β) = H(1−β)). The whole point of VAL-051 is to recover the direction that pooled-entropy loses. A_dir is the right statistic for this question. We additionally report the pooled entropy A-score as a null-comparator.

---

## 6. Primary hypotheses (pre-locked)

**H1 (primary):** On AIBL holdout, A_dir(AD) > A_dir(HC), one-sided.
**Test:** Mann-Whitney U one-sided.
**Decision threshold:** α = 0.05.
**Effect size:** Cohen's d on A_dir with 10,000-bootstrap 95% CI.

**H2 (cross-platform replication):** On AddNeuroMed GSE144858, A_dir(AD) > A_dir(HC), one-sided.
**Test:** Mann-Whitney U one-sided. Panel CpGs intersected with 450K probes; report coverage fraction.
**Decision threshold:** α = 0.05.

**H3 (sex-stratified primary):** H1 restricted to females-only AND males-only.
**Test:** MWU one-sided each.

**H4 (secondary — bimodality):** On holdout, Var(panel CpG β-scores within sample) differs between AD and HC.
**Test:** Levene's test for equality of variance, two-sided.
**Rationale:** The inflammation-vs-exhaustion simultaneity predicted by VAL-050's bidirectional pattern should manifest as ELEVATED within-sample variance across panel CpGs in AD. This is a direct test of the "two-mechanism" hypothesis.

---

## 7. Pre-locked outcome matrix (4 × 2 = 8 outcomes)

| | AIBL holdout (H1) | AddNeuroMed (H2) | Interpretation |
|---|---|---|---|
| 1 | d > 0.3, p < 0.05 | d > 0.2, p < 0.10 | **Full replication** — panel + directional weighting recovers AD signal across platforms. Publish. |
| 2 | d > 0.3, p < 0.05 | d < 0.1 or p > 0.20 | **AIBL-internal only** — panel is AIBL-specific. Either platform-dependent or population-dependent. Needs AD cohort on EPIC for cross-validation. |
| 3 | 0.1 < d < 0.3, p 0.05-0.15 | any | **Direction-positive-weak** — framework prediction is right but effect size modest. Consistent with true AD blood signal ceiling at d ≈ 0.2. |
| 4 | d < 0.1 | — | **Null** — directional weighting does not recover signal even on a training-derived panel. Framework at the panel-entropy level does not detect AD in blood. Motivates VAL-052 cell-type-stratified approach. |
| 5 | negative | — | **Anti-direction** — AD signal in panel CpGs is inconsistent with training-set direction in the same cohort's holdout. This is a catastrophic result for the panel but informative (would indicate training-set overfitting). |

All outcomes are publishable. All are hash-sealed before analysis.

---

## 8. AddNeuroMed details

**Cohort:** GSE144858, Illumina 450K, n ~= 300 (93 AD + ~111 MCI + ~96 HC per published description)
**Source:** Smith et al. 2018/2021, published in Alzheimer's & Dementia
**Preprocessing:** 450K BMIQ-normalized β values from GEO supplementary files
**Panel transfer:** 18 starting CpGs intersected with 450K probe list; expected ~18/18 because the IMM_CPGS_RAW 29-CpG panel is a 450K panel that was INTERSECTED to EPIC for VAL-050. On 450K we should recover all 18. Confirmed at data-load time; report exact coverage.

**Replication scoring:** Use AIBL-training-derived directions (not re-selected on AddNeuroMed). Use AIBL-training-derived (β̄_HC, σ_HC) for standardization — NOT AddNeuroMed's own controls — because that would be population-specific leakage and would break the "frozen panel" claim. Report sensitivity analysis with AddNeuroMed-own standardization as a secondary.

---

## 9. Pre-registered expected outcome (not used in decision)

Based on VAL-050 per-CpG FDR results and inflammation-vs-exhaustion simultaneity:

- Expected A_dir d on holdout: 0.30-0.50 (Outcome 1 or Outcome 3)
- Expected A_dir d on AddNeuroMed: 0.15-0.35 (smaller due to platform + population)
- Expected bimodality H4: AD variance > HC variance, ratio ~1.3-1.5
- Expected sex asymmetry: preserved (females stronger) but less dramatic than VAL-050 pooled because directional scoring recovers male signal too

Most likely: **Outcome 1 on AIBL holdout** (d ~ 0.4), **Outcome 3 on AddNeuroMed** (d ~ 0.2). This is a publishable "directional-panel replicates within-cohort, direction-positive-weak cross-platform" result.

---

## 10. QC gates (locked)

**Sample exclusion** (identical to VAL-050 for consistency):
- Missing disease status → exclude
- Missing sex → exclude
- Fewer than 12 of 18 panel CpGs valid → exclude

**CpG exclusion from panel** (training-set only):
- Failed Rule A criteria → excluded from panel
- No post-hoc re-inclusion regardless of holdout result

**Outlier handling:** No outliers removed from holdout. Report median + MAD as robustness check.

---

## 11. Seal procedure

Before H1/H2/H3/H4 execution:

1. SHA-256 of this prereg
2. SHA-256 of `aibl_manifest.json`, `aibl_imm_betas.json` (VAL-050 carryover)
3. SHA-256 of `val051_split.py` (split script)
4. SHA-256 of `val051_select.py` (panel selection script)
5. SHA-256 of `val051_analyze.py` (holdout + AddNeuroMed scorer)
6. SHA-256 of `val051_split_map.json` (produced by split script BEFORE selection)

All 6 hashes recorded in `VAL_051_SEAL.txt`. AddNeuroMed data stream is logged separately (matrix SHA).

**Once the split map is hashed, NOTHING about the split changes.**

---

## 12. Scientific integrity commitments

- No post-hoc re-split
- No post-hoc panel modification
- No post-hoc threshold adjustment
- No removing a CpG after seeing holdout
- No re-running with different α or sidedness
- AddNeuroMed is one-shot — no iteration, no "let me try with adjusted standardization" unless pre-specified as sensitivity (which is)
- All outcomes publishable

Signed for submission (Heath W. Mahaffey): ___________________________ Date: ___________
