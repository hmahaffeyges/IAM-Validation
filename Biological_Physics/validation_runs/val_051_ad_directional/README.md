# VAL-051 — AD-Directional Immune Panel, Holdout Recovery

**Date:** 2026-04-23
**Status:** Pre-registered, hash-sealed, executed. Outcome 1 (FULL RECOVERY on AIBL holdout).
**Motivating finding:** VAL-050 pooled-entropy null (d=+0.08, p=0.32) on AIBL AD. 7 of 18 panel CpGs individually FDR<0.05 with bidirectional pattern.
**This study:** Test whether directional weighting on a training-selected subset recovers the signal.

---

## Headline result

On a sealed 20% holdout of AIBL (n=33 AD, n=95 HC, n=20 MCI) never seen during panel selection:

| Metric | Cohen's d | p (one-sided) | AUC |
|---|---|---|---|
| **A<sub>dir</sub> (Rule A, 7 CpGs)** | **+0.624** | **0.0013** | **0.677** |
| A<sub>dir</sub> (Rule B, all 18, directional) | +0.464 | 0.011 | 0.634 |
| A<sub>entropy</sub> pooled (VAL-050 metric, null-comparator) | +0.056 | 0.42 | 0.50 |

**The pooled-entropy null holds on the same holdout samples** → this is not a subset artifact, it's a metric-specific recovery.

Sex-stratified: **Male d = +0.51, p = 0.04**; **Female d = +0.71, p = 0.003**. Both sexes significant.

---

## The Directional-Score Principle

Pooled-β entropy A-score is direction-blind — it works when a disease shifts the entire panel uniformly. Many diseases do NOT do that. AD in the immune class splits its signal: some CpGs go up (inflammation / IFN response), others go down (T-cell exhaustion). Pooled averaging cancels them.

**Directional score** (A<sub>dir</sub> = mean across panel of direction<sub>i</sub> × z-score<sub>i</sub>) recovers the signal. Directions (+1 / −1) are assigned per CpG from the sign of Δβ on a training split; z-scores standardize against training-HC mean/SD.

Entropy A-score and directional score are **complementary, not redundant.** Report both. Tier by validation status.

---

## Files

| File | Role |
|---|---|
| `VAL_051_PREREG.md` | Pre-registration: decision rules, frozen constants, outcome matrix |
| `VAL_051_SEAL.txt` | SHA-256 hashes sealed before holdout access (2026-04-23 07:23:53 UTC) |
| `val051_split.py` | Deterministic 80/20 stratified split by disease × sex, seed=42 |
| `val051_split_map.json` | Per-GSM training/holdout assignments |
| `val051_select.py` | Panel selection on TRAINING SET ONLY (Rule A: \|Δβ\|>0.015 AND q<sub>FDR</sub><0.10) |
| `val051_panel_ruleA.json` | 7-CpG selected panel with training stats |
| `val051_panel_ruleB.json` | All 18 with training stats (directional weighting) |
| `val051_analyze.py` | Holdout scoring, H1/H3/H4, 4×2 outcome decision |
| `VAL_051_RESULTS.json` | Full results including per-sample holdout A-scores |
| `VAL_051_REPORT.md` | Human-readable report with all tables and interpretation |

**Input data** (AIBL manifest + β-matrix) are in `../val_050_aibl/aibl_manifest.json` and `../val_050_aibl/aibl_imm_betas.json`. Not duplicated here.

---

## Reproduction

```bash
# From this directory, symlink or copy AIBL inputs
ln -s ../val_050_aibl/aibl_manifest.json .
ln -s ../val_050_aibl/aibl_imm_betas.json .

# Run in order
python3 val051_split.py      # → val051_split_map.json (already committed)
python3 val051_select.py     # → val051_panel_ruleA.json, val051_panel_ruleB.json
python3 val051_analyze.py    # → VAL_051_RESULTS.json

# All stdlib Python 3.9+. No external packages required.
# Seed = 42 everywhere. Outputs are byte-identical on re-run.
```

## Seal verification

```bash
sha256sum -c <(awk 'NR>2 && NF==2 {print $1"  "$2}' VAL_051_SEAL.txt)
```

All 10 hashes should verify.

---

## Pre-locked outcome matrix (VAL_051_PREREG.md §7)

| | AIBL holdout (H1) | AddNeuroMed (H2) | Result |
|---|---|---|---|
| 1 | d > 0.3, p < 0.05 | pending | **ACHIEVED on AIBL holdout** |
| 2 | d > 0.3, p < 0.05 | d < 0.1 | AIBL-internal only |
| 3 | 0.1 < d < 0.3 | any | Direction-positive-weak |
| 4 | d < 0.1 | — | Null even with directional |
| 5 | d < 0 | — | Anti-direction (overfit) |

**Outcome 1 on AIBL holdout arm is confirmed.** VAL-052 (AddNeuroMed GSE144858, 450K, n=300) is the pending cross-platform replication.

---

## The 7-CpG AD-Directional Panel (Rule A)

| CpG | Δβ (train) | Direction in AD |
|---|---|---|
| cg16867657 | +0.0246 | UP |
| cg25809905 | −0.0277 | DOWN |
| cg22454769 | +0.0207 | UP |
| cg09809672 | −0.0203 | DOWN |
| cg26614073 | −0.0194 | DOWN |
| cg00431549 | −0.0152 | DOWN |
| cg02228185 | −0.0330 | DOWN |

2 UP + 5 DOWN. Bidirectionality is the signature that breaks pooled-β averaging.

---

## Limitations

- Internal 80/20 split — cross-platform (VAL-052 AddNeuroMed) is the next step.
- 7-CpG panel from a 18-CpG starting set. Genome-wide selection (VAL-056) would likely reach AUC 0.72–0.75.
- Holdout n=33 AD; CI on d is wide [+0.24, +1.06].
- d=0.62 → per-patient sensitivity at 95% specificity ≈ 25-30%. Deployment is cohort screening / serial monitoring, not single-shot diagnosis.
- Age effect not formally regressed (pooled null on same cohort argues against as primary driver; VAL-053 will confirm).

---

## Citations

- **Nabais et al. 2021** — AIBL GSE153712 source, *Genome Biology* 22:90
- **Xu et al. 2020** — IMM_CPGS starting panel, *JNCI* 112(1):87-94
- **Fransquet et al. 2020** — CD4+ ~4× signal motivation for VAL-055
- **Yang et al. 2024** — Sex-differential AD methylation, confirms H3 sex asymmetry
- **Zhang et al. 2022, Sugden et al. 2024** — Published AD blood-methylation AUC range 0.67-0.79
