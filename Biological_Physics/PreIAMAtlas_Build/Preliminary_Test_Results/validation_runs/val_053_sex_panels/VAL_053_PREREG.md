# VAL-053 Pre-Registration — Sex-Specific AD Panel Selection

**Date:** 2026-04-23
**Parent:** VAL-051 — AD-directional immune panel, holdout recovery
**Motivating finding:** VAL-051 Rule A panel (unified across sexes) produced Male d=+0.51, Female d=+0.71 on sealed holdout. Female signal is 1.38× the male signal. Separate selection per sex may push both higher by removing cross-sex interference.

---

## 1. Scope

Build TWO panels — a female-train-only panel and a male-train-only panel — using the VAL-051 80/20 split (already sealed). Score each on its own-sex holdout. Compare to VAL-051 unified-panel performance.

**Not re-splitting.** The VAL-051 split_map is inherited; no new randomization. This prevents p-hacking by split variation.

---

## 2. Frozen constants (inherited)

- VAL-051 split_map.json (seed=42 stratified disease × sex)
- 18-CpG starting panel IMM_CPGS_EPIC_18
- H_min(immune, methyl) = 0.838889
- Rule A criterion: |Δβ| > 0.015 AND q_FDR < 0.10
- Seed = 42, N_boot = N_perm = 10,000

---

## 3. Panel selection (training only, per-sex)

**Panel-F:** Select on female training (72 AD + 217 HC)
**Panel-M:** Select on male training (56 AD + 159 HC)

Apply Rule A identically to each subset. Direction assigned from sign of Δβ in that sex's training.

---

## 4. Hypotheses (pre-locked)

**H1 (primary):** Panel-F on female holdout (n=19 AD vs n=55 HC) gives Cohen's d > VAL-051 female unified (d_unified = +0.705).

**H2 (primary):** Panel-M on male holdout (n=14 AD vs n=40 HC) gives Cohen's d > VAL-051 male unified (d_unified = +0.512).

**H3 (secondary):** Panel-F applied to male holdout: does female-trained panel generalize to males?

**H4 (secondary):** Panel-M applied to female holdout: does male-trained panel generalize to females?

**H5 (secondary):** Jaccard overlap of Panel-F and Panel-M CpGs. How much of the AD signature is sex-shared vs sex-specific?

---

## 5. Outcome matrix

| Panel-F vs unified on Female | Panel-M vs unified on Male | Interpretation |
|---|---|---|
| Better | Better | Sex-specific panels clearly beneficial — build both into EDEAR |
| Better | Same/worse | Female benefits from sex-specific; male doesn't. Sex-asymmetric response. |
| Same/worse | Better | Male benefits; female saturates with unified. |
| Same/worse | Same/worse | Unified panel is good enough; don't bother with sex-specific. |

All four outcomes publishable.

---

## 6. Seal procedure

All inputs hash-sealed pre-analysis:
- This prereg
- VAL-051 split_map.json (inherited, sealed)
- aibl_manifest.json, aibl_imm_betas.json (inherited, sealed)
- val053_select_sex.py, val053_analyze_sex.py

Sealed to VAL_053_SEAL.txt before execution.

---

## 7. Scientific integrity

- No re-split
- Rule A criteria identical for both sexes (no tuning per sex)
- If Panel-F or Panel-M selects zero CpGs (no survivors at Rule A threshold for that sex alone), that IS the result — no threshold relaxation.
- Both outcomes publishable whether positive or null.
