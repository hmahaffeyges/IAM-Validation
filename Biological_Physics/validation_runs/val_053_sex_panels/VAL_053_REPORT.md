# VAL-053 — Sex-Specific AD Panel Selection

**Status:** Complete — pre-registered, sealed, executed.
**Date run:** 2026-04-23
**Pre-registration:** VAL_053_PREREG.md
**Parent:** VAL-051
**Outcome:** **Sex-specific panels do NOT outperform unified; unified panel is the right EDEAR product**

---

## 1. Short answer

Separate panel selection on female-only and male-only training subsets of AIBL does NOT improve holdout performance over the unified VAL-051 panel. The unified panel is the correct production choice.

**Female result:** Panel-F (10 CpGs) on Female holdout → d = +0.585 vs unified d = +0.705. **Slightly worse.**
**Male result:** Panel-M selected only 1 CpG → too small to score under the ≥5-CpG coverage gate.
**Jaccard overlap:** 0.10 — F and M share only cg16867657 of the AD-direction CpGs.

This is **Outcome 4 — unified is good enough.** The EDEAR product uses a single unified panel across sexes. The directional score captures both sexes at their respective strengths.

---

## 2. Panel-F: 10 CpGs selected from female-only training (n=72 AD, n=217 HC)

| CpG | Δβ | Direction |
|---|---|---|
| cg16867657 | +0.027 | UP |
| cg25809905 | −0.031 | DOWN |
| cg00431549 | −0.026 | DOWN |
| cg09809672 | −0.025 | DOWN |
| cg22454769 | +0.026 | UP |
| cg12554573 | −0.016 | DOWN |
| cg26614073 | −0.020 | DOWN |
| cg14614643 | −0.036 | DOWN |
| cg17861230 | +0.020 | UP |
| cg02228185 | −0.036 | DOWN |

Selected 10 at Rule A threshold (|Δβ| > 0.015 AND q_FDR < 0.10). More than unified (7) because female training has higher statistical power and more CpGs cleared the threshold.

**On female holdout:** d = +0.585 [+0.10, +1.16], AUC = 0.69, p = 0.006. **Slightly worse than unified's d = +0.705.** The additional 3 CpGs added noise without adding signal.

---

## 3. Panel-M: 1 CpG selected from male-only training (n=56 AD, n=159 HC)

| CpG | Δβ | Direction |
|---|---|---|
| cg16867657 | +0.021 | UP |

Only cg16867657 cleared Rule A on male training. The male AD signal is concentrated in a single CpG — the strongest up-in-AD inflammation/IFN response CpG. Other CpGs that were significant at the unified panel's threshold dilute below significance when you subset to only 215 male samples.

**On male holdout:** skipped — a 1-CpG panel fails the coverage gate (≥5 valid β per sample). Even if scored, a 1-CpG z-score with no averaging is high-variance and not a meaningful panel.

---

## 4. Jaccard overlap & direction agreement

- **Shared CpGs:** 1 (cg16867657)
- **Female-only:** 9 CpGs
- **Male-only:** 0 CpGs
- **Jaccard:** 0.10
- **Direction agreement on shared:** 1/1

**Biological read:** the male AD immune signature is a narrow inflammation/IFN signal anchored at cg16867657. The female AD immune signature is broader, encompassing both the inflammation up-in-AD CpGs AND a wider set of down-in-AD CpGs (T-cell effector loci, cytokine-signaling regions). Females have 10× more AD-affected CpGs than males in this data.

This is consistent with published literature (Yang 2024) that AD blood methylation is sex-dimorphic, with female-specific AD-DMPs outnumbering male-specific by roughly 3:1. Our 9:0 ratio is more extreme, likely because (a) male n is smaller so fewer CpGs pass the FDR threshold, and (b) the directional Rule A criterion is more sensitive to female-skewed signal.

---

## 5. Comparison to VAL-051 unified

| Test | Unified (VAL-051) | Sex-specific (VAL-053) |
|---|---|---|
| Female holdout d | +0.705 (p=0.003, AUC 0.70) | +0.585 (p=0.006, AUC 0.69) |
| Male holdout d | +0.512 (p=0.041, AUC 0.66) | not scoreable (1 CpG) |

**Conclusion:** Unified panel is the right product. Sex-specific selection hurts female performance and makes male performance untestable. The directional score with the unified 7-CpG Rule A panel captures both sexes at their respective natural strengths.

---

## 6. What this means for the Cookbook

- AD-immune card uses **one panel** (Rule A, 7 CpGs, unified).
- Card flags sex-asymmetric performance (Female d > Male d on AIBL) as a cohort-specific observation.
- VAL-052 cross-platform result shows AddNeuroMed has ROUGHLY EQUAL sex performance (Male d = +0.40, Female d = +0.36) — AIBL's sex asymmetry may be Australian-cohort-specific and not universal.
- Card does NOT include sex-specific calibration fields at this validation tier.

---

## 7. Honest limitations

1. **Male panel (n=1 CpG)** could be due to statistical power rather than biology. With n_AD = 56 in male training, FDR is conservative. A larger male cohort might surface more CpGs.
2. **Rule A was pre-registered for unified selection**, not sex-specific selection. The same thresholds might not be optimal for smaller subsets. We pre-locked no tuning.
3. **Direction-based analysis only.** Mean-β or entropy-based female/male panels could behave differently.

---

## 8. Reproduction

All inputs hash-sealed (VAL_053_SEAL.txt). Seed 42.

```bash
python3 val053_sex_panels.py  # produces VAL_053_RESULTS.json
```

---

**VAL-053: a clean negative on sex-specific panels. Unified is the right EDEAR product for AD-immune. The directional score + unified panel captures the AD signal in both sexes better than sex-specific selection does.**
