# VAL-053 — Sex-Specific AD Panel Selection

**Date:** 2026-04-23
**Status:** Complete — pre-registered, hash-sealed, executed.
**Parent:** VAL-051
**Outcome:** **Unified panel wins — sex-specific panels don't outperform.**

---

## Headline

Separate panel selection on female-only and male-only AIBL training subsets does NOT outperform the unified VAL-051 panel.

- **Panel-F** (10 CpGs from female-only training): d = +0.585 on female holdout vs unified d = +0.705 on same holdout. **Slightly worse.**
- **Panel-M** (1 CpG from male-only training — only cg16867657): fails ≥5-CpG coverage gate. Not deployable.
- **Jaccard overlap** F vs M = 0.10 (1 shared CpG).

**Biology:** the male AD immune signature is narrow and inflammation/IFN-anchored (cg16867657). The female signature is broader, encompassing both inflammation and T-cell effector loci. Consistent with Yang 2024 sex-dimorphic AD methylation literature.

**EDEAR consequence:** deploy the unified 7-CpG panel. Sex-specific calibration is a future research direction, not a production requirement.

---

## Files

| File | Role |
|---|---|
| `VAL_053_PREREG.md` | Pre-registration with sex-stratified selection protocol |
| `VAL_053_SEAL.txt` | SHA-256 hashes sealed before analysis |
| `val053_sex_panels.py` | Sex-stratified selection (80% training) + holdout scoring (20%) |
| `VAL_053_RESULTS.json` | Full results: Panel-F, Panel-M, Jaccard, per-sex holdout performance |
| `VAL_053_REPORT.md` | Human-readable report |

---

## Reproduction

```bash
# AIBL data + VAL-051 split map and panel live in ../val_050_aibl/ and ../val_051_ad_directional/
cp ../val_050_aibl/aibl_manifest.json .
cp ../val_050_aibl/aibl_imm_betas.json .
cp ../val_051_ad_directional/val051_split_map.json .
cp ../val_051_ad_directional/val051_panel_ruleA.json .

python3 val053_sex_panels.py   # ~30 seconds
```

All stdlib Python 3.9+. Seed 42. Outputs byte-identical.

---

## Sources

- Yang J, et al. **Sex-specific DNA methylation differences in Alzheimer's disease.** 2024 reference (sex-dimorphic AD-DMP analysis).
- Xu 2020 JNCI for panel starting set.
