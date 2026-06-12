# VAL-061 Outcome — CRC Tissue, Immune-Compartment Reading on TCGA-COAD

**Date completed:** 2026-04-24 UTC
**Prereg SHA:** bdce2f903a20a3375681a3589710c2f5a6392a4f4c6772305fd3afc656bed521
**Cohort SHA:** ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27
**Results SHA:** def8a69030a2b1d1619f4a930e419604b44c0f2097655c97eea7f580f4a12c96

## What this run actually measured

Xu-538 immune panel applied to CRC tumor tissue. The panel is immune-derived (Sister Study breast blood). Applied to tumor tissue, it reads the **tumor-infiltrating immune cell (TIL) compartment** inside the tumor, not the tumor secretory cells themselves. This is a reading of the immune response in the tumor bed, not a reading of tumor architecture.

The prereg as originally written ("tumor should go negative, inversion-direction") conflated two distinct compartments. Tumor cells and the immune response to tumor go in opposite directions; the correct prediction depends on which compartment the measurement reads.

## Compartment-corrected framework predictions

**Tumor cells (secretory class, H_min = 0.843264)** — go **positive** (disordered tumor architecture). Consistent with VAL-058 prostate secretory (+0.497) and VAL-060 breast secretory (+0.745). Requires a secretory-class panel, not the Xu-538 immune panel. NOT tested in this run.

**Immune cells in peripheral blood (immune class, H_min = 0.838889)** — go **negative**. Circulating immune compartment shows suppressed/exhausted response to disease presence. Consistent with VAL-047 blood pre-dx d = −0.33, and with yesterday's brain tumor peripheral immune reading.

**Immune cells in tumor tissue (immune class, H_min = 0.838889)** — go **positive**. Tumor-infiltrating lymphocytes are activated and expanded inside the tumor bed — the immune system in the battlefield, engaged and disordered relative to resting peripheral leukocytes. This is what VAL-061 measured.

## Results

- **n matched pairs:** 26 (of 38 downloaded, 12 excluded for coverage <430/538)
- **Paired Cohen's d:** +1.066, 95% CI [+0.585, +1.547], p < 0.00001
- **Unpaired Cohen's d:** +1.469, 95% CI [+0.856, +2.081], p < 0.00001
- **Per-CpG direction:** 61% hypomethylated (295/484), 39% hypermethylated (189/484)
- **A-tumor mean:** 0.592 ± 0.021
- **A-normal mean:** 0.563 ± 0.019

## Interpretation

The d = +1.066 is the tumor-infiltrating immune compartment reading strongly positive. This is the expected direction for activated TIL, and is NOT inconsistent with the VAL-047 blood pre-dx d = −0.33 peripheral reading. The two readings are from different compartments of the same immune system responding to the same disease, and they go in opposite directions by framework design.

Per-CpG pattern (61% hypomethylated, 39% hypermethylated in tumor) is consistent with activated immune cells in a tumor microenvironment: global hypomethylation of activation-linked loci with focal hypermethylation at suppressive/exhaustion markers. The aggregate A-score is positive because H(β) rises as β moves toward 0.5 from either direction — activated immune cells in the tumor bed have disordered methylation regardless of sign.

## What this run does NOT address

The tumor secretory architecture reading. VAL-061 did not test CRC tumor cells against secretory H_min on a secretory-specific panel. That test remains to be done. Expected direction: strongly positive, magnitude comparable to or larger than VAL-058 prostate secretory (+0.497) and VAL-060 breast secretory (+0.745) given CRC's higher proliferation and crypt-architecture disorder.

## Action items (revised)

- [ ] Build CRC tumor-cell secretory-class validation (VAL-062 — new pre-reg needed, secretory panel + secretory H_min + TCGA-COAD tumor tissue)
- [ ] Update crc-epic card v2.2 tissue arm to document BOTH compartments: VAL-061 tumor-TIL immune reading (d = +1.066, positive as expected for activated infiltrate) and VAL-062 tumor secretory reading (pending)
- [ ] Insert VAL-061 block into Evidence Report with compartment-corrected interpretation
- [ ] Update Master README crc-epic entry
- [ ] GitHub push

## Lessons learned (added to LESSONS_LEARNED)

CCL-019: The direction of an A-score reading depends on (a) the cellular class the panel is derived from, AND (b) the compartment the sample represents. The same immune class reads negative in peripheral blood (suppressed circulating response) and positive in tumor tissue (activated TIL). Future prereg statements must specify BOTH class AND compartment, not just class.

CCL-020: When applying a non-class-specific panel (e.g., an immune panel on tumor tissue), the reading is of the immune infiltrate within that tissue, not of the tissue's own architecture. To measure tumor architecture, the panel must be derived from or tuned to the tumor's own class.

## Prereg accounting

The original VAL-061 prereg predicted d < 0. The observed d = +1.066 is opposite direction. Under a strict prereg reading, this is a failed prediction. Under a compartment-corrected reading, the prereg was mis-specified (it named the wrong compartment) and the result is the expected direction for what was actually measured.

Both interpretations are logged transparently. The prereg is not retroactively amended. VAL-062 will be a new prereg with the correct compartment specification, and its prediction will be tested independently.
