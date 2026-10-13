# DEV-FIELD-01 — is the tissue beside a tumor already off the healthy middle? (sealed before any array was read; SCORED ONCE 2026-10-13)

**The question (author, 2026-10-13).** In the cohort era, tissue next to tumors read high: "What looked healthy under the microscope,
was actually not healthy... the sign of the disorder in the system, and in the entire organ." HYP-SATURATION-01 falsifier 2 makes it
testable: the field comes before the tumor. The confound a referee raises first is cell mix - tissue beside a tumor has a different
mixture of cells - so the reading must be one that a change of mix cannot move.

**Data: GSE287331** (GEO metadata checked 2026-10-13, nothing downloaded): breast, Illumina EPIC v1 (GPL21145), 446 arrays, each
with raw IDATs, in ONE series - the condition DEV-GROUP-01 forced. Five tissues, a distance gradient:

| code | tissue | arrays |
|---|---|---|
| HDB | healthy donor breast (no cancer) | 182 |
| CUB | contralateral, unaffected breast of a patient | 68 |
| OQ | the same breast, another quadrant | 67 |
| AN | adjacent to the tumor | 60 |
| TU | tumor | 69 |

Found by the 2026-10-13 data search (candidate_datasets_2026-10-13.csv). Not in CANON/data_spent.csv; never read.

**The reading.** Met-A at the composition-blind breast sites, the list sealed in DEV-TISSUE-BREAST-01 (sealed_breast_tissue.json).
The list was fixed from the atlas alone (sites where every breast cell type reads within 0.05 of luminal epithelium), so it carries
no information from any array, and a shift in cell mix cannot move it. Stage 1 calibration, unchanged.

**The slide check changed the design (metadata only, `data/DEV_FIELD_01/field_meta_01.py`, output `field_meta_01_output.txt`).**
Arrays sit on 63 slides. Every healthy-donor array is on slides with no patient tissue, and every adjacent-tissue array is on slides
of its own. So healthy donor against adjacent tissue cannot be separated from slide against slide - the trap DEV-GROUP-01 measured,
where two healthy groups differed by a cancer-sized amount untared. The patient's other breast (CUB), the other quadrant of the same
breast (OQ) and the tumor (TU) DO share slides: 66 patients have all three, and for 31 of them all three sit on one slide.

**Draft bars, for the author before any seal:**
| | bar | what it tests |
|---|---|---|
| F1 field (primary) | in the 31 one-slide patients, OQ reads above the same woman's CUB in more patients than chance (one-sided sign test, p < 0.05) | the breast with the cancer is off the middle; the other breast is the woman's own control |
| F2 positive control | in the same 31, TU above OQ beyond chance (sign test, p < 0.05) | the reading sees the tumor |
| F3 replication | F1 in the other 35 patients, each array tared to its own slide's median | the same field under a tare |
| F4 descriptive only | AN and HDB, each against the spread of slide medians among the 24 healthy-donor slides | reported, not decided: slide-confounded |

**What breaks the picture:** F1 not met with F2 met - inside one woman, on one slide, the breast with the cancer reads like the breast
without it on a mix-proof reading, and the adjacent-normal signal of the cohort era was cell mix or batch. Note that a field reaching
the OTHER breast too (a whole-body field, falsifier 8) would weaken F1; F4 is the only view of that, and it is confounded.

**Planted test, run once before the seal** (`data/DEV_FIELD_01/planted_field_01.py`, bars committed first; output
`planted_field_01_summary.txt`). The sealed blind-site reading and planting rule on the spent TCGA breast normals, two women on one plate
standing in for one woman on one slide (conservative), 1,000 draws of 31 pairs per case: with nothing planted F1 is met in 3.6 % of
draws (bar: at most 10 %); a 1 % fidelity loss is found in 100 % (bar: at least 80 %); 0.5 % is also found in every draw.

**Slide position, checked because the reading is that sensitive** (`field_positions_01.py`, output `field_positions_01_output.txt`):
in the 31 one-slide patients the other-quadrant array sits above the other-breast array in 15 and below in 16, so array position cannot
produce F1.

**The seal** (`score_field_01.py seal`, metadata only; `sealed_field_01.json`, with the scorer's sha256): all 6,000 sealed breast sites
exist on EPIC v1; F1 and F2 on 31 patients, met at 21 or more; F3 on the other 35, met at 23 or more; F4 descriptive. The scorer refuses
to score unless the seal is committed unchanged and its own hash matches.

**Amendment, 2026-10-13, after calibration began and before any reading (author).** "There is still a chance both breasts could read
higher, they are both the same organs and the same system." Correct: F1 compares a woman's two breasts, so a field reaching both makes F1
flat, and the sealed label would have called that a break of the picture. Changed, with the bars, pairs, sites and critical counts
unchanged: F1 not met with F2 met now reads **no difference between a woman's two breasts; a field covering both breasts or the whole
body is not ruled out**. Added F5, descriptive and never decided: each woman's other breast against the spread of the healthy-donor slide
medians, which is the only view this series gives of a whole-body field, and it is slide-confounded. The first seal is kept as
`sealed_field_01_v1_SUPERSEDED.json`; the resealed scorer carries a new hash.

**Scoring crash, 2026-10-13, recorded before the rerun.** The first `score` run computed the readings and crashed writing its JSON
(numpy integers are not serializable), after writing three result tables. Those tables were deleted unopened; no result was seen. The
fix is serialization only (`default=int` on the two JSON calls; the diff is those two lines), so the scorer was resealed with its new
hash; the seal is otherwise identical (pairs, sites, critical counts, rule, amendment).

## Outcome, scored once (2026-10-13)

`score_field_01.py score` on all 446 arrays (Stage 1 unchanged, box run; checksums in the job record); outputs `scored_field_01.json`,
`scored_field_01_F1F2.csv`, `scored_field_01_F3.csv`, `scored_field_01_arrays.csv`.

| | result | bar | verdict |
|---|---|---|---|
| F1 the breast with the cancer vs her other breast, one slide | 20 of 31 read higher, median ratio 1.0024 | 21 of 31 | **NOT MET** |
| F2 tumor vs the same breast, one slide | 24 of 31, median 1.031 | 21 of 31 | MET |
| F3 the same as F1, other 35 women, slide-tared | 17 of 35, median 1.000 | 23 of 35 | NOT MET |

**Verdict under the amended seal: no difference between a woman's two breasts.** Inside one woman, on one slide, the breast with the
cancer reads like her other breast, while the reading still sees the tumor. That is not a break of the field picture: a field covering
both breasts reads exactly like this.

**What F4 and F5 show, descriptive and NOT decided** (median entropy at the blind sites):
| healthy donors | her other breast | same breast, other quadrant | adjacent | tumor |
|---|---|---|---|---|
| 0.329 | 0.389 | 0.391 | 0.375 | 0.399 |

All 68 of the women's other-breast arrays read above every one of the 24 healthy-donor slide medians (0.314 to 0.340). Every patient
tissue sits about 18 % above the healthy donors, and the tumor only about 3 % above the women's other breasts. **This cannot be read as
a body-wide field yet**, for three reasons that all point the same way: the healthy donors sit on slides of their own; healthy donor
tissue is usually collected differently from surgical tissue; and the series lists no age, while disorder in methylation rises with age.
It is the strongest hint we have of a field covering both breasts, and it needs a series where healthy donors and patients share slides
and ages are known.

**In plain words, the result.** Inside each woman, the breast with the cancer did not read different from her other breast. Both
breasts of every patient read well above the healthy donors, by far more than the tumor stands above them - but the healthy donors were
measured on separate slides, probably collected differently and of unknown age, so that gap is a lead, not a result.

**Prior reading:** GSE287331 has not been read. The sealed site list comes from DEV-TISSUE-BREAST-01, whose arrays are spent; only the
atlas-derived list is reused, no array.

**In plain words.** Tissue beside a tumor often looks normal under a microscope. We want to know whether its DNA already reads less
faithful than a healthy woman's breast, using only sites that do not change when the mix of cell types changes, so the answer cannot
be an artifact of what the sample contained. This series is unusually good for it: healthy donors, the patient's other breast,
another part of the same breast, the tissue right next to the tumor, and the tumor, all measured together. If the reading climbs
step by step toward the tumor, the disorder spreads through the organ before the tumor. If the tissue next to the tumor reads
healthy, the old signal was cell mix. The slide check showed the cleanest comparison is inside one woman: the breast with the cancer against her other breast, on the same slide. **What it would take** to settle the body-wide question: a series with healthy donors and patients on shared slides, with ages, sealed first.

**Synthetic test:** DEV_FIELD_01/planted_field_01.py, DEV_FIELD_01/planted_field_01_summary.txt

**Not a milestone:** scored once; F1 not met, a body-wide field not ruled out, the hint confounded.

**Milestone:** F2, the sealed positive control (the tumor seen in a second laboratory's patients), 2026-10-13; listed on the front page and in `CANON/results_register.json`. F1 not met is listed with it.

<!-- PLAIN:BEGIN (generated by CANON/results_to_tex.py from CANON/results_register.json; do not edit by hand) -->
**In plain words** (Breast tumors seen again, in a second laboratory's patients): A new group of women with breast cancer, from a different laboratory. Each tumor was read against the tissue of the same breast on the same slide, by a rule written and sealed before any array was opened, and the reading saw the tumor in more of them than the rule required. The study also asked a harder question: whether the healthy-looking tissue next to a tumor already differs from the woman's other breast. It fell one woman short of the bar, so that question stays open.
<!-- PLAIN:END -->
