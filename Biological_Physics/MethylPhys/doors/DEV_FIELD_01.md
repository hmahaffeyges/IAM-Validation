# DEV-FIELD-01 — is the tissue beside a tumor already off the healthy middle? (design, 2026-10-13; NOT sealed, nothing read)

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

**Draft bars, for the author before any seal** (each against the healthy donors, chance = 5 % per array):
| | bar | what it tests |
|---|---|---|
| F1 sham | HDB split at random into two halves; the second half reads above the first half's 95th percentile in at most 10 % | the band is honest |
| F2 field | AN reads above the full HDB 95th percentile in more arrays than chance allows (binomial p < 0.05) | the field exists |
| F3 gradient | median order HDB <= CUB <= OQ <= AN <= TU, by a one-sided trend test (p < 0.05) | the field fades with distance |
| F4 positive control | TU above the HDB 95th percentile beyond chance | the reading sees what it should |

**What breaks the picture:** F2 not met with F1 and F4 met - the tissue next to a tumor is healthy on a mix-proof reading, and the
adjacent-normal signal was cell mix. F3 not met with F2 met: the field exists but does not fade with distance.

**Before the seal, still to do:** (1) metadata-only check of slides and processing batch per tissue, so no tissue sits on its own
slides; (2) a planted test on spent arrays (the TCGA breast normals): a 1 % fidelity loss planted in a third of them must meet F2,
and none planted must not; (3) the author's word on the bars.

**Prior reading:** GSE287331 has not been read. The sealed site list comes from DEV-TISSUE-BREAST-01, whose arrays are spent; only the
atlas-derived list is reused, no array.

**In plain words.** Tissue beside a tumor often looks normal under a microscope. We want to know whether its DNA already reads less
faithful than a healthy woman's breast, using only sites that do not change when the mix of cell types changes, so the answer cannot
be an artifact of what the sample contained. This series is unusually good for it: healthy donors, the patient's other breast,
another part of the same breast, the tissue right next to the tumor, and the tumor, all measured together. If the reading climbs
step by step toward the tumor, the disorder spreads through the organ before the tumor. If the tissue next to the tumor reads
healthy, the old signal was cell mix. **What it would take:** the batch check, the planted test, the seal, then one download and read.

**Synthetic test:** planned (TCGA breast normals, planted 1 % loss); not written yet.

**Not a milestone:** a design; nothing read, nothing sealed.
