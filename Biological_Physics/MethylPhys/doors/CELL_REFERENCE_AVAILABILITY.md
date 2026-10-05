# Which blood cells can get their own healthy reference (2026-10-05)

Source: the atlas v2 roster (`atlas/sources/roster_v2/atlas_v2_roster.csv`, all sources) and GEO GSE186458 sample titles (Loyfer 2023,
which ships .beta and read-level .pat files for every sample, hg19 and hg38; verified from the GEO supplementary file names:
506 `.pat.gz` with 506 `.pat.gz.csi` indexes, e.g. `GSM5652313_Blood-Granulocytes-Z000000TZ.pat.gz` and `...hg38.pat.gz`). Loyfer counts are WGBS samples; array counts are the rest.

| cell | atlas v2 samples (all sources) | Loyfer WGBS samples (read level) | array references | notes |
|---|---|---|---|---|
| neutrophils | 12 | 0 as neutrophils; 3 Blood-Granulocytes (a neutrophil-dominated mixture, OUT of atlas v2 as a mixture) | Salas 2018, 2022 | + GSE128731: 2 donors x 4 WGBS runs (Box Run 2) |
| monocytes | 14 | 3 | Salas 2018, 2022 | |
| NK cells | 13 | 3 | Salas 2018, 2022 | |
| B cells | 9 | 3 (+2 memory B) | Salas 2018 | |
| CD4 T cells | 8 | 3 (+ central memory 3, effector memory 3, naive 1) | Salas 2018 | + GSE128731 CD4 T WGBS |
| CD8 T cells | 9 | 3 (+ effector 3, effector memory 2, naive 2) | Salas 2018 | |
| eosinophils | 4 | 0 | Salas 2022 | arrays only |
| basophils | 6 | 0 | Salas 2022 | arrays only |
| regulatory T cells | 3 | 0 | Salas 2022 | arrays only |
| erythrocyte progenitors | 3 | 3 (bone marrow) | — | not circulating |

**Reading.** For single-molecule references (IAM-A), Loyfer gives 3 healthy samples for each of monocytes, NK, B, CD4 T and CD8 T cells,
all from one laboratory. That is enough to build a first reference, not to test it: each needs held-out healthy samples from another
laboratory before it can be commissioned (the new-cell rule). Neutrophils are the exception that is ready: Loyfer's granulocytes build the
floor, and GSE128731 provides another laboratory's purified neutrophils for the test. CD4 T cells are next (GSE128731 has them too).
Eosinophils, basophils and regulatory T cells have array references only.
