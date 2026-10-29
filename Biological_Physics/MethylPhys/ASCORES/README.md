# The A-score database

_Generated from `CANON/ascore_sources.json` by `CANON/ascore_db.py`; do not edit by hand._

One table per instrument, one row per sample reading, each read from a committed file and linked to its record. It is how we learn
where healthy samples and tumors sit on each instrument's gauge, and whether a line of concern shows before the tumor line
(manifest PHY-069). Readings of different kinds (an untared A, a tared A, a tumor-over-own-normal ratio, a held error) are kept
apart and are never pooled.

**Past the healthy line** counts the samples beyond the healthy line of the reading they were made with: Met-A's Normal band (0.95 to
1.05) and the C-Score's upper limit on the chain's tare (1.152) from their commissioning, and, for a tumor read against the same
patient's normal tissue, the line its sealed test set from pairs of normal tissue, where chance alone puts about 5 % of tumors past it.
Healthy rows show how rarely a healthy sample crosses its line; tumor rows show how often a tumor does. A line is never applied to a
reading made another way (an earlier tare), so those rows show none.

| instrument | reading | specimen | condition | samples | median | middle 90 percent | past the healthy line | file |
|---|---|---|---|---|---|---|---|---|
| Met-A | A, untared | isolated neutrophils | healthy | 6 | 1.0009 | 0.99428 to 1.0056 | 0 of 6 (0 %) | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | healthy | 13 | 1.0027 | 0.91902 to 1.0619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | patient (see record) | 16 | 0.9641 | 0.91625 to 0.99295 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | patient (see record) | 38 | 1.0082 | 0.97722 to 1.0351 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | healthy | 26 | 1.001 | 0.97897 to 1.0261 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | breast tissue | tumor | 53 | 1.0649 | 0.9781 to 1.2235 | 23 of 53 (43 %) above 1.094 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | colon tissue | tumor | 88 | 1.0658 | 0.9804 to 1.1658 | 28 of 88 (32 %) above 1.092 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | liver tissue | tumor | 50 | 1.1039 | 0.93635 to 1.3255 | 22 of 50 (44 %) above 1.125 | [Met_A.csv](Met_A.csv) |
| C-Score | C_rel, tared to the series | whole blood | healthy | 553 | 1.0003 | 0.7543 to 1.332 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | isolated neutrophils | healthy | 62 | 1.0032 | 0.84518 to 1.2189 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, chain tare | whole blood | healthy | 231 | 0.99669 | 0.90289 to 1.1027 | 3 of 231 (1 %) | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | patient (see record) | 38 | 1.044 | 0.76941 to 1.419 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | healthy | 26 | 1.0312 | 0.74043 to 1.2753 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| IAM-A | held error eps, per interior opportunity (commissioned counter) | purified cell types (56) | healthy | 56 | 0.0447 | 0.03546 to 0.05224 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | normal tissue of a cancer patient | 5 | 0.035141 | 0.032699 to 0.038031 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | normal tissue of a cancer patient | 8 | 0.038905 | 0.035798 to 0.040128 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | tumor | 5 | 0.042157 | 0.036901 to 0.04588 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | tumor | 8 | 0.04093 | 0.034506 to 0.04639 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | normal tissue of a cancer patient | 6 | 0.013831 | 0.010367 to 0.018707 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | tumor | 6 | 0.017143 | 0.010224 to 0.021938 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, arrays | whole blood | healthy | 100 | 2.395 | 2.247 to 2.5047 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, islands territory, arrays | whole blood | healthy | 100 | 3.7239 | 3.5067 to 3.8767 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, domains territory, arrays | whole blood | healthy | 100 | 2.186 | 2.0695 to 2.2709 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | not stated | leukemia | 5 | 2.195 | 2.1719 to 2.3016 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | cell line | control line | 1 | 2.0064 | 2.0064 to 2.0064 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | cell line | knockout | 1 | 1.998 | 1.998 to 1.998 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | sorted b cells | cord-blood B cells | 3 | 2.3834 | 2.3811 to 2.384 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | leukemia | 3 | 0.67377 | 0.64334 to 0.70321 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | leukemia | 3 | 0.62711 | 0.58457 to 0.68111 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | leukemia | 3 | 0.5064 | 0.35774 to 0.53291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | leukemia | 3 | 0.71765 | 0.71095 to 0.73597 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | leukemia | 3 | 0.67836 | 0.64861 to 0.70631 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | leukemia | 3 | 0.23681 | 0.23227 to 0.27173 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | leukemia | 3 | 0.51504 | 0.46639 to 0.58693 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | leukemia | 3 | 0.8962 | 0.88765 to 0.91069 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | leukemia | 3 | 0.997 | 0.9916 to 0.99853 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | plasma | leukemia | 3 | 1.0065 | 0.97284 to 1.0177 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | plasma | leukemia | 3 | 1.029 | 0.99246 to 1.0509 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney glomerular endothelium | plasma | leukemia | 1 | 1.068 | 1.068 to 1.068 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | leukemia | 3 | 1.0253 | 0.96203 to 1.0662 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | plasma | leukemia | 1 | 1.0671 | 1.0671 to 1.0671 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | plasma | leukemia | 3 | 1.0056 | 0.95979 to 1.0864 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | leukemia | 2 | 0.95645 | 0.95092 to 0.96198 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | plasma | leukemia | 2 | 1.0018 | 0.9865 to 1.0172 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | plasma | leukemia | 2 | 1.0369 | 0.99937 to 1.0744 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | plasma | leukemia | 1 | 1.0688 | 1.0688 to 1.0688 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | plasma | leukemia | 1 | 0.9918 | 0.9918 to 0.9918 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | not stated | leukemia | 28 | 0.74391 | 0.61415 to 0.83308 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | not stated | leukemia | 28 | 0.66314 | 0.59236 to 0.73647 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | not stated | leukemia | 28 | 0.35356 | 0.27938 to 0.56103 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | not stated | leukemia | 28 | 0.81828 | 0.65836 to 0.93061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | not stated | leukemia | 28 | 0.75338 | 0.62409 to 0.84997 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | not stated | leukemia | 28 | 0.2431 | 0.19645 to 0.38771 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | not stated | leukemia | 28 | 0.63109 | 0.46895 to 0.74208 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (not stated) | not stated | leukemia | 28 | 0.903 | 0.81963 to 0.96384 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | not stated | leukemia | 28 | 0.9972 | 0.99157 to 1.0035 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | not stated | leukemia | 28 | 1.0053 | 0.92173 to 1.0627 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | not stated | leukemia | 28 | 1.0074 | 0.91587 to 1.0948 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | not stated | leukemia | 21 | 0.9968 | 0.9005 to 1.0871 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | not stated | leukemia | 5 | 1.0093 | 0.93838 to 1.1024 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | not stated | leukemia | 1 | 0.9446 | 0.9446 to 0.9446 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | not stated | leukemia | 10 | 1.0006 | 0.92166 to 1.1367 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | not stated | leukemia | 10 | 0.99485 | 0.90026 to 1.0611 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | not stated | leukemia | 21 | 1.0003 | 0.9425 to 1.0587 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon fibroblasts | not stated | leukemia | 7 | 1.0073 | 0.92313 to 1.0561 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | not stated | leukemia | 20 | 1.0191 | 0.94798 to 1.1631 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | not stated | leukemia | 1 | 1.0267 | 1.0267 to 1.0267 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | not stated | leukemia | 17 | 0.9855 | 0.8926 to 1.0421 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | not stated | leukemia | 1 | 1.1054 | 1.1054 to 1.1054 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | not stated | leukemia | 1 | 1.0166 | 1.0166 to 1.0166 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | not stated | leukemia | 2 | 0.9702 | 0.94077 to 0.99963 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tongue epithelium | not stated | leukemia | 1 | 0.9993 | 0.9993 to 0.9993 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | not stated | leukemia | 5 | 0.9749 | 0.91532 to 1.0153 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | control line | 1 | 0.51762 | 0.51762 to 0.51762 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | control line | 1 | 0.54608 | 0.54608 to 0.54608 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | control line | 1 | 0.5516 | 0.5516 to 0.5516 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | control line | 1 | 0.51879 | 0.51879 to 0.51879 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | control line | 1 | 0.51191 | 0.51191 to 0.51191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | control line | 1 | 0.42341 | 0.42341 to 0.42341 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | control line | 1 | 0.43707 | 0.43707 to 0.43707 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | control line | 1 | 0.7654 | 0.7654 to 0.7654 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | control line | 1 | 0.9988 | 0.9988 to 0.9988 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | plasma | control line | 1 | 0.9965 | 0.9965 to 0.9965 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | plasma | control line | 1 | 1.082 | 1.082 to 1.082 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | plasma | control line | 1 | 0.9548 | 0.9548 to 0.9548 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | control line | 1 | 1.0114 | 1.0114 to 1.0114 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | plasma | control line | 1 | 1.0573 | 1.0573 to 1.0573 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | plasma | control line | 1 | 0.9636 | 0.9636 to 0.9636 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | plasma | control line | 1 | 0.9287 | 0.9287 to 0.9287 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | plasma | control line | 1 | 0.9389 | 0.9389 to 0.9389 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | plasma | control line | 1 | 0.9177 | 0.9177 to 0.9177 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | knockout | 1 | 1.0775 | 1.0775 to 1.0775 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | knockout | 1 | 1.15 | 1.15 to 1.15 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | knockout | 1 | 0.71473 | 0.71473 to 0.71473 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | knockout | 1 | 1.105 | 1.105 to 1.105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | knockout | 1 | 1.0913 | 1.0913 to 1.0913 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | knockout | 1 | 0.52802 | 0.52802 to 0.52802 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | knockout | 1 | 0.83695 | 0.83695 to 0.83695 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | knockout | 1 | 1.2618 | 1.2618 to 1.2618 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | knockout | 1 | 1.0024 | 1.0024 to 1.0024 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | knockout | 1 | 1.0736 | 1.0736 to 1.0736 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | plasma | knockout | 1 | 0.9437 | 0.9437 to 0.9437 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | knockout | 1 | 0.9512 | 0.9512 to 0.9512 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | plasma | knockout | 1 | 1.0706 | 1.0706 to 1.0706 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | plasma | knockout | 1 | 1.1043 | 1.1043 to 1.1043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | plasma | knockout | 1 | 1.0212 | 1.0212 to 1.0212 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | cord-blood B cells | 3 | 0.70302 | 0.69898 to 0.73942 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | cord-blood B cells | 3 | 0.58503 | 0.55325 to 0.60002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | cord-blood B cells | 3 | 0.21721 | 0.19201 to 0.35815 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | cord-blood B cells | 3 | 0.79448 | 0.75862 to 0.85068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | cord-blood B cells | 3 | 0.72874 | 0.71622 to 0.76389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | cord-blood B cells | 3 | 0.16695 | 0.14812 to 0.23822 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | cord-blood B cells | 3 | 0.65717 | 0.617 to 0.68542 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | cord-blood B cells | 3 | 0.7906 | 0.77116 to 0.79411 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | cord-blood B cells | 3 | 0.9966 | 0.99111 to 1.0049 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cardiomyocyte | plasma | cord-blood B cells | 2 | 1.0173 | 1.0135 to 1.0212 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | plasma | cord-blood B cells | 2 | 1.0366 | 0.99762 to 1.0755 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | cord-blood B cells | 3 | 1.0113 | 0.9726 to 1.0136 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | plasma | cord-blood B cells | 3 | 1.041 | 0.9996 to 1.0907 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | plasma | cord-blood B cells | 3 | 1.0055 | 0.99434 to 1.0509 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | plasma | cord-blood B cells | 2 | 0.96895 | 0.93885 to 0.99905 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | not stated | leukemia | 1 | 1.005 | 1.005 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | not stated | leukemia | 1 | 1.0302 | 1.0302 to 1.0302 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted b cells | leukemia | 13 | 0.79849 | 0.73379 to 0.90462 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted b cells | leukemia | 13 | 0.72463 | 0.66439 to 0.8066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted b cells | leukemia | 13 | 0.45802 | 0.28467 to 0.53007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted b cells | leukemia | 13 | 0.86641 | 0.77743 to 1.0568 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted b cells | leukemia | 13 | 0.8142 | 0.74186 to 0.9192 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted b cells | leukemia | 13 | 5.5245 | 4.5633 to 7.8091 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted b cells | leukemia | 13 | 0.64439 | 0.51326 to 0.86324 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | b cells | sorted b cells | leukemia | 11 | 0.9035 | 0.82735 to 0.93225 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Coupling-A | coupling-A | b cells | sorted b cells | leukemia | 13 | 1.0917 | 1.0325 to 1.1138 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Met-A | Met-A sorted cell (atlas) | b cells | sorted b cells | leukemia | 13 | 1.5055 | 1.3983 to 1.7442 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted b cells | leukemia | 13 | 0.9955 | 0.99236 to 1.0055 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted b cells | leukemia | 13 | 0.9665 | 0.89758 to 1.0479 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted adipocytes | healthy | 3 | 1.1307 | 1.1256 to 1.1386 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted adipocytes | healthy | 3 | 1.1004 | 1.1003 to 1.1054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted adipocytes | healthy | 3 | 0.21125 | 0.19427 to 0.25929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted adipocytes | healthy | 3 | 1.3825 | 1.3583 to 1.4132 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted adipocytes | healthy | 3 | 1.1553 | 1.1489 to 1.1613 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted adipocytes | healthy | 3 | 0.1903 | 0.17525 to 0.23058 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted adipocytes | healthy | 3 | 1.2238 | 1.1977 to 1.2521 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | adipocytes | sorted adipocytes | healthy | 3 | 1.0094 | 1.0066 to 1.0226 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | adipocytes | sorted adipocytes | healthy | 3 | 0.84396 | 0.8332 to 0.86829 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted adipocytes | healthy | 3 | 0.9999 | 0.99558 to 1.0032 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted adipocytes | healthy | 3 | 0.9343 | 0.87805 to 0.95698 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted adipocytes | healthy | 3 | 1.0135 | 0.99631 to 1.0504 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted adipocytes | healthy | 3 | 0.9882 | 0.9873 to 0.99081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted adipocytes | healthy | 3 | 1.0475 | 1.0294 to 1.1217 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted adipocytes | healthy | 3 | 0.9872 | 0.97766 to 1.0372 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted aorta endothelium | healthy | 2 | 1.0491 | 1.0265 to 1.0717 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted aorta endothelium | healthy | 2 | 1.0207 | 0.99208 to 1.0492 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted aorta endothelium | healthy | 2 | 0.19412 | 0.19013 to 0.19812 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted aorta endothelium | healthy | 2 | 1.2872 | 1.2868 to 1.2877 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted aorta endothelium | healthy | 2 | 1.0694 | 1.0422 to 1.0965 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted aorta endothelium | healthy | 2 | 0.16711 | 0.16435 to 0.16986 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted aorta endothelium | healthy | 2 | 1.124 | 1.1161 to 1.1319 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | aorta endothelium | sorted aorta endothelium | healthy | 2 | 0.9585 | 0.92322 to 0.99378 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | aorta endothelium | sorted aorta endothelium | healthy | 2 | 1.0098 | 0.98689 to 1.0327 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted aorta endothelium | healthy | 2 | 0.995 | 0.99212 to 0.99788 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted aorta endothelium | healthy | 2 | 0.93285 | 0.90571 to 0.95998 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted aorta endothelium | healthy | 2 | 0.9937 | 0.93952 to 1.0479 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted aorta endothelium | healthy | 2 | 0.993 | 0.98256 to 1.0034 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted aorta endothelium | healthy | 2 | 1.0677 | 1.0637 to 1.0717 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted aorta endothelium | healthy | 2 | 0.973 | 0.93502 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted vascular saphenous endothelium | healthy | 3 | 0.98006 | 0.94923 to 1.0651 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted vascular saphenous endothelium | healthy | 3 | 0.95501 | 0.91322 to 1.0462 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted vascular saphenous endothelium | healthy | 3 | 0.21979 | 0.21814 to 0.22207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted vascular saphenous endothelium | healthy | 3 | 1.1617 | 1.1249 to 1.2954 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted vascular saphenous endothelium | healthy | 3 | 1.0037 | 0.97216 to 1.0861 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted vascular saphenous endothelium | healthy | 3 | 0.19606 | 0.19009 to 0.19907 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted vascular saphenous endothelium | healthy | 3 | 1.0272 | 0.99958 to 1.1296 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | vascular saphenous endothelium | sorted vascular saphenous endothelium | healthy | 3 | 1.0054 | 0.99883 to 1.0373 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | vascular saphenous endothelium | sorted vascular saphenous endothelium | healthy | 3 | 0.7461 | 0.73658 to 0.77845 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted vascular saphenous endothelium | healthy | 3 | 1.0013 | 1.0011 to 1.0035 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted vascular saphenous endothelium | healthy | 3 | 0.8744 | 0.85829 to 0.94181 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted vascular saphenous endothelium | healthy | 3 | 0.9609 | 0.93057 to 0.97287 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted vascular saphenous endothelium | healthy | 3 | 1.0013 | 0.96413 to 1.0152 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted vascular saphenous endothelium | healthy | 3 | 1.0672 | 1.0563 to 1.0749 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted vascular saphenous endothelium | healthy | 3 | 1.0262 | 0.98219 to 1.0394 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney glomerular endothelium | healthy | 3 | 1.0041 | 0.92378 to 1.0208 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney glomerular endothelium | healthy | 3 | 0.98404 | 0.89098 to 1.0008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney glomerular endothelium | healthy | 3 | 0.2206 | 0.2025 to 0.22155 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney glomerular endothelium | healthy | 3 | 1.1786 | 1.0995 to 1.2385 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney glomerular endothelium | healthy | 3 | 1.0299 | 0.9448 to 1.0423 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney glomerular endothelium | healthy | 3 | 0.19803 | 0.17861 to 0.1981 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney glomerular endothelium | healthy | 3 | 1.0479 | 0.97879 to 1.0887 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney glomerular endothelium | sorted kidney glomerular endothelium | healthy | 3 | 1.0349 | 1.0038 to 1.0443 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney glomerular endothelium | sorted kidney glomerular endothelium | healthy | 3 | 0.75493 | 0.73754 to 0.75883 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney glomerular endothelium | healthy | 3 | 0.9994 | 0.99787 to 1.0025 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney glomerular endothelium | healthy | 3 | 0.8897 | 0.88691 to 0.91166 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney glomerular endothelium | healthy | 3 | 1.0012 | 0.9841 to 1.0158 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney glomerular endothelium | healthy | 3 | 1.0199 | 0.98759 to 1.036 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney glomerular endothelium | healthy | 3 | 1.0164 | 0.96555 to 1.0277 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney glomerular endothelium | healthy | 3 | 0.9561 | 0.94323 to 0.97662 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney tubular endothelium | healthy | 3 | 0.93373 | 0.93265 to 0.9478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney tubular endothelium | healthy | 3 | 0.90007 | 0.89462 to 0.92003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney tubular endothelium | healthy | 3 | 0.22317 | 0.21262 to 0.22736 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney tubular endothelium | healthy | 3 | 1.1052 | 1.0905 to 1.118 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney tubular endothelium | healthy | 3 | 0.95543 | 0.95527 to 0.97079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney tubular endothelium | healthy | 3 | 0.18591 | 0.18232 to 0.20247 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney tubular endothelium | healthy | 3 | 0.97263 | 0.93729 to 0.98771 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney tubular endothelium | sorted kidney tubular endothelium | healthy | 3 | 1.0255 | 1.0241 to 1.027 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney tubular endothelium | sorted kidney tubular endothelium | healthy | 3 | 0.77506 | 0.72703 to 0.80766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney tubular endothelium | healthy | 3 | 1.0001 | 0.99848 to 1.0022 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney tubular endothelium | healthy | 3 | 0.9269 | 0.90656 to 0.98558 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney tubular endothelium | healthy | 3 | 1.0149 | 0.97692 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney tubular endothelium | healthy | 3 | 1.0034 | 0.9863 to 1.0443 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney tubular endothelium | healthy | 3 | 1.0011 | 0.99759 to 1.0245 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney tubular endothelium | healthy | 3 | 1.0049 | 0.97916 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar endothelium | healthy | 3 | 1.0395 | 1.0248 to 1.0449 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar endothelium | healthy | 3 | 1.0195 | 0.99242 to 1.0243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar endothelium | healthy | 3 | 0.22615 | 0.19415 to 0.23106 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar endothelium | healthy | 3 | 1.2281 | 1.2124 to 1.3068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar endothelium | healthy | 3 | 1.0534 | 1.0483 to 1.0707 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar endothelium | healthy | 3 | 0.19838 | 0.16947 to 0.20727 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar endothelium | healthy | 3 | 1.0908 | 1.0747 to 1.1584 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar endothelium | sorted lung alveolar endothelium | healthy | 3 | 0.9922 | 0.9337 to 1.0062 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar endothelium | sorted lung alveolar endothelium | healthy | 3 | 0.7626 | 0.7536 to 0.76541 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar endothelium | healthy | 3 | 1.0006 | 1.0002 to 1.0097 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar endothelium | healthy | 3 | 0.9146 | 0.85961 to 0.94727 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar endothelium | healthy | 3 | 0.9704 | 0.91343 to 1.0442 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar endothelium | healthy | 3 | 0.9419 | 0.93974 to 1.0252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung alveolar endothelium | healthy | 3 | 1.0399 | 1.0161 to 1.0518 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung alveolar endothelium | healthy | 3 | 0.9676 | 0.96544 to 1.0083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreas endothelium | healthy | 3 | 1.0746 | 1.0663 to 1.0799 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreas endothelium | healthy | 3 | 1.0607 | 1.0483 to 1.0613 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreas endothelium | healthy | 3 | 0.19587 | 0.19491 to 0.2004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreas endothelium | healthy | 3 | 1.2926 | 1.2861 to 1.3059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreas endothelium | healthy | 3 | 1.0986 | 1.0887 to 1.1047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreas endothelium | healthy | 3 | 0.16496 | 0.16147 to 0.17262 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreas endothelium | healthy | 3 | 1.0939 | 1.0919 to 1.1212 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreas endothelium | sorted pancreas endothelium | healthy | 3 | 1.0506 | 1.0085 to 1.0528 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreas endothelium | sorted pancreas endothelium | healthy | 3 | 0.84448 | 0.789 to 0.84936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreas endothelium | healthy | 3 | 1.0001 | 0.99605 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreas endothelium | healthy | 3 | 0.9662 | 0.94766 to 0.99392 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreas endothelium | healthy | 3 | 1.0065 | 0.95286 to 1.0172 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreas endothelium | healthy | 3 | 0.9871 | 0.93409 to 1.0189 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreas endothelium | healthy | 3 | 1.0404 | 0.98775 to 1.0426 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreas endothelium | healthy | 3 | 0.984 | 0.97671 to 1.0815 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon fibroblasts | healthy | 2 | 1.2186 | 1.2067 to 1.2304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon fibroblasts | healthy | 2 | 1.1856 | 1.1824 to 1.1888 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon fibroblasts | healthy | 2 | 0.22609 | 0.22578 to 0.22641 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon fibroblasts | healthy | 2 | 1.4745 | 1.4733 to 1.4756 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon fibroblasts | healthy | 2 | 1.2471 | 1.2323 to 1.2619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon fibroblasts | healthy | 2 | 0.20316 | 0.20296 to 0.20337 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon fibroblasts | healthy | 2 | 1.313 | 1.3088 to 1.3172 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon fibroblasts | sorted colon fibroblasts | healthy | 2 | 0.9906 | 0.98385 to 0.99735 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon fibroblasts | sorted colon fibroblasts | healthy | 2 | 0.92575 | 0.92285 to 0.92865 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon fibroblasts | healthy | 2 | 0.99295 | 0.99164 to 0.99425 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon fibroblasts | healthy | 2 | 0.9425 | 0.92135 to 0.96365 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon fibroblasts | healthy | 2 | 1.0226 | 1.0173 to 1.0279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon fibroblasts | healthy | 2 | 0.9631 | 0.95896 to 0.96724 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon fibroblasts | healthy | 2 | 1.0599 | 1.0395 to 1.0803 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon fibroblasts | healthy | 2 | 0.9867 | 0.96762 to 1.0058 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted heart fibroblasts | healthy | 3 | 1.1879 | 1.1527 to 1.2122 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted heart fibroblasts | healthy | 3 | 1.1973 | 1.1649 to 1.2153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted heart fibroblasts | healthy | 3 | 0.2028 | 0.18576 to 0.20312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted heart fibroblasts | healthy | 3 | 1.6459 | 1.5799 to 1.6543 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted heart fibroblasts | healthy | 3 | 1.1811 | 1.1491 to 1.2118 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted heart fibroblasts | healthy | 3 | 0.17747 | 0.16003 to 0.17868 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted heart fibroblasts | healthy | 3 | 1.4417 | 1.3876 to 1.4515 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | heart fibroblasts | sorted heart fibroblasts | healthy | 3 | 0.8859 | 0.88473 to 0.89508 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | heart fibroblasts | sorted heart fibroblasts | healthy | 3 | 0.75967 | 0.74465 to 0.76502 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted heart fibroblasts | healthy | 3 | 0.991 | 0.98884 to 0.99577 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted heart fibroblasts | healthy | 3 | 0.9013 | 0.8797 to 0.93964 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted heart fibroblasts | healthy | 3 | 0.9246 | 0.92226 to 0.99561 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted heart fibroblasts | healthy | 3 | 0.956 | 0.947 to 0.9758 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted heart fibroblasts | healthy | 3 | 1.0599 | 1.0176 to 1.0807 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted heart fibroblasts | healthy | 3 | 1.0198 | 0.98632 to 1.0251 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted striated muscle | healthy | 2 | 1.1593 | 1.154 to 1.1646 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted striated muscle | healthy | 2 | 1.1337 | 1.129 to 1.1384 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted striated muscle | healthy | 2 | 0.18732 | 0.18301 to 0.19163 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted striated muscle | healthy | 2 | 1.4014 | 1.3887 to 1.4141 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted striated muscle | healthy | 2 | 1.1869 | 1.1817 to 1.1921 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted striated muscle | healthy | 2 | 0.16335 | 0.16037 to 0.16633 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted striated muscle | healthy | 2 | 1.2333 | 1.2207 to 1.2458 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | striated muscle | sorted striated muscle | healthy | 2 | 1.0544 | 1.0423 to 1.0665 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | striated muscle | sorted striated muscle | healthy | 2 | 0.9128 | 0.90499 to 0.9206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted striated muscle | healthy | 2 | 0.996 | 0.99546 to 0.99654 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted striated muscle | healthy | 2 | 0.9691 | 0.96298 to 0.97522 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted striated muscle | healthy | 2 | 0.96045 | 0.95959 to 0.96131 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted striated muscle | healthy | 2 | 0.9923 | 0.98375 to 1.0009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted striated muscle | healthy | 2 | 1.0621 | 1.0244 to 1.0998 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted striated muscle | healthy | 2 | 0.98805 | 0.93607 to 1.04 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted smooth muscle | healthy | 3 | 1.29 | 1.085 to 1.3152 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted smooth muscle | healthy | 3 | 1.3257 | 1.1003 to 1.3357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted smooth muscle | healthy | 3 | 0.20347 | 0.19849 to 0.23202 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted smooth muscle | healthy | 3 | 1.8064 | 1.4338 to 1.811 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted smooth muscle | healthy | 3 | 1.2797 | 1.0871 to 1.3124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted smooth muscle | healthy | 3 | 0.17877 | 0.16969 to 0.20444 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted smooth muscle | healthy | 3 | 1.5675 | 1.2657 to 1.5855 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | smooth muscle | sorted smooth muscle | healthy | 3 | 0.874 | 0.86077 to 0.89947 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | smooth muscle | sorted smooth muscle | healthy | 3 | 0.86884 | 0.75982 to 0.8782 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted smooth muscle | healthy | 3 | 0.9996 | 0.99465 to 1.0005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted smooth muscle | healthy | 3 | 0.9969 | 0.94803 to 0.99879 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted smooth muscle | healthy | 3 | 0.9612 | 0.93996 to 1.0325 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted smooth muscle | healthy | 3 | 0.9903 | 0.94008 to 1.0453 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted smooth muscle | healthy | 3 | 1.0675 | 1.0149 to 1.0761 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted smooth muscle | healthy | 3 | 0.9851 | 0.91337 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cardiomyocyte | healthy | 3 | 0.86952 | 0.84146 to 0.87643 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cardiomyocyte | healthy | 3 | 0.85 | 0.82294 to 0.85459 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cardiomyocyte | healthy | 3 | 0.20048 | 0.19976 to 0.20404 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cardiomyocyte | healthy | 3 | 1.0653 | 1.0291 to 1.0819 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cardiomyocyte | healthy | 3 | 0.88425 | 0.85574 to 0.89048 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cardiomyocyte | healthy | 3 | 0.17798 | 0.17706 to 0.17911 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cardiomyocyte | healthy | 3 | 0.92054 | 0.89991 to 0.94528 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cardiomyocyte | sorted cardiomyocyte | healthy | 3 | 0.9498 | 0.94773 to 0.95259 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cardiomyocyte | sorted cardiomyocyte | healthy | 3 | 0.74678 | 0.73017 to 0.76684 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cardiomyocyte | healthy | 3 | 1.0033 | 1.0018 to 1.0042 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted cardiomyocyte | healthy | 3 | 1.0029 | 0.9858 to 1.0478 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted cardiomyocyte | healthy | 3 | 0.9541 | 0.94447 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted cardiomyocyte | healthy | 3 | 0.9712 | 0.94276 to 0.97642 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cardiomyocyte | healthy | 3 | 1.0893 | 1.0351 to 1.0955 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted cardiomyocyte | healthy | 3 | 0.9944 | 0.93977 to 1.009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted oligodendrocytes | healthy | 3 | 0.83856 | 0.82549 to 0.85003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted oligodendrocytes | healthy | 3 | 0.83081 | 0.82071 to 0.83638 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted oligodendrocytes | healthy | 3 | 0.20053 | 0.17711 to 0.2262 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted oligodendrocytes | healthy | 3 | 0.99751 | 0.97586 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted oligodendrocytes | healthy | 3 | 0.85535 | 0.84187 to 0.87182 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted oligodendrocytes | healthy | 3 | 0.17452 | 0.15237 to 0.19937 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted oligodendrocytes | healthy | 3 | 0.85476 | 0.85194 to 0.86353 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | oligodendrocytes | sorted oligodendrocytes | healthy | 3 | 0.9154 | 0.90793 to 0.94402 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | oligodendrocytes | sorted oligodendrocytes | healthy | 3 | 0.755 | 0.72503 to 0.75703 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted oligodendrocytes | healthy | 3 | 0.9937 | 0.99343 to 1.0007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted oligodendrocytes | healthy | 3 | 0.9375 | 0.91401 to 0.9969 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted oligodendrocytes | healthy | 3 | 0.9404 | 0.92654 to 0.94292 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted oligodendrocytes | healthy | 3 | 0.9635 | 0.95018 to 1.0633 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted oligodendrocytes | healthy | 3 | 1.01 | 1.0041 to 1.0144 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted oligodendrocytes | healthy | 3 | 0.9918 | 0.9774 to 0.99702 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cortical neurons | healthy | 3 | 0.82649 | 0.8023 to 0.97304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cortical neurons | healthy | 3 | 0.82818 | 0.80627 to 0.96632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cortical neurons | healthy | 3 | 0.21262 | 0.21036 to 0.24822 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cortical neurons | healthy | 3 | 0.97949 | 0.91906 to 1.094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cortical neurons | healthy | 3 | 0.84274 | 0.82285 to 1.0021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cortical neurons | healthy | 3 | 0.18478 | 0.18348 to 0.21471 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cortical neurons | healthy | 3 | 0.84706 | 0.7924 to 0.9396 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cortical neurons | sorted cortical neurons | healthy | 3 | 0.7885 | 0.7714 to 0.84349 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cortical neurons | sorted cortical neurons | healthy | 3 | 0.86063 | 0.77314 to 0.92823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cortical neurons | healthy | 3 | 0.9971 | 0.99467 to 0.99773 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted cortical neurons | healthy | 3 | 0.9233 | 0.89063 to 0.95165 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted cortical neurons | healthy | 3 | 0.9644 | 0.91841 to 0.97817 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted cortical neurons | healthy | 3 | 0.9789 | 0.9384 to 0.98205 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cortical neurons | healthy | 3 | 0.9812 | 0.95114 to 1.0623 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted cortical neurons | healthy | 3 | 0.949 | 0.89968 to 0.95809 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted hepatocyte | healthy | 3 | 0.87292 | 0.86797 to 0.88275 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted hepatocyte | healthy | 3 | 0.85959 | 0.84181 to 0.86479 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted hepatocyte | healthy | 3 | 0.23173 | 0.21087 to 0.24269 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted hepatocyte | healthy | 3 | 1.0742 | 1.0293 to 1.0824 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted hepatocyte | healthy | 3 | 0.88535 | 0.88513 to 0.8969 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted hepatocyte | healthy | 3 | 0.20486 | 0.1884 to 0.21671 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted hepatocyte | healthy | 3 | 0.95029 | 0.91858 to 0.96153 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | hepatocyte | sorted hepatocyte | healthy | 3 | 0.8843 | 0.87755 to 0.9239 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | hepatocyte | sorted hepatocyte | healthy | 3 | 0.80683 | 0.79288 to 0.82108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted hepatocyte | healthy | 3 | 1.0039 | 0.99994 to 1.0063 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted hepatocyte | healthy | 3 | 0.9686 | 0.95681 to 0.96959 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted hepatocyte | healthy | 3 | 1.0265 | 1.0222 to 1.0558 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted hepatocyte | healthy | 3 | 0.9472 | 0.9391 to 0.96151 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted hepatocyte | healthy | 3 | 0.9805 | 0.97546 to 0.99931 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted hepatocyte | healthy | 3 | 0.9268 | 0.89053 to 0.9646 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic duct | healthy | 3 | 0.82873 | 0.80252 to 0.85431 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic duct | healthy | 3 | 0.82171 | 0.76912 to 0.8234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic duct | healthy | 3 | 0.18355 | 0.17432 to 0.18784 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic duct | healthy | 3 | 0.96697 | 0.93048 to 1.0075 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic duct | healthy | 3 | 0.84969 | 0.82458 to 0.87658 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic duct | healthy | 3 | 0.15895 | 0.15283 to 0.1605 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic duct | healthy | 3 | 0.81913 | 0.80996 to 0.8787 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic duct | sorted pancreatic duct | healthy | 3 | 0.9588 | 0.95007 to 0.97464 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic duct | sorted pancreatic duct | healthy | 3 | 0.81583 | 0.77192 to 0.82161 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic duct | healthy | 3 | 0.9985 | 0.99499 to 0.99985 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic duct | healthy | 3 | 0.9492 | 0.94101 to 1.0037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic duct | healthy | 3 | 0.9488 | 0.93107 to 0.97265 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic duct | healthy | 3 | 0.9724 | 0.96205 to 1.0431 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic duct | healthy | 3 | 1.0343 | 1.0032 to 1.0529 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic duct | healthy | 3 | 0.9548 | 0.94589 to 1.0649 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic acinar | healthy | 3 | 1.0857 | 1.0274 to 1.0901 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic acinar | healthy | 3 | 1.0859 | 1.0469 to 1.1067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic acinar | healthy | 3 | 0.17636 | 0.16645 to 0.18558 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic acinar | healthy | 3 | 1.4502 | 1.3666 to 1.4619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic acinar | healthy | 3 | 1.0865 | 1.0309 to 1.0998 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic acinar | healthy | 3 | 0.1521 | 0.1521 to 0.16797 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic acinar | healthy | 3 | 1.2394 | 1.219 to 1.2725 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic acinar | sorted pancreatic acinar | healthy | 3 | 0.8983 | 0.8263 to 0.91207 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic acinar | sorted pancreatic acinar | healthy | 3 | 0.84536 | 0.78105 to 0.86192 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic acinar | healthy | 3 | 1 | 0.9991 to 1.0043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic acinar | healthy | 3 | 0.9336 | 0.93279 to 0.99678 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic acinar | healthy | 3 | 0.9456 | 0.9312 to 0.97872 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic acinar | healthy | 3 | 0.9661 | 0.94333 to 0.97465 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic acinar | healthy | 3 | 1.1509 | 1.088 to 1.1513 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic acinar | healthy | 3 | 1.0167 | 0.98043 to 1.0195 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic delta | healthy | 3 | 0.97647 | 0.93519 to 0.98964 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic delta | healthy | 3 | 0.96605 | 0.92011 to 0.98264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic delta | healthy | 3 | 0.20163 | 0.20046 to 0.20245 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic delta | healthy | 3 | 1.1804 | 1.118 to 1.2066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic delta | healthy | 3 | 0.99534 | 0.95568 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic delta | healthy | 3 | 0.16748 | 0.1652 to 0.17472 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic delta | healthy | 3 | 0.99305 | 0.96876 to 1.0154 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic delta | sorted pancreatic delta | healthy | 3 | 1.0545 | 1.0232 to 1.0596 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic delta | sorted pancreatic delta | healthy | 3 | 0.79697 | 0.73754 to 0.79783 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic delta | healthy | 3 | 0.9963 | 0.99522 to 0.99846 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic delta | healthy | 3 | 0.9674 | 0.9539 to 0.98225 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic delta | healthy | 3 | 1.0059 | 0.95847 to 1.0113 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic delta | healthy | 3 | 0.9904 | 0.97339 to 1.0827 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic delta | healthy | 3 | 0.9624 | 0.93927 to 1.0061 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic delta | healthy | 1 | 1.0117 | 1.0117 to 1.0117 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic beta | healthy | 3 | 0.92301 | 0.89563 to 0.95496 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic beta | healthy | 3 | 0.90098 | 0.87341 to 0.93335 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic beta | healthy | 3 | 0.21266 | 0.21026 to 0.21594 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic beta | healthy | 3 | 1.0921 | 1.0524 to 1.1525 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic beta | healthy | 3 | 0.94378 | 0.91659 to 0.97389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic beta | healthy | 3 | 0.18123 | 0.18111 to 0.18573 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic beta | healthy | 3 | 0.93854 | 0.90767 to 0.99778 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic beta | sorted pancreatic beta | healthy | 3 | 1.0103 | 0.99311 to 1.0305 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic beta | sorted pancreatic beta | healthy | 3 | 0.84695 | 0.82568 to 0.85172 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic beta | healthy | 3 | 1.0022 | 0.99833 to 1.0041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic beta | healthy | 3 | 0.9815 | 0.95621 to 0.99725 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic beta | healthy | 3 | 1.0083 | 0.97104 to 1.0374 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic beta | healthy | 3 | 1.0592 | 1.0345 to 1.084 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic beta | healthy | 3 | 0.9401 | 0.93272 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic alpha | healthy | 3 | 0.93361 | 0.90972 to 0.93395 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic alpha | healthy | 3 | 0.90664 | 0.89048 to 0.90821 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic alpha | healthy | 3 | 0.22261 | 0.22193 to 0.22778 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic alpha | healthy | 3 | 1.1384 | 1.0996 to 1.1386 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic alpha | healthy | 3 | 0.94974 | 0.92589 to 0.94984 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic alpha | healthy | 3 | 0.19633 | 0.19317 to 0.1989 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic alpha | healthy | 3 | 0.99138 | 0.95768 to 1.0018 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic alpha | sorted pancreatic alpha | healthy | 3 | 1.0253 | 1.011 to 1.028 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic alpha | sorted pancreatic alpha | healthy | 3 | 0.74826 | 0.72824 to 0.75265 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic alpha | healthy | 3 | 1.0042 | 1.0001 to 1.0045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic alpha | healthy | 3 | 0.9785 | 0.96617 to 0.99443 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic alpha | healthy | 3 | 0.9537 | 0.86127 to 0.9591 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic alpha | healthy | 3 | 1.0416 | 1.0139 to 1.0454 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic alpha | healthy | 3 | 1.0099 | 0.98515 to 1.0448 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney glomerular epithelium | healthy | 2 | 0.9116 | 0.8822 to 0.941 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney glomerular epithelium | healthy | 2 | 0.89721 | 0.86962 to 0.9248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney glomerular epithelium | healthy | 2 | 0.18862 | 0.18795 to 0.1893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney glomerular epithelium | healthy | 2 | 1.0697 | 1.0432 to 1.0962 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney glomerular epithelium | healthy | 2 | 0.93491 | 0.90363 to 0.96619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney glomerular epithelium | healthy | 2 | 0.16282 | 0.15927 to 0.16637 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney glomerular epithelium | healthy | 2 | 0.92777 | 0.91007 to 0.94547 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney glomerular epithelium | sorted kidney glomerular epithelium | healthy | 2 | 0.992 | 0.9821 to 1.0019 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney glomerular epithelium | sorted kidney glomerular epithelium | healthy | 2 | 0.94581 | 0.91559 to 0.97602 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney glomerular epithelium | healthy | 2 | 0.99985 | 0.99926 to 1.0004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney glomerular epithelium | healthy | 2 | 0.9497 | 0.92891 to 0.97049 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney glomerular epithelium | healthy | 2 | 0.9899 | 0.98027 to 0.99953 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney glomerular epithelium | healthy | 2 | 1.0567 | 1.0355 to 1.0778 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney glomerular epithelium | healthy | 2 | 1.012 | 1.0113 to 1.0128 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney glomerular epithelium | healthy | 2 | 0.97025 | 0.95976 to 0.98074 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney tubular epithelium | healthy | 3 | 0.86989 | 0.81368 to 0.91878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney tubular epithelium | healthy | 3 | 0.84866 | 0.78145 to 0.90549 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney tubular epithelium | healthy | 3 | 0.20803 | 0.19092 to 0.22228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney tubular epithelium | healthy | 3 | 1.0377 | 0.91869 to 1.0948 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney tubular epithelium | healthy | 3 | 0.88991 | 0.83793 to 0.93923 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney tubular epithelium | healthy | 3 | 0.18345 | 0.17001 to 0.19852 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney tubular epithelium | healthy | 3 | 0.9146 | 0.81121 to 0.95794 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney tubular epithelium | sorted kidney tubular epithelium | healthy | 3 | 0.9737 | 0.90791 to 0.98927 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney tubular epithelium | sorted kidney tubular epithelium | healthy | 3 | 0.73804 | 0.71985 to 0.75721 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney tubular epithelium | healthy | 3 | 0.9985 | 0.99364 to 1.0049 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney tubular epithelium | healthy | 3 | 0.9666 | 0.93618 to 1.0088 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney tubular epithelium | healthy | 2 | 0.95635 | 0.94929 to 0.96341 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney tubular epithelium | healthy | 3 | 1.0195 | 1.0029 to 1.0409 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney tubular epithelium | healthy | 3 | 1.0614 | 1.0052 to 1.1077 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney tubular epithelium | healthy | 3 | 0.9931 | 0.94558 to 1.0169 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted thyroid epithelium | healthy | 3 | 0.80298 | 0.79585 to 0.80671 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted thyroid epithelium | healthy | 3 | 0.76842 | 0.76184 to 0.77181 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted thyroid epithelium | healthy | 3 | 0.21121 | 0.20887 to 0.22128 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted thyroid epithelium | healthy | 3 | 0.92621 | 0.91871 to 0.93952 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted thyroid epithelium | healthy | 3 | 0.82303 | 0.81597 to 0.82637 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted thyroid epithelium | healthy | 3 | 0.18363 | 0.18306 to 0.1939 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted thyroid epithelium | healthy | 3 | 0.81225 | 0.80384 to 0.82376 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | thyroid epithelium | sorted thyroid epithelium | healthy | 3 | 0.9983 | 0.99713 to 1.0097 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | thyroid epithelium | sorted thyroid epithelium | healthy | 3 | 0.73552 | 0.73507 to 0.73983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted thyroid epithelium | healthy | 3 | 1.0013 | 0.99797 to 1.0017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted thyroid epithelium | healthy | 3 | 0.9951 | 0.98178 to 1.0006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted thyroid epithelium | healthy | 3 | 0.9998 | 0.9413 to 1.0118 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted thyroid epithelium | healthy | 3 | 1.0083 | 0.99021 to 1.0444 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted thyroid epithelium | healthy | 3 | 0.9222 | 0.90672 to 0.95919 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted fallopien tubes epithelium | healthy | 3 | 0.83544 | 0.82771 to 0.84597 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted fallopien tubes epithelium | healthy | 3 | 0.80111 | 0.79678 to 0.82946 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted fallopien tubes epithelium | healthy | 3 | 0.19047 | 0.18898 to 0.25476 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted fallopien tubes epithelium | healthy | 3 | 1.0194 | 1.0096 to 1.0232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted fallopien tubes epithelium | healthy | 3 | 0.85 | 0.84316 to 0.85794 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted fallopien tubes epithelium | healthy | 3 | 0.16784 | 0.16269 to 0.21939 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted fallopien tubes epithelium | healthy | 3 | 0.8921 | 0.88696 to 0.89255 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | fallopien tubes epithelium | sorted fallopien tubes epithelium | healthy | 3 | 0.9378 | 0.88722 to 0.95238 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | fallopien tubes epithelium | sorted fallopien tubes epithelium | healthy | 3 | 0.7179 | 0.69125 to 0.73999 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted fallopien tubes epithelium | healthy | 3 | 0.9959 | 0.99221 to 0.9968 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted fallopien tubes epithelium | healthy | 3 | 0.9257 | 0.91481 to 0.94046 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted fallopien tubes epithelium | healthy | 3 | 1.0206 | 0.99072 to 1.023 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted fallopien tubes epithelium | healthy | 3 | 0.9874 | 0.92656 to 1.0163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted fallopien tubes epithelium | healthy | 3 | 1.0212 | 0.99789 to 1.028 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted endometrium epithelium | healthy | 3 | 0.87847 | 0.86832 to 1.0027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted endometrium epithelium | healthy | 3 | 0.86016 | 0.8484 to 0.99149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted endometrium epithelium | healthy | 3 | 0.18939 | 0.17327 to 0.20289 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted endometrium epithelium | healthy | 3 | 1.0587 | 1.0539 to 1.1763 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted endometrium epithelium | healthy | 3 | 0.8971 | 0.8854 to 1.0304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted endometrium epithelium | healthy | 3 | 0.16428 | 0.14328 to 0.18169 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted endometrium epithelium | healthy | 3 | 0.93616 | 0.92567 to 0.99628 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | endometrium epithelium | sorted endometrium epithelium | healthy | 3 | 0.9201 | 0.89112 to 0.93882 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | endometrium epithelium | sorted endometrium epithelium | healthy | 3 | 0.72973 | 0.69227 to 0.81051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted endometrium epithelium | healthy | 3 | 1.0018 | 1.0004 to 1.0077 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted endometrium epithelium | healthy | 3 | 0.9685 | 0.94699 to 0.97498 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted endometrium epithelium | healthy | 3 | 0.9762 | 0.9564 to 0.99348 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted endometrium epithelium | healthy | 3 | 0.9599 | 0.92462 to 1.0027 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted endometrium epithelium | healthy | 3 | 1.0173 | 1.0052 to 1.0612 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted erythrocyte progenitors | healthy | 3 | 1.5553 | 1.5246 to 1.5734 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted erythrocyte progenitors | healthy | 3 | 1.5314 | 1.4848 to 1.5409 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted erythrocyte progenitors | healthy | 3 | 0.18951 | 0.17855 to 0.19994 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted erythrocyte progenitors | healthy | 3 | 2.1646 | 2.116 to 2.2058 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted erythrocyte progenitors | healthy | 3 | 1.5515 | 1.5253 to 1.5711 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted erythrocyte progenitors | healthy | 3 | 0.17212 | 0.14665 to 0.18435 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted erythrocyte progenitors | healthy | 3 | 1.9037 | 1.8187 to 1.9679 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | erythrocyte progenitors | sorted erythrocyte progenitors | healthy | 3 | 0.7374 | 0.72138 to 0.73965 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | erythrocyte progenitors | sorted erythrocyte progenitors | healthy | 3 | 0.74692 | 0.72741 to 0.85149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted erythrocyte progenitors | healthy | 3 | 0.9997 | 0.99889 to 1.0048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted erythrocyte progenitors | healthy | 3 | 1.0375 | 1.0017 to 1.0472 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted erythrocyte progenitors | healthy | 3 | 0.986 | 0.97358 to 0.9878 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted erythrocyte progenitors | healthy | 3 | 0.9045 | 0.89694 to 0.91521 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted erythrocyte progenitors | healthy | 3 | 0.977 | 0.94856 to 1.0118 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted erythrocyte progenitors | healthy | 3 | 0.9167 | 0.9149 to 0.9365 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cd4 t cells | healthy | 3 | 0.80439 | 0.7923 to 0.85843 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cd4 t cells | healthy | 3 | 0.69578 | 0.6729 to 0.76176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cd4 t cells | healthy | 3 | 0.31133 | 0.24394 to 0.31215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cd4 t cells | healthy | 3 | 0.87382 | 0.87075 to 0.95667 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cd4 t cells | healthy | 3 | 0.82923 | 0.81942 to 0.88098 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cd4 t cells | healthy | 3 | 0.25439 | 0.21493 to 0.27102 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cd4 t cells | healthy | 3 | 0.80542 | 0.80302 to 0.84881 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.9282 | 0.92712 to 0.94638 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.6925 | 0.6694 to 0.75763 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cd4 t cells | healthy | 3 | 1.0072 | 0.99298 to 1.0085 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.996 | 0.99267 to 1.0645 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cd8 t cells | healthy | 3 | 0.99039 | 0.88514 to 1.0294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cd8 t cells | healthy | 3 | 0.76109 | 0.68696 to 0.77078 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cd8 t cells | healthy | 3 | 0.31616 | 0.31096 to 0.37599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cd8 t cells | healthy | 3 | 1.1259 | 0.98432 to 1.212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cd8 t cells | healthy | 3 | 1.0112 | 0.90952 to 1.051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cd8 t cells | healthy | 3 | 0.27609 | 0.26452 to 0.31827 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cd8 t cells | healthy | 3 | 1.0217 | 0.90437 to 1.0695 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cd8 t cells | sorted cd8 t cells | healthy | 3 | 0.9518 | 0.89168 to 0.97547 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cd8 t cells | sorted cd8 t cells | healthy | 3 | 0.75742 | 0.68366 to 0.76676 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cd8 t cells | healthy | 3 | 0.9959 | 0.99563 to 1.0076 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted cd8 t cells | healthy | 3 | 1.0961 | 1.004 to 1.111 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t central memory cd4 | healthy | 3 | 0.92355 | 0.91644 to 0.95589 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t central memory cd4 | healthy | 3 | 0.69526 | 0.67718 to 0.73389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t central memory cd4 | healthy | 3 | 0.23048 | 0.22811 to 0.2311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t central memory cd4 | healthy | 3 | 1.0779 | 1.062 to 1.1334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t central memory cd4 | healthy | 3 | 0.94917 | 0.94347 to 0.97937 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t central memory cd4 | healthy | 3 | 0.19411 | 0.19149 to 0.20171 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t central memory cd4 | healthy | 3 | 0.9679 | 0.96657 to 0.99358 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t central memory cd4 | sorted t central memory cd4 | healthy | 3 | 1.0593 | 1.0216 to 1.0604 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t central memory cd4 | sorted t central memory cd4 | healthy | 3 | 0.78507 | 0.75222 to 0.8234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t central memory cd4 | healthy | 3 | 0.9998 | 0.99872 to 1.0016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted t central memory cd4 | healthy | 3 | 1.0059 | 0.94164 to 1.0071 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t effector cell cd8 | healthy | 3 | 0.97655 | 0.95454 to 1.0122 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t effector cell cd8 | healthy | 3 | 0.71944 | 0.70343 to 0.77705 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t effector cell cd8 | healthy | 3 | 0.29797 | 0.27875 to 0.30592 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t effector cell cd8 | healthy | 3 | 1.1272 | 1.0795 to 1.1979 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t effector cell cd8 | healthy | 3 | 1.0012 | 0.98096 to 1.0347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t effector cell cd8 | healthy | 3 | 0.25658 | 0.23968 to 0.25676 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t effector cell cd8 | healthy | 3 | 1.0205 | 0.96132 to 1.0802 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t effector cell cd8 | sorted t effector cell cd8 | healthy | 3 | 1.0169 | 0.91592 to 1.0405 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t effector cell cd8 | sorted t effector cell cd8 | healthy | 3 | 0.82295 | 0.82256 to 0.8616 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t effector cell cd8 | healthy | 3 | 0.9969 | 0.99438 to 0.99726 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted t effector cell cd8 | healthy | 3 | 1.0108 | 1.0089 to 1.0588 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted t effector cell cd8 | healthy | 3 | 0.9783 | 0.95076 to 0.99288 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t effector memory cd4 | healthy | 3 | 1.0644 | 1.0289 to 1.0666 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t effector memory cd4 | healthy | 3 | 0.93742 | 0.89501 to 0.95032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t effector memory cd4 | healthy | 3 | 0.24249 | 0.24244 to 0.25295 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t effector memory cd4 | healthy | 3 | 1.3066 | 1.2391 to 1.3106 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t effector memory cd4 | healthy | 3 | 1.082 | 1.051 to 1.0864 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t effector memory cd4 | healthy | 3 | 0.20742 | 0.20414 to 0.21616 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t effector memory cd4 | healthy | 3 | 1.1678 | 1.0949 to 1.1681 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t effector memory cd4 | sorted t effector memory cd4 | healthy | 3 | 0.9412 | 0.90331 to 0.98791 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t effector memory cd4 | sorted t effector memory cd4 | healthy | 3 | 0.77313 | 0.7676 to 0.79855 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t effector memory cd4 | healthy | 3 | 0.9946 | 0.99433 to 0.99712 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted t effector memory cd4 | healthy | 3 | 1.0469 | 0.96464 to 1.0532 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted t effector memory cd4 | healthy | 3 | 0.9737 | 0.96272 to 0.98441 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted t effector memory cd4 | healthy | 3 | 0.9943 | 0.98449 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted effector memory cd8 t cells | healthy | 1 | 1.0595 | 1.0595 to 1.0595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted effector memory cd8 t cells | healthy | 1 | 0.74034 | 0.74034 to 0.74034 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted effector memory cd8 t cells | healthy | 1 | 0.28196 | 0.28196 to 0.28196 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted effector memory cd8 t cells | healthy | 1 | 1.2785 | 1.2785 to 1.2785 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted effector memory cd8 t cells | healthy | 1 | 1.0803 | 1.0803 to 1.0803 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted effector memory cd8 t cells | healthy | 1 | 0.24442 | 0.24442 to 0.24442 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted effector memory cd8 t cells | healthy | 1 | 1.1461 | 1.1461 to 1.1461 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | effector memory cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 0.9094 | 0.9094 to 0.9094 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | effector memory cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 0.73758 | 0.73758 to 0.73758 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted effector memory cd8 t cells | healthy | 1 | 0.9948 | 0.9948 to 0.9948 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 1.0814 | 1.0814 to 1.0814 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted naive cd4 t cells | healthy | 1 | 0.61564 | 0.61564 to 0.61564 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted naive cd4 t cells | healthy | 1 | 0.43466 | 0.43466 to 0.43466 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted naive cd4 t cells | healthy | 1 | 0.20865 | 0.20865 to 0.20865 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted naive cd4 t cells | healthy | 1 | 0.63156 | 0.63156 to 0.63156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted naive cd4 t cells | healthy | 1 | 0.64294 | 0.64294 to 0.64294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted naive cd4 t cells | healthy | 1 | 0.1819 | 0.1819 to 0.1819 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted naive cd4 t cells | healthy | 1 | 0.57884 | 0.57884 to 0.57884 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | naive cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 0.8329 | 0.8329 to 0.8329 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | naive cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 0.67147 | 0.67147 to 0.67147 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted naive cd4 t cells | healthy | 1 | 0.9969 | 0.9969 to 0.9969 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 1.0262 | 1.0262 to 1.0262 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted naive cd8 t cells | healthy | 1 | 0.57175 | 0.57175 to 0.57175 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted naive cd8 t cells | healthy | 1 | 0.40766 | 0.40766 to 0.40766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted naive cd8 t cells | healthy | 1 | 0.21646 | 0.21646 to 0.21646 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted naive cd8 t cells | healthy | 1 | 0.5793 | 0.5793 to 0.5793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted naive cd8 t cells | healthy | 1 | 0.59657 | 0.59657 to 0.59657 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted naive cd8 t cells | healthy | 1 | 0.1894 | 0.1894 to 0.1894 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted naive cd8 t cells | healthy | 1 | 0.52697 | 0.52697 to 0.52697 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | naive cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 0.8031 | 0.8031 to 0.8031 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | naive cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 0.63896 | 0.63896 to 0.63896 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted naive cd8 t cells | healthy | 1 | 0.9964 | 0.9964 to 0.9964 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 1.0111 | 1.0111 to 1.0111 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted nk cells | healthy | 3 | 0.80829 | 0.77107 to 0.9738 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted nk cells | healthy | 3 | 0.70856 | 0.70493 to 0.75017 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted nk cells | healthy | 3 | 0.27612 | 0.27351 to 0.28583 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted nk cells | healthy | 3 | 0.87508 | 0.83 to 1.1104 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted nk cells | healthy | 3 | 0.83593 | 0.79647 to 1.0038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted nk cells | healthy | 3 | 0.24276 | 0.22479 to 0.25069 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted nk cells | healthy | 3 | 0.79512 | 0.73567 to 0.99878 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | nk cells | sorted nk cells | healthy | 3 | 0.9105 | 0.89619 to 0.95775 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | nk cells | sorted nk cells | healthy | 3 | 0.70767 | 0.70417 to 0.83281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted nk cells | healthy | 3 | 1.0003 | 0.99607 to 1.0013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted nk cells | healthy | 3 | 0.9064 | 0.89785 to 0.92224 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted nk cells | healthy | 1 | 1.0463 | 1.0463 to 1.0463 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted monocytes | healthy | 3 | 0.75462 | 0.74462 to 0.79298 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted monocytes | healthy | 3 | 0.6662 | 0.65385 to 0.69439 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted monocytes | healthy | 3 | 0.18876 | 0.18156 to 0.23943 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted monocytes | healthy | 3 | 0.84628 | 0.84335 to 0.90662 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted monocytes | healthy | 3 | 0.77945 | 0.77031 to 0.82101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted monocytes | healthy | 3 | 0.15765 | 0.14798 to 0.20535 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted monocytes | healthy | 3 | 0.76359 | 0.74799 to 0.776 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | monocytes | sorted monocytes | healthy | 3 | 0.8997 | 0.8799 to 0.91788 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | monocytes | sorted monocytes | healthy | 3 | 0.7519 | 0.7425 to 0.79834 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted monocytes | healthy | 3 | 0.9987 | 0.99861 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted monocytes | healthy | 3 | 1.0928 | 1.0527 to 1.1077 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted monocytes | healthy | 3 | 0.9953 | 0.99449 to 1.0154 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon macrophages | healthy | 2 | 0.97606 | 0.91485 to 1.0373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon macrophages | healthy | 2 | 0.8187 | 0.73635 to 0.90105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon macrophages | healthy | 2 | 0.2312 | 0.21792 to 0.24449 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon macrophages | healthy | 2 | 1.1058 | 1.0216 to 1.19 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon macrophages | healthy | 2 | 1.0122 | 0.95189 to 1.0726 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon macrophages | healthy | 2 | 0.20679 | 0.19303 to 0.22055 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon macrophages | healthy | 2 | 0.99829 | 0.91687 to 1.0797 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon macrophages | sorted colon macrophages | healthy | 2 | 0.89075 | 0.87702 to 0.90448 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon macrophages | sorted colon macrophages | healthy | 2 | 0.87179 | 0.87149 to 0.87209 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon macrophages | healthy | 2 | 1.0048 | 1.0044 to 1.0051 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon macrophages | healthy | 1 | 0.9072 | 0.9072 to 0.9072 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon macrophages | healthy | 2 | 0.98435 | 0.94965 to 1.019 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon macrophages | healthy | 2 | 0.977 | 0.92822 to 1.0258 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon macrophages | healthy | 1 | 1.0155 | 1.0155 to 1.0155 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted colon macrophages | healthy | 2 | 1.0399 | 1.0051 to 1.0747 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon macrophages | healthy | 1 | 0.9282 | 0.9282 to 0.9282 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar macrophages | healthy | 2 | 0.79668 | 0.77766 to 0.81571 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar macrophages | healthy | 2 | 0.6649 | 0.65794 to 0.67186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar macrophages | healthy | 2 | 0.19789 | 0.19504 to 0.20074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar macrophages | healthy | 2 | 0.86533 | 0.8419 to 0.88877 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar macrophages | healthy | 2 | 0.83225 | 0.81229 to 0.85221 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar macrophages | healthy | 2 | 0.17376 | 0.16711 to 0.18041 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar macrophages | healthy | 2 | 0.77957 | 0.74641 to 0.81273 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar macrophages | sorted lung alveolar macrophages | healthy | 2 | 0.8507 | 0.84035 to 0.86105 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar macrophages | sorted lung alveolar macrophages | healthy | 2 | 0.8924 | 0.85005 to 0.93474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar macrophages | healthy | 2 | 1.0011 | 0.99472 to 1.0076 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar macrophages | healthy | 2 | 1.008 | 0.99018 to 1.0258 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar macrophages | healthy | 2 | 0.9617 | 0.96134 to 0.96206 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar macrophages | healthy | 1 | 0.9876 | 0.9876 to 0.9876 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung interstitial macrophages | healthy | 3 | 0.76998 | 0.74158 to 0.84819 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung interstitial macrophages | healthy | 3 | 0.67711 | 0.59274 to 0.69373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung interstitial macrophages | healthy | 3 | 0.21435 | 0.19255 to 0.21582 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung interstitial macrophages | healthy | 3 | 0.84871 | 0.80393 to 0.93186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung interstitial macrophages | healthy | 3 | 0.80099 | 0.77465 to 0.88365 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung interstitial macrophages | healthy | 3 | 0.19032 | 0.16945 to 0.1925 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung interstitial macrophages | healthy | 3 | 0.77831 | 0.73313 to 0.839 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung interstitial macrophages | sorted lung interstitial macrophages | healthy | 3 | 0.847 | 0.84439 to 0.86005 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung interstitial macrophages | sorted lung interstitial macrophages | healthy | 3 | 0.70454 | 0.70434 to 0.77389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung interstitial macrophages | healthy | 3 | 1.0043 | 0.99719 to 1.0088 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung interstitial macrophages | healthy | 3 | 0.9713 | 0.93512 to 0.97256 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung interstitial macrophages | healthy | 2 | 0.9833 | 0.94775 to 1.0189 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung interstitial macrophages | healthy | 1 | 0.9977 | 0.9977 to 0.9977 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted lung interstitial macrophages | healthy | 1 | 1.0377 | 1.0377 to 1.0377 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung interstitial macrophages | healthy | 1 | 0.9858 | 0.9858 to 0.9858 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted b cells | healthy | 4 | 0.87722 | 0.77662 to 1.0198 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted b cells | healthy | 4 | 0.69226 | 0.62619 to 0.89161 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted b cells | healthy | 4 | 0.27803 | 0.23415 to 0.31587 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted b cells | healthy | 4 | 0.97922 | 0.83487 to 1.2285 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted b cells | healthy | 4 | 0.90295 | 0.80093 to 1.0384 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted b cells | healthy | 4 | 0.27028 | 0.21603 to 2.8595 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted b cells | healthy | 4 | 0.8361 | 0.77166 to 0.92822 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | b cells | sorted b cells | healthy | 4 | 0.87995 | 0.81288 to 0.95544 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | b cells | sorted b cells | healthy | 4 | 0.68907 | 0.62347 to 1.8717 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted b cells | healthy | 4 | 0.99885 | 0.99582 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted b cells | healthy | 4 | 0.9742 | 0.96156 to 1.0043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted memory b cells | healthy | 4 | 0.9918 | 0.96169 to 1.0208 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted memory b cells | healthy | 4 | 0.82049 | 0.6983 to 0.92072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted memory b cells | healthy | 4 | 0.22834 | 0.21299 to 0.2434 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted memory b cells | healthy | 4 | 1.1859 | 1.135 to 1.2349 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted memory b cells | healthy | 4 | 1.0127 | 0.98557 to 1.0415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted memory b cells | healthy | 4 | 0.63989 | 0.19743 to 1.0824 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted memory b cells | healthy | 4 | 0.92036 | 0.77583 to 1.0687 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | memory b cells | sorted memory b cells | healthy | 4 | 0.88975 | 0.78753 to 1.0152 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | memory b cells | sorted memory b cells | healthy | 4 | 1.3665 | 0.72604 to 1.9766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted memory b cells | healthy | 4 | 0.99755 | 0.99298 to 0.99787 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted memory b cells | healthy | 2 | 0.9713 | 0.95447 to 0.98813 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tonsil palatine epithelium | healthy | 3 | 1.0382 | 1.0032 to 1.0535 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tonsil palatine epithelium | healthy | 3 | 1.0051 | 0.98378 to 1.0375 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tonsil palatine epithelium | healthy | 3 | 0.19037 | 0.18941 to 0.19431 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tonsil palatine epithelium | healthy | 3 | 1.3633 | 1.3143 to 1.4063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tonsil palatine epithelium | healthy | 3 | 1.0466 | 1.01 to 1.0576 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tonsil palatine epithelium | healthy | 3 | 0.17092 | 0.16921 to 0.17293 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tonsil palatine epithelium | healthy | 3 | 1.2087 | 1.1598 to 1.2363 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tonsil palatine epithelium | sorted tonsil palatine epithelium | healthy | 3 | 1.0066 | 0.95962 to 1.013 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tonsil palatine epithelium | sorted tonsil palatine epithelium | healthy | 3 | 0.73051 | 0.7236 to 0.75055 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tonsil palatine epithelium | healthy | 3 | 1.0027 | 0.99901 to 1.0027 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tonsil palatine epithelium | healthy | 3 | 0.9473 | 0.90365 to 0.97565 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tonsil palatine epithelium | healthy | 3 | 0.9731 | 0.92729 to 0.99011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tonsil palatine epithelium | healthy | 3 | 0.9759 | 0.94656 to 1.0407 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tonsil palatine epithelium | healthy | 3 | 1.0704 | 1.0083 to 1.0811 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tonsil palatine epithelium | healthy | 3 | 0.9368 | 0.91025 to 0.94256 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.027 | 1.0014 to 1.0525 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.0091 | 0.97964 to 1.0386 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.18816 | 0.18802 to 0.1883 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.3528 | 1.2871 to 1.4184 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.033 | 1.0119 to 1.054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.16417 | 0.16236 to 0.16597 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.182 | 1.1129 to 1.2512 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tonsil pharyngeal epithelium | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.98485 | 0.94458 to 1.0251 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tonsil pharyngeal epithelium | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.93591 | 0.91147 to 0.96035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.0076 | 1.0036 to 1.0116 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.94005 | 0.94 to 0.9401 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.0072 | 1.0062 to 1.0083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.98645 | 0.96849 to 1.0044 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.0247 | 0.99167 to 1.0577 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.91755 | 0.90698 to 0.92812 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tongue epithelium | healthy | 3 | 0.92043 | 0.90284 to 0.9246 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tongue epithelium | healthy | 3 | 0.90272 | 0.89041 to 0.91562 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tongue epithelium | healthy | 3 | 0.19742 | 0.19311 to 0.20173 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tongue epithelium | healthy | 3 | 1.1647 | 1.1613 to 1.191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tongue epithelium | healthy | 3 | 0.93146 | 0.91017 to 0.93234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tongue epithelium | healthy | 3 | 0.17247 | 0.17017 to 0.18046 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tongue epithelium | healthy | 3 | 1.0314 | 1.0192 to 1.0546 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tongue epithelium | sorted tongue epithelium | healthy | 3 | 1.0094 | 0.99806 to 1.0432 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tongue epithelium | sorted tongue epithelium | healthy | 3 | 0.70774 | 0.7015 to 0.73838 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tongue epithelium | healthy | 3 | 1.0042 | 0.99826 to 1.0049 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tongue epithelium | healthy | 3 | 0.9637 | 0.93238 to 0.98206 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tongue epithelium | healthy | 3 | 0.9825 | 0.94794 to 0.99375 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tongue epithelium | healthy | 3 | 0.9992 | 0.99137 to 1.0274 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tongue epithelium | healthy | 3 | 1.0957 | 1.0913 to 1.1088 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tongue epithelium | healthy | 3 | 0.9117 | 0.89451 to 0.94401 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung bronchus epithelium | healthy | 3 | 0.92693 | 0.91339 to 0.96448 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung bronchus epithelium | healthy | 3 | 0.89275 | 0.88157 to 0.94257 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung bronchus epithelium | healthy | 3 | 0.1914 | 0.18905 to 0.19678 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung bronchus epithelium | healthy | 3 | 1.1633 | 1.1526 to 1.243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung bronchus epithelium | healthy | 3 | 0.9409 | 0.92608 to 0.97407 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung bronchus epithelium | healthy | 3 | 0.17011 | 0.16746 to 0.18032 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung bronchus epithelium | healthy | 3 | 1.0353 | 1.0248 to 1.0979 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung bronchus epithelium | sorted lung bronchus epithelium | healthy | 3 | 1.0042 | 1.0027 to 1.0253 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung bronchus epithelium | sorted lung bronchus epithelium | healthy | 3 | 0.68651 | 0.6662 to 0.71703 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung bronchus epithelium | healthy | 3 | 1.0009 | 0.99928 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung bronchus epithelium | healthy | 3 | 0.9407 | 0.9407 to 0.99578 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung bronchus epithelium | healthy | 3 | 1.0171 | 0.97156 to 1.0243 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung bronchus epithelium | healthy | 3 | 0.9814 | 0.94306 to 0.98932 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung bronchus epithelium | healthy | 3 | 1.0488 | 1.0425 to 1.0871 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung bronchus epithelium | healthy | 3 | 0.8874 | 0.86634 to 0.92484 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted prostate epithelium | healthy | 3 | 0.96723 | 0.96205 to 0.96728 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted prostate epithelium | healthy | 3 | 0.95227 | 0.95115 to 0.96524 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted prostate epithelium | healthy | 3 | 0.2006 | 0.18656 to 0.21649 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted prostate epithelium | healthy | 3 | 1.1094 | 1.0892 to 1.1714 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted prostate epithelium | healthy | 3 | 0.99764 | 0.98044 to 0.99936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted prostate epithelium | healthy | 3 | 0.1775 | 0.15983 to 0.19153 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted prostate epithelium | healthy | 3 | 0.96564 | 0.95391 to 1.0184 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | prostate epithelium | sorted prostate epithelium | healthy | 3 | 0.9435 | 0.92181 to 1.0151 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | prostate epithelium | sorted prostate epithelium | healthy | 3 | 0.7715 | 0.76171 to 0.77272 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted prostate epithelium | healthy | 3 | 0.997 | 0.99322 to 1.0062 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted prostate epithelium | healthy | 3 | 0.9266 | 0.89906 to 0.94568 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted prostate epithelium | healthy | 3 | 1.0041 | 0.99528 to 1.0257 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted prostate epithelium | healthy | 3 | 0.9533 | 0.9245 to 1.0475 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted prostate epithelium | healthy | 3 | 1.0237 | 0.96439 to 1.0417 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted prostate epithelium | healthy | 3 | 0.9161 | 0.88577 to 0.92852 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted bladder epithelium | healthy | 3 | 0.95882 | 0.93804 to 0.97145 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted bladder epithelium | healthy | 3 | 0.94195 | 0.91388 to 0.94963 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted bladder epithelium | healthy | 3 | 0.17708 | 0.16636 to 0.17995 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted bladder epithelium | healthy | 3 | 1.1687 | 1.1504 to 1.1738 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted bladder epithelium | healthy | 3 | 0.97847 | 0.95696 to 0.99489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted bladder epithelium | healthy | 3 | 0.15492 | 0.14064 to 0.15763 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted bladder epithelium | healthy | 3 | 1.0064 | 0.99681 to 1.0262 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | bladder epithelium | sorted bladder epithelium | healthy | 3 | 0.9913 | 0.98572 to 1.0144 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | bladder epithelium | sorted bladder epithelium | healthy | 3 | 0.76341 | 0.7579 to 0.83181 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted bladder epithelium | healthy | 3 | 0.992 | 0.99182 to 0.9965 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted bladder epithelium | healthy | 3 | 0.9545 | 0.89888 to 0.97556 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted bladder epithelium | healthy | 3 | 0.999 | 0.9522 to 1.0103 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted bladder epithelium | healthy | 3 | 0.9311 | 0.87557 to 0.96602 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted bladder epithelium | healthy | 3 | 1.0394 | 1.032 to 1.0474 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted bladder epithelium | healthy | 3 | 0.9701 | 0.95444 to 1.0001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted breast luminal epithelium | healthy | 3 | 0.89183 | 0.86782 to 0.93088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted breast luminal epithelium | healthy | 3 | 0.88168 | 0.83826 to 0.92546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted breast luminal epithelium | healthy | 3 | 0.26015 | 0.21304 to 0.29277 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted breast luminal epithelium | healthy | 3 | 1.0277 | 0.96809 to 1.0573 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted breast luminal epithelium | healthy | 3 | 0.91075 | 0.89699 to 0.95512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted breast luminal epithelium | healthy | 3 | 0.21556 | 0.18788 to 0.25484 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted breast luminal epithelium | healthy | 3 | 0.89265 | 0.84655 to 0.89634 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | breast luminal epithelium | sorted breast luminal epithelium | healthy | 3 | 0.915 | 0.86613 to 0.91833 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | breast luminal epithelium | sorted breast luminal epithelium | healthy | 3 | 0.76674 | 0.7182 to 0.81009 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted breast luminal epithelium | healthy | 3 | 0.9965 | 0.99218 to 0.99902 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted breast luminal epithelium | healthy | 3 | 0.8954 | 0.87218 to 0.9251 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted breast luminal epithelium | healthy | 3 | 0.9482 | 0.94694 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted breast luminal epithelium | healthy | 3 | 1.0041 | 0.94677 to 1.0289 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted breast luminal epithelium | healthy | 3 | 1.0185 | 0.95658 to 1.0257 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted breast luminal epithelium | healthy | 3 | 0.9476 | 0.94238 to 0.96389 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted breast basal epithelium | healthy | 3 | 0.8366 | 0.80335 to 0.86514 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted breast basal epithelium | healthy | 3 | 0.82404 | 0.78701 to 0.85702 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted breast basal epithelium | healthy | 3 | 0.26074 | 0.25603 to 0.26358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted breast basal epithelium | healthy | 3 | 0.98073 | 0.93102 to 1.0012 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted breast basal epithelium | healthy | 3 | 0.8524 | 0.81927 to 0.88383 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted breast basal epithelium | healthy | 3 | 0.22147 | 0.21617 to 0.22447 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted breast basal epithelium | healthy | 3 | 0.83822 | 0.79797 to 0.85482 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | breast basal epithelium | sorted breast basal epithelium | healthy | 3 | 0.899 | 0.89351 to 0.90143 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | breast basal epithelium | sorted breast basal epithelium | healthy | 3 | 0.79568 | 0.78044 to 0.80706 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted breast basal epithelium | healthy | 3 | 0.9994 | 0.99841 to 1.0007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted breast basal epithelium | healthy | 3 | 0.9194 | 0.89168 to 0.92147 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted breast basal epithelium | healthy | 3 | 0.9172 | 0.88615 to 0.95482 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted breast basal epithelium | healthy | 3 | 0.9788 | 0.97673 to 0.99374 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted breast basal epithelium | healthy | 3 | 0.9889 | 0.92086 to 1.0121 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted breast basal epithelium | healthy | 3 | 1.0291 | 1.0268 to 1.033 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar epithelium | healthy | 3 | 0.95081 | 0.94233 to 0.96061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar epithelium | healthy | 3 | 0.94566 | 0.93446 to 0.95047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar epithelium | healthy | 3 | 0.27551 | 0.27028 to 0.28534 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar epithelium | healthy | 3 | 1.1442 | 1.1344 to 1.1575 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar epithelium | healthy | 3 | 0.96656 | 0.95711 to 0.97644 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar epithelium | healthy | 3 | 0.24782 | 0.2393 to 0.25697 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar epithelium | healthy | 3 | 1.0124 | 1.0078 to 1.032 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar epithelium | sorted lung alveolar epithelium | healthy | 3 | 0.9565 | 0.94705 to 0.96478 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar epithelium | sorted lung alveolar epithelium | healthy | 3 | 0.82321 | 0.81855 to 0.83419 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar epithelium | healthy | 3 | 1.0018 | 1.0015 to 1.0037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar epithelium | healthy | 3 | 0.9694 | 0.92782 to 0.97588 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar epithelium | healthy | 3 | 0.9717 | 0.9384 to 0.97755 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar epithelium | healthy | 3 | 0.9525 | 0.91407 to 0.95628 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung alveolar epithelium | healthy | 3 | 1.0854 | 1.0116 to 1.0951 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung alveolar epithelium | healthy | 3 | 0.9106 | 0.90421 to 0.91735 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric fundus epithelium | healthy | 3 | 0.97454 | 0.92074 to 0.97719 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric fundus epithelium | healthy | 3 | 0.95031 | 0.88944 to 0.95319 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric fundus epithelium | healthy | 3 | 0.18422 | 0.18075 to 0.19572 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric fundus epithelium | healthy | 3 | 1.2536 | 1.1532 to 1.2921 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric fundus epithelium | healthy | 3 | 0.9827 | 0.93433 to 0.98447 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric fundus epithelium | healthy | 3 | 0.16237 | 0.16207 to 0.17506 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric fundus epithelium | healthy | 3 | 1.1099 | 1.0226 to 1.1493 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric fundus epithelium | sorted gastric fundus epithelium | healthy | 3 | 0.9544 | 0.92497 to 0.95971 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric fundus epithelium | sorted gastric fundus epithelium | healthy | 3 | 0.70167 | 0.68971 to 0.71715 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric fundus epithelium | healthy | 3 | 0.9985 | 0.99715 to 1.0053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric fundus epithelium | healthy | 3 | 0.9109 | 0.892 to 0.92476 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric fundus epithelium | healthy | 3 | 0.9566 | 0.95021 to 0.98459 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric fundus epithelium | healthy | 3 | 0.9435 | 0.93261 to 0.94386 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric fundus epithelium | healthy | 3 | 1.0295 | 1.0229 to 1.0798 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric fundus epithelium | healthy | 3 | 0.9092 | 0.90479 to 0.94511 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric body epithelium | healthy | 3 | 0.95434 | 0.9137 to 0.96028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric body epithelium | healthy | 3 | 0.93335 | 0.88666 to 0.93517 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric body epithelium | healthy | 3 | 0.18619 | 0.17024 to 0.19523 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric body epithelium | healthy | 3 | 1.2332 | 1.1398 to 1.2532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric body epithelium | healthy | 3 | 0.96097 | 0.92799 to 0.97046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric body epithelium | healthy | 3 | 0.16547 | 0.14836 to 0.17469 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric body epithelium | healthy | 3 | 1.0917 | 1.0045 to 1.0924 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric body epithelium | sorted gastric body epithelium | healthy | 3 | 0.9479 | 0.94025 to 0.96887 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric body epithelium | sorted gastric body epithelium | healthy | 3 | 0.71009 | 0.70567 to 0.73345 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric body epithelium | healthy | 3 | 1.0019 | 0.99587 to 1.0045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric body epithelium | healthy | 3 | 0.8689 | 0.86449 to 0.93127 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric body epithelium | healthy | 3 | 0.9908 | 0.94427 to 1.0147 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric body epithelium | healthy | 3 | 0.9364 | 0.91777 to 0.94486 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric body epithelium | healthy | 3 | 1.0821 | 1.0424 to 1.1075 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric body epithelium | healthy | 3 | 0.9306 | 0.93024 to 0.93564 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric antrum epithelium | healthy | 3 | 0.91957 | 0.91515 to 0.9507 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric antrum epithelium | healthy | 3 | 0.89274 | 0.88946 to 0.9266 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric antrum epithelium | healthy | 3 | 0.18266 | 0.17298 to 0.19215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric antrum epithelium | healthy | 3 | 1.1819 | 1.1418 to 1.24 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric antrum epithelium | healthy | 3 | 0.93038 | 0.92912 to 0.95903 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric antrum epithelium | healthy | 3 | 0.16453 | 0.15466 to 0.16887 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric antrum epithelium | healthy | 3 | 1.0441 | 1.0038 to 1.0953 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric antrum epithelium | sorted gastric antrum epithelium | healthy | 3 | 0.9684 | 0.94302 to 0.97416 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric antrum epithelium | sorted gastric antrum epithelium | healthy | 3 | 0.69206 | 0.69135 to 0.71554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric antrum epithelium | healthy | 3 | 0.9999 | 0.99522 to 1.0012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric antrum epithelium | healthy | 3 | 0.8792 | 0.86066 to 0.8927 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric antrum epithelium | healthy | 3 | 0.9823 | 0.92551 to 0.98842 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric antrum epithelium | healthy | 3 | 0.9555 | 0.94857 to 1.0008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric antrum epithelium | healthy | 3 | 1.1008 | 1.0609 to 1.1164 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric antrum epithelium | healthy | 3 | 0.9281 | 0.9119 to 0.95186 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon epithelium | healthy | 3 | 0.84131 | 0.82947 to 0.90358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon epithelium | healthy | 3 | 0.83214 | 0.82276 to 0.90424 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon epithelium | healthy | 3 | 0.27682 | 0.1931 to 0.2785 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon epithelium | healthy | 3 | 1.0384 | 0.96776 to 1.1104 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon epithelium | healthy | 3 | 0.85491 | 0.8452 to 0.91399 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon epithelium | healthy | 3 | 0.23826 | 0.16869 to 0.24191 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon epithelium | healthy | 3 | 0.91317 | 0.8408 to 0.97513 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon epithelium | sorted colon epithelium | healthy | 3 | 0.8883 | 0.88452 to 0.96219 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon epithelium | sorted colon epithelium | healthy | 3 | 0.77386 | 0.7691 to 0.85035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon epithelium | healthy | 3 | 0.9979 | 0.9961 to 0.99925 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon epithelium | healthy | 3 | 0.9532 | 0.93844 to 0.97669 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon epithelium | healthy | 3 | 0.9517 | 0.93118 to 1.0017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon epithelium | healthy | 3 | 0.958 | 0.8986 to 1.0062 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon epithelium | healthy | 3 | 1.1166 | 1.0805 to 1.145 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon epithelium | healthy | 3 | 0.9292 | 0.89959 to 0.93604 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon enteroendocrine | healthy | 2 | 0.86051 | 0.8554 to 0.86563 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon enteroendocrine | healthy | 2 | 0.84345 | 0.8416 to 0.84529 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon enteroendocrine | healthy | 2 | 0.20759 | 0.20347 to 0.21171 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon enteroendocrine | healthy | 2 | 1.0277 | 1.0227 to 1.0327 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon enteroendocrine | healthy | 2 | 0.87866 | 0.87318 to 0.88414 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon enteroendocrine | healthy | 2 | 0.1817 | 0.17555 to 0.18785 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon enteroendocrine | healthy | 2 | 0.90519 | 0.89181 to 0.91857 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon enteroendocrine | sorted colon enteroendocrine | healthy | 2 | 0.9142 | 0.91357 to 0.91483 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon enteroendocrine | sorted colon enteroendocrine | healthy | 2 | 0.85735 | 0.82804 to 0.88667 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon enteroendocrine | healthy | 2 | 0.99965 | 0.99862 to 1.0007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon enteroendocrine | healthy | 2 | 0.96625 | 0.95693 to 0.97557 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon enteroendocrine | healthy | 2 | 0.9499 | 0.93964 to 0.96016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon enteroendocrine | healthy | 2 | 0.97185 | 0.97027 to 0.97342 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon enteroendocrine | healthy | 2 | 1.1204 | 1.1171 to 1.1236 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon enteroendocrine | healthy | 2 | 0.91615 | 0.90809 to 0.92421 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted small intestine epithelium | healthy | 3 | 0.87347 | 0.86731 to 0.91032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted small intestine epithelium | healthy | 3 | 0.86721 | 0.8499 to 0.91504 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted small intestine epithelium | healthy | 3 | 0.25513 | 0.20867 to 0.27431 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted small intestine epithelium | healthy | 3 | 1.0565 | 1.0489 to 1.1066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted small intestine epithelium | healthy | 3 | 0.89123 | 0.88101 to 0.92203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted small intestine epithelium | healthy | 3 | 0.22721 | 0.18663 to 0.23336 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted small intestine epithelium | healthy | 3 | 0.93665 | 0.93002 to 0.96046 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | small intestine epithelium | sorted small intestine epithelium | healthy | 3 | 0.9421 | 0.93976 to 0.95092 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | small intestine epithelium | sorted small intestine epithelium | healthy | 3 | 0.73445 | 0.72538 to 0.80919 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted small intestine epithelium | healthy | 3 | 0.9959 | 0.99338 to 1.0003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted small intestine epithelium | healthy | 3 | 0.9226 | 0.9199 to 0.95248 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted small intestine epithelium | healthy | 3 | 1.0358 | 1.0233 to 1.0363 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted small intestine epithelium | healthy | 3 | 0.9884 | 0.96959 to 0.99623 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted small intestine epithelium | healthy | 3 | 1.134 | 1.0855 to 1.1491 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted small intestine epithelium | healthy | 3 | 0.9091 | 0.90244 to 0.97048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Coupling-A | coupling-A | b cells | sorted b cells | healthy | 1 | 1.1266 | 1.1266 to 1.1266 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A | memory b cells | sorted memory b cells | healthy | 2 | 1.0233 | 1.0168 to 1.0299 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| C-Score | C-Score each cell | memory b cells | sorted memory b cells | healthy | 2 | 0.9936 | 0.96606 to 1.0211 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | adipocytes | healthy | 3 | 0.99949 | 0.97219 to 1.0288 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | aorta endothelium | healthy | 2 | 1.0028 | 0.93517 to 1.0705 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | vascular saphenous endothelium | healthy | 3 | 1.0028 | 0.99285 to 1.0052 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney glomerular endothelium | healthy | 3 | 0.99553 | 0.99131 to 1.0119 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney tubular endothelium | healthy | 3 | 0.98755 | 0.98294 to 1.0262 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar endothelium | healthy | 3 | 0.99787 | 0.94138 to 1.0628 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreas endothelium | healthy | 3 | 1.001 | 0.97891 to 1.0207 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon fibroblasts | healthy | 2 | 1.0008 | 0.96485 to 1.0367 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | heart fibroblasts | healthy | 3 | 0.99361 | 0.98925 to 1.0154 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | striated muscle | healthy | 2 | 1 | 0.99752 to 1.0025 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | smooth muscle | healthy | 3 | 1.0135 | 0.96819 to 1.023 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cardiomyocyte | healthy | 3 | 0.99899 | 0.99331 to 1.0074 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | oligodendrocytes | healthy | 3 | 1.0204 | 0.95826 to 1.0285 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cortical neurons | healthy | 3 | 0.99596 | 0.97154 to 1.032 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | hepatocyte | healthy | 3 | 1.0048 | 0.96457 to 1.033 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic duct | healthy | 3 | 1.0009 | 0.99468 to 1.0047 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic acinar | healthy | 3 | 0.99608 | 0.99126 to 1.0116 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic delta | healthy | 3 | 1.0095 | 0.97056 to 1.0233 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic beta | healthy | 3 | 0.99833 | 0.99764 to 1.0035 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic alpha | healthy | 3 | 0.99314 | 0.99273 to 1.0122 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney glomerular epithelium | healthy | 2 | 1 | 0.99643 to 1.0036 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney tubular epithelium | healthy | 3 | 0.99211 | 0.99091 to 1.0147 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | thyroid epithelium | healthy | 3 | 0.99848 | 0.9943 to 1.0068 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | fallopien tubes epithelium | healthy | 3 | 0.98876 | 0.94181 to 1.0691 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | endometrium epithelium | healthy | 3 | 1.005 | 0.96166 to 1.0359 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | erythrocyte progenitors | healthy | 3 | 1.0128 | 0.96384 to 1.0281 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cd4 t cells | healthy | 3 | 1.0092 | 0.96121 to 1.0333 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cd8 t cells | healthy | 3 | 1.0333 | 0.93211 to 1.0476 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t central memory cd4 | healthy | 3 | 0.98935 | 0.97855 to 1.0294 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t effector cell cd8 | healthy | 3 | 0.98081 | 0.97315 to 1.0413 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t effector memory cd4 | healthy | 3 | 0.99797 | 0.97172 to 1.0303 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | nk cells | healthy | 3 | 0.99811 | 0.98137 to 1.0202 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | monocytes | healthy | 3 | 0.98737 | 0.97344 to 1.0362 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon macrophages | healthy | 2 | 1.0003 | 0.97672 to 1.024 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar macrophages | healthy | 2 | 1 | 0.99967 to 1.0003 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung interstitial macrophages | healthy | 3 | 1.0004 | 0.99043 to 1.0094 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | b cells | healthy | 3 | 1.0061 | 0.9826 to 1.0133 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | memory b cells | healthy | 2 | 1 | 0.99236 to 1.0077 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tonsil palatine epithelium | healthy | 3 | 0.99817 | 0.99677 to 1.0045 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tonsil pharyngeal epithelium | healthy | 2 | 1.0016 | 0.95019 to 1.0531 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tongue epithelium | healthy | 3 | 1.0076 | 0.96831 to 1.027 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung bronchus epithelium | healthy | 3 | 0.97903 | 0.97901 to 1.0365 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | prostate epithelium | healthy | 3 | 0.99764 | 0.97481 to 1.0274 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | bladder epithelium | healthy | 3 | 1.0119 | 0.96332 to 1.0292 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | breast luminal epithelium | healthy | 3 | 0.99491 | 0.93748 to 1.0693 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | breast basal epithelium | healthy | 3 | 1.0035 | 0.98185 to 1.0159 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar epithelium | healthy | 3 | 1.0072 | 0.9742 to 1.0212 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric fundus epithelium | healthy | 3 | 1.0037 | 0.96707 to 1.0311 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric body epithelium | healthy | 3 | 0.99354 | 0.98405 to 1.0207 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric antrum epithelium | healthy | 3 | 0.98433 | 0.98138 to 1.0301 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon epithelium | healthy | 3 | 1.0243 | 0.94429 to 1.0407 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon enteroendocrine | healthy | 2 | 1.0002 | 0.98447 to 1.0158 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | small intestine epithelium | healthy | 3 | 1.0057 | 0.97555 to 1.0209 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | adipocytes | healthy | 3 | 1.5036 | 1.4824 to 1.5448 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | aorta endothelium | healthy | 2 | 1.3658 | 1.3342 to 1.3974 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | vascular saphenous endothelium | healthy | 3 | 1.5923 | 1.5731 to 1.6599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney glomerular endothelium | healthy | 3 | 1.6022 | 1.5693 to 1.6142 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney tubular endothelium | healthy | 3 | 1.6477 | 1.5439 to 1.716 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar endothelium | healthy | 3 | 1.6229 | 1.6011 to 1.6247 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreas endothelium | healthy | 3 | 1.7625 | 1.6502 to 1.7926 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon fibroblasts | healthy | 2 | 1.4457 | 1.4406 to 1.4509 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | heart fibroblasts | healthy | 3 | 1.6034 | 1.5721 to 1.6149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | striated muscle | healthy | 2 | 1.3897 | 1.3779 to 1.4015 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | smooth muscle | healthy | 3 | 1.7204 | 1.5052 to 1.7402 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cardiomyocyte | healthy | 3 | 1.5479 | 1.5131 to 1.5924 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | oligodendrocytes | healthy | 3 | 1.5928 | 1.5183 to 1.5947 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cortical neurons | healthy | 3 | 1.7951 | 1.6216 to 1.9551 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | hepatocyte | healthy | 3 | 1.3563 | 1.3337 to 1.382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic duct | healthy | 3 | 1.4981 | 1.4174 to 1.5121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic acinar | healthy | 3 | 1.4997 | 1.3814 to 1.5457 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic delta | healthy | 3 | 1.725 | 1.5916 to 1.7269 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic beta | healthy | 3 | 1.4608 | 1.4265 to 1.4697 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic alpha | healthy | 3 | 1.5942 | 1.55 to 1.6032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney glomerular epithelium | healthy | 2 | 1.3281 | 1.2859 to 1.3704 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney tubular epithelium | healthy | 3 | 1.5751 | 1.5347 to 1.6153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | thyroid epithelium | healthy | 3 | 1.5603 | 1.5602 to 1.5691 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | fallopien tubes epithelium | healthy | 3 | 1.5193 | 1.4642 to 1.5679 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | endometrium epithelium | healthy | 3 | 1.5566 | 1.4777 to 1.7277 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | erythrocyte progenitors | healthy | 3 | 1.4783 | 1.4395 to 1.6839 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cd4 t cells | healthy | 3 | 1.5426 | 1.5061 to 1.687 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cd8 t cells | healthy | 3 | 1.6865 | 1.5373 to 1.7425 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t central memory cd4 | healthy | 3 | 1.5933 | 1.5274 to 1.6671 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t effector cell cd8 | healthy | 3 | 1.6397 | 1.6391 to 1.7204 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t effector memory cd4 | healthy | 3 | 1.6467 | 1.6345 to 1.7047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | effector memory cd8 t cells | healthy | 1 | 1.6494 | 1.6494 to 1.6494 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | naive cd4 t cells | healthy | 1 | 1.4799 | 1.4799 to 1.4799 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | naive cd8 t cells | healthy | 1 | 1.4488 | 1.4488 to 1.4488 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | nk cells | healthy | 3 | 1.5713 | 1.5457 to 1.8529 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | monocytes | healthy | 3 | 1.6688 | 1.6549 to 1.7911 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon macrophages | healthy | 2 | 1.3825 | 1.3819 to 1.383 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar macrophages | healthy | 2 | 1.3123 | 1.2485 to 1.3762 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung interstitial macrophages | healthy | 3 | 1.4952 | 1.4923 to 1.6442 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | b cells | healthy | 3 | 1.5061 | 1.4064 to 1.5968 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | memory b cells | healthy | 2 | 1.6185 | 1.5737 to 1.6634 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tonsil palatine epithelium | healthy | 3 | 1.5496 | 1.5374 to 1.5947 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tonsil pharyngeal epithelium | healthy | 2 | 1.4003 | 1.3621 to 1.4384 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tongue epithelium | healthy | 3 | 1.4665 | 1.4542 to 1.5324 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung bronchus epithelium | healthy | 3 | 1.4547 | 1.4099 to 1.5189 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | prostate epithelium | healthy | 3 | 1.5843 | 1.5581 to 1.5853 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | bladder epithelium | healthy | 3 | 1.5725 | 1.5583 to 1.712 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | breast luminal epithelium | healthy | 3 | 1.6331 | 1.5306 to 1.7287 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | breast basal epithelium | healthy | 3 | 1.6545 | 1.6223 to 1.6783 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar epithelium | healthy | 3 | 1.5777 | 1.5689 to 1.6014 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric fundus epithelium | healthy | 3 | 1.487 | 1.4613 to 1.5159 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric body epithelium | healthy | 3 | 1.5114 | 1.4977 to 1.5604 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric antrum epithelium | healthy | 3 | 1.4766 | 1.4713 to 1.5214 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon epithelium | healthy | 3 | 1.6089 | 1.5982 to 1.7728 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon enteroendocrine | healthy | 2 | 1.3422 | 1.2952 to 1.3892 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | small intestine epithelium | healthy | 3 | 1.5366 | 1.519 to 1.6891 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | CD | monocytes | 20 | 0.98907 | 0.96126 to 1.0113 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | CD | monocytes | 20 | 0.95399 | 0.91931 to 0.98108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | CD | monocytes | 20 | 0.91073 | 0.87719 to 0.97846 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | CD | monocytes | 20 | 1.0214 | 0.97839 to 1.0724 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | CD | monocytes | 20 | 0.98522 | 0.95699 to 1.0149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | CD | monocytes | 20 | 0.93464 | 0.90761 to 0.95538 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | CD | monocytes | 20 | 0.93149 | 0.88937 to 0.95621 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | CD | monocytes | 20 | 0.96164 | 0.93306 to 0.99021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | CD | monocytes | 20 | 0.95091 | 0.91448 to 0.98177 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | CD | monocytes | 20 | 0.9758 | 0.92014 to 1.0236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | CD | monocytes | 20 | 0.93204 | 0.9049 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | CD | monocytes | 20 | 0.96696 | 0.94186 to 0.98791 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | CD | monocytes | 20 | 0.9262 | 0.90623 to 0.94942 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | CD | monocytes | 20 | 0.95537 | 0.94256 to 0.99369 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | CD | monocytes | 20 | 0.99624 | 0.97713 to 1.0348 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | CD | monocytes | 20 | 0.9532 | 0.93295 to 0.98855 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | CD | monocytes | 20 | 0.94111 | 0.92097 to 0.97059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | CD | monocytes | 20 | 0.95891 | 0.92404 to 0.98383 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | CD | monocytes | 20 | 0.98599 | 0.94195 to 1.0295 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | CD | monocytes | 20 | 0.99679 | 0.94381 to 1.0524 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | CD | monocytes | 20 | 0.97009 | 0.95322 to 1.0026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | monocytes | 20 | 1.0018 | 0.95188 to 1.0357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | CD | monocytes | 20 | 0.9691 | 0.95808 to 1.0157 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | CD | monocytes | 20 | 0.98785 | 0.94669 to 1.073 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | CD | monocytes | 20 | 0.94014 | 0.9016 to 0.97363 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | CD | monocytes | 20 | 0.96891 | 0.94397 to 0.99317 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | CD | monocytes | 20 | 0.96316 | 0.9488 to 0.98775 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | CD | monocytes | 20 | 0.954 | 0.93488 to 0.97966 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | CD | monocytes | 20 | 0.94822 | 0.93011 to 0.97092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | CD | monocytes | 20 | 0.9519 | 0.93278 to 0.9758 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | CD | monocytes | 20 | 0.94332 | 0.90492 to 1.0213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | CD | monocytes | 20 | 0.99692 | 0.97478 to 1.0301 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | CD | monocytes | 18 | 0.95186 | 0.92212 to 0.97694 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | CD | monocytes | 20 | 0.96962 | 0.92512 to 1.0333 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | CD | monocytes | 20 | 0.98878 | 0.95846 to 1.0362 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | CD | monocytes | 20 | 0.958 | 0.94189 to 1.0057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | CD | monocytes | 20 | 0.96299 | 0.93202 to 1.0191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | CD | monocytes | 20 | 1.0175 | 0.97438 to 1.0954 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | CD | monocytes | 20 | 0.98854 | 0.96529 to 1.0534 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | CD | monocytes | 20 | 0.9906 | 0.9521 to 1.0872 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | CD | monocytes | 20 | 0.94247 | 0.91985 to 0.97723 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | CD | monocytes | 20 | 0.96823 | 0.9421 to 0.99989 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | CD | monocytes | 20 | 0.99943 | 0.97142 to 1.0141 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | CD | monocytes | 20 | 0.97398 | 0.94435 to 1.002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | CD | monocytes | 20 | 0.98873 | 0.96015 to 1.0166 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | CD | monocytes | 20 | 0.99403 | 0.94732 to 1.0406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | CD | monocytes | 20 | 1.0252 | 1.0008 to 1.0856 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | CD | monocytes | 20 | 1.0006 | 0.95557 to 1.0283 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | CD | monocytes | 20 | 0.96262 | 0.94547 to 0.9875 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | CD | monocytes | 20 | 0.95279 | 0.93381 to 0.97923 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | CD | monocytes | 20 | 0.90797 | 0.8755 to 0.97252 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | CD | monocytes | 20 | 1.0149 | 0.97489 to 1.062 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | CD | monocytes | 20 | 1.007 | 0.99623 to 1.0182 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | CD | monocytes | 20 | 0.9893 | 0.91856 to 1.0658 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | unknown | CD | monocytes | 5 | 0.8777 | 0.85486 to 0.97624 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | CD | monocytes | 20 | 0.9871 | 0.87726 to 1.1126 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | unknown | CD | monocytes | 2 | 1.0474 | 0.97945 to 1.1154 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | CD | wh blood | 19 | 0.969 | 0.94198 to 1.0043 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | CD | wh blood | 19 | 0.93513 | 0.89912 to 0.9902 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | CD | wh blood | 19 | 0.92507 | 0.86613 to 0.96614 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | CD | wh blood | 19 | 0.99362 | 0.93977 to 1.0454 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | CD | wh blood | 19 | 0.9749 | 0.95385 to 1.0066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | CD | wh blood | 19 | 0.93064 | 0.90237 to 0.96236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | CD | wh blood | 19 | 0.91503 | 0.8849 to 0.94908 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | CD | wh blood | 19 | 0.93196 | 0.89516 to 0.95967 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | CD | wh blood | 19 | 0.94177 | 0.90853 to 0.96743 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | CD | wh blood | 19 | 0.95575 | 0.90895 to 0.98936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | CD | wh blood | 19 | 0.92883 | 0.89975 to 0.98424 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | CD | wh blood | 19 | 0.95634 | 0.91618 to 0.99028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | CD | wh blood | 19 | 0.89569 | 0.86885 to 0.92936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | CD | wh blood | 18 | 0.94829 | 0.86989 to 1.0101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | CD | wh blood | 13 | 0.95702 | 0.92377 to 0.98195 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | CD | wh blood | 11 | 0.97795 | 0.95226 to 0.99632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | CD | wh blood | 19 | 0.92765 | 0.89211 to 0.97167 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | CD | wh blood | 19 | 0.99358 | 0.95095 to 1.0389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | CD | wh blood | 19 | 0.93244 | 0.8893 to 0.98929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | CD | wh blood | 19 | 0.91648 | 0.8811 to 0.94905 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | CD | wh blood | 19 | 0.92304 | 0.86352 to 0.98182 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | CD | wh blood | 19 | 0.95274 | 0.87693 to 1.0197 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | CD | wh blood | 19 | 0.95248 | 0.88987 to 1.0173 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | CD | wh blood | 18 | 0.95204 | 0.90253 to 1.0053 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | CD | wh blood | 12 | 0.96584 | 0.91133 to 0.99696 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | CD | wh blood | 19 | 0.94379 | 0.90396 to 0.97908 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | wh blood | 19 | 0.96517 | 0.8634 to 1.0338 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | CD | wh blood | 19 | 0.94808 | 0.90414 to 0.99304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | CD | wh blood | 19 | 0.94444 | 0.89596 to 1.0169 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | CD | wh blood | 12 | 0.96784 | 0.93444 to 0.99091 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | CD | wh blood | 19 | 0.94135 | 0.87286 to 0.99136 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | CD | wh blood | 19 | 0.96393 | 0.9266 to 0.99632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | CD | wh blood | 12 | 0.96444 | 0.92941 to 0.99136 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | CD | wh blood | 19 | 0.9644 | 0.9328 to 0.98542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | CD | wh blood | 19 | 0.95553 | 0.91619 to 0.99178 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | CD | wh blood | 18 | 0.95359 | 0.91061 to 0.98888 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | CD | wh blood | 19 | 0.95315 | 0.91983 to 0.98177 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | CD | wh blood | 19 | 0.9484 | 0.91181 to 0.97966 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | CD | wh blood | 19 | 0.92749 | 0.88508 to 0.99914 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | CD | wh blood | 19 | 0.97422 | 0.90657 to 1.0266 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | CD | wh blood | 18 | 0.94085 | 0.89069 to 0.98011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | CD | wh blood | 19 | 0.93645 | 0.87643 to 1.0093 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | CD | wh blood | 19 | 0.94221 | 0.87277 to 0.99825 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | CD | wh blood | 19 | 0.95438 | 0.87731 to 1.0202 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | CD | wh blood | 19 | 0.93228 | 0.89955 to 0.96164 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | CD | wh blood | 19 | 0.93826 | 0.88516 to 0.9994 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | CD | wh blood | 19 | 0.96371 | 0.90588 to 1.0502 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | CD | wh blood | 19 | 0.96299 | 0.91629 to 1.0219 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | CD | wh blood | 19 | 0.95851 | 0.89559 to 1.0364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | CD | wh blood | 19 | 0.92249 | 0.89009 to 0.95574 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | CD | wh blood | 19 | 0.93606 | 0.89834 to 0.99215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | CD | wh blood | 8 | 0.95942 | 0.94292 to 0.97686 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | CD | wh blood | 19 | 0.98356 | 0.95082 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | CD | wh blood | 11 | 0.96321 | 0.93339 to 0.98545 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | CD | wh blood | 19 | 0.94921 | 0.91163 to 0.99423 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | CD | wh blood | 17 | 0.9597 | 0.92274 to 0.98874 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | CD | wh blood | 19 | 0.98191 | 0.9513 to 0.99783 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | CD | wh blood | 13 | 0.93018 | 0.8982 to 0.97082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | CD | wh blood | 19 | 0.95229 | 0.891 to 1.0171 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | CD | wh blood | 19 | 1.0084 | 0.92379 to 1.0926 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | CD | wh blood | 19 | 0.96766 | 0.87722 to 1.0296 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | CD | wh blood | 12 | 0.94582 | 0.9012 to 0.98199 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | CD | wh blood | 13 | 0.95508 | 0.92621 to 0.98467 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | CD | wh blood | 15 | 0.9605 | 0.92057 to 0.98581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | CD | wh blood | 19 | 0.9658 | 0.93354 to 0.99102 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | CD | wh blood | 19 | 0.95252 | 0.91313 to 0.98589 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | CD | wh blood | 19 | 0.91948 | 0.86412 to 0.9625 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | CD | wh blood | 19 | 0.99457 | 0.94189 to 1.0378 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | CD | wh blood | 19 | 1.0126 | 0.99892 to 1.0242 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | CD | wh blood | 18 | 1.0142 | 0.93627 to 1.0846 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | CD | wh blood | 16 | 1.0025 | 0.93695 to 1.0735 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | CD | wh blood | 12 | 0.9649 | 0.93495 to 1.0817 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | CD | wh blood | 13 | 1.0261 | 0.99056 to 1.1091 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | CD | wh blood | 11 | 0.9991 | 0.8794 to 1.0889 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | CD | wh blood | 8 | 0.9808 | 0.89114 to 1.0028 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | CD | wh blood | 19 | 0.9985 | 0.94568 to 1.0533 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | CD | wh blood | 18 | 0.9682 | 0.90681 to 1.0206 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | CD | CD4 | 20 | 0.97407 | 0.93732 to 1.0149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | CD | CD4 | 20 | 0.94529 | 0.90234 to 0.98004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | CD | CD4 | 20 | 0.91397 | 0.87795 to 0.96315 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | CD | CD4 | 20 | 0.99977 | 0.93971 to 1.0564 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | CD | CD4 | 20 | 0.97906 | 0.94987 to 1.0169 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | CD | CD4 | 20 | 0.93934 | 0.90703 to 0.95591 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | CD | CD4 | 20 | 0.92644 | 0.89089 to 0.94396 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | CD | CD4 | 20 | 0.94515 | 0.9132 to 0.97177 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | CD | CD4 | 20 | 0.94114 | 0.90876 to 0.96001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | CD | CD4 | 20 | 0.95066 | 0.92329 to 0.99819 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | CD | CD4 | 20 | 0.93113 | 0.89008 to 0.97619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | CD | CD4 | 20 | 0.96429 | 0.93256 to 1.0019 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | CD | CD4 | 20 | 0.91124 | 0.87511 to 0.94079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | CD | CD4 | 20 | 0.94172 | 0.90181 to 0.98599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | CD | CD4 | 20 | 0.97843 | 0.93619 to 1.022 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | CD | CD4 | 20 | 0.93125 | 0.89453 to 0.97779 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | CD | CD4 | 20 | 0.93212 | 0.88549 to 0.96729 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | CD | CD4 | 20 | 0.94445 | 0.89053 to 0.99398 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | CD | CD4 | 20 | 0.97513 | 0.91671 to 1.0525 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | CD | CD4 | 20 | 0.97001 | 0.91848 to 1.0456 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | CD | CD4 | 19 | 0.97155 | 0.91881 to 1.0259 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | CD | CD4 | 20 | 0.94663 | 0.91273 to 0.97524 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | CD4 | 20 | 0.98486 | 0.92478 to 1.0678 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | CD | CD4 | 20 | 0.956 | 0.91117 to 1.0069 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | CD | CD4 | 20 | 0.96056 | 0.90973 to 1.0491 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | CD | CD4 | 20 | 0.94703 | 0.90417 to 0.98387 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | CD | CD4 | 20 | 0.9771 | 0.93645 to 1.0092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | CD | CD4 | 20 | 0.964 | 0.94415 to 0.98764 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | CD | CD4 | 20 | 0.95899 | 0.92317 to 0.98401 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | CD | CD4 | 19 | 0.95839 | 0.92394 to 0.98371 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | CD | CD4 | 20 | 0.95371 | 0.92408 to 0.96818 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | CD | CD4 | 20 | 0.95432 | 0.91973 to 0.97859 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | CD | CD4 | 20 | 0.92954 | 0.90142 to 0.9954 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | CD | CD4 | 20 | 0.97705 | 0.94338 to 1.0179 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | CD | CD4 | 19 | 0.94752 | 0.90967 to 0.98236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | CD | CD4 | 20 | 0.95142 | 0.90413 to 1.0314 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | CD | CD4 | 20 | 0.95268 | 0.90532 to 1.0163 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | CD | CD4 | 20 | 0.96273 | 0.91521 to 1.0146 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | CD | CD4 | 20 | 0.9432 | 0.91246 to 0.97021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | CD | CD4 | 20 | 0.9412 | 0.90138 to 0.99727 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | CD | CD4 | 20 | 0.98009 | 0.92425 to 1.076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | CD | CD4 | 20 | 0.96934 | 0.92166 to 1.0473 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | CD | CD4 | 20 | 0.96096 | 0.90417 to 1.0411 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | CD | CD4 | 20 | 0.92184 | 0.88835 to 0.95581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | CD | CD4 | 20 | 0.94267 | 0.90472 to 0.98478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | CD | CD4 | 20 | 0.98642 | 0.95645 to 1.0153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | CD | CD4 | 20 | 0.96173 | 0.92111 to 1.0057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | CD | CD4 | 19 | 0.96059 | 0.93041 to 0.98838 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | CD | CD4 | 20 | 0.9737 | 0.95125 to 1.0024 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | CD | CD4 | 20 | 0.97587 | 0.92466 to 1.0411 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | CD | CD4 | 20 | 1.0088 | 0.9425 to 1.0903 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | CD | CD4 | 20 | 0.98455 | 0.93099 to 1.0528 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | CD | CD4 | 20 | 0.96044 | 0.92748 to 0.9861 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | CD | CD4 | 20 | 0.95527 | 0.92538 to 0.98029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | CD | CD4 | 20 | 0.91037 | 0.87584 to 0.9576 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | CD | CD4 | 20 | 0.9937 | 0.94402 to 1.0445 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | CD | CD4 | 20 | 1.0103 | 0.98957 to 1.0286 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | CD | CD4 | 17 | 0.9866 | 0.94124 to 1.0606 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | CD | CD4 | 20 | 1.0349 | 0.95063 to 1.1449 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | UC | monocytes | 20 | 0.9845 | 0.95584 to 1.0148 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | UC | monocytes | 20 | 0.9407 | 0.90903 to 0.99236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | UC | monocytes | 20 | 0.93144 | 0.88568 to 0.95862 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | UC | monocytes | 20 | 1.0152 | 0.95931 to 1.0822 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | UC | monocytes | 20 | 0.98274 | 0.95434 to 1.0097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | UC | monocytes | 20 | 0.92236 | 0.89177 to 0.96721 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | UC | monocytes | 20 | 0.91334 | 0.88166 to 0.96191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | UC | monocytes | 20 | 0.95293 | 0.91995 to 0.99593 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | UC | monocytes | 20 | 0.93501 | 0.90668 to 0.99374 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | UC | monocytes | 20 | 0.94713 | 0.90751 to 1.0286 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | UC | monocytes | 20 | 0.94016 | 0.89299 to 0.98542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | UC | monocytes | 20 | 0.95516 | 0.92929 to 1.0067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | UC | monocytes | 20 | 0.92005 | 0.8814 to 0.95876 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | UC | monocytes | 20 | 0.94534 | 0.91068 to 0.99156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | UC | monocytes | 20 | 0.99333 | 0.96163 to 1.0397 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | UC | monocytes | 20 | 0.94521 | 0.90746 to 0.99003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | UC | monocytes | 20 | 0.93258 | 0.89724 to 0.97403 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | UC | monocytes | 20 | 0.94852 | 0.87956 to 0.99845 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | UC | monocytes | 20 | 0.97951 | 0.90044 to 1.0323 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | UC | monocytes | 20 | 0.97972 | 0.90003 to 1.0528 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | UC | monocytes | 20 | 0.9574 | 0.91287 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | monocytes | 20 | 0.98785 | 0.90281 to 1.0474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | UC | monocytes | 20 | 0.9749 | 0.93057 to 1.023 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | UC | monocytes | 20 | 0.96678 | 0.90213 to 1.0554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | UC | monocytes | 20 | 0.96644 | 0.94337 to 1.0018 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | UC | monocytes | 20 | 0.95842 | 0.93741 to 0.99406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | UC | monocytes | 20 | 0.94568 | 0.92938 to 0.98562 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | UC | monocytes | 20 | 0.94568 | 0.9255 to 0.97823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | UC | monocytes | 20 | 0.94505 | 0.92576 to 0.98607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | UC | monocytes | 20 | 0.93891 | 0.893 to 1.0021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | UC | monocytes | 20 | 0.98622 | 0.94093 to 1.0381 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | UC | monocytes | 20 | 0.97459 | 0.89966 to 1.046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | UC | monocytes | 20 | 0.95456 | 0.91291 to 0.9981 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | UC | monocytes | 20 | 0.95223 | 0.89933 to 1.0156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | UC | monocytes | 20 | 0.99273 | 0.92044 to 1.0829 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | UC | monocytes | 20 | 0.98175 | 0.93369 to 1.0456 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | UC | monocytes | 20 | 0.98279 | 0.90859 to 1.0665 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | UC | monocytes | 20 | 0.9327 | 0.90424 to 0.97499 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | UC | monocytes | 20 | 0.95725 | 0.8996 to 1.0164 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | UC | monocytes | 20 | 0.99074 | 0.96326 to 1.0269 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | UC | monocytes | 20 | 0.96979 | 0.92003 to 1.0174 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | UC | monocytes | 20 | 0.9813 | 0.9555 to 1.0225 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | UC | monocytes | 20 | 0.98388 | 0.90651 to 1.0492 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | UC | monocytes | 20 | 1.0177 | 0.94351 to 1.1015 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | UC | monocytes | 20 | 0.98728 | 0.91018 to 1.0443 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | UC | monocytes | 20 | 0.95874 | 0.93506 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | UC | monocytes | 20 | 0.94469 | 0.92721 to 0.9847 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | UC | monocytes | 20 | 0.92847 | 0.88326 to 0.95446 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | UC | monocytes | 20 | 1.0085 | 0.9563 to 1.0682 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | UC | monocytes | 20 | 1.012 | 0.9945 to 1.0208 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | UC | monocytes | 18 | 1.0156 | 0.91095 to 1.1029 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | unknown | UC | monocytes | 2 | 0.9516 | 0.948 to 0.9552 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | UC | monocytes | 20 | 0.9518 | 0.84885 to 1.0426 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | healthy | 125 | 0.98989 | 0.94988 to 1.0427 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | healthy | 125 | 0.96539 | 0.90611 to 1.0412 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | healthy | 125 | 0.94743 | 0.87555 to 1.2535 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | healthy | 125 | 1.027 | 0.95414 to 1.1205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | healthy | 125 | 0.97286 | 0.91198 to 1.0149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | healthy | 125 | 0.93723 | 0.89531 to 0.97165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | healthy | 125 | 0.93961 | 0.89405 to 0.99642 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | healthy | 125 | 0.96813 | 0.91303 to 1.0737 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | healthy | 125 | 0.93433 | 0.88488 to 0.97877 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | healthy | 125 | 0.9423 | 0.90006 to 1.0048 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | healthy | 125 | 0.92828 | 0.87281 to 0.98999 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | healthy | 125 | 0.97339 | 0.93745 to 1.0212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | healthy | 125 | 0.92969 | 0.87734 to 1.1157 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | healthy | 110 | 0.9627 | 0.90779 to 1.0135 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | healthy | 93 | 0.97564 | 0.92987 to 1.0411 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | healthy | 35 | 0.95051 | 0.92945 to 0.98162 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | healthy | 125 | 0.96516 | 0.90731 to 1.096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | healthy | 125 | 1.0004 | 0.95664 to 1.0484 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | healthy | 125 | 0.96106 | 0.90248 to 1.0731 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | healthy | 125 | 0.95096 | 0.89123 to 1.1082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | healthy | 125 | 0.97316 | 0.88529 to 1.0358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | healthy | 125 | 0.94921 | 0.8993 to 1.0415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | healthy | 125 | 1.0002 | 0.91746 to 1.0751 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | healthy | 119 | 0.99527 | 0.93672 to 1.1532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | healthy | 65 | 0.98479 | 0.92393 to 1.0707 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | healthy | 125 | 0.96941 | 0.91998 to 1.0546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | healthy | 125 | 0.96343 | 0.90407 to 1.0567 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | healthy | 125 | 0.98973 | 0.92457 to 1.0619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | healthy | 125 | 0.99128 | 0.90964 to 1.064 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | healthy | 42 | 0.96064 | 0.94098 to 0.99216 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | healthy | 124 | 0.93975 | 0.90206 to 0.98796 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | healthy | 125 | 0.9619 | 0.89552 to 1.0087 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | healthy | 44 | 0.9589 | 0.93676 to 0.98488 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | healthy | 125 | 0.95584 | 0.82912 to 0.99429 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | healthy | 124 | 0.96959 | 0.93233 to 1.0456 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | healthy | 115 | 0.97293 | 0.93128 to 1.0281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | healthy | 125 | 0.96359 | 0.92932 to 1.0433 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | healthy | 125 | 0.96868 | 0.92759 to 1.0315 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | healthy | 125 | 0.94615 | 0.89974 to 1.0164 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | healthy | 125 | 0.9963 | 0.94193 to 1.0515 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | healthy | 117 | 0.95799 | 0.91764 to 1.0054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | healthy | 123 | 0.98116 | 0.90705 to 1.0506 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | healthy | 124 | 0.9309 | 0.88816 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | healthy | 125 | 0.9967 | 0.91119 to 1.1092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | healthy | 125 | 0.96229 | 0.91317 to 1.0522 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | healthy | 125 | 0.9721 | 0.91013 to 1.0498 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | healthy | 125 | 0.99901 | 0.92255 to 1.0793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | healthy | 125 | 0.98581 | 0.92564 to 1.0545 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | healthy | 125 | 0.97263 | 0.91478 to 1.0457 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | healthy | 125 | 0.94364 | 0.89207 to 1.0745 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | healthy | 125 | 0.97181 | 0.90998 to 1.0732 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | healthy | 125 | 0.97545 | 0.86525 to 1.0232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | healthy | 34 | 0.94599 | 0.9241 to 0.97284 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | healthy | 125 | 0.96407 | 0.91914 to 1.0116 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | healthy | 105 | 0.97199 | 0.9347 to 1.0148 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | healthy | 125 | 0.96819 | 0.86693 to 1.0112 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | healthy | 125 | 1.0063 | 0.91348 to 1.1029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | healthy | 125 | 1.0054 | 0.93913 to 1.0929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | healthy | 125 | 0.94818 | 0.87367 to 1.0473 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | healthy | 74 | 0.9643 | 0.90925 to 1.0018 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | healthy | 93 | 0.96821 | 0.92724 to 1.0176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | healthy | 97 | 0.97034 | 0.93543 to 1.033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | healthy | 125 | 0.95541 | 0.89711 to 0.99399 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | healthy | 122 | 0.96734 | 0.93018 to 1.0452 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | healthy | 125 | 0.94366 | 0.87247 to 1.2516 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | healthy | 125 | 1.0206 | 0.95492 to 1.0945 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | healthy | 125 | 1.0057 | 0.9916 to 1.0198 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | healthy | 72 | 0.99965 | 0.91448 to 1.0928 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | healthy | 55 | 1.0531 | 0.93815 to 1.1733 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | healthy | 44 | 0.96915 | 0.89299 to 1.0611 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | healthy | 25 | 1.0685 | 0.983 to 1.1725 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | healthy | 48 | 0.97305 | 0.91382 to 1.0632 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | basophils | unknown | CD | monocytes | 15 | 0.96351 | 0.91259 to 1.0015 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | CD | monocytes | 14 | 0.95288 | 0.92973 to 0.97738 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | CD | monocytes | 10 | 0.96495 | 0.93115 to 0.98228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | CD | monocytes | 17 | 0.9801 | 0.93516 to 0.99591 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | CD | monocytes | 12 | 0.9826 | 0.93136 to 0.99208 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | CD | monocytes | 11 | 0.96554 | 0.94379 to 0.98969 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | CD | monocytes | 10 | 0.96383 | 0.93381 to 0.98482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | CD | monocytes | 17 | 0.95999 | 0.93118 to 0.98072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | CD | monocytes | 18 | 0.96136 | 0.92585 to 0.98003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | CD | monocytes | 11 | 0.95747 | 0.92854 to 0.97776 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | CD | monocytes | 14 | 0.96338 | 0.93812 to 0.98157 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | CD | monocytes | 10 | 0.96393 | 0.91606 to 0.97835 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | CD | monocytes | 12 | 0.95472 | 0.92307 to 0.97294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | CD | monocytes | 13 | 0.95758 | 0.93519 to 0.97936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | CD | monocytes | 3 | 0.9827 | 0.91691 to 1.0239 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | monocytes | 1 | 0.9933 | 0.9933 to 0.9933 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | UC | CD8 | 19 | 0.99962 | 0.94095 to 1.0338 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | UC | CD8 | 19 | 0.96451 | 0.88856 to 1.0908 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | UC | CD8 | 19 | 0.92753 | 0.83969 to 0.98833 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | UC | CD8 | 19 | 1.0286 | 0.96374 to 1.0793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | UC | CD8 | 19 | 0.99722 | 0.95366 to 1.0277 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | UC | CD8 | 19 | 0.94813 | 0.89486 to 0.98291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | UC | CD8 | 19 | 0.93682 | 0.88281 to 0.97878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | UC | CD8 | 19 | 0.96569 | 0.8871 to 1.0031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | UC | CD8 | 19 | 0.95044 | 0.9007 to 0.98425 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | UC | CD8 | 19 | 0.96544 | 0.92079 to 1.0115 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | UC | CD8 | 19 | 0.94666 | 0.90068 to 0.97797 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | UC | CD8 | 19 | 0.9842 | 0.92507 to 1.0196 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | UC | CD8 | 19 | 0.92982 | 0.84473 to 0.964 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | UC | CD8 | 13 | 0.98 | 0.89857 to 1.0061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | UC | CD8 | 19 | 0.9651 | 0.86873 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | UC | CD8 | 19 | 1.001 | 0.92245 to 1.0652 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | UC | CD8 | 19 | 0.96333 | 0.8748 to 1.0362 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | UC | CD8 | 19 | 0.96074 | 0.855 to 0.99868 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | UC | CD8 | 18 | 0.97335 | 0.858 to 1.0218 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | UC | CD8 | 19 | 1.013 | 0.8807 to 1.0894 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | UC | CD8 | 19 | 0.99038 | 0.90407 to 1.0724 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | UC | CD8 | 16 | 1.0006 | 0.91929 to 1.0386 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | UC | CD8 | 19 | 0.96021 | 0.89173 to 1.0138 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | CD8 | 19 | 1.0231 | 0.87822 to 1.0999 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | UC | CD8 | 19 | 0.99868 | 0.89664 to 1.0609 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | UC | CD8 | 19 | 0.98554 | 0.90078 to 1.0988 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | UC | CD8 | 18 | 0.96833 | 0.91011 to 1.0059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | UC | CD8 | 18 | 0.99228 | 0.93431 to 1.0334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | UC | CD8 | 19 | 0.97526 | 0.93538 to 1.0051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | UC | CD8 | 17 | 0.9693 | 0.92172 to 0.99554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | UC | CD8 | 16 | 0.97724 | 0.92341 to 0.99833 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | UC | CD8 | 18 | 0.96314 | 0.9226 to 0.99541 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | UC | CD8 | 17 | 0.96678 | 0.9147 to 0.99009 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | UC | CD8 | 18 | 0.94033 | 0.89683 to 1.0035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | UC | CD8 | 19 | 1.0091 | 0.93036 to 1.0513 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | UC | CD8 | 16 | 0.96905 | 0.90394 to 0.99008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | UC | CD8 | 18 | 0.97936 | 0.88625 to 1.0421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | UC | CD8 | 18 | 0.98505 | 0.88458 to 1.0442 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | UC | CD8 | 18 | 0.98338 | 0.89916 to 1.0299 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | UC | CD8 | 19 | 0.9504 | 0.88597 to 0.99387 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | UC | CD8 | 18 | 0.96647 | 0.88747 to 1.0118 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | UC | CD8 | 19 | 1.0138 | 0.90432 to 1.0876 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | UC | CD8 | 19 | 1.0127 | 0.90527 to 1.0911 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | UC | CD8 | 19 | 0.9826 | 0.89765 to 1.0595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | UC | CD8 | 19 | 0.94309 | 0.86298 to 0.98236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | UC | CD8 | 18 | 0.96412 | 0.89795 to 1.003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | UC | CD8 | 19 | 1.0042 | 0.95971 to 1.0251 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | UC | CD8 | 19 | 0.98447 | 0.91492 to 1.0219 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | UC | CD8 | 19 | 0.98492 | 0.95091 to 1.0124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | UC | CD8 | 19 | 0.99637 | 0.90205 to 1.0645 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | UC | CD8 | 18 | 1.0373 | 0.92963 to 1.0948 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | UC | CD8 | 19 | 1.0202 | 0.89118 to 1.0885 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | UC | CD8 | 19 | 0.96833 | 0.92117 to 1.007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | UC | CD8 | 17 | 0.9688 | 0.91841 to 0.99221 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | UC | CD8 | 19 | 0.92496 | 0.83817 to 0.98476 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | UC | CD8 | 19 | 1.0186 | 0.96283 to 1.0627 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | UC | CD8 | 19 | 1.0072 | 0.99697 to 1.0184 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | UC | CD8 | 13 | 1.0197 | 0.906 to 1.0903 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | UC | CD8 | 18 | 0.9959 | 0.93486 to 1.0342 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | unknown | UC | CD8 | 4 | 0.97305 | 0.9382 to 0.99949 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | CD8 | 7 | 1.0413 | 0.98924 to 1.1096 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | unknown | UC | CD8 | 2 | 0.95125 | 0.94823 to 0.95427 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | UC | CD8 | 14 | 0.97465 | 0.92136 to 1.0203 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | UC | CD8 | 7 | 0.9547 | 0.92611 to 0.98815 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | UC | CD8 | 12 | 0.96829 | 0.93985 to 0.99155 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | UC | CD8 | 10 | 0.99493 | 0.93249 to 1.0264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | UC | CD8 | 9 | 0.96599 | 0.95051 to 0.98557 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | UC | CD8 | 8 | 0.97436 | 0.94474 to 0.99592 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | UC | CD8 | 8 | 0.95871 | 0.93568 to 0.98331 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | UC | CD8 | 15 | 0.97328 | 0.92395 to 0.99907 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | UC | CD8 | 7 | 0.9454 | 0.88444 to 0.9771 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | UC | CD8 | 8 | 0.96487 | 0.91391 to 0.99767 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | UC | CD8 | 12 | 0.96208 | 0.93495 to 0.99251 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | UC | CD8 | 13 | 0.97291 | 0.94444 to 0.99413 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | UC | CD8 | 3 | 1.0447 | 1.0375 to 1.0867 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | UC | CD8 | 9 | 1.0607 | 0.94176 to 1.1003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | UC | CD8 | 9 | 0.9754 | 0.87036 to 1.0263 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | UC | CD8 | 1 | 0.9624 | 0.9624 to 0.9624 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | UC | CD8 | 6 | 0.9884 | 0.86438 to 1.0292 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | basophils | unknown | UC | monocytes | 17 | 0.9243 | 0.87574 to 1.0107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | UC | monocytes | 17 | 0.97623 | 0.92299 to 1.0149 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | UC | monocytes | 19 | 0.92334 | 0.8687 to 0.98316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | UC | monocytes | 17 | 0.95245 | 0.91767 to 0.99679 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | UC | monocytes | 17 | 0.94975 | 0.9025 to 0.99079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | UC | monocytes | 19 | 0.94711 | 0.88627 to 1.0206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | UC | monocytes | 19 | 0.95166 | 0.89612 to 1.0026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | UC | monocytes | 14 | 0.96538 | 0.92502 to 1.002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | UC | monocytes | 5 | 0.92264 | 0.89503 to 0.93763 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | UC | monocytes | 3 | 1.0572 | 0.98313 to 1.0763 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | UC | monocytes | 2 | 1.0769 | 1.0668 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | UC | monocytes | 4 | 0.90555 | 0.8261 to 0.97318 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | UC | monocytes | 4 | 1.1155 | 1.0376 to 1.172 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | UC | monocytes | 10 | 1.0259 | 0.94651 to 1.0793 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | UC | monocytes | 4 | 0.99875 | 0.98638 to 1.0838 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | UC | monocytes | 12 | 1.0065 | 0.92212 to 1.0683 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | UC | monocytes | 3 | 0.9156 | 0.91434 to 0.95475 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | UC | monocytes | 12 | 0.95743 | 0.91592 to 0.98797 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | UC | monocytes | 8 | 0.9604 | 0.9167 to 0.99201 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | UC | monocytes | 11 | 0.97304 | 0.91575 to 1.0041 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | UC | monocytes | 8 | 0.96603 | 0.92644 to 0.98069 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | UC | monocytes | 11 | 0.95027 | 0.91292 to 0.97985 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | UC | monocytes | 13 | 0.95974 | 0.92631 to 0.98625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | CD | monocytes | 6 | 0.96715 | 0.9106 to 1.0328 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | UC | monocytes | 7 | 0.96409 | 0.91634 to 0.97861 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | UC | monocytes | 6 | 0.95491 | 0.90871 to 0.97243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | UC | wh blood | 21 | 0.9653 | 0.95088 to 0.99946 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | UC | wh blood | 21 | 0.92703 | 0.91123 to 0.96683 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | UC | wh blood | 21 | 0.91448 | 0.8675 to 0.96284 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | UC | wh blood | 21 | 0.98611 | 0.92813 to 1.0391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | UC | wh blood | 21 | 0.97904 | 0.9538 to 0.99654 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | UC | wh blood | 21 | 0.94098 | 0.91586 to 0.95134 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | UC | wh blood | 21 | 0.92049 | 0.89856 to 0.9419 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | UC | wh blood | 21 | 0.92757 | 0.91082 to 0.95899 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | UC | wh blood | 21 | 0.94415 | 0.92258 to 0.96172 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | UC | wh blood | 21 | 0.9439 | 0.9239 to 0.97311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | UC | wh blood | 21 | 0.92585 | 0.90544 to 0.96492 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | UC | wh blood | 21 | 0.95799 | 0.92717 to 0.97972 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | UC | wh blood | 21 | 0.89783 | 0.86568 to 0.91743 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | UC | wh blood | 21 | 0.93202 | 0.88529 to 0.98716 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | UC | wh blood | 21 | 0.92465 | 0.88603 to 0.95281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | UC | wh blood | 21 | 0.98219 | 0.94014 to 1.0305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | UC | wh blood | 21 | 0.92214 | 0.88831 to 0.9615 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | UC | wh blood | 21 | 0.91273 | 0.8736 to 0.93568 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | UC | wh blood | 21 | 0.90348 | 0.87832 to 0.96396 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | UC | wh blood | 21 | 0.92609 | 0.89898 to 0.99068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | UC | wh blood | 21 | 0.93038 | 0.90733 to 0.99337 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | UC | wh blood | 21 | 0.93929 | 0.91921 to 0.98448 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | UC | wh blood | 21 | 0.93623 | 0.9039 to 0.97181 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | wh blood | 21 | 0.93214 | 0.90138 to 1.0028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | UC | wh blood | 21 | 0.94129 | 0.90339 to 0.97293 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | UC | wh blood | 21 | 0.93524 | 0.90748 to 0.98304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | UC | wh blood | 21 | 0.93115 | 0.88247 to 0.96864 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | UC | wh blood | 21 | 0.9693 | 0.92558 to 0.98532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | UC | wh blood | 21 | 0.96328 | 0.94409 to 0.97971 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | UC | wh blood | 21 | 0.95489 | 0.9208 to 0.97108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | UC | wh blood | 21 | 0.94877 | 0.91576 to 0.96809 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | UC | wh blood | 21 | 0.95397 | 0.92192 to 0.96799 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | UC | wh blood | 21 | 0.94933 | 0.91368 to 0.96675 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | UC | wh blood | 21 | 0.92079 | 0.88121 to 0.95863 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | UC | wh blood | 21 | 0.96645 | 0.93809 to 1.0029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | UC | wh blood | 21 | 0.9354 | 0.90179 to 0.95935 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | UC | wh blood | 21 | 0.92656 | 0.90121 to 0.97333 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | UC | wh blood | 21 | 0.92536 | 0.88425 to 0.96678 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | UC | wh blood | 21 | 0.93788 | 0.89471 to 0.98392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | UC | wh blood | 21 | 0.93495 | 0.90362 to 0.95471 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | UC | wh blood | 21 | 0.93218 | 0.89935 to 0.9674 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | UC | wh blood | 21 | 0.95133 | 0.91988 to 1.0113 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | UC | wh blood | 21 | 0.95027 | 0.9143 to 0.99512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | UC | wh blood | 21 | 0.94416 | 0.9034 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | UC | wh blood | 21 | 0.91914 | 0.89204 to 0.94197 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | UC | wh blood | 21 | 0.93039 | 0.90794 to 0.96878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | UC | wh blood | 21 | 0.97545 | 0.95702 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | UC | wh blood | 21 | 0.94626 | 0.92382 to 0.97249 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | UC | wh blood | 18 | 0.95646 | 0.92509 to 0.96958 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | UC | wh blood | 21 | 0.97636 | 0.95336 to 1.0003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | UC | wh blood | 16 | 0.93126 | 0.89819 to 0.95158 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | UC | wh blood | 21 | 0.93013 | 0.9073 to 0.98758 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | UC | wh blood | 21 | 0.98695 | 0.94605 to 1.0518 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | UC | wh blood | 21 | 0.93729 | 0.90931 to 1.0035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | UC | wh blood | 21 | 0.96415 | 0.93374 to 0.98488 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | UC | wh blood | 21 | 0.95235 | 0.91947 to 0.96529 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | UC | wh blood | 21 | 0.91084 | 0.86615 to 0.95603 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | UC | wh blood | 21 | 0.98455 | 0.92844 to 1.0316 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | UC | wh blood | 21 | 1.0105 | 1.0055 to 1.0211 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | UC | wh blood | 19 | 1.0309 | 0.97674 to 1.0617 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | UC | wh blood | 17 | 1.0216 | 0.93192 to 1.1387 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | UC | wh blood | 17 | 0.989 | 0.90768 to 1.0806 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | UC | wh blood | 17 | 1.045 | 0.932 to 1.1884 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | unknown | UC | wh blood | 11 | 1.033 | 0.9495 to 1.0907 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | UC | wh blood | 2 | 0.83645 | 0.83037 to 0.84252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | UC | wh blood | 9 | 1.0947 | 0.96956 to 1.1562 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | UC | wh blood | 16 | 1.0363 | 0.92263 to 1.1351 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | UC | wh blood | 9 | 1.075 | 0.9975 to 1.1399 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | UC | wh blood | 21 | 0.9803 | 0.9072 to 1.0694 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | UC | wh blood | 20 | 0.99335 | 0.92956 to 1.0378 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | UC | wh blood | 13 | 0.925 | 0.87718 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | UC | CD8 | 7 | 0.97031 | 0.94334 to 0.98432 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | CD | wh blood | 13 | 1.0386 | 0.93934 to 1.0857 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | unknown | CD | wh blood | 8 | 1.0692 | 0.954 to 1.1301 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | CD | wh blood | 3 | 0.9398 | 0.83126 to 0.94403 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | CD | wh blood | 8 | 0.9812 | 0.90979 to 1.0864 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | CD | wh blood | 12 | 1.0233 | 0.91057 to 1.1612 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | CD | wh blood | 8 | 1.0956 | 1.0186 to 1.1605 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | CD | wh blood | 9 | 0.9429 | 0.88328 to 0.9895 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | healthy | 20 | 0.94248 | 0.928 to 0.97492 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | healthy | 67 | 0.86943 | 0.80673 to 0.95456 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | healthy | 30 | 1.0514 | 0.97818 to 1.1443 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | healthy | 25 | 1.0048 | 0.9103 to 1.0524 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | healthy | 28 | 1.004 | 0.91341 to 1.1171 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | healthy | 36 | 1.0219 | 0.95088 to 1.0919 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | UC | wh blood | 8 | 0.93968 | 0.91402 to 0.96745 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | UC | wh blood | 12 | 0.94598 | 0.92257 to 0.96709 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | UC | wh blood | 13 | 1.0162 | 0.89368 to 1.0542 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | unknown | healthy | 13 | 1.0097 | 0.9502 to 1.1686 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | healthy | 11 | 1.0731 | 0.9905 to 1.1391 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | healthy | 39 | 0.9808 | 0.88372 to 1.0807 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | healthy | 62 | 0.9724 | 0.8938 to 1.0474 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | healthy | 55 | 0.9735 | 0.924 to 1.0911 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | healthy | 10 | 0.96565 | 0.92889 to 1.0252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | CD | CD8 | 18 | 0.97382 | 0.94504 to 1.0312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | CD | CD8 | 18 | 0.94332 | 0.89947 to 1.0156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | CD | CD8 | 18 | 0.91952 | 0.88302 to 0.98565 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | CD | CD8 | 18 | 0.9865 | 0.93513 to 1.0489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | CD | CD8 | 18 | 0.98383 | 0.95955 to 1.0305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | CD | CD8 | 18 | 0.94688 | 0.91626 to 0.98871 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | CD | CD8 | 18 | 0.92709 | 0.89802 to 0.97151 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | CD | CD8 | 18 | 0.94272 | 0.90872 to 0.99534 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | CD | CD8 | 18 | 0.94399 | 0.92144 to 0.98207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | CD | CD8 | 18 | 0.95098 | 0.92132 to 0.98893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | CD | CD8 | 18 | 0.94856 | 0.90366 to 0.98571 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | CD | CD8 | 18 | 0.97253 | 0.93891 to 1.005 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | CD | CD8 | 18 | 0.90879 | 0.87312 to 0.97236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | CD | CD8 | 15 | 0.95713 | 0.9164 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | CD | CD8 | 4 | 0.96281 | 0.94552 to 0.97127 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | CD | CD8 | 11 | 0.95485 | 0.9285 to 1.0017 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | CD | CD8 | 6 | 0.95331 | 0.93093 to 0.96803 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | CD | CD8 | 18 | 0.93807 | 0.90014 to 0.994 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | CD | CD8 | 18 | 0.97646 | 0.94758 to 1.0381 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | CD | CD8 | 18 | 0.93217 | 0.89387 to 1.0063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | CD | CD8 | 18 | 0.92667 | 0.88362 to 0.98733 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | CD | CD8 | 18 | 0.93132 | 0.87537 to 1.0085 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | CD | CD8 | 18 | 0.9618 | 0.89461 to 1.0428 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | CD | CD8 | 18 | 0.95464 | 0.90296 to 1.0188 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | CD | CD8 | 14 | 0.9653 | 0.92935 to 1.031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | CD | CD8 | 10 | 0.96085 | 0.91812 to 1.0309 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | CD | CD8 | 18 | 0.94342 | 0.91812 to 0.98807 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | CD8 | 18 | 0.97426 | 0.8975 to 1.0642 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | CD | CD8 | 18 | 0.95969 | 0.91441 to 1.0417 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | CD | CD8 | 18 | 0.9472 | 0.89648 to 1.0248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | CD | CD8 | 9 | 0.96356 | 0.94178 to 0.99684 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | CD | CD8 | 18 | 0.94903 | 0.91144 to 0.99068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | CD | CD8 | 18 | 0.98255 | 0.95075 to 1.0218 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | CD | CD8 | 8 | 0.95771 | 0.93749 to 0.99718 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | CD | CD8 | 18 | 0.96853 | 0.94228 to 0.99633 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | CD | CD8 | 18 | 0.96277 | 0.92483 to 0.99463 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | CD | CD8 | 14 | 0.96223 | 0.92975 to 1.0044 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | CD | CD8 | 18 | 0.95766 | 0.92826 to 0.98865 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | CD | CD8 | 18 | 0.95954 | 0.92804 to 0.99461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | CD | CD8 | 18 | 0.94993 | 0.89282 to 0.99228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | CD | CD8 | 18 | 0.97914 | 0.93732 to 1.0408 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | CD | CD8 | 14 | 0.94578 | 0.92047 to 0.99408 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | CD | CD8 | 18 | 0.94519 | 0.89276 to 1.0206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | CD | CD8 | 18 | 0.95054 | 0.90223 to 1.019 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | CD | CD8 | 18 | 0.95138 | 0.91359 to 1.0307 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | CD | CD8 | 18 | 0.94822 | 0.91512 to 0.99165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | CD | CD8 | 18 | 0.94399 | 0.90458 to 1.0013 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | CD | CD8 | 18 | 0.95612 | 0.89998 to 1.0358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | CD | CD8 | 18 | 0.96923 | 0.91568 to 1.0437 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | CD | CD8 | 18 | 0.94977 | 0.90447 to 1.0109 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | CD | CD8 | 18 | 0.92445 | 0.88393 to 0.98193 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | CD | CD8 | 18 | 0.94388 | 0.90842 to 0.99807 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | CD | CD8 | 5 | 0.95555 | 0.92942 to 0.96239 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | CD | CD8 | 18 | 0.98778 | 0.95586 to 1.0224 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | CD | CD8 | 8 | 0.94964 | 0.92608 to 0.99185 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | CD | CD8 | 18 | 0.96443 | 0.92665 to 1.0046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | CD | CD8 | 14 | 0.96079 | 0.93531 to 0.9986 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | CD | CD8 | 18 | 0.96761 | 0.94775 to 1.0094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | CD | CD8 | 11 | 0.94675 | 0.92087 to 0.98907 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | CD | CD8 | 18 | 0.95905 | 0.9071 to 1.0205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | CD | CD8 | 18 | 0.98731 | 0.9321 to 1.0917 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | CD | CD8 | 18 | 0.96487 | 0.90369 to 1.0405 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | CD | CD8 | 7 | 0.93407 | 0.91927 to 0.98469 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | CD | CD8 | 11 | 0.94727 | 0.92406 to 0.99111 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | CD | CD8 | 13 | 0.96044 | 0.93665 to 1.0008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | CD | CD8 | 18 | 0.97203 | 0.93697 to 1.0026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | CD | CD8 | 17 | 0.96191 | 0.9296 to 0.99345 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | CD | CD8 | 18 | 0.91478 | 0.88022 to 0.98017 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | CD | CD8 | 18 | 0.97799 | 0.93631 to 1.0367 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | CD | CD8 | 18 | 1.0099 | 1.0017 to 1.018 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | CD | CD8 | 11 | 1.0068 | 0.97055 to 1.0717 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | CD | CD8 | 3 | 1.0399 | 0.99274 to 1.08 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | unknown | CD | CD8 | 12 | 1.0145 | 0.91202 to 1.1026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | CD | CD8 | 7 | 1.0669 | 0.98149 to 1.1464 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | CD | CD8 | 18 | 0.96665 | 0.88812 to 1.0426 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | CD | CD8 | 10 | 1.0728 | 0.95717 to 1.1807 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | CD | CD8 | 10 | 0.96125 | 0.91773 to 1.071 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | unknown | CD | CD8 | 5 | 0.9669 | 0.90582 to 1.0279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | CD8 | 8 | 1.1287 | 0.97308 to 1.1612 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | CD | CD8 | 7 | 0.9796 | 0.91352 to 1.0218 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | CD | CD8 | 10 | 0.96015 | 0.88034 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | CD | monocytes | 9 | 1.0313 | 0.90388 to 1.0902 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | basophils | unknown | CD | CD4 | 14 | 0.95291 | 0.91087 to 0.97596 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | CD | CD4 | 15 | 0.94983 | 0.92435 to 0.96742 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | CD | CD4 | 10 | 0.95972 | 0.93776 to 0.97461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | CD | CD4 | 14 | 0.95978 | 0.92269 to 0.99202 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | CD | CD4 | 12 | 0.96287 | 0.93879 to 0.97832 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | CD | CD4 | 10 | 0.96132 | 0.93288 to 0.97317 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | CD | CD4 | 8 | 0.9528 | 0.93304 to 0.96842 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | CD | CD4 | 11 | 0.94944 | 0.92386 to 0.96434 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | CD | CD4 | 9 | 0.94619 | 0.9009 to 0.96628 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | CD | CD4 | 12 | 0.93796 | 0.9048 to 0.96562 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | CD | CD4 | 15 | 0.94928 | 0.92076 to 0.9672 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | CD | CD4 | 15 | 0.95952 | 0.93198 to 0.97207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | CD | CD4 | 3 | 0.91 | 0.8443 to 0.94753 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | CD | CD4 | 7 | 0.9309 | 0.87283 to 0.99859 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | CD4 | 3 | 1.1042 | 1.0789 to 1.1126 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | CD | CD4 | 7 | 0.9302 | 0.89148 to 1.0771 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | unknown | CD | CD4 | 2 | 1.0555 | 1.0442 to 1.0669 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | CD | CD4 | 6 | 1.0036 | 0.9743 to 1.0852 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | CD | CD4 | 10 | 1.0285 | 0.97847 to 1.0949 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | CD | CD4 | 9 | 1.0055 | 0.93652 to 1.0231 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | CD | CD4 | 5 | 0.9574 | 0.889 to 0.98574 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | CD | wh blood | 3 | 0.96873 | 0.96726 to 0.97479 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | CD | CD8 | 8 | 0.98285 | 0.90902 to 1.0501 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | UC | monocytes | 7 | 0.9359 | 0.89863 to 0.96984 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | unknown | UC | monocytes | 1 | 1.0573 | 1.0573 to 1.0573 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | unknown | UC | monocytes | 3 | 0.9909 | 0.98667 to 1.0118 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | healthy | 11 | 0.95584 | 0.9441 to 0.97576 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | UC | CD8 | 2 | 0.95844 | 0.94604 to 0.97084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | UC | CD8 | 5 | 0.95671 | 0.93219 to 0.98241 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | UC | CD8 | 3 | 1.0104 | 0.9609 to 1.041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | UC | CD8 | 8 | 1.0736 | 1.0058 to 1.1895 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | UC | CD8 | 8 | 0.9798 | 0.90883 to 1.1606 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | UC | CD8 | 3 | 1.0236 | 1.0115 to 1.042 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | CD | wh blood | 5 | 1.0113 | 0.96556 to 1.0613 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | CD | wh blood | 4 | 0.9787 | 0.93291 to 1.0193 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | UC | CD4 | 19 | 0.97679 | 0.94362 to 1.0108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | UC | CD4 | 19 | 0.93915 | 0.90194 to 1.0019 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | UC | CD4 | 19 | 0.91566 | 0.8733 to 0.96316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | UC | CD4 | 19 | 0.9927 | 0.94752 to 1.0533 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | UC | CD4 | 19 | 0.98237 | 0.95221 to 1.0099 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | UC | CD4 | 19 | 0.93602 | 0.89955 to 0.96358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | UC | CD4 | 19 | 0.92132 | 0.88744 to 0.95162 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | UC | CD4 | 19 | 0.94294 | 0.90501 to 0.97431 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | UC | CD4 | 19 | 0.93761 | 0.90578 to 0.97259 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | UC | CD4 | 19 | 0.94391 | 0.91206 to 1.002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | UC | CD4 | 19 | 0.92347 | 0.88576 to 0.96106 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | UC | CD4 | 19 | 0.95837 | 0.93008 to 1.0053 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | UC | CD4 | 19 | 0.90833 | 0.87375 to 0.92791 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | UC | CD4 | 14 | 0.95315 | 0.92718 to 0.97757 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | UC | CD4 | 19 | 0.93543 | 0.89906 to 0.97123 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | UC | CD4 | 19 | 0.97046 | 0.9437 to 1.0201 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | UC | CD4 | 19 | 0.92195 | 0.89856 to 0.97556 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | UC | CD4 | 19 | 0.92519 | 0.88591 to 0.95474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | UC | CD4 | 19 | 0.92904 | 0.88831 to 0.9939 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | UC | CD4 | 19 | 0.95452 | 0.90085 to 1.0504 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | UC | CD4 | 19 | 0.95075 | 0.90806 to 1.0448 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | UC | CD4 | 18 | 0.96045 | 0.92479 to 1.0042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | UC | CD4 | 12 | 0.97355 | 0.9245 to 1.0074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | UC | CD4 | 19 | 0.93601 | 0.90743 to 0.98206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | CD4 | 19 | 0.9689 | 0.91086 to 1.0592 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | UC | CD4 | 19 | 0.95902 | 0.92881 to 0.99893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | UC | CD4 | 19 | 0.93854 | 0.90515 to 1.0358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | UC | CD4 | 19 | 0.93605 | 0.90224 to 0.99121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | UC | CD4 | 19 | 0.97214 | 0.9481 to 1.0086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | UC | CD4 | 19 | 0.95982 | 0.93892 to 1.0002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | UC | CD4 | 19 | 0.95201 | 0.92617 to 0.98047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | UC | CD4 | 18 | 0.95122 | 0.92677 to 0.98186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | UC | CD4 | 19 | 0.94801 | 0.9232 to 0.97393 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | UC | CD4 | 19 | 0.94796 | 0.92471 to 0.98074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | UC | CD4 | 19 | 0.92512 | 0.88229 to 0.98243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | UC | CD4 | 19 | 0.96951 | 0.94195 to 1.034 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | UC | CD4 | 17 | 0.93988 | 0.91149 to 0.97807 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | UC | CD4 | 19 | 0.93212 | 0.90148 to 1.0152 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | UC | CD4 | 19 | 0.93333 | 0.89889 to 1.0102 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | UC | CD4 | 19 | 0.946 | 0.91389 to 1.0135 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | UC | CD4 | 19 | 0.937 | 0.90375 to 0.96408 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | UC | CD4 | 19 | 0.92627 | 0.90191 to 0.98989 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | UC | CD4 | 19 | 0.95546 | 0.91235 to 1.059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | UC | CD4 | 19 | 0.96051 | 0.9236 to 1.038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | UC | CD4 | 19 | 0.93743 | 0.91011 to 1.0286 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | UC | CD4 | 19 | 0.91393 | 0.88162 to 0.9452 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | UC | CD4 | 19 | 0.93262 | 0.89969 to 0.98902 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | UC | CD4 | 19 | 0.98207 | 0.9558 to 1.0205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | UC | CD4 | 19 | 0.94706 | 0.92378 to 1.0083 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | UC | CD4 | 16 | 0.95382 | 0.93183 to 0.98373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | UC | CD4 | 19 | 0.96992 | 0.94691 to 1.0076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | UC | CD4 | 19 | 0.95514 | 0.91207 to 1.0413 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | UC | CD4 | 19 | 0.98026 | 0.951 to 1.0728 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | UC | CD4 | 19 | 0.96701 | 0.91624 to 1.0554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | UC | CD4 | 14 | 0.94784 | 0.92269 to 0.97292 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | UC | CD4 | 15 | 0.95676 | 0.93438 to 0.97818 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | UC | CD4 | 19 | 0.95494 | 0.91539 to 0.97911 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | UC | CD4 | 19 | 0.95199 | 0.92345 to 0.98032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | UC | CD4 | 19 | 0.91291 | 0.87077 to 0.95664 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | UC | CD4 | 19 | 0.98831 | 0.94846 to 1.0415 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | UC | CD4 | 19 | 1.0113 | 1.0017 to 1.0243 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | unknown | UC | CD4 | 12 | 1.021 | 0.93271 to 1.0755 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | UC | CD4 | 12 | 1.0115 | 0.92095 to 1.0677 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | UC | CD4 | 19 | 1.0554 | 0.97836 to 1.1628 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | UC | CD4 | 10 | 1.0014 | 0.95188 to 1.0854 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | CD | monocytes | 5 | 0.95797 | 0.92158 to 0.97912 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | CD | monocytes | 4 | 1.0652 | 1.0003 to 1.0933 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | UC | CD4 | 10 | 0.96288 | 0.9303 to 0.98682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | UC | CD4 | 12 | 0.96292 | 0.93838 to 0.98518 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | UC | CD4 | 11 | 0.96524 | 0.93535 to 0.98938 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | UC | CD4 | 10 | 0.95291 | 0.92282 to 0.97787 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | UC | CD4 | 12 | 0.94396 | 0.90945 to 0.9888 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | UC | CD4 | 9 | 0.941 | 0.89838 to 1.0449 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | CD | CD8 | 2 | 0.9694 | 0.94636 to 0.99244 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | basophils | unknown | UC | CD4 | 16 | 0.94025 | 0.90452 to 0.97039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | UC | CD4 | 6 | 1.0669 | 0.98302 to 1.0944 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | UC | CD4 | 2 | 0.92745 | 0.89798 to 0.95693 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | UC | CD4 | 2 | 1.053 | 0.98477 to 1.1211 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | unknown | CD | CD8 | 3 | 0.8861 | 0.87224 to 0.97484 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | unknown | CD | CD8 | 2 | 0.99715 | 0.95958 to 1.0347 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | CD | CD4 | 8 | 1.0088 | 0.92615 to 1.08 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | CD | CD4 | 3 | 1.0875 | 1.0646 to 1.1159 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | healthy | 17 | 1.0426 | 0.99078 to 1.132 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | healthy | 32 | 0.9722 | 0.88655 to 1.0813 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | healthy | 4 | 0.95792 | 0.94399 to 0.98232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | healthy | 4 | 0.96826 | 0.95452 to 0.98806 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | UC | monocytes | 2 | 0.93784 | 0.92753 to 0.94815 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | UC | monocytes | 3 | 0.92745 | 0.90062 to 0.9664 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | UC | monocytes | 4 | 1.0209 | 0.95718 to 1.0373 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | UC | wh blood | 13 | 0.9571 | 0.8977 to 1.0688 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | unknown | UC | wh blood | 4 | 1.0245 | 0.93954 to 1.1133 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | CD | CD8 | 4 | 1.024 | 0.9771 to 1.059 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | unknown | CD | CD8 | 1 | 1.0883 | 1.0883 to 1.0883 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | unknown | CD | CD8 | 1 | 1.0001 | 1.0001 to 1.0001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | unknown | CD | CD8 | 5 | 0.9683 | 0.94794 to 0.96976 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | unknown | UC | wh blood | 1 | 0.9819 | 0.9819 to 0.9819 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | wh blood | 4 | 1.1016 | 0.97474 to 1.1636 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | UC | wh blood | 6 | 0.9077 | 0.8697 to 0.97805 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | healthy | 5 | 1.0083 | 0.88192 to 1.05 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | unknown | UC | monocytes | 1 | 1.1747 | 1.1747 to 1.1747 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | UC | CD4 | 10 | 0.93212 | 0.9074 to 0.9546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | UC | CD4 | 10 | 1.0051 | 0.94414 to 1.0511 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | UC | CD4 | 5 | 1.04 | 1.0282 to 1.1473 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | unknown | UC | CD4 | 9 | 0.9644 | 0.93288 to 1.0451 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | CD | monocytes | 1 | 1.0887 | 1.0887 to 1.0887 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | CD | monocytes | 1 | 0.998 | 0.998 to 0.998 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | unknown | CD | monocytes | 1 | 0.989 | 0.989 to 0.989 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | unknown | UC | monocytes | 1 | 1.057 | 1.057 to 1.057 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | UC | monocytes | 2 | 1.0119 | 0.98733 to 1.0365 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | monocytes | 2 | 1.0477 | 0.9973 to 1.0981 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | unknown | UC | monocytes | 1 | 1.0281 | 1.0281 to 1.0281 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | unknown | healthy | 2 | 1.0674 | 1.0537 to 1.0811 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | UC | CD4 | 3 | 0.96798 | 0.95686 to 0.96892 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | UC | CD4 | 1 | 0.95267 | 0.95267 to 0.95267 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | UC | CD4 | 5 | 0.9518 | 0.93548 to 0.96251 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | UC | CD4 | 1 | 0.96247 | 0.96247 to 0.96247 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | nk cells | unknown | UC | CD4 | 5 | 0.992 | 0.97424 to 1.0734 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | unknown | CD | CD4 | 3 | 1.0282 | 0.99067 to 1.0298 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | unknown | healthy | 3 | 0.9785 | 0.97283 to 1.0747 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | unknown | healthy | 4 | 0.96735 | 0.86535 to 0.97585 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | UC | CD8 | 1 | 1.0306 | 1.0306 to 1.0306 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | unknown | healthy | 2 | 1.0066 | 0.97681 to 1.0363 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | UC | wh blood | 7 | 0.9223 | 0.88301 to 0.99621 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | UC | CD4 | 1 | 1.0541 | 1.0541 to 1.0541 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | unknown | UC | CD4 | 1 | 0.9601 | 0.9601 to 0.9601 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | UC | CD4 | 2 | 0.9264 | 0.90489 to 0.94791 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | UC | wh blood | 4 | 0.96324 | 0.95442 to 0.97522 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | unknown | UC | wh blood | 5 | 0.97444 | 0.95145 to 0.9759 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | UC | wh blood | 7 | 0.93724 | 0.91844 to 0.97304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | unknown | UC | wh blood | 7 | 0.9523 | 0.93541 to 0.97781 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | UC | wh blood | 1 | 0.95125 | 0.95125 to 0.95125 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | UC | wh blood | 7 | 0.94785 | 0.92541 to 0.97132 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | unknown | UC | wh blood | 4 | 0.94688 | 0.93693 to 0.96363 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | unknown | UC | wh blood | 7 | 0.94076 | 0.92249 to 0.96649 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | UC | wh blood | 7 | 0.92041 | 0.90185 to 0.95474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | UC | wh blood | 7 | 0.94202 | 0.92561 to 0.96735 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | CD | monocytes | 1 | 0.94506 | 0.94506 to 0.94506 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | CD | monocytes | 2 | 0.93572 | 0.91065 to 0.96079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | CD | monocytes | 1 | 1.08 | 1.08 to 1.08 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | unknown | CD | monocytes | 1 | 0.8998 | 0.8998 to 0.8998 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | CD | monocytes | 1 | 1.1411 | 1.1411 to 1.1411 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | unknown | healthy | 1 | 1.1017 | 1.1017 to 1.1017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | unknown | UC | wh blood | 1 | 0.9687 | 0.9687 to 0.9687 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | unknown | UC | wh blood | 1 | 0.9972 | 0.9972 to 0.9972 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | unknown | UC | wh blood | 1 | 1.0385 | 1.0385 to 1.0385 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | unknown | CD | wh blood | 2 | 0.9703 | 0.93016 to 1.0104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | CD | wh blood | 3 | 1.0584 | 1.0404 to 1.0909 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | unknown | healthy | 1 | 0.9565 | 0.9565 to 0.9565 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | unknown | UC | CD8 | 1 | 1.1396 | 1.1396 to 1.1396 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | unknown | UC | CD8 | 1 | 1.0556 | 1.0556 to 1.0556 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | unknown | UC | monocytes | 1 | 0.9915 | 0.9915 to 0.9915 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | unknown | UC | monocytes | 2 | 1.0892 | 1.0525 to 1.1258 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | unknown | healthy | 4 | 1.0263 | 0.93558 to 1.0987 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | unknown | UC | CD8 | 1 | 1.0203 | 1.0203 to 1.0203 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | CD | CD8 | 3 | 0.95092 | 0.94834 to 0.96705 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | UC | CD8 | 1 | 0.97881 | 0.97881 to 0.97881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | UC | CD8 | 1 | 0.98361 | 0.98361 to 0.98361 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | unknown | CD | CD4 | 3 | 0.96662 | 0.95955 to 0.96691 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | UC | CD4 | 1 | 1.1165 | 1.1165 to 1.1165 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | unknown | CD | CD8 | 1 | 0.9494 | 0.9494 to 0.9494 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | unknown | CD | CD4 | 1 | 1.1285 | 1.1285 to 1.1285 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | CD | CD8 | 2 | 0.95572 | 0.9534 to 0.95805 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | UC | monocytes | 1 | 0.92165 | 0.92165 to 0.92165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | UC | monocytes | 1 | 0.93248 | 0.93248 to 0.93248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | unknown | CD | CD4 | 1 | 0.96067 | 0.96067 to 0.96067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | unknown | CD | CD4 | 1 | 0.97042 | 0.97042 to 0.97042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | unknown | CD | CD4 | 1 | 1.005 | 1.005 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | CD | CD4 | 2 | 0.9246 | 0.89697 to 0.95223 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | unknown | UC | CD8 | 1 | 1.0271 | 1.0271 to 1.0271 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | unknown | UC | monocytes | 1 | 0.987 | 0.987 to 0.987 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | unknown | UC | CD8 | 1 | 0.9841 | 0.9841 to 0.9841 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | metabolic HCC case | 221 | 1.0132 | 0.9714 to 1.0539 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | metabolic HCC case | 221 | 0.99559 | 0.94598 to 1.0445 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | metabolic HCC case | 221 | 0.9677 | 0.90036 to 1.0327 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | metabolic HCC case | 221 | 1.0167 | 0.96314 to 1.0663 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | metabolic HCC case | 221 | 1.0167 | 0.97784 to 1.0584 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | metabolic HCC case | 221 | 1.0232 | 0.98688 to 1.0571 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | metabolic HCC case | 221 | 1.0263 | 0.9886 to 1.0602 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | metabolic HCC case | 221 | 1.0163 | 0.96181 to 1.061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | metabolic HCC case | 221 | 1.0132 | 0.97906 to 1.0465 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | metabolic HCC case | 221 | 0.99869 | 0.96798 to 1.0291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | metabolic HCC case | 221 | 0.99389 | 0.94593 to 1.047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | metabolic HCC case | 221 | 1.0123 | 0.97256 to 1.0467 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | metabolic HCC case | 202 | 1.0208 | 0.97772 to 1.0682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | metabolic HCC case | 202 | 1.0496 | 1.0029 to 1.1005 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | metabolic HCC case | 221 | 1.0306 | 0.96625 to 1.0881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | metabolic HCC case | 221 | 1.0032 | 0.95605 to 1.0418 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | metabolic HCC case | 202 | 1.0055 | 0.97528 to 1.0305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | metabolic HCC case | 202 | 1.0022 | 0.97283 to 1.0281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | metabolic HCC case | 217 | 1.0077 | 0.97132 to 1.0347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | metabolic HCC case | 202 | 1.0119 | 0.98363 to 1.0357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | metabolic HCC case | 221 | 1.0139 | 0.94675 to 1.0713 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | metabolic HCC case | 221 | 0.99114 | 0.93771 to 1.0453 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | metabolic HCC case | 221 | 1.0073 | 0.9425 to 1.0603 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | metabolic HCC case | 221 | 1.0325 | 0.96306 to 1.0873 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | metabolic HCC case | 221 | 1.0002 | 0.94428 to 1.0456 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | metabolic HCC case | 221 | 0.9994 | 0.94339 to 1.0461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | metabolic HCC case | 202 | 1.0099 | 0.97239 to 1.0513 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | metabolic HCC case | 221 | 0.99059 | 0.93576 to 1.0416 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | metabolic HCC case | 221 | 1.0177 | 0.96948 to 1.0672 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | metabolic HCC case | 215 | 1.0189 | 0.9707 to 1.0698 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | metabolic HCC case | 221 | 0.98825 | 0.93929 to 1.0334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | metabolic HCC case | 221 | 0.9918 | 0.93562 to 1.0426 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | metabolic HCC case | 221 | 0.99418 | 0.92599 to 1.048 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | metabolic HCC case | 221 | 0.99603 | 0.94325 to 1.0476 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | metabolic HCC case | 214 | 1.0021 | 0.96799 to 1.0305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | metabolic HCC case | 221 | 1.0082 | 0.97321 to 1.0394 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | metabolic HCC case | 221 | 1.006 | 0.96982 to 1.0391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | metabolic HCC case | 202 | 1.0031 | 0.97271 to 1.0299 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | metabolic HCC case | 202 | 1.0025 | 0.9746 to 1.0268 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | metabolic HCC case | 202 | 1.0049 | 0.97429 to 1.0293 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | metabolic HCC case | 202 | 1.0052 | 0.97874 to 1.0293 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | metabolic HCC case | 214 | 1.0047 | 0.96996 to 1.0337 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | metabolic HCC case | 221 | 1.0057 | 0.97757 to 1.0332 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | metabolic HCC case | 221 | 1.0105 | 0.9725 to 1.0422 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | metabolic HCC case | 202 | 1.0175 | 0.97693 to 1.0625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | metabolic HCC case | 221 | 1.0042 | 0.96794 to 1.0352 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | metabolic HCC case | 221 | 1.0099 | 0.97542 to 1.0392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | metabolic HCC case | 221 | 1.0108 | 0.97324 to 1.0421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | metabolic HCC case | 221 | 0.98109 | 0.92904 to 1.0332 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | metabolic HCC case | 202 | 1.011 | 0.96726 to 1.0602 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | metabolic HCC case | 221 | 0.99729 | 0.94584 to 1.0463 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | metabolic HCC case | 221 | 1.0016 | 0.95822 to 1.039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | metabolic HCC case | 221 | 0.98608 | 0.93071 to 1.0361 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | metabolic HCC case | 221 | 0.99186 | 0.94335 to 1.0292 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | metabolic HCC case | 221 | 0.98459 | 0.92246 to 1.0443 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | metabolic HCC case | 221 | 0.99677 | 0.94306 to 1.0489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | metabolic HCC case | 221 | 0.99072 | 0.93896 to 1.0379 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | metabolic HCC case | 221 | 0.97316 | 0.9142 to 1.0312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | metabolic HCC case | 221 | 0.9914 | 0.93148 to 1.0474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | metabolic HCC case | 221 | 0.97655 | 0.91939 to 1.0334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | metabolic HCC case | 221 | 1.0213 | 0.95555 to 1.0723 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | metabolic HCC case | 202 | 1.0472 | 1.0018 to 1.0959 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | metabolic HCC case | 221 | 0.98552 | 0.9393 to 1.0267 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | metabolic HCC case | 202 | 1.0061 | 0.9729 to 1.0365 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | metabolic HCC case | 221 | 1.0069 | 0.97775 to 1.0332 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | metabolic HCC case | 213 | 1.0082 | 0.97301 to 1.0368 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | metabolic HCC case | 221 | 1.0026 | 0.9581 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | metabolic HCC case | 221 | 1.0078 | 0.97218 to 1.038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | metabolic HCC case | 221 | 1.0013 | 0.97131 to 1.0318 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | metabolic HCC case | 202 | 1.0071 | 0.97837 to 1.0299 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | metabolic HCC case | 221 | 0.98904 | 0.94853 to 1.0214 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | metabolic HCC case | 221 | 0.99578 | 0.9405 to 1.0433 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | metabolic HCC case | 202 | 1.0065 | 0.97906 to 1.0317 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | metabolic HCC case | 202 | 1.0169 | 0.97309 to 1.0661 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | metabolic HCC case | 221 | 0.97317 | 0.9114 to 1.0352 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | metabolic HCC case | 221 | 0.98829 | 0.93501 to 1.0322 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | metabolic HCC case | 215 | 0.99971 | 0.95324 to 1.0373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | metabolic HCC case | 217 | 1.0076 | 0.97194 to 1.0362 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | metabolic HCC case | 202 | 1.0083 | 0.97924 to 1.0313 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | metabolic HCC case | 217 | 1.0091 | 0.9756 to 1.0364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | metabolic HCC case | 202 | 1.0177 | 0.9778 to 1.0586 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | metabolic HCC case | 221 | 1.0053 | 0.95939 to 1.0419 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | metabolic HCC case | 202 | 1.0518 | 1.0045 to 1.101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | metabolic HCC case | 221 | 1.0119 | 0.97351 to 1.0427 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | metabolic HCC case | 221 | 0.97002 | 0.90249 to 1.0342 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | metabolic HCC case | 221 | 1.0146 | 0.96745 to 1.06 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | metabolic HCC case | 221 | 1.0038 | 0.9725 to 1.0399 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | metabolic HCC case | 221 | 1.001 | 0.9946 to 1.0108 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | metabolic HCC case | 38 | 0.9778 | 0.87689 to 1.0619 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | metabolic HCC case | 14 | 0.97755 | 0.89596 to 1.0128 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | metabolic HCC case | 215 | 0.9848 | 0.89665 to 1.0606 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | whole blood | metabolic HCC case | 10 | 1.0293 | 0.93932 to 1.1154 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | metabolic HCC case | 208 | 0.9795 | 0.90465 to 1.0533 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | metabolic HCC case | 9 | 1.0112 | 0.98422 to 1.0826 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | metabolic HCC case | 197 | 1.0119 | 0.93818 to 1.0967 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | metabolic HCC case | 207 | 0.9856 | 0.90517 to 1.0689 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | metabolic HCC case | 213 | 1.0135 | 0.92462 to 1.0917 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | metabolic HCC case | 210 | 0.98975 | 0.90663 to 1.0765 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | metabolic HCC case | 213 | 1.0166 | 0.93888 to 1.0916 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | metabolic HCC case | 108 | 0.9896 | 0.90421 to 1.0738 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | metabolic HCC case | 221 | 1.0163 | 0.9423 to 1.116 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | metabolic HCC case | 161 | 1.0371 | 0.9589 to 1.0973 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | metabolic HCC case | 194 | 0.98955 | 0.90099 to 1.083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | metabolic HCC case | 75 | 0.991 | 0.92153 to 1.0806 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | metabolic HCC case | 196 | 1.0093 | 0.9241 to 1.0898 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | metabolic HCC case | 168 | 0.97825 | 0.88896 to 1.0708 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | metabolic HCC case | 161 | 0.9847 | 0.8924 to 1.0897 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | metabolic HCC case | 166 | 0.9887 | 0.91595 to 1.0817 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | healthy | 302 | 1.0055 | 0.95232 to 1.0506 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | healthy | 302 | 0.98826 | 0.92144 to 1.0448 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | healthy | 302 | 0.9634 | 0.90407 to 1.0372 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | healthy | 302 | 1.0076 | 0.93833 to 1.0667 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | healthy | 302 | 1.0099 | 0.95677 to 1.0532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | healthy | 302 | 1.0172 | 0.97098 to 1.0509 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | healthy | 302 | 1.0172 | 0.95307 to 1.0566 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | healthy | 302 | 1.0065 | 0.93654 to 1.063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | healthy | 302 | 1.0081 | 0.96333 to 1.0398 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | healthy | 302 | 0.99278 | 0.94663 to 1.0275 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | healthy | 302 | 0.9914 | 0.94187 to 1.0421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | healthy | 302 | 1.0043 | 0.96197 to 1.0485 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | healthy | 250 | 1.0151 | 0.95907 to 1.0662 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | healthy | 250 | 1.0435 | 0.97468 to 1.0928 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | healthy | 302 | 1.0235 | 0.91889 to 1.0818 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | healthy | 302 | 0.99543 | 0.91116 to 1.0502 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | healthy | 250 | 1.0033 | 0.97508 to 1.0316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | healthy | 250 | 0.99955 | 0.97376 to 1.0264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | healthy | 299 | 0.99949 | 0.96279 to 1.0359 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | healthy | 250 | 1.0101 | 0.98205 to 1.0367 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | healthy | 302 | 1.0053 | 0.91233 to 1.0733 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | healthy | 302 | 0.9852 | 0.92906 to 1.0435 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | healthy | 302 | 0.998 | 0.90409 to 1.0619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | healthy | 302 | 1.0225 | 0.92214 to 1.0881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | healthy | 302 | 0.99085 | 0.90096 to 1.0493 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | healthy | 302 | 0.98905 | 0.9036 to 1.0509 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | healthy | 250 | 1.0053 | 0.95815 to 1.0465 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | healthy | 302 | 0.98059 | 0.90911 to 1.0414 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | healthy | 301 | 1.0086 | 0.95056 to 1.0661 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | healthy | 288 | 1.0101 | 0.93799 to 1.0674 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | healthy | 302 | 0.98372 | 0.93523 to 1.0307 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | healthy | 302 | 0.97959 | 0.89497 to 1.0473 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | healthy | 302 | 0.98577 | 0.93358 to 1.0491 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | healthy | 302 | 0.98623 | 0.92242 to 1.0483 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | healthy | 286 | 0.99653 | 0.96378 to 1.0294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | healthy | 302 | 1.0005 | 0.94054 to 1.0417 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | healthy | 302 | 1.0002 | 0.96063 to 1.0392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | healthy | 250 | 0.99956 | 0.97048 to 1.0302 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | healthy | 250 | 1.0002 | 0.97355 to 1.027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | healthy | 250 | 1.0022 | 0.97485 to 1.0299 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | healthy | 250 | 1.0029 | 0.97498 to 1.0306 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | healthy | 287 | 0.997 | 0.96221 to 1.0355 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | healthy | 302 | 1.0022 | 0.96925 to 1.0311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | healthy | 302 | 1.003 | 0.9635 to 1.0428 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | healthy | 250 | 1.0113 | 0.96139 to 1.0587 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | healthy | 302 | 0.99649 | 0.95928 to 1.0347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | healthy | 302 | 1.0042 | 0.96971 to 1.0388 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | healthy | 302 | 1.003 | 0.96451 to 1.0436 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | healthy | 302 | 0.97845 | 0.92636 to 1.0284 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | healthy | 250 | 1.0047 | 0.95065 to 1.0553 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | healthy | 302 | 0.98815 | 0.92927 to 1.0523 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | healthy | 302 | 0.99423 | 0.94703 to 1.0382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | healthy | 302 | 0.97773 | 0.88869 to 1.0379 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | healthy | 302 | 0.98287 | 0.91239 to 1.0364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | healthy | 302 | 0.97388 | 0.9027 to 1.0459 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | healthy | 302 | 0.98866 | 0.93503 to 1.0417 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | healthy | 302 | 0.98424 | 0.90676 to 1.0406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | healthy | 302 | 0.96509 | 0.90575 to 1.033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | healthy | 302 | 0.98513 | 0.92491 to 1.0485 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | healthy | 302 | 0.96713 | 0.91618 to 1.0333 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | healthy | 302 | 1.0091 | 0.92001 to 1.0756 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | healthy | 250 | 1.0413 | 0.97833 to 1.0899 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | healthy | 302 | 0.97759 | 0.9203 to 1.0281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | healthy | 250 | 1.0019 | 0.96778 to 1.0387 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | healthy | 302 | 1.0033 | 0.96164 to 1.0339 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | healthy | 285 | 1.0014 | 0.9661 to 1.0366 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | healthy | 302 | 0.99474 | 0.94406 to 1.0425 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | healthy | 301 | 1.001 | 0.96558 to 1.0367 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | healthy | 302 | 0.99723 | 0.96043 to 1.0276 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | healthy | 250 | 1.0043 | 0.97614 to 1.0316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | healthy | 301 | 0.98575 | 0.92925 to 1.0218 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | healthy | 302 | 0.98526 | 0.91572 to 1.05 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | healthy | 250 | 1.0054 | 0.97523 to 1.0337 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | healthy | 250 | 1.0107 | 0.95296 to 1.0621 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | healthy | 302 | 0.96188 | 0.90062 to 1.0357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | healthy | 302 | 0.97948 | 0.90669 to 1.0408 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | healthy | 296 | 0.9914 | 0.93893 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | healthy | 300 | 0.99972 | 0.96612 to 1.0373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | healthy | 250 | 1.0053 | 0.97532 to 1.0308 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | healthy | 300 | 1.0021 | 0.96879 to 1.0356 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | healthy | 250 | 1.0127 | 0.96207 to 1.0587 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | healthy | 302 | 0.99921 | 0.94987 to 1.0396 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | healthy | 250 | 1.0444 | 0.9781 to 1.097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | healthy | 302 | 1.004 | 0.96563 to 1.0433 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | healthy | 302 | 0.9647 | 0.90692 to 1.0369 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | healthy | 302 | 1.0072 | 0.94844 to 1.0589 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | healthy | 302 | 0.99735 | 0.96274 to 1.0426 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | healthy | 302 | 1.0021 | 0.9944 to 1.0097 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | healthy | 294 | 0.974 | 0.8997 to 1.0777 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | healthy | 285 | 0.9687 | 0.89678 to 1.0614 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | healthy | 226 | 0.9681 | 0.88803 to 1.0512 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | healthy | 248 | 1.0167 | 0.9433 to 1.1113 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | healthy | 225 | 0.9953 | 0.91106 to 1.0821 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | healthy | 288 | 0.9902 | 0.90445 to 1.0808 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | healthy | 225 | 1.0351 | 0.9357 to 1.1354 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | healthy | 282 | 1.0074 | 0.92282 to 1.0883 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | healthy | 271 | 0.9923 | 0.9154 to 1.083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | healthy | 283 | 0.9938 | 0.91327 to 1.1046 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | healthy | 287 | 1.0228 | 0.93948 to 1.1173 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | healthy | 210 | 0.9889 | 0.92068 to 1.0709 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | healthy | 259 | 0.9956 | 0.911 to 1.0816 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | healthy | 302 | 1.0158 | 0.93261 to 1.0976 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | healthy | 268 | 1.0074 | 0.9209 to 1.0948 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | whole blood | metabolic HCC case | 5 | 0.957 | 0.9441 to 1.0026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | healthy | 119 | 0.9928 | 0.89469 to 1.1023 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | healthy | 74 | 0.9958 | 0.88558 to 1.0893 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | metabolic HCC case | 25 | 0.9814 | 0.908 to 1.0428 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | metabolic HCC case | 27 | 1.0178 | 0.93775 to 1.0978 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | whole blood | metabolic HCC case | 4 | 1.0903 | 0.99852 to 1.1229 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | healthy | 53 | 1.0387 | 0.95442 to 1.141 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | metabolic HCC case | 4 | 0.997 | 0.92563 to 1.0379 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | healthy | 52 | 0.9787 | 0.89715 to 1.0667 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | whole blood | metabolic HCC case | 3 | 1.0276 | 0.93544 to 1.0285 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | whole blood | healthy | 5 | 0.9532 | 0.90498 to 0.9706 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | whole blood | healthy | 17 | 1.0421 | 0.94406 to 1.1669 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | whole blood | healthy | 8 | 1.0012 | 0.88728 to 1.1447 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | whole blood | metabolic HCC case | 7 | 1.0542 | 1.0325 to 1.1036 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | metabolic HCC case | 10 | 1.0278 | 0.97314 to 1.0611 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | whole blood | metabolic HCC case | 1 | 1.0615 | 1.0615 to 1.0615 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | healthy | 5 | 1.0293 | 0.98624 to 1.104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | whole blood | healthy | 4 | 1.0482 | 1.021 to 1.0779 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | healthy | 12 | 1.019 | 0.9509 to 1.0806 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | whole blood | healthy | 3 | 1.0074 | 0.98643 to 1.0262 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | healthy | 3 | 1.0544 | 0.99779 to 1.0911 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | healthy | 12 | 0.97025 | 0.90218 to 1.045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | whole blood | healthy | 1 | 0.8474 | 0.8474 to 0.8474 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | metabolic HCC case | 3 | 1.0046 | 0.97337 to 1.0149 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | metabolic HCC case | 7 | 0.9922 | 0.92742 to 1.0577 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | healthy | 5 | 0.9948 | 0.96832 to 1.0947 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | metabolic HCC case | 5 | 0.9404 | 0.8254 to 1.0159 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | healthy | 36 | 1.0283 | 0.90773 to 1.0928 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric fundus epithelium | whole blood | metabolic HCC case | 3 | 0.9979 | 0.98116 to 1.0738 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | metabolic HCC case | 10 | 0.98015 | 0.90895 to 1.0677 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tongue epithelium | whole blood | metabolic HCC case | 6 | 0.993 | 0.97528 to 1.0938 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | whole blood | healthy | 4 | 1.0162 | 0.96768 to 1.0237 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cardiomyocyte | whole blood | healthy | 1 | 0.95 | 0.95 to 0.95 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | whole blood | healthy | 1 | 0.9827 | 0.9827 to 0.9827 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tongue epithelium | whole blood | healthy | 4 | 0.94325 | 0.93801 to 0.98308 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | fallopien tubes epithelium | whole blood | metabolic HCC case | 3 | 1.0916 | 0.94553 to 1.092 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | metabolic HCC case | 2 | 0.99475 | 0.99425 to 0.99524 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | healthy | 5 | 0.932 | 0.87852 to 1.0611 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | healthy | 5 | 1.027 | 0.9537 to 1.0643 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | whole blood | metabolic HCC case | 4 | 0.9812 | 0.92122 to 1.129 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | whole blood | metabolic HCC case | 4 | 0.9883 | 0.94546 to 1.022 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | whole blood | metabolic HCC case | 5 | 0.9968 | 0.91814 to 1.0711 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric body epithelium | whole blood | metabolic HCC case | 1 | 0.9668 | 0.9668 to 0.9668 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tonsil palatine epithelium | whole blood | metabolic HCC case | 1 | 0.9422 | 0.9422 to 0.9422 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | whole blood | metabolic HCC case | 2 | 1.0112 | 1.0062 to 1.0162 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | whole blood | metabolic HCC case | 1 | 1.0519 | 1.0519 to 1.0519 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | whole blood | healthy | 6 | 0.99095 | 0.97023 to 1.0514 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric fundus epithelium | whole blood | healthy | 1 | 1.074 | 1.074 to 1.074 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | healthy | 19 | 1.0202 | 0.94519 to 1.1208 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | bladder epithelium | whole blood | metabolic HCC case | 1 | 0.9758 | 0.9758 to 0.9758 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric body epithelium | whole blood | healthy | 1 | 1.061 | 1.061 to 1.061 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | healthy | 4 | 0.9683 | 0.9495 to 1.1103 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney glomerular epithelium | whole blood | healthy | 1 | 0.8901 | 0.8901 to 0.8901 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | whole blood | healthy | 1 | 1.0467 | 1.0467 to 1.0467 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | whole blood | healthy | 5 | 1.0254 | 0.99232 to 1.0605 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | whole blood | healthy | 2 | 0.9459 | 0.88065 to 1.0111 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | healthy | 6 | 0.9855 | 0.90305 to 1.0799 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | metabolic HCC case | 1 | 1.0161 | 1.0161 to 1.0161 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | whole blood | healthy | 19 | 0.9447 | 0.9054 to 1.0436 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | healthy | 28 | 0.96195 | 0.86523 to 1.0479 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | whole blood | healthy | 11 | 1.0394 | 0.92815 to 1.149 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | whole blood | healthy | 3 | 0.9396 | 0.88587 to 0.95949 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | whole blood | healthy | 9 | 0.9706 | 0.9043 to 1.1039 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | whole blood | healthy | 3 | 0.9787 | 0.94432 to 0.98041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | whole blood | healthy | 1 | 0.996 | 0.996 to 0.996 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | whole blood | healthy | 1 | 0.903 | 0.903 to 0.903 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic alpha | whole blood | healthy | 1 | 0.9874 | 0.9874 to 0.9874 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | cardioembolic | 109 | 1.0079 | 0.94755 to 1.1035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | cardioembolic | 109 | 1.0334 | 0.95071 to 1.205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | cardioembolic | 109 | 1.0709 | 0.93356 to 1.2253 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | cardioembolic | 109 | 1.0135 | 0.94424 to 1.1141 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | cardioembolic | 109 | 0.99777 | 0.94073 to 1.0688 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | cardioembolic | 109 | 1.0036 | 0.93088 to 1.0823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | cardioembolic | 109 | 1.0011 | 0.9092 to 1.0981 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | cardioembolic | 109 | 1.0093 | 0.94305 to 1.1264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | cardioembolic | 109 | 1.0038 | 0.94392 to 1.0802 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | cardioembolic | 109 | 1.0147 | 0.95502 to 1.1059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | cardioembolic | 109 | 1.0067 | 0.9345 to 1.0983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | cardioembolic | 109 | 1.0087 | 0.94873 to 1.1107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | cardioembolic | 109 | 1.0023 | 0.91304 to 1.1744 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | cardioembolic | 87 | 1.0022 | 0.9354 to 1.1033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | cardioembolic | 75 | 0.98756 | 0.90174 to 1.0611 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | cardioembolic | 57 | 0.99008 | 0.91591 to 1.0229 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | cardioembolic | 109 | 1.0387 | 0.95384 to 1.2368 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | cardioembolic | 109 | 1.0652 | 0.9893 to 1.2255 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | cardioembolic | 107 | 1.0606 | 0.96924 to 1.2258 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | cardioembolic | 109 | 1.0207 | 0.93364 to 1.199 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | cardioembolic | 104 | 1.0537 | 0.95193 to 1.1591 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | cardioembolic | 109 | 1.0488 | 0.94637 to 1.1991 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | cardioembolic | 109 | 1.0495 | 0.96325 to 1.205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | cardioembolic | 92 | 1.0062 | 0.91554 to 1.0959 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | cardioembolic | 73 | 0.99651 | 0.90326 to 1.0927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | cardioembolic | 109 | 1.0481 | 0.97372 to 1.1798 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | cardioembolic | 109 | 1.0588 | 0.95426 to 1.2138 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | cardioembolic | 108 | 1.0454 | 0.94617 to 1.2098 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | cardioembolic | 109 | 1.0544 | 0.97162 to 1.2421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | cardioembolic | 70 | 0.9848 | 0.90046 to 1.0478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | cardioembolic | 97 | 1.0162 | 0.93557 to 1.0983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | cardioembolic | 101 | 0.99803 | 0.92509 to 1.0801 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | cardioembolic | 70 | 0.98698 | 0.9005 to 1.0543 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | cardioembolic | 109 | 0.99947 | 0.95507 to 1.0632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | cardioembolic | 97 | 1.0051 | 0.92316 to 1.0765 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | cardioembolic | 95 | 0.99993 | 0.91902 to 1.0716 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | cardioembolic | 99 | 0.99384 | 0.91513 to 1.0673 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | cardioembolic | 98 | 1.0027 | 0.92046 to 1.0744 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | cardioembolic | 106 | 1.0142 | 0.9308 to 1.122 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | cardioembolic | 109 | 1.0431 | 0.96848 to 1.1714 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | cardioembolic | 95 | 1.0085 | 0.91554 to 1.0809 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | cardioembolic | 94 | 1.0592 | 0.97523 to 1.1908 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | cardioembolic | 94 | 1.0178 | 0.92947 to 1.115 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | cardioembolic | 106 | 1.0639 | 0.97011 to 1.2178 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | cardioembolic | 108 | 1.024 | 0.95345 to 1.1768 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | cardioembolic | 104 | 1.0713 | 0.97586 to 1.1926 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | cardioembolic | 109 | 1.0652 | 0.97528 to 1.2755 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | cardioembolic | 109 | 1.0579 | 0.97558 to 1.2478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | cardioembolic | 109 | 1.036 | 0.94043 to 1.2625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | cardioembolic | 109 | 1.0103 | 0.93784 to 1.1846 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | cardioembolic | 102 | 1.0314 | 0.96002 to 1.1344 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | cardioembolic | 47 | 0.98193 | 0.89713 to 1.0144 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | cardioembolic | 109 | 1.0075 | 0.95535 to 1.0628 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | cardioembolic | 69 | 0.98696 | 0.90959 to 1.0585 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | cardioembolic | 109 | 1.0328 | 0.9471 to 1.141 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | cardioembolic | 91 | 1.0028 | 0.92777 to 1.0704 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | cardioembolic | 109 | 1.0021 | 0.95136 to 1.0607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | cardioembolic | 109 | 1.047 | 0.94423 to 1.1953 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | cardioembolic | 104 | 1.0812 | 0.98217 to 1.2823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | cardioembolic | 109 | 1.0354 | 0.94347 to 1.1713 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | cardioembolic | 71 | 0.98334 | 0.89459 to 1.0715 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | cardioembolic | 79 | 0.9962 | 0.91416 to 1.0614 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | cardioembolic | 81 | 0.9893 | 0.90681 to 1.0579 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | cardioembolic | 109 | 1.0216 | 0.93413 to 1.1076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | cardioembolic | 97 | 1.0048 | 0.9247 to 1.076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | cardioembolic | 109 | 1.0697 | 0.933 to 1.2217 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | cardioembolic | 109 | 1.0075 | 0.94431 to 1.0949 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | cardioembolic | 109 | 1.0052 | 0.99496 to 1.016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | whole blood | cardioembolic | 63 | 0.9887 | 0.90521 to 1.1044 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | whole blood | cardioembolic | 27 | 0.9973 | 0.88873 to 1.097 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | cardioembolic | 40 | 1.0519 | 0.94483 to 1.1278 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | cardioembolic | 56 | 1.025 | 0.95242 to 1.1427 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | cardioembolic | 36 | 1.0297 | 0.88065 to 1.1664 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | whole blood | cardioembolic | 7 | 1.0718 | 0.91326 to 1.1431 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | whole blood | cardioembolic | 33 | 1.0077 | 0.92132 to 1.1278 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | cardioembolic | 96 | 1.0078 | 0.87282 to 1.1405 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | cardioembolic | 109 | 1.0007 | 0.92026 to 1.0774 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | whole blood | cardioembolic | 53 | 0.9729 | 0.89146 to 1.0634 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | whole blood | cardioembolic | 31 | 0.9809 | 0.8906 to 1.1191 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | cardioembolic | 68 | 1.0128 | 0.95116 to 1.0896 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | cardioembolic | 14 | 0.9798 | 0.85342 to 1.0479 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | whole blood | cardioembolic | 6 | 0.9985 | 0.92548 to 1.0624 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | cardioembolic | 73 | 0.9901 | 0.91216 to 1.0832 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | cardioembolic | 18 | 1.0303 | 0.89029 to 1.1859 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | cardioembolic | 19 | 1.0185 | 0.90269 to 1.092 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | cardioembolic | 42 | 1.0331 | 0.91361 to 1.1504 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | whole blood | cardioembolic | 73 | 0.9772 | 0.90194 to 1.0652 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | whole blood | cardioembolic | 52 | 0.99465 | 0.91801 to 1.1042 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | cardioembolic | 46 | 1.0128 | 0.9283 to 1.0661 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | cardioembolic | 14 | 0.9771 | 0.92649 to 1.086 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | cardioembolic | 9 | 1.0488 | 1.0053 to 1.1323 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | cardioembolic | 13 | 1.0214 | 0.88364 to 1.0899 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | whole blood | cardioembolic | 80 | 0.9789 | 0.89494 to 1.0662 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | cardioembolic | 10 | 1.0641 | 1.0138 to 1.112 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | small-vessel disease | 57 | 1.0256 | 0.96266 to 1.1036 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | small-vessel disease | 57 | 1.0545 | 0.94763 to 1.1894 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | small-vessel disease | 57 | 1.1119 | 0.92279 to 1.2743 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | small-vessel disease | 57 | 1.0265 | 0.94087 to 1.1228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | small-vessel disease | 57 | 1.0074 | 0.95692 to 1.0707 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | small-vessel disease | 57 | 1.0229 | 0.96486 to 1.0955 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | small-vessel disease | 57 | 1.0279 | 0.96291 to 1.1125 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | small-vessel disease | 57 | 1.0281 | 0.95864 to 1.1234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | small-vessel disease | 57 | 1.017 | 0.97294 to 1.0812 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | small-vessel disease | 57 | 1.0237 | 0.96521 to 1.1131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | small-vessel disease | 57 | 1.0063 | 0.9387 to 1.0831 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | small-vessel disease | 57 | 1.0279 | 0.96433 to 1.105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | small-vessel disease | 57 | 1.0396 | 0.93534 to 1.1507 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | small-vessel disease | 46 | 1.0239 | 0.91484 to 1.1267 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | small-vessel disease | 19 | 0.99215 | 0.95455 to 1.0144 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | small-vessel disease | 34 | 1.0028 | 0.9518 to 1.0778 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | small-vessel disease | 24 | 0.99447 | 0.95688 to 1.0267 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | small-vessel disease | 57 | 1.0555 | 0.9514 to 1.1927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | small-vessel disease | 57 | 1.0759 | 0.97783 to 1.2371 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | small-vessel disease | 57 | 1.0887 | 0.96908 to 1.2482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | small-vessel disease | 57 | 1.0527 | 0.93801 to 1.1651 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | small-vessel disease | 55 | 1.0764 | 0.95696 to 1.2012 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | small-vessel disease | 57 | 1.0617 | 0.94017 to 1.1844 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | small-vessel disease | 57 | 1.0554 | 0.95867 to 1.2039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | small-vessel disease | 46 | 1.0229 | 0.95175 to 1.109 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | small-vessel disease | 33 | 1.0056 | 0.9404 to 1.1 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | small-vessel disease | 57 | 1.0571 | 0.96944 to 1.1761 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | small-vessel disease | 57 | 1.068 | 0.93961 to 1.2033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | small-vessel disease | 57 | 1.0617 | 0.96063 to 1.1941 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | small-vessel disease | 57 | 1.0576 | 0.96009 to 1.2624 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | small-vessel disease | 32 | 0.99237 | 0.94768 to 1.0602 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | small-vessel disease | 53 | 1.04 | 0.95993 to 1.1198 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | small-vessel disease | 53 | 1.0219 | 0.94517 to 1.0932 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | small-vessel disease | 30 | 1 | 0.95257 to 1.0677 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | small-vessel disease | 57 | 1.0119 | 0.97208 to 1.0713 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | small-vessel disease | 50 | 1.0278 | 0.95678 to 1.1011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | small-vessel disease | 47 | 1.0186 | 0.95505 to 1.0764 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | small-vessel disease | 52 | 1.0214 | 0.95124 to 1.0866 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | small-vessel disease | 51 | 1.0265 | 0.95687 to 1.0998 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | small-vessel disease | 57 | 1.027 | 0.92605 to 1.1256 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | small-vessel disease | 57 | 1.0442 | 0.94269 to 1.1489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | small-vessel disease | 48 | 1.0288 | 0.95223 to 1.1108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | small-vessel disease | 52 | 1.0844 | 0.96977 to 1.2309 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | small-vessel disease | 50 | 1.0488 | 0.94218 to 1.1545 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | small-vessel disease | 57 | 1.0797 | 0.95825 to 1.2397 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | small-vessel disease | 57 | 1.043 | 0.95374 to 1.1572 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | small-vessel disease | 57 | 1.0884 | 0.98012 to 1.2339 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | small-vessel disease | 57 | 1.073 | 0.96547 to 1.2891 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | small-vessel disease | 57 | 1.0746 | 0.96826 to 1.2553 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | small-vessel disease | 57 | 1.0285 | 0.926 to 1.2573 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | small-vessel disease | 57 | 1.0352 | 0.95036 to 1.1416 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | small-vessel disease | 54 | 1.0509 | 0.96335 to 1.1736 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | small-vessel disease | 20 | 0.99596 | 0.95083 to 1.0334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | small-vessel disease | 57 | 1.0201 | 0.9657 to 1.0766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | small-vessel disease | 31 | 1.0001 | 0.95376 to 1.0703 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | small-vessel disease | 57 | 1.0446 | 0.96471 to 1.1385 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | small-vessel disease | 44 | 1.0216 | 0.96014 to 1.0822 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | small-vessel disease | 57 | 1.0112 | 0.96272 to 1.0711 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | small-vessel disease | 42 | 1.0368 | 0.95626 to 1.0917 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | small-vessel disease | 57 | 1.0524 | 0.94772 to 1.1864 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | small-vessel disease | 55 | 1.0821 | 0.9603 to 1.2734 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | small-vessel disease | 57 | 1.0509 | 0.93022 to 1.1666 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | small-vessel disease | 33 | 0.99877 | 0.93036 to 1.0822 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | small-vessel disease | 38 | 1.0121 | 0.96248 to 1.0761 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | small-vessel disease | 40 | 1.0023 | 0.96055 to 1.0651 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | small-vessel disease | 57 | 1.045 | 0.97107 to 1.1216 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | small-vessel disease | 49 | 1.023 | 0.95675 to 1.1016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | small-vessel disease | 57 | 1.109 | 0.92341 to 1.2676 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | small-vessel disease | 57 | 1.0204 | 0.94419 to 1.1057 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | small-vessel disease | 57 | 1.0059 | 0.9963 to 1.0203 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | whole blood | small-vessel disease | 28 | 0.96725 | 0.9049 to 1.0734 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | small-vessel disease | 17 | 1.0129 | 0.93568 to 1.102 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | small-vessel disease | 42 | 0.9868 | 0.90963 to 1.0577 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | small-vessel disease | 29 | 1.0345 | 0.92286 to 1.1566 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | small-vessel disease | 39 | 1.0192 | 0.93559 to 1.1088 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | whole blood | small-vessel disease | 43 | 0.968 | 0.89962 to 1.068 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | small-vessel disease | 36 | 0.99765 | 0.91297 to 1.1193 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | small-vessel disease | 8 | 1.0098 | 0.92453 to 1.1118 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | small-vessel disease | 51 | 1.0155 | 0.89375 to 1.1184 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | small-vessel disease | 57 | 1.0138 | 0.91562 to 1.1101 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | whole blood | small-vessel disease | 23 | 0.9841 | 0.90538 to 1.07 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | whole blood | small-vessel disease | 13 | 1.0196 | 0.92612 to 1.0854 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | whole blood | cardioembolic | 11 | 0.9981 | 0.86865 to 1.0702 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | cardioembolic | 4 | 0.9558 | 0.88955 to 1.1122 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | cardioembolic | 35 | 0.9668 | 0.83664 to 1.1024 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | cardioembolic | 11 | 0.9838 | 0.90775 to 1.1147 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | whole blood | cardioembolic | 37 | 1.0001 | 0.92034 to 1.1249 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | small-vessel disease | 25 | 1.038 | 0.93506 to 1.1549 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | whole blood | small-vessel disease | 25 | 1.0023 | 0.95338 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | whole blood | small-vessel disease | 5 | 0.9095 | 0.7896 to 1.0318 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | whole blood | small-vessel disease | 30 | 0.96745 | 0.89028 to 1.049 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | small-vessel disease | 11 | 0.9879 | 0.93045 to 1.1252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | small-vessel disease | 4 | 0.99225 | 0.91091 to 1.0121 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | small-vessel disease | 10 | 1.0042 | 0.88433 to 1.1674 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | small-vessel disease | 20 | 0.99165 | 0.80234 to 1.0683 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | small-vessel disease | 6 | 1.0119 | 0.92495 to 1.1667 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | small-vessel disease | 32 | 0.97955 | 0.86685 to 1.0597 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | small-vessel disease | 10 | 1.0365 | 0.90532 to 1.1277 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | small-vessel disease | 5 | 1.1009 | 0.96388 to 1.1101 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | cardioembolic | 36 | 0.98512 | 0.90475 to 1.0066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | cardioembolic | 8 | 0.98134 | 0.95481 to 1.001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | cardioembolic | 7 | 0.98553 | 0.94958 to 0.99526 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | cardioembolic | 25 | 0.98537 | 0.9369 to 0.99504 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | cardioembolic | 6 | 0.98238 | 0.94328 to 0.99477 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | cardioembolic | 24 | 0.98872 | 0.94505 to 0.99889 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | cardioembolic | 9 | 0.98311 | 0.95327 to 1.0191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | cardioembolic | 10 | 0.98389 | 0.95525 to 0.99498 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | cardioembolic | 29 | 0.9981 | 0.89 to 1.1371 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | cardioembolic | 32 | 0.99395 | 0.86721 to 1.0779 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | whole blood | cardioembolic | 2 | 1.0038 | 0.99066 to 1.0169 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | cardioembolic | 18 | 1.0177 | 0.92952 to 1.1022 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | cardioembolic | 26 | 0.9938 | 0.9129 to 1.0882 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | cardioembolic | 10 | 0.99695 | 0.90831 to 1.0706 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | whole blood | cardioembolic | 8 | 1.0439 | 0.93631 to 1.0812 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | cardioembolic | 15 | 0.9694 | 0.90535 to 1.0871 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | small-vessel disease | 22 | 1.0008 | 0.93327 to 1.0409 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | whole blood | small-vessel disease | 1 | 0.9124 | 0.9124 to 0.9124 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | whole blood | small-vessel disease | 8 | 1.0343 | 0.97637 to 1.0955 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | cardioembolic | 14 | 0.9412 | 0.85674 to 1.0521 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | cardioembolic | 3 | 0.9383 | 0.91517 to 1.0802 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | small-vessel disease | 16 | 1.009 | 0.97762 to 1.1098 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | small-vessel disease | 7 | 0.974 | 0.85255 to 1.0051 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | small-vessel disease | 8 | 1.0537 | 0.96992 to 1.1444 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | small-vessel disease | 20 | 0.9756 | 0.89348 to 1.0349 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | small-vessel disease | 12 | 0.98695 | 0.90793 to 1.0468 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | whole blood | cardioembolic | 2 | 0.9932 | 0.93965 to 1.0468 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tonsil pharyngeal epithelium | whole blood | cardioembolic | 3 | 1.0228 | 0.94783 to 1.0732 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | cardioembolic | 33 | 0.9917 | 0.91268 to 1.1291 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | cardioembolic | 28 | 1.0103 | 0.92483 to 1.0776 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | small-vessel disease | 15 | 0.99244 | 0.95024 to 1.0225 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | small-vessel disease | 15 | 0.99033 | 0.95684 to 1.0248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | whole blood | small-vessel disease | 8 | 1.0164 | 0.88252 to 1.1248 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | whole blood | cardioembolic | 7 | 1.0469 | 0.89926 to 1.1271 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | cardioembolic | 10 | 1.0135 | 0.83124 to 1.1527 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | whole blood | small-vessel disease | 6 | 0.97475 | 0.92245 to 0.9936 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | small-vessel disease | 15 | 0.9994 | 0.92798 to 1.1275 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | small-vessel disease | 4 | 1.0442 | 0.96758 to 1.0825 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | cardioembolic | 3 | 1.0242 | 0.88182 to 1.1163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | small-vessel disease | 3 | 0.9361 | 0.88129 to 0.94879 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | fallopien tubes epithelium | whole blood | cardioembolic | 1 | 0.8852 | 0.8852 to 0.8852 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | small-vessel disease | 7 | 0.99328 | 0.94983 to 1.021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | small-vessel disease | 6 | 0.99527 | 0.94705 to 1.023 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | small-vessel disease | 5 | 0.9969 | 0.94504 to 1.0226 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | small-vessel disease | 5 | 0.99221 | 0.94222 to 1.0206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | small-vessel disease | 5 | 1.0073 | 0.93734 to 1.0414 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | small-vessel disease | 9 | 0.99363 | 0.95423 to 1.02 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | basophils | whole blood | small-vessel disease | 8 | 1.0191 | 0.95181 to 1.0851 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | atherotrombotic | 18 | 1.0293 | 0.97349 to 1.0585 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | atherotrombotic | 18 | 1.0609 | 0.97109 to 1.1111 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | atherotrombotic | 18 | 1.1196 | 1.0024 to 1.1877 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | atherotrombotic | 18 | 1.0221 | 0.96805 to 1.0652 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | atherotrombotic | 18 | 1.009 | 0.96328 to 1.0291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | atherotrombotic | 18 | 1.0271 | 0.97402 to 1.0461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | atherotrombotic | 18 | 1.037 | 0.97337 to 1.0591 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | atherotrombotic | 18 | 1.033 | 0.97477 to 1.0751 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | atherotrombotic | 18 | 1.022 | 0.97625 to 1.0406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | atherotrombotic | 18 | 1.0318 | 0.98434 to 1.0533 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | atherotrombotic | 18 | 1.0051 | 0.93794 to 1.0348 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | atherotrombotic | 18 | 1.0314 | 0.97421 to 1.0583 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | atherotrombotic | 18 | 1.0511 | 0.95721 to 1.0996 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | atherotrombotic | 16 | 1.0523 | 0.95116 to 1.0989 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | atherotrombotic | 18 | 1.0546 | 0.97846 to 1.1228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | atherotrombotic | 18 | 1.0707 | 0.99517 to 1.1404 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | atherotrombotic | 18 | 1.0963 | 0.99924 to 1.1509 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | atherotrombotic | 18 | 1.0528 | 0.97243 to 1.1177 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | atherotrombotic | 18 | 1.091 | 0.99273 to 1.1515 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | atherotrombotic | 18 | 1.0695 | 0.96077 to 1.1293 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | atherotrombotic | 18 | 1.0689 | 0.9779 to 1.1258 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | atherotrombotic | 18 | 1.0609 | 0.99642 to 1.1054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | atherotrombotic | 18 | 1.0739 | 0.95984 to 1.1469 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | atherotrombotic | 18 | 1.0682 | 0.98276 to 1.1151 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | atherotrombotic | 18 | 1.0627 | 0.98146 to 1.1355 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | atherotrombotic | 18 | 1.0594 | 0.98891 to 1.0847 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | atherotrombotic | 18 | 1.0261 | 0.97245 to 1.0569 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | atherotrombotic | 18 | 1.0157 | 0.97525 to 1.0347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | atherotrombotic | 18 | 1.0332 | 0.98331 to 1.0614 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | atherotrombotic | 18 | 1.0249 | 0.96981 to 1.0483 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | atherotrombotic | 18 | 1.0334 | 0.98015 to 1.0624 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | atherotrombotic | 18 | 1.0233 | 0.94738 to 1.0568 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | atherotrombotic | 18 | 1.0474 | 0.96738 to 1.1009 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | atherotrombotic | 18 | 1.1087 | 0.9958 to 1.1698 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | atherotrombotic | 18 | 1.0681 | 0.97937 to 1.1162 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | atherotrombotic | 18 | 1.0954 | 0.99084 to 1.1459 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | atherotrombotic | 18 | 1.0363 | 0.9783 to 1.0907 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | atherotrombotic | 18 | 1.1003 | 1.0031 to 1.1467 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | atherotrombotic | 18 | 1.0767 | 0.98975 to 1.1575 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | atherotrombotic | 18 | 1.0763 | 0.98742 to 1.1348 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | atherotrombotic | 18 | 1.0316 | 0.94902 to 1.1364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | atherotrombotic | 18 | 1.0384 | 0.96275 to 1.0814 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | atherotrombotic | 18 | 1.0686 | 0.99237 to 1.1073 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | atherotrombotic | 18 | 1.0169 | 0.98476 to 1.0382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | atherotrombotic | 18 | 1.0492 | 0.98967 to 1.0876 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | atherotrombotic | 18 | 1.0097 | 0.97741 to 1.0315 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | atherotrombotic | 18 | 1.0693 | 0.96931 to 1.119 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | atherotrombotic | 18 | 1.084 | 0.99723 to 1.2047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | atherotrombotic | 18 | 1.053 | 0.94967 to 1.1111 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | atherotrombotic | 18 | 1.0474 | 0.97675 to 1.0848 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | atherotrombotic | 17 | 1.0309 | 0.97951 to 1.0634 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | atherotrombotic | 18 | 1.1176 | 1.0031 to 1.1869 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | atherotrombotic | 18 | 1.0156 | 0.96797 to 1.0529 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | atherotrombotic | 18 | 1.0089 | 1.0003 to 1.0195 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | atherotrombotic | 4 | 1.0284 | 1.0136 to 1.0945 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | whole blood | atherotrombotic | 12 | 1.0326 | 0.96594 to 1.1216 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | atherotrombotic | 1 | 0.9368 | 0.9368 to 0.9368 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | whole blood | atherotrombotic | 1 | 1.0022 | 1.0022 to 1.0022 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | atherotrombotic | 6 | 1.0286 | 0.91032 to 1.1139 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | atherotrombotic | 10 | 0.99125 | 0.90257 to 1.1295 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | whole blood | atherotrombotic | 15 | 0.9887 | 0.91361 to 1.0827 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | whole blood | atherotrombotic | 2 | 1.0259 | 0.9873 to 1.0646 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | atherotrombotic | 9 | 0.9911 | 0.8993 to 1.0631 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | atherotrombotic | 9 | 0.9823 | 0.9232 to 1.0409 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | atherotrombotic | 5 | 0.9879 | 0.90238 to 1.0103 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | atherotrombotic | 10 | 0.9572 | 0.88359 to 1.0462 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | atherotrombotic | 3 | 0.9838 | 0.96823 to 0.98407 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | atherotrombotic | 7 | 0.9962 | 0.93077 to 1.0464 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | atherotrombotic | 18 | 1.0073 | 0.92513 to 1.0513 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | atherotrombotic | 3 | 1.0023 | 0.95784 to 1.0163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | whole blood | atherotrombotic | 4 | 1.0243 | 0.90385 to 1.156 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | whole blood | atherotrombotic | 2 | 0.94175 | 0.91597 to 0.96754 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | whole blood | atherotrombotic | 12 | 0.9919 | 0.93417 to 1.0411 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | atherotrombotic | 2 | 1.0396 | 0.9289 to 1.1503 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | whole blood | cardioembolic | 7 | 1.0771 | 0.94783 to 1.1004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | whole blood | cardioembolic | 12 | 1.0357 | 0.94433 to 1.0927 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | atherotrombotic | 6 | 0.98842 | 0.946 to 1.0153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | atherotrombotic | 11 | 1.0248 | 0.95759 to 1.0476 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | atherotrombotic | 7 | 0.99787 | 0.96146 to 1.0273 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | atherotrombotic | 15 | 1.0278 | 0.97003 to 1.0686 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | atherotrombotic | 10 | 1.0233 | 0.95632 to 1.0682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | atherotrombotic | 10 | 1.0004 | 0.95829 to 1.0336 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | atherotrombotic | 5 | 0.98476 | 0.94505 to 1.0152 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | atherotrombotic | 10 | 1.0026 | 0.9554 to 1.0385 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | atherotrombotic | 16 | 1.0218 | 0.97152 to 1.0599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | atherotrombotic | 16 | 1.035 | 0.97824 to 1.0677 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | atherotrombotic | 7 | 0.9891 | 0.94973 to 1.0299 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | atherotrombotic | 10 | 1.0082 | 0.96279 to 1.0471 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | atherotrombotic | 15 | 1.0228 | 0.97666 to 1.0581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | atherotrombotic | 13 | 1.0405 | 0.97611 to 1.0715 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | atherotrombotic | 5 | 0.9881 | 0.94937 to 1.0163 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | atherotrombotic | 11 | 1.0298 | 0.95723 to 1.0581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | atherotrombotic | 12 | 1.0157 | 0.97105 to 1.0524 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | atherotrombotic | 13 | 1.016 | 0.96279 to 1.0405 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | b cells | whole blood | atherotrombotic | 13 | 1.0233 | 0.92072 to 1.106 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | atherotrombotic | 16 | 1.0514 | 0.93057 to 1.1302 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | whole blood | atherotrombotic | 8 | 0.9748 | 0.92023 to 1.0304 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | whole blood | small-vessel disease | 5 | 1.0081 | 0.89618 to 1.0537 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | whole blood | cardioembolic | 1 | 1.0838 | 1.0838 to 1.0838 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | whole blood | cardioembolic | 1 | 1.1041 | 1.1041 to 1.1041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | small-vessel disease | 6 | 0.97565 | 0.92043 to 1.0486 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | whole blood | cardioembolic | 7 | 1.0269 | 0.95567 to 1.0784 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | whole blood | cardioembolic | 2 | 1.0629 | 1.0596 to 1.0662 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | whole blood | small-vessel disease | 8 | 1.0243 | 0.95644 to 1.0641 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | small-vessel disease | 6 | 0.9498 | 0.9235 to 0.9859 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | atherotrombotic | 7 | 0.9897 | 0.92961 to 1.0591 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | atherotrombotic | 13 | 1.0255 | 0.93366 to 1.1188 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | atherotrombotic | 2 | 0.95965 | 0.94635 to 0.97296 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | atherotrombotic | 2 | 0.96081 | 0.94051 to 0.98112 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | whole blood | atherotrombotic | 3 | 1.0017 | 0.95148 to 1.0391 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon fibroblasts | whole blood | atherotrombotic | 1 | 0.9038 | 0.9038 to 0.9038 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | atherotrombotic | 3 | 0.9466 | 0.87919 to 0.98458 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | whole blood | atherotrombotic | 9 | 0.9897 | 0.95172 to 1.0623 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | atherotrombotic | 5 | 1.0273 | 1.0149 to 1.1212 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | whole blood | atherotrombotic | 3 | 1.1247 | 0.97827 to 1.1415 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | whole blood | atherotrombotic | 2 | 0.9965 | 0.99551 to 0.99749 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | whole blood | atherotrombotic | 2 | 1.0187 | 0.99719 to 1.0402 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | whole blood | atherotrombotic | 9 | 0.9651 | 0.91844 to 1.0284 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | whole blood | small-vessel disease | 8 | 0.8992 | 0.86559 to 1.0209 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | atherotrombotic | 2 | 0.9834 | 0.95406 to 1.0127 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | whole blood | atherotrombotic | 1 | 1.006 | 1.006 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | whole blood | atherotrombotic | 1 | 0.9995 | 0.9995 to 0.9995 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | whole blood | atherotrombotic | 1 | 0.9041 | 0.9041 to 0.9041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | atherotrombotic | 7 | 0.9969 | 0.93542 to 1.083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | whole blood | atherotrombotic | 2 | 1.0711 | 1.0281 to 1.114 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric fundus epithelium | whole blood | cardioembolic | 1 | 0.9885 | 0.9885 to 0.9885 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | whole blood | cardioembolic | 1 | 1.0507 | 1.0507 to 1.0507 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | cardioembolic | 3 | 0.99081 | 0.94476 to 0.99218 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | lung alveolar endothelium | whole blood | cardioembolic | 1 | 0.9063 | 0.9063 to 0.9063 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | whole blood | atherotrombotic | 2 | 0.9484 | 0.91834 to 0.97846 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | atherotrombotic | 1 | 0.9478 | 0.9478 to 0.9478 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | small-vessel disease | 5 | 1.0716 | 0.8489 to 1.1002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | small-vessel disease | 1 | 1.0422 | 1.0422 to 1.0422 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | whole blood | cardioembolic | 2 | 0.9589 | 0.9562 to 0.9616 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | whole blood | cardioembolic | 2 | 0.9872 | 0.91286 to 1.0615 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | whole blood | small-vessel disease | 3 | 1.065 | 1.0245 to 1.1101 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | whole blood | small-vessel disease | 2 | 1.0461 | 0.99877 to 1.0935 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | atherotrombotic | 1 | 0.98954 | 0.98954 to 0.98954 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | atherotrombotic | 1 | 0.98351 | 0.98351 to 0.98351 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | atherotrombotic | 1 | 0.98563 | 0.98563 to 0.98563 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | atherotrombotic | 1 | 0.98336 | 0.98336 to 0.98336 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | atherotrombotic | 2 | 0.91095 | 0.8497 to 0.97219 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | atherotrombotic | 3 | 1.0111 | 0.99544 to 1.039 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | atherotrombotic | 2 | 1.0875 | 1.0847 to 1.0902 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | atherotrombotic | 3 | 0.9434 | 0.91397 to 0.98543 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | whole blood | atherotrombotic | 1 | 1.029 | 1.029 to 1.029 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | whole blood | atherotrombotic | 1 | 0.975 | 0.975 to 0.975 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | whole blood | atherotrombotic | 1 | 0.8972 | 0.8972 to 0.8972 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | atherotrombotic | 2 | 1.0217 | 1.0022 to 1.0413 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | whole blood | small-vessel disease | 1 | 0.9272 | 0.9272 to 0.9272 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon fibroblasts | whole blood | cardioembolic | 2 | 0.96465 | 0.88279 to 1.0465 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | whole blood | atherotrombotic | 1 | 1.0401 | 1.0401 to 1.0401 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | atherotrombotic | 1 | 0.9518 | 0.9518 to 0.9518 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | whole blood | small-vessel disease | 1 | 1.065 | 1.065 to 1.065 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic delta | whole blood | small-vessel disease | 1 | 0.9195 | 0.9195 to 0.9195 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric body epithelium | whole blood | small-vessel disease | 1 | 0.9418 | 0.9418 to 0.9418 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney glomerular endothelium | whole blood | small-vessel disease | 1 | 1.0066 | 1.0066 to 1.0066 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | Normal | 117 | 1.0355 | 0.96179 to 1.1112 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | Normal | 117 | 1.0839 | 1.0054 to 1.1683 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | Normal | 117 | 1.3693 | 1.218 to 1.5632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | Normal | 117 | 1.0045 | 0.94078 to 1.1077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | Normal | 117 | 1.0076 | 0.9437 to 1.0868 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | Normal | 117 | 0.93447 | 0.88889 to 1.0056 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | Normal | 117 | 0.96399 | 0.91192 to 1.0417 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | Normal | 117 | 1.0729 | 0.99104 to 1.1538 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | Normal | 117 | 0.92501 | 0.87436 to 0.99946 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | Normal | 114 | 0.94421 | 0.89997 to 1.011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | Normal | 115 | 0.92293 | 0.8829 to 0.98976 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | Normal | 116 | 1.0715 | 1.0014 to 1.1428 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | tissue | Normal | 117 | 1.0639 | 0.98013 to 1.1501 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | tissue | Normal | 117 | 0.97665 | 0.90094 to 1.0898 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | Normal | 117 | 1.0937 | 1.0015 to 1.168 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | Normal | 117 | 0.98968 | 0.92652 to 1.0622 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | tissue | Normal | 116 | 0.98876 | 0.93945 to 1.0599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | tissue | Normal | 115 | 0.94848 | 0.90354 to 1.0191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | Normal | 117 | 0.99923 | 0.94126 to 1.0679 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | tissue | Normal | 117 | 0.96813 | 0.927 to 1.0311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | Normal | 117 | 1.1067 | 1.0235 to 1.1893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | Normal | 116 | 1.0921 | 1.0158 to 1.1833 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | Normal | 117 | 1.1257 | 1.0354 to 1.2131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | Normal | 117 | 1.1284 | 1.0293 to 1.2088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | Normal | 117 | 1.0912 | 0.9977 to 1.1691 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | Normal | 114 | 1.0592 | 0.9795 to 1.1284 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | tissue | Normal | 117 | 1.0077 | 0.93698 to 1.0875 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | Normal | 116 | 1.1169 | 1.0135 to 1.2043 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | Normal | 117 | 1.0928 | 1.0125 to 1.1797 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | Normal | 117 | 1.0214 | 0.95458 to 1.1034 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | Normal | 114 | 1.0678 | 0.98127 to 1.1686 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | Normal | 114 | 1.086 | 0.99986 to 1.1674 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | Normal | 117 | 1.1726 | 1.0989 to 1.2692 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | Normal | 114 | 1.1618 | 1.0604 to 1.2582 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | tissue | Normal | 116 | 1.0029 | 0.94465 to 1.0656 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | Normal | 117 | 0.94154 | 0.89599 to 1.0133 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | Normal | 117 | 1.0102 | 0.96212 to 1.0773 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | tissue | Normal | 114 | 0.98191 | 0.93633 to 1.0498 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | tissue | Normal | 117 | 0.99502 | 0.94841 to 1.0494 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | tissue | Normal | 116 | 0.99843 | 0.95052 to 1.0554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | tissue | Normal | 117 | 0.99245 | 0.94757 to 1.0523 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | Normal | 117 | 1.0271 | 0.96935 to 1.0934 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | Normal | 114 | 0.95203 | 0.91228 to 1.0177 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | Normal | 117 | 1.0164 | 0.95425 to 1.0911 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | tissue | Normal | 117 | 1.0168 | 0.94128 to 1.1027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | Normal | 117 | 1.0093 | 0.94803 to 1.0795 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | Normal | 117 | 0.99915 | 0.9436 to 1.0674 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | Normal | 117 | 1.0274 | 0.9636 to 1.104 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | Normal | 116 | 1.0309 | 0.96037 to 1.092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | tissue | Normal | 117 | 0.98605 | 0.92502 to 1.0753 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | Normal | 103 | 1.1768 | 1.0827 to 1.3027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | Normal | 117 | 0.98059 | 0.93198 to 1.0533 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | Normal | 115 | 1.0991 | 1.0161 to 1.1886 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | Normal | 115 | 0.96241 | 0.90975 to 1.0341 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | Normal | 114 | 1.1497 | 1.0471 to 1.2715 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | Normal | 114 | 1.0218 | 0.9546 to 1.093 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | Normal | 117 | 1.0837 | 1.012 to 1.1642 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | Normal | 107 | 1.1196 | 1.0384 to 1.2298 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | Normal | 112 | 1.1359 | 1.0606 to 1.2391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | Normal | 100 | 1.0934 | 1.0284 to 1.207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | Normal | 117 | 1.1077 | 1.0107 to 1.1823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | tissue | Normal | 117 | 0.99683 | 0.91504 to 1.103 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | Normal | 112 | 1.0722 | 0.98251 to 1.1733 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | tissue | Normal | 116 | 1.0189 | 0.95867 to 1.0903 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | Normal | 117 | 0.96231 | 0.91823 to 1.0254 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | tissue | Normal | 117 | 0.99615 | 0.94338 to 1.0654 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | Normal | 116 | 1.0698 | 1.0006 to 1.1415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | Normal | 116 | 1.0045 | 0.94572 to 1.0706 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | Normal | 114 | 0.94985 | 0.91179 to 1.0086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | tissue | Normal | 116 | 0.96698 | 0.92039 to 1.0283 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | Normal | 117 | 0.89749 | 0.8483 to 0.97724 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | Normal | 116 | 1.1458 | 1.0415 to 1.2272 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | tissue | Normal | 117 | 1.011 | 0.96119 to 1.0749 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | tissue | Normal | 117 | 1.0631 | 0.97705 to 1.1412 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | Normal | 102 | 1.0691 | 0.99996 to 1.1682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | Normal | 103 | 1.0296 | 0.96037 to 1.1049 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | Normal | 116 | 1.0055 | 0.95219 to 1.0821 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | Normal | 116 | 0.98544 | 0.93185 to 1.0536 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | tissue | Normal | 117 | 0.98277 | 0.94282 to 1.0443 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | Normal | 117 | 1.0269 | 0.96977 to 1.0872 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | tissue | Normal | 117 | 1.0658 | 0.98585 to 1.1479 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | Normal | 117 | 0.9499 | 0.89444 to 1.0183 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | tissue | Normal | 117 | 1.0126 | 0.93104 to 1.1222 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | Normal | 117 | 1.0352 | 0.97287 to 1.1077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | Normal | 117 | 1.3587 | 1.2096 to 1.5491 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | Normal | 117 | 0.9839 | 0.92263 to 1.0815 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | Normal | 117 | 0.9997 | 0.99254 to 1.0079 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | Normal | 89 | 1.0191 | 0.93912 to 1.104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | Normal | 110 | 1.018 | 0.92763 to 1.1079 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | Normal | 117 | 0.9691 | 0.89134 to 1.0779 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | Normal | 117 | 1.0217 | 0.92524 to 1.1439 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | Normal | 36 | 0.96925 | 0.8706 to 1.0692 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | tissue | Normal | 102 | 0.9981 | 0.90719 to 1.0901 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | tissue | Normal | 67 | 1.0316 | 0.93996 to 1.1577 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | Normal | 117 | 1.0373 | 0.9589 to 1.1208 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | Normal | 88 | 1.0056 | 0.91148 to 1.0735 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | Normal | 117 | 1.0085 | 0.9236 to 1.1101 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | tissue | Normal | 114 | 1.012 | 0.92861 to 1.1013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | Normal | 117 | 1.0112 | 0.91676 to 1.0783 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | Tumor | 285 | 1.0402 | 0.96969 to 1.1245 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | Tumor | 285 | 1.0378 | 0.9592 to 1.1378 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | Tumor | 285 | 1.1772 | 1.0098 to 1.5739 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | Tumor | 285 | 1.0108 | 0.93278 to 1.1423 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | Tumor | 285 | 1.0374 | 0.95151 to 1.1013 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | Tumor | 281 | 0.97692 | 0.87355 to 1.0333 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | Tumor | 282 | 0.99134 | 0.90804 to 1.0486 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | Tumor | 282 | 1.0414 | 0.97063 to 1.1397 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | Tumor | 280 | 0.96549 | 0.853 to 1.0202 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | Tumor | 274 | 0.96212 | 0.87535 to 1.0114 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | Tumor | 277 | 0.96309 | 0.86797 to 1.0285 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | Tumor | 278 | 1.0347 | 0.96861 to 1.1266 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | tissue | Tumor | 115 | 1.04 | 0.96767 to 1.1437 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | tissue | Tumor | 118 | 0.9637 | 0.87968 to 1.0892 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | Tumor | 282 | 1.0386 | 0.94639 to 1.156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | Tumor | 264 | 0.99053 | 0.92067 to 1.0537 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | tissue | Tumor | 111 | 0.98161 | 0.9289 to 1.0588 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | tissue | Tumor | 105 | 0.9511 | 0.89367 to 1.0248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | Tumor | 221 | 1.0021 | 0.93912 to 1.0677 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | tissue | Tumor | 115 | 0.97094 | 0.91733 to 1.0415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | Tumor | 278 | 1.0466 | 0.94683 to 1.1795 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | Tumor | 277 | 1.0524 | 0.97471 to 1.1599 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | Tumor | 275 | 1.0446 | 0.95028 to 1.1987 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | Tumor | 282 | 1.0587 | 0.95971 to 1.1982 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | Tumor | 272 | 1.0283 | 0.94727 to 1.151 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | Tumor | 273 | 1.0152 | 0.9415 to 1.1206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | tissue | Tumor | 115 | 0.99659 | 0.92956 to 1.0838 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | Tumor | 273 | 1.0192 | 0.90652 to 1.1812 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | Tumor | 267 | 1.0855 | 1.0049 to 1.1918 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | Tumor | 174 | 1.0306 | 0.958 to 1.1195 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | Tumor | 272 | 1.0129 | 0.9398 to 1.14 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | Tumor | 278 | 1.0593 | 0.96017 to 1.2459 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | tissue | Tumor | 114 | 0.99071 | 0.93538 to 1.0668 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | Tumor | 272 | 0.9802 | 0.88262 to 1.0291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | Tumor | 274 | 1.0459 | 0.97962 to 1.0998 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | tissue | Tumor | 105 | 0.977 | 0.92376 to 1.0519 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | tissue | Tumor | 111 | 0.98194 | 0.93589 to 1.0472 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | tissue | Tumor | 111 | 0.98561 | 0.93153 to 1.0525 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | tissue | Tumor | 112 | 0.98073 | 0.92777 to 1.0489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | Tumor | 131 | 1.0197 | 0.97113 to 1.0953 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | Tumor | 275 | 0.97831 | 0.89851 to 1.0263 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | Tumor | 270 | 1.0388 | 0.97756 to 1.1037 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | tissue | Tumor | 115 | 1.0063 | 0.93577 to 1.1093 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | Tumor | 263 | 1.0265 | 0.96637 to 1.0915 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | Tumor | 271 | 1.0285 | 0.96134 to 1.0894 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | Tumor | 271 | 1.0412 | 0.97647 to 1.1102 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | Tumor | 272 | 0.99683 | 0.93122 to 1.0756 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | tissue | Tumor | 115 | 0.98225 | 0.91074 to 1.0676 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | Tumor | 255 | 1.0574 | 0.9646 to 1.2607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | Tumor | 261 | 1.0099 | 0.93483 to 1.0739 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | Tumor | 264 | 0.98383 | 0.90473 to 1.0375 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | Tumor | 270 | 1.0431 | 0.94268 to 1.2109 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | Tumor | 270 | 0.98697 | 0.92089 to 1.0646 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | Tumor | 273 | 1.0302 | 0.95813 to 1.13 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | Tumor | 270 | 1.0336 | 0.93147 to 1.2056 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | Tumor | 278 | 1.0401 | 0.94715 to 1.1704 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | tissue | Tumor | 118 | 0.97921 | 0.90104 to 1.0963 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | Tumor | 263 | 1.0238 | 0.94533 to 1.1294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | tissue | Tumor | 111 | 1.0108 | 0.96023 to 1.0978 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | Tumor | 277 | 0.98111 | 0.90916 to 1.0288 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | tissue | Tumor | 115 | 0.98858 | 0.92947 to 1.0707 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | Tumor | 278 | 1.0314 | 0.95362 to 1.1356 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | Tumor | 245 | 1.023 | 0.96163 to 1.088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | Tumor | 274 | 0.95488 | 0.88732 to 1.0087 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | tissue | Tumor | 111 | 0.96203 | 0.91062 to 1.028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | Tumor | 217 | 0.95247 | 0.82709 to 1.0247 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | Tumor | 278 | 1.0521 | 0.93729 to 1.2061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | tissue | Tumor | 112 | 1.0018 | 0.95992 to 1.0741 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | tissue | Tumor | 115 | 1.0527 | 0.98702 to 1.1538 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | Tumor | 191 | 1.0155 | 0.95454 to 1.0942 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | Tumor | 218 | 1.0103 | 0.93751 to 1.0744 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | tissue | Tumor | 112 | 0.97653 | 0.92662 to 1.0464 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | Tumor | 221 | 1.0263 | 0.97411 to 1.0889 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | tissue | Tumor | 116 | 1.0581 | 0.98926 to 1.1552 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | Tumor | 281 | 1.0047 | 0.88672 to 1.0847 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | tissue | Tumor | 118 | 0.9969 | 0.91379 to 1.1176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | Tumor | 267 | 1.042 | 0.97581 to 1.1055 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | Tumor | 285 | 1.1743 | 1.0096 to 1.5612 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | Tumor | 285 | 1.002 | 0.91686 to 1.1227 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | Tumor | 285 | 0.9984 | 0.98814 to 1.0104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | Tumor | 230 | 1.0076 | 0.91641 to 1.1107 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | Tumor | 258 | 0.9989 | 0.91219 to 1.1087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | Tumor | 186 | 0.9987 | 0.90415 to 1.0832 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | Tumor | 252 | 1.0076 | 0.92466 to 1.1073 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | Tumor | 253 | 0.9918 | 0.90596 to 1.0944 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | Tumor | 206 | 1.0203 | 0.93443 to 1.1186 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | Tumor | 274 | 1.0252 | 0.9251 to 1.1191 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | Tumor | 236 | 1.0088 | 0.91997 to 1.1033 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | Tumor | 276 | 0.98525 | 0.87243 to 1.0892 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | Tumor | 281 | 1.0343 | 0.9279 to 1.1305 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | Normal | 53 | 1.0302 | 0.92096 to 1.1098 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | Tumor | 271 | 1.0238 | 0.9379 to 1.1514 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | Tumor | 269 | 1.0449 | 0.93837 to 1.2213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | Tumor | 265 | 1.0247 | 0.94211 to 1.1474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | Tumor | 261 | 1.0088 | 0.91308 to 1.1757 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | Tumor | 253 | 0.97806 | 0.88728 to 1.1464 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | Tumor | 252 | 1.0319 | 0.94793 to 1.1391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | Tumor | 263 | 0.98319 | 0.91336 to 1.0773 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | Tumor | 135 | 1.0103 | 0.92531 to 1.1278 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | tissue | Normal | 40 | 0.98795 | 0.92247 to 1.0586 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | Tumor | 170 | 1.0272 | 0.94642 to 1.1091 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | tissue | Tumor | 35 | 0.9887 | 0.91912 to 1.1043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | tissue | Tumor | 16 | 0.9844 | 0.92292 to 1.091 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | tissue | Tumor | 70 | 1.0035 | 0.91536 to 1.1238 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | tissue | Normal | 34 | 0.95765 | 0.88931 to 1.0155 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | Normal | 32 | 1.0041 | 0.93433 to 1.0966 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | Tumor | 81 | 0.9678 | 0.895 to 1.0586 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | Normal | 52 | 0.9729 | 0.88421 to 1.0867 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | Normal | 38 | 0.9041 | 0.84293 to 1.0137 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | tissue | Normal | 17 | 0.9636 | 0.8968 to 1.0914 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | Tumor | 113 | 0.9351 | 0.82872 to 1.0331 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | Tumor | 80 | 1.0129 | 0.92972 to 1.1113 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | tissue | Tumor | 2 | 0.97055 | 0.90381 to 1.0373 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | tissue | Tumor | 3 | 0.9492 | 0.92508 to 0.98421 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | Normal | 13 | 1.0283 | 0.96466 to 1.1362 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | tissue | Tumor | 7 | 0.9655 | 0.94157 to 0.99047 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | tissue | Tumor | 17 | 0.9848 | 0.88094 to 1.0713 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | tissue | Normal | 3 | 0.9886 | 0.97717 to 0.98905 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | Normal | 3 | 1.1084 | 1.0053 to 1.2514 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | Normal | 5 | 1.0076 | 0.92522 to 1.1382 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | Tumor | 72 | 0.91795 | 0.78036 to 1.0818 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | Normal | 2 | 0.9689 | 0.92588 to 1.0119 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | Normal | 1 | 0.9654 | 0.9654 to 0.9654 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | Normal | 1 | 0.9676 | 0.9676 to 0.9676 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | Normal | 2 | 0.99175 | 0.95237 to 1.0311 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | Normal | 1 | 0.9772 | 0.9772 to 0.9772 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | tissue | Normal | 1 | 0.9165 | 0.9165 to 0.9165 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | Tumor | 116 | 1.0137 | 0.89765 to 1.1116 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | Tumor | 8 | 0.96585 | 0.92139 to 1.0184 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | Tumor | 6 | 0.98155 | 0.96012 to 1.0332 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | Tumor | 160 | 0.99215 | 0.89954 to 1.0595 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | tissue | Tumor | 6 | 1.0367 | 0.9273 to 1.1111 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | tissue | Tumor | 1 | 1.0452 | 1.0452 to 1.0452 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | Tumor | 36 | 1.016 | 0.94027 to 1.1029 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | Tumor | 147 | 0.977 | 0.89455 to 1.0745 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | tissue | Tumor | 2 | 0.979 | 0.89845 to 1.0595 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | Tumor | 82 | 0.98175 | 0.88597 to 1.0974 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | tissue | Tumor | 94 | 1.0379 | 0.91749 to 1.1326 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | Tumor | 87 | 1.037 | 0.94767 to 1.1013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | Tumor | 74 | 0.97475 | 0.87292 to 1.073 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | bladder epithelium | tissue | Tumor | 89 | 0.989 | 0.87516 to 1.1137 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | tissue | Tumor | 62 | 1.0866 | 0.98942 to 1.2199 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | tissue | Tumor | 146 | 1.0215 | 0.86423 to 1.1728 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | tissue | Tumor | 23 | 0.9994 | 0.90171 to 1.1401 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | Tumor | 146 | 1.0088 | 0.939 to 1.0863 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | tissue | Tumor | 75 | 1.0002 | 0.89952 to 1.13 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | tissue | Tumor | 4 | 1.0314 | 1.0068 to 1.1251 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric fundus epithelium | tissue | Tumor | 18 | 1.0547 | 0.96674 to 1.168 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | tissue | Tumor | 7 | 0.9721 | 0.91421 to 1.0279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | Tumor | 47 | 0.967 | 0.89347 to 1.0181 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | GBM | 71 | 1.012 | 0.92645 to 1.0688 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | GBM | 71 | 0.9921 | 0.91606 to 1.0568 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | GBM | 71 | 1.0731 | 0.96884 to 1.1849 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | GBM | 71 | 1.0368 | 0.90644 to 1.1533 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | GBM | 71 | 0.98284 | 0.88335 to 1.0382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | GBM | 71 | 0.92934 | 0.81993 to 0.98873 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | GBM | 71 | 0.96339 | 0.85989 to 1.0128 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | GBM | 71 | 1.0001 | 0.91873 to 1.0566 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | GBM | 71 | 0.91912 | 0.80396 to 0.98154 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | GBM | 71 | 0.923 | 0.80083 to 0.97607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | GBM | 71 | 0.90973 | 0.80133 to 0.97498 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | GBM | 71 | 0.97062 | 0.90242 to 1.0297 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | GBM | 71 | 0.99753 | 0.92436 to 1.0663 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | GBM | 69 | 0.95111 | 0.87775 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | GBM | 71 | 0.9928 | 0.91631 to 1.0605 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | GBM | 71 | 0.98478 | 0.91688 to 1.0455 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | GBM | 71 | 0.98407 | 0.91982 to 1.0528 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | GBM | 71 | 1.0129 | 0.93365 to 1.0805 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | GBM | 71 | 0.97368 | 0.90549 to 1.0341 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | GBM | 71 | 0.93619 | 0.86351 to 1.0127 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | GBM | 71 | 0.95885 | 0.88657 to 1.0327 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | GBM | 68 | 1.0766 | 0.99634 to 1.1381 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | GBM | 71 | 0.95253 | 0.88028 to 1.012 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | GBM | 71 | 0.93094 | 0.85399 to 1.0195 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | GBM | 71 | 0.99774 | 0.93999 to 1.0687 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | GBM | 71 | 0.94676 | 0.87173 to 1.0236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | GBM | 71 | 0.9434 | 0.86104 to 0.99175 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | GBM | 71 | 0.98532 | 0.90075 to 1.0368 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | GBM | 71 | 0.90775 | 0.81462 to 0.96735 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | GBM | 69 | 1.0014 | 0.95721 to 1.0556 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | GBM | 68 | 0.99661 | 0.93989 to 1.0432 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | GBM | 69 | 1.0071 | 0.94488 to 1.0555 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | GBM | 69 | 0.99761 | 0.93827 to 1.0431 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | GBM | 71 | 0.94924 | 0.88502 to 1.0074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | GBM | 71 | 1.0037 | 0.9365 to 1.0713 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | GBM | 67 | 0.96809 | 0.90659 to 1.0183 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | GBM | 70 | 0.96118 | 0.8991 to 1.0312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | GBM | 69 | 0.92825 | 0.86003 to 0.98298 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | GBM | 71 | 0.98258 | 0.91264 to 1.0583 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | GBM | 71 | 0.95607 | 0.88389 to 1.0044 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | GBM | 71 | 0.96657 | 0.89969 to 1.0305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | GBM | 71 | 0.94488 | 0.86422 to 1.0193 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | GBM | 71 | 0.95158 | 0.88135 to 1.0379 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | GBM | 71 | 0.91308 | 0.83927 to 0.98657 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | GBM | 71 | 0.98914 | 0.89597 to 1.0601 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | GBM | 71 | 0.96759 | 0.88265 to 1.0422 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | GBM | 71 | 0.91943 | 0.79865 to 0.98578 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | GBM | 71 | 0.97311 | 0.8985 to 1.0282 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | GBM | 40 | 0.98959 | 0.89408 to 1.0357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | GBM | 71 | 0.90734 | 0.782 to 0.9824 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | GBM | 13 | 0.84371 | 0.73352 to 0.92919 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | GBM | 71 | 0.99609 | 0.92517 to 1.0722 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | GBM | 71 | 0.95048 | 0.88697 to 1.0294 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | GBM | 71 | 0.90763 | 0.81411 to 0.99577 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | GBM | 14 | 0.97497 | 0.86415 to 1.0521 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | GBM | 15 | 0.98094 | 0.86399 to 1.0441 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | GBM | 13 | 0.99447 | 0.88294 to 1.0507 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | GBM | 71 | 0.9411 | 0.86788 to 0.99867 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | GBM | 69 | 1.0044 | 0.95255 to 1.0497 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | GBM | 71 | 1.0715 | 0.96697 to 1.183 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | GBM | 71 | 1.0162 | 0.89824 to 1.1295 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | GBM | 71 | 1.0077 | 0.9961 to 1.0171 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | GBM | 30 | 1.0122 | 0.93303 to 1.1225 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | GBM | 22 | 1.013 | 0.92509 to 1.1083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | GBM | 47 | 0.9973 | 0.93265 to 1.076 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | GBM | 63 | 0.9921 | 0.88603 to 1.1454 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | tissue | GBM | 65 | 0.9715 | 0.7648 to 1.1029 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | GBM | 71 | 0.9781 | 0.8877 to 1.0703 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | GBM | 61 | 1.0041 | 0.866 to 1.1234 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | GBM | 57 | 0.9653 | 0.87262 to 1.0529 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | GBM | 62 | 1.0057 | 0.93565 to 1.0852 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | astrocytes | tissue | GBM | 52 | 0.96215 | 0.85916 to 1.1954 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | GBM | 49 | 1.0253 | 0.9305 to 1.1229 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | GBM | 41 | 1.0169 | 0.9475 to 1.1388 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | tissue | GBM | 26 | 1.0041 | 0.87077 to 1.1423 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | GBM | 54 | 0.9805 | 0.91114 to 1.0415 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | GBM | 30 | 1.0296 | 0.87331 to 1.156 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | GBM | 13 | 1.0546 | 0.92414 to 1.1282 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | GBM | 50 | 0.98475 | 0.86843 to 1.1186 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | tissue | GBM | 59 | 0.9586 | 0.8322 to 1.0779 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | GBM | 59 | 0.984 | 0.88094 to 1.0994 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | GBM | 30 | 0.9876 | 0.92434 to 1.0613 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | GBM | 26 | 0.9937 | 0.88867 to 1.1737 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | tissue | GBM | 2 | 0.95155 | 0.94925 to 0.95384 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | pancreatic cancer | 10 | 1.0701 | 0.97807 to 1.1014 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | pancreatic cancer | 10 | 1.0497 | 0.96322 to 1.1103 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | pancreatic cancer | 10 | 1.18 | 0.96307 to 1.2711 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | pancreatic cancer | 10 | 1.0537 | 0.97289 to 1.108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | pancreatic cancer | 10 | 1.0637 | 0.98913 to 1.0772 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | pancreatic cancer | 10 | 0.98612 | 0.92773 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | pancreatic cancer | 10 | 1.0053 | 0.93572 to 1.0313 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | pancreatic cancer | 10 | 1.0501 | 0.95682 to 1.0983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | pancreatic cancer | 10 | 0.98216 | 0.91779 to 0.99526 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | pancreatic cancer | 10 | 0.98059 | 0.93371 to 1.0123 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | pancreatic cancer | 10 | 0.97139 | 0.90367 to 0.99594 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | pancreatic cancer | 10 | 1.0448 | 0.96179 to 1.0676 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | pancreatic cancer | 10 | 1.0386 | 0.91808 to 1.1024 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | pancreatic cancer | 6 | 1.0434 | 1.0183 to 1.0662 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | pancreatic cancer | 10 | 1.0493 | 0.92338 to 1.1362 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | pancreatic cancer | 10 | 1.0767 | 0.97245 to 1.1361 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | pancreatic cancer | 10 | 1.0655 | 0.9542 to 1.1458 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | pancreatic cancer | 10 | 1.0635 | 0.94159 to 1.1462 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | pancreatic cancer | 9 | 1.0587 | 0.97707 to 1.1347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | pancreatic cancer | 10 | 1.0467 | 0.97222 to 1.1157 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | pancreatic cancer | 10 | 1.0324 | 0.96117 to 1.1482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | pancreatic cancer | 5 | 1.1238 | 1.083 to 1.1373 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | pancreatic cancer | 10 | 1.0294 | 0.91444 to 1.1001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | pancreatic cancer | 10 | 1.0646 | 0.9919 to 1.1439 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | pancreatic cancer | 10 | 1.086 | 0.95744 to 1.1622 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | pancreatic cancer | 10 | 1.0685 | 0.98832 to 1.2089 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | pancreatic cancer | 8 | 1.0113 | 0.96313 to 1.0274 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | pancreatic cancer | 8 | 1.0655 | 0.99974 to 1.0829 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | pancreatic cancer | 10 | 0.99148 | 0.93749 to 1.0075 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | pancreatic cancer | 7 | 1.061 | 1.0086 to 1.0828 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | pancreatic cancer | 5 | 1.0605 | 1.0366 to 1.0729 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | pancreatic cancer | 8 | 1.0429 | 0.96223 to 1.0676 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | pancreatic cancer | 7 | 1.0614 | 1.0128 to 1.0777 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | pancreatic cancer | 9 | 0.99862 | 0.93342 to 1.0358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | pancreatic cancer | 10 | 1.0918 | 0.99079 to 1.1533 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | pancreatic cancer | 5 | 1.0387 | 1.023 to 1.0466 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | pancreatic cancer | 8 | 1.051 | 0.97458 to 1.1062 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | pancreatic cancer | 7 | 1.0261 | 0.99252 to 1.0426 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | pancreatic cancer | 10 | 1.0755 | 0.95862 to 1.1927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | pancreatic cancer | 10 | 1.0109 | 0.89416 to 1.0546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | pancreatic cancer | 9 | 1.0485 | 0.96271 to 1.1113 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | pancreatic cancer | 10 | 1.0297 | 0.93828 to 1.1495 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | pancreatic cancer | 10 | 1.0386 | 0.93988 to 1.1461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | pancreatic cancer | 10 | 0.99548 | 0.90983 to 1.1249 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | pancreatic cancer | 10 | 1.0196 | 0.91369 to 1.088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | pancreatic cancer | 9 | 1.0402 | 0.95129 to 1.1105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | pancreatic cancer | 10 | 1.0143 | 0.96539 to 1.0327 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | pancreatic cancer | 10 | 1.0317 | 0.9434 to 1.0904 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | pancreatic cancer | 2 | 1.0682 | 1.0585 to 1.0778 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | pancreatic cancer | 10 | 0.96894 | 0.91769 to 0.99719 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | pancreatic cancer | 10 | 1.0645 | 0.97517 to 1.1626 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | pancreatic cancer | 9 | 1.062 | 0.97577 to 1.1354 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | pancreatic cancer | 10 | 1.0248 | 0.95404 to 1.0776 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | pancreatic cancer | 10 | 1.0054 | 0.91235 to 1.049 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | pancreatic cancer | 6 | 1.0643 | 1.0419 to 1.0813 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | pancreatic cancer | 10 | 1.1772 | 0.96372 to 1.2665 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | pancreatic cancer | 10 | 1.0424 | 0.97036 to 1.0926 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | pancreatic cancer | 10 | 1.0002 | 0.99367 to 1.007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | pancreatic cancer | 6 | 0.97735 | 0.96215 to 1.0586 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | pancreatic cancer | 5 | 1.0174 | 0.8911 to 1.0558 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | pancreatic cancer | 7 | 0.9702 | 0.90484 to 1.0575 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | pancreatic cancer | 7 | 0.9412 | 0.88773 to 0.98913 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | pancreatic cancer | 10 | 0.97875 | 0.9323 to 1.0803 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | pancreatic cancer | 7 | 0.9558 | 0.91514 to 1.0206 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | pancreatic cancer | 10 | 0.99785 | 0.9127 to 1.0476 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | pancreatic cancer | 10 | 0.99065 | 0.89867 to 1.0312 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | pancreatic cancer | 6 | 0.8636 | 0.84588 to 0.97747 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | pancreatic cancer | 7 | 0.9852 | 0.93221 to 1.1131 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | pancreatic cancer | 9 | 1.0069 | 0.92728 to 1.0288 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | pancreatic cancer | 10 | 0.9399 | 0.84672 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | pancreatic cancer | 6 | 1.0987 | 0.9347 to 1.1384 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | pancreatic cancer | 4 | 0.9658 | 0.89458 to 0.99341 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | pancreatic cancer | 7 | 1.006 | 0.94764 to 1.05 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | tissue | pancreatic cancer | 7 | 1.0894 | 0.87424 to 1.1256 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | pancreatic cancer | 10 | 1.0134 | 0.87943 to 1.0912 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | pancreatic cancer | 9 | 1.0382 | 0.98308 to 1.1137 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | pancreatic cancer | 9 | 0.9647 | 0.8825 to 1.0176 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | pancreatic cancer | 5 | 1.0604 | 0.98102 to 1.161 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | pancreatic cancer | 1 | 0.8764 | 0.8764 to 0.8764 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | pancreatic cancer | 10 | 1.0895 | 0.97782 to 1.1562 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric body epithelium | tissue | Tumor | 5 | 1.0634 | 0.98104 to 1.0909 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | Adjacent | 29 | 1.0353 | 0.96472 to 1.0791 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | Adjacent | 29 | 0.98712 | 0.92346 to 1.0467 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | Adjacent | 29 | 1.075 | 0.93538 to 1.1929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | Adjacent | 29 | 1.0272 | 0.95613 to 1.0911 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | Adjacent | 29 | 1.0256 | 0.98381 to 1.068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | Adjacent | 29 | 0.97476 | 0.9436 to 1.0252 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | Adjacent | 29 | 0.98132 | 0.9517 to 1.0214 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | Adjacent | 29 | 1.0177 | 0.95255 to 1.0732 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | Adjacent | 29 | 0.96383 | 0.93802 to 1.0139 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | Adjacent | 29 | 0.95497 | 0.93231 to 0.9882 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | Adjacent | 29 | 0.96151 | 0.92116 to 1.0038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | Adjacent | 29 | 0.98646 | 0.94123 to 1.0273 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | Adjacent | 29 | 1.0019 | 0.91809 to 1.0701 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | Adjacent | 26 | 0.97874 | 0.93073 to 1.0195 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | Adjacent | 19 | 0.99612 | 0.94484 to 1.0351 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | Adjacent | 29 | 1.0033 | 0.91943 to 1.0658 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | Adjacent | 29 | 1.013 | 0.93611 to 1.0772 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | Adjacent | 29 | 0.998 | 0.9081 to 1.0776 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | Adjacent | 29 | 1.0144 | 0.92335 to 1.0898 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | Adjacent | 29 | 0.98348 | 0.90689 to 1.094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | Adjacent | 29 | 0.98078 | 0.91204 to 1.0681 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | Adjacent | 29 | 0.95904 | 0.8924 to 1.0802 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | Adjacent | 26 | 1.0457 | 0.9539 to 1.117 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | Adjacent | 9 | 1.0087 | 0.98544 to 1.0602 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | Adjacent | 29 | 0.97698 | 0.91141 to 1.0269 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | Adjacent | 29 | 0.97885 | 0.90961 to 1.1002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | Adjacent | 29 | 1.0173 | 0.91793 to 1.0969 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | Adjacent | 29 | 0.97911 | 0.9014 to 1.1122 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | Adjacent | 28 | 0.98223 | 0.94958 to 1.0064 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | Adjacent | 28 | 1.0313 | 0.98968 to 1.0682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | Adjacent | 29 | 0.95852 | 0.92953 to 1.0053 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | Adjacent | 27 | 1.0168 | 0.96542 to 1.0451 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | Adjacent | 25 | 1.0048 | 0.95576 to 1.0375 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | Adjacent | 28 | 1.0093 | 0.96392 to 1.0388 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | Adjacent | 28 | 1.0101 | 0.95829 to 1.0512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | Adjacent | 29 | 0.96123 | 0.90818 to 1.0169 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | Adjacent | 29 | 1.0089 | 0.92907 to 1.1079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | Adjacent | 24 | 0.99504 | 0.94427 to 1.0203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | Adjacent | 28 | 0.98558 | 0.89988 to 1.0826 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | Adjacent | 27 | 0.97622 | 0.92274 to 1.0232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | Adjacent | 29 | 0.98746 | 0.90099 to 1.1131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | Adjacent | 29 | 0.97666 | 0.92635 to 1.0422 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | Adjacent | 29 | 0.98455 | 0.91624 to 1.0662 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | Adjacent | 29 | 0.96979 | 0.89181 to 1.0701 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | Adjacent | 29 | 0.98108 | 0.90283 to 1.0601 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | Adjacent | 29 | 0.94246 | 0.87849 to 1.0388 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | Adjacent | 29 | 0.98784 | 0.91723 to 1.054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | Adjacent | 29 | 0.96997 | 0.91209 to 1.0578 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | Adjacent | 29 | 0.98153 | 0.94675 to 1.0107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | Adjacent | 29 | 0.96604 | 0.91409 to 1.0312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | Adjacent | 23 | 1.0108 | 0.9532 to 1.0426 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | Adjacent | 29 | 0.95405 | 0.93133 to 0.99739 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | Adjacent | 19 | 0.9675 | 0.93659 to 0.99523 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | Adjacent | 29 | 0.98456 | 0.91308 to 1.1075 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | Adjacent | 29 | 0.98888 | 0.9124 to 1.0983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | Adjacent | 29 | 0.96349 | 0.9112 to 1.0427 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | Adjacent | 17 | 0.99782 | 0.93711 to 1.0457 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | Adjacent | 19 | 1.0034 | 0.95054 to 1.0301 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | Adjacent | 19 | 1.0087 | 0.95858 to 1.0449 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | Adjacent | 29 | 0.96919 | 0.94173 to 1.0522 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | Adjacent | 26 | 1.0132 | 0.96584 to 1.0467 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | Adjacent | 29 | 1.0703 | 0.93377 to 1.1864 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | Adjacent | 29 | 1.0188 | 0.95936 to 1.0796 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | Adjacent | 29 | 1.0042 | 0.99432 to 1.0155 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | Adjacent | 14 | 1.0048 | 0.94434 to 1.1456 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | Adjacent | 27 | 1.0124 | 0.93518 to 1.0833 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | tissue | Adjacent | 10 | 0.98685 | 0.94717 to 1.0622 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | tissue | Adjacent | 5 | 1.0791 | 1.0367 to 1.1376 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | Adjacent | 2 | 0.9538 | 0.94489 to 0.96271 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | Adjacent | 19 | 0.9847 | 0.90438 to 1.0467 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | Adjacent | 24 | 0.9849 | 0.86924 to 1.0487 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | Adjacent | 27 | 0.9907 | 0.91208 to 1.084 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | Adjacent | 28 | 1.0156 | 0.92508 to 1.1085 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | Adjacent | 27 | 0.9244 | 0.85409 to 1.0342 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | tissue | Adjacent | 4 | 1.0081 | 0.94919 to 1.0858 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | tissue | Adjacent | 3 | 1.0178 | 1.0001 to 1.0785 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | Adjacent | 24 | 0.97705 | 0.88796 to 1.1102 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | tissue | Adjacent | 24 | 1.0183 | 0.89883 to 1.0929 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | Adjacent | 25 | 0.973 | 0.91628 to 1.0634 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | Adjacent | 28 | 0.99655 | 0.93359 to 1.0402 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | Adjacent | 21 | 0.975 | 0.913 to 1.0489 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | Adjacent | 6 | 0.97395 | 0.84205 to 1.0151 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | Adjacent | 29 | 0.9417 | 0.8774 to 1.0907 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | GBM | 9 | 0.9638 | 0.92682 to 1.0102 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | tissue | GBM | 1 | 1.0185 | 1.0185 to 1.0185 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | tissue | GBM | 2 | 1.0065 | 0.9479 to 1.065 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | adjacent normal pancreas | 5 | 1.0947 | 1.0746 to 1.1566 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | adjacent normal pancreas | 5 | 1.0528 | 1.051 to 1.1465 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | adjacent normal pancreas | 5 | 1.2331 | 1.1641 to 1.3612 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | adjacent normal pancreas | 5 | 1.0879 | 1.067 to 1.1821 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | adjacent normal pancreas | 5 | 1.0673 | 1.0513 to 1.1031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | adjacent normal pancreas | 5 | 1.0009 | 0.99644 to 1.0038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | adjacent normal pancreas | 5 | 1.0269 | 1.0151 to 1.04 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | adjacent normal pancreas | 5 | 1.0824 | 1.0703 to 1.1541 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | adjacent normal pancreas | 5 | 0.98892 | 0.98645 to 0.99902 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | adjacent normal pancreas | 5 | 0.9863 | 0.98335 to 1.0407 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | adjacent normal pancreas | 5 | 0.97941 | 0.96181 to 0.99949 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | adjacent normal pancreas | 5 | 1.0356 | 1.0202 to 1.0905 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | adjacent normal pancreas | 5 | 1.0929 | 1.0586 to 1.1691 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | adjacent normal pancreas | 3 | 1.0373 | 1.0368 to 1.05 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | adjacent normal pancreas | 5 | 1.0878 | 1.0586 to 1.2031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | adjacent normal pancreas | 5 | 1.1122 | 1.0711 to 1.2132 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | adjacent normal pancreas | 5 | 1.1034 | 1.0744 to 1.2224 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | adjacent normal pancreas | 5 | 1.1237 | 1.0779 to 1.2216 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | adjacent normal pancreas | 4 | 1.0935 | 1.0759 to 1.1447 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | adjacent normal pancreas | 5 | 1.0379 | 1.0366 to 1.1625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | adjacent normal pancreas | 5 | 1.0345 | 1.0307 to 1.22 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | adjacent normal pancreas | 3 | 1.1278 | 1.1047 to 1.1297 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | adjacent normal pancreas | 5 | 1.0491 | 1.0279 to 1.1733 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | adjacent normal pancreas | 5 | 1.0619 | 1.0551 to 1.2076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | adjacent normal pancreas | 5 | 1.134 | 1.0841 to 1.2476 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | adjacent normal pancreas | 5 | 1.075 | 1.0581 to 1.3084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | adjacent normal pancreas | 4 | 1.0143 | 1.0068 to 1.0333 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | adjacent normal pancreas | 4 | 1.0714 | 1.0558 to 1.081 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | adjacent normal pancreas | 5 | 0.98323 | 0.96847 to 1.0074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | adjacent normal pancreas | 4 | 1.0598 | 1.0528 to 1.0742 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | adjacent normal pancreas | 3 | 1.0471 | 1.0462 to 1.0509 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | adjacent normal pancreas | 4 | 1.0503 | 1.0458 to 1.0751 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | adjacent normal pancreas | 4 | 1.0586 | 1.0516 to 1.0785 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | adjacent normal pancreas | 4 | 1.0068 | 0.97663 to 1.0571 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | adjacent normal pancreas | 5 | 1.1124 | 1.0844 to 1.2538 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | adjacent normal pancreas | 3 | 1.0259 | 1.0245 to 1.0314 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | adjacent normal pancreas | 4 | 1.0856 | 1.066 to 1.1424 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | adjacent normal pancreas | 4 | 1.0277 | 1.0208 to 1.0595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | adjacent normal pancreas | 4 | 1.0933 | 1.0726 to 1.1474 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | adjacent normal pancreas | 5 | 1.042 | 1.0162 to 1.1193 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | adjacent normal pancreas | 4 | 1.0626 | 1.0467 to 1.1106 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | adjacent normal pancreas | 5 | 1.0508 | 1.0218 to 1.2305 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | adjacent normal pancreas | 5 | 1.0665 | 1.0349 to 1.2218 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | adjacent normal pancreas | 5 | 1.0107 | 0.98983 to 1.2011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | adjacent normal pancreas | 5 | 1.0581 | 1.0394 to 1.15 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | adjacent normal pancreas | 4 | 1.0496 | 1.0403 to 1.1042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | adjacent normal pancreas | 5 | 1.0092 | 0.99412 to 1.0518 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | adjacent normal pancreas | 5 | 1.0261 | 1.0106 to 1.1481 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | adjacent normal pancreas | 2 | 1.0541 | 1.054 to 1.0542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | adjacent normal pancreas | 5 | 0.97764 | 0.95483 to 1.0098 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | adjacent normal pancreas | 5 | 1.0646 | 1.0628 to 1.238 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | adjacent normal pancreas | 4 | 1.0708 | 1.0514 to 1.1275 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | adjacent normal pancreas | 5 | 1.0196 | 1.0084 to 1.1244 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | adjacent normal pancreas | 5 | 0.98848 | 0.98469 to 1.0132 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | adjacent normal pancreas | 3 | 1.0543 | 1.052 to 1.0618 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | adjacent normal pancreas | 5 | 1.2281 | 1.1615 to 1.353 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | adjacent normal pancreas | 5 | 1.0753 | 1.049 to 1.1424 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | adjacent normal pancreas | 5 | 1.0009 | 0.99116 to 1.0107 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | adjacent normal pancreas | 2 | 1.0629 | 1.0568 to 1.069 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | tissue | adjacent normal pancreas | 3 | 0.9756 | 0.8712 to 0.99756 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | adjacent normal pancreas | 4 | 0.97035 | 0.93787 to 1.0002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | adjacent normal pancreas | 4 | 0.95865 | 0.94593 to 0.99433 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | adjacent normal pancreas | 4 | 1.0421 | 0.96301 to 1.0543 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | adjacent normal pancreas | 4 | 0.9995 | 0.94716 to 1.0594 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | adjacent normal pancreas | 5 | 1.0522 | 0.9285 to 1.0942 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | adjacent normal pancreas | 2 | 0.9959 | 0.98024 to 1.0116 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | adjacent normal pancreas | 2 | 1.0174 | 0.99409 to 1.0407 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | adjacent normal pancreas | 5 | 0.9173 | 0.88992 to 0.98834 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | adjacent normal pancreas | 2 | 1.0171 | 0.99694 to 1.0372 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | adjacent normal pancreas | 1 | 0.9936 | 0.9936 to 0.9936 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | adjacent normal pancreas | 5 | 1.0218 | 1.018 to 1.1447 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | adjacent normal pancreas | 5 | 0.9476 | 0.9252 to 1.0356 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | adjacent normal pancreas | 5 | 1.0066 | 0.98714 to 1.0821 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | adjacent normal pancreas | 5 | 0.9918 | 0.88538 to 1.053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | adjacent normal pancreas | 4 | 1.1029 | 1.059 to 1.1175 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | adjacent normal pancreas | 5 | 0.9568 | 0.89258 to 1.0164 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | Adjacent | 6 | 0.9843 | 0.91643 to 1.0279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | tissue | Adjacent | 2 | 0.96445 | 0.89457 to 1.0343 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | Adjacent | 22 | 0.9639 | 0.89064 to 1.0683 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | tissue | Adjacent | 2 | 1.0447 | 0.98872 to 1.1006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | Adjacent | 18 | 1.0122 | 0.92782 to 1.1252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | Adjacent | 10 | 1.0252 | 0.93029 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | Adjacent | 3 | 0.9522 | 0.94185 to 1.0164 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | tissue | Adjacent | 3 | 0.9318 | 0.92622 to 1.045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | tissue | Adjacent | 6 | 1.0478 | 0.9686 to 1.1459 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | tissue | Adjacent | 4 | 1.0787 | 0.96308 to 1.125 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | Adjacent | 7 | 0.9332 | 0.8557 to 1.0313 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | Adjacent | 15 | 1.0205 | 0.95617 to 1.0633 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | Adjacent | 18 | 1.0572 | 1.0231 to 1.1411 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | tissue | Adjacent | 6 | 1.0483 | 1.0226 to 1.0992 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | tissue | Adjacent | 1 | 0.9565 | 0.9565 to 0.9565 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | Adjacent | 12 | 0.9412 | 0.88777 to 1.022 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | tissue | Adjacent | 2 | 1.0476 | 1.047 to 1.0483 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | GBM | 9 | 0.99919 | 0.94478 to 1.0555 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | GBM | 6 | 1.0153 | 0.9621 to 1.0507 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | tissue | GBM | 8 | 1.0236 | 0.96354 to 1.0825 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | GBM | 4 | 1.0161 | 0.97907 to 1.0412 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | tissue | pancreatic cancer | 2 | 1.0417 | 0.97726 to 1.1062 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | pancreatic cancer | 3 | 1.0093 | 0.90148 to 1.013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | pancreatic cancer | 8 | 0.9881 | 0.89789 to 1.0772 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | tissue | pancreatic cancer | 5 | 1.0354 | 0.92348 to 1.0414 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | pancreatic cancer | 5 | 0.9606 | 0.93246 to 1.0252 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | pancreatic cancer | 4 | 0.97145 | 0.91082 to 1.0565 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | pancreatic cancer | 3 | 1.0055 | 0.92117 to 1.0656 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | Adjacent | 4 | 0.93545 | 0.89761 to 0.99701 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | Adjacent | 4 | 1.0042 | 0.99412 to 1.0415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | tissue | Adjacent | 6 | 0.9489 | 0.84925 to 1.0437 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon fibroblasts | tissue | Adjacent | 7 | 1.0102 | 0.95941 to 1.0408 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | Adjacent | 1 | 0.9447 | 0.9447 to 0.9447 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | tissue | Adjacent | 5 | 1.0913 | 1.0181 to 1.1493 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | Adjacent | 5 | 0.9431 | 0.9188 to 0.95374 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | Adjacent | 8 | 0.933 | 0.8869 to 1.1326 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | GBM | 6 | 0.9797 | 0.91392 to 1.0532 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | GBM | 2 | 0.9994 | 0.99292 to 1.0059 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | GBM | 4 | 0.94745 | 0.83836 to 1.0297 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | adjacent normal pancreas | 2 | 1.0252 | 1.0081 to 1.0422 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | tissue | adjacent normal pancreas | 1 | 1.0984 | 1.0984 to 1.0984 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | tissue | adjacent normal pancreas | 1 | 1.0987 | 1.0987 to 1.0987 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | adjacent normal pancreas | 1 | 0.8683 | 0.8683 to 0.8683 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | adjacent normal pancreas | 3 | 0.9862 | 0.96595 to 1.0369 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | tissue | adjacent normal pancreas | 1 | 1.0067 | 1.0067 to 1.0067 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | adjacent normal pancreas | 2 | 1.0562 | 1.0432 to 1.0693 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | tissue | adjacent normal pancreas | 1 | 0.9625 | 0.9625 to 0.9625 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | tissue | adjacent normal pancreas | 2 | 1.0511 | 1.0448 to 1.0574 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | tissue | GBM | 1 | 1.0293 | 1.0293 to 1.0293 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | tissue | pancreatic cancer | 3 | 1.0748 | 1.0564 to 1.1084 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | pancreatic cancer | 1 | 1.0033 | 1.0033 to 1.0033 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | tissue | Tumor | 1 | 1.0225 | 1.0225 to 1.0225 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | cardiomyocyte | tissue | GBM | 7 | 0.97034 | 0.88611 to 1.0151 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | GBM | 7 | 1.0236 | 0.98117 to 1.1524 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | tissue | GBM | 8 | 0.97684 | 0.92957 to 1.0391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | GBM | 8 | 0.98974 | 0.93142 to 1.0617 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | tissue | GBM | 6 | 0.98939 | 0.96269 to 1.0473 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | tissue | GBM | 8 | 0.98234 | 0.93987 to 1.0468 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | tissue | GBM | 5 | 0.99621 | 0.94605 to 1.0421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | regulatory t cells | tissue | adjacent normal pancreas | 2 | 0.9233 | 0.91196 to 0.93464 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | tissue | Tumor | 1 | 0.9979 | 0.9979 to 0.9979 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | tissue | Adjacent | 1 | 0.9549 | 0.9549 to 0.9549 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | bladder epithelium | tissue | pancreatic cancer | 3 | 0.9339 | 0.83985 to 1.0982 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | tissue | Adjacent | 1 | 0.9694 | 0.9694 to 0.9694 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cardiomyocyte | tissue | Adjacent | 2 | 0.93525 | 0.93376 to 0.93673 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | tissue | Adjacent | 3 | 1.0647 | 0.99153 to 1.0923 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | tissue | Adjacent | 1 | 1.003 | 1.003 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | fallopien tubes epithelium | tissue | Adjacent | 2 | 0.97435 | 0.9654 to 0.9833 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | Non-tumor brain | 5 | 1.0009 | 0.98345 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | Non-tumor brain | 5 | 0.94367 | 0.9388 to 0.9759 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | Non-tumor brain | 5 | 0.92605 | 0.90169 to 0.95563 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | Non-tumor brain | 5 | 1.0142 | 1.0062 to 1.0595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | Non-tumor brain | 5 | 1.004 | 0.98793 to 1.0192 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | Non-tumor brain | 5 | 0.98119 | 0.9587 to 0.98224 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | Non-tumor brain | 5 | 0.98165 | 0.9767 to 0.99392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | Non-tumor brain | 5 | 0.97921 | 0.97571 to 0.99836 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | Non-tumor brain | 5 | 0.97072 | 0.95547 to 0.97808 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | Non-tumor brain | 5 | 0.94968 | 0.94201 to 0.95935 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | Non-tumor brain | 5 | 0.94529 | 0.92087 to 0.97593 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | Non-tumor brain | 5 | 0.94956 | 0.94675 to 0.97084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | Non-tumor brain | 5 | 0.95556 | 0.94501 to 0.98241 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | Non-tumor brain | 5 | 0.95124 | 0.9414 to 0.96921 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | tissue | Non-tumor brain | 2 | 0.97342 | 0.96199 to 0.98486 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | Non-tumor brain | 2 | 0.96766 | 0.96186 to 0.97347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | tissue | Non-tumor brain | 2 | 0.96795 | 0.95357 to 0.98233 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | Non-tumor brain | 5 | 0.94668 | 0.94384 to 0.97689 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | Non-tumor brain | 5 | 0.93835 | 0.93809 to 0.96451 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | Non-tumor brain | 5 | 0.93659 | 0.92948 to 0.96155 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | Non-tumor brain | 5 | 0.95757 | 0.95059 to 0.98368 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | Non-tumor brain | 5 | 0.93642 | 0.92773 to 0.96127 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | Non-tumor brain | 5 | 0.9358 | 0.93327 to 0.95521 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | Non-tumor brain | 5 | 0.91715 | 0.90987 to 0.94768 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | Non-tumor brain | 5 | 1.013 | 1.0014 to 1.0369 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | Non-tumor brain | 2 | 1.0065 | 1.0035 to 1.0094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | Non-tumor brain | 5 | 0.90395 | 0.88868 to 0.92971 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | Non-tumor brain | 5 | 0.93205 | 0.92869 to 0.95605 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | Non-tumor brain | 5 | 0.9521 | 0.94589 to 0.98185 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | Non-tumor brain | 5 | 0.90687 | 0.90415 to 0.93322 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | tissue | Non-tumor brain | 2 | 0.97237 | 0.96472 to 0.98003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | Non-tumor brain | 5 | 0.95762 | 0.94594 to 0.96725 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | Non-tumor brain | 5 | 1.0051 | 0.98968 to 1.0156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | tissue | Non-tumor brain | 2 | 0.97673 | 0.96837 to 0.9851 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | Non-tumor brain | 2 | 0.97479 | 0.96571 to 0.98387 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | Non-tumor brain | 5 | 0.95417 | 0.93442 to 0.95736 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | Non-tumor brain | 5 | 0.98021 | 0.96797 to 0.98714 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | Non-tumor brain | 5 | 0.97179 | 0.96348 to 0.9833 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | Non-tumor brain | 5 | 0.97598 | 0.9648 to 0.98635 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | Non-tumor brain | 5 | 0.9779 | 0.96697 to 0.99081 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | Non-tumor brain | 5 | 0.92969 | 0.90755 to 0.96213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | Non-tumor brain | 5 | 0.94602 | 0.9428 to 0.98176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | Non-tumor brain | 5 | 0.97081 | 0.9572 to 0.98038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | Non-tumor brain | 5 | 0.92571 | 0.91763 to 0.94806 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | Non-tumor brain | 5 | 0.93793 | 0.93093 to 0.94964 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | Non-tumor brain | 5 | 0.91849 | 0.9001 to 0.94858 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | Non-tumor brain | 5 | 0.96683 | 0.94682 to 0.97727 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | Non-tumor brain | 5 | 0.9356 | 0.92984 to 0.96223 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | Non-tumor brain | 5 | 0.91516 | 0.90942 to 0.94409 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | Non-tumor brain | 5 | 0.92868 | 0.92483 to 0.95839 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | Non-tumor brain | 5 | 0.8879 | 0.88359 to 0.9148 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | Non-tumor brain | 5 | 0.95023 | 0.94683 to 0.97677 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | Non-tumor brain | 5 | 0.88932 | 0.87274 to 0.91582 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | tissue | Non-tumor brain | 2 | 0.95528 | 0.94621 to 0.96434 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | Non-tumor brain | 5 | 0.97076 | 0.9661 to 0.98395 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | tissue | Non-tumor brain | 2 | 0.95782 | 0.94992 to 0.96573 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | Non-tumor brain | 5 | 0.92584 | 0.925 to 0.94272 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | Non-tumor brain | 4 | 0.96493 | 0.95306 to 0.97723 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | Non-tumor brain | 5 | 0.95607 | 0.95299 to 0.96767 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | Non-tumor brain | 2 | 0.95409 | 0.94164 to 0.96654 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | Non-tumor brain | 5 | 0.94244 | 0.93252 to 0.96989 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | tissue | Non-tumor brain | 2 | 0.98546 | 0.97291 to 0.99801 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | Non-tumor brain | 5 | 0.91249 | 0.90578 to 0.93587 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | Non-tumor brain | 5 | 0.92425 | 0.92224 to 0.94261 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | Non-tumor brain | 2 | 0.96702 | 0.96021 to 0.97383 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | Non-tumor brain | 2 | 0.96209 | 0.95306 to 0.97112 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | Non-tumor brain | 2 | 0.97845 | 0.96885 to 0.98804 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | Non-tumor brain | 5 | 0.95619 | 0.92329 to 0.96975 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | Non-tumor brain | 5 | 0.97512 | 0.96267 to 0.98823 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | Non-tumor brain | 5 | 0.92491 | 0.89929 to 0.95302 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | Non-tumor brain | 5 | 1.0173 | 1.0102 to 1.0543 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | Non-tumor brain | 5 | 1.0068 | 0.99866 to 1.0083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | tissue | Non-tumor brain | 5 | 1.0324 | 0.94166 to 1.1203 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | Non-tumor brain | 5 | 1.0085 | 0.99748 to 1.0287 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | astrocytes | tissue | Non-tumor brain | 5 | 0.9463 | 0.85824 to 0.9842 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | tissue | Non-tumor brain | 4 | 0.9343 | 0.86572 to 0.95256 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | Non-tumor brain | 5 | 1.0021 | 0.97892 to 1.0363 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | Non-tumor brain | 3 | 0.919 | 0.91783 to 0.98632 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | Non-tumor brain | 4 | 0.99205 | 0.93349 to 1.0166 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | tissue | Non-tumor brain | 5 | 1.1052 | 0.96436 to 1.1625 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | Non-tumor brain | 1 | 0.9099 | 0.9099 to 0.9099 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | Non-tumor brain | 1 | 1.0258 | 1.0258 to 1.0258 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | tissue | Non-tumor brain | 4 | 0.8932 | 0.78909 to 1.0569 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | Non-tumor brain | 5 | 0.9422 | 0.9136 to 1.1279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | Non-tumor brain | 4 | 0.9545 | 0.95249 to 0.9594 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | tissue | Non-tumor brain | 4 | 0.9765 | 0.90404 to 1.0366 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | juice circulating cancer cells | 4 | 1.0802 | 1.0524 to 1.0922 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | juice circulating cancer cells | 4 | 1.0819 | 1.057 to 1.0868 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | juice circulating cancer cells | 4 | 1.2243 | 1.1356 to 1.3759 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | juice circulating cancer cells | 4 | 1.0484 | 1.006 to 1.1096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | juice circulating cancer cells | 4 | 1.0543 | 1.0303 to 1.0682 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | juice circulating cancer cells | 4 | 0.97652 | 0.95397 to 1.0198 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | juice circulating cancer cells | 4 | 0.98069 | 0.97703 to 1.0222 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | juice circulating cancer cells | 4 | 1.059 | 1.0361 to 1.0774 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | juice circulating cancer cells | 4 | 0.9784 | 0.94885 to 1.0217 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | juice circulating cancer cells | 4 | 1.0025 | 0.95246 to 1.0454 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | juice circulating cancer cells | 4 | 0.98734 | 0.95292 to 1.0131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | juice circulating cancer cells | 4 | 1.0579 | 1.0531 to 1.0927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | juice circulating cancer cells | 4 | 1.0463 | 1.0166 to 1.0596 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | juice circulating cancer cells | 4 | 1.0993 | 1.0664 to 1.1083 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | juice circulating cancer cells | 4 | 1.1566 | 1.1399 to 1.1706 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | juice circulating cancer cells | 4 | 1.1112 | 1.0821 to 1.1258 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | juice circulating cancer cells | 4 | 1.0805 | 1.0492 to 1.1118 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | juice circulating cancer cells | 4 | 1.0915 | 1.0739 to 1.0992 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | juice circulating cancer cells | 4 | 1.076 | 1.0533 to 1.0953 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | juice circulating cancer cells | 4 | 1.1014 | 1.0697 to 1.124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | juice circulating cancer cells | 4 | 1.0811 | 1.0473 to 1.1058 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | juice circulating cancer cells | 4 | 1.096 | 1.0731 to 1.115 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | juice circulating cancer cells | 4 | 1.1226 | 1.0953 to 1.1525 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | juice circulating cancer cells | 4 | 1.1201 | 1.1091 to 1.1519 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | juice circulating cancer cells | 4 | 1.0245 | 0.99233 to 1.0479 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | juice circulating cancer cells | 4 | 1.0727 | 1.0651 to 1.0894 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | juice circulating cancer cells | 4 | 1.0134 | 0.97965 to 1.0224 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | juice circulating cancer cells | 3 | 1.0747 | 1.0528 to 1.0922 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | juice circulating cancer cells | 4 | 1.0596 | 1.0378 to 1.0791 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | juice circulating cancer cells | 4 | 1.0805 | 1.0518 to 1.1042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | juice circulating cancer cells | 4 | 1.0402 | 1.0117 to 1.0866 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | juice circulating cancer cells | 4 | 1.1372 | 1.1044 to 1.1773 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | juice circulating cancer cells | 4 | 1.0592 | 1.0253 to 1.0849 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | juice circulating cancer cells | 4 | 1.1417 | 1.1105 to 1.1728 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | juice circulating cancer cells | 4 | 1.0529 | 1.0217 to 1.0666 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | juice circulating cancer cells | 4 | 1.105 | 1.0878 to 1.1127 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | juice circulating cancer cells | 4 | 1.1239 | 1.0916 to 1.1491 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | juice circulating cancer cells | 4 | 1.1084 | 1.0874 to 1.1313 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | juice circulating cancer cells | 4 | 1.1048 | 1.0535 to 1.1412 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | juice circulating cancer cells | 4 | 1.0478 | 1.0162 to 1.0607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | juice circulating cancer cells | 4 | 1.092 | 1.0688 to 1.1193 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | juice circulating cancer cells | 4 | 1.0122 | 0.99424 to 1.039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | juice circulating cancer cells | 4 | 1.0626 | 1.0566 to 1.0935 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | juice circulating cancer cells | 4 | 0.97425 | 0.94994 to 1.0273 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | juice circulating cancer cells | 4 | 1.11 | 1.0997 to 1.1463 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | juice circulating cancer cells | 4 | 1.1891 | 1.1687 to 1.2078 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | juice circulating cancer cells | 4 | 1.0732 | 1.0147 to 1.1021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | juice circulating cancer cells | 4 | 1.0698 | 1.0467 to 1.0899 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | juice circulating cancer cells | 4 | 1.2156 | 1.129 to 1.3687 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | juice circulating cancer cells | 4 | 1.0425 | 0.99693 to 1.0953 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | juice circulating cancer cells | 4 | 0.99815 | 0.99341 to 1.0078 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | tissue | juice circulating cancer cells | 2 | 1.0737 | 1.0703 to 1.0772 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | juice circulating cancer cells | 3 | 1.0045 | 0.93313 to 1.0454 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | juice circulating cancer cells | 4 | 1.057 | 0.97171 to 1.163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | tissue | juice circulating cancer cells | 1 | 0.9518 | 0.9518 to 0.9518 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | juice circulating cancer cells | 4 | 0.99005 | 0.96874 to 1.0097 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar epithelium | tissue | juice circulating cancer cells | 3 | 1.0139 | 0.98609 to 1.0627 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | juice circulating cancer cells | 3 | 0.942 | 0.84579 to 0.9447 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | tissue | juice circulating cancer cells | 1 | 0.9786 | 0.9786 to 0.9786 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | tissue | juice circulating cancer cells | 4 | 1.0168 | 0.96778 to 1.0994 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | juice circulating cancer cells | 3 | 1.0072 | 0.97093 to 1.043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | juice circulating cancer cells | 4 | 1.0848 | 1.0399 to 1.1281 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | juice circulating cancer cells | 2 | 1.0351 | 1.0283 to 1.0418 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic duct | tissue | juice circulating cancer cells | 2 | 0.92555 | 0.91983 to 0.93126 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | juice circulating cancer cells | 3 | 1.0073 | 0.98399 to 1.1405 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | juice circulating cancer cells | 3 | 1.027 | 0.97075 to 1.0425 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | juice circulating cancer cells | 4 | 1.0764 | 0.99408 to 1.0939 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | tissue | adjacent normal pancreas | 1 | 1.0075 | 1.0075 to 1.0075 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar macrophages | tissue | Tumor | 3 | 0.9314 | 0.82169 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | Non-tumor brain | 4 | 0.94945 | 0.91434 to 1.0232 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | basophils | tissue | juice circulating cancer cells | 2 | 1.0649 | 1.0484 to 1.0813 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | juice circulating cancer cells | 1 | 1.0421 | 1.0421 to 1.0421 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | tissue | juice circulating cancer cells | 1 | 1.0258 | 1.0258 to 1.0258 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | juice circulating cancer cells | 2 | 1.092 | 1.0866 to 1.0974 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | tissue | juice circulating cancer cells | 1 | 1.0957 | 1.0957 to 1.0957 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | tissue | juice circulating cancer cells | 1 | 1.0432 | 1.0432 to 1.0432 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | tissue | juice circulating cancer cells | 1 | 1.0462 | 1.0462 to 1.0462 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | juice circulating cancer cells | 2 | 1.0591 | 1.0518 to 1.0664 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | juice circulating cancer cells | 2 | 1.0616 | 1.0601 to 1.063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | juice circulating cancer cells | 2 | 1.1115 | 1.1027 to 1.1203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | tissue | juice circulating cancer cells | 1 | 1.045 | 1.045 to 1.045 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | tissue | juice circulating cancer cells | 1 | 1.0395 | 1.0395 to 1.0395 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | tissue | juice circulating cancer cells | 1 | 1.0497 | 1.0497 to 1.0497 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | tissue | juice circulating cancer cells | 1 | 1.0712 | 1.0712 to 1.0712 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | juice circulating cancer cells | 1 | 1.0422 | 1.0422 to 1.0422 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | juice circulating cancer cells | 1 | 1.0394 | 1.0394 to 1.0394 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | juice circulating cancer cells | 2 | 1.0541 | 1.0484 to 1.0597 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | GMP (bone marrow) | tissue | juice circulating cancer cells | 2 | 1.0015 | 0.98305 to 1.0199 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | tissue | pancreatic cancer | 2 | 1.0019 | 0.95591 to 1.0479 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | tissue | pancreatic cancer | 2 | 1.0432 | 1.0263 to 1.0601 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | tissue | pancreatic cancer | 3 | 0.9531 | 0.93051 to 1.0013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | tissue | pancreatic cancer | 1 | 1.0034 | 1.0034 to 1.0034 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | bladder epithelium | tissue | GBM | 4 | 0.9849 | 0.94313 to 1.0316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | tissue | GBM | 3 | 0.9742 | 0.94767 to 0.9899 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | tonsil pharyngeal epithelium | tissue | Tumor | 3 | 0.8519 | 0.81212 to 1.007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | tissue | Adjacent | 4 | 1.0172 | 0.99677 to 1.0259 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | tissue | pancreatic cancer | 1 | 1.0166 | 1.0166 to 1.0166 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | adjacent normal pancreas | 1 | 1.0545 | 1.0545 to 1.0545 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | adjacent normal pancreas | 1 | 0.98674 | 0.98674 to 0.98674 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | adjacent normal pancreas | 1 | 1.0447 | 1.0447 to 1.0447 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | adjacent normal pancreas | 1 | 1.0515 | 1.0515 to 1.0515 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | primary culture cancer cells | 1 | 1.0618 | 1.0618 to 1.0618 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | primary culture cancer cells | 1 | 1.0763 | 1.0763 to 1.0763 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | primary culture cancer cells | 1 | 1.3607 | 1.3607 to 1.3607 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | primary culture cancer cells | 1 | 1.0465 | 1.0465 to 1.0465 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | primary culture cancer cells | 1 | 0.9996 | 0.9996 to 0.9996 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | primary culture cancer cells | 1 | 0.88512 | 0.88512 to 0.88512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | primary culture cancer cells | 1 | 0.93856 | 0.93856 to 0.93856 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | primary culture cancer cells | 1 | 1.042 | 1.042 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | primary culture cancer cells | 1 | 0.87732 | 0.87732 to 0.87732 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | primary culture cancer cells | 1 | 0.88878 | 0.88878 to 0.88878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | primary culture cancer cells | 1 | 0.87793 | 0.87793 to 0.87793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | primary culture cancer cells | 1 | 1.016 | 1.016 to 1.016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | primary culture cancer cells | 1 | 1.0652 | 1.0652 to 1.0652 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | primary culture cancer cells | 1 | 0.9945 | 0.9945 to 0.9945 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | primary culture cancer cells | 1 | 1.0915 | 1.0915 to 1.0915 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | primary culture cancer cells | 1 | 1.0869 | 1.0869 to 1.0869 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | primary culture cancer cells | 1 | 1.0843 | 1.0843 to 1.0843 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | primary culture cancer cells | 1 | 1.1042 | 1.1042 to 1.1042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | primary culture cancer cells | 1 | 1.0642 | 1.0642 to 1.0642 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | primary culture cancer cells | 1 | 1.0024 | 1.0024 to 1.0024 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | primary culture cancer cells | 1 | 1.0733 | 1.0733 to 1.0733 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | primary culture cancer cells | 1 | 1.1844 | 1.1844 to 1.1844 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | primary culture cancer cells | 1 | 1.0499 | 1.0499 to 1.0499 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | primary culture cancer cells | 1 | 1.0328 | 1.0328 to 1.0328 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | primary culture cancer cells | 1 | 1.1172 | 1.1172 to 1.1172 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | primary culture cancer cells | 1 | 1.1046 | 1.1046 to 1.1046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | primary culture cancer cells | 1 | 0.93418 | 0.93418 to 0.93418 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | primary culture cancer cells | 1 | 1.0258 | 1.0258 to 1.0258 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | primary culture cancer cells | 1 | 0.89057 | 0.89057 to 0.89057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | primary culture cancer cells | 1 | 1.056 | 1.056 to 1.056 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | primary culture cancer cells | 1 | 1.0569 | 1.0569 to 1.0569 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | primary culture cancer cells | 1 | 1.0556 | 1.0556 to 1.0556 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | primary culture cancer cells | 1 | 1.067 | 1.067 to 1.067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | primary culture cancer cells | 1 | 1.0186 | 1.0186 to 1.0186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | primary culture cancer cells | 1 | 1.1447 | 1.1447 to 1.1447 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | tissue | primary culture cancer cells | 1 | 1.0208 | 1.0208 to 1.0208 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | primary culture cancer cells | 1 | 1.0663 | 1.0663 to 1.0663 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | primary culture cancer cells | 1 | 0.96628 | 0.96628 to 0.96628 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | primary culture cancer cells | 1 | 1.1255 | 1.1255 to 1.1255 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | primary culture cancer cells | 1 | 1.0087 | 1.0087 to 1.0087 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | primary culture cancer cells | 1 | 1.0631 | 1.0631 to 1.0631 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | primary culture cancer cells | 1 | 1.0452 | 1.0452 to 1.0452 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | primary culture cancer cells | 1 | 1.0491 | 1.0491 to 1.0491 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | primary culture cancer cells | 1 | 1.008 | 1.008 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | primary culture cancer cells | 1 | 1.0429 | 1.0429 to 1.0429 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | primary culture cancer cells | 1 | 1.077 | 1.077 to 1.077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | primary culture cancer cells | 1 | 0.89542 | 0.89542 to 0.89542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | primary culture cancer cells | 1 | 1.0323 | 1.0323 to 1.0323 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | primary culture cancer cells | 1 | 0.86338 | 0.86338 to 0.86338 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | primary culture cancer cells | 1 | 1.1096 | 1.1096 to 1.1096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | primary culture cancer cells | 1 | 1.0909 | 1.0909 to 1.0909 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | primary culture cancer cells | 1 | 0.95919 | 0.95919 to 0.95919 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | primary culture cancer cells | 1 | 0.94644 | 0.94644 to 0.94644 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | primary culture cancer cells | 1 | 1.0661 | 1.0661 to 1.0661 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | primary culture cancer cells | 1 | 1.3579 | 1.3579 to 1.3579 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | primary culture cancer cells | 1 | 1.0166 | 1.0166 to 1.0166 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | primary culture cancer cells | 1 | 0.996 | 0.996 to 0.996 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | bladder epithelium | tissue | primary culture cancer cells | 1 | 0.8751 | 0.8751 to 0.8751 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast luminal epithelium | tissue | primary culture cancer cells | 1 | 1.1674 | 1.1674 to 1.1674 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | tissue | primary culture cancer cells | 1 | 1.1913 | 1.1913 to 1.1913 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | tissue | primary culture cancer cells | 1 | 1.1218 | 1.1218 to 1.1218 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon epithelium | tissue | primary culture cancer cells | 1 | 1.0043 | 1.0043 to 1.0043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | primary culture cancer cells | 1 | 0.939 | 0.939 to 0.939 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | primary culture cancer cells | 1 | 0.9445 | 0.9445 to 0.9445 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | primary culture cancer cells | 1 | 1.0043 | 1.0043 to 1.0043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | tissue | primary culture cancer cells | 1 | 1.1466 | 1.1466 to 1.1466 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | tissue | primary culture cancer cells | 1 | 1.05 | 1.05 to 1.05 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | tissue | primary culture cancer cells | 1 | 1.0324 | 1.0324 to 1.0324 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | tissue | primary culture cancer cells | 1 | 0.972 | 0.972 to 0.972 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | tissue | primary culture cancer cells | 1 | 0.9565 | 0.9565 to 0.9565 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | primary culture cancer cells | 1 | 1.1194 | 1.1194 to 1.1194 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cardiomyocyte | tissue | Non-tumor brain | 1 | 1.0097 | 1.0097 to 1.0097 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | Non-tumor brain | 1 | 1.0121 | 1.0121 to 1.0121 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | tissue | pancreatic cancer | 1 | 0.9644 | 0.9644 to 0.9644 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | tissue | pancreatic cancer | 1 | 1.1119 | 1.1119 to 1.1119 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | tissue | pancreatic cancer | 1 | 0.9914 | 0.9914 to 0.9914 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | bladder epithelium | tissue | juice circulating cancer cells | 1 | 0.9352 | 0.9352 to 0.9352 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon enteroendocrine | tissue | juice circulating cancer cells | 2 | 1.0155 | 1.0015 to 1.0296 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | tissue | juice circulating cancer cells | 1 | 0.9724 | 0.9724 to 0.9724 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric antrum epithelium | tissue | juice circulating cancer cells | 1 | 0.9177 | 0.9177 to 0.9177 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric body epithelium | tissue | juice circulating cancer cells | 1 | 1.0754 | 1.0754 to 1.0754 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | gastric fundus epithelium | tissue | juice circulating cancer cells | 1 | 0.9406 | 0.9406 to 0.9406 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | tissue | juice circulating cancer cells | 1 | 1.0311 | 1.0311 to 1.0311 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | tissue | Non-tumor brain | 1 | 1.0016 | 1.0016 to 1.0016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar endothelium | tissue | Non-tumor brain | 1 | 0.8651 | 0.8651 to 0.8651 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | tissue | pancreatic cancer | 1 | 1.014 | 1.014 to 1.014 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | tissue | pancreatic cancer | 1 | 0.9892 | 0.9892 to 0.9892 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | tissue | Adjacent | 1 | 0.9826 | 0.9826 to 0.9826 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | tissue | Adjacent | 1 | 1.0185 | 1.0185 to 1.0185 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | tissue | Adjacent | 1 | 1.0705 | 1.0705 to 1.0705 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | tissue | Adjacent | 1 | 1.0778 | 1.0778 to 1.0778 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | tissue | pancreatic cancer | 1 | 1.0469 | 1.0469 to 1.0469 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | tissue | pancreatic cancer | 1 | 1.0083 | 1.0083 to 1.0083 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | tissue | pancreatic cancer | 1 | 1.0658 | 1.0658 to 1.0658 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | tissue | pancreatic cancer | 1 | 1.0559 | 1.0559 to 1.0559 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample | whole sample | tissue | primary cancer associated fibroblasts | 3 | 1.0936 | 1.057 to 1.1115 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | tissue | primary cancer associated fibroblasts | 3 | 1.1133 | 1.1065 to 1.129 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | tissue | primary cancer associated fibroblasts | 3 | 1.3409 | 1.2962 to 1.3893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | tissue | primary cancer associated fibroblasts | 3 | 1.0973 | 1.0511 to 1.1248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | tissue | primary cancer associated fibroblasts | 3 | 1.0332 | 0.98143 to 1.0592 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 0.94131 | 0.86943 to 0.95146 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 0.98102 | 0.92499 to 0.98763 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 1.0924 | 1.0622 to 1.0936 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 0.92786 | 0.85902 to 0.93869 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 0.94921 | 0.88603 to 0.95603 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | tissue | primary cancer associated fibroblasts | 3 | 0.94469 | 0.85454 to 0.96077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | tissue | primary cancer associated fibroblasts | 3 | 1.0442 | 1.0041 to 1.0581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | tissue | primary cancer associated fibroblasts | 3 | 1.093 | 1.0776 to 1.0971 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | tissue | primary cancer associated fibroblasts | 3 | 1.0488 | 1.0081 to 1.0535 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.1042 | 1.1028 to 1.1072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | tissue | primary cancer associated fibroblasts | 3 | 1.1197 | 1.0957 to 1.1205 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | tissue | primary cancer associated fibroblasts | 3 | 1.117 | 1.1025 to 1.1244 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.1266 | 1.1176 to 1.1287 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | tissue | primary cancer associated fibroblasts | 3 | 1.1317 | 1.1192 to 1.1391 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | tissue | primary cancer associated fibroblasts | 3 | 1.0357 | 1.0105 to 1.0422 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.0887 | 1.0681 to 1.0894 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | tissue | primary cancer associated fibroblasts | 3 | 1.2045 | 1.1888 to 1.2058 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | tissue | primary cancer associated fibroblasts | 3 | 1.0695 | 1.0612 to 1.0723 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | tissue | primary cancer associated fibroblasts | 3 | 1.0848 | 1.0647 to 1.0871 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.1401 | 1.1381 to 1.1503 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | tissue | primary cancer associated fibroblasts | 3 | 1.1182 | 1.1138 to 1.1219 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | tissue | primary cancer associated fibroblasts | 3 | 0.98528 | 0.92538 to 0.98561 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | tissue | primary cancer associated fibroblasts | 3 | 1.0636 | 1.0132 to 1.0659 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | tissue | primary cancer associated fibroblasts | 3 | 0.91762 | 0.84994 to 0.93484 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | tissue | primary cancer associated fibroblasts | 3 | 1.0982 | 1.0677 to 1.1019 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | tissue | primary cancer associated fibroblasts | 3 | 1.0776 | 1.0435 to 1.0785 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.0685 | 1.0218 to 1.0832 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | tissue | primary cancer associated fibroblasts | 3 | 1.1143 | 1.0769 to 1.1194 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.0306 | 0.95411 to 1.0365 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.1617 | 1.158 to 1.1623 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | tissue | primary cancer associated fibroblasts | 3 | 1.1145 | 1.1062 to 1.1193 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.0082 | 0.96507 to 1.0092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | tissue | primary cancer associated fibroblasts | 3 | 1.1518 | 1.1418 to 1.1548 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | tissue | primary cancer associated fibroblasts | 3 | 1.0388 | 0.99999 to 1.0437 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | tissue | primary cancer associated fibroblasts | 3 | 1.0903 | 1.0705 to 1.0955 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.0618 | 1.0446 to 1.0705 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | tissue | primary cancer associated fibroblasts | 3 | 1.0653 | 1.0519 to 1.0734 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | tissue | primary cancer associated fibroblasts | 3 | 1.0203 | 1.0012 to 1.0266 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | tissue | primary cancer associated fibroblasts | 3 | 1.0689 | 1.0412 to 1.0765 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | tissue | primary cancer associated fibroblasts | 3 | 1.0839 | 1.0738 to 1.0873 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | tissue | primary cancer associated fibroblasts | 3 | 0.93085 | 0.86443 to 0.94778 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | tissue | primary cancer associated fibroblasts | 3 | 1.0233 | 0.96839 to 1.0368 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | tissue | primary cancer associated fibroblasts | 3 | 0.90601 | 0.84589 to 0.9179 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.1121 | 1.0972 to 1.1189 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | tissue | primary cancer associated fibroblasts | 3 | 1.1024 | 1.0828 to 1.1091 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | tissue | primary cancer associated fibroblasts | 3 | 1.0065 | 0.971 to 1.017 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | tissue | primary cancer associated fibroblasts | 3 | 0.99925 | 0.96632 to 1.0138 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | tissue | primary cancer associated fibroblasts | 3 | 1.1092 | 1.0866 to 1.1109 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | tissue | primary cancer associated fibroblasts | 3 | 1.3372 | 1.2926 to 1.3831 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | tissue | primary cancer associated fibroblasts | 3 | 1.0606 | 1.0062 to 1.0865 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | tissue | primary cancer associated fibroblasts | 3 | 1.0012 | 0.9967 to 1.0026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | tissue | primary cancer associated fibroblasts | 3 | 1.0135 | 0.99298 to 1.0357 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | tissue | primary cancer associated fibroblasts | 3 | 0.9316 | 0.88408 to 1.0204 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | tissue | primary cancer associated fibroblasts | 2 | 0.9656 | 0.96416 to 0.96704 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | tissue | primary cancer associated fibroblasts | 3 | 1.0163 | 1.0022 to 1.0212 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | tissue | primary cancer associated fibroblasts | 3 | 1.0722 | 0.99624 to 1.1023 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | tissue | primary cancer associated fibroblasts | 1 | 0.8928 | 0.8928 to 0.8928 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | tissue | primary cancer associated fibroblasts | 3 | 1.0265 | 1.0054 to 1.096 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | tissue | primary cancer associated fibroblasts | 3 | 1.0896 | 0.96171 to 1.1713 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | tissue | primary cancer associated fibroblasts | 1 | 0.8117 | 0.8117 to 0.8117 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | aorta endothelium | unknown | healthy | 39 | 1.0576 | 0.9762 to 1.2963 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | unknown | healthy | 28 | 0.9683 | 0.94097 to 1.051 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar endothelium | unknown | healthy | 39 | 0.9736 | 0.85 to 1.0149 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | unknown | healthy | 41 | 0.9526 | 0.8738 to 1.02 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | unknown | healthy | 41 | 0.962 | 0.8954 to 1.0327 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular saphenous endothelium | unknown | healthy | 19 | 0.9397 | 0.9033 to 1.0664 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98978 | 0.95355 to 1.0577 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.97953 | 0.95508 to 1.0478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.1848 | 1.1673 to 1.2495 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0174 | 0.96739 to 1.1512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.92802 | 0.89737 to 0.97085 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.92323 | 0.8929 to 0.96519 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.96842 | 0.93856 to 1.0104 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0141 | 0.97537 to 1.0862 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.91735 | 0.87681 to 0.95567 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9124 | 0.88131 to 0.95455 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.88737 | 0.87538 to 0.95886 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.97132 | 0.93274 to 1.0425 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0455 | 1.0127 to 1.1204 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.95124 | 0.92817 to 0.99563 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9976 | 0.96915 to 1.0413 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0372 | 1.0072 to 1.1107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.99989 | 0.95862 to 1.0466 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0148 | 0.99098 to 1.0698 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.037 | 1.0107 to 1.1126 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0028 | 0.97533 to 1.0308 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.91948 | 0.88919 to 0.95554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0095 | 0.96724 to 1.0844 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0678 | 1.03 to 1.189 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0205 | 0.98929 to 1.056 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.92686 | 0.89549 to 0.95521 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0216 | 0.99861 to 1.0684 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9979 | 0.95934 to 1.0646 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.93402 | 0.90953 to 0.96888 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.91541 | 0.88771 to 0.94566 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.87199 | 0.83806 to 0.89868 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9956 | 0.97284 to 1.0587 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98261 | 0.95618 to 1.0403 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0031 | 0.97531 to 1.0548 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98672 | 0.96165 to 1.0408 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.96483 | 0.93096 to 1.0089 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98707 | 0.95135 to 1.0585 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.97083 | 0.94235 to 1.0144 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0082 | 0.9748 to 1.0489 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.91748 | 0.8894 to 0.94099 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0484 | 1.025 to 1.1169 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0117 | 0.97211 to 1.0632 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0013 | 0.96903 to 1.0573 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98439 | 0.94892 to 1.0393 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.96868 | 0.94226 to 1.0342 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.94906 | 0.91257 to 1.0131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0199 | 0.97408 to 1.0958 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0233 | 1.0029 to 1.0701 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.88438 | 0.85452 to 0.90789 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.95431 | 0.91711 to 1.0078 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.98019 | 0.95196 to 1.0241 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.89013 | 0.85359 to 0.9215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.85304 | 0.83746 to 0.89277 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0312 | 0.98804 to 1.1105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.97843 | 0.93097 to 1.0282 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.88728 | 0.8574 to 0.9242 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | heritable Pulmonary arterial hypertension (hPAH) | 18 | 0.96439 | 0.93501 to 1.0153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.97775 | 0.94888 to 1.023 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.99451 | 0.95635 to 1.0418 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.92983 | 0.91207 to 0.95536 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.001 | 0.97436 to 1.0583 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.1824 | 1.1642 to 1.2478 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0082 | 0.95963 to 1.12 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0009 | 0.9863 to 1.0117 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 2 | 1.0656 | 1.0656 to 1.0656 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | aorta endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 1.0009 | 0.9189 to 1.1244 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 18 | 1.0076 | 0.9736 to 1.0146 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | unknown | heritable Pulmonary arterial hypertension (hPAH) | 20 | 1.0124 | 0.9149 to 1.0762 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | unknown | heritable Pulmonary arterial hypertension (hPAH) | 5 | 1.1513 | 1.0622 to 1.1513 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9153 | 0.8399 to 1.0201 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9614 | 0.8454 to 0.9909 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 5 | 0.9328 | 0.9328 to 0.9763 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 21 | 0.9462 | 0.8665 to 1.0419 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.98552 | 0.94451 to 1.0494 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.98741 | 0.9406 to 1.0402 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.2096 | 1.1383 to 1.3186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0027 | 0.91056 to 1.122 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.91556 | 0.87773 to 0.96996 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.90742 | 0.8737 to 0.95886 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.95318 | 0.92001 to 1.0049 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0081 | 0.98197 to 1.0777 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.89518 | 0.85301 to 0.94635 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.90064 | 0.86072 to 0.94959 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.88179 | 0.85381 to 0.95452 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96812 | 0.93005 to 1.0319 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0569 | 1.0161 to 1.1161 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.95296 | 0.92841 to 0.99898 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.98709 | 0.96229 to 1.0395 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0405 | 1.0047 to 1.1032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99328 | 0.96288 to 1.0438 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0238 | 0.98423 to 1.0686 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0524 | 1.0082 to 1.1082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.009 | 0.96715 to 1.036 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.91871 | 0.88375 to 0.95487 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99543 | 0.96603 to 1.0723 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0722 | 1.0333 to 1.1627 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 17 | 1.0243 | 0.97877 to 1.1249 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0117 | 0.98439 to 1.0615 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.93125 | 0.89418 to 0.95664 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0319 | 0.9907 to 1.0815 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1 | 0.96114 to 1.0526 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.92122 | 0.90505 to 0.96453 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.92084 | 0.88142 to 0.9479 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.85782 | 0.80258 to 0.88818 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0041 | 0.97531 to 1.0577 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.98756 | 0.96079 to 1.036 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.9949 | 0.97248 to 1.0502 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99164 | 0.95821 to 1.0416 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96788 | 0.91342 to 1.0064 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99469 | 0.95654 to 1.0406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96516 | 0.93321 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0062 | 0.96439 to 1.0485 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.91509 | 0.88211 to 0.93393 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0569 | 1.0133 to 1.1124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99714 | 0.97919 to 1.0635 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0066 | 0.97398 to 1.0542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.95921 | 0.94449 to 1.0354 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96337 | 0.93718 to 1.0246 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.9245 | 0.90033 to 1.0047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0104 | 0.97914 to 1.0858 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0131 | 0.99111 to 1.0735 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.86938 | 0.82513 to 0.91145 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.93537 | 0.90992 to 1.0026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96609 | 0.94613 to 1.0229 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.86387 | 0.82873 to 0.91975 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.85172 | 0.81126 to 0.87819 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.0178 | 0.98794 to 1.0974 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96276 | 0.92682 to 1.0196 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.88053 | 0.85316 to 0.92071 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.96127 | 0.92581 to 1.0105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.97633 | 0.94535 to 1.0229 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.97803 | 0.94832 to 1.0374 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.92898 | 0.88333 to 0.97288 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.002 | 0.9674 to 1.0531 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 1.2073 | 1.1366 to 1.3149 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.99351 | 0.90143 to 1.0977 | no line set for this reading | [Territories.csv](Territories.csv) |
| C-Score | C-Score whole genome | whole genome | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.9988 | 0.9902 to 1.0065 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | aorta endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 25 | 1.0116 | 0.9024 to 1.1637 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | central memory th1 cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 22 | 0.9968 | 0.9112 to 1.0529 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 26 | 0.979 | 0.91925 to 1.0162 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.97225 | 0.9344 to 1.0393 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 10 | 0.926 | 0.9243 to 0.9523 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung alveolar endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 23 | 0.923 | 0.8715 to 1.0481 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.9381 | 0.8954 to 1.0515 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 28 | 0.9725 | 0.91614 to 1.0326 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular saphenous endothelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 9 | 0.9668 | 0.9112 to 1.0214 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cytotoxic cd4 t cells (cd28-) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 19 | 1.0134 | 0.9239 to 1.0571 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | unknown | healthy | 6 | 1.03 | 1.0076 to 1.0524 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | unknown | healthy | 39 | 0.9946 | 0.9202 to 1.038 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | unknown | healthy | 17 | 0.9282 | 0.8525 to 1.0279 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | unknown | heritable Pulmonary arterial hypertension (hPAH) | 3 | 1.0185 | 1.0185 to 1.0185 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 9 | 0.9973 | 0.9154 to 1.0536 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular saphenous endothelium | unknown | heritable Pulmonary arterial hypertension (hPAH) | 11 | 0.9622 | 0.8986 to 0.9766 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 8 | 1.0439 | 1.0351 to 1.0772 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 6 | 1.0177 | 1.0144 to 1.021 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic beta | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 0.9602 | 0.9602 to 0.9602 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 0.8664 | 0.8664 to 0.8664 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | unknown | healthy | 7 | 0.9688 | 0.90483 to 0.9901 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | colon macrophages | unknown | heritable Pulmonary arterial hypertension (hPAH) | 8 | 1.0463 | 0.9792 to 1.0769 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | heritable Pulmonary arterial hypertension (hPAH) | 8 | 0.95926 | 0.94045 to 0.98392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 0.9926 | 0.9926 to 0.9926 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MEP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 0.9566 | 0.9566 to 0.9566 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | MPP (bone marrow) | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 7 | 0.9885 | 0.9804 to 1.0132 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 1.0102 | 1.0102 to 1.0102 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | hepatocyte | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 5 | 0.9871 | 0.9 to 0.9871 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 3 | 1.0012 | 1.0012 to 1.0012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 5 | 0.9817 | 0.9817 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | heart fibroblasts | unknown | idiopathic Pulmonary arterial hypertension (iPAH) | 14 | 0.95666 | 0.93 to 1.0029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | hepatocyte | unknown | heritable Pulmonary arterial hypertension (hPAH) | 1 | 0.9465 | 0.9465 to 0.9465 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | unknown | heritable Pulmonary arterial hypertension (hPAH) | 3 | 0.9096 | 0.89898 to 0.9096 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | unknown | healthy | 4 | 1.0331 | 0.9216 to 1.1446 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | unknown | healthy | 2 | 0.9822 | 0.9822 to 0.9822 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | unknown | healthy | 2 | 1.1165 | 1.1165 to 1.1165 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | 2 new lgg | 21 | 0.98247 | 0.94981 to 1.0101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | 2 new lgg | 21 | 0.96265 | 0.93277 to 0.99419 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | 2 new lgg | 21 | 0.95489 | 0.91833 to 0.99622 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | 2 new lgg | 21 | 0.99208 | 0.94947 to 1.027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | 2 new lgg | 21 | 0.98442 | 0.95366 to 1.0097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | 2 new lgg | 21 | 0.99093 | 0.97571 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | 2 new lgg | 21 | 0.98457 | 0.96557 to 1.0035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | 2 new lgg | 21 | 0.98316 | 0.95316 to 1.0186 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | 2 new lgg | 21 | 0.98309 | 0.969 to 1.0013 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | 2 new lgg | 21 | 0.97848 | 0.95614 to 0.99339 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | 2 new lgg | 21 | 0.96132 | 0.93742 to 1.0012 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | 2 new lgg | 21 | 0.99175 | 0.96692 to 1.0151 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | 2 new lgg | 21 | 0.9767 | 0.94512 to 1.0074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | 2 new lgg | 21 | 0.96557 | 0.94556 to 0.98958 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | 2 new lgg | 21 | 0.9813 | 0.96663 to 1.0051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | 2 new lgg | 21 | 0.96829 | 0.93197 to 1.0097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | 2 new lgg | 21 | 0.97006 | 0.93179 to 1.0094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | 2 new lgg | 21 | 0.95484 | 0.92688 to 0.99605 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | 2 new lgg | 21 | 0.98129 | 0.95075 to 1.0215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | 2 new lgg | 21 | 0.95782 | 0.92915 to 0.99385 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | 2 new lgg | 21 | 0.97371 | 0.93439 to 1.003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | 2 new lgg | 21 | 0.96498 | 0.93319 to 0.99608 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | 2 new lgg | 21 | 0.98478 | 0.957 to 1.0198 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | 2 new lgg | 21 | 0.98094 | 0.94772 to 1.0137 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | 2 new lgg | 21 | 0.96517 | 0.94047 to 0.99997 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | 2 new lgg | 21 | 0.96896 | 0.92749 to 1.0006 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | 2 new lgg | 21 | 0.9695 | 0.93877 to 1.0113 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | 2 new lgg | 21 | 0.97382 | 0.93126 to 1.0084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | 2 new lgg | 21 | 0.98055 | 0.96871 to 1.0045 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | 2 new lgg | 21 | 0.97937 | 0.95761 to 0.99797 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | 2 new lgg | 21 | 0.98678 | 0.96863 to 1.0124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | 2 new lgg | 21 | 0.9801 | 0.96634 to 1.0064 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | 2 new lgg | 21 | 0.98629 | 0.97468 to 1.0048 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | 2 new lgg | 21 | 0.98064 | 0.96346 to 1.0055 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | 2 new lgg | 21 | 0.97694 | 0.96144 to 1.0056 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | 2 new lgg | 21 | 0.98385 | 0.9691 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | 2 new lgg | 21 | 0.98123 | 0.96269 to 1.0099 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | 2 new lgg | 21 | 0.95229 | 0.9206 to 0.9881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | 2 new lgg | 21 | 0.97809 | 0.94084 to 1.0154 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | 2 new lgg | 21 | 0.9702 | 0.9504 to 1.0016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | 2 new lgg | 21 | 0.95091 | 0.92925 to 0.98766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | 2 new lgg | 21 | 0.95869 | 0.93346 to 0.98643 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | 2 new lgg | 21 | 0.95421 | 0.91501 to 0.99696 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | 2 new lgg | 21 | 0.95902 | 0.93863 to 0.9978 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | 2 new lgg | 21 | 0.95123 | 0.92621 to 0.98986 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | 2 new lgg | 21 | 0.95072 | 0.90901 to 0.99358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | 2 new lgg | 21 | 0.96946 | 0.93015 to 1.0104 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | 2 new lgg | 21 | 0.97532 | 0.92727 to 1.0203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | 2 new lgg | 21 | 0.97466 | 0.94454 to 1.0143 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | 2 new lgg | 21 | 0.96027 | 0.93279 to 0.99284 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | 2 new lgg | 21 | 0.98366 | 0.96816 to 1.0011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | 2 new lgg | 21 | 0.98127 | 0.96529 to 1.0068 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | 2 new lgg | 21 | 0.97415 | 0.95204 to 1.0035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | 2 new lgg | 21 | 0.98396 | 0.96608 to 1.0097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | 2 new lgg | 21 | 0.98115 | 0.96455 to 1.0021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | 2 new lgg | 21 | 0.95108 | 0.93831 to 0.97779 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | 2 new lgg | 21 | 0.97114 | 0.93268 to 1.0057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | 2 new lgg | 21 | 0.95132 | 0.90995 to 0.99703 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | 2 new lgg | 21 | 0.97091 | 0.93343 to 1.0002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | 2 new lgg | 21 | 0.96471 | 0.94703 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | 2 new lgg | 21 | 0.9809 | 0.96205 to 1.0044 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | 2 new lgg | 21 | 0.98631 | 0.97291 to 1.0094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | 2 new lgg | 21 | 0.95477 | 0.93644 to 0.97032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | 2 new lgg | 21 | 0.98201 | 0.96621 to 1.007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | 2 new lgg | 21 | 0.95433 | 0.91958 to 0.99557 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | 2 new lgg | 21 | 0.99256 | 0.95517 to 1.0251 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | 2 new lgg | 21 | 0.9859 | 0.956 to 1.0154 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | 2 new lgg | 21 | 1.0031 | 0.9867 to 1.0152 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | 2 new lgg | 12 | 1.0048 | 0.95894 to 1.0955 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | 2 new lgg | 1 | 1.0784 | 1.0784 to 1.0784 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | 2 new lgg | 21 | 1.0209 | 0.9636 to 1.0975 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | 2 new lgg | 1 | 1.0172 | 1.0172 to 1.0172 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | 2 new lgg | 11 | 1.0559 | 0.9893 to 1.1227 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | 2 new lgg | 19 | 1.0362 | 0.96344 to 1.1053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | 2 new lgg | 19 | 0.9706 | 0.9328 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | 2 new lgg | 21 | 1.0336 | 0.9273 to 1.1043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | 2 new lgg | 21 | 1.0091 | 0.8896 to 1.0602 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | 2 new lgg | 21 | 0.985 | 0.9208 to 1.0189 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | 2 new lgg | 21 | 1.0133 | 0.9729 to 1.0847 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | 2 new lgg | 21 | 0.9954 | 0.9228 to 1.1071 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | 2 new lgg | 18 | 1.0162 | 0.95097 to 1.0667 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | 2 new lgg | 16 | 0.9882 | 0.94988 to 1.0152 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | 2 new lgg | 16 | 1.0021 | 0.9664 to 1.0322 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | 2 new lgg | 16 | 0.98771 | 0.97714 to 1.0083 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | 2 new lgg | 16 | 0.98373 | 0.97072 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | 2 new lgg | 16 | 0.98971 | 0.97244 to 1.0071 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | 2 new lgg | 16 | 0.97897 | 0.95208 to 1.0101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | 2 new lgg | 16 | 0.98483 | 0.96989 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | 2 new lgg | 16 | 0.98535 | 0.97153 to 1.0037 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | 2 new lgg | 16 | 0.98598 | 0.97135 to 1.0077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | 2 new lgg | 16 | 0.98813 | 0.97468 to 1.0073 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | 2 new lgg | 16 | 0.98422 | 0.95031 to 1.0158 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | 2 new lgg | 16 | 0.97671 | 0.93966 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | 2 new lgg | 16 | 0.99842 | 0.95963 to 1.0297 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | 2 new lgg | 16 | 0.9848 | 0.96548 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | 2 new lgg | 16 | 0.98776 | 0.97431 to 1.0048 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | 2 new lgg | 16 | 0.98831 | 0.97322 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | 2 new lgg | 16 | 0.98235 | 0.94604 to 1.0129 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | 2 new lgg | 16 | 0.9861 | 0.97489 to 1.0076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | 2 new lgg | 16 | 0.98611 | 0.95621 to 1.0162 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | 2 new lgg | 16 | 0.99984 | 0.958 to 1.027 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | b cells | whole blood | 2 new lgg | 3 | 1.0053 | 0.95679 to 1.0053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | 2 new lgg | 14 | 0.94025 | 0.86379 to 1.1354 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | 2 new lgg | 11 | 0.9751 | 0.9086 to 1.0497 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | 2 new lgg | 14 | 0.97605 | 0.8544 to 1.0417 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | 2 new lgg | 20 | 0.9949 | 0.93254 to 1.0821 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | 2 new lgg | 18 | 1.0078 | 0.92754 to 1.0221 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | 1 new gbm | 59 | 0.99532 | 0.97079 to 1.036 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | 1 new gbm | 59 | 0.96731 | 0.93539 to 1.0235 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | 1 new gbm | 59 | 0.95677 | 0.90714 to 1.0417 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | 1 new gbm | 59 | 1.0023 | 0.9709 to 1.0537 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | 1 new gbm | 59 | 0.99819 | 0.97278 to 1.0326 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | 1 new gbm | 59 | 0.99824 | 0.97757 to 1.0214 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | 1 new gbm | 59 | 0.99137 | 0.97142 to 1.0243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | 1 new gbm | 59 | 0.99053 | 0.96113 to 1.0433 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | 1 new gbm | 59 | 0.99024 | 0.97154 to 1.016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | 1 new gbm | 59 | 0.98112 | 0.96314 to 1.014 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | 1 new gbm | 59 | 0.97775 | 0.94296 to 1.0042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | 1 new gbm | 59 | 1.0022 | 0.97737 to 1.0357 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | 1 new gbm | 43 | 0.99708 | 0.96362 to 1.0415 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | 1 new gbm | 43 | 1.0148 | 0.99587 to 1.0625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | 1 new gbm | 59 | 0.98431 | 0.94848 to 1.038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | 1 new gbm | 59 | 0.97267 | 0.93629 to 1.0428 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | 1 new gbm | 43 | 0.99248 | 0.97453 to 1.0167 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | 1 new gbm | 43 | 0.9903 | 0.97451 to 1.0124 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | 1 new gbm | 59 | 0.98771 | 0.96661 to 1.0174 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | 1 new gbm | 44 | 0.99668 | 0.97943 to 1.0203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | 1 new gbm | 59 | 0.97993 | 0.93603 to 1.0436 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | 1 new gbm | 59 | 0.96952 | 0.93527 to 1.0228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | 1 new gbm | 59 | 0.96527 | 0.91875 to 1.0222 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | 1 new gbm | 59 | 0.99144 | 0.95042 to 1.051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | 1 new gbm | 59 | 0.96927 | 0.92532 to 1.0385 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | 1 new gbm | 59 | 0.97517 | 0.93741 to 1.055 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | 1 new gbm | 43 | 0.98759 | 0.95918 to 1.029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | 1 new gbm | 59 | 0.97079 | 0.93857 to 1.0482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | 1 new gbm | 59 | 0.99815 | 0.96335 to 1.0406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | 1 new gbm | 55 | 0.99303 | 0.95614 to 1.0433 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | 1 new gbm | 59 | 0.97028 | 0.94803 to 1.0181 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | 1 new gbm | 59 | 0.96988 | 0.93179 to 1.061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | 1 new gbm | 59 | 0.97315 | 0.93091 to 1.0382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | 1 new gbm | 59 | 0.98003 | 0.93854 to 1.0565 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | 1 new gbm | 55 | 0.98791 | 0.9675 to 1.0162 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | 1 new gbm | 59 | 0.9845 | 0.95964 to 1.034 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | 1 new gbm | 59 | 0.99234 | 0.966 to 1.0332 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | 1 new gbm | 43 | 0.98975 | 0.9702 to 1.0175 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | 1 new gbm | 43 | 0.98929 | 0.97417 to 1.013 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | 1 new gbm | 43 | 0.98993 | 0.97337 to 1.0164 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | 1 new gbm | 43 | 0.9908 | 0.97609 to 1.0155 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | 1 new gbm | 55 | 0.98925 | 0.96654 to 1.0203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | 1 new gbm | 59 | 0.99436 | 0.9775 to 1.0167 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | 1 new gbm | 59 | 0.99033 | 0.96891 to 1.0281 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | 1 new gbm | 43 | 0.99616 | 0.96702 to 1.0397 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | 1 new gbm | 59 | 0.98663 | 0.96213 to 1.0208 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | 1 new gbm | 59 | 0.99284 | 0.97086 to 1.0225 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | 1 new gbm | 59 | 0.98953 | 0.96673 to 1.0245 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | 1 new gbm | 59 | 0.96076 | 0.92536 to 0.99088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | 1 new gbm | 43 | 0.98703 | 0.95413 to 1.0384 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | 1 new gbm | 59 | 0.9845 | 0.94188 to 1.0554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | 1 new gbm | 59 | 0.9851 | 0.95093 to 1.0206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | 1 new gbm | 59 | 0.96341 | 0.91632 to 1.0246 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | 1 new gbm | 59 | 0.96385 | 0.9309 to 1.0375 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | 1 new gbm | 59 | 0.96528 | 0.9151 to 1.0488 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | 1 new gbm | 59 | 0.96323 | 0.94393 to 1.0112 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | 1 new gbm | 59 | 0.96031 | 0.92376 to 1.0204 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | 1 new gbm | 59 | 0.95643 | 0.91019 to 1.0326 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | 1 new gbm | 59 | 0.97062 | 0.92946 to 1.033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | 1 new gbm | 59 | 0.97517 | 0.93174 to 1.0657 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | 1 new gbm | 59 | 0.98508 | 0.95009 to 1.0416 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | 1 new gbm | 43 | 1.0142 | 0.99187 to 1.0594 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | 1 new gbm | 59 | 0.96836 | 0.93836 to 1.0268 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | 1 new gbm | 43 | 0.99151 | 0.96602 to 1.0233 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | 1 new gbm | 59 | 0.99048 | 0.97415 to 1.0123 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | 1 new gbm | 55 | 0.98999 | 0.9684 to 1.0227 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | 1 new gbm | 59 | 0.98331 | 0.95731 to 1.03 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | 1 new gbm | 59 | 0.99072 | 0.97057 to 1.0233 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | 1 new gbm | 59 | 0.98798 | 0.96989 to 1.0084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | 1 new gbm | 43 | 0.99237 | 0.97768 to 1.0168 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | 1 new gbm | 59 | 0.9593 | 0.93451 to 0.99865 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | 1 new gbm | 59 | 0.98086 | 0.94323 to 1.0581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | 1 new gbm | 43 | 0.99358 | 0.97637 to 1.0213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | 1 new gbm | 43 | 0.99537 | 0.95717 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | 1 new gbm | 59 | 0.96283 | 0.91317 to 1.0442 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | 1 new gbm | 59 | 0.97122 | 0.93474 to 1.052 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | 1 new gbm | 57 | 0.97901 | 0.93836 to 1.0174 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | 1 new gbm | 59 | 0.99026 | 0.96977 to 1.022 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | 1 new gbm | 43 | 0.99288 | 0.97888 to 1.0131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | 1 new gbm | 57 | 0.99009 | 0.97127 to 1.0213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | 1 new gbm | 43 | 0.99695 | 0.96798 to 1.0399 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | 1 new gbm | 59 | 0.97049 | 0.94743 to 0.99606 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | 1 new gbm | 43 | 1.0121 | 0.99324 to 1.0635 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | 1 new gbm | 59 | 0.99253 | 0.97092 to 1.029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | 1 new gbm | 59 | 0.95819 | 0.90863 to 1.0417 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | 1 new gbm | 59 | 1.0009 | 0.97353 to 1.0472 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | 1 new gbm | 59 | 1.005 | 0.97473 to 1.0316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | 1 new gbm | 59 | 1.0054 | 0.99724 to 1.0124 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | 1 new gbm | 22 | 1.0373 | 0.95517 to 1.1488 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | 1 new gbm | 51 | 0.9759 | 0.90975 to 1.0859 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | 1 new gbm | 59 | 1.007 | 0.92198 to 1.0813 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | 1 new gbm | 41 | 1.0177 | 0.9345 to 1.1038 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | 1 new gbm | 35 | 0.971 | 0.8874 to 1.0507 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | 1 new gbm | 47 | 0.9954 | 0.88475 to 1.0541 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | 1 new gbm | 49 | 0.9999 | 0.94354 to 1.0754 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | 1 new gbm | 53 | 1.0041 | 0.92536 to 1.0972 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | 1 new gbm | 58 | 1.016 | 0.94342 to 1.0781 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | 1 new gbm | 58 | 1.0359 | 0.95192 to 1.1104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | 1 new gbm | 32 | 1.001 | 0.93749 to 1.1176 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | 1 new gbm | 56 | 0.9764 | 0.93607 to 1.0535 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | 1 new gbm | 59 | 1.0248 | 0.94296 to 1.1145 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | 1 new gbm | 56 | 1.0006 | 0.9248 to 1.0777 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | 1 new gbm | 58 | 1.0194 | 0.93667 to 1.0889 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | 1 new gbm | 27 | 0.9846 | 0.86321 to 1.0677 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | 1 new gbm | 19 | 1.0429 | 0.9466 to 1.0853 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | 1 new gbm | 41 | 1.0108 | 0.9587 to 1.1094 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | 2 new lgg | 6 | 1.0476 | 1.0341 to 1.0787 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | 2 new lgg | 12 | 1.0213 | 0.9703 to 1.1518 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | 4 rec lgg, now gbm | 16 | 0.99315 | 0.94358 to 1.0033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | 4 rec lgg, now gbm | 16 | 0.97266 | 0.92557 to 0.99403 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | 4 rec lgg, now gbm | 16 | 0.965 | 0.91543 to 1.0049 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | 4 rec lgg, now gbm | 16 | 1.0028 | 0.93905 to 1.0108 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | 4 rec lgg, now gbm | 16 | 0.99537 | 0.95459 to 1.0045 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.99446 | 0.97254 to 1.0065 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.99003 | 0.96389 to 1.0016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.99409 | 0.94912 to 1.0077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.98742 | 0.96412 to 0.99751 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.98223 | 0.95255 to 0.9903 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | 4 rec lgg, now gbm | 16 | 0.96462 | 0.93211 to 1.0072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | 4 rec lgg, now gbm | 16 | 1.0004 | 0.96788 to 1.0248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | 4 rec lgg, now gbm | 16 | 0.98507 | 0.94165 to 0.99522 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | 4 rec lgg, now gbm | 16 | 0.97245 | 0.93073 to 0.99096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.9884 | 0.95507 to 0.99916 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.98211 | 0.93133 to 0.99902 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | 4 rec lgg, now gbm | 16 | 0.97645 | 0.9096 to 1.0034 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.96879 | 0.91158 to 0.99237 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.99486 | 0.9456 to 1.0081 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.96806 | 0.92291 to 0.98771 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | 4 rec lgg, now gbm | 16 | 0.97816 | 0.93173 to 1.0007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.97637 | 0.93071 to 0.99403 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | 4 rec lgg, now gbm | 16 | 0.9945 | 0.94228 to 1.014 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | 4 rec lgg, now gbm | 16 | 0.97698 | 0.93044 to 0.99044 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | 4 rec lgg, now gbm | 16 | 0.97518 | 0.92242 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.98449 | 0.92546 to 1.011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | 4 rec lgg, now gbm | 16 | 0.9833 | 0.93075 to 1.0089 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | 4 rec lgg, now gbm | 16 | 0.98028 | 0.96138 to 0.99949 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | 4 rec lgg, now gbm | 16 | 0.99435 | 0.96465 to 1.0203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | 4 rec lgg, now gbm | 16 | 0.99267 | 0.97388 to 1.0028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99085 | 0.95981 to 1.0086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | 4 rec lgg, now gbm | 16 | 0.98723 | 0.95263 to 1.0041 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99229 | 0.96648 to 1.0091 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | 4 rec lgg, now gbm | 16 | 0.9903 | 0.9584 to 1.0075 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.95111 | 0.90369 to 0.99174 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99057 | 0.93165 to 1.0163 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | 4 rec lgg, now gbm | 16 | 0.97879 | 0.93807 to 0.99291 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | 4 rec lgg, now gbm | 16 | 0.95944 | 0.90675 to 0.98203 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.9653 | 0.92818 to 0.99212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | 4 rec lgg, now gbm | 16 | 0.97091 | 0.911 to 0.99638 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | 4 rec lgg, now gbm | 16 | 0.97235 | 0.92714 to 0.98491 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | 4 rec lgg, now gbm | 16 | 0.96083 | 0.90749 to 0.97866 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.96436 | 0.8968 to 0.98795 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.98142 | 0.91754 to 1.0032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | 4 rec lgg, now gbm | 16 | 0.98646 | 0.91609 to 1.0077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | 4 rec lgg, now gbm | 16 | 0.99201 | 0.9411 to 1.0035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | 4 rec lgg, now gbm | 16 | 0.97237 | 0.92968 to 0.99092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | 4 rec lgg, now gbm | 16 | 0.9895 | 0.96293 to 0.99591 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | 4 rec lgg, now gbm | 16 | 0.98496 | 0.94713 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | 4 rec lgg, now gbm | 16 | 0.9896 | 0.95541 to 1.0075 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | 4 rec lgg, now gbm | 16 | 0.98717 | 0.95869 to 0.9927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.96054 | 0.92905 to 0.97363 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.98389 | 0.93428 to 1.0051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | 4 rec lgg, now gbm | 16 | 0.96517 | 0.89469 to 1.0006 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | 4 rec lgg, now gbm | 16 | 0.97561 | 0.92534 to 1.0022 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | 4 rec lgg, now gbm | 16 | 0.97778 | 0.93447 to 0.99495 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99051 | 0.95939 to 1.0099 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99391 | 0.96817 to 1.0105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | 4 rec lgg, now gbm | 16 | 0.96171 | 0.94221 to 0.98816 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | 4 rec lgg, now gbm | 16 | 0.99135 | 0.95971 to 1.0107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | 4 rec lgg, now gbm | 16 | 0.96539 | 0.91713 to 1.0039 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | 4 rec lgg, now gbm | 16 | 1.003 | 0.94512 to 1.0105 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | 4 rec lgg, now gbm | 16 | 0.9826 | 0.97108 to 1.007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | 4 rec lgg, now gbm | 16 | 1.0043 | 0.99733 to 1.0129 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | 4 rec lgg, now gbm | 10 | 1.0351 | 0.9168 to 1.0818 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | 4 rec lgg, now gbm | 12 | 0.96195 | 0.9246 to 1.0802 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | 4 rec lgg, now gbm | 16 | 1.0154 | 0.9297 to 1.1597 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | 4 rec lgg, now gbm | 6 | 0.9348 | 0.9007 to 1.0163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | 4 rec lgg, now gbm | 11 | 1.0376 | 0.9449 to 1.1317 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | 4 rec lgg, now gbm | 14 | 1.0089 | 0.8933 to 1.0648 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.9852 | 0.9498 to 1.0351 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | 4 rec lgg, now gbm | 15 | 0.9931 | 0.9348 to 1.0627 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | 4 rec lgg, now gbm | 16 | 0.9789 | 0.9075 to 1.1003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | 4 rec lgg, now gbm | 16 | 1.0961 | 1.0061 to 1.1455 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | 4 rec lgg, now gbm | 11 | 1.0096 | 0.9541 to 1.0634 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | 4 rec lgg, now gbm | 16 | 1.0242 | 0.9702 to 1.161 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | 4 rec lgg, now gbm | 16 | 0.9822 | 0.91355 to 1.1002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | 4 rec lgg, now gbm | 16 | 0.9706 | 0.93475 to 1.1514 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | 4 rec lgg, now gbm | 16 | 1.0341 | 0.9418 to 1.1004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | 4 rec lgg, now gbm | 14 | 0.98507 | 0.94001 to 1.0039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | 4 rec lgg, now gbm | 14 | 0.986 | 0.9568 to 0.99366 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | 4 rec lgg, now gbm | 14 | 0.99181 | 0.95874 to 1.0016 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | 4 rec lgg, now gbm | 14 | 0.98955 | 0.9599 to 0.99873 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | 4 rec lgg, now gbm | 14 | 0.9596 | 0.93181 to 1.0346 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | 4 rec lgg, now gbm | 4 | 1.0232 | 1.0034 to 1.0568 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | CMP (bone marrow) | whole blood | 1 new gbm | 2 | 1.0155 | 1.0155 to 1.0155 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | L-MPP (bone marrow) | whole blood | 1 new gbm | 2 | 1.0103 | 1.0103 to 1.0103 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | whole blood | 1 new gbm | 5 | 0.9012 | 0.9012 to 1.0239 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | whole blood | 1 new gbm | 3 | 0.983 | 0.983 to 0.983 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | 1 new gbm | 3 | 1.0734 | 0.97728 to 1.0734 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | 1 new gbm | 3 | 0.9856 | 0.9289 to 0.9856 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | breast basal epithelium | whole blood | 1 new gbm | 4 | 1.0207 | 0.94403 to 1.0461 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99768 | 0.97481 to 1.0096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | 4 rec lgg, now gbm | 8 | 1.0151 | 0.99939 to 1.0246 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99401 | 0.98555 to 1.0028 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99347 | 0.98309 to 0.99631 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | 4 rec lgg, now gbm | 10 | 0.99427 | 0.97058 to 1.0063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | 4 rec lgg, now gbm | 8 | 0.9893 | 0.97222 to 0.99811 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99292 | 0.98279 to 0.99737 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.9935 | 0.98318 to 0.99714 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99352 | 0.9845 to 0.99878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99486 | 0.98563 to 0.99904 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99661 | 0.97682 to 1.0054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | 4 rec lgg, now gbm | 8 | 0.98808 | 0.96565 to 1.0011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | 4 rec lgg, now gbm | 8 | 1.0126 | 0.99405 to 1.0247 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99553 | 0.97797 to 0.99846 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.9962 | 0.98816 to 0.99986 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | 4 rec lgg, now gbm | 8 | 0.9973 | 0.98706 to 1.004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | 4 rec lgg, now gbm | 8 | 0.99876 | 0.97063 to 1.0086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99445 | 0.98611 to 0.99927 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | 4 rec lgg, now gbm | 8 | 0.99886 | 0.97674 to 1.0066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | 4 rec lgg, now gbm | 8 | 1.012 | 0.99161 to 1.0225 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | 4 rec lgg, now gbm | 7 | 0.9918 | 0.94801 to 1.0095 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | whole blood | 4 rec lgg, now gbm | 2 | 1.0963 | 1.0963 to 1.0963 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | whole blood | 3 rec lgg, still lgg | 18 | 0.97983 | 0.94678 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | whole blood | 3 rec lgg, still lgg | 18 | 0.94628 | 0.92187 to 1.0117 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | whole blood | 3 rec lgg, still lgg | 18 | 0.93289 | 0.90016 to 1.0316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | whole blood | 3 rec lgg, still lgg | 18 | 0.99041 | 0.94741 to 1.021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | whole blood | 3 rec lgg, still lgg | 18 | 0.98577 | 0.95493 to 1.0103 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | CMP (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.98891 | 0.97249 to 1.0029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | GMP (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.98335 | 0.96152 to 0.99881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | HSC (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.97296 | 0.94865 to 1.0236 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | L-MPP (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.98057 | 0.96429 to 0.99461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MEP (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.97015 | 0.95281 to 0.99642 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | MPP (bone marrow) | whole blood | 3 rec lgg, still lgg | 18 | 0.97458 | 0.94223 to 1.0131 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | adipocytes | whole blood | 3 rec lgg, still lgg | 18 | 0.98476 | 0.96374 to 1.0242 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | aorta endothelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97466 | 0.94283 to 1.011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | astrocytes | whole blood | 3 rec lgg, still lgg | 12 | 0.99936 | 0.95613 to 1.022 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | b cells | whole blood | 3 rec lgg, still lgg | 18 | 0.96222 | 0.9371 to 1.0133 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | basophils | whole blood | 3 rec lgg, still lgg | 18 | 0.94824 | 0.92422 to 1.0047 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | bladder epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97644 | 0.96575 to 0.99721 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast basal epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97688 | 0.96772 to 0.99341 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | breast luminal epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.97622 | 0.96017 to 1.0007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cardiomyocyte | whole blood | 3 rec lgg, still lgg | 12 | 0.98347 | 0.976 to 1.001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd4 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.95706 | 0.92122 to 1.0247 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd56bright nk cells | whole blood | 3 rec lgg, still lgg | 18 | 0.94715 | 0.92604 to 1.0238 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 central memory t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9406 | 0.91527 to 1.0139 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.96735 | 0.94174 to 1.0302 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cd8 temra t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9443 | 0.92209 to 1.0087 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | central memory th1 cells | whole blood | 3 rec lgg, still lgg | 18 | 0.94987 | 0.92788 to 1.0251 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon enteroendocrine | whole blood | 3 rec lgg, still lgg | 12 | 0.96605 | 0.94085 to 0.99459 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.94867 | 0.92467 to 1.0163 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon fibroblasts | whole blood | 3 rec lgg, still lgg | 18 | 0.97203 | 0.95381 to 1.0234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | colon macrophages | whole blood | 3 rec lgg, still lgg | 16 | 0.96519 | 0.93995 to 1.0044 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cortical neurons | whole blood | 3 rec lgg, still lgg | 18 | 0.95605 | 0.93397 to 1.0099 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | cytotoxic cd4 t cells (cd28-) | whole blood | 3 rec lgg, still lgg | 18 | 0.94692 | 0.91693 to 1.0279 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | effector memory cd8 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.95017 | 0.93012 to 1.0323 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | embryonic stem cells | whole blood | 3 rec lgg, still lgg | 18 | 0.95362 | 0.9227 to 1.031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | endometrium epithelium | whole blood | 3 rec lgg, still lgg | 16 | 0.97143 | 0.95842 to 0.99485 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | eosinophils | whole blood | 3 rec lgg, still lgg | 18 | 0.96924 | 0.94464 to 1.0078 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | erythrocyte progenitors | whole blood | 3 rec lgg, still lgg | 18 | 0.97532 | 0.95417 to 1.0153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | fallopien tubes epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97189 | 0.9624 to 0.99542 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric antrum epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97458 | 0.96535 to 0.99555 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric body epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97401 | 0.96439 to 0.99304 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | gastric fundus epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97818 | 0.96979 to 0.99639 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | heart fibroblasts | whole blood | 3 rec lgg, still lgg | 16 | 0.97068 | 0.9616 to 1.0005 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | hepatocyte | whole blood | 3 rec lgg, still lgg | 18 | 0.98189 | 0.97261 to 1.006 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular endothelium | whole blood | 3 rec lgg, still lgg | 18 | 0.97569 | 0.96108 to 1.0061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney glomerular epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97194 | 0.94426 to 1.0064 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular endothelium | whole blood | 3 rec lgg, still lgg | 18 | 0.97086 | 0.95616 to 1.0011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | kidney tubular epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.98051 | 0.96323 to 1.0052 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar endothelium | whole blood | 3 rec lgg, still lgg | 18 | 0.9766 | 0.96006 to 1.0079 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.95612 | 0.92319 to 0.99348 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung alveolar macrophages | whole blood | 3 rec lgg, still lgg | 12 | 0.96213 | 0.93147 to 0.99792 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung bronchus epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.95325 | 0.93431 to 1.0358 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | lung interstitial macrophages | whole blood | 3 rec lgg, still lgg | 18 | 0.95918 | 0.93956 to 0.99857 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory b cells | whole blood | 3 rec lgg, still lgg | 18 | 0.93318 | 0.91312 to 1.0088 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | memory cd4 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.94375 | 0.92356 to 1.0076 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | microglia | whole blood | 3 rec lgg, still lgg | 18 | 0.93403 | 0.90725 to 1.0229 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | monocytes | whole blood | 3 rec lgg, still lgg | 18 | 0.95532 | 0.92382 to 1.0018 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive b cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9376 | 0.9152 to 1.0015 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd4 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.92912 | 0.89747 to 1.0105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | naive cd8 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9491 | 0.92215 to 1.0275 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | neutrophils | whole blood | 3 rec lgg, still lgg | 18 | 0.95053 | 0.91484 to 1.0332 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | nk cells | whole blood | 3 rec lgg, still lgg | 18 | 0.96447 | 0.93321 to 1.0227 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocyte precursors | whole blood | 3 rec lgg, still lgg | 12 | 0.99785 | 0.95604 to 1.0182 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | oligodendrocytes | whole blood | 3 rec lgg, still lgg | 18 | 0.95303 | 0.9242 to 1.0121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreas endothelium | whole blood | 3 rec lgg, still lgg | 12 | 0.96842 | 0.95714 to 0.99554 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic acinar | whole blood | 3 rec lgg, still lgg | 18 | 0.9785 | 0.96326 to 0.99826 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic alpha | whole blood | 3 rec lgg, still lgg | 16 | 0.97345 | 0.96133 to 0.99801 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic beta | whole blood | 3 rec lgg, still lgg | 18 | 0.96433 | 0.94909 to 1.0092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic delta | whole blood | 3 rec lgg, still lgg | 18 | 0.9771 | 0.95956 to 1.0074 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | pancreatic duct | whole blood | 3 rec lgg, still lgg | 18 | 0.97747 | 0.9603 to 0.99364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | prostate epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97806 | 0.96593 to 0.99787 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | regulatory t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9424 | 0.93267 to 0.97748 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | small intestine epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.95257 | 0.93213 to 1.0241 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | smooth muscle | whole blood | 3 rec lgg, still lgg | 12 | 0.97635 | 0.96752 to 0.99929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | striated muscle | whole blood | 3 rec lgg, still lgg | 12 | 0.96215 | 0.94063 to 1.0038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t central memory cd4 | whole blood | 3 rec lgg, still lgg | 18 | 0.93211 | 0.90109 to 1.0254 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector cell cd8 | whole blood | 3 rec lgg, still lgg | 18 | 0.94718 | 0.92267 to 1.0232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | t effector memory cd4 | whole blood | 3 rec lgg, still lgg | 16 | 0.9521 | 0.93822 to 0.9915 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | thyroid epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.97459 | 0.95993 to 1.0063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tongue epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97798 | 0.96837 to 0.99707 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil palatine epithelium | whole blood | 3 rec lgg, still lgg | 18 | 0.98116 | 0.96426 to 1.006 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | tonsil pharyngeal epithelium | whole blood | 3 rec lgg, still lgg | 12 | 0.97324 | 0.94952 to 1.0059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular endothelium | whole blood | 3 rec lgg, still lgg | 18 | 0.95497 | 0.93827 to 0.97481 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular leptomeningeal cells | whole blood | 3 rec lgg, still lgg | 12 | 0.9938 | 0.95405 to 1.0255 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A each cell | vascular saphenous endothelium | whole blood | 3 rec lgg, still lgg | 18 | 0.97686 | 0.96296 to 1.0081 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | whole blood | 3 rec lgg, still lgg | 18 | 0.93399 | 0.90203 to 1.0303 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | whole blood | 3 rec lgg, still lgg | 18 | 0.99264 | 0.95205 to 1.0173 | no line set for this reading | [Territories.csv](Territories.csv) |
| Met-A | Met-A | neutrophils | whole blood | 3 rec lgg, still lgg | 18 | 0.98425 | 0.96989 to 1.0084 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | whole blood | 3 rec lgg, still lgg | 18 | 1.0021 | 0.99649 to 1.01 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | basophils | whole blood | 3 rec lgg, still lgg | 18 | 0.9858 | 0.91422 to 1.1023 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd56bright nk cells | whole blood | 3 rec lgg, still lgg | 18 | 0.9976 | 0.93007 to 1.0524 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 central memory t cells | whole blood | 3 rec lgg, still lgg | 4 | 0.93555 | 0.9076 to 0.9635 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | whole blood | 3 rec lgg, still lgg | 10 | 1.0083 | 0.96793 to 1.0756 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | eosinophils | whole blood | 3 rec lgg, still lgg | 17 | 0.9929 | 0.9054 to 1.0543 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | whole blood | 3 rec lgg, still lgg | 10 | 1.0469 | 0.9512 to 1.099 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory cd4 t cells | whole blood | 3 rec lgg, still lgg | 18 | 0.98025 | 0.8731 to 1.1142 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | microglia | whole blood | 3 rec lgg, still lgg | 18 | 0.9871 | 0.92259 to 1.0753 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | whole blood | 3 rec lgg, still lgg | 18 | 1.0235 | 0.89831 to 1.1689 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | whole blood | 3 rec lgg, still lgg | 15 | 1.0396 | 0.9686 to 1.2195 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | whole blood | 3 rec lgg, still lgg | 13 | 1.0045 | 0.95894 to 1.1009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | whole blood | 3 rec lgg, still lgg | 15 | 1.0114 | 0.9435 to 1.0572 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | whole blood | 3 rec lgg, still lgg | 18 | 0.99255 | 0.9146 to 1.0549 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | regulatory t cells | whole blood | 3 rec lgg, still lgg | 17 | 1.0192 | 0.92596 to 1.0785 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular endothelium | whole blood | 3 rec lgg, still lgg | 17 | 1.0364 | 0.8807 to 1.1501 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | small intestine epithelium | whole blood | 1 new gbm | 2 | 1.0833 | 1.0833 to 1.0833 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | 4 rec lgg, now gbm | 1 | 1.0383 | 1.0383 to 1.0383 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | 4 rec lgg, now gbm | 3 | 0.9693 | 0.9693 to 1.0421 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | HSC (bone marrow) | whole blood | 3 rec lgg, still lgg | 8 | 1.1116 | 0.9817 to 1.145 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | effector memory cd8 t cells | whole blood | 3 rec lgg, still lgg | 8 | 0.9578 | 0.9048 to 1.0468 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | whole blood | 3 rec lgg, still lgg | 6 | 1.0156 | 0.9865 to 1.0759 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | whole blood | 3 rec lgg, still lgg | 9 | 0.9955 | 0.96998 to 1.0554 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t central memory cd4 | whole blood | 1 new gbm | 2 | 0.9675 | 0.9675 to 0.9675 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tongue epithelium | whole blood | 1 new gbm | 1 | 0.9567 | 0.9567 to 0.9567 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | whole blood | 3 rec lgg, still lgg | 1 | 1.1375 | 1.1375 to 1.1375 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | whole blood | 3 rec lgg, still lgg | 2 | 0.919 | 0.919 to 0.919 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung bronchus epithelium | whole blood | 3 rec lgg, still lgg | 3 | 0.9988 | 0.98521 to 0.9988 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | adipocytes | whole blood | 3 rec lgg, still lgg | 1 | 0.8714 | 0.8714 to 0.8714 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | whole blood | 3 rec lgg, still lgg | 1 | 1.1054 | 1.1054 to 1.1054 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | whole blood | 3 rec lgg, still lgg | 1 | 0.9828 | 0.9828 to 0.9828 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | whole blood | 3 rec lgg, still lgg | 1 | 1.1153 | 1.1153 to 1.1153 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | whole blood | 1 new gbm | 2 | 0.9033 | 0.9033 to 0.9033 | no line set for this reading | [C_Score.csv](C_Score.csv) |

150771 readings from 34 sources. Status and record for every row are in the tables.

**Where Normal begins and ends:** [Bands.csv](Bands.csv) lists, for each instrument, reading, cell and platform, the band the gauge drew
(copied from the bundles) beside the spread of the healthy readings themselves (2.5th to 97.5th percentile, with how many healthy samples).
