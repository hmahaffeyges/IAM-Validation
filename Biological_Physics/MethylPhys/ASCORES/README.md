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

4815 readings from 26 sources. Status and record for every row are in the tables.

**Where Normal begins and ends:** [Bands.csv](Bands.csv) lists, for each instrument, reading, cell and platform, the band the gauge drew
(copied from the bundles) beside the spread of the healthy readings themselves (2.5th to 97.5th percentile, with how many healthy samples).
