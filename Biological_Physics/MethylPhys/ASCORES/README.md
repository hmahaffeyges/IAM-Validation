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
| Met-A | A, untared | isolated neutrophils | healthy | 6 | 1.001 | 0.9943 to 1.006 | 0 of 6 (0 %) | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | healthy | 13 | 1.003 | 0.919 to 1.062 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | purified neutrophils | patient (see record) | 16 | 0.9641 | 0.9163 to 0.9929 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | patient (see record) | 38 | 1.008 | 0.9772 to 1.035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | A, tared to the same laboratory | granulocytes | healthy | 26 | 1.001 | 0.979 to 1.026 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | breast tissue | tumor | 53 | 1.065 | 0.9781 to 1.224 | 23 of 53 (43 %) above 1.094 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | colon tissue | tumor | 88 | 1.066 | 0.9804 to 1.166 | 28 of 88 (32 %) above 1.092 | [Met_A.csv](Met_A.csv) |
| Met-A | R, tumor over the same patient's normal tissue | liver tissue | tumor | 50 | 1.104 | 0.9364 to 1.326 | 22 of 50 (44 %) above 1.125 | [Met_A.csv](Met_A.csv) |
| C-Score | C_rel, tared to the series | whole blood | healthy | 553 | 1 | 0.7543 to 1.332 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | isolated neutrophils | healthy | 62 | 1.003 | 0.8452 to 1.219 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, chain tare | whole blood | healthy | 231 | 0.9967 | 0.9029 to 1.103 | 3 of 231 (1 %) | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | patient (see record) | 38 | 1.044 | 0.7694 to 1.419 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C_rel, tared to the series | granulocytes | healthy | 26 | 1.031 | 0.7404 to 1.275 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| IAM-A | held error eps, per interior opportunity (commissioned counter) | purified cell types (56) | healthy | 56 | 0.0447 | 0.03546 to 0.05224 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | normal tissue of a cancer patient | 5 | 0.03514 | 0.0327 to 0.03803 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | normal tissue of a cancer patient | 8 | 0.03891 | 0.0358 to 0.04013 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | colon tissue | tumor | 5 | 0.04216 | 0.0369 to 0.04588 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | held error eps, earlier counter (end CpGs counted), conversion-corrected | oral tissue | tumor | 8 | 0.04093 | 0.03451 to 0.04639 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | normal tissue of a cancer patient | 6 | 0.01383 | 0.01037 to 0.01871 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| IAM-A | island gain error (PROC-CHANNEL-01 rule) | colon tissue | tumor | 6 | 0.01714 | 0.01022 to 0.02194 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, arrays | whole blood | healthy | 100 | 2.395 | 2.247 to 2.505 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, islands territory, arrays | whole blood | healthy | 100 | 3.724 | 3.507 to 3.877 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, domains territory, arrays | whole blood | healthy | 100 | 2.186 | 2.069 to 2.271 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | not stated | leukemia | 5 | 2.195 | 2.172 to 2.302 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | cell line | control line | 1 | 2.006 | 2.006 to 2.006 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | cell line | knockout | 1 | 1.998 | 1.998 to 1.998 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Ceiling | A at the ceiling, Met-A whole sample, sequencing | sorted b cells | cord-blood B cells | 3 | 2.383 | 2.381 to 2.384 | no line set for this reading | [Ceiling.csv](Ceiling.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | leukemia | 3 | 0.6738 | 0.6433 to 0.7032 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | leukemia | 3 | 0.6271 | 0.5846 to 0.6811 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | leukemia | 3 | 0.5064 | 0.3577 to 0.5329 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | leukemia | 3 | 0.7177 | 0.711 to 0.736 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | leukemia | 3 | 0.6784 | 0.6486 to 0.7063 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | leukemia | 3 | 0.2368 | 0.2323 to 0.2717 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | leukemia | 3 | 0.515 | 0.4664 to 0.5869 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | leukemia | 3 | 0.8962 | 0.8877 to 0.9107 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | leukemia | 3 | 0.997 | 0.9916 to 0.9985 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | plasma | leukemia | 3 | 1.006 | 0.9728 to 1.018 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | plasma | leukemia | 3 | 1.029 | 0.9925 to 1.051 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney glomerular endothelium | plasma | leukemia | 1 | 1.068 | 1.068 to 1.068 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | leukemia | 3 | 1.025 | 0.962 to 1.066 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | plasma | leukemia | 1 | 1.067 | 1.067 to 1.067 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | plasma | leukemia | 3 | 1.006 | 0.9598 to 1.086 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | leukemia | 2 | 0.9565 | 0.9509 to 0.962 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | plasma | leukemia | 2 | 1.002 | 0.9865 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | plasma | leukemia | 2 | 1.037 | 0.9994 to 1.074 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | plasma | leukemia | 1 | 1.069 | 1.069 to 1.069 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | plasma | leukemia | 1 | 0.9918 | 0.9918 to 0.9918 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | not stated | leukemia | 28 | 0.7439 | 0.6141 to 0.8331 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | not stated | leukemia | 28 | 0.6631 | 0.5924 to 0.7365 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | not stated | leukemia | 28 | 0.3536 | 0.2794 to 0.561 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | not stated | leukemia | 28 | 0.8183 | 0.6584 to 0.9306 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | not stated | leukemia | 28 | 0.7534 | 0.6241 to 0.85 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | not stated | leukemia | 28 | 0.2431 | 0.1964 to 0.3877 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | not stated | leukemia | 28 | 0.6311 | 0.4689 to 0.7421 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (not stated) | not stated | leukemia | 28 | 0.903 | 0.8196 to 0.9638 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | not stated | leukemia | 28 | 0.9972 | 0.9916 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | not stated | leukemia | 28 | 1.005 | 0.9217 to 1.063 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | not stated | leukemia | 28 | 1.007 | 0.9159 to 1.095 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | not stated | leukemia | 21 | 0.9968 | 0.9005 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | not stated | leukemia | 5 | 1.009 | 0.9384 to 1.102 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocyte precursors | not stated | leukemia | 1 | 0.9446 | 0.9446 to 0.9446 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreas endothelium | not stated | leukemia | 10 | 1.001 | 0.9217 to 1.137 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | not stated | leukemia | 10 | 0.9949 | 0.9003 to 1.061 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | not stated | leukemia | 21 | 1 | 0.9425 to 1.059 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon fibroblasts | not stated | leukemia | 7 | 1.007 | 0.9231 to 1.056 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | not stated | leukemia | 20 | 1.019 | 0.948 to 1.163 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd4 t cells | not stated | leukemia | 1 | 1.027 | 1.027 to 1.027 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | not stated | leukemia | 17 | 0.9855 | 0.8926 to 1.042 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | not stated | leukemia | 1 | 1.105 | 1.105 to 1.105 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | endometrium epithelium | not stated | leukemia | 1 | 1.017 | 1.017 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | not stated | leukemia | 2 | 0.9702 | 0.9408 to 0.9996 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | tongue epithelium | not stated | leukemia | 1 | 0.9993 | 0.9993 to 0.9993 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | not stated | leukemia | 5 | 0.9749 | 0.9153 to 1.015 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | control line | 1 | 0.5176 | 0.5176 to 0.5176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | control line | 1 | 0.5461 | 0.5461 to 0.5461 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | control line | 1 | 0.5516 | 0.5516 to 0.5516 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | control line | 1 | 0.5188 | 0.5188 to 0.5188 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | control line | 1 | 0.5119 | 0.5119 to 0.5119 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | control line | 1 | 0.4234 | 0.4234 to 0.4234 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | control line | 1 | 0.4371 | 0.4371 to 0.4371 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | control line | 1 | 0.7654 | 0.7654 to 0.7654 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | control line | 1 | 0.9988 | 0.9988 to 0.9988 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 temra t cells | plasma | control line | 1 | 0.9965 | 0.9965 to 0.9965 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cortical neurons | plasma | control line | 1 | 1.082 | 1.082 to 1.082 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | embryonic stem cells | plasma | control line | 1 | 0.9548 | 0.9548 to 0.9548 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | control line | 1 | 1.011 | 1.011 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | plasma | control line | 1 | 1.057 | 1.057 to 1.057 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | oligodendrocytes | plasma | control line | 1 | 0.9636 | 0.9636 to 0.9636 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | prostate epithelium | plasma | control line | 1 | 0.9287 | 0.9287 to 0.9287 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | striated muscle | plasma | control line | 1 | 0.9389 | 0.9389 to 0.9389 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | vascular leptomeningeal cells | plasma | control line | 1 | 0.9177 | 0.9177 to 0.9177 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | knockout | 1 | 1.078 | 1.078 to 1.078 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | knockout | 1 | 1.15 | 1.15 to 1.15 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | knockout | 1 | 0.7147 | 0.7147 to 0.7147 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | knockout | 1 | 1.105 | 1.105 to 1.105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | knockout | 1 | 1.091 | 1.091 to 1.091 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | knockout | 1 | 0.528 | 0.528 to 0.528 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | knockout | 1 | 0.8369 | 0.8369 to 0.8369 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | knockout | 1 | 1.262 | 1.262 to 1.262 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | knockout | 1 | 1.002 | 1.002 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | knockout | 1 | 1.074 | 1.074 to 1.074 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | heart fibroblasts | plasma | knockout | 1 | 0.9437 | 0.9437 to 0.9437 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | memory b cells | plasma | knockout | 1 | 0.9512 | 0.9512 to 0.9512 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | pancreatic acinar | plasma | knockout | 1 | 1.071 | 1.071 to 1.071 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | smooth muscle | plasma | knockout | 1 | 1.104 | 1.104 to 1.104 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector cell cd8 | plasma | knockout | 1 | 1.021 | 1.021 to 1.021 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | plasma | cord-blood B cells | 3 | 0.703 | 0.699 to 0.7394 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | plasma | cord-blood B cells | 3 | 0.585 | 0.5533 to 0.6 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | plasma | cord-blood B cells | 3 | 0.2172 | 0.192 to 0.3581 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | plasma | cord-blood B cells | 3 | 0.7945 | 0.7586 to 0.8507 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | plasma | cord-blood B cells | 3 | 0.7287 | 0.7162 to 0.7639 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | plasma | cord-blood B cells | 3 | 0.1669 | 0.1481 to 0.2382 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | plasma | cord-blood B cells | 3 | 0.6572 | 0.617 to 0.6854 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | mixed specimen (plasma) | plasma | cord-blood B cells | 3 | 0.7906 | 0.7712 to 0.7941 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| C-Score | C-Score whole genome | whole genome | plasma | cord-blood B cells | 3 | 0.9966 | 0.9911 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cardiomyocyte | plasma | cord-blood B cells | 2 | 1.017 | 1.014 to 1.021 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | colon macrophages | plasma | cord-blood B cells | 2 | 1.037 | 0.9976 to 1.075 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | erythrocyte progenitors | plasma | cord-blood B cells | 3 | 1.011 | 0.9726 to 1.014 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | lung interstitial macrophages | plasma | cord-blood B cells | 3 | 1.041 | 0.9996 to 1.091 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive b cells | plasma | cord-blood B cells | 3 | 1.006 | 0.9943 to 1.051 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | t effector memory cd4 | plasma | cord-blood B cells | 2 | 0.9689 | 0.9388 to 0.9991 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | naive cd8 t cells | not stated | leukemia | 1 | 1.005 | 1.005 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | kidney tubular endothelium | not stated | leukemia | 1 | 1.03 | 1.03 to 1.03 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted b cells | leukemia | 13 | 0.7985 | 0.7338 to 0.9046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted b cells | leukemia | 13 | 0.7246 | 0.6644 to 0.8066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted b cells | leukemia | 13 | 0.458 | 0.2847 to 0.5301 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted b cells | leukemia | 13 | 0.8664 | 0.7774 to 1.057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted b cells | leukemia | 13 | 0.8142 | 0.7419 to 0.9192 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted b cells | leukemia | 13 | 5.524 | 4.563 to 7.809 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted b cells | leukemia | 13 | 0.6444 | 0.5133 to 0.8632 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | b cells | sorted b cells | leukemia | 11 | 0.9035 | 0.8274 to 0.9323 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Coupling-A | coupling-A | b cells | sorted b cells | leukemia | 13 | 1.092 | 1.032 to 1.114 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Met-A | Met-A sorted cell (atlas) | b cells | sorted b cells | leukemia | 13 | 1.506 | 1.398 to 1.744 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted b cells | leukemia | 13 | 0.9955 | 0.9924 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted b cells | leukemia | 13 | 0.9665 | 0.8976 to 1.048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted adipocytes | healthy | 3 | 1.131 | 1.126 to 1.139 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted adipocytes | healthy | 3 | 1.1 | 1.1 to 1.105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted adipocytes | healthy | 3 | 0.2112 | 0.1943 to 0.2593 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted adipocytes | healthy | 3 | 1.383 | 1.358 to 1.413 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted adipocytes | healthy | 3 | 1.155 | 1.149 to 1.161 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted adipocytes | healthy | 3 | 0.1903 | 0.1753 to 0.2306 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted adipocytes | healthy | 3 | 1.224 | 1.198 to 1.252 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | adipocytes | sorted adipocytes | healthy | 3 | 1.009 | 1.007 to 1.023 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | adipocytes | sorted adipocytes | healthy | 3 | 0.844 | 0.8332 to 0.8683 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted adipocytes | healthy | 3 | 0.9999 | 0.9956 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted adipocytes | healthy | 3 | 0.9343 | 0.878 to 0.957 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted adipocytes | healthy | 3 | 1.014 | 0.9963 to 1.05 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted adipocytes | healthy | 3 | 0.9882 | 0.9873 to 0.9908 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted adipocytes | healthy | 3 | 1.048 | 1.029 to 1.122 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted adipocytes | healthy | 3 | 0.9872 | 0.9777 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted aorta endothelium | healthy | 2 | 1.049 | 1.027 to 1.072 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted aorta endothelium | healthy | 2 | 1.021 | 0.9921 to 1.049 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted aorta endothelium | healthy | 2 | 0.1941 | 0.1901 to 0.1981 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted aorta endothelium | healthy | 2 | 1.287 | 1.287 to 1.288 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted aorta endothelium | healthy | 2 | 1.069 | 1.042 to 1.097 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted aorta endothelium | healthy | 2 | 0.1671 | 0.1643 to 0.1699 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted aorta endothelium | healthy | 2 | 1.124 | 1.116 to 1.132 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | aorta endothelium | sorted aorta endothelium | healthy | 2 | 0.9585 | 0.9232 to 0.9938 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | aorta endothelium | sorted aorta endothelium | healthy | 2 | 1.01 | 0.9869 to 1.033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted aorta endothelium | healthy | 2 | 0.995 | 0.9921 to 0.9979 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted aorta endothelium | healthy | 2 | 0.9328 | 0.9057 to 0.96 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted aorta endothelium | healthy | 2 | 0.9937 | 0.9395 to 1.048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted aorta endothelium | healthy | 2 | 0.993 | 0.9826 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted aorta endothelium | healthy | 2 | 1.068 | 1.064 to 1.072 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted aorta endothelium | healthy | 2 | 0.973 | 0.935 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted vascular saphenous endothelium | healthy | 3 | 0.9801 | 0.9492 to 1.065 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted vascular saphenous endothelium | healthy | 3 | 0.955 | 0.9132 to 1.046 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted vascular saphenous endothelium | healthy | 3 | 0.2198 | 0.2181 to 0.2221 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted vascular saphenous endothelium | healthy | 3 | 1.162 | 1.125 to 1.295 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted vascular saphenous endothelium | healthy | 3 | 1.004 | 0.9722 to 1.086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted vascular saphenous endothelium | healthy | 3 | 0.1961 | 0.1901 to 0.1991 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted vascular saphenous endothelium | healthy | 3 | 1.027 | 0.9996 to 1.13 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | vascular saphenous endothelium | sorted vascular saphenous endothelium | healthy | 3 | 1.005 | 0.9988 to 1.037 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | vascular saphenous endothelium | sorted vascular saphenous endothelium | healthy | 3 | 0.7461 | 0.7366 to 0.7784 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted vascular saphenous endothelium | healthy | 3 | 1.001 | 1.001 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted vascular saphenous endothelium | healthy | 3 | 0.8744 | 0.8583 to 0.9418 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted vascular saphenous endothelium | healthy | 3 | 0.9609 | 0.9306 to 0.9729 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted vascular saphenous endothelium | healthy | 3 | 1.001 | 0.9641 to 1.015 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted vascular saphenous endothelium | healthy | 3 | 1.067 | 1.056 to 1.075 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted vascular saphenous endothelium | healthy | 3 | 1.026 | 0.9822 to 1.039 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney glomerular endothelium | healthy | 3 | 1.004 | 0.9238 to 1.021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney glomerular endothelium | healthy | 3 | 0.984 | 0.891 to 1.001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney glomerular endothelium | healthy | 3 | 0.2206 | 0.2025 to 0.2215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney glomerular endothelium | healthy | 3 | 1.179 | 1.1 to 1.239 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney glomerular endothelium | healthy | 3 | 1.03 | 0.9448 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney glomerular endothelium | healthy | 3 | 0.198 | 0.1786 to 0.1981 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney glomerular endothelium | healthy | 3 | 1.048 | 0.9788 to 1.089 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney glomerular endothelium | sorted kidney glomerular endothelium | healthy | 3 | 1.035 | 1.004 to 1.044 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney glomerular endothelium | sorted kidney glomerular endothelium | healthy | 3 | 0.7549 | 0.7375 to 0.7588 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney glomerular endothelium | healthy | 3 | 0.9994 | 0.9979 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney glomerular endothelium | healthy | 3 | 0.8897 | 0.8869 to 0.9117 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney glomerular endothelium | healthy | 3 | 1.001 | 0.9841 to 1.016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney glomerular endothelium | healthy | 3 | 1.02 | 0.9876 to 1.036 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney glomerular endothelium | healthy | 3 | 1.016 | 0.9656 to 1.028 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney glomerular endothelium | healthy | 3 | 0.9561 | 0.9432 to 0.9766 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney tubular endothelium | healthy | 3 | 0.9337 | 0.9326 to 0.9478 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney tubular endothelium | healthy | 3 | 0.9001 | 0.8946 to 0.92 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney tubular endothelium | healthy | 3 | 0.2232 | 0.2126 to 0.2274 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney tubular endothelium | healthy | 3 | 1.105 | 1.091 to 1.118 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney tubular endothelium | healthy | 3 | 0.9554 | 0.9553 to 0.9708 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney tubular endothelium | healthy | 3 | 0.1859 | 0.1823 to 0.2025 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney tubular endothelium | healthy | 3 | 0.9726 | 0.9373 to 0.9877 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney tubular endothelium | sorted kidney tubular endothelium | healthy | 3 | 1.026 | 1.024 to 1.027 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney tubular endothelium | sorted kidney tubular endothelium | healthy | 3 | 0.7751 | 0.727 to 0.8077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney tubular endothelium | healthy | 3 | 1 | 0.9985 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney tubular endothelium | healthy | 3 | 0.9269 | 0.9066 to 0.9856 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney tubular endothelium | healthy | 3 | 1.015 | 0.9769 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney tubular endothelium | healthy | 3 | 1.003 | 0.9863 to 1.044 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney tubular endothelium | healthy | 3 | 1.001 | 0.9976 to 1.024 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney tubular endothelium | healthy | 3 | 1.005 | 0.9792 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar endothelium | healthy | 3 | 1.04 | 1.025 to 1.045 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar endothelium | healthy | 3 | 1.019 | 0.9924 to 1.024 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar endothelium | healthy | 3 | 0.2261 | 0.1941 to 0.2311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar endothelium | healthy | 3 | 1.228 | 1.212 to 1.307 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar endothelium | healthy | 3 | 1.053 | 1.048 to 1.071 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar endothelium | healthy | 3 | 0.1984 | 0.1695 to 0.2073 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar endothelium | healthy | 3 | 1.091 | 1.075 to 1.158 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar endothelium | sorted lung alveolar endothelium | healthy | 3 | 0.9922 | 0.9337 to 1.006 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar endothelium | sorted lung alveolar endothelium | healthy | 3 | 0.7626 | 0.7536 to 0.7654 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar endothelium | healthy | 3 | 1.001 | 1 to 1.01 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar endothelium | healthy | 3 | 0.9146 | 0.8596 to 0.9473 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar endothelium | healthy | 3 | 0.9704 | 0.9134 to 1.044 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar endothelium | healthy | 3 | 0.9419 | 0.9397 to 1.025 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung alveolar endothelium | healthy | 3 | 1.04 | 1.016 to 1.052 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung alveolar endothelium | healthy | 3 | 0.9676 | 0.9654 to 1.008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreas endothelium | healthy | 3 | 1.075 | 1.066 to 1.08 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreas endothelium | healthy | 3 | 1.061 | 1.048 to 1.061 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreas endothelium | healthy | 3 | 0.1959 | 0.1949 to 0.2004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreas endothelium | healthy | 3 | 1.293 | 1.286 to 1.306 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreas endothelium | healthy | 3 | 1.099 | 1.089 to 1.105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreas endothelium | healthy | 3 | 0.165 | 0.1615 to 0.1726 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreas endothelium | healthy | 3 | 1.094 | 1.092 to 1.121 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreas endothelium | sorted pancreas endothelium | healthy | 3 | 1.051 | 1.008 to 1.053 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreas endothelium | sorted pancreas endothelium | healthy | 3 | 0.8445 | 0.789 to 0.8494 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreas endothelium | healthy | 3 | 1 | 0.996 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreas endothelium | healthy | 3 | 0.9662 | 0.9477 to 0.9939 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreas endothelium | healthy | 3 | 1.006 | 0.9529 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreas endothelium | healthy | 3 | 0.9871 | 0.9341 to 1.019 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreas endothelium | healthy | 3 | 1.04 | 0.9878 to 1.043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreas endothelium | healthy | 3 | 0.984 | 0.9767 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon fibroblasts | healthy | 2 | 1.219 | 1.207 to 1.23 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon fibroblasts | healthy | 2 | 1.186 | 1.182 to 1.189 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon fibroblasts | healthy | 2 | 0.2261 | 0.2258 to 0.2264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon fibroblasts | healthy | 2 | 1.474 | 1.473 to 1.476 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon fibroblasts | healthy | 2 | 1.247 | 1.232 to 1.262 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon fibroblasts | healthy | 2 | 0.2032 | 0.203 to 0.2034 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon fibroblasts | healthy | 2 | 1.313 | 1.309 to 1.317 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon fibroblasts | sorted colon fibroblasts | healthy | 2 | 0.9906 | 0.9839 to 0.9973 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon fibroblasts | sorted colon fibroblasts | healthy | 2 | 0.9257 | 0.9229 to 0.9286 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon fibroblasts | healthy | 2 | 0.9929 | 0.9916 to 0.9943 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon fibroblasts | healthy | 2 | 0.9425 | 0.9214 to 0.9637 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon fibroblasts | healthy | 2 | 1.023 | 1.017 to 1.028 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon fibroblasts | healthy | 2 | 0.9631 | 0.959 to 0.9672 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon fibroblasts | healthy | 2 | 1.06 | 1.039 to 1.08 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon fibroblasts | healthy | 2 | 0.9867 | 0.9676 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted heart fibroblasts | healthy | 3 | 1.188 | 1.153 to 1.212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted heart fibroblasts | healthy | 3 | 1.197 | 1.165 to 1.215 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted heart fibroblasts | healthy | 3 | 0.2028 | 0.1858 to 0.2031 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted heart fibroblasts | healthy | 3 | 1.646 | 1.58 to 1.654 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted heart fibroblasts | healthy | 3 | 1.181 | 1.149 to 1.212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted heart fibroblasts | healthy | 3 | 0.1775 | 0.16 to 0.1787 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted heart fibroblasts | healthy | 3 | 1.442 | 1.388 to 1.452 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | heart fibroblasts | sorted heart fibroblasts | healthy | 3 | 0.8859 | 0.8847 to 0.8951 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | heart fibroblasts | sorted heart fibroblasts | healthy | 3 | 0.7597 | 0.7446 to 0.765 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted heart fibroblasts | healthy | 3 | 0.991 | 0.9888 to 0.9958 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted heart fibroblasts | healthy | 3 | 0.9013 | 0.8797 to 0.9396 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted heart fibroblasts | healthy | 3 | 0.9246 | 0.9223 to 0.9956 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted heart fibroblasts | healthy | 3 | 0.956 | 0.947 to 0.9758 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted heart fibroblasts | healthy | 3 | 1.06 | 1.018 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted heart fibroblasts | healthy | 3 | 1.02 | 0.9863 to 1.025 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted striated muscle | healthy | 2 | 1.159 | 1.154 to 1.165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted striated muscle | healthy | 2 | 1.134 | 1.129 to 1.138 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted striated muscle | healthy | 2 | 0.1873 | 0.183 to 0.1916 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted striated muscle | healthy | 2 | 1.401 | 1.389 to 1.414 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted striated muscle | healthy | 2 | 1.187 | 1.182 to 1.192 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted striated muscle | healthy | 2 | 0.1633 | 0.1604 to 0.1663 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted striated muscle | healthy | 2 | 1.233 | 1.221 to 1.246 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | striated muscle | sorted striated muscle | healthy | 2 | 1.054 | 1.042 to 1.066 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | striated muscle | sorted striated muscle | healthy | 2 | 0.9128 | 0.905 to 0.9206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted striated muscle | healthy | 2 | 0.996 | 0.9955 to 0.9965 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted striated muscle | healthy | 2 | 0.9691 | 0.963 to 0.9752 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted striated muscle | healthy | 2 | 0.9605 | 0.9596 to 0.9613 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted striated muscle | healthy | 2 | 0.9923 | 0.9838 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted striated muscle | healthy | 2 | 1.062 | 1.024 to 1.1 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted striated muscle | healthy | 2 | 0.9881 | 0.9361 to 1.04 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted smooth muscle | healthy | 3 | 1.29 | 1.085 to 1.315 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted smooth muscle | healthy | 3 | 1.326 | 1.1 to 1.336 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted smooth muscle | healthy | 3 | 0.2035 | 0.1985 to 0.232 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted smooth muscle | healthy | 3 | 1.806 | 1.434 to 1.811 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted smooth muscle | healthy | 3 | 1.28 | 1.087 to 1.312 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted smooth muscle | healthy | 3 | 0.1788 | 0.1697 to 0.2044 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted smooth muscle | healthy | 3 | 1.567 | 1.266 to 1.585 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | smooth muscle | sorted smooth muscle | healthy | 3 | 0.874 | 0.8608 to 0.8995 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | smooth muscle | sorted smooth muscle | healthy | 3 | 0.8688 | 0.7598 to 0.8782 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted smooth muscle | healthy | 3 | 0.9996 | 0.9947 to 1 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted smooth muscle | healthy | 3 | 0.9969 | 0.948 to 0.9988 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted smooth muscle | healthy | 3 | 0.9612 | 0.94 to 1.032 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted smooth muscle | healthy | 3 | 0.9903 | 0.9401 to 1.045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted smooth muscle | healthy | 3 | 1.067 | 1.015 to 1.076 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted smooth muscle | healthy | 3 | 0.9851 | 0.9134 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cardiomyocyte | healthy | 3 | 0.8695 | 0.8415 to 0.8764 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cardiomyocyte | healthy | 3 | 0.85 | 0.8229 to 0.8546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cardiomyocyte | healthy | 3 | 0.2005 | 0.1998 to 0.204 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cardiomyocyte | healthy | 3 | 1.065 | 1.029 to 1.082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cardiomyocyte | healthy | 3 | 0.8842 | 0.8557 to 0.8905 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cardiomyocyte | healthy | 3 | 0.178 | 0.1771 to 0.1791 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cardiomyocyte | healthy | 3 | 0.9205 | 0.8999 to 0.9453 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cardiomyocyte | sorted cardiomyocyte | healthy | 3 | 0.9498 | 0.9477 to 0.9526 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cardiomyocyte | sorted cardiomyocyte | healthy | 3 | 0.7468 | 0.7302 to 0.7668 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cardiomyocyte | healthy | 3 | 1.003 | 1.002 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted cardiomyocyte | healthy | 3 | 1.003 | 0.9858 to 1.048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted cardiomyocyte | healthy | 3 | 0.9541 | 0.9445 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted cardiomyocyte | healthy | 3 | 0.9712 | 0.9428 to 0.9764 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cardiomyocyte | healthy | 3 | 1.089 | 1.035 to 1.096 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted cardiomyocyte | healthy | 3 | 0.9944 | 0.9398 to 1.009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted oligodendrocytes | healthy | 3 | 0.8386 | 0.8255 to 0.85 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted oligodendrocytes | healthy | 3 | 0.8308 | 0.8207 to 0.8364 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted oligodendrocytes | healthy | 3 | 0.2005 | 0.1771 to 0.2262 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted oligodendrocytes | healthy | 3 | 0.9975 | 0.9759 to 1.004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted oligodendrocytes | healthy | 3 | 0.8554 | 0.8419 to 0.8718 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted oligodendrocytes | healthy | 3 | 0.1745 | 0.1524 to 0.1994 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted oligodendrocytes | healthy | 3 | 0.8548 | 0.8519 to 0.8635 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | oligodendrocytes | sorted oligodendrocytes | healthy | 3 | 0.9154 | 0.9079 to 0.944 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | oligodendrocytes | sorted oligodendrocytes | healthy | 3 | 0.755 | 0.725 to 0.757 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted oligodendrocytes | healthy | 3 | 0.9937 | 0.9934 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted oligodendrocytes | healthy | 3 | 0.9375 | 0.914 to 0.9969 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted oligodendrocytes | healthy | 3 | 0.9404 | 0.9265 to 0.9429 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted oligodendrocytes | healthy | 3 | 0.9635 | 0.9502 to 1.063 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted oligodendrocytes | healthy | 3 | 1.01 | 1.004 to 1.014 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted oligodendrocytes | healthy | 3 | 0.9918 | 0.9774 to 0.997 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cortical neurons | healthy | 3 | 0.8265 | 0.8023 to 0.973 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cortical neurons | healthy | 3 | 0.8282 | 0.8063 to 0.9663 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cortical neurons | healthy | 3 | 0.2126 | 0.2104 to 0.2482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cortical neurons | healthy | 3 | 0.9795 | 0.9191 to 1.094 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cortical neurons | healthy | 3 | 0.8427 | 0.8229 to 1.002 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cortical neurons | healthy | 3 | 0.1848 | 0.1835 to 0.2147 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cortical neurons | healthy | 3 | 0.8471 | 0.7924 to 0.9396 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cortical neurons | sorted cortical neurons | healthy | 3 | 0.7885 | 0.7714 to 0.8435 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cortical neurons | sorted cortical neurons | healthy | 3 | 0.8606 | 0.7731 to 0.9282 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cortical neurons | healthy | 3 | 0.9971 | 0.9947 to 0.9977 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted cortical neurons | healthy | 3 | 0.9233 | 0.8906 to 0.9516 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted cortical neurons | healthy | 3 | 0.9644 | 0.9184 to 0.9782 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted cortical neurons | healthy | 3 | 0.9789 | 0.9384 to 0.9821 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cortical neurons | healthy | 3 | 0.9812 | 0.9511 to 1.062 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted cortical neurons | healthy | 3 | 0.949 | 0.8997 to 0.9581 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted hepatocyte | healthy | 3 | 0.8729 | 0.868 to 0.8827 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted hepatocyte | healthy | 3 | 0.8596 | 0.8418 to 0.8648 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted hepatocyte | healthy | 3 | 0.2317 | 0.2109 to 0.2427 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted hepatocyte | healthy | 3 | 1.074 | 1.029 to 1.082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted hepatocyte | healthy | 3 | 0.8853 | 0.8851 to 0.8969 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted hepatocyte | healthy | 3 | 0.2049 | 0.1884 to 0.2167 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted hepatocyte | healthy | 3 | 0.9503 | 0.9186 to 0.9615 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | hepatocyte | sorted hepatocyte | healthy | 3 | 0.8843 | 0.8776 to 0.9239 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | hepatocyte | sorted hepatocyte | healthy | 3 | 0.8068 | 0.7929 to 0.8211 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted hepatocyte | healthy | 3 | 1.004 | 0.9999 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted hepatocyte | healthy | 3 | 0.9686 | 0.9568 to 0.9696 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted hepatocyte | healthy | 3 | 1.026 | 1.022 to 1.056 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted hepatocyte | healthy | 3 | 0.9472 | 0.9391 to 0.9615 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted hepatocyte | healthy | 3 | 0.9805 | 0.9755 to 0.9993 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted hepatocyte | healthy | 3 | 0.9268 | 0.8905 to 0.9646 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic duct | healthy | 3 | 0.8287 | 0.8025 to 0.8543 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic duct | healthy | 3 | 0.8217 | 0.7691 to 0.8234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic duct | healthy | 3 | 0.1835 | 0.1743 to 0.1878 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic duct | healthy | 3 | 0.967 | 0.9305 to 1.008 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic duct | healthy | 3 | 0.8497 | 0.8246 to 0.8766 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic duct | healthy | 3 | 0.159 | 0.1528 to 0.1605 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic duct | healthy | 3 | 0.8191 | 0.81 to 0.8787 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic duct | sorted pancreatic duct | healthy | 3 | 0.9588 | 0.9501 to 0.9746 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic duct | sorted pancreatic duct | healthy | 3 | 0.8158 | 0.7719 to 0.8216 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic duct | healthy | 3 | 0.9985 | 0.995 to 0.9999 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic duct | healthy | 3 | 0.9492 | 0.941 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic duct | healthy | 3 | 0.9488 | 0.9311 to 0.9726 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic duct | healthy | 3 | 0.9724 | 0.962 to 1.043 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic duct | healthy | 3 | 1.034 | 1.003 to 1.053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic duct | healthy | 3 | 0.9548 | 0.9459 to 1.065 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic acinar | healthy | 3 | 1.086 | 1.027 to 1.09 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic acinar | healthy | 3 | 1.086 | 1.047 to 1.107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic acinar | healthy | 3 | 0.1764 | 0.1665 to 0.1856 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic acinar | healthy | 3 | 1.45 | 1.367 to 1.462 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic acinar | healthy | 3 | 1.087 | 1.031 to 1.1 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic acinar | healthy | 3 | 0.1521 | 0.1521 to 0.168 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic acinar | healthy | 3 | 1.239 | 1.219 to 1.273 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic acinar | sorted pancreatic acinar | healthy | 3 | 0.8983 | 0.8263 to 0.9121 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic acinar | sorted pancreatic acinar | healthy | 3 | 0.8454 | 0.781 to 0.8619 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic acinar | healthy | 3 | 1 | 0.9991 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic acinar | healthy | 3 | 0.9336 | 0.9328 to 0.9968 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic acinar | healthy | 3 | 0.9456 | 0.9312 to 0.9787 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic acinar | healthy | 3 | 0.9661 | 0.9433 to 0.9747 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic acinar | healthy | 3 | 1.151 | 1.088 to 1.151 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic acinar | healthy | 3 | 1.017 | 0.9804 to 1.019 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic delta | healthy | 3 | 0.9765 | 0.9352 to 0.9896 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic delta | healthy | 3 | 0.966 | 0.9201 to 0.9826 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic delta | healthy | 3 | 0.2016 | 0.2005 to 0.2024 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic delta | healthy | 3 | 1.18 | 1.118 to 1.207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic delta | healthy | 3 | 0.9953 | 0.9557 to 1.007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic delta | healthy | 3 | 0.1675 | 0.1652 to 0.1747 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic delta | healthy | 3 | 0.993 | 0.9688 to 1.015 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic delta | sorted pancreatic delta | healthy | 3 | 1.054 | 1.023 to 1.06 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic delta | sorted pancreatic delta | healthy | 3 | 0.797 | 0.7375 to 0.7978 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic delta | healthy | 3 | 0.9963 | 0.9952 to 0.9985 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic delta | healthy | 3 | 0.9674 | 0.9539 to 0.9822 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic delta | healthy | 3 | 1.006 | 0.9585 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic delta | healthy | 3 | 0.9904 | 0.9734 to 1.083 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic delta | healthy | 3 | 0.9624 | 0.9393 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted pancreatic delta | healthy | 1 | 1.012 | 1.012 to 1.012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic beta | healthy | 3 | 0.923 | 0.8956 to 0.955 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic beta | healthy | 3 | 0.901 | 0.8734 to 0.9334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic beta | healthy | 3 | 0.2127 | 0.2103 to 0.2159 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic beta | healthy | 3 | 1.092 | 1.052 to 1.153 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic beta | healthy | 3 | 0.9438 | 0.9166 to 0.9739 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic beta | healthy | 3 | 0.1812 | 0.1811 to 0.1857 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic beta | healthy | 3 | 0.9385 | 0.9077 to 0.9978 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic beta | sorted pancreatic beta | healthy | 3 | 1.01 | 0.9931 to 1.03 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic beta | sorted pancreatic beta | healthy | 3 | 0.8469 | 0.8257 to 0.8517 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic beta | healthy | 3 | 1.002 | 0.9983 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic beta | healthy | 3 | 0.9815 | 0.9562 to 0.9972 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic beta | healthy | 3 | 1.008 | 0.971 to 1.037 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic beta | healthy | 3 | 1.059 | 1.035 to 1.084 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic beta | healthy | 3 | 0.9401 | 0.9327 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted pancreatic alpha | healthy | 3 | 0.9336 | 0.9097 to 0.934 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted pancreatic alpha | healthy | 3 | 0.9066 | 0.8905 to 0.9082 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted pancreatic alpha | healthy | 3 | 0.2226 | 0.2219 to 0.2278 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted pancreatic alpha | healthy | 3 | 1.138 | 1.1 to 1.139 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted pancreatic alpha | healthy | 3 | 0.9497 | 0.9259 to 0.9498 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted pancreatic alpha | healthy | 3 | 0.1963 | 0.1932 to 0.1989 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted pancreatic alpha | healthy | 3 | 0.9914 | 0.9577 to 1.002 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | pancreatic alpha | sorted pancreatic alpha | healthy | 3 | 1.025 | 1.011 to 1.028 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | pancreatic alpha | sorted pancreatic alpha | healthy | 3 | 0.7483 | 0.7282 to 0.7527 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted pancreatic alpha | healthy | 3 | 1.004 | 1 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted pancreatic alpha | healthy | 3 | 0.9785 | 0.9662 to 0.9944 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted pancreatic alpha | healthy | 3 | 0.9537 | 0.8613 to 0.9591 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted pancreatic alpha | healthy | 3 | 1.042 | 1.014 to 1.045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted pancreatic alpha | healthy | 3 | 1.01 | 0.9852 to 1.045 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney glomerular epithelium | healthy | 2 | 0.9116 | 0.8822 to 0.941 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney glomerular epithelium | healthy | 2 | 0.8972 | 0.8696 to 0.9248 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney glomerular epithelium | healthy | 2 | 0.1886 | 0.1879 to 0.1893 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney glomerular epithelium | healthy | 2 | 1.07 | 1.043 to 1.096 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney glomerular epithelium | healthy | 2 | 0.9349 | 0.9036 to 0.9662 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney glomerular epithelium | healthy | 2 | 0.1628 | 0.1593 to 0.1664 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney glomerular epithelium | healthy | 2 | 0.9278 | 0.9101 to 0.9455 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney glomerular epithelium | sorted kidney glomerular epithelium | healthy | 2 | 0.992 | 0.9821 to 1.002 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney glomerular epithelium | sorted kidney glomerular epithelium | healthy | 2 | 0.9458 | 0.9156 to 0.976 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney glomerular epithelium | healthy | 2 | 0.9998 | 0.9993 to 1 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney glomerular epithelium | healthy | 2 | 0.9497 | 0.9289 to 0.9705 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney glomerular epithelium | healthy | 2 | 0.9899 | 0.9803 to 0.9995 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney glomerular epithelium | healthy | 2 | 1.057 | 1.036 to 1.078 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney glomerular epithelium | healthy | 2 | 1.012 | 1.011 to 1.013 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney glomerular epithelium | healthy | 2 | 0.9703 | 0.9598 to 0.9807 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted kidney tubular epithelium | healthy | 3 | 0.8699 | 0.8137 to 0.9188 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted kidney tubular epithelium | healthy | 3 | 0.8487 | 0.7814 to 0.9055 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted kidney tubular epithelium | healthy | 3 | 0.208 | 0.1909 to 0.2223 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted kidney tubular epithelium | healthy | 3 | 1.038 | 0.9187 to 1.095 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted kidney tubular epithelium | healthy | 3 | 0.8899 | 0.8379 to 0.9392 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted kidney tubular epithelium | healthy | 3 | 0.1835 | 0.17 to 0.1985 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted kidney tubular epithelium | healthy | 3 | 0.9146 | 0.8112 to 0.9579 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | kidney tubular epithelium | sorted kidney tubular epithelium | healthy | 3 | 0.9737 | 0.9079 to 0.9893 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | kidney tubular epithelium | sorted kidney tubular epithelium | healthy | 3 | 0.738 | 0.7199 to 0.7572 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted kidney tubular epithelium | healthy | 3 | 0.9985 | 0.9936 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted kidney tubular epithelium | healthy | 3 | 0.9666 | 0.9362 to 1.009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted kidney tubular epithelium | healthy | 2 | 0.9564 | 0.9493 to 0.9634 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted kidney tubular epithelium | healthy | 3 | 1.02 | 1.003 to 1.041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted kidney tubular epithelium | healthy | 3 | 1.061 | 1.005 to 1.108 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted kidney tubular epithelium | healthy | 3 | 0.9931 | 0.9456 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted thyroid epithelium | healthy | 3 | 0.803 | 0.7959 to 0.8067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted thyroid epithelium | healthy | 3 | 0.7684 | 0.7618 to 0.7718 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted thyroid epithelium | healthy | 3 | 0.2112 | 0.2089 to 0.2213 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted thyroid epithelium | healthy | 3 | 0.9262 | 0.9187 to 0.9395 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted thyroid epithelium | healthy | 3 | 0.823 | 0.816 to 0.8264 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted thyroid epithelium | healthy | 3 | 0.1836 | 0.1831 to 0.1939 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted thyroid epithelium | healthy | 3 | 0.8123 | 0.8038 to 0.8238 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | thyroid epithelium | sorted thyroid epithelium | healthy | 3 | 0.9983 | 0.9971 to 1.01 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | thyroid epithelium | sorted thyroid epithelium | healthy | 3 | 0.7355 | 0.7351 to 0.7398 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted thyroid epithelium | healthy | 3 | 1.001 | 0.998 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted thyroid epithelium | healthy | 3 | 0.9951 | 0.9818 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted thyroid epithelium | healthy | 3 | 0.9998 | 0.9413 to 1.012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted thyroid epithelium | healthy | 3 | 1.008 | 0.9902 to 1.044 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted thyroid epithelium | healthy | 3 | 0.9222 | 0.9067 to 0.9592 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted fallopien tubes epithelium | healthy | 3 | 0.8354 | 0.8277 to 0.846 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted fallopien tubes epithelium | healthy | 3 | 0.8011 | 0.7968 to 0.8295 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted fallopien tubes epithelium | healthy | 3 | 0.1905 | 0.189 to 0.2548 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted fallopien tubes epithelium | healthy | 3 | 1.019 | 1.01 to 1.023 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted fallopien tubes epithelium | healthy | 3 | 0.85 | 0.8432 to 0.8579 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted fallopien tubes epithelium | healthy | 3 | 0.1678 | 0.1627 to 0.2194 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted fallopien tubes epithelium | healthy | 3 | 0.8921 | 0.887 to 0.8925 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | fallopien tubes epithelium | sorted fallopien tubes epithelium | healthy | 3 | 0.9378 | 0.8872 to 0.9524 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | fallopien tubes epithelium | sorted fallopien tubes epithelium | healthy | 3 | 0.7179 | 0.6913 to 0.74 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted fallopien tubes epithelium | healthy | 3 | 0.9959 | 0.9922 to 0.9968 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted fallopien tubes epithelium | healthy | 3 | 0.9257 | 0.9148 to 0.9405 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted fallopien tubes epithelium | healthy | 3 | 1.021 | 0.9907 to 1.023 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted fallopien tubes epithelium | healthy | 3 | 0.9874 | 0.9266 to 1.016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted fallopien tubes epithelium | healthy | 3 | 1.021 | 0.9979 to 1.028 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted endometrium epithelium | healthy | 3 | 0.8785 | 0.8683 to 1.003 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted endometrium epithelium | healthy | 3 | 0.8602 | 0.8484 to 0.9915 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted endometrium epithelium | healthy | 3 | 0.1894 | 0.1733 to 0.2029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted endometrium epithelium | healthy | 3 | 1.059 | 1.054 to 1.176 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted endometrium epithelium | healthy | 3 | 0.8971 | 0.8854 to 1.03 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted endometrium epithelium | healthy | 3 | 0.1643 | 0.1433 to 0.1817 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted endometrium epithelium | healthy | 3 | 0.9362 | 0.9257 to 0.9963 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | endometrium epithelium | sorted endometrium epithelium | healthy | 3 | 0.9201 | 0.8911 to 0.9388 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | endometrium epithelium | sorted endometrium epithelium | healthy | 3 | 0.7297 | 0.6923 to 0.8105 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted endometrium epithelium | healthy | 3 | 1.002 | 1 to 1.008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted endometrium epithelium | healthy | 3 | 0.9685 | 0.947 to 0.975 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted endometrium epithelium | healthy | 3 | 0.9762 | 0.9564 to 0.9935 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted endometrium epithelium | healthy | 3 | 0.9599 | 0.9246 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted endometrium epithelium | healthy | 3 | 1.017 | 1.005 to 1.061 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted erythrocyte progenitors | healthy | 3 | 1.555 | 1.525 to 1.573 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted erythrocyte progenitors | healthy | 3 | 1.531 | 1.485 to 1.541 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted erythrocyte progenitors | healthy | 3 | 0.1895 | 0.1785 to 0.1999 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted erythrocyte progenitors | healthy | 3 | 2.165 | 2.116 to 2.206 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted erythrocyte progenitors | healthy | 3 | 1.552 | 1.525 to 1.571 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted erythrocyte progenitors | healthy | 3 | 0.1721 | 0.1467 to 0.1844 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted erythrocyte progenitors | healthy | 3 | 1.904 | 1.819 to 1.968 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | erythrocyte progenitors | sorted erythrocyte progenitors | healthy | 3 | 0.7374 | 0.7214 to 0.7397 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | erythrocyte progenitors | sorted erythrocyte progenitors | healthy | 3 | 0.7469 | 0.7274 to 0.8515 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted erythrocyte progenitors | healthy | 3 | 0.9997 | 0.9989 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted erythrocyte progenitors | healthy | 3 | 1.038 | 1.002 to 1.047 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted erythrocyte progenitors | healthy | 3 | 0.986 | 0.9736 to 0.9878 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted erythrocyte progenitors | healthy | 3 | 0.9045 | 0.8969 to 0.9152 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted erythrocyte progenitors | healthy | 3 | 0.977 | 0.9486 to 1.012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted erythrocyte progenitors | healthy | 3 | 0.9167 | 0.9149 to 0.9365 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cd4 t cells | healthy | 3 | 0.8044 | 0.7923 to 0.8584 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cd4 t cells | healthy | 3 | 0.6958 | 0.6729 to 0.7618 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cd4 t cells | healthy | 3 | 0.3113 | 0.2439 to 0.3121 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cd4 t cells | healthy | 3 | 0.8738 | 0.8708 to 0.9567 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cd4 t cells | healthy | 3 | 0.8292 | 0.8194 to 0.881 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cd4 t cells | healthy | 3 | 0.2544 | 0.2149 to 0.271 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cd4 t cells | healthy | 3 | 0.8054 | 0.803 to 0.8488 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.9282 | 0.9271 to 0.9464 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.6925 | 0.6694 to 0.7576 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cd4 t cells | healthy | 3 | 1.007 | 0.993 to 1.009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted cd4 t cells | healthy | 3 | 0.996 | 0.9927 to 1.064 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted cd8 t cells | healthy | 3 | 0.9904 | 0.8851 to 1.029 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted cd8 t cells | healthy | 3 | 0.7611 | 0.687 to 0.7708 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted cd8 t cells | healthy | 3 | 0.3162 | 0.311 to 0.376 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted cd8 t cells | healthy | 3 | 1.126 | 0.9843 to 1.212 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted cd8 t cells | healthy | 3 | 1.011 | 0.9095 to 1.051 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted cd8 t cells | healthy | 3 | 0.2761 | 0.2645 to 0.3183 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted cd8 t cells | healthy | 3 | 1.022 | 0.9044 to 1.07 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | cd8 t cells | sorted cd8 t cells | healthy | 3 | 0.9518 | 0.8917 to 0.9755 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | cd8 t cells | sorted cd8 t cells | healthy | 3 | 0.7574 | 0.6837 to 0.7668 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted cd8 t cells | healthy | 3 | 0.9959 | 0.9956 to 1.008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted cd8 t cells | healthy | 3 | 1.096 | 1.004 to 1.111 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t central memory cd4 | healthy | 3 | 0.9235 | 0.9164 to 0.9559 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t central memory cd4 | healthy | 3 | 0.6953 | 0.6772 to 0.7339 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t central memory cd4 | healthy | 3 | 0.2305 | 0.2281 to 0.2311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t central memory cd4 | healthy | 3 | 1.078 | 1.062 to 1.133 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t central memory cd4 | healthy | 3 | 0.9492 | 0.9435 to 0.9794 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t central memory cd4 | healthy | 3 | 0.1941 | 0.1915 to 0.2017 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t central memory cd4 | healthy | 3 | 0.9679 | 0.9666 to 0.9936 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t central memory cd4 | sorted t central memory cd4 | healthy | 3 | 1.059 | 1.022 to 1.06 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t central memory cd4 | sorted t central memory cd4 | healthy | 3 | 0.7851 | 0.7522 to 0.8234 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t central memory cd4 | healthy | 3 | 0.9998 | 0.9987 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted t central memory cd4 | healthy | 3 | 1.006 | 0.9416 to 1.007 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t effector cell cd8 | healthy | 3 | 0.9766 | 0.9545 to 1.012 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t effector cell cd8 | healthy | 3 | 0.7194 | 0.7034 to 0.777 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t effector cell cd8 | healthy | 3 | 0.298 | 0.2788 to 0.3059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t effector cell cd8 | healthy | 3 | 1.127 | 1.079 to 1.198 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t effector cell cd8 | healthy | 3 | 1.001 | 0.981 to 1.035 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t effector cell cd8 | healthy | 3 | 0.2566 | 0.2397 to 0.2568 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t effector cell cd8 | healthy | 3 | 1.021 | 0.9613 to 1.08 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t effector cell cd8 | sorted t effector cell cd8 | healthy | 3 | 1.017 | 0.9159 to 1.04 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t effector cell cd8 | sorted t effector cell cd8 | healthy | 3 | 0.8229 | 0.8226 to 0.8616 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t effector cell cd8 | healthy | 3 | 0.9969 | 0.9944 to 0.9973 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted t effector cell cd8 | healthy | 3 | 1.011 | 1.009 to 1.059 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted t effector cell cd8 | healthy | 3 | 0.9783 | 0.9508 to 0.9929 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted t effector memory cd4 | healthy | 3 | 1.064 | 1.029 to 1.067 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted t effector memory cd4 | healthy | 3 | 0.9374 | 0.895 to 0.9503 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted t effector memory cd4 | healthy | 3 | 0.2425 | 0.2424 to 0.2529 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted t effector memory cd4 | healthy | 3 | 1.307 | 1.239 to 1.311 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted t effector memory cd4 | healthy | 3 | 1.082 | 1.051 to 1.086 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted t effector memory cd4 | healthy | 3 | 0.2074 | 0.2041 to 0.2162 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted t effector memory cd4 | healthy | 3 | 1.168 | 1.095 to 1.168 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | t effector memory cd4 | sorted t effector memory cd4 | healthy | 3 | 0.9412 | 0.9033 to 0.9879 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | t effector memory cd4 | sorted t effector memory cd4 | healthy | 3 | 0.7731 | 0.7676 to 0.7985 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted t effector memory cd4 | healthy | 3 | 0.9946 | 0.9943 to 0.9971 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted t effector memory cd4 | healthy | 3 | 1.047 | 0.9646 to 1.053 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted t effector memory cd4 | healthy | 3 | 0.9737 | 0.9627 to 0.9844 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted t effector memory cd4 | healthy | 3 | 0.9943 | 0.9845 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted effector memory cd8 t cells | healthy | 1 | 1.059 | 1.059 to 1.059 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted effector memory cd8 t cells | healthy | 1 | 0.7403 | 0.7403 to 0.7403 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted effector memory cd8 t cells | healthy | 1 | 0.282 | 0.282 to 0.282 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted effector memory cd8 t cells | healthy | 1 | 1.278 | 1.278 to 1.278 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted effector memory cd8 t cells | healthy | 1 | 1.08 | 1.08 to 1.08 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted effector memory cd8 t cells | healthy | 1 | 0.2444 | 0.2444 to 0.2444 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted effector memory cd8 t cells | healthy | 1 | 1.146 | 1.146 to 1.146 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | effector memory cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 0.9094 | 0.9094 to 0.9094 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | effector memory cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 0.7376 | 0.7376 to 0.7376 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted effector memory cd8 t cells | healthy | 1 | 0.9948 | 0.9948 to 0.9948 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted effector memory cd8 t cells | healthy | 1 | 1.081 | 1.081 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted naive cd4 t cells | healthy | 1 | 0.6156 | 0.6156 to 0.6156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted naive cd4 t cells | healthy | 1 | 0.4347 | 0.4347 to 0.4347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted naive cd4 t cells | healthy | 1 | 0.2087 | 0.2087 to 0.2087 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted naive cd4 t cells | healthy | 1 | 0.6316 | 0.6316 to 0.6316 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted naive cd4 t cells | healthy | 1 | 0.6429 | 0.6429 to 0.6429 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted naive cd4 t cells | healthy | 1 | 0.1819 | 0.1819 to 0.1819 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted naive cd4 t cells | healthy | 1 | 0.5788 | 0.5788 to 0.5788 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | naive cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 0.8329 | 0.8329 to 0.8329 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | naive cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 0.6715 | 0.6715 to 0.6715 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted naive cd4 t cells | healthy | 1 | 0.9969 | 0.9969 to 0.9969 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted naive cd4 t cells | healthy | 1 | 1.026 | 1.026 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted naive cd8 t cells | healthy | 1 | 0.5717 | 0.5717 to 0.5717 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted naive cd8 t cells | healthy | 1 | 0.4077 | 0.4077 to 0.4077 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted naive cd8 t cells | healthy | 1 | 0.2165 | 0.2165 to 0.2165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted naive cd8 t cells | healthy | 1 | 0.5793 | 0.5793 to 0.5793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted naive cd8 t cells | healthy | 1 | 0.5966 | 0.5966 to 0.5966 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted naive cd8 t cells | healthy | 1 | 0.1894 | 0.1894 to 0.1894 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted naive cd8 t cells | healthy | 1 | 0.527 | 0.527 to 0.527 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | naive cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 0.8031 | 0.8031 to 0.8031 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | naive cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 0.639 | 0.639 to 0.639 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted naive cd8 t cells | healthy | 1 | 0.9964 | 0.9964 to 0.9964 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted naive cd8 t cells | healthy | 1 | 1.011 | 1.011 to 1.011 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted nk cells | healthy | 3 | 0.8083 | 0.7711 to 0.9738 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted nk cells | healthy | 3 | 0.7086 | 0.7049 to 0.7502 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted nk cells | healthy | 3 | 0.2761 | 0.2735 to 0.2858 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted nk cells | healthy | 3 | 0.8751 | 0.83 to 1.11 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted nk cells | healthy | 3 | 0.8359 | 0.7965 to 1.004 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted nk cells | healthy | 3 | 0.2428 | 0.2248 to 0.2507 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted nk cells | healthy | 3 | 0.7951 | 0.7357 to 0.9988 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | nk cells | sorted nk cells | healthy | 3 | 0.9105 | 0.8962 to 0.9577 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | nk cells | sorted nk cells | healthy | 3 | 0.7077 | 0.7042 to 0.8328 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted nk cells | healthy | 3 | 1 | 0.9961 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted nk cells | healthy | 3 | 0.9064 | 0.8979 to 0.9222 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted nk cells | healthy | 1 | 1.046 | 1.046 to 1.046 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted monocytes | healthy | 3 | 0.7546 | 0.7446 to 0.793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted monocytes | healthy | 3 | 0.6662 | 0.6539 to 0.6944 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted monocytes | healthy | 3 | 0.1888 | 0.1816 to 0.2394 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted monocytes | healthy | 3 | 0.8463 | 0.8434 to 0.9066 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted monocytes | healthy | 3 | 0.7794 | 0.7703 to 0.821 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted monocytes | healthy | 3 | 0.1577 | 0.148 to 0.2054 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted monocytes | healthy | 3 | 0.7636 | 0.748 to 0.776 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | monocytes | sorted monocytes | healthy | 3 | 0.8997 | 0.8799 to 0.9179 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | monocytes | sorted monocytes | healthy | 3 | 0.7519 | 0.7425 to 0.7983 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted monocytes | healthy | 3 | 0.9987 | 0.9986 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted monocytes | healthy | 3 | 1.093 | 1.053 to 1.108 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted monocytes | healthy | 3 | 0.9953 | 0.9945 to 1.015 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon macrophages | healthy | 2 | 0.9761 | 0.9149 to 1.037 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon macrophages | healthy | 2 | 0.8187 | 0.7363 to 0.9011 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon macrophages | healthy | 2 | 0.2312 | 0.2179 to 0.2445 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon macrophages | healthy | 2 | 1.106 | 1.022 to 1.19 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon macrophages | healthy | 2 | 1.012 | 0.9519 to 1.073 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon macrophages | healthy | 2 | 0.2068 | 0.193 to 0.2206 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon macrophages | healthy | 2 | 0.9983 | 0.9169 to 1.08 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon macrophages | sorted colon macrophages | healthy | 2 | 0.8907 | 0.877 to 0.9045 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon macrophages | sorted colon macrophages | healthy | 2 | 0.8718 | 0.8715 to 0.8721 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon macrophages | healthy | 2 | 1.005 | 1.004 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon macrophages | healthy | 1 | 0.9072 | 0.9072 to 0.9072 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon macrophages | healthy | 2 | 0.9843 | 0.9497 to 1.019 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon macrophages | healthy | 2 | 0.977 | 0.9282 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon macrophages | healthy | 1 | 1.016 | 1.016 to 1.016 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted colon macrophages | healthy | 2 | 1.04 | 1.005 to 1.075 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon macrophages | healthy | 1 | 0.9282 | 0.9282 to 0.9282 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar macrophages | healthy | 2 | 0.7967 | 0.7777 to 0.8157 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar macrophages | healthy | 2 | 0.6649 | 0.6579 to 0.6719 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar macrophages | healthy | 2 | 0.1979 | 0.195 to 0.2007 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar macrophages | healthy | 2 | 0.8653 | 0.8419 to 0.8888 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar macrophages | healthy | 2 | 0.8322 | 0.8123 to 0.8522 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar macrophages | healthy | 2 | 0.1738 | 0.1671 to 0.1804 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar macrophages | healthy | 2 | 0.7796 | 0.7464 to 0.8127 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar macrophages | sorted lung alveolar macrophages | healthy | 2 | 0.8507 | 0.8403 to 0.861 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar macrophages | sorted lung alveolar macrophages | healthy | 2 | 0.8924 | 0.85 to 0.9347 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar macrophages | healthy | 2 | 1.001 | 0.9947 to 1.008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar macrophages | healthy | 2 | 1.008 | 0.9902 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar macrophages | healthy | 2 | 0.9617 | 0.9613 to 0.9621 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar macrophages | healthy | 1 | 0.9876 | 0.9876 to 0.9876 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung interstitial macrophages | healthy | 3 | 0.77 | 0.7416 to 0.8482 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung interstitial macrophages | healthy | 3 | 0.6771 | 0.5927 to 0.6937 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung interstitial macrophages | healthy | 3 | 0.2144 | 0.1926 to 0.2158 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung interstitial macrophages | healthy | 3 | 0.8487 | 0.8039 to 0.9319 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung interstitial macrophages | healthy | 3 | 0.801 | 0.7746 to 0.8837 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung interstitial macrophages | healthy | 3 | 0.1903 | 0.1694 to 0.1925 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung interstitial macrophages | healthy | 3 | 0.7783 | 0.7331 to 0.839 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung interstitial macrophages | sorted lung interstitial macrophages | healthy | 3 | 0.847 | 0.8444 to 0.86 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung interstitial macrophages | sorted lung interstitial macrophages | healthy | 3 | 0.7045 | 0.7043 to 0.7739 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung interstitial macrophages | healthy | 3 | 1.004 | 0.9972 to 1.009 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung interstitial macrophages | healthy | 3 | 0.9713 | 0.9351 to 0.9726 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung interstitial macrophages | healthy | 2 | 0.9833 | 0.9477 to 1.019 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung interstitial macrophages | healthy | 1 | 0.9977 | 0.9977 to 0.9977 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd8 t cells | sorted lung interstitial macrophages | healthy | 1 | 1.038 | 1.038 to 1.038 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung interstitial macrophages | healthy | 1 | 0.9858 | 0.9858 to 0.9858 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted b cells | healthy | 4 | 0.8772 | 0.7766 to 1.02 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted b cells | healthy | 4 | 0.6923 | 0.6262 to 0.8916 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted b cells | healthy | 4 | 0.278 | 0.2342 to 0.3159 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted b cells | healthy | 4 | 0.9792 | 0.8349 to 1.228 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted b cells | healthy | 4 | 0.9029 | 0.8009 to 1.038 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted b cells | healthy | 4 | 0.2703 | 0.216 to 2.86 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted b cells | healthy | 4 | 0.8361 | 0.7717 to 0.9282 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | b cells | sorted b cells | healthy | 4 | 0.88 | 0.8129 to 0.9554 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | b cells | sorted b cells | healthy | 4 | 0.6891 | 0.6235 to 1.872 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted b cells | healthy | 4 | 0.9989 | 0.9958 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted b cells | healthy | 4 | 0.9742 | 0.9616 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted memory b cells | healthy | 4 | 0.9918 | 0.9617 to 1.021 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted memory b cells | healthy | 4 | 0.8205 | 0.6983 to 0.9207 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted memory b cells | healthy | 4 | 0.2283 | 0.213 to 0.2434 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted memory b cells | healthy | 4 | 1.186 | 1.135 to 1.235 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted memory b cells | healthy | 4 | 1.013 | 0.9856 to 1.042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted memory b cells | healthy | 4 | 0.6399 | 0.1974 to 1.082 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted memory b cells | healthy | 4 | 0.9204 | 0.7758 to 1.069 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | memory b cells | sorted memory b cells | healthy | 4 | 0.8898 | 0.7875 to 1.015 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | memory b cells | sorted memory b cells | healthy | 4 | 1.366 | 0.726 to 1.977 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted memory b cells | healthy | 4 | 0.9975 | 0.993 to 0.9979 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted memory b cells | healthy | 2 | 0.9713 | 0.9545 to 0.9881 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tonsil palatine epithelium | healthy | 3 | 1.038 | 1.003 to 1.053 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tonsil palatine epithelium | healthy | 3 | 1.005 | 0.9838 to 1.037 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tonsil palatine epithelium | healthy | 3 | 0.1904 | 0.1894 to 0.1943 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tonsil palatine epithelium | healthy | 3 | 1.363 | 1.314 to 1.406 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tonsil palatine epithelium | healthy | 3 | 1.047 | 1.01 to 1.058 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tonsil palatine epithelium | healthy | 3 | 0.1709 | 0.1692 to 0.1729 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tonsil palatine epithelium | healthy | 3 | 1.209 | 1.16 to 1.236 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tonsil palatine epithelium | sorted tonsil palatine epithelium | healthy | 3 | 1.007 | 0.9596 to 1.013 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tonsil palatine epithelium | sorted tonsil palatine epithelium | healthy | 3 | 0.7305 | 0.7236 to 0.7506 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tonsil palatine epithelium | healthy | 3 | 1.003 | 0.999 to 1.003 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tonsil palatine epithelium | healthy | 3 | 0.9473 | 0.9037 to 0.9757 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tonsil palatine epithelium | healthy | 3 | 0.9731 | 0.9273 to 0.9901 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tonsil palatine epithelium | healthy | 3 | 0.9759 | 0.9466 to 1.041 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tonsil palatine epithelium | healthy | 3 | 1.07 | 1.008 to 1.081 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tonsil palatine epithelium | healthy | 3 | 0.9368 | 0.9103 to 0.9426 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.027 | 1.001 to 1.053 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.009 | 0.9796 to 1.039 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.1882 | 0.188 to 0.1883 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.353 | 1.287 to 1.418 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.033 | 1.012 to 1.054 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.1642 | 0.1624 to 0.166 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.182 | 1.113 to 1.251 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tonsil pharyngeal epithelium | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.9849 | 0.9446 to 1.025 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tonsil pharyngeal epithelium | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.9359 | 0.9115 to 0.9604 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.008 | 1.004 to 1.012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.9401 | 0.94 to 0.9401 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.007 | 1.006 to 1.008 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.9865 | 0.9685 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 1.025 | 0.9917 to 1.058 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tonsil pharyngeal epithelium | healthy | 2 | 0.9176 | 0.907 to 0.9281 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted tongue epithelium | healthy | 3 | 0.9204 | 0.9028 to 0.9246 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted tongue epithelium | healthy | 3 | 0.9027 | 0.8904 to 0.9156 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted tongue epithelium | healthy | 3 | 0.1974 | 0.1931 to 0.2017 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted tongue epithelium | healthy | 3 | 1.165 | 1.161 to 1.191 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted tongue epithelium | healthy | 3 | 0.9315 | 0.9102 to 0.9323 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted tongue epithelium | healthy | 3 | 0.1725 | 0.1702 to 0.1805 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted tongue epithelium | healthy | 3 | 1.031 | 1.019 to 1.055 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | tongue epithelium | sorted tongue epithelium | healthy | 3 | 1.009 | 0.9981 to 1.043 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | tongue epithelium | sorted tongue epithelium | healthy | 3 | 0.7077 | 0.7015 to 0.7384 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted tongue epithelium | healthy | 3 | 1.004 | 0.9983 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted tongue epithelium | healthy | 3 | 0.9637 | 0.9324 to 0.9821 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted tongue epithelium | healthy | 3 | 0.9825 | 0.9479 to 0.9938 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted tongue epithelium | healthy | 3 | 0.9992 | 0.9914 to 1.027 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted tongue epithelium | healthy | 3 | 1.096 | 1.091 to 1.109 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted tongue epithelium | healthy | 3 | 0.9117 | 0.8945 to 0.944 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung bronchus epithelium | healthy | 3 | 0.9269 | 0.9134 to 0.9645 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung bronchus epithelium | healthy | 3 | 0.8928 | 0.8816 to 0.9426 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung bronchus epithelium | healthy | 3 | 0.1914 | 0.1891 to 0.1968 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung bronchus epithelium | healthy | 3 | 1.163 | 1.153 to 1.243 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung bronchus epithelium | healthy | 3 | 0.9409 | 0.9261 to 0.9741 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung bronchus epithelium | healthy | 3 | 0.1701 | 0.1675 to 0.1803 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung bronchus epithelium | healthy | 3 | 1.035 | 1.025 to 1.098 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung bronchus epithelium | sorted lung bronchus epithelium | healthy | 3 | 1.004 | 1.003 to 1.025 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung bronchus epithelium | sorted lung bronchus epithelium | healthy | 3 | 0.6865 | 0.6662 to 0.717 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung bronchus epithelium | healthy | 3 | 1.001 | 0.9993 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung bronchus epithelium | healthy | 3 | 0.9407 | 0.9407 to 0.9958 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung bronchus epithelium | healthy | 3 | 1.017 | 0.9716 to 1.024 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung bronchus epithelium | healthy | 3 | 0.9814 | 0.9431 to 0.9893 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung bronchus epithelium | healthy | 3 | 1.049 | 1.042 to 1.087 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung bronchus epithelium | healthy | 3 | 0.8874 | 0.8663 to 0.9248 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted prostate epithelium | healthy | 3 | 0.9672 | 0.962 to 0.9673 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted prostate epithelium | healthy | 3 | 0.9523 | 0.9511 to 0.9652 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted prostate epithelium | healthy | 3 | 0.2006 | 0.1866 to 0.2165 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted prostate epithelium | healthy | 3 | 1.109 | 1.089 to 1.171 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted prostate epithelium | healthy | 3 | 0.9976 | 0.9804 to 0.9994 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted prostate epithelium | healthy | 3 | 0.1775 | 0.1598 to 0.1915 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted prostate epithelium | healthy | 3 | 0.9656 | 0.9539 to 1.018 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | prostate epithelium | sorted prostate epithelium | healthy | 3 | 0.9435 | 0.9218 to 1.015 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | prostate epithelium | sorted prostate epithelium | healthy | 3 | 0.7715 | 0.7617 to 0.7727 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted prostate epithelium | healthy | 3 | 0.997 | 0.9932 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted prostate epithelium | healthy | 3 | 0.9266 | 0.8991 to 0.9457 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted prostate epithelium | healthy | 3 | 1.004 | 0.9953 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted prostate epithelium | healthy | 3 | 0.9533 | 0.9245 to 1.048 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted prostate epithelium | healthy | 3 | 1.024 | 0.9644 to 1.042 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted prostate epithelium | healthy | 3 | 0.9161 | 0.8858 to 0.9285 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted bladder epithelium | healthy | 3 | 0.9588 | 0.938 to 0.9714 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted bladder epithelium | healthy | 3 | 0.9419 | 0.9139 to 0.9496 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted bladder epithelium | healthy | 3 | 0.1771 | 0.1664 to 0.18 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted bladder epithelium | healthy | 3 | 1.169 | 1.15 to 1.174 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted bladder epithelium | healthy | 3 | 0.9785 | 0.957 to 0.9949 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted bladder epithelium | healthy | 3 | 0.1549 | 0.1406 to 0.1576 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted bladder epithelium | healthy | 3 | 1.006 | 0.9968 to 1.026 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | bladder epithelium | sorted bladder epithelium | healthy | 3 | 0.9913 | 0.9857 to 1.014 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | bladder epithelium | sorted bladder epithelium | healthy | 3 | 0.7634 | 0.7579 to 0.8318 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted bladder epithelium | healthy | 3 | 0.992 | 0.9918 to 0.9965 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted bladder epithelium | healthy | 3 | 0.9545 | 0.8989 to 0.9756 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted bladder epithelium | healthy | 3 | 0.999 | 0.9522 to 1.01 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted bladder epithelium | healthy | 3 | 0.9311 | 0.8756 to 0.966 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted bladder epithelium | healthy | 3 | 1.039 | 1.032 to 1.047 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted bladder epithelium | healthy | 3 | 0.9701 | 0.9544 to 1 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted breast luminal epithelium | healthy | 3 | 0.8918 | 0.8678 to 0.9309 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted breast luminal epithelium | healthy | 3 | 0.8817 | 0.8383 to 0.9255 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted breast luminal epithelium | healthy | 3 | 0.2601 | 0.213 to 0.2928 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted breast luminal epithelium | healthy | 3 | 1.028 | 0.9681 to 1.057 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted breast luminal epithelium | healthy | 3 | 0.9107 | 0.897 to 0.9551 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted breast luminal epithelium | healthy | 3 | 0.2156 | 0.1879 to 0.2548 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted breast luminal epithelium | healthy | 3 | 0.8927 | 0.8466 to 0.8963 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | breast luminal epithelium | sorted breast luminal epithelium | healthy | 3 | 0.915 | 0.8661 to 0.9183 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | breast luminal epithelium | sorted breast luminal epithelium | healthy | 3 | 0.7667 | 0.7182 to 0.8101 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted breast luminal epithelium | healthy | 3 | 0.9965 | 0.9922 to 0.999 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted breast luminal epithelium | healthy | 3 | 0.8954 | 0.8722 to 0.9251 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted breast luminal epithelium | healthy | 3 | 0.9482 | 0.9469 to 1.017 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted breast luminal epithelium | healthy | 3 | 1.004 | 0.9468 to 1.029 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted breast luminal epithelium | healthy | 3 | 1.018 | 0.9566 to 1.026 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted breast luminal epithelium | healthy | 3 | 0.9476 | 0.9424 to 0.9639 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted breast basal epithelium | healthy | 3 | 0.8366 | 0.8034 to 0.8651 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted breast basal epithelium | healthy | 3 | 0.824 | 0.787 to 0.857 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted breast basal epithelium | healthy | 3 | 0.2607 | 0.256 to 0.2636 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted breast basal epithelium | healthy | 3 | 0.9807 | 0.931 to 1.001 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted breast basal epithelium | healthy | 3 | 0.8524 | 0.8193 to 0.8838 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted breast basal epithelium | healthy | 3 | 0.2215 | 0.2162 to 0.2245 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted breast basal epithelium | healthy | 3 | 0.8382 | 0.798 to 0.8548 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | breast basal epithelium | sorted breast basal epithelium | healthy | 3 | 0.899 | 0.8935 to 0.9014 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | breast basal epithelium | sorted breast basal epithelium | healthy | 3 | 0.7957 | 0.7804 to 0.8071 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted breast basal epithelium | healthy | 3 | 0.9994 | 0.9984 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted breast basal epithelium | healthy | 3 | 0.9194 | 0.8917 to 0.9215 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted breast basal epithelium | healthy | 3 | 0.9172 | 0.8861 to 0.9548 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted breast basal epithelium | healthy | 3 | 0.9788 | 0.9767 to 0.9937 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted breast basal epithelium | healthy | 3 | 0.9889 | 0.9209 to 1.012 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted breast basal epithelium | healthy | 3 | 1.029 | 1.027 to 1.033 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted lung alveolar epithelium | healthy | 3 | 0.9508 | 0.9423 to 0.9606 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted lung alveolar epithelium | healthy | 3 | 0.9457 | 0.9345 to 0.9505 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted lung alveolar epithelium | healthy | 3 | 0.2755 | 0.2703 to 0.2853 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted lung alveolar epithelium | healthy | 3 | 1.144 | 1.134 to 1.158 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted lung alveolar epithelium | healthy | 3 | 0.9666 | 0.9571 to 0.9764 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted lung alveolar epithelium | healthy | 3 | 0.2478 | 0.2393 to 0.257 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted lung alveolar epithelium | healthy | 3 | 1.012 | 1.008 to 1.032 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | lung alveolar epithelium | sorted lung alveolar epithelium | healthy | 3 | 0.9565 | 0.947 to 0.9648 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | lung alveolar epithelium | sorted lung alveolar epithelium | healthy | 3 | 0.8232 | 0.8185 to 0.8342 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted lung alveolar epithelium | healthy | 3 | 1.002 | 1.002 to 1.004 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted lung alveolar epithelium | healthy | 3 | 0.9694 | 0.9278 to 0.9759 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted lung alveolar epithelium | healthy | 3 | 0.9717 | 0.9384 to 0.9775 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted lung alveolar epithelium | healthy | 3 | 0.9525 | 0.9141 to 0.9563 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted lung alveolar epithelium | healthy | 3 | 1.085 | 1.012 to 1.095 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted lung alveolar epithelium | healthy | 3 | 0.9106 | 0.9042 to 0.9173 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric fundus epithelium | healthy | 3 | 0.9745 | 0.9207 to 0.9772 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric fundus epithelium | healthy | 3 | 0.9503 | 0.8894 to 0.9532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric fundus epithelium | healthy | 3 | 0.1842 | 0.1808 to 0.1957 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric fundus epithelium | healthy | 3 | 1.254 | 1.153 to 1.292 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric fundus epithelium | healthy | 3 | 0.9827 | 0.9343 to 0.9845 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric fundus epithelium | healthy | 3 | 0.1624 | 0.1621 to 0.1751 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric fundus epithelium | healthy | 3 | 1.11 | 1.023 to 1.149 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric fundus epithelium | sorted gastric fundus epithelium | healthy | 3 | 0.9544 | 0.925 to 0.9597 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric fundus epithelium | sorted gastric fundus epithelium | healthy | 3 | 0.7017 | 0.6897 to 0.7171 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric fundus epithelium | healthy | 3 | 0.9985 | 0.9971 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric fundus epithelium | healthy | 3 | 0.9109 | 0.892 to 0.9248 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric fundus epithelium | healthy | 3 | 0.9566 | 0.9502 to 0.9846 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric fundus epithelium | healthy | 3 | 0.9435 | 0.9326 to 0.9439 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric fundus epithelium | healthy | 3 | 1.03 | 1.023 to 1.08 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric fundus epithelium | healthy | 3 | 0.9092 | 0.9048 to 0.9451 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric body epithelium | healthy | 3 | 0.9543 | 0.9137 to 0.9603 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric body epithelium | healthy | 3 | 0.9334 | 0.8867 to 0.9352 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric body epithelium | healthy | 3 | 0.1862 | 0.1702 to 0.1952 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric body epithelium | healthy | 3 | 1.233 | 1.14 to 1.253 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric body epithelium | healthy | 3 | 0.961 | 0.928 to 0.9705 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric body epithelium | healthy | 3 | 0.1655 | 0.1484 to 0.1747 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric body epithelium | healthy | 3 | 1.092 | 1.005 to 1.092 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric body epithelium | sorted gastric body epithelium | healthy | 3 | 0.9479 | 0.9403 to 0.9689 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric body epithelium | sorted gastric body epithelium | healthy | 3 | 0.7101 | 0.7057 to 0.7334 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric body epithelium | healthy | 3 | 1.002 | 0.9959 to 1.005 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric body epithelium | healthy | 3 | 0.8689 | 0.8645 to 0.9313 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric body epithelium | healthy | 3 | 0.9908 | 0.9443 to 1.015 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric body epithelium | healthy | 3 | 0.9364 | 0.9178 to 0.9449 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric body epithelium | healthy | 3 | 1.082 | 1.042 to 1.107 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric body epithelium | healthy | 3 | 0.9306 | 0.9302 to 0.9356 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted gastric antrum epithelium | healthy | 3 | 0.9196 | 0.9152 to 0.9507 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted gastric antrum epithelium | healthy | 3 | 0.8927 | 0.8895 to 0.9266 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted gastric antrum epithelium | healthy | 3 | 0.1827 | 0.173 to 0.1921 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted gastric antrum epithelium | healthy | 3 | 1.182 | 1.142 to 1.24 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted gastric antrum epithelium | healthy | 3 | 0.9304 | 0.9291 to 0.959 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted gastric antrum epithelium | healthy | 3 | 0.1645 | 0.1547 to 0.1689 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted gastric antrum epithelium | healthy | 3 | 1.044 | 1.004 to 1.095 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | gastric antrum epithelium | sorted gastric antrum epithelium | healthy | 3 | 0.9684 | 0.943 to 0.9742 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | gastric antrum epithelium | sorted gastric antrum epithelium | healthy | 3 | 0.6921 | 0.6913 to 0.7155 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted gastric antrum epithelium | healthy | 3 | 0.9999 | 0.9952 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted gastric antrum epithelium | healthy | 3 | 0.8792 | 0.8607 to 0.8927 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted gastric antrum epithelium | healthy | 3 | 0.9823 | 0.9255 to 0.9884 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted gastric antrum epithelium | healthy | 3 | 0.9555 | 0.9486 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted gastric antrum epithelium | healthy | 3 | 1.101 | 1.061 to 1.116 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted gastric antrum epithelium | healthy | 3 | 0.9281 | 0.9119 to 0.9519 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon epithelium | healthy | 3 | 0.8413 | 0.8295 to 0.9036 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon epithelium | healthy | 3 | 0.8321 | 0.8228 to 0.9042 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon epithelium | healthy | 3 | 0.2768 | 0.1931 to 0.2785 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon epithelium | healthy | 3 | 1.038 | 0.9678 to 1.11 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon epithelium | healthy | 3 | 0.8549 | 0.8452 to 0.914 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon epithelium | healthy | 3 | 0.2383 | 0.1687 to 0.2419 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon epithelium | healthy | 3 | 0.9132 | 0.8408 to 0.9751 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon epithelium | sorted colon epithelium | healthy | 3 | 0.8883 | 0.8845 to 0.9622 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon epithelium | sorted colon epithelium | healthy | 3 | 0.7739 | 0.7691 to 0.8504 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon epithelium | healthy | 3 | 0.9979 | 0.9961 to 0.9992 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon epithelium | healthy | 3 | 0.9532 | 0.9384 to 0.9767 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon epithelium | healthy | 3 | 0.9517 | 0.9312 to 1.002 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon epithelium | healthy | 3 | 0.958 | 0.8986 to 1.006 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon epithelium | healthy | 3 | 1.117 | 1.081 to 1.145 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon epithelium | healthy | 3 | 0.9292 | 0.8996 to 0.936 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted colon enteroendocrine | healthy | 2 | 0.8605 | 0.8554 to 0.8656 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted colon enteroendocrine | healthy | 2 | 0.8434 | 0.8416 to 0.8453 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted colon enteroendocrine | healthy | 2 | 0.2076 | 0.2035 to 0.2117 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted colon enteroendocrine | healthy | 2 | 1.028 | 1.023 to 1.033 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted colon enteroendocrine | healthy | 2 | 0.8787 | 0.8732 to 0.8841 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted colon enteroendocrine | healthy | 2 | 0.1817 | 0.1756 to 0.1878 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted colon enteroendocrine | healthy | 2 | 0.9052 | 0.8918 to 0.9186 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | colon enteroendocrine | sorted colon enteroendocrine | healthy | 2 | 0.9142 | 0.9136 to 0.9148 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | colon enteroendocrine | sorted colon enteroendocrine | healthy | 2 | 0.8574 | 0.828 to 0.8867 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted colon enteroendocrine | healthy | 2 | 0.9996 | 0.9986 to 1.001 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted colon enteroendocrine | healthy | 2 | 0.9663 | 0.9569 to 0.9756 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted colon enteroendocrine | healthy | 2 | 0.9499 | 0.9396 to 0.9602 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted colon enteroendocrine | healthy | 2 | 0.9718 | 0.9703 to 0.9734 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted colon enteroendocrine | healthy | 2 | 1.12 | 1.117 to 1.124 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted colon enteroendocrine | healthy | 2 | 0.9162 | 0.9081 to 0.9242 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Met-A | Met-A whole sample | whole sample | sorted small intestine epithelium | healthy | 3 | 0.8735 | 0.8673 to 0.9103 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | identity sites of the cells found | sorted small intestine epithelium | healthy | 3 | 0.8672 | 0.8499 to 0.915 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | islands | sorted small intestine epithelium | healthy | 3 | 0.2551 | 0.2087 to 0.2743 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | domains | sorted small intestine epithelium | healthy | 3 | 1.056 | 1.049 to 1.107 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A whole sample, by place | the rest | sorted small intestine epithelium | healthy | 3 | 0.8912 | 0.881 to 0.922 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Territories | territory | islands | sorted small intestine epithelium | healthy | 3 | 0.2272 | 0.1866 to 0.2334 | no line set for this reading | [Territories.csv](Territories.csv) |
| Territories | territory | domains | sorted small intestine epithelium | healthy | 3 | 0.9366 | 0.93 to 0.9605 | no line set for this reading | [Territories.csv](Territories.csv) |
| IAM-A | IAM-A | small intestine epithelium | sorted small intestine epithelium | healthy | 3 | 0.9421 | 0.9398 to 0.9509 | no line set for this reading | [IAM_A.csv](IAM_A.csv) |
| Met-A | Met-A sorted cell (atlas) | small intestine epithelium | sorted small intestine epithelium | healthy | 3 | 0.7345 | 0.7254 to 0.8092 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| C-Score | C-Score whole genome | whole genome | sorted small intestine epithelium | healthy | 3 | 0.9959 | 0.9934 to 1 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | neutrophils | sorted small intestine epithelium | healthy | 3 | 0.9226 | 0.9199 to 0.9525 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | monocytes | sorted small intestine epithelium | healthy | 3 | 1.036 | 1.023 to 1.036 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | b cells | sorted small intestine epithelium | healthy | 3 | 0.9884 | 0.9696 to 0.9962 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | cd4 t cells | sorted small intestine epithelium | healthy | 3 | 1.134 | 1.085 to 1.149 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| C-Score | C-Score each cell | nk cells | sorted small intestine epithelium | healthy | 3 | 0.9091 | 0.9024 to 0.9705 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Coupling-A | coupling-A | b cells | sorted b cells | healthy | 1 | 1.127 | 1.127 to 1.127 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A | memory b cells | sorted memory b cells | healthy | 2 | 1.023 | 1.017 to 1.03 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| C-Score | C-Score each cell | memory b cells | sorted memory b cells | healthy | 2 | 0.9936 | 0.9661 to 1.021 | no line set for this reading | [C_Score.csv](C_Score.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | adipocytes | healthy | 3 | 0.9995 | 0.9722 to 1.029 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | aorta endothelium | healthy | 2 | 1.003 | 0.9352 to 1.07 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | vascular saphenous endothelium | healthy | 3 | 1.003 | 0.9929 to 1.005 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney glomerular endothelium | healthy | 3 | 0.9955 | 0.9913 to 1.012 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney tubular endothelium | healthy | 3 | 0.9876 | 0.9829 to 1.026 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar endothelium | healthy | 3 | 0.9979 | 0.9414 to 1.063 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreas endothelium | healthy | 3 | 1.001 | 0.9789 to 1.021 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon fibroblasts | healthy | 2 | 1.001 | 0.9648 to 1.037 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | heart fibroblasts | healthy | 3 | 0.9936 | 0.9893 to 1.015 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | striated muscle | healthy | 2 | 1 | 0.9975 to 1.002 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | smooth muscle | healthy | 3 | 1.013 | 0.9682 to 1.023 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cardiomyocyte | healthy | 3 | 0.999 | 0.9933 to 1.007 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | oligodendrocytes | healthy | 3 | 1.02 | 0.9583 to 1.029 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cortical neurons | healthy | 3 | 0.996 | 0.9715 to 1.032 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | hepatocyte | healthy | 3 | 1.005 | 0.9646 to 1.033 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic duct | healthy | 3 | 1.001 | 0.9947 to 1.005 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic acinar | healthy | 3 | 0.9961 | 0.9913 to 1.012 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic delta | healthy | 3 | 1.01 | 0.9706 to 1.023 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic beta | healthy | 3 | 0.9983 | 0.9976 to 1.004 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | pancreatic alpha | healthy | 3 | 0.9931 | 0.9927 to 1.012 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney glomerular epithelium | healthy | 2 | 1 | 0.9964 to 1.004 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | kidney tubular epithelium | healthy | 3 | 0.9921 | 0.9909 to 1.015 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | thyroid epithelium | healthy | 3 | 0.9985 | 0.9943 to 1.007 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | fallopien tubes epithelium | healthy | 3 | 0.9888 | 0.9418 to 1.069 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | endometrium epithelium | healthy | 3 | 1.005 | 0.9617 to 1.036 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | erythrocyte progenitors | healthy | 3 | 1.013 | 0.9638 to 1.028 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cd4 t cells | healthy | 3 | 1.009 | 0.9612 to 1.033 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | cd8 t cells | healthy | 3 | 1.033 | 0.9321 to 1.048 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t central memory cd4 | healthy | 3 | 0.9893 | 0.9785 to 1.029 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t effector cell cd8 | healthy | 3 | 0.9808 | 0.9731 to 1.041 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | t effector memory cd4 | healthy | 3 | 0.998 | 0.9717 to 1.03 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | nk cells | healthy | 3 | 0.9981 | 0.9814 to 1.02 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | monocytes | healthy | 3 | 0.9874 | 0.9734 to 1.036 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon macrophages | healthy | 2 | 1 | 0.9767 to 1.024 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar macrophages | healthy | 2 | 1 | 0.9997 to 1 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung interstitial macrophages | healthy | 3 | 1 | 0.9904 to 1.009 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | b cells | healthy | 3 | 1.006 | 0.9826 to 1.013 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | memory b cells | healthy | 2 | 1 | 0.9924 to 1.008 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tonsil palatine epithelium | healthy | 3 | 0.9982 | 0.9968 to 1.005 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tonsil pharyngeal epithelium | healthy | 2 | 1.002 | 0.9502 to 1.053 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | tongue epithelium | healthy | 3 | 1.008 | 0.9683 to 1.027 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung bronchus epithelium | healthy | 3 | 0.979 | 0.979 to 1.036 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | prostate epithelium | healthy | 3 | 0.9976 | 0.9748 to 1.027 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | bladder epithelium | healthy | 3 | 1.012 | 0.9633 to 1.029 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | breast luminal epithelium | healthy | 3 | 0.9949 | 0.9375 to 1.069 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | breast basal epithelium | healthy | 3 | 1.004 | 0.9819 to 1.016 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | lung alveolar epithelium | healthy | 3 | 1.007 | 0.9742 to 1.021 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric fundus epithelium | healthy | 3 | 1.004 | 0.9671 to 1.031 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric body epithelium | healthy | 3 | 0.9935 | 0.984 to 1.021 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | gastric antrum epithelium | healthy | 3 | 0.9843 | 0.9814 to 1.03 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon epithelium | healthy | 3 | 1.024 | 0.9443 to 1.041 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | colon enteroendocrine | healthy | 2 | 1 | 0.9845 to 1.016 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Coupling-A | coupling-A, own sample left out of the reference | small intestine epithelium | healthy | 3 | 1.006 | 0.9756 to 1.021 | no line set for this reading | [Coupling_A.csv](Coupling_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | adipocytes | healthy | 3 | 1.504 | 1.482 to 1.545 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | aorta endothelium | healthy | 2 | 1.366 | 1.334 to 1.397 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | vascular saphenous endothelium | healthy | 3 | 1.592 | 1.573 to 1.66 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney glomerular endothelium | healthy | 3 | 1.602 | 1.569 to 1.614 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney tubular endothelium | healthy | 3 | 1.648 | 1.544 to 1.716 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar endothelium | healthy | 3 | 1.623 | 1.601 to 1.625 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreas endothelium | healthy | 3 | 1.763 | 1.65 to 1.793 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon fibroblasts | healthy | 2 | 1.446 | 1.441 to 1.451 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | heart fibroblasts | healthy | 3 | 1.603 | 1.572 to 1.615 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | striated muscle | healthy | 2 | 1.39 | 1.378 to 1.401 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | smooth muscle | healthy | 3 | 1.72 | 1.505 to 1.74 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cardiomyocyte | healthy | 3 | 1.548 | 1.513 to 1.592 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | oligodendrocytes | healthy | 3 | 1.593 | 1.518 to 1.595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cortical neurons | healthy | 3 | 1.795 | 1.622 to 1.955 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | hepatocyte | healthy | 3 | 1.356 | 1.334 to 1.382 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic duct | healthy | 3 | 1.498 | 1.417 to 1.512 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic acinar | healthy | 3 | 1.5 | 1.381 to 1.546 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic delta | healthy | 3 | 1.725 | 1.592 to 1.727 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic beta | healthy | 3 | 1.461 | 1.427 to 1.47 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | pancreatic alpha | healthy | 3 | 1.594 | 1.55 to 1.603 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney glomerular epithelium | healthy | 2 | 1.328 | 1.286 to 1.37 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | kidney tubular epithelium | healthy | 3 | 1.575 | 1.535 to 1.615 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | thyroid epithelium | healthy | 3 | 1.56 | 1.56 to 1.569 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | fallopien tubes epithelium | healthy | 3 | 1.519 | 1.464 to 1.568 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | endometrium epithelium | healthy | 3 | 1.557 | 1.478 to 1.728 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | erythrocyte progenitors | healthy | 3 | 1.478 | 1.44 to 1.684 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cd4 t cells | healthy | 3 | 1.543 | 1.506 to 1.687 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | cd8 t cells | healthy | 3 | 1.686 | 1.537 to 1.743 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t central memory cd4 | healthy | 3 | 1.593 | 1.527 to 1.667 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t effector cell cd8 | healthy | 3 | 1.64 | 1.639 to 1.72 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | t effector memory cd4 | healthy | 3 | 1.647 | 1.635 to 1.705 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | effector memory cd8 t cells | healthy | 1 | 1.649 | 1.649 to 1.649 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | naive cd4 t cells | healthy | 1 | 1.48 | 1.48 to 1.48 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | naive cd8 t cells | healthy | 1 | 1.449 | 1.449 to 1.449 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | nk cells | healthy | 3 | 1.571 | 1.546 to 1.853 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | monocytes | healthy | 3 | 1.669 | 1.655 to 1.791 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon macrophages | healthy | 2 | 1.382 | 1.382 to 1.383 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar macrophages | healthy | 2 | 1.312 | 1.248 to 1.376 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung interstitial macrophages | healthy | 3 | 1.495 | 1.492 to 1.644 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | b cells | healthy | 3 | 1.506 | 1.406 to 1.597 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | memory b cells | healthy | 2 | 1.619 | 1.574 to 1.663 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tonsil palatine epithelium | healthy | 3 | 1.55 | 1.537 to 1.595 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tonsil pharyngeal epithelium | healthy | 2 | 1.4 | 1.362 to 1.438 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | tongue epithelium | healthy | 3 | 1.467 | 1.454 to 1.532 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung bronchus epithelium | healthy | 3 | 1.455 | 1.41 to 1.519 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | prostate epithelium | healthy | 3 | 1.584 | 1.558 to 1.585 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | bladder epithelium | healthy | 3 | 1.573 | 1.558 to 1.712 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | breast luminal epithelium | healthy | 3 | 1.633 | 1.531 to 1.729 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | breast basal epithelium | healthy | 3 | 1.654 | 1.622 to 1.678 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | lung alveolar epithelium | healthy | 3 | 1.578 | 1.569 to 1.601 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric fundus epithelium | healthy | 3 | 1.487 | 1.461 to 1.516 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric body epithelium | healthy | 3 | 1.511 | 1.498 to 1.56 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | gastric antrum epithelium | healthy | 3 | 1.477 | 1.471 to 1.521 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon epithelium | healthy | 3 | 1.609 | 1.598 to 1.773 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | colon enteroendocrine | healthy | 2 | 1.342 | 1.295 to 1.389 | no line set for this reading | [Met_A.csv](Met_A.csv) |
| Met-A | Met-A sorted cell (atlas), sequencing units (band not met) | small intestine epithelium | healthy | 3 | 1.537 | 1.519 to 1.689 | no line set for this reading | [Met_A.csv](Met_A.csv) |

4815 readings from 26 sources. Status and record for every row are in the tables.

**Where Normal begins and ends:** [Bands.csv](Bands.csv) lists, for each instrument, reading, cell and platform, the band the gauge drew
(copied from the bundles) beside the spread of the healthy readings themselves (2.5th to 97.5th percentile, with how many healthy samples).
