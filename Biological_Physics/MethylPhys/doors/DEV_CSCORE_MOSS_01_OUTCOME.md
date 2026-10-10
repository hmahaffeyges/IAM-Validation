# DEV-CSCORE-MOSS-01 — outcome (2026-10-10; development)

Read by the sealed rule (`data/DEV_CSCORE_MOSS_01/insilico_moss_01.py readmix`; rows `moss_mixes_01_rows.csv`), 9 real mix arrays, chain Stage 1
and `conductor_v3.run_neutrophil(specimen="constructed DNA mixture")`, untared.

| mix | tissue | A | C | C in silico (median) |
|---|---|---|---|---|
| leukocytes alone | 0 | 1.0035 | 0.9981 | — |
| Mix15 | 4 % liver | 1.0129 | 1.0645 | 0.9963 |
| Mix17 | 6 % colon | 1.0088 | 1.1369 | 0.9938 |
| Mix16 | 8 % lung | 1.0024 | 1.0893 | 1.0086 |
| Mix11 | 3.5 % liver, 5 % colon | 1.0124 | 1.0666 | 1.0029 |
| Mix13 | 5 % lung, 3.5 % colon | 1.0061 | 1.0372 | 1.0058 |
| Mix9 | 10 % liver, 3.5 % lung | 1.0127 | 1.1365 | 1.0390 |
| Mix14 | 10 % lung, 3.5 % neurons | 1.0058 | 1.0639 | 1.0132 |
| Mix10 | 5 % liver, 10 % neurons | 1.0039 | 1.1080 | 1.0165 |
| Mix12 | 10 % colon, 5 % neurons | 1.0058 | 1.0320 | 1.0078 |

**Bar 1 (Met-A specificity): met, 9/9.** Every mix Normal and within 0.010 of the leukocyte array.
**Bar 2 (C-score specificity): not met, 6/9** (3 above 1.10). The rise does not follow the tissue dose (6 % colon 1.137; 15 % colon plus neurons
1.032) and the in-silico builds of the same mixes stay near 1, so it is array-to-array variation of the untared C, the same open problem as the
C-score repeat bar. The C-score's same-run tare (DEV-CSCORE-TARE-01) needs >= 3 same-run healthy references; this series has one.
