# Ready-to-test board (2026-10-11; keep current: update the stage the day it changes)

Stages: **0** in hand and chain-readable (IDAT, `data/READY_TO_TEST/s3_format_check.csv`) · **1** design written · **2** synthetic tests done
(planted loss AND gain, no-effect, composition, power) · **3** sealed · **4** scored and reviewed. One cancer at a time; nothing tumour-side is opened
before stage 3.

| set | what | stage | next step |
|---|---|---|---|
| ENCODE MCF-7, T47D (DEV-FINGERPRINT-02) | breast cancer lines vs breast normals | 3 | score one pair at a time when the box finishes |
| TCGA-BRCA (DEV-TISSUE-BREAST-01) | 91 patients' breast tumour and own normal tissue, 450K | 2 (partly) | null and spread on the 97 real normals; then seal |
| GSE131013 | colon: healthy donors, adjacent normal, tumour | 0 | check patient pairing; reuse the tissue composition design |
| GSE288652 | colon adenomas, low and high grade (pre-cancer) | 0 | after GSE131013: does Met-A rise before cancer? |
| GSE166212 | colorectal cancer, EPIC | 0 | check for normal tissue in the series |
| GSE304513 | liver tissue | 0 | read the design |
| cfDNA: GSE298812 liver, GSE277078 lung, GSE207998 breast, GSE119260 prostate, GSE292312 child brain, GSE282216, GSE287137, GSE303824-26, GSE311578, GSE325660 | plasma cell-free DNA, IDAT | 0 | needs a plasma healthy reference and composition (blood cells plus tissue); development track |
| AML GSE159907, GSE175758; CLL GSE136724, GSE295456; GSE237299 | blood cancers, IDAT | 0 | needs references for the cells involved (blasts, B cells); development track |
| GSE69914 (breast) | submitter betas only, no IDAT, no patient IDs | not chain-readable | set aside |
| GSE262273/4, GSE275017, GSE302507 (cfDNA) | .pat / tsv, not IDAT | read-level or tables | check whether IAM-A can read the .pat sets |
