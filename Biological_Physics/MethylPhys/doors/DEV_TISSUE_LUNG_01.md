# DEV-TISSUE-LUNG-01 — Met-A on lung tumours against the same patient's normal lung (development; design, 2026-10-12; NOT sealed)

**Data.** GSE94785 (Kettunen, Hernandez-Vargas et al., Int J Cancer 2017, PMID 28722770): Illumina 450K, raw IDATs, 28 men with lung cancer, each
tumour and the same patient's normal lung, plus 6 controls. Never read by any version of the instrument (CANON/data_spent.csv, repository search).
**Layout** (`data/DEV_TISSUE_LUNG_01/samples_lung_01.py`, output `layout_lung_01_output.txt`): every tumour shares a slide with its own normal,
the design of DEV-TISSUE-COLON-01. The controls sit on a slide of their own.

**Reading (as DEV-TISSUE-COLON-01 and -BREAST-01).** Pair ratio R = Met-A(tumour) / Met-A(own normal, same slide) at composition-blind sites. GEO
gives no tumour type, so the blind sites are those where BOTH atlas lung epithelia (alveolar and bronchial) agree with every other panel cell:
neither tumour type nor tissue mix can move them.

**Inputs, before any synthetic test** (`fetch_lung_01.py`, `calib_lung_01.py`): tumour arrays refused before any request until
`sealed_lung_01.json` is committed unchanged; lock tested three ways (refused with no seal, refused with an uncommitted seal, allowed after a
committed one; HEAD restored). Normal and control arrays only until the seal.

**Before any seal:** synthetic tests on the real normals (spread, composition, independence, planted 5 %, power, doubled slide offset), as COLON-02.
**Not a milestone:** design (development).
