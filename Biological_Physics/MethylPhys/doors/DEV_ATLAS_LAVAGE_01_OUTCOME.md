# DEV-ATLAS-LAVAGE-01 — outcome (2026-10-10; development)

`data/DEV_ATLAS_LAVAGE_01/score_lavage_01.py` on 70 GSE133062 arrays (chain Stage 1), rows `lavage_01_rows.csv`. One mechanical change before
scoring: counts recorded as 'na' are treated as missing (4 eosinophil counts); method and bars unchanged.

| cell | n | mean abs error vs count | bias (atlas − count) | within 0.05 | r |
|---|---|---|---|---|---|
| neutrophils | 70 | **0.053** | +0.049 | 31/70 | 0.45 |
| lymphocytes | 70 | 0.041 | +0.018 | 50/70 | 0.90 |
| macrophages | 70 | 0.087 | −0.070 | 24/70 | 0.87 |
| eosinophils | 66 | 0.006 | +0.003 | 66/66 | 0.05 |

**Bars: not met** (neutrophil mean error 0.053 > 0.02; 39 samples outside 0.05).

**What it shows.** The atlas moves 5–7 points of the macrophage signal onto neutrophils. Lymphocytes and macrophages track the counts (r 0.90,
0.87), but with offsets. The simulation predicted 0.010 and missed this, for two reasons it did not model: only 4,722 of the 8,000 atlas loci are on
the EPIC array (median), and the lavage macrophages are not the atlas's alveolar macrophage template (the simulation drew 'people' from the
atlas itself). The lung myeloid templates do not separate neutrophils from macrophages on real lavage arrays. That is a limit of atlas v2 in this
tissue; blood composition is a separate question and is not tested here. DEV-ATLAS-LAVAGE-02 (lymphocytes, second laboratory) stays sealed as
written.
