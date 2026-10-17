# DEV-DECONV-V21-01: the atlas v2.1 deconvolution tool on real mixtures (release track B)

**The instrument's job:** given a mixed sample, report each cell's share with its error bar.

**Tool:** chain Stage D21 (`chain/stage_deconv_v21.py`): the atlas's best values and 95 % ranges, marker sites chosen per cell from the
atlas, a weighted non-negative fit with a laboratory term, and a 95 % bar per share from resampling the sites.

**Sealed 2026-10-16** (`data/DEV_DECONV_V21_01/sealed_deconv_v21.json`), before any mixture was built; the author chose the test set.
200 mixtures of the six blood cells (neutrophils, monocytes, B, CD4 T, CD8 T, NK), each built from real held-out arrays of laboratories
that built nothing, with shares drawn at random (seed 2026).

| bar | rule |
|---|---|
| B1 | mean error per cell at most 0.03 |
| B2 | the true share inside its 95 % bar for at least 80 % of shares |

Both met: commissioned on six-cell blood mixtures; otherwise development. Reported alongside, not scored: the Salas laboratory's wet-lab
DNA mixtures (same group as two build sources).

**Synthetic test:** chain/stage_deconv_v21.py, DEV_DECONV_V21_01/planted_deconv_v21.txt

The planted test on the six-cell panel PASSED first (pure cells read at least 0.993 of themselves; mixture error 0.003; a 5 % share seen in 20 of 20).

**Sources:** public: the atlas v2.1 compact copy in this repository (best values and 95 % ranges, built from public GEO, ENCODE and ArrayExpress data); public held-out GEO arrays of six sorted healthy blood cells (the DEV-ATLAS-USE-01 lists), which provide the mixture components; the Salas laboratory's published wet-lab mixtures, which provide known shares (reported only, not a test set).

**Reference sets:** atlas v2.1 (DEV_ATLAS_V21_01.md); held-out arrays from the sealed DEV-ATLAS-USE-01 lists (731 arrays, independent laboratories).

**Prior reading:** these held-out arrays were read once in DEV-ATLAS-USE-01 (marker and mixture tests against Moss's reference), and no mixture of them has been built for this tool; the Salas mixtures GSE110554 and GSE182379 were read in earlier composition tests (CANON/data_spent.csv) and are reported here, never scored.

**What it would take to succeed:** both bars met on the 200 sealed mixtures.

**In plain words:** real healthy blood cells from outside laboratories are mixed in known amounts by computer, and the tool must say how much of each is in the mix, with an honest error bar.

## Outcome (2026-10-16), scored once: B1 MET, B2 NOT MET

Box job e8e72ed7 (code 47c18ab3); planted test PASSED first on the box. 200 mixtures of 584 held-out arrays from independent laboratories
(`data/DEV_DECONV_V21_01/deconv_scored_v21.json`, `deconv_mixtures_v21.csv`).

| bar | result | verdict |
|---|---|---|
| B1 mean error per cell <= 0.03 | **0.029** (B 0.013, monocytes 0.018, neutrophils 0.026, CD4 T 0.032, NK 0.035, CD8 T 0.048) | MET |
| B2 true share inside its 95 % bar >= 80 % | **0.42** | NOT MET |

**What it means.** The shares are right to about 3 points on outside-laboratory cells. The error bars are too narrow: they come from
resampling sites, which measures how well the mix fits, but not how much one donor's or one laboratory's cells differ from the atlas's
average cell. The same gap U1 found. The tool stays development; the next step is an error bar that carries the measured donor and
laboratory spread, tested on mixtures not built from these arrays.

## Diagnostic after scoring (2026-10-16; nothing rescored, the outcome above stands)

Each pure held-out array read alone (box job 91ef36cf, `data/DEV_DECONV_V21_01/diag_pure_v21.csv`). In real mixtures CD8 T read 0.76 x true and
NK 1.16 x true. Read alone, pure cells from five of the six laboratories read as themselves at 0.93-1.00 (CD8: GSE130030 1.00, GSE87640 0.93;
CD4: four laboratories 0.95-0.99). One laboratory, **GSE59065**, reads CD8 0.74 and CD4 0.90, its T cells carrying neutrophil (0.06) and
monocyte (0.04) signal; it supplied 100 of the 133 CD8 and 99 of the 156 CD4 arrays in the mixture pool, and it is one of the three laboratories
behind most of U1's misses. Summing the atlas's CD8 subsets instead of the parent CD8 profile made CD8 worse (0.55), so it is not the
panel. **What it points to:** the sorted cells of one laboratory, not the atlas or the tool. The planted test with the atlas's donor and
laboratory spread planted (version 2) gives error 0.004 and covers 82 % of true shares; the old version covered 89 % on the same planted
data, so the spread terms were not the cause. Next: a sample-shape check that flags an array that does not read as its own cell, sealed
before use, then a fresh mixture test on E-MTAB-2145 (Cambridge; read once in U1, never mixed).
