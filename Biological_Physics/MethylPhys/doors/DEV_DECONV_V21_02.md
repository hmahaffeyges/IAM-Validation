# DEV-DECONV-V21-02: the deconvolution tool, version 2, on fresh mixtures (release track B)

**The instrument's job:** given a mixed sample, report each cell's share with its error bar.

**Why a second test.** DEV-DECONV-V21-01 scored the first version once: shares right (mean error 0.029), error bars too narrow (0.42).
The diagnostic after it traced the misses to one laboratory's sorted T cells (GSE59065), not to the atlas. Version 2 reads each cell
with the spread the atlas itself fits (the average's uncertainty, the person-to-person spread and the laboratory spread, Stage A21's array
predictive), and first checks every sample: an instrument should know when it cannot read a sample.

**Sealed 2026-10-16** (`data/DEV_DECONV_V21_01/sealed_deconv_v21_02.json`), before any array of the test laboratory was read by the tool.
Sample check: a pure array must read at least 0.80 of its own cell, or it is listed and not used. 200 mixtures of the passing arrays of
E-MTAB-2145 (Cambridge; neutrophils, monocytes, B, CD4 T, CD8 T; 450K).

| bar | rule |
|---|---|
| B1 | mean error per cell at most 0.03 |
| B2 | the true share inside its 95 % bar for at least 80 % of shares |

Both met: commissioned on six-cell blood mixtures read on 450K; otherwise development.

**Synthetic test:** chain/stage_deconv_v21.py, DEV_DECONV_V21_01/planted_deconv_v2.txt

Version 2's planted test (the atlas's own person-to-person and laboratory spread planted) PASSED.

**Sources:** public: the atlas v2.1 compact copy in this repository (best values, ranges and the fit's person-to-person spread); E-MTAB-2145 on ArrayExpress (sorted healthy blood cells, 450K raw arrays), which provides the mixture components.

**Prior reading:** E-MTAB-2145's 30 arrays were read once in U1, each alone against its cell's range; none has been read by this tool or mixed.

**What it would take to succeed:** both bars met on the 200 sealed mixtures.

**In plain words:** healthy blood cells from a laboratory the tool has never seen are mixed in known amounts by computer; the tool must first confirm each sample looks like its own cell, then say how much of each cell is in every mix, with an honest error bar.

## Outcome: NOT MET (scored once, 2026-10-16, box job 41e7d7d7; `data/DEV_DECONV_V21_01/deconv_v2_scored.json`)

The planted test passed first on the box. Then the sealed test:

| bar | result | needed |
|---|---|---|
| B1 mean error | **0.016** (met) | at most 0.03 |
| B2 true share inside its bar | **0.78** (not met) | at least 0.80 |

**The sample check worked.** Neutrophils, CD4 T, CD8 T and monocytes passed (own-cell share 0.75-1.00; one monocyte array at 0.75 was
refused). All six B-cell arrays from this laboratory read only 0.29-0.68 B cells and were refused, so B cells were left out of the mixtures,
as the rule says. The tool told us it could not read those samples instead of reporting a wrong answer.

**Where the error bars are too narrow:** monocytes (true share inside the bar 0.38 of the time) and CD4 T cells (0.64). Neutrophils 0.93,
CD8 T 0.81. The shares themselves are accurate (error 0.016, half the bar).

**What it would take:** error bars that widen for monocytes and CD4 T cells, the two cells whose shares trade off most in blood. This is
development; the shares (B1) are the deliverable that already works, twice now (0.029 on the first test, 0.016 here).
