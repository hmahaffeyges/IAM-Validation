# Reference data - the calibrated arrays the instrument constants were measured on

These are the Stage-1 calibrated arrays the pipeline map, the presence floors and the detection panel were measured on, published beside the constants rather than described.
A reader does not have to re-run Stage 1 to check a number: load the file, apply the formula, compare.

| file | laboratory | arrays | loci | what was measured from it |
|---|---|---|---|---|
| [`stage1_betas_GSE87571.pkl.xz`](stage1_betas_GSE87571.pkl.xz) | Uppsala | 80 (40 build panel + 40 held out) | 125,323 | pipeline map (fitted here; Uppsala is the map's source), presence floors (with the other three, 160 arrays), detection panel (12 arrays); the retired September layers (record) |
| [`stage1_betas_GSE42861.pkl.xz`](stage1_betas_GSE42861.pkl.xz) | Karolinska | 78 | 125,263 | pipeline-map transfer check (median 1.0097), presence floors (160-array panel), detection panel (12 arrays); the retired September layers (record) |
| [`stage1_betas_GSE111629.pkl.xz`](stage1_betas_GSE111629.pkl.xz) | UCLA | 80 | 123,913 | presence floors (160-array panel), detection panel (12 arrays); the retired September layers (record) |
| [`stage1_betas_GSE125105.pkl.xz`](stage1_betas_GSE125105.pkl.xz) | Munich | 80 | 115,691 | presence floors (160-array panel), detection panel (12 arrays) - refused at intake since 2026-09-27 (low-signal input; FINDING_GSE125105_LOW_SIGNAL.md); the retired September layers (record) |

All four were processed through **one** pipeline - [`stage_1_idat_calibration.py`](../chain/stage_1_idat_calibration.py) (methylprep noob) - from the raw
IDATs fetched from GEO, which is why the constants measured on them transfer. Each file has a manifest beside it
carrying the sample list, the build/held-out split and the seed, the locus count, the Stage 1 call and a sha256 of
the pickle. Loading: `pickle.load(lzma.open(path,"rb"))` gives a DataFrame of beta, loci x samples.

The arrays are public GEO submissions; no identifiers beyond the GSM accessions are held here. GSE125105 (Munich) is retained as the record of a low-signal laboratory (`../doors/FINDING_GSE125105_LOW_SIGNAL.md`); the intake gate refuses its arrays.
