# Reference data - the calibrated betas every measured constant was fitted on

These are the arrays the healthy reference was measured on, published beside the constants rather than described.
A reader does not have to re-run Stage 1 to check a number: load the file, apply the formula, compare.

| file | laboratory | arrays | loci | what was measured from it |
|---|---|---|---|---|
| `stage1_betas_GSE87571.pkl.xz` | Uppsala | 80 (40 build panel + 40 held out) | 125,323 | laboratory zero, healthy band, per-entry references, sky residual scale |
| `stage1_betas_GSE42861.pkl.xz` | Karolinska | 78 | 125,263 | laboratory zero, healthy band, per-entry references, sky residual scale |
| `stage1_betas_GSE111629.pkl.xz` | UCLA | 80 | 123,913 | laboratory zero, healthy band, per-entry references, sky residual scale |
| `stage1_betas_GSE125105.pkl.xz` | Munich | 80 | 115,691 | laboratory zero, healthy band, per-entry references, sky residual scale |

All four were processed through **one** pipeline - `stage_1_idat_calibration.py` (methylprep noob) - from the raw
IDATs fetched from GEO, which is why the constants measured on them transfer. Each file has a manifest beside it
carrying the sample list, the build/held-out split and the seed, the locus count, the Stage 1 call and a sha256 of
the pickle. Loading: `pickle.load(lzma.open(path,"rb"))` gives a DataFrame of beta, loci x samples.

The healthy donors are public GEO submissions; no identifiers beyond the GSM accessions are held here.
