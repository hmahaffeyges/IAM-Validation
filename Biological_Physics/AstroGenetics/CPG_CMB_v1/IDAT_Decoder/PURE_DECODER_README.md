# Pure-Python IDAT decoder (no methylprep / no pinned env)

`idat_parse.py` + `idat_decoder_pure.py` decode an Illumina IDAT pair (Grn+Red)
to per-CpG beta using only the standard library, numpy, and pandas — no
methylprep/minfi, no pinned Python/numpy/pandas. This removes the chain's most
fragile dependency (methylprep 1.7.1 requires Python 3.10/3.11 + numpy<2 +
pandas<2; it will not run on a modern host) so the decode step cannot break on a
version bump in the field.

## Validation
Decoded GSM3228562 (EPIC) and compared to the trusted methylprep raw beta:
865,918 probes, mean|diff| = 0.00000, max = 0.00000, r = 1.000000 — bit-exact.

## Use
```python
from idat_decoder_pure import decode_to_beta
beta, meta = decode_to_beta(grn_path, red_path, manifest_csv)   # -> pd.Series {cg: beta}
```
`beta` feeds run_pipeline directly (the Stage-1 calibrated-beta entry).

## The one data dependency: the array manifest
A static manifest CSV with columns Name, AddressA_ID, AddressB_ID,
Infinium_Design_Type, Color_Channel. EPIC and HM450 use the same columns, so the
same code handles both — pass the matching manifest. This is DATA shipped with
the instrument, not a software dependency that can break.

## Scope (current)
Produces beta (the chain's critical input). Stage-0 QC byproducts the older
methylprep-wrapping `idat_decoder.py` lists (poobah detection-p, control-probe
intensities, chrX/Y sex intensities) are the next increment; per-address bead
counts are already available from the parser. `idat_decoder.py` is retained for
reference, unused at runtime.
