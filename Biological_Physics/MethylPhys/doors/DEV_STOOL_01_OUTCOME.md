# DEV-STOOL-01 — can stool carry enough colon-lining molecules for IAM-A and Met-A? (development, 2026-10-02)

**Run.** Box 1. (a) Marker regions for lower-GI epithelium (colon + small intestine treated as one cell type, since stool selects both) against the
Loyfer 2023 atlas, two backgrounds: every cell type ("all") and the cells that shed into stool ("stool": blood, immune, gut stroma, liver, squamous).
Both polarities searched. (b) 2-kb tile scan of the early-onset CRC tumour/normal pairs (6 usable pairs) for where the tumour's copy-error rise sits.

**(a) Markers.**
| background | strict regions | loose regions | CpGs |
|---|---|---|---|
| all cell types | 0 | 4 | 30 |
| stool-relevant cells | 1 (13 CpGs) | 28 | 214 + 13 |
All found regions are methylated in lower-GI epithelium (median β 0.79–0.89) and low in the background (0.10–0.19); the opposite polarity returned none.
Merging colon with small intestine and narrowing the background to stool-relevant cells raises the usable set from 1 region (6 CpGs, earlier run)
to 29 regions, 227 CpGs. That is enough sites to pull molecules from a deep stool library; whether enough tumour molecules survive in stool is not
assessed here (needs a stool methylome).

**(b) Tumour tiles: not assessed at this depth.** Only 80 tiles reached ≥ 20 qualifying molecules in both tissues in ≥ 5 pairs; 64 of them are on
chr1/chr16/chr10 pericentromeric repeats. Pooled over those tiles the tumour copy error is 1.16× normal (0.0539 vs 0.0466), the same direction as
PROC-TUMOUR-01, but where the rise sits genome-wide cannot be read from these libraries. Needs deeper WGBS or pooling at gene/region level.

**Next.** Score the 29 stool-background regions in the tumour pairs (region-level pooling instead of 2-kb tiles); look for a public stool or
colonic-effluent methylome to count lower-GI molecules per gram.
