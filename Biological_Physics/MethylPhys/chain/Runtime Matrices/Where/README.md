# Where: the positions reference (v1)

Built by `build_where_ref.py` from public files (UCSC hg19 sequence, cpgIslandExt, refGene) and the chain's sequencing key and Law Sphere units. `where_units_v1.parquet`: every Law Sphere unit (islands, then 50-CpG blocks) with chromosome, hg19 start and end, CpG count, array probe count (0 = arrays blind), the UCSC CpG island, the nearest gene by transcription start and the signed distance to it. `where_regions_v1.parquet`: the same for every Sequencing Globe region (1,000 consecutive CpG indexes). `where_ref_v1.json`: sizes, sources and the key check (all 813,951 array probes on their recorded chromosome and position).

Sources: UCSC Genome Browser hg19 (Kent et al. 2002, Genome Res. 12:996-1006); the CpG index numbering of wgbstools (Loyfer et al. 2023, Nature 613:355-364).

## The Locator's gene relation (v2)

Built by `build_where_genes_v2.py` from v1 and public UCSC hg19 files (refGene, knownGene with kgXref, cpgIslandExt). `where_units_v2.parquet` and `where_regions_v2.parquet` add to every place: the genes whose promoter it lies in (transcription start +/- 2 kb), the genes whose body it lies in, the nearest gene and signed distance, the relation (promoter, gene body, intergenic), each from NCBI RefSeq and from UCSC known genes, and the CpG-island context (island; shore, within 2 kb; shelf, 2-4 kb; open sea). The definitions follow Koldobskiy et al. 2021 (Nature Biomedical Engineering, Methods), so gene lists compare directly with that study's. `where_ref_v2.json`: definitions, sources and counts, and the check that v1's nearest gene agrees at every place.

Sources: UCSC Genome Browser hg19 (Kent et al. 2002, Genome Res. 12:996-1006); NCBI RefSeq (O'Leary et al. 2016, Nucleic Acids Res. 44:D733-D745); UCSC known genes (Hsu et al. 2006, Bioinformatics 22:1036-1046).

## The healthy reference and measured null (edit 259)

`where_rates_v1_part1.npz` to `where_rates_v1_part3.npz`: each Law Sphere unit's healthy relative rate on the loss side (held errors) and the gain side (stray marks), for 57 cell types over 237,850 units, split by groups of 19 cell types (every part holds every unit) so each file stays under the repository's upload limit. `where_null_v1.json`: the measured null, in bins of expected count, from 156 healthy atlas libraries, each read against the reference built without it (leave one out): each bin's variance inflation and upper-tail quantiles.

Built by `build_where_null.py build` (then `split`) from the per-unit counts the one button writes with `--law-counts-out` for each healthy GSE186458 library (Loyfer et al. 2023; molecule files: Zenodo 10.5281/zenodo.24518180). Read by `stage_where.py`, which the one button runs after the Law Sphere.

| File | SHA-256 |
|---|---|
| where_rates_v1_part1.npz | c0bd371ea48e580e59a8b02c1ad0e073b55f07b00c1167b794609366445c3fc3 |
| where_rates_v1_part2.npz | c17c72f081037781d97ea54283f61cda472dd8ae1eb561358dadc4df9485ddd9 |
| where_rates_v1_part3.npz | 1157d17af0b1afb1d7d35e098be50094ac8ca7c4588ae8efe042277dea7f91b8 |
| where_null_v1.json | 1059ff2fbed78b37ca52b250176f67f65e03873de7a049abb4eee4e5b3b484a1 |
