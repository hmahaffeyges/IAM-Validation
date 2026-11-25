# Where: the positions reference (v1)

Built by `build_where_ref.py` from public files (UCSC hg19 sequence, cpgIslandExt, refGene) and the chain's sequencing key and Law Sphere units. `where_units_v1.parquet`: every Law Sphere unit (islands, then 50-CpG blocks) with chromosome, hg19 start and end, CpG count, array probe count (0 = arrays blind), the UCSC CpG island, the nearest gene by transcription start and the signed distance to it. `where_regions_v1.parquet`: the same for every Sequencing Globe region (1,000 consecutive CpG indexes). `where_ref_v1.json`: sizes, sources and the key check (all 813,951 array probes on their recorded chromosome and position).

Sources: UCSC Genome Browser hg19 (Kent et al. 2002, Genome Res. 12:996-1006); the CpG index numbering of wgbstools (Loyfer et al. 2023, Nature 613:355-364).

## The Locator's gene relation (v2)

Built by `build_where_genes_v2.py` from v1 and public UCSC hg19 files (refGene, knownGene with kgXref, cpgIslandExt). `where_units_v2.parquet` and `where_regions_v2.parquet` add to every place: the genes whose promoter it lies in (transcription start +/- 2 kb), the genes whose body it lies in, the nearest gene and signed distance, the relation (promoter, gene body, intergenic), each from NCBI RefSeq and from UCSC known genes, and the CpG-island context (island; shore, within 2 kb; shelf, 2-4 kb; open sea). The definitions follow Koldobskiy et al. 2021 (Nature Biomedical Engineering, Methods), so gene lists compare directly with that study's. `where_ref_v2.json`: definitions, sources and counts, and the check that v1's nearest gene agrees at every place.

Sources: UCSC Genome Browser hg19 (Kent et al. 2002, Genome Res. 12:996-1006); NCBI RefSeq (O'Leary et al. 2016, Nucleic Acids Res. 44:D733-D745); UCSC known genes (Hsu et al. 2006, Bioinformatics 22:1036-1046).
