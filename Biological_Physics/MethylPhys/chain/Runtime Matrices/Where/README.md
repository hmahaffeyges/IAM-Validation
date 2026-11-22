# Where: the positions reference (v1)

Built by `build_where_ref.py` from public files (UCSC hg19 sequence, cpgIslandExt, refGene) and the chain's sequencing key and Law Sphere units. `where_units_v1.parquet`: every Law Sphere unit (islands, then 50-CpG blocks) with chromosome, hg19 start and end, CpG count, array probe count (0 = arrays blind), the UCSC CpG island, the nearest gene by transcription start and the signed distance to it. `where_regions_v1.parquet`: the same for every Sequencing Globe region (1,000 consecutive CpG indexes). `where_ref_v1.json`: sizes, sources and the key check (all 813,951 array probes on their recorded chromosome and position).

Sources: UCSC Genome Browser hg19 (Kent et al. 2002, Genome Res. 12:996-1006); the CpG index numbering of wgbstools (Loyfer et al. 2023, Nature 613:355-364).
