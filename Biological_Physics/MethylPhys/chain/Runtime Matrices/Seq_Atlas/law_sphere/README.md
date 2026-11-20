IAM's Law Sphere runtime reference (development, DEV-SPHERE-01; edit 204). Built on a Google spot machine from the IAM Sequencing Atlas pieces
(seq_sphere_runtime.py, register SEQSPHERE plan). Units: each UCSC hg19 CpG island is one unit; outside islands, blocks of 50 consecutive CpGs.
A unit is on the sphere when its across-type variance beats its own within-type noise (person-to-person SD from the pieces plus counting
b(1-b)/reads; chi-square 99 %, df 57): 237,850 units. units.npz: island bounds and block size. seq_sphere_ref_part*.parquet: per unit, its job
(territory: the family of the type where it departs most from its own level), and per type the pooled level and the share of its CpGs held.
positions.npz (edit 208): each unit's place on the sphere (the sequencing sphere's own-rule layout, construction v5), for the Law Sphere page.
