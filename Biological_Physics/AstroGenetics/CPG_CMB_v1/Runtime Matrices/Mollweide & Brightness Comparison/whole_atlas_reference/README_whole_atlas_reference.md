# Whole-450K Cosmic Methylome Background reference

`iamatlas_whole450k_reference.npz` — the single full-methylome reference (all 483,092 atlas
CpGs) the patient's WHOLE-ATLAS brilliance map is compared against. This is distinct from the
8 per-class references (Plate 1): the per-class panels compare patient-vs-class (8 panels);
the whole-atlas map compares the WHOLE patient methylome vs the WHOLE 450K reference (1 map).

Arrays (atlas row order, aligned to iamatlas_cpg_to_healpix_nside128.npy):
- `mean`    per-CpG posterior mean beta, averaged across the 8 class references (nan-aware)
- `sd`      per-CpG posterior sd, averaged across the 8 class references
- `cpg_ids` the 483,092 atlas CpG ids

Built from the 8 class brightness CSVs in IAM_Atlas/iamatlas_class_archives/. Coverage 100%
(every CpG covered by >=1 class). The patient whole-atlas departure is z=(beta_patient-mean)/sd
projected onto the NSIDE=128 grid as one Mollweide.
