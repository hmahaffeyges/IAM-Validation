# Public candidates checked 2026-10-10 (search only; nothing downloaded; all public GEO with IDATs)

| set | design | what the chain can read | limit |
|---|---|---|---|
| GSE268145 | BJ fibroblasts (hTERT), serial passage, two trajectories, PDL 4–38+, 66 EPIC arrays | Met-A on the fibroblast's own identity sites, read against its own early passages (development; commissioned Met-A is neutrophils only) | arrays only: no IAM-A; hTERT immortalised |
| GSE225944 | methionine replaced by homocysteine; MeWo (methionine independent) and A101D (dependent) melanoma, 16 EPIC | same, each line against its own control (the independent line is the built-in negative control) | cancer lines, arrays only, no healthy reference |
| GSE297935 | blood, 51 BWS (IC2 loss, some multi-locus), 16 controls, 67 EPIC | Met-A on whole blood (commissioned path) | C-score cannot see it: 0 of the 6,000 identity sites lie in IC2, IC1, MEST, PLAGL1, GNAS or PEG3 (EPIC v1 manifest, hg19). Imprinted sites read ~0.5 and the site rule excludes them |
| GSE237503 | blood leukocytes, 13 BWS, 2 SRS, 4 controls, EPIC | as above | as above |

The imprinting sets are a negative control for Met-A (a few loci lost; the identity reading should stay Normal), not a C-score test.
A C-score test needs clustered change AT identity sites: the Moss in vitro mixes (DEV-CSCORE-MOSS-01) are being built in silico first.
