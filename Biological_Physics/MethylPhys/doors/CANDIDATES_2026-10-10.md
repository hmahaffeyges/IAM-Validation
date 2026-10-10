# Public candidates checked 2026-10-10 (search only; nothing downloaded; all public GEO with IDATs)

| set | design | what the chain can read | limit |
|---|---|---|---|
| GSE268145 | BJ fibroblasts (hTERT), serial passage, two trajectories, PDL 4–38+, 66 EPIC arrays | Met-A on the fibroblast's own identity sites, read against its own early passages (development; commissioned Met-A is neutrophils only) | arrays only: no IAM-A; hTERT immortalised |
| GSE225944 | methionine replaced by homocysteine; MeWo (methionine independent) and A101D (dependent) melanoma, 16 EPIC | same, each line against its own control (the independent line is the built-in negative control) | cancer lines, arrays only, no healthy reference |
| GSE297935 | blood, 51 BWS (IC2 loss, some multi-locus), 16 controls, 67 EPIC | Met-A on whole blood (commissioned path) | C-score cannot see it: 0 of the 6,000 identity sites lie in IC2, IC1, MEST, PLAGL1, GNAS or PEG3 (EPIC v1 manifest, hg19). Imprinted sites read ~0.5 and the site rule excludes them |
| GSE237503 | blood leukocytes, 13 BWS, 2 SRS, 4 controls, EPIC | as above | as above |

The imprinting sets are a negative control for Met-A (a few loci lost; the identity reading should stay Normal), not a C-score test.
A C-score test needs clustered change AT identity sites: the Moss in vitro mixes (DEV-CSCORE-MOSS-01) are being built in silico first.

## Second search, 2026-10-10 afternoon (search only; nothing downloaded)
**Opposite SAM lever (GNMT knockout, high liver SAM): no public methylome.** GNMT sets on GEO are expression arrays only (GSE9809,
GSE34838, GSE63027, GSE63062); the SRA bisulfite hits for the gene name (PRJNA193508) are Schwann cells.

**Fatty liver and inflammation without the Mat1a SAM loss** (separates the steatohepatitis route from the SAM route of DEV-SAM-LEVER-01):
| set | design | read level | to check before any design |
|---|---|---|---|
| GSE233768 | C57BL/6J, high-fat high-cholesterol high-fructose diet 28 weeks (NASH) vs normal diet, WGBS, 8 libraries | yes (WGBS) | liver SAM in this model; group sizes |
| GSE85772 | fast-food diet with and without exercise, liver RRBS, 2-3 replicates (PMID 27858497) | yes (RRBS) | SAM; whether steatohepatitis is present |
| GSE231727 | high-fat and alcohol diet, hepatocyte RRBS, 8 libraries (PMID 37415213) | yes (RRBS) | alcohol lowers SAM, so a second SAM route |

**The copier's own kinetics (bears on the central open problem: deriving φ from restore and loss rates):**
| set | design |
|---|---|
| GSE131098 | Hammer-seq (PMID 32581343): HeLa, EdU pulse, hairpin bisulfite on parent and daughter strands of newly copied DNA, chased to 24 h; per-CpG maintenance kinetics on single molecules |
| GSE116482 | DNMT1-only cells (PMID 32690947): imprecise DNMT1 maintenance and weak de novo activity measured at base resolution |
Both measure, on single molecules, the restore rate that sets the holding energy E_hold = k_BT ln(g/f). A prediction of the healthy copy
error from measured kinetics alone, with nothing taken from the copy error itself, would be the physics-derived number the cellular rung lacks.
