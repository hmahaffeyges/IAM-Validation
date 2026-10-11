# Neuron check: public sorted human neuron methylation data (metadata only)

Checked 2026-10-11 (Pacific). Sources: GEO series and sample text records, GEO RAW.tar filelists, SRA run tables, the S3 key list (13,998 keys), CANON/data_spent.csv (186 rows, read from the repo), and the methods sections of Loyfer 2023, Rizzardi 2019 and Mendizabal 2019. No sample-level values were opened and no reads were downloaded. Per-sample and per-series rows are in neuron_check.csv (700 rows).

## Limits (read first)

1. **Loyfer donor count is provisional.** GEO gives sample codes, not donor IDs. The 8 Cortex-Neuron samples carry 6 distinct age/sex pairs (19M, 25M x3, 38M, 39M, 56M, 65F), so they come from 6 to 8 donors. The real count is in Loyfer Supplementary Table 1, which I did not read. The same applies to whether the neuron and oligodendrocyte samples share donors.
2. **The Moss-Loyfer same-donor match is inferred.** It rests on age, sex and source (KI) matching, not on donor IDs.
3. **Feinberg (GSE96612) donor age, sex and PMI are not in GEO.** They are in the paper's Supplementary Table 1, which I did not read. The paper states the 6 donors were "not affected with neurological or psychiatric disease".
4. **Depth figures are raw SRA bases divided by 3.1 Gb**, before trimming and deduplication. Read them as upper bounds. Mendizabal reports >25x per sample after processing. Loyfer reports a mean of at least 30x across the whole atlas; per-sample values were not read.
5. **For the Mill sets (GSE234520, GSE279509), diagnosis is not in GEO.** Neither series' healthy status is established.
6. **Donor overlap between the Exeter (Mill) and Maastricht (Pishva) sets was not checked.** Both draw on UK brain-bank cohorts (BDR / UKBBN).
7. **The search was GEO, SRA and ENA only.** Synapse/PsychENCODE, NDA, EGA and figshare were not searched beyond what GEO records point to. Those are controlled access in any case, except the Tian 2023 figshare pseudobulk noted in project memory, which is single-nucleus data and hg38.
8. **S3 and spent status.** No candidate accession string appears in any S3 key. Three GSE186458 sample IDs do appear, but only as granulocyte result files (GSM5652313-15 under results/DEV_Q0_HEALTHY_01/). No Loyfer neuron or oligodendrocyte sample is in S3. The register lists only GSE122126 (Moss), marked spent. Loyfer and Lister are not in the register, but project memory records that both were already used (see the table).

## 1. IAM-A: WGBS of sorted healthy neurons

| Rank | Series | Lab (rule 9) | Healthy neuron donors | Region | Library / reads | Raw reads | Verdict |
|---|---|---|---|---|---|---|---|
| Reference | GSE186458 (Loyfer) | HU Kaplan/Dor; brain from H. Druid, Karolinska autopsies | 10 neuron samples: 8 Cortex-Neuron, 1 "Neuron", 1 Cerebellum-Neuron; 9 M, 1 F, ages 19-65; 6-8 cortical donors (provisional) | cortex (region unspecified) | Swift Accel-NGS Methyl-Seq, 2x150, NovaSeq; mean >=30x | FASTQ only at EGA EGAS00001006791 (controlled). Public hg19 and hg38 .pat.gz (250-322 MB each) and .beta on GEO | **Best healthy reference.** Already used to build references (atlas v2 .beta; PROC-CHANNEL-01 .pat windows), so it cannot serve as a test |
| 1 (test) | GSE96612 (superseries GSE96615) | JHU Feinberg (Rizzardi, Hickey, Hansen, Feinberg) | 6 donors, no neurological or psychiatric disease (paper); 23 NeuN+ libraries: BA9 6, HC 6, NAcc 6, BA24 5; 22 matching NeuN- | 4 regions; UMB NeuroBioBank | NEBNext + post-ligation bisulfite, 2x125, HiSeq 2500; NeuN+ median 64 Gb (~21x raw) | SRA public, SRP101887. NeuN+ total 680 GB; BA9 alone 188 GB (6 runs) | **Best independent IAM-A C4 lab.** Shares 0 contributors with Loyfer. FANS NeuN, purity >95% on re-sort. Needs FASTQ, then the pinned hg19 pipeline. Age, sex and PMI only in the paper supplement |
| 2 (test) | GSE107729 (superseries GSE108066) | GaTech Yi (sole GEO contributor) | 25 control NeuN+ donors (18 M / 7 F, ages 25-80; PMI 10-25 h for 20 of them); 20 of these also have OLIG2+ | BA46 DLPFC; several brain banks | in-house WGBS, 2x151, HiSeq X Ten; median 284 Gb raw | SRA public, SRP126677. 25 control NeuN+ = 3.0 TB | Strong backup or second test lab. Controls come from a schizophrenia case-control study. Large transfer |
| weak | GSE47966 (Lister 2013) | Salk Ecker | 2 donors (53F UMB_579, 55M UMB_797); NeuN+ and NeuN- | dorsal PFC | MethylC-Seq, SE101, HiSeq 2000; ~108 Gb per NeuN+ sample | SRA public, SRP026048. Processed allC is **hg18**, which explains the 2026-09-28 mapping failure | Fewer than 3 donors. Usable only by realigning the FASTQ |
| weak | GSE50852 | MSSM Dracheva | 3 normal males; neuron and non-neuron | OFC | ERRBS, SE50, ~9 Gb | SRA public | Reduced representation, 3 donors |
| weak | GSE109559 | Wuerzburg Haaf (shares Kraus and Giese with GSE66351, so same lab by rule 9) | 3 females; NeuN+ and NeuN- | BA10 | RRBS, SE76, GA II, ~0.8 Gb | SRA public | Too shallow; reduced representation |

**Excluded:**
- GSE330111 (Dracheva/Mukamel, 2026): WGBS + oxBS of Glu and MGE-GABA neurons from 103 donors. FASTQ is under NDA controlled access; GEO holds only per-cytosine counts (hg38).
- GSE339321: Nanopore native calls, so not bisulfite or EM chemistry.
- GSE235818: RRoxBS; the submitter states no raw data were deposited.
- GSE110728, GSE145153 and GSE112524: targeted padlock or SeqCap capture, not genome-wide.
- GSE137364: RRBS of unsorted caudate.
- GSE88716: the neurons are mouse.
- ENCODE motor-neuron WGBS series: neurons derived in vitro.

**No EM-seq or PBAT data from sorted human neurons was found.**

## 2. Met-A: Illumina arrays of sorted healthy neurons (chain v3 reads EPIC v1 only)

| Rank | Series | Lab | Platform / IDAT | Healthy neuron arrays | Notes |
|---|---|---|---|---|---|
| 1 | GSE306226 + GSE306227 (CEAM) | Maastricht Pishva (Lunnon, Smith, Harvey are Exeter co-authors) | EPIC v1, IDATs (1.35 + 1.27 GB) | 20 (BDR, PFC) + 19 (UKBBN, cingulate); design states "neurologically healthy" | **Best array lab.** Neurons, oligodendrocytes, microglia and astrocytes from the same donors. Shares 0 GEO contributors with Exeter-Mill, but the institutional link means possible donor overlap |
| 2 | GSE112179 (superseries GSE112525) | Labrie (Toronto/VARI) | EPIC v1, IDATs (1.64 GB) | 33 control NeuN+ arrays from 28 donors (of 100 arrays); PMI and age given | **Second independent array lab.** Same donors as the GSE112524 SeqCap data, which is targeted, so there is no WGBS link |
| 3 | GSE234520 / GSE279509 | Exeter Mill/Hannon | EPIC v1, IDATs (1.65 / 11.2 GB) | 28 NeuN+ arrays (27 donors) plus 9 SATB2+; 182 NeuN+ arrays (178 donors) | Diagnosis not in GEO; healthy subset must be resolved from the papers. Also has SOX10+ and NeuN-/SOX10- arrays from the same donors |
| link | GSE122126 (Moss 2018) | HU Kaplan/Dor (same lab as Loyfer) | 1 x 450K (31M), 2 x EPIC (19M, 38M), IDATs | 3 normal cortical neurons | Age and sex match Loyfer Neuron-Z000000TH (31M), Cortex-Neuron-Z000000TD (19M) and Cortex-Neuron-Z000000TF (38M), so probably the same donors (provisional). **The only same-donor neuron pair across array and WGBS.** Spent, same lab as the reference, and only 2 EPIC donors |

**450K with IDATs, refused by chain v3:**
- GSE66351 (Saarland): 16 control and 15 AD neurons.
- GSE98203 (Dracheva): 29 control neurons, OFC.
- GSE119532 (Karolinska): 7 control donors, NeuN+ from white matter, BS and oxBS.

**No IDATs:**
- GSE41826: RAW.tar holds manifests only.
- GSE50798.
- GSE156996.

**Not adult:** GSE289184, SATB2+/- from fetal and child donors.

## 3. Neurons and glia from the same donors; neuron-derived tumours

**WGBS (genome-wide), neurons plus glia from the same donors:**
- GSE96612: NeuN+ and NeuN-, 6 donors.
- GSE107729: NeuN+ and OLIG2+, 41 donors, 20 of them controls.
- Loyfer: neurons plus 4 oligodendrocyte samples whose age/sex pairs match neuron samples (provisional).
- GSE47966: 2 donors.

**Reduced representation (RRBS/ERRBS):** GSE50852 and GSE109559, 3 donors each.

**Arrays:** GSE306226/7 (four cell types), GSE234520 and GSE279509 (NeuN+/SOX10+/DN), and GSE66351 (450K).

**Neuron-derived tumours, arrays only:**
- GSE288337: central neurocytoma, 133 arrays, mixed 450K/EPIC/EPIC v2, IDATs.
- GSE331022: sellar neurocytoma, 20 arrays.
- GSE218542: ganglioglioma, 86 arrays on 450K.

No public WGBS of neuron-derived tumours was found. Neuroblastoma sets exist but derive from peripheral neural crest, not cortical neurons.

## Missing

- A second laboratory with **public FASTQ** from sorted healthy neurons and donor age/sex/PMI in GEO. Only Yi (GSE107729) has all of these, and its controls come from a disease study. Feinberg's donor metadata is in the paper only.
- **Same-donor EPIC v1 + WGBS neurons outside the Loyfer/Moss lab.** None was found. The Moss pair has 2 EPIC donors, is spent, and is the same lab as the reference.
- Any EM-seq or PBAT data from sorted human neurons.
- Public raw reads for the largest recent neuron WGBS set (GSE330111, 103 donors). It is under NDA controlled access.
- WGBS of neuron-derived tumours.

## Next steps

1. Read Loyfer Supplementary Table 1 to get donor IDs, per-sample depth and the neuron-oligodendrocyte pairing, and to confirm the Moss donor match.
2. Read Rizzardi Supplementary Table 1 for donor age, sex and PMI.
3. Read the Hannon 2024 and 2025 methods to find the healthy subset of the Mill sets.
4. For IAM-A C4: a box job on GSE96612 BA9 NeuN+, 6 runs, 188 GB, run through the pinned bwa-meth / wgbstools hg19 pipeline. Optionally add HC or NAcc for within-donor region replicates.
