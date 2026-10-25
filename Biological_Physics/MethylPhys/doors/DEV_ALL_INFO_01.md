# DEV-ALL-INFO-01: the published information-theory study of pediatric leukemia, replicated through the chain (development)

**The instrument's job:** the chain reads each leukemia and each control by itself, against its own reference and the physics floor, and
the readings are set beside what the study found by comparing groups.

**The study:** an information-theory analysis of whole-genome bisulfite sequencing in pediatric pre-B acute lymphoblastic leukemia
(Nature Biomedical Engineering 2021). Each 3-kb region was modeled with a one-dimensional Ising model fitted to the reads (informME), giving
a mean methylation level, a normalized methylation entropy (0 ordered to 1 random) and a methylation sensitivity index, and leukemias were
compared with normal controls (sorted pre-B2 precursors from an earlier study, and cord-blood CD19+ B cells) by the Jensen-Shannon distance.

**What it found, written here before the chain reads any of its samples:**
1. Leukemia epigenomes are more stochastic than the controls: higher normalized methylation entropy at specific regulatory regions and
   genes.
2. Gains in mean methylation, gains in entropy and losses of methylation sensitivity inside the bivalent domains of normal B cells, across
   four cytogenetic subtypes.
3. Higher promoter entropy goes with lower gene expression and with higher expression variability.
4. The genes with discordant methylation converge on genes involved in the leukemia's translocations, with UHRF1 as a central node.
5. Knocking out UHRF1 in the Reh line lost methylation and raised entropy genome-wide; in gene-poor heterochromatic B domains, where
   methylation went almost to zero, the entropy fell instead.

**Data spent (author: "Spend the whole series"):** GEO GSE116229, all 36 WGBS samples with reads (31 leukemias, the Reh control and UHRF1 knockout,
3 cord-blood CD19+ samples; `data/DEV_ALL_INFO_01/all_samples.csv`). The study's pre-B2 controls are controlled-access and are not used (public data only); the
controls here are its 3 public cord-blood samples. Its RNA sequencing is not read.

**Pass 1:** IAM-A's count on every sample at the eraser recount's fixed 10 million read pairs (`erase_gh.sh`, unchanged), each through
`run_sample.py`. Compared with findings 1 and 5: do the leukemias read beyond the cord-blood controls on the held error and loose share,
and does the knockout. **Pass 2:** full depth through a split alignment, then the chain's regional readings (Met-A on sequencing, the
territories, the globe, the distance to the ceiling) set region by region beside findings 2 to 5.

## Pass 2: the split alignment (written and checked before any full-depth sample is aligned)

A full-depth sample (about 150 to 220 million read pairs) does not fit one GitHub machine's 6-hour limit, so its read pairs are cut into up to
20 contiguous ranges (`data/DEV_ERASE_01/erase_blocks.py`, `chunk`: every pair exactly once, across run boundaries), each aligned on its
own machine by the committed pipeline unchanged (`erase_gh.sh`, `CHUNK`, `PATONLY`), then the chunks' molecules merged
(`merge_pats.py`, a streaming merge in genome order, identical molecules summed) and read once by the one command through the pipeline's
saved-molecules route (`.github/workflows/split_align.yml`). **Checks:** the chunk plan covers 157 of 157 pairs of a two-run example once; a
saved molecule file split into three sorted pieces and merged is byte-identical to the original (4,216,305 lines), and a piece merged with
itself doubles its molecules exactly. **Disclosed:** duplicates are marked within each chunk; a duplicate pair whose copies fall in two
chunks is kept. **First real test:** the Reh control (GSM4914798) at full depth, against its pass-1 count at 10 million pairs.

**The aligner index, built once.** Every alignment job spent about 6,000 s rebuilding the same hg19 + lambda + pUC19 index (pass-1 timings:
6,006, 6,276 and 6,425 s of about 9,500 s). `build_index.yml` builds it once through the same block of `erase_gh.sh` and keeps it as release
files with the SHA-256 of the whole stream; each job streams it in and checks it, and builds it itself if the release is missing or the
check fails. The index is the same either way, so alignments are unchanged.

## Pass 1, read

`data/DEV_ALL_INFO_01/pass1_01/all_pass1_summary.txt` (scored by `pass1_score.py`, written before the counts; 34 of 36 samples here, the Reh
pair counted earlier in DEV-ERASE-01 at the same depth). Each sample is counted against its own molecules at 10 million read pairs; no
sample is compared with a control to make its reading.

| Group | n | held error, median | loose share, median |
|---|---|---|---|
| leukemias | 31 | 0.04121 | 0.1208 |
| cord-blood CD19+ B cells | 3 | 0.03668 | 0.0900 |
| Reh control (DEV-ERASE-01) | 1 | 0.03216 | 0.0490 |
| Reh, UHRF1 knocked out (DEV-ERASE-01) | 1 | 0.09407 | 1.0 |

- **Beside finding 1 (leukemias more stochastic):** the leukemias' held error is higher than the cord-blood B cells' (Mann-Whitney p 0.022,
  with only 3 controls); the loose share does not separate the groups (p 0.24) but spreads widely across the leukemias (0.015 to 0.558; six
  above 0.2), the samples where whole molecules have left the held class.
- **Beside finding 5 (the knockout):** knocking out UHRF1 multiplies the held error 2.9 times and leaves no tile held, the largest change in
  the series.
- **Not yet compared:** findings 2 to 4 are regional (bivalent domains, promoters, translocation genes); they need pass 2 (full depth and the
  regional readings). The controls are cord-blood B cells, not the study's sorted pre-B2 precursors (controlled access), so the comparison
  is with a related, not the same, normal cell.
