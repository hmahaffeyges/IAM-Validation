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

**First split alignment, stopped by a script fault:** the Reh control's 20 chunks (run 40157833589) each stopped after the index step: a comment added to `erase_gh.sh` for the chunk mode had swallowed the two commands after it (`rm -rf $S/fq/*; i=0`), leaving the block counter unset. No chunk aligned; the merge refused (0 of 20). Fixed and sent again; no other alignment ran on that version of the script.

**The merge's file search:** with the molecule step fixed, all 20 of the Reh control's chunks wrote their molecules, but the merge found none: the chunk artifacts keep their `eout/` folder, so the files arrive under `chunks/eout/`, where the merge did not look. Fixed (the merge searches the whole download), and a merge-only mode added (`chunksfrom`: merge an earlier run's chunks without aligning again), so the aligned chunks were merged, not realigned.

**Chunk 19's short fetch:** the merge-only run found 19 of 20 chunks: chunk 19's first read file came back short (10,559,762 reads against the second file's 10,998,061) with no error, and the pipeline stopped it as written (exit 30). Fixed: a short fetch is fetched once more; a rerun can align only named chunks (`onlychunks`) and the merge takes the rest from the earlier run (`chunksfrom`). Chunk 19 alone is realigned. **The knockout's chunk 19 stopped the same way** (its first read file short, 9,488,351 reads against 10,477,113), so it is not transient: the last stretch of each run holds spots whose first read is empty, which `--split-files` leaves out of the first file only. The fetch now uses `--split-3`, which writes only true pairs to the two files (for a fully paired stretch, such as every pass-1 range, the output is the same); both chunk 19s are realigned with it.

**Last-chunk fetch, matched pairs:** cord-blood control 3's last chunk (split alignment run 40185918188) came back with 7,060,310 read-1
records against 7,217,487 read-2 records, with no error, so the merge refused it, as it should. The fetch now checks each block: where its
two read files differ in length, it keeps the true pairs record by record (read names checked at the last kept record; it fails if they
differ) and logs how many unpaired records were dropped; the fetch's error log goes into the run log. Tested on constructed read files
before use. The dropped records are reported in each chunk's log.

**Pass 2, the first leukemia at full depth** (`data/DEV_ALL_INFO_01/pass2/GSM4914781_full_vs_pass1.txt`): ALL-45's held error at full depth
agrees with its pass-1 count at 10 million pairs within a few percent (values in the file), so the split alignment reproduces the count.
**Specimen:** the full reading ran with the pipeline's default specimen, plasma, because the sample list gave none. Each sample now carries
its stated specimen in `data/DEV_ALL_INFO_01/all_samples.csv`: the leukemias "not stated" (GEO records them only as primary diagnosis
samples), the cord-blood CD19+ B cells "sorted b cells" (read against the atlas's B cells), the Reh line and its knockout "cell line". Stage
0 intake now reads a cell line on the full-atlas route, flagged, as bone marrow is (read and flag); it refused cell lines before. Every
full-depth sample is read again from its merged molecules with its stated specimen; no alignment is repeated.

**Pass 2, the check on four samples** (`data/DEV_ALL_INFO_01/pass2/full_vs_pass1.txt`): the Reh control, the UHRF1 knockout, cord-blood
control 1 and ALL-45 at full depth each reproduce their pass-1 held error closely (ratios in the file), and the knockout still reads about
three times its control. The rest of the leukemias go out at full depth, four at a time; each full-depth sample is read again from its
merged molecules with its stated specimen.

**Re-reads from two runs:** a sample whose last chunk was re-sent has its chunks in two runs, so the split-alignment workflow takes a second
source run (`chunksalso`); the first re-reads of the Reh control, the knockout and cord-blood control 1 found 19 chunks missing and refused,
as they should.

**Sorted cells on sequencing:** cord-blood control 1's re-read as "sorted b cells" stopped at IAM-A's intake, because the sequenced cell
defaulted to neutrophils, and so read nothing after it. Three fixes: the sequenced cell now follows the stated specimen's atlas cell when it
is not given (`run_sample.py --seq-cell`); the globe names the atlas cell for a sorted, isolated or purified specimen and reads cell lines
and marrow as mixed; Met-A for each cell, for the whole sample and the territories now run for an isolated atlas cell too, against the cells
the IAM-Deconvolver finds. Tested on a small saved molecule file read as sorted B cells: IAM-A reads B cells and the globe is drawn; the
file was too shallow for the cell shares (marker sites withheld), so the atlas readings are checked on the control's own re-read.
**The Reh line and its knockout, re-read as cell lines** (full-atlas route): IAM-A's held error is unchanged from the full-depth count, the
knockout still about three times its control; the Deconvolver finds no single healthy cell type in a cultured line, as expected.

**Pass 1 against full depth, seven samples** (`data/DEV_ALL_INFO_01/pass2/full_vs_pass1.txt`, which replaces the four-sample table): four
samples reproduce their pass-1 held error closely; three (cord-blood controls 2 and 3, ALL-100) read 6 to 9 % lower at full depth. The
counter is not the cause: pass 1's own molecule files for cord-blood controls 1 and 3, recounted with today's chain, give exactly their
pass-1 values. Pass 1 read the first 10 million pairs of each run in the archive's order, so for some runs those first reads differ from the
run as a whole (as when lanes of different quality are stacked). **Full depth stands; pass 1 is superseded** for each sample as its full
depth lands, and the group comparison is made again on full depth. The earlier statement that full depth reproduces pass 1 held for four
samples only.

**No silent skips:** when the cell shares are withheld (too few marker sites measured), Met-A for each cell, the whole sample, the territories
and the C-Score for each cell now each state that they were not read and why, instead of being left out. Checked before any further sample
is sent: the pipeline script's own reading command run on a 10-million-pair molecule file under each specimen in the sample list (cell
line, not stated, sorted B cells); every reading is present or states its reason.

**The ceiling's A, captured** (`data/DEV_ALL_INFO_01/full_depth/full_depth_rows.csv`, written by `CANON/bundle_rows.py` from the newest
bundle of each full-depth sample; values copied, nothing computed): the samples read at full depth so far, each reading with its A and the A
its ceiling sits at; the ceiling's A is listed in the A-score database, kept apart from array readings (sequencing reads Met-A lower against the array-built atlas: the chemistry offset in the plan's Met-A-on-sequencing branch).
Rebuilt as each sample lands.

## A quarter of each run: quarter-depth test (written and committed before any quarter reading is made)

**The instrument's job:** a quarter of a sample's read pairs, spread across its run, reads the sample the same as all of them. If it does,
the remaining samples of this study are aligned at a quarter of the machine time (author: "whatever we can do to cut time and still have
a respectable presentation"), and the paper says so.
**Prior reading:** the 10 samples already read at full depth (5 leukemias, the Reh control and knockout, 3 cord-blood controls); their
chunks are re-read, nothing new is fetched. **Data:** `data/DEV_ALL_INFO_01/quarter_01/q4_runs.json` (each sample's full-depth run).
**Method:** `split_align.yml` merge only (chunksfrom that run, keepchunks 0,4,8,12,16: 5 of the 20 contiguous chunks), the same merge
and the same one command as full depth (no constructed data: the full-depth readings are the known answer).
**Synthetic test:** DEV_ALL_INFO_01/quarter_01/q4_synthetic.py, DEV_ALL_INFO_01/quarter_01/q4_synthetic.txt
**Bars (at most three):**
1. IAM-A within 0.02 of the full-depth reading, all 10 samples.
2. Met-A whole sample keeps its order across the 10 (rank correlation at least 0.9). *Amended before any quarter reading, disclosed:*
   as written, Met-A within 0.02 of full depth; the synthetic (below) read it 0.0218 lower at a quarter.
3. IAM-A's order across the 10 kept (rank correlation at least 0.9).
**Scorer:** `data/DEV_ALL_INFO_01/quarter_01/q4_score.py`.

**Quarter-depth synthetic, read** (`data/DEV_ALL_INFO_01/quarter_01/q4_synthetic.txt`; run 1, `q4_synthetic_run1.txt`, on a shallow file whose
quarter the sample-shape check withheld, which also showed the one command's report stopping on the check's own wording, fixed at its
source in `stage_q2_selfref.py` and `stage_v4.py`). GSM4914785's 20 chunks merged (spent) and a random quarter of its molecules, both
read by the one command: **IAM-A unchanged (0.8124 against 0.8122); Met-A whole sample 0.0218 lower at a quarter**, past the 0.02 bar. Met-A
on sequencing falls with fewer reads per site (a site read once or twice looks fully held or fully bare, so it carries less entropy), so
bar 2 is amended to the order across samples, and **every sample of this study is read at the same depth**: if the test passes, the 10
already read at full depth are re-merged from 5 of their chunks, and the remaining samples are aligned on those 5 chunks only.

