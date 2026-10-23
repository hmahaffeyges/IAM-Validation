# DEV-METAA-ATLAS-01: Met-A and the C-Score against IAM-Atlas v2.1 (development, written 2026-10-21 before any reading; not sealed)

**The instrument's job:** a healthy sample of any blood cell, read against that cell in IAM-Atlas v2.1 and tared against healthy samples of the
same cell run the same way, reads Normal; a planted loss of the cell's pattern moves it.

**Why (author, 2026-10-21).** Met-A is commissioned on neutrophils against purified arrays of that cell (6 healthy arrays). Purified arrays are
not available for most cell types, while IAM-Atlas v2.1 holds a healthy reference, with its uncertainty, for all 79. The atlas sits slightly
softer than one array (DEV-REFERENCES-01: at the neutrophil identity sites its entropy is -5 % to +12 % from the purified arrays, by cell), so no
single correction factor carries over; the same-run tare cancels any reference offset, whatever its size or sign. Readings made this way are
printed and flagged "read against IAM-Atlas v2.1; not commissioned by our standards", and the purified-cell route stays the one to test against
as purified arrays become available.

**Built from what, tested on what.** IAM-Atlas v2.1 was fitted on, among 21 sources, both purified-blood array studies (GSE110554, 35 arrays;
GSE167998, 56 arrays), including the 6 neutrophil arrays of Met-A's own reference. Those arrays are therefore not test data here. Test data:
- isolated cells: GSE184269, an independent laboratory not in the atlas; its sorted monocytes, B, CD4, CD8 and NK arrays are unread
  (CANON/data_spent.csv: partly spent, only its whole bloods and granulocytes read);
- whole blood: the Group-A healthy controls (GSE318173, not in the atlas), read both ways.

**Design, in order.**
1. Identity sites per cell, from the atlas alone, by one fixed rule written before reading (the neutrophil rule of Met-A, applied to each cell).
2. Synthetic first, through the chain: atlas-healthy samples of each cell must read 1; a planted 1 % loss must move the reading by the shift
   the chain predicts.
3. Isolated cells (GSE184269): each sorted array read against its atlas cell, tared against the other arrays of the same cell in that series.
4. Whole blood (Group-A): Met-A read against the atlas cells weighted by the person's own shares, and against the commissioned purified
   reference; reported side by side person by person (share in Normal, agreement). The C-Score likewise, with a planted clustered loss.

**What this is not.** Not a commissioning: the bars of a sealed test are written only when the exploration shows the route works ("We explore
WITHOUT sealing until we are certain"). Not a replacement of the commissioned neutrophil reading.

**Design additions (2026-10-21, before any reading; from the globe's change rule, ATL-063).** The report globe found that the atlas's 95 %
range is how sure the atlas is of the healthy average, not how much a healthy person varies (it flagged 84 % of a healthy blood), and that
each site's healthy width (the person-to-person spread, donor_sd, with the atlas's own uncertainty) is the right yardstick (2.1 % of a
healthy blood beyond 3 widths in its own spread). For this test:
1. **Identity sites by width.** Each cell's identity sites are chosen where that cell's healthy width is small, by one rule fixed before reading.
2. **Weighting by width, tested not assumed.** Met-A read with every site equal (as commissioned) and with sites weighted by one over the
   healthy width squared; both reported, synthetic first.
3. **No self-centering for Met-A.** Centering a sample on its own spread cancels a loss at every site (as the self-tare does,
   DEV-SELFTARE-03); Met-A's offset from the atlas is cancelled only by the same-run tare against healthy references of the same cell.
4. **The C-Score on the atlas.** The C-Score reads how change is laid out, not its size, so the globe's change map (each site in its healthy
   width, in the sample's own spread) is a whole-genome departure map against the atlas; the C-Score's clustering statistic runs on it
   unchanged, with healthy and planted clustered losses, and is compared with the commissioned C-Score on the same arrays.

## Synthetic test 1, run once (2026-10-21)

`data/DEV_METAA_ATLAS_01/synthetic_01.py`, committed before it ran (`cfc419fc`), through `chain/stage_metaa_atlas.py`; output
`synthetic_01_summary.txt`, rows `synthetic_01_rows.csv`.

- 6,000 identity sites for each of the six panel cells, by Met-A's own rule on the atlas.
- **The offset is real and large before the tare:** untared A was 0.813, 0.998, 1.168 and 1.324 for laboratory offsets of -0.03, 0, +0.03
  and +0.06.
- **B1 met:** after the same-run tare every healthy array of every cell and laboratory read Normal (lowest share 1.000).
- **B2 met, and weaker than it looks:** the planted 1 % loss changed A_rel by exactly the predicted amount in every cell (1.000). The
  prediction is the same arithmetic on the same array, so this shows the tare does not absorb a real loss; it does not test the size
  independently. Synthetic sites are independent and noise is only the healthy spread, so real arrays will be harder.

Next (design step 3): the sorted cells of GSE184269, an independent laboratory the atlas never saw, on GitHub, each array read against
its atlas cell and tared against the other arrays of the same cell in that series.

## Design step 3, run once (2026-10-21; GitHub explore run 39680799865; exploration, not sealed)

GSE184269's 109 sorted arrays (an independent laboratory, never read before) through chain Stage 1 and `chain/stage_metaa_atlas.py`, tared
leave-one-out within each cell type: `data/DEV_METAA_ATLAS_01/sorted_01/sorted_cells_01_summary.txt`. Every array read. **Not ready:** after
the tare only 37 to 75 % read Normal (tared spread 4.2 to 8.6 %), and a planted 1 % loss moves a reading by about one spread.

**Why, measured (`slide_offset_02.py`, on the committed readings):** each donor's five cell types sit on one slide, and their readings move
together (correlations 0.53 to 0.87). Removing each reading's slide offset, taken only from the same slide's other cells, cuts the spread by
about a third (Normal 72 to 92 %), so the slide (here also the donor) is part of it, not all. The X is not the cause: each cell's 6,000
identity sites hold 0 to 27 on the X. Such a correction is not a proposed reading: with one donor per slide it would also hide a loss
shared by all of a donor's cells (the self-tare trap). Next: what is left (the sorted subsets are naive cells, the atlas cells are not
always; age; sites where this laboratory's arrays differ from the atlas), before any sealed test.

## Exploration 3, planned (written 2026-10-21 before it runs; not sealed)

`data/DEV_METAA_ATLAS_01/sorted_purity_03.py` on GitHub: is the sorted cells' spread their sorting purity? Each array again through chain
Stage 1, `stage_metaa_atlas.read` and the commissioned IAM-Deconvolver (`stage_v4.stage_d21`); per cell type the correlation of the tared
reading with the off-target share, and Normal in the purest half against the rest. Nothing is fitted or corrected.

**Exploration 3 outcome (2026-10-21, GitHub explore run 39704005104; `data/DEV_METAA_ATLAS_01/sorted_purity_03/`).** Is the spread of the
sorted cells against the atlas their sorting purity? No, not in four of the five cell types. The arrays are nearly pure by the
IAM-Deconvolver (median own-group share 0.98 to 1.00). Off-target share and the tared reading are not related in monocytes, naive CD4 T,
naive CD8 T or NK cells (Spearman +0.13 to +0.24, p 0.32 to 0.59). Naive B cells show a weak relation (+0.46, p 0.026). And the purest half
of each type is as wide as the whole: its spread is 0.9 to 1.8 times what a planted 1 % loss moves. So a purity cut would not make Met-A on the
atlas sensitive enough; the spread comes from between people or between runs, not from contamination. Next in this plan: the healthy-width
site choice and weighting (design additions above), synthetic first.

**Design additions 1 and 2 closed (2026-10-21, on the atlas alone, no array read; `data/DEV_METAA_ATLAS_01/width_weight_04.py`).** Addition 1 is already the stage's rule: each cell's
6,000 identity sites are the tightest by healthy width. That leaves almost nothing for weighting (addition 2) to do. Across the six panel
cells the widths at the identity sites run only from 0.0038 to 0.0091 (medians 0.0046 to 0.0066). Weighting by one over the width squared
keeps 5,322 to 5,825 effective sites of 6,000, and it changes what a 1 % loss moves the reading by less than 1 % in every cell (for example
neutrophils +0.05858 equal, +0.05907 weighted). The option was not added to the chain. This also shows where the sorted cells' 4 to 9 % spread
cannot come from: per-site person-to-person variation of this size averages away over 6,000 sites. It is a whole-array offset (step 3 found
it moves with the slide). The same-run tare exists to cancel exactly that, but in GSE184269 each slide holds one donor, so every array could
only be tared against other slides. **Next:** a sorted-cell series where several donors' cells of one type share a run, so the tare is
tested the way the chain uses it.

## Exploration 5, planned (written 2026-10-21 before it runs; not sealed)

`data/DEV_METAA_ATLAS_01/same_slide_05.py` on GitHub: GSE166844, purified blood cells of 30 volunteers from a general-population cohort
(an independent laboratory, not among the atlas's sources, never read before). 141 sorted arrays (B, CD4 T, CD8 T, monocytes, granulocytes
against the atlas's neutrophils), and each slide holds two donors. Each array goes through chain Stage 1 and `stage_metaa_atlas.read`,
unchanged. Two tares are compared, and neither uses the array's own donor: across runs (as step 3), and against the other donor on the same
slide. The second tests the same-run tare the way the chain means it, without the self-tare trap. Reported for each: Normal share, spread,
and what a planted 1 % loss moves the reading by, in spreads. Each array's values at the identity sites are kept for later explorations.

**Exploration 5 withdrawn before reading (2026-10-21).** GEO holds no IDATs for GSE166844, only signal tables without the control probes
chain Stage 1's calibration needs. The run stopped at its first three arrays, as written, and nothing was read.

## Exploration 6, planned (written 2026-10-21 before it runs; not sealed)

`data/DEV_METAA_ATLAS_01/same_run_06.py` on GitHub: GSE168779, purified monocytes of 130 healthy children (14 to 24 months), one array
each, 67 not vaccinated and 63 given BCG at birth; an independent laboratory, not among the atlas's sources, never read. Ten slides hold
four or more unvaccinated arrays, so each can be tared the way the chain tares: against at least three healthy monocytes on its own slide.
Every array goes through chain Stage 1 and `stage_metaa_atlas.read`, unchanged. Scored on the unvaccinated arrays, with four tares compared
(same slide or other slides; controls only or every array). Reported for each: Normal share, spread, and what a planted 1 % loss moves the
reading by, in spreads.

## Exploration 6, run once (2026-10-21; explore run 39723551088, commit 54390f59; 130 of 130 arrays read)

Output: `data/DEV_METAA_ATLAS_01/same_run_06/same_run_06_summary.txt` (copied from the run's report in issue 32; the per-array rows and values are in the run's GitHub output). The
67 unvaccinated children's monocytes, read against the atlas's (adult) monocytes:

| Tare (never the array itself) | Tared | Normal | Spread | Planted 1 % loss |
|---|---|---|---|---|
| Same slide, healthy controls (at least 3) | 50 | 100 % | 0.0151 | +0.0538, 3.6 spreads |
| Same slide, every array | 67 | 100 % | 0.0155 | +0.0539, 3.5 spreads |
| Other slides, controls | 67 | 96 % | 0.0208 | +0.0540, 2.6 spreads |
| Other slides, every array | 67 | 96 % | 0.0205 | +0.0540, 2.6 spreads |

The 42 vaccinated children tared on their slide: median 1.0016, 98 % Normal.

**What it shows.** Read against the atlas and tared the way the chain tares (healthy cells of the same type run the same way), a healthy
monocyte from an independent laboratory reads Normal, though the atlas's monocytes are adults' and these are infants'. The tare cancels both
gaps. The slide matters: the same-slide tare is tighter than the cross-slide tare and makes a 1 % loss stand out more. This answers step 3's
open question (GSE184269 had one donor per slide, so it could only tare across slides). Not sealed; a sealed test on a new series comes next.

**Exploration 7, written before it runs (2026-10-22).** Found while planning the next sealed test: the chain's commissioned Met-A reads every
array only after Stage T step 1, the self-tare (each design mapped onto the reference arrays' scale by the array's own fixed-site anchors;
no other person; adopted 2026-10-04). Explorations 6 and the sealed test METAA-ATLAS-SEAL-01 read Met-A against the atlas without that step,
so they did not run the whole chain. The sealed result stands as scored. `data/DEV_METAA_ATLAS_01/selftare_07.py` reads both spent series
(GSE168779's 67 unvaccinated controls, GSE267660's 31 healthy donors) with and without step 1, each tared on the laboratory and on the slide,
to learn whether the chain's own step shrinks the spread across slides. Only if it does does a sealed test with step 1 and the laboratory
tare come next; public series rarely hold the four or more healthy arrays per slide a same-slide test needs (screened 2026-10-22: GSE201754,
GSE252024, GSE189426 have at most 4, 3 and 2 healthy arrays on any slide).

**WITHDRAWN 2026-10-22: not a chain result.** The script called chain stages directly instead of running the one command (chain/MethylPhys_Interface/run_sample.py), so Stage 0 and Stage T step 1 were skipped. Author: "If the chain didnt produce it, it did NOT happen." Every number above is withdrawn; the work is rerun through the one command (to-do step 35).

**Exploration 8, written before it runs (2026-10-22; exploratory, not sealed; both series spent).** Explorations 6 and 7 again, the chain's
way. Every reading is the one command: `run_sample.py --specimen "isolated monocytes"` on each array's IDATs (Stage 0, Stage 1, Stage T step 1,
then Met-A against IAM-Atlas v2.1 monocytes, `development.metaa_atlas`), run over each series by the chain's series runner
(`chain/MethylPhys_Interface/run_series.py`, workflow `series.yml`: pass 1 without references; pass 2 with run_sample.py's own tare against
the other healthy arrays of the series, and of the same slide; the constructed 1 % loss by `stage_metaa_atlas.plant`, read by run_sample.py).
Scorer `data/DEV_METAA_ATLAS_01/chain_08_score.py` reads only the table of bundles. Series: GSE168779's 67 unvaccinated controls
(`chain_08_gse168779.csv`) and GSE267660's 31 healthy donors (`chain_08_gse267660.csv`), each its own run; first a 4-array smoke run
(`chain_08_smoke.csv`, one slide). Chain changes for it: intake accepts isolated, sorted or purified cells of any IAM-Atlas v2.1 type and
reads them by Met-A against the atlas, flagged not commissioned (`stage_0_intake.atlas_cell_of`); the commissioned Met-A is not applied to them.

## Exploration 8, adult monocytes, run once through the one command (2026-10-22; series run 39834679796)
Output: `data/DEV_METAA_ATLAS_01/run_08/chain_08_gse267660_summary.txt` and `_rows.csv` (31 healthy adult monocyte arrays of GSE267660, every
reading and tare by run_sample.py). Stage 0 admitted all 31 (one with a penalty). Tared against the rest of the laboratory's healthy arrays:
31 tared, 90.3 % Normal, spread 0.0401, a planted 1 % loss moves the reading 1.45 spreads. Tared against healthy arrays on the same slide
(the chain's Stage T rule): 9 of 31 had at least 3 on their slide, 100 % Normal, spread 0.0145, a planted 1 % loss 4.09 spreads. The withdrawn
sealed test's pattern holds through the chain: the laboratory-wide tare is too loose for a 1 % loss; the same-slide tare is tight enough.
The children's monocytes (GSE168779) are still running.

## Exploration 8, children's monocytes, run once through the one command (2026-10-22; series run 39834626943)
Output: `data/DEV_METAA_ATLAS_01/run_08/chain_08_gse168779_summary.txt` and `_rows.csv` (67 healthy children's monocyte arrays of GSE168779).
Stage 0 admitted all 67. Laboratory tare: 67 tared, 97.0 % Normal, spread 0.0197, a planted 1 % loss 2.76 spreads. Same-slide tare: 50 of 67
tared, 100 % Normal, spread 0.0145, a planted 1 % loss 3.73 spreads. With the adult series (100 % Normal, 4.09 spreads on its own slide), the
same-slide tare reads every healthy monocyte Normal in both laboratories and lifts a 1 % loss well clear of the healthy spread. Next: the
sealed test of the same-slide tare, on a series not yet read with at least 3 healthy arrays on most slides.
