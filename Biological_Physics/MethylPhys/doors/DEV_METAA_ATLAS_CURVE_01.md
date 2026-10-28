# DEV-METAA-ATLAS-CURVE-01: Met-A with IAM-Atlas v2.1 as the ruler (plan 14.14)

Written and committed before any monocyte array is read. Development; nothing here is commissioned.

**The instrument's job:** a healthy purified monocyte array, read against the atlas's monocytes on its own response curve, reads Normal.

**Why:** Met-A step 3 (DEV-METAA-HEALTHY-01) found that the commissioned identity sites, chosen on the same arrays they are scored against,
read above each array's own response curve, so the commissioned mark could not be predicted. The atlas is the ruler (author): here the sites
are chosen on IAM-Atlas v2.1 and the arrays read are ones the atlas does not hold.

**The reading** (`stage_metaa_atlas.read_on_curve`, `run_sample.py --dev-metaa-atlas monocytes --dev-metaa-curve`): identity sites by Met-A's
own rule on the atlas (`identity_sites`: healthy width at most 0.05, best value 0.75-0.95 or 0.05-0.25, 3,000 per side). The array's own
response curve against the atlas's monocyte best value, measured at every other atlas site it read (binned means, step 3 version 7's bins),
and its own spread about that curve, bin by bin. The array's healthy mark is the expected entropy at the identity sites of a reading on its
curve with its own spread; Met-A is its own mean entropy there over that mark. No cohort, no same-run references, nothing fitted.

**Prior reading:** none of the arrays. GSE189426, GSE201754 and GSE196071 had only their sample labels and file lists read; none is in the
atlas roster or the spent list. GSE110554 and GSE167998 (spent, atlas sources) are named in the B and T section only to place a laboratory; nothing of theirs is read in this record.

**Data:** purified monocytes from healthy donors on the EPIC chip, raw IDATs, read by the one command (`series.yml`, specimen "isolated
monocytes"). Development: GSE189426's 15 healthy donors (`data/DEV_METAA_ATLAS_CURVE_01/mono_dev_samples.csv`). Confirmation, untouched until
the development bars are met: GSE201754's 31 and GSE196071's 12 healthy donors, three laboratories in all
(`data/DEV_METAA_ATLAS_CURVE_01/mono_confirm_samples.csv`).

**Synthetic test:** DEV_METAA_ATLAS_CURVE_01/curve_synthetic.py, DEV_METAA_ATLAS_CURVE_01/curve_synthetic.txt

**Bars (at most three):**
1. **Synthetic:** constructed arrays from the atlas's own monocyte column through two different curved responses read within 0.01 of 1; a
   planted 5 % loss of the pattern reads the value the planted loss predicts, within half a hundredth.
2. **The job:** at least 90 % of the healthy arrays read Normal (0.95-1.05), in development and then in the confirmation.
3. **Selection:** across the healthy arrays, the median of each array's mean reading above its own curve at the identity sites lies within
   half a hundredth of zero (the commissioned sites, chosen on their own arrays, read well above it in step 3).

**Bar 1, read: met on its third run** (`data/DEV_METAA_ATLAS_CURVE_01/curve_synthetic.txt`; runs 1 and 2 kept beside it). Run 1 put the
array's mark at the entropy of its curve's mean reading alone, which left every array low by about the effect of its own scatter; run 2
took the scatter from each bin's mean entropy, which also averages over the levels inside a bin and left every array high by about as much.
Run 3 takes the array's own spread about its curve and reads all three cases within the bars.

**Scorer:** `data/DEV_METAA_ATLAS_CURVE_01/mono_curve_score.py`, committed before the development arrays land (series run 40202293656).

**Reported, not scored:** the patients' arrays in the same series (giant cell arteritis, systemic sclerosis, undifferentiated arthritis),
read and flagged after the healthy bars.

**Development, read: bars 2 and 3 met** (`data/DEV_METAA_ATLAS_CURVE_01/dev_01/mono_curve_dev_summary.txt`; series run 40202293656, scored
from its shards' bundles). Every one of the 15 healthy monocyte arrays reads Normal on its own curve, and the identity sites chosen on the
atlas sit on each array's curve (median above-curve in the file), where the commissioned sites, chosen on their own arrays, read well above
it. Disclosed: the scorer's folder pattern was fixed before it printed any array (it looked for a folder named pass1; the shards are saved
as pass1-0, pass1-1 and so on). **Next, as written:** the 43 confirmation arrays from the other two laboratories.

**Confirmation, read: bar 2 not met as written; bar 3 met** (`data/DEV_METAA_ATLAS_CURVE_01/confirm_01/mono_curve_confirm_summary.txt`;
series run 40204936759; GSE201754 and GSE196071, the two laboratories reserved). 36 of the 43 healthy arrays were read and 35 of those read
Normal; bar 2 counts all 43, so 35 of 43. The 7 not read were withheld by the reading itself, each with too few of its 6,000 identity sites
past the array's own detection check (the counts are in each array's bundle); no array was read and called abnormal for that. The identity
sites sit on each array's own curve in both laboratories (bar 3). **Not commissioned:** bar 2 as written is not met. Stated for the next
test: arrays the reading withholds for detection are reported apart from the arrays it reads, and the bar is set on the arrays read.

## B and T cells (written and committed before any B or T array is read; plan branch for B and T cells)

**The instrument's job:** a healthy sorted B or T cell array, read against its own subtype's column of IAM-Atlas v2.1 on its own
response curve, reads Normal. Same reading, same bars as the monocytes; B and T cells carry the same priority (author).

**Prior reading:** GSE110554 and GSE167998 (spent, atlas sources) are named only to place GSE174666's laboratory; nothing of theirs is read here.

None of the B or T arrays has been read. Six series had only their sample labels and file lists read; none is in the atlas roster or the
spent list. GSE288076 (naive CD4 T cells of preterm neonates) is left out: not adult blood.

**Data, one subtype to one atlas column** (stated mapping, from each array's own labels; ambiguous subsets such as mixed effector-memory
and TEMRA sorts, PBMC, NK and monocytes left out): development `data/DEV_METAA_ATLAS_CURVE_01/bt_development_samples.csv`, 78 arrays from
one laboratory (GSE252045: naive and switched-memory B, naive and central-memory CD4, naive, central-memory, effector-memory and TEMRA
CD8); confirmation `data/DEV_METAA_ATLAS_CURVE_01/bt_confirmation_samples.csv`, 35 arrays from three other laboratories (GSE192451
naive and memory B, GSE146093 healthy donors' CD4, GSE184500 controls' CD8). GSE174666 (43 arrays) shares its contributors with the
sorted-blood reference series the atlas was built from (GSE110554, GSE167998), so it is not independent of the atlas: read and reported
apart (`data/DEV_METAA_ATLAS_CURVE_01/bt_samelab_samples.csv`), not counted in the confirmation. New in the chain for this test: `run_series.py` takes a per-array `atlas_cell`
column, so one run reads every subtype against its own column.

**Synthetic test:** DEV_METAA_ATLAS_CURVE_01/bt_synthetic.py, DEV_METAA_ATLAS_CURVE_01/bt_01/bt_synthetic_naive_cd8_t_cells.txt, DEV_METAA_ATLAS_CURVE_01/bt_01/bt_synthetic_naive_b_cells.txt

**Bars (at most three):**
1. **Synthetic:** the monocyte synthetic on two of these columns (naive CD8 T, naive B): two curved responses read within 0.01 of 1 and a
   planted 5 % loss within half a hundredth of its prediction. **Read: met on both** (the files above).
2. **The job:** at least 90 % of the arrays read Normal (0.95-1.05), in development and then in the confirmation. **Quality rule, stated
   now:** arrays the reading withholds (too few identity sites past the array's own detection check) are reported apart with their reason
   and not counted; a laboratory with more than a quarter withheld is reported as not assessed.
3. **Selection:** the median of each array's mean reading above its own curve at the identity sites lies within half a hundredth of zero.

**Scorer:** `data/DEV_METAA_ATLAS_CURVE_01/bt_curve_score.py`, committed before any array lands.

**B and T cells, development read, and a correction** (`data/DEV_METAA_ATLAS_CURVE_01/bt_02/bt_curve_dev_summary.txt`). **Correction first:**
GSE252045 is one of IAM-Atlas v2.1's own sources (`DEV_ATLAS_V21_01/final_samples_v21.csv`); 63 of the 78 development arrays are atlas inputs
(`bt_02/bt_overlap.txt`). The record above said none of the B or T arrays had been read: wrong for this laboratory, which the atlas read. The
check before writing looked at the atlas's v2 roster and its README, not the v2.1 sample list (LESSONS C44). The confirmation list (three
laboratories) and the same-laboratory list hold no atlas inputs and no atlas source (same file), and remain unread.
**The read, as it stands (mostly in-sample): bar 2 not met** (61 of 78 Normal), **bar 3 met** (median above-curve -0.0009). The CD8 columns
read Normal (TEMRA 6 of 6, effector memory 7 of 7, naive 13 of 15, central memory 15 of 16); CD4 central memory (2 of 9, median 0.9516) and
memory B (5 of 8, median 0.9593) read low, even on arrays the atlas was built from. **What follows:** the confirmation is not read; first,
why those two columns read low on their own inputs (both pool this array laboratory with sequencing laboratories through the lab terms).

**Why two columns read low** (`data/DEV_METAA_ATLAS_CURVE_01/bt_02/bt_columns.txt`, added after the development read, disclosed; nothing
scored). Every B and T column built from arrays with at most one sequencing sample reads 0.980 to 1.002; the two that read low carry the most
sequencing: memory B cells (2 of 16 samples) 0.9593, CD4 central memory (3 of 10) 0.9516. Likely cause: in a column where sequencing carries
real weight, the atlas's laboratory terms do not fully remove the chemistry difference between sequencing and arrays (the same difference
that reads sequencing samples lower against the array-built atlas). **Next:** measure that shift at the identity sites from the atlas's
own inputs before any change; the confirmation lists stay unread until it is understood.

## The sequencing-array shift at the B and T identity sites (diagnostic; written and committed before any of its readings exists)

**The question:** do sequencing samples read lower than arrays against the same IAM-Atlas v2.1 column, and by how much, column by column?
**Data:** the atlas's own 17 sequencing inputs to its B and T columns (`data/DEV_METAA_ATLAS_CURVE_01/shift_01/seq_inputs.csv`; the
published sequencing atlas of human cell types, GSE186458; spent, atlas inputs), against the same columns' array readings from the
development read above (`bt_02/bt_curve_dev_summary.txt`; spent). **Prior reading:** every sample here is an atlas input; nothing new is
spent. **The reading:** each sample's molecule file (hg19 .pat.gz from GEO) read once by the one command through the committed pipeline's
saved-molecules route (`.github/workflows/geo_pat.yml`, erase_gh.sh with PATFILE and FULL=1), its own cell type as the specimen.
**Compared** by `shift_01/shift_compare.py`: per column, the sequencing inputs' Met-A, the arrays' median and the difference. Nothing is
scored or tuned; the result decides whether Met-A on sequencing needs its own reference before the B-cell confirmation is read, and sets the
size of the chemistry offset on these cells (the plan branch for Met-A on sequencing).
**First dispatch refused at intake** (series run 40710832318): the sample list was not in the pipeline's format, so every sample stopped
before any molecule was read (rc 19); no reading was made. The list was rewritten in that format and sent again. **Second dispatch refused at intake** (series run 40725273902): the specimen was given as the bare cell name, which intake
refuses; a sorted cell is named as such ("sorted naive b cells"). Nothing was read; the list now gives each specimen that way.

**The shift, read** (`data/DEV_METAA_ATLAS_CURVE_01/shift_01/shift_summary.txt`; series run 40731928429; all 17 read). **Correction first:**
the comparison script's first scoring read each sample's mixed-specimen cell reading instead of its sorted-cell reading against its own atlas
column; it was corrected to read the sorted-cell field and rerun (disclosed; nothing else changed). **Result:** every healthy sequencing input
reads far below 1 against its own column of IAM-Atlas v2.1, while arrays of the same cells read close to 1. The atlas's own sequencing inputs
therefore cannot be read by Met-A against the atlas as built; IAM-A on the same files reads Normal, so the files are sound. **What follows:**
Met-A on sequencing needs its own healthy reference before any sequencing sample (the leukemia study included) is read on Met-A's gauge; the
cause of the offset (a site's value from few reads, or the chemistry) is measured next on the 153 healthy sequencing readings now running.

## The shift's cause: are the atlas's levels blurred toward the middle? (written and committed before the rerun)

**The question:** the atlas columns are built mostly from arrays, which pull every level slightly toward the middle; sequencing counts
molecules. If the atlas's level at an identity site is blurred, a healthy sequencing sample is more extreme there than the atlas says,
holds less disorder than the expectation and reads below 1, and more so the deeper it is (the read-count damping that offsets the gap
fades with depth), as the shift readings showed. **The prediction:** at the column's methylated identity sites the sample's median level is
above the atlas's, and at its unmethylated sites below. **The reading:** the same 17 sequencing inputs (`shift_01/seq_inputs.csv`), read again
by the one command on the chain that records each reading's levels at its identity sites (`stage_metaa_atlas`, `levels_at_identity`,
recorded only, never used in A). **Compared** by `shift_02/levels_compare.py`. Nothing scored; the answer decides what Met-A on sequencing
is read against.
