# DEV-TERRITORIES-01: reading cancer's own territories (development)

**The instrument's job:** a gain at the islands every healthy cell keeps unmarked, or a loss in the loosely held domains, is read where it
happens, against IAM-Atlas v2.1 in the person's own mix; a healthy sample reads Normal.

**Why.** The commissioned Met-A, the C-Score and Group-A read each cell's 6,000 identity sites, whose healthy levels are by rule 0.05-0.25 or
0.75-0.95, so none reads the two places where early cancer is known to act. Both are on the arrays and in the atlas. Two fixed rules on the
atlas, over the sites measured in all 79 cell types (`chain/stage_metaa_atlas.territory_sets`; nothing chosen by hand, nothing fitted):

- **islands:** the highest of the 79 values below 0.2: 66,986 sites;
- **domains:** mean of the 79 at least 0.75, none below 0.2, at least one between 0.2 and 0.7: 100,101 sites.

The reading (`territory_reading`, run by `run_sample.py --dev-territories`): mean H(beta) over mean H(e) at each set, e the atlas mixed in the
sample's own Deconvolver shares, with the mean level change beta - e. Plants by the chain for tests: `--dev-plant-territory`.

**Synthetic test 1** (`data/DEV_TERRITORIES_01/synthetic_01.sh`, written before it ran; output `synthetic_01_summary.txt`): the healthy whole
blood GSE235717 GSM7508429 (spent), then gains planted at 20 % of the islands and losses at 20 % of the domains, each at two sizes.

| sample | islands A | islands level change | domains A | domains level change |
|---|---|---|---|---|
| healthy | 1.17607 | +0.01211 | 1.11469 | -0.02004 |
| island gain, 10 % | 1.40054 | +0.03075 | 1.1147 | -0.02004 |
| island gain, 30 % | 1.6293 | +0.06802 | 1.11472 | -0.02004 |
| domain loss, 10 % | 1.17605 | +0.01211 | 1.21691 | -0.03759 |
| domain loss, 30 % | 1.176 | +0.01211 | 1.32304 | -0.07271 |

**Against what was written before it ran:** the plants are read as expected: each raises its own set's reading, larger plants further, with
the right sign of level change, and leaves the other set unchanged to the fourth decimal. **Healthy does not read near 1:** the islands read
1.176 and the domains 1.115, untared. On this array the healthy islands sit 0.012 above the atlas and the domains 0.020 below it: an array
compresses toward the middle (low values read high, high values read low) where sequencing does not. That is the scale difference the atlas
ruler is meant to remove, so these readings cannot be read for a person until it is built, or until a healthy band is measured.
A real offset of healthy blood against the atlas cannot be told apart from the scale difference in this test.

**In plain words.** The two new readings see planted cancer-like changes clearly and keep them apart; on a real array, healthy blood already
sits off the atlas at these sites, which is the chip's scale, so the ruler comes first.

**What it would take to finish:** the atlas ruler; then the healthy spread of both readings on healthy arrays; then spent tumor and
normal arrays. **Not a milestone:** a synthetic test on one blood, nothing commissioned.

## Test 2: Met-A as a two-level instrument

**The instrument's job:** the whole-sample reading (every atlas site, `stage_metaa_atlas.read_whole`) moves when anything in the sample departs;
each cell's reading (its identity sites) says which cell. Both on by default in `run_sample.py`, with the territories.

`data/DEV_TERRITORIES_01/whole_02.sh`, written before it ran; output `whole_02_summary.txt`. Healthy whole blood GSM7508429 (spent), untared;
plants by the chain. 773,556 sites carry an expectation.

| sample | whole sample | its identity-site part | islands part | domains part | the rest | neutrophils' own (identity sites) |
|---|---|---|---|---|---|---|
| healthy | 1.11291 | 1.15739 | 1.17899 | 1.12544 | 1.10423 | 1.12224 |
| neutrophil identity-site loss, 5 % | 1.11369 | 1.18196 | 1.18291 | 1.12713 | 1.1042 | 1.23072 |
| neutrophil whole-pattern loss, 5 % | 1.18621 | 1.25794 | 1.34547 | 1.20188 | 1.1685 | 1.21996 |
| island gain, 10 % | 1.12813 | 1.20068 | 1.41105 | 1.12545 | 1.10405 | 1.17345 |
| domain loss, 10 % | 1.12878 | 1.18753 | 1.17897 | 1.24224 | 1.10404 | 1.17062 |

**Against what was written before it ran:** every plant raises the whole-sample reading above healthy (standard error over sites about 0.0002);
the island plant moves the islands part most and the domain plant the domains part most; the neutrophil identity-site loss moves neutrophils'
own reading most (1.122 to 1.231) while the whole sample barely moves (a loss at 6,000 of 773,556 sites), which is why both levels are kept.
The island and domain plants also move neutrophils' own reading (to 1.173 and 1.171), because some identity sites lie in those sets.
**Healthy reads 1.113 untared** on every part: the array's scale against the atlas, the same offset as test 1, for the atlas ruler to remove.

## The atlas ruler, first version (two anchors per probe design)

`chain/builders/build_atlas_ruler.py` builds `Runtime Matrices/Stage_T/atlas_ruler_v1.json`: for each probe design, sites every one of the 79
cell types holds below 0.10 (outside CpG islands and their 2 kb shores) and above 0.90, on both the 450K and EPIC v1 chips, autosomes (861 and
1,101 low, 2,553 and 3,690 high, for designs I and II); targets the atlas's means there. `stage_t_selftare.ruler_map` maps the array's own anchor
means onto them by the self-tare's straight line; `run_sample.py --dev-ruler` (off by default) applies it before the whole-sample Met-A and the
territories. On the healthy whole blood GSM7508429 (test 3, `whole_03.sh`, output `whole_03_healthy_summary.txt`; the planted runs follow): the type II low anchors read 0.064 against
the atlas's 0.039, so the line lowers every low value; the whole-sample reading goes from 1.113 untared to 0.938, the islands part from 1.179 to
0.877, the domains part from 1.125 to 0.939. **The straight line overcorrects:** the array's departure from the atlas is not a straight line across
the range. Next: a ruler with many marks (sites every cell type holds alike at several levels between 0 and 1), synthetic first.

**On sequencing, in sequencing units (chain change, nothing scored).** The territories compare a sample with IAM-Atlas v2.1 mixed in its
shares; the atlas is on the array scale, whose unmarked sites sit near 0.055 where sequencing reads 0, which put every healthy sequencing
sample's islands far below 1 (DEV-METAA-ATLAS-CURVE-01, the shift's cause). On sequencing the mixed expectation is now put in sequencing
units by the atlas's own measured conversion (`stage_metaa_atlas.seq_units`), as Met-A's is; arrays are unchanged. Healthy sorted B and T
cells from two new laboratories are read and reported on it in the sequencing commissioning test (not scored there).

**The territories' healthy band on sequencing (written and committed before any reading on the amended chain).** **The instrument's job:**
a healthy cell reads near 1 on each territory. **Data:** IAM-Atlas v2.1's 153 healthy sequencing inputs (spent, atlas inputs;
`DEV_ALL_INFO_01/coupling_02/healthy_seq.csv`), each read again by the one command (`geo_pat.yml`, its sorted cell as the specimen) on the
chain that reads the territories in sequencing units.
**Prior reading:** GSE235717 GSM7508429 (spent) is the healthy blood of the first synthetic above; the 153 atlas sequencing inputs were read earlier the same day on the previous rule (the coupling reference, Met-A's sequencing band), and their territory readings on that rule are not used here.
**Synthetic test:** DEV_TERRITORIES_01/seq_band_01/terr_seq_synthetic.py, DEV_TERRITORIES_01/seq_band_01/terr_seq_synthetic.txt
**Run 1 not met as written** (`seq_band_01/terr_seq_synthetic_run1.txt`): the healthy naive B cell plant read islands 3.00 (CD8 T cells 0.99).
The cause was in the one command, not the territories: for a sorted cell the whole-sample readings (Met-A whole sample, the territories,
the C-Score) were mixed from the six-cell blood panel's shares, so a naive B cell was compared with the panel's B-cell column, whose island
levels differ. A declared sorted cell (specimen "sorted ...") is now read against its own atlas column (`run_sample.py`, the bundle says so
in `whole_sample_reference`); mixed specimens are unchanged. **Run 2** (no bar changed): met, healthy plants 0.99 to 1.01 on both territories.
**Built** by `seq_band_01/terr_seq_band.py`: for the islands and the domains, the 2.5th
to 97.5th percentile of the healthy readings and the median; **bar:** the median healthy reading within 0.05 of 1 on each. The same pass records the C-Score's healthy spread on sequencing (whole genome: how many healthy samples sit within their derived limit, C's median and 97.5th percentile), not scored. **Then:** the
healthy B and T cells of the sequencing commissioning test (DEV-METAA-ATLAS-CURVE-01) read against these bands.

**The territories' healthy band on sequencing, read: not met** (`seq_band_01/territories_seq_band_summary.txt`; 152 of the 153 healthy
sequencing inputs read by the one command, one stopped on GitHub). Islands: healthy median 13.9359 (2.5th to 97.5th percentile 1.4057 to
27.5715); domains: median 1.1323 (0.8345 to 1.3065). Bar (median within 0.05 of 1) not met on either. **What it shows:** on sequencing the
territories are read against IAM-Atlas v2.1, whose unmarked levels carry the array floor (the shift's cause, read in DEV-METAA-ATLAS-CURVE-01).
At the islands every healthy cell keeps unmarked, sequencing reads near 0, so the healthy disorder built from array levels is far from what a
healthy sequenced cell shows. The synthetic planted its truth on the same array levels and could not see it. The territories on sequencing
are to be read against the IAM Sequencing Atlas (each cell's own healthy disorder, counted at every CpG); their band is built
again on it, synthetic first. **Recorded, not scored:** the C-Score on sequencing, whole genome: 145 of 152 healthy within their derived
limit; C median 1.0010, 97.5th percentile 1.0085.
