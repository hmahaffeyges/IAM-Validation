# DEV-GLOBE-COMMISSION-01: commissioning the IAM-Globe (plan, written 2026-10-22 before any of it is scored)

**The instrument's job:** a healthy sample's globe stays nearly dark; a change planted along the genome lights up where it was planted.

The globe is drawn by the one command (`chain/MethylPhys_Interface/run_sample.py`) on every reading, and every bundle records it:
`globe/blocks_changed` of `globe/blocks` (blocks of 10 neighboring atlas sites, each site's difference in its healthy width and the
sample's own spread, a block changed beyond the 5 % family-wise line), with the X and Y counted alone (`blocks_changed_on_XY`) and Met-A's
6,000 sites alone (`metaa_blocks_changed`). Nothing in the rule is fitted to the data it is tested on. The healthy example blood
(GSM7508429) reads 45 of 77,355 blocks changed, 1 of them on the X and Y (`website/art/out/explorer/healthy-blood.json`).

## Step 1: exploration on data already read (not sealed)

The Group-A rerun through the one command (series run of GSE318173's 278 healthy controls, `score_groupa_chain.py`) writes a bundle per
blood, so the globe of 278 healthy bloods from one laboratory comes from it with no new reading. From those bundles: the share of blocks
changed per healthy blood (its median, 95th percentile and largest), on the autosomes and on the X and Y, by sex and by slide. The planted
half of that run carries a 1 % neutrophil loss at Met-A's 6,000 sites, which the whole-atlas globe is not built to see (it is spread
thinly); its Met-A layer count is reported. These arrays are spent for Group-A, so nothing here can seal.

## Step 2: synthetic plants (not sealed)

Clustered losses planted along the genome in the spent healthy bloods (runs of 30 neighboring atlas sites moved toward 0.5, as
`report_globe_v4`'s own planted check does), at three sizes, each read by the one command from the planted values: the share of planted
blocks found, and the share of newly changed blocks that lie inside the planted runs.

## Step 3: the sealed test (bars fixed after steps 1 and 2, before any reading)

On healthy whole bloods from a laboratory none of this has read. At most three bars, of this form (the numbers set from steps 1 and 2 and
written into the seal before the series is opened):

1. **Healthy stays dark:** at least 95 % of healthy bloods have no more than a fixed share of blocks changed.
2. **A planted change lights:** a planted clustered loss of a fixed size is found in at least a fixed share of its blocks.
3. **It lights where it was planted:** at least 90 % of the newly changed blocks lie inside the planted runs.

If it meets every bar, the globe is commissioned on whole blood, EPIC v1, as a picture with measured limits; scoring a diagnosis from it
stays outside its scope.
