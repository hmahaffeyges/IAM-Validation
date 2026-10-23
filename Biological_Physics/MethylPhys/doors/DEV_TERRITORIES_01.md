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
