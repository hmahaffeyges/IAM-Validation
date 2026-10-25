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
atlas roster or the spent list.

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
