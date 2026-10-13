# DEV-CEILING-02 — where a tumor sits between its own healthy tissue and the derived ceiling (development, 2026-10-13; NOT sealed)

**The author's question.** Is there consistent evidence for a tumor's distance from the ceiling that goes with a cancer diagnosis -
measured in several cell types and by several methods, and some day derived?

**Why the question can be asked here at all.** Both ends of the gauge are derived: the floor by thermal kicks against one ATP, the
ceiling by a coin flip at every identity site (one bit per site, reached when methylated sites average beta = 1/2). So a tumor's
position can be written without units, as the fraction u of the headroom between its own normal tissue (u = 0) and the ceiling
(u = 1) that it has used. Both arms are read separately: the methylated arm approaches the ceiling from above, the unmethylated arm
from below.

**Prior reading:** every set here is already read, sealed and scored - DEV-TISSUE-BREAST-01, DEV-TISSUE-COLON-01, DEV-TISSUE-LIVER-01
and DEV-TISSUE-LUNG-01. This is development use: a first descriptive look on their sealed sites and pairs. It cannot change any of
their verdicts and none of them may later serve as a held-out test of a ceiling claim.

**The rule, written before running** (`data/DEV_CEILING_02/ceiling_position_02.py`, committed first): per cancer, the median u with a
bootstrap 95 % interval; "consistent across cancers" only if all four methylated-arm intervals overlap and the medians' spread (SD /
mean) is under 0.25. A planted check had to give u = 0, 1 and 1/2 on made-up numbers before any tumor was read; it did, exactly.

## Outcome, run once: NOT CONSISTENT across cancers, and every cancer moved the same way

Numbers from `ceiling_position_02_summary.txt` and `ceiling_counts_02_output.txt`; 219 of 219 pairs read.

| cancer | pairs | methylated arm, median u [95 % CI] | moved toward the ceiling | unmethylated arm, median u |
|---|---|---|---|---|
| liver | 50 | +0.1286 [+0.0906, +0.2130] | 42 of 50 | +0.0153 |
| breast | 53 | +0.0769 [+0.0483, +0.1322] | 50 of 53 | -0.0142 |
| colorectal | 88 | +0.0574 [+0.0474, +0.0685] | 80 of 88 | +0.0131 |
| lung | 28 | +0.0419 [+0.0133, +0.0894] | 21 of 28 | -0.0002 |

By the rule written first this is **not consistent**: the liver and lung intervals do not overlap and the medians spread by 0.495
against a limit of 0.25. What does hold across all four: the methylated arm moved toward the ceiling in 193 of 219 tumors, no tumor
was past it, and every cancer used only a small fraction of its headroom - between about 4 and 13 percent at the median. The
unmethylated arm barely moves in any of them. The lung set, whose sealed test read no fingerprint, sits lowest.

**Why this is not yet the answer.** Bulk tissue is a lower bound. Every tumor sample is a mixture, and the normal cells in it pull the
reading back toward u = 0, so a cancer whose samples are less pure will read lower even if its tumor cells sit at the same place. The
ordering above could be a purity ordering. That is the first thing to rule in or out, and it is measurable.

**In plain words.** We asked how far each tumor has moved from healthy toward the point where its DNA pattern would be a coin flip.
Nearly every tumor in four cancers moved that way, none reached the end, and they used between a twentieth and an eighth of the room.
But the four cancers did not land in the same place, so there is no single distance yet. Tumor samples are diluted by healthy cells,
which makes the measured distance smaller than the real one, so the next step is to correct for that and see whether the cancers
line up.

**What it would take to finish:** each sample's tumor purity (published for the two TCGA sets, breast and liver), the position
corrected for it, and the sequenced cancer cell lines - pure tumor cells read molecule by molecule - as the other bracket. If the
corrected positions converge, the claim is written as a seal and tested once on cancers nobody has opened. If they do not, the
distance is a property of each cancer, not a law, and the record says so.

**Synthetic test:** DEV_CEILING_02/ceiling_position_02.py, DEV_CEILING_02/ceiling_position_02_summary.txt

**Not a milestone:** a first descriptive look on spent data, and its own consistency rule was not met.
