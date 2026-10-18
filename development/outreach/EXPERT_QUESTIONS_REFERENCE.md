# What methylation researchers ask, and how IAM answers (working reference, not for the site)

Source: an "Ask the Experts: DNA methylation studies" feature (vendor-sponsored, about 2021-22: it calls EPIC v1 the newest array), three
researchers: a genomics core-facility head, a population epigenetics fellow, and an epigenetics laboratory director who works on sorted cells.
Saved 2026-10-18 at the author's request, so the researcher package answers what they will be most curious about. Numbers in the source were
in a damaged font and were decoded by hand.

## The eight questions they were asked
1. When would you consider DNA methylation?
2. What can you measure with it?
3. How do you tell correlation from causation?
4. What matters about samples?
5. Replicates and controls?
6. Intra-specimen comparisons, and how is a healthy baseline set?
7. Processing raw intensity files?
8. Advice for a first study?

## What they emphasized, and what IAM has

| Their point | What IAM has (2026-10-18) |
|---|---|
| **Baseline.** Usually the average of a control group; the sorted-cell lab says the same person's own earlier sample is "much more powerful", because everyone has a different baseline. | IAM-A reads each sample against the line its own loose share predicts, no control group (commissioned 2026-10-17, sealed test on healthy CD8 T cells; one command reproduces it exactly, A 0.9083). Met-A on arrays: three same-run healthy references. |
| **Cell mix.** Whole-blood differences mostly reflect cell proportions; that lab sorts cells. | IAM-Atlas v2.1 and the IAM-Deconvolver measure the mix (commissioned for neutrophil, monocyte, CD4 and CD8 shares); Met-A is commissioned on isolated neutrophils for this reason; the composition guard declines samples whose mix it cannot trust. |
| **Batch effects.** Corrected afterward (ComBat, RUV, SVA). | The same-run tare divides by healthy references from the same run, so the batch cancels; the atlas fits each source's offset beside the biology (map-making's destriping). |
| **Sensitivity.** Arrays detect changes of about 20-30 %; subtler needs sequencing. | The C-Score's commissioning detected a 5 % shift on 5 % of sites when clustered (detection limit C 1.5 at 5 % of sites clustered): it reads the pattern across many sites, not one site at a time. |
| **Early detection, liquid biopsy, precancer.** Named by all three as the biggest opportunity. | On the test plan: cfDNA (short-fragment counter checked first), and 140 rectal biopsies with and without adenomas reserved for the field test (TEST_PLAN_TRACKS_2026-10-17.md). |
| **Aging clocks vs disease.** | Scattered drift is ordinary aging (0 of 120 planted scattered changes read above the C-Score band); clustered drift is what the C-Score flags. |
| **X inactivation, imprinting.** | The v2.1 uncertainty map found the X chromosome on its own. Imprinted regions are not among Met-A's identity sites (they read near 0.5). |
| **Replication in external data; blind analysis.** | Every rule sealed before its data; clean-clone release check; held-out sets; records and accessions in the repository. |

## Where IAM changes the question
Correlation vs causation: they reach for statistics (Mendelian randomization, longitudinal cohorts). IAM does not prove causation and should
not claim to. It adds a physical scale instead of an association: drift measured in bits, between a floor and a saturation point derived from
physics, not fitted to a cohort.

## Where they will push
1. Sample sizes (the IAM-A grand test read 8 samples).
2. The C-Score's far-right example is planted, not yet a real tumor.
3. Narrow scopes: Met-A on EPIC v1 only; nothing yet on 450K, EPIC v2, FFPE tissue or across ancestry (self-reference should help with ancestry; untested).
4. The law across 37 orders of magnitude: the hardest sell to this audience.

**How to present it to them:** lead with the problems they named (own baseline, batch, cell mix, sensitivity), each shown on real data; then the
physics as the reason it works. The first thing they will want is to run it on their own arrays.

## "Try to break it" (planned document for the researcher package)
For an outside lab, in their terms:
1. What each instrument should read on samples they already know, written down in advance (healthy sorted neutrophils read Normal on Met-A;
   known mixtures recovered as their true shares; a known clustered change reads above the C-Score band; a healthy sorted CD8 T-cell sample reads
   Normal on IAM-A against its own reference).
2. What counts as a failure for each.
3. How to report it publicly (a GitHub issue form), so every failure becomes part of the record.

The author (2026-10-18): once the package is ready, he will reach out to groups like these; "if they have doubt, which they should", they can
download the chain, run it on their own data with their own known results, test the physics, and try to break it. "That is what we want."
