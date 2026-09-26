# What would make this chain more sensitive, ranked — and what it would cost

**Written 2026-09-25** on the author's instruction, before the next round of cohort testing: *"it does no good
to test one now and potentially miss something in the cohort that we could have actually detected had we just
been smart and spent the necessary time fine tuning and calibrating."* That is the right order of work, so
this document exists to make the choice explicit rather than implicit.

Two lists. The first is everything that would improve what the chain can see; the second is the cosmology
methods on the shelf. They overlap at the top, because the single largest improvement available is both.

**How impact is judged.** Not by elegance. By whether it changes a number the chain *reports*, or lets it
report something it currently withholds, on a specimen a clinic could actually send. An improvement that
makes a figure prettier is not on this list.

**Read the Blocker column first.** Several items are not compute — they are data that does not exist yet, and
no amount of cleverness substitutes.

---

## List A — chain enhancements

| # | enhancement | impact | effort | blocker |
|---|---|---|---|---|
| **A1** | **Use the atlas's full cell-type covariance in the separation** (generalised least squares instead of independent uncertainties) | **Highest.** Changes every fraction and therefore every A. It is what lets the solver say *these two components are individually uncertain but their sum is well determined* — which is the blood-separability situation exactly, and the reason five classes currently read 0.0 %. Also the most likely route to per-cell reporting. | 2–4 days | The covariance must be recovered from the per-class MCMC archives (165 MB, present in the repo). Nothing else is missing. |
| **A2** | **Band the joint haematopoietic-progenitor component** — *attempted 2026-09-25, [NOT COMMISSIONED](PROC_BAND_01_OUTCOME.md): B1, B3, B4 met (r=0.534 against immune, so it IS a second axis), B2 failed on one laboratory in four (Uppsala 0.646 against a bar of 0.70-0.90). Cause measured: the joint age curve was fitted on 234 arrays where immune's had 1,379, and Uppsala is the only cohort spanning the decades where that curve is worst* | **High.** Converts the departure statistic from *one* banded axis into two (corrected 2026-09-25: the chain reads progenitor and stem_adult JOINTLY on whole blood by its own s108 rule, so three was wrong), with proper chi-square thresholds. That is what the Mahalanobis design was for: catching a specimen that moves one component while another holds steady. Also removes a refusal printed on every report today. | 1 day, not the 2-3 h first estimated: the joint component has no age term and no laboratory zero, and both must be derived | None. Both classes sit above their presence floor in blood (12.8 % and 3.8 % on GSM2333901) and all four laboratories' healthy arrays are on disk. **This is the cheapest high-impact item on the list.** |
| **A3** | **Per-laboratory bands** — *measured 2026-09-25, [NOT COMMISSIONED](PROC_LABBAND_01_OUTCOME.md): at 80 arrays per laboratory a 1.40x width ratio is ORDINARY (permutation p = 0.123, null median 1.245), and applying per-laboratory widths makes the out-of-sample tail WORSE for three laboratories of four. Needs the full cohorts* | **High.** Today one pooled band is used for every laboratory, and the laboratories differ: false-alarm rates run 4.4 % to 9.8 % against a nominal 5 %. That is why GSM1051533, a healthy control, reads ELEVATED. A per-laboratory band converts a caveat into a correct threshold. | 1–2 days | None — the same four cohorts. Needs a pre-registration because it changes a reported tier. |
| **A4** | *ADOPTED 2026-09-25, [PROC-FOREIGN-01](PROC_FOREIGN_01_OUTCOME.md) - all six bars met; the immune tier is withheld when more than 2.07 % of a specimen is assigned outside the blood lineage. Verified end to end: a 20 % secretory mixture that would have printed ELEVATED now prints no tier.* | **Withhold the immune tier when the specimen carries foreign material** | **High, and it is a correctness fix.** Measured: immune A″ rises ~0.003 per 1 % of non-haematopoietic material — 0.15 σ of the healthy band per 1 %. At 20 % foreign, every healthy donor read ABOVE_BAND on a gauge that is supposed to report immune fidelity. A tumour-bearing specimen is exactly the case. | 1 day | The bound above which to withhold has not been measured. Needs one dilution series and a pre-registration. |
| **A5** | **Merge the atlas's duplicate lineage labels** | **High for per-cell reporting, nil before it.** The same lineage appears under several atlas entries whose per-cell readings differ by more than anything biological — purely by which reference panel defined their markers. Until they are merged, nobody can answer *which cell moved*. | 3–5 days | A merge rule plus a re-measured per-entry reference. |
| **A6** | **Per-entry healthy bands wide enough to carry a tier word** | Moderate. 20 % of atlas entries have a healthy spread wider than the gauge's NORMAL band, so their tier is withheld and only the number is printed. | 2–3 days | Depends on A5. |
| **A7** | **Calibrate the bisulfite-conversion threshold** | Moderate. Currently reported, not applied, because every healthy array fell below the literature threshold — so applying it would refuse everyone. PROC-STAGE0-04 is pre-registered and waiting on the author, because a gate that refuses a patient's specimen is his to commission. | 1 day | **Author's decision**, not compute. |
| **A8** | **Re-derive the sealed intake distribution on the robust background** | Moderate. The sealed Stage 0 result used a mean-based background that the robust estimator replaced; bounded and checked (11 of 12 arrays rose or held, one fell by <1e-5), so the verdicts stand — but the published distribution should be the one the chain now computes. | 3 h | None. |
| **A9** | **Adopt the v0_3 marker panel and re-seal the anchors** | Moderate. Closes the marker-panel migration; prerequisite for per-cell work. The panel is in the tree marked TRIAL. | 1–2 days | None; both anchor matrices are downloaded. |
| **A10** | **Bands for secretory, cycling, terminal, stromal, stem_pluri** | High if achievable, **but not from blood.** Measured 2026-09-23: none of them clears its presence floor in healthy blood, and the detection limit is 2 %. A healthy band cannot be fitted to values the instrument cannot resolve. | — | **Acquisition, not compute:** healthy *tissue* or cfDNA with a commissioned laboratory zero. This is the one place where "just spend the time" does not apply. |
| **A11** | **A serial cohort — two draws from one person** | High, and unlocks the single most sensitive design available (see B5). | — | **Acquisition.** Public extracts do not carry repeat-draw metadata. |

---

## List B — cosmology methods still on the shelf

Every entry here is already listed in the report's own **Safeguards** tab as `NOT_BUILT`, so the shelf is
visible on every run rather than living in a document.

| # | method | borrowed from | what it would buy | effort | blocker |
|---|---|---|---|---|---|
| **B1** | **Generalised least squares on the full covariance** | CMB likelihood analysis, where the covariance is never diagonal | Same as A1 — the largest single improvement available to the chain. Cosmologists would not dream of throwing away the off-diagonal terms; we are. | 2–4 days | none |
| **B2** | **Angular power spectrum of the residual sky** — *measured 2026-09-25, [NOT COMMISSIONED as a reference](PROC_CLS_01_OUTCOME.md) but the finding stands: the residual sky IS spatially structured, 3.5x its own permutation null at l 2-8, decaying to nothing by l~200. Structure is LARGE-scale, not pixel-scale. B3 failed in both directions (Munich 0.650, UCLA 0.906), which points at the per-laboratory residual scales - do A3 first* | the CMB power spectrum itself | **One number per specimen, computable today on data already on disk:** is a departure locally clustered along the genome, or spread across it? A focal lesion and a systemic process should look different, and nothing in the chain currently asks. Highest ratio of insight to effort on either list. | 1–2 days | none |
| **B3** | **Per-patient posterior for the composition** (sampling instead of a point fit) | parameter estimation from a CMB likelihood | Directly attacks the defect that motivated Stage 2c: a constrained point estimate pins a trace component at exactly zero, while a posterior has a tail. Would give every fraction a credible interval instead of a number. | 4–7 days | compute cost per specimen; needs a bar on runtime |
| **B4** | **Internal linear combination on the residual sky** (not only on the composition) | CMB foreground cleaning | Today the needlet solver gives a second opinion on *composition*. Applied to the residual sky it would separate a departure from the genomic-correlation background, which is what the 57 σ mottling is. | 3–5 days | none |
| **B5** | **Difference maps** | CMB detector differencing | The technical term cancels: paired detection limits drop to ~0.0056 at island scale — an order of magnitude below published island effects. | 2 days once data exists | **A11** — no serial cohort |
| **B6** | **A real beam with a stated resolution** | the instrument beam | The plate smooths at a fixed scale chosen by eye. A measured beam would let the chain state the genomic resolution of a departure instead of implying one. | 2 days | none |
| **B7** | **Cross-spectra between class panels** | CMB temperature–polarisation cross-spectra | Is a departure shared across cell classes (systemic) or confined to one (focal)? Cheap once B2 exists. | 1 day after B2 | B2 |
| **B8** | **Apodised masks instead of a binary presence floor** | the galaxy mask | A class at 1.9 % is masked and a class at 2.1 % is fully trusted. A graded mask would weight by how well the class is actually constrained. | 2 days | none |
| **B9** | **Degeneracy / Fisher analysis of the composition** | the banana degeneracy in cosmological parameter space | Which composition solutions are genuinely distinguishable, rather than assuming the reported one is unique. Turns "the solvers disagree" into "these two are degenerate along this direction". | 3 days | none |

---

## The prerequisite three items share (measured 2026-09-25)

PROC-BAND-01, PROC-CLS-01 and PROC-LABBAND-01 asked three different questions and failed for one reason:
**the published 80-array panels are too thin**, and the full-cohort Stage 1 output they replaced was never
preserved. The joint age curve came out non-monotone on 234 arrays where immune's was fitted on 1,379; the
sky reference failed on per-laboratory residual scales each built from 40 arrays; and a 1.40x width ratio
at n=80 is indistinguishable from four identical laboratories.

**One input unblocks three items: Stage 1 on the full four cohorts, 1,379 arrays.** About 1,400 IDAT pairs,
~15 GB and a day of calibration. Nothing else on either list has that leverage, and A2, A3 and B2 should all
wait behind it rather than be retried on the panels that have now refused them three times.

## What to do now, and what to do later

**Now, before any new cohort — in this order:**

1. **A2** (band progenitor and stem_adult) — 2–3 hours, no new data, converts the departure statistic from
   one axis to three. Nothing else on either list is this cheap.
2. **B2** (angular power spectrum) — 1–2 days, no new data, and it answers a question the chain cannot
   currently ask about any specimen.
3. **A4** (withhold the immune tier under foreign material) — a correctness fix on a number the chain
   reports today, and the exact confound a tumour-bearing specimen presents.
4. **A1 / B1** (the full covariance) — the largest improvement, and the one that makes several others
   possible. Start it once the three above are sealed.

**Later, deliberately:** A3 and A5–A9 are real work with real payoff but none of them changes what the chain
can *see* as much as A1 does. A10 and A11 are acquisition and should be pursued as data requests, not as
compute.

**A rule for all of it.** Every item here that changes a reported number gets a pre-registration with its bar
fixed before the run, and is adopted only if it clears that bar on data it did not help choose — the same
discipline that made PROC-SMALL-01 trustworthy. An improvement adopted because it made the numbers look
better is how a tool starts lying.

## Held-out disease test, 2026-09-26

[PROC-EPIC-01](PROC_EPIC_01_OUTCOME.md) is the first test of the commissioned chain against disease data. The colorectal arm replicated on genuinely held-out EPIC-Italy blood (313 arrays entered the bars, of 516 held out) (d = +0.60, p = 0.0004 at > 5 years; loudest at 2-8 years); the breast arm did not (d = -0.32, p = 0.93), and the pre-atlas 'replication' that reported it had re-used 146 of its 224 breast cases from its own discovery set. The composition guard adopted the day before was shown not to be confounded with disease status (4.0 % of cases withheld against 3.7 % of controls).

**The binding limit is now cohorts, not method.** The > 8 year question rests on 12 held-out colorectal arrays; no analysis of this cohort can settle it.

[PROC-PARTIAL-01](PROC_PARTIAL_01_OUTCOME.md) closes the question of scoring a non-blood class's FIDELITY from an ordinary blood draw: NOT COMMISSIONED, and not close. At a 2 % fraction the recovered profile is a mean beta of 12.4, where a beta must lie between 0 and 1. The estimator's arithmetic is exact (machine-precision recovery on a synthetic host); what fails is the composition model, which reconstructs real blood at the identity loci with a +0.067 bias that 1/f amplification turns into 3.35 at f = 0.02. Averaging over 30,000 loci reduces noise by root n and does nothing to a bias. Detection and quantification are unaffected: presence to ~2 %, fraction above ~5 %.

## Which substrate next, 2026-09-26

[SUBSTRATE_STRATEGY.md](SUBSTRATE_STRATEGY.md) answers it from the rule PROC-PARTIAL-01 measured: a class is scoreable when the reference reconstructs the specimen at its identity loci to within about f x (the class's signal). That makes the substrate question a question about FRACTION. Tissue (tumour lineage 0.5-0.9) is the decisive case and needs no new calibration, because a paired tumour-versus-adjacent-normal comparison cancels the laboratory zero, the age term and the donor. Stool is out (bacterial-dominated, low human fraction). Plasma cfDNA can carry the IMMUNE reading but not the tumour one, and needs a scale map for fragmented DNA.

[PROC-TISSUE-01](PROC_TISSUE_01_OUTCOME.md) did not license the stool route: the field effect that would have justified collecting colonocytes was not observed, and the cohort turned out to be bulk mucosa (23 % epithelial), so the epithelial gauge was never tested. FOURTH failure today with the same cause: the reference. Thin panels sank PROC-BAND-01, PROC-CLS-01 and PROC-LABBAND-01; a +0.067 misfit against real blood sank PROC-PARTIAL-01; and here the atlas represents colon poorly enough that a 23 %-epithelial specimen has no epithelial class to select. Every route forward passes through the atlas.

## What the instrument can read, measured

[ATLAS_READABILITY.md](ATLAS_READABILITY.md) stops treating the recurring reference failure as a surprise and measures it. The atlas is an IMMUNE atlas with eight labels on it: immune holds 51 of the 115 cell types; stem_adult and stem_pluri hold ONE each. Three class pairs are collinear - stem_adult vs progenitor at r = +0.989, progenitor vs immune at +0.958, cycling vs secretory at +0.955 - which is the cosmologist's situation exactly and the argument for spending the covariance: two components at r = 0.989 are individually unidentifiable while their SUM is well determined, and the chain already hand-codes that one case in section 108 instead of getting it from the off-diagonal terms.
