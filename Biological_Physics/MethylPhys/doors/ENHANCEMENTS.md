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

## Correction, 2026-09-26: the covariance is NOT recoverable from the MCMC archives

This list has said that the full cell-type covariance is recoverable from the per-class MCMC archives. **It is not, and this was checked rather than assumed.** The archives hold only marginals (`cpg_id, mean, sd, ci_lo, ci_hi`); the build script's own correctness note states that *each CpG's posterior is INDEPENDENT in the model*; and the eight classes were run as separate jobs. There are no joint draws, so **no cross-class covariance at an address was ever estimated**. Any plan that budgeted days for mining it was budgeting for a quantity that does not exist.

What replaces it is better aimed anyway. The covariance that governs the composition fit is the **residual** covariance on real specimens - reference error, biological variation, technical noise and **model misspecification** - and it is the last term that closed PROC-PARTIAL-01 (+0.067 beta), which a posterior covariance would not have contained at all. It is estimable today from the 318 calibrated healthy arrays already on disk. See [PROC_COV_01_PREREG.md](PROC_COV_01_PREREG.md).

## The per-cell reading already exists

[PER_CELL_SCORING.md](PER_CELL_SCORING.md): all 115 cell types are ALREADY scored on their own discriminative markers against the H_min of their architecture class - verified in iamatlas_a_scoring.score_per_celltype, whose docstring says so verbatim. Nothing is pooled before deconvolution. The pooled class gauge is a separate later stage (stage_b_identity) that builds only immune and haematopoietic progenitor, and it is defensible exactly where it is used and nowhere else. THE GAP IS BANDS: only immune has a measured healthy band; every other entry is an UNMEASURED PLACEHOLDER, so a per-cell A can be computed but not PLACED. That is a calibration task on data in hand - the 732-array Uppsala panel now calibrating, cross-checked against the 318 published arrays - and it is what turns 115 computed numbers into 115 readable ones.

## The chain runs two deconvolutions

[TWO_FIT_FINDING.md](TWO_FIT_FINDING.md): walther_iam_deconvolver performs TWO independent NNLS solves - one against the 8 pooled class columns producing class_fractions, one against the 114 cell-type columns producing celltype_fractions - and they are not related by summation. Every REPORTED composition number comes from the pooled solve. On 48 healthy arrays the pooled fit puts immune at 0.8233 where the cells sum to 0.9136, moving ~8 points of immune mass into progenitor and stem_adult, which is exactly what the measured collinearity (immune/progenitor r=+0.958, stem_adult/progenitor +0.989) predicts. The section-108 joint-component rule exists to paper over a degeneracy the pooled fit CREATES and the cell-level fit does not. Switching the reported composition to the cell-level solve needs its own pre-registration with an invariance bar, because it changes numbers in sealed procedures.

## The chain verified against constructed truth

[PROC-SYNTH-01](PROC_SYNTH_01_OUTCOME.md) ran the author's requested verification THROUGH the chain. The architecture passes: composition recovered exactly, and a pure cell type named and scored against its class's floor to 1e-6. But the per-cell A is CONFOUNDED WITH FRACTION - the same cell reads 0.7447 to 1.1784 depending only on how much of it is present, a spread 8.3x the immune band width. Immune in whole blood works because the substrate nearly fixes the fraction (p5-p95 span 0.1746), and even there fraction explains 19.3 per cent of the reading's variance and moves it by 0.0435 against a 0.0524 band. PER-CELL BANDS MUST THEREFORE BE FRACTION-CONDITIONED, which is the thing to fix before they are measured. It also cost PROC-EPIC-01 21 per cent of its colorectal effect, which survives adjustment at d = +0.476, p = 0.0062.

## The per-cell A is on the wrong surface, 2026-09-26

[REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) is the highest-priority item on this list and it displaces the fraction-conditioned bands proposed the night before. Every cell's own atlas mean must read A = 1.0 by design; only 11 of 115 do, and 65 read below 0.5. The cause is that the per-cell path scores each cell on its DISCRIMINATIVE MARKER panel, which is 77-100 per cent near-binary, so mean per-CpG entropy is ~0 by construction. On identity loci the same references read 0.89-1.14, median 0.9876. The fix is a CONSTRUCTION - per-cell identity loci, mirroring how the eight class panels were built - because a cell's class loci would make all 51 immune cells read identically.

## Documentation catch-up after the per-cell surface, 2026-09-26 — author: "not yet, but keep it on the list"

Parts of the chain found today that no document names, to be written into every place that lists the
chain's parts once the per-cell bands are rebuilt on the identity surface:

| found | where it lives | what it is |
|---|---|---|
| [`percell_reference_v0_3.json`](../chain/Runtime%20Matrices/Percell_Reference/percell_reference_v0_3.json) | `chain/Runtime Matrices/Percell_Reference/` | per-cell per-laboratory healthy bands with a held-out check, built 2026-09-22; the folder was NOT on the chain's search path, so the chain never loaded it |
| [`iamatlas_percell_identity_loci_v1_0.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_percell_identity_loci_v1_0.json) | `chain/Runtime Matrices/A_Scoring_Module/` | 102 per-cell identity panels at the class floor, the surface the per-cell A is now read on |
| `_score_one_identity` | [`iamatlas_a_scoring.py`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_a_scoring.py) | the per-cell A on identity loci with the class gauge's formula |
| [`percell_reference_identity_v1_0.json`](../chain/Runtime%20Matrices/Percell_Reference/percell_reference_identity_v1_0.json) | `chain/Runtime Matrices/Percell_Reference/` | the per-cell bands rebuilt on the identity surface |

Documents to update: the SOP (`sop/MethylPhys_CPG_SOP.md`), the OM (`manual/MethylPhys_CPG_Operations_Manual.pdf`
via its build), [`REVIEWER_MANIFEST.md`](REVIEWER_MANIFEST.md), [`COMPONENT_MAP.md`](COMPONENT_MAP.md), [`RUNBOOK.md`](RUNBOOK.md), [`CHAIN_SEQUENCE.md`](CHAIN_SEQUENCE.md), the chain
inventory, and the report itself. **The report restructure leads with the physics**: per-cell A, fraction,
class floor and band on the front page; cohort-relative surfaces (Mahalanobis departure, cellular age, hull,
marker union) demoted to a labelled section or removed. The A-score is the instrument; nothing in the report
is to be phrased in another group's methodology.

## Deconvolver repair, 2026-09-26

Breast read 0.000 in breast tissue. Four causes found and fixed by measurement - no per-cell marker quota, grand-mean filling across twelve source families of wildly different coverage, duplicate cells solved as separate columns, and raw (unmapped) betas into the solve - plus the author's uniqueness markers as a union with the variance set. Healthy blood now reads median 0.000 non-blood (p90 0.023) across 48 arrays and four laboratories; 10 % spiked Breast is recovered at 0.102. Record: [`DECONVOLVER_REPAIR_2026-09-26.md`](DECONVOLVER_REPAIR_2026-09-26.md). Two questions closed by measurement: unclassified blood subtypes are NOT the leak (an unmerged NK twin was), and a substrate-only atlas is NOT less noisy (fit residual 0.073 vs 0.063) and cannot detect shed tissue at all.

**Next (author, 2026-09-26): the MATCHED FILTER as a detection stage.** Component separation today weights every locus equally; the healthy-blood residual is a reproducible, structured misfit (PROC-COV-01), which is exactly what a covariance-weighted template filter is built against. Test as a pre-registered detection-limit measurement on constructed spikes (0.5-5 %) against the 48-array healthy null, matched filter vs NNLS, at a fixed false-positive rate. No cohort needed.

**PROC-MF-01 outcome (2026-09-26): NOT COMMISSIONED.** The full-covariance matched filter cannot be estimated from 36 arrays x 1,506 markers and ties NNLS; per-locus inverse-variance weighting (the diagonal control) lowers the detection limit 2-10x on all four cells but detects without estimating (constant -0.03 offset = the structured misfit on the template). **Next: PROC-MF-02** - inverse-variance detection with the null median subtracted and sigma from the null spread, same six bars applied to it. See [`PROC_MF_01_OUTCOME.md`](PROC_MF_01_OUTCOME.md).

## Standing to-do, 2026-09-26 — the author: "I dont want to forget something"

The report and its documents FOLLOW the chain; nothing here is done until it is read on the rendered page.

**A. Detection stage**
1. ~~PROC-MF-02~~ **sealed NOT COMMISSIONED** (B1-B6 met, limits 0.5-1 %; B7 failed - the threshold does not transfer to a 783-marker EPIC matrix). ~~PROC-MF-03~~ **sealed NOT COMMISSIONED** (B1-B6 met again; B7 failed: the fifth laboratory's null is 6-25x wider at full markers, and its Breast/Prostate lines were set by two controls elevated on both those cells - overturns MF-02's marker-set diagnosis). **PROC-MF-04** if commissioned: per-laboratory WEIGHTS from the lab's own panel, the chain's composition gate ahead of the panel, p99 line rule with minimum n. See [`PROC_MF_03_OUTCOME.md`](PROC_MF_03_OUTCOME.md). See [`PROC_MF_02_OUTCOME.md`](PROC_MF_02_OUTCOME.md). Original item: inverse-variance (diagonal) weighted detection, leave-one-laboratory-out, null-median centred, σ from the null spread; the six MF-01 bars applied to *this* detector. Passes → detection stage ahead of the per-cell A, reporting (f̂, σ, detected yes/no at ≤ 1 FP in 48) per foreign cell.

**B. Report, PDF, SOP — following the deconvolver repair**
2. HTML Cells tab: tag family members inline in the class-table rows; print a family's A once; show `exclusive_markers` per cell; detection column once MF-02 passes.
3. Red Flags tab: `NOT_RESOLVABLE_ON_PLATFORM` (informational); a family reported as several cells (must be impossible — flag it); a foreign cell detected above the pre-registered false-positive rate (the shedding flag).
4. Troubleshooting tab: "cell reads fraction 0 — is it resolvable on this platform?"; "two cells show identical fractions — a resolution family"; "tissue cell found in blood — check detection significance before reading its A".
5. Safeguards / [`cmb_tools.py`](../chain/cmb_tools.py): `INVVAR_DETECT` (after MF-02) with a per-run check; `ILC_SKY` re-aimed or retired now that the solve block does the separation; a check that the deconvolver's solve block matches the commissioned one.
6. Operations Manual PDF: rebuild via `build_om.sh`, then READ the rendered pages — per-tab figures and `om_data.py RULES` must carry the four deconvolver causes and the resolvability rules.
7. SOP: LESSON-DECON-01 (coverage, filling, twins, mapped input) beside LESSON-SURFACE-01; pre-registration convention that future evidence files are plain names, not paths.
8. Documentation catch-up ("not yet" on 2026-09-26): SOP, OM, REVIEWER_MANIFEST, COMPONENT_MAP, RUNBOOK, CHAIN_SEQUENCE naming the per-cell identity surface, [`percell_reference_identity_v1_0.json`](../chain/Runtime%20Matrices/Percell_Reference/percell_reference_identity_v1_0.json), the Percell_Reference path, the solve block and twin rules. When the deconvolver stops moving.

**C. Chain**
9. CD4/CD8 — r 0.978, 33 and 8 exclusive loci; needs loci this block lacks (the 2,543-locus immune-subset source, second block).
10. Second-block solve for the nine finer blood subsets (~9 % of an array).
11. Twin/family thresholds (0.98 / 0.985 / margin 0.15 / 1 % coverage / 2,000 loci) as a runtime matrix, not constants.
12. Per-cell A on a family printed once.
13. Coverage floor on any class-selection rule (PROC-TISSUE-01's lesson).
14. EPIC platform block — today's block is 450K; GSE292312 and EPIC-Italy need the same measurement on 865k loci.

**D. Procedures waiting on the chain settling**
15. PROC-BRAIN-01 redo — clean single-provenance CSF run, uncontended machine, repaired deconvolver.
16. Gastric — six stomach entries, never tested; now two resolution families (diff / undiff).
17. Breast shedding in real patient blood — needs the detection stage first.
18. Per-cell bands on the identity surface rebuilt on the 732-array Uppsala calibration.

**E. Housekeeping**
19. Data bundles in the store: `MethylPhys_data_2026-09-26.zip` (2.57 GB) and `_part2.zip` (3.06 GB) — the author keeps them at `~/MethylPhys_data/`.
20. `guarded_push.sh` lives at `chain/`, not `kit/` — the run-book should say so.

**PROC-MF-02 outcome (2026-09-26): NOT COMMISSIONED on B7 alone.** See the register row B-10 and the outcome document.

**PROC-MF-03 outcome (2026-09-26): NOT COMMISSIONED on B7.** Register row B-11. GSE51032 is a 450K array, not EPIC - corrected in the outcome documents.
