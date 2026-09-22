# Row 9 working note - the report (unsealed build)

## 2026-09-22 - MethylPhys CPG builder (CPG_Engine/MethylPhys_Interface/build_methylphys.py) replaces cpg_report_v3.py
Author's direction: the v3 table was 'very blah'; the report is now the researcher interface - dark, tabbed, printable, Clinician/Researcher toggle, eleven tabs
(Reading, Every cell, Departure, Sky, Healthy reference, Integrity, Chain, Physics, Story, Record, Run). BREACH stays as a gauge reading on the CLASS gauge (a temperature);
no comparison to any prior cohort; no condition named; no years. Vocabulary guard on the four measurement tabs.
- Conductor now passes `cells_all` (all 115 per-cell A) and matches the laboratory key by accession prefix (departure false-alarm rate was printing 'not measured' for Uppsala - fixed).
- First read of GSM2333901 against the spec: (a) stem_adult and progenitor printed the SAME A_mapped 0.9262 - the conductor scores them as one joint haematopoietic-progenitor component
  on shared identity loci; collapsed to one gauge row, labelled as joint. (b) The same lineage appears under many atlas entries with a two-fold spread in per-cell A on one healthy array
  (Neu 1.158, granulocytes 1.123, Neutrophils_EPIC 1.121, Neutrophils_reinius 1.117, Neutro 1.083 ... neutrophil 0.545): on the marker surface, WHICH REFERENCE PANEL DEFINED THE CELL is a
  larger effect than anything biological in a healthy sample. Consequence: the per-cell healthy range must be per atlas ENTRY (its own markers), never per cell name; the report should
  group aliases visibly. (c) Class composition arrives in percent, cell fractions in [0,1] - still an open conductor item; the builder scales.
- Per-cell healthy reference (H_ref(entry), p10-p90 per laboratory) in build from 80 healthy arrays per laboratory (seed 2029; GSE87571/42861/111629/125105), raw IDAT -> Stage 1 noob,
  Stage 1 betas checkpointed per laboratory as artifacts + manifest the moment they exist (author: 'I wish you would have saved the betas'). Until then the cell table prints 'pending'.
- Substrate x specimen grid (5 substrates x whole blood / plasma cfDNA / tissue / CSF / urine / stool) reserved: every cell prints its commissioning status. Not yet in the builder.

## 2026-09-22, later - the per-cell healthy reference exists; four author corrections; two of my own errors found by reading output

**Per-cell healthy reference built** (`CPG_Engine/Runtime Matrices/Percell_Reference/percell_reference_v0.json`, builder `MethylPhys_Interface/build_percell_reference.py`).
All **115 atlas entries x 4 laboratories = 460 entry-laboratory cells**, from 318 healthy arrays (Uppsala 80, UCLA 80, Munich 80, Karolinska 78; seed 2029, 40 build + 40 held out per laboratory),
raw IDAT -> Stage 1 noob. Per entry: H_ref = median of mean_i H(beta_i) over that entry's markers on the build panel, with p10/p90, and A percentiles against the class H_min.
Every laboratory's Stage 1 beta matrix is a checkpointed artifact with its GSM list, seed, Stage 1 version and SHA-256 - the rule from this morning is now satisfied for this layer.
**Status: EXPLORATION, UNSEALED** (sealing rule: seal a working tool against a bar, not a build).

**Two errors of mine, both caught by reading the output rather than trusting it:**
1. *Plain mean propagated NaN.* The first build used `numpy.mean` over the per-CpG entropies, so a single missing marker voided the whole array; `dropna` then silently dropped most arrays and only
   24 of 115 entries got a reference from one laboratory. `nanmean` fixed it: 115/115 entries, all four laboratories.
2. *The reference was on the wrong beta scale.* I applied the Stage 1s pipeline map when building it. `cpg_conductor.stage_a_cells` scores the per-cell surface on the **RAW** calibrated betas -
   only the class gauge receives the mapped ones. The mismatch manufactured spurious 'below range' calls. The diagnosis came from the printed immune block of one healthy
   Uppsala array: **13 of the 20 immune rows printed read 'below'** their own healthy range, including every T-cell and NK entry - which cannot be true of a typical healthy array. The pre-fix
   total across all 115 cells was **not counted** (only the immune block was printed), so no pre-fix total is claimed here; it necessarily exceeded 13. Post-fix, measured over all 115:
   **101 within, 14 below**, with cells including CD14_monocytes and CD56_NK-cells flipping from 'below' to 'within'. Recorded in the builder's header.
   RULE: a reference is built on exactly the scale the stage that consumes it receives.

**Held out**: a held-out array falls inside its entry's own p10-p90 with median 0.75 (IQR 0.70-0.82; nominal 0.80; >= 0.80 in 35 % of entry-laboratory cells). Not yet at the bar. Before this is sealed
it needs either a wider interval definition or the alias merge below; as it stands the per-cell range is shown with its status, and no tier word is printed on a cell.

**The alias spread is now measured, not suspected.** On the marker surface, one lineage's atlas entries sit far apart: neutrophil-lineage pooled A_p50 runs `Neu` 1.142, `Neutro` 1.104,
`granulocytes` 1.128, `Neutrophils_reinius` 1.129, `Neutrophils_EPIC` 1.105 - and `neutrophil` **0.551**. That spread (~0.59) dwarfs anything biological in a healthy sample, and it is a property of
*which reference panel defined the entry's markers*, not of the cells. Consequences: (a) an entry is only ever read against **its own** reference, never against another entry's; (b) duplicate-label
entries must be merged to one lineage before any statement about 'which cell moved'; (c) the merge rule is an open item.

**Author corrections applied to the report (2026-09-22):**
- **Floor vs ceiling, corrected.** I had written the 1.07/1.10 lines as approaching 'the ceiling', and described the ceiling as where identity is lost. Wrong: **H_min is the floor** (A = 1.00, health;
  BREACH at 1.10 is losing the floor) and **1/H_min is the ceiling** (saturation - the instrument running out of scale). Now a table on How-to-read contrasting the two, plus the correction that the
  ceiling varies by class **and by substrate**.
- **Saturation chart added**: the frozen 40-value H_min table (8 classes x 5 substrates) rendered from `cpg_gauge_engine.H_MIN_TABLE`, each cell showing the floor and the ceiling it implies. Read across
  a row and the argument for five substrates is visible (terminal: WPS floor 0.959 -> ceiling 1.043, almost no range; fragment size 0.625 -> 1.600).
- **Warburg line at 1.07 explained** from `tier_breakpoints.json`'s own `physics_meaning` and `customer_paragraph`: a boundary LINE, not a tier band, past which adding metabolic fuel can accelerate
  rather than correct the drift. Stated with its provenance: inherited from the Issue 002 tier system and **not re-derived on the commissioned chain** - a named open item.
- **Coverage tab added**: the five substrates (with their published single-substrate AUCs and the specimen each requires) x seven specimens, as a grid. One cell is lit - methylation x whole blood -
  and every other cell prints what it needs (pipeline map, laboratory zero, healthy band). With the statement that a reserved cell is 'not yet tested', never 'cannot'.
- **Benign well-differentiated growths**: the astro-genetics passage is on How-to-read (identity vs cell number; lipoma/fibroid/nevus; shape-independence), with the colorectal/breast progression series
  marked design-record history rather than a result of this chain.
- **Links**: every chain stage now has a collapsible fold (goes in / comes out / why it exists / what it refuses / commissioned by) with the live file, hash and GitHub link; the atlas, the 115->8 class
  map, the calibration provenance, the CpG->HEALPix mapping, the RUNBOOK, CHAIN_COMMISSIONING and HANDOFF are all linked; NILC is linked with its vindication stated; Issue 003 is a deep-dive link at the
  bottom of six tabs. **The sky mapping was verified against the atlas plate mapping: 483,092 of 483,092 CpGs to the same pixel** - the patient's plate and Plates 1-4 are on one projection, and the
  plates are embedded on the Sky tab as the reference skies. The brightness CSVs are described as superseded (the sky now weights the expectation by the sample's own composition) rather than retired.

### 2026-09-22 - the gauge semantics corrected from the author's two gauge figures (figB cell, figC cosmic)

My wording had **A = 1.00 'sitting on the floor'**. Wrong, and the author corrected it: A = 1.00 is the **healthy reference, the ideal baseline in the middle of the NORMAL band** - not an edge of
anything. Three distinct things, now stated as a table on How-to-read and consistent with both of his gauge figures, which are embedded there (`Plates/CPG_Gauge_Cell.png`, `Plates/CPG_Gauge_Cosmic.png`):
1. **A = 1.00** - the calibration point. (His cell gauge: BELOW NORMAL/suppressed < 0.95, NORMAL 0.95-1.05, MARGINAL 1.05-1.07, Warburg line 1.07, DETECTABLE 1.07-1.10, BREACH >= 1.10. Our measured
   commissioned band, central 95 % of 1,379 healthy donors = 0.954-1.041, sits inside his canonical NORMAL band - worth noting as independent agreement.)
2. **H_min(class)** - the constant in the denominator, in bits: the entropy level below which that architecture cannot hold the pattern that makes it that cell type. It is **not a mark on the A axis**;
   it is the unit the axis is drawn in. Readings well below 1.00 are the suppressed / inverted direction (his figure puts post-chemo and immunosuppressed near 0.90) - a direction the report had been
   under-representing relative to the drift direction.
3. **1/H_min** - the ceiling, saturation. Class- **and** substrate-specific, and it may sit either above or below breach.
**The consequence, which his cosmic gauge teaches better than any prose: the ceiling can sit BELOW breach.** An isolated white dwarf reads above healthy yet is structurally capped below breach because it
has no mechanism to gain mass; only a collapse-capable core reaches A_IAM = 1 (Chandrasekhar / TOV / Schwarzschild), which is why his figure rescales gravitational saturation onto cellular breach at 1.10.
**Measured from the frozen 40-value table: 15 of the 40 class-substrate combinations have a ceiling below breach 1.10** - nucleosome occupancy caps 7 of 8 classes (floors all 0.98-0.99); **fuzziness** caps the three
stem/progenitor classes (stem_pluri, stem_adult, progenitor) while **WPS caps a different three - terminal, stem_adult, progenitor** - and does NOT cap stem_pluri (WPS ceiling 1.105, just above the
line); and on methylation, the one lit column, **pluripotent stem is capped at 1.018**, so a pluripotent-stem breach cannot be read on methylation at all. The chart now
flags every capped cell. Ceiling-capped is not safe and not healthy: it means *that substrate cannot tell you*, which is the five-substrate argument as a structural limit rather than a preference.
**Open question raised by the same arithmetic:** the breakpoints file carries reference clusters past breach - senescent 1.24-1.27, malignant 1.28-1.32. The highest methylation ceiling of any class is
terminal at 1.294. So the senescent range sits entirely below that one ceiling and above every other class's, while the malignant range **straddles it**: 1.28-1.294 is reachable on the terminal class
only, and **1.294-1.32 exceeds every methylation ceiling there is** - no class can produce those readings on this surface. (My first wording said malignant was 'reachable on the terminal class and on no
other', which is wrong for its upper part; corrected.) Which class and which surface those clusters were measured on must be stated wherever they are quoted - the marker-union surface has a different
denominator and different limits. Carried here as corpus reference values, not re-measured.

### 2026-09-22 - Issue 002 read on the clusters and the ceilings (author: 'I would read the section in Issue002 to be sure')

Read `Papers/IAMPerformance_GAPEIssue002.pdf` pp. 12, 21, 22, 27, 28, 33, 62.

**1. The saturation chart is his, from April, and it reproduces exactly.** Issue 002 p12 is the SATURATION WALL CHART - all 40 class-by-substrate combinations, each with its ceiling
`1/H_min`, flagged **SAT** (saturates below BREACH 1.10), **TGT** (tight ceiling, A_max < 1.15) or unflagged (full headroom). Parsed from the PDF and compared against
`cpg_gauge_engine.H_MIN_TABLE`: **40 rows, every ceiling equal to 1/H_min to three decimals, 15 SAT, 2 TGT, zero mismatches** - the same cells and the same flags I had derived this
afternoon. His framing is the better one and the report now uses his vocabulary and credits the source: the direct analogue of the Dennard scaling walls (frequency, power, cost) in
semiconductor physics. The 2 TGT cells are adult-stem methylation (1.145) and pluripotent-stem WPS (1.105).

**2. The malignant cluster is terminal-class and measured.** pp. 21/22/27: lower-grade glioma **A = 1.2846** and glioblastoma **A = 1.256** on methylation, both Ceccarelli 2016 TCGA
(n = 516, n = 149), against a healthy frontal-cortex neuron reference at A = 0.9692; terminal-class cancers carry the largest departures in the 28-cancer panel (dA ~ 0.22-0.27). So the
breakpoints file's `malignant_cells` 1.28-1.32 has a real source at its lower end, on the one class whose methylation ceiling (1.294) can host it. The consequence is sharper than
'inconsistent': **the largest cancer signal in the panel sits within 0.01 of its own class ceiling** - the regime where a second substrate stops being a refinement and becomes the only
way to keep measuring. Issue 002 says exactly this on p22, and discloses that its own LGG/GBM nucleosome and WPS values are placeholders at the class ceilings pending reanalysis of the
Corces 2018 bigWigs (G-2026-P023).

**3. A = 1.00 is 'the architectural commitment point, not a mathematical floor'** (p22, his words), and under the unfloored formula his healthy references sit slightly *below* it, at
A ~ 0.97, because the healthy beta gives an entropy just under the MCMC central estimate of H_min. On this chain healthy reads 1.00 - not because the formula changed, but because the
laboratory zero and the age term are measured from healthy donors of that laboratory and decade, which places the healthy population at 1.00 by construction. Same instrument, same floor;
one quotes the raw ratio, the other quotes it after two measured offsets. **Any A quoted from Issue 002 must say which of the two it is.** Now stated on the report.

**TWO RECONCILIATION ITEMS RAISED (for the Issue 003 register):**
- **RECON W1 - two different 'Warburg' numbers.** `tier_breakpoints.json` carries `WARBURG_TRANSITION` as a boundary line at **1.07**, pre-breach, described as where metabolic support
  can start to accelerate rather than correct the drift. Issue 002 pp. 33/62 use the same word for a **post-breach zone boundary at ~1.15** ('Ceiling 1.10 -> Warburg ~1.15 -> glucose
  inversion ~1.25 -> no return ~1.40+'), explicitly flagged there as qualitative therapeutic-window boundaries pending G-2026-P025, not diagnostic tiers. Two objects, one name. The report
  currently prints only the 1.07 line, with its provenance and the statement that it has not been re-derived on this chain. The naming must be resolved before both documents are read together.
- **RECON W2 - two different 'ceiling' quantities in Issue 002.** p12's wall chart equals 1/H_min exactly. The per-class post-breach tables on pp. 33 and 62 print a column also called
  'Ceiling (A)' whose values do **not** equal 1/H_min except for nucleosome occupancy: secretory methylation 1.27 (1/H_min = 1.186), cycling methylation 1.30 (1.168), cycling fuzziness
  1.28 (1.221), WPS 1.18 (1.594). Whatever those numbers are, they are not the saturation ceiling. Not carried onto the report until identified.

### TEST PLAN - what to run on the commissioned chain to settle this ourselves (author: 'we need to plan on testing some of this out ourselves with the finished trusted chain')

Ordered so that each step is runnable when it is reached, cheapest first. None of it is started.

1. **T-CEIL - the ceiling conformance test. Runnable today, costs nothing, and it is the sharpest falsifiable statement in the whole framework.** The claim is that no reading can exceed
   1/H_min for its class and substrate. Run the commissioned identity gauge over every array we hold - the 1,379 healthy donors, the 318 Stage 1 rebuild arrays, the commissioning arrays -
   and assert that no A ever exceeds its class ceiling. A single reading above it falsifies either the floor value or the formula. Expected: none, because the arithmetic forbids it; the
   test's real value is that it turns an arithmetic identity into a standing regression test that any future change to a floor or a scale has to pass.
2. **T-TISSUE - commission solid tissue, which every remaining test needs.** Today tissue runs end to end and reads plausible composition, but no tissue laboratory has a pipeline map, a
   laboratory zero or a healthy band, so every class prints NOT REPORTABLE. TCGA solid-tissue normals are public, one processing pipeline, hundreds of arrays per tissue: enough for the
   three layers. This is the gate on T-GLIOMA, T-SAT and T-CLUSTER.
3. **T-GLIOMA - reproduce the largest claim in the panel on the commissioned chain.** Ceccarelli 2016 LGG (n = 516) and GBM (n = 149) methylation arrays are public through GDC. Prediction
   to seal *before* running: terminal-class A on the identity gauge, with the tissue map and a TCGA laboratory zero, lands above breach and near the Issue 002 values (1.285 / 1.256).
   This is the cleanest available test of whether the pre-atlas cohort-mean instrument and the commissioned identity gauge agree on the biggest signal in the corpus - and if they disagree,
   the size and direction of the disagreement is the finding.
4. **T-CLUSTER - place the senescent and malignant clusters properly.** Re-measure both ranges on the commissioned chain and state, for each, the class and the surface. Closes the open
   question of the 1.32 upper bound, which exceeds every class's methylation ceiling and therefore cannot be a methylation reading.
5. **T-SAT - the saturation ordering test, which is the five-substrate argument made falsifiable on one substrate.** Within methylation alone the ceilings differ by class: terminal 1.294
   against pluripotent stem 1.018. Prediction: in a tissue where both classes are present, a departure large enough to breach on the terminal class will read as saturated (pinned at 1.018)
   on the pluripotent-stem component of the same array. If the stem component instead reads freely above its ceiling, the floor table is wrong.
6. **T-WARBURG - parked, and honestly.** Testing 1.07 as a metabolic-intervention boundary needs a cohort with metabolic intervention and outcome, which no public methylation dataset
   provides. Until then the line is reported with its provenance and the statement that it has not been re-derived here. RECON W1 must be closed first anyway, since the two documents
   currently disagree about which number the word names.

### 2026-09-22 - precedence paragraph corrected: a summary of a search is not the search

**What went wrong.** The Europe PMC query `HEALPix AND methylation` returned **2 records** (and `HEALPix AND genome` returned **4**) - all structural-biology papers where
HEALPix is the cryo-EM orientation-sampling scheme and methylation appears incidentally. I read them as irrelevant, which they are, and then wrote *"returned no hits for
HEALPix with methylation"* into the Sky tab and into a memory note. Irrelevant is not zero. A review caught it against the tool's own hitCount, and it was caught on the one
paragraph explicitly framed as written *the way a referee will read it* - the worst possible place for a count that does not match the retrieved data.

**Corrected.** The paragraph now prints the counts per query pair (0, 0, 0, 2, 4, and the >1,500 abbreviation collision on CMB), says what the non-zero hits actually are, and
states plainly that this is **a bounded search, not a proof of absence**. The two HEALPix hits earn their place: they are how we know the pixelisation is already used in
biology, which is a better sentence than a zero would have been.

**Rule.** When a precedence or absence claim rests on a search, the document carries the **counts and the queries**, never a characterisation of them. "We found nothing
relevant" and "the query returned nothing" are different claims and only one of them is checkable.

### 2026-09-22 - two measurements behind the Sky tab, and a locality scare that was my own error

**PROJECTION LOCALITY (measured, and it matters).** Asked whether the sphere is necessary, I measured whether the CpG-to-pixel mapping actually
preserves genomic locality. First attempt used the **atlas row index** as a proxy for genomic order and reported the mapping as nearly random
(median neighbour gap ~100,000 rows). That was wrong: `IAMAtlasREBUILD.csv` rows are **alphabetical by CpG ID**, not genomic - the mapping itself was
built from the real manifest (CHR + MAPINFO). Re-measured against `external_manifests/EPIC_plus_HM450_combined_manifest_normalized.csv`:

| quantity | value |
|---|---|
| pixels whose CpGs are genomically contiguous | **100.0 %** (196,608 of 196,608) |
| pixels whose CpGs are all on one chromosome | **100.0 %** |
| median bp span within a pixel | **511 bp** (90th pct 20,481) |
| median genomic-order gap between neighbouring pixels | 1,256 CpGs (25th pct 40) |
| neighbouring pairs within 10 CpGs / beyond 10,000 | 25.0 % / 0.9 % |
| the same for a random assignment | ~161,000 |

So the projection is a genuine locality-preserving space-filling reindexing at 511 bp per pixel. **This retroactively supports the 57-sigma
interpretation**: the smoothed mottling really is the genomic correlation of methylation, because adjacent pixels really are adjacent genome.
Had the first (wrong) measurement stood, that interpretation would have had to be withdrawn. Lesson: a row index is not a coordinate - check what a
file is sorted by before using its order as a proxy for anything.

**DIFFERENCE-MAP SENSITIVITY (measured on the Uppsala panel, n=80).** Between-person beta SD per address: median 0.0289 (IQR 0.0182-0.0463). The 5th
percentile, 0.0089, is an **upper bound on the technical term** (nothing can be quieter than the noise). Paired 2-sigma detection limits, from
technical noise x root 2: 1 address 0.025; 5 addresses 0.011; **20 addresses (an island) 0.0056**; 100 addresses 0.0025; 1,000 addresses 0.0008.
Reported island-scale disease effects run several to twenty points, so an island-scale paired difference should resolve changes an order of
magnitude below published effect sizes. **Caveats that must travel with it:** the technical term is inferred from cross-sectional data because
**no repeat draws of the same person exist in anything held here** (the EPIC-Italy foundation cohort is 460 distinct participants, 0 repeats - checked),
and the cancellation only holds within one laboratory and pipeline. A serial cohort is now roadmap item 1.

**Report changes in this pass:** presence-floor row corrected (below its floor a class **is not there**, and the galaxy-mask analogy is explicitly
inexact - the galaxy hides a real sky, an absent class has nothing behind the mask); a section on what else the MCMC gives and the unused
cell-type covariance; the brightness/brilliance lineage as the first CMB-derived tool, superseded not retired; the author's difference-map
paragraph verbatim with the measured table; the sphere question answered with the locality numbers plus the linear-track / Hilbert-curve
recommendation for clinical reading; an **Acknowledgement** section crediting the cosmology community ('we are the messenger, not the inventor');
and a new **Roadmap** tab, 20 items from the translation map and this chain's own measurements, banana degeneracy included.

### 2026-09-22 - the composition question answered across 40 donors, and the departure statistic named

**"62.6 per cent neutrophils - is it always that way, and where are the other immune cells?"** (author). Answered by running the composition step
over all 40 healthy Uppsala build-panel arrays rather than defending one array. Placed-entry distribution, with the textbook white-cell
differential beside it:

| atlas entry | donors placing it | median % | range | textbook differential |
|---|---|---|---|---|
| Neutrophils | 40/40 | 47.0 | 31.9-68.7 | 40-70 |
| CD4 T cells | 38/40 | 18.7 | 1.2-34.3 | lymphocytes 20-45 total |
| CD8 T cells | 32/40 | 6.3 | 0.6-30.2 | " |
| CD19 B cells | 34/40 | 2.3 | 0.4-7.2 | " |
| CD56 NK cells | 40/40 | 13.4 | 4.3-32.5 | 2-10 |
| CD14 monocytes | 40/40 | 8.4 | 3.5-14.3 | 2-10 |
| GMP | 25/40 | 4.6 | 0.1-16.2 | not counted clinically |

Median 7 entries placed per donor (range 5-9 of 115). **So the other immune cells are all there** - the single array the author was reading (62.6 %)
is a high-normal donor inside a 31.9-68.7 % range, not a solver with one answer. Two honest departures now stated on the Reading tab: **NK reads
high** (median 13 % vs textbook 2-10), most likely the lymphoid non-separability PROC-SEP-03 measured, i.e. an NK panel absorbing T-cell signal -
a limit of the reference, not a finding about the donor; and a minority of donors place a **trace of something implausible** (gastric or glial
entries under 2 %, in 1-15 of 40 donors) - the conservative solver's false placements at its evidence threshold, reported rather than hidden.
Eosinophils and basophils have **no atlas entry at all**, so they are missing from the reference rather than from the sample.

**The departure statistic is now named and explained on its own tab.** It is a **Mahalanobis distance**: distance measured in units of how much
healthy people vary, accounting for the fact that the axes move together. Provenance stated honestly - **P. C. Mahalanobis, Indian Statistical
Institute, 1936, on skull measurements; not from cosmology.** But cosmology is among its heaviest users under another name: every parameter fit's
chi-squared, (d-m)^T C^-1 (d-m), *is* a squared Mahalanobis distance, so what cosmology contributed is the discipline of building it on a properly
measured covariance and not trusting it until the covariance itself is measured - the same pattern as HEALPix, which also came from outside
cosmology before cosmology made it standard. The tab also states what is thin about it today: with one commissioned class band the distance is just
|z| of that class and the covariance has nothing to act on.

### 2026-09-22 - the Story tab replaced by the author's own explainer, with three supersessions flagged

The author judged `What_Is_Astro_Genetics.tex` (Zenodo 10.5281/zenodo.18702042, May 2026) better than the Story tab I had written, and he is right -
it is his voice and the two correspondences are stated far more precisely than my paraphrase. The tab is now his text: who this is for; the
one-sentence version ('biology has a measurement problem that cosmology solved the tooling for'); stargazers and trailblazers with the two
correspondences (CMB <-> architectural drift as redundantly encoded classical information, Zurek; and the load-bearing one, inhomogeneous horizon
decoherence <-> inhomogeneous floor crossings, both writing irreversibly to a surface at a fixed multiple of the Landauer cost); the virial/balance
argument and the DNA-as-ledger passage; and the ceiling-is-informational-not-gravitational argument, which is what lets the account travel to a cell
whose mass bends nothing measurable.

**Three passages superseded by this chain's own measurements, flagged [updated] in place rather than silently edited:**
1. *'At A = 1.0 the system sits exactly at its floor'* - the author corrected this himself today: A = 1.00 is the healthy reference in the MIDDLE of
   the NORMAL band; H_min is the denominator constant and losing it is the failure event; the ceiling 1/H_min is saturation, a third thing.
2. *'the class-specific H_min anchor is held internal'* - no longer true or desirable: the floors, the calibration code and the bootstrap
   cross-check are public, and were in the author's own Zenodo deposit under an open licence from April 2026, so the withholding was never in force.
3. *the 27-of-28 TCGA figure and the cosmological ratios* - pre-atlas surface, cohort level, before the gauge switch / pipeline map / lab zero /
   presence floors existed. Part of the record, not results of the commissioned chain; reproducing them is a Roadmap item. Quoting them as this
   instrument's performance would be the exact stale-data failure the protocol exists to prevent.

Also added: a 'two names' note - **astro-genetics** is the programme (cosmology's tools pointed at the epigenome); **physics of methylation:
Landauer metrology** is the narrower field name the manual and the methods paper carry, because a methods paper should claim only what it measures.

### 2026-09-22 - the author inspected the interface's three controls; all three were broken

He asked what the Clinician/Researcher toggle changes, what the print button produces, and whether the Run tab links what a researcher needs.
I checked instead of answering, and the answers were: nothing, not enough, and no.

| control | what it did | fixed to |
|---|---|---|
| audience toggle | set a `researcher` class on `<body>` and stored it in localStorage - **and no CSS rule or element ever referenced it.** Decorative. | the tab table gained an audience column; 8 of 15 tabs (Healthy reference, Coverage, Safeguards, Integrity, Chain, Roadmap, Record, Run) and all provenance/SHA material are `.resr` and hidden in clinician view by a real CSS rule. Nothing is hidden from the researcher. |
| print | printed the five measurement tabs (the third column of the tab table was already a print flag, so my first reading of this as 'one tab' was wrong) - but **no provenance, and `<details>` folds printed collapsed** | print is audience-aware: clinician gets the five measurement tabs, researcher additionally gets Healthy reference, Safeguards, Integrity, Chain and Coverage; all folds forced open; tables/figures/gauges given `page-break-inside:avoid`; and every external link prints its URL after the text, so a paper copy is still traceable |
| Run tab | **zero links**, and it advertised `build_methylphys.py --idat-grn ... --idat-red ...` - **a command that did not exist**; the builder only ever took `--bundle` | 25 linked files at this commit with SHA badges (the kit guards, the RUNBOOK, lab_zero, the atlas and its provenance, the manifest, the band, the age curve, the per-cell reference), and `MethylPhys_Interface/run_sample.py` written so the advertised command is real: IDAT pair or a `cpg_id,beta` CSV -> Stage 1 -> run_full -> report, with the commissioning requirement and the refusal behaviour stated in a table |

**Tested before pushing, because this is the second time I have advertised a command:** `run_sample.py --betas` on a cached array returns immune
A'' 0.9951 IN_BAND NORMAL - identical to the conductor's own reading for that array - and the same call without `--lab` returns 5 refusals and no
placement, which is the fail-closed path working from the entry point a stranger would use.

**Lesson:** a control on a page is a claim. The toggle, the print button and the command block were all claims I had not checked, and two of the
three were false. Any interactive element ships with a check that it does what its label says.

## 2026-09-22 - item 1 was not an alias problem. The marker panels are not exclusive, and the cause is the selection criterion.

The author's item 1 was "merge the atlas's duplicate labels to one lineage per entry, so we can say which cell moved." Measuring it first changed
what the item is.

**What the marker file says about itself.** `iamatlas_celltype_markers_v0_2.json` records its own method:
`selection_method: one_vs_rest_top_N`, criterion `|target_celltype_mean - mean(other_celltype_means)|`, top 100 per cell type, 115 cell types.
That criterion compares each cell type against the **mean of all the others**. A CpG that is extreme in a handful of cell types therefore scores
highly for **every** one of them, because the mean of the rest is dragged toward the middle. The criterion selects globally extreme CpGs
repeatedly, rather than uniquely distinguishing ones.

**Measured consequence.**

| quantity | value |
|---|---|
| distinct marker CpGs / panel slots | 6,738 / 11,369 |
| markers belonging to more than one entry's panel | **2,278 (33.8 %)** |
| most panels one marker serves | **11** |
| median fraction of an entry's panel exclusive to it | **0.37** (10th percentile 0.06) |
| entries with a panel under 25 % exclusive | **36 of 115** |
| worst | **macrophage 0.0 %** - every marker it has also belongs to another entry; Cortical_neurons, dendritic, erythroblast, small_intestine, tcell all ~1 % |
| pairs sharing >= half their markers | 50, involving 26 of 115 entries |
| ...of those, pairs spanning DIFFERENT architecture classes | **28** - e.g. Cortical_neurons/stem_pluri share 91 markers; small_intestine/tcell share 82; erythroblast/tcell 81 |

**Why a name-based merge would have been wrong.** The name-normalised groups (10 of them) are not the pairs that share markers. `Neu` and
`neutrophil` are both immune neutrophil entries and share **zero** markers; `Neutrophils_EPIC` and `Neutrophils_reinius` share 49. Merging by name
would have pooled panels that measure different things, and left the cross-class overlaps untouched. The earlier note attributing the neutrophil
spread to "which reference panel defined the entry" was directionally right and mechanistically wrong: the spread is driven by panel
**non-exclusivity** (those entries measure 17-18 % exclusive panels), not by which laboratory named the cell.

**What was done, and what deliberately was not.** Measured and published as
`CPG_Engine/Runtime Matrices/Celltype_Marker/percell_exclusivity_v0.json` (per entry: n markers, n exclusive, exclusivity, the most panels any of
its markers serves, and an `individual_claim_ok` flag at a 0.25 threshold). The report now prints an **exclusivity column** for all 115 entries and
**withholds the individual direction claim for the 36 entries under the threshold**, printing the number with its exclusivity in amber beside it
and saying why. **The runtime marker file is unchanged**: the sealed foundation-cohort anchors reproduce on it, so repairing the criterion requires
a re-seal - the same situation as the chrX-removed trial copy in September.

**The repair, for the roadmap.** Replace `|target - mean(others)|` with a **nearest-rival margin**: require each marker to beat its
closest competing cell type by a stated margin, and cap how many panels a marker may serve. Then re-select, re-seal the anchors, and re-measure the
per-entry references. Until that is done, per-cell readings are honest for 79 entries and explicitly withheld for 36.

## 2026-09-22 - the author asked whether the Chain tab lists everything the chain uses. It did not.

**Audited rather than answered.** The conductor *was* linked on the Chain tab (not on the Run tab), but an audit of the live tree against the
rendered page found **20 load-bearing files linked nowhere** - including `IAMAtlasREBUILD.csv` itself, which engine code names seven times, plus the
array manifest, the four per-laboratory sky residual scales, the Mahalanobis healthy reference, `walther_clinical.py` (named eight times),
`idat_parse.py`, `stage_0_intake.py` and the exclusivity file written an hour earlier.

**Fixed by generating the list instead of maintaining one.** `MethylPhys_Interface/build_chain_inventory.py` enumerates the whole live tree,
classifies each file, and writes `Runtime Matrices/chain_inventory_v1.json`; a new **Files** tab renders it. Roles are measured, not asserted:
*in the chain* = resolved by `cpg_conductor._find()` or loaded by a module that is; *not in the chain* = present and callable but never reached from
`run_full()`. **152 files**, each with what it is, why it is there, its size and its SHA-256: 18 chain, 47 reference and calibration, 5 interface,
40 guards and doors, 28 record-side, 13 superseded. Anything without a description is emitted as **UNDESCRIBED and counted on the page**, so a gap
is visible rather than silent - the first run showed 81, which is why the descriptions were written from each file's own header rather than guessed.

**Two things the audit settled.** `run_full()` does **not** call `stage_8_matching` - confirmed by reading its body, so the removal holds - but the
conductor's own module docstring still lists the disease matrix as "(Stage C)", a stale line to fix. And the **Files tab is exempt from the
vocabulary guard**, deliberately and for the same reason the Healthy-reference tab is: an inventory that cannot name
`disease_cell_signature_matrix_v1_13.csv` is a false inventory. Every measurement tab stays guarded.

## The lineage grouping the author asked for already exists, and the chain does not read it

`Runtime Matrices/Collinearity_Groups/iamatlas_collinearity_groups_v0_1.json`, built **2026-06-26**: complete-linkage clustering at centred-cosine
0.95 in departure-from-consensus space, 111 cells to **94 groups, 10 of them multi-member**, with a `low_confidence` flag for groups spanning more
than one tissue super-family. Its own note states the point exactly: *cells within a group are methylation-collinear and not individually
identifiable by deconvolution.* The multi-member groups are the biologically sensible ones - the six gastric entries; CD4T with CD8T (three times
over, in different naming conventions); HSC/L-MPP/MPP; CMP/MEP; dendritic/macrophage; eosinophil/monocyte/neutrophil; CM/Hep/neuron.

**This is a different problem from the marker non-exclusivity measured earlier today, and both matter:**
- **Atlas collinearity** (this file): the reference genuinely cannot separate these entries. The honest unit of a per-cell claim is the *group*.
- **Marker non-exclusivity** (percell_exclusivity_v0): the *panel* is not discriminative even where the atlas profiles differ.
  Cortical_neurons and stem_pluri share 91 markers but sit in different collinearity groups - the atlas can tell them apart; their panels cannot.

So the answer to "which cell moved" is: **you can say which group moved, and within a group you cannot** - and separately, for 36 entries the panel
is too shared to carry an individual claim at all. Wiring the group column into the per-cell table is the next step, and it needs no re-seal.

## 2026-09-22 - the lineage group column, and the VAL findings schema written BEFORE the first run

**Group column wired (no re-seal needed).** Every per-cell row now prints its collinearity group. Measured on the
115 entries: **27 rows belong to a multi-member group** and say so ("not separable; read at group level"),
**84 are singletons** (separable in the atlas), and 4 are absent from the grouping. The tab now states the two
limits separately, because they are independent: the *atlas* cannot separate the members of a group, and *panels*
can be non-exclusive even where the atlas separates fine - Cortical_neurons and stem_pluri share 91 markers while
sitting in different groups. A per-cell claim needs both a single-member group (or a group-level claim) and an
exclusive enough panel. One bug caught by reading the output: the group values are dicts carrying
members/classes/singleton, so my first version printed the dict *keys* as the member list and flagged all 111
rows as group-level. Reading the rendered row is what caught it.

**VAL findings schema, written before the first run (`CPG_Engine/val_finding.py`, schema `val_finding_v1`).**
The author asked for this now: capture everything a researcher will want, and exactly what a future disease
matrix would need - which cells, which direction, what magnitude, per condition. A record designed after the runs
is shaped by whatever was convenient to save, so this one is designed first.

| block | holds | why |
|---|---|---|
| instrument | sha256 of all **14** reference layers plus the repo commit | a reading is only meaningful against a stated instrument; a re-seal changes these hashes, so two findings are comparable only if the fingerprints match - checkable rather than assumed |
| samples | one row per array: arm, age, class readings, departure, sky, refusals | the unit is a per-sample absolute reading; storing samples means an arm difference can never quietly become the result |
| cells / groups | per entry and per group: placement, median by arm, **direction**, **magnitude in units of that entry's healthy spread**, **prevalence**, exclusivity, claim level | the answer to "which cells are doing what, how much, how often". Magnitude in healthy spreads so a loose entry and a tight one compare; prevalence separates "most moved a little" from "a few moved a lot" |
| bars | each pre-registered bar with threshold, measured value, PASS / FAIL AS SEALED | scored against what was written before the run |
| not_assessable | what could not be read, and why | a class below its presence floor is absent, not normal |
| matrix_evidence | per-group direction/magnitude/prevalence for this condition, tagged with the instrument fingerprint | the disease-matrix precursor. **Accumulating evidence, not a matching rule** - the chain never reads it back to classify |

Two rules are in the writer rather than left to the operator: a finding cannot be written without its instrument
fingerprint, and the claim level per entry comes from the reference (individual / group_only /
withheld_panel_shared), not from the operator's choice.

**Exercised end to end** on the eleven commissioning arrays (`VAL-DRYRUN_finding.json`): 11 samples, 115 entries,
94 groups, 14 layers fingerprinted, direction and magnitude computed per entry per arm. **A flaw in my own filter
caught on the dry run:** the first version carried all 94 groups as matrix evidence for a run with no condition
at all, because it tested for a non-zero magnitude and every group has one. Now two conditions are required - the
finding must name a condition, and a group must show an actual departure - so a healthy or technical run
contributes nothing, and the record says "nothing - this finding names no condition".

A new **Findings** tab reads these records and explains each block; it is exempt from the vocabulary guard for the
same reason the Files tab is - a finding must be able to name the condition it measured.

## 2026-09-22 - the June cards and the straw-man wall read, and the findings schema extended to v1.1

The author asked that the old disease matrix, the cards and the residual maps be read before the findings schema
is settled, so we look for the right things. Read in full: `IAM_Disease_Wall_CROWN_JEWEL_v1_12.html`,
`ad-immune_card_v3_1.json`, `breast-epic_card_v3_1.json`, `immune-atlas_card_v2_0.json`, and the residual /
bimodality / PCA map column structures.

**Nothing numeric was imported.** Their effect sizes are case-versus-control Cohen's d on the pre-atlas surface -
the statistic this chain does not use. What was taken is the *vocabulary of observations*: the kinds of thing
worth recording. Eight fields were added because of what those files record:

| field added | why - what the old set recorded |
|---|---|
| `window` | the wall's rows are disease x PHASE x substrate (">10 yr pre-dx", "5-10 yr", "0-2 yr", at dx) and its headline is a trajectory across them. Without a window field, findings at different distances from diagnosis pool into one number and the trajectory cannot be reconstructed |
| class `direction` | **the sharpest thing in the whole set.** The AD card's specificity arm: on the SAME Mahalanobis metric, Alzheimer's departed outward, PSP/CBD departed inward ("BELOW_NORMAL architectural compaction direction"), FTD sat between. *Direction, not magnitude, separated three conditions* |
| `compartments` + opposition | the wall splits immune into lymphoid and myeloid because they "move in opposite directions near diagnosis". A pooled immune number averages that away |
| `per_cpg` residual map | every card carried one (cpg, d per cohort, concordant_strong, mean_abs_d, CHR, MAPINFO). Ours is the same object on an absolute footing: per-CpG mean residual z against the healthy reference per arm |
| `per_cpg` bimodality | their breast map decomposed each CpG into bc_hc, bc_case, delta_bc, mean/sd beta, delta_var, loss_of_bimodality. A locus can hold its mean while splitting into two populations - a different event from a mean shift |
| `coverage` | the cards refused to match below 80 % coverage of their residual map (INSUFFICIENT_COVERAGE) |
| `covariates` | the immune-atlas card carries age, smoking and sex foreground layers - and discloses that smoking subtraction was NOT applied at beta level. Smoking is a large blood methylation effect; a finding that does not say whether it was handled invites a confounded comparison |
| `specificity_arm`, `cell_of_origin`, `conjunction_rule`, `honest_limitations` | the cards' own devices: the contrast conditions run on the same instrument, the declared cell of origin (the wall's gold ring), "card never fires on one tile", and a limitations list at the end |

**Three defects in my own code, caught on the dry runs rather than reasoned about:**
1. The sky residual arrives as a **pandas Series indexed by CpG id**, not a bare array - my `isinstance(list, ndarray)` check silently skipped it, so the whole per-CpG layer was empty while reporting no error. Now aligned on the CpG index so samples with different coverage still stack.
2. The **opposition flag fired on healthy noise**: the first run reported opposite lymphoid/myeloid signs in *both* arms with the sides swapped - which is what sampling scatter looks like. It now carries the arm size, the member-magnitude IQR, and a `separation_clears_member_scatter` test; on these healthy arms it correctly reports **false** in both.
3. Ranked by |mean z| alone the top CpGs were **mean 30.8 with sd 42.6** - high-variance probes, not consistent departures. A second ranking by |mean z| / sd was added; that is the list to carry into a candidate panel, and with two cohorts the intersection of the two consistency lists is this chain's equivalent of the cards' `concordant_strong`.

## 2026-09-22 - the report is not a change log. Author's correction, and a guard so it cannot recur.

**What he objected to, correctly:** "why are you adding stuff like this on the freaking interface we are presenting
to researchers??? I dont need a log of all the changes. We are handing them a finished product not a log of my
mistakes or changes."

He is right, and the reasoning I had used for it was already void: I marked three passages of the imported story
text `[updated]` on the argument that the source document is public and a reader might hold both - and he had
already told me that paper is **not** public. Even if it were, a researcher handed an instrument wants the
instrument, not its construction history.

**Scanned all 17 tabs** for the same voice rather than only fixing the three he quoted: nine patterns
(`[updated]` markers, "my first/earlier version", "the author corrected", "no longer true", dated change
narrative, "found independently", "withdrawn", "stale-data failure", first-person error references). **13 hits
across 5 tabs.** Three were real:

| where | was | now |
|---|---|---|
| Story | a block headed "Three places where this chain's own measurements supersede the May 2026 text", with three `[updated]` passages, one of them dated and attributed | two plain statements of what is true: where A = 1.00, H_min and the ceiling sit; and that the floors, the calibration code and the bootstrap are public and linked |
| Files | the build log described as "every defect found while building, including the ones in my own output" | "Engineering log for the report and interface: what was measured while building it, what failed, and what each failure changed. Read it for the construction history; nothing in it is needed to read a result." |
| Physics | "A report generated in June 2026 printed ... ; the same error, found independently, was the defect in the retired sky formula" | the same point as a principle: using the posterior SD of a mean as a normal range is a category error, and here is what it costs. No incident, no date |

The remaining hits were false positives and stay: "the **corrected** value A''" is the name of the age- and
laboratory-corrected reading; the detection rule is a policy of the instrument; and one file's description quotes
its own title.

**The guard.** A rule in a document does not survive; the fix is in the builder. `no_changelog()` runs on **every
tab, including the two exempt from the vocabulary guard** - those exemptions are for naming conditions and files,
not for telling a reader about earlier drafts - and refuses to write the report if that voice appears. Tested both
ways before pushing: it fires on `[updated]`, on "the author corrected this himself" and on "my first version was
wrong", and does **not** fire on "The corrected value A'' is placed in the band" or on the flatness lesson's own
title. One boundary bug was caught in that test: the first pattern matched "he corrected" inside "**the** corrected
value" and blocked the Reading tab; fixed with a lookbehind.

The division of labour is now explicit and enforced: **this log carries the construction history; the report
states what is true now.**

## 2026-09-22 - the marker repair, measured. A trial panel, and a finding about the atlas rather than the criterion.

The plan was: replace the one-vs-rest mean criterion with a nearest-rival margin, cap how many panels a marker may
serve, re-select, re-seal, re-measure. The selection is built and measured. **The measurement found something
larger than the criterion.**

**The atlas is sparse per address, and wildly uneven per cell type.** No address carries all 115 cell-type means -
the median address has **25 of 115** measured (range 6-84). And the pool per cell type runs from **252 addresses**
(megakaryocyte, eosinophil, monocyte, neutrophil, erythroblast, small_intestine and three others) to **482,421**
(HSC, GMP, L-MPP, stem_pluri). A cell type with 252 known addresses cannot have a 100-marker exclusive panel: the
panel would be 40 % of everything the atlas knows about it. **That, and not only the selection rule, is why the
old panels overlapped.**

**The nearest-rival margins are small.** Computed against the rivals measured at each address: median **0.0016**,
90th percentile 0.0126, 99th 0.1006. At a 5-point beta margin only **74 of 115** cell types have 100 candidate
addresses; at 0.10, 53; at 0.15, 35. There is no threshold at which all 115 entries get a clean panel.

**The sweep, on 40 healthy Uppsala arrays (real betas, Stage 1, 125,323 loci):**

| panel | entries scored | median size | exclusivity | duplicate-lineage disagreement | healthy spread | spread > band |
|---|---|---|---|---|---|---|
| v0_2 (current, sealed) | 115 | 100 | **0.37** | **0.3548** | 0.0657 | 20/115 |
| nearest-rival, cap 2, N<=100 | 102 | 100 | 0.94 | 0.1765 | 0.0807 | 37/102 |
| **nearest-rival, cap 2, N<=300** | **106** | **159** | **0.83** | **0.1204** | **0.0741** | **32/106** |
| nearest-rival, cap 2, N<=1000 | 106 | 159 | 0.72 | 0.1437 | 0.0734 | 31/106 |

**The duplicate-lineage test is the one that matters** - it is the author's question, "which cell moved", made
measurable: two atlas entries naming the same lineage should read the same on the same array. Median disagreement
falls from **0.355 to 0.120**, and the individual pairs are stark: Mono vs CD14_monocytes 0.452 -> 0.074,
eosinophil vs Eosinophils_reinius 0.580 -> 0.158, neutrophil vs Neutrophils_reinius 0.604 -> 0.292, CD8T vs
CD8_T-cells 0.224 -> 0.098. One pair got worse (Bcell vs CD19_B-cells, 0.086 -> 0.167).

**The honest cost:** per-entry healthy spread widens from 0.066 to 0.074, and entries whose spread exceeds the
class NORMAL band go from 20 to 32. Part of v0_2's apparent tightness was an artefact of sharing - correlated
panels produce correlated readings, which look tighter without being more about the cell. N<=300 recovers most of
the loss; N<=1000 buys nothing and costs exclusivity.

**Chosen parameters: margin >= 0.05 against the nearest measured rival, at most 2 panels per marker, up to 300
markers, 20 minimum to be individually resolvable.** Result: 93 of 115 entries individually resolvable, median
panel 159, median exclusivity 0.83, no marker serving more than 2 panels by construction.

**22 entries are not individually resolvable at this margin** - Kera_undiff and Neutro get zero markers;
Epi_basal, Leu, Eosino, Epi_suprabasal, BE, erythroblast, tcell under six. Only 6 of the 22 are in a multi-member
collinearity group, so this is a **third, independent** limit on a per-cell claim: the June grouping says which
entries the atlas cannot separate *in profile*; exclusivity says which panels are shared; and this says which
entries the atlas has too few addresses to characterise at all.

**Written as `iamatlas_celltype_markers_v0_3_TRIAL.json`, NOT adopted.** The sealed anchors reproduce on v0_2, and
only 2,572 of the 11,369 old marker slots survive the new criterion, so the anchors will not reproduce on v0_3.
Adoption needs, in order: (1) recompute the two foundation-cohort anchors from raw GEO on v0_3 and seal them as a
new anchor set, keeping the v0_2 anchors as the historical seal; (2) re-measure the per-entry healthy references
on v0_3 from the four laboratory panels; (3) re-run the kit guards; (4) only then switch the report to v0_3, with
the 22 unresolvable entries printing their number and withholding the individual claim.
