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
