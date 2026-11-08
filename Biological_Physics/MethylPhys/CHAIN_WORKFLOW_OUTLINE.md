# The Chain Workflow: outline

<!-- Rule (author 2026-11-06): every physics value and every important number in this document comes only from CANON/iam_canon.json, written as a {{canon:KEY}} or {{canon:KEY:FORMAT}} token and filled by CANON/canon_fill.py when the document is built; none is typed by hand. -->

This is the outline for the chain workflow document: plan item 60.16, Phase H, the last step of the list. The document will be a visual,
step-by-step flowchart of the whole chain, laid out like an emergency action plan. Each box asks what the chain checks first, and each
answer says where the sample goes next ("if this, go here"). It starts with the kind of laboratory and sample and runs through intake, the
cell mix, the right atlas, every instrument, the globes and the sphere, to the report. It contains no code. A researcher can use it to
process and score their own samples, and to build and test their own chain.

Every box below points to the Decoder section it will be written from (`manual/decoder_text.md`), so every statement in the finished
document traces to a committed record. The document is written once the chain, the laboratory work and the commissioning are settled.
Until then this outline grows with the chain.

**Every box will carry the same five things:**
1. the question it asks;
2. the answers and where each one goes next;
3. what it measures, and the physics behind it in words;
4. every gate, with its limit and where the limit came from;
5. every refusal, with the reason the chain prints.

**The physics root.** Every instrument rests on one fact: a cell pays to keep each methyl mark against the kicks of heat. A mark kept by
that cost is held; one that has slipped is disorder. Every reading compares a sample's disorder with that of healthy cells of the same kind,
measured the same way. The opening page states this once, from `CANON/physics_root.md`. Every box then names which part of it that box uses.

## Part A. Before the chain: what do you have?

**A1. How was the DNA measured?**
- **An array** (an IDAT pair, or a calibrated table of values): go to Part B.
- **Sequencing** (molecules in a `.pat` file, an aligned BAM, or a per-site count table such as Bismark coverage, a bigWig level and
  coverage pair, or a `.beta` file): go to Part C.
- **Neither:** stop. The chain has no reader for it.
- Decoder: §1, §10, §10b, §21.

**A2. What is the specimen?**
- whole blood or buffy coat;
- a sorted, isolated or purified cell that names an atlas cell;
- plasma (cell-free DNA);
- tissue;
- a constructed mixture (test material only);
- **anything else is refused before a byte is read**, with the reason.
- Decoder: §2, `specimen_refusal`, `is_plasma`, `is_tissue`, `atlas_cell_of`.

**A3. Is it the whole sample?** A sequencing run is read in full depth, every read pair of every run. A run stored in genome order and
read from its first read pairs holds only the first chromosomes, so the chain never reads "the first N pairs" of a run. A partial
sample is refused at intake by the genome-coverage check (box C3).
- Decoder: §21, `stage_seq_values.genome_coverage`.

**A4. Where does healthy come from?** Only from the atlases: IAM-Atlas v2.1 for arrays, the IAM Sequencing Atlas for sequenced
samples. Never from other samples in the same run, laboratory or study, and never from a fixed band. Each array's own line and noise is
measured on the array itself and applied to the atlas (each reading's healthy prediction), and each cell's healthy width comes from the
atlas's own donors.
- Decoder: §16 (against the atlas), `stage_metaa_atlas.predicted_healthy`, `stage_metaa_atlas.cell_prediction`.
- The reference door checks both atlases against their checksum manifests before any reading (`reference_door.py`).

## Part B. The array path

1. **B1. Arrival, manifest and integrity** (Stage 0.1 to 0.3). Each step can quarantine, and a quarantine stops the run with exit code 2.
   Decoder: §2.
2. **B2. Which platform?** The platform is read from the IDAT header's address count: under 800,000 is 450K, under 1,080,000 is EPIC v1,
   otherwise EPIC v2.
   - **EPIC v1:** the commissioned route.
   - **450K:** read and flagged. The atlas readers read it, and the commissioned readers whose EPIC references a 450K chip only half
     covers refuse it themselves.
   - **EPIC v2:** refused on the main route before the quality decode, with the reason. A development route exists (`--dev-epic-v2`).
   - Decoder: §2, `_infer_platform`; §1 step 3.
3. **B3. The quality gates** (Stage 0.4 to 0.9): control probes, detection p-values, bead counts, call rate, sex and platform coverage,
   then the decision gate. A quarantine prints five parts: what was measured, the limit, where the limit comes from, what it means and
   what to do. Decoder: §2, §3.
4. **B4. Calibration** (Stage 1). The IDAT pair becomes one value per site. Stage 1's own quality numbers are recorded and never gate.
   Decoder: §4.
5. **B5. Coverage of the identity sites.** Met-A withholds its reading when fewer than 90 % of its sites were measured. Decoder: §1 step 7,
   §5.
6. Then go to Part D.

## Part C. The sequencing path

1. **C1. Intake of the file** (Stage Q0): the checks in its table, stopping at the first refusal. Decoder: §10.
2. **C2. Which genome?** Each reading counts on its own genome build, and the chain keeps the builds apart. A table on GRCh38 is lifted to
   the atlas's numbering. Decoder: §10b; §21 (`--build seq-table`).
3. **C3. Is the whole genome covered?** For each autosome, the chain takes the share of sites with a value and compares it with the median
   autosome's share. An autosome under half the median refuses the sample, and the refusal names the missing chromosomes. Decoder: §21,
   `genome_coverage` (DEV-SEQ-INTAKE-01).
4. **C4. The library's floor**: false "methylated" calls at the sites no healthy cell marks, measured for every library. Decoder: §21,
   `library_floor`.
5. **C5. Molecules or levels?** A file that keeps molecules (`.pat`, BAM) can be read by IAM-A and Coupling-A. A file that keeps only
   levels (a count table, a bigWig) cannot, and the report says so. Decoder: §9, §15.
6. Then go to Part D.

## Part D. Choosing the ruler

1. **D1. Which atlas?**
   - An array is read against IAM-Atlas v2.1.
   - A sequenced sample is read against the IAM Sequencing Atlas.
   - Each atlas is built from healthy cells measured with its own platform's chemistry.
   - Decoder: §8, §16, §21.
2. **D2. A single cell type or a mix?** The chain does not take it from the label. The label picks only the route (array or sequencing;
   blood, plasma, tissue or isolated cells); it never picks the cell. Every sample, a sorted one included, goes to the cell mix (Part E),
   and the IAM-Deconvolver says what is in it. A sample whose label disagrees with what the Deconvolver finds is read as found, and the
   report records the disagreement.
   - Decoder: §1 (`run_sample.py::main`), `stage_0_intake.LABEL_RULE`, `stage_0_intake.label_check`, `stage_0_intake.is_isolated`; the gate refuses a cell chosen from a
     label (rule 23b-h).

## Part E. The cell mix: the IAM-Deconvolver

1. **E1. What cells are in the sample, and in what shares?** The chain fits the sample's cell shares against the atlas's healthy cells.
   It refuses a mix it cannot trust instead of reading a change in the mix as biology. Decoder: §7.
2. **E2. Cells the atlas does not hold:** the reading names the cell and reads the rest. Megakaryocytes, a large source of plasma DNA,
   have a provisional column in the IAM Sequencing Atlas (from single megakaryocytes of two healthy donors); IAM-Atlas v2.1 holds none.
   Decoder: §20, `read_in_mix`.
3. **E3. Each cell found is read against its own healthy profile.** The results come per cell found (`cells_found`, first in the bundle
   and the report): each cell's share, its readings against its own column of the atlas, each divided by its own healthy prediction, and
   how much of the reading is the cell's own. In a mix, a cell's reading moves with a change in that cell and with a change in the cells
   around it, in proportion to the DNA each holds at the cell's identity sites. Above 90 % of the sample the row reads the cell itself;
   below it, the sample at that cell's sites. A change is given to one cell only in an isolated sample, molecule by molecule on
   sequencing, or in one person read against their own earlier draws with the shares measured each time.
   Decoder: `stage_0_intake.cells_found`, `stage_metaa_atlas.cells_found`, `stage_metaa_atlas.cell_prediction`.

## Part F. The instruments

Each instrument box carries five things:
1. the question it answers;
2. what it reads;
3. its physics in words, then its formula;
4. its gauge: the floor, the healthy middle and the ceiling;
5. where it is commissioned, with each use's record; everything else is read and flagged.

1. **F1. Met-A:** how well the cell's identity sites are held. Decoder: §5 (Stage M), §16 (against the atlas).
2. **F2. The C-Score:** do the changes cluster along the genome, or are they scattered? Decoder: §5 (Stage MC).
3. **F3. IAM-A:** the held error against the loose share, from molecules, against the sample's own reference. Decoder: §9.
4. **F4. The territories** (islands, domains, the rest). Decoder: §15, §16.
5. **F5. Coupling-A:** is a mark kept site by site, or by a stretch of DNA as a unit? Decoder: §15 (`dev_stages`).
6. **F6. Group-A:** has a whole group moved? Decoder: §11.
7. **F7. The tumor ceiling gauge:** how far a sample is from fully scrambled. Decoder: §16, §21.
8. **The tares: retired.** The same-run tare (arrays from the same slide) and the same-laboratory tare (sequencing) took healthy from
   other samples. They stay in the code only so the first commissionings' records still rebuild (`--ref-table`); no new reading uses them,
   and the gate refuses a reference built from other samples. Met-A, the C-Score and Group-A were first commissioned with them and are
   being re-commissioned against the atlas alone. Decoder: §5 (Stage T, history).

## Part G. The maps: globes and spheres

1. **G1.** The report's globe: the sample laid over its healthy reference, with everything that matches quieted. Decoder: §14b.
2. **G2.** The whole-genome globe. Decoder: §17.
3. **G3.** The information sphere: every site placed by how it behaves across the cell types. Decoder: §18.
4. **G4.** The reference globes of both atlases. Decoder: §19, §21.

These are pictures, not readings. Every number stays with its instrument.

## Part H. More than one draw

1. **H1.** Two draws of one person (Stage 12b), refused unless both draws carry the same identifier hash, platform and pipeline.
   Decoder: §13.
2. **H2.** One blood draw read as cells and plasma. Decoder: §20.

## Part I. The report

1. **I1.** One page with a researcher's view and a plain view, and two PDFs. The identifier is replaced by its hash, and one row is added
   to the evidence ledger. Decoder: §14.
2. **I2.** Exit codes:
   - 0: a report was written, which may still withhold a reading and says why;
   - 2: the sample was quarantined at intake;
   - 3: the machine is missing a file it needs.
   - Decoder: §1.

## Decoder text brought up to date while writing this outline

These were found while writing this outline and corrected in the Decoder from the code. The `--array-type` help text in `run_sample.py` is changed after the B and T and Johns Hopkins readings finish, so those readings stay on one chain version.

1. **§1, "The order it runs", step 2** says the chain reads only whole blood, buffy coat and isolated neutrophils. The intake code
   (`stage_0_intake.specimen_refusal`) also accepts sorted, isolated and purified cells of any IAM-Atlas v2.1 type, plasma, tissue and
   constructed mixtures.
2. **§1, step 7** says EPIC is the only platform the chain reads. A 450K array now advances flagged (`ADVANCE_FLAGGED_450K`): the atlas
   readers read it, and the commissioned readers refuse it themselves. Step 3 is correct: EPIC v2 is refused on the main route, and its
   development route (`--dev-epic-v2`) should be named there. The `--array-type` help text ("reads EPIC_v1 only") should say the same.
3. **§1** lists only `.pat` and BAM for sequencing. The chain also reads per-site tables (`--build seq-table`), bigWig tracks and `.beta`
   files, and it now checks genome coverage (box C3).
4. **§2, the specimen check** listed only blood, neutrophils, buffy coat and mixtures; it now names isolated atlas cells, plasma and tissue, as the code accepts.
