# The Chain Decoder

## Why the chain works: IAM's physics

Every reading this chain makes rests on one physical fact. A methyl mark on DNA is a stored record, and keeping a record against heat
costs energy. Landauer's principle sets the least energy that can hold one bit against thermal noise. IAM's Law is that this price
governs every physical record: a qubit, a transistor, the large-scale structure of the universe, and a methyl mark. Two consequences
make the instruments possible.

1. **The scale is derived, not learned.** The energy a cell spends holding a mark sets the least error it can reach: the floor. A site
   whose mark is lost entirely reads as a coin flip, one bit of disorder: the ceiling. Counting single DNA molecules in 56 healthy human
   cell types, the energy that holds a methyl mark is 3.08 k_B T, about 0.147 of the ATP the cell pays to keep each site (M = 20.94 k_B T). The restore odds behind that share follow the same rule as a slowly ramped write; restored suddenly, a mark would be a coin flip.
   A healthy cell sits at a measured place between the floor and the ceiling, so a reading in bits means the same thing in any
   laboratory, for any cell type, before anything has been trained. A candidate derives that place from M and the price of the methyl
   donor (one SAM costs about 2.0 ATP): restoring paid against thermal kicks gives a held error of
   0.0445 and 3.07 k_B T, against the measured 0.0453 (candidate, not yet tested).
2. **Disorder is counted, and counting has exact statistics.** A sample's disorder at a site is the binary entropy of its measured level,
   and the disorder that heat and counting alone would give at that site's number of reads follows exactly. So every instrument sets a
   measured disorder beside an expectation derived from the physics and from the sample's own reads. Nothing is fitted to the sample.

**How each part of the chain uses it.**
- **Met-A** reads a sample's disorder at its cells' identity sites against the disorder healthy cells of the same kind carry there, on
  the floor-to-ceiling scale; 1 is healthy.
- **IAM-A** reads the held error on single molecules against what the cell's own loosely held share predicts, on a line whose form comes
  from the holding physics. The sample is its own reference.
- **The C-Score** asks whether departures from the expected disorder are scattered along the genome, as ordinary wear is, or clustered,
  as when a whole region loses its holding.
- **The territories** make the same comparison at the islands every healthy cell holds unmethylated and the domains some cell types hold
  loosely.
- **Coupling-A** asks whether neighboring marks on one molecule were lost together, against its counted floor and ceiling.
- **The tumor gauge** reads the distance from the measured disorder to the entropy ceiling.
- **Group-A** asks whether a whole group has moved against the healthy samples of its own series.
- **The atlases and the IAM-Deconvolver are the rulers.** They say which healthy cells a sample is made of and where healthy disorder
  sits for each, so the physics can be applied to each person's own mix of cells.
- **The library floor.** Every laboratory's chemistry adds a small rate of false "methylated" calls. The chain measures it at the sites
  no healthy cell marks, and reads it as a property of the library, not of the cell.

## The chain in numbers

Read from the code at build time: the one command can load **38 Python files** holding **611 functions** (nested ones included); 611 of them have their entry in this Decoder, and a push is refused when one does not. Called: how many of a file's functions each traced run used (the array run on a healthy whole blood; the sequencing run on the release check's planted BAM).

| file | lines | functions | called, array run | called, sequencing run |
|---|---|---|---|---|
| `MethylPhys_Interface/run_sample.py` | 1,789 | 21 | 11 | 10 |
| `stage_0_intake.py` | 1,281 | 48 | 33 | 1 |
| `stage_0_1_qc_handoff.py` | 194 | 7 | 7 | 0 |
| `stage_1_idat_calibration.py` | 232 | 6 | 6 | 0 |
| `conductor_v3.py` | 321 | 20 | 18 | 0 |
| `stage_m_met_a.py` | 102 | 7 | 3 | 0 |
| `MethylPhys_Interface/report_v4.py` | 574 | 40 | 17 | 17 |
| `MethylPhys_Interface/report_v3.py` | 297 | 9 | 9 | 9 |
| `MethylPhys_Interface/report_figs_v4.py` | 157 | 8 | 6 | 2 |
| `dev_stages.py` | 400 | 30 | 1 | 0 |
| `stage_v4.py` | 192 | 11 | 4 | 2 |
| `stage_deconv_v21.py` | 221 | 7 | 4 | 0 |
| `stage_q2_selfref.py` | 613 | 49 | 0 | 11 |
| `stage_atlas_commission.py` | 62 | 8 | 1 | 0 |
| `MethylPhys_Interface/report_pdf_v4.py` | 123 | 8 | 6 | 6 |
| `MethylPhys_Interface/info_sphere_viewer.py` | 194 | 5 | 0 | 0 |
| `stage_t_selftare.py` | 90 | 5 | 0 | 0 |
| `stage_seq_values.py` | 256 | 14 | 0 | 0 |
| `dev_comp_methods.py` | 216 | 17 | 0 | 0 |
| `deconv_v2.py` | 74 | 3 | 0 | 0 |
| `sky_statistics.py` | 108 | 6 | 0 | 0 |
| `stage_q0_intake.py` | 189 | 8 | 0 | 0 |
| `stage_q_iam_a.py` | 174 | 9 | 0 | 0 |
| `stage_coupling.py` | 30 | 2 | 0 | 0 |
| `stage_provisional.py` | 98 | 7 | 0 | 0 |
| `serial_mode.py` | 67 | 6 | 0 | 0 |
| `stage_groupa.py` | 215 | 9 | 0 | 0 |
| `MethylPhys_Interface/report_globe_v4.py` | 489 | 25 | 0 | 0 |
| `MethylPhys_Interface/globe_viewer.py` | 145 | 7 | 0 | 0 |
| `stage_metaa_atlas.py` | 1,414 | 70 | 0 | 0 |
| `reference_door.py` | 81 | 5 | 0 | 0 |
| `stage_q2_globe.py` | 159 | 15 | 0 | 0 |
| `stage_info_sphere.py` | 216 | 16 | 0 | 0 |
| `stage_law_sphere.py` | 373 | 24 | 0 | 0 |
| `MethylPhys_Interface/reference_globes.py` | 279 | 15 | 0 | 0 |
| `MethylPhys_Interface/run_series.py` | 199 | 7 | 0 | 0 |
| `MethylPhys_Interface/run_draw.py` | 91 | 5 | 0 | 0 |
| `stage_seq_atlas.py` | 998 | 52 | 0 | 0 |

## The one command's options

Read from `run_sample.py`'s own option definitions at build time: **136 options**. Without `--commissioned-only` the full development chain runs on every sample (each development reading flagged); the last column says whether that default turns the option on. A push is refused when an option has no description.

| option | default | on in the default chain | what it does |
|---|---|---|---|
| `--age` | none |  | declared age in years (optional; recorded when given, read by no stage). Float: records carry 72.0 and 54.5 as readily as 72 |
| `--alignment-qc` | None |  | Stage Q0: JSON from the alignment step with conversion_rate and/or duplicate_fraction |
| `--array-type` | None |  | declared array type; omit to take it from the IDAT header (Stage 0.1). The chain reads EPIC_v1 only |
| `--assembly` | 'hg19' |  | assembly of --bam (mm10: --dev-count-only only, 2026-11-14) |
| `--atlas-v2` | os.environ.get('CPG_ATLAS_V2_PARQUET') |  | atlas v2 parquet for the atlas-based development flags (not stored in the repository) |
| `--bam` | None |  | chain v4 Stage Q2: an aligned whole-genome bisulfite BAM (bwa-meth or Bismark), counted by the chain (needs --ref-fasta) |
| `--banner` | None |  | a statement printed at the top of the report and both PDFs, e.g. that a known change was planted into a real array for an example (2026-10-18) |
| `--betas` | none |  | a per-site table of values (CSV: site id, value) read as an array; the Deconvolver and every reading run on it |
| `--build` | None |  | build a reference or a reference globe instead of reading a sample (2026-10-22: the chain builds its own references and draws every globe): info-sphere = the information sphere's layout from --build-input (development, DEV-SPHERE-01); explorer-purified / explorer-noise = the Globe Explorer's reference globes (--build-input: the noise maps folder); atlas-sphere / atlas-masks = the atlas globe's textures and slider pictures (reference_globes.py) |
| `--build-about` | None |  | with --build info-globe: the explanation beside the picture (default: the one in run_sample.py) |
| `--build-block` | None |  | --build seq-atlas (2026-10-29, development): the block of hg19 CpGs, 0-based, end excluded |
| `--build-cell` | None |  | --build provisional: the cell's name |
| `--build-construction` | None |  | --build info-sphere: v1 (default) or v5, the level taken out (DEV-SPHERE-01, development) |
| `--build-depth` | None |  | --build constructed-draw: reads per site in each half (default 30,15) |
| `--build-groups` | None |  | with --build info-sphere: JSON {name: [site ids]} of site groups to test for landing together ('pool': the sites random groups are drawn from) |
| `--build-input` | None |  | with --build info-sphere: a matrix of methylation levels, sites x cell types (.parquet or .csv, site id first) |
| `--build-mix` | None |  | --build constructed-draw: atlas tissues added to the plasma half and their shares (tests only) |
| `--build-name` | None |  | the atlas's name on a picture built from --build-input (info-globe) |
| `--build-out` | None |  | with --build: where the built reference is written (.npz) |
| `--build-random` | 500 |  | with --build info-sphere: random groups per named group for its line (default 500) |
| `--build-samples` | None |  | --build provisional: the samples CSV (gsm, file) of per-CpG read files to pool |
| `--build-with-mk` | off |  | DEVELOPMENT: --build seq-deconv-ref keeps the provisional megakaryocyte column |
| `--bundle` | none |  | also write the full bundle as JSON - every stage's output, for a test harness or an integration that needs more than the report |
| `--cell-panel` | None | yes | the cells the IAM-Deconvolver splits the sample into: 'atlas' (all of IAM-Atlas v2.1's cell types; the default chain's choice, with the six-cell commissioned reading written beside it as cell_shares_blood6), 'blood' (the six blood cells it is commissioned on; the choice with --commissioned-only), or as before 'atlas' (all of IAM-Atlas v2.1's cell types, development: every share flagged), or a comma-separated list of atlas cell names (development); Met-A for each cell reads every cell the panel finds (2026-10-22) |
| `--commissioned-only` | off |  | read only the commissioned instruments; without it every built development reading runs, each flagged (2026-11-08, author: nothing built sits switched off) |
| `--covariate` | [] |  | a phenotype or covariate to record with this run, repeatable - e.g. --covariate diagnosis=case --covariate stage=II --covariate cohort=EPIC_Italy. Kept in the custody record and the bundle; never printed in report prose |
| `--covariates` | none |  | a JSON object of covariates, merged with any --covariate flags |
| `--dev-allow-partial` | off |  | development only: read IAM-A on a --pat-max-bytes cut (e.g. to reproduce position v1) |
| `--dev-allow-partial-genome` | off |  | development: skip the genome-coverage intake check (a synthetic that reads part of a genome on purpose); every reading is flagged |
| `--dev-count-genome` | 'hg19' |  | --dev-count-only: the genome the .pat was made on (recorded) |
| `--dev-count-only` | off |  | DEVELOPMENT (to-do 14): with --pat, record Stage Q2's count (held error and opportunities) even where Stage Q0 refuses an IAM-A reading, flagged; for tests on samples without a healthy reference (cell lines, mouse) |
| `--dev-count-sites` | None |  | a comma list counts each set from the same file (2026-11-16, e.g. islands,epitoc2: the clock and its islands meter in one reading); DEVELOPMENT (DEV-METAA-HEALTHY-01 step 3 version 2): with --dev-count-only, also count the held error only at Met-A's commissioned identity sites of CELL (e.g. neutrophils), by Stage Q2's own rules (stage_q2_selfref.count_pat_sites); CELL = islands (2026-11-15, register T1): the islands territory, both the loss and the stray-mark side; CELL = epitoc2 (register C3): the mitotic clock's 163 CpGs and 25 CpGs either side |
| `--dev-count-tiles` | None |  | DEVELOPMENT (DEV-ERASE-01 graded maintenance test, 2026-10-25): with --dev-count-only, also write the held error tile by tile (2,000 consecutive CpGs per tile, two independent halves of the molecules) |
| `--dev-curve-cal` | off |  | DEVELOPMENT (2026-11-08, DEV-METAA-HEALTHY-01): arrays only; each probe design's response curve measured on this array's own calibrators (sites where every cell found holds the same atlas level, never islands or domains) and mapped onto IAM-Atlas v2.1 before the cell shares (stage_t_selftare.curve_cal) |
| `--dev-deconv-drop` | None |  | DEVELOPMENT, diagnosis only: hold these cells out of the Deconvolver's sequencing reference (the IAM Sequencing Atlas), so their share goes to the cells that remain; flagged in the bundle |
| `--dev-deconv-ref` | None | yes | DEVELOPMENT, diagnosis only: read a sequenced sample's cell shares against this Deconvolver reference instead of the committed one (e.g. one built with --build-with-mk); flagged in the bundle |
| `--dev-floor-class` | off |  | DEVELOPMENT (2026-11-07): the islands read with the sample's and the reference's floors each counted at the islands read, where the sample's own mix expects them unmarked, both at the higher (stage_metaa_atlas.FLOOR_CLASS) |
| `--dev-floor-equal` | off |  | DEVELOPMENT: read a sequenced sample and its reference at a common library floor (the higher of the two) |
| `--dev-genome-globe` | off | yes | DEVELOPMENT - not commissioned (DEV-IAMA-GLOBE-01): with --pat, the whole-genome globe (methylation level, held error and the change against the sample's own prediction; with --prior-pat, against an earlier reading too) |
| `--dev-globe-v21` | off |  | DEVELOPMENT (plan 35.30): draw a sequenced sample's globe and information sphere against IAM-Atlas v2.1 instead of the IAM Sequencing Atlas (for comparison) |
| `--dev-iama-position` | off |  | development: read --pat with the retired position reading (Stage Q) instead of the commissioned self-reference (Stage Q2) |
| `--dev-info-sphere` | off | yes | DEVELOPMENT - not commissioned (DEV-SPHERE-01): the sample's change on the information sphere (page OUT_info_sphere.html; needs the layout from --build info-globe). On by default since 2026-10-23 (author: 'it still gets created by the chain and we learn each time it does') |
| `--dev-input-floors` | None |  | DEVELOPMENT: an atlas input floors file in the committed format (gsm, cell, half, floor, calls), for synthetics |
| `--dev-law-sphere` | off | yes | DEVELOPMENT (DEV-SPHERE-01, edit 205): with --pat, IAM's Law Sphere: each sequencing-sphere unit from the sample's molecules against the law's expected point (14 layers); on in the full chain |
| `--dev-line-censored` | off | yes | DEVELOPMENT (2026-11-07, DEV-METAA-HEALTHY-01): the array's line at the anchor sites measured with readings at exactly 0 or 1 counted as censored |
| `--dev-line-wide` | off | yes | DEVELOPMENT (2026-11-08, DEV-METAA-HEALTHY-01): the array's line measured on every site this sample's own mix holds firmly (below 0.05 or above 0.95), the identity sites of the cells found set aside, instead of only the ruler's anchors |
| `--dev-metaa-allsites` | off |  | DEVELOPMENT (2026-10-22): also read Met-A for each cell over every atlas site, each weighted by the shift a loss of that cell would make (stage_metaa_atlas.read_in_mix_allsites), recorded beside the identity-site reading |
| `--dev-metaa-atlas` | None |  | DEVELOPMENT - not commissioned (DEV-METAA-ATLAS-01, 2026-10-22): Met-A of an isolated CELL (an IAM-Atlas v2.1 cell name, e.g. monocytes) against that cell in IAM-Atlas v2.1, read on the self-tared values (Stage T step 1, the same map the commissioned Met-A reads); written under bundle['development']['metaa_atlas'], not part of the reading |
| `--dev-metaa-curve` | off |  | DEVELOPMENT (plan 14.14, 2026-10-25): with --dev-metaa-atlas CELL, read Met-A with the atlas as the ruler on the array's own response curve (stage_metaa_atlas.read_on_curve) instead of the same-run tare |
| `--dev-metaa-sites` | off |  | DEVELOPMENT: with Met-A on sequencing, write every identity site's reads, levels and measured and expected disorder beside the bundle (NAME_metaa_sites.csv.gz), for site-by-site diagnosis |
| `--dev-metaa-specific` | None |  | DEVELOPMENT (39.9): also read Met-A for each cell on sites held by that cell and differing by at least DELTA from every other cell of the panel |
| `--dev-one-scale` | off |  | DEVELOPMENT (2026-11-09): arrays; the per-cell readings (Met-A, C-Score, departures) on Stage 1's values, one scale with the array's own measured line (the self-tare map rescales only its 6,963 anchor probes and leaves the rest raw, so 3 to 7 percent of identity sites were corrected twice); the self-tare path's per-cell Met-A kept beside it as met_a_each_cell_selftare |
| `--dev-plant` | None |  | TESTS ONLY (2026-10-22, author: tests run through the chain): plant a CONSTRUCTED loss of CELL's pattern (an IAM-Atlas v2.1 cell) in this array's values after Stage 1, before every reading, by stage_metaa_atlas.plant; in whole blood the loss is scaled by CELL's share as the IAM-Deconvolver reads it on the unplanted values. Every report and bundle says PLANTED; the bundle records the plant (key 'planted') |
| `--dev-plant-ends` | None |  | TESTS ONLY (2026-11-10, the self-calibration synthetic): on a .pat input, at each molecule's outermost call a C read T with probability END_CT and a T read C with END_TC, and at every call a C read T with ALL_CT (stage_seq_values.plant_ends); a PLANTED banner |
| `--dev-plant-territory` | None |  | TESTS ONLY: a constructed gain at islands or loss in domains, planted by the chain |
| `--dev-pred-raw` | off | yes | development (2026-11-07): the healthy prediction of an array on its own values, not on the atlas ruler's |
| `--dev-provisional` | None | yes | DEVELOPMENT (plan 40.18): with the full atlas panel, a second deconvolution with provisional columns (stage_provisional) beside IAM-Atlas v2.1, recorded as cell_shares_provisional and flagged; the atlas's own shares are unchanged |
| `--dev-ruler` | off | yes | DEVELOPMENT (plan 0.2): put the array on IAM-Atlas v2.1's scale (stage_t_selftare.ruler_map) before the whole-sample Met-A and the territories. Off until the many-level ruler passes (the two-anchor line overcorrects: healthy 1.113 untared, 0.938 on it) |
| `--dev-selftare-ii` | off |  | no-op alias, kept so recorded commands still run: self-tare II (DEV-SELFTARE-02) is Stage T step 1 since 2026-10-04 and always runs (bundle tare.selftare_ii); the flag only copies that record to bundle['development']['selftare_ii'] and changes nothing else |
| `--dev-seq-atlas-dir` | None |  | DEVELOPMENT (2026-11-09): on sequencing, read against a rebuilt IAM Sequencing Atlas in DIR (DIR/array_sites/ for the in-mix readings, DIR/deconv_reference_v1.parquet for the Deconvolver) instead of the committed one; e.g. the atlas counted with the molecules' end calls left out, for samples read with --dev-seq-trim-ends; the bundle names it and the reference door records it as not door-checked |
| `--dev-seq-beta` | None |  | DEVELOPMENT (40.10): read a .beta file of per-CpG counts (hg19 CpG numbering) as values at IAM-Atlas v2.1's sites; every reader but IAM-A (no molecules); with --dev-seq-values N, sites with fewer than N reads have no value; or CALL.bw,COV.bw, a GRCh38 bigWig pair of per-CpG levels and coverage (2026-10-24) |
| `--dev-seq-copies` | off | yes | DEVELOPMENT: on sequencing, count the sample's calls as copies of fewer molecules, by its own copy distribution measured at the library-floor sites (linear-amplification libraries; DEV-ONE-DRAW-01) |
| `--dev-seq-counted-mix` | off |  | DEVELOPMENT (edit 226): Met-A whole and in the mix, and the territories, with the expectation from the reference's own reads at a common depth (counted_mix); on with --dev-seq-half-cols |
| `--dev-seq-donors` | None |  | DEVELOPMENT: a per-donor counts file in the committed format (cell, probe, cpg, gsm, half, m, n), for synthetics |
| `--dev-seq-end-cal` | off |  | DEVELOPMENT (2026-11-12, with --dev-seq-self-cal; the self-calibration's eighth form): before the cell shares, a first Deconvolver pass finds the cells, the library's two molecule-end error rates are measured where those cells agree on the atlas (agree_rates), every site's level is corrected for them (end_corrected_levels), and the shares and every reading then use the corrected levels |
| `--dev-seq-half-cols` | None |  | DEVELOPMENT (edit 222): a folder with array_sites_halfA/B; with --dev-seq-ref-half every in-mix reading (Met-A whole and in the mix, territories, C-Score) is read against that donor half only |
| `--dev-seq-level` | None | yes | DEVELOPMENT: on sequencing, the healthy level the expectation uses: pooled (methylated over all reads of the cell's donors; the default) or median (of the donors' own levels; the earlier rule) |
| `--dev-seq-noise-weight` | off |  | DEVELOPMENT (2026-11-09): on sequencing, the IAM-Deconvolver weights each site by its own read count's counting noise (from the model's prediction, as each sky pixel by its noise); off until its synthetic is met |
| `--dev-seq-ref` | None | yes | DEVELOPMENT (plan 35.30): with --dev-metaa-atlas on sequencing values, the healthy levels from the IAM Sequencing Atlas's Met-A reference (auto: Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet) instead of IAM-Atlas v2.1 |
| `--dev-seq-ref-half` | None |  | with --dev-seq-ref: that sealed donor half only (for a sample of the other half) |
| `--dev-seq-self-cal` | off |  | DEVELOPMENT (2026-11-10, sequencing step 1): on a .pat input, Met-A's healthy expectation is the IAM Sequencing Atlas's inner level (its donors without their molecule-end calls) moved by this library's own measured end error per side (stage_metaa_atlas.self_calibrated); nothing cut, nothing borrowed from other samples |
| `--dev-seq-self-cal-agree` | off |  | DEVELOPMENT (2026-11-11, with --dev-seq-self-cal): the fifth form; the library's two molecule-end error rates measured where every found cell agrees (all at most 0.1 or all at least 0.9 on IAM-Atlas v2.1; stage_metaa_atlas.agree_rates), put into the mixed forward model (mix_forward); every read records A_agree beside the other forms |
| `--dev-seq-self-cal-data` | off |  | DEVELOPMENT (2026-11-10, with --dev-seq-self-cal): Met-A read on this library's own calls with its measured molecule-end error taken out, against the IAM Sequencing Atlas's inner level (stage_metaa_atlas.read_in_mix); every read also records A_without, A_expectation_side and A_data_side |
| `--dev-seq-self-cal-disorder` | off |  | DEVELOPMENT (2026-11-10, with --dev-seq-self-cal): Met-A as the observed disorder minus the disorder this library's measured molecule-end error adds (in expectation under counting), against the IAM Sequencing Atlas's inner level; every read records A_disorder_side beside the other forms |
| `--dev-seq-self-cal-dump` | None |  | DEVELOPMENT (2026-11-12, with --dev-seq-self-cal): write each read cell's per-site self-calibration inputs (reads, end reads, end and all marked calls, the measured level, the atlas's inner and published levels mixed in the sample's shares) to PREFIX_<cell>.csv.gz, for diagnosis; changes no reading |
| `--dev-seq-self-cal-mix` | off |  | DEVELOPMENT (2026-11-10, with --dev-seq-self-cal): Met-A's expectation is every found cell's IAM Sequencing Atlas inner level mixed in this sample's shares, with this library's measured molecule-end error flipping calls in proportion to each state's share at each site (stage_metaa_atlas.mix_forward); every read records A_mix_forward beside the other forms |
| `--dev-seq-self-cal-own` | off |  | DEVELOPMENT (2026-11-12, with --dev-seq-self-cal): the sixth form; the disorder the library's molecule-end error adds, computed at the library's own level at each of the cell's identity sites, the end error measured at those sites by mixed atlas level (stage_metaa_atlas.own_level_added), taken out of the observed disorder; every read records A_own_level beside the other forms |
| `--dev-seq-self-cal-reach` | None |  | DEVELOPMENT (2026-11-12, with --dev-seq-self-cal; the seventh form): Met-A read on the library's calls past each side's reach (marked side: past the last call; unmarked side: past the last N calls), against the IAM Sequencing Atlas's level past the same reach (its inner references at one and N calls); nothing corrected; every read records A_reach beside the other forms |
| `--dev-seq-trim-ends` | None |  | DEVELOPMENT (2026-11-09): on a .pat input, the N outermost CpG calls at each end of every molecule left out before any reading (library chemistry at read ends; stage_seq_values.trim_pat) |
| `--dev-seq-values` | None |  | DEVELOPMENT (to-do step 40): with --pat, also make a value at each IAM-Atlas v2.1 site from the molecules (methylated calls over calls, sites with at least MIN_CALLS calls; chain/stage_seq_values.py), so the IAM-Deconvolver, Met-A for each cell, Met-A against the atlas and the globe read the same sequencing run IAM-A reads. The commissioned array readings (Met-A on neutrophils, the C-Score) are not applied to sequencing values; every reading is flagged |
| `--dev-shares` | None |  | DEVELOPMENT, diagnosis only: read every share-based reading against these declared shares instead of the Deconvolver's (which are kept under cell_shares_read) |
| `--dev-terr-counted` | off | yes | DEVELOPMENT: on sequencing, the islands' expectation from the reference's own reads in the sample's mix (expected_H_mix_counts), against --dev-seq-ref-half's donors when given |
| `--dev-terr-counts` | None |  | DEVELOPMENT: a per-half island counts file in the committed format (cell, probe, mA, nA, mB, nB), for synthetics |
| `--dev-terr-sites` | off |  | DEVELOPMENT: write each territory set's per-site table beside the bundle (NAME_terr_sites_islands.csv.gz, NAME_terr_sites_domains.csv.gz) |
| `--dev-territories` | off | yes | DEVELOPMENT (plan 48): disorder at cancer's own territories (islands unmarked in every cell type; loosely held domains), against the atlas in the sample's mix. On by default since 2026-10-23 (author: 'wire in the territories score') |
| `--dev-thin` | None |  | TESTS ONLY (step 40): read a seeded thinned copy of --pat, each molecule kept with this probability (stage_seq_values.thin_pat), so a deep sample stands in for a shallow one; IAM-A and the sequencing values both read the thinned copy; recorded and bannered |
| `--dev-tiles` | 'table' |  | DEVELOPMENT (plan 14.9): Stage Q2's tiles from the commissioned table (default) or from the genome dictionary's chromosome ranges |
| `--dev-whole-sites` | off |  | DEVELOPMENT: write Met-A for the whole sample's per-site table (level, expectation, reads, weight, disorder, territory) beside the bundle (NAME_whole_sites.csv.gz) |
| `--engine` | 'v3' |  | kept so recorded commands still run: v3 names the conductor (conductor_v3.py), which chain v4 runs; the class-floor engine was retired 2026-10-03 |
| `--explorer-spec` | None |  | also write the report globe's layers as a Globe Explorer spec (JSON) to this path, for the site's Globe Explorer (website/art/make_explorer.py sample), so every sample globe on the site is the one this command drew |
| `--full-chain` | off |  | every development reading that applies to the input, beside the commissioned ones, each flagged; a reading that cannot run is recorded with its reason under bundle['full_chain'] (2026-10-29; the replays use it) |
| `--genome-region` | 1000 |  | with --dev-genome-globe: CpGs per region (default 1000) |
| `--geo-dir` | None |  | where --geo-pat files are written (default: a temporary folder) |
| `--geo-pat` | None |  | fetch this GEO sample's .pat file and read it as --pat; the chain takes the file on the build it reads (hg19), never another build (stage_q0_intake.geo_pat; Decoder: genome builds) |
| `--grn` | none |  | the array's green-channel raw file (IDAT, .gz allowed), with --red; Stage 1 calibrates it |
| `--iama-ref-table` | None |  | CSV of same-run healthy references for IAM-A (column A, optional id, pipeline): same laboratory, kit and pipeline; >= 3 rows -> IAM-A median tare |
| `--id` | None |  | the sample's name in the report and bundle |
| `--intake-log` | None |  | append intake and integrity records here |
| `--lambda-bam` | None |  | PHY-091 amendment 3 (development): the same reads aligned to the lambda spike-in genome alone; with --lambda-fasta, the meter read on lambda |
| `--lambda-fasta` | None |  | the lambda genome FASTA (NC_001416.1) for --lambda-bam |
| `--law-card` | None |  | IAM's Law Sphere input card from a file (any keys of stage_law_sphere.CARD) |
| `--law-counts` | None |  | with --build law-sphere-sweep or law-sphere-fit: the saved counts |
| `--law-counts-out` | None |  | with the Law Sphere: save the sample's per-unit molecule counts, so any card can be read again with --build law-sphere-sweep/-fit |
| `--law-set` | None |  | IAM's Law Sphere input card: set one input (repeatable; JSON values, e.g. coins=1.9, held="law", by_type={...}); keys in stage_law_sphere.CARD |
| `--law-sphere-out` | None |  | with the Law Sphere: save the per-unit layers |
| `--law-vary` | None |  | with --build law-sphere-sweep or law-sphere-fit: the input varied and its range |
| `--law-vary-cell` | None |  | with --law-vary: vary the input for this cell type only (by_type) |
| `--ledger` | None |  | append one flat row per run to this JSONL evidence ledger; defaults to evidence_ledger.jsonl beside the report |
| `--manifest-dir` | None |  | write the immutable per-sample manifest here |
| `--metaa-atlas-ref-A` | None |  | with --dev-metaa-atlas: comma-separated development.metaa_atlas.A of >= 3 healthy samples of the same cell run the same way (never this sample); gives A_rel and its state (stage_metaa_atlas.tare) |
| `--no-bundle` | off |  | do not write the machine-readable bundle beside the report (it is written by default: a run that leaves only HTML cannot be pooled later) |
| `--no-info-sphere` | none |  | skip the information sphere reading |
| `--no-intake` | off |  | skip Stage 0 (recorded in the report as skipped) |
| `--no-territories` | none |  | skip the territory readings |
| `--out` | 'methylphys_report.html' |  | the report (HTML) to write |
| `--pat` | None |  | chain v4 Stage Q2 (IAM-A against the sample's own reference); with --dev-iama-position the retired v3 Stage Q: a wgbstools .pat / .pat.gz file; read with the loyfer_pat_v1 extractor (stage_q_iam_a.pat_site_table) |
| `--pat-max-bytes` | None |  | development: read only the first N bytes of --pat; IAM-A is then refused (P is measured on whole files) unless --dev-allow-partial |
| `--patient-id` | None |  | already-hashed identifier; a cleartext one is hashed here |
| `--prior-betas` | None |  | stage 12b (difference map, DEV-TOOLKIT-ADDED-01): the Stage 1 beta vector of an earlier draw of the same person (as written by --save-betas); needs --prior-bundle |
| `--prior-bundle` | None |  | stage 12b: the bundle of that earlier draw; the difference is refused unless both draws carry the same identifier hash, array type and pipeline |
| `--prior-geo-pat` | None |  | the same for --prior-pat |
| `--prior-name` | 'the earlier reading' |  | with --prior-pat: what the other file is, as the globe names it |
| `--prior-pat` | None |  | with --dev-genome-globe: a .pat file of an earlier reading of the same person or specimen |
| `--red` | none |  | the array's red-channel raw file (IDAT, .gz allowed), with --grn |
| `--ref-fasta` | None |  | the reference FASTA the --bam was aligned to (indexed) |
| `--save-betas` | None |  | also write this specimen's Stage 1 beta vector (after the detection mask) to this path as a two-column file cpg_id,beta (.parquet or .csv); recorded in the bundle |
| `--seq-cell` | None |  | cell type of the sequenced specimen (Stage Q); default: for --pat and --bam, the cell the IAM-Deconvolver found most of among those IAM-A holds a reference for (2026-11-07: never the specimen's words); a --site-table has no Deconvolver, so name the cell |
| `--seq-chemistry` | 'whole-genome bisulfite' |  | chemistry of the sequencing input (IAM-A is commissioned on whole-genome bisulfite) |
| `--seq-pipeline` | None |  | the read-level pipeline that produced --site-table (required with it); IAM-A is refused unless a position P was frozen for it |
| `--sesame-rscript` | os.environ.get('METHYLPHYS_SESAME_RSCRIPT') |  | Rscript of an environment with Bioconductor sesame (--dev-epic-v2) |
| `--sex` | None |  | declared sex (F/M), optional; when given Stage 0.8 compares it with the array, otherwise the array's sex is recorded |
| `--site-table` | None |  | v3 Stage Q: a per-site CSV with columns pos,opp_A,err_A,opp_B,err_B; needs --seq-pipeline |
| `--slide-ref-A` | None |  | comma-separated Met-A before the median tare (met_a.A, self-tared since 2026-10-04) of >= 3 same-run healthy reference arrays (same slide, else same batch) - v3 Stage T median tare, whole blood and isolated neutrophils |
| `--slide-ref-table` | None |  | CSV of same-run healthy references with columns A and C (optional id): Met-A and the C-score from pass 1 (met_a.A, self-tared since 2026-10-04; before the median tare). >= 3 rows -> median tare (nothing is fitted) |
| `--specimen` | 'whole blood' |  | what the sample is (whole blood, plasma, tissue, sorted cells, ...); picks the route only, never the cell: the Deconvolver decides what is in it |
| `--workers` | 4 |  | processes for counting --bam |

## How to read the Decoder

The chain is the code that turns one sample into one report. This part of the manual takes it apart in the order a real run
executes it: a traced run of the one command on a real healthy whole-blood array (GSE235717, GSM7508429; `chain/trace/`), and a
second traced run on a small constructed sequencing file for IAM-A. Every function the one command can load has its own entry, in
the file where it lives, in the order the run first enters that file.

Each stage opens with what it is for, the physics in words and then the formula, what it decides, every gate with its measured value,
its limit, where the limit came from and whether it is provisional, the files it reads, and how to break it. Each function entry
then says what the function does and what it decides. Under every entry the manual prints, read from the code itself and checked
at every push: the file and line, the arguments, the module constants it reads, the data files it names, and how many times each
traced run called it.

Every stage carries one of two labels, so the physics can be told apart from the bookkeeping:

- **Standard method**: a technique used elsewhere in the field (array calibration, deconvolution of a cell mix, normalization
  against references run the same way). IAM uses it as a tool. It is not evidence for IAM's law, and a reviewer can check it
  against the literature.
- **From IAM's physics**: a quantity that follows from the cost of holding a mark against thermal noise. The entry gives the
  derivation and the test that could have failed.

**Names.** The gauge has three marks. The **floor** is the least entropy a held mark can carry, derived from thermal kicks against
about one ATP per site (M = 20.94). The **ceiling** (saturation) is every identity site fully scrambled. **Healthy = 1**
is Met-A's healthy reference, measured once on purified healthy cells of the same type, and, for IAM-A, IAM-A's baseline, computed at the
sample's own loose share. Some file names in the chain predate these names: `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`
holds Met-A's healthy reference, not the floor. The Decoder uses the canon's names and says which file holds what.

**One reference per reading.** An entropy is read only against a reference made the same way: the atlas measured with the sample's own chemistry, with the array's own line and noise measured on the array (the same-run tare that once served is retired). Met-A's healthy
reference is purified arrays read by Stage 1; IAM-Atlas v2.1 blends 21 sources, and on Met-A's sites its neutrophil reads 1.062 against
Met-A's reference although the values differ by 0.007 on average (`doors/data/DEV_REFERENCES_01/`): the entropy of a blend is higher than
the average entropy of its parts.

**Built from what, tested on what.** A reviewer's first question about any reference is whether it was tested on the data it was
built from. Every reference entry names both.

## Where IAM's Law is in the chain

The law: holding a mark costs energy paid at the surface that holds it, at that surface's own temperature, never less than k_B T ln 2 per bit
(Landauer). For a cell the surface is the DNA, at 310 K, and the drive is one ATP: M = ΔG_ATP / RT = 20.94. Every IAM quantity
the chain reads comes from that one statement; everything else in the chain is standard method.

- **The floor (derived).** A mark held behind a wall of M thermal kicks is lost with odds 1/(1 + e^M): millions of times below any reading.
- **The ceiling (derived).** One bit per site, every identity site scrambled: H(β) = 1 bit at β = 0.5 (Met-A's `_ceiling` caveat).
- **The healthy level (derived from M alone).** A mark kept by a constant push of two coins per restore (one S-adenosylmethionine) against
  thermal kicks (first passage under a drive) has odds(w) = (w/2) / (1 − (1 − e^−w)/w) restores per loss at w = 2M: held error
  0.0445 per site and held share φ = E_hold/M = 0.1464 from M alone. In mature blood no eraser share is
  needed (a lead); in stem and progenitor types the measured eraser share is part of that type's distance from IAM-A's one line. Read through
  the counter at the sample's own clustering it gives IAM-A's derived intercept, 0.0359 at the healthy clustering (Section 9), and through H(β)
  the healthy level at fully held sites, which arrays cannot resolve (Section 5). The chain reads against the measured references; the
  derivation explains them, and the measured IAM-A line sits 1.12 times the derived level in the atlas laboratory and 0.96 to 1.03 times it in
  the blind-test laboratory.
- **What every IAM-A reading carries.** Besides A, each reading keeps the count's run lengths (loss runs against the
  independent expectation), the held error in firm and loose tiles (the line tested tile by tile), and the meter's own fields. On a molecule
  file (.pat) the meter's fields are empty: conversion failure and the T-to-C slip are counted only from an aligned BAM, and the lambda
  spike-in only on the count-only route. None of these fields is used by A or the state; they are recorded.
- **The derived line beside A.** An IAM-A reading from a molecule file whose clustering (coupling) lies inside the forward table's range also gives A_derived: the held error over the line whose
  intercept is the held error from M alone (0.0445, no eraser input in mature blood, a lead) read through this counter
  at the sample's own clustering (Runtime Matrices/IAM_A_SelfRef/derived_line_v1.json), with the measured slope. A (the measured line) stays
  the default while the two laboratories disagree: the atlas laboratory reads 1.12 times the derived level, the blind-test laboratory
  0.96-1.03 times.
- **Islands and clock sites in every reading.** Each sequencing reading also counts, by the same rules as the held error, the held and
  absent sides at the islands territory (sites every healthy cell type keeps unmarked, so what is counted there is the library's own meter,
  register T1) and at the epiTOC2 mitotic-clock CpGs with 25 CpGs either side (register C3), written as iam_a.sites. No reading uses them yet.
- **The healthy band on the derived line.** Where A_derived exists the reading also gives state_derived: A_derived against band v2
  (Runtime Matrices/IAM_A_SelfRef/band_v2.json), the 2.5th to 97.5th percentile of A_derived over the healthy atlas libraries read by the one
  button, by the same rule as band v1. Band v1 on the measured line stays the default and keeps its commissioning record; band v2 is in
  commissioning, because the libraries that set it also set the line. The report prints A_derived and state_derived beside A, marked in commissioning
  (report_v4._iama_derived). Readings from an aligned BAM get neither (the BAM count has no clustering), and the report says so.
- **Met-A sequencing at held sites, against the derived level.** Each sequencing Met-A reading also gives A_derived_held: the library's
  disorder over the cell's held identity sites (atlas level at least 0.80) over the disorder of 1 - eps0 (0.0445, the held
  error from M alone) at each site's own read count. It sits beside A; the per-type reference stays measured, because each type's own
  position (A8) is not yet derived.
- **The balance (derived form).** Lost with probability f and restored with g, the held error settles at ε = f/(f + g), E_hold = kT ln[(1 − ε)/ε]
  (Section 9).
  Established neighbours: the floor 1/(1 + e^M) is the one-step copy limit of Sartori and Pigolotti (2013); England's (2013) bound
  from detailed balance says that holding a mark at error ε costs at least kT ln(1/ε) of heat per interval, about 3 kT at the
  derived level and one coin at the floor. Both rest on Landauer's price of a bit at kT.
- **The eraser (measured, by cell type).** The share of losses the erasers cause is measured in eraser-removal libraries (PHY-085, PHY-086,
  PHY-090): 13.6 % in stem cells, about 5 % in hematopoietic progenitors, near zero in mature B cells; its price per erasure is calculated
  (PHY-088). IAM-A has one line for every cell type: in mature blood the share drops out (a lead), and in stem and progenitor types it is one
  contributor to that type's distance from the line, inside each type's own level, which is open (register A8).
- **The bit.** Met-A reads disorder in bits, H(β); IAM-A reads the held error, the share of held sites wrong at any moment.

## Inputs: every number the chain reads, with its physics

This section lists every number the chain reads, so that a researcher can rebuild the chain's arithmetic, or write a new chain, from the
Decoder alone. One entry per input: a file under `chain/Runtime Matrices` that the stage files open, or a constant of the canon
(`CANON/iam_canon.json`). Each entry says what kind of number it is, which register rows decide its status, what it is, its physics (step
by step for the derived and calculated ones, enough to recompute it; for the measured ones, how it was measured and why it stays measured),
which function and stage read it, the command or calculation that reproduces it, and what is still open.

**The five kinds.**

- **Derived**: follows from IAM's Law at the DNA (M = ΔG_ATP / RT) and established physics, with no measured input beyond the coin and the
  temperature.
- **Calculated**: arithmetic or established mathematics on stated inputs (a genome's CpG table, a geometry, a published bound, a count's null).
- **Measured**: counted from healthy cells, or taken from the literature, and kept measured: nature sets it and no derivation is claimed.
- **Calibrated**: the instrument's own response (the meter), measured once on references or on each array and library, never derived.
- **Development**: a reference or rule that is read and flagged but not commissioned, or retired and kept for earlier records.

**How the canon reaches the chain.** The chain's stage files do not open `CANON/iam_canon.json` while they run. Every number a reading uses
is in a stamped file under `Runtime Matrices` (or in the atlas folders), and the canon holds the same numbers with their derivation so that
`CANON/canon_check.py` can refuse any file that disagrees. Each canon entry below names the file through which its value reaches a reading,
or says that it reaches none and is printed by the Decoder and the report only.

**Where the files sit.** Paths are relative to `Biological_Physics/MethylPhys/chain`. Every run records the size and a SHA-256 prefix of
every file under `Runtime Matrices` in its bundle (`run_sample.py::_versions`), whether or not that run reads it, and the reference door
checks the atlases against their checksums before any reading (`reference_door.py`).

### The canon's constants: IAM's Law at the DNA, and what it is compared with

#### Input: dG_ATP (canon)

**Kind:** measured

**Register:** A10; A16 (each type's own coin, open)

**What it is:** The free energy one ATP hydrolysis delivers in the cell, the coin: 54000 J/mol.

**Physics / equation:** Measured, not derived: the nameplate value of cellular ATP hydrolysis (Nelson and Cox, Lehninger). Tissue measurements by 31P-NMR and magnetic resonance spectroscopy span 53.5 to 61.7 kJ/mol (the canon's band). It stays measured because it is set by the cell's own concentrations of ATP, ADP and phosphate; free ADP is not measured in blood cells, so a coin per cell type is not yet an input (register A16).

**How the chain uses it:** Reaches every reading only through M (below). No stage file reads it directly.

**Reproduce it:** Read the canon's `band_note` and the cited sources; M follows as below.

**Status and open parts:** Stays measured. Each cell type's own coin is open (A16); the lever sweep of the cell simulator finds that the coin would have to change 1.42 times to make the spread of healthy types, against 1.15 across tissues, so the coin does not set each type's level (A8).

#### Input: T_cell (canon)

**Kind:** measured

**Register:** A11

**What it is:** The temperature of the surface that holds the record, the nucleus at body temperature: 310.15 K.

**Physics / equation:** Measured (37 °C). It stays measured: IAM's Law prices the hold at the holding surface's own temperature, and that temperature is a fact of the body. Through the derived passage (odds about M + 1/2) a nucleus 1 K warmer raises the held error by a factor 1.003, so temperature differences between cells cannot make the spread of healthy types (register A8).

**How the chain uses it:** Reaches every reading only through M.

**Reproduce it:** None to compute.

**Status and open parts:** Stays measured.

#### Input: R and k_B (canon)

**Kind:** calculated

**Register:** none (SI constants)

**What it is:** The gas constant R = 8.314462618 J/(mol K) and Boltzmann's constant k_B = 1.380649e-23 J/K, exact in SI since the 2019 redefinition.

**Physics / equation:** R = N_A k_B. A kT is one thermal kick: 2.9681e-21 J is k_B T ln 2, Landauer's price of one bit at 310.15 K.

**How the chain uses it:** Reach the chain only through M.

**Reproduce it:** `python3 -c "print(6.02214076e23*1.380649e-23)"` gives R.

**Status and open parts:** Exact by definition.

#### Input: M_cell (canon)

**Kind:** derived

**Register:** A1 (uses M); A4; A10; A11

**What it is:** IAM's Law at the DNA: the drive ratio of the cell's memory, M = 20.94 thermal kicks per coin (30.2 in Landauer units, M / ln 2).

**Physics / equation:** M = ΔG_ATP / (R T) = 54000 / (8.314462618 × 310.15) = 20.94. The moles cancel: one ATP carries 20.94 kT. Every IAM quantity the chain uses below comes from this one number; the rest of the chain is standard method.

**How the chain uses it:** No stage file reads it; it is built into the derived files below (the gauge's floor, the forward table behind IAM-A's derived intercept) and printed by the Decoder and the report.

**Reproduce it:** `python3 -c "print(54000/(8.314462618*310.15))"` → 20.9405.

**Status and open parts:** Derived from two measured inputs (coin and temperature), which stay measured.

#### Input: H_min_cell (canon): the floor

**Kind:** derived

**Register:** A1; M1

**What it is:** The gauge's floor: the least disorder a held mark can carry, 2.5e-08 bits, at held error 1/(1 + e^M) = 8.05 × 10⁻¹⁰.

**Physics / equation:** A mark held behind a wall of M kicks, one coin per copy, is lost with probability ε_min = 1/(1 + e^M) = 1/(1 + e^20.94) = 8.05 × 10⁻¹⁰. Its disorder is H(ε_min) = −ε log₂ ε − (1 − ε) log₂(1 − ε) = 2.5 × 10⁻⁸ bits. Established neighbor: the one-step copy limit of Sartori and Pigolotti (2013), lowest error 1/(1 + e^x) for discrimination x kT (register E3).

**How the chain uses it:** Read as the gauge's left end from `Runtime Matrices/Gauge/cell_gauge_marks_v1.json` by `report_figs_v4.gauge_png`; no reading divides by it.

**Reproduce it:** `python3 -c "import math; M=54000/(8.314462618*310.15); e=1/(1+math.exp(M)); print(e, -(e*math.log2(e)+(1-e)*math.log2(1-e)))"`.

**Status and open parts:** Derived; nothing open.

#### Input: SAM_cost_ATP (canon): the price of one restore

**Kind:** derived

**Register:** A2

**What it is:** The coins spent to make one S-adenosylmethionine, the methyl donor of each restore: 2.0 (range 1.967 to 2.033).

**Physics / equation:** Making SAM spends ATP → AMP + 2 Pi = (ATP → ADP + Pi) + (ADP → AMP + Pi). Adenylate kinase (ATP + AMP ⇌ 2 ADP) links the two bonds, so SAM / coin = 2 − ΔG_AK / coin, exactly 2 at adenylate-kinase equilibrium whatever the concentrations; the range allows adenylate kinase within a factor two of equilibrium. Counts only the ATP spent making SAM. The textbook standard free energies give 2.12 (canon SAM_cost_ATP_textbook), kept as a check.

**How the chain uses it:** Built into the drive w = 2M of the paid hold (below); no stage file reads it.

**Reproduce it:** `Biological_Physics/MethylPhys/minicell/sam_cost_ak.py`.

**Status and open parts:** Derived, given adenylate kinase at equilibrium.

#### Input: W_restore and odds_restore (canon): the paid hold

**Kind:** derived

**Register:** A3; A4; PHY-083; blind test PHY-084

**What it is:** The push behind each restore, W = 2M = 41.88 kT, and the odds it buys, 21.45 restores per loss.

**Physics / equation:** A lost mark is restored by a constant push of two coins (register A3: a constant push is the best of 2,000 push profiles two coins can buy). The restore is a driven first passage: with the push w = W/kT = 2M = 41.88 across the step and the Einstein relation fixing the noise, the ratio of restores to losses is odds(w) = (w/2) / (1 − (1 − e^−w)/w) = 20.94 / (1 − 1/41.88) = 21.45. For a strong push odds ≈ M + 1/2, which is why the result depends only weakly on the coin and the temperature.

**How the chain uses it:** Built into eps0_candidate and so into the forward table behind IAM-A's derived intercept; no stage file reads it.

**Reproduce it:** `python3 -c "import math; M=54000/(8.314462618*310.15); w=2*M; print(w, (w/2)/(1-(1-math.exp(-w))/w))"` → 41.881, 21.4528.

**Status and open parts:** Derived. The identification of a mark's loss with the reverse step of the push is the physics claim the blind test checked (PHY-084 met, B cells on watch).

#### Input: eps0_candidate (canon): the held error from M alone

**Kind:** derived

**Register:** A4; A6; S1; PHY-083; PHY-084; F033

**What it is:** The share of held sites lost at any moment when the hold is paid by M alone: 0.0445 per site (0.04386 to 0.04524 over the SAM price's range, canon eps0_candidate_SAM).

**Physics / equation:** Each interval a held mark is lost with probability f and restored with g; the balance settles at ε = f/(f + g) = 1/(1 + g/f) = 1/(1 + odds(2M)) = 1/22.45 = 0.04454. The balance is set by the ratio, not by how often the cell divides; division only sets how fast it is reached. This is a per-site loss. The counter reads a lone-loss share on qualifying molecules, a different quantity, so it is compared with a measurement only after planting molecules at this loss and reading them with the same counter (the forward table, Section 9).

**How the chain uses it:** Not read by any stage file at run time. It is the planted loss of the forward table (`doors/data/DEV_CHAIN_Q2_01/forward_line_synthetic.py`) that gives IAM-A's derived intercept, and the held level 1 − ε0 behind Met-A's derived height (register M2).

**Reproduce it:** `python3 -c "import math; M=54000/(8.314462618*310.15); w=2*M; o=(w/2)/(1-(1-math.exp(-w))/w); print(1/(1+o))"` → 0.044538.

**Status and open parts:** Derived. Blind test PHY-084 (an unseen laboratory, GSE173787, 54 libraries) met by the standard written before the seal, B cells on watch (pull −2.45), none beyond 3. Compare the measured healthy pool, canon eps_healthy_pool.

#### Input: E_hold_candidate and phi_candidate (canon): the holding energy and the held share

**Kind:** derived

**Register:** A4; P4; E7

**What it is:** The energy that holds one mark, E_hold = 3.066 kT, and the share of one coin it is, φ = E_hold / M = 0.1464.

**Physics / equation:** E_hold = kT ln[(1 − ε0)/ε0] = kT ln(odds) = ln 21.45 = 3.066 kT. φ(M) = ln(odds(2M)) / M = 0.1464; for a strong push φ ≈ ln(M)/M, within 1 % at the cell's M. Measured on 56 healthy cell types: 3.084 kT and 0.1473 (canon E_hold_meth, phi). In plain words: the cell pays in coins worth 20.94 kicks each, and each mark is held at about 3 kicks' worth. The driven-passage odds are the same function as a write ramped slowly against its own clock (kept over lost = (x/2)/(1 - (1 - e^-x)/x)), with x = w = 2M here. Sudden (x -> 0) gives one restore per loss, a coin flip; the cell at x = 2M gives about 21, and φ = ln(odds)/M follows. φ is a share of the coin, not the energy half that a sudden capacitor charge keeps.

**How the chain uses it:** Printed by the Decoder and the report; no stage file reads it.

**Reproduce it:** `python3 -c "import math; print(math.log(21.4528), math.log(21.4528)/20.9405)"`.

**Status and open parts:** Derived.

#### Input: England_hold_heat_kT (canon): England's bound, the established neighbor

**Kind:** calculated

**Register:** E7; A4

**What it is:** The least heat that holding a mark at error ε costs per hold interval, by England's bound: 3.11 kT at ε0.

**Physics / equation:** England (2013, J. Chem. Phys. 139, 121923, eq. 6), from microscopic reversibility: β⟨ΔQ⟩ + ln π(reverse) + ΔS_int ≥ 0, so holding a mark at error ε costs at least kT ln(1/ε) per interval even when nothing changes. At ε0: ln(1/0.04454) = 3.111 kT, which is E_hold plus ln(1/(1 − ε0)) = 3.066 + 0.046 kT: an identity on the same ε0, not a second estimate. At the floor ε = 1/(1 + e^M): ln(1 + e^M) = M, exactly one coin. England supplies the inequality (no M, no ATP, no methylation); IAM supplies M → odds → ε0.

**How the chain uses it:** Printed only; no stage file reads it.

**Reproduce it:** `python3 -c "import math; e=0.0445379; print(math.log(1/e), math.log(1/(1-e)))"`.

**Status and open parts:** Calculated (an established bound applied). Still to check: that a held mark's loss is England's reverse transition, and the internal entropy term of a marked against a bare site.

#### Input: E_hold_meth, phi, eps0_meth and H_ref_cell (canon): the measured holding energy

**Kind:** measured

**Register:** A4 (comparison); M2 (H_ref_cell caution)

**What it is:** What 56 healthy cell types actually hold, from the chain's own counts of 153 healthy samples (GSE186458): E_hold = 3.084 kT per interior opportunity (SD 0.12 across types), φ = 0.1473, held error 0.04376, and its disorder H_ref = 0.2593 bits.

**Physics / equation:** ε measured by the counter (Section 9); E_hold = ln[(1 − ε)/ε]; φ = E_hold / M; H_ref = H(ε). Measured: this is the healthy reference height as read, and it stays measured because it carries each type's own level (register A8). H_ref_cell is in sequencing units per interior opportunity, not an array level: it is compared with the derived height only through the same counter.

**How the chain uses it:** Printed by the Decoder in Section 9; no stage file reads it. The file is `doors/data/DEV_CHAIN_Q2_01/holding_energy_q2.json` (`holding_energy_q2.py`).

**Reproduce it:** `python3 doors/data/DEV_CHAIN_Q2_01/holding_energy_q2.py` from `Biological_Physics/MethylPhys`.

**Status and open parts:** Measured. The earlier instrument-corrected 3.77 kT was made on an older counter and is not carried.

#### Input: eps_healthy_pool (canon)

**Kind:** measured

**Register:** A4; A6

**What it is:** The healthy lone-loss intercept pooled over 197 samples (153 + 54 atlas samples, neurons out): 0.04526 (95 % range 0.04375 to 0.04657).

**Physics / equation:** Measured: the intercept of the lone-loss reading at zero loose share across the pooled healthy samples. It is the measured quantity the derived ε0 is compared with on the lone-loss axis, not an input of any reading.

**How the chain uses it:** Not read by the chain.

**Reproduce it:** `doors/data/PHY_050_PERCALL/m_or_m_plus_1.py` (summary `m_or_m_plus_1_summary.txt`).

**Status and open parts:** Measured.

#### Input: A_healthy_type_range (canon)

**Kind:** measured

**Register:** A8

**What it is:** Where healthy cell types sit on IAM-A, read by the one command on 31 atlas libraries: A from 0.77 (cortical neurons) to 1.07 (striated muscle).

**Physics / equation:** Measured: each type sits at its own fixed A, with no rise over donor age in any non-dividing type; T-cell memory sits above naive inside the same donors. This spread, about 81 % of IAM-A's healthy spread, is each type's own level, the open item of the cell's derivation.

**How the chain uses it:** Not read by the chain. It is why IAM-A is commissioned for CD8 T cells only and every other type is labeled development.

**Reproduce it:** `run_sample.py` on the 31 libraries (GSE186458) with its defaults.

**Status and open parts:** Measured. Deriving each type's level is register A8 (open); ruled out so far: the coin, the temperature, the SAM price (energy levers), division rate, a duty-cycle crew form, cooperativity.

#### Input: strand_coupling_maintenance_only and strand_coupling_wild_type (canon)

**Kind:** measured

**Register:** A12

**What it is:** Hairpin (both-strand) counts: fully lost pairs over independent strands, 5.08 to 8.21 with maintenance only and 5.43 to 5.56 in wild type (GSE116482).

**Physics / equation:** Measured; in the three-state balance the ratio equals the restore-rate ratio (half-lost over fully lost pair), which supports one SAM per restore (register A2). The reading is interpretation. Distinct from nearest-neighbor coupling along one strand (register CP1).

**How the chain uses it:** Not read by the chain.

**Reproduce it:** `doors/data/DEV_MINICELL_01/hairpin_timecourse/strand_pairs.py`.

**Status and open parts:** Measured.

#### Input: Met_A_floor_EPIC_neutrophil and Met_A_floor_450K_neutrophil (canon)

**Kind:** measured

**Register:** M5

**What it is:** Met-A's healthy reference for neutrophils on EPIC v1: H_ref = 0.33026 bits; the 450K value is pending (a rebuild from purified 450K neutrophils, GSE88824).

**Physics / equation:** The mean over 6 purified healthy EPIC neutrophil arrays (Salas, GSE110554) of the mean H(β) over the 6,000 identity sites, calibrated by this chain's Stage 1. It stays measured: identity sites are not fully held (healthy levels 0.75 to 0.95 or 0.05 to 0.25), so their healthy disorder is a fact of the cell type, not a consequence of M alone. The key name 'floor' is older than the gauge's marks: it is the middle mark, healthy = 1.

**How the chain uses it:** Reaches the chain through `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json` (key `floor`), read by `stage_m_met_a.read`.

**Reproduce it:** See that file's entry.

**Status and open parts:** Measured; the 6-array base is noted. Held-out precision SD 0.020, 0.983 to 1.045.

#### Input: Met_A_site_rule (canon)

**Kind:** measured

**Register:** M4

**What it is:** Met-A's site rule: across-array SD at most 0.05; own mean β 0.75 to 0.95 (methylated) or 0.05 to 0.25 (unmethylated); at most 3,000 per side, most stable first.

**Physics / equation:** The held sides follow from the physics (a loss moves a held site toward 0.5, where H is largest); the small spread from the reading itself; the window's ends and the count are operational and stated (Section 5, 'Why these sites').

**How the chain uses it:** Applied when the identity sites were chosen; the chosen sites are in `metA_floors_v1_3.json` and, for every atlas cell, by `stage_metaa_atlas.identity_sites`.

**Reproduce it:** Apply the rule to the 6 purified arrays' Stage 1 values.

**Status and open parts:** Measured per cell type on the atlas (M4); partly derivable through A8.

#### Input: MetA_array_held_height (canon)

**Kind:** derived

**Register:** M2; S1; A4

**What it is:** Met-A's healthy level at fully held sites on the array scale, from M alone: 0.9545 array units.

**Physics / equation:** Step 1: the held level in sequencing units is 1 − ε0 = 1 − 0.04454 = 0.95546 (S1, derived as A6). Step 2: arrays read sequencing levels through the atlas's measured transfer (`doors/data/DEV_ATLAS_V21_01/transfer_bounded_v21.json`: monotone, 42 knots, range 0.0089 to 0.9590, worst checked bin 0.018); between its knots 0.955 → 0.9543 and 0.9638 → 0.9590 it maps 0.95546 to about 0.9545. Step 3: that is 0.0044 below the array's saturation 0.959 and inside the transfer's own calibration error, so no array test can resolve it. The strong test is on sequencing (S1, PHY-084 met).

**How the chain uses it:** Not read by any stage file. Met-A on arrays keeps the atlas's measured reference; the derived height is recorded in the canon and the register beside it, flagged not resolvable on arrays.

**Reproduce it:** Interpolate the transfer's knots at 0.95546.

**Status and open parts:** Derived; not resolvable on arrays (register M2). Printing it beside each array reading is a chain item not yet in the code.

#### Input: Normal_band (canon)

**Kind:** development

**Register:** A9 (IAM-A's band); OTH-059

**What it is:** The first commissioning's design tolerance for Met-A, 0.95 to 1.05, superseded as the definition of Normal and kept so that earlier readings reproduce.

**Physics / equation:** None: a tolerance set by hand and confirmed on healthy arrays. Normal is now each cell's measured healthy width (the atlas's own donors), with 1 the middle. IAM-A has its own measured band and never borrows this one.

**How the chain uses it:** Still printed by `stage_m_met_a.read` (constant `NORMAL`) and `conductor_v3._state` until the measured widths (`Runtime Matrices/Met_A_Widths/donor_widths_v1.json`, read by `stage_metaa_atlas.width_state`) replace it.

**Reproduce it:** None.

**Status and open parts:** Superseded; to be removed from the printed state when every reading carries its measured width.

#### Input: islands_meter_slope, islands_meter_intercept and islands_writer_error_range (canon)

**Kind:** measured

**Register:** T1; A17; PHY-093

**What it is:** The islands as each library's own meter: islands stray-mark error = 1.01 × meter + 0.0001 (slope 0.69 to 1.22), and the writer's error at the islands with the meter removed, −0.0002 to 0.0014 per opportunity.

**Physics / equation:** Measured on 53 healthy blood libraries (GSE173787): ordinary least squares of each library's islands error on its meter (conv_fail + tc_err), 2,000 bootstrap resamples. The islands, sites every healthy cell type keeps unmarked, read the meter one for one: the writer leaves essentially no stray marks there, so the islands are an in-sample meter readable from molecule files alone.

**How the chain uses it:** Not subtracted from any reading. The islands count is made by `run_sample.py` (every sequencing reading counts the islands by default, `_site_counts`) (sites from `stage_metaa_atlas.territory_sets`).

**Reproduce it:** `doors/data/DEV_ERASE_01/conv_01/p093_islands_slope.py` on `p093_read.csv`.

**Status and open parts:** Measured (exploration). Reading every library's meter from its islands is a chain item not yet in the code.

#### Input: clock_sites_writer_range (canon)

**Kind:** measured

**Register:** C3; PHY-094

**What it is:** The writer's stray-mark level at the epiTOC2 mitotic-clock sites with the meter removed: 0.0164 (monocytes) to 0.0311 (CD8 T) per opportunity, about twice the genome-wide writer level (0.0112 to 0.0147).

**Physics / equation:** Measured on 54 healthy blood libraries: the clock sites are islands where the shield fails, so they carry the writer's error and an accumulated part. The tick per round needs an age series of one type in one laboratory.

**How the chain uses it:** Not read by any reading. The count at these sites is made by `run_sample.py` (every sequencing reading counts the clock sites by default, `_site_counts`) (`Runtime Matrices/Clock_Sites/epitoc2_sites.csv`).

**Reproduce it:** `doors/data/DEV_ERASE_01/conv_01/` (PHY-094 reading, `p094_read.csv`).

**Status and open parts:** Measured; the clock tick from the writer's error is register C3 (open, lead).

#### Input: P_neutrophil_IAM_A (canon)

**Kind:** development

**Register:** none (retired position form)

**What it is:** The retired position form of IAM-A: 1.1492, H(ε of the other donors)/H(ε0) over 3 granulocyte donors of the published sequencing atlas, valid for that atlas's own pipeline only.

**Physics / equation:** None now: superseded by the self-reference reading (Section 9).

**How the chain uses it:** Read only by the retired Stage Q (`stage_q_iam_a.py`, a development flag) from `Runtime Matrices/IAM_A_Positions/iama_positions_v2.json`.

**Reproduce it:** None.

**Status and open parts:** Retired; kept so earlier records reproduce.

### IAM-A's files (Section 9)

#### Input: Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json

**Kind:** measured

**Register:** A4; A6; A7; A8; F033

**What it is:** IAM-A's line, ε_predicted = firm + slope × L, on the raw held-error axis this counter writes: firm 0.040265 and slope 0.034625 (the Decoder prints them as 0.04026 and 0.03462). No scale factor anywhere.

**Physics / equation:** Measured once on 153 healthy samples of 56 cell types (GSE186458), counted by this same counter. Per cell type: the median loose share L and the opportunity-weighted held error; across cell types a Theil–Sen line (the median of pairwise slopes). Median distance of a type from the line 0.0025. What the numbers mean: the intercept is derived from M alone through the counter (register A4, A6): 0.0359 at the healthy clustering 1.33, so the stored firm is 1.12 times the derived level in this, the atlas laboratory, while the blind-test laboratory reads 0.96 to 1.03 times it (F033). The slope is measured; about 40 % of it is mixing inside a sample and follows from the pump (A7, partly derived), and the rest is cell type, the same open item as each type's own level (A8).

**How the chain uses it:** `stage_q2_selfref.line` loads the newest `selfref_line_v*.json` (or the version asked for) and refuses it if its payload no longer matches its chain stamp; `predict`, `residual` and `reading` use it; `stage_v4.stage_q2` reads every sample through it, and so does the mixed-specimen path of `run_sample.py`.

**Reproduce it:** `cd Biological_Physics/MethylPhys/chain` && `python3 -c "import stage_q2_selfref as Q; J=Q.line(1); print(J['firm'], J['slope'])"`. To rebuild it: `stage_q2_selfref.build_line(per_sample)` on the 153 healthy counts named in the file's stamp (`chain_stamp.inputs`). Check one reading by hand: the test library's held error 0.034412 at loose share 0.0304 gives 0.034412 / (0.040265 + 0.034625 × 0.0304) = 0.8329.

**Status and open parts:** Measured; the chain's default line, loaded by version (`LINE_DEFAULT = 1`), never as the newest file. The derived intercept is read beside it (`derived_reading`, derived_line_v1.json). Making a derived line the default waits on the 12 % between-laboratory offset (open: A5 lead, A8, the meter) and comes together with band v2.

#### Input: Runtime Matrices/IAM_A_SelfRef/derived_line_v1.json

**What it is:** the forward table of the line whose intercept is the held error from M alone (0.0445), read through this
counter at the sample's own clustering (firm_by_coupling), with the measured slope; and eps0 itself, which Met-A sequencing's held-site
reading uses.

**How the chain uses it:** `stage_q2_selfref.derived_reading` (stamp checked) gives A_derived beside A when the sample's coupling lies inside
the table's range; outside it, or on a BAM count (no coupling), A_derived is not given and the reason is recorded.
`stage_metaa_atlas.derived_held` reads eps0 from it. Frozen (FROZEN_INPUTS_v4.json).

**Status and open parts:** Derived intercept, measured slope; reported beside A, not the default. Becoming the default needs a rehearsal and
the author's decision, because it changes the commissioned reading.

#### Input: Runtime Matrices/IAM_A_SelfRef/band_v2.json

**What it is:** the healthy band on the derived line: the 2.5th to 97.5th percentile of A_derived over the healthy atlas libraries read by
the one button, by the same rule as band v1.

**How the chain uses it:** `derived_reading` gives state_derived against it, written to the bundle. Frozen (FROZEN_INPUTS_v4.json).

**Status and open parts:** In commissioning: the libraries that set it also set the line (a rehearsal reference). One pooled band; each
type's own position is in development.

#### Analysis: per-type band against band v2, like for like (DEV-IAMA-BAND-01, COM-IAMA-054, 2026-11-20)

**What it is:** an analysis, not a chain stage. The question: does judging each healthy cell type against its own measured level (A8's substitute, `type_positions_v1.json`) make IAM-A more sensitive without calling healthy libraries abnormal, compared with one pooled band (band v2)?

**How it was done:** 152 IAM Sequencing Atlas libraries (GSE186458), each with A_derived from the one-button joint run. For every library, all libraries from its donor stand-in (GEO lab + age + sex; GEO gives no donor id) are removed and both bands are rebuilt. Pooled band: band v2's rule (2.5th-97.5th percentile of all remaining libraries). Per-type band: centre m = median A_derived of the remaining libraries of the same type; half-width w = t(0.975, df) x s x sqrt(1 + 1/n), s = the pooled within-type relative spread (0.035, df 94), the GUM prediction interval for one new library. Sensitivity is calculated: the prediction does not depend on held error, so a rise r multiplies A_derived by (1 + r).

**Results (123 libraries scored; 16 types with fewer than 2 libraries left out):** per-type band 113 of 123 inside (91.9 %), +/-8.5 %, flags an 8.5 % held-error rise in all 43 types. Pooled band 115 of 123 (93.5 %), one window 0.874-1.178, flags a rise of 0.7 % to 46.9 % depending on the type (10 % in 18 of 43); it misses whole healthy types (erythrocyte progenitors 3 of 3 below, central memory CD4 2 of 3 above). The per-type misses are single libraries just past the edge (|z| 2.1-3.4).

**Physics and basis:** the line's intercept is derived from M alone (A1-A4: M = 20.94, two coins, the slow-write rule, held error 0.04454); the slope is partly derived (A7); each type's level (A8) is MEASURED, used as a substitute value (a pseudo-measurement). This is not a derivation of A8.

**Status and open parts:** exploration on spent data, not commissioning. Coverage 91.9 % is under the 95 % built for (a between-lab term, about -5 %, PHY-084, is not in it). Donor stand-in, not donor ids (COM-IAMA-039 owed). Correction recorded: a first statement called band v2 'commissioned', scored it in-sample and quoted 17.8 % from A = 1; the numbers above replace it. Full record: `doors/DEV_IAMA_BAND_01.md`; rerun: `doors/data/DEV_IAMA_BAND_01/per_type_band_loo.py`.

#### Input: Runtime Matrices/Replication_Timing/hg19_cpg_timing_thirds_{blood,endothelial,epithelial}_v1.npz

**What they are:** the same CpG timing thirds as hg19_cpg_timing_thirds_v1.npz, each built from one family's own ENCODE UW Repli-seq lines:
blood (GM06990, GM12801, GM12812, GM12813, GM12878, K562), endothelial (HUVEC), epithelial (NHEK, MCF-7, HeLa-S3, HepG2). Built on GitHub by edit 157.

**How the chain uses them:** `stage_q2_selfref.count_pat_timing_sets` counts Stage Q2's held and gain errors by each family's thirds in one pass;
every sequencing reading records them under `iam_a.sites.timing_by_family`. They never change a reading. Frozen (FROZEN_INPUTS_v4.json).

**Status and open parts:** Development (register A8, exploration): does the late excess of immune and endothelial types survive their own timing?

#### Input: Runtime Matrices/Replication_Timing/hg19_cpg_timing_thirds_v1.npz

**What it is:** each hg19 CpG's constitutive replication timing (the mean of the 15 ENCODE UW Repli-seq wavelet signals at the CpG), cut into
early, mid and late thirds by CpG, stored as runs of the CpG index. Built on GitHub from the public tracks by edit 153.

**How the chain uses it:** `stage_q2_selfref.count_pat_timing` counts Stage Q2's held and gain errors by the third of each molecule's first CpG,
by `count_pat_sites`' rules; every sequencing reading records the counts under `iam_a.sites.timing`. It never changes a reading.
Frozen (FROZEN_INPUTS_v4.json).

**Status and open parts:** Development (register A8L2, the location test): does the spread of held error between cell types live in
late-copied DNA? The timing is from cell lines, not each type's own.

#### Input: Runtime Matrices/IAM_A_SelfRef/band_v1.json

**Kind:** measured

**Register:** A9; COM-IAMA-054

**What it is:** IAM-A's Normal band v1: 0.7787 to 1.0551.

**Physics / equation:** Measured by nature: the 2.5th to 97.5th percentile of 156 pooled healthy readings A = held error / line prediction (153 from GSE186458, 3 from GSE49627), each through the shape check first. It describes the healthy spread inside the set the line was measured on; it would narrow (from about 0.86 to 1.16 toward 0.94 to 1.07) once each type's own level is derived (A8).

**How the chain uses it:** `stage_q2_selfref.band` loads it and asserts its chain stamp; `stage_v4.stage_q2` gives the state (Normal, above, below) and `stage_v4.counting_error` the counting range's share of its width.

**Reproduce it:** `cd Biological_Physics/MethylPhys/chain` && `python3 -c "import stage_q2_selfref as Q; B=Q.band(1); print(B['lo'], B['hi'], B['n'])"`; rebuild with `stage_q2_selfref.build_band(healthy)`.

**Status and open parts:** Commissioned on healthy CD8 T cells with line v1 (sealed grand test: 7 of 8 read inside). Band v2 is to be rebuilt on the derived line from the joint run's healthy atlas libraries; until then v1 is shown.

#### Input: Runtime Matrices/IAM_A_SelfRef/shape_reference_v1.json

**Kind:** measured

**Register:** COM-IAMA-053 (a gate; no derivation row)

**What it is:** The sample-shape gate: per healthy cell type the lowest 10th percentile of tile means, and the margin 0.0543.

**Physics / equation:** Measured from the reference, never typed: the margin is the 90th percentile (SHAPE_Q), over the 53 cell types with at least 2 healthy samples, of the within-type spread (max − min) of the tile 10th percentile. The physics reason: a sample whose genome has let go more than any healthy sample of its type is not in a healthy state the line can be read from.

**How the chain uses it:** `stage_q2_selfref.shape_reference` (stamp checked) and `shape_check`: a sample is read only if tile_p10 ≥ min_p10 − margin; `stage_v4.stage_q2` refuses otherwise, with the five-part explanation. Not applied to a mixed specimen.

**Reproduce it:** `stage_q2_selfref.build_shape_reference(records)` on the healthy counts.

**Status and open parts:** Measured; a cell type with no healthy reference is not read.

#### Input: Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json

**Kind:** calculated

**Register:** none (genome table)

**What it is:** Each autosome's range of CpG indices in the hg19 CpG dictionary (wgbstools init_genome hg19).

**Physics / equation:** Calculated from the genome sequence: no physics. The counter's tiles are 2,000 consecutive CpG indices, so the tile grid depends only on this numbering; a partial last tile is dropped.

**How the chain uses it:** `stage_q2_selfref.hg19_tiles` with TILE_SOURCE "dictionary" (`run_sample.py --dev-tiles dictionary`, development; the default uses the table HG19_CPGS in the code); Stage Q0 check Q0.2 refuses a `.pat` line outside its chromosome's range.

**Reproduce it:** `wgbstools init_genome hg19`, then read the ranges.

**Status and open parts:** Calculated; nothing open.

#### Input: Runtime Matrices/IAM_A_Positions/iama_positions_v*.json (v1, v2)

**Kind:** development

**Register:** none (retired)

**What it is:** The retired position form of IAM-A (canon P_neutrophil_IAM_A).

**Physics / equation:** None now.

**How the chain uses it:** Read only by the retired Stage Q (`stage_q_iam_a.py`); `run_sample.py::_versions` records v2's hash in every bundle.

**Reproduce it:** None.

**Status and open parts:** Retired, kept so earlier records reproduce.

#### Input: Runtime Matrices/Coupling/coupling_reference_v1.json

**Kind:** measured

**Register:** CP1

**What it is:** Each cell type's healthy nearest-neighbor coupling and the Normal band of coupling-A.

**Physics / equation:** Measured from IAM-Atlas v2.1's healthy sequencing inputs. The quantity: on held molecules, the adjacent pairs of unmethylated calls over what random placement gives each molecule from its own length L and its own number k of unmethylated calls, Σ k(k − 1)/L. Its null, 1 for independent losses, is counted exactly (statistics). Healthy values rise with differentiation: naive T 1.29 to 1.33, central memory 1.43 to 1.44, effector 1.52 to 1.58. IAM's Law is not yet in it; the candidate (a lead) is that the restore crew works along a stretch, so coupling = 1 + (mean skip length − 1) × (share of losses from skips).

**How the chain uses it:** `stage_coupling.reference` loads it; `stage_coupling.read` gives coupling-A = coupling / healthy coupling of the sample's cell type, with floor 0 and ceiling coupling_max / healthy, both counted, and the state. Development, flagged.

**Reproduce it:** Count the healthy inputs with `run_sample.py --dev-count-only` and take each type's coupling.

**Status and open parts:** Development; IAM's Law not yet in it (CP1). If each type's level carries a history term, the healthy reference is to be split by history.

#### Input: Runtime Matrices/Clock_Sites/epitoc2_sites.csv

**Kind:** measured

**Register:** C3; PHY-094

**What it is:** The 163 CpGs of the epiTOC2 mitotic clock (Teschendorff, Genome Medicine 2020), the sites the count reads with 25 CpGs either side.

**Physics / equation:** A published site list (Polycomb-target island CpGs unmethylated in fetal tissue that gain methylation with stem-cell divisions); epiTOC2's own model is β_t = 1 − (1 − δ/2)^t (1 − β_0). Measured, not derived. In IAM's terms these are islands where the shield against the writer fails, so the writer's one-step errors are kept there (C3, lead).

**How the chain uses it:** `run_sample.py --dev-count-only --dev-count-sites epitoc2`: the held error and the stray-mark (gain) side counted at these sites only, by Stage Q2's own rules (`stage_q2_selfref.count_pat_sites`). Development.

**Reproduce it:** Download the epiTOC2 site list from the paper's supplement.

**Status and open parts:** Measured site list; the tick per round from the writer's error is open (C3).

#### Input: Runtime Matrices/Info_Sphere/layout_v5.npz

**Kind:** calculated

**Register:** SP1

**What it is:** The information sphere's frozen layout: every site's id, its point on the globe, the cell type it marks most, the sign and size of that departure and its margin over the next type, the cell list and families, and the atlas's SHA-256.

**Physics / equation:** Established mathematics on the atlas's levels: u = (√b₁…√b_S, √(1−b₁)…√(1−b_S))/√S, a unit vector; distances are the Fisher–Rao (Bhattacharyya) angle arccos(u_i · u_j); construction v5 removes each site's own level (a flat profile at its mean) before projecting onto the three leading directions. IAM's Law is not yet in it.

**How the chain uses it:** `stage_info_sphere.load_layout` (LAYOUT), `groups_of` (territories: each family's clearly marking sites, margin ≥ CLEAR = 0.10, by direction) and `sample_reading` (each territory's mean change t_g in the sample's own spread). Built by `run_sample.py --build info-sphere` (`stage_info_sphere.save_layout`).

**Reproduce it:** `python3 MethylPhys_Interface/run_sample.py --build info-sphere --build-construction v5 --build-input MATRIX --build-out LAYOUT --bundle B.json`.

**Status and open parts:** Development: construction test 7 passed; sample test 1 met on distance, not on territories (0 of 27); the null test ATL-028 is pending (SP1).

### Met-A's and the array stages' files (Sections 2 to 5)

#### Input: Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json

**Kind:** measured

**Register:** M5; M4

**What it is:** Met-A's commissioned reference: the 6,000 EPIC identity sites of the healthy neutrophil and the healthy reference H_ref (key `floor`, the middle mark) = 0.33026 bits.

**Physics / equation:** Measured: mean H(β) over the identity sites on 6 purified healthy neutrophil arrays (GSE110554; GSE167998 re-deposits the same arrays, each physical array counted once), calibrated by Stage 1. Met-A = mean_i H(β_i) / H_ref. It stays measured because identity sites are cell-type marks, not fully held sites; the law's derived height applies at fully held sites only (register M2).

**How the chain uses it:** `stage_m_met_a._floors` (FLOORS) and `stage_m_met_a.read`; `conductor_v3.stage_m_isolated`; the site list also serves `run_sample.py --dev-count-sites` on sequencing.

**Reproduce it:** `cd Biological_Physics/MethylPhys/chain` && `python3 stage_m_met_a.py BETAS.csv neutrophils` on a purified array's Stage 1 values (two columns, cpg_id, beta).

**Status and open parts:** Measured; commissioned on isolated neutrophils, EPIC v1. Being re-commissioned with IAM-Atlas v2.1 as the only healthy reference.

#### Input: Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv

**Kind:** measured

**Register:** M5

**What it is:** The held-out precision of Met-A's reference: each of the 6 arrays read with the sites re-chosen on the other 5 (0.983 to 1.045, SD 0.020).

**Physics / equation:** Measured; a leave-one-out check of the reference, printed with every isolated reading.

**How the chain uses it:** `stage_m_met_a.precision` (LOO).

**Reproduce it:** Re-choose the sites on 5 arrays by the site rule and read the sixth.

**Status and open parts:** Measured.

#### Input: Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json

**Kind:** measured

**Register:** M4; C1

**What it is:** The 6,000 neutrophil identity sites in genomic order, the healthy mean and shrunk SD of H at each, and the C-Score's block length (10) and healthy baseline 1.0103.

**Physics / equation:** Measured on the purified neutrophil arrays. The departure map z_i = (H(β_i) − H(ref_i)) / s_i is built on it; the C-Score's baseline is statistics (register C1).

**How the chain uses it:** `conductor_v3.ref`; the departure map of Stage M and the C-Score of Stage MC.

**Reproduce it:** Recompute the per-site H means and SDs on the 6 arrays.

**Status and open parts:** Measured.

#### Input: Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json

**Kind:** measured

**Register:** none (standard method)

**What it is:** Stage A's mix matrix: 963 marker sites and 8 blood groups' means at each, and the 8 groups' healthy means at the 6,000 identity sites.

**Physics / equation:** Measured from 91 purified-cell EPIC arrays (GSE110554, GSE167998) through Stage 1. Standard method (non-negative least squares); no IAM physics. The expectation e_i = Σ_g f_g μ_g,i makes Met-A in whole blood independent of the person's mix.

**How the chain uses it:** `conductor_v3._bc`, `stage_a_composition`, `stage_m_blood`.

**Reproduce it:** `chain_tests/blood_comp.py`; the rebuild `metaa_rebuild` reproduced every group mean to 5 × 10⁻⁶.

**Status and open parts:** Measured; standard method.

#### Input: Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json and noise_gate_EPIC_v1.json

**Kind:** calibrated

**Register:** M3 (the array's own response)

**What it is:** The array's own noise: 48,528 sites every purified blood group holds fixed (each group's mean ≤ 0.03 or ≥ 0.97 and SD ≤ 0.02), and the gate N_max = 0.149.

**Physics / equation:** Calibrated: N is the mean entropy at sites where no blood cell has anything to say, so it measures the instrument, not the cells. N_max is the top of the reference arrays' range.

**How the chain uses it:** `conductor_v3.noise_sites`, `noise_gate`, `noise_index`.

**Reproduce it:** Apply the site rule to the 91 purified arrays; N_max from the reference arrays.

**Status and open parts:** Calibrated; stays measured per array.

#### Input: Runtime Matrices/Intake/intake_thresholds_v*.json (v1, v2)

**Kind:** calibrated

**Register:** none (standard method)

**What it is:** Stage 0's limits: control-probe, detection p-value, bead count and call rate thresholds (v1), and the three control-probe limits of the QC hand-off (v2: bisulfite conversion, hybridization, extension).

**Physics / equation:** Instrument limits, no IAM physics: each one states its measured value, its limit and the source of the limit.

**How the chain uses it:** `stage_0_intake._intake_thresholds` loads v1 at import (Stage 0 refuses to start without it); the control-probe gates read v2. `run_sample.py::_versions` records v1's hash.

**Reproduce it:** Read the file; each limit's source is printed with every stop (`run_sample._why`).

**Status and open parts:** Calibrated; a bisulfite limit marked provisional in v2 records a failure without stopping the array.

#### Input: Runtime Matrices/Gauge/cell_gauge_marks_v1.json

**Kind:** derived

**Register:** A1; M1

**What it is:** The gauge's marks: the floor 1/(1 + e^M), the Normal band around healthy 1, and saturation (every identity site fully scrambled).

**Physics / equation:** Floor and ceiling derived (A1, M1): floor H(1/(1 + e^M)) = 2.5e-08 bits; ceiling one bit per site at β = 0.5. Healthy = 1 is measured (each reading's own reference).

**How the chain uses it:** `report_figs_v4.gauge_png` draws every mark from it; nothing is typed in the drawing code.

**Reproduce it:** Compute the floor from M (canon H_min_cell).

**Status and open parts:** Derived; already in the chain.

#### Input: Runtime Matrices/Met_A_Widths/donor_widths_v1.json

**Kind:** measured

**Register:** A9 (the same idea for Met-A); M4

**What it is:** Each cell's healthy width on Met-A, measured on IAM-Atlas v2.1's own donors read by the one command.

**Physics / equation:** Measured by nature: Normal is 1 ± k widths with k = t(0.975, n − 1) × √(1 + 1/n) for n donors, the prediction interval for one new reading.

**How the chain uses it:** `stage_metaa_atlas.width_state` (arrays).

**Reproduce it:** Read the atlas donors through `run_sample.py` and take each cell's SD about 1.

**Status and open parts:** Measured; replaces the 0.95 to 1.05 tolerance as each cell's reading carries it.

#### Input: Runtime Matrices/Atlas_V21/sequencing_term_v1.json

**Kind:** calibrated

**Register:** M3; S1 (the bridge between scales)

**What it is:** The atlas's measured conversion of an array-scale level to sequencing units for the published sequencing atlas: sequencing = −0.0902 + 1.1026 × atlas level, clipped to 0 to 1.

**Physics / equation:** Calibrated: arrays read unmarked sites near 0.055 where sequencing reads 0 (each array's offset, M3), so a sequencing sample is compared with the atlas only after this term.

**How the chain uses it:** `stage_metaa_atlas.seq_units`: Met-A for a sorted cell, for each cell in a mix, and the territories, on sequencing.

**Reproduce it:** Fit atlas level against the published sequencing atlas at the shared sites (the atlas's own build record).

**Status and open parts:** Calibrated.

#### Input: Runtime Matrices/CScore_Limits/cell_factors_v1.json and platform_term_450k.json

**Kind:** measured

**Register:** C1

**What it is:** The C-Score's limits: each cell's factor over the derived limit (neutrophils 1.0314 from 57 donors in 7 laboratories; monocytes 1.0062 from 40 donors in 6), and the 450K platform term.

**Physics / equation:** The derived limit is statistics, not IAM's Law: for scattered departures the block statistic is 1 with a band from independence (0.894 to 1.111 over 6,000 sites in blocks of 10). Healthy neutrophils carry a little real clustering in every laboratory, which the measured factor carries.

**How the chain uses it:** `stage_metaa_atlas._cs_cell_factor` and `cscore_limits`.

**Reproduce it:** One-sided 95 % prediction limit of C over the derived limit on the atlas's donors.

**Status and open parts:** Measured; statistics.

### The IAM Sequencing Atlas's files (Section 21) and the sequenced sample's references

#### Input: Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet (and .json)

**Kind:** measured

**Register:** M4; S1; Part I (Met-A sequencing)

**What it is:** Met-A's healthy reference on sequencing: per cell and identity site (cell, probe, cpg), the healthy level and the two sealed halves' values.

**Physics / equation:** Counted, not fitted, from the 153 healthy sequencing inputs behind IAM-Atlas v2.1 (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`), each with its sealed donor half. Measured: the healthy level at a cell's identity sites is the atlas's per-type level. At fully held sites the law predicts 1 − ε0 in sequencing units (register S1, derived as A6); identity sites are not all fully held, so the reference stays the measured level.

**How the chain uses it:** `stage_seq_atlas.metaa_reference_of` (METAA_REF), `noise_check`, `noise_at_identity`, `beta_vs_pat`, `identity_side_table`; `run_sample.py --dev-seq-ref auto`, on by default for sequencing input ('the IAM Sequencing Atlas as ruler').

**Reproduce it:** `run_sample.py --build seq-metaa-ref` on the atlas pieces (the build named in the file's .json).

**Status and open parts:** Measured. Using the derived held level where a reading is at fully held sites, recorded beside the atlas level, is a chain item still to check.

#### Input: Runtime Matrices/Seq_Atlas/deconv_reference_v1.parquet and deconv_reference_mk_dev.parquet (and .json)

**Kind:** measured

**Register:** D1

**What it is:** The IAM-Deconvolver's reference for sequenced samples: per site and cell, μ and sd (spread 'full'), without (v1) and with (mk_dev) the provisional megakaryocyte column.

**Physics / equation:** Counted, not fitted, from the 153 healthy sequencing inputs behind IAM-Atlas v2.1 (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`), each with its sealed donor half. Standard method (weighted non-negative least squares); not a law quantity.

**How the chain uses it:** `stage_seq_atlas.deconv_reference_of` (DECONV_REF) as `stage_v4.stage_d21` takes it; `run_sample.py` reads with `deconv_reference_mk_dev.parquet` by default (`--dev-deconv-ref`), or a rebuilt atlas's `deconv_reference_v1.parquet` with `--dev-seq-atlas-dir`.

**Reproduce it:** `run_sample.py --build seq-deconv-ref` (`--build-with-mk` keeps the megakaryocyte column).

**Status and open parts:** Measured; commissioned on arrays for four blood cells (share error 0.016); sequencing shares are development.

#### Input: Runtime Matrices/Seq_Atlas/array_sites/ (meth, reads, donor_sd and mu _part1-3.parquet, array_sites.json)

**Kind:** measured

**Register:** M4; T2

**What it is:** The IAM Sequencing Atlas at IAM-Atlas v2.1's sites, in three parts per column: methylated reads, reads, the donors' spread, and the pooled level.

**Physics / equation:** Counted, not fitted, from the 153 healthy sequencing inputs behind IAM-Atlas v2.1 (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`), each with its sealed donor half. Levels are uint16 × 10,000, 65535 blank, nothing filled; the pooled level is meth / reads with its exact 95 % range.

**How the chain uses it:** `stage_seq_atlas` (ARRAY_COLS): the in-mix readings on sequencing, `domain_disorder` (each cell's level, disorder and donor spread at the domains), and `sphere_columns`, which turns these counts into the sphere columns below.

**Reproduce it:** `run_sample.py --build seq-array-cols` on the atlas pieces.

**Status and open parts:** Measured. The domains' healthy level is derivable as IAM-A's held error at each site's loose share (T2, a lead); today it is read against the atlas.

#### Input: Runtime Matrices/Seq_Atlas/sphere_cols/ (mu, mu_lo, mu_hi, mu_se, donor_sd _part1-3.parquet, sites.parquet, sphere_cols.json, sphere_cols_build.json)

**Kind:** calculated

**Register:** SP1; M4

**What it is:** The sky sphere's input in IAM-Atlas v2.1's layout: each cell's pooled level, the 95 % range of its average, its standard error and the donors' spread.

**Physics / equation:** Calculated from the array_sites counts (`range_of_average`); nothing fitted. A site measured by fewer of a cell's donors is not told apart.

**How the chain uses it:** `stage_seq_atlas.bridge` (SPHERE_DIR) and the report globe; `sphere_columns` writes it.

**Reproduce it:** `run_sample.py --build seq-sphere-cols`.

**Status and open parts:** Calculated, development.

#### Input: Runtime Matrices/Seq_Atlas/noise_cols/

**Kind:** calculated

**Register:** none (noise maps, statistics)

**What it is:** The IAM Sequencing Atlas's two noise maps at IAM-Atlas v2.1's sites: halfA, halfB (the sealed donor halves) and readA, readB (two random halves of every donor's reads, seed 20261029). The chain folder holds the README only; the columns are built with `--build seq-noise-cols`.

**Physics / equation:** Map 1 = donor half A − half B (donor difference plus counting); map 2 = read half A − read half B (counting only); their mean squares' difference is the healthy donor-to-donor part, measured.

**How the chain uses it:** `stage_seq_atlas.noise_maps`, `noise_check`, `noise_minkowski`, `noise_null` (NOISE_DIR).

**Reproduce it:** `run_sample.py --build seq-noise-cols`.

**Status and open parts:** Calculated, development.

#### Input: Runtime Matrices/Seq_Atlas/held_reference_v1.parquet (and .json)

**Kind:** measured

**Register:** A6; S1; A8

**What it is:** Each healthy cell's held error tile by tile (2,000-CpG tiles), as counts: lone errors and their chances, all donors and each sealed half.

**Physics / equation:** Counted, not fitted, from the 153 healthy sequencing inputs behind IAM-Atlas v2.1 (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`), each with its sealed donor half. Counts by Stage Q2's own rules (`count_pat_tile_errors`, even and odd lines). Measured: the per-type rates carry each type's own level (A8), which is not derived yet.

**How the chain uses it:** `stage_seq_atlas.iama_in_mix` (HELD_REF): IAM-A for a mixed sample, A = Σ_tiles held errors / Σ_tiles (opportunities × Σ_cells share × the cell's healthy held-error rate there); healthy reads 1. Built by `held_reference`.

**Reproduce it:** `run_sample.py --dev-count-only --dev-count-tiles` on each input, then `run_sample.py --build seq-held` (`stage_seq_atlas.held_reference(samples, out)`).

**Status and open parts:** Measured, development. The one-line reading of Section 9 needs no atlas; this reference serves the reading in a mix.

#### Input: Runtime Matrices/Seq_Atlas/floor_sites_v1.parquet and input_floors_v1.parquet (and .json)

**Kind:** measured

**Register:** A15; A17; T1; S2

**What it is:** The library floor: the sites where every cell of the IAM Sequencing Atlas has at least 20 reads and a pooled level ≤ 0.05 (unmarked) or ≥ 0.95 (marked), and every atlas input's own floor there.

**Physics / equation:** Measured. An opposite call at these sites (methylated at an unmarked site, unmethylated at a marked one) is counted as the library's own. Physics on each side: at unmarked sites it is the writer's one selection step, 1/(1 + D) with D 47.4 to 80 (0.0123 to 0.0207, A15), plus the meter (conv_fail + tc_err, A17); at marked sites a held loss also reads as an opposite call, so the marked side carries the held error.

**How the chain uses it:** `stage_seq_atlas.library_floor`, `floor_by_position`, `identity_side_table` (FLOOR_SITES); `input_floors`, `floors_of`, `half_floor` (INPUT_FLOORS). Development, flagged.

**Reproduce it:** `run_sample.py --build seq-floor`, then `--build seq-input-floors`.

**Status and open parts:** Measured; the common and class library floors failed their own tests and stay off by default.

#### Input: Runtime Matrices/Seq_Atlas/metaa_half_counts/, metaa_donor_counts/, metaa_inner_reference/ and terr_half_counts/ (*_v1.parquet, metaa_inner_reference_t3)

**Kind:** measured

**Register:** M4; S2; T1

**What it is:** Counts behind Met-A and the territories on sequencing: each cell's methylated and total reads at its identity sites per sealed half (metaa_half_counts) and per donor (metaa_donor_counts), the same without the molecules' end calls (metaa_inner_reference; t3 trims three), and the island probes' reads per half (terr_half_counts).

**Physics / equation:** Counted, not fitted, from the 153 healthy sequencing inputs behind IAM-Atlas v2.1 (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`), each with its sealed donor half. Counts, never ratios. The end calls are the library's own chemistry, the end meter calibrated per library (S2).

**How the chain uses it:** `stage_seq_atlas.metaa_half_counts` and `metaa_reference_of` (METAA_COUNTS_DIR), `metaa_donors_of` (METAA_DONORS_DIR), `metaa_inner_of` and `ref_compare` (METAA_INNER), `terr_counts_of` (TERR_DIR; the counted island expectation, on by default).

**Reproduce it:** `run_sample.py --build seq-metaa-counts`, `--build seq-metaa-donors`, `--build seq-metaa-inner`, `--build seq-terr-halves`.

**Status and open parts:** Measured, development.

#### Input: Runtime Matrices/Provisional_References/megakaryocytes_v*.json and .parquet (v1, v2)

**Kind:** development

**Register:** D1

**What it is:** A provisional megakaryocyte column for the Deconvolver, beside IAM-Atlas v2.1 (which has none).

**Physics / equation:** Measured from megakaryocyte sequencing reads: version 2 takes the level from the shrunk proportion (m + 1/2)/(n + 1), carries the counting spread through the atlas's inverse transfer, and keeps sites with at least 5 reads. Not a law quantity.

**How the chain uses it:** `stage_v4.stage_d21_provisional` (`stage_provisional.with_provisional`), on by default as `--dev-provisional megakaryocytes`; `run_sample._id_collides` reads the column names so a typed id cannot erase a cell's name.

**Reproduce it:** `stage_provisional.column_v2`.

**Status and open parts:** Development; recorded beside the atlas's shares, never in place of them.

#### Input: doors/data/DEV_ATLAS_V21_01/transfer_v21_smooth.json (outside Runtime Matrices)

**Kind:** calibrated

**Register:** M2; M3

**What it is:** The smooth transfer curve between sequencing levels and the array scale with which IAM-Atlas v2.1 was built.

**Physics / equation:** Calibrated: measured once on cells both platforms measured. Through it, the law's held level 1 − ε0 = 0.95546 maps to about 0.9545 array units (the bounded table `transfer_bounded_v21.json`), 0.0044 below saturation 0.959.

**How the chain uses it:** `stage_m_met_a.derived_baseline` (TRANSFER): the PCHIP curve through its knots, inverted on a grid of 200,001 points. Development. (The Deconvolver's laboratory spread comes from `transfer_v21_final.json` in the same folder, Section 7.)

**Reproduce it:** Interpolate its knots.

**Status and open parts:** Calibrated; it lives in a test-record folder and belongs in `Runtime Matrices`.

### Every other file under Runtime Matrices

#### Input: Runtime Matrices: every other file (no IAM quantity rides on it)

**Kind:** development

**Register:** none

**What it is:** Atlas_V21/sex_chromosome_sites_v21.json, sky_order_v21.npz, territories_seq_band_v1.json; Celltype_Marker/ (README.md, detection_panel_v3.json, iamatlas_celltype_markers_v0_2.json, iamatlas_celltype_markers_v0_3_TRIAL.json, percell_exclusivity_v0.json, twin_family_thresholds_v1.json); Development/ (dev_foreign_placenta_EPIC_v1.json, dev_metA_450K_neutrophils_v0.json, dev_selftare_450K_v0.json); Directional Panel/ (bidirectional_decomposition.py, directional_panels_v1_0.json); Met_A_Floors/README.md, metA_floors_v1_2_ALLCELLS_development.json, neutrophil_reference_v1_1.json; Patient_CMB/ (README.md, build_healpix_mapping.py, iamatlas_cpg_to_healpix_nside128.npz, presence_floors_v1.json); Report/report_text.json; Stage_T/atlas_ruler_v1.json, selftare_typeII_EPIC_v1.json; trace_detection_panel_v1.json; the README files of the Seq_Atlas folders.

**Physics / equation:** None of these carries a number derived from IAM's Law. They are site lists, display orders, report text, retired references and development panels, read by other modules (`reference_globes.py`, `report_v4.py`, `stage_metaa_atlas.py`, `dev_stages.py`, `stage_t_selftare.py` and others), each described in the Decoder entry of the function that loads it.

**How the chain uses it:** Recorded by size and hash in every bundle (`run_sample.py::_versions`). `metA_floors_v1_2_ALLCELLS_development.json` is named by `stage_m_met_a.py` as kept for development and not read there.

**Reproduce it:** Each file's builder is named in its own Decoder entry.

**Status and open parts:** Each is described where it is loaded; none needs a derivation.

## The Deconvolver decides what is in a sample, in every instrument

**The rule.** A specimen's words (whole blood, plasma, tissue, "sorted CD4 T cells") choose only the route a sample takes: array or sequencing,
and which kind of specimen. What is in the sample, and which atlas reference each part is read against, come only from the IAM-Deconvolver's
measured shares (Stage D21), which run first on every input that has values, a sequencing file included. Every cell found at 1 % or more is
read against its own atlas reference (Met-A for each cell). A single-cell reading (the sorted-cell Met-A, IAM-A, coupling-A) reads the cell
the Deconvolver found most of. A label the Deconvolver disagrees with is recorded in the bundle (`label_check`), never obeyed. A cell set by
hand (`--dev-metaa-atlas`, `--seq-cell`) is recorded as the operator's (`operator_cell`), not as a reading of the sample.

**Why it is written here.** From 2026-10-22 to 10-28 three paths in `run_sample.py` took the cell from the specimen's words, and `run_series.py`
let a sample list's `atlas_cell` column do the same. The cross-laboratory tests on sorted cells were then scored by their labels: one healthy
"CD4" sample the Deconvolver read as 64 % monocytes was scored as CD4, and outside "sorted" samples, which hold 24 to 80 % of the labeled cell,
were read as one cell. Days of results compared labels, not samples. The physics reads the molecules in the tube; only the Deconvolver says
what those molecules came from. `CANON/chain_gate.py` rule 23b-h now refuses chain code that sets a cell or a mix from the specimen's words,
and any scorer that reads the label's cell without the chain's `label_check`.

## The atlases are the only healthy reference, in every instrument

Two questions are kept apart. What does healthy look like? That comes only from the atlases, built from cells of healthy donors so that healthy is certain; never from other samples in the same run, laboratory or study, which are only labeled healthy and can carry the very signal being looked for. What did the instrument do to this measurement? That is calibration, a different job, answered by calibrators whose answer is already known, never by the sample's neighbors. Clinical laboratories calibrate this way: reference DNA of known methylation (fully methylated, unmethylated and mixes of the two) is run beside each sample, and it tells what the instrument did without saying what healthy is. Cosmology keeps the same separation: the sky model comes from the theory, and each detector is calibrated on sources whose answer is already known, never on the signal being measured. For public data, the atlas's donors play the calibrators' part: the same healthy cell types, measured in many laboratories, show what laboratories do to a measurement, and each cell's Normal width is measured from them. And because nothing is borrowed from the sample's neighbors, nothing the instrument does can cancel unseen: every reading compares like with like, against a reference that is certainly healthy.

**The rule.** Every comparison a reading makes with healthy is made against IAM-Atlas v2.1 (arrays) or the IAM Sequencing Atlas (sequencing),
the references this work built and checked for exactly that purpose, in the mix of cells the Deconvolver measured in the sample. Nothing else is
a healthy reference: not a study's samples labeled healthy, normal or control, not the other arrays of the same run or slide, not the other
samples of the same laboratory, and not a patient's own non-tumor tissue (labeled "normal" in a public series, it is tissue from an organ that
grew a cancer). Anything that differs between laboratories, kits or runs is measured from the sample's own reads, at the sites being read (for
example its false-call floor at the islands, `--dev-floor-class`), never borrowed from other people's samples. A person compared with their own
earlier sample is self-reference, which is a separate question, and both readings are still read against the atlas.

**Why it is written here.** A tare (a reading divided by the median of other samples) entered the chain early, for arrays, to remove a run's
offset; Met-A on neutrophils, the C-Score and Group-A were commissioned with it. On sequencing it spread: a same-laboratory tare, a test of
healthy CD4 T cells tared on the other donors, and a tumor read against the same person's "normal" tissue. Each looked as though it worked, because
a sample divided by its neighbors comes out near 1 by construction: the same-laboratory tare put the medians at 1.0001 and 0.9889 while only 38 to
52 % of the people read inside Normal. A tare against unchecked samples hides the very differences the instruments exist to see, and it imports
whatever is wrong with those samples. The author's rule, stated when this was found: healthy for comparisons always comes from the atlases,
because that is what they were created for, and no instrument is tared with anything that is not certainly healthy.

**The atlas goes through the same processing as the sample.** Because a comparison is fair only when both sides have passed through the same instrument and the same pipeline. Cosmologists work this way: they do not clean the instrument out of the data before comparing it with theory; they pass the theory, as simulated skies, through a model of the same instrument and the same processing (beams, filtering, masks, map-making), compare the data with theory processed identically, and carry the reference's own uncertainty (noise maps, half-mission maps, null tests) into every comparison. The atlas is used the same way. It is built from public data of purified healthy cells, and it serves as the model of healthy with its own measured uncertainty, compared to a sample only after both have taken the same path. On arrays, the atlas's donor arrays were calibrated from their raw files by the chain's own first stage, the stage every sample goes through, and each array's own measured response is then applied to the atlas before the comparison (the healthy prediction). On sequencing, the calls at the two ends of a DNA molecule carry the laboratory's library chemistry, a bias long known in bisulfite sequencing; so the IAM Sequencing Atlas is being counted again from its donors' molecules exactly as a sample is read, and tested on libraries from several laboratories. In the chain: Stage 1 for both on arrays, then the healthy prediction (`stage_metaa_atlas.predicted_healthy`, `cell_prediction`, `territory_prediction`); on sequencing, `stage_seq_values.trim_pat` and `pat_to_lbeta` for sample and donors alike (development, `--dev-seq-trim-ends`, `--dev-seq-atlas-dir`).

**How the code holds it.**
- `CANON/chain_gate.py` rule 23b-i refuses any line added to chain code or a test's code that builds a reference from other samples (a tare
  call, a division by the median of the others or of a healthy, normal or control group, a tumor divided by a normal); its control cases C22 to
  C25 are lines written in the form the scorers used where this happened (not copies of those lines), each refused; C26, a reading
  against the atlas's own prediction, must be accepted.
- `CANON/kiss_check.py` rules e and f refuse a test written before it runs that lacks a `**Reference:**` line naming an atlas, names any other
  reference, or has no bar asked of each person or sample (a median alone can be met while most people are outside Normal).
- `CANON/commissioning_check.py` refuses a commissioning from this date on that is not on the atlases alone; each older commissioning states its
  reference, and those made with a same-run tare (Met-A on neutrophils, the C-Score, Group-A) carry "being re-commissioned with IAM-Atlas v2.1 as
  the only healthy reference" in the catalogue until each passes again: its synthetic over the real data's range, a rehearsal on spent data per
  person, and a fresh test per person, all against the atlas (plan 60.29; sequencing, plan 60.30). **Lock 1, the reference door** (`reference_door.py`): every run checks the atlases against their checksums before any reading and records in its bundle the atlas it read and every tare applied. **The daily reference line:** from 2026-11-07 every result in a day log names its healthy reference, an atlas or "none (not a reading against healthy)"; the day-log check refuses a result that names neither.

### `reference_door.py::_sha`

The SHA-256 checksum of one file, read in 4 MB blocks.

*Code:* `reference_door.py` line 20 (5 lines) · arguments: p · calls: not called on the traced runs

### `reference_door.py::_seq_files`

Every reference file under chain/Runtime Matrices/Seq_Atlas (parquet and json), in a fixed order.

*Code:* `reference_door.py` line 26 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `SEQ`

### `reference_door.py::build`

Writes reference_manifest.json: each IAM Sequencing Atlas file the chain reads, with its size and checksum. Re-run only when such a file is rebuilt, and committed with it.

*Code:* `reference_door.py` line 33 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `MANIFEST`, `SEQ`

### `reference_door.py::check`

Lock 1 of the anti-drift locks. Before any reading, each IAM-Atlas v2.1 file is checked against compact_manifest.json and each IAM Sequencing Atlas file against reference_manifest.json; a file that differs, or a file in Seq_Atlas the manifest does not list, stops the run (REFERENCE_REFUSED, exit 7). A reading made against an altered reference is not a reading against the atlas.

*Code:* `reference_door.py` line 38 (22 lines) · arguments: none · calls: not called on the traced runs · constants read: `MANIFEST`, `SEQ`, `V21` · files named: `compact_manifest.json`, `reference_manifest.json`

### `reference_door.py::record`

The bundle's reference block: the atlases checked, whether they passed, and every tare a reading in the bundle applied. The array instruments commissioned before the atlas-only rule still carry a same-run healthy tare; it is named here until they are re-commissioned on the atlas, so no tare can sit in a reading unrecorded. A limit term calibrated on outside healthy samples (the 450K C-Score platform term) is named the same way, under calibrated_terms.


**What each sequencing reading is referenced to (no reading is tared with same-run healthy samples).** A laboratory rarely has three other healthy
samples of the same cell run the same way, and samples merely labeled healthy cannot be trusted to be, so no sequencing reading uses them.
Each reading is referenced to one of three things, and every laboratory or library difference is measured from the sample's own reads:

| Reading | Healthy reference | Laboratory and library part |
|---|---|---|
| IAM-A | the sample itself: its own line, held error against its own loose share | none needed (self-referenced) |
| CP1 run lengths, islands meter, clock sites, copy timing, TERR/GLB line locality | the sample against itself | none needed |
| IAM-A C-Score, whole-genome sequencing globe | each region against the sample's own held error | none needed |
| Deconvolver | the IAM Sequencing Atlas (identity: which cells, what shares) | the sample's own reads |
| Coupling-A | the atlas: the type's healthy coupling | the sample's own reads |
| Met-A sequencing (sorted cell, whole sample, each cell in the mix), Met-A-side C-Score, territories | the IAM Sequencing Atlas in the sample's own Deconvolver mix; for an atlas library, the other sealed donor half | the islands meter (the sample's own false-mark rate), the library floor at the sites read, the expectation at the sample's own read count |
| IAM's Law Sphere | the law's expected point, with the sample's own line and the atlas angle beside it | the counting floor at the sample's own read counts |

The bundle carries this table as `reference_rule`, and each atlas-referenced reading's state says "atlas ruler; laboratory part from this sample's own
reads (no same-run healthy tare)". The word "untared" in older bundles, and the optional same-run tare in the code, are left from the array method
this rule retired; on sequencing neither applies, and "untared" never meant a missing step. Group-A alone compares groups.

### The three Met-A readings on a sequenced sample, and why they can differ on one library

Met-A asks whether a cell's methylation record has become more disordered than a healthy cell of the same type holds it (normalized methylation entropy against a cell-type-matched healthy reference). On a sequenced sample the chain reads it three ways. Each uses a different set of sites and a different healthy expectation, so on one library the three need not agree.

| Reading | Sites read | Healthy expectation | Question it answers |
|---|---|---|---|
| Sorted-cell Met-A | The identity sites of the library's own cell type (Met-A's site rule on IAM-Atlas v2.1) | That type's healthy people in the IAM Sequencing Atlas: each person's expected disorder at the sample's own read depth, averaged over the people, so the differences between healthy people enter as they are | Has this cell type's identity record moved? |
| Met-A for each cell in the mix | Each found cell's identity sites | The atlas mixed in the sample's own cell shares (IAM-Deconvolver) at those sites | In a mixed sample, which cell has moved? |
| Whole-sample Met-A | Every IAM-Atlas v2.1 site, each weighted by how firmly a healthy sample of this mix holds it, \|0.5 - e\| | The atlas mixed in the sample's own cell shares, each type's level pooled over its healthy people | Has the sample as a whole departed? |

**Why "whole sample" on a sorted library.** The name comes from blood, where a sample is a mix of cells. On a sorted library the same route still runs, with the IAM-Deconvolver's shares: most of the share goes to the library's own type and a few percent to close relatives (on a sorted heart-muscle library, for example, a few percent to lung epithelium and heart fibroblasts). Sorted-cell Met-A uses the library's own type only.

**Status.** Whole-sample Met-A is development. On healthy atlas libraries never used to build the reference it reads below sorted-cell Met-A in most types; two causes are being tested (the pooled healthy expectation, and the mix weights on a sorted library), and the measurements are in the development log.

Source: `stage_metaa_atlas.METAA_SEQ_READINGS` (the SOP prints the same text).

### The Deconvolver and Met-A's tare on sorted sequenced samples

#### How the sequencing Deconvolver works (Stage D21 on the IAM Sequencing Atlas)

- **The rule:** Splits a sample into the 57 healthy cell types of the IAM Sequencing Atlas. For every marker CpG it solves the sample's level as a non-negative mix of the types' healthy levels (non-negative least squares), each site weighted by the atlas's person-to-person spread at that site times the current shares, plus a laboratory term; three passes, each re-weighting from the last shares. Shares are normalized to 1. 95 % intervals from 200 resamplings of the marker sites, each also drawing that person's cells from the atlas's spread (seed 2026).
- **Why it is needed:** Every reading that compares a sample with its healthy expectation needs to know which healthy cells it is made of. In whole blood, plasma and tissue the Deconvolver decides what is there: no cell is assumed absent (the rule of 2026-11-07).
- **Physics:** Not a step of IAM's Law: linear unmixing (a sample's level at a site is the share-weighted sum of its cells' levels). The cosmology analogue is component separation of foregrounds (credited: Tegmark, de Oliveira-Costa & Hamilton 2003, Phys. Rev. D 68:123523).
- **Where the data comes from:** deconv_reference_v1.parquet (chain Runtime Matrices/Seq_Atlas): for each of 5,699 marker CpGs (named by array probe; mapped to the sequencing CpG index by atlas/v2.1/sites_sequencing_key.parquet) each type's healthy level (|mu) and person-to-person spread (|sd), from the IAM Sequencing Atlas's own donors. Markers: for each type, the 50 sites where it sits most above and the 50 most below its NEAREST other type, in units of their combined spread (stage_deconv_v21.markers_compute, PER_CELL = 50). The sample's level and read count at each marker.
- **The numbers:** 57 types, 5,699 markers (2026-10-30 build: 56 types, 5,599 markers; megakaryocytes added later as an identification-only column). On the build day 8 of 9 planted mixes read their shares; the ninth gave 8 % to close relatives; a pure cell gives others 1-2 % from counting alone (development log 2026-10-30). On 25 pure never-used healthy libraries (2026-11-21): median non-main share 0.006, 95th percentile 0.053, largest 0.305 (spill25.csv).
- **Method and standard:** Weighted non-negative least squares (scipy.optimize.nnls) with iterative re-weighting; intervals by resampling (bootstrap). Error bars are labeled 'information, not calibrated'. A share outside its own 95 % range is flagged 'not resolved from its close relatives'.
- **Known-answer test:** Planted mixtures through the one button (2026-11-21, plantmix_14.csv): planted share recovered 0/0.043/0.134/0.159/0.216/0.320/0.522 (heart + liver) and 0/0.046/0.082/0.126/0.177/0.263/0.459 (oligodendrocytes + acinar) at 0/0.05/0.10/0.15/0.20/0.30/0.50.
- **Limits:** Near-identical types trade share (kidney tubular and glomerular epithelium: podocyte libraries read tubular 0.48-0.69 and glomerular 0.21-0.26); types missing from the atlas appear as spread 'ghost' shares; same laboratory as the atlas.
- **Where the result goes:** bundle cell_shares (each cell's share, 95 % interval, resampled median, status); used by Met-A, IAM-A in the mix, the mixture check, each cell's Met-A, the globe's weights, and the report.
- **Code:** stage_v4.stage_d21; stage_deconv_v21.deconvolve, _solve, markers, markers_compute (LAB_SD 0.02, PER_CELL 50, BOOT 200, SEED 2026); stage_seq_atlas.deconv_reference_of, DECONV_REF, DECONV_DROP
- **How to rebuild it:** From the IAM Sequencing Atlas's per-type levels and spreads: run markers_compute (50 above and 50 below the nearest other type per type), keep those sites' |mu and |sd as deconv_reference_v1.parquet; deconvolve each sample with stage_deconv_v21.deconvolve (seed 2026).

| A researcher may ask | Answer |
|---|---|
| How are the marker sites chosen? | For each of the 57 types, the 50 CpGs where it sits furthest above, and the 50 furthest below, its nearest other type, measured in their combined person-to-person spread; 5,699 sites in all. |
| How is the mix solved? | Weighted non-negative least squares: each site weighted by the expected spread of a mixed healthy sample (the types' spreads times the shares) plus a laboratory term, re-weighted three times. |
| What do the 95 % intervals mean? | 200 resamplings of the marker sites, each also drawing that person's cells from the atlas's spread; labeled information, not calibrated. |
| Does it assume which cells are in a sample? | No. In whole blood, plasma and tissue every atlas type is allowed and the data decide. A sorted label is never used to choose a cell; a disagreeing label is only flagged. |
| How well does it recover known mixes? | Planted two-cell mixtures read 0.043-0.522 for planted 0.05-0.50 (heart + liver) and 0.046-0.459 (oligodendrocytes + acinar); pure healthy libraries give others a median 0.006. |
| What happens with a cell type the atlas lacks? | Its signal spreads over related types as small 'ghost' shares; an identification-only column (as for megakaryocytes) is added only when real samples show it. |

#### Met-A's tare on sequencing: the Deconvolver's spill inside a tissue family

- **The rule:** SORTED (isolated, purified) sequenced samples only; in whole blood, plasma and tissue Met-A keeps the Deconvolver's shares and no cell is assumed absent. Inside Met-A's healthy expectation: the Deconvolver's largest share is the main cell; a share in the main cell's tissue family (stage_0_intake.lineage_group) counts as the main cell; a cell in another family keeps its share only at or above SPILL_FLOOR = 0.05; megakaryocytes are left out (identification only); the rest is renormalized. The sorted label is recorded but never used: when it names a different type, the Deconvolver's call is used and the disagreement is flagged (tested: reading against the label's type moved Met-A away from 1). The Deconvolver, its reported shares, IAM-A in the mix and the mixture check are unchanged.
- **Why it is needed:** On healthy pure libraries the Deconvolver names the main cell but gives a few per cent to related references in the same tissue family (colon epithelium -> colon enteroendocrine 0.190; heart fibroblasts -> smooth muscle 0.238; tongue base -> tonsil palatine 0.305). Taken as real cells, those shares make Met-A's expected disorder too high at the sites where those references differ, so a healthy sample read low: median whole-sample Met-A 0.938 on 31 never-used healthy libraries (sorted-cell Met-A on the same libraries 1.004). In grid terms: the meter and relay are right; the setpoint was built with phantom load added in.
- **Physics:** Not a step of IAM's Law: a calibration of the reference that Met-A (the disorder of the record against its healthy expectation) is read against. Met-A's expectation is the entropy of the atlas mixed in the sample's shares; entropy is steep near 0 and 1, so a few per cent of a related type at a firmly held site raises the expected disorder much more than it changes the level (the reason small shares bias Met-A, recorded 2026-11-21).
- **Where the data comes from:** (1) The Deconvolver's shares for the sample (Stage D21, deconv_reference_v1, 5,699 marker CpGs, 57 types). (2) The chain's tissue families (stage_0_intake.lineage_group, the same map the mixture check uses). (3) SPILL_FLOOR from spill25.csv: every non-main share on 25 pure healthy libraries of atlas types (GSE186458, never used to build the atlas; real types from each GEO file name, lib49_real_types.csv).
- **The numbers:** Spill on the 25 pure libraries: 326 non-main shares, median 0.006, 95th percentile 0.053, largest 0.305; 15 of 19 shares of 0.05 or more stay in the main cell's family. Whole-sample Met-A on the 31 never-used atlas-type libraries: median 0.938 with the Deconvolver's shares, 1.016 with the rule; correlation with sorted-cell Met-A 0.18 -> 0.51; 27 of 31 within the GUM bar. Outside: GSM5652341 Prostate-Epithelial 0.865 (z -6.27, k 4.3); GSM5652237 Liver-Hepatocytes 1.200 (z 2.33, k 2.0); GSM5652330 Tongue_base-Epithelial 1.166 (z 4.84, k 4.3); GSM5652261 Kidney-Glomerular-Podocytes 1.208 (z 5.30, k 4.3).
- **Method and standard:** Bar by the GUM (JCGM 100:2008): u = each cell type's person-to-person standard deviation of held-out sorted-cell Met-A (held-out pass 8, each atlas library read against the other donor half), pooled 0.086 over 137 libraries where a type has fewer than 3; coverage factor Student's t at n - 1 degrees of freedom for the type's library count (k = 4.30 for 3 libraries; Student 1908), 2 for large n. Not yet in u (both widen the bar): each read's counting noise and the shares' own uncertainty. The floor is the 95th percentile of measured spill, set from spill25.csv before the run; the prediction was written first.
- **Known-answer test:** Inject and recover (jobs/plant_mix.py: two healthy libraries' molecules mixed at known fractions, seeded binomial thinning): cardiomyocyte GSM5652214 + hepatocyte GSM5652236, and oligodendrocytes GSM5652222 + pancreatic acinar GSM5652246, at 0/0.05/0.10/0.15/0.20/0.30/0.50, read through the one button. With the rule, pair 1 reads 0.984/1.055/0.969/0.993/0.994/0.998/1.015 and pair 2 1.016/1.098/1.019/1.015/1.005/0.999/0.988 (with the Deconvolver's shares: 0.894/0.897/0.922/0.956/0.973/0.996/0.996 and 0.952/0.969/0.977/0.980/0.974/0.974/0.976). The chain's tare_shares is checked to give exactly the tested mixes.
- **Limits:** (1) A real admixture below 0.05 from another family reads as pure (the detection limit; the 5 % plantings read 1.055 and 1.098). (2) A real mixture inside one family (e.g. fibroblasts with smooth muscle) is read as the main cell: its size is not yet measured (owed: a planted within-family mixture). (3) A large false share from another family stays (cortex neuron GSM5652232: striated muscle 0.244 kept, Met-A 0.861). (4) The within-family name can be wrong (podocytes called kidney tubular epithelium) while the family is right: Met-A's mix is unaffected, the name is. (5) Same laboratory as the atlas; a second laboratory's healthy libraries are owed. (6) Sequencing only; arrays untested. (7) The tissue families are broad (all 26 epithelia are one family, hepatocytes and pancreatic acinar included): stromal (6): adipocytes, cardiomyocyte, colon fibroblasts, heart fibroblasts, smooth muscle, striated muscle; endothelial (6): aorta endothelium, kidney glomerular endothelium, kidney tubular endothelium, lung alveolar endothelium, pancreas endothelium, vascular saphenous endothelium; B cells (2): b cells, memory b cells; epithelial (26): bladder epithelium, breast basal epithelium, breast luminal epithelium, colon enteroendocrine, colon epithelium, endometrium epithelium, fallopien tubes epithelium, gastric antrum epithelium, gastric body epithelium, gastric fundus epithelium, hepatocyte, kidney glomerular epithelium, kidney tubular epithelium, lung alveolar epithelium, lung bronchus epithelium, pancreatic acinar, pancreatic alpha, pancreatic beta, pancreatic delta, pancreatic duct, prostate epithelium, small intestine epithelium, thyroid epithelium, tongue epithelium, tonsil palatine epithelium, tonsil pharyngeal epithelium; T cells (8): cd4 t cells, cd8 t cells, effector memory cd8 t cells, naive cd4 t cells, naive cd8 t cells, t central memory cd4, t effector cell cd8, t effector memory cd4; monocytes and macrophages (4): colon macrophages, lung alveolar macrophages, lung interstitial macrophages, monocytes; neural (2): cortical neurons, oligodendrocytes; erythrocyte progenitors (1): erythrocyte progenitors; granulocytes (1): granulocytes; nk cells (1): nk cells. (8) Labels can be wrong (outside 'sorted' samples were 24-80 % their label, 2026-11-06/07; three podocyte libraries read further from 1 against their label's type, 1.208 -> 1.271 and 1.016 -> 1.092): the label is never used, only flagged. (9) The test sets' 'real types' are GEO file names: the best available truth, still labels.
- **Where the result goes:** bundle met_a_tare (main cell, family, floor, shares counted as main, shares left out, the mix); every Met-A reading of a sorted sequenced sample (whole sample, each cell in the mix, departure and C-Score in the mix) uses that mix; the report states the shares the tare moved under Met-A.
- **Code:** stage_metaa_atlas.tare_shares, SPILL_FLOOR, ID_ONLY; MethylPhys_Interface/run_sample.py (the edit-236 lines in the Met-A block: the sorted-sample switch and the mix); stage_0_intake.lineage_group
- **How to rebuild it:** 1. List every non-main Deconvolver share on pure healthy libraries whose real type (GEO name) is an atlas type -> spill25.csv; the floor is its 95th percentile. 2. For each library, apply the rule to its Deconvolver shares -> tare_rules.json. 3. Read each library through the one button with those shares (--dev-shares) on Google -> whole-sample Met-A. 4. Judge each against 1 with u from held-out pass 8 and Student's t coverage. 5. Plant two-library mixtures (plant_mix.py, seeds = CRC32 of the job name) and read them the same way.

| A researcher may ask | Answer |
|---|---|
| Why not fix the Deconvolver instead? | Its job is identification, and it names the main cell correctly; its shares are still reported unchanged. The bias comes only from how Met-A used near-miss shares between related references, so the correction sits in Met-A's tare. |
| Was the 0.05 floor tuned to make Met-A read 1? | No. It is the 95th percentile (0.053) of every non-main share measured on 25 pure healthy libraries (spill25.csv), set before the run; the prediction was written in the design sheet before any read. |
| Doesn't this hide real mixtures? | Not across families at 10 % or more: planted mixtures keep their mix and read 0.969-1.019. Below 0.05 a cross-family admixture reads as pure (detection limit). Inside one family a real mixture is read as the main cell; its effect is not yet measured and is owed. |
| Why is sorted-cell Met-A the comparison? | It reads only the type's identity sites against each healthy donor separately, so it does not depend on the shares. It is the independent route; the rule moved whole-sample Met-A toward it (correlation 0.18 -> 0.51). |
| How were the pass/fail bars set? | By the GUM (JCGM 100:2008): each type's person-to-person standard deviation from held-out readings, with Student's t coverage for small library counts. No chosen number. |
| Which libraries, and were they used to build the atlas? | 49 sorted GSE186458 libraries never used to build the IAM Sequencing Atlas; real types from each GEO file name; 31 are atlas types and are the test set. |
| Why do four libraries fall outside? | GSM5652341 Prostate-Epithelial 0.865 (z -6.27, k 4.3); GSM5652237 Liver-Hepatocytes 1.200 (z 2.33, k 2.0); GSM5652330 Tongue_base-Epithelial 1.166 (z 4.84, k 4.3); GSM5652261 Kidney-Glomerular-Podocytes 1.208 (z 5.30, k 4.3). Two are read against a different site or the wrong type within the family, one is the prostate reference (thin donors), one matches its own sorted-cell reading. |
| Is this independent of the atlas's own laboratory? | Not yet: every library here comes from the atlas's laboratory. A second laboratory's healthy libraries (already aligned cord-blood B cells and CD4 T cells) are the owed independent check. |
| Does it change array readings? | No. It applies only to sorted sequenced samples; arrays are untested and unchanged. |
| Can it be rebuilt from public data? | Yes: the libraries are public (GEO GSE186458), the chain is public, and the tables, job scripts and seeds are in the development record. |
| Isn't trusting the sorted label what went wrong before? | Yes, and it was tested here too. A step that let a sorted label name the main cell inside its family was tried on three podocyte libraries: read against the label's type, Met-A moved away from 1 (1.208 -> 1.271, 1.016 -> 1.092), so the molecules sided with the Deconvolver. The label is never used; a disagreeing label is only flagged. |
| Why only sorted samples? | Because only there is one cell type expected; in whole blood, plasma and tissue a few per cent of any cell may be real (e.g. liver DNA in plasma), so dropping small shares there would assume absence. The rule was tested only on sorted libraries. |
| What is a tissue family? | The chain's existing grouping (stage_0_intake.lineage_group), also used by the mixture check: stromal (6): adipocytes, cardiomyocyte, colon fibroblasts, heart fibroblasts, smooth muscle, striated muscle; endothelial (6): aorta endothelium, kidney glomerular endothelium, kidney tubular endothelium, lung alveolar endothelium, pancreas endothelium, vascular saphenous endothelium; B cells (2): b cells, memory b cells; epithelial (26): bladder epithelium, breast basal epithelium, breast luminal epithelium, colon enteroendocrine, colon epithelium, endometrium epithelium, fallopien tubes epithelium, gastric antrum epithelium, gastric body epithelium, gastric fundus epithelium, hepatocyte, kidney glomerular epithelium, kidney tubular epithelium, lung alveolar epithelium, lung bronchus epithelium, pancreatic acinar, pancreatic alpha, pancreatic beta, pancreatic delta, pancreatic duct, prostate epithelium, small intestine epithelium, thyroid epithelium, tongue epithelium, tonsil palatine epithelium, tonsil pharyngeal epithelium; T cells (8): cd4 t cells, cd8 t cells, effector memory cd8 t cells, naive cd4 t cells, naive cd8 t cells, t central memory cd4, t effector cell cd8, t effector memory cd4; monocytes and macrophages (4): colon macrophages, lung alveolar macrophages, lung interstitial macrophages, monocytes; neural (2): cortical neurons, oligodendrocytes; erythrocyte progenitors (1): erythrocyte progenitors; granulocytes (1): granulocytes; nk cells (1): nk cells. |

Source: `stage_metaa_atlas.METAA_TARE_TEXT`, generated from the cell instruments database (the SOP prints the same text).

*Code:* `reference_door.py` line 61 (16 lines) · arguments: o · calls: not called on the traced runs

## Sex and the X chromosome, in every instrument

**Why sex matters to a methylation reading.** A woman has two X chromosomes and silences one in every cell by methylating it (X inactivation). At the switch sites of hundreds of X genes a man's single X reads near 0 and a woman's two copies average about 0.5. The Y exists only in men. A comparison between people of different sex can therefore light up the X without any disease. Each instrument handles this as follows (numbers computed from the chain's own files by `CANON/sex_chromosomes.py`):

| Instrument | X and Y |
|---|---|
| Stage 0, intake | Predicts the array's sex from its X and Y probe brightness (step 0.8: female when log2 Y minus log2 X is below -2.0). When a sex is declared and the array disagrees, the array is stopped and the reason printed: a swapped sample is caught before any reading. |
| Met-A | 6,000 neutrophil identity sites: 32 on the X, 0 on the Y. They were chosen where the six purified reference arrays, one of them a woman's, agree. Group-A's sealed test read 75 women and 188 men the same after the tare (-0.36 %, 1.2 standard errors). |
| C-Score | The same 6,000 sites (32 on the X, 0 on the Y), in genome order; it reads how departures cluster, read against the same healthy reference. |
| Noise index | 48,528 sites, 95 on the X and 15 on the Y. Even if every one of them read at maximum noise, the index could move by at most 0.0023 bits (0.23 % of its 1-bit scale). |
| Stage A, the blood mix | 963 markers, 0 on the X or Y. |
| IAM-Deconvolver | 600 markers, 0 on the X, 0 on the Y. |
| IAM-A | Each sample is read against its own predicted error, so no other person, of either sex, enters its reading. Aligned files (BAM) are counted on chromosomes 1 to 22 only. From a .pat file both counts use chromosomes 1 to 22. |
| Group-A | Compares groups run in the same series; its sealed bar 2 is a women-against-men null (above). |
| IAM-Atlas v2.1 and the globes | 813,951 atlas sites, 14,759 on the X and 3 on the Y. The atlas's donors are women and men, so it is 2.5 to 3.6 times less sure on the X and Y than elsewhere (the noise globe shows it). Every change count on a globe is also given on the X and Y, and the page prints the sex Stage 0 read. When a tumor is drawn against its own normal tissue (same person, same sex), its change is counted with and without the X and Y: in a tumor an X change can be real biology. |

## 1. The one command: `run_sample.py`

**What it is for.** One sample in, one report out. A researcher runs one command on an IDAT pair (an Illumina EPIC v1 or 450K array), a
calibrated beta table, or sequencing data (a wgbstools `.pat` file, an aligned bisulfite BAM, or, in development, per-CpG counts as a `.beta`
file or a call and coverage bigWig pair, `--dev-seq-beta`; a per-site table such as Bismark coverage is first converted with `--build seq-table`). `run_sample.py` decides which stages
run, in what order, and stops the run the moment a stage refuses the sample. It computes no reading itself.

**Label: standard method** (orchestration). Nothing here is physics.

**The order it runs.**

1. Parse the command and refuse impossible combinations (no input; two sequencing inputs; a BAM without its reference FASTA; a site
   table without its pipeline; both kinds of reference list).
2. **Specimen check** (`stage_0_intake.specimen_refusal`). The chain accepts whole blood, buffy coat (read on the whole-blood path,
   labeled preliminary), isolated, sorted or purified neutrophils, isolated, sorted or purified cells of any IAM-Atlas v2.1 type (read by
   Met-A against the atlas, flagged where not commissioned), plasma, tissue, and constructed DNA mixtures of blood cells (test material).
   Any other specimen is refused before a byte is read.
3. **Stage 0, intake**, for an IDAT pair: 0.1 arrival, 0.2 manifest, 0.3 integrity hash. A quarantine at any of these stops the run
   (exit code 2) before calibration. An EPIC v2 header is refused here: the chain's references are EPIC v1, and Stage 1's calibrator
   does not read EPIC v2. EPIC v2 has a development route only (`--dev-epic-v2`, through SeSAMe).
4. **The QC hand-off** (`stage_0_1_qc_handoff.decode_qc_inputs`) reads the array's control probes, probe intensities, negative-control
   background, bead counts and sex probes straight from the IDAT files. If the array manifest cannot be loaded, the machine is at
   fault, not the array: exit code 3, and the array is not judged. If the decoder fails on the files, the array is quarantined as a
   corrupt IDAT.
5. **Stage 0, the QC gates**: 0.4 control probes, 0.5 detection p-values, 0.6 bead counts, 0.7 call rate, 0.8 sex, 0.7b platform
   coverage, then 0.9 the decision gate. A quarantine prints every reason in five parts (measured, limit, where the limit comes from,
   what it means, what to do) and stops the run.
6. **Stage 1, calibration** (`stage_1_idat_calibration.calibrate_idat_to_beta`): the IDAT pair becomes one beta value per site, with
   the probes at background removed. Stage 1's own quality numbers are recorded beside Stage 0's, never over them, and never gate. Its
   control medians (hybridization, negative and non-polymorphic controls, and two medians over all BS Conversion I and II probes) are
   recorded and not graded: the two bisulfite medians are of different control sets, not a conversion efficiency, which is Stage 0's.
7. **Stage 0.7b again**, now with the calibrated values: `run_sample.py` counts the share of Met-A's 6,000 neutrophil identity sites
   the array measured and hands it to 0.7b. On EPIC it records `NA_EPIC` and passes. On a 450K array below 80 % the gate's failure stays
   in the record, but the array advances flagged (`ADVANCE_FLAGGED_450K`): the commissioned Met-A and C-Score, whose EPIC references a 450K
   chip only half covers, refuse it themselves, and the atlas readers read it, each with its own coverage. Met-A itself withholds its reading when fewer than 90 % of its sites were measured (Stage M).
8. **The conductor** (`conductor_v3.run_neutrophil`): the platform check, Stage A (the cell mix), Stage M (Met-A), Stage MC (the
   C-Score) and Stage T (the same-run tare).
9. **Chain v4 additions** when the array was read: Stage D21 (`stage_v4.stage_d21`, the IAM-Deconvolver's cell shares against
   IAM-Atlas v2.1) and the three skies (the C-Score's departures on the sphere, the healthy neutrophil of IAM-Atlas v2.1 at the same
   sites, and the overlay).
10. **IAM-A** when sequencing data is given: Stage Q0 intake on a `.pat` file, then Stage Q2 (`stage_v4.count_sequencing` and
    `stage_v4.stage_q2`), the held error and the loose share counted in one pass and read against the sample's own reference.
    The retired position reading (Stage Q) runs only with `--site-table` or `--dev-iama-position`, labeled development.
11. **Stage 12b**, the difference between two draws of the same person, when `--prior-betas` is given; refused unless both draws carry
    the same identifier hash, array type and pipeline.
12. **Development readings**, only when a `--dev-*` flag asks for them, written under `bundle["development"]` and never part of the
    reading.
13. **The report** (`report_v4.build`): one HTML page with a researcher view and a plain view, and two PDFs. Then the identifier is
    replaced by its hash everywhere in the bundle and the ledger, the bundle is written as JSON, and one row is appended to the
    evidence ledger.

**Exit codes.** 0 a report was written (it may still withhold a reading, and says why); 2 the sample was quarantined at intake and
nothing was scored; 3 the machine is missing a file it needs (the array manifest) and the array was not judged.

**What it reads.** The command line; the IDAT pair or beta table or sequencing file; `--ref-table` (the same-run healthy references:
column A for Met-A, column C for the C-Score); `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json` for the 0.7b coverage count.
Everything else is read by the stages.

**How to break it.** Give two sequencing inputs, a BAM without `--ref-fasta`, a specimen outside the list, an EPIC v2 array, a
truncated IDAT: each must stop with a named reason, never a traceback. Release checks V1 to V10 run the one command end to end.

### `run_sample.py::_site_counts`

Register T1 and C3: in every sequencing reading, the held and absent sides counted by Stage Q2's own rules
(count_pat_sites) at the islands territory (sites every healthy cell type keeps unmarked, so what is counted there is the library's own
meter) and at the epiTOC2 mitotic-clock CpGs with 25 CpGs either side. Written as iam_a.sites; it never changes a reading.

*Code:* `MethylPhys_Interface/run_sample.py` line 623 (34 lines) · arguments: pat · calls: not called on the traced runs · files named: `Runtime Matrices/Clock_Sites/epitoc2_sites.csv`, `_v1.npz`, `epitoc2_sites.csv`, `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `run_sample.py::_bio_root`

Finds the `Biological_Physics` folder by walking up from this file until it finds a folder holding `MethylPhys`, rather than by
counting levels. Written after a 2026-09-22 move changed every script's depth by one and a counted path silently stopped finding
the record folder. Decides nothing about a sample.

*Code:* `MethylPhys_Interface/run_sample.py` line 39 (16 lines) · arguments: start · calls: array run 1, bam run 1

### `run_sample.py::_assign_run_id`

Gives the run its identifier, `RUN-YYYYMMDD-NN`: the UTC date and the next number for that day, counted from the ledger the run is
about to append to. A run makes no claim and passes no bar; it is one execution on one specimen, and the identifier lets a reading be
found again.

*Code:* `MethylPhys_Interface/run_sample.py` line 59 (19 lines) · arguments: ledger_path · calls: array run 1, bam run 1

### `run_sample.py::_covariates`

Collects the phenotype and covariates for the run: a JSON object from `--covariates`, then each `--covariate KEY=VALUE` on top. They
are kept in the bundle and the custody record and are never printed in the report's prose. No stage reads them.

*Code:* `MethylPhys_Interface/run_sample.py` line 80 (16 lines) · arguments: a · calls: array run 1, bam run 1

### `run_sample.py::_sha12`

The first 12 hexadecimal characters of a file's SHA-256, used to fingerprint each frozen input in `_versions`. Returns nothing if the
file cannot be read.

*Code:* `MethylPhys_Interface/run_sample.py` line 98 (10 lines) · arguments: path · calls: array run 18, bam run 18

### `run_sample.py::_versions`

Records everything that can change a reading: the UTC time, the Python and methylprep versions, the chain's git commit and whether
the working copy had uncommitted changes, and a short hash and size of each frozen matrix and stage file. Two runs months apart can
be pooled only if these match.

*Code:* `MethylPhys_Interface/run_sample.py` line 110 (55 lines) · arguments: chain_dir · calls: array run 1, bam run 1 · files named: `Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`, `Runtime Matrices/IAM_A_Positions/iama_positions_v2.json`, `Runtime Matrices/Intake/intake_thresholds_v1.json`, `Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`, `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json`, `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`, `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `compact_manifest.json`, `in compact_manifest.json`, `transfer_v21_final.json`

### `run_sample.py::_why`

Prints and keeps one stop's explanation in five parts (what was measured, the limit, where the limit comes from and whether it is
provisional, what it means, what to do), for every stop outside Stage 0's own gates: a refused specimen, the early intake stops, EPIC v2,
a missing manifest (exit code 3), a corrupt IDAT, Stage 1's platform refusal and the quarantine after calibration. The report shows the same
rows under "Why the chain stopped or flagged this sample". Release check V11 fails if any exit with code 2 or 3 has no explanation in its
own block.

*Code:* `MethylPhys_Interface/run_sample.py` line 167 (8 lines) · arguments: rec, check, outcome, measured, limit, source, means, todo, provisional · calls: not called on the traced runs

### `run_sample.py::_hash_id`

Replaces a typed identifier with the first 32 characters of its SHA-256. An identifier that already looks like a hash (16 or more
hexadecimal characters) is kept. The engine never stores a cleartext identifier.

*Code:* `MethylPhys_Interface/run_sample.py` line 177 (5 lines) · arguments: x · calls: array run 1, bam run 1

### `run_sample.py::_id_collides`

True when the typed sample id is the name of an IAM-Atlas v2.1 cell type or of a provisional column. Then the id is hashed in its own field only and the bundle's text is not searched for it, so a cell's name is never erased (found when the id 'megakaryocytes' erased the cell's name in provisional reference test 1).

*Code:* `MethylPhys_Interface/run_sample.py` line 184 (12 lines) · arguments: sid · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `run_sample.py::_redact`

Walks the whole bundle (every string, dictionary key and list) and replaces each whole-token occurrence of the typed identifier with
its hash, so neither the bundle nor the ledger carries the typed name. The printed report keeps the typed name for the operator.

*Code:* `MethylPhys_Interface/run_sample.py` line 197 (11 lines) · arguments: obj, typed, hid · calls: array run 4, bam run 4

### `run_sample.py::_redact.f`

The recursive step of `_redact`: a string gets the replacement, a dictionary and a list are walked, anything else is returned as is.

**`--dev-count-only` (development, to-do 14).** With `--pat`, Stage Q2's count (the held error and its opportunities) is recorded in the
bundle as `iam_a_count_only`, flagged not commissioned, even where Stage Q0 refuses an IAM-A reading (a cell line, a mouse sample: no healthy
reference). IAM-A itself stays refused. The firm-rate recount reads the held error this way, so its count is the one command's. The held error
does not depend on the genome; the loose share uses hg19 tiles and is marked meaningless on another genome (`--dev-count-genome`). **With `--bam` (2026-11-14, PHY-091):** Stage Q2 counts the aligned file directly (`stage_q2_selfref.count_bam` over the autosomes the reference
FASTA carries, so mm10 works through `--assembly mm10`) and records, besides the held error and the stray marks, the meter's own false marks:
`conv_fail`, the share of non-CpG cytosines read as methylated (incomplete bisulfite conversion; the known-zero channel each library carries),
and `sub_err`, the A/T substitution rate, and (2026-11-15, PHY-091 amendment 2) `tc_err`, the rate of the one substitution that can fake
a mark: a T read as C on top-strand reads, an A read as G on bottom-strand reads, at non-CpG positions where bisulfite plays no part, base quality
20 or more. `sub_err` counts every A/T mismatch in any direction, and in a mouse strain also its genetic differences from the reference, so it
is only an upper bound on the meter's sequencing false marks; `tc_err` is the measured one. Physics: none of IAM's Law; this is the meter. With `--lambda-bam` and `--lambda-fasta` (2026-11-15, PHY-091 amendment 3) it adds `lambda_meter`: the share of cytosines on the unmethylated lambda spike-in read as C, the meter alone, since genomic non-CpG sites also carry real CpH methylation in neurons and stem or progenitor cells (reads shorter than 50 aligned bases excluded; synthetic test `doors/data/DEV_ERASE_01/conv_01/lambda_synthetic.py`, GitHub). No IAM-A reading is made. A stray mark (a lone methylated call on a molecule held unmarked) minus the
false marks of the same library is the unmarked side's write error (register A17). The strand of each read comes from Bismark's XM/XG, or XB, or bwa-meth's YD:Z:f/r (`stage_q2_selfref._calls_of`; YD added 2026-11-14:
before that every bwa-meth read was skipped on the BAM route, found while designing PHY-091). Synthetic test, through the one command on a
synthetic bwa-meth BAM with planted conversion failure: `doors/data/DEV_ERASE_01/conv_01/conv_synthetic.py` (GitHub); for `tc_err`, planted T->C with a low-quality share to be excluded: `doors/data/DEV_ERASE_01/conv_01/tc_synthetic.py` (GitHub). With `--dev-count-sites islands` (2026-11-15, register T1) the same count is also made at the islands territory (stage_metaa_atlas.territory_sets: the sites every healthy cell type in IAM-Atlas v2.1 keeps below 0.2), both sides, so the stray-mark rate there can be set against the writer's one-step error plus the laboratory's meter; any other value names a cell's Met-A identity sites. Since 2026-11-16 the option takes a comma list (for example `islands,epitoc2`): each set is counted from the same file, the first set still fills `count_sites`, and every set is kept under `count_sites_all`, so the mitotic clock and its islands meter come from one reading (synthetic: `doors/data/DEV_ERASE_01/conv_01/multi_sites_synthetic.py`, GitHub: the islands count of a two-set run must equal the islands-only run exactly). With `--dev-count-sites epitoc2` (2026-11-15, register C3) the count is made at the 163 CpGs of the epiTOC2 mitotic clock (Teschendorff 2020, Genome Medicine; Runtime Matrices/Clock_Sites/epitoc2_sites.csv) and 25 CpGs either side in genome order, so the writer's stray-mark rate there can be set against the clock's per-round gain (about 5e-5): their ratio is the share of stray marks kept each round of writing and erasing.

*Code:* `MethylPhys_Interface/run_sample.py` line 202 (5 lines) · arguments: v · calls: array run 865, bam run 359

### `run_sample.py::_metaa_atlas`

`--dev-metaa-atlas CELL` (development, not commissioned): maps the values by Stage T step 1 (the self-tare the commissioned Met-A reads), refuses
unless both designs were mapped, reads Met-A against CELL in IAM-Atlas v2.1 (`stage_metaa_atlas.read`) and, with `--metaa-atlas-ref-A` (at least
3 healthy samples of the same cell run the same way, never this one), tares it (`stage_metaa_atlas.tare`). Written under bundle
`development.metaa_atlas`; the reading, gauge and tare are untouched.

*Code:* `MethylPhys_Interface/run_sample.py` line 210 (36 lines) · arguments: a, b · calls: not called on the traced runs

### `run_sample.py::_law_sphere`

`--dev-law-sphere` with `--pat` or `--geo-pat` (development, on in the full chain; edit 205): after IAM-A's count and the
Deconvolver's shares, reads every sequencing-sphere unit from the sample's molecules by IAM-A's own rule and sets it against the law's
expected point in the sample's own cells (`stage_law_sphere.reading`, Section 18b). Written under bundle `development.law_sphere`: the 14
layers summarised per job and for the whole sample; `--law-sphere-out` saves them per unit. Not run without a molecule file, IAM-A's loose
share or the Deconvolver's shares (the reason is written).
With `--out`, also writes the Law Sphere page `<report>_law_sphere.html` (edit 208). In the full chain it runs on every sequenced sample, including a file fetched from GEO with --geo-pat (edit 209). It also reads samples IAM-A does not read as one cell type (mixtures), in their own shares; without IAM-A's loose share it takes the sphere's own (edit 210).

*Code:* `MethylPhys_Interface/run_sample.py` line 522 (30 lines) · arguments: a, cnt, o · calls: not called on the traced runs · files named: `positions.npz`

### `run_sample.py::_genome_globe`

`--dev-genome-globe` with `--pat` or `--geo-pat` (development, not commissioned; the file must be hg19, Section 10b): counts the file's molecules into regions of `--genome-region` CpGs on the
autosomes (`stage_q2_globe.count_pat_regions`) and draws the sample against its own predicted held error (`self_globe`) and, with `--prior-pat`,
against that second file (`pair_globe`, named by `--prior-name`). Written under bundle `development.genome_globe`; with `--explorer-spec`,
the Globe Explorer file. In the full chain it runs on every sequenced sample, including a file fetched from GEO with --geo-pat (edit 209). It draws for mixtures too, each region against the sample's own prediction from its loose share (edit 211).

*Code:* `MethylPhys_Interface/run_sample.py` line 553 (23 lines) · arguments: a, cnt, sid · calls: not called on the traced runs

### `run_sample.py::_build`

`--build`: the chain builds a reference instead of reading a sample. `info-sphere` (development): the information sphere's layout from a
matrix of levels (sites x cell types), with its report (variance kept, sky covered and masked, distance agreement, edge ratio, and each named
site group's concentration against random groups) to `--bundle` and the layout to `--build-out`.

*Code:* `MethylPhys_Interface/run_sample.py` line 254 (266 lines) · arguments: a · calls: not called on the traced runs · constants read: `_ISV_ABOUT` · files named: `--build seq-metaa-counts needs --build-samples INPUTS.csv`, `--build seq-metaa-donors needs --build-samples INPUTS.csv`, `--build seq-metaa-inner needs --build-input TRIMMED_DONOR_COUNTS.parquet`, `--build seq-terr-halves needs --build-samples INPUTS.csv`, `--build seq-wg-layers needs --build-input TEXTURES.npz`, `--build seq-wg-sphere needs --build-input TEXTURES.npz`, `--build-samples B and --bundle OUT.json`, `--build-samples NOISE_MAPS.npz`, `MORE.npz`, `_layout.npz`, `_summary.txt`, `and --bundle OUT.json`

### `run_sample.py::_dev_run`

Runs the development readings named by `--dev-*` flags (direction, trace cell, foreign cell, brightness interval, NILC-e, atlas e,
B-cell reading, sky with the block-shuffle null, EPIC v2 through SeSAMe). Each writes its own record under `bundle["development"]`,
labeled DEVELOPMENT, and an error in one is caught and recorded. None of them changes the reading, the gauge or the tare.

*Code:* `MethylPhys_Interface/run_sample.py` line 578 (42 lines) · arguments: a, o, beta, intake · calls: array run 1, bam run 1

### `run_sample.py::main`

The orchestration described above, from parsing the command to writing the ledger row. Decides which stages run, stops the run with
exit code 2 or 3 at the stops listed, carries the same-run references (columns A and C) into the conductor, adds the v4 stages, and
hands everything to the report.

For a declared sorted cell (specimen "sorted ...", "isolated ..." or "purified ..." naming an atlas cell), the whole-sample readings (Met-A whole sample, the territories, the C-Score) are read against that cell's own atlas column instead of the blood panel's mixed shares; the bundle records it as `whole_sample_reference`.

**The default chain (2026-11-08, author: nothing built sits switched off).** Without `--commissioned-only`, every built development reading runs on every sample, each flagged, including on arrays the wide line, and on sequencing input the IAM Sequencing Atlas as ruler, the donors' levels, the counted island expectation, the copy correction and the megakaryocyte column. Off: the two library floors (failed their own tests), the measured middle (held), `--dev-curve-cal` and `--dev-one-scale` (until their synthetics). The table of options above says which are on.

**One scale (2026-11-09).** The self-tare map rescales only the 6,963 probes in its own table and leaves the rest at Stage 1's values; the per-cell readings read those values while the array's line is measured on sites almost all outside that table, so 3 to 7 % of each cell's identity sites were corrected twice. `--dev-one-scale` reads them on Stage 1's values with the same line, the old path's per-cell Met-A beside it. The curve calibration (`--dev-curve-cal`) first took the same short design table; it now takes the ruler's full one (`stage_metaa_atlas._ruler`). The default chain (no `--commissioned-only`) reads with every atlas cell (`--cell-panel` left unset becomes 'atlas'); on arrays the six-cell commissioned reading is written beside it as `cell_shares_blood6`. Isolated / sorted neutrophils get the atlas readings (cells found, Met-A per cell, whole-sample Met-A, C-Score per cell, territories) like every other specimen, beside the first commissioning's neutrophil path, which still runs and is flagged as being re-commissioned.

*Code:* `MethylPhys_Interface/run_sample.py` line 658 (1129 lines) · arguments: none · calls: array run 1, bam run 1 · constants read: `BIO`, `CONTROL_SITES`, `CONTROL_SITES_ABOUT`, `ENG`, `_ISV_ABOUT` · files named: `.gz`, `DIR/deconv_reference_v1.parquet`, `FILE.json`, `FILE.npz`, `MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `OUT.csv`, `OUT.npz`, `Runtime Matrices/Clock_Sites/epitoc2_sites.csv`, `Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet`, `_bundle.json`, `a wgbstools .pat / .pat.gz`, `array_sites/mu_part1.parquet`

### `run_sample.py::main._panel`

Turns `--cell-panel` into the IAM-Deconvolver's panel: nothing for `blood` (its six commissioned cells), every IAM-Atlas v2.1 cell type for
`atlas`, or the named cells, refusing any name the atlas does not hold.

*Code:* `MethylPhys_Interface/run_sample.py` line 1290 (8 lines) · arguments: none · calls: not called on the traced runs

### `run_sample.py::main._stopped`

True when an intake record's status begins with QUARANTINE. Intake checks it after every step, so a quarantine ends intake at its
cause. Without it a later step overwrote the status, and the verdict named a downstream symptom (seen 2026-09-23: an array-type
mismatch was reported as a detection failure).

*Code:* `MethylPhys_Interface/run_sample.py` line 1055 (5 lines) · arguments: r · calls: array run 3

### `run_sample.py::main._ctx`

Builds the comparison record Stage 12b needs from a draw: the identifier hash, the array type and the calibration pipeline.

*Code:* `MethylPhys_Interface/run_sample.py` line 1706 (2 lines) · arguments: b, it, pipe · calls: not called on the traced runs

## 2. Stage 0, intake: `stage_0_intake.py`

**What it is for.** Deciding whether this pair of files is a usable array from the specimen it claims to be, before anything is
calibrated or read. A quarantine here means nothing about the biology was looked at.

**Label: standard method.** These are the array maker's and the field's quality checks (control probes, detection against background,
bead counts, call rate, a sex check from the X and Y signals), applied to one array on its own. None of the limits comes from IAM's
physics. Every limit lives in a named constant or in `Runtime Matrices/Intake/intake_thresholds_v1.json`, never as a bare number
in a gate.

**The specimen check (before Stage 0).** `specimen_refusal` accepts whole blood (and the words blood, peripheral blood, venous blood),
isolated, sorted or purified neutrophils, buffy coat (read on the whole-blood path, labeled as its own specimen, preliminary), the
constructed DNA mixtures used as test material, isolated, sorted or purified cells of any IAM-Atlas v2.1 type (`atlas_cell_of`; read by
Met-A against the atlas, flagged where not commissioned), plasma (`is_plasma`) and tissue (`is_tissue`). PBMC, bone marrow, cell lines,
an unstated specimen and anything else are refused with the reason, and nothing is read.

**The steps, in the order the run calls them.**

| Step | What it measures | Outcome | Limit and source |
|---|---|---|---|
| 0.1 arrival | the manifest's required fields (Sentrix ID, array type, identifier, intake date, specimen); both channels present; each IDAT at least 1 MB; the array type in the IDAT header against the declared one | quarantine on a missing field, a missing channel, a truncated file, or a 450K/EPIC family swap; a note on an EPIC v1/v2 subtype difference; a soft warning on the same Sentrix ID taken in within 24 h | fixed rules (SOP 11): `MIN_PLAUSIBLE_IDAT_BYTES` 1,000,000; header address count below 800,000 is 450K, below 1,080,000 EPIC v1, above EPIC v2 |
| 0.2 manifest | builds the immutable per-sample record; the identifier must already be a hash | quarantine on a cleartext identifier (spaces, an @, under 16 characters) or a missing core field | fixed rules (SOP 12) |
| 0.3 integrity | SHA-256 of both files against earlier intakes of the same Sentrix ID in this intake log | identical bytes again: re-transmission, the run stops; same Sentrix ID with different bytes: a re-run of a fresh array, read | fixed rule (SOP 13) |
| 0.4 control probes | bisulfite conversion (C/(C+U) per matched BS Conversion I pair, each pair in its own channel, median over pairs), hybridization (the spread between the high and low concentration controls, with medium between them, green) and extension (green C and G against red A and T) controls, from the array's own control addresses | a failed hybridization or extension stops the array; a high/low swap with the spread intact is recorded, not a stop; a bisulfite conversion below the line stops the array | from `intake_thresholds_v2.json`, set 2026-10-21 (PROC-STAGE0-04) on 337 healthy whole-blood arrays of three laboratories (GSE318173, GSE315366, GSE184269): bisulfite at least **0.80** (the author's choice, the 2.5th percentile with each laboratory counted equally; healthy medians 0.93 to 0.97 by laboratory; stops 1 of the 337, GSM5583015 at 0.576, low on every pair in its own channel); hybridization spread at least 2.0 (healthy 2.69 and above) and extension balance 0.2 to 5.0 (healthy 0.45 to 1.22), stopping none of the 337 |
| 0.5 detection | per probe, p = 1 − Φ((I − μ_bg)/σ_bg) against this array's negative-control background; the share of probes with p ≤ 0.01 | above 0.99 passes; 0.93 to 0.99 is read with a penalty flag; below 0.93 stops | `intake_thresholds_v1.json` detection |
| 0.6 bead counts | the share of probes resting on at least 3 beads | below 0.995 a penalty flag; never stops | `BEAD_COUNT_MIN` 3; `intake_thresholds_v1.json` bead |
| 0.7 call rate | the share of probes passing both detection and bead count | 0.98 or more passes; 0.93 to 0.98 penalty; below 0.93 stops | `intake_thresholds_v1.json` call_rate; the 0.93 line was set by the author on 2026-09-27 from 48 arrays, 12 from each of four commissioning laboratories |
| 0.7b coverage | the platform tag; on a 450K array, the share of the commissioned neutrophil identity sites measured | on 450K, below 0.80 the gate fails and stays in the record; when it is the only failure, `run_sample.py` advances the array flagged (`ADVANCE_FLAGGED_450K`): the commissioned Met-A and C-Score refuse 450K themselves and the atlas readers read it, each with its own coverage; on EPIC, `NA_EPIC` (passes) | `HM450_COVERAGE_MIN` 0.80 |
| 0.8 sex | median log2 intensity of the Y probes minus the X probes | Y − X below −2 is female, otherwise male; a mismatch with a declared sex stops the array; with no declared sex the array's sex is recorded | `SEX_GETSEX_CUTOFF` −2.0, the rule of the minfi package's getSex (SOP 18) |
| 0.9 decision gate | every step's outcome | any hard failure: QUARANTINE; any penalty: PROCEED_WITH_PENALTY; else PROCEED | a detection or call rate that could not be measured is a hard failure, never a pass (added 2026-09-27 after a low-signal laboratory had walked through for a week) |

**Every stop explains itself.** `explain_stage0` attaches, for every check that stopped, penalized or deferred the array: what was
measured on this array, the limit, where the limit comes from and whether it is provisional, what it means and what to do.
`run_sample.py` prints these on a quarantine. Release check V9 builds each kind of stop and fails if any one is not explained.

**Built from what, tested on what.** The detection and call-rate lines were measured on 48 arrays, 12 from each of the four
laboratories of Met-A's commissioning panel (`intake_thresholds_v1.json`, `_meta`): Uppsala 0.985, Karolinska 0.975, UCLA 0.953,
Munich 0.878. The detected fraction, not the signal-to-background ratio, separated the low-signal laboratory. The control-probe limits
are provisional and have not been set from healthy arrays; setting them is a to-do step. Until then a provisional limit is printed
with the word PROVISIONAL wherever it is used.

**How to break it.** Hand it a single channel, a 900 KB IDAT, a 450K pair declared as EPIC, the same pair twice under one Sentrix ID,
a declared sex that disagrees with the array, or a low-signal array: each must quarantine with its reason printed in five parts. The
file's own self-test (`python3 stage_0_intake.py`) runs every gate on constructed inputs.

### `stage_0_intake.py::_intake_thresholds`

Loads `intake_thresholds_v1.json` from the chain's `Runtime Matrices/Intake` folder (never from a RETIRED folder) when the module is
imported. Stage 0 refuses to start without it, so no gate can run on a number typed into the code.

*Code:* `stage_0_intake.py` line 35 (8 lines) · arguments: none · calls: array run 1, bam run 1 · files named: `intake_thresholds_v2.json`

### `stage_0_intake.py::normalise_specimen`

Lowercases the specimen text, turns underscores into spaces and collapses repeated spaces, so `Whole_Blood` and `whole blood` are the
same specimen.

*Code:* `stage_0_intake.py` line 76 (2 lines) · arguments: specimen · calls: array run 5

### `stage_0_intake.py::atlas_cells`

The 79 cell names of IAM-Atlas v2.1 (its best-value columns), read once.

*Code:* `stage_0_intake.py` line 82 (7 lines) · arguments: none · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `stage_0_intake.py::atlas_cell_of`

An isolated, sorted or purified cell of any IAM-Atlas v2.1 type other than neutrophils (case-insensitive): its atlas name, else nothing. Such a
specimen is accepted at intake and takes the single-cell route (development, not commissioned); the commissioned Met-A is not applied. Since
2026-11-07 the name it returns is the specimen's claim only: it chooses the route and is checked by `label_check`; it never sets the cell a
reading uses (rule 23b-h).

*Code:* `stage_0_intake.py` line 90 (9 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `ATLAS_CELL_WORDS`, `SEQ_ATLAS_ONLY_CELLS`

### `stage_0_intake.py::is_isolated`

A sample of isolated, sorted or purified cells, of a named cell or none ('sorted cells'), routed to the atlas readings like whole blood: the IAM-Deconvolver decides which cells it holds and each cell found is read against its own atlas reference. The label picks this route and never the cell. Isolated neutrophils keep the commissioned neutrophil route. Added when a constructed 'sorted cells' sample got no atlas readings at all.

*Code:* `stage_0_intake.py` line 201 (7 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `ACCEPTED_SPECIMENS`, `ATLAS_CELL_WORDS`

### `stage_0_intake.py::cells_found`

The cells the IAM-Deconvolver found in the sample, largest share first, each at or above 1 % (`CELL_FLOOR`, the presence gate).

*Code:* `stage_0_intake.py` line 116 (5 lines) · arguments: cell_shares, floor · calls: not called on the traced runs · constants read: `CELL_FLOOR`

### `stage_0_intake.py::chain_cell`

The cell the Deconvolver found most of, or nothing when it did not run. The only source of a single-cell reading's cell when the operator set
none.

*Code:* `stage_0_intake.py` line 162 (4 lines) · arguments: cell_shares · calls: not called on the traced runs

### `stage_0_intake.py::lineage_group`

The group a cell is pooled into before the mixture test: a blood cell by its lineage (`LINEAGE`: T cells, B cells, monocytes with the
macrophages), any other atlas cell by its tissue family (`FAMILY`: epithelial, endothelial, neural, stromal, from the atlas's own family
of each type), and anything else as its own group. Used by `mixture_check` (overhaul step E2c).

*Code:* `stage_0_intake.py` line 143 (3 lines) · arguments: c · calls: not called on the traced runs · constants read: `FAMILY_OF`, `LINEAGE_OF`

### `stage_0_intake.py::mixture_check`

One cell type or a mixture? The cells found are pooled into lineage groups (`LINEAGE`: the T-cell subsets, the B cells, monocytes with the
macrophages, neurons with oligodendrocytes, and the sibling kidney, stomach, islet, breast and tonsil types; any other cell is its own group),
and the sample is a mixture when a second group holds more than 0.12 (`MIX_FLOOR`). The floor is measured: in 154 sorted atlas libraries the
largest second share was 0.112, always a sibling type. A planted 100-million-molecule blood mixture is recovered within 0.02 per group.
Written to the bundle as `mixture_check`.
Since overhaul step E2c, blood cells keep these lineage groups (a blood mixture must stay a mixture) and every other atlas cell is pooled by
tissue family (`FAMILY`: epithelial, endothelial, neural, stromal; `lineage_group`). Reason: read outside the reference, 19 of 49 sorted
atlas libraries never used before were refused although the Deconvolver's first group was the true type; the second group was a sibling of
the same tissue family. With family pooling all 154 atlas inputs stay single, the planted blood mixture stays a mixture, and 12 of the 19 are
read; the other 7 carry a second family above 0.12 and stay refused. `MIX_FLOOR` is unchanged.

*Code:* `stage_0_intake.py` line 148 (13 lines) · arguments: cell_shares, floor · calls: not called on the traced runs · constants read: `MIX_FLOOR`

### `stage_0_intake.py::label_check`

The specimen's claimed cell against what the Deconvolver found: the claimed cell, its measured share, the cells found, the chain's cell, and
whether they agree. Written to the bundle as `label_check`; nothing is read from the claim.

**450K arrays and the atlas readers (2026-10-22).** On a 450K array the conductor refuses the commissioned Met-A and C-Score (no 450K healthy
neutrophil reference; `conductor_v3.platform_refusal`), but `run_sample.py` still runs the atlas readers, which work by site name on IAM-Atlas
v2.1: the IAM-Deconvolver's shares, Met-A for each cell and the globe, each flagged development and each reporting how many of its sites the
array measured. The bundle records this under `atlas_readers_on_450k`.

*Code:* `stage_0_intake.py` line 167 (12 lines) · arguments: specimen, cell_shares · calls: not called on the traced runs · constants read: `LABEL_RULE`

### `stage_0_intake.py::is_plasma`

True for plasma cell-free DNA (plasma, cfDNA, cell-free DNA): accepted at intake since 2026-10-22 and read as a development specimen. The
IAM-Deconvolver splits it on IAM-Atlas v2.1's full panel (every cell and tissue type the atlas holds, since plasma DNA comes from cells all over
the body) and Met-A reads each source it finds against the atlas in the sample's own shares; the commissioned Met-A and C-Score read whole
blood and are not applied.

*Code:* `stage_0_intake.py` line 183 (3 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `PLASMA_WORDS`

### `stage_0_intake.py::is_tissue`

True for a solid tissue, a tumor, a biopsy, or a specimen not named ('unknown', for blind runs). Such a sample is read on IAM-Atlas v2.1's full panel, flagged, as plasma is: the IAM-Deconvolver says what it is made of, and Met-A reads each cell it finds against the atlas in the sample's own shares. The commissioned readers read blood and are not applied. Bone marrow, a marrow aspirate and a blood-and-marrow mix are read on this route too, with the marrow progenitors kept in the mix (read and flag; it was refused at intake before).

*Code:* `stage_0_intake.py` line 188 (12 lines) · arguments: specimen · calls: not called on the traced runs · constants read: `TISSUE_WORDS`

### `stage_0_intake.py::specimen_refusal`

Returns nothing when the specimen is accepted, otherwise the plain refusal: which specimen, why the chain has no reference for it,
and what is accepted. Called by `run_sample.py` before any file is read.

*Code:* `stage_0_intake.py` line 209 (9 lines) · arguments: specimen · calls: array run 2 · constants read: `ACCEPTED_SPECIMENS`, `REFUSED_SPECIMEN_REASONS`

### `stage_0_intake.py::_infer_platform`

Turns the IDAT header's address count into a platform: below 800,000 is 450K, below 1,080,000 EPIC v1, otherwise EPIC v2.

*Code:* `stage_0_intake.py` line 230 (8 lines) · arguments: nsnps · calls: array run 2

### `stage_0_intake.py::read_idat_nsnps`

Reads only the start of the green IDAT: the four-byte `IDAT` mark, the field table, and field 1000 (the number of addresses). It never
decodes intensities. Returns the count and a status (OK, NOT_IDAT, NO_NSNPS_FIELD, FILE_NOT_FOUND or a read error).

*Code:* `stage_0_intake.py` line 242 (24 lines) · arguments: path · calls: array run 2

### `stage_0_intake.py::step_0_1_idat_arrival`

Step 0.1: checks the manifest's required fields and array type, that both channels exist, that each file is at least 1 MB, and that the
header's platform family agrees with the declared one; warns on a repeat intake within 24 hours. Stamps a run identifier and the
arrival time, and logs the result.

*Code:* `stage_0_intake.py` line 270 (89 lines) · arguments: manifest_entry, grn_path, red_path, intake_log_path · calls: array run 1 · constants read: `MIN_PLAUSIBLE_IDAT_BYTES`, `REQUIRED_MANIFEST_FIELDS`, `VALID_ARRAY_TYPES`

### `stage_0_intake.py::_log_intake`

Appends one row per intake step to the intake log, when one is given: time, run identifier, Sentrix ID, declared and detected array
type, specimen, declared sex and age, status and flags.

*Code:* `stage_0_intake.py` line 363 (10 lines) · arguments: intake_log_path, result · calls: array run 1

### `stage_0_intake.py::_is_duplicate_within_24h`

True when the intake log already holds the same Sentrix ID from the last 24 hours. Only a warning; the array still advances.

*Code:* `stage_0_intake.py` line 375 (21 lines) · arguments: intake_log_path, sentrix_id · calls: array run 1

### `stage_0_intake.py::_is_null_answer`

For storing questionnaire answers: a blank or "prefer not to say" becomes no answer; "No" is kept as a real answer.

*Code:* `stage_0_intake.py` line 458 (3 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NULL_FOR_STORAGE`

### `stage_0_intake.py::_answer_present`

For the questionnaire's context modes: true only for an affirmative answer (not blank, not "prefer not to say", not No, none or not
applicable).

*Code:* `stage_0_intake.py` line 463 (5 lines) · arguments: v · calls: not called on the traced runs · constants read: `_NEGATIVE`, `_NULL_FOR_STORAGE`

### `stage_0_intake.py::_age_from_dob`

Age in decimal years from a date of birth and the intake date, rounded to 0.1.

*Code:* `stage_0_intake.py` line 470 (8 lines) · arguments: dob, on_date · calls: not called on the traced runs

### `stage_0_intake.py::parse_questionnaire`

Maps a digital questionnaire's answers to the stored covariate record: each field, the age from the date of birth, a smoking bin, and
a list of context modes. The record is stored with the manifest. No stage of the current chain reads these covariates or modes; the
stage names in its field table belong to an earlier engine.

*Code:* `stage_0_intake.py` line 480 (18 lines) · arguments: answers, intake_date · calls: not called on the traced runs · constants read: `COVARIATE_FIELDS`, `_CONTEXT_MODE_TRIGGERS`

### `stage_0_intake.py::_looks_like_cleartext_pii`

True when an identifier looks like a name rather than a hash: empty, containing a space or an @, shorter than 16 characters once
dashes and underscores are removed, or not purely letters and digits.

*Code:* `stage_0_intake.py` line 500 (9 lines) · arguments: patient_id · calls: array run 1

### `stage_0_intake.py::_write_manifest`

Writes the per-sample manifest as `patient_manifest_<run id>.json` in the manifest folder, when one is given.

*Code:* `stage_0_intake.py` line 511 (8 lines) · arguments: manifest_dir, record · calls: array run 1 · files named: `Write the immutable per-sample patient_manifest.json`

### `stage_0_intake.py::step_0_2_manifest_creation`

Step 0.2: builds the immutable manifest from the 0.1 record (and questionnaire answers, when given), quarantines a cleartext
identifier or a missing core field, and writes the file.

*Code:* `stage_0_intake.py` line 521 (40 lines) · arguments: step_0_1_result, questionnaire_answers, manifest_dir, intake_date · calls: array run 1

### `stage_0_intake.py::_sha256_file`

The SHA-256 of a whole file, read in 1 MB pieces.

*Code:* `stage_0_intake.py` line 569 (6 lines) · arguments: path · calls: array run 2

### `stage_0_intake.py::_prior_hashes_for_sentrix`

The pairs of file hashes earlier intakes recorded for the same Sentrix ID in the integrity log.

*Code:* `stage_0_intake.py` line 577 (14 lines) · arguments: integrity_log_path, sentrix_id · calls: array run 1

### `stage_0_intake.py::_log_integrity`

Appends one row to the integrity log (also used for the decision gate's verdict row).

*Code:* `stage_0_intake.py` line 593 (6 lines) · arguments: integrity_log_path, row · calls: array run 1

### `stage_0_intake.py::step_0_3_integrity_hash`

Step 0.3: hashes both files and compares them with earlier intakes of the same Sentrix ID. Identical bytes again is a re-transmission
and stops the run; the same Sentrix ID with different bytes is flagged as a fresh re-run and advances.

*Code:* `stage_0_intake.py` line 601 (37 lines) · arguments: record, grn_path, red_path, integrity_log_path · calls: array run 1

### `stage_0_intake.py::extract_control_probes`

Gets the control-probe medians by class from the IDAT pair through the QC hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`), the
same decoder Stage 1 uses. The array's own controls only.

*Code:* `stage_0_intake.py` line 659 (7 lines) · arguments: grn_path, red_path, array_type · calls: not called on the traced runs

### `stage_0_intake.py::validate_control_probes`

Step 0.4's arithmetic. Bisulfite: `bs1 / (bs1 + bs2)` against 0.80, where the QC hand-off passes in the matched-pair conversion
efficiency and its complement under the keys `bisulfite_conversion_I_median` and `bisulfite_conversion_II_median` (so the ratio is the
efficiency itself; the key names are older than the pair method). Hybridization: the spread, the larger of the high- and low-concentration
medians over the smaller, at least 2, with the medium control between them. A high control reading where the low one should (and the
reverse) with the spread intact is recorded as a swapped order, not a stop: on two whole slides of GSE184269 High and Low trade values
exactly (21,304 and 20,040; 6,514 and 6,845) while Medium is unchanged (12,822 and 12,834), which a failed hybridization cannot do. Extension: the green-channel (C and G) extension median over the red-channel (A and T) median, between 0.2 and 5;
the record calls these "meth" and "unmeth", which they are not. The three limits are read from `intake_thresholds_v2.json`; while a bisulfite
limit is marked provisional in that file, a bisulfite failure alone is recorded and the array advances (the rule before 2026-10-21). Records the
raw medians the gates tested.

*Code:* `stage_0_intake.py` line 668 (53 lines) · arguments: control_summary · calls: array run 2 · constants read: `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `EXTENSION_BALANCE_RANGE`, `HYB_HIGH_LOW_MIN_RATIO`

### `stage_0_intake.py::step_0_4_control_probe_validation`

Step 0.4: runs `validate_control_probes` on the hand-off's control summary and stamps the verdict and its numbers. With no decoder
available it marks the check deferred rather than inventing a pass.

*Code:* `stage_0_intake.py` line 723 (20 lines) · arguments: record, grn_path, red_path, control_summary · calls: array run 1

### `stage_0_intake.py::_normal_sf`

The normal survival function, 1 − Φ(z), from scipy when it is installed, otherwise from the complementary error function.

*Code:* `stage_0_intake.py` line 762 (9 lines) · arguments: z · calls: array run 2

### `stage_0_intake.py::compute_detection_p`

Per probe, the detection p-value 1 − Φ((I − μ_bg)/σ_bg), with μ_bg and σ_bg this array's negative-control background.

*Code:* `stage_0_intake.py` line 773 (6 lines) · arguments: probe_intensities, mu_bg, sigma_bg · calls: array run 2

### `stage_0_intake.py::validate_detection_p`

The share of probes with detection p ≤ 0.01, graded against the thresholds file: above 0.99 PASS, from 0.93 BORDERLINE, below FAIL.

*Code:* `stage_0_intake.py` line 781 (15 lines) · arguments: detection_p_array · calls: array run 2 · constants read: `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `DETECTION_P_THRESHOLD` · files named: `intake_thresholds_v1.json`

### `stage_0_intake.py::step_0_5_detection_pvalue_qc`

Step 0.5: computes and grades detection. Without intensities the check is deferred, and a deferred detection never advances.

*Code:* `stage_0_intake.py` line 798 (22 lines) · arguments: record, probe_intensities, neg_control_stats · calls: array run 1

### `stage_0_intake.py::validate_bead_count`

The share of probes with at least 3 beads: 0.995 or more PASS, otherwise a warning. Never stops an array.

*Code:* `stage_0_intake.py` line 830 (8 lines) · arguments: bead_counts, min_beads · calls: array run 1 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`

### `stage_0_intake.py::step_0_6_bead_count_qc`

Step 0.6: grades the bead counts and flags a low count. Always advances.

*Code:* `stage_0_intake.py` line 840 (14 lines) · arguments: record, bead_counts · calls: array run 1

### `stage_0_intake.py::validate_call_rate`

The share of probes passing both detection and bead count, graded: 0.98 or more PASS, from 0.93 BORDERLINE, below FAIL.

*Code:* `stage_0_intake.py` line 864 (9 lines) · arguments: n_passing_both, n_total · calls: array run 2 · constants read: `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`

### `stage_0_intake.py::step_0_7_call_rate`

Step 0.7: combines the detection and bead masks probe by probe and grades the call rate. Without masks the check is deferred, and a
deferred call rate never advances.

*Code:* `stage_0_intake.py` line 875 (22 lines) · arguments: record, detection_pass_mask, bead_pass_mask · calls: array run 1

### `stage_0_intake.py::step_0_7b_platform_coverage`

Step 0.7b: stamps the platform tag. On a 450K array it stops below 80 % of the reference sites; on EPIC it records `NA_EPIC` and passes.

*Code:* `stage_0_intake.py` line 906 (25 lines) · arguments: record, reference_cpg_coverage · calls: array run 2 · constants read: `HM450_COVERAGE_MIN`

### `stage_0_intake.py::predict_sex`

Female when the median log2 Y intensity minus the median log2 X intensity is below −2, otherwise male (minfi's getSex rule).

*Code:* `stage_0_intake.py` line 940 (2 lines) · arguments: log2_x_median, log2_y_median, cutoff · calls: array run 1 · constants read: `SEX_GETSEX_CUTOFF`

### `stage_0_intake.py::validate_sex`

Compares the predicted sex with a declared one. No declared sex: the array's sex is recorded and nothing is compared. A mismatch stops
the array.

*Code:* `stage_0_intake.py` line 944 (7 lines) · arguments: log2_x_median, log2_y_median, declared_sex · calls: array run 1

### `stage_0_intake.py::step_0_8_sex_check`

Step 0.8: runs the sex check on the hand-off's X and Y medians and stamps the result.

*Code:* `stage_0_intake.py` line 953 (15 lines) · arguments: record, sex_intensities · calls: array run 1

### `stage_0_intake.py::_fmt`

Formats a measured number for an explanation, or says "not measured".

*Code:* `stage_0_intake.py` line 982 (2 lines) · arguments: x, nd · calls: array run 2

### `stage_0_intake.py::explain_stage0`

Builds the five-part explanation of every check that stopped, penalized or deferred the array (measured on this array, the limit, the
limit's source and whether it is provisional, what it means, what to do). Changes no decision.

*Code:* `stage_0_intake.py` line 984 (60 lines) · arguments: record, hard_fail, borderline, deferred · calls: array run 2 · constants read: `BEAD_COUNT_MIN`, `BEAD_PASS_FRACTION`, `BS_CONVERSION_MIN`, `BS_THRESHOLD_CALIBRATED`, `CALL_RATE_BORDERLINE`, `CALL_RATE_PASS`, `DETECTION_BORDERLINE_FRACTION`, `DETECTION_PASS_FRACTION`, `EXTENSION_BALANCE_RANGE`, `HM450_COVERAGE_MIN`, `HYB_HIGH_LOW_MIN_RATIO`, `SEX_GETSEX_CUTOFF`, `_TH_SRC` · files named: `Runtime Matrices/Intake/intake_thresholds_v2.json`

### `stage_0_intake.py::explain_stage0.add`

Adds one explanation to the list.

*Code:* `stage_0_intake.py` line 986 (2 lines) · arguments: check, outcome, measured, limit, source, provisional, means, todo · calls: array run 2

### `stage_0_intake.py::step_0_9_decision_gate`

Step 0.9: sorts every step's outcome into hard failures, penalties and deferrals and gives the verdict (QUARANTINE, PROCEED_WITH_PENALTY
or PROCEED). A deferred detection or call rate is a hard failure; a deferred bead count is recorded.
Attaches the explanations and logs the verdict.

*Code:* `stage_0_intake.py` line 1046 (80 lines) · arguments: record, verdict_log_path · calls: array run 2

## 3. The QC hand-off: `stage_0_1_qc_handoff.py`

**What it is for.** Stage 0's gates 0.4 to 0.8 need intensities, and intensities come from decoding the IDAT files. This module decodes
the pair just far enough to give those gates real numbers from the array itself, using the same decoder Stage 1 uses (methylprep) and
the control addresses of the array's own manifest. Nothing from any other sample enters.

**Label: standard method** (Illumina's control probes, read by design type the way the minfi package reads them).

**What it returns.**

- **Control summary.** Bisulfite conversion: for each matched BS Conversion I pair (C1 with U1, C2 with U2, ...), the converted signal
  over converted plus unconverted, C/(C+U), and the median over the pairs. Each pair is read in its own channel, the one where the pair's
  total C + U is brighter: on EPIC v1, pairs 1 and 2 are green probes and 3 to 5 red. Until 2026-10-20 every pair was read in green, which
  put three of five at background on both probes and gave every healthy array about 0.7 (on 61 healthy arrays, mostly GSE318173: median
  0.738 as written, 0.970 read correctly, range 0.927 to 0.978; `doors/data/PROC_STAGE0_04/bs_channel_explore.py`). Read correctly, other
  laboratories differ: GSE315366 0.899 to 0.941, GSE184269 0.576 to 0.975 (`doors/data/GROUPA_COMMISSIONING/diag_ctrl.py`). The sum C + U is used to find the channel because
  it stays bright whether conversion worked (C bright) or failed (U bright). Each pair's channel and contrast are recorded. Hybridization:
  the medians of the high-, medium- and low-concentration controls (green). Extension: the median of the C and G controls (green) and of the A and T controls (red).
- **Negative-control background.** The negative controls summed over both channels, summarized robustly as the median and 1.4826 times
  the median absolute deviation (MAD). The mean and standard deviation are returned beside them. The robust form was adopted on
  2026-09-23 after a few EPIC negative controls carrying real signal (one at 29,110 against a median of 134) pushed the standard
  deviation to 1,676 and failed half the probes of a good array; on a clean 450K array the two estimates agree.
- **Per-probe total intensity.** A type II probe is read at its one address in both channels; a type I probe at its two addresses in
  its own color only (the other color would add only background).
- **Bead counts**, per probe the smaller of the two channels' counts (and of the two addresses for type I).
- **Sex intensities**: log2 of the median total intensity of the X probes and of the Y probes.

**How to break it.** A truncated or corrupt IDAT makes the decoder fail; `run_sample.py` then quarantines the array as a corrupt IDAT.
A missing manifest on the machine is an environment failure (exit code 3), never a judgment of the array.

### `stage_0_1_qc_handoff.py::_manifest`

Loads the array type's manifest from methylprep once and keeps it for the rest of the run.

*Code:* `stage_0_1_qc_handoff.py` line 26 (9 lines) · arguments: array_type · calls: array run 2 · constants read: `_MANIFEST`

### `stage_0_1_qc_handoff.py::_channel_means`

Reads one IDAT file (gzipped or plain) with methylprep and returns the mean intensity at every address and, when asked, the bead count.

*Code:* `stage_0_1_qc_handoff.py` line 37 (11 lines) · arguments: path, channel, want_beads · calls: array run 2

### `stage_0_1_qc_handoff.py::_median`

The median intensity over a list of control addresses, ignoring addresses the file lacks; nothing if none are present.

*Code:* `stage_0_1_qc_handoff.py` line 50 (3 lines) · arguments: series, addresses · calls: array run 4

### `stage_0_1_qc_handoff.py::decode_qc_inputs`

Builds everything Stage 0's gates read, as listed above, from the two files and the manifest. Returns the probe identifiers in the same
order as the intensities and bead counts, so Stage 1's call rate can line them up later.

*Code:* `stage_0_1_qc_handoff.py` line 55 (139 lines) · arguments: grn_path, red_path, array_type · calls: array run 1

### `stage_0_1_qc_handoff.py::decode_qc_inputs.ctl_addr`

The manifest addresses of one control type, optionally narrowed to extended types containing (or not containing) a word.

*Code:* `stage_0_1_qc_handoff.py` line 62 (9 lines) · arguments: control_type, contains, excludes · calls: array run 5

### `stage_0_1_qc_handoff.py::decode_qc_inputs._at`

The intensity at each probe's address, with a gap where the probe has no such address.

*Code:* `stage_0_1_qc_handoff.py` line 149 (6 lines) · arguments: series, addr · calls: array run 8

### `stage_0_1_qc_handoff.py::decode_qc_inputs.sex_median`

log2 of the median total intensity over the X (or Y) probes the array measured, and how many probes that was.

*Code:* `stage_0_1_qc_handoff.py` line 180 (5 lines) · arguments: which · calls: array run 2

## 4. Stage 1, calibration: `stage_1_idat_calibration.py`

**What it is for.** Turning one array's raw intensities into one calibrated methylation value (beta) per site, from this array alone,
and removing every probe that reads at background. Without calibration the methylated peak sits near 0.85 instead of 0.95 and every
later number moves.

**Label: standard method.** Noob (normal-exponential out-of-band background correction with dye-bias correction), as implemented in the
methylprep package, run on one array at a time with its own control and out-of-band probes. No cohort and no reference enters.

**The formula.** For each probe, beta = M / (M + U + offset) after noob correction of the methylated (M) and unmethylated (U)
intensities. Detection: methylprep's poobah p-value, each probe against this array's own out-of-band background; a probe with p above
0.05 is removed before any stage reads a value ("no measurement at background").

**What it pins.** The commissioned numbers were made with methylprep 1.7.1 and three Illumina manifests, fixed by their SHA-256
(`METHYLPREP_VERSION`, `MANIFEST_SHA256`). Stage 1 refuses to run on another version or a changed manifest, before and after
methylprep fetches its files. Development work can override this with `IAM_ALLOW_UNPINNED_STAGE1=1`; the results are then not the
commissioned chain's.

**What else it returns.** The detection summary (probes measured, detected, removed); the control medians; and the array's own noise,
from its SNP (rs) probes. Those probes cluster at 0, 0.5 and 1 by genotype, so their spread around each cluster is pure measurement
noise: σ²(β) = a + b·β(1 − β), with a the mean variance of the two homozygous clusters and b = 4·(variance of the heterozygous
cluster − a). The sky uses this as the array's own noise term.

**How to break it.** Run with methylprep 1.7.2, or edit one byte of a cached manifest: Stage 1 must refuse. An EPIC v2 pair reaching
Stage 1 is refused as an array type it does not calibrate; a truncated file is quarantined as a corrupt IDAT.

### `stage_1_idat_calibration.py::_check_pins`

Refuses to run unless methylprep is version 1.7.1 and every cached manifest matches its pinned SHA-256 (or the development override is
set). Called before calibration and again after methylprep fetches any missing manifest.

*Code:* `stage_1_idat_calibration.py` line 38 (11 lines) · arguments: none · calls: array run 2 · constants read: `MANIFEST_SHA256`, `METHYLPREP_VERSION`

### `stage_1_idat_calibration.py::_ensure_methylprep`

Imports methylprep, installing the pinned version into the running Python once if it is missing, and explains what to do when the
install or import fails.

*Code:* `stage_1_idat_calibration.py` line 49 (43 lines) · arguments: none · calls: array run 1 · constants read: `METHYLPREP_VERSION` · files named: `pip install -r Biological_Physics/MethylPhys/chain/requirements.txt`

### `stage_1_idat_calibration.py::_detect_array_type`

Reads the green IDAT with methylprep and maps its address count to methylprep's array type; also returns the barcode.

*Code:* `stage_1_idat_calibration.py` line 94 (10 lines) · arguments: grn_path · calls: array run 1

### `stage_1_idat_calibration.py::calibrate_idat_to_beta`

Copies the pair into a private temporary folder with a one-row sample sheet (so methylprep processes this array alone), runs noob with
poobah detection and the control and SNP probes exported, keeps the cg probes with poobah p ≤ 0.05, and returns the betas with the
record described above. The temporary folder is always deleted.

*Code:* `stage_1_idat_calibration.py` line 106 (106 lines) · arguments: grn_path, red_path, array_type, verbose, mask_detection, return_mask, return_snp · calls: array run 1 · files named: `_processed.csv`, `control_probes.pkl`, `samplesheet.csv`, `the tag beta_scale_maps_v1.json`

### `stage_1_idat_calibration.py::calibrate_idat_to_beta.med`

The median of one control type in one channel, from methylprep's exported control table.

*Code:* `stage_1_idat_calibration.py` line 176 (2 lines) · arguments: t, col · calls: array run 7

### `stage_1_idat_calibration.py::_stage_idat`

Copies an IDAT into the staging folder under the name methylprep expects, unzipping it if it is gzipped.

*Code:* `stage_1_idat_calibration.py` line 214 (8 lines) · arguments: src, dst · calls: array run 2

## 5. The conductor: `conductor_v3.py`, and Met-A's reading: `stage_m_met_a.py`

**What it is for.** The conductor takes one array's calibrated values and makes the array reading: the platform check, Stage T step 1
(self-tare II), Stage A (the cell mix), Stage M (Met-A), the noise index, Stage T step 2 (the same-run tare), and Stage MC (the C-Score).
It returns one record per stage, each with its reason when it gives no number.

**The order inside `run_neutrophil`.**

1. Refuse a specimen outside the list (as `run_sample.py` already did) and an array that is not EPIC v1.
2. **Stage T step 1, self-tare II**: map this array's values onto the reference arrays' scale by its own fixed sites (below).
3. Isolated neutrophils: **Stage M** against the healthy reference. Whole blood (and buffy coat, and constructed mixtures): **Stage A**,
   then **Stage M** against the expectation for this blood's own mix.
4. The **noise index** N, from the unmapped values.
5. **Stage T step 2**, the median tare against the same-run healthy references.
6. The **noise gate**: no gauge state when N cannot be formed, or when N is above the reference arrays' range and the reading is not tared.
7. The scope label: commissioned on isolated neutrophils; preliminary (labeled development) on whole blood and buffy coat.
8. **Stage MC**, the C-Score, and its own same-run tare; the commissioned label on whole blood.

### The physics: what Met-A measures

**Label: from IAM's physics.** A cell holds a methylation pattern against thermal noise by paying for it. A site held fully methylated
or fully unmethylated carries no uncertainty; a site that has drifted toward 50 % carries up to one bit. The entropy of one site's
beta value is

  H(β) = −β log₂ β − (1 − β) log₂(1 − β),

and Met-A is the mean entropy at the cell type's identity sites divided by the healthy reference:

  isolated neutrophils: Met-A = mean_i H(β_i) / H_ref, with H_ref = 0.33026 bits (6 healthy neutrophil arrays);
  whole blood: Met-A = mean_i H(β_i) / mean_i H(e_i), with e_i = Σ_g f_g μ_g,i.

Here f_g is this blood's share of cell group g (Stage A), and μ_g,i is the healthy mean of group g at site i, from purified cells. In
whole blood the denominator is what a healthy blood with exactly this person's cell mix would read, so a change in the mix does not move
Met-A; only a change in how well the cells hold their marks does.

**The gauge.** Healthy is 1 (the healthy reference, measured). Normal is each cell's measured healthy width on each instrument (the atlas's own donors read against the atlas, confirmed by the A-score database), never a chosen band; the earlier 0.95 to 1.05 design tolerance is superseded (author), and the printed state still uses it until the measured widths replace it. The floor (derived: thermal kicks against about one
ATP per site, M = 20.94) sits millions of times below any reading; the ceiling (derived: one bit per site, every identity site
scrambled) sits about three times above healthy. What the physics fixes is the scale and its two ends. The healthy reference is measured
once and never fitted to the sample. Its level at fully held sites follows from M alone: the paid hold of two ATP per restore (one S-adenosylmethionine) loses
0.0445 per site, so a fully held site reads 1 − ε0 in sequencing units (derived, the same quantity as IAM-A's derived
baseline; blind test PHY-084 met), and the atlas's measured transfer maps it to about 0.9545 on the array scale,
0.0044 below the array's saturation 0.959 and inside the transfer's calibration error, so no array test can resolve it. Met-A's identity
sites are cell-type marks, not fully held sites, and their healthy reference stays measured. The chain reads against the measured reference;
the derivation explains it and does not replace it.

**The ceiling caveat (`_ceiling`).** H(β) peaks at β = 0.5. Met-A rises with loss of the pattern only while the methylated sites stay
above 0.5. Every reading records the mean β of the cell's methylated sites and flags `past_entropy_ceiling` when it has fallen below 0.5,
because past that point further loss lowers Met-A.

**The physics, step by step.** From IAM's Law to Met-A's reading on arrays, each step marked with its kind and register row.

1. **M and the gauge's ends (derived, A1 and M1).** M = ΔG_ATP / RT = 20.94. The floor is the disorder of a mark lost with probability
   1/(1 + e^M), 2.5e-08 bits; the ceiling is one bit per site, H(β) = 1 at β = 0.5, every identity site scrambled.
2. **The bit (calculated).** A site's disorder is H(β) = −β log₂ β − (1 − β) log₂(1 − β), with β clipped to [10⁻⁶, 1 − 10⁻⁶] (`stage_m_met_a._H`).
   A held site sits near 0 or 1; a loss moves it toward 0.5, so its disorder rises.
3. **The healthy level at fully held sites (derived, S1 and M2).** The paid hold of M alone loses ε0 = 0.0445 per site (A4),
   so a fully held site reads 1 − ε0 = 0.95546 in sequencing units (S1, derived as A6; PHY-084 met), disorder H(ε0) = 0.2627 bits. Arrays read
   it through the atlas's measured transfer (calibrated): about 0.9545 array units, 0.0044 below the array's
   saturation 0.959 and inside the transfer's calibration error. No array test resolves it (M2), so on arrays the atlas stays the reference and
   the derived value is kept beside it, flagged not resolvable.
4. **Identity sites are cell-type marks (measured, M4).** Met-A reads each cell at its own identity sites (healthy level 0.75 to 0.95 or 0.05 to
   0.25, spread at most 0.05, at most 3,000 a side). They are not fully held, so their healthy disorder is a fact of the cell type, measured on
   the atlas; part of it may follow once each type's own level is derived (A8).
5. **The healthy reference (measured, M5).** For isolated neutrophils H_ref = 0.33026 bits, the mean over 6 purified
   healthy EPIC arrays (`metA_floors_v1_3.json`). In whole blood the reference is the expectation e_i = Σ_g f_g μ_g,i for this blood's own mix
   (Stage A, standard method). Met-A = mean_i H(β_i) / H_ref, or / mean_i H(e_i).
6. **The array's own response (calibrated, M3).** Each array's offset at unmarked sites (about 5 points, nearly constant across types) is the
   meter, measured per array and never derived; on the unmarked side the cells' own contribution is the writer's one selection step,
   1/(1 + D) with D 47.4 to 80, 0.0123 to 0.0207 (A15, accounted). The noise index N reads the instrument at 48,528 sites no cell has anything
   to say about (calibrated).
7. **Normal (measured).** Normal is each cell's measured healthy width on the atlas's own donors (`donor_widths_v1.json`); the 0.95 to 1.05
   tolerance is superseded and still printed by `stage_m_met_a.read` until the widths replace it.
8. **The development path (`stage_m_met_a.derived_baseline`).** The healthy mark from the cell's own held errors: 1 − ε_loss where the cell holds
   the mark and ε_gain where it is held absent, each put on the array scale through the transfer (`transfer_v21_smooth.json`), the mark the
   mean entropy over the sites. Nothing fitted; the array's own blur not included.
9. **What is open.** Each type's own level (A8); the per-type comparison of the derived held level with the Sequencing Atlas (S1, carried to
   A8); printing the derived held height beside each array reading (a chain item).


### Why these sites: Met-A's site rule

Met-A reads each cell at that cell's own identity sites. The rule is fixed (`stage_m_met_a.py`; for every atlas cell,
`stage_metaa_atlas.identity_sites`): healthy spread at most 0.05, healthy level 0.75 to 0.95 (held methylated) or 0.05 to 0.25 (held
unmethylated), at most 3,000 a side, the tightest first. Each part of it is one of three kinds:

- **Held sites. Label: from IAM's physics.** Disorder H(β) is lowest where a mark is held and highest at 0.5. A loss of hold moves a held
  site toward 0.5, so its disorder rises; a site already near 0.5 is at the ceiling, where a loss cannot show (the ceiling caveat above). So
  Met-A reads the sites a cell holds, on both sides: held methylated and held unmethylated, because a loss moves either toward 0.5.
- **A small healthy spread. Label: from the reading itself.** Where healthy people differ, their ordinary differences look like a loss. Sites
  where healthy people agree keep the healthy reading tight, so a loss stands out against it.
- **The window's ends (0.95 and 0.05) and the count (3,000 a side). Label: operational, stated.** Neither is derived. The ends are part of the
  rule as first written, and no reason for them is recorded; the count gives a stable reading. Both are tested rather than defended; the ends
  against windows reaching 1 and 0 (to-do 39.8).

**Tested: 6,000 identity sites against every atlas site (DEV-EACH-CELL-01 exploration 4).** Reading every site, each counted by the shift a
loss of the cell would make there, is far more sensitive to a loss across a cell's whole pattern (45 to 187 of its own errors, against 6 to 30
for the identity sites) but moves every cell alike, so it cannot say which cell is losing its pattern. The identity sites say which cell (other
cells move 15 to 26 % as much) when the loss sits at that cell's own sites. A loss across a cell's whole pattern cannot be put on one cell by
either, because most held sites are held by its relatives too. So Met-A for each cell keeps the identity sites; sites specific to each cell
are tested next (to-do 39.9).

**Each cell has its own set.** On the atlas, every cell's sites come from its own column by the same rule, so the CD4 T cells' sites differ
from the neutrophils'. Sets overlap where cells hold the same sites, and that overlap is why a loss in one cell moves its relatives a little:
in a whole blood, up to 26 % of the planted cell's move (DEV-EACH-CELL-01 synthetic test 2).

### Stage A, the cell mix (whole blood)

**Label: standard method** (reference-based deconvolution by non-negative least squares, used in the field for over a decade).

The blood's shares of 8 cell groups (neutrophils, eosinophils, basophils, monocytes, B, NK, CD4 T, CD8 T) are solved from 963 marker
sites: minimize |y − M f| with f ≥ 0, then f is scaled to sum to 1. M holds each group's mean β at each marker, from 91 purified-cell
EPIC arrays of two published purified-blood-cell studies (GSE110554 and GSE167998), calibrated by this chain's own Stage 1.

- **Why its own reader, not the atlas:** the expectation e must come from the same platform and the same reference as the profiles it
  multiplies, or any platform difference leaks into Met-A. IAM-Atlas v2.1 and the Deconvolver report the shares separately (Stage D21).
- **Why no marker is an identity site:** solving the mix from the sites Met-A reads would let a change in the cells' marks move the mix
  that is supposed to explain them.
- **Gate:** at least 90 % of the 963 markers measured (867); otherwise no mix, no Met-A, and the reason gives the count.
- **Built from, tested on:** built from the 91 purified arrays (`chain_tests/blood_comp.py`). Rebuilt from GEO by the builder's own rule
  on 2026-10-03, reproducing every group mean to 5 × 10⁻⁶, and again from a fresh clone (`metaa_rebuild`, PASS 2026-10-18).

### Stage M, Met-A

- **Identity sites:** 6,000 EPIC sites the healthy neutrophil holds fixed, chosen on the purified neutrophil arrays (across-array SD
  ≤ 0.05; mean β 0.75 to 0.95, or 0.05 to 0.25; up to 3,000 per state). The isolated and whole-blood paths read the same 6,000.
- **The healthy reference (H_ref = 0.33026 bits):** the mean entropy at those sites on 6 purified healthy neutrophil arrays, each
  physical array counted once. The file holding it is still named `metA_floors_v1_3.json`, and its key is `floor`; it is the middle
  mark, not the floor.
- **Held-out precision:** each of the 6 arrays read with the sites re-chosen on the other 5 (`metA_floors_v1_3_loo.csv`) gives 0.983 to
  1.045. This is printed with every isolated reading.
- **Gate:** at least 90 % of the 6,000 sites measured; otherwise no Met-A, and the reason gives the count. In whole blood, a neutrophil
  share of 0 also withholds the reading.
- **Detection at this person's mix:** the reading records how far a 1 % loss of the neutrophils' pattern, planted at this blood's own
  neutrophil share, would move it (`shift_per_1pct_loss`). Stage T turns this into the person's detection limit.
- **The departure map** for the C-Score: per site, z_i = (H(β_i) − H(ref_i)) / s_i, with ref the healthy neutrophil (isolated) or the
  expectation e (blood), and s_i the shrunk healthy SD of H among neutrophils.

### The noise index and its gate

The array's own noise is read where no blood cell has anything to say: 48,528 EPIC sites that every purified blood group holds fixed (each
group's mean ≤ 0.03 or ≥ 0.97 and SD ≤ 0.02; none an identity site). N is their mean entropy, read before self-tare II. **Gate:** with fewer
than 90 % of these sites measured the noise is unknown and no gauge state is shown, tared or not. With N above N_max = 0.149 (the top of the
reference arrays' range, `noise_gate_EPIC_v1.json`), the state is shown only after the same-run tare.

### Stage T, the tare

**Retired.** Healthy comes only from the atlases; Stage T runs only when `--ref-table` or `--slide-ref-A` is given, to rebuild the first commissionings' records, and the gate refuses a reference built from other samples for any new reading. Kept below as it was built.

**Label: standard method** (normalization against references run the same way).

- **Step 1, self-tare II** (`dev_stages.selftare_map`, on every reading since 2026-10-04). Each probe design (type I, type II) is mapped
  onto the 6 reference arrays' scale by this array's own anchors: β' = L_r + (β − L)(U_r − L_r)/(U − L), with L and U this array's mean β
  over its own fixed low and high sites of that design, and L_r and U_r the same means on the reference arrays (type I 0.0186 and 0.9820;
  type II 0.0555 and 0.9486). It maps only the 6,963 sites Stages A and M read (the 6,000 identity sites and the 963 markers). A design
  whose anchors are missing, or closer than 0.1, is left unmapped. No other array enters; nothing is fitted.
- **Step 2, the median tare.** A_rel = A / median(A of at least 3 healthy arrays run the same way: same slide, else same batch). Normal is
  0.95 ≤ A_rel ≤ 1.05. The reference spread is the SD of the references' A over their median. The **detection limit** is 2 × spread /
  shift per 1 % loss: the smallest loss of the cell's pattern, in percent, this person's reading could show. A sample that is itself in
  the reference list is excluded from it. Fewer than 3 references: the reading is printed as untared, and the reason says what to give.
- **Why:** untared Met-A in whole blood carries a laboratory offset of several percent. The tare cancels it, which is why Met-A needs no
  cohort: it compares the person with a prediction for their own blood and centers it on their own run.

### Stage MC, the C-Score

**Label: standard statistic (register C1: statistics, not IAM's Law), read on the departure map Met-A's physics makes.** Drift that comes from holding marks against noise is scattered: each site
fails on its own. Loss that comes from a process acting on a region, such as a tumor's, is clustered along the genome.

- **The statistic:** take the departure map z in genomic order, average blocks of 10 consecutive sites, and take the variance of the
  block means times the block length over the variance of z. For scattered departures it is 1. Divided by the healthy median of that
  number (1.0103, from the 6 reference arrays held out one at a time), healthy reads about 1.
- **Its tare:** the same rule as Met-A's: C_rel = C / median(C of at least 3 same-run healthy references).
- **The band (commissioned on whole blood, 2026-10-15):** 0.877 to 1.152, the 2.5 to 97.5 % range of tared C on 617 healthy arrays from
  19 studies. One-sided: above 1.152 is a clustered departure; below 0.877 is less clustered than healthy, not a departure.

**How to break it.** Plant a 1 % loss of the neutrophils' pattern scattered across sites: Met-A must move and the C-Score must not (seen
on Group-A's 278 arrays: Met-A by up to 0.036, the C-Score by at most 0.006). Plant the same loss in runs of neighboring sites: the C-Score
must rise. Give 2 references: the tare must refuse. Remove 15 % of the identity sites: Met-A must withhold its reading and say how many
were measured.

**The physics, step by step.** Where IAM's Law is in the C-Score, and where it is not.

1. **The departure map carries the physics (from Met-A).** z_i = (H(β_i) − H(ref_i)) / s_i at the identity sites in genomic order: each site's
   disorder against its healthy reference, the quantity of Section 5.
2. **The statistic is standard (calculated, C1).** Average blocks of 10 consecutive sites; C = variance of the block means × 10 / variance of z.
   For independent departures C = 1, with a band 0.894 to 1.111 from independence over 6,000 sites in blocks of 10: statistics, not IAM's Law.
   Holding marks against thermal noise predicts scattered failure, so a clustered departure is what the statistic is there to find.
3. **The healthy baseline (measured).** 1.0103, from the 6 reference arrays held out one at a time; the commissioned band 0.877 to 1.152 on
   617 healthy whole bloods, its upper edge wider by whole blood's measured clustering. Each cell's limit factor (`cell_factors_v1.json`) and the
   450K platform term are measured.
4. **What is open.** Reading the healthy clustering from its derivation rather than a stored number (Part I action).

### `conductor_v3.py::ref`

Loads the neutrophil reference (`neutrophil_reference_v1_2.json`) once: the 6,000 sites in genomic order, the healthy mean and shrunk SD
of H at each, and the C-Score's block length and healthy baseline.

*Code:* `conductor_v3.py` line 62 (4 lines) · arguments: none · calls: array run 4 · constants read: `RM`, `_REF` · files named: `neutrophil_reference_v1_2.json`

### `conductor_v3.py::noise_sites`

Loads the 48,528 noise sites once.

*Code:* `conductor_v3.py` line 70 (6 lines) · arguments: none · calls: array run 1 · constants read: `RM`, `_NS` · files named: `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`, `noise_sites_EPIC_v1.json`

### `conductor_v3.py::noise_gate`

Loads N_max and its rule once.

*Code:* `conductor_v3.py` line 78 (5 lines) · arguments: none · calls: array run 1 · constants read: `RM`, `_NG` · files named: `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`, `noise_gate_EPIC_v1.json`

### `conductor_v3.py::noise_index`

N = mean H(β) over the noise sites this array measured, or nothing when fewer than 90 % were measured; with the counts.

*Code:* `conductor_v3.py` line 84 (6 lines) · arguments: beta · calls: array run 1 · constants read: `H`, `MIN_NOISE_FRACTION`

### `conductor_v3.py::_bc`

Loads the blood composition matrix (`blood_composition_EPIC_v1.json`) once: the 963 markers and 8 group means at each, and the 8 groups'
healthy means at the 6,000 identity sites.

*Code:* `conductor_v3.py` line 92 (4 lines) · arguments: none · calls: array run 4 · constants read: `RM`, `_BC` · files named: `blood_composition_EPIC_v1.json`

### `conductor_v3.py::min_markers`

The number of markers Stage A needs: 90 % of 963, rounded up (867).

*Code:* `conductor_v3.py` line 97 (2 lines) · arguments: none · calls: array run 2 · constants read: `MIN_MARKER_FRACTION`

### `conductor_v3.py::stage_a_composition`

Stage A: solves the 8 shares by non-negative least squares on the measured markers and scales them to sum to 1; records the fit's mean
absolute residual. Refuses, with the count, below 867 markers.
How it agrees with the chain's other cell-mix reader, Stage D21 (the IAM-Deconvolver): on Group-A's 278 healthy whole bloods (GSE318173), the
neutrophil shares correlate at 0.960; the Deconvolver reads 0.021 lower on average, and 95 % of arrays agree within -0.113 to +0.071
(`doors/data/GROUPA_COMMISSIONING/two_readers_01_summary.txt`, 2026-10-22). Stage A's share sets Met-A's whole-blood expectation; the
Deconvolver's shares are the ones printed on the report.

*Code:* `conductor_v3.py` line 100 (15 lines) · arguments: beta · calls: array run 1

### `conductor_v3.py::_clustering`

The C-Score's statistic on a departure map: variance of block means of 10 sites (× √10) over the site variance; nothing with fewer than 10
blocks.

*Code:* `conductor_v3.py` line 116 (5 lines) · arguments: z, w · calls: array run 1

### `conductor_v3.py::_state`

Normal from 0.95 to 1.05, otherwise above or below Normal: the first commissioning's band, still printed until each cell's healthy width, measured from the atlas's own donors, replaces it.

*Code:* `conductor_v3.py` line 122 (2 lines) · arguments: A · calls: array run 1

### `conductor_v3.py::stage_m_blood`

Stage M in whole blood: the expectation e for this blood's mix, Met-A = mean H(β) / mean H(e) over the measured identity sites, the shift a
1 % loss would make at this neutrophil share, the ceiling check and the departure map. Withholds the reading, with the reason, when there is
no mix, no neutrophils, or under 90 % of the sites.

*Code:* `conductor_v3.py` line 125 (26 lines) · arguments: beta, comp · calls: array run 1 · constants read: `BUILD`, `H`

### `conductor_v3.py::_ceiling`

The mean β of the neutrophil's methylated identity sites, and whether it has fallen below 0.5, past which Met-A no longer rises with loss.

*Code:* `conductor_v3.py` line 152 (7 lines) · arguments: x, mu · calls: array run 1

### `conductor_v3.py::stage_m_isolated`

Stage M on isolated neutrophils: Met-A against the healthy reference (`stage_m_met_a.read`), the 1 % shift, the ceiling check and the
departure map against the healthy neutrophil. Keeps the untared state as `state_own_floor` and marks the reading untared until Stage T.

*Code:* `conductor_v3.py` line 160 (12 lines) · arguments: beta · calls: not called on the traced runs · constants read: `H`

### `conductor_v3.py::stage_mc_cscore`

Stage MC: the C-Score from the departure map, divided by the healthy baseline, with the share of sites more than 3 SD out. With no map,
no C-Score and the reason.

*Code:* `conductor_v3.py` line 173 (12 lines) · arguments: z · calls: array run 1 · files named: `divided by the healthy median clustering of neutrophil_reference_v1_2.json`

### `conductor_v3.py::_ref_records`

Turns the same-run references into records (A, neutrophil share, N, C), skipping empty values and the sample itself.

*Code:* `conductor_v3.py` line 186 (13 lines) · arguments: ref_A, sample_id · calls: array run 2

### `conductor_v3.py::stage_t_tare`

Stage T step 2: the median tare of Met-A, the reference spread and the detection limit; untared, with the reason, under 3 references.

*Code:* `conductor_v3.py` line 200 (16 lines) · arguments: A, ref_A, shift_1pct, sample_id · calls: array run 1 · constants read: `MIN_REFS`

### `conductor_v3.py::stage_t_cscore`

The C-Score's same-run tare and its call against the band (above 1.152 a clustered departure).

*Code:* `conductor_v3.py` line 219 (18 lines) · arguments: cs, ref_A, sample_id · calls: array run 1 · constants read: `CSCORE_COMMISSIONING`, `CSCORE_DEV_BAND`, `MIN_REFS`

### `conductor_v3.py::stage_t_selftare_ii`

Stage T step 1: runs `dev_stages.selftare_map` on the array and returns the mapped values with the anchors and maps; unmapped values and a
NOT_RUN record if the map file is missing.

*Code:* `conductor_v3.py` line 238 (9 lines) · arguments: beta · calls: array run 1

### `conductor_v3.py::platform_refusal`

Nothing for an EPIC v1 array; otherwise the refusal: another array type from Stage 0, EPIC v2 probe names, or 700,000 probes or fewer.

*Code:* `conductor_v3.py` line 248 (10 lines) · arguments: beta, array_type · calls: array run 1 · constants read: `ACCEPTED_ARRAY_TYPES`

### `conductor_v3.py::plant_neutrophil_loss`

A constructed 1 % loss for tests, Group-A's sealed rule: at the neutrophil sites, beta + f_neu x 1 % x (0.5 - mu_NEU), f_neu the array's own
neutrophil fraction. Used by the series runner's `--plant neutrophil`; never by a reading.

*Code:* `conductor_v3.py` line 315 (6 lines) · arguments: beta, f_neu, loss · calls: not called on the traced runs

### `conductor_v3.py::run_neutrophil`

The whole conductor, in the order above. Returns the array reading's records (mix, Met-A, tare, C-Score) and the departure map for the sky. Since the healthy reference comes only from the atlases, this first-commissioning path runs on every array, flagged: its neutrophil Met-A and whole-blood C-Score carry their first commissioning as `commissioning_first` with the status 'being re-commissioned against IAM-Atlas v2.1 alone', never as a current commissioning; the chain's reading of each cell is `cells_found` and `met_a_each_cell`. Group-A still reads this path's whole-blood Met-A (to be moved to the per-cell readings).

*Code:* `conductor_v3.py` line 259 (54 lines) · arguments: beta, specimen, ref_A, array_type, sample_id · calls: array run 1 · constants read: `BUILD`, `CSCORE_COMMISSIONING`, `METAA_COMMISSIONING`, `MIN_NOISE_FRACTION`

### `dev_stages.py::_json`

Loads a file from `Runtime Matrices/Development` once and keeps it; the self-tare matrix's old name reads the moved file
(`stage_t_selftare._json`), so older scripts still run.

*Code:* `dev_stages.py` line 38 (7 lines) · arguments: name · calls: array run 1 · constants read: `DEVDIR`, `_C` · files named: `dev_selftare_typeII_EPIC_v1.json`

### `stage_t_selftare.py::curve_cal`

Each array's response curve, per probe design, measured on its own calibrators and mapped onto IAM-Atlas v2.1 before the cell shares (`run_sample.py --dev-curve-cal`, development). Calibrators: sites where every cell found holds the same atlas level, never the islands or domains (cancer's territories) and never a cell's identity sites; binned by atlas level, one point per bin, made monotone. Added when the GMP share was traced to a laboratory-dependent departure on each probe design, which the two-point line of the self-tare cannot take out. Like a CMB detector calibrated on its own data against known calibrators. Nothing comes from other samples.

*Code:* `stage_t_selftare.py` line 58 (31 lines) · arguments: beta, cells, design · calls: not called on the traced runs · constants read: `CURVE_AGREE`, `CURVE_BIN`, `CURVE_MIN_SITES`

### `stage_t_selftare.py::_json`

Loads the self-tare matrix once, `Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json`: the four fixed-site sets, each site's probe design,
and the reference arrays' anchors. Moved unchanged from `Runtime Matrices/Development/` on 2026-10-22 (finding F23; the same sha256) and
pinned by the release check as a frozen input (F1): Stage T step 1 runs on every reading, so its matrix is frozen like every other reference.

*Code:* `stage_t_selftare.py` line 15 (4 lines) · arguments: name · calls: not called on the traced runs · constants read: `MATRIX`, `_C` · files named: `selftare_typeII_EPIC_v1.json`

### `stage_seq_values.py::values_from_bigwig`

Reads a public pair of GRCh38 bigWig tracks (per-CpG methylation level and read coverage, the form consortia such as BLUEPRINT release whole-genome bisulfite calls in) at each atlas site's GRCh38 position (the C of the CpG, or the next base where the caller reports the reverse strand), giving a value at each site for every reader but IAM-A, which needs molecules. Used by `run_sample.py --dev-seq-beta CALL.bw,COV.bw`.

*Code:* `stage_seq_values.py` line 206 (22 lines) · arguments: call_bw, cov_bw, min_calls · calls: not called on the traced runs · constants read: `KEY38` · files named: `atlas/v2.1/sites_key_hg38.parquet`

### `stage_seq_values.py::plant_ends`

TESTS ONLY: a molecule file with known errors planted: at each molecule's outermost calls a methylated call read unmethylated or the reverse, at set rates, and optionally a change at every call wherever it sits (a change in the cell). For the self-calibration synthetic.

*Code:* `stage_seq_values.py` line 90 (23 lines) · arguments: path, end_ct, end_tc, all_ct, dest, seed · calls: not called on the traced runs

### `stage_seq_values.py::end_calls_from_pat`

From a molecule file, at every atlas site, the calls made at the outermost CpG of each molecule end, and how many were methylated. Counts only.

*Code:* `stage_seq_values.py` line 68 (21 lines) · arguments: path, n_end · calls: not called on the traced runs

### `stage_seq_values.py::end_calls_from_pat.flush`

Inside end_calls_from_pat: adds the buffered end calls to the per-site counts.

*Code:* `stage_seq_values.py` line 76 (4 lines) · arguments: none · calls: not called on the traced runs

### `stage_seq_values.py::key`

Loads the frozen site key (`atlas/v2.1/sites_sequencing_key.parquet`: each IAM-Atlas v2.1 site's EPIC v1 probe name, hg19 position and CpG index
in the .pat files' numbering) once, sorted by CpG index. Development (to-do step 40).

*Code:* `stage_seq_values.py` line 23 (5 lines) · arguments: none · calls: not called on the traced runs · constants read: `KEY`

### `stage_seq_values.py::construct_draw`

TESTS ONLY: a constructed blood draw with a known answer, built by the chain from spent values (`run_sample.py --build constructed-draw
--build-input BLOOD_VALUES --build-mix TISSUE:SHARE,... --build-depth BUFFY,PLASMA --build-out DIR`). Buffy coat: the blood's own levels;
plasma: that blood in the remaining share plus each named atlas tissue's best value at its share. Each half is drawn as k of n reads per atlas
site (seeded) and written as a `.beta` count file with a truth record. With `--dev-plant CELL:LOSS` a loss of that cell's pattern is planted in the blood's levels before the reads are counted, scaled by
the cell's six-cell share as the one command scales a plant, so the counting noise falls on the planted levels as it would on a real loss. Made because the open matched set (GSE186888) is reserved unread for
the one-draw test that counts (author 2026-10-22).

*Code:* `stage_seq_values.py` line 173 (29 lines) · arguments: blood, add, depth_buffy, depth_plasma, out_dir, seed, plant · calls: not called on the traced runs · constants read: `HERE`, `HG19_CPGS` · files named: `truth.json`

### `stage_seq_values.py::values_from_beta`

A value at each atlas site from a `.beta` file, the standard binary per-CpG count file (methylated reads and all reads at every CpG, hg19 CpG
numbering, the numbering of the frozen site key); a file of another length is refused with both counts. Per-site counts only, so every reader
runs but IAM-A, which needs molecules (`run_sample.py --dev-seq-beta`, `run_draw.py --buffy-seqbeta / --plasma-seqbeta`). Checked on a
constructed file (the example blood GSM7508429 drawn at 30 reads per site): the IAM-Deconvolver's six shares within 1.2 points of the array's.

*Code:* `stage_seq_values.py` line 157 (14 lines) · arguments: path, min_calls · calls: not called on the traced runs · constants read: `FLAG`, `HG19_CPGS`

### `stage_seq_values.py::genome_coverage`

The intake check every sequenced sample passes before any reading: for each autosome, the share of IAM-Atlas v2.1's sites on it that carry a
value, against the median autosome's share; an autosome under half the median refuses the sample (GENOME_COVERAGE_REFUSED). Why it exists: a
run stored on SRA in genome order and read from its first read pairs holds only the first chromosomes (GSE98837's first 140 million pairs were
chromosomes 1 to 8), and the readings made from it are wrong in a way no later stage names. Comparing chromosomes with each other lets a thin
library (plasma) pass and stops a partial one (DEV-SEQ-INTAKE-01). The rule behind it: never read the first N pairs of a run; read it whole,
or check its order first.

*Code:* `stage_seq_values.py` line 232 (24 lines) · arguments: values, key, rule · calls: not called on the traced runs · constants read: `GENOME_COVERAGE_RULE` · files named: `sites_sequencing_key.parquet`

### `stage_seq_values.py::values_from_pat`

A value at each atlas site from sequencing molecules, for `--dev-seq-values MIN_CALLS`: every call at the exact CpG the site names is counted,
and the value is methylated calls over calls, kept only where there are at least MIN_CALLS calls; the summary gives how many sites have a value
and the calls per site, so a reading can say what depth it needed. Every line counts (IAM-A's own count keeps its line split). Development.

*Code:* `stage_seq_values.py` line 29 (25 lines) · arguments: path, min_calls, keep · calls: not called on the traced runs · constants read: `FLAG`

### `stage_seq_values.py::values_from_pat.flush`

Adds the buffered calls to their sites in one step, matching each call's CpG index to the key exactly; calls at CpGs that are not atlas sites
are dropped.

*Code:* `stage_seq_values.py` line 32 (5 lines) · arguments: none · calls: not called on the traced runs

### `stage_seq_values.py::pat_to_lbeta`

A molecule file counted into a per-CpG count file (`.lbeta`, uint16 methylated and all calls at every hg19 CpG, the layout the atlas build reads). Run as `run_sample.py --build seq-pat-lbeta --pat FILE [--dev-seq-trim-ends N] --build-out OUT.lbeta`; with a trim, the file is trimmed once by `trim_pat` and then counted. Untrimmed it equals the published `.beta` of the same molecules below that file's cap of 255. Used to count the IAM Sequencing Atlas's own donors with the trim a sample is read with, so sample and reference take the same path (the CMB rule: the theory goes through the instrument's processing, not the data cleaned toward the theory). A file with no molecules (empty or unreadable) is refused and nothing is written.

*Code:* `stage_seq_values.py` line 130 (25 lines) · arguments: path, out, n · calls: not called on the traced runs · constants read: `FLAG`, `HG19_CPGS`

### `stage_seq_values.py::pat_to_lbeta.flush`

Inside `pat_to_lbeta`: adds the buffered calls (CpG position, methylated or not, molecule count) into the two per-CpG totals in one vectorized step every five million calls. Counts only.

*Code:* `stage_seq_values.py` line 138 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_values.py::trim_pat`

A copy of a molecule file with the N outermost CpG calls at each end of every molecule left out, for `run_sample.py --dev-seq-trim-ends N` (development): the calls where library chemistry at the read ends lands, found by the floor by molecule position (one outside laboratory's end calls read 9 times its interior floor, reaching about 4 CpGs in). Geneticists call this trimming for M-bias. The bundle records what was kept.

*Code:* `stage_seq_values.py` line 114 (14 lines) · arguments: path, n, dest · calls: not called on the traced runs

### `stage_seq_values.py::thin_pat`

Tests only, for `--dev-thin FRACTION`: a seeded thinned copy of a .pat file, each molecule kept with probability FRACTION (a line counting w
molecules keeps a binomial draw from w), so a deep sample stands in for a shallower one; it returns the molecules before and after.

*Code:* `stage_seq_values.py` line 55 (11 lines) · arguments: path, frac, dest, seed · calls: not called on the traced runs

### `stage_t_selftare.py::anchors`

This array's mean β over each of the four fixed-site sets (type I low and high, type II low and high), with how many it measured.

*Code:* `stage_t_selftare.py` line 21 (7 lines) · arguments: beta, st · calls: not called on the traced runs

### `stage_t_selftare.py::selftare_map`

Stage T step 1, as described above: maps each design's values by β' = L_r + (β − L)(U_r − L_r)/(U − L) at the sites whose design is
recorded and returns the maps' anchors and slopes. Moved unchanged from `dev_stages.py` on 2026-10-22 (finding F23); `dev_stages` keeps
both names for older scripts.

*Code:* `stage_t_selftare.py` line 42 (12 lines) · arguments: beta, st · calls: not called on the traced runs · files named: `Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json`

### `stage_m_met_a.py::derived_baseline`

Met-A's healthy mark derived from the cell's own held errors (development, DEV-METAA-HEALTHY-01 step 3): at each identity site the healthy level is 1 - eps_loss where the cell holds the mark and eps_gain where it is held absent, put on the array scale through IAM-Atlas v2.1's measured transfer, and the mark is the mean entropy over the sites. Nothing is fitted; the array's own measurement blur is not included.

*Code:* `stage_m_met_a.py` line 89 (13 lines) · arguments: held, eps_loss, eps_gain · calls: not called on the traced runs · constants read: `TRANSFER`

### `stage_m_met_a.py::_floors`

Loads `metA_floors_v1_3.json` (the 6,000 identity sites and the healthy reference H_ref) once.

*Code:* `stage_m_met_a.py` line 29 (4 lines) · arguments: none · calls: array run 1 · constants read: `FLOORS`, `_F`

### `stage_m_met_a.py::_H`

The entropy of a beta value in bits, with β clipped to [10⁻⁶, 1 − 10⁻⁶] so a value of exactly 0 or 1 gives 0, not an error.

*Code:* `stage_m_met_a.py` line 34 (3 lines) · arguments: b · calls: array run 7

### `stage_m_met_a.py::platform_of`

The platform from the values themselves: EPIC v2 when any probe name carries the v2 design suffix; else EPIC when more than 700,000 probes,
else 450K.

*Code:* `stage_m_met_a.py` line 40 (6 lines) · arguments: beta · calls: array run 2 · constants read: `_V2_NAME`

### `stage_m_met_a.py::available`

The cell types with a healthy reference on each platform that are in scope (neutrophils).

*Code:* `stage_m_met_a.py` line 47 (3 lines) · arguments: platform · calls: not called on the traced runs · constants read: `CELLS_IN_SCOPE`

### `stage_m_met_a.py::precision`

The held-out readings of the reference arrays for one cell type: how many, the share in Normal, the SD and the range.

*Code:* `stage_m_met_a.py` line 51 (6 lines) · arguments: platform, cell · calls: not called on the traced runs · constants read: `LOO`, `NORMAL`

### `stage_m_met_a.py::read`

Met-A on a purified specimen of one cell type: refuses a cell outside scope, a platform without a reference, or under 90 % of the sites;
otherwise A = mean H(β) / H_ref with its state and the held-out precision.

*Code:* `stage_m_met_a.py` line 58 (19 lines) · arguments: beta, cell, platform, specimen · calls: not called on the traced runs · constants read: `CELLS_IN_SCOPE`, `NORMAL`, `SITE_COVERAGE_MIN`

## 6. Chain v4: the cell shares, IAM-A and the sky: `stage_v4.py`

**What it is for.** Bringing the instruments commissioned after chain v3 into the one run: Stage D21 (each blood cell's share, from
the IAM-Deconvolver against IAM-Atlas v2.1), Stage Q2 (IAM-A on sequencing data) and the sky (a picture of the C-Score's departures).
Nothing here is fitted at run time; every reference is a committed, stamped file.

**Stage D21's gate.** The six-cell blood panel (neutrophils, monocytes, B, CD4 T, CD8 T, NK) is read on the sites that separate those
cells in IAM-Atlas v2.1. With fewer than half of those marker sites measured, the shares are withheld and the reason gives the count.
The 50 % line is operational, not measured. Shares of neutrophils, monocytes, CD4 T and CD8 T cells carry the commissioned label; B and NK
cells are printed as development. The error bars are information, not calibrated intervals.

**Stage Q2's order.** Count the molecules (from a `.pat` file or an aligned BAM), refuse a file with no qualifying molecules or tiles,
run the sample-shape check, then read A against the sample's own reference and band v1. The commissioned label attaches to CD8 T cells
(the parent type and its subsets) read by whole-genome bisulfite; every other cell type or chemistry is labeled development.

**The sky.** The C-Score's departure map, in the C-Score's own genomic order, laid on a HEALPix sphere in NESTED order: consecutive
indices fill compact patches, so a run of neighboring departures (what the C-Score scores) shows as a patch and scattered departures as
speckle. NSIDE is the smallest power of 2 giving at least one pixel per two sites. Departures are drawn in units of the sample's own
spread (median and 1.4826 × MAD), which leaves the C-Score unchanged, because the C-Score is unchanged by shifting or rescaling every
departure. The pixel-to-angle conversion is the standard HEALPix NESTED algorithm (Górski et al. 2005), written out here so the chain
does not need healpy; it matches healpy to 10⁻¹⁴ radians. **Label: standard method** (the HEALPix sphere of cosmology, used as a
display). Nothing is scored from the picture.

### `stage_v4.py::_ex`

The five-part explanation of a stop, for the cell shares and IAM-A (the same parts as `run_sample._why`).

*Code:* `stage_v4.py` line 23 (2 lines) · arguments: check, outcome, measured, limit, source, means, todo, provisional · calls: not called on the traced runs

### `stage_v4.py::atlas_panel`

Every cell type of IAM-Atlas v2.1, read from the atlas's own columns, for `--cell-panel atlas`: the IAM-Deconvolver splits the sample into all
of them instead of the six blood cells. Development: the Deconvolver is commissioned on the six-cell panel only, so every share from another
panel is flagged.

*Code:* `stage_v4.py` line 28 (4 lines) · arguments: none · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `stage_v4.py::stage_d21`

Stage D21: loads the panel's reference and markers once, counts the markers measured, withholds below half, otherwise runs the
Deconvolver and labels each cell's share commissioned or development.

Each share whose best fit lies outside its own 95 % range is marked `resolved: false` ("not resolved from its close relatives in the atlas").
On the six-cell panel this does not happen; on the full panel close relatives (neutrophils, eosinophils and basophils; monocytes, macrophages
and microglia; T-cell subsets) can trade a share between them, so the resampled range and the single best fit disagree (development). `counts` (development): passed to the Deconvolver when `--dev-seq-noise-weight` is given on a sequenced sample; the record then says `noise_weighting`.

*Code:* `stage_v4.py` line 33 (31 lines) · arguments: beta, panel, sequencing, seqref, counts · calls: array run 1 · constants read: `MIN_MARKER_SHARE`, `SEQ_READ_MARKERS`, `_REF`

### `stage_v4.py::count_sequencing`

Counts a `.pat` file (gzipped or plain) on the hg19 tiles, or an aligned BAM against its reference FASTA, with Stage Q2's one counter.

*Code:* `stage_v4.py` line 80 (5 lines) · arguments: pat, bam, ref_fasta, workers, assembly · calls: bam run 1

### `stage_v4.py::counting_error`

IAM-A's counting error, stated by every reading (finding F38, 2026-10-22): the held error is holes counted among held-error chances, so its
counting error is binomial, √(ε(1 − ε)/n) for n chances, and A carries the same relative error. The record gives the chances, A's 95 % range
from counting alone, and that range's width as a share of the Normal band's width. It is recorded only: it never refuses a reading and never
changes A or its state, so a shallow sample, plasma included, is read and says itself how much of the band its counting uses (author: "let the
instrument tell us...not our assumptions"). For example, at a held error of 0.02: ten million chances use 3 % of the band's width, one million
9 %, a hundred thousand 30 %, ten thousand 94 % (`stage_v4.counting_error` on those counts).

*Code:* `stage_v4.py` line 86 (12 lines) · arguments: rec, A, B · calls: not called on the traced runs

### `stage_v4.py::stage_q2`

Stage Q2 as above: the refusals, the shape check, the reading A with its predicted value, and the commissioned or development label.

The reading's `count` also carries the nearest-neighbor coupling (`coupling`, `coupling_err_share`, `coupling_pairs`), recorded beside
the held error, never used in A.

The count also holds the run-length spectrum of losses on the same held molecules (`loss_runs`: observed runs of 1, 2, 3, 4 and 5 or more
consecutive losses, and their exact expectation when each molecule's own losses fall at random among its calls; `loss_run_ratio` =
observed runs / expected runs: 1 for independent point losses, below 1 when losses come in stretches), the tile-by-tile line
(`held_err_firm_tiles`, `held_err_loose_tiles`, `tile_line_n`, `held_err_by_tile_call`) and the library's end damage (`end_damage`,
`meter`), all from `stage_q2_selfref.count_pat`. This function's count list passes them on to the bundle with every reading. Development, recorded only, never used in A (register CP1; book
Part VI, 'Run-length spectrum of losses').

*Code:* `stage_v4.py` line 99 (31 lines) · arguments: rec, cell, chemistry, source · calls: bam run 1 · constants read: `CD8_TYPES` · files named: `Runtime Matrices/IAM_A_SelfRef/shape_reference_v1.json`

### `stage_q2_selfref.py::_runs_expected`

The expected number of runs of exactly 1, 2, 3, 4 and 5 or more consecutive losses when a molecule's own k losses fall at random among its
L calls: a run inside the molecule needs both neighbours unlost, one at either end needs one; divided by the number of ways to place k
losses. Exact (checked against every arrangement of small molecules); nothing fitted. Development, recorded only (register CP1).

*Code:* `stage_q2_selfref.py` line 211 (13 lines) · arguments: L, k · calls: not called on the traced runs · constants read: `_RUNS_CACHE`

### `stage_q2_selfref.py::_loss_runs`

The run-length spectrum of losses on held molecules: observed runs by length beside the expectation above, and loss_run_ratio = observed
runs / expected runs (1 for independent losses, below 1 when losses come in stretches). Recorded only, never used in A (register CP1).

*Code:* `stage_q2_selfref.py` line 225 (7 lines) · arguments: RL, RE · calls: not called on the traced runs

### `stage_q2_selfref.py::_tile_line`

The locality test of IAM-A's line inside one sample: the held error pooled in tiles judged firm and loose, the tile's mean call from the even
half of the sample and its held error from the odd half, so no half judges itself; also by four bands of tile mean call. If the line is local,
firm tiles read the line at loose share 0 and loose tiles at loose share 1. Development, recorded only, never used in A (TERR, GLB).

*Code:* `stage_q2_selfref.py` line 233 (16 lines) · arguments: tile_calls, TE · calls: not called on the traced runs

### `stage_v4.py::_compress`

Takes every other bit of an integer: the step of the HEALPix algorithm that splits a NESTED pixel index into its x and y coordinates.

*Code:* `stage_v4.py` line 137 (3 lines) · arguments: v · calls: array run 6

### `stage_v4.py::nest_pix2ang`

HEALPix NESTED pixel centers to colatitude and longitude.

*Code:* `stage_v4.py` line 140 (10 lines) · arguments: nside, ipix · calls: array run 3 · constants read: `JPLL`, `JRLL`

### `stage_v4.py::cscore_sky`

Draws the C-Score's departures on the sphere in a Mollweide view and returns a summary (NSIDE, sites, pixels drawn, the share of
departures beyond 3). Without a departure map it draws nothing and says why.

*Code:* `stage_v4.py` line 151 (38 lines) · arguments: z, out_png, title, vlim, readings · calls: array run 1

### `stage_v4.py::png_data_uri`

Embeds a figure in the HTML report as a data URI.

*Code:* `stage_v4.py` line 190 (2 lines) · arguments: path · calls: not called on the traced runs

## 7. The IAM-Deconvolver: `stage_deconv_v21.py`

**What it is for.** Given a mixed sample's array values, each cell's share, with a 95 % interval.

**Label: standard method** (weighted non-negative least squares against reference cell profiles), with IAM-Atlas v2.1 as the reference.

**The reference.** For each site and cell, the atlas's best value μ and the spread a new person's cell has around it:
sd² = ((hi − lo)/3.92)² + donor_sd² + lab_sd² + 0.01². The first term is the uncertainty of the atlas's average (its 95 % range); donor_sd
is the person-to-person spread the atlas fit measured; lab_sd is the laboratory-to-laboratory spread, the SD of the fit's own
array-laboratory offsets (18 laboratories, 0.0065 when sealed); 0.01² is a floor. Version 1 used only the first term and is kept as
`SPREAD="mean_only"`.

**The markers.** For each cell, the 50 sites where it sits furthest above, and the 50 furthest below, its nearest other cell in the panel,
in units of their combined spread: z = |μ_c − μ_nearest| / √(sd_c² + sd_nearest²).

**The solver.** Non-negative least squares on the markers, each site weighted by 1 / √(Σ_k f_k² sd_k² + lab²), starting from equal shares
and refined over three passes. The shares are scaled to sum to 1 and the raw sum is reported.

**The intervals.** 200 refits, each on resampled marker sites and on cell profiles redrawn from the atlas's spread (a new person's cells
are not the average cell). The 2.5th and 97.5th percentiles give the interval. On held-out data they covered the truth 78 % of the time,
below the 95 % their name implies: they are information, not calibrated intervals (see below).

**Built from, tested on.** Built from IAM-Atlas v2.1 alone. Tested by planted mixtures drawn from the atlas's own spread with a laboratory
shift (the file's `planted` test), then by the sealed real test: 200 mixtures of held-out purified arrays from E-MTAB-2145, a laboratory
the atlas never saw. Commissioned 2026-10-17 on the shares of neutrophils, monocytes, CD4 T and CD8 T cells: mean share error 0.016, every
number reproduced from a fresh clone (on AWS 2026-10-18 and on GitHub 2026-10-20). Both sealed tests carried two bars: the mean share
error at most 0.03 (met: 0.029 for version 1, 0.016 for version 2), and the true share inside the 95 % interval at least 80 % of the time
(not met: 0.42, then 0.78). After both scores, the author's limits study (2026-10-17) held the tool to its job, the share, so the
interval bar no longer gates it; the commissioning note discloses that the decision came after the scores, and the scores stand
unchanged. The intervals are therefore printed as information, labeled not calibrated.

**What it reads outside its own folder.** The laboratory spread comes from `doors/data/DEV_ATLAS_V21_01/transfer_v21_final.json`, a file in a
test-record folder. It is a runtime input and belongs in the chain's references (a to-do item of this Decoder).

**How to break it.** Read a pure array of one cell: its own share should be at least 0.80. Plant a 5 % share of one cell into a mixture:
its interval should exclude 0 in most mixtures. Remove half the markers: the shares must be withheld.

**The physics, step by step.** The IAM-Deconvolver carries no quantity of IAM's Law (register D1: measured, atlas identity, not a law
quantity). Its place in the physics is that it decides which healthy reference every other reading uses.

1. **The reference (measured).** For each site and cell, the atlas's best value μ and the spread a new person's cell has around it,
   sd² = ((hi − lo)/3.92)² + donor_sd² + lab_sd² + 0.01².
2. **The solver (standard method).** Weighted non-negative least squares on 50 markers above and 50 below each cell's nearest neighbor, shares
   scaled to sum to 1; 200 refits for the interval (information, not calibrated).
3. **What it hands on.** Every cell found at 1 % or more is read against its own atlas reference (Met-A for each cell), and the single-cell
   readings read the cell found most of. IAM-A's one-line reading needs no cell type for its prediction; the cell named here sets only the
   shape check's reference and the commissioned label.
4. **Status.** Commissioned on arrays for neutrophils, monocytes, CD4 T and CD8 T cells, share error 0.016; nothing of it is to be derived.

### `stage_deconv_v21.py::reference`

Reads μ, its range and the donor spread for the named cells from the atlas's compact parquet files (only those columns; 65,535 marks a
missing value) and builds sd as above.

*Code:* `stage_deconv_v21.py` line 26 (23 lines) · arguments: cells, atlas, spread · calls: array run 1 · constants read: `ATLAS`, `SPREAD` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_deconv_v21.py::markers`

The marker sites for the panel, as above.

The sites are kept in a cache on each machine under a key made from the reference's own numbers, its cells and `per_cell`, so an array after the first reads them instead of recomputing them (78 of 364 seconds on one array); a changed atlas, provisional column or panel makes a new key, so a stale list can never be used.

*Code:* `stage_deconv_v21.py` line 49 (13 lines) · arguments: R, per_cell · calls: array run 1 · constants read: `PER_CELL`

### `stage_deconv_v21.py::markers_compute`

The marker sites themselves, as `markers` computed them before the cache: for each cell, the sites that separate it most from its nearest other cell in the panel, in both directions.

*Code:* `stage_deconv_v21.py` line 62 (8 lines) · arguments: R, per_cell · calls: not called on the traced runs · constants read: `PER_CELL`

### `stage_deconv_v21.py::_solve`

The three-pass weighted non-negative least squares. On sequencing with `--dev-seq-noise-weight` (development), each site's own counting noise p(1-p)/n joins its weight, n the site's read count and p the model's prediction from the current shares, not the site's own noisy value (the way each sky pixel is weighted by its noise in a CMB fit); one more pass is made.

*Code:* `stage_deconv_v21.py` line 70 (11 lines) · arguments: y, A, S, lab_sd, passes, n · calls: array run 201

### `stage_deconv_v21.py::deconvolve`

The shares, raw sum, 95 % intervals and sites used for one sample.

Since 2026-10-22 it also records each cell's `resampled_median`, the middle of the resampled shares, beside the best fit, so a wide panel's two
estimates can be compared against known mixtures (development; the commissioned six-cell shares are the best fit, unchanged). `counts` (development): each site's read count on sequencing, passed to `_solve` for the counting noise, and resampled with the sites.

*Code:* `stage_deconv_v21.py` line 81 (18 lines) · arguments: beta, R, M, lab_sd, boot, seed, counts · calls: array run 1 · constants read: `BOOT`, `LAB_SD`, `SEED`

### `stage_deconv_v21.py::sample`

Inside the planted test only: builds a planted array, each cell's DNA drawn from the atlas's donor spread, mixed in known shares, plus a
laboratory shift and array noise.

*Code:* `stage_deconv_v21.py` line 116 (2 lines) · arguments: f, lab · calls: not called on the traced runs

### `stage_deconv_v21.py::sample_check`

For the second sealed test: a pure array passes when it reads at least 0.80 of its own cell. Failing arrays are listed and not mixed.

*Code:* `stage_deconv_v21.py` line 193 (2 lines) · arguments: x, cell, R, M · calls: not called on the traced runs · constants read: `SELF_MIN`

## 8. IAM-Atlas v2.1's commissioning rules: `stage_atlas_commission.py`

**What it is for.** The scorers IAM-Atlas v2.1 was commissioned by, and the table that says which uses of the atlas are commissioned.
The chain loads it for the laboratory spread (`lab_sd`) and for that table.

**Label: standard method** (a posterior predictive interval for a new sample from a new laboratory, scored by coverage).

**The uses.** U1, reference profiles: a healthy cell from any laboratory reads inside the atlas's range for that cell. Commissioned for
cortical neurons, lung bronchus epithelium and naive CD4 T cells on 450K. U2a, blind sites: commissioned for colon and lung. U3,
composition: development. A use not listed is development wherever it appears.

**The bars (as amended and sealed 2026-10-16, before the atlas existed).** U1: at least 95 % of a cell's healthy held-out samples, all
laboratories pooled, cover 80 % to 97 % of their sites inside the atlas's 90 % predictive range. U2: a tissue passes when its epithelium
is assessed and at least 95 % of its assessed cells reach a share of 0.90. U3: neutrophil mean absolute error ≤ 0.02 and at least 95 % of
samples within 0.05. A sample the chain withholds is listed, not scored.

### `stage_atlas_commission.py::lab_sd`

The SD of the atlas fit's array-laboratory offsets, read from `transfer_v21_final.json` at run time, never typed.

*Code:* `stage_atlas_commission.py` line 17 (4 lines) · arguments: terms · calls: array run 1 · constants read: `TERMS`

### `stage_atlas_commission.py::predictive_interval`

The 90 % predictive range for a new sample from a new laboratory. Arrays: N(μ, sd² + lab² + floor). Sequencing: the array value carried
through the platform map, plus binomial counting noise at the sample's coverage, the laboratory scatter and the floor.

*Code:* `stage_atlas_commission.py` line 21 (11 lines) · arguments: draws, sd_draws, kind, g, cov, scatter, level, seed, lab · calls: not called on the traced runs · constants read: `FLOOR`, `NOISE`, `RNG_SEED`

### `stage_atlas_commission.py::u1_coverage`

The share of a sample's sites that fall inside the predictive range.

*Code:* `stage_atlas_commission.py` line 32 (2 lines) · arguments: lo, hi, y · calls: not called on the traced runs

### `stage_atlas_commission.py::u1_verdict`

U1's verdict for one cell: commissioned when at least 95 % of its scored samples have coverage between 0.80 and 0.97.

*Code:* `stage_atlas_commission.py` line 34 (6 lines) · arguments: cov_by_sample, withheld, lo_bar, hi_bar · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::u2_share`

The share of a tissue set's sites within tolerance of the anchor.

*Code:* `stage_atlas_commission.py` line 40 (2 lines) · arguments: set_mean, anchor, tol · calls: not called on the traced runs

### `stage_atlas_commission.py::u2_verdict`

U2's verdict for one tissue.

*Code:* `stage_atlas_commission.py` line 42 (4 lines) · arguments: shares, bar, epithelium · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::u3_neutrophil`

U3's neutrophil error: the mean absolute error, the share within 0.05, the largest error, and whether both bars are met.

*Code:* `stage_atlas_commission.py` line 46 (4 lines) · arguments: est, truth, withheld_mask · calls: not called on the traced runs · constants read: `SHARE`

### `stage_atlas_commission.py::atlas_v21_use_status`

Commissioned or development for one use of the atlas, read from the table above.

*Code:* `stage_atlas_commission.py` line 56 (6 lines) · arguments: use, key · calls: not called on the traced runs · constants read: `ATLAS_V21_COMMISSIONING`

## 9. IAM-A, each sample against its own reference: `stage_q2_selfref.py`

**What it is for.** Reading, from single DNA molecules, how well a cell holds its methylation compared with what its own state predicts.
No other person enters the reading.

**Label: from IAM's physics.** At a held site, writing (on SAM) and erasing (on oxygen and 2-oxoglutarate) are both running. Each
interval a mark is lost with probability f and restored with probability g, so the share of held sites that are wrong at any moment
settles at the **held error**

  ε = f / (f + g),  and the holding energy is E_hold = kT ln[(1 − ε)/ε].

The balance is set by the ratio, not by how often the cell divides; division only sets how fast the balance is reached. Measured this way
on 56 healthy cell types, E_hold is 3.08 kT per interior opportunity (`holding_energy_q2.json`).

**What is counted (one counter, two readers of the same molecules).**

- **The held error ε.** On molecules with at least 6 CpG calls and at least 80 % methylated (a held run), the isolated unmethylated
  interior calls (a T between two Cs) over the interior calls. The first and last call of a molecule are never counted, because a hole
  there cannot be told from the end of the run.
- **The loose share L.** The genome is cut into tiles of 2,000 consecutive CpGs. A tile is loose when its molecules' mean methylation is
  below 0.70. L is the share of tiles (each with at least 200 calls) that are loose: how much of its genome this cell has let go.
- **No molecule is used twice.** Molecules alternate: even ones are counted for L, odd ones for ε. The reference (from L) and the
  quantity it judges (ε) never share a molecule. In a `.pat` file the alternation is by line, and a line carries every identical
  molecule with its count, so all copies of one pattern go to the same side.

**IAM-A's baseline (calibrated).** Across healthy cell types, ε rises in a straight line with L:

  ε_predicted = firm + slope × L,  with firm = 0.04026 and slope = 0.03462 (`selfref_line_v1.json`).

This line was measured once on 153 healthy samples of 56 cell types (a public whole-genome bisulfite atlas of purified human cells,
GSE186458), counted by this same counter. A Theil–Sen fit was used, the median of pairwise slopes, which is robust to outlying cell types.
The median distance of a cell type from the line is 0.0025. The line is stored on the raw axis this counter writes; there is no scale
factor anywhere. What the intercept means physically is derived from M alone (PHY-083, register A4): the paid hold loses 0.0445 per site.
A per-site loss and the counter's lone-loss reading are different quantities, so the comparison is made like for like: molecules planted at
that loss are read by this same counter (the forward table, register A6), which gives 0.0359 at the healthy clustering 1.33 and 0.0364 to
0.0333 over clustering 0.94 to 2.27, with no eraser input (in mature blood the eraser share drops out, register A5, a lead). The stored firm,
measured in the atlas laboratory, is 1.12 times that level; the blind-test laboratory's healthy samples read 0.96 to 1.03 times it. The
spread of healthy cell types about this one line, about 81 % of its healthy spread, is each type's own level, open (register A8); the
measured eraser share of stem and progenitor types is one contributor to it. The slope 0.034625 is measured; the part of it that is mixing
inside a sample, about 40 %, follows from the pump (register A7, partly derived). The reading still uses the measured constants.

**The reading.** IAM-A = ε measured / ε predicted at the sample's own L. 1 is on IAM-A's baseline; above 1 is more holes than its own
state predicts. **Band v1: 0.7787 to 1.0551**, the 2.5th to 97.5th percentile of 156 pooled healthy readings.

**Built from, tested on.** The line, the shape reference and 153 of the band's 156 readings come from the same 153 healthy samples
(GSE186458); the band's other 3 readings are from a second laboratory (GSE49627). So the band describes the healthy spread inside the
set the line was measured on. What commissioned IAM-A was the sealed grand test on held-out samples from a third study: healthy CD8 T
cells, of which 8 were read (5 declined by the shape check) and 7 read Normal (bar: at least 75 % inside, at least 8 read).

**The sample-shape check (a gate).** A sample is read only if the 10th percentile of its tile means is at most 0.0543 below the lowest
healthy sample of its own cell type. The margin is the 90th percentile, over the 53 cell types with at least 2 healthy samples, of the
spread of that percentile within a type. It is computed from the reference, never typed. A cell type with no healthy reference is not read.

**The stamp.** Every output this stage writes carries a chain stamp: the stage, a SHA-256 of the chain's code, a SHA-256 of the result,
its inputs, the Master Testing Manifest rows it belongs to, and the time. The line, band and shape reference are refused if they were
edited after the chain wrote them, because the result's hash no longer matches its stamp.

**How to break it.** Plant isolated holes into held runs of a constructed BAM: ε and IAM-A must rise (release check V7 does this through
the one command). Hand it a cell type with no healthy reference: it must not read. Edit one number in `band_v1.json`: the chain must refuse it.

**The physics, step by step.** From IAM's Law to the number IAM-A prints, each step marked derived, calculated, measured, calibrated or
open, with the register row that decides it. Every number can be recomputed with a calculator or the one-line commands of the Inputs section.

1. **The law at the DNA (derived from two measured inputs).** Holding a mark costs energy paid at the surface that holds it, at that surface's
   own temperature. For the cell the surface is the DNA at T = 310.15 K (measured, A11) and the coin is one ATP,
   ΔG_ATP = 54000 J/mol (measured, A10). M = ΔG_ATP / RT = 54000 / (8.314462618 × 310.15) = 20.94 thermal kicks per coin.
2. **The floor (derived, A1).** A mark held behind a wall of M kicks with one coin per copy is lost with probability 1/(1 + e^M) = 8.05 × 10⁻¹⁰,
   disorder 2.5e-08 bits: the gauge's left end, millions of times below any reading. Established neighbor: the one-step copy
   limit 1/(1 + e^x) of Sartori and Pigolotti (E3).
3. **The price of one restore (derived, A2).** One S-adenosylmethionine costs two coins (ATP → AMP + 2 Pi, adenylate kinase at equilibrium), so
   each restore is pushed by w = W/kT = 2M = 41.88. A constant push is the best profile two coins can buy (derived, A3).
4. **The paid hold (derived, A4).** A driven first passage over the push gives the ratio of restores to losses
   odds(w) = (w/2) / (1 − (1 − e^−w)/w) = 21.45. Lost with probability f and restored with g each interval, the held error
   settles at ε = f/(f + g) = 1/(1 + odds): ε0 = 0.0445 per site from M alone. The balance is a ratio, so it does not depend on
   how often the cell divides. Blind test PHY-084 (an unseen laboratory, 54 libraries) met, B cells on watch.
5. **The holding energy and the held share (derived, A4 and P4).** E_hold = kT ln[(1 − ε0)/ε0] = ln(odds) = 3.066 kT and
   φ(M) = E_hold / M = 0.1464 (about ln(M)/M); measured on 56 healthy types 3.084 kT and 0.1473.
6. **England's bound, the established neighbor (calculated, E7).** By England (2013, eq. 6) holding a mark at error ε costs at least kT ln(1/ε) of
   heat per interval, even when the mark never changes: 3.11 kT at ε0, which is E_hold + ln(1/(1 − ε0)) on the same ε0
   (an identity, not a second estimate), and exactly M, one coin, at the floor. Still to check: that a held mark's loss is England's reverse step.
7. **The counter (calibrated, A14).** Two readers of the same molecules, alternating so that no molecule is used twice (even lines or molecules
   to L, odd to ε). The held error ε: on held runs (at least 6 CpG calls, at least 80 % methylated) the isolated unmethylated interior calls (a T
   between two Cs) over the interior calls; the first and last call are never counted. The loose share L: of the tiles of 2,000 consecutive
   CpGs with at least 200 calls, the share whose mean call is below 0.70. A per-site loss e and the counter's lone-loss share are different
   quantities: with no qualifying rule a molecule shows e(1 − e)² (0.0407 at e = ε0; `stage_q2_selfref.held_err_length`), and the qualifying
   rule (a short molecule carrying a second loss fails to qualify) and clustering lower the raw held error further.
8. **The forward table: ε0 through the counter (derived, A4 and A6; F033).** Molecules planted at ε0 per site, with no eraser input, and read by
   this counter (`doors/data/DEV_CHAIN_Q2_01/forward_line_synthetic.py`; 3 samples of 120,000 planted held molecules per point, nine GitHub
   runs 43991012677 to 43991157818) give the raw held error at each clustering (coupling → held error): 0.94 → 0.0364, 1.24 → 0.0360,
   1.42 → 0.0357, 1.52 → 0.0352, 1.75 → 0.0348, 1.89 → 0.0346, 2.13 → 0.0341, 2.15 → 0.0337, 2.27 → 0.0333.
   At the healthy clustering 1.33 the level from M alone is 0.0359 (length-corrected 0.0400). No eraser share enters: in mature blood it drops
   out (A5, a lead: all eight type-laboratory readings of PHY-084 sit 0.96 to 1.17 of this level with none). In stem and progenitor types the
   measured eraser share (13.6 % in stem cells, about 5 % in hematopoietic progenitors) is part of that type's distance from the line, a
   contributor to each type's own level (A8), not a separate item.
9. **The line (measured; intercept derived, slope partly derived).** ε_predicted = firm + slope × L, one line for every cell type, read against
   each sample's own loose share; no atlas enters the prediction. The chain's line v1 (`selfref_line_v1.json`) is measured: firm 0.04026 and
   slope 0.034625 on 153 healthy samples of 56 types. Its firm is 1.12 times the level from M alone in the atlas laboratory; the blind-test
   laboratory's healthy samples read 0.96 to 1.03 times it (F033). About 40 % of the slope is mixing inside a sample and follows from the pump
   when the restore push scales with the held share of the neighborhood (A7, partly derived, replicated in a second laboratory, median ratio
   1.03); the rest is cell type and completes with A8.
10. **The reading (measured against the line).** IAM-A = ε measured / ε predicted at the sample's own L (`stage_q2_selfref.reading`). The chain
    reads A on the measured line v1 (the default); A on the derived line of step 8 is read beside it where the sample's clustering is in range.
11. **The band (measured, A9).** Band v1, 0.7787 to 1.0551: the 2.5th to 97.5th percentiles of 156 pooled healthy readings on line v1, commissioned
    on healthy CD8 T cells. Band v2 is to be rebuilt on the derived line from the joint run's healthy atlas libraries, and the default line
    switches to the derived line together with it. `stage_v4.counting_error` gives each reading's binomial counting range,
    √(ε(1 − ε)/n) for n chances, as a share of the band's width.
12. **The shape check (measured gate).** A sample is read only if its tiles' 10th percentile is no more than the margin 0.0543 below the lowest
    healthy sample of its type (`shape_reference_v1.json`), because a genome let go beyond any healthy sample of its type is not a state the
    line can be read from.
13. **Coupling and its null (calculated null; physics open, CP1).** `stage_q2_selfref._coupling`: adjacent pairs of unmethylated calls on held
    molecules over Σ k(k − 1)/L, the number random placement gives a molecule of L calls with k losses; 1 for independent losses, counted
    exactly. Healthy: naive T 1.29 to 1.33, central memory 1.43 to 1.44, effector 1.52 to 1.58. It matters for step 8: at the same error share,
    clustered losses lower the lone-error count, which is why the derived intercept is read at the sample's own coupling. coupling-A
    (`stage_coupling.read`) is the coupling over the healthy coupling of the type (`coupling_reference_v1.json`). IAM's Law is not yet in it;
    the candidate (a lead) is a restore crew working along stretches, coupling = 1 + (mean skip length − 1) × (share of losses from skips).
14. **Run lengths (calculated; development, CP1).** `stage_q2_selfref._runs_expected` gives the exact expected number of runs of 1, 2, 3, 4 and 5
    or more consecutive losses when a molecule's own k losses fall at random among its L calls; `_loss_runs` gives the observed runs and
    loss_run_ratio = observed / expected: 1 for independent point losses, below 1 when losses come in stretches. Synthetic: independent 0.999,
    paired 0.538. It separates double loss (both strands lost before the crew arrives; an A8 candidate) from independent loss.
15. **The tile line (calculated; development, T2).** `stage_q2_selfref._tile_line`: the held error pooled in tiles judged firm (mean call ≥ 0.70)
    and loose, each tile's mean call from the even half and its held error from the odd half, and in four bands of tile mean call. If the line
    is local, firm tiles read the line at L = 0 (about 0.0403) and loose tiles at L = 1 (about 0.0749). Synthetic met exactly (firm 0.027455
    against 0.02745 planted).
16. **The meter (calibrated, measured per library; A17, T1, S2).** The meter's own false marks are measured and recorded, never subtracted from a
    reading. conv_fail: the share of non-CpG cytosines read as C (incomplete bisulfite conversion; on genomic sites it also counts real CpH
    methylation in neurons and stem or progenitor cells). tc_err: T read as C on the top strand, A as G on the bottom, at non-CpG positions,
    base quality at least 20. The meter = conv_fail + tc_err, a quarter to a third of the raw unmarked-side rate in healthy blood
    (A17, PHY-091). conv_fail_lambda: every cytosine of an unmethylated lambda spike-in read as C (`stage_q2_selfref.lambda_meter`), free of
    real CpH. end_damage: the lone-loss rate one and two calls from a molecule's nearer end minus four to six calls in (`meter_end_damage`);
    above 0.002 the library is a noisy meter, still read, never a healthy reference. The islands meter (PHY-093): sites every healthy type keeps
    unmarked read the meter one for one, islands error = 1.01 × meter + 0.0001, so each
    library's islands are its in-sample meter.
17. **Where these fields are recorded now.** `stage_q2_selfref.count_pat` computes the coupling, the run lengths, the tile line, the end damage and
    the distance tally on every `.pat` count; `count_bam` adds conv_fail, sub_err and tc_err. The main reading's count (`stage_v4.stage_q2`)
    carries the coupling, the run lengths, the tile line and the meter's fields; on a molecule file the meter's fields are empty (they come
    from an aligned BAM), and the lambda meter is read only on the count-only route (`--dev-count-only`).
18. **What is open.** Each cell type's own level (A8, the deciding item; it also completes A7's slope); the 12 % between-laboratory offset that
    keeps the measured line the default (A5 lead, A8, the meter); band v2; the mitotic clock's tick from the writer's error (C3); the domains'
    level from the line (T2); each type's own coin (A16); IAM's Law inside the coupling (CP1); and inside the information sphere (SP1).


**Nearest-neighbor coupling, a second reading on the same count.** Every count also records whether a molecule's lost marks fall one at a
time or together: the pairs of neighboring CpGs unmethylated together on held molecules, over what each molecule's own length and error count
give at random (`_coupling`); 1 is independent. It is written beside the held error because clustered errors lower the lone-error count IAM-A
uses. In development: recorded and flagged. Its gauge, coupling-A (the sample's coupling over the healthy coupling of its cell type,
`stage_coupling.read`, against `Runtime Matrices/Coupling/coupling_reference_v1.json`), is read and flagged; its null, 1, is counted exactly,
and IAM's Law is not yet in it (register CP1).

### `stage_q2_selfref.py::qual_iso_expect`

The expected number of isolated losses on a qualifying molecule of L calls at a per-call loss e (development, plan 40.11). Exact: a walk over every pattern of the molecule's calls, keeping the losses so far and the last two calls, then only the patterns the qualifying rule keeps (at least 6 calls, at least 80 % methylated). Nothing fitted.

*Code:* `stage_q2_selfref.py` line 105 (14 lines) · arguments: L, e · calls: not called on the traced runs

### `stage_q2_selfref.py::held_err_length`

The held error with the qualifying rule's length bias removed (development, recorded only, plan 40.11). Short molecules carrying a second loss fail to qualify, so the raw held error falls with molecule length. From the qualifying molecules counted at each length, it finds the per-call loss at which their expected isolated losses equal those counted, and reports it as the held error a molecule shows with no qualifying rule, e (1 - e)^2.

*Code:* `stage_q2_selfref.py` line 119 (13 lines) · arguments: by_len · calls: not called on the traced runs

### `stage_q2_selfref.py::_dispersion`

Register A8, development, recorded only: how the held losses spread across held molecules, as observed variance of each molecule's
losses about L p over the binomial L p (1 - p). An even per-site rate keeps it near 1 (a little below, from the held rule's cut); a
damaged share of cells raises it. Read across cell types; never changes A.

*Code:* `stage_q2_selfref.py` line 191 (8 lines) · arguments: DV · calls: not called on the traced runs

### `stage_q2_selfref.py::_coupling`

Nearest-neighbor coupling on held molecules (development, recorded only, plan 49.5): the adjacent pairs of unmethylated calls, set against what random placement gives each molecule from its own length L and its own number k of unmethylated calls, k (k - 1) / L. 1 when errors fall independently, above 1 when they cluster along the molecule. Nothing fitted. Clustered errors lower the lone-error count IAM-A uses at the same error share, so the two are reported together.

*Code:* `stage_q2_selfref.py` line 200 (9 lines) · arguments: K · calls: not called on the traced runs

### `stage_q2_selfref.py::lambda_meter`

The meter's own false marks read on the unmethylated lambda spike-in (2026-11-15, PHY-091 amendment 3). Every cytosine of the lambda genome is unmethylated, so every C read as C there, CpG or not, is the meter: incomplete bisulfite conversion. Genomic non-CpG sites cannot give that alone, because neurons and stem or progenitor cells carry real CpH methylation. Same read rules and end skip as the genomic count; reads with fewer than 50 aligned bases are left out (partial alignments of genomic reads onto the small lambda genome). Returns None when the reference or alignment has no lambda contig. Where IAM's Law is: it is the meter, not the law; it lets the unmarked side's write error, the writer's one selection step 1/(1+D), be read with the meter removed in any cell type. Synthetic: doors/data/DEV_ERASE_01/conv_01/lambda_synthetic.py.

*Code:* `stage_q2_selfref.py` line 466 (18 lines) · arguments: bam, fa · calls: not called on the traced runs · constants read: `LAMBDA_CONTIGS`

### `stage_q2_selfref.py::meter_end_damage`

The library-noise rule, applied to every library alike: the library's own molecule-end damage, the lone-loss rate one and two calls from the molecule's nearer end minus four to six calls in, on the same held molecules (count_pat's held_by_distance). Healthy chromatin has no reason to lose marks at the ends of the fragments a library happens to cut, a copier does. Above METER_END_MAX (0.002) the library is flagged a noisy meter: still read and scored, never a healthy reference. Recorded in every count as end_damage and meter. Development.

*Code:* `stage_q2_selfref.py` line 134 (13 lines) · arguments: by_dist, limit · calls: not called on the traced runs · constants read: `METER_END_MAX`

### `stage_q2_selfref.py::count_pat_tile_errors`

The held error tile by tile (DEV-ERASE-01 graded maintenance test): held molecules and lone errors by Stage Q2's own rules, each molecule in the tile of its first CpG (blocks of 2,000 consecutive CpG indices, so no genome table is needed), split into two halves that share no molecule (even and odd lines), so one half can measure a tile's level and the other remove the measurement's own scatter from the slope.

Since 2026-11-10 it also records, for development only, the same held error by distance from the molecule's nearer end (held_by_distance: 1 is the second call in, 6 means six or more), so the meter's end error inside the counted interior can be measured; the reading is unchanged. Since 2026-11-14 the tile file written by `run_sample.py --dev-count-tiles` also carries each tile's calls and mean call (count_pat_tile_calls), columns `calls` and `mean_call`; readers that take only the four count columns are unchanged.

*Code:* `stage_q2_selfref.py` line 250 (15 lines) · arguments: lines, tile · calls: not called on the traced runs

### `stage_q2_selfref.py::count_pat_tile_calls`

Each tile's calls and methylated calls over all its molecules, by the same tiling as count_pat_tile_errors (each molecule in the tile of its first CpG, blocks of 2,000 consecutive CpG indices). The tile's mean call is methylated over calls; its looseness is one minus that. Added 2026-11-14 for PHY-087 step 2, so a tile's looseness comes from the chain, not from a script reading the molecule file.

*Code:* `stage_q2_selfref.py` line 266 (12 lines) · arguments: lines, tile · calls: not called on the traced runs

### `stage_q2_selfref.py::chain_sha`

The SHA-256 over every `.py` file of the chain folder, names and contents, in sorted order.

*Code:* `stage_q2_selfref.py` line 29 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `HERE`

### `stage_q2_selfref.py::_canon`

A result in canonical JSON (sorted keys, no spaces), so the same result always hashes the same.

*Code:* `stage_q2_selfref.py` line 33 (1 lines) · arguments: obj · calls: bam run 4

### `stage_q2_selfref.py::payload_sha`

The SHA-256 of a result without its stamp.

*Code:* `stage_q2_selfref.py` line 34 (1 lines) · arguments: obj · calls: bam run 4

### `stage_q2_selfref.py::file_sha`

The SHA-256 of a file, read in 16 MB pieces.

*Code:* `stage_q2_selfref.py` line 35 (5 lines) · arguments: path, chunk · calls: not called on the traced runs

### `stage_q2_selfref.py::_rows`

The manifest rows a result belongs to, given directly or through the `IAM_MANIFEST` environment variable.

*Code:* `stage_q2_selfref.py` line 45 (3 lines) · arguments: manifest · calls: not called on the traced runs

### `stage_q2_selfref.py::stamp`

Attaches the chain stamp to a result, hashing it exactly as it will be stored (JSON turns number keys into text).

*Code:* `stage_q2_selfref.py` line 48 (8 lines) · arguments: obj, inputs, manifest · calls: not called on the traced runs · constants read: `STAGE`

### `stage_q2_selfref.py::write_json`

Writes a stamped result and reads it back to confirm the stored file verifies.

*Code:* `stage_q2_selfref.py` line 56 (6 lines) · arguments: obj, path, inputs, manifest · calls: not called on the traced runs

### `stage_q2_selfref.py::write_csv`

Writes a table and its stamp beside it, as `<file>.stamp.json`.

*Code:* `stage_q2_selfref.py` line 62 (4 lines) · arguments: df, path, inputs, manifest · calls: not called on the traced runs · constants read: `STAGE`

### `stage_q2_selfref.py::hg19_tiles`

The hg19 tiles of 2,000 consecutive CpGs on each autosome, in CpG-index coordinates; a partial last tile is dropped. With TILE_SOURCE "dictionary" (run_sample.py --dev-tiles dictionary, development, plan 14.9) the tiles follow the genome dictionary's chromosome ranges instead of the table's counts, which are short from chr14 on.

*Code:* `stage_q2_selfref.py` line 69 (18 lines) · arguments: none · calls: not called on the traced runs · constants read: `HG19_CPGS`, `TILE_SOURCE` · files named: `hg19_cpg_chrom_ranges.json`, `s range from Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`

### `stage_q2_selfref.py::_finish`

Turns tile counts and hole counts into the record: L, the number of tiles, ε and its opportunities, and the 10th, 50th and 90th
percentiles of the tile means for the shape check. When the gain-side counts are given, it adds gain_err and gain_opp.

*Code:* `stage_q2_selfref.py` line 87 (11 lines) · arguments: tile_calls, e_iso, e_opp, extra, g_iso, g_opp · calls: bam run 1

### `stage_q2_selfref.py::_mol`

The loose share counted molecule by molecule (development, recorded only): of the molecules with at least six calls on the tile half of the file, the share below the 0.70 cut the tile loose share uses and the share held neither marked nor unmarked (between 20 and 80 % methylated). Unlike the tile loose share, both add linearly when cell types mix, so a mix of cells reads the same share as one of them (synthetic test cfDNA 2).

*Code:* `stage_q2_selfref.py` line 98 (7 lines) · arguments: M · calls: not called on the traced runs

### `stage_q2_selfref.py::count_pat_site_classes`

Every call at a set of sites, grouped (for Met-A: the cell's held and held-absent identity sites), split by the molecule's class from all its calls by Stage Q2's rules (held, held absent, loose, short): the site level and each class's share and level, so a site's level is shown as the loose share beside the held error (development, step 3 version 3). A synthetic file with planted classes is recovered.

*Code:* `stage_q2_selfref.py` line 352 (24 lines) · arguments: lines, groups · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::timing_runs`

Loads the frozen replication-timing thirds (Runtime Matrices/Replication_Timing/hg19_cpg_timing_thirds_v1.npz): the CpG index where each run
starts, its label (0 no timing value, 1 early, 2 mid, 3 late) and the two cuts. Development (register A8L2).

*Code:* `stage_q2_selfref.py` line 300 (6 lines) · arguments: path · calls: not called on the traced runs · files named: `Runtime Matrices/Replication_Timing/hg19_cpg_timing_thirds_v1.npz`, `hg19_cpg_timing_thirds_v1.npz`

### `stage_q2_selfref.py::count_pat_timing_sets`

count_pat_timing for several timing references in one pass over the molecule file (the 15-line mean and each family's own lines); the same rules.
Recorded under iam_a.sites.timing_by_family; never changes a reading. Development (register A8).

*Code:* `stage_q2_selfref.py` line 328 (23 lines) · arguments: lines, sets · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_pat_timing`

Stage Q2's held and gain counts by the replication-timing third of each molecule's first CpG, by count_pat_sites' own rules (odd lines, molecule
classes from all their calls, the isolated error at interior calls). Every sequencing reading records it under iam_a.sites.timing; it never
changes a reading. Development (register A8L2, the location test).

*Code:* `stage_q2_selfref.py` line 307 (20 lines) · arguments: lines, starts, labels · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_pat_sites`

The held error counted only at a chosen set of sites (development, Met-A's derived baseline, step 3 version 2): Stage Q2's own rules unchanged, the same held and held-absent molecules classed from all their calls, only the calls counted restricted to the sites. A synthetic file with planted rates 0.06 and 0.05 at a site set returns them once the neighbors' rate is divided out.

*Code:* `stage_q2_selfref.py` line 279 (20 lines) · arguments: lines, sites · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_pat`

The `.pat` reader: even lines are added to their tile; odd lines that qualify as held runs add their interior calls and isolated holes. Since DEV-METAA-HEALTHY-01 it also counts the gain side, the mirror image: on molecules held unmarked (at least 6 calls, at most 20 % methylated, QUAL_MAX_METH), a lone methylated call between two unmethylated ones over the interior calls (gain_err, gain_opp). The loss side and every earlier output are unchanged.

*Code:* `stage_q2_selfref.py` line 149 (41 lines) · arguments: lines, tiles · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::_cpg_index`

The positions of every CpG in a chromosome's sequence.

*Code:* `stage_q2_selfref.py` line 379 (2 lines) · arguments: seq · calls: not called on the traced runs

### `stage_q2_selfref.py::_calls_of`

One aligned read's methylation calls at CpGs, from the Bismark XM tag or, for gemBS, from the read against the reference. Skips the first
and last 3 aligned bases. Also counts non-CpG conversion (a check on bisulfite conversion) and, on every 20th read, A/T mismatches (a check
on sequencing error).

*Code:* `stage_q2_selfref.py` line 381 (33 lines) · arguments: r, ref, ref0, sub · calls: not called on the traced runs

### `stage_q2_selfref.py::_merge`

Merges the calls of the two reads of a pair into one molecule; a CpG on which they disagree is dropped.

*Code:* `stage_q2_selfref.py` line 414 (6 lines) · arguments: a, b · calls: not called on the traced runs

### `stage_q2_selfref.py::count_bam_region`

Counts one 10 Mb region of an alignment file: primary, mapped, non-duplicate reads with mapping quality at least 10, pairs merged, then
the same alternation and the same two counts as the `.pat` reader. Also counts calls per qualifying molecule for the response curve.

*Code:* `stage_q2_selfref.py` line 420 (36 lines) · arguments: args · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::count_bam_region.done`

Counts one finished molecule: even molecules to their tile, odd held runs to ε.

*Code:* `stage_q2_selfref.py` line 427 (16 lines) · arguments: calls · calls: not called on the traced runs · constants read: `QUAL_MAX_METH`

### `stage_q2_selfref.py::combine_bam`

Adds the regions' counts into one record, with the conversion-failure and substitution rates.

*Code:* `stage_q2_selfref.py` line 456 (9 lines) · arguments: parts, assembly · calls: bam run 1

### `stage_q2_selfref.py::count_bam`

Splits the autosomes into 10 Mb regions and counts them in parallel.

*Code:* `stage_q2_selfref.py` line 485 (5 lines) · arguments: bam, fa, workers, assembly, chroms, span · calls: bam run 1 · constants read: `AUTOSOMES`

### `stage_q2_selfref.py::build_line`

How the line was built: per cell type, the median L and the opportunity-weighted ε, then a Theil–Sen line across cell types. Not called at
run time; the line is read from its frozen file.

*Code:* `stage_q2_selfref.py` line 492 (10 lines) · arguments: per_sample · calls: not called on the traced runs

### `stage_q2_selfref.py::line`

Loads the frozen line and refuses it if it was edited after the chain wrote it.

*Code:* `stage_q2_selfref.py` line 503 (6 lines) · arguments: version · calls: bam run 1 · constants read: `LINE_DEFAULT`, `LINE_DIR`

### `stage_q2_selfref.py::predict`

ε predicted at a loose share: firm + slope × L.

*Code:* `stage_q2_selfref.py` line 509 (2 lines) · arguments: loose_share, version · calls: not called on the traced runs

### `stage_q2_selfref.py::residual`

A count against the line: measured ε, predicted ε, and their difference.

*Code:* `stage_q2_selfref.py` line 511 (5 lines) · arguments: rec, version · calls: bam run 1

### `stage_q2_selfref.py::build_shape_reference`

How the shape reference was built: per cell type, the lowest healthy 10th percentile, and the margin. Not called at run time.

*Code:* `stage_q2_selfref.py` line 548 (8 lines) · arguments: records · calls: not called on the traced runs · constants read: `SHAPE_Q`

### `stage_q2_selfref.py::shape_reference`

Loads the frozen shape reference and refuses it if edited.

*Code:* `stage_q2_selfref.py` line 557 (6 lines) · arguments: version · calls: bam run 2 · constants read: `LINE_DIR`, `SHAPE_DEFAULT`

### `stage_q2_selfref.py::shape_check`

The sample-shape gate described above, with the measured percentile, the limit and the reason.

**The X and Y on the two routes (Decoder finding F37, measured 2026-10-21).** On the `.pat` route the held error is counted on every
chromosome and the loose share on the autosomes; the aligned-read route counts both on the autosomes. On 17 healthy samples of the committed
line, counting the held error on the autosomes only moved IAM-A by a mean of −0.32 % (range −0.71 % to +0.09 %), at most about 2.5 % of the
band's width (`doors/DEV_IAMA_SEX_01.md`). The commissioned line and band were built on the `.pat` route and are kept.

*Code:* `stage_q2_selfref.py` line 563 (9 lines) · arguments: rec, cell, version, ref · calls: bam run 1

### `stage_q2_selfref.py::reading`

IAM-A = measured ε / predicted ε.

*Code:* `stage_q2_selfref.py` line 574 (4 lines) · arguments: rec, version · calls: bam run 1

### `stage_q2_selfref.py::build_band`

How band v1 was built: the 2.5th to 97.5th percentile of pooled healthy readings. Not called at run time.

*Code:* `stage_q2_selfref.py` line 578 (6 lines) · arguments: healthy, version · calls: not called on the traced runs

### `stage_q2_selfref.py::band`

Loads band v1 and refuses it if edited.

*Code:* `stage_q2_selfref.py` line 588 (2 lines) · arguments: version · calls: bam run 1 · constants read: `LINE_DIR`

### `stage_q2_selfref.py::grand_verdict`

The sealed grand test's scorer: each healthy sample shape-checked, each read sample inside or outside the band; met when at least 75 % of at
least 8 read samples are inside.

*Code:* `stage_q2_selfref.py` line 601 (12 lines) · arguments: records, version, B, ref · calls: not called on the traced runs · constants read: `GRAND_MIN_READ`, `GRAND_SHARE`

### `stage_q2_selfref.py::derived_line`

Loads the derived line (Runtime Matrices/IAM_A_SelfRef/derived_line_v1.json): the intercept as a table over the
sample's clustering (coupling), from the held error derived from M alone (0.0445, no eraser input) read through this
counter on planted molecules, and the measured slope. Refuses the file if it was edited after the chain wrote it (its stamp).

*Code:* `stage_q2_selfref.py` line 518 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `LINE_DIR` · files named: `derived_line_v1.json`

### `stage_q2_selfref.py::derived_reading`

A on the derived line: held error / (intercept at the sample's own coupling, interpolated in the table, + measured
slope x loose share). Never extrapolated: outside the table's coupling range it returns no A_derived and says why. Recorded beside A.

When band_v2.json exists (refused if edited after the chain wrote it), the reading also carries state_derived (Normal, above Normal or
below Normal on band v2) and the band it used, marked 'in commissioning'.

*Code:* `stage_q2_selfref.py` line 524 (18 lines) · arguments: rec · calls: not called on the traced runs · constants read: `LINE_DIR` · files named: `band_v2.json`

### `stage_metaa_atlas.py::sequencing_reference`

For a sequenced sample, replaces the legacy state 'untared' in every atlas-referenced reading (Met-A sequencing, the territories, the Met-A-side C-Score) with what the reading is referenced to (the atlas ruler, the laboratory part from the sample's own reads) and puts SEQ_REFERENCE_RULE, one line per sequencing instrument, in the bundle as reference_rule; numbers are unchanged.

*Code:* `stage_metaa_atlas.py` line 45 (21 lines) · arguments: o · calls: not called on the traced runs · constants read: `SEQ_REFERENCE_RULE`, `SEQ_REFERENCE_STATE`

### `stage_metaa_atlas.py::sequencing_reference.fix`

Inside sequencing_reference: one reading's state. If the reading holds a value and its state is missing or the legacy 'untared', the state becomes SEQ_REFERENCE_STATE and the reading's name is recorded as changed; a reading not read is left as it is.

*Code:* `stage_metaa_atlas.py` line 49 (3 lines) · arguments: d, name · calls: not called on the traced runs · constants read: `SEQ_REFERENCE_STATE`

### `stage_metaa_atlas.py::sequencing_reference.deep`

Inside sequencing_reference (edit 242): walks every nested reading and note of Met-A sequencing, the territories and the Met-A-side C-Score. A nested state 'untared' becomes SEQ_REFERENCE_STATE; a note that begins 'untared' loses that word, and a note naming same-run healthy samples as the comparison names the IAM Sequencing Atlas in the sample's own mix instead. The rule text (SEQ_REFERENCE_RULE) keeps its record of the retired word.

*Code:* `stage_metaa_atlas.py` line 53 (9 lines) · arguments: d, name · calls: not called on the traced runs · constants read: `SEQ_REFERENCE_STATE`

### `stage_metaa_atlas.py::_cnt`

One cell's reference reads at the given sites, methylated and total, from the sequencing atlas columns in use (the held-out donor half when --dev-seq-half-cols is given); NaN where the cell has none.

*Code:* `stage_metaa_atlas.py` line 952 (17 lines) · arguments: cell, idx · calls: not called on the traced runs · constants read: `ATLAS`, `NA`, `SEQ_ATLAS` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::counted_mix`

The sample's disorder and a healthy sample of this mix's expected disorder, both at a common depth at each site: the smallest of the sample's reads, 20, and every kept cell's reference reads. The sample's reads are drawn down without replacement; the expectation draws each read from a cell of the mix and from that cell's own reference reads without replacement, so no level is taken as exact. A is the ratio of their sums with equal site weights (weights taken from the same thin reads would bias it). Used by Met-A for the whole sample, Met-A for each cell in the mix and the territories when COUNTED_MIX is on (--dev-seq-counted-mix, or --dev-seq-half-cols); the plug-in reading is kept as A_plugin.

*Code:* `stage_metaa_atlas.py` line 970 (27 lines) · arguments: xs, counts, F, w, Mk, Nk, m_max · calls: not called on the traced runs · constants read: `TERR_HALF`, `TERR_M`, `TERR_SHARE_MIN`

### `stage_metaa_atlas.py::derived_held`

Met-A at a cell's held identity sites against the level derived from M alone: held sites are those where the
cell's own atlas level is at least 0.80; the expectation at each is the disorder of 1 - eps0 at that site's read count (expected_H_counts),
eps0 read from derived_line_v1.json. Returns A_derived_held with its site count, or None with the reason (no derived line; under 200 sites).

*Code:* `stage_metaa_atlas.py` line 582 (16 lines) · arguments: x, n, mu · calls: not called on the traced runs · constants read: `HELD_MIN` · files named: `derived_line_v1.json`, `read from derived_line_v1.json`

### `stage_q2_selfref.py::commissioned_files_ok`

True when the line, band and shape reference the chain loads are byte-for-byte the files IAM-A was commissioned on (their payload
checksums are kept in IAMA_COMMISSIONING['files']); otherwise False with the files that differ. `stage_v4.stage_q2` gives the commissioned
label only when it is True, and says which file changed when it is not.

### run_sample.py: refusal MIXED_SAMPLE

When no cell was named and the Deconvolver finds a mixture (`mixture_check`), IAM-A is not read: it reads one cell type, and reading a mixture
as its largest cell would mix several types' levels into one A. The record lists the groups found. A named cell (`--seq-cell`) is read as
before. IAM-A in a mixture (`iam_a_mix`) is in development.

### run_sample.py: refusal NO_CHAIN_CELL

When the Deconvolver's cells hold no IAM-A shape reference (for example granulocytes or megakaryocytes, which the shape reference
does not yet carry), IAM-A is not read and the record says 'IAM-A not read' with the cells found; no cell is taken from the specimen's
words.

*Code:* `stage_q2_selfref.py` line 593 (8 lines) · arguments: none · calls: not called on the traced runs · constants read: `IAMA_COMMISSIONING`, `LINE_DIR`

## 10. Stage Q0, intake for sequencing files: `stage_q0_intake.py`

**What it is for.** Stage 0 for a `.pat` file: an array cannot reach Met-A without Stage 0, and a `.pat` file cannot reach IAM-A without
Stage Q0. Each check stops the file with a named reason, and every measured value is recorded whether or not it stops the file.

**Label: standard method** (file integrity, genome build, coverage and conversion checks, as in the ENCODE whole-genome bisulfite standard).

| Check | What it measures | Stops the file when | Limit and source |
|---|---|---|---|
| Q0.7 specimen | the cell type | on the chain v4 route, the cell type has no healthy shape reference | rule |
| Q0.1 readable | the file decompresses to its end; every line has 4 fields (chromosome, CpG index, a pattern of C, T and dots, a count) | any malformed line or a truncated file | rule |
| Q0.2 build | every line's CpG index lies inside its chromosome's range in the hg19 CpG dictionary (`hg19_cpg_chrom_ranges.json`) | any line outside | rule |
| Q0.4 whole genome | molecules on every autosome | an autosome has none | rule |
| Q0.4 coverage | interior calls on molecules with at least 6 calls | never (recorded) | — |
| Q0.5 read length | the share of molecules with at least 6 calls | not set: recorded only | to be set on healthy files |
| Q0.3 conversion | the bisulfite conversion rate from the alignment record, when one is given | below 0.98 | the ENCODE whole-genome bisulfite standard |
| Q0.6 duplicates | the duplicate fraction from the alignment record, when given | not set: recorded only | to be set on healthy files |

**Not run on a BAM.** Q0's file checks are written for the `.pat` format. On the `--bam` route the intake record says so and why, and
Stage Q2's counter records the bisulfite conversion failure (from non-CpG calls) and the substitution rate on A and T; the sample-shape
check refuses a cell type with no healthy reference.

### `stage_q0_intake.py::_ranges`

Loads the hg19 CpG-index range of each chromosome, if the file is present.

*Code:* `stage_q0_intake.py` line 35 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `RANGES`

### `stage_q0_intake.py::scan_pat`

One pass over the file, gzipped (including multi-member bgzip) or plain: completeness, malformed lines, molecules per chromosome, calls,
molecules with at least 6 calls, interior opportunities, and lines outside their chromosome's range.

*Code:* `stage_q0_intake.py` line 40 (36 lines) · arguments: path · calls: not called on the traced runs · constants read: `MIN_CALLS` · files named: `hg19_cpg_chrom_ranges.json`

### `stage_q0_intake.py::scan_pat.line`

Parses and counts one line.

*Code:* `stage_q0_intake.py` line 45 (11 lines) · arguments: l · calls: not called on the traced runs · constants read: `MIN_CALLS`

### `stage_q0_intake.py::intake`

Runs the checks in the table and returns the record with each check's value, rule and result, and the decision: stop at the first
refusal, or proceed.

*Code:* `stage_q0_intake.py` line 130 (59 lines) · arguments: path, specimen, cell, alignment_qc, selfref_cells · calls: not called on the traced runs · constants read: `ACCEPTED`, `AUTOSOMES`, `GEO_BUILD`, `MIN_CALLS`, `Q0_LIMITS`

### `stage_q0_intake.py::intake.chk`

Records one check and, when it stops the file, the first refusal's code and reason, with its five-part explanation.

*Code:* `stage_q0_intake.py` line 144 (10 lines) · arguments: step, ok, value, rule, code, stop · calls: not called on the traced runs

## 10b. Genome builds: which genome each reading counts on, and how the chain keeps them apart (2026-10-22, lesson C34)

**Why it matters.** A sequencing file (.pat) names each CpG by its index in one genome assembly's list of CpGs. hg19 and hg38 number the
genome differently, so the same index points at a different CpG on each. A file made on hg38 and counted on hg19 does not fail loudly: every
count lands on the wrong site, and a globe drawn from it looks like a real picture. The withdrawn IAM-A genome globes (DEV-IAMA-GLOBE-01 steps
2 to 5) did exactly this: GEO holds each healthy-cell sample of the published sequencing atlas twice, `..._name.hg38.pat.gz` and
`..._name.pat.gz` (hg19); the test's own download took the first name in alphabetical order, the hg38 one, and counted it on hg19 positions.

**The rule.** Every .pat file is read on **hg19** (wgbstools 0.1.0's CpG index, `stage_q0_intake.GEO_BUILD`): Stage Q0's index ranges, Stage
Q2's tiles and the whole-genome globe's layout are all hg19 indexes. An aligned BAM is different: Stage Q2 reads the bases themselves against
the reference sequence given, on hg19 or hg38 (`--assembly`, default hg19; mm10 for `--dev-count-only` only, 2026-11-14), so the only requirement is that the BAM and its reference are the
same build (IAM-A's line was built this way from ENCODE's GRCh38 files and from the published atlas's hg19 .pat files). The arrays carry their
own probe names and have no build question.

**How the chain enforces it, three times over.**
1. **The chain fetches the file.** `run_sample.py --geo-pat GSM` (and `--prior-geo-pat` for a second reading) lists the sample's files on
   GEO and takes the one that declares no other build (`stage_q0_intake.geo_pat`); if none or several remain it stops and lists what GEO holds.
   The bundle records which file it took, the others it saw, and the file's checksum (`geo_fetch`). This record is the one place a typed id
   is not hashed: a public GEO accession is the file's address, not a person's id, and hashing it would leave an address that points nowhere.
2. **Stage Q0 stops a wrong build twice.** First by the name: a file named as hg38, GRCh38, hg18 or a mouse build is stopped before a line is
   read, and the refusal says which build it is and what to do (`stage_q0_intake.build_named`). Then by the content: every line's CpG index
   must lie inside its chromosome's hg19 range (check Q0.2); the hg38 file of the withdrawn steps had 48,859,268 lines outside them.
3. **The push gate refuses side downloads.** Rule 23b-g: a changed test or picture script that downloads a .pat file itself, instead of
   letting the chain fetch it, is refused (controls C13 and C14).

### `stage_q0_intake.py::build_named`

The build a file's name declares when it is not hg19 (hg38, GRCh38, hg18, mm9, mm10, GRCm38, GRCm39), else nothing; read from the name alone.

*Code:* `stage_q0_intake.py` line 81 (5 lines) · arguments: path · calls: not called on the traced runs · constants read: `OTHER_BUILDS`

### `stage_q0_intake.py::_geo_wait`

How long the chain waits before asking GEO again for a sample's file: the server's own Retry-After when it sends one, otherwise one
minute doubling to a ten-minute cap, over eight tries. GEO refuses by request rate per address, for longer than a few minutes.

*Code:* `stage_q0_intake.py` line 87 (8 lines) · arguments: t, err · calls: not called on the traced runs

### `stage_q0_intake.py::geo_pat`

A GEO sample's .pat file on the build the chain reads, as described above, with its name, address, the other files GEO holds, its size
and checksum; stops, with what GEO holds, when it cannot pick exactly one.

*Code:* `stage_q0_intake.py` line 96 (33 lines) · arguments: gsm, dest, build, retries · calls: not called on the traced runs · constants read: `GEO_BUILD`

## 11. Group-A, has a whole group moved: `stage_groupa.py`

**What it is for.** One person's whole-blood Met-A cannot show a 1 % change: healthy people differ by about 2 % after the tare. The
average of a group can. Group-A compares two groups read in one series.

**Label: standard method** (two tared means, Welch's standard error; register GA1: statistics, not IAM's Law), applied to one physical
reading per person, each against its own expected value for that person's cell mix. The physics is in the readings it compares (Met-A).
Its commissioned rule tares each reading against same-run healthy arrays, a reference the atlas rule retires; Group-A is not yet
re-commissioned on IAM-Atlas v2.1, and its development route (`compare-cells`) reads each cell against the atlas alone.

**How it reads.** Each person is read by the one command, unchanged. Each reading is tared against healthy arrays run the same way: the
median of the healthy references on the same slide when the slide holds at least 3, else of the same series. A sample is never its own
reference. Then the two groups' tared means are compared:

  difference d = mean₁ − mean₂, standard error SE = √(s₁²/n₁ + s₂²/n₂), t = d / SE.

|t| ≥ 2 is a group that moved. |t| < 2 is no difference found, and the smallest difference the design could have found (2 × SE) is
printed. Nothing is fitted.

**The physics, step by step.** Group-A is statistics (register GA1: not IAM's Law, like the C-Score); the physics is in the readings it
compares.

1. **Each person's reading (physics, Section 5).** Met-A from the one command, unchanged: disorder at the identity sites over the healthy
   expectation for that person's mix.
2. **The comparison (standard method).** d = mean₁ − mean₂, SE = √(s₁²/n₁ + s₂²/n₂), t = d / SE; |t| ≥ 2 is a group that moved.
3. **The reference (open).** The commissioned rule tares each reading against same-run healthy arrays, a reference the atlas rule retires.
   Group-A is not yet re-commissioned on IAM-Atlas v2.1; its development route (`collect-cells`, `compare-cells`) compares each cell found
   against the atlas alone, with no tare, because two groups of one series share that series' own array line and it cancels in the difference.
   Where the derivation could enter is through Met-A's held level (M2), which arrays cannot resolve.


**Built from, tested on.** Sealed before any array was opened, then scored once on 278 healthy controls of a laboratory no instrument had
read (GSE318173): 263 read; women and men 0.36 % apart at 1.2 standard errors; a 1 % loss planted into half of them found at 7.3 standard
errors. Commissioned 2026-10-20 for groups of whole bloods run in one series, EPIC v1. Release check V10 reproduces bars 2 and 3 to every
printed digit from the committed rows. Groups in any other specimen are read and labeled preliminary.

**Why one series.** Across laboratories, untared readings differ by several percent. Two groups from two laboratories would differ by
their laboratories. The download is checked against the size GEO states: an empty or short body (GEO sometimes sends one with no error) is fetched again, up to the retries, and never read. An empty folder page from GEO is fetched again before the sample is taken to hold no molecule files.

### `stage_groupa.py::_find`

A person's bundle in the bundles folder (named ID_bundle.json, or ID.json as the series runner's pass 1 writes it), or in any folder below it (sequencing runs land each bundle in its own folder).

*Code:* `stage_groupa.py` line 85 (7 lines) · arguments: bundle_dir, i · calls: not called on the traced runs · files named: `_bundle.json`, `s pass-1 bundles are named GSM.json`

### `stage_groupa.py::tare`

Each reading over the median of its healthy references (same slide when there are at least 3 there, else the series), excluding itself;
records which scope was used.

*Code:* `stage_groupa.py` line 33 (22 lines) · arguments: r, healthy · calls: not called on the traced runs

### `stage_groupa.py::welch`

The two groups' means, counts, difference, standard error and t.

*Code:* `stage_groupa.py` line 56 (4 lines) · arguments: a, b · calls: not called on the traced runs

### `stage_groupa.py::compare`

Tares, compares and calls the result, with the smallest findable difference and the commissioning label.

*Code:* `stage_groupa.py` line 61 (17 lines) · arguments: rows, group_col, a, b, healthy, specimen, platform · calls: not called on the traced runs · constants read: `PRELIMINARY`

### `stage_groupa.py::collect`

Builds one row per person from the run's report bundles and the researcher's sample list (id, slide, group column, optional
healthy_reference).

*Code:* `stage_groupa.py` line 93 (11 lines) · arguments: bundle_dir, samples_csv, out_csv, reading · calls: not called on the traced runs · constants read: `READINGS`

### `stage_groupa.py::collect_cells`

Development (Group-A against IAM-Atlas v2.1 alone): one row per person and per cell the IAM-Deconvolver found (the bundle's cells_found rows), with the cell's share, its Met-A against the atlas (each cell divided by its own healthy prediction) and its C-Score; no tare. `python3 stage_groupa.py collect-cells BUNDLE_DIR SAMPLES.csv OUT.csv`.

*Code:* `stage_groupa.py` line 106 (13 lines) · arguments: bundle_dir, samples_csv, out_csv · calls: not called on the traced runs

### `stage_groupa.py::compare_cells`

Development: Group-A per cell found, against the atlas alone. For each cell read in at least 3 people of each group at a share of at least 5 %, the two groups' Met-A means are compared (welch: difference, standard error, t; groups differ when |t| >= 2). Two groups of one series share that series' own array line, so it cancels in their difference and nothing but the atlas is used as healthy. `python3 stage_groupa.py compare-cells CELLS.csv GROUP_COLUMN A B`.

*Code:* `stage_groupa.py` line 120 (10 lines) · arguments: rows, group_col, a, b, min_share, min_n · calls: not called on the traced runs

## 12. The retired position reading, Stage Q: `stage_q_iam_a.py`

**Retired.** IAM-A was first read against a frozen position P per cell type, measured on purified granulocytes. That reading was replaced
on 2026-10-17 by Stage Q2, each sample against its own reference. The one command runs Stage Q only with `--site-table` or
`--dev-iama-position`, and labels it development. It is kept so earlier records can be reproduced. Its text still uses the retired term
"copy error"; what it counts is the held error described in section 9.

### `stage_groupa.py::compare_cells_sites`

Which cell changed between two groups, site by site. In one array every cell's row blends that cell with its neighbors in proportion to their DNA, so a change in one cell moves every row. At each site, each person's value is the sum of the cells found, each weighted by its share (the IAM-Deconvolver's), with each cell's own value in the series shared by both groups and each cell's difference between the groups estimated by least squares, as a mean difference is estimated. Each cell's difference is then read against its own atlas pattern at its own identity sites, each site weighted by its own measured noise in the series (the matched filter of the CMB maps); a 1 % loss of the cell's pattern reads +1 %. Healthy enters only from the atlas. Standard errors from resampling the people. Reads the saved values of run_sample.py --keep-values. Development.

*Code:* `stage_groupa.py` line 135 (47 lines) · arguments: rows, values_dir, group_col, a, b, plant, seed · calls: not called on the traced runs · constants read: `WEIGHTED` · files named: `_betas.parquet`

### `stage_groupa.py::compare_cells_sites.est`

Inside compare_cells_sites: for a set of people (all of them, or one resample), the per-site least-squares solution and each cell's noise-weighted reading against its own atlas pattern at its own identity sites.

*Code:* `stage_groupa.py` line 167 (6 lines) · arguments: ix · calls: not called on the traced runs

### `stage_q_iam_a.py::positions`

Loads the frozen positions file (`iama_positions_v2.json`).

*Code:* `stage_q_iam_a.py` line 35 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `POS`, `_P`

### `stage_q_iam_a.py::_H`

The entropy of a rate in bits.

*Code:* `stage_q_iam_a.py` line 40 (1 lines) · arguments: e · calls: not called on the traced runs

### `stage_q_iam_a.py::_pat_lines`

Streams the lines of a `.pat` file, gzipped or plain, optionally cut at a byte count (the retired version 1 read only part of each file).

*Code:* `stage_q_iam_a.py` line 42 (27 lines) · arguments: path, max_bytes, chunk · calls: not called on the traced runs

### `stage_q_iam_a.py::pat_site_table`

The per-site table (opportunities and holes in two halves of the molecules) from a `.pat` file.

*Code:* `stage_q_iam_a.py` line 72 (22 lines) · arguments: path, max_bytes · calls: not called on the traced runs · constants read: `PAT_PIPELINE`

### `stage_q_iam_a.py::_genomic_order`

Sorts a site table by chromosome and CpG index.

*Code:* `stage_q_iam_a.py` line 95 (8 lines) · arguments: T · calls: not called on the traced runs

### `stage_q_iam_a.py::cscore`

The retired IAM-A clustering score over the site table, pooled and per half.

*Code:* `stage_q_iam_a.py` line 105 (14 lines) · arguments: site_table, eps, halves · calls: not called on the traced runs · constants read: `CSCORE_BLOCK_SITES`, `CSCORE_MIN_BLOCKS`

### `stage_q_iam_a.py::cscore.one`

The clustering score of one table.

*Code:* `stage_q_iam_a.py` line 108 (8 lines) · arguments: o, k · calls: not called on the traced runs · constants read: `CSCORE_BLOCK_SITES`, `CSCORE_MIN_BLOCKS`

### `stage_q_iam_a.py::read`

The retired reading, IAM-A = H(ε) / (P × H(ε₀)), refused for a pipeline other than the one P was measured on, or for part of a file.

*Code:* `stage_q_iam_a.py` line 121 (27 lines) · arguments: site_table, cell, pipeline, mask, allow_partial · calls: not called on the traced runs · constants read: `IAMA_BAND_STATE`, `MIN_HALF_OPPORTUNITIES`, `MIN_OPPORTUNITIES`

### `stage_q_iam_a.py::tare`

The retired reading's same-run tare.

*Code:* `stage_q_iam_a.py` line 152 (22 lines) · arguments: rec, refs, sample_id · calls: not called on the traced runs · constants read: `IAMA_BAND_STATE`, `MIN_REFS`

## 13. Two draws of one person, Stage 12b: `serial_mode.py`

**What it is for.** Comparing two draws of the same person. `run_sample.py` calls two of these functions when `--prior-betas` is given;
the rest are tools.

**Label: standard method.**

### `serial_mode.py::check_same_person`

Refuses unless the two draws carry the same identifier hash, array type and pipeline.

*Code:* `serial_mode.py` line 7 (8 lines) · arguments: now, prior · calls: not called on the traced runs

### `serial_mode.py::delta_cells`

The change in each cell's A for cells present in both draws; cells that appeared or disappeared are listed, not differenced. Reads the
per-cell layout of an older bundle (`cells_all`); a tool, not called by the one command.

*Code:* `serial_mode.py` line 16 (16 lines) · arguments: now, prior, floor · calls: not called on the traced runs

### `serial_mode.py::delta_sky`

β now minus β before, at the sites both draws measured. No expectation, no composition.

*Code:* `serial_mode.py` line 33 (8 lines) · arguments: beta_now, beta_prior, floor_beta · calls: not called on the traced runs

### `serial_mode.py::snp_noise_floor_A`

The two arrays' own noise (Stage 1's SNP-probe model, σ²(β) = a + b·β(1 − β)) carried through to A over the cell's identity sites:
ΔA ≈ 2.576 × |dH/dβ| × √(var_now + var_prior) / H_ref at the mean β, the 99 % two-sided change the noise alone could make. A lower bound
on the smallest change that means anything. It reads the identity sites from an older bundle layout (`loci` of a per-cell entry)
that chain v4 bundles do not carry; it is a tool, not a stage.

*Code:* `serial_mode.py` line 42 (13 lines) · arguments: pci_entry, beta_now, beta_prior, snp_now, snp_prior, hmin · calls: not called on the traced runs

### `serial_mode.py::snp_noise_floor_A.var_mean`

For one array, the variance of the mean β over the identity sites from its noise model, Σ(a + b·β(1 − β)) / n², and the mean β.

*Code:* `serial_mode.py` line 49 (3 lines) · arguments: beta, snp · calls: not called on the traced runs

### `serial_mode.py::trajectory`

A person's readings in draw order with their dates and the step between draws: numbers in order, no fit, no slope. Reads the same older
per-cell layout as `delta_cells`.

*Code:* `serial_mode.py` line 56 (11 lines) · arguments: bundles · calls: not called on the traced runs

## 14. The report: `report_v4.py`, `report_v3.py`, `report_figs_v4.py`, `report_pdf_v4.py`

**What it is for.** Writing what the stages measured, and nothing they did not. One self-contained HTML page per sample, and the same
report as two PDFs, built from the same rows and pictures so the formats cannot disagree.

**Label: standard method** (presentation). The report computes no reading. Every number it prints comes from the run's bundle or from
a committed reference file.

**Its parts, in page order.**

1. **The researcher view**: the reading at a glance (each instrument, the question it answers, its reading, its state and its
   commissioning status); the physics behind each reading with this run's own numbers; the cell shares; Met-A on its
   gauge; the C-Score with its sky and the overlay of the sample on its healthy reference; IAM-A on its gauge.
2. **The plain view**: one gauge and one sentence per reading, and the sky.
3. **The technical record** (`report_v3.build`): every stage's record, the red flags, the safeguards, troubleshooting, and the
   development toolkit table.

**The gauge** (`report_figs_v4.gauge_png`). Left end the floor (derived), the Normal band around healthy 1 (measured), right end
saturation, every identity site fully scrambled (derived). Every mark is read from `Runtime Matrices/Gauge/cell_gauge_marks_v1.json`;
nothing is typed in the drawing code. On IAM-A's scale healthy sits close to the floor, so a magnified strip shows the band. A withheld
state draws no gauge at all. The C-Score has no derived ends, because it is a clustering ratio: its bar runs from scattered to clustered,
with its band.

**Commissioned or not, on every line.** A commissioned reading carries its record. Anything else is still shown and says so: "This score
is preliminary until [instrument] is officially commissioned on [cell]" (`report_v4.untested`).

**Red flags** (`report_v3.red_flags`): STOP (refused), FLAG (a commissioned call such as a clustered departure), WITHHELD (a reading not
given, with its reason), CAUTION and NOTE. They are written into the bundle as well as the page.

**Safeguards** (`report_v3.safeguards`), each PASS, FAIL or NOT_RUN on this run:
- **Rendered-claim scan:** no disease, cohort or population words in the report's prose.
- **Formula self-test:** H(0.5) = 1, H(0.25) = 0.81128, H(10⁻⁶) under 3 × 10⁻⁵.
- **Anchors**, **deconvolver conformance** and **atlas separability.**

**How to break it.** Write a disease word into a reading's text: the claim scan must fail. Remove a mark from the gauge file: the gauge
must not draw.

### `report_v4.py::_law_sphere_sec`

The report's IAM's Law Sphere section (edit 206): every Law Sphere output for the whole sample and by job, beside its healthy
(law) value, its unit and its basis (derived, calculated, measured, conjecture), and the whole-genome globe's numbers; 'not run' with
the reason when the sample has no molecule file.

*Code:* `MethylPhys_Interface/report_v4.py` line 515 (25 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`, `LS_BASIS`

### `report_v4.py::_tl`

The words the report prints for what an atlas-referenced reading is referenced to: on sequencing, 'atlas ruler; laboratory part from this sample's own reads'; on arrays, the legacy 'untared'.

*Code:* `MethylPhys_Interface/report_v4.py` line 347 (2 lines) · arguments: o · calls: not called on the traced runs

### `report_v4.py::_uri`

A figure file as a data URI, so the page is self-contained.

*Code:* `MethylPhys_Interface/report_v4.py` line 38 (1 lines) · arguments: path · calls: array run 6, bam run 2

### `report_v4.py::_H`

The entropy of a rate in bits, for the physics rows.

*Code:* `MethylPhys_Interface/report_v4.py` line 39 (1 lines) · arguments: p · calls: bam run 2

### `report_v4.py::_sci`

A number in HTML scientific notation (×10ⁿ).

*Code:* `MethylPhys_Interface/report_v4.py` line 40 (2 lines) · arguments: v · calls: bam run 2

### `report_v4.py::_marks`

Loads the gauge marks file.

*Code:* `MethylPhys_Interface/report_v4.py` line 42 (1 lines) · arguments: none · calls: array run 3, bam run 3 · constants read: `CH` · files named: `cell_gauge_marks_v1.json`

### `report_v4.py::_href`

Met-A's healthy reference in bits per site (0.33026), read from `metA_floors_v1_3.json`, never typed.

*Code:* `MethylPhys_Interface/report_v4.py` line 45 (2 lines) · arguments: none · calls: not called on the traced runs · files named: `metA_floors_v1_3.json`

### `report_v4.py::cellname`

A cell type's display name.

*Code:* `MethylPhys_Interface/report_v4.py` line 47 (2 lines) · arguments: c · calls: array run 30, bam run 1 · constants read: `_NAMES`

### `report_v4.py::untested`

The line printed beside a reading outside what its instrument was commissioned on.

*Code:* `MethylPhys_Interface/report_v4.py` line 49 (5 lines) · arguments: what, inst, shown, noun · calls: array run 7 · constants read: `_NAMES`

### `report_v4.py::chip`

The colored status label: commissioned, refused or withheld, not run, or development.

*Code:* `MethylPhys_Interface/report_v4.py` line 54 (4 lines) · arguments: status · calls: array run 12, bam run 5 · constants read: `E`

### `report_v4.py::_glance`

The at-a-glance table as HTML.

*Code:* `MethylPhys_Interface/report_v4.py` line 59 (3 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::glance_rows`

The at-a-glance rows, from the bundle: the Deconvolver's top three shares, Met-A (tared when it can be), the C-Score, and IAM-A, each
with its state and status. The PDFs use the same rows.

*Code:* `MethylPhys_Interface/report_v4.py` line 63 (70 lines) · arguments: o · calls: array run 2, bam run 2

### `report_v4.py::_shares`

The cell-shares section: each share with its 95 % interval and status, or the reason they were withheld.

*Code:* `MethylPhys_Interface/report_v4.py` line 134 (12 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_meta`

The Met-A section: its gauge (tared reading; an untared whole-blood reading has no gauge position) and its numbers.

*Code:* `MethylPhys_Interface/report_v4.py` line 147 (9 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_cscore`

The C-Score section: its bar, its sky and the overlay.

*Code:* `MethylPhys_Interface/report_v4.py` line 157 (9 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_iama`

The IAM-A section: its gauge, or the refusal.

*Code:* `MethylPhys_Interface/report_v4.py` line 167 (14 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::_iama_derived`

IAM-A on the derived line, printed beside A in the IAM-A section and marked in commissioning: A_derived, its state on band v2, and the
held error against the derived prediction; where it is not given, the reason (a BAM count, or clustering outside the derived line's table).
A and its state are unchanged.

*Code:* `MethylPhys_Interface/report_v4.py` line 183 (13 lines) · arguments: q · calls: not called on the traced runs · constants read: `E`

### `report_v4.py::_layout`

The section 'How the losses are laid out along the molecules': run lengths (loss_run_ratio, with runs of two seen and expected), the line tile
by tile (held error in firm and loose tiles), the spread over molecules (loss_dispersion), and the islands meter and clock sites (iam_a.sites).
Information only: no state uses them. Met-A's held-site score is not printed until its null test on healthy libraries is read.

*Code:* `MethylPhys_Interface/report_v4.py` line 198 (25 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`

### `report_v4.py::_figs`

Draws this sample's gauge and C-Score pictures beside the report.

*Code:* `MethylPhys_Interface/report_v4.py` line 237 (22 lines) · arguments: o, base · calls: array run 1, bam run 1

### `report_v4.py::_physics`

The physics section as HTML.

*Code:* `MethylPhys_Interface/report_v4.py` line 264 (6 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `E`, `PHYS_INTRO`

### `report_v4.py::physics_rows`

Each reading set in the physics it rests on, with this run's numbers. IAM-A: ε measured, ε predicted at the sample's own L, saturation
0.5/ε_pred, and the floor 1/(1 + e^M) as a fraction of ε_pred. Met-A: the mean entropy over the identity sites against its expectation,
what healthy 1 is on this run (the median of the same-run references, or the committed healthy reference), and on isolated neutrophils
the reading in bits per site. The C-Score: the clustering ratio and its band. The 0.330263 bits per site it prints is typed in this
function rather than read from `metA_floors_v1_3.json` (a fix on the Decoder's list).

*Code:* `MethylPhys_Interface/report_v4.py` line 271 (32 lines) · arguments: o · calls: array run 2, bam run 2 · constants read: `E`

### `report_v4.py::_skypair`

The overlay of the sample on its healthy reference, with how sure the healthy reference is at those sites.

*Code:* `MethylPhys_Interface/report_v4.py` line 308 (8 lines) · arguments: o · calls: array run 1 · constants read: `SKY_CAP`

### `report_v4.py::_plain`

The plain view: one gauge and one sentence per reading.

*Code:* `MethylPhys_Interface/report_v4.py` line 317 (19 lines) · arguments: o, F · calls: array run 1, bam run 1 · constants read: `E`

### `report_v4.py::build`

Builds the whole page: the technical record from `report_v3.build`, the figures, both views, and the two PDFs.

*Code:* `MethylPhys_Interface/report_v4.py` line 541 (33 lines) · arguments: o, out, sid · calls: array run 1, bam run 1 · constants read: `CSS`, `E`, `_SEQ`

### `report_v3.py::_gauge`

The gauge as inline HTML, drawn from the gauge marks file.

*Code:* `MethylPhys_Interface/report_v3.py` line 51 (38 lines) · arguments: A, lo, hi, w, label, instrument, top · calls: array run 1, bam run 2 · files named: `CANON/foundation.json`, `Runtime Matrices/Gauge/cell_gauge_marks_v1.json`, `drawn from Runtime Matrices/Gauge/cell_gauge_marks_v1.json`

### `report_v3.py::_H`

The entropy of a rate in bits, for the formula self-test.

*Code:* `MethylPhys_Interface/report_v3.py` line 90 (3 lines) · arguments: b · calls: array run 3, bam run 3

### `report_v3.py::_load`

Loads a chain file, or nothing if it is missing.

*Code:* `MethylPhys_Interface/report_v3.py` line 95 (3 lines) · arguments: rel · calls: array run 3, bam run 3 · constants read: `CHAIN`

### `report_v3.py::red_flags`

The red flags, from the bundle only.

*Code:* `MethylPhys_Interface/report_v3.py` line 101 (26 lines) · arguments: o · calls: array run 1, bam run 1

### `report_v3.py::safeguards`

The safeguards listed above.

*Code:* `MethylPhys_Interface/report_v3.py` line 129 (44 lines) · arguments: o, prose · calls: array run 1, bam run 1 · constants read: `CLAIM_SCAN` · files named: `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json`, `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`

### `report_v3.py::troubleshooting`

For each stop on this run, what it usually means and what to do.

*Code:* `MethylPhys_Interface/report_v3.py` line 175 (17 lines) · arguments: o · calls: array run 1, bam run 1

### `report_v3.py::toolkit_table`

The development stages and whether each ran on this sample.

*Code:* `MethylPhys_Interface/report_v3.py` line 194 (12 lines) · arguments: o · calls: array run 1, bam run 1 · constants read: `NOT_BUILT`, `TOOLKIT_STAGES`

### `report_v3.py::_table`

A plain HTML table.

*Code:* `MethylPhys_Interface/report_v3.py` line 208 (4 lines) · arguments: rows, head · calls: array run 7, bam run 7

### `report_v3.py::build`

The technical record: intake, Stage 1, the mix, Met-A, the tare, the noise gate, the C-Score, IAM-A, the red flags, the safeguards,
troubleshooting and the toolkit table, the run-it-yourself command, and a copy of the bundle inside the page. Returns the page's path,
size and any refusals; the bundle file itself is written by `run_sample.py`.

*Code:* `MethylPhys_Interface/report_v3.py` line 214 (83 lines) · arguments: o, out, sid · calls: array run 1, bam run 1 · constants read: `FLAG_ORDER`, `STAGES_2B` · files named: `cell_gauge_marks_v1.json`

### `report_figs_v4.py::sci`

A number in scientific notation for a figure.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 13 (2 lines) · arguments: v · calls: not called on the traced runs

### `report_figs_v4.py::_bar`

Draws one gauge bar: the gradient from the floor through healthy to the ceiling, the band, the needle and the ticks.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 15 (13 lines) · arguments: ax, lo_v, hi_v, band, A, fl, cl, y0, h, grad, ticks, mid · calls: array run 2, bam run 2

### `report_figs_v4.py::gauge_png`

The floor-to-ceiling gauge for Met-A or IAM-A, with the magnified strip when healthy sits near the floor.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 29 (20 lines) · arguments: out, name, A, floor, ceiling, band, subject, state, unit, ends · calls: array run 1, bam run 1

### `report_figs_v4.py::cscore_png`

The C-Score bar, scattered to clustered; the scale stretches to hold the reading.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 50 (10 lines) · arguments: out, C, band_hi, band_lo, subject, state · calls: array run 1

### `report_figs_v4.py::atlas_sky_png`

The healthy reference of the sample's cell from IAM-Atlas v2.1, at the same sites, in the same order and on the same pixels as the
sample's C-Score sky, so the two line up site for site.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 61 (40 lines) · arguments: out, ids, cell, atlas_dir, nest_pix2ang, title · calls: array run 1 · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `sites.parquet`

### `report_figs_v4.py::changes_sky_png`

The sample laid over its healthy reference at Met-A's 6,000 identity sites: the healthy cell in grays, color only on blocks that changed,
by the globe's rule (blocks of 10 neighboring sites in genome order, each block's mean departure beyond the 5 % family-wise line), so the
flat picture and the globe count the same change (healthy example blood: 0 of 597 blocks; the planted clustered loss: 44). Each pixel keeps
the block value of largest size, so a changed block is never averaged away.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 102 (55 lines) · arguments: out, z, cell, atlas_dir, nest_pix2ang, readings, thr, title · calls: array run 1 · files named: `import pyarrow.parquet`, `sites.parquet`

### `report_figs_v4.py::changes_sky_png.pext`

The block value of largest size in each pixel, its sign kept, so a changed block is never averaged away by an unchanged neighbor.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 128 (5 lines) · arguments: x · calls: not called on the traced runs

### `report_figs_v4.py::changes_sky_png.pmean`

The mean of the values falling in each pixel.

*Code:* `MethylPhys_Interface/report_figs_v4.py` line 126 (2 lines) · arguments: x · calls: array run 2

### `report_pdf_v4.py::_cap`

In the PDF report, a list reading longer than 400 characters (for example Met-A for every atlas cell type) shows its first eight entries and the count of the rest; the HTML report keeps the whole list. Table rows may split across pages (2026-11-12, after the release check found every PDF failing on a row taller than a page).

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 44 (5 lines) · arguments: x, n · calls: not called on the traced runs

### `report_pdf_v4.py::P`

A paragraph for the PDF, keeping only the tags the PDF library understands.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 25 (4 lines) · arguments: t, st · calls: array run 90, bam run 49 · constants read: `S`

### `report_pdf_v4.py::_img`

A figure scaled to a width, or nothing if missing.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 30 (4 lines) · arguments: path, w · calls: array run 8, bam run 6

### `report_pdf_v4.py::_head`

The PDF's heading block, with the research-use line and how to read the commissioning status.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 35 (8 lines) · arguments: o, sid, when, view · calls: array run 2, bam run 2

### `report_pdf_v4.py::_tbl`

A PDF table.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 50 (5 lines) · arguments: rows, widths, head · calls: array run 3, bam run 2

### `report_pdf_v4.py::build_pdfs`

The researcher and plain PDFs, from the same rows and pictures as the page.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 61 (62 lines) · arguments: o, F, out, sid, when · calls: array run 1, bam run 1

### `report_pdf_v4.py::build_pdfs.doc`

Writes one PDF.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 64 (3 lines) · arguments: path, story · calls: array run 2, bam run 2

## 14b. The report's globe: `report_globe_v4.py`

**What it is for.** A reviewer turns a globe in the HTML report and sees where the sample differs from its healthy reference, with a
slider that fades only the healthy background, so the change can be seen by itself. The change is fixed by the measurement: each site's
difference from the healthy reference in that site's healthy width (the atlas's person-to-person spread and its own uncertainty), centered
and scaled by the sample's own spread, then read in blocks of 10 neighboring sites against the fixed 5 % family-wise line (below). Healthy
blood: 45 of 77,355 blocks. Tried first and dropped the same day: the atlas's 95 % range alone flagged 84 % of a healthy blood (it is how
sure the atlas is of the healthy average, not how much a healthy person varies), and single sites beyond 3 lit 2.1 % of a healthy blood's
sites across the whole globe. **A picture, not a reading:** nothing is scored from it, and Met-A and the
C-Score keep their commissioned purified-array reference.

**Label: standard method** (display). The layout is the atlas's sky sphere: every IAM-Atlas v2.1 site in genome order at NESTED pixel
floor(i × NPIX / n), NSIDE 256, the same pixels as the atlas's cell pages and the public globe. The order is frozen in
`Runtime Matrices/Atlas_V21/sky_order_v21.npz` (built by `atlas/tools/sky_order_v21.py` with Illumina's EPIC v1 B4 and 450K manifests,
checked against the cell pages: 785,718 measured neutrophil pixels on both), so the chain draws it without downloading a manifest.

**The reference, by specimen.** An isolated cell type is compared with that cell in the atlas. Whole blood and buffy coat are compared
with the atlas's healthy blood for this person's own mix: the six panel cells weighted by their Stage D21 shares, scaled to sum to 1.
Comparing blood with one cell would light up every site where blood differs from that cell.

**The information sphere's expected healthy distance.** On a sequenced sample the sphere reports, beside its measured distance, the
distance a healthy sample of the same depth should show (fisher_rao_expected_rms_rad) and the ratio of the two (fisher_rao_over_expected).
At each site the expected angle squared is 1/(4n) for the sample's own n reads, at any level, plus the reference's own uncertainty
mu_se^2 / (4e(1 - e)) in the sample's shares; the counting part is summed exactly over the possible read outcomes,
since at a few reads a site 1/(4n) is up to a third too small. A healthy sample reads near 1; the measured distance alone mostly measures depth.

**Every in-mix reading against one donor half.** For a healthy atlas input to be read held out, every healthy level it is compared with must come from
donors it does not belong to. `--dev-seq-ref-half` did that only for sorted-cell Met-A and the islands; `--dev-seq-half-cols DIR` with it serves Met-A for the
whole sample, Met-A in the mix, the territories and the C-Score from that half's own columns (array_sites_halfA or _halfB, the same layout as the sequencing
atlas's columns, built from the atlas's per-CpG donor-half tables). The folder is outside the reference door, so the bundle records each file's checksum
(seq_half_columns). Development; without the flag nothing changes.

**A cell the atlas holds from one library.** The globe measures each site's change in its healthy width, which includes the person-to-person
spread of every cell in the sample's mix. Naive CD8, effector memory CD8 and naive CD4 T cells come from one library each, so they have no
spread of their own; when they held more than 1 % of a sample's shares, every site lost its width and the information sphere's territories
were empty (6 of 48 sorted libraries). Such a cell now borrows its lineage siblings' spread, the median at each site, and the globe lists
the cells that borrowed (width_borrowed).

**A specimen that names no cell.** The one button reads a sorted molecule file with the specimen 'sorted cells' and no cell named.
That label names no atlas cell, so the reference is the sample's own cells as the IAM-Deconvolver found them, weighted by their Stage D21
shares (in a sorted library the top share is about 0.98). Before this rule the globe looked for a cell called 'sorted cells', refused, and
the information sphere had no site-by-site change to place (152 of 156 atlas libraries in the joint run). With no shares it refuses and
says why.

**Cells the atlas does not hold (no-cell route).** When the Deconvolver's shares include cells the atlas has no column for, they are left out
and the rest rescaled if together they hold at most 1 % of the shares (the coverage rule Met-A uses at every site); the reference names them.
Above 1 % the globe refuses and names them.

**Four layers.** The sample; the reference; the difference (sample minus reference) at every site both hold; and Met-A's 6,000 neutrophil
identity sites as the C-Score's departure map, centered and scaled by the sample's own median and spread as the report's overlay, drawn as
dots of radius 3 texture pixels at their own places so that a run of departing sites shows as a patch. The page prints the share of the
measured globe above the chosen size, from the equal-area sky pixels.

**What a reader will see on healthy blood.** The atlas blends 21 sources and sits about 0.006 closer to 0.5 than one array
(DEV-REFERENCES-01), so a faint even difference everywhere (median 0.018 on the real healthy example, 2.9 % of the globe above 0.10). In a
woman, the X chromosome as a patch: one X is held methylated, and the reference pools both sexes (the last 2 % of the genome order differed 7
to 11 times as often as the rest on the first real blood). A small loss spread over many sites does not show on the whole-genome layer; it
shows on the departure layer.

**How to break it.** Lay the atlas's own healthy neutrophil on the globe as an isolated neutrophil sample: every difference must be 0
(release check V12). Give whole-blood shares that do not sum to 1: the weights must rescale. Edit the atlas site list: the globe must refuse
to draw, because the frozen order no longer matches.

**Two more layers (2026-10-21).** Met-A's own reference, the purified arrays: at its 6,000 identity sites, the purified group profiles
weighted by this blood's Stage A fractions (or the neutrophil profile for isolated neutrophils), with the sample's difference from it, drawn
as dots. And each cell the IAM-Deconvolver read, named with its share of this sample, as that cell's healthy map in IAM-Atlas v2.1. Each
reading is shown against the reference its instrument reads against.

### `stage_metaa_atlas.py::cscore_whole_departures`

The whole-genome departure map behind the sequencing C-Score's whole_genome reading: at every atlas site the sample measured, H(beta) minus
H(e), e the atlas mixed in the sample's own Deconvolver shares, in genome order. The sky plate draws it for a sequenced sample, which has no
array C-Score map (before this the plate was never drawn on sequencing). The neutrophil overlays stay with the array map. Development.

*Code:* `stage_metaa_atlas.py` line 811 (11 lines) · arguments: beta, shares · calls: not called on the traced runs · constants read: `CS_BLOCK`, `MIX_COVER`

### `report_globe_v4.py::_dots`

Draws a sparse texture as dots: every texture pixel within 3 of a measured one takes the nearest one's value.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 166 (7 lines) · arguments: v · calls: not called on the traced runs · constants read: `R_DOT`

### `report_globe_v4.py::purified`

Met-A's purified-array reference at its 6,000 sites: the Stage A fractions times the purified group profiles for blood, the neutrophil profile
for isolated neutrophils; nothing for other specimens.

**One viewer for every globe (`globe_viewer.py`, 2026-10-21).** A globe is a spec: its pictures (each stored once), and groups of layers. A
layer is a picture shown as it is, the fixed change with a slider that fades only the background (the change and its count never move), or a
series of pictures a slider steps through (the noise maps). The same code draws the globe inside a report, the report's own full page
(written beside the report as `<name>_globe.html` and linked from it), and the site's Globe Explorer (`atlas/sphere.html`). It needs only
the standard library, so the site build imports it unchanged; release check V12 and the site build both check that its script parses.

**The map-layer slider (2026-10-22, lesson C35).** A map layer (a value picture: a cell's methylation, how sure the atlas is) carries a
lossless value picture beside it. Sliding right keeps only the sites at or above the chosen value; **every other pixel goes dark**, the sites
below the value, the sky with no site, and the rim of each dot alike, so at the right end the globe is dark. Release check V13 requires at
least 90 % of the globe dark at every map layer's right end, layers drawn as dots included.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 174 (9 lines) · arguments: fractions, specimen · calls: not called on the traced runs · constants read: `BLOOD`, `CH` · files named: `blood_composition_EPIC_v1.json`

### `globe_viewer.py::inline`

The reverse of dedupe: each layer's picture written back in place of its asset name, so groups taken from several globes can be joined into one.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 13 (10 lines) · arguments: spec · calls: not called on the traced runs

### `globe_viewer.py::dedupe`

Stores every picture of a spec once under a short name; layers refer to it by name (a reference is both a layer and a difference's gray).

*Code:* `MethylPhys_Interface/globe_viewer.py` line 24 (13 lines) · arguments: spec · calls: not called on the traced runs

### `globe_viewer.py::dedupe.one`

Gives one inlined picture its name, reusing the name if the same picture was already stored.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 27 (4 lines) · arguments: v · calls: not called on the traced runs

### `globe_viewer.py::fragment`

The globe inside a report: one section with the controls, the viewer and the specs inlined, so the report stays one file that works offline.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 118 (4 lines) · arguments: specs, size · calls: not called on the traced runs · constants read: `CONTROLS`, `VIEWER_JS`

### `globe_viewer.py::page`

A full dark page for one or more globes with a globe chooser: a report's own globe page, or the site's Globe Explorer.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 123 (8 lines) · arguments: specs, title, back, size, intro · calls: not called on the traced runs · constants read: `CONTROLS`, `VIEWER_JS`

### `globe_viewer.py::write_files`

Writes a spec's inlined pictures as files for the site and returns the spec pointing at them, so each globe loads only when it is chosen.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 132 (13 lines) · arguments: spec, d, url · calls: not called on the traced runs

### `globe_viewer.py::write_files.one`

Writes one inlined picture to its file and returns its address.

*Code:* `MethylPhys_Interface/globe_viewer.py` line 135 (4 lines) · arguments: v · calls: not called on the traced runs

### `report_globe_v4.py::_to_sky`

One value per site in genome order to a NESTED map: the mean per pixel, nothing where no site was measured (the atlas tools' own rule).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 81 (5 lines) · arguments: v, npix · calls: not called on the traced runs

### `report_globe_v4.py::_borrow_sd`

For a cell whose person-to-person spread (donor_sd) is missing at every site, a type the atlas holds from one library,
the median spread of its lineage siblings at each site (stage_0_intake.LINEAGE); returns the spreads and which cells borrowed from whom.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 87 (15 lines) · arguments: sd_, cells, mu_fn · calls: not called on the traced runs

### `report_globe_v4.py::_mu`

Reads the atlas best values of the named cells from the compact copy (`atlas/v2.1/mu_part*.parquet`), 65535 decoded as no value.

**The change is read in the C-Score's blocks, beyond a fixed family-wise line.** Judged site by site, a change beyond 3 spreads lit 2.1 %
of a healthy blood's sites, spread over the whole globe (16,372 of 773,556), while a loss that runs along the genome (many small moves the same
way) passes 3 at hardly any single site. So both change layers read blocks of 10 neighboring measured sites in genome order, the C-Score's
block. On the whole genome each block's mean is taken in the sample's own block spread (neighboring CpGs move together, so a healthy block
spreads about twice as wide as independent sites would); on Met-A's 6,000 nearly independent sites the block mean is in standard errors. A
block has changed beyond the line pure chance crosses less than once in 20 genomes (5 % family-wise, two-sided; 4.98 for the genome's 77,355
blocks, 3.93 for Met-A's 597): FW_LINE. Healthy blood: 45 of 77,355 genome blocks and 0 of 597 Met-A blocks; the planted clustered example:
44 Met-A blocks, in clumps; 200 planted runs of 30 sites in a copy of the healthy blood: 229 of their 756 blocks found. The slider ends on
pure black, so only the changes stay lit.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 103 (12 lines) · arguments: cells, stem · calls: not called on the traced runs · constants read: `ATLAS`, `NA` · files named: `import pyarrow.parquet`

### `report_globe_v4.py::FW_LINE`

The fixed line for n blocks: the size a normal block crosses by chance less than once in 20 genomes, two-sided (5 % family-wise).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 74 (4 lines) · arguments: n · calls: not called on the traced runs

### `report_globe_v4.py::_vmask`

A map's values (0 to 1) as a half-size 8-bit picture for the page: 0 where no site, 1 to 255 the value. With it every map layer has a slider
that keeps only the sites at least as methylated as the chosen value, and prints the share of the measured globe kept (2026-10-21; before,
the map layers had no tool). A browser that color-manages the picture may shift a code by a step or two, about 1 % of the scale at the
slider's line: a viewing tool, never a reading.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 135 (6 lines) · arguments: v · calls: not called on the traced runs

### `report_globe_v4.py::sex_sites`

Loads the frozen list of IAM-Atlas v2.1's sites on the X (14,759) and Y (3) chromosomes, `Runtime Matrices/Atlas_V21/sex_chromosome_sites_v21.json`,
built by `atlas/tools/sex_sites_v21.py` from the two Illumina manifests Stage 1 is pinned to (it refuses any other manifest).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 46 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `SEX_SITES`, `_XY`

### `report_globe_v4.py::_xy_index`

The X and Y site names as a pandas Index, built once, so a block's sex-chromosome check is a hashed lookup (DEV-SPEED-01: np.isin on site names had taken about 10 s per call).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 53 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `_XY`

### `report_globe_v4.py::_xy_blocks`

How many changed blocks hold at least one X or Y site. Every change count on the globe is printed with this count beside it.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 58 (6 lines) · arguments: ids_in_order, mi, nb, changed · calls: not called on the traced runs · constants read: `BLOCK`

### `report_globe_v4.py::key_css`

The parchment scale of the methylation maps as a color bar, for the color key under every map layer.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 65 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `HERE` · files named: `Planck_Parchment_RGB.txt`

### `report_globe_v4.py::build_pair`

DEVELOPMENT, not commissioned: one person's sample (a tumor) against the same person's own other sample (its adjacent normal) on the atlas's
layout, by the globe's block rule without the atlas: each site's difference in the pair's own spread, blocks of 10 neighboring sites, the
block mean in its own block spread, changed beyond the 5 % family-wise line. Run as `python3 report_globe_v4.py pair SAMPLE REFERENCE NAME
OUT.html` on chain Stage 1 values. A pair with no change lights 0 of 77,356 blocks; 200 planted regional runs of 30 sites light 620.
`drop` leaves listed sites out before anything is computed (the X and Y, so a difference of sex is not counted as a change). The record also
gives the changed blocks above and below the reference and the share of the globe's measured pixels the change lights.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 435 (44 lines) · arguments: beta, ref_beta, ref_name, out_html, title, drop · calls: not called on the traced runs · constants read: `ATLAS`, `BLOCK`, `HERE`, `MV`, `ORDER`, `PAIR_ABOUT` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `sites.parquet`

### `report_globe_v4.py::build_pair.rgb`

Draws a methylation map for the pair's globe on the atlas pages' parchment scale; gray where no site was measured.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 457 (2 lines) · arguments: m · calls: not called on the traced runs

### `report_globe_v4.py::build._code`

Codes a change map for the page: 0 where no site was measured, 128 where the sites are unchanged, and 128 plus or minus at least one step for a
changed site, its size up to a fixed full scale (10 spreads), so the page can fade the unchanged background without ever touching the change.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 288 (6 lines) · arguments: v, full · calls: not called on the traced runs

### `report_globe_v4.py::_to_sky_extreme`

One value per site to the sky map, keeping in each pixel the value of largest size with its sign, so a changed site is never averaged away by
an unchanged neighbor sharing its pixel.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 116 (6 lines) · arguments: v, npix · calls: not called on the traced runs

### `report_globe_v4.py::_lookup`

For each pixel of the 1024 × 512 texture, the nearest sky pixel center (the chain's `stage_v4.nest_pix2ang`).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 123 (7 lines) · arguments: nside · calls: not called on the traced runs · constants read: `CH`

### `report_globe_v4.py::_png_b64`

A grayscale texture as base64 PNG, for the difference and departure codes (0 = no site; 1 to 255 = −0.5 to +0.5, or −5 to +5 spreads).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 131 (3 lines) · arguments: arr, mode · calls: not called on the traced runs

### `report_globe_v4.py::_jpg_b64`

A color texture as base64 JPEG, for the sample and reference layers.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 143 (3 lines) · arguments: rgb · calls: not called on the traced runs

### `report_globe_v4.py::reference`

The healthy reference for a specimen: the matching atlas cell, or for blood the six panel cells weighted by the Stage D21 shares and scaled
to 1; refuses with the reason when there are no shares or no specimen.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 147 (18 lines) · arguments: specimen, shares · calls: not called on the traced runs · constants read: `ATLAS_NAME`, `BLOOD`, `PANEL`

### `report_globe_v4.py::_paint_blocks`

Drawing only. A changed block of 10 sites (the Met-A layer, each cell's layer) is painted over the stretch of the genome it covers, from its
first placed site to its last, at their places on the genome-order sky. Met-A's 6,000 sites lie far apart along the genome, so a block drawn as
10 dots landed as 10 specks in different places; painted over its stretch (a median of about 0.15 % of the sphere) a clustered loss shows as
patches where it sits on the genome. Which blocks changed, and every count, are untouched; a block whose sites lie far apart gives a large patch
from only 10 measured sites, and the layer's note says so.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 184 (14 lines) · arguments: sky, vals, posidx, npix, n · calls: not called on the traced runs

### `report_globe_v4.py::_patch`

Drawing only. On the whole-genome change layer a block of 10 neighboring sites covers only a few texture pixels; each changed pixel is drawn as a
patch of the dots' radius (R_DOT), the larger change winning where two meet. Unchanged and unmeasured pixels and every count are untouched.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 199 (8 lines) · arguments: v · calls: not called on the traced runs · constants read: `R_DOT`

### `report_globe_v4.py::_cell_change`

One cell's change layer on the globe (development, 2026-10-22): the departure behind Met-A for that cell
(`stage_metaa_atlas.departure_in_mix`), in the sample's own spread at that cell's identity sites, read in blocks of 10 of those sites in genome
order; a block changed beyond the 5 % family-wise line is lit, drawn as dots. Each cell the IAM-Deconvolver read gets one, in the group
"each cell's change". It shows a loss that clusters along the genome; a loss spread evenly over every site of a cell is canceled by the
sample's own centering and shows in that cell's Met-A number instead (the example blood with a 5 % even CD4 T loss: 2 of 597 blocks
before and after).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 208 (19 lines) · arguments: d, ids_ordered, npix, look · calls: not called on the traced runs · constants read: `BLOCK`, `R_DOT`

### `report_globe_v4.py::build`

Checks the frozen layout against the atlas site list, builds the four layers, the shares above each size, and the page fragment (canvas,
controls and script, no outside library). Returns the record kept in the bundle, marked `scored: False`.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 228 (191 lines) · arguments: beta, specimen, shares, out_html, z, fractions, sex, cell_departures, counts · calls: not called on the traced runs · constants read: `ABOUT`, `ATLAS`, `ATLAS_NAME`, `BLOCK`, `CH`, `HERE`, `MV`, `ORDER`, `PANEL`, `R_DOT`, `SEXNOTE`, `SEXWORD` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `neutrophil_reference_v1_2.json`, `sites.parquet`, `sky_order_v21.npz`

### `report_globe_v4.py::build.rgb`

Colors a sky map for the texture with the atlas pages' parchment scale, 0 to 1; gray where no site was measured.

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 285 (2 lines) · arguments: m · calls: not called on the traced runs

## 15. The development readings: `dev_stages.py`, `dev_comp_methods.py`, `deconv_v2.py`, `sky_statistics.py`

**What they are for.** Readings under development, run only when a `--dev-*` flag asks for them. Each writes its own record under
`bundle["development"]`, labeled "DEVELOPMENT - not commissioned", and none changes the reading, the gauge or the tare. They are in the
chain so a development result is made by the same code a researcher can run. The one exception is `dev_stages.selftare_map` (section 5),
which has been Stage T step 1 on every reading since 2026-10-04 and still lives in this file.

**Labels.** Direction, brightness and the per-cell B reading follow Met-A's physics; the composition methods and the sky statistics
are standard methods (deconvolution, the angular power spectrum of a sphere with a shuffled null).

### `dev_stages.py::_H`

The entropy of a beta value in bits.

*Code:* `dev_stages.py` line 33 (3 lines) · arguments: b · calls: not called on the traced runs

### `dev_stages.py::_bc`

Loads the blood composition matrix once.

*Code:* `dev_stages.py` line 47 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `RM`, `_C` · files named: `blood_composition_EPIC_v1.json`

### `dev_stages.py::_rec`

A development record with its label.

*Code:* `dev_stages.py` line 52 (2 lines) · arguments: stage · calls: not called on the traced runs · constants read: `DEV_LABEL`

### `dev_stages.py::selftare_ii`

The development form of self-tare II: Met-A re-read on the mapped values beside the reading. Kept for its earlier records; the one command
no longer calls it, because step 1 now runs on every reading.

*Code:* `dev_stages.py` line 60 (9 lines) · arguments: beta, specimen · calls: not called on the traced runs

### `dev_stages.py::anchor_range`

Development (DEV-SELFTARE-03, preliminary): the span between the array's type II fixed-site anchors, R = U − L, the number the self-tare
divides by. A loss of the neutrophils' pattern that reaches every site is mapped away by the self-tare but narrows this span: tared against at
least 3 same-run healthy references, R_rel = R / median(R_refs), and the loss is (1 − R_rel) / f_neu with a limit of 2 × the references' spread /
f_neu. On synthetic bloods planted 1, 2 and 5 % read back as 0.97, 1.95 and 4.94 %. Not called by the one command yet; changes no reading.

*Code:* `dev_stages.py` line 72 (18 lines) · arguments: beta, f_neu, refs, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::site_uncertainty`

This array's own per-site uncertainty: the SD of β against the reference value over its fixed sites, per design and state.

*Code:* `dev_stages.py` line 92 (8 lines) · arguments: beta, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::s_for`

The per-site uncertainty for a list of sites, by design and state (low when the reference value is below 0.5).

*Code:* `dev_stages.py` line 102 (6 lines) · arguments: sites, ref_values, su, st · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::direction`

D = mean over the identity sites of |r − 0.5| − |β − 0.5|: above 0, the sites have moved toward 0.5, toward disorder; below 0, away from it.

*Code:* `dev_stages.py` line 111 (6 lines) · arguments: beta, ref, sites · calls: not called on the traced runs

### `dev_stages.py::direction_record`

The direction record at the neutrophil identity sites against the cell's own reference (the expectation in whole blood), tared on the
same-run references' D.

*Code:* `dev_stages.py` line 119 (15 lines) · arguments: beta, specimen, comp, refs_D · calls: not called on the traced runs

### `dev_stages.py::trace_cell`

In an isolated-neutrophil specimen, another blood group's contamination, called above 3 × this array's own standard error.

*Code:* `dev_stages.py` line 137 (15 lines) · arguments: beta, su, line · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::foreign_cell`

In whole blood, a non-blood template solved beside the 8 blood groups, called above 3 × this array's own standard error.

*Code:* `dev_stages.py` line 155 (22 lines) · arguments: beta, template, su, line · calls: not called on the traced runs · files named: `dev_foreign_placenta_EPIC_v1.json`, `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::brightness`

A 95 % interval on Met-A by Monte Carlo from this array's per-site uncertainty.

*Code:* `dev_stages.py` line 180 (17 lines) · arguments: beta, specimen, comp, n_draw, su, seed · calls: not called on the traced runs · files named: `dev_selftare_typeII_EPIC_v1.json`

### `dev_stages.py::_mapping`

Loads the CpG-to-HEALPix map of the development sky (NSIDE 128).

*Code:* `dev_stages.py` line 200 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `RM`, `_C` · files named: `iamatlas_cpg_to_healpix_nside128.npz`

### `dev_stages.py::residual_z`

The development sky's per-site departure: z = (β − Σ f_g μ_g) / √(Σ f_g² (sd² + s_bp²) + 0.02²), with s_bp the atlas's between-person SD.

*Code:* `dev_stages.py` line 208 (6 lines) · arguments: beta, fractions, AP · calls: not called on the traced runs · constants read: `PARENTS`

### `dev_stages.py::_pixmap`

The mean departure in each pixel.

*Code:* `dev_stages.py` line 216 (3 lines) · arguments: zs, pix, npix · calls: not called on the traced runs

### `dev_stages.py::block_shuffle_order`

A permutation that moves whole runs of consecutive sites within each chromosome: the null keeps short-range structure along the genome and
breaks where it sits.

*Code:* `dev_stages.py` line 221 (7 lines) · arguments: chr_keys, run, rng · calls: not called on the traced runs · constants read: `SKY_RUN`

### `dev_stages.py::sky_from_z`

The development sky's band powers against the block-shuffle null and the look-elsewhere statistic by simulation.

*Code:* `dev_stages.py` line 230 (14 lines) · arguments: z, n_null, n_lee, seed, free_null · calls: not called on the traced runs · constants read: `SKY_LEE_N`, `SKY_NULL_N`, `SKY_RUN`

### `dev_stages.py::sky_from_z.shuf`

The band powers of one block-shuffled sky.

*Code:* `dev_stages.py` line 235 (2 lines) · arguments: none · calls: not called on the traced runs

### `dev_stages.py::_has_cols`

Whether a parquet file holds the named columns.

*Code:* `dev_stages.py` line 246 (4 lines) · arguments: path, cols · calls: not called on the traced runs · files named: `import pyarrow.parquet`

### `dev_stages.py::load_atlas_parents`

Loads the atlas v2 parent blood cells' means, SDs and between-person SDs.

*Code:* `dev_stages.py` line 296 (9 lines) · arguments: atlas_parquet · calls: not called on the traced runs · constants read: `PARENTS`, `_C`

### `dev_stages.py::sky`

The development residual sky of one whole blood, with its null.

*Code:* `dev_stages.py` line 307 (7 lines) · arguments: beta, fractions, atlas_parquet · calls: not called on the traced runs

### `dev_stages.py::_atlas_e`

Builds the atlas_e solver once on the 12 array-measured circulating cells of atlas v2, with the neutrophil identity sites removed.

*Code:* `dev_stages.py` line 321 (11 lines) · arguments: atlas_parquet · calls: not called on the traced runs · constants read: `ATLAS_E_CELLS`, `_C`

### `dev_stages.py::atlas_e`

The atlas_e composition, summed into the 8 blood groups.

*Code:* `dev_stages.py` line 334 (7 lines) · arguments: beta, atlas_parquet · calls: not called on the traced runs · constants read: `BLOOD8`

### `dev_stages.py::nilc_e`

The NILC-e composition on the atlas_e templates, with each share's noise.

*Code:* `dev_stages.py` line 343 (7 lines) · arguments: beta, atlas_parquet · calls: not called on the traced runs · constants read: `BLOOD8`

### `dev_stages.py::percell_b`

A B-cell Met-A against the expectation for this blood's mix, on the development B-cell reference.

*Code:* `dev_stages.py` line 352 (14 lines) · arguments: beta, fractions, atlas_parquet · calls: not called on the traced runs · constants read: `PARENTS`, `RM` · files named: `metA_floors_v1_2_ALLCELLS_development.json`

### `dev_stages.py::epicv2_calibrate`

An EPIC v2 IDAT pair through SeSAMe (an R package, run in its own environment) to β values on EPIC v1 probe names, replicate probes of one
CpG averaged and background-masked probes dropped. Development only: the chain's references are EPIC v1.

*Code:* `dev_stages.py` line 372 (19 lines) · arguments: grn, red, rscript, prep · calls: not called on the traced runs · constants read: `SESAME_R` · files named: `betas.csv`

### `dev_comp_methods.py::load_atlas`

Loads an atlas parquet file indexed by site.

*Code:* `dev_comp_methods.py` line 46 (4 lines) · arguments: path · calls: not called on the traced runs

### `dev_comp_methods.py::atlas_cells`

The cell names in an atlas table.

*Code:* `dev_comp_methods.py` line 52 (2 lines) · arguments: A · calls: not called on the traced runs

### `dev_comp_methods.py::subset_atlas`

The columns of the named cells.

*Code:* `dev_comp_methods.py` line 56 (3 lines) · arguments: A, keep · calls: not called on the traced runs

### `dev_comp_methods.py::build_deconv`

An atlas v2 solver on the named cells, with the neutrophil identity sites removed from its pool.

*Code:* `dev_comp_methods.py` line 61 (4 lines) · arguments: A, keep, exclude_sites · calls: not called on the traced runs · constants read: `POOL1`, `SOLVER`

### `dev_comp_methods.py::twin_merge`

The atlas's own twin rule: a cell with no array measurement whose marker profile correlates with an array-measured cell above the
threshold is that cell on another platform, and the two merge.

*Code:* `dev_comp_methods.py` line 67 (28 lines) · arguments: A, D, roster, twin_tab, thr, candidates · calls: not called on the traced runs · files named: `records/10_twin_test_sample_level.csv`, `twin_family_thresholds_v1.json`

### `dev_comp_methods.py::mixture_twin`

A cell whose template is reproduced by a non-negative, sum-to-1 mix of the other cells, above the twin correlation, cannot be told apart
from that mix in a specimen.

*Code:* `dev_comp_methods.py` line 97 (18 lines) · arguments: D, candidates, r_thr · calls: not called on the traced runs

### `dev_comp_methods.py::crlb`

The Cramér–Rao bound on each share's SD for a solver's design at its markers.

*Code:* `dev_comp_methods.py` line 117 (10 lines) · arguments: D, f, sigma · calls: not called on the traced runs

### `dev_comp_methods.py::NNLS8.__init__`

Stage A's solver as a class: the 963 markers and 8 groups.

*Code:* `dev_comp_methods.py` line 130 (2 lines) · arguments: self, bc · calls: not called on the traced runs

### `dev_comp_methods.py::NNLS8.deconvolve`

Stage A's shares, for comparison with the other methods.

*Code:* `dev_comp_methods.py` line 132 (4 lines) · arguments: self, beta · calls: not called on the traced runs

### `dev_comp_methods.py::ledoit_wolf`

Analytic Ledoit–Wolf shrinkage of a covariance toward a multiple of the identity; no parameter to choose.

*Code:* `dev_comp_methods.py` line 138 (7 lines) · arguments: X · calls: not called on the traced runs

### `dev_comp_methods.py::NILC.__init__`

The NILC solver: sites, cells, means, variances and the noise model.

*Code:* `dev_comp_methods.py` line 150 (4 lines) · arguments: self, sites, cells, mu, v, cov, sigma, tech_var, tech_cov, offset · calls: not called on the traced runs

### `dev_comp_methods.py::NILC._solve`

One weighted, constrained solve.

*Code:* `dev_comp_methods.py` line 154 (11 lines) · arguments: self, m, f · calls: not called on the traced runs

### `dev_comp_methods.py::NILC.deconvolve`

The NILC shares in two passes.

*Code:* `dev_comp_methods.py` line 165 (9 lines) · arguments: self, beta, passes · calls: not called on the traced runs

### `dev_comp_methods.py::to_groups`

Sums atlas cells into the 8 blood groups.

*Code:* `dev_comp_methods.py` line 176 (7 lines) · arguments: fr · calls: not called on the traced runs · constants read: `SUBGROUP`

### `dev_comp_methods.py::met_a`

Whole-blood Met-A with another method's shares and profiles, for comparison with Stage M.

*Code:* `dev_comp_methods.py` line 185 (8 lines) · arguments: beta, fractions, profiles, ns · calls: not called on the traced runs · constants read: `H`

### `dev_comp_methods.py::ceiling_flag`

The entropy-ceiling flag for that comparison reading.

*Code:* `dev_comp_methods.py` line 195 (16 lines) · arguments: beta, profiles, fractions, ns · calls: not called on the traced runs

### `dev_comp_methods.py::read_beta`

Reads one β file.

*Code:* `dev_comp_methods.py` line 213 (3 lines) · arguments: path · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2.__init__`

The atlas v2 solver: loci measured by every modeled cell; for each cell, markers chosen against its nearest other cell (margin at least
0.20, the top 200 by margin over pooled SD); variance v = μ_sd² + donor_sd².

*Code:* `deconv_v2.py` line 11 (34 lines) · arguments: self, atlas_parquet, pooled_one_sample, margin, top, min_markers, sigma, drop, markers, k_near, per_pair, pair_margin, A · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2._solve`

One joint solve with f ≥ 0 and Σ f = 1, weights 1 / (Σ_c f_c² v_ci + σ²) iterated 5 times.

*Code:* `deconv_v2.py` line 46 (7 lines) · arguments: self, y, M, V · calls: not called on the traced runs

### `deconv_v2.py::DeconvV2.deconvolve`

The shares, and each cell's presence by 200 bootstrap resamples of the markers.

*Code:* `deconv_v2.py` line 54 (20 lines) · arguments: self, beta, n_boot, seed, present_floor, affine · calls: not called on the traced runs

### `sky_statistics.py::bandpowers`

The mean power in each of six bands of the angular spectrum.

*Code:* `sky_statistics.py` line 30 (3 lines) · arguments: cl, bands · calls: not called on the traced runs · constants read: `BANDS`

### `sky_statistics.py::apodised_mask`

A pixel weight that rises smoothly from 0 at the edge of the measured footprint to 1 a set angle inside it, so a sharp edge does not
spread power between multipoles.

*Code:* `sky_statistics.py` line 35 (20 lines) · arguments: good, apod_deg, taper · calls: not called on the traced runs

### `sky_statistics.py::_weights`

Keeps the last mask's weights.

*Code:* `sky_statistics.py` line 57 (5 lines) · arguments: good, apod_deg, taper · calls: not called on the traced runs · constants read: `_W`

### `sky_statistics.py::masked_spectrum`

The pseudo angular power spectrum of a sky with masked pixels, with a hard or apodized mask.

*Code:* `sky_statistics.py` line 64 (13 lines) · arguments: pix, lmax, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `MASKS`

### `sky_statistics.py::shuffled_null`

The band powers of skies made by shuffling the sky's own measured pixel values among its measured pixels: the mask, the count and the
distribution are kept, and only the arrangement is destroyed, so a difference from this null is spatial structure.

*Code:* `sky_statistics.py` line 79 (16 lines) · arguments: pix, n_perm, lmax, rng, bands, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `BANDS`, `MASKS`

### `sky_statistics.py::sky_spectrum_record`

One sky's spectrum, band powers, measured fraction, share of pixels with |z| above 2, and its null.

*Code:* `sky_statistics.py` line 97 (11 lines) · arguments: pix, n_perm, lmax, rng, mask, apod_deg, taper · calls: not called on the traced runs · constants read: `BANDS`

## 16. Met-A against IAM-Atlas v2.1 (development): `stage_metaa_atlas.py`

**What it is for.** Reading an isolated cell of any type the atlas holds by Met-A, against that cell in IAM-Atlas v2.1 instead of against
purified arrays. Run by the one command with `--dev-metaa-atlas CELL`, and on its own when intake accepts an isolated, sorted or purified atlas
cell (Section 1, `stage_0_intake.atlas_cell_of`). Development, not commissioned: written under `development.metaa_atlas`.

**Label: from IAM's physics.** The same reading as Met-A: mean entropy H(β) at the cell's identity sites over the entropy the reference
predicts there. A held site sits near 0 or 1; a loss moves it toward 0.5, where H is largest, so a loss raises the reading.

**Label: standard.** The identity sites follow Met-A's own site rule, applied to the atlas: healthy width at most 0.05 (the person-to-person
spread combined with the atlas's own uncertainty, half its 95 % range over 1.96), and a best value from 0.75 to 0.95 (methylated side) or 0.05
to 0.25 (unmethylated side); where more than 3,000 qualify on a side, the 3,000 of smallest width. A reading needs at least 90 % of them
measured. The tare (a median of at least 3 same-run healthy references, as Stage T step 2; never the sample itself) is retired and runs only to rebuild earlier records; each reading is divided by its own healthy prediction instead.

**The territories, read on every run.** Besides each cell's identity sites, this stage reads two fixed site sets the identity readings do not
reach (`territory_sets`, `territory_reading`): the islands, sites every healthy cell type keeps unmarked, and the domains, marked sites some
healthy cell types hold loosely, each on the A gauge (a reading over its healthy reference) against IAM-Atlas v2.1 at each site's own read count. In development: read and
flagged on arrays and sequencing (INSTRUMENTS.md, In development).

**The physics, step by step.** The same steps as Met-A on arrays (Section 5), with IAM-Atlas v2.1 as the reference and two additions.

1. **The reference is the atlas's measured level (M4).** Each cell's identity sites by Met-A's own rule applied to the atlas; each reading divided
   by its own healthy prediction (the atlas in this sample's mix, passed through the array's own measured line and noise).
2. **On sequencing, one scale (calibrated).** The atlas column is put in sequencing units first, sequencing = −0.0902 + 1.1026 × atlas level
   (`sequencing_term_v1.json`), because arrays read unmarked sites near 0.055 where sequencing reads 0 (M3).
3. **The islands (measured, T1).** Sites every healthy cell type keeps unmarked carry only the meter: islands error = 1.01
   × meter + 0.0001 (PHY-093); the writer leaves essentially no stray marks there.
4. **The domains (a lead, T2).** Marked sites some healthy types hold loosely: derivable as IAM-A's held error at the site's own loose share
   (intercept from M, A6; slope A7). Today read against the atlas (measured).
5. **What is open.** Each type's own level (A8); the per-site loose share the domains' derivation needs (T2).

### `report_v4.py::_fam_of`

Each atlas cell type's family (stage_info_sphere.FAMILIES), for the report's unexpected-cell note.

*Code:* `MethylPhys_Interface/report_v4.py` line 342 (5 lines) · arguments: none · calls: not called on the traced runs

### `report_v4.py::detail_tables`

The one source of the report's instrument sections, drawn by the HTML report and the researcher PDF alike: for each instrument (atlas and ruler, unexpected cells, Met-A whole sample, Met-A for each cell, commissioned Met-A, territories, the four C-Scores, IAM-A, Group-A, the globe, the information sphere) a title, what it reads, a table, and a note that says the state or why it did not fire. Every instrument always appears.

*Code:* `MethylPhys_Interface/report_v4.py` line 352 (94 lines) · arguments: o · calls: not called on the traced runs · constants read: `BLOOD_FAMILIES`, `DEV`

### `report_v4.py::_f`

Prints a number to a fixed number of decimals, or the value as text.

*Code:* `MethylPhys_Interface/report_v4.py` line 349 (1 lines) · arguments: v, d · calls: not called on the traced runs

### `report_v4.py::report_text`

Loads the report's explanations from Runtime Matrices/Report/report_text.json, which CANON/report_text.py generates from CANON/catalogue.json (the one source); read once per run.

*Code:* `MethylPhys_Interface/report_v4.py` line 451 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `REPORT_TEXT`, `_RT` · files named: `generated from CANON/catalogue.json`

### `report_v4.py::explain`

The explanation of one report section as (label, text) pairs, in a fixed order: what it measures, where it comes from, what it is compared to, the physics, why it matters, the link to the sky where there is one, and its status.

*Code:* `MethylPhys_Interface/report_v4.py` line 471 (11 lines) · arguments: sec · calls: not called on the traced runs · constants read: `EXPLAIN_LABELS`, `_SEQ`

### `report_v4.py::_is_seq`

True when the sample was read from sequenced molecules (IAM-A counted reads, or sequencing values). Sets which explanation the report prints (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 456 (3 lines) · arguments: o · calls: not called on the traced runs

### `report_v4.py::_ikey`

Sorts the sequencing instruments in chain order (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 460 (2 lines) · arguments: k · calls: not called on the traced runs

### `report_v4.py::_seq_block`

One sequencing instrument's explanation: the field's terms, how it is measured, IAM's Law (the steps it rests on, with their status), where the method comes from, and the credit for every method it borrows. Text from report_text.json's sequencing part, generated from the cell instruments database (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 462 (5 lines) · arguments: ins · calls: not called on the traced runs · constants read: `SEQ_LABELS`

### `report_v4.py::explain_seq`

The sequencing instruments printed under one report section (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 467 (4 lines) · arguments: sec · calls: not called on the traced runs · files named: `from report_text.json`

### `report_v4.py::instruments_seq`

Every sequencing instrument with its explanation, in chain order: the report's appendix 'How each instrument works' (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 482 (4 lines) · arguments: none · calls: not called on the traced runs

### `report_v4.py::_instruments_seq_html`

The appendix in the HTML report, printed only for a sequencing sample (edit 235).

*Code:* `MethylPhys_Interface/report_v4.py` line 486 (6 lines) · arguments: none · calls: not called on the traced runs · constants read: `E`, `_SEQ`

### `report_v4.py::_explain_html`

Draws one section's explanation as a shaded block under its table in the HTML report.

*Code:* `MethylPhys_Interface/report_v4.py` line 492 (3 lines) · arguments: sec · calls: not called on the traced runs · constants read: `E`

### `report_v4.py::_intro_html`

Draws the report's 'How to read this report' section (the law, the ratio to healthy, the person's own reference, the methods' origin in sky mapping, and what each status means) from the same source.

*Code:* `MethylPhys_Interface/report_v4.py` line 495 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `E`

### `report_pdf_v4.py::_explain`

Draws one section's explanation in the researcher PDF, from the same source as the HTML report.

*Code:* `MethylPhys_Interface/report_pdf_v4.py` line 56 (4 lines) · arguments: R, sec · calls: not called on the traced runs

### `report_v4.py::detail_tables._cl`

Adds one row to the distance-to-the-ceiling table for a reading that carries its ceiling record: the reading, its saturation, its distance to the ceiling, its turnover share, and whether its methylated sites have passed 0.5.

*Code:* `MethylPhys_Interface/report_v4.py` line 401 (2 lines) · arguments: name, A, c · calls: not called on the traced runs

### `report_v4.py::_sections_v5`

All the new sections in their order, after the cell shares.

*Code:* `MethylPhys_Interface/report_v4.py` line 499 (8 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`

### `stage_t_selftare.py::ruler_map`

The atlas ruler: the self-tare's straight line per probe design onto IAM-Atlas v2.1's means at sites every cell type holds alike (islands and shores never anchors). Development; off unless run_sample.py is given --dev-ruler.

*Code:* `stage_t_selftare.py` line 32 (8 lines) · arguments: beta · calls: not called on the traced runs · constants read: `RULER`, `_C` · files named: `Runtime Matrices/Stage_T/atlas_ruler_v1.json`

### `stage_metaa_atlas.py::_flag`

The development flag on every atlas reading, naming the ruler used (IAM-Atlas v2.1 or the IAM Sequencing Atlas).

*Code:* `stage_metaa_atlas.py` line 29 (2 lines) · arguments: none · calls: not called on the traced runs · constants read: `FLAG`, `SEQ_ATLAS`

### `stage_metaa_atlas.py::agree_rates`

The library's two molecule-end error rates (a false T on marked molecules, a false C on unmarked ones), measured where every found cell agrees: the atlas sites at which the found cells with an IAM-Atlas v2.1 value hold at least 98 percent of the sample's DNA (by its shares) and every one of them, in sequencing units, is at most 0.1 or at least 0.9, chosen on the atlas and never on the library. On each set the library's end calls minus its inner calls give one equation (end level minus inner level = unmarked rate x (1 - inner level) - marked rate x inner level); the two sets solve both rates exactly. Physics: the error is the chemistry of the library's molecule ends, the same at every CpG, so it is measured where the state of every molecule is known. Added after four synthetics failed most for a minority cell on the marked side: sides set by the read cell's own level at its identity sites hold mostly the other cells' opposite-state molecules. Development.

*Code:* `stage_metaa_atlas.py` line 459 (31 lines) · arguments: cells, counts, sc, lo, hi, shares · calls: not called on the traced runs · constants read: `_AGREE_CACHE`

### `stage_metaa_atlas.py::own_level_added`

The disorder a library's molecule-end error adds at the read cell's identity sites, computed at the library's own level at each site (its inner calls), not at the atlas's. The end error is measured at the same sites as the library's end-call level minus its inner-call level, in ten bins of the mixed atlas inner level. Physics: biology is the same at a molecule's end and its middle, so within a bin the difference is the library's chemistry at exactly those sites and in their actual state mix; and the disorder an error adds is steep near 0 and 1, so it must be computed where the site truly sits. Counts only; nothing fitted, no constant chosen. Added after five forms: rates measured exactly at agreement sites still left the reading moved, and a library's end error differs between site sets. Development.

*Code:* `stage_metaa_atlas.py` line 511 (20 lines) · arguments: sites, counts, sc, e_bin, bins · calls: not called on the traced runs

### `stage_metaa_atlas.py::mix_forward`

The self-calibration's fourth form, the Planck forward model on the whole mixed expectation: every cell found contributes its IAM Sequencing Atlas inner level (its donors without their own molecule-end calls), mixed in the sample's shares as Met-A mixes the atlas; the library's measured end error then flips calls in proportion to each state's share at each site (a false T on the marked share, a false C on the unmarked share), on that site's share of end calls. An end error acts on every molecule in the mix and on each site in proportion to its level, which the first three forms did not model. Development.

### run_sample.py --dev-seq-self-cal-dump

With --dev-seq-self-cal, writes each read cell's per-site inputs to the self-calibration (reads, end reads, end and all marked calls, the measured level, and the IAM Sequencing Atlas's inner and published levels mixed in the sample's shares) to PREFIX_<cell>.csv.gz. For diagnosis: it changes no reading. Added to put a real library's inputs, site by site, into the offline model of the calibration, after the sixth form overcorrected on one library. Development.

### run_sample.py --dev-seq-self-cal-own

With --dev-seq-self-cal, Met-A is the observed disorder minus the disorder the library's end error adds (own_level_added), read against the atlas's inner level mixed in the sample's shares. Every read records A_own_level and the measured bin differences beside the earlier forms. Development.

### run_sample.py --dev-seq-self-cal-agree

With --dev-seq-self-cal, Met-A's expectation is every found cell's inner level mixed in the sample's shares, with the library's two end-error rates from agree_rates flipping calls in proportion to each state's share at each site (mix_forward). Every read records A_agree and the agreement-site counts beside the earlier forms. Development.

### run_sample.py --dev-seq-self-cal-mix

With --dev-seq-self-cal, Met-A's expectation is mix_forward's: every found cell's inner level mixed in the sample's shares, with this library's measured end error applied in proportion to each state's share. Every read records A_mix_forward beside the other forms. Development.

### run_sample.py --dev-seq-self-cal-disorder

With --dev-seq-self-cal, Met-A is the observed disorder at the cell's identity sites minus the disorder this library's measured molecule-end error adds (the counting expectation at the atlas's inner level moved by the library's end error, minus the same expectation without it), read against the atlas's inner level. An error flips whole reads at a few sites, so it is taken out as disorder, in expectation, not as a shift of each site's level. Every read records A_without, A_expectation_side, A_data_side and A_disorder_side. Development.

### run_sample.py --dev-plant-ends

TESTS ONLY (the self-calibration synthetics). On a .pat input, known errors are planted in a copy of the file (stage_seq_values.plant_ends):
END_CT and END_TC at the outermost CpG call at each end of every molecule, ALL_CT at every call. The plant is applied first, before
--dev-seq-trim-ends, so a planted end error sits on the library's real outermost call and a trim of that call removes it (2026-11-12). The
report carries a PLANTED banner.

### run_sample.py --dev-seq-end-cal

DEVELOPMENT (the self-calibration's eighth form; with --dev-seq-self-cal). Before the cell shares, a first Deconvolver pass finds the cells
(share at least 0.01); the library's two molecule-end error rates are measured where those cells agree on the atlas (agree_rates); every
site's level is corrected for them (end_corrected_levels); the shares and every reading then use the corrected levels. The bundle's end_cal
records the first-pass cells, the rates and the mean level change. **Where IAM's Law is:** the instrument, not the law.

*Code:* `stage_metaa_atlas.py` line 532 (19 lines) · arguments: sites, cells, F, counts, sc, unmarked, marked · calls: not called on the traced runs · constants read: `_INNER_CACHE`

### `stage_metaa_atlas.py::end_corrected_levels`

A library's molecule-end error is a bias on a site's level, linear in it: with f the share of the site's calls that are molecule-end calls,
the observed level is m_obs = m (1 - f e_m) + (1 - m) f e_u (e_m the false-T rate on marked molecules, e_u the false-C rate on unmarked
ones), so m = (m_obs - f e_u) / (1 - f (e_u + e_m)) exactly in expectation. Every site's level is corrected so and clipped to 0-1; sites
whose denominator would fall below 0.05 keep their observed level. Counts only; nothing fitted. Development.

### run_sample.py --seq-cell

The cell type IAM-A reads. For --pat and --bam, when not given, the cell the IAM-Deconvolver found most of among those IAM-A holds a reference for (2026-11-07: never the specimen's words). A --site-table has no Deconvolver, so its cell must be named; with none, Stage Q refuses it ('no frozen IAM-A position'). The genome-coverage intake stop now prints and keeps the same five parts as every other stop (2026-11-12, release check V11).

### run_sample.py --dev-seq-self-cal-reach

With --dev-seq-self-cal (the seventh form, development), Met-A is read on the library's calls past each side's reach along the molecule: at sites
the read cell holds marked, past the last call at each end; at sites it holds unmarked, past the last N calls. The healthy level is the IAM
Sequencing Atlas's own level past the same reach (its donors counted with the same trims), every found cell mixed in the sample's shares. Nothing
is modeled or corrected: both sides of the comparison pass through the same cut, because the atlas's donors carry the same molecule-end fault as
the samples. In grid terms: a distance relay's zone reach, set per side from the library's own fault profile, with the reading taken only beyond
it. Every read records A_reach beside the other forms. Development.

### run_sample.py --dev-seq-self-cal-data

With --dev-seq-self-cal, Met-A is read on the library's own calls with its measured molecule-end error taken out (each site moved back by the library's end share there times its measured excess on that site's side), against the IAM Sequencing Atlas's inner level, which holds no end calls. Added after the first synthetic: a ratio of disorders moves unequally when the same error enters a library that departs from 1 and its expectation. Every read records A_without, A_expectation_side and A_data_side. Development.

*Code:* `stage_metaa_atlas.py` line 491 (16 lines) · arguments: beta, counts, sc, rates · calls: not called on the traced runs · files named: `deconv_trim_identity.csv`

### `stage_metaa_atlas.py::self_calibrated`

Each sequencing library calibrated on its own calls, as each CMB detector was calibrated on sources in its own maps. Biology does not depend on where a CpG sits on a molecule, so the difference between a library's molecule-end calls and its inner calls at the same cell's identity sites is its chemistry alone, measured per side (a false T where the cell holds a mark, a false C where it holds none). The healthy expectation is the atlas's inner level moved by that measured end error at each site's own share of end calls. Nothing is cut and nothing is borrowed from other samples. Development (run_sample.py --dev-seq-self-cal).

*Code:* `stage_metaa_atlas.py` line 327 (29 lines) · arguments: cell, level, counts, sc, lo, hi · calls: not called on the traced runs

### `stage_metaa_atlas.py::floor_raise`

The probability that raises a library's floor from f to F: (F - f) / (1 - f).

*Code:* `stage_metaa_atlas.py` line 1007 (3 lines) · arguments: f_low, F · calls: not called on the traced runs

### `stage_metaa_atlas.py::_Hj`

The disorder of j methylated of m reads after each unmethylated read is turned methylated with probability u (a library's floor raised).

*Code:* `stage_metaa_atlas.py` line 998 (8 lines) · arguments: m, u · calls: not called on the traced runs

### `stage_metaa_atlas.py::ceiling_counted`

The distance to the entropy ceiling on sequencing: the reading the sample would give with every site at one half, read at its own reads by the reading's own expectation; distance, headroom used and the share of sites that crossed one half.

*Code:* `stage_metaa_atlas.py` line 871 (12 lines) · arguments: A, A_sat, x, e · calls: not called on the traced runs

### `stage_metaa_atlas.py::_cs_cell_factor`

The C-Score limit factor of one cell, from `Runtime Matrices/CScore_Limits/cell_factors_v1.json`: measured on IAM-Atlas v2.1's own healthy donors (the cell found at 85 % or more), the one-sided 95 % prediction limit of C over the derived limit, for cells whose donors span five or more laboratories (neutrophils 1.0314, 57 donors, 7 laboratories; monocytes 1.0062, 40 donors, 6). Healthy neutrophils carry a little real clustering in every laboratory, so the derived limit, which assumes none, is slightly tight for them. None for every other cell: the derived limit stands.

*Code:* `stage_metaa_atlas.py` line 1233 (6 lines) · arguments: cell · calls: not called on the traced runs · constants read: `HERE`, `IDENTITY_OF` · files named: `cell_factors_v1.json`, `s C-Score limit factor from Runtime Matrices/CScore_Limits/cell_factors_v1.json`

### `stage_metaa_atlas.py::cscore_limits`

Adds each C-Score reading's limit, whether it sits within it, and the rule used. Each cell: the derived limit. The whole genome: the derived limit, times the 450K platform term on a 450K array (measured once on spent healthy 450K bloods and frozen, Runtime Matrices/CScore_Limits/platform_term_450k.json; calibrated).

Each cell's limit is the derived limit times `_cs_cell_factor` where the atlas's donors measure one (limit_rule says which).

*Code:* `stage_metaa_atlas.py` line 1240 (16 lines) · arguments: cs, array_type · calls: not called on the traced runs · constants read: `HERE` · files named: `platform_term_450k.json`

### `stage_metaa_atlas.py::cscore_limit`

The C-Score's derived healthy upper limit (development): with departures independent from site to site, C over nb = n // 10 blocks follows a scaled Beta((nb - 1) / 2, (n - nb) / 2); the limit is its 95th percentile. Nothing is measured.

*Code:* `stage_metaa_atlas.py` line 1227 (4 lines) · arguments: n_sites, q, w · calls: not called on the traced runs · constants read: `CS_BLOCK`

### `stage_metaa_atlas.py::read_on_curve`

Met-A with IAM-Atlas v2.1 as the ruler on the array's own response curve (plan 14.14, development): identity sites chosen on the atlas; the array's mean reading and its spread at each atlas level, measured at every other site it read; its healthy mark is the expected entropy at the identity sites of a reading on that curve with that spread; Met-A is its own mean entropy there over the mark. No cohort and no same-run references; also returns how far the array reads above its own curve at the identity sites.

*Code:* `stage_metaa_atlas.py` line 359 (28 lines) · arguments: beta, cell · calls: not called on the traced runs · constants read: `CURVE_EDGES`, `RULE`

### `stage_metaa_atlas.py::_he`

The healthy expectation of disorder at each site: on sequencing, the expected entropy of k of n reads at that site's own read count (expected_H_counts), so a site read from one molecule, which can only read 0 or 1, is not taken as order; on arrays, the entropy of the expected level.

*Code:* `stage_metaa_atlas.py` line 930 (3 lines) · arguments: es, counts · calls: not called on the traced runs

### `stage_metaa_atlas.py::_ruler`

Loads the atlas ruler's frozen file once: every site's probe design (I or II), the anchor sites per design (held below about 0.05 or above about 0.95 in every one of the 79 cell types) and the atlas's values there.

*Code:* `stage_metaa_atlas.py` line 1091 (4 lines) · arguments: none · calls: not called on the traced runs · constants read: `RULER`

### `stage_metaa_atlas.py::_line_censored`

Development (2026-11-07, `--dev-line-censored`, off by default). An array reports values between 0 and 1, so a site whose true reading
lies below 0 or above 1 is reported at the bound. Taking plain means of the anchor sites then pulls the measured line inward, and the healthy
prediction built on it misses on arrays whose line runs past a bound (Met-A synthetic test 3, the high tail). This measures the same line, x =
a + b e with noise s, from the same anchor sites, counting each reading at 0 or 1 as known only to be at or beyond the bound (a censored-normal
likelihood). When no anchor is at a bound it returns the plain estimate unchanged. It measures the instrument at sites where every healthy
cell agrees; nothing is fitted to any reading of the sample.

*Code:* `stage_metaa_atlas.py` line 1099 (13 lines) · arguments: x, e, a0, b0, s0 · calls: not called on the traced runs

### `stage_metaa_atlas.py::_line_censored.nll`

The negative log-likelihood the line is measured by: anchors strictly between 0 and 1 count by their normal density around a + b e; an anchor at 0 counts by the chance of a reading at or below 0, one at 1 by the chance at or above 1.

*Code:* `stage_metaa_atlas.py` line 1107 (3 lines) · arguments: t · calls: not called on the traced runs

### `stage_metaa_atlas.py::array_blur`

Measures the array's own blur, per probe design, at its anchor sites: the straight line x = a + b e through the means of its low and high anchors against the atlas's means there, and its noise, 1.4826 times the median absolute residual from that line. Nothing is fitted to any reading; these are measurements of the array where every healthy cell agrees (DEV-METAA-HEALTHY-01).


With `--dev-line-wide` (LINE_WIDE, development): the line is measured on every site this sample's own mix holds firmly, below 0.05 or above 0.95, with the identity sites of the cells found set aside (passed as exclude), about 110,000 sites instead of the ruler's 8,000 anchors. Added when the width diagnostic showed the anchors' line moving from person to person enough to move a cell's Met-A by about 0.25 %.

*Code:* `stage_metaa_atlas.py` line 1118 (20 lines) · arguments: xs, es, exclude · calls: not called on the traced runs · constants read: `LINE_CENSORED`, `LINE_WIDE`

### `stage_metaa_atlas.py::predicted_healthy`

Met-A's healthy reference predicted from the array itself: at each site, the disorder this array would show if the sample were exactly the atlas in its own mix, E[H(clip(a + b e + noise, 0, 1))] over the array's measured noise by 20-point Gauss-Hermite quadrature; A_pred = sum w E[H] / sum w H(e) with the whole-sample reading's weights. A healthy sample should read A / A_pred = 1, with no healthy cohort and no reference arrays. Blind spot, stated: a loss spread evenly over every site, the anchors included, looks like a weaker array (DEV-METAA-HEALTHY-01).

*Code:* `stage_metaa_atlas.py` line 1139 (16 lines) · arguments: xs, es, w, blur, min_sites, exclude · calls: not called on the traced runs · constants read: `GH_NODES`

### `stage_metaa_atlas.py::cell_prediction`

Each cell's Met-A divided by its own healthy prediction, as the whole-sample reading already is: the array's own line and noise per probe design, measured once on the whole array, applied to the atlas in this sample's mix at that cell's identity sites (predicted_healthy). The row keeps A_raw and A_pred beside the ratio; a healthy cell reads 1. Added when the width synthetic showed each cell's reading carrying the array's line and noise (center 0.89, spread up to ten times the true person spread). Arrays only; on sequencing the expectation already counts the reads.

*Code:* `stage_metaa_atlas.py` line 1320 (24 lines) · arguments: each, beta, shares, blur · calls: not called on the traced runs · constants read: `MIX_COVER`

### `stage_metaa_atlas.py::width_state`

The state of one cell's Met-A against that cell's healthy width measured on IAM-Atlas v2.1's own donors read by the one command (`Runtime Matrices/Met_A_Widths/donor_widths_v1.json`): Normal within 1 plus or minus k widths, k = t(0.975, n - 1) x sqrt(1 + 1/n) for its n donors (the prediction interval for one new reading), else above or below Normal, with the distance from 1 in widths. 1 is the middle; a cell whose donors read their middle more than 2 widths from 1 carries that note and is not commissioned. Given only at a share the cell held among its donors (share_min); below it the row says the width is not measured there. Arrays only. Its synthetic: the third new-people test, met.

*Code:* `stage_metaa_atlas.py` line 1371 (14 lines) · arguments: cell, a, share · calls: not called on the traced runs · constants read: `IDENTITY_OF`, `WIDTHS`

### `stage_metaa_atlas.py::_sorted_route`

True when the specimen's words choose the sorted-sample route (isolated, sorted or purified cells, of a named cell or none). The words choose only the route; which cell is read comes from the IAM-Deconvolver.

*Code:* `stage_metaa_atlas.py` line 1269 (5 lines) · arguments: specimen · calls: not called on the traced runs

### `stage_metaa_atlas.py::metaa_status`

Each row of cells_found carries met_a_status: 'commissioned (record)' only when the sample took the sorted route, the row is the top cell found, that (cell, platform) is in METAA_ATLAS_COMMISSIONED's scope (COMMISSION-METAA-SORTED-01: neutrophils on Infinium 450K, monocytes on Infinium EPIC v1, against IAM-Atlas v2.1 alone), and a state was measured against the cell's width; otherwise 'development'. The report shows it per row.

*Code:* `stage_metaa_atlas.py` line 1275 (8 lines) · arguments: o, row, is_top · calls: not called on the traced runs · constants read: `METAA_ATLAS_COMMISSIONED`

### `stage_metaa_atlas.py::cscore_status`

Each cells_found row's `c_status`: 'commissioned (COMMISSION-MONO450-01 ...)' when the row is the top cell found in a sorted sample, its C-Score limit is scaled by the cell's factor measured on the atlas's own donors, and (cell, platform) is in CSCORE_ATLAS_COMMISSIONED's scope (sorted monocytes on Infinium 450K); otherwise 'development'. Met-A's `met_a_status` names the record of its own scope (METAA_ATLAS_COMMISSIONED's records: monocytes on 450K by COMMISSION-MONO450-01). The report's per-cell table shows both.

*Code:* `stage_metaa_atlas.py` line 1284 (6 lines) · arguments: o, row, is_top · calls: not called on the traced runs · constants read: `CSCORE_ATLAS_COMMISSIONED`

### `stage_metaa_atlas.py::cells_found`

The sample's result, cell by cell (the author: we score individual cells found by the IAM-Deconvolver, not bundles of cells). One row per cell the Deconvolver found (exactly the cells the chain did not mark not found; no threshold chosen), with each instrument's reading of that cell against its own atlas reference in the sample's own shares: Met-A, the C-Score with its limit, and IAM-A when IAM-A read that cell. The whole-sample reading stays a side line. Each row states its own fraction, the cell's share: the share synthetic measured that a cell's reading in a mix moves with a change in the cell and a change around it in proportion to the DNA each holds at its identity sites, so of a change in that row, about that share would be the cell's own. Above 90 % the row reads the cell itself.

*Code:* `stage_metaa_atlas.py` line 1291 (27 lines) · arguments: o · calls: not called on the traced runs

### `stage_metaa_atlas.py::read_whole`

Met-A for the whole sample over every atlas site: the atlas mixed in the sample's own shares is the expectation, each site counts by how firmly that mix holds it (|0.5 - e|), and the reading is weighted disorder over weighted expected disorder. Reported overall and split by place (identity sites of the cells found, islands, domains, the rest), with the shift a 1 % loss would make. The per-cell readings say which cell; this one says whether the sample has departed. Development, flagged, on by default.


With `--dev-pred-raw` (PRED_RAW; development, 2026-11-07) the healthy prediction of an array, and the A it is compared with, are read on the array's own values even when the whole-sample A is read on the atlas ruler (which `--full-chain` turns on): the healthy prediction measures the array's own line, and measuring it again on ruler-mapped values handles the array's blur twice (DEV-METAA-HEALTHY-01, the capture after synthetic test 6).

*Code:* `stage_metaa_atlas.py` line 1159 (59 lines) · arguments: beta, shares, counts, raw · calls: not called on the traced runs · constants read: `COUNTED_MIX`, `MIX_COVER`, `PRED_RAW`, `SEQ_ATLAS`, `WHOLE_SITE_OUT`

### `stage_metaa_atlas.py::read_whole.one`

The weighted reading over one group of sites (all, or one place), with its standard error over sites; a group with fewer than 500 sites is not read.

*Code:* `stage_metaa_atlas.py` line 1184 (6 lines) · arguments: m · calls: not called on the traced runs

### `stage_metaa_atlas.py::tare_shares`

Met-A's mix for its healthy expectation on a SORTED sequenced sample (edit 236): shares in the main cell's tissue family count as the main cell (the Deconvolver's main cell; a disagreeing sorted label is flagged, never used), other families only at or above SPILL_FLOOR = 0.05, megakaryocytes left out. Whole blood, plasma and tissue keep the Deconvolver's shares. Full rule, numbers, method (GUM), known-answer test, limits and how to rebuild it: the section 'The Deconvolver and Met-A's tare on sorted sequenced samples' above (stage_metaa_atlas.METAA_TARE_TEXT).

*Code:* `stage_metaa_atlas.py` line 1389 (24 lines) · arguments: shares, floor, declared, support · calls: not called on the traced runs · constants read: `ID_ONLY`, `SPILL_FLOOR`

### `stage_metaa_atlas.py::territory_sets`

The two sets where early cancer is known to act, by fixed rules on IAM-Atlas v2.1 over the sites measured in all 79 cell types: islands (every cell type below 0.2) and loosely held domains (mean at least 0.75, none below 0.2, at least one between 0.2 and 0.7). Read column by column so memory stays near one column.

*Code:* `stage_metaa_atlas.py` line 831 (17 lines) · arguments: none · calls: not called on the traced runs · constants read: `ATLAS`, `NA` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::ceiling_reading`

Distance to the entropy ceiling (CAN-005), derived and unfitted: a reading's saturation A_sat = sum w / sum w H(e) (every site at 1 bit), the distance (A_sat - A) / (A_sat - 1) (1 at the healthy expectation, 0 at the ceiling), and the turnover share, the weighted share of sites whose level has crossed 0.5 to the far side of their healthy state, past which the reading falls as loss continues. Attached to Met-A for the whole sample (and each place), each cell's Met-A, and each territory.

*Code:* `stage_metaa_atlas.py` line 849 (21 lines) · arguments: xs, es, w · calls: not called on the traced runs · files named: `CANON/foundation.json`

### `stage_metaa_atlas.py::territory_prediction`

Each territory set's reading on arrays divided by the array's own healthy prediction, as the whole-sample and per-cell readings already are: the array's line and noise per probe design (measured once on the whole array) applied to the atlas in this sample's mix at the set's sites, on the array's own values. Adds A_raw, A_pred and ratio beside A; a healthy sample reads 1. Added when healthy arrays, a sorted neutrophil among them, read the islands at 0.84 without it (1.0005 with it). Arrays only.

*Code:* `stage_metaa_atlas.py` line 1346 (22 lines) · arguments: terr, beta, shares, blur · calls: not called on the traced runs · constants read: `MIX_COVER`

### `stage_metaa_atlas.py::territory_reading`

The sample's disorder at each set against the atlas mixed in its own shares: mean H(beta) over mean H(e), the mean level change, the standard error over sites, and how much of the set the array measures. Development, flagged.

*Code:* `stage_metaa_atlas.py` line 884 (35 lines) · arguments: beta, shares, counts · calls: not called on the traced runs · constants read: `COUNTED_MIX`, `MIX_COVER`, `SEQ_ATLAS`, `TERR_COUNTED`, `TERR_RULES`, `TERR_SITE_OUT`

### `stage_metaa_atlas.py::plant_territory`

Tests only: a constructed gain at a seeded fraction of the islands, or loss at the domains, so a test of the territory readings runs through the one command.

*Code:* `stage_metaa_atlas.py` line 920 (7 lines) · arguments: beta, name, frac, size, seed · calls: not called on the traced runs

### `stage_metaa_atlas.py::absent_cells`

The cells among a sample's shares whose column holds no value at any site of the ruler in use (cached per ruler). On sequencing the provisional megakaryocyte column, built on the whole genome, has none at IAM-Atlas v2.1's sites. Used by `_cover_den`.

*Code:* `stage_metaa_atlas.py` line 432 (11 lines) · arguments: cells · calls: not called on the traced runs · constants read: `SEQ_ATLAS`, `SEQ_LEVEL`, `_ABSENT`

### `stage_metaa_atlas.py::_cover_den`

The share a site's cover is measured against (MIX_COVER, 0.99): the sample's shares less those of cells with no column at any site (`absent_cells`). Gaps inside a column still count against the cover. Added when every sequenced library with megakaryocytes found at 1 % or more lost every site of every reading (27 of 49 trim-reading bundles). `read_whole` names the cells set aside and their share (`absent_from_ruler`); the per-cell row of such a cell says it is not a cell of IAM-Atlas v2.1.

*Code:* `stage_metaa_atlas.py` line 444 (6 lines) · arguments: F · calls: not called on the traced runs

### `stage_metaa_atlas.py::reach_read`

The seventh self-calibration form's reading: the sample's calls past each side's reach (marked side past one call, unmarked side past N) at
each identity site, the atlas's inner levels past the same reach (metaa_inner_reference at one call and at N calls) mixed in the sample's shares,
and each site's remaining call count, for Met-A as observed disorder over the counting expectation. Returns nothing, with the reason, when the
atlas reference at N calls has not been built. Counts only; nothing fitted. Development.

*Code:* `stage_metaa_atlas.py` line 552 (28 lines) · arguments: cells, shares, sites, side_level, x_all, counts, sc, n_unmarked · calls: not called on the traced runs · files named: `_v1.parquet`

### `stage_metaa_atlas.py::read_in_mix_allsites`

Met-A for each cell over every IAM-Atlas v2.1 site (development; `run_sample.py --dev-metaa-allsites`): each site counts by the shift a loss
of that cell would make there, share × |0.5 − μ| (the chain's own shift rule), so nothing is chosen. Records the reading, its shift per 1 %
loss, its standard error over sites and its effective number of sites. The standard error is within one sample; how much healthy people
differ is measured on many healthy samples, not here.

*Code:* `stage_metaa_atlas.py` line 728 (30 lines) · arguments: beta, shares · calls: not called on the traced runs · constants read: `MIX_COVER`

### `stage_metaa_atlas.py::departure_in_mix`

The per-site departure behind Met-A for one cell of a mixed sample: at the cell's identity sites,
H(β) − H(e), e the atlas mixed in the sample's own shares by `read_in_mix`'s rule; by site id. Used by the globe's layer for each cell.

*Code:* `stage_metaa_atlas.py` line 713 (13 lines) · arguments: beta, shares, cell · calls: not called on the traced runs · constants read: `MIX_COVER`

### `stage_metaa_atlas.py::_H`

The entropy of a beta value in bits, clipped to [10⁻⁶, 1 − 10⁻⁶].

*Code:* `stage_metaa_atlas.py` line 75 (2 lines) · arguments: b · calls: not called on the traced runs

### `stage_metaa_atlas.py::_col`

One cell's atlas column (best value, the ends of its 95 % range, or the person-to-person spread) over the atlas's sites, read once.

*Code:* `stage_metaa_atlas.py` line 78 (31 lines) · arguments: stem, cell · calls: not called on the traced runs · constants read: `ATLAS`, `NA`, `SEQ_ATLAS`, `SEQ_LEVEL` · files named: `import pyarrow.parquet`, `meth_part1.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::width`

Each site's healthy width: the person-to-person spread and the atlas's uncertainty (half the 95 % range over 1.96), combined in quadrature.

*Code:* `stage_metaa_atlas.py` line 110 (4 lines) · arguments: cell · calls: not called on the traced runs

### `stage_metaa_atlas.py::identity_sites`

The cell's identity sites by the rule above (up to 3,000 per side): methylated side, then unmethylated side.

*Code:* `stage_metaa_atlas.py` line 118 (16 lines) · arguments: cell · calls: not called on the traced runs · constants read: `IDENTITY_OF`, `RULE`, `SEQ_ATLAS`

### `stage_metaa_atlas.py::expected_H_counts`

**Label: from IAM's physics (counting).** On sequencing, a site's level is k of the n reads that cover it. With few reads the level is 0 or 1
far more often than the true level p, and the disorder of 0 or 1 is zero, so a healthy site's measured disorder at depth n is not H(p) but
E[H(k/n)] = Σₖ C(n,k) pᵏ(1−p)ⁿ⁻ᵏ H(k/n). Met-A's healthy expectation on sequencing values is this sum at each site's own read count, with the
atlas value as p; nothing is fitted. Checked against a 200,000-draw simulation (2026-10-22). Above 400 reads H(p) is used (the bias there is
negligible). It was added after the depth test (DEV-ONE-DRAW-01 exploration 1) found Met-A on sequencing values reading lower as depth fell. With `run_sample.py --dev-seq-copies` (development) and the sample's copy distribution measured (`COPIES`), the expectation is taken under that copy process instead (`stage_seq_copies.expected_H_copies`): a linear-amplification library's calls are copies of fewer molecules.

*Code:* `stage_metaa_atlas.py` line 160 (4 lines) · arguments: p, n · calls: not called on the traced runs · constants read: `COPIES`

### `stage_metaa_atlas.py::_expected_H_binom`

The exact expected entropy of k of n reads at true level p, k binomial (the expectation `expected_H_counts` returns when no copy distribution is set; above N_EXACT reads, H(p)). Derived, nothing fitted.

*Code:* `stage_metaa_atlas.py` line 165 (11 lines) · arguments: p, n · calls: not called on the traced runs · constants read: `N_EXACT`

### `stage_metaa_atlas.py::read`

Met-A against the atlas cell, untared, on the values given (the one command passes the self-tared values), with the shift a 1 % loss would
make; refuses, with the reason, a cell with no qualifying site or a sample measuring under 90 % of them. On sequencing the healthy level is the pooled one (SEQ_LEVEL: methylated over all reads of the cell's donors), not the median of the donors' own levels, which sits at exactly 0 or 1 where a donor has few reads.

Each reading also records `levels_at_identity`: the sample's and the atlas column's median level at the column's methylated and
unmethylated identity sites (recorded only, never used in A), to measure whether the atlas's levels sit nearer the middle than sequencing.

On sequencing (read counts given), the atlas column is first put in sequencing units by the atlas's own measured source term for the
published sequencing atlas (`Runtime Matrices/Atlas_V21/sequencing_term_v1.json`: sequencing = -0.0902 + 1.1026 x atlas level, clipped to
0 to 1): the atlas is on the array scale, whose unmarked sites sit near 0.055 where sequencing reads 0. Arrays are read unchanged.

*Code:* `stage_metaa_atlas.py` line 239 (80 lines) · arguments: beta, cell, specimen, counts, seqref, half · calls: not called on the traced runs · constants read: `FLOOR_EQUAL`, `RULE`, `SAMPLE_FLOOR`, `SELFCAL`, `SEQ_COUNTED`, `SEQ_LEVEL`, `SITE_OUT`

### `stage_metaa_atlas.py::tare`

A_rel = A over the median of at least 3 same-run healthy references (never the sample), the references' spread and the detection limit.

*Code:* `stage_metaa_atlas.py` line 388 (10 lines) · arguments: A, refs · calls: not called on the traced runs · constants read: `RULE`

### `stage_metaa_atlas.py::specific_sites`

DEVELOPMENT (to-do 39.9): Met-A's site rule plus one condition, that the cell's atlas value differs by at least a stated amount from every other
cell of the panel, then the tightest 3,000 a side. The count is recorded and never padded: CD4 and CD8 T cells hold nearly the same sites (at a
difference of 0.2 they keep 66 and 25 of 6,000). `read_in_mix(..., specific=DELTA)` reads on these sites (`run_sample.py --dev-metaa-specific`).

*Code:* `stage_metaa_atlas.py` line 135 (14 lines) · arguments: cell, others, delta · calls: not called on the traced runs · constants read: `RULE`

### `stage_metaa_atlas.py::read_in_mix`

Met-A for each cell of a whole blood (development, flagged in the report as preliminary until commissioned): for each cell the IAM-Deconvolver
read, its identity sites by Met-A's rule on the atlas; the healthy expectation at those sites, IAM-Atlas v2.1's best values mixed in this
person's own cell shares; A = mean entropy of the sample over mean entropy of the expectation; and the shift a 1 % loss of that cell's pattern
would make at this person's share of it, so a cell present in small numbers shows a small shift. Read on the values Stage T step 1 maps, as
the commissioned Met-A is; untared. Known from exploration (GRP-014): a loss at every one of a cell's sites is partly taken up by the
Deconvolver's share of that cell, so the reading sees less loss than was planted.

On a wide panel (`--cell-panel atlas`) the expectation at each site is mixed over the cells that have an atlas value there, renormalized, and a
site counts when those cells hold at least `MIX_COVER` (99 %) of the sample's shares: before this rule (2026-10-22), one small cell without a
value dropped the site, and on the 79-cell panel most cells lost about half their identity sites.

*Code:* `stage_metaa_atlas.py` line 599 (112 lines) · arguments: beta, shares, counts, specific · calls: not called on the traced runs · constants read: `COPY_FLOOR`, `COUNTED_MIX`, `MIX_COVER`, `NOT_FOUND_SHARE`, `READ_COVER_MIN`, `READ_COVER_MIN_SEQ`, `RULE`, `SELFCAL`, `SELFCAL_AGREE`, `SELFCAL_DATA`, `SELFCAL_DUMP`, `SELFCAL_ENTROPY`, `SELFCAL_MIX`, `SELFCAL_OWN`, `SELFCAL_REACH`, `SEQ_ATLAS`

### `stage_metaa_atlas.py::plant_all`

TESTS ONLY: a constructed loss of a cell's whole pattern, every atlas site the array measured where the cell has a value moved the given share of
the way to 0.5 (`run_sample.py --dev-plant CELL:LOSS:all`). A real loss of hold reaches every held site, not only the identity sites.

*Code:* `stage_metaa_atlas.py` line 399 (5 lines) · arguments: beta, cell, loss · calls: not called on the traced runs

### `stage_metaa_atlas.py::plant_runs`

TESTS ONLY: a constructed clustered loss, 10 runs of 30 neighboring identity sites of a cell in genome order (the frozen sky order), chosen by
a seeded draw, each moved the given share of the way to 0.5 (`run_sample.py --dev-plant CELL:LOSS:runs`). The shape the globe's block rule is
built to see.

*Code:* `stage_metaa_atlas.py` line 406 (13 lines) · arguments: beta, cell, loss, seed · calls: not called on the traced runs · constants read: `ATLAS`, `HERE` · files named: `chain/Runtime Matrices/Atlas_V21/sky_order_v21.npz`, `import pyarrow.parquet`, `sites.parquet`, `sky_order_v21.npz`

### `stage_metaa_atlas.py::plant`

A constructed loss for tests: every identity site the array measured moved 1 % of the way to 0.5, the shift `read` predicts. Used by the
series runner's `--plant metaa_atlas` and, scaled by the neutrophil share found, `--plant neutrophil_atlas`; never by a reading.

*Code:* `stage_metaa_atlas.py` line 420 (5 lines) · arguments: beta, cell, loss · calls: not called on the traced runs

## 17. The whole-genome globe (development): `stage_q2_globe.py`

**What it is for.** Drawing a sequenced sample's whole genome, molecule by molecule, so where it holds its marks can be seen: the sample
against its own predicted held error, and against a second reading. Run by the one command with `--pat FILE --dev-genome-globe`
(`--prior-pat` for the second reading). Development: written under `development.genome_globe`.

**Label: from IAM's physics.** The held error is counted exactly as Stage Q2 counts it (Section 9): on molecules that hold their marks, how
often one unmethylated call sits between two methylated ones. The prediction is the sample's own: IAM-A's baseline at its loose share.

**Layout.** hg19's autosomal CpGs in the .pat files' own index order (chr1 to chr22; the X and Y are left off, as the loose share leaves them,
so a difference of sex is never drawn), cut into regions of 1,000 CpGs; region i of n at NESTED pixel floor(i × NPIX / n), NSIDE 256, and every
pixel of its span carries it. Nothing fitted. A region is drawn when it holds at least 200 held-error chances.

**The change.** Each region's departure in counting noise, scaled by the sample's own spread of departures (never below 1: counting is the
floor); changed beyond the family-wise line the report globes use.

### `stage_q2_globe.py::site_owner`

The atlas sites that have a CpG index, with their atlas rows, for counting on the atlas's layout.

*Code:* `stage_q2_globe.py` line 19 (9 lines) · arguments: none · calls: not called on the traced runs · constants read: `ATLAS`, `_OWN` · files named: `import pyarrow.parquet`, `sites.parquet`, `sites_sequencing_key.parquet`

### `stage_q2_globe.py::cscore_regions`

The IAM-A C-Score on the sequencing globe's regions (edit 216). Physics: IAM's Law prices each held mark independently, so losses of held marks should fall like independent coin flips; the C-Score asks whether they do. Same definition as stage_q_iam_a.cscore: C = sum over blocks of (k - eps o)^2 / sum eps (1 - eps) o, eps the sample's pooled held error, here with the globe's regions (1,000 CpGs by default) as the blocks. Independent losses give 1 with null spread sqrt(2 / regions). Each region's term is the globe's own departure, so the globe shows where C comes from. Healthy regions also differ by nature, so the healthy band is measured on the atlas libraries, not assumed. Development; on in the full chain.

*Code:* `stage_q2_globe.py` line 147 (12 lines) · arguments: R, min_opp · calls: not called on the traced runs · constants read: `MIN_OPP`, `REGION`

### `stage_q2_globe.py::_owner_of`

The atlas row of each CpG index, by binary search.

*Code:* `stage_q2_globe.py` line 28 (3 lines) · arguments: cpg, idx, rows · calls: not called on the traced runs

### `stage_q2_globe.py::count_pat_sites`

Per atlas site: held errors, held-error chances, methylated calls and calls, by Stage Q2's rule and line split; with the genome totals.

*Code:* `stage_q2_globe.py` line 31 (21 lines) · arguments: lines · calls: not called on the traced runs

### `stage_q2_globe.py::count_pat_sites.flush`

Adds the buffered counts to the site totals.

*Code:* `stage_q2_globe.py` line 35 (4 lines) · arguments: none · calls: not called on the traced runs

### `stage_q2_globe.py::departures`

Blocks of neighboring counted atlas sites: each block's z of observed against predicted errors, and which pass the family-wise line.

*Code:* `stage_q2_globe.py` line 52 (7 lines) · arguments: err, opp, p · calls: not called on the traced runs · constants read: `BLOCK`, `FLAG`, `ORDER`

### `stage_q2_globe.py::count_pat_regions`

The same counts per region of 1,000 consecutive CpG indexes on the autosomes.

*Code:* `stage_q2_globe.py` line 67 (22 lines) · arguments: lines, region · calls: not called on the traced runs · constants read: `N_CPG`, `REGION`

### `stage_q2_globe.py::count_pat_regions.flush`

Adds the buffered counts to the region totals.

*Code:* `stage_q2_globe.py` line 71 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `N_CPG`

### `stage_q2_globe.py::_z_self`

Each region's held errors against the number its chances predict at the sample's own predicted held error, in counting noise.

*Code:* `stage_q2_globe.py` line 90 (2 lines) · arguments: R, p · calls: not called on the traced runs · constants read: `MIN_OPP`

### `stage_q2_globe.py::_z_pair`

Each region's held error against the other reading's, in both regions' counting noise; with which regions both readings drew.

*Code:* `stage_q2_globe.py` line 92 (4 lines) · arguments: A, B · calls: not called on the traced runs · constants read: `MIN_OPP`

### `stage_q2_globe.py::_own_spread`

The sample's (or pair's) own spread of region departures: MAD × 1.4826 over the drawn regions, never below 1.

*Code:* `stage_q2_globe.py` line 96 (5 lines) · arguments: z, ok · calls: not called on the traced runs

### `stage_q2_globe.py::_page_layers`

The globe's layers: methylation level, held error, and the change (departures over the own spread, beyond the line), with their keys.

*Code:* `stage_q2_globe.py` line 101 (26 lines) · arguments: z, ok, err_rate, meth, p, change_name, about_ref, scale · calls: not called on the traced runs · constants read: `HERE`, `REGION` · files named: `Planck_Parchment_RGB.txt`

### `stage_q2_globe.py::_page_layers.rgb`

Turns a coded map into the page's colors.

*Code:* `stage_q2_globe.py` line 111 (2 lines) · arguments: m · calls: not called on the traced runs

### `stage_q2_globe.py::self_globe`

The sample against itself (its own predicted held error), with the count of changed regions.

*Code:* `stage_q2_globe.py` line 131 (7 lines) · arguments: R, loose_share, gid, title · calls: not called on the traced runs · constants read: `ABOUT`, `REGION`

### `stage_q2_globe.py::pair_globe`

The sample against another reading, with the count of changed regions.

*Code:* `stage_q2_globe.py` line 138 (7 lines) · arguments: A, B, gid, title, name_b · calls: not called on the traced runs · constants read: `ABOUT`, `REGION`

## 18. The information sphere (development): `stage_info_sphere.py`

**What it is for.** Placing sites by how they behave across cell types instead of by their address, so how far apart states are can be seen.
Built by the one command: `run_sample.py --build info-sphere` (Section 1, `_build`). Development (DEV-SPHERE-01).

**Label: established mathematics.** Each site's levels b in S cell types become u = (√b₁…√b_S, √(1−b₁)…√(1−b_S))/√S, a point on a unit sphere;
the distance between two sites is the Fisher-Rao (Bhattacharyya) angle arccos(u_i · u_j). The globe keeps the three leading principal
directions of those points, scaled to length 1 (construction v1, the one that kept distances); nothing is tuned. The sky the sites do not
cover is masked, as the microwave sky masks the galaxy.

**The physics, step by step.** The information sphere is established mathematics; IAM's Law is not yet in it (register SP1).

1. **The geometry (calculated).** Each site's levels in S cell types become u = (√b₁…√b_S, √(1 − b₁)…√(1 − b_S))/√S on a unit sphere; the
   Fisher–Rao (Bhattacharyya) angle arccos(u_i · u_j) is the distance between sites; construction v5 removes each site's own level first.
2. **The sample's reading (statistics).** Each territory's mean change in the sample's own spread of territories, against the line chance
   crosses less than once in 20 samples.
3. **Where the law could enter (open, SP1).** Through the healthy expectation at each site: the held level from M (A4, M2), the islands' meter
   (T1), the clustering statistic (C1) and the cell shares (D1). Construction test 7 passed; sample test 1 met on distance, not on territories
   (0 of 27); the null test ATL-028 is pending.

### `stage_info_sphere.py::counting_floor`

The expected squared Fisher-Rao angle at each site of a healthy sample read n times at its true level, summed exactly over every
read outcome (binomial); the large-n value 1/(4n) above 200 reads.

*Code:* `stage_info_sphere.py` line 169 (10 lines) · arguments: n, e, cap · calls: not called on the traced runs

### `stage_info_sphere.py::info_vectors`

The square-root vectors of a sites × cell types matrix.

*Code:* `stage_info_sphere.py` line 18 (3 lines) · arguments: B · calls: not called on the traced runs

### `stage_info_sphere.py::fr_angle`

The Fisher-Rao angle between pairs of sites, in the full space.

*Code:* `stage_info_sphere.py` line 22 (2 lines) · arguments: U, i, j · calls: not called on the traced runs

### `stage_info_sphere.py::globe`

Construction v1: centered, projected on the three leading directions, scaled to 1; with the share of the spread the three directions keep.

*Code:* `stage_info_sphere.py` line 25 (4 lines) · arguments: U · calls: not called on the traced runs

### `stage_info_sphere.py::_spread`

Interleaves the bits of a pixel coordinate, for the NESTED pixel number.

*Code:* `stage_info_sphere.py` line 40 (4 lines) · arguments: v · calls: not called on the traced runs

### `stage_info_sphere.py::level_free`

Construction v5 (development): each site's square-root vector minus that of a flat profile at the site's own mean level, scaled to length 1, so
the globe places sites by how they depart from their own level across the cell types, not by the level itself (DEV-SPHERE-01 test 8 found the
level dominates on real cells). Sites with no departure have no direction and are counted.

*Code:* `stage_info_sphere.py` line 30 (8 lines) · arguments: B, U · calls: not called on the traced runs

### `stage_info_sphere.py::ang2pix_nest`

The standard HEALPix NESTED angle to pixel; checked by a round trip through `stage_v4.nest_pix2ang` on all 196,608 pixels at NSIDE 128.

*Code:* `stage_info_sphere.py` line 44 (17 lines) · arguments: nside, theta, phi · calls: not called on the traced runs

### `stage_info_sphere.py::to_pixels`

The pixel of each globe point.

*Code:* `stage_info_sphere.py` line 62 (2 lines) · arguments: X, nside · calls: not called on the traced runs · constants read: `NSIDE`

### `stage_info_sphere.py::concentration`

How tightly a group of sites lands together: the length of the mean of its globe points (1 = one point, near 0 = spread over the sphere).
**Label: standard.** This is the mean resultant length of directional statistics, the field's usual measure of how concentrated points on a
sphere are; it is not ours. What the information sphere adds is where the points come from (the square-root map of each site's methylation
across cell types) and the tools that read the map.

*Code:* `stage_info_sphere.py` line 65 (3 lines) · arguments: X, idx · calls: not called on the traced runs

### `stage_info_sphere.py::edge_ratio`

The mean jump of the mean-level map across HEALPix base-face edges over the jump within a face (about 1: no grid edges).

*Code:* `stage_info_sphere.py` line 69 (14 lines) · arguments: mb, nside · calls: not called on the traced runs · constants read: `NSIDE`

### `run_sample.py::_build._R`

With --build info-globe: the concentration R of a set of sites on the information sphere (the length of their mean direction, 0 spread evenly, 1 all in one place), reported for each family and each direction in the build record; at least 20 sites, else none.

*Code:* `MethylPhys_Interface/run_sample.py` line 431 (1 lines) · arguments: m · calls: not called on the traced runs

### `stage_info_sphere.py::save_layout`

Freezes the information sphere's layout (every site's id, direction and role, and the atlas file's fingerprint) in Runtime Matrices/Info_Sphere,
written by `--build info-globe`, so every sample is placed on the same sphere.

*Code:* `stage_info_sphere.py` line 154 (6 lines) · arguments: path, ids, X, cell, sgn, size, margin, cells, cell_family, atlas_sha · calls: not called on the traced runs

### `stage_info_sphere.py::load_layout`

Reads the frozen layout back.

*Code:* `stage_info_sphere.py` line 161 (2 lines) · arguments: path · calls: not called on the traced runs · constants read: `LAYOUT`

### `stage_info_sphere.py::groups_of`

The territories a sample is read on: each family's clearly marking sites (margin at least 0.10), split by direction (methylated there, or
unmethylated there), the same groups the picture colors.

*Code:* `stage_info_sphere.py` line 164 (4 lines) · arguments: lay · calls: not called on the traced runs · constants read: `CLEAR`

### `stage_info_sphere.py::sample_reading`

One sample on the information sphere. It takes the change the address globe already read at each site (the site's difference from the
sample's own expected healthy self, in that site's healthy width and the sample's own spread), so both globes read the same change. Each
territory's mean change is put in standard errors, then in the sample's own spread of territories (median and spread over the territories
read, as the address globe reads its blocks in its own block spread). A territory has changed beyond the line chance crosses less than once
in 20 samples (5 % family-wise over the territories read). Also the Fisher-Rao angle between the sample and its expectation at each site, and
its root mean square: the sample's distance from its own expected healthy self. Nothing fitted.

*Code:* `stage_info_sphere.py` line 180 (36 lines) · arguments: x_by_id, lay, families, b_by_id, e_by_id, floor_by_id · calls: not called on the traced runs · constants read: `MIN_GROUP`

### `stage_info_sphere.py::roles`

What groups a site with its neighbors on the information sphere. With each site's own level taken out (construction v5), a site's direction
is set by how it departs from that level across the cell types, so the cell where it departs most is the identity it carries: returned with
the sign (methylated there, or unmethylated), the size of that departure, and the margin over the next cell. The 15 families that color the
picture are a *stated* grouping of the atlas's 79 cell types by tissue and lineage (FAMILIES); nothing is computed from them. Measured, not
assumed: a cell's methylated and unmethylated sites gather on opposite sides of the sphere (T cells: two groups, each tight, R 0.88 and 0.86).

*Code:* `stage_info_sphere.py` line 137 (10 lines) · arguments: B, cells · calls: not called on the traced runs · constants read: `FAMILIES`

### `info_sphere_viewer.py::_b64`

Packs an array into text so the page carries its own data and needs no other file.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 49 (1 lines) · arguments: a · calls: not called on the traced runs

### `info_sphere_viewer.py::territories`

Each family's density of clearly marking sites (margin at least CLEAR = 0.10, *stated*) in every direction of a 128 x 256 grid over the sphere,
with a von Mises-Fisher kernel of 6 degrees (*stated*): the spherical form of a smoothing window.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 16 (9 lines) · arguments: X, fam, nfam, H, W, sigma_deg, chunk · calls: not called on the traced runs

### `info_sphere_viewer.py::territory_texture`

Colors each direction by the family most over-represented there: its share of the local density divided by its share of all clearly marking
sites (the enrichment), so a large family does not color the sphere by size alone. The glow grows with the enrichment and the density; it
dims where two families are nearly equal, which draws the dark lanes between territories; empty sky stays dark.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 26 (9 lines) · arguments: T, counts, colors · calls: not called on the traced runs

### `info_sphere_viewer.py::territory_peaks`

Where to write each family's name: up to two directions where it is most enriched and leads, at least 55 degrees apart (*stated*).

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 36 (12 lines) · arguments: E, top, min_sep_deg, min_enrich, per_family · calls: not called on the traced runs

### `info_sphere_viewer.py::write`

Writes the information sphere as one page with everything inside it: every site's direction, cell, sign, size and margin; the territory
picture; the labels; and the page's own drawing code. The drawing turns the sphere, shows the territories, the sites as points, a wire
graticule for depth, and reads which family's territory is under the pointer. Its page (2026-10-23, checked control by control): redraws only while turning or after a control changes; a family shown alone or one cell type chosen turns the sphere to face it and stops the turn; the readout states how strongly a family's sites gather, how many sites a chosen cell has at the slider's setting, and, on a sample with no changed territory, why nothing remains at full fade.

*Code:* `MethylPhys_Interface/info_sphere_viewer.py` line 51 (34 lines) · arguments: path, X, cell, sgn, size, margin, cells, cell_family, families, title, about, sample · calls: not called on the traced runs · constants read: `CLEAR`, `PAGE`

### `stage_info_sphere.py::build`

The layout and its report: variance kept, sky covered and masked, distance agreement on 5,000 seeded pairs, edge ratio, and for each named
group its concentration against the 99th percentile of 500 random groups of the same size.

*Code:* `stage_info_sphere.py` line 84 (27 lines) · arguments: B, groups, n_random, seed, nside, construction · calls: not called on the traced runs · constants read: `NSIDE`

## 18b. IAM's Law Sphere (development): `stage_law_sphere.py`

**What it is for.** Showing a sequenced sample against IAM's Law itself, unit by unit and job by job, with one layer per result the
law's inputs allow. The information sphere places units by how they behave across the healthy cell types; the Law Sphere reads each unit
from the sample's own molecules and compares it with the state the law predicts for the sample's own cells. Development (DEV-SPHERE-01).

**Label: established mathematics on IAM's derived values.** The angle between two Bernoulli states b and e is the Fisher-Rao angle
arccos(sqrt(b e) + sqrt((1 - b)(1 - e))); fidelity is its cos^2. The law's expected level at a unit is the share-weighted sum over the
sample's cells of held CpGs at 1 - p (p: IAM-A's own predicted held error from the sample's loose share) and unmarked CpGs at the
writer's stray rate 1/(1 + D), less the eraser's clean-up, plus the meter. Each angle is judged against its counting floor (edit 198).
The hold layers use IAM-A's own counter (isolated errors over opportunities on held molecules, the book's definition of the held error),
so they compare like with like with the law's 0.0445 (A4). The implied coins per restore invert the slow-write rule, the identity with
A4's driven passage. The Koide balance assumes the conjectured identification of held and spent parts.

**The input card.** Every law input is named, with its default from the cell simulator's law card: temperature, the coin's
size, coins per restore, a set push (suddenness), the source of the held error (IAM-A's line or the law), IAM-A's line, the
writer's selection D, the eraser's share and clean-up, conversion, and per-type and per-job settings. `--law-set`, `--law-card`;
`--build law-sphere-sweep` and `law-sphere-fit` read saved counts again (`--law-counts`, `--law-vary`, `--law-vary-cell`).

**The 16 layers.** Angle to the law; that angle over its floor; fidelity to the law; the power-factor reading (cos of the angle, an analogy); the angle between the law's and the atlas's expected points; angle to the healthy atlas; hold against IAM-A's
line; hold against the law; energy held per mark (kT); share of each coin kept as hold (phi); implied coins per restore; Koide balance;
stray marks against the law; loose share; agreement of the file's two halves over their floor (the instrument's own check); depth.

### `stage_law_sphere.py::card_of`

The Law Sphere's input card: the defaults (stage_law_sphere.CARD, from the cell simulator's law card) with overrides from a dict or
KEY=VALUE strings; unknown keys are refused.

*Code:* `stage_law_sphere.py` line 40 (15 lines) · arguments: over · calls: not called on the traced runs · constants read: `CARD`

### `stage_law_sphere.py::odds_of_push`

Held over lost for a write of x kicks, (x/2)/(1 - (1 - e^-x)/x): the slow-write rule, identical to A4's driven passage.

*Code:* `stage_law_sphere.py` line 55 (3 lines) · arguments: x · calls: not called on the traced runs

### `stage_law_sphere.py::eps_of`

The held error a card predicts for one cell type: IAM-A's line, or the law from the coin, temperature and coins per restore (or a
set push), the eraser's share, and a per-type multiplier.

*Code:* `stage_law_sphere.py` line 64 (11 lines) · arguments: card, cell, loose_share · calls: not called on the traced runs · constants read: `R_GAS`

### `stage_law_sphere.py::loss_of_eps`

The per-call loss whose isolated losses equal the counter's held error (independent losses assumed).

*Code:* `stage_law_sphere.py` line 58 (6 lines) · arguments: eps · calls: not called on the traced runs

### `stage_law_sphere.py::save_counts`

Saves a sample's per-unit molecule counts with its shares and loose share, so any card can be read again without the file.

*Code:* `stage_law_sphere.py` line 238 (6 lines) · arguments: path, C, ref, shares, loose_share, conv, sample · calls: not called on the traced runs · constants read: `COUNT_KEYS`

### `stage_law_sphere.py::load_counts`

Reads saved counts back onto the sphere's units.

*Code:* `stage_law_sphere.py` line 244 (6 lines) · arguments: path, ref · calls: not called on the traced runs · constants read: `COUNT_KEYS`

### `stage_law_sphere.py::sweep`

One card input over a range (or one type's), every summary at each value.

*Code:* `stage_law_sphere.py` line 250 (9 lines) · arguments: C, ref, shares, loose_share, key, values, base, conv, cell · calls: not called on the traced runs

### `stage_law_sphere.py::fit`

The value of one card input (whole or one type's) that brings the sample nearest its counting floor, whole and per job, with the curve.

*Code:* `stage_law_sphere.py` line 259 (13 lines) · arguments: C, ref, shares, loose_share, key, lo, hi, n, base, conv, cell, target · calls: not called on the traced runs

### `stage_law_sphere.py::fit.best`

The minimum of one summary over the fit's grid, and whether it lies at the grid's edge.

*Code:* `stage_law_sphere.py` line 264 (4 lines) · arguments: get · calls: not called on the traced runs

### `stage_law_sphere.py::departure`

Each unit's departure from its healthy value for one output: the law's value where it gives one; the angles from what counting
alone gives; loose share and depth from the sample's own median.

*Code:* `stage_law_sphere.py` line 282 (11 lines) · arguments: key, Lay · calls: not called on the traced runs · constants read: `LAW_VALUE`

### `stage_law_sphere.py::texture`

The mean of the units' values in each cell of an equirectangular grid on the sphere (the information sphere's direction convention).

*Code:* `stage_law_sphere.py` line 293 (8 lines) · arguments: X, v, H, W · calls: not called on the traced runs

### `stage_law_sphere.py::spec`

The Law Sphere page's globe-viewer spec: one group per family of outputs, one coded-difference layer per output with its scale,
its share above each step and its note.

*Code:* `stage_law_sphere.py` line 301 (26 lines) · arguments: rec, Lay, positions, sample · calls: not called on the traced runs · constants read: `GROUPS`, `LAYERS`

### `stage_law_sphere.py::unit_map`

CpG index to sphere unit: each island one unit, then blocks of 50 consecutive CpGs outside islands.

*Code:* `stage_law_sphere.py` line 100 (7 lines) · arguments: isl_first, isl_last, block, n · calls: not called on the traced runs

### `stage_law_sphere.py::load_ref`

Reads the runtime reference: island bounds, block size, and the per-unit table of jobs, levels and held shares.

*Code:* `stage_law_sphere.py` line 108 (7 lines) · arguments: ref_dir · calls: not called on the traced runs · constants read: `REF_DIR` · files named: `units.npz`

### `stage_law_sphere.py::count_pat_units`

IAM-A's molecule rule per unit: levels from even lines; from odd lines held, unmarked and loose molecules, isolated errors
and gains on interior calls, with their opportunities.

*Code:* `stage_law_sphere.py` line 116 (34 lines) · arguments: lines, umap · calls: not called on the traced runs

### `stage_law_sphere.py::_theta`

The Fisher-Rao angle between two Bernoulli states.

*Code:* `stage_law_sphere.py` line 151 (2 lines) · arguments: b, e · calls: not called on the traced runs

### `stage_law_sphere.py::push_for_odds`

Inverts the slow-write rule held/lost = (x/2)/(1 - (1 - e^-x)/x) for the push x in kicks.

*Code:* `stage_law_sphere.py` line 156 (3 lines) · arguments: odds · calls: not called on the traced runs · constants read: `_OG`, `_XG`

### `stage_law_sphere.py::reading`

One sample on the Law Sphere: the law's expected point in the sample's own shares, the 14 layers per unit, and per job and
whole-sample summaries, each against its counting floor.

*Code:* `stage_law_sphere.py` line 160 (75 lines) · arguments: C, ref, shares, loose_share, conv, card · calls: not called on the traced runs · constants read: `CARD`, `CONV_DEFAULT`, `LAYERS`, `MIN_CALLS`, `MIN_OPP`, `R_GAS`

### `stage_law_sphere.py::type_positions`

Reads type_positions_v1.json: each healthy cell type's measured position on IAM-A's derived line (A8-POSITION), with its relative uncertainty and library count. Measured, not derived: a substitute for the underived A8, like a scheduled value for a missing meter. The between-lab term is not in it (edit 212).

*Code:* `stage_law_sphere.py` line 329 (3 lines) · arguments: none · calls: not called on the traced runs · constants read: `REF_DIR` · files named: `type_positions_v1.json`

### `stage_law_sphere.py::hold_alarm`

The Law Sphere's hold alarm (edit 212). Physics: the held record is the potential side of IAM's Law in the cell (the slow close, A4); the level layers see the hold only as 1 - q, so they cannot confirm a reference for it. The alarm counts the sample's held error from the same molecules (IAM-A's rule) and compares it with three references: the law (two coins, 0.04454), IAM-A's own derived line at the sample's loose share, and that line times the type's measured position. z = log(held / reference) / sqrt(se^2 + u^2), with se from a block jackknife over the sphere's units (the measured spread between units and molecules, about four times plain counting noise) and u the type's band uncertainty (GUM). |z| <= 2 reads within. Mixtures are not read: the line is defined for one cell type. Development; on in the full chain with the Law Sphere.

*Code:* `stage_law_sphere.py` line 332 (20 lines) · arguments: C, ref, cell, predicted_derived, blocks · calls: not called on the traced runs · constants read: `EPS_LAW`

### `stage_law_sphere.py::pileup_mask`

A unit read deeper than PILEUP_X (10) times the sample's median calls per unit is a depth pile-up, usually a repeat where reads from many places stack, and is left out of every pooled hold reading and the hold alarm (edit 218). A chain rule, not physics. Before this the whole-sample and per-job hold pooled only units with at least 200 held chances, which at atlas depth were mostly pile-ups (held error 0.079 on the known-answer library), so a healthy library read 1.73 times its own line. With pile-ups out the same library reads 0.04253, against IAM-A's 0.042375 over the whole genome; 5 and 20 times the median give the same to 0.0001. The whole-sample and per-job records also carry the halves (the angle between the two strands' levels over its counting floor).

*Code:* `stage_law_sphere.py` line 77 (4 lines) · arguments: calls · calls: not called on the traced runs · constants read: `PILEUP_X`

### `stage_law_sphere.py::unused_islands`

Stray writing at the islands no healthy cell type writes (edit 215). Physics: the write side of IAM's Law; a mark there is a stray from the writer (selection 1/(1 + D), D = 60, after the eraser's 18 % clean-up) plus the meter (conversion failure). The set (unused_islands_v1.json) is measured: 8,567 IAM Sequencing Atlas islands below 5 % methylated in all 58 types at >= 50 calls, 516 of them covered by no array probe. Reports the sample's pooled stray rate against the law and against the atlas's own healthy level in the sample's cell shares, z from counting only (a library floor is not yet in it). Development.

*Code:* `stage_law_sphere.py` line 354 (19 lines) · arguments: C, shares, conv · calls: not called on the traced runs · constants read: `CONV_DEFAULT`, `D_SELECT`, `ERASER_CLEAN`, `REF_DIR` · files named: `unused_islands_v1.json`

### `stage_law_sphere.py::count_pat_units.flush`

Adds the buffered (CpG, weight) pairs into the per-unit sums and empties the buffers.

*Code:* `stage_law_sphere.py` line 121 (5 lines) · arguments: none · calls: not called on the traced runs

### `stage_law_sphere.py::reading.summ`

The summary of every layer over a set of units (the whole sample, or one job), pooled before the ratio is taken.

*Code:* `stage_law_sphere.py` line 201 (24 lines) · arguments: m · calls: not called on the traced runs

## 19. The reference globes and the series runner: `reference_globes.py`, `run_series.py`

**What they are for.** `reference_globes.py` draws the site's reference globes, so every globe on the site comes from the chain: Met-A's
purified-array reference, the atlas's noise, the 79 atlas cells' textures and their slider pictures (`run_sample.py --build explorer-purified`,
`explorer-noise`, `atlas-sphere`, `atlas-masks`). `run_series.py` runs the one command over a series of arrays on GitHub (`series.yml`):
pass 1 reads each array without references; pass 2 reads it again with run_sample.py's own tare against the other healthy arrays of the series
or slide. It reads, tares and scores nothing itself; a scorer reads only the bundles.

### `reference_globes.py::wg_layers`

The Globe Viewer's layers of the Sequencing Atlas Globe beyond the level and range pictures, each the picture of a tool already measured at every CpG; nothing is read or scored. Per cell: the slider codes for the level and the range; noise map 1 (donor halves) and noise map 2 (read halves) from the noise tools' maps, on one fixed scale (+/-0.05, NZ_LIM) so cells and maps compare; the null, map 2's own values shuffled among the pixels the tools use, one of the shuffles the null test compares against. One mask for the globe: the sex chromosomes the tools leave out, the pixels no cell measured, the pixels used. Run by `run_sample.py --build seq-wg-layers`.

*Code:* `MethylPhys_Interface/reference_globes.py` line 224 (55 lines) · arguments: out, tex_files, noise_files, seed · calls: not called on the traced runs · constants read: `GREY`, `HW_MAX`, `NZ_LIM` · files named: `Writes layers.json`, `globe_wg_noise_maps_n256.npz`, `layers.json`

### `reference_globes.py::wg_layers.tex`

Draws a noise map as a picture on the fixed scale, blue below zero and red above; pixels with no value are gray.

*Code:* `MethylPhys_Interface/reference_globes.py` line 243 (3 lines) · arguments: m, path · calls: not called on the traced runs · constants read: `GREY`, `NZ_LIM`

### `reference_globes.py::wg_layers.code`

Writes the one-byte code the viewer's slider reads: 0 for no measured CpG, else 1 plus the value over its top times 254, every other row and column, the rule atlas_masks uses for the other atlas globes.

*Code:* `MethylPhys_Interface/reference_globes.py` line 246 (3 lines) · arguments: m, hi, path · calls: not called on the traced runs

### `reference_globes.py::wg_sphere`

The Globe Viewer's pictures of the Sequencing Atlas Globe at every CpG, drawn from the published browser-sized textures (NSIDE 256, every pixel the average of the CpGs it holds, about 36 of the 28,217,448) by the same drawing rule as the other atlas globes: the level 0 to 1 and the half-width of the 95 % range 0 to 0.15. A picture of the atlas; nothing is read or scored from it. Run by `run_sample.py --build seq-wg-sphere`.

*Code:* `MethylPhys_Interface/reference_globes.py` line 183 (35 lines) · arguments: out, tex_files · calls: not called on the traced runs · constants read: `GREY`, `HERE`, `HW_MAX` · files named: `Planck_Parchment_RGB.txt`, `globe_wg_textures_n256.npz`, `index.json`

### `reference_globes.py::wg_sphere.tex`

Draws one map as a picture: each picture pixel takes the value of the nearest sphere pixel, colored on the given scale; pixels with no measured CpG are gray. The same rule atlas_sphere uses for the other atlas globes.

*Code:* `MethylPhys_Interface/reference_globes.py` line 199 (3 lines) · arguments: m, cmap, lo, hi, path · calls: not called on the traced runs · constants read: `GREY`

### `reference_globes.py::slug`

A cell name as a file name.

*Code:* `MethylPhys_Interface/reference_globes.py` line 16 (1 lines) · arguments: c · calls: not called on the traced runs

### `reference_globes.py::purified`

Met-A's purified-array group profiles at its 6,000 neutrophil identity sites, drawn as dots on the atlas's layout.

*Code:* `MethylPhys_Interface/reference_globes.py` line 18 (17 lines) · arguments: none · calls: not called on the traced runs · constants read: `CH`, `GROUP`, `HERE` · files named: `Planck_Parchment_RGB.txt`, `blood_composition_EPIC_v1.json`, `import pyarrow.parquet`, `sites.parquet`

### `reference_globes.py::noise`

The atlas's noise maps: per blood cell, the noise left in an average of k posterior draws at every site, on one log scale.

*Code:* `MethylPhys_Interface/reference_globes.py` line 36 (33 lines) · arguments: d · calls: not called on the traced runs · files named: `noise.json`, `uncertainty.json`

### `reference_globes.py::manifests`

Finds Illumina's 450K and EPIC manifests in methylprep's download folder (it differs between installs).

*Code:* `MethylPhys_Interface/reference_globes.py` line 70 (33 lines) · arguments: none · calls: not called on the traced runs

### `reference_globes.py::atlas_sphere`

The 79 cells' globe textures (best value and how sure), laid out by the frozen sky order; each cell's measured pixels must equal its cell
page's count, or the build stops.

*Code:* `MethylPhys_Interface/reference_globes.py` line 104 (34 lines) · arguments: out, atlas, man, cells_index, name, pages · calls: not called on the traced runs · constants read: `GREY`, `HERE`, `HW_MAX`, `REPO` · files named: `Planck_Parchment_RGB.txt`, `import pyarrow.parquet`, `index.json`, `s measured-pixel count equal to website/art/out/cells/index.json`, `sites.parquet`, `sphere.json`

### `reference_globes.py::atlas_sphere.tex`

One texture from a sky map.

*Code:* `MethylPhys_Interface/reference_globes.py` line 117 (3 lines) · arguments: m, cmap, lo, hi, path · calls: not called on the traced runs · constants read: `GREY`

### `reference_globes.py::atlas_masks`

The value pictures for the atlas globe's sliders (best value, and half-width of the 95 % range), losslessly coded; checked against the textures.

*Code:* `MethylPhys_Interface/reference_globes.py` line 163 (18 lines) · arguments: OUT, A · calls: not called on the traced runs · constants read: `ATLAS`, `HW_MAX` · files named: `import pyarrow.parquet`, `sphere.json`

### `reference_globes.py::atlas_masks.code`

Codes a map 1 to 255 on its scale, 0 where no site.

*Code:* `MethylPhys_Interface/reference_globes.py` line 169 (2 lines) · arguments: v, hi · calls: not called on the traced runs

### `run_series.py::matrix_values`

Reads a series published only as a GEO value table (no IDATs): each array's column becomes a values file (cpg, beta) read by run_sample.py with --betas, the replays' values route; nothing in the values is changed, and the chip-level intake checks, which need IDATs, do not run. Synthetic check: a matrix holding the example healthy blood gives the same Met-A (0.9845), C-Score (0.9686) and cell shares as reading the same values directly.

*Code:* `MethylPhys_Interface/run_series.py` line 51 (26 lines) · arguments: src, gsms, d · calls: not called on the traced runs · files named: `_matrix_values.csv`

### `run_series.py::fetch`

An array's two IDATs from GEO, retried.

*Code:* `MethylPhys_Interface/run_series.py` line 24 (26 lines) · arguments: gsm, d, row · calls: not called on the traced runs

### `run_series.py::run`

Runs the one command with the given flags and keeps its log.

*Code:* `MethylPhys_Interface/run_series.py` line 78 (3 lines) · arguments: args, log · calls: not called on the traced runs · constants read: `PY`, `RS`

### `run_series.py::refs_for`

The pass-1 readings of the other arrays of the scope (series, slide, or slide when it holds at least 3 others, else series), as the one
command's reference inputs; never the array itself.

*Code:* `MethylPhys_Interface/run_series.py` line 82 (8 lines) · arguments: gsm, slide, P1, scope, out · calls: not called on the traced runs

### `run_series.py::bundle_row`

One bundle's numbers for the table: Met-A, its tare, the C-Score, atlas Met-A and its tare, the intake verdict, the reason when not read.

*Code:* `MethylPhys_Interface/run_series.py` line 91 (6 lines) · arguments: p · calls: not called on the traced runs

### `run_series.py::read`

One shard of a pass, as described above; with `--plant`, the constructed loss is planted in the array's own saved values and read by the one
command with the same references (Group-A's design: the planted and unplanted halves both by the values route). A per-array atlas cell (samples column atlas_cell) replaces --dev-metaa-atlas CELL for that array, so one series run reads each sorted subtype against its own IAM-Atlas v2.1 column.

`--plant neutrophil_atlas` (pass 1, development; Group-A against the atlas alone): with a `half` column, both halves are read again from their own saved values, so they take the same route; the planted half carries a constructed 1 % loss of the neutrophils' pattern scaled by the neutrophil share the IAM-Deconvolver found in that array (`stage_metaa_atlas.plant`). No references are used.

`--keep-values DIR` (pass 1): each array's calibrated values (its `--save-betas` file) are also copied to DIR as GSM_betas.parquet before the work folder is removed; used to build healthy bloods from the atlas's own donors.

*Code:* `MethylPhys_Interface/run_series.py` line 98 (83 lines) · arguments: a, extra · calls: not called on the traced runs · constants read: `HERE` · files named: `_betas.parquet`, `_planted.csv`, `_planted.json`, `_values.csv`, `_values.json`

### `run_series.py::table`

One row per bundle, for the scorer.

*Code:* `MethylPhys_Interface/run_series.py` line 182 (7 lines) · arguments: a · calls: not called on the traced runs · files named: `rows.csv`

## 20. One blood draw: `run_draw.py` (development, to-do step 40)

**What it is for.** One blood draw, split into its buffy coat (the person's own white cells) and its plasma (cell-free DNA), read end to end:
`run_draw.py` runs the one command, `run_sample.py`, on each half with `--cell-panel atlas` (and `--dev-seq-values` for sequencing), so every
instrument with enough to read reads, then writes the draw's joint record from the two bundles alone. The GitHub workflow `one_draw.yml` runs
it from public GEO samples with one click. Nothing in it is commissioned; every reading keeps its own flag.

### `run_draw.py::is_blood`

Whether an atlas cell type is a blood or bone-marrow cell, by its name.

*Code:* `MethylPhys_Interface/run_draw.py` line 21 (1 lines) · arguments: c · calls: not called on the traced runs · constants read: `BLOOD_WORDS`

### `run_draw.py::half_args`

The one command's input flags for one half: an IDAT pair, a values file, a .pat file or a GEO sample (the last two with sequencing values).

*Code:* `MethylPhys_Interface/run_draw.py` line 23 (8 lines) · arguments: a, half · calls: not called on the traced runs

### `run_draw.py::read_half`

Runs the one command on one half, on the full atlas panel, and returns its bundle, or the reason it was not read.

*Code:* `MethylPhys_Interface/run_draw.py` line 32 (10 lines) · arguments: a, half, specimen, extra · calls: not called on the traced runs · constants read: `HERE`, `RS` · files named: `bundle.json`

### `run_draw.py::sources`

A bundle's sources: each cell type's share with its 95 % interval, and its Met-A for each cell where read.

*Code:* `MethylPhys_Interface/run_draw.py` line 43 (4 lines) · arguments: B · calls: not called on the traced runs

### `run_draw.py::main`

Reads both halves and writes `draw.json` and `draw_summary.txt`: each half's sources and Met-A, IAM-A with its counting range, and the
plasma's sources the person's own buffy coat does not explain: every non-blood source, and every blood source whose plasma share is above
its buffy-coat share by more than its 95 % interval.

*Code:* `MethylPhys_Interface/run_draw.py` line 48 (41 lines) · arguments: none · calls: not called on the traced runs · files named: `draw.json`, `draw_summary.txt`

### `stage_metaa_atlas.py::_clustering`

The commissioned C-Score's statistic, unchanged: the departures in genome order, the variance of the means of blocks of 10 neighboring sites (times the square root of 10) over the variance of the sites. Departures independent from site to site give 1; departures that run along the genome give more. Only the layout counts, not the size. *Derived* (the variance of a mean of independent values).

*Code:* `stage_metaa_atlas.py` line 763 (7 lines) · arguments: z, w · calls: not called on the traced runs · constants read: `CS_BLOCK`

### `stage_metaa_atlas.py::_genome_order`

Sorts site ids into genome order, the order of IAM-Atlas v2.1's site list, which the planted runs and the globe use.

*Code:* `stage_metaa_atlas.py` line 771 (6 lines) · arguments: ids · calls: not called on the traced runs · constants read: `ATLAS` · files named: `import pyarrow.parquet`, `s sites.parquet`, `sites.parquet`

### `stage_metaa_atlas.py::cscore_in_mix`

The C-Score for each cell found in a mixed sample and for the whole genome (development, flagged). For every cell the IAM-Deconvolver finds: its departure map at its identity sites (the map its globe layer draws), read by the commissioned statistic. The whole-genome version reads the same at every atlas site the sample measured. Untared: healthy samples of the same series are the comparison until a tare for these readings is built. Below 1 % of the sample the C is flagged, not withheld (*stated*): there a cell's departure map is mostly the other cells'. Which cells are there is the IAM-Deconvolver's answer. It also reports the whole-genome map's correlation between neighboring sites at lags 1 to 9 (neighbor_corr), in genome order: near zero where departures are independent, as the derived limit assumes.

*Code:* `stage_metaa_atlas.py` line 778 (31 lines) · arguments: beta, shares, deps · calls: not called on the traced runs · constants read: `CS_BLOCK`, `CS_FLAG_SHARE`, `MIX_COVER`

### `stage_provisional.py::pool_hg38`

Pools per-CpG methylated and total read files (GEO, GRCh38) at IAM-Atlas v2.1's sites, through the public EPIC hg38 annotation; a position listed twice is summed.

*Code:* `stage_provisional.py` line 22 (17 lines) · arguments: samples_csv, work · calls: not called on the traced runs · constants read: `ANN`, `ATLAS` · files named: `sites_sequencing_key.parquet`

### `stage_provisional.py::column`

A provisional reference column from pooled reads: the level through the atlas's own inverse transfer (as the atlas converted its sequencing readings), the spread from counting (q = (m + 1/2)/(n + 1)), the median donor spread of the 79 atlas cells, and the atlas reference's laboratory and floor terms. Nothing fitted.

*Code:* `stage_provisional.py` line 40 (12 lines) · arguments: P · calls: not called on the traced runs · constants read: `ATLAS`, `SMOOTH` · files named: `sites.parquet`

### `stage_provisional.py::build`

Builds and saves a provisional column with its provenance (samples, annotation and column checksums); run by run_sample.py --build provisional.

*Code:* `stage_provisional.py` line 79 (6 lines) · arguments: cell, samples_csv, work, out · calls: not called on the traced runs · constants read: `ANN`, `DIR`, `FLAG`, `SMOOTH`

### `stage_provisional.py::load`

Loads a saved provisional column, if one was built.

*Code:* `stage_provisional.py` line 86 (2 lines) · arguments: cell · calls: not called on the traced runs · constants read: `DIR`

### `stage_provisional.py::with_provisional`

The atlas reference with provisional columns added, as a copy; IAM-Atlas v2.1 itself is never changed.

*Code:* `stage_provisional.py` line 89 (9 lines) · arguments: R, cells · calls: not called on the traced runs

### `stage_v4.py::stage_d21_provisional`

A second deconvolution with provisional columns beside IAM-Atlas v2.1, recorded as cell_shares_provisional and flagged; the atlas's own shares are unchanged (development, plan 40.18). If it fails (for example on a constructed file with almost no methylation profile), run_sample.py records it under cell_shares_provisional with its reason and the run goes on (2026-11-15, PHY-093 synthetic); it is a development reading and never stops a count.

*Code:* `stage_v4.py` line 65 (13 lines) · arguments: beta, panel, cells, sequencing · calls: not called on the traced runs · constants read: `_REF`

### `stage_provisional.py::column_v2`

Version 2 of a provisional column (after test 1): the level from the shrunk proportion (m + 1/2)/(n + 1), the counting spread carried through the atlas's inverse transfer by the two ends of its interval (no local slope, so thin sites at the ends are not made narrow), and only sites with at least 5 reads.

*Code:* `stage_provisional.py` line 55 (19 lines) · arguments: P · calls: not called on the traced runs · constants read: `ATLAS`, `MIN_READS_V2`, `SMOOTH` · files named: `sites.parquet`

### `stage_provisional.py::_name_version`

Reads a column name with its version ('megakaryocytes@2'); a bare name is version 1. The version is recorded in the bundle, the cell keeps its plain name.

*Code:* `stage_provisional.py` line 75 (3 lines) · arguments: cell · calls: not called on the traced runs

### `stage_coupling.py::read`

Nearest-neighbor coupling on the A gauge (a reading over its healthy reference) (in development, plan 49.5). coupling-A is the sample's coupling (Stage Q2) over the healthy
coupling of its cell type (`Runtime Matrices/Coupling/coupling_reference_v1.json`, built from IAM-Atlas v2.1's healthy sequencing inputs).
The gauge runs from 0 (errors never neighbors) to the ceiling `coupling_max` over the same healthy coupling (every molecule's errors in one
run), both counted. Output `coupling_a`: A, healthy, floor, ceiling, band, state, flag; without a reference, the reason.

*Code:* `stage_coupling.py` line 17 (13 lines) · arguments: q, cell, ref · calls: not called on the traced runs · constants read: `FLAG`

### `report_v4.py::_coupling`

The report's nearest-neighbor coupling section: coupling-A's state and Normal band beside the healthy coupling of the sample's cell,
flagged in development, with its gauge; without a reference, the coupling and the reason it has no gauge position.

*Code:* `MethylPhys_Interface/report_v4.py` line 225 (11 lines) · arguments: o · calls: not called on the traced runs · constants read: `E`

### `stage_coupling.py::reference`

Loads the healthy coupling reference (`Runtime Matrices/Coupling/coupling_reference_v1.json`): each cell type's healthy coupling and the
Normal band of coupling-A. Returns nothing until the reference is built, and the reading then says so.

*Code:* `stage_coupling.py` line 14 (2 lines) · arguments: path · calls: not called on the traced runs · constants read: `REF`

### `stage_metaa_atlas.py::seq_units`

An atlas level, on the array scale, in sequencing units: a + b x level from the conversion the atlas's own build measured for the published
sequencing atlas (`Runtime Matrices/Atlas_V21/sequencing_term_v1.json`), clipped to 0 to 1. Every comparison of a sequencing sample against
the atlas goes through it: Met-A for a sorted cell, Met-A for each cell in a mix, and the territories.

*Code:* `stage_metaa_atlas.py` line 151 (5 lines) · arguments: p · calls: not called on the traced runs · constants read: `SEQ_TERM`

## 21. The IAM Sequencing Atlas (development): `stage_seq_atlas.py`

The reference for sequenced samples, kept apart from IAM-Atlas v2.1 (arrays), plan 35.30. Measured, not fitted: every value is counted from
healthy donors' reads at every one of the 28,217,448 CpGs of the hg19 numbering the .pat and .beta files share. The one command builds it
one block at a time (`run_sample.py --build seq-atlas --build-samples CSV --build-block START:END --build-out PARQUET`); the blocks together
are the atlas. Its inputs (`doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv`) are the 153 healthy sequencing inputs behind IAM-Atlas
v2.1, each with its GEO .beta file, cell and sealed donor half. Each block is a parquet with the CpG index and, per cell, these columns
(uint16; levels times 10,000; 65535 is blank, where no donor of the cell has a read, and nothing is filled):
`donors`, `reads`, `meth` (methylated reads; with `reads` it gives the pooled level and its exact 95 % range), `level` (median of the donors'
levels), `spread` (the donors' standard deviation), `disorder` (each donor's binary entropy, averaged), `halfA`, `halfB` (the pooled level in
each sealed donor half; their difference is noise map 1), `readA`, `readB` (the pooled level in two random halves of every donor's reads,
seed 20261029; their difference is noise map 2, defined for a cell with one donor).

**The physics, step by step.** The IAM Sequencing Atlas is measured: every value is counted from healthy donors' reads, nothing fitted.
Where IAM's Law enters a sequenced reading:

1. **Held sites (derived, S1).** In sequencing the unmarked sites read at 0 with no array offset, so the healthy held-site level is IAM-A's
   derived baseline itself: 1 − ε0 = 0.95546 per site from M alone (A4), read like for like through the counter at the sample's own clustering
   (A6). Blind test PHY-084 met.
2. **Identity sites (measured, M4).** Met-A on sequencing reads the atlas's measured per-type level (`metaa_seq_reference_v1.parquet`).
   Using the derived held level where a reading is at fully held sites, recorded beside the atlas level, is still to check in this stage.
3. **The held error by tile (measured, A8).** `held_reference_v1.parquet` holds each healthy cell's lone errors and chances per 2,000-CpG tile,
   so IAM-A in a mix (`iama_in_mix`) reads a sample against its own mix of the cells' healthy rates. The per-type rates carry each type's own
   level, open (A8).
4. **The library floor (measured; A15, A17).** At sites where every healthy cell agrees completely, an opposite call on the unmarked side is the
   writer's one selection step (1/(1 + D), A15) plus the meter (conv_fail + tc_err, A17); on the marked side a held loss also reads as one.
5. **The end calls (calibrated per library, S2).** The calls at a molecule's two ends carry the library's chemistry; the inner reference
   (`metaa_inner_reference_v1.parquet`) holds the donors' counts without them.
6. **Noise maps (statistics).** Donor half A − half B and read half A − read half B; their mean squares' difference is the healthy
   donor-to-donor part, measured.

### `stage_seq_atlas.py::ref_compare`

Development, run by `run_sample.py --build ref-compare --build-cell CELL[,SEQ_CELL] --bundle OUT.json`. At one cell's Met-A identity sites, the array atlas's healthy level (IAM-Atlas v2.1) against the IAM Sequencing Atlas's pooled healthy donors (interior calls, sites with at least 10 reads): mean per-site disorder in bits, overall and by side. It shows how much of an array reference's height is the array's own response and how much the cells hold (2026-11-12).

*Code:* `stage_seq_atlas.py` line 979 (18 lines) · arguments: cell, seq_cell, min_reads · calls: not called on the traced runs

### `stage_seq_atlas.py::domain_disorder`

Each cell's mean pooled level, its disorder and the donor spread at the domains (`--build seq-domain-disorder`).

*Code:* `stage_seq_atlas.py` line 163 (13 lines) · arguments: none · calls: not called on the traced runs · constants read: `ARRAY_COLS` · files named: `seq_atlas_inputs.csv`

### `stage_seq_atlas.py::metaa_inner_of`

One cell's rows of the inner reference, by site.

*Code:* `stage_seq_atlas.py` line 788 (6 lines) · arguments: cell, path · calls: not called on the traced runs · constants read: `METAA_INNER`

### `stage_seq_atlas.py::metaa_inner_reference`

The IAM Sequencing Atlas's level at every Met-A identity site without the atlas's own molecule-end calls: its donors' calls after a one-call trim at each molecule end, and the end calls themselves (published minus trimmed). Built by run_sample.py --build seq-metaa-inner. Counts only. Development.

*Code:* `stage_seq_atlas.py` line 772 (15 lines) · arguments: trimmed_counts, out, published · calls: not called on the traced runs · constants read: `METAA_DONORS_DIR` · files named: `metaa_donor_counts_v1.parquet`

### `stage_seq_atlas.py::merge_pieces`

Piece folders joined block by block into one folder, so each builder sees every cell (`--build seq-merge-pieces`).

*Code:* `stage_seq_atlas.py` line 177 (22 lines) · arguments: piece_dirs, out_dir · calls: not called on the traced runs · files named: `pyarrow.parquet`

### `stage_seq_atlas.py::half_floor`

The mean library floor of the atlas inputs in one sealed half.

*Code:* `stage_seq_atlas.py` line 390 (8 lines) · arguments: half · calls: not called on the traced runs · constants read: `FLOORS_FILE`, `INPUT_FLOORS`

### `stage_seq_atlas.py::floors_of`

The recorded library floor of each named atlas input.

*Code:* `stage_seq_atlas.py` line 383 (6 lines) · arguments: gsms · calls: not called on the traced runs · constants read: `FLOORS_FILE`, `INPUT_FLOORS` · files named: `input_floors_v1.parquet`

### `stage_seq_atlas.py::input_floors`

Every atlas input's library floor from its own .beta at the floor sites (`--build seq-input-floors`).

*Code:* `stage_seq_atlas.py` line 365 (17 lines) · arguments: samples_csv, out · calls: not called on the traced runs · constants read: `FLOOR_SITES`

### `stage_seq_atlas.py::minkowski`

The three Minkowski functionals of a sphere map at each threshold: the share above, the boundary length, and the patches above minus below.

*Code:* `stage_seq_atlas.py` line 598 (18 lines) · arguments: ring, nu · calls: not called on the traced runs · constants read: `MINK_NU`

### `stage_seq_atlas.py::minkowski.comps`

The number of connected patches among the selected pixels (8 neighbors).

*Code:* `stage_seq_atlas.py` line 611 (3 lines) · arguments: sel · calls: not called on the traced runs

### `stage_seq_atlas.py::sky_minkowski`

A site-ordered map on the sphere, its Minkowski functionals, and those of shuffles of its own pixels.

*Code:* `stage_seq_atlas.py` line 617 (24 lines) · arguments: v, order, masked, n_perm, rng · calls: not called on the traced runs

### `stage_seq_atlas.py::noise_minkowski`

Every cell's two noise maps read by sky_minkowski (`--build seq-noise-minkowski`).

*Code:* `stage_seq_atlas.py` line 642 (12 lines) · arguments: cols, man, n_perm · calls: not called on the traced runs · constants read: `NOISE_DIR` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_seq_atlas.py::read_block`

One input's methylated and all reads over a block of CpGs: a .beta file (uint8 pairs) read locally or straight from GEO by byte range, with
retries, or an .lbeta (uint16 pairs, this stage's converted files). A file that is not the full hg19 length is refused.

*Code:* `stage_seq_atlas.py` line 24 (23 lines) · arguments: src, start, end · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::build_block`

The atlas over one block: for each cell, every column above, from its donors' counts. Returns the table and a summary (each cell's donors,
halves, CpGs with reads and median reads; sites at the source files' 255-read cap).

*Code:* `stage_seq_atlas.py` line 129 (27 lines) · arguments: samples, start, end, source · calls: not called on the traced runs · constants read: `BLANK`, `SEED`

### `stage_seq_atlas.py::build_block.pooled`

The pooled level of a cell's donors in one sealed half (all their methylated reads over all their reads), for the donor-half noise map.

*Code:* `stage_seq_atlas.py` line 143 (3 lines) · arguments: sel · calls: not called on the traced runs

### `stage_seq_atlas.py::hg19_dictionary`

The pipeline's hg19 CpG dictionary (wgbstools init_genome's CpG.bed.gz: chromosome, 1-based position, index), checked for 28,217,448 CpGs.

*Code:* `stage_seq_atlas.py` line 48 (5 lines) · arguments: cpg_bed · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::lift_to_hg19`

One donor's GRCh38 per-CpG read tables, pooled on GRCh38 and moved to the hg19 numbering (UCSC hg38ToHg19, pyliftover; a position that does
not move to one place is dropped and counted), each landing on the hg19 CpG whose C or G it is, so both strands add to one CpG. Written as an
.lbeta; the report counts positions read, moved and landed. Run by `run_sample.py --build seq-lift`.

*Code:* `stage_seq_atlas.py` line 91 (29 lines) · arguments: files, out, cpg_bed, chain · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `dev_stages.py::atlas_frame`

The table the round-1 tools read (each cell's level, uncertainty and donor spread by site), served from the atlas the chain is using
(IAM-Atlas v2.1, or the IAM Sequencing Atlas for a sequenced sample); a cell the active atlas lacks stops the reading with its name.

*Code:* `dev_stages.py` line 252 (17 lines) · arguments: cells · calls: not called on the traced runs · constants read: `_C`

### `dev_stages.py::_sky_in_mix_core`

The residual sky against the sample's own mix of atlas cells (the Deconvolver's shares): each site's departure over its expected scatter
(the cells' spreads, plus the array's 0.02 or, on sequencing, counting at the site's reads), mapped on the sphere, read against the
within-chromosome block-shuffle null with the look-elsewhere statistic.

*Code:* `dev_stages.py` line 271 (23 lines) · arguments: beta, shares, counts · calls: not called on the traced runs

### `dev_stages.py::sky_in_mix`

The residual sky against the sample's own mix (edit 239): identification-only cells (stage_metaa_atlas.ID_ONLY, the megakaryocyte column) are left out of the healthy expectation, as in Met-A's mix rule, and recorded under left_out_id_only; the rest is _sky_in_mix_core unchanged.

*Code:* `dev_stages.py` line 392 (8 lines) · arguments: beta, shares, counts · calls: not called on the traced runs

### `stage_seq_atlas.py::iama_in_mix`

IAM-A for a mixed sample: the sample's held errors over the number its own shares predict, tile by tile, from each cell's healthy held-error
rate in the IAM Sequencing Atlas's held-error reference and the sample's own opportunities; healthy reads 1.

*Code:* `stage_seq_atlas.py` line 268 (26 lines) · arguments: tiles, shares, half, ref · calls: not called on the traced runs · constants read: `HELD_MIN_OPP`, `HELD_REF`

### `stage_metaa_atlas.py::expected_H_counted`

The expected disorder of a healthy sample's k of n reads when the healthy level is itself counted (m of N reads in the reference): the
beta-binomial mean of the entropy, Jeffreys' half counts; a reference level of 0 then predicts the small disorder such a level really gives.

*Code:* `stage_metaa_atlas.py` line 177 (13 lines) · arguments: mref, nref, n · calls: not called on the traced runs · constants read: `N_EXACT`

### `stage_seq_atlas.py::metaa_half_counts`

Each cell's methylated and total reads at its Met-A identity sites in each sealed donor half, summed over the healthy inputs' .beta files.

*Code:* `stage_seq_atlas.py` line 471 (20 lines) · arguments: samples_csv, out, ref · calls: not called on the traced runs · constants read: `HG19_CPGS`, `METAA_REF` · files named: `seq_atlas_inputs.csv`

### `stage_seq_atlas.py::noise_at_identity`

At each cell's Met-A identity sites, by side: noise map 1 (donor halves) and noise map 2 (read halves) as mean squares, and map 1 against the
counting its two halves' own reads predict (`run_sample.py --build seq-noise-identity`).

*Code:* `stage_seq_atlas.py` line 837 (22 lines) · arguments: path · calls: not called on the traced runs · constants read: `BLANK`, `METAA_COUNTS_DIR`, `METAA_REF` · files named: `metaa_half_counts_v1.parquet`

### `stage_seq_atlas.py::beta_vs_pat`

One healthy input's published .beta counts against counts taken from its own molecule file at its cell's Met-A identity sites, and the
share of molecules covering two or more of those sites (`run_sample.py --build seq-beta-vs-pat`).

*Code:* `stage_seq_atlas.py` line 807 (29 lines) · arguments: beta_src, pat_path, cell, ref · calls: not called on the traced runs · constants read: `HG19_CPGS`, `METAA_REF`

### `stage_seq_atlas.py::table_to_lbeta`

A published per-CpG table (hg19 or hg38; plain counts or Bismark coverage) written as .lbeta on the hg19 numbering, both strands added to one
CpG (`run_sample.py --build seq-table`).

*Code:* `stage_seq_atlas.py` line 58 (32 lines) · arguments: src, out, cpg_bed, fmt, genome · calls: not called on the traced runs · constants read: `HG19_CPGS`, `LEVEL_N`, `TABLE_FORMATS`

### `stage_metaa_atlas.py::donor_pair`

The sample's disorder and one healthy person's expected disorder at a common depth (the smaller of the sample's reads and the deepest
donor's): both sides' reads thinned without replacement to that depth, so both are exact.

*Code:* `stage_metaa_atlas.py` line 216 (22 lines) · arguments: k, n, Md, Nd, u_s, u_d · calls: not called on the traced runs · constants read: `N_EXACT`

### `stage_metaa_atlas.py::donor_pair.thin`

The exact expected entropy of K methylated of N reads thinned without replacement to m reads (the hypergeometric), remembered per (K, N, m).

*Code:* `stage_metaa_atlas.py` line 226 (6 lines) · arguments: K, N, m, u · calls: not called on the traced runs

### `stage_metaa_atlas.py::expected_H_mix_counts`

The expected disorder of a healthy sample of a mix, from each cell's reference reads: every read from a cell of the mix, drawn without
replacement from that cell's reads, so no level is treated as exact (exactly unbiased).

*Code:* `stage_metaa_atlas.py` line 1017 (24 lines) · arguments: Mk, Nk, F, m, block, u · calls: not called on the traced runs

### `stage_metaa_atlas.py::_floor_probes`

The library-floor sites by probe (Seq_Atlas floor_sites_v1, side 'unmarked': every healthy cell of the IAM Sequencing Atlas unmarked there). The class floor (--dev-floor-class) counts a sample's and its reference's false-call rate at those of them that lie inside the islands read, so a call there is a false call and never the islands' own low methylation.

*Code:* `stage_metaa_atlas.py` line 1043 (7 lines) · arguments: none · calls: not called on the traced runs · constants read: `HERE`, `_FP` · files named: `floor_sites_v1.parquet`, `import pyarrow.parquet`

### `stage_metaa_atlas.py::islands_counted`

The islands read with that expectation, against the donor half given, both sides at a common depth of at most 20 reads.

*Code:* `stage_metaa_atlas.py` line 1051 (35 lines) · arguments: xs, counts, F · calls: not called on the traced runs · constants read: `FLOOR_CLASS`, `FLOOR_EQUAL`, `SAMPLE_FLOOR`, `TERR_HALF`, `TERR_M`, `TERR_SHARE_MIN` · files named: `terr_half_counts_v1.parquet`

### `stage_metaa_atlas.py::thin_H`

The exact expected disorder of a site's reads drawn down to fewer reads without replacement.

*Code:* `stage_metaa_atlas.py` line 1011 (5 lines) · arguments: k, n, m, u · calls: not called on the traced runs

### `report_globe_v4.py::use_atlas`

Read the globe's healthy reference from another atlas in v2.1's layout (the sequencing atlas for a sequenced sample).

*Code:* `MethylPhys_Interface/report_globe_v4.py` line 37 (5 lines) · arguments: path, name · calls: not called on the traced runs · constants read: `ATLAS`, `ATLAS_NAME` · files named: `the same sites.parquet`

### `stage_seq_atlas.py::bridge`

Each cell's sequencing level against IAM-Atlas v2.1's at the sites both measure (`--build seq-bridge`).

*Code:* `stage_seq_atlas.py` line 667 (19 lines) · arguments: cols · calls: not called on the traced runs · constants read: `BLANK`, `SPHERE_DIR` · files named: `import pyarrow.parquet`

### `stage_seq_atlas.py::genome_spectrum`

The correlation of a map's values at two sites by the genome distance between them, in distance bands.

*Code:* `stage_seq_atlas.py` line 690 (13 lines) · arguments: v, chrom, pos, bins, offsets · calls: not called on the traced runs · constants read: `GD_BINS`, `GD_OFFSETS`

### `stage_seq_atlas.py::noise_spectrum`

Every cell's two noise maps read by genome_spectrum (`--build seq-noise-spectrum`).

*Code:* `stage_seq_atlas.py` line 704 (11 lines) · arguments: cols · calls: not called on the traced runs · files named: `import pyarrow.parquet`, `sites.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::sky_mask_sites`

The sites the sky statistics leave out: the sex chromosomes and the probes that cannot be placed uniquely in the genome.

*Code:* `stage_seq_atlas.py` line 573 (7 lines) · arguments: ids · calls: not called on the traced runs · constants read: `POOR_MAP` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::sky_null`

A map in site order on the sky sphere at NSIDE 128, its band powers against its own shuffled null; the bands above it.

*Code:* `stage_seq_atlas.py` line 581 (14 lines) · arguments: v, order, masked, n_perm, rng · calls: not called on the traced runs

### `stage_seq_atlas.py::noise_null`

Every cell's two noise maps read by sky_null (`--build seq-noise-null`).

*Code:* `stage_seq_atlas.py` line 655 (11 lines) · arguments: cols, man · calls: not called on the traced runs · constants read: `NOISE_DIR` · files named: `import pyarrow.parquet`, `sites.parquet`

### `stage_seq_atlas.py::noise_maps`

The atlas's two noise maps at v2.1's sites per cell (donor halves; read halves) and a summary row per cell.

*Code:* `stage_seq_atlas.py` line 537 (17 lines) · arguments: cols · calls: not called on the traced runs · constants read: `BLANK`, `NOISE_DIR` · files named: `import pyarrow.parquet`

### `stage_seq_atlas.py::noise_check`

The noise columns against the committed Met-A reference at the identity sites: the share of values identical.

*Code:* `stage_seq_atlas.py` line 555 (15 lines) · arguments: cols, ref · calls: not called on the traced runs · constants read: `BLANK`, `METAA_REF`, `NOISE_DIR` · files named: `pyarrow.parquet`, `sites.parquet`

### `reference_globes.py::noise_sphere`

The sequencing atlas's two noise maps drawn on the sky sphere per cell (`--build seq-noise-sphere`).

*Code:* `MethylPhys_Interface/reference_globes.py` line 139 (23 lines) · arguments: out, cols, man, lim · calls: not called on the traced runs · constants read: `ATLAS`, `GREY` · files named: `import pyarrow.parquet`, `noise_sphere.json`, `sites.parquet`

### `reference_globes.py::noise_sphere.tex`

One noise map as a sky texture on a diverging scale, grey where blank.

*Code:* `MethylPhys_Interface/reference_globes.py` line 153 (3 lines) · arguments: m, path · calls: not called on the traced runs · constants read: `GREY`

### `stage_seq_atlas.py::range_of_average`

Each cell's pooled level at a site and the 95 % range of the cell's average: donor spread over the square root of the donors times the t
quantile, or counting only for one donor.

*Code:* `stage_seq_atlas.py` line 494 (19 lines) · arguments: meth, reads, donor_sd, k · calls: not called on the traced runs

### `stage_seq_atlas.py::sphere_columns`

The sky sphere's input from the IAM Sequencing Atlas, in IAM-Atlas v2.1's layout (`--build seq-sphere-cols`).

*Code:* `stage_seq_atlas.py` line 514 (20 lines) · arguments: out, cols, inputs · calls: not called on the traced runs · constants read: `ARRAY_COLS`, `BLANK` · files named: `pyarrow.parquet`, `seq_atlas_inputs.csv`, `sites.parquet`, `sphere_cols.json`

### `stage_seq_atlas.py::terr_half_counts`

Each cell's reads at every island site in each sealed donor half (`--build seq-terr-halves`).

*Code:* `stage_seq_atlas.py` line 719 (14 lines) · arguments: samples_csv, out · calls: not called on the traced runs · constants read: `HG19_CPGS` · files named: `pyarrow.parquet`, `seq_atlas_inputs.csv`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::terr_counts_of`

The given cells' reads at the island sites, from one donor half or both.

*Code:* `stage_seq_atlas.py` line 734 (17 lines) · arguments: cells, half, probes, own_cell · calls: not called on the traced runs · constants read: `TERR_DIR`, `TERR_FILE` · files named: `terr_half_counts_v1.parquet`

### `stage_metaa_atlas.py::expected_H_donors`

One healthy person's expected disorder at k of n reads, from healthy donors' own counts: n of each donor's reads drawn without replacement
(exactly unbiased where the donor has at least n reads), averaged over donors; differences between healthy people enter as they are.

*Code:* `stage_metaa_atlas.py` line 191 (24 lines) · arguments: Md, Nd, n · calls: not called on the traced runs · constants read: `N_EXACT`

### `stage_seq_atlas.py::metaa_donor_counts`

Each healthy input's own methylated and total reads at its cell's Met-A identity sites, one row per input and site (`--build seq-metaa-donors`).

*Code:* `stage_seq_atlas.py` line 755 (14 lines) · arguments: samples_csv, out, ref · calls: not called on the traced runs · constants read: `HG19_CPGS`, `METAA_REF`

### `stage_seq_atlas.py::metaa_donors_of`

One cell's per-donor counts at its identity sites (sites by donors), the donor half's donors only when a half is given.

*Code:* `stage_seq_atlas.py` line 795 (11 lines) · arguments: cell, half, exclude · calls: not called on the traced runs · constants read: `DONORS_FILE`, `METAA_DONORS_DIR` · files named: `metaa_donor_counts_v1.parquet`

### `stage_seq_atlas.py::array_columns`

The IAM Sequencing Atlas at every site of IAM-Atlas v2.1, in v2.1's layout (`run_sample.py --build seq-array-cols`): each cell's counted
level and donor spread, so a sequenced sample's in-mix readings mix sequencing levels; the sites themselves are still chosen on v2.1.

*Code:* `stage_seq_atlas.py` line 235 (29 lines) · arguments: piece_dirs, outdir, fields · calls: not called on the traced runs · constants read: `ARRAY_FIELDS`, `BLANK` · files named: `pyarrow.parquet`, `s sites.parquet`, `sites.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::deconv_reference`

The IAM-Deconvolver's reference for sequenced samples (`run_sample.py --build seq-deconv-ref`): each sequencing atlas cell's counted level
and donor spread at the array sites, marker sites chosen on the sequencing atlas by the Deconvolver's own rule; only the markers stored.

*Code:* `stage_seq_atlas.py` line 200 (29 lines) · arguments: piece_dirs, out, per_cell, with_mk · calls: not called on the traced runs · constants read: `BLANK`, `DECONV_SD_FLOOR` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::deconv_reference_of`

That reference as Stage D21 takes it (levels, spreads, cells, markers); nothing when the file is absent, and the shares then say they were
read against IAM-Atlas v2.1.

*Code:* `stage_seq_atlas.py` line 295 (8 lines) · arguments: path · calls: not called on the traced runs · constants read: `DECONV_DROP`, `DECONV_REF`

### `stage_seq_atlas.py::metaa_reference`

Each cell's IAM Sequencing Atlas values at its Met-A identity sites (`run_sample.py --build seq-metaa-ref`): the same sites Met-A reads on
arrays, chosen by its site rule on IAM-Atlas v2.1, so Met-A stays one instrument, with every stored column from the built pieces, as one
long table (cell, probe, cpg, columns). Small enough to sit in the chain (`Runtime Matrices/Seq_Atlas/metaa_seq_reference_v1.parquet`).

*Code:* `stage_seq_atlas.py` line 306 (27 lines) · arguments: piece_dirs, out · calls: not called on the traced runs · constants read: `METAA_COLS` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::floor_sites`

The library floor's sites (`run_sample.py --build seq-floor`): among IAM-Atlas v2.1's sites, the CpGs where every cell of the IAM Sequencing
Atlas has at least 20 reads and reads at or below 0.05 or at or above 0.95. There a healthy cell carries no information, so an opposite call
is the library's own (unconverted DNA, sequencing error, damage).

*Code:* `stage_seq_atlas.py` line 337 (24 lines) · arguments: piece_dirs, out, rule · calls: not called on the traced runs · constants read: `BLANK`, `FLOOR_RULE` · files named: `pyarrow.parquet`, `sites_sequencing_key.parquet`

### `stage_seq_atlas.py::library_floor`

A sequenced sample's library floor: at those sites, the share of opposite calls on each side and pooled, from the sample's own calls;
recorded on every sequenced sample as `development.library_floor`, flagged.

*Code:* `stage_seq_atlas.py` line 399 (12 lines) · arguments: beta, calls, path · calls: not called on the traced runs · constants read: `FLOOR_SITES`

### `stage_seq_atlas.py::identity_side_table`

The sites `floor_by_position` reads when `run_sample.py --build seq-floor-position` is given `--build-cell CELL`: that cell's Met-A identity sites in the IAM Sequencing Atlas's Met-A reference, each with the side the cell holds it on (level at most 0.2 unmarked, at least 0.8 marked, between left out). So the read-end split is made where Met-A reads, not only at the floor. CELL is the cell the IAM-Deconvolver found in that library, never its label. Counts only.

*Code:* `stage_seq_atlas.py` line 412 (10 lines) · arguments: cell, out, lo, hi · calls: not called on the traced runs · constants read: `FLOOR_SITES` · files named: `metaa_seq_reference_v1.parquet`

### `stage_seq_atlas.py::floor_by_position`

The library floor counted from the molecules themselves, split by where each call sits on its molecule: at the floor sites (every healthy cell agrees completely), the share of opposite calls by the call's distance in CpGs from the nearer end of its molecule, and apart, beside a gap between the two reads of a pair. Biology does not depend on where a CpG sits on a molecule; library chemistry at the read ends does (the M-bias of end repair, as geneticists call it). It is the detector-split null test of the CMB missions on sequencing: split the data by a property the sky cannot depend on, and whatever differs is the instrument. Run alone by `run_sample.py --build seq-floor-position --build-input PAT`. The counts are also given per side (`by_side`: a false C at an unmarked site and a false T at a marked one, which are different errors), so a library's own error profile can be built into its expected value. With `--geo-pat GSM` the route fetches the sample's molecule file from GEO itself.

*Code:* `stage_seq_atlas.py` line 423 (45 lines) · arguments: pat, path, max_d · calls: not called on the traced runs · constants read: `FLOOR_SITES`

### `stage_seq_atlas.py::floor_by_position.mh`

The site-matched ratio inside `floor_by_position`: over the floor sites holding both end calls and inner calls, the end calls' opposite rate against the inner calls' at the same sites (a Mantel-Haenszel ratio, each site weighted by its calls), so the mix of sites cancels. Counts only.

*Code:* `stage_seq_atlas.py` line 457 (5 lines) · arguments: items · calls: not called on the traced runs

### `stage_seq_atlas.py::metaa_reference_of`

One cell's rows of that reference, indexed by probe, levels 0 to 1 (blank as missing); nothing when the file or the cell is absent, and the
reading then says it was read against IAM-Atlas v2.1.

*Code:* `stage_seq_atlas.py` line 860 (16 lines) · arguments: cell, path · calls: not called on the traced runs · constants read: `BLANK`, `METAA_COUNTS_DIR`, `METAA_REF` · files named: `metaa_half_counts_v1.parquet`

### `stage_seq_atlas.py::held_reference`

Each healthy cell's held error region by region, stored beside the sequencing atlas (`run_sample.py --build seq-held`): from each healthy
sample's held-error counts by tile (`--dev-count-only --dev-count-tiles`; a tile is 2,000 consecutive CpGs), summed by cell, as counts, never
ratios: donors, lone errors and their chances in all, in each sealed donor half, and in each half of every donor's molecules. A tile's held
error is its lone errors over its chances. It is the healthy held error a plasma's own mix is built from: each cell's, weighted by its share.

*Code:* `stage_seq_atlas.py` line 877 (25 lines) · arguments: samples, out · calls: not called on the traced runs

### `stage_seq_atlas.py::compare_pair`

How far two samples sit from each other (`run_sample.py --build seq-pair`): at the CpGs where both have at least 5 reads, the mean level
difference, and the summed squared differences over the squared difference counting predicts (about 1 for the same cell read twice).

*Code:* `stage_seq_atlas.py` line 903 (16 lines) · arguments: a_path, b_path, min_reads · calls: not called on the traced runs · constants read: `HG19_CPGS`

### `stage_seq_atlas.py::compare`

How far one sample sits from every cell of the atlas (`run_sample.py --build seq-compare`): at the CpGs where the sample has at least 5 reads
and every cell has a level, the mean distance between the sample's level and each cell's, cells ranked nearest first; and for each cell, the
mean distance between its two donor halves at the same CpGs. Reads the built pieces in two passes (levels first, halves only at the compared
CpGs) so a 16 GB machine holds it. Also a depth-weighted score per cell: each CpG's squared difference over the scatter counting gives it
(the sample's and the column's binomial scatter and the column's donors' spread squared), averaged, about 1 for a healthy cell of that column;
ranked as `weighted_ranked`, so a column pooled from sparse reads is not pushed away by its own counting scatter.

*Code:* `stage_seq_atlas.py` line 920 (53 lines) · arguments: lbeta, piece_dirs, min_reads · calls: not called on the traced runs · constants read: `BLANK`, `HG19_CPGS` · files named: `pyarrow.parquet`

### `stage_seq_atlas.py::geo_url`

The hg19 .beta file of a published sequencing sample on GEO (used when the inputs list gives no file).

*Code:* `stage_seq_atlas.py` line 974 (3 lines) · arguments: gsm, title · calls: not called on the traced runs

### `stage_seq_atlas.py::_u16`

Values to the stored uint16 form (times 10,000, clipped; blank 65535).

*Code:* `stage_seq_atlas.py` line 121 (2 lines) · arguments: x, scale · calls: not called on the traced runs · constants read: `BLANK`

### `stage_seq_atlas.py::_H`

Binary entropy in bits, 0 at a level of 0 or 1.

*Code:* `stage_seq_atlas.py` line 124 (4 lines) · arguments: p · calls: not called on the traced runs
