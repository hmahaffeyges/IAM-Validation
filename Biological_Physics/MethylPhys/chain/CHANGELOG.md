# Chain changelog

## 2026-10-09 - Met-A on neutrophils COMMISSIONED (author approval)

- `conductor_v3.py`: `BUILD` names Met-A on neutrophils (EPIC v1) commissioned 2026-10-09 and every other stage development;
  `METAA_COMMISSIONING` record (scope, stages, detection limits 2 % / 5 %, what is not commissioned) attached to `met_a` when A is read.
- `report_v3.py`: green commissioning box in the Stage 5 section with both detection limits; `run_sample.py`: ledger label follows.
- Note: `doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md`. No number, frozen input or locked result changed.

## 2026-10-08 - chain v3 development round 3 (DEVELOPMENT - not commissioned)

Records: `development/METHYLPHYS_DEVELOPMENT_LOG.md` (2026-10-08 entries), `doors/DEV_IAMA_P_WHOLE_01.md`, `doors/DEV_IAMA_INTAKE_01.md`. No locked result changed.
- `conductor_v3.py`: `stage_t_cscore()` - the Met-A C-score is tared like A: `C_rel` = C / median C of >= 3 same-run healthy references
  (records carrying `C`), shown against the development band 0.751-1.409 (DEV-CSCORE-TARE-01). Untared C is still printed.
- New frozen input `Runtime Matrices/IAM_A_Positions/iama_positions_v2.json`: neutrophil P = 1.1492 measured on the WHOLE Loyfer granulocyte
  files (v1, P = 1.099 on the first 60 MB of each file, moved to `superseded`). `stage_q_iam_a.py` reads v2 and refuses a reading of part
  of a file (`allow_partial` / `--dev-allow-partial` for development only).
- New frozen input `Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json` (wgbstools 0.1.0 hg19 dictionary, 28,217,448 CpGs).
- New `stage_q0_intake.py`, Stage Q0 (IAM-A intake): readable file, hg19 build, whole genome, specimen (rules, stop now); conversion,
  read length, duplicates (recorded; limits set from healthy files before any test file). Called by `run_sample.py --pat` before Stage Q;
  `--alignment-qc` passes the alignment record. `release_check_v3.py` E11: five negative controls; E4 reads Stage Q's own position file.

## 2026-10-04 - chain v3 development round 2 (DEVELOPMENT - not commissioned)

Each check below was written in a dated `doors/DEV_*.md` note before the data were read; results are under the line in the same note and in
`doors/CHAIN_COMMISSIONING.md` (round 2 table). No frozen input changed. No locked result changed.
- `stage_0_intake.py`: sex and age optional (`NOT_DECLARED`); `ACCEPTED_SPECIMENS` (blood only) and `specimen_refusal()` - other specimens stop with `SPECIMEN_REFUSED`.
- `conductor_v3.py`: build label `DEVELOPMENT - not commissioned (chain v3, neutrophils only)`; gauge state withheld with the reason when noise-site coverage is below 90 %; `refusal_code` carried to the report.
- `MethylPhys_Interface/run_sample.py`: specimen refusal; identifiers hashed in bundle and ledger (`_hash_id`), typed id kept on the report; development flags (`--dev-selftare-ii`, `--dev-direction`, `--dev-trace`, `--dev-foreign`, `--dev-brightness`, `--dev-nilc`, `--dev-atlas-e`, `--dev-percell-b`, `--dev-sky`, `--dev-epic-v2`; inputs `--atlas-v2`, `--sesame-rscript`), each labelled DEVELOPMENT on the report.
- `stage_q_iam_a.py`: streamed reading of whole `.pat` files; IAM-A C-score (blocks of 1,000 sites); error count in the output.
- `MethylPhys_Interface/report_v3.py`: refusal codes; C-score line; development section; development stage statuses RAN / NOT_RUN / ERROR.
- New: `dev_stages.py` (self-tare on type II fixed sites, physics-only direction, block-shuffle sky null, stages 3b/3c/11b on own noise), `dev_comp_methods.py`, `dev_epicv2_sesame.R` (EPIC v2 development read, behind `--atlas-v2`; EPIC v2 stays refused by default), `Runtime Matrices/Development/` (development inputs, not frozen).
- `release_check_v3.py`: E1 is a whole-blood array with no sex or age; new E6-E10 (specimen refusal, hashed ids, noise-coverage withhold, C-score line, development flags labelled).
- `build_chain_sequence.py`, `chain_sequence.json`, `doors/CHAIN_SEQUENCE.md`, `TOOLKIT.md`, SOP v3, operator chapter updated to match.

---

## 2026-10-03 - chain v2 retired; chain v3 is the only engine

The class-floor engine (v2) is retired from the public repository and archived privately (one archive, paths preserved, sha256
recorded): its conductor, deconvolver, gauge engine, class floors and tiers, disease matrix and matching, synthetic patient
generator, propagate and inventory generators, report builder, the old SOP and the Edition 003 operations manual with their
generators, the v2 kit procedures and their results, the legacy example runs and the v2 Record folders.
`run_sample.py`: the legacy branch and its flags (`--lab`, `--lab-zero`, `--pipeline`) are removed; `--engine` accepts only `v3`
(kept so recorded commands still run); the bundle now carries `versions` (short hashes of every v3 frozen input and chain module).
New: `TOOLKIT.md` (stages 3, 3b, 3c, 4, 10, 11, 11b, 12, 12b, built and not wired into v3), `toolkit_foreign_detection.py`
(stage 3c, extracted from the retired conductor), `toolkit_surface_brightness.py` (stage 11b, extracted from the retired v1 conductor),
`sky_statistics.py` (stage 12, from the PROC-CLS-01 script), `stage_2c_trace_detection.py` restored as stage 3b,
`FROZEN_INPUTS_v3.json`, `release_check_v3.py`; `build_all.py`, `build_chain_sequence.py` and `guarded_push.sh` rebuilt for v3;
the operations manual is now built from the v3 operator chapter (`manual/build_manual_v3.py`). No frozen input changed.

---

# CPG pipeline update — disease-wall matcher + RUN-everything sweep

## 1. Per-cell directional matcher  (chain/disease_matching.py (archived privately) (the v1 conductor the v1 clinical conductor (archived privately) was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called), 1521 -> 1554 lines)
Replaced the absolute-magnitude cosine (whose |dep|>=0.15 floor gated out subtle
pre-dx directional signal) with a weighted directional matcher over each disease's
SIGNAL cells (|Cohen d| >= 0.20). 'cosine' now carries directional concordance in
[-1,+1] for downstream compatibility; the report is untouched.
- Hardened specificity gate: the neutrophil-to-lymphocyte / progenitor-expansion axis
  (myeloid-up, progenitor-up, lymphoid-down) is now NON_SPECIFIC_GENERIC. A tissue/origin
  cell or a lineage break makes a match SPECIFIC.
- STRONG needs dc>=0.70, coverage>=0.40, >=3 moved cells, mean |dep|>=0.15.
- Verified: injected myeloma -> flags multiple_myeloma (specific, strong); injected AML ->
  correctly NON_SPECIFIC (blood pattern alone cannot name AML vs CML vs reactive); the three
  subtle bundles produce no false concern.

## 2. RUN-everything residual sweep  (stage_5_second_chain.py, 284 -> 349 lines)
The second chain no longer fires only on the per-cell top match. It now ALWAYS sweeps every
available residual map (breast, AD, immune cross-disease universal alarm) for a whole-blood-
compatible patient, independent of the per-cell rank. This is the fix for the breast pre-dx
miss: that case's signal is distributional (secretory homogenization), so the per-cell matcher
correctly leaves it quiet and the matched filter must screen it on its own.
- A DETECTION is a POSITIVE-rho fire (consistent direction). A negative-rho fire is an
  anti-correlation, NOT a detection. Null check across EPIC-Italy healthy controls confirmed
  healthy blood anti-correlates with AD/immune maps (rho ~ -0.1 to -0.17) and sits at zero on
  the breast map, while breast cases fire breast positive (+0.058, +0.093).
- The confirmation verdict now flags only the top SPECIFIC, concern-worthy per-cell match;
  non-specific generic-axis matches are handled by the report's Mode 1 line, never escalated.

## 3. Report  (`cpg_report_builder.py` (retired 2026-09-26))
- New 'C - RUN-everything residual sweep' table in the Confirmation section (per-map rho, CI,
  CpGs, detection). None-safe for the no-per-cell-flag case.
- Updated the non-specific Mode 1 line to the generic-axis wording (was myeloid/lymphoid).

## 4. Doctor-facing mahalanobis cleanup (2 surgical strings)
- breast-epic_card_v3_1.json honest_limitations: "Universal Mahalanobis may capture..." ->
  "The universal architectural screen (now the residual matched-filter sweep) may capture...".
  Honest point preserved; only the removed-mechanism name corrected.
- flowchart_v4.html component list: mahalanobis -> matched_filter.
- LEFT INTACT: the matrix CSV and breast-card validation records that cite Mahalanobis d-values
  (e.g. d=+1.876/+2.097). Those are accurate validation HISTORY, not stale mechanism, and were
  not altered. A holistic card v3.2 reconciliation (matched filter throughout) is a separate
  discussion, not a unilateral edit.

## Validation summary
- breast pre-dx (GSM1235926): per-cell matcher quiet (correct); residual sweep fires breast
  (+0.058, CI [+0.001,+0.114]); AD + immune null. CAUGHT.
- healthy control (GSM1235534): second chain gate closed, no false flag.
- positive control (injected MM): flags multiple_myeloma, confirmed.

## 5. Systemic stress / inflammatory wellness signal  (chain/disease_matching.py (archived privately) (the v1 conductor the v1 clinical conductor (archived privately) was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called) + report)
New detect_systemic_stress_pattern(patient_departure): a wellness-level read (NONE / MILD /
NOTABLE) of the neutrophil-to-lymphocyte axis (myeloid + progenitor up, lymphoid down). It is
NEVER a disease call. Fires only on a coherent, real-magnitude pattern (n>=4 axis cells, mean
|dep|>=0.10, coherence>=0.60); flat noise and incoherent departures stay quiet.
- Wired into run_pipeline -> bundle["systemic_stress"].
- Calibration: flat-healthy and incoherent -> NONE; breast pre-dx / HCC / CRC -> NOTABLE.
  Large healthy-cohort calibration is honest future work.

## 6. Patient straw man  (deliverables/IAM_Patient_StrawMan.html)
The companion to the crown jewel: the patient's own per-cell architecture on the SAME eight-class
grid, so the two prints can be laid side by side.
- Top row = the patient's bloodwork: A-departure per scorable cell, GREEN = healthy band
  (A 0.93-1.07), red = elevated, blue = suppressed, intensity by magnitude. A bright outline
  marks a CONFIDENT departure (95% CI clears the band); a hatch marks mild / CI-uncertain.
- Beneath it, every disease the patient flagged is pulled straight from the crown jewel (here the
  full breast trajectory) for direct shape comparison. The wellness stress banner sits on top.
- Builders included under builders/ so the wall regenerates for any patient bundle.

## 6b. Patient straw man tier correction
Patient cells are now coloured by the FIVE gauge tiers (cpg_gauge.py / tier_breakpoints v1.3),
not a flat green band: SUPPRESSED <0.95, NORMAL 0.95-1.04 (green/healthy), ELEVATED 1.04-1.07
(amber), SIGNIFICANTLY_ELEVATED 1.07-1.10 (orange, past the Warburg line), BREACH >=1.10 (red).
Each cell shows its A-score; a white outline marks a confident departure (95% CI clears NORMAL),
a hatch marks CI-uncertain. This matches the report gauge exactly.
NOTE: the tier file puts the NORMAL->ELEVATED edge at 1.04 (used here), not 1.05.

---

## Stage 4.5 — AD directional detector wired in (bidirectional decomposition)

ROOT CAUSE: the residual matched filter (SOP 8.2) was acting as AD's detector. Its fixed
whole-blood baseline folds each patient's cell composition into the departure, so it
false-fired AD positive on healthy blood (rho ~ +0.55-0.60 on cases AND controls alike).
AD's validated per-patient detector is the Stage 4.5 directional composite (sealed VAL-051
Rule A, composition-independent z-score vs a frozen per-CpG reference). It was designed in
but Route C stood down in lean v1 and the module files were never placed.

CHANGES:
- Runtime Matrices/Directional Panel/: placed bidirectional_decomposition.py +
  directional_panels_v1_0.json (sealed VAL-051 Rule A 7-CpG immune panel).
- chain/disease_matching.py (archived privately) (the v1 conductor the v1 clinical conductor (archived privately) was retired 2026-09-25 to `RETIRED_2026-09/v1_conductor_2026-09/`; this is the one function the live chain called): Stage 4.5 wired into run_pipeline (computes per-class directional
  composite after Stage 4, feeds stage_8); config path -> Directional Panel.
- stage_5_second_chain.py: AD removed from _RESIDUAL_SWEEP_DISEASES (breast + immune-alarm
  only); AD directional read surfaced; gate fires on composite > 0.40 (the directional
  threshold), independent of the module's narrower cancellation flag.
- `cpg_report_builder.py` (retired 2026-09-26): AD directional subsection + machine-readable snapshot keys.
- flowchart_v4.html: Stage 4.5 node (what/why/the-trouble); matched-filter node corrected
  (offset-robust only vs UNIFORM shifts, never patterned composition shifts).

VERIFIED end-to-end (AIBL AD GSM4649829): composite +2.017, flags, second chain fires,
AD section renders. Healthy 58M reads -0.102 (no AD).

HONEST SCOPE: Rule A reference is AIBL-trained; discriminates within AIBL (AD +0.329 vs
HC +0.079) but does not transfer to other-platform cohorts (GIFT AD -1.228). Flag requires
composite > 0.40, so a non-transferring cohort is a MISS, never a false alarm. Physics cure
(re-anchor reference to IAMAtlas A-score departure, keep only the direction) is open VAL-053.

KNOWN OPEN (pre-existing, not from this change): per-cell deconvolution shows stem_adult/
progenitor breach reads in peripheral blood (artifact suspected); patient report does not
yet render the strawman / reference wall (separate pipeline). Both next.

---

## Patient straw man + crown-jewel reference wall wired into the report

`cpg_report_builder.py` (retired 2026-09-26) now renders the pattern-recognition straw man: the patient's own
per-cell A-score architecture on the eight-class grid, the disease rows it flagged pulled
from the crown jewel for side-by-side comparison, and the full reference disease-signature
wall (crown jewel, 81 rows) collapsed beneath it. The patient is measured by physics
(A = H(beta)/H_min, derived floor, no cohort); the wall supplies only the DIRECTION each
disease moves each cell -- the demarcation line. No per-disease derived panels: the matrix
v1.8 / crown jewel IS the reference, and the patient pattern is matched against it.

CHANGES:
- `cpg_report_builder.py` (retired 2026-09-26): added _exec_render (runs the sealed builders/render_patient_wall.py
  and `render_strawman_v2.py` (retired 2026-09-26) WITHOUT modifying them -- strips their file-I/O, injects data,
  captures HTML) and _strawman_section (builds the patient wall from the EXISTING bundle,
  no chain re-run; embeds both walls collapsed in iframes). 1097 -> 1229.
- Record/crown_jewel_and_patient_strawman/strawman_data_v2.json: NEW. The crown-jewel data,
  generated from disease_cell_signature_matrix_v1_8.csv (build_strawman + enrich_strawman).
  81 disease rows, 49 VAL-anchored. Shipped so the report builder needs no regeneration.

BREACH ARTIFACT (partial): the straw man's present-cell gate (above floor AND deconvolver
fraction >= 0.001) excludes the zero-fraction reads that inflate the class ranking -- on the
AIBL AD case, 19 such reads (HSC, CMP, L-MPP, MPP, erythroblast, megakaryocyte ...) are
dropped, leaving the 5 cells actually present. STILL OPEN: the class-level "Cellular
departure ranking" and cell census do not yet apply this gate, so they still show the
artifact (stem_adult 1.142). Fix pending design decision on threshold.
