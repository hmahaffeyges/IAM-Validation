# Operations Manual — chain v3, neutrophils (operator chapter)

**Build:** development v3 (2026-10-01), DEVELOPMENT - not commissioned; this chapter re-checked against the code on 2026-10-03 and updated for development round 2 on 2026-10-04; Stage T updated on 2026-10-04 for self-tare II then the median tare, adopted by the author (`boxruns/run1/JOBS.md` job A; `doors/DEV_SELFTARE_02.md`; development log 2026-10-04, DEV-PAIRED-01). Until then Stage T was the median tare alone (DEV-TARE-02, 2026-10-02). Self-tare II was wired into Stage T on 2026-10-04 (development log DEV-SELFTARE-03). The procedure is in `sop/MethylPhys_CPG_SOP_v3.md`; this chapter covers running it and reading the output.
The PDF manual in this folder (`MethylPhys_CPG_Operations_Manual.pdf`) is built from this chapter, the generated chain sequence and the toolkit list by `build_manual_v3.py`. The class-floor engine (v2) and its manual were retired on 2026-10-03 and are archived privately.
Paths are relative to `Biological_Physics/MethylPhys/`.

## Before a run
1. Use EPIC v1 IDAT pairs (Grn and Red) of whole blood or isolated / sorted / purified neutrophils. Any other specimen (PBMC, other sorted fractions, bone marrow, cell lines, tissue, unspecified) is refused at intake with a report. Sex and age are optional: give them when known and they are recorded; with a declared sex Stage 0.8 compares it with the array.
2. Put at least 3 healthy reference specimens of the same specimen type on the same slide (else the same batch), run the same way, so each reading can be tared. This applies to whole blood and to isolated neutrophils. Stage T is self-tare II (each array tared on its own fixed sites; no references needed), then the median tare against these references.
3. Use Python 3.11 with the versions in `chain/requirements.txt` (methylprep 1.7.1, numpy 1.26.4, pandas 1.5.3, scipy 1.17.1; reportlab for the manual): `pip install -r chain/requirements.txt`.
4. Stage 1 needs the Illumina manifest. methylprep downloads it on first use into `$HOME/.methylprep_manifest_files/` (`HOME` must be writable; network access to `array-manifest-files.s3.amazonaws.com`); offline, place the files there by hand (`doors/RUNBOOK.md` section 1). In a batch, run the first array alone so the download is not raced.
5. The frozen inputs are in the repository under `chain/Runtime Matrices/` (`Met_A_Floors/`, `IAM_A_Positions/`, `Intake/`); their hashes are in `chain/FROZEN_INPUTS_v3.json`.

## Run
From `chain/MethylPhys_Interface/` (run on the box: Stage 1 needs the manifest):
```
python run_sample.py --grn <Grn> --red <Red> --engine v3 --specimen "whole blood" --array-type EPIC_v1 \
  --id <id> --out <id>.html          # add --sex <F|M> --age <years> when known
```
- Isolated neutrophils: `--specimen "isolated neutrophils"`.
- Pass 2 (the tare): run each specimen again, whole blood and isolated alike, with `--slide-ref-A a1,a2,a3`, where those are the references' A values from pass 1 (`met_a.A`, self-tared, before the median tare), or with `--slide-ref-table refs.csv` (column `A`, optional column `id`; a row whose `id` is the specimen's own `--id` is left out). Fewer than 3 references: the median tare does not run and the reading stays untared.
- Self-tare II (Stage T step 1, adopted 2026-10-04): always runs (wired into Stage T on 2026-10-04); no flag is needed. It maps each probe design onto the reference arrays' scale by this array's own low and high fixed-site anchors before Met-A is read (`tare.selftare_ii`). The printed A_rel is the median tare of the self-tared A. The noise index reads the betas before self-tare II. `--dev-selftare-ii` is a no-op alias, kept so recorded commands still run.
- A beta table already calibrated by this chain's Stage 1 (two-column CSV `cpg_id,beta`; Stage 0 does not run): `python run_sample.py --betas <table>.csv --specimen "whole blood" --id <id> --out <id>.html`.
- Second draw of the same person (stage 12b): run the first draw with `--save-betas <id>_betas.parquet --patient-id <hash>`; run the second with the same `--patient-id` plus `--prior-betas <id>_betas.parquet --prior-bundle <id>_bundle.json`.
- Identifiers: the bundle and the ledger carry the sha256 hash of `--id`; the report keeps the id you typed.
- Development flags (DEVELOPMENT - not commissioned; never part of the reading; `--dev-selftare-ii` is a no-op alias since the adopted Stage T step 1 was wired in on 2026-10-04): `--dev-selftare-ii --dev-direction --dev-trace --dev-foreign --dev-brightness` (no extra input); `--dev-nilc --dev-atlas-e --dev-percell-b --dev-sky` with `--atlas-v2 <IAMAtlas_v2.parquet>` (`--dev-sky` needs healpy); `--dev-epic-v2 --sesame-rscript <Rscript>` for an EPIC v2 IDAT pair. Each writes `development.<stage>` into the bundle and a report section.
- Sequencing (Stage Q, IAM-A): `python run_sample.py --pat <file>.pat.gz --id <id> --out <id>.html`, or `python run_sample.py --site-table <sites>.csv --seq-pipeline loyfer_pat_v1 --id <id> --out <id>.html`.

Output: `<id>.html`, `<id>_bundle.json` beside it, and one row appended to `evidence_ledger.jsonl` in the same folder (`--ledger` to
change; `--no-bundle` writes neither the bundle nor the row). Exit 0 when a report is written (a refusal still writes one); exit 2 on a Stage 0 QUARANTINE, with no report, no bundle and no ledger row (a command-line error also exits 2).

Batch runners (written for the compute box; they read box paths and roster files and are not general tools):
`chain_tests/run_chain_acceptance.py` (pass 1 every specimen; pass 2 re-runs each whole-blood specimen with the other specimens of its
group as references; isolated specimens are not re-run) and `doors/data/DEV_REPL_V3_01_run/run_proc_repl_v3_01.py` (pass 2 with
`--slide-ref-table` built from the other arrays on the same slide, else the batch; exact commands in `COMMANDS.md` beside it).

Checks: `python chain/release_check_v3.py` (or `python kit/release_check.py`) - exit 0 only when every check passes (F1, F1b, S1-S4, E1-E10, M1). The IDAT checks E1-E3 and E6 need the manifest (run on the box).
Manual: `python manual/build_manual_v3.py` rebuilds the PDF.

## Reading the report
| field | meaning |
|---|---|
| Refused (specimen) | `SPECIMEN_REFUSED`: the specimen has no reference in chain v3; the refusal names it. No reading, Stage 1 does not run |
| Refused | the platform check: not an EPIC v1 vector (array type, EPIC v2 probe names, or 700,000 probes or fewer); an EPIC v2 IDAT pair is refused at intake, before Stage 1. No reading |
| Stage 0 intake | verdict PROCEED / PROCEED_WITH_PENALTY / QUARANTINE (QUARANTINE produces no report), call rate, flags; `not run` for `--betas`, `--pat`, `--site-table` or `--no-intake` |
| Stage 1 | poobah detection, call rate and controls: recorded, not gated |
| Stage A composition | the 8 blood groups (groups at 1 % or more are listed). Whole blood is read when neutrophils are ≥ 20 % and ≥ 867 of the 963 markers are measured |
| Stage M Met-A | isolated cells: A against the own floor, drawn on the gauge as untared until tared. Whole blood untared: A is a number only, no gauge position. Shift per 1 % loss of the neutrophil pattern |
| Stage T tare | adopted 2026-10-04: self-tare II (β′ = Lr + (β − L)(Ur − Lr)/(U − L) per probe design, from this array's own fixed-site anchors L, U and the reference arrays' Lr, Ur; nothing fitted), then the median tare. Printed: A_rel = A ÷ median of the references (self-tared A; self-tare II wired in on 2026-10-04, record `tare.selftare_ii`); number of references and their median; reference spread; detection limit (% loss of the pattern) = 2 × spread ÷ shift per 1 % loss. **Normal = 0.95–1.05** |
| Noise index N | mean H(β) on the 48,528 noise sites. Above 0.149 on an untared reading: state `withheld`, A printed as a number. Fewer than 90 % of the noise sites measured: N is not formed and the state is withheld, tared or not, with the counts and the reason |
| Methylated sites mean β | below 0.5: past the entropy ceiling; read β, not A |
| Stage MC C-score | genomic clustering of the departures (healthy = 1), with the healthy held-out range. Development: no band yet |
| Stage Q IAM-A | sequencing only: IAM-A, copy error eps, position P, eps0, the two halves, opportunities; the IAM-A C-score (development: independent copy errors give 1; band not set) |
| Development stages | only with a development flag: one row per flagged stage, labelled DEVELOPMENT - not commissioned |
| Withheld | what the build does not print, and why |
| Stage 12b difference map | only with `--prior-betas`/`--prior-bundle`: per-address difference to an earlier draw of the same person, or the refusal naming what differs (identifier hash, array type, pipeline) |
| Red flags | STOP / WITHHELD / CAUTION / NOTE, from the bundle; also written to the bundle as `red_flags` |
| Safeguards | rendered-claim scan (no disease, cohort or population words in the prose), formula self-test, anchors (Normal band, floor, C baseline, N_max read from the frozen files), deconvolver conformance, atlas separability (NOT_RUN until stage 3 is wired) |
| Troubleshooting, Integrity, Chain file inventory, Run it yourself | what to do for each condition on this specimen; IDAT hashes and chain commit; every frozen input and module with its hash; the exact command |
| The chain, Toolkit stages | SOP 2b stage table; each toolkit stage PASS / FAIL / REFUSED / NOT_RUN / NOT_BUILT on this specimen |

## Faults
| symptom | cause | action |
|---|---|---|
| QUARANTINE_INCOMPLETE_MANIFEST | array type not declared and not readable from the IDAT header (sex and age are optional since 2026-10-04) | supply `--array-type` |
| `SPECIMEN_REFUSED` | the specimen is not whole blood or isolated / sorted / purified neutrophils | none: that specimen needs its own reference first |
| `withheld: only <n> of the 48,528 noise sites were measured` | fewer than 90 % of the noise sites passed Stage 1 detection | re-hybridise or check the array's signal; the state cannot be shown without the array's own noise |
| ENVIRONMENT_MISSING_MANIFEST (exit 3; before 2026-10-03 this showed as QUARANTINE_CORRUPT_IDAT) with PermissionError or a connection error on `.methylprep_manifest_files` | the manifest is not in its cache and could not be downloaded; the IDAT is not at fault and is not judged | make `HOME` writable and allow the download, or place the manifest files by hand; re-run |
| Stage 0 QUARANTINE, other hard failures (`call_rate`, `detection`, `ctrl_qc`, `sex`, `integrity`, `hm450_coverage`, `intake_deferred:...`) | the array failed intake | none: no reading by rule; the flags name the check |
| `neutrophil fraction <f> < 0.2: fraction reported, A withheld` | neutrophils < 20 % of the whole blood | none: the fraction is reported and A is withheld by rule |
| `only <n> of 963 composition markers measured` / `only <n> of 6000 ... sites measured` | fewer than 90 % measured | none: A withheld |
| `untared: <n> same-run reference arrays (>= 3 required)` | fewer than 3 references supplied | run the references, then pass them with `--slide-ref-A` or `--slide-ref-table` |
| `tare.selftare_ii` NOT_RUN, `dev_selftare_typeII_EPIC_v1.json not found` (the reading then uses β without self-tare II) | the self-tare II file is missing from `chain/Runtime Matrices/Development/` | restore it from the repository |
| `withheld: noise index <N> > 0.149 and no same-run tare` | the array is noisier than the reference arrays; the gauge is not drawn | tare it against same-run references |
| refusal `... (450K or incomplete vector) ... (450K neutrophil floor pending)` | fewer than 700,001 probes after Stage 1 (a 450K table, or an EPIC array that lost probes at detection) | none: v3 reads EPIC v1 only. A 450K IDAT pair never gets here: it quarantines at Stage 0 (0.1 array-type mismatch when `--array-type EPIC_v1` is given, else 0.7b coverage) |
| refusal `... (no frozen neutrophil floor for this platform)` | declared array type is not EPIC_v1, or EPIC v2 probe names | none: no floor for that platform |
| IAM-A refusal `position for neutrophils was measured on loyfer_pat_v1, not <pipeline>` / `too few opportunities` | another read-level pipeline, or fewer than 100,000 opportunities | none: P is valid only for its own pipeline |
