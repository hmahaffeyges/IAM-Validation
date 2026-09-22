# Author copies received 2026-09-23

Eight files were sent from the author's own working folder. Five were **byte-identical** to the repository copies:
`cpg_intake_form.html`, `idat_decoder_pure.py`, `idat_parse.py`, `run_batch.py`, `stage_1_calibration.py`.

Three differed, and in every case the repository copy is the one with the later fix, so the repository copies
stand and the author copies are kept here for comparison only:

| file | difference |
|---|---|
| `stage_0_intake.py` | the repository copy carries the 2026-09-19 hard-fail guard (PROC-STAGE0-01): a `QUARANTINE_INCOMPLETE_MANIFEST` or `QUARANTINE_MISSING_CHANNEL` sample used to fall through to PROCEED |
| `stage_1_idat_calibration.py` | the repository copy stamps `meta["pipeline"]` (LESSON-SCALE-01), which is the key `beta_scale_maps_v1.json` is looked up by; without it a reading is UNMAPPED and not reportable |
| `stage_5_second_chain.py` | the repository copy resolves the class archive by searching the tree rather than a fixed `IAM_Atlas/` path |

Nothing from these copies was merged. Use the files in `chain/`.
