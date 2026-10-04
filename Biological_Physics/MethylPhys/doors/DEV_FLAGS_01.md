# DEV-FLAGS-01 - development flags for the stages likely to work, and the stages kept out (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Author decision N.** Stages likely to work go behind a development flag and are actively worked; stages that cannot work as designed stay
out and are listed in the SOP with why.

**Flagged (each writes `development.<stage>` into the bundle and a "Development (behind flags)" section into the report, labelled
DEVELOPMENT - not commissioned; none changes the reading, the gauge or the tare).**
`--dev-nilc` (NILC-e, stage 4), `--dev-atlas-e` (atlas_e, stage 3; needs `--atlas-v2 <parquet>`), `--dev-percell-b` (stage 5 B cells, development
floor), `--dev-sky` (stage 11 sky map + stage 12 statistics with the block-shuffle null; needs healpy), `--dev-direction` (stage 10, DEV-DIRECTION-02),
`--dev-trace` (3b), `--dev-foreign` (3c), `--dev-brightness` (11b), `--dev-selftare-ii` (DEV-SELFTARE-02), `--dev-epic-v2` (EPIC v2 through the
SeSAMe calibrator, DEV-EPIC-V2-01).

**Check (set now).** On the GSE250556 replicate arrays and on the release-check constructed specimen: with every flag on, the reading (A, A_rel,
state, C) is identical to the run without flags, and each flagged block is written and labelled. Bar: 100 %.

**Kept out (listed in the SOP with why).** The class-era stage 10 panel (`bidirectional_decomposition.py`: z against other arrays' mean and SD, a
disease sign); the class-era 3c panel (`detection_panel_v3.json`: a line from other arrays, class-era beta scale); the class-era 11b
(`toolkit_surface_brightness.py`: class archives); the toolkit NILC module N1 (`nilc_celltype_deconvolver.py` on the class-era atlas: every truth bar
missed in DEV-NILC-01); null N7 of the null runner (its generator was retired).

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 9bcbcf99. Records: `data/DEV_FLAGS_01/flags01.csv`.

- \measured GSE250556, 63 arrays, run twice through `run_sample.py --betas` (tared on three given references), without flags and with every flag
  (`--atlas-v2` given): reading identical (A, state, A_rel, tare state, C) on **63 of 63**; every development block labelled on 63 of 63; the development
  section in the report on 63 of 63. Blocks: selftare_ii:OK; direction:OK; trace:OK; foreign:OK; brightness:OK; nilc:OK; atlas_e:OK; percell_b:OK; sky:OK.
- \observed `--dev-epic-v2` is checked on its own (DEV-EPIC-V2-01): it needs an IDAT pair.
- The release check carries the same test on the constructed specimen (E10).
