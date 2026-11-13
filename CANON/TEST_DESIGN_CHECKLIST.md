# Test design checklist (canonical; read by CANON/design_check.py)

Author, 2026-11-13: "there is a whole checklist that should be reviewed when making decisions but that should happen during the design
phase when creating the synthetic test so that it never ends up on the wrong machine or the wrong data is not used ... something in code
that rejects the very first synthetic test without it."

Every new test (a manifest row whose record adds a sealed rule or a synthetic test) has a design sheet, design/<ROW>.json, written and
committed BEFORE its first synthetic test. The gate refuses the record, and the launchers refuse its jobs, until the sheet is complete.
Answers are evidence (numbers, file names, quoted protocol lines), not reassurance. Times are Pacific.

| Key | Question |
|---|---|
| physics.where_iam_law | Where is IAM's Law in this test? (physics first) |
| data.set | Which data, by accession? |
| data.protocol_read | The library method as deposited, quoted (PBAT, non-directional, low input, EM-seq, end clipping are flagged by code) |
| data.spent_first | Can data already read answer it? If not, why not? |
| data.alternatives | What other sets were considered, and why this one? |
| kiss.smallest | Is this the smallest set that answers the question? |
| kiss.samples_why | Why each sample (or group) is in |
| machines.platform | github or google (GitHub first) |
| machines.github_free_at_design | Free GitHub slots counted when designing |
| machines.why_not_github | Required for google: why not GitHub |
| machines.s_per_pair | Measured seconds per read pair on that platform, and the file it was measured in |
| machines.pairs | Pairs per library |
| machines.peak_mem_gb | Measured peak memory per job, and its source (or 'not measured' and parallel 1) |
| machines.machine_mem_gb | The machine's memory |
| machines.parallel_per_machine | Jobs per machine (parallel x peak must fit memory) |
| machines.expected_hours | Expected wall time per library, computed from s_per_pair x pairs plus fetch |
| machines.time_limit_hours | The job's limit (at least 1.3 x expected) |
| machines.cost | Expected cost |
| save.resume | ckpt, skip-done or merge: how a reclaimed or failed machine keeps its work |
| watchers.count | Exactly one watcher per wave |
| watchers.stop_condition | What ends the watcher |
| canary.instrument_check | Numeric pass rule for the canary before any full run: alignment jobs qc_failed_max, mapped_min, s_per_pair_max; count-only jobs rows_min, halves_corr_min, hours_per_sample_max |
| cosmologist | What would a cosmologist do: the null test or self-calibration in this design |
