# DEV-CEILING-AUDIT-01 — has any reading of ours crossed the derived ceiling? (audit, 2026-10-13)

**Why ask.** The gauge's right-hand mark is derived, not measured: the reading rises with pattern loss only while the specimen's
methylated sites stay above beta = 1/2. Past that the specimen is at one bit per site, and the reading turns over and falls as loss
continues (`CANON/foundation.json`, entry `entropy_ceiling`). A specimen past the ceiling therefore reads LOW while being badly
damaged. If any sealed reading of ours sat past it, that reading was under-reported.

**What was audited.** Every committed reading file that carries the chain's ceiling diagnostic:

| file | readings | past the ceiling |
|---|---|---|
| `DEV_BASE_CHAIN_01/readings_all.csv` | 7,490 | **0** |
| `DEV_REPL_V3_01_run/proc_repl_v3_01_box_readings.csv` | 64 | **0** |
| `DEV_REPL_V3_01_run/pass1.csv` | 64 | **0** |
| `DEV_REPL_V3_01_run/pass2.csv` | 63 | **0** |

**Result: of 7,681 readings that record the flag, none is past the ceiling.** The sorted-cell path has recorded this diagnostic
since `conductor_v3._ceiling` was written, and it has never fired.

**The gap, stated plainly.** The composition-aware path - the one behind the breast, colorectal and liver tissue results - never
recorded the flag, and the per-sample outputs it saved do not keep the mean beta of the methylated sites. So those three sealed
results **cannot be audited from their saved files**; auditing them would require re-reading the calibrated arrays.

**Why that gap can only have cost us detections, never manufactured one.** Each of those tests asked whether tumors read ABOVE the
healthy range. A specimen past the ceiling reads LOWER than its damage warrants. So an unnoticed past-ceiling tumor would have been
counted as NOT detected - a false negative. The direction of the error is the safe one: the sealed counts may understate how many
tumors leave the healthy range, and cannot overstate it. That argument is about the direction of a known inequality, not an estimate
of how many were affected, which remains unknown until the arrays are re-read.

**The fix, already in.** `chain/dev_comp_methods.ceiling_flag` records the same diagnostic for the composition-aware reading: the
mean beta of the sites the estimated MIXTURE expects to be methylated, and whether it has crossed one half. Every future tissue or
whole-blood reading carries it.

**Still open.** Re-read the three sealed tissue sets with the flag recorded and report how many specimens, if any, sat past the
ceiling. Free (the calibrated arrays are on disk) but it is a re-read of sealed data, so it is reported as an audit of those results
and cannot change their sealed verdicts.

**In plain words.** Our gauge has a top end that physics fixes: past a certain amount of damage, the needle stops climbing and starts
coming back down, so a very badly damaged sample can read deceptively low. We checked every reading we have that records this, 7,681
of them, and not one was past that point. For the three tumor-tissue results the check was never recorded, so we cannot answer it from
the files we saved - but the direction of the error is the safe one: it could only have made us MISS tumors, never invent one. The
missing check is now recorded for every future reading of that kind.

**Not a milestone:** an audit of existing records, with one part still open.
