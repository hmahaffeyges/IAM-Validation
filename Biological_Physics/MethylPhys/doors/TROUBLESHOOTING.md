# Troubleshooting — what the chain does when something is wrong, and what to do about it

This page is for the person running a sample, not for the person who wrote the chain. Every message below is
one the chain actually prints; every number is one it actually applies. Search this page for the exact string
you were given.

**Read this first, because it explains most of what follows.** The chain has three different ways of not giving
you an answer, and they mean different things:

| what you see | what it means | what to do |
|---|---|---|
| `QUARANTINE` | Stage 0 refused the specimen. **No report is written and nothing is scored.** The run exits with code 2. | Fix the cause below. Nothing downstream is worth looking at until you do. |
| `NOT REPORTABLE` / `UNSET` / `NOT ASSESSABLE` | The measurement ran; the chain will not put a number on it because a reference it needs does not exist. | Supply the missing reference (usually a laboratory zero). The value is not wrong, it is unplaced. |
| `DEFERRED` | A check could not be made. **It is not a pass.** | Read which check, and whether you care. Nothing is being hidden: the reason is printed. |

A fourth thing you may see is `PROVISIONAL_…`, which means a threshold exists in the SOP but has never been
measured against healthy specimens, so the chain reports the value and does not refuse on it. There is exactly
one of these today (bisulfite conversion) and it is described at the end.

---

## 1. Stage 0 refused my sample

Every refusal names itself. Find the string you got.

| status | what the chain found | what to do |
|---|---|---|
| `QUARANTINE_INCOMPLETE_MANIFEST` | A required manifest field is missing or empty. | The seven required fields are exact: `sentrix_id`, `array_type`, `patient_id`, `intake_date`, `substrate`, `declared_sex`, `declared_chronological_age`. Pass `--sex` and `--age` on the command line; a donor with no published age cannot clear this gate. |
| `QUARANTINE_MANIFEST_INVALID` | A field is present but not acceptable. The flags say which: `UNKNOWN_ARRAY_TYPE:x` or `CLEARTEXT_PII:patient_id_not_hashed`. | `array_type` must be one of **`HM450K`, `EPIC_v1`, `EPIC_v2`** — `450k` and `450K` are rejected. `patient_id` must be a hashed token: at least 16 characters, alphanumeric (dashes and underscores allowed), no spaces, no `@`. `run_sample.py` hashes a cleartext identifier for you; if you build the manifest yourself, hash it. |
| `QUARANTINE_MISSING_CHANNEL` | One of the two IDAT files is absent. | Both `--grn` and `--red` are required, and both must exist. A missing Red channel is not recoverable from the Grn. |
| `QUARANTINE_TRUNCATED_UPLOAD` | An IDAT is smaller than 1 MB. | The file transfer did not finish. Re-fetch it. The flag prints the size it found. |
| `QUARANTINE_ARRAY_TYPE_MISMATCH` | What you declared and what the file's header says are different array families. | Believe the header. Omit `--array-type` and the chain reads it from the file: a 450K array reports 622,399 addresses, an EPIC v1 about 1,051,943. Declaring the wrong one is the most common way to get nonsense out of an otherwise good sample. |
| `QUARANTINE_CORRUPT_IDAT` | The decoder reached the file and failed on it. The flag names the exception. | Re-fetch. Note that a file can pass the 1 MB size floor and still be truncated inside its gzip stream — one array in a 732-pair public download was exactly this. A size check is not an integrity check. |
| `integrity` in the hard-failure list, with `RE_TRANSMISSION_DETECTED` | The same two files, byte for byte, were already taken in against this custody log. | This is SOP §13 working: it is how a duplicate submission is caught. For a legitimate re-run, use a different `--intake-log`, or none. If you did not expect a duplicate, find out who submitted the first one. |
| `sex` in the hard-failure list | The array's chrX and chrY intensities do not agree with the sex you declared. | Check the declared sex first: this call is right on 729 of 731 arrays in a public healthy cohort, so it is usually the metadata that is wrong, not the array. If the donor's sex is genuinely unknown, the gate cannot pass — it has nothing to compare against. |
| `detection` or `call_rate` in the hard-failure list | Too many probes are indistinguishable from the array's own background. | This is a specimen or hybridisation problem, not a configuration one. On 731 healthy arrays the worst detection was 0.9951 against a 0.99 threshold, so a genuine failure here is a long way from normal. Look at the array, not the chain. |
| `hm450_coverage` in the hard-failure list | Fewer than 80 % of the reference CpGs survived calibration. | Usually the wrong array type, or a platform whose probes do not overlap the reference. Check the type first. |

**`PROCEED_WITH_PENALTY` is not a refusal.** It means a borderline flag was raised and the sample was scored
anyway. In the healthy cohort, all eight penalised arrays were borderline on bead count (between 0.990 and
0.995 of probes at three or more beads, against a 0.995 threshold). The flag is printed on the report; the
number stands.

### The thresholds, and what healthy looks like against them

Measured on 731 healthy whole-blood arrays (PROC-STAGE0-02), so you can tell an unusual sample from a
mis-configured one:

| gate | threshold | healthy median | healthy worst |
|---|---|---|---|
| detection p (0.5) | ≥ 0.99 of probes at p < 0.01 | 0.9994 | 0.9951 |
| call rate (0.7) | ≥ 0.98 | 0.9983 | 0.9889 |
| bead count (0.6) | ≥ 0.995 of probes at ≥ 3 beads | 0.9988 | 0.9900 |
| reference coverage (0.7b) | ≥ 0.80 | > 0.99 | — |
| sex (0.8) | log2(Y) − log2(X) < −2 → female | agrees with the published label on 729 of 731 | — |

If your array is far from these, it is the array. If it is at zero or one, it is the configuration.

---

## 2. It ran, but the chain will not give me a number

This is the `NOT REPORTABLE` family, and each one prints its own reason. They are refusals to *place* a
measurement, not failures to make it.

| reason printed | why | what to do |
|---|---|---|
| `no laboratory zero (Issue 003 s3.5; lab_zero.py) -> no placement, no tier` | Your laboratory has never been measured, so the chain does not know where its healthy centre sits. Between-laboratory offsets are as large as 0.046 in A — bigger than most effects anyone wants to see. | Commission the laboratory once: 40 healthy arrays of any age mix through the same Stage 1, then `lab_zero.py`. Panels under 40 are refused by design. Then pass `--lab-zero`. |
| `sky: no commissioned residual scale for this laboratory -> not rendered` | Same cause: the sky needs that laboratory's residual scale, built from the same 40-array panel. | Same fix. |
| `UNMAPPED` | No pipeline map was applied, so the β values are not on the scale the floors were calibrated on. | Pass `--pipeline` with a map name that exists in `beta_scale_maps_v1.json` (`stage1_noob_450K` is the default and is the right one for raw IDATs through this chain). Without a map, an absolute reading is meaningless and the chain says so rather than printing it. |
| `no band for this component yet` | That cell class has no measured healthy band, so it gets no placement and no tier word. | Nothing to fix. The class fraction and A are still printed. |
| `cellular age in years: not reported` | One array resolves age to about 50 years, which is not a number worth printing. | Nothing to fix. The age-matched healthy reference is what the chain uses instead, and that is the useful part. |
| `NOT ASSESSABLE · f=0.000 < 0.020` on a sky panel | That class is below its measured presence floor in this specimen — the chain cannot see enough of it to say anything. | Nothing to fix. Below its floor, a class is not there. |

---

## 3. It will not start at all

These are environment problems, not chain problems, and each has a one-line fix.

| symptom | cause | fix |
|---|---|---|
| Calibration hangs for minutes with no output, or fails on a manifest download | methylprep wants to download the array manifest into your home directory and cannot | Point `HOME` at a writable cache directory for the run: `HOME=/path/to/cache python3 run_sample.py …`. The first run downloads the manifest once; every later run reads it from there. Per-array calibration is about 26 seconds once the manifest is local. |
| `atlas not found: …IAMAtlasREBUILD.csv.xz` | The atlas is stored compressed | Nothing to do — the runner decompresses it once (605 MB) and says so. If you are short of disk, that is the file to plan for. |
| `Missing optional dependency 'pyarrow'` from the synthetic generator | The generator writes its cohort as parquet | `pip install pyarrow`. |
| A batch script dies with a process-pool error | Some sandboxes forbid process pools | Use threads. The published batch scripts do. |
| `ModuleNotFoundError: stage_1_idat_calibration` | The chain directory is not on the path | Run `run_sample.py` from its own directory, or let it resolve itself — it ascends to find the tree. If you moved files, run `build_chain_sequence.py`; it will tell you what is no longer reachable. |

---

## 4. It ran and the number looks wrong

Before suspecting the gauge, check the three layers in order. They are separate on purpose and a
reading is only absolute when all three are in place.

1. **Floor** — `H_min` per class, calibrated by MCMC on the reference scale. Never re-derived per pipeline.
   Reproduce it in fifteen seconds: [10.5281/zenodo.22905819](https://doi.org/10.5281/zenodo.22905819).
2. **Pipeline map** — one affine map per pipeline, because the same healthy blood reads β̄ = 0.737 on the
   reference scale and 0.815 through Stage 1 noob from raw IDATs. The offset is additive and large. A
   within-pipeline comparison cancels it and never sees it; an absolute reading does not.
3. **Laboratory zero** — measured, not modelled, on 40 healthy arrays. Four cohorts on one scale sit at
   0, +0.024, −0.021 and −0.046 in A.

A reading that skips layer 2 or 3 is not a small error. If your healthy controls do not sit near A″ = 1, one
of those two is missing, and the chain will have printed which.

**The check that catches this in one line:** run a handful of your own healthy samples. Their median A″
should land near 1.00. On the four commissioned cohorts it does — that is how the scale map and the zero were
verified in the first place.

---

## 5. The bisulfite gate says PROVISIONAL

`PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` means what it says: SOP §14 carries a 0.95 threshold for bisulfite
conversion efficiency, and every one of 731 healthy arrays in a public cohort sits below it (median 0.798,
worst 0.635). A gate that refuses every healthy specimen is not measuring specimen quality, so the chain
prints the measured value and does not refuse on it, and the decision gate records it as deferred rather than
passed.

Nothing for you to do. The threshold will be set from the healthy distribution across several cohorts
(PROC-STAGE0-04) and the flag `BS_THRESHOLD_CALIBRATED` flipped in the same commit. Until then, treat a low
bisulfite number as information about your array chemistry, not as a refusal.

---

## 6. Checking the chain itself

If you suspect the installation rather than the sample, these run in seconds and fail loudly.

| command | what it proves |
|---|---|
| `python3 chain/build_chain_sequence.py` | Prints the step order **derived from the code**, both interfaces, and anything documented as a chain step that nothing calls. If a document disagrees with this output, the document is wrong. |
| `python3 kit/link_check.py` | Every relative path in every live document resolves. A path in a document is a claim. |
| `python3 kit/release_check.py` | The guards, each with its result. A guard that has not run prints NOT RUN and is never shown as a pass. |
| `python3 kit/test_tiers.py`, `test_gauge_switch.py`, `test_patient_sky.py`, `test_lab_zero.py` | The tier boundaries, the reported gauge, the sky and the panel rule behave as sealed. |
| `shasum -c MANIFEST.sha256` | Your copy of the chain is intact (in the offline bundle). |

---

## 7. How the failures above were found, which is how you should look for yours

Four things cost hours, and all four have the same shape.

**A gate that cannot read its input never fires.** The header reader opened IDAT files raw, and public
downloads are gzipped — so the array-type gate silently never ran on any public data. It did not error. It
returned "header unreadable" and everything continued. If a check never reports a failure, test it with input
you know is bad.

**A value that fails to propagate looks like a value that is wrong.** Intake dropped `patient_id` between two
steps, and the next step read the absence as a cleartext identifier and quarantined *every* array. The message
blamed the data. The cause was two functions disagreeing about a field name. When a gate refuses 100 % of
anything, suspect the wiring before the specimens.

**A refusal that does not stop the run gets reported as something else.** When a quarantine was allowed to
continue, the next step overwrote the status, and an array-type mismatch surfaced two gates later as a
detection failure. Read the *first* refusal, not the last one.

**"Deferred" is the dangerous word.** A decode failure was caught and logged as a deferred check, which meant
a corrupt array would have been calibrated and scored with six gates unmeasured. Anything that turns a failure
into a silence is worse than no check at all. This is why the report prints NOT RUN in place of a pass, and why
deferred gates are listed by name on every reading.

**And the general one:** verify against rendered output, not source. Every one of these was found by reading
what the chain actually printed — the PDF page, the HTML tab, the log line — and not by reading the code that
was supposed to print it.

---

## Where to go next

- [`RUNBOOK.md`](RUNBOOK.md) — how to run a sample, end to end
- [`CHAIN_SEQUENCE.md`](CHAIN_SEQUENCE.md) — the step order, derived from the code
- [`../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md`](../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md) — the procedure each gate implements, with a link to its code under every step
- [`PROC_STAGE0_02_OUTCOME.md`](PROC_STAGE0_02_OUTCOME.md) — the 732-array run these numbers come from
- [`REVIEWER_MANIFEST.md`](REVIEWER_MANIFEST.md) — every file, with what is not published and why
