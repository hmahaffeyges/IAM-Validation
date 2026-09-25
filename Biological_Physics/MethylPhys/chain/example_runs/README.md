# Runs, and what every kind of report in this repository is

One folder per run. Each folder is `RUN-YYYYMMDD-NN/` and holds that run's report plus a `RUN.md` naming the
specimen, the cohort and what the chain reported.

## The three things that get a number, and why they are different

| prefix | what it is | makes a claim? | where |
|---|---|---|---|
| `VAL-###` | the pre-atlas validation record | yes, historically | [`Record/`](../../../Record/) - **frozen**; nothing new joins it |
| `PROC-XXX-##` | a **test of the instrument**: a question, bars fixed in a pre-registration *before* any data is read, and a sealed outcome that scores every bar | yes | [`doors/`](../../doors/) |
| `RUN-YYYYMMDD-NN` | **one execution on one specimen** | **no** | here |

A run is evidence, not a test. It cannot pass or fail, because nothing was pre-registered about it - it is
the chain doing its job on one array. Filing a run as a VAL would imply a claim nobody made.

## A cohort test, when we get there

A test across a cohort is a **procedure**, not a pile of runs: it gets a `PROC-` identifier, a
pre-registration with its bar fixed before the first array is read, and an outcome document that scores that
bar. The runs it produces live here and are cited by the outcome. So the answer to "what will the cohort
tests be called" is: `PROC-<COHORT>-##`, with the runs underneath it.

## What the report itself contains

18 tabs. Seven carry **this specimen** - Reading, Every cell, Departure, Sky, Integrity, Safeguards, Run.
Nine are **reference material** identical in every report and say so at the top. Two differ only in a
provenance line. The Sky tab's plate is the only image drawn from this specimen's own data, and it is
captioned `THIS SPECIMEN`.

Two tabs are worth knowing about before you read a result:

- **Red flags** - every refusal, withheld number, deferred intake check and tripped guard in one place,
  ordered STOP / WITHHELD / CAUTION / NOTE, with the same list as JSON for a program to read.
- **Safeguards** - includes the register of every method borrowed from CMB analysis with its state on this
  run: PASS, FAIL, NOT_RUN, NOT_APPLICABLE or NOT_BUILT. A FAIL there is also a red flag.

The **Run** tab states whether every canonical document was current when the report was built, from
`chain/propagate.py`'s own status file.

## Regenerating the index

```
python3 chain/build_run_index.py
```

`RUN_INDEX.csv` is generated from every evidence ledger in the tree - never typed.
