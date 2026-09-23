# PROC-E2E-01 — the commissioned chain, end to end, against the test package's own documented outputs

**Pre-registered 2026-09-23, before any array was run.** The test package
([`TEST_DATA_MANIFEST.md`](../chain/TEST_DATA/TEST_DATA_MANIFEST.md)) documents what each of its arrays
produced in June 2026, which makes it a bar rather than a demonstration: the chain has since gained Stage 0
intake, the identity gauge, the pipeline map, the laboratory zero and a rebuilt report, and the question is
whether it still reproduces what the record says it produced.

## What is being tested

Nine IDAT pairs, four cohorts, two platforms, three substrates — whole blood (healthy and rheumatoid
arthritis), colon adenoma tissue, and colorectal carcinoma tissue at two stages. Every array runs through
[`run_sample.py`](../chain/MethylPhys_Interface/run_sample.py): the ten intake steps, Stage 1 calibration from the raw IDAT pair, the eleven conductor
stages, and the report.

## Metadata correction, recorded before the run

The test manifest describes the three Uppsala arrays as "healthy 58M", "healthy 67F" and "healthy 43M". GEO's
own sample characteristics say **GSM2333901 = 72 M, GSM2333905 = 74 M, GSM2333950 = 81 M**. The published
metadata is authoritative, so this run uses it — which means any age-corrected number from the June run was
corrected against the wrong age, and a difference in this run is expected for that reason before any other.
GSE166212 publishes neither age nor sex for its two arrays.

## The bars

| bar | what must happen |
|---|---|
| **B1 intake behaves** | every array either clears intake or is refused with a cause consistent with its own metadata. The two GSE166212 arrays publish no age, so `QUARANTINE_INCOMPLETE_MANIFEST` is the *expected* result; they are then run with `--no-intake` to exercise the measurement path, and that is recorded on their reports. |
| **B2 calibration is stable** | for every array in `betas_cache.pkl`, Stage 1 from the raw IDATs reproduces the cached betas to within 1e-9 on the shared loci. |
| **B3 composition reproduces** | each documented class fraction reproduces within **±3 percentage points**. Deconvolution is what those numbers measure, and it is the part of the chain that has not changed surface. |
| **B4 the substrate claims reproduce** | secretory below 2 % in every whole-blood array; above 8 % in every tissue array; secretory *and* cycling both higher in the stage-4 carcinoma than in the stage-1; both EPIC arrays complete the chain. |
| **B5 the adjudicator fix holds** | GSM2333950 reads inside the band (the false d = 42.9 does not return) and GSM2333905 keeps a genuine stem_adult elevation. |
| **B6 the report renders** | 18 tabs on every array, the sky plate present, the Stage 0 block present and naming its decision, every refusal printed with its reason, zero condition-name or build-history vocabulary hits, and no dead internal reference. |

**B7 is not a bar, it is a measurement.** The documented A-scores come from the pre-atlas marker-union gauge;
the chain now reports the identity gauge, and the two are known to move in opposite directions with age
(RECON D2). Every A is recorded beside its documented value and any difference is reported as a surface
difference with both numbers stated. **No A-score comparison can fail this procedure, and none may be used to
claim agreement either.**

## Decision rule, fixed now

All six bars met → the chain is confirmed end to end on the documented package and the run is published as the
commissioning evidence a reviewer can repeat. Any bar failed → the failure is named with the array that failed
it, the outcome says so in its first paragraph, and nothing is re-run with a changed threshold. No tolerance in
this document moves after results are visible.
