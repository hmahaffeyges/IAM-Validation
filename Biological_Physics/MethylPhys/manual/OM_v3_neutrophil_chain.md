# Operations Manual — chain v3, neutrophils (operator chapter)

**Build:** development v3 (2026-10-01), not commissioned. The procedure is in `sop/MethylPhys_CPG_SOP_v3.md`; this chapter covers running it and reading the output.
The PDF manual in this folder (`MethylPhys_CPG_Operations_Manual.pdf`) is built from this chapter, the generated chain sequence and the toolkit list by `build_manual_v3.py`. The class-floor engine (v2) and its manual were retired on 2026-10-03 and are archived privately.

## Before a run
1. Use EPIC v1 IDAT pairs (Grn and Red), with declared sex and age.
2. Put at least 3 healthy reference specimens on the same slide or batch, run the same way, so whole-blood readings can be tared.
3. Use Python 3.11 with methylprep 1.7.1. The repository includes the frozen files under `chain/Runtime Matrices/Met_A_Floors/`.

## Run
`python run_sample.py --grn <Grn> --red <Red> --engine v3 --specimen "whole blood" --array-type EPIC_v1 --sex <F|M> --age <years> --id <id> --out <id>.html`
Then run each whole-blood specimen again with `--slide-ref-A a1,a2,a3`, where those are the references' untared A values. The batch runner
`chain_tests/run_chain_acceptance.py` does both passes.

## Reading the report
| field | meaning |
|---|---|
| Stage 0 verdict | PROCEED / PROCEED_WITH_PENALTY / QUARANTINE (QUARANTINE produces no reading) |
| Composition | the 8 blood groups. Neutrophils must be ≥ 50 % for a reading |
| Met-A | isolated cells: the reading. Whole blood: "untared", a number only |
| A_rel | whole-blood reading after the tare. **Normal = 0.95–1.05** |
| C-score | genomic clustering of the departures (healthy ≈ 1). Development: no band yet |
| Withheld | what the build does not print, and why |

## Faults
| symptom | cause | action |
|---|---|---|
| QUARANTINE_INCOMPLETE_MANIFEST | sex or age not declared | supply `--sex` and `--age` |
| "not the dominant cell" | neutrophils < 50 % | none: the fraction is reported and A is withheld by rule |
| "untared: 0 same-slide reference arrays" | no references supplied | run the references, then pass them with `--slide-ref-A` |
| refusal "no frozen neutrophil floor for this platform" | the specimen is not EPIC | 450K floor pending |
