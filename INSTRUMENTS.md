# The IAM instrument catalogue

*Generated from CANON/catalogue.json; every status line comes from the Master Testing Manifest.*

## Why a self-reference matters

Until now every instrument needed a reference: healthy cells of the same kind, often from the same laboratory, to say what normal looks like. That reference is the hardest thing in this field to get, and it is why most tests work in one lab and stumble in the next. On 2026-10-15 we found that across 56 kinds of healthy human cells, a cell's tagging mistakes follow one straight line set by how loosely it holds its own DNA. So a healthy cell's expected reading can come from its own sample. An instrument that carries its own reference does not need a library of other people's cells, does not inherit another lab's quirks, and can be pointed at any tissue, any lab, even any species that tags its DNA. That is the step we are now commissioning.

## How they work together

Each instrument answers a different question about the same sample, so together they leave no blind side. The atlas says whose DNA it is. Met-A says how far a tissue's tagging level has drifted toward the ceiling. The C-Score says whether the pattern has started to clump. IAM-A says whether each mark is still held at its own healthy rate. Group-A says whether a whole group has moved, for studies rather than single patients. A tumor that hides from one reading still has to answer to the others.

## IAM-Atlas v2.1

**The question:** Whose DNA is this? Which cells are in a sample, and in what shares?

A map of 79 kinds of healthy human cells, built from seven public sources on arrays and sequencing, fitted as one model so every spot on the genome carries its value and its uncertainty. Like a state estimator across the whole interconnection: many utilities' meters, each weighted by how far it can be trusted.

**Reads:** arrays or sequencing. **Where it stands:** running now.

- ATL-005 Atlas v2.1 joint fit: running now
- ATL-060 Head to head: atlas v2.1 against the source atlases' own references, on their own job: sealed, waiting for data
- ATL-001 Atlas commissioning as a source of healthy references: designed; last run 2026-10-12: progressed

**Next:** The fit finishes; then its sealed commissioning bars and the sealed tests on Loyfer's and Moss's own jobs are scored.

## Met-A

**The question:** How far has a tissue's tagging level drifted toward the ceiling?

Reads the methylation level at sites a healthy cell holds firmly, against a healthy reference, on cheap and plentiful arrays. Commissioned on isolated neutrophils; whole blood is in development.

**Reads:** arrays. **Where it stands:** commissioned (isolated neutrophils).

- COM-METAA-003 Met-A commissioning on isolated neutrophils, EPIC v1: run; last run 2026-10-11: progressed
- COM-METAA-039 Met-A read against the sample's own line: run; last run 2026-10-15: not met
- COM-METAA-032 Whole blood back into Met-A scope: designed

**Next:** Test whether Met-A can read against the sample's own loose share, which would shrink or remove its same-lab calibration and open it to new cell types.

## C-Score (the clustering meter)

**The question:** Has the pattern started to clump?

Cancer does not only shift tagging levels; it clumps them into runs. The C-Score measures that clumping against a healthy band. The band held in two laboratories that never helped set it.

**Reads:** arrays (whole-genome sequencing too). **Where it stands:** passing its tests.

- COM-CSCORE-011 C-score healthy band test: run; last run 2026-10-13: met
- COM-CSCORE-012 C-score last check: same-person repeatability on unspent replicate arrays: sealed, waiting for data; last run 2026-10-13: not met

**Next:** One check left: the same person measured twice on unused public arrays. Searching for that data.

## IAM-A

**The question:** Is each mark still held at its own healthy rate?

Counts tagging mistakes one DNA molecule at a time, the most direct reading of informational fidelity we have, and reads them against the sample's own line. The physics end of the catalogue: the floor and ceiling are derived, and since 2026-10-15 the healthy middle comes from the sample itself.

**Reads:** single-molecule sequencing (bisulfite, EM-seq; nanopore next). **Where it stands:** running now.

- PHY-045 The loose end read off the line: every healthy cell's held error from its own loose share: run; last run 2026-10-15: met
- PHY-048 The loose-end line at the one-Landauer-bit cut: run; last run 2026-10-15: met
- PHY-049 The loose share from each sample's own molecules: run; last run 2026-10-15: progressed
- PHY-043 Tumor backbone against the writer's nameplate: running now
- COM-IAMA-052 IAM-A atlas-free commissioning: one grand sealed test on fresh data: planned

**Next:** The tumor pairs read on both sides of the DNA, a second laboratory, then one grand sealed test on fresh data to commission it.

## Group-A

**The question:** Has a whole group moved (a trial arm, an exposure, a cohort)?

Single blood readings are too noisy for one person on arrays, but a group average is not. Our honesty test showed two healthy groups from two labs can differ as much as a tumor would, so Group-A compares groups only when they were run in the same series.

**Reads:** arrays. **Where it stands:** in development.

- GRP-004 DEV-GROUP-01 step 2: group comparison, one cancer at a time, patients and healthy controls in the same series: planned

**Next:** One cancer at a time, patients and healthy controls from the same series.

## Candidate instruments

Readings still on the drawing board, each with its blind spot and next step: [Biological_Physics/MethylPhys/INSTRUMENTS.md](Biological_Physics/MethylPhys/INSTRUMENTS.md).

