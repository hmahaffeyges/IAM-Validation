# What is in this repository, and what should not be

**Measured 2026-09-25** from `git ls-files`, not estimated. The author's concern: *"its kind of a mess right
now … way too easy to lose control of a repo and the chain when there is so much complexity."* He is right,
and the numbers say where.

## Where the weight is

| area | files | MB | what it is | keep? |
|---|---|---|---|---|
| `Biological_Physics/MethylPhys` | 360 | **726** | the live chain, atlas, manual, SOP, procedures | **yes** — this is the instrument |
| `Biological_Physics/RETIRED_2026-09` | 1,121 | **377** | the pre-build card folders and the July tree, retired wholesale | **no** — see below |
| `camb_validation/chains` | 28 | **377** | cosmology MCMC chains (CAMB) | move off |
| `Biological_Physics/Record` | 1,037 | **359** | the pre-atlas validation record, VAL evidence, procedure data | **yes, thinned** |
| `mgcamb_validation/chains` | 89 | **137** | cosmology MCMC chains (MGCAMB) | move off |
| `docs/papers` | 50 | 40 | the paper library | yes |
| `tests/pantheon_plus.zip` | 1 | 32 | supernova dataset | move off |

Inside the chain itself, the weight is data rather than code:

| item | MB | note |
|---|---|---|
| `atlas/iamatlas_class_archives` (8 files) | 165 | the per-class MCMC archives — **needed for enhancement A1**, so keep until that is done |
| `chain/TEST_DATA` (35 files) | 103 | the eleven-array test package, the bar PROC-E2E-01 is scored against |
| `atlas/IAMAtlasREBUILD.csv.xz` | 101 | the atlas; irreducible |
| `chain/example_runs` | 46 | run reports — grows with every run, see below |
| `atlas/external_manifests` | 44 | Illumina manifests; the vendor host is unreachable from the sandbox, so these must stay |
| `kit/CPG_Issue003_ReproductionKit.zip` | 44 | **a zip of files that are already in the tree** |
| `reference_data/stage1_betas_*.pkl.xz` (4) | 117 | the four laboratories' healthy betas — the calibration inputs; keep |

## Recommendations, in the order I would do them

**1. Retire `RETIRED_2026-09` out of the repository entirely — 1,121 files, 377 MB.**
It is the single largest cleanup and the lowest risk: nothing live reads it, and the link checker already
excludes it because its internal links have been broken since before the current tree. Two honest options:
publish it as one Zenodo archive with a DOI and leave a one-line pointer, or keep it as a git tag and delete
the working copy. Either way the history is preserved and a reviewer cloning the repository stops downloading
a third of a gigabyte of superseded material.

**2. Move the cosmology MCMC chains off — 513 MB across `camb_validation/` and `mgcamb_validation/`.**
Chains are exactly what a data archive is for, and they are already described by the papers that used them.
The scripts and the convergence summaries stay in the repository; the samples go to Zenodo, beside the
H_min calibration deposit that is already there.

**3. Delete `CPG_Issue003_ReproductionKit.zip` — 44 MB.**
It is a zip of files the repository already contains. A reproduction kit that duplicates the tree is a second
copy to keep in sync, which is the exact failure mode you are worried about. If a single downloadable bundle
is wanted, generate it on demand — the delta bundles already work this way.

**4. Rename `chain/Disease Cards : Residual Maps`.**
A colon in a directory name is not portable — it breaks on Windows checkouts and inside URLs. Also
`chain/README's` (an apostrophe) and the directories with spaces, which already forced percent-encoding into
every generated link.

**5. Thin `Record/` rather than move it.**
1,037 files is a lot, but this is the validation record and the rule is that when the tree and the record
disagree the record wins. What can go: the `OLD/` sub-folders of superseded evidence reports, and per-VAL
intermediates that the sealed outcome documents already quote.

**6. Two folders that are neither chain nor record.**
`chain/Crown Jewel and Patient Strawman` (2.7 MB) and `chain/Disease Cards : Residual Maps` (7 MB) are
presentation material from the pre-atlas era, sitting inside the live chain directory where a reader will
take them for chain components. They belong under `Record/` or in the retired archive.

## What the repository should be, stated positively

Four things, and nothing else at the top of the chain directory:

1. **The chain** — the modules a run executes, their runtime matrices, and the test package that proves it.
2. **The two canonical documents** — the Chain of Custody SOP and the [Issue 003](../manual/IAMPerformance_GAPEIssue003_RC1.pdf) operator manual. Every other
   document in `doors/` is either generated from the tree or is a procedure's pre-registration and outcome.
3. **The record** — procedures, their evidence, and the runs.
4. **The paper library.**

Everything else is either derived (and should be generated), superseded (and should be archived with a DOI),
or large data (and belongs in a data archive with a citation).

[`propagate.py`](../chain/propagate.py) already enforces the second point: a document that drifts from the tree fails the gate. The
cleanups above are what make the first, third and fourth honest.
