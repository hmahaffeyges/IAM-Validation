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

## Two corrections to my first pass (measured 2026-09-25)

**The cosmology chains stay.** The author's instruction: they are there for any cosmologist to pull. Removed
from the recommendations below.

**I was wrong about `RETIRED_2026-09`.** I recommended retiring it out of the repository. Measured: of its
1,121 files, 493 are VAL-named and **63 of those have no same-named file anywhere live** - including
`val_111_prereg.md`, `val_111_outcome.md`, `VAL-112_113_outcome.md` and the VAL-119 to VAL-122 bladder
artifacts. Those are pre-registrations and outcomes for early VALs that exist nowhere else. The rule is that
when the tree and the record disagree the record wins, and this folder *is* record. **Keep it.** What it
needs is an index, not an exit.

**And the duplication is not where I implied.** Byte-identical duplicates inside MethylPhys: **3 groups,
4.3 MB** - a figure shared with a plate, and two pairs of test-run plates. That is all. The real mess is
folder shape and same-name-different-content, listed below.

## Recommendations, in the order I would do them

**1. Index `RETIRED_2026-09` rather than removing it.** It holds 63 VAL files that exist nowhere else. One
generated index at its root - which VAL, what the file is, which live document supersedes it if any - turns
1,121 files from a heap into an archive. Nothing moves.

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

## The MethylPhys folders, which is what you actually asked about

67 directories. What is wrong with them, measured:

| problem | what | fix |
|---|---|---|
| **six `__pycache__` directories are committed** | `MethylPhys_Interface/`, four under `Runtime Matrices/`, one more | delete and add to `.gitignore`; nothing should ever have tracked them |
| **a colon in a directory name** | `chain/Disease Cards : Residual Maps` (20 files, 7 MB, nested four deep) | rename - a colon breaks Windows checkouts and URLs |
| **an apostrophe in a directory name** | `chain/README's` (2 files) | rename to `chain/readme_archive/`, or fold both files into `chain/README.md` |
| **presentation material inside the live chain** | `chain/Crown Jewel and Patient Strawman` (2.7 MB), `chain/Disease Cards : Residual Maps` | move under `Record/` - a reader takes anything in `chain/` for a chain component |
| **two files with the same name and different content** | `VAL_INDEX.csv` (19.5 KB in `kit/`, 52.9 KB in `kit/results/`) and `switching_order.py` (21.5 KB in `kit/`, 26.4 KB in `manual/`) | resolve each: one is current, the other is either a subset or stale |
| **a zip of the tree** | `kit/CPG_Issue003_ReproductionKit.zip`, 44 MB | delete; generate bundles on demand |
| **nine `README.md` files** | one per folder | this one is *fine* - each describes its own directory, which is the pattern that works |

That is the whole list. The folder count is high but most of it is `Runtime Matrices/` holding one
subdirectory per runtime file, which is legible; the actual defects are the seven rows above.

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
