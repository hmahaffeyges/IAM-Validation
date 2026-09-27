# doors/ — the record of how the chain was built, and what is next

This folder holds the procedures (one pre-registration and one outcome each), the findings, the register, the ledger, the plan and
the runbook. **New to the project? Read [`../START_HERE.md`](../START_HERE.md) first**; it says which document is canonical for what.

**The claim, in one paragraph.** A cell maintains its methylation pattern by irreversible information writing, and at body
temperature that writing has a thermodynamic cost. For each of eight cellular architecture classes there is a floor entropy
`H_min` below which a cell of that class does not keep its identity; it was fitted once by MCMC on 37 published reference cell
methylomes (April 2026) and is frozen. A cell's reading is `A = H(mean β over its identity loci) / H_min`. Healthy is A = 1.00 by
the physics; the tier scale is the tolerance. Nothing about any other person enters a reading.

| | file |
|---|---|
| **What is next** | [`PLAN.md`](PLAN.md) — one line per item, done items come off |
| **The ledger** | [`ENHANCEMENTS.md`](ENHANCEMENTS.md) — what was found, decided, closed and why |
| **The register** | [`CHAIN_COMMISSIONING.md`](CHAIN_COMMISSIONING.md) — every row of the chain and the procedure that commissioned, sealed or removed it |
| **Run one specimen** | [`RUNBOOK.md`](RUNBOOK.md) |
| **The step order, as the code calls it** | [`CHAIN_SEQUENCE.md`](CHAIN_SEQUENCE.md) (generated) |
| **Where every file lives and what reads it** | [`COMPONENT_MAP.md`](COMPONENT_MAP.md), [`REPO_INVENTORY.md`](REPO_INVENTORY.md) (generated) |
| **What each report tab says and where it comes from** | [`REPORT_TAB_REFERENCE.md`](REPORT_TAB_REFERENCE.md) (generated from a fresh render on every build) |
| **For a reviewer** | [`REVIEWER_MANIFEST.md`](REVIEWER_MANIFEST.md) (generated) |
| **Procedures** | `PROC_*_PREREG.md` (written before data is read; never moves after) and `PROC_*_OUTCOME.md` (written after the render is read) |
| **Findings** | `FINDING_*.md` — e.g. [`FINDING_GSE125105_LOW_SIGNAL.md`](FINDING_GSE125105_LOW_SIGNAL.md) |
| **The translation map** | [`CMB_TO_METHYLOME_MAP.md`](CMB_TO_METHYLOME_MAP.md) — the CMB-analysis modules mapped to their methylome analogs, with what was built and what was reversed |

**Why a sky.** The atlas is a reference map with a per-pixel uncertainty (posterior mean and sd at every CpG); a specimen is one
observation whose residual against what its own composition predicts is read pixel by pixel. That is the Planck workflow, and it
is why the CMB toolkit — HEALPix, Mollweide, matched filters, residual maps — transfers.

*Research and development stage. Nothing here is clinical validation.*

## Planning and housekeeping

- [`ENHANCEMENTS.md`](ENHANCEMENTS.md) - everything that would make the chain more sensitive, ranked by impact on the mission, with effort and blockers. Two lists: chain enhancements and the cosmology methods still on the shelf.
- [`REPO_INVENTORY.md`](REPO_INVENTORY.md) - what is in this repository, measured, and what should not be.

## Part II

[`PART_II_CHAPTER_NOTES.md`](../manual/PART_II_CHAPTER_NOTES.md) is the outline for Part II - the chapters, including the biological write-head, that are written after the instrument is commissioned. It had no reader in the tree until 2026-09-25.
