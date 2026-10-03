# Biological physics — the cell as an encoding surface

A cell holds its identity as a methylation pattern, one bit per CpG copy, written and copied at body temperature. IAM's Law prices each
maintained bit at `k_B T ln 2` (2.97 × 10⁻²¹ J at 310.15 K). The instrument here reads how well a cell is holding its own pattern: its
copy error against the same cell type when healthy, on one gauge where **A = 1 is healthy**, the thermal floor lies to the left and more
error lies to the right. Nothing about any other person enters a reading. The physics and every number are in Part 4 of the book
([`docs/book`](../docs/book)).

Prior art: Sanchez and Mackenzie (2016) showed that cytosine methylation obeys Landauer's bound and used it to filter thermal background.
Here the same bound is the ruler.

| folder | contents |
|---|---|
| [`MethylPhys/`](MethylPhys/) | **the instrument**, chain v3 (development build, neutrophils on EPIC v1 and on single molecules). Start at [`MethylPhys/README.md`](MethylPhys/README.md); operator procedure [`MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`](MethylPhys/sop/MethylPhys_CPG_SOP_v3.md) |
| [`MethylPhys/doors/`](MethylPhys/doors/) | one pre-registration and one outcome record per procedure, including those that failed |
| [`Salmonid/`](Salmonid/) | the fish work: copy error at water temperature (in development) |
| [`Record/`](Record/) | the April–June 2026 validation runs on the earlier class-floor method, kept as the record of how the work developed; not evidence for the current instrument |
| RETIRED_2026-09/ (archived privately), RETIRED_2026-10/ (archived privately) | earlier chain versions, reports, manuals and pages, kept unchanged with an index |

The instrument is not commissioned and is not a diagnostic test. It gives no medical advice.
