> **[RECORD] — superseded draft (2026-10-01).** Kept as written. Classes are now only the floor a cell is divided by (CLASS_HISTORY.md); current names and constants are in CANON/GLOSSARY.md.

# Architecture-class assignment — a written rule, tested against the existing classes (DRAFT, 2026-09-28)

**Status: DRAFT for the author's review. Not in the chain. No A is read on any new cell until this is accepted.**

Author, 2026-09-28: *"We have to have a definition and we have to have strict agreed upon methodology for classification that
matches exactly the other cells in the other classes and how they were determined … then we use the same procedure. That is the
only defensible method of choosing, not my opinion."*


## The founding definition (the methylation report Day-2 session, 2026-04-06; author's transcript `MPHYS_Day2AIChat.txt`, lines 73, 265, 293, 1315)
A class was defined as a **regime**, not a list: *"The question isn't how many cell types, it's how many distinct inversion
regimes exist"* (l. 293); cell types that *"share the same dominant regulatory mechanism and therefore the same n_bio"* are one
class (l. 73) — the the methylation report equivalent of the quantum-processor report/the semiconductor report's architecture or ISA class, where every architecture has one dominant error
source and its own Dennard-type transition (l. 159–171, 265). So the defining property of a class is its **dominant failure
mode (inversion)**, with its own n_bio and floor:

| class | dominant inversion (Issue 002, [`mphys002_lib.py`](../manual/mphys002_lib.py)) |
|---|---|
| stem_pluri | Differentiation Dose Inversion |
| stem_adult | Niche Depletion |
| progenitor | Replication Throughput Ceiling |
| cycling | Replication Ceiling |
| immune | Cytokine Saturation |
| secretory | Secretory Overload |
| stromal | Stiffness Coupling |
| terminal | Oxidative Stress Inversion |

**What that makes R1–R8:** the questions below are the operational test for *which inversion governs a cell* — potency (the
differentiation / niche / throughput regimes), haematopoietic signalling (cytokine), mechanical-matrix function (stiffness),
post-mitotic OxPhos load (oxidative stress), product secretion (overload), continuous replication (replication ceiling). A cell
is assigned by its governing inversion; R1–R8 are how that is decided without opinion. Where no question fits cleanly the cell
is flagged for review of its facts or the rule.

**The number of classes is eight, by definition.** The Day-2 session opened with ten (senescent and cancer included) and guessed at 12–18; senescent and cancer were then recognised as **states** — defined by having crossed an inversion (l. 872) — not regimes. The MCMC (G-002) was *given* eight classes and fitted their floors; no run in the record compares class counts ([CLASS_HISTORY.md](CLASS_HISTORY.md) §4, corrected 2026-09-28). A new cell is assigned to one of the eight; a different count would come only from the split-and-merge test in PLAN, never from a quiet reassignment.

## What the record held
The classes were assigned by **example lists**, not a criterion: G-002's 37 reference cells
([`reference_cells_37.csv`](../hmin_calibration/reference_cells_37.csv)), each class's `what_includes` in Issue 002
(`manual/mphys002_lib.py`), and the v1 map ([`IAMAtlasREBUILD_celltype_to_class.json`](../atlas/IAMAtlasREBUILD_celltype_to_class.json)).

## The rule those lists imply (biology only; no methylation data enters)
Facts per cell: lineage, potency, whether it divides under normal adult conditions, whether its defining function is a secreted
product. First match wins:

| | question | class |
|---|---|---|
| R1 | pluripotent? | stem_pluri |
| R2 | self-renewing multipotent tissue stem? | stem_adult |
| R3 | committed, dividing precursor? | progenitor |
| R4 | haematopoietic lineage (mature)? | immune |
| R5 | mesenchymal: connective, vascular, smooth muscle, adipose, pericyte? | stromal |
| R6 | does not divide under normal adult conditions? | terminal |
| R7 | defining function a secreted product (enzyme, hormone, bile, milk, pigment)? | secretory |
| R8 | otherwise — dividing lining epithelium | cycling |

Code: [`atlas/class_assignment/classify.py`](../atlas/class_assignment/classify.py); facts:
[`cell_facts_v0.csv`](../atlas/class_assignment/cell_facts_v0.csv) (a literature citation is still **to attach** on every row);
full result per label: [`class_test_report.csv`](../atlas/class_assignment/class_test_report.csv).

## Test against the existing records
| record | result |
|---|---|
| G-002, 37 reference cells (the cells the floors were fitted on) | **37 of 37 reproduced** |
| Issue 002 `what_includes` (25 items) | 22 reproduced, 3 conflicts |
| v1 map (115 labels) | 94 reproduced, **18 are mixtures or tissues, not cells**, 3 conflicts |

**Conflicts — for the author, not changed silently:**
1. *Thyroid*: Issue 002 lists it under both cycling and secretory; the rule says secretory (hormone), as v1 does.
2. *Skin (melanoma)* in Issue 002's cycling list: a cancer, not a cell; the cell (melanocyte) is secretory in v1 and by the rule.
3. *Cholangiocytes*: Issue 002 lists them under adult stem; v1 and the rule say secretory.
4. *v1 `Ductal`* (cycling) vs *v1 `Pancreatic_duct_cells`* (secretory): the same cell in two classes. The rule says secretory.
5. *v1 `megakaryocyte`, `nRBC`* (progenitor): both are the ends of maturation and no longer divide; the rule says immune.

**Mixtures carried as v1 "cells":** `whole_blood`, `PBMC`, `Leu`, `IC`, `Lym`, `Mye`, `Gran`, `granulocytes`, `Epi`, `Gland`,
`Breast`, `Prostate`, `Kidney`, `Lung_cells`, `Upper_GI`, `Uterus_cervix`, `Left_atrium`, `stromal_other`. In a cell atlas a
mixture competes with its own components; v2 carries them to the tissue atlas or flags them as legacy.

## New cells under the same rule
astrocytes → terminal (**borderline**: divides after injury; consistent with v1 `Glia` = terminal) · microglia → immune ·
oligodendrocytes → terminal · OPC → progenitor · pericytes, VLMC, brain and 7-bed endothelium, fibroblasts, smooth muscle,
osteoblasts, pleural mesothelium → stromal · podocytes, striated muscle → terminal · islet alpha, delta, enteroendocrine,
thyroid, prostate, gallbladder → secretory · kidney tubular, female reproductive, oral/oesophageal epithelia → cycling ·
tissue macrophages, basophils, eosinophils, Treg, naive/memory subsets → immune · H1/H9/HUES64 → stem_pluri.
Loyfer's "kidney glomerular epithelium" was mapped to podocyte — to confirm against its sorting markers.

## Third audit, blocked: the G-002 posterior check
G-002's likelihood is ((H(β̄)/H_min(class) − 1)/0.02)² per reference cell, so a new cell's fit to each class could be read from its
β̄ against the eight fitted floors. It discriminates only where floors differ — terminal (0.773) and pluripotent (0.982) — the
other six lie within ~±2 %. **Blocked:** the 37 β̄ values come from `MPHYS_WEB_v4.py`'s published database, "cited to primary
sources", with no stated locus set or statistic (H1 ESC is listed at 0.42, below H1's genome-wide CpG methylation), so a matching
β̄ cannot yet be computed for a new cell. Tracing each of the 37 values to its paper, figure and region set is an open item — it is
also the provenance a reviewer will ask of the frozen floors themselves.

## Before this leaves DRAFT
1. The author accepts or amends R1–R8 and the fact values.
2. A citation is attached to every fact row.
3. Each of the five conflicts is decided and recorded.
4. The two data audits run on every new cell (identity-locus yield in its class band; nearest existing cells): flags only.
