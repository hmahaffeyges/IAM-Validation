# PROC-BRAIN-01 — outcome: brain-derived cells ARE found in cerebrospinal fluid, in every patient. Whether they can be SCORED is not assessable on the current surface.

**Sealed 2026-09-26** against the bars fixed in [`PROC_BRAIN_01_PREREG.md`](PROC_BRAIN_01_PREREG.md) and
amended — before any array was scored — to record that the CSF arm is 24 specimens rather than 181.
**This is the second run.** The first is recorded as VOID in the pre-registration, for operator reasons set
out there; nothing from it is quoted here. This run was one process, a fresh output path, no edit to the
chain while it ran, and every specimen through `run_sample.py`.
Evidence: [`PROC_BRAIN_01.json`](../kit/results/PROC_BRAIN_01.json) ·
[`PROC_BRAIN_01_scored.json`](../kit/results/PROC_BRAIN_01_scored.json) ·
script [`PROC_BRAIN_01.py`](../kit/PROC_BRAIN_01.py).

GSE292312, 24 CSF cfDNA specimens from paediatric CNS tumour patients, EPIC, 370,346 CpGs submitted of
EPIC's ~865,000, pipeline map `geo_author_processed_EPIC`.

| bar | result |
|---|---|
| **B1** terminal is found — median fraction > 0.0312 | **MET.** Median **0.3012**, range 0.0802–0.6290 — about **ten times** the gate |
| **B2** found in most patients — ≥ 12 of 24 | **MET. 24 of 24.** |
| **B3** the author's April rule — median terminal A > 1.05 | **NOT ASSESSABLE** — see below |
| **B4** not the absence artefact | **MET.** No specimen fell below the gate, so no absent-class reading could contaminate B3 |
| **B5** specific to terminal, not everything rising together | **MET**, with a caveat below. Terminal 0.1765 against a median of 0.0820 across the other non-haematopoietic classes |
| **B6** the instrument has not moved | **MET.** Max abs delta A_mapped = 0.00e+00 on PROC-BAND-01's published 318 arrays |

## The presence result is decisive, and the threshold was not chosen for it

The gate — 0.0312 — is the **maximum** terminal fraction across 845 EPIC-Italy whole bloods with no CNS
disease, where it gives **0 of 845** false positives. It was measured before this cohort was selected. Every
one of 24 CSF specimens clears it, with a median ten times higher.

That is what the author's mechanism predicts: the barrier is why the blood null is empty, and shedding into
the compartment the barrier encloses is why presence there means something. **It needs only the
deconvolution**, which PROC-SYNTH-01 verified recovers a known composition with 0.0000 error.

## Why B3 cannot be read, and why the bar was not rewritten

The bar names the terminal **A**. On the morning of this run, the 115-cell reference audit established that
the per-cell A is computed on each cell's **discriminative marker panel**, and those panels are **77–100 %
near-binary**. Mean per-CpG entropy over binary addresses is ≈ 0 by construction, so every cell's *own atlas
reference* reads far below 1.0 — `Cortical_neurons` at **0.0099**. Scored instead on identity loci, the same
references read **0.89–1.14, median 0.9876**, which is what an absolute floor is supposed to give for a
healthy reference. The repository's own Jensen-bound rule already disqualifies marker panels for this
computation; the per-cell path was simply never held to it.

So the quantity B3 names is not currently measured on a defensible surface. **The bar is left exactly as
pre-registered** — not loosened, not moved, not reinterpreted — and recorded as not assessable. It becomes
assessable when per-cell identity loci exist, and that construction is the item this procedure hands forward.

## The caveat on B5, which the numbers demand

B5 passes on its stated criterion. But the one class that individually exceeds terminal is **`stem_pluri` at
0.2295**, and PROC-SYNTH-01 showed that a pure `stem_pluri` specimen is **misidentified as
`Cortical_neurons`** — a terminal cell — at r = +0.817. So the honest reading is that **brain-derived
material is found, split between `terminal` and `stem_pluri` by a known confusion**, rather than that
terminal specifically is elevated. That does not weaken B1 or B2, which are about finding the material at
all.

## What this licenses, and what it does not

**Licensed:** the instrument finds brain-derived cells in a liquid specimen, in every patient tested, at a
threshold calibrated to give no false positives in 845 unrelated bloods. That was the stated purpose of this
step, and it licenses the plasma procedure the author designed in April.

**Not licensed, and stated here so no later document can imply otherwise:**

- **CSF is inside the barrier.** This does not test the barrier-breakdown claim. That needs plasma, and it
  was written into the pre-registration before any array was read.
- **No fidelity claim.** B3 is unassessable, so nothing here says anything about how well those cells are
  holding their state.
- **No control specimens of either substrate exist in this series**, so the comparator is a whole-blood null
  and the substrate difference is a limitation of that comparison.
- **The 157 primary tumour tissue arrays were NOT RUN.** At full locus coverage each specimen costs about
  five minutes, so that arm is roughly sixteen hours and belongs to a separate run. Recorded as not done
  rather than partially done.
