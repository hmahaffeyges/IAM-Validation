# How the eight architecture classes came to be — the record

Written 2026-09-28 from the author's the methylation report Day-2 session transcript (6 April 2026, `MPHYS_Day2AIChat.txt`, line numbers below),
the G-002 sampler (`hmin_calibration/mphys_mcmc_g002.py`) and its deposit (10.5281/zenodo.22905819). This is the answer to
"why eight?" — stated as the record has it, including what the record does not show.

## 1. A class is a failure regime, borrowed from the quantum-processor report and the semiconductor report
The classes came into the methylation report from the two engines built before it. In the quantum-processor report (quantum processors) and the semiconductor report (semiconductor chips)
an *architecture class* is a family that shares one dominant error source and so one Dennard-type transition. the methylation report took the same
definition for cells: *"The question isn't how many cell types, it's how many distinct inversion regimes exist"* (l.293). Cells
that share the same dominant regulatory mechanism — and therefore the same failure mode — are one class (l.73).

| class | governing inversion (`manual/mphys002_lib.py`) |
|---|---|
| stem_pluri | Differentiation Dose Inversion |
| stem_adult | Niche Depletion |
| progenitor | Replication Throughput Ceiling |
| cycling | Replication Ceiling |
| immune | Cytokine Saturation |
| secretory | Secretory Overload |
| stromal | Stiffness Coupling |
| terminal | Oxidative Stress Inversion |

## 2. The count was reasoned, not fitted: ten, then eight
- The engine entering Day 2 held **ten** classes and 22 reference cells (l.53): the eight above plus **senescent** and **cancer**.
- At l.293 the count stood at "8 distinct inversions across 10 classes" (cycling and progenitor then shared the Replication
  Ceiling; senescent carried all of them), with a forecast of 12–18 classes once more data came in. That forecast was never tested.
- **Ten became eight at l.872:** *"We need 8 validated n_bio values — one per non-pathological class. Senescent and cancer don't need
  n_bio because those classes are defined by having crossed the inversion threshold."* Senescence and cancer are **states a cell
  reaches**, not regimes a cell lives in. Later in the session: the cancer *"is a departure from the class, not a class itself"* (l.2146).

## 3. What the MCMC did: it fitted the eight floors
G-002 (proposed l.917, run l.931–935) was given the eight classes and asked one question: *which H_min per class makes the
reference cells of that class read A = 1.00?*
- Data: 37 reference cells with a defined class, each with a published mean β. Parameters: 8 (one H_min per class).
- Likelihood: Σ ((H(β̄_cell)/H_min(class) − 1)/0.02)². Prior: uniform on [0.60, 1.00] (a Gaussian prior centred on the published
  value is computed in the code but not returned, so the run used the flat prior).
- Sampler: emcee, 32 walkers, 500 burn-in + 5,000 production steps, 5 independent chains, R-hat < 1.001 on all eight floors.
- Result: six of eight floors inside 2σ of the published single-cell values; immune moved 6.44σ (0.795 → 0.838889) because the
  sampler read six immune cells instead of one neutrophil (l.935, l.972).
- Reproduced 2026-09-22 from the deposit: every floor inside its own posterior SD (largest difference 0.000245).
- The same eight classes were then calibrated on the four other substrates (nucleosome occupancy, fuzziness, WPS, fragment size).

## 4. What the record does NOT show
**No run in the record chose the number eight.** Every sampler — G-002, its reproduction, the four-substrate calibrations — was
*given* eight classes; none compares seven, eight, nine or ten. "The MCMC chose eight" is the claim a reviewer will test, and it is
not in the record. What the MCMC established is that eight floors, fitted to the reference cells so assigned, converge and
reproduce. The number eight rests on the definition in §1 and the reasoning in §2.

Two further things a reviewer will ask, answered in advance:
- **The floors are defined so that the 37 reference cells read 1.00.** A reference cell reading 1.00 is therefore not evidence;
  a cell that was *not* in the 37 reading 1.00 is. That is the test every atlas v2 cell takes (acceptance A1–A9).
- **The 37 mean β values** come from the the methylation report web engine's published database (cited to primary sources) with no stated locus set
  or statistic. Tracing each to its paper and region set is open ([`CLASS_ASSIGNMENT_RULE_DRAFT.md`](CLASS_ASSIGNMENT_RULE_DRAFT.md)).

## 5. What would give the count a measured basis (PLAN)
Fit the floors with the classes split and merged where the Day-2 session expected structure — immune into lymphoid and myeloid,
cycling by tissue, progenitor with stem_adult — and compare on cells held out of the fit: does a ninth floor read held-out cells
closer to 1.00 than eight do? Until that runs, eight is a definition with converged floors, not a measured count.

## 6. What a class does in the instrument today
One thing: it names the floor a cell is divided by. Nothing pools cells by class and no class has an A of its own
([`CLASS_USE_INVENTORY.md`](CLASS_USE_INVENTORY.md), `kit/class_guard.py`). A new cell is assigned to a class by the written rule in
[`CLASS_ASSIGNMENT_RULE_DRAFT.md`](CLASS_ASSIGNMENT_RULE_DRAFT.md) — which inversion governs it — never by opinion.
