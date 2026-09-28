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

**In plain words (author, 2026-09-28): eight trapdoors.** A class is not a group of cells; it is one of eight trapdoors out of existence, each with its own shape and size, so that only some cells fit through. Every cell stands over one door - its governing inversion. Cells over the same door need not look alike; they share the way out, not the address. A cell can stand at the edge of its door, elevated, in overdrive, and still be that cell; failure is falling through. Senescence and cancer are not doors - they are where a cell lands after it falls.

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
- **Tried to break it, twice, and it held.** A bootstrap over the reference cells (PROC-HMIN-BOOT-01, 2026-09-20) put all eight methylation floors inside the MCMC's 95 % interval, mean relative difference 0.060 %, max 0.095 % (`chain/CPG_Lessons_Learned_2026-06-29.md`; record `Record/PROC_data/PROC-HMIN-BOOT-01/`). The April bootstrap did the same for the 32 non-methylation floors (24/32 in CI, 0.168 %).
- **Tested on data it was not fitted to.** VAL-003 (April 2026, TCGA Pan-Cancer): adjacent-normal tissue read above its floor in **28 of 28** cancer types, 4,092 matched pairs, p = 1.32e-15, +20.2 % mean elevation (`kit/results/VAL_INDEX.json`). Each pair is one person's tumour-adjacent tissue against the fixed point. Pre-commissioning record.
- The same eight classes were then calibrated on the four other substrates (nucleosome occupancy, fuzziness, WPS, fragment size).

## 4. What the record does NOT show
**No run in the record chose the number eight.** Every sampler — G-002, its reproduction, the four-substrate calibrations — was
*given* eight classes; none compares seven, eight, nine or ten. "The MCMC chose eight" is the claim a reviewer will test, and it is
not in the record. What the MCMC, the bootstrap and VAL-003 established is that eight floors, fitted to the reference cells so assigned, converge,
reproduce, survive resampling and predict the direction in 28 of 28 cancer types. Whether seven or nine would do as well is the untested part. The number eight rests on the definition in §1 and the reasoning in §2.

Two further things a reviewer will ask, answered in advance:
- **The floors are defined so that the 37 reference cells read 1.00.** A reference cell reading 1.00 is therefore not evidence;
  a cell that was *not* in the 37 reading 1.00 is. That is the test every atlas v2 cell takes (acceptance A1–A9).
- **The 37 mean β values** come from the the methylation report web engine's published database (cited to primary sources) with no stated locus set
  or statistic. Tracing each to its paper and region set is open ([`CLASS_ASSIGNMENT_RULE_DRAFT.md`](CLASS_ASSIGNMENT_RULE_DRAFT.md)).

## 5. What would give the count a measured basis (PLAN)
Fit the floors with the classes split and merged where the Day-2 session expected structure — immune into lymphoid and myeloid,
cycling by tissue, progenitor with stem_adult — and compare on cells held out of the fit: does a ninth floor read held-out cells
closer to 1.00 than eight do? Until that runs, eight is a definition with converged floors, not a measured count.

**The pass/fail rule for a new class (author, 2026-09-28: a future scientist may decide there are more than eight).** A proposed split is a class only if both hold: (1) **definition** - the split-off cells have their own governing inversion; (2) **measurement** - its fitted floor differs from the parent class's floor by more than the tier tolerance, with neither floor inside the other's 95 % posterior interval, scored on cells held out of the fit. A split whose floor lands inside the parent's bars changes no reading and is not a class. Written before CLASS-COUNT-01 runs; it does not move afterwards.

## 6. What a class does in the instrument today
One thing: it names the floor a cell is divided by. Nothing pools cells by class and no class has an A of its own
([`CLASS_USE_INVENTORY.md`](CLASS_USE_INVENTORY.md), `kit/class_guard.py`). A new cell is assigned to a class by the written rule in
[`CLASS_ASSIGNMENT_RULE_DRAFT.md`](CLASS_ASSIGNMENT_RULE_DRAFT.md) — which inversion governs it — never by opinion.

## 7. Regime and the 1-bit bound (2026-09-28, corrected by the author the same day)
A failure regime is not the cell's maximum entropy. The maximum is 1 bit per locus for every cell and cannot tell classes apart; the regime is the route - which mechanism drives the cell off its floor.

**Elevated is not failure.** A cell can read well above the NORMAL bars and still be ordered - in computational overdrive, still that cell. Failure is where the cell stops being that cell: senescence and beyond. The 1-bit bound (A = 1/H_min, 1.018 for stem_pluri) is arithmetic saturation of H(mean beta) at a coin flip. It is not a floor, not a loss of fidelity and not a failure point, and it is not printed on the report (author's ruling, 2026-09-27).

**Open instrument question, recorded as a fact and not interpreted:** on H(mean beta)/H_min the highest A a stem_pluri cell can return is 1.018, inside NORMAL, so on this construction the gauge cannot show a pluripotent cell above the ELEVATED line whatever its state. Whether that is a limit of the construction for that class is for the author; nothing is changed on the strength of it.

**Hypothesis - range of operation follows commitment (author and assistant, 2026-09-28; reasoning, test named).** The range between a class's floor and the 1-bit bound is widest for terminal cells (to A = 1.294: post-mitotic, identity locked deep, must stay the same cell for a lifetime under oxidative load) and narrowest for pluripotent cells (to 1.018: identity is to stay undecided, nearly every address poised, a transient state whose every move is commitment). the quantum-processor report/the semiconductor report form: a latched bit with a large noise margin vs a sense amplifier held at its metastable point. The six middle classes sit within about 2 % of each other (1.145-1.192) and do not order by lifespan (cycling above stromal), so the claim is about the two ends only. Test: atlas v2 cells scored against their own class, held-out - do pluripotent cells stay in the narrow window and terminal cells use the wide one.

## 8. Do cells over the same door share a genome pattern? (exploratory, 2026-09-28)
Asked because a pattern other than H_min would be independent support. 56 Loyfer cells, 651k CpGs, no H_min used ([`class_structure.json`](class_structure.json), [`class_shared_sets.csv`](class_shared_sets.csv), `class_shared_sky.png`).
- Nearest neighbour: germ layer 0.96, cell family 0.95, class 0.86, organ 0.64 - class structure is mostly lineage.
- CpGs unmethylated in every cell of the class and methylated in >= 90 % of the others: immune 398 (one lineage), terminal 1, stromal 1, cycling 0, secretory 0. Neurons + oligodendrocytes share 202, cardiomyocytes + striated muscle 43, all four terminal cells 1.
- Identity sets cluster along the genome (median spacing immune 1,516 vs 2,122 random; neural 2,037 vs 2,953; muscle 7,439 vs 17,736).
**Reading:** identity has an address in the genome; a door does not. This answers the reviewer's first objection - the classes are not lineages renamed: terminal, secretory and cycling cut across lineages and share no identity mark. The independent evidence for a class must therefore be functional: whether every cell over a door keeps the machinery of that door's inversion open (NRF2/antioxidant for terminal, ER/UPR for secretory, replication/MMR for cycling, YAP/TAZ for stromal, cytokine signalling for immune). Not yet run; needs curated pathway gene sets.
