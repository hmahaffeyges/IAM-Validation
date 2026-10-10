# DEV-HAMMER-01 — can the held copy error be predicted from measured maintenance kinetics? (development; step 1, 2026-10-10)

**Question (book open item D1).** On a held molecule a site opposite a methylated parent fails with probability f per copy; a site opposite
an unmethylated parent is gained with probability g. At steady state ε = f/(f+g), E_hold = k_BT ln(g/f). Can f and g measured on newly copied
DNA predict the ε the cells hold, with nothing taken from ε?
**Data checked (nothing scored):** Hammer-seq, GSE131098 (Ming et al. 2020, Cell Res 30:980; HeLa, EdU pulse, hairpin bisulfite of parent and
daughter strands, chase 4 min-24 h; per-CpG MM/MU/UU/UM event files, 64-106 MB each); same lab, same cells, deep WGBS SRR9328506 (270 M pairs),
from which Stage Q could read HeLa's held ε.

**Step 1, simulation (`development/sims/hammer_01.py`, output `hammer_01_output.txt`).** Known f, g; Hammer-seq's measurement as published
(parent strand = the strand with more methylated CpGs, ties at random; the paper calls this assignment arbitrary).
| truth (ε 0.032) | estimate from events at 24 h | ε predicted |
|---|---|---|
| f 0.010, g 0.30 | f 0.019, g 0.027 | 0.42 |
| f 0.020, g 0.60 | f 0.037, g 0.155 | 0.19 |
| f 0.005, g 0.15 | f 0.010, g 0.006 | 0.62 |
Off by 6-20x at every time point. On a held molecule a true gain (parent U, daughter M) leaves the daughter the more methylated strand, so it is
recorded as a failure (MU): failures are doubled and gains nearly vanish. Raw reads do not fix this: which half of a hairpin is the parent is not
in the sequence.

**Step 1, the deeper finding (algebra, no data).** At steady state the joint state of parent and daughter at a site is symmetric
(π_M f = π_U g), so g/f = (1-ε)/ε identically. Any f and g measured in the same cells at steady state reproduce ε by construction. Measured
kinetics can test whether the two-state picture holds (for example, the speed of recovery after a perturbation must equal f+g), but they cannot
predict ε independently. A prediction of ε0 needs the rates from outside the cell's steady state: the writer's discrimination measured on its
own (enzyme kinetics on hemimethylated against unmethylated DNA), or a physical bound on it from the energy per renewal.

**Decision: DEV-HAMMER-01 as a prediction of ε0 is closed before download.** Open item D1 is restated: derive ε0 from the writer's measured
discrimination, independent of any cell. Possible use of Hammer-seq kept: a test of the two-state picture (recovery speed after UHRF1 depletion
against f+g from steady state), to be designed and simulated separately.
