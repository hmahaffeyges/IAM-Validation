# DEV-RUNLOSS-01 — a molecule reading of run-type loss (written 2026-10-10; simulation first, prediction sealed before reading)

**Why.** IAM-A reads scattered copy error on molecules still held (Stage Q: ≥ 80 % methylated, isolated errors). In the decitabine series
(DEV-LINK-IAMA-METAA-02) treated cells lost 40–65 % of readable molecules while IAM-A on the survivors rose only 4–12 %. A trapped DNMT1 leaves the
rest of a stretch uncopied, so the daughter strand reads unmethylated end to end: loss in runs. IAM-A cannot see that by design. This reading can.

**Definition** (`data/DEV_RUNLOSS_01/runloss_01.py`, nothing fitted). Territory = molecules with ≥ 6 CpG calls whose CpGs average vehicle β ≥ 0.8.
K = share ≥ 80 % methylated, L = share ≤ 20 % methylated, d_eff = 1 − β_T(sample)/β_T(vehicle). L_scat(d_eff) = L if the same loss were
scattered (computed on the vehicle's own molecules). **Excess run loss = L − L_scat.**

**Simulation (vehicle Veh_EM_1 molecules, 493,784 territory molecules):**
| planted | d_eff | L | L_scat | excess |
|---|---|---|---|---|
| scattered 0.10 / 0.22 / 0.40 | 0.0999 / 0.2197 / 0.3997 | 0.0042 / 0.0097 / 0.0453 | 0.0043 / 0.0095 / 0.0461 | −0.0000 / +0.0002 / −0.0008 |
| run 0.10 / 0.22 | 0.1006 / 0.2203 | 0.1032 / 0.2226 | 0.0043 / 0.0095 | +0.0990 / +0.2132 |
| run 0.10 + scattered 0.10 | 0.1902 | 0.1043 | 0.0075 | +0.0968 |
It separates the two kinds of loss and recovers the run part of a mixture.

**Prediction for the decitabine series (sealed; d_eff, L and K not yet computed on treated runs; the share of Stage Q-readable molecules was seen):**
1. Cross-platform: d_eff from the EM-seq molecules equals the array loss at methylated identity sites, 1 − β_hi(dose)/β_hi(vehicle) =
   0.22 (30 nM) and 0.39 (300 nM) from the EPIC means 0.90 / 0.70 / 0.55, within ± 0.05 (different site sets).
2. Run type: excess L ≥ 0.5 × d_eff at both doses (most of the loss in runs).
Development: whichever way it reads, a second experiment (another drug or laboratory, with arrays) is needed before this reading is commissioned.
