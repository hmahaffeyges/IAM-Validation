# DEV-LINK-IAMA-METAA-02 — the IAM-A → Met-A derivation on a decitabine dose series (development, 2026-10-09)

**Data.** GSE237665 (one laboratory, HCT116 colon cancer cells): vehicle, decitabine 30 nM and 300 nM, 2 replicates each, read by
EM-seq (GSE237662: SRR25322247-52, 2×150, ~25-33 Gb) and by EPIC v1 (GSE237553: HCT116_DMSO_1/2, DAC30_1/2, DAC300_1/2). Decitabine
traps DNMT1, so maintenance copying fails in proportion to dose: a known, graded loss of the kind the derivation models.

**What is read.** Everything relative to the same cells untreated (the cell's own healthy-equivalent reference; HCT116 is not a healthy
cell, so no state is given).
- IAM-A_rel = H(ε_treated) ÷ H(ε_vehicle), ε by Stage Q's rule on the EM-seq molecules (pinned pipeline, 25 M pairs per run, Q0).
- Met-A_rel = mean H(β) over HCT116 identity sites ÷ the vehicle mean. Identity sites chosen on the two vehicle arrays only, by Met-A's
  rule (vehicle SD ≤ 0.05; mean β 0.75-0.95 or 0.05-0.25; up to 3,000 each).
- The derivation's step dε/dδ measured on the vehicle EM-seq molecules (in-silico loss, as in DEV-LINK-01); the Met-A curves A and B
  computed from the vehicle identity-site means.

**Prediction (from DEV-LINK-IAMA-METAA-01, nothing fitted).** At each dose, measured Met-A_rel lies between curve A and curve B at the
measured IAM-A_rel (allowance: the two vehicle arrays' spread). Below A: Met-A does not see the copy error. Above B: something beyond
copy error moves Met-A (cells switching state). IAM-A_rel above the reverse limit from Met-A falsifies the relation.

**Reading rules added after simulation (DEV-SYNTH-LEVERS-01 §2), before any dose-series data are read.** At each dose report the share of
molecules IAM-A can read and the identity-site mean β. The derivation is tested only at doses where that share is ≥ 0.70 and the
identity sites stay on their own side of β = 0.5; a dose outside that range is recorded as outside the relation's range, not as a result.
