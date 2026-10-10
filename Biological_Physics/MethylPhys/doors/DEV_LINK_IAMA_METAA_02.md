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

**Met-A side, read 2026-10-10 01:40 UTC, before any EM-seq (IAM-A) reading of this series exists** (`doors/data/DEV_LINK_IAMA_METAA_02/
metaa_dose_02.py`, committed 792d7dd before running; rows `metaa_dose_02_rows.csv`; 3,000 high + 3,000 low identity sites from the vehicles).

| arrays | Met-A_rel | methylated identity sites, mean β | unmethylated, mean β |
|---|---|---|---|
| vehicle (2) | 0.9986, 1.0014 | 0.898 | 0.091 |
| DAC 30 nM (2) | 1.3944, 1.4565 | 0.704 | 0.090 |
| DAC 300 nM (2) | 1.6425, 1.6995 | 0.549 | 0.121 |

Both doses keep the identity sites on their own side of β = 0.5 (300 nM narrowly). The IAM-A side and curves A and B come from the box EM-seq
runs and are compared with these readings as they stand.
