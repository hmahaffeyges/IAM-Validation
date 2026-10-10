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

**IAM-A window, sealed 2026-10-10 02:00 UTC before any EM-seq reading** (`predict_iama_window_02.py`, nothing fitted). Each curve is
inverted at the measured Met-A_rel for the loss δ it needs. The step δ → IAM-A_rel is **Stage Q's measured response** to that loss
(`insilico_loss_02.py`: every methylated call lost with probability δ on real molecules, read by Stage Q's own code), as this note requires.
An earlier version at 01:50 used the simple form ε = ε_v + δ(1 − ε_v). Stage Q reads only molecules ≥ 80 % methylated and only isolated errors,
so its reading rises about half as fast (δ 0.10: 1.55 measured against 2.21 simple, 74 % of molecules read). That version was replaced
before any data, because it would have failed on the instrument, not on the physics.

Response so far on a stand-in (healthy neutrophil EM-seq-free WGBS, SRR9888330, ε_v 0.045; `insilico_standin_neutrophil_SRR9888330.csv`);
the final window is the same script on the HCT116 vehicle molecules when they land:

| dose | Met-A_rel | δ_B | δ_A | IAM-A_rel window | status |
|---|---|---|---|---|---|
| 30 nM | 1.4255 | 0.073 | 0.188 | ≥ 1.44; δ_A is beyond the readable range (≥ 70 % read to δ 0.10, IAM-A_rel 1.55) | **the test** |
| 300 nM | 1.6710 | 0.129 | — | δ_B already beyond the readable range | outside the relation's range; recorded, not scored |

**Prediction at 30 nM:** IAM-A_rel 1.44–1.55, with ≥ 70 % of molecules read. Below 1.44 with ≥ 70 % read: Met-A moved more than copy
error allows (curve B), so something beyond copy error moves it. Fewer than 70 % read: outside the range, no result.

**Final window, sealed 2026-10-10 03:30 UTC, after the vehicle run Veh_EM_1 (SRR25322252, ε 0.0508 by the box, finished 03:16) and before any
treated run is read.** Stage Q's response measured on its molecules (`insilico_vehicle_SRR25322252.csv`, first 1,500,000 lines as for the
stand-in; ε_v 0.0500 on that slice). Readable (≥ 70 % of molecules read) to δ 0.10, where IAM-A_rel is 1.438.

| dose | δ_B | IAM-A_rel at δ_B | readable ceiling | **prediction** |
|---|---|---|---|---|
| 30 nM (mean of 2 arrays) | 0.0732 | 1.351 | 1.438 at δ 0.10 | **IAM-A_rel 1.35–1.44, ≥ 70 % read** |
| 30 nM, per array | 0.0670 / 0.0796 | 1.328 / 1.374 | | |
| 300 nM | 0.129 | beyond the readable range | | recorded, not scored |

Read: below 1.35 with ≥ 70 % read: Met-A moved more than copy error allows, so something beyond copy error moves it. Above 1.44 with ≥ 70 % read:
not possible on this instrument, so it would point to a fault. Fewer than 70 % read: outside the range, no result.
