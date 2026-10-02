# DEV-TARE-02 — Stage T without a fitted term (development, 2026-10-02)

**Change.** Stage T is a median tare against >= 3 healthy references of the same specimen type run the same way (same slide, else same batch):
A_rel = A / median(reference A). The DEV-NOISE-02 correction A = a + b f_neu + c N, fitted on the reference records after looking at the data, is removed:
nothing in the chain is fitted. The noise index N (DEV-NOISE-01) becomes a gate: if N > N_max = 0.149 (top of the six reference arrays' range;
`chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`) and the reading is untared, the gauge state is withheld and A is printed as a number only.
Planned physical tare (OPEN until a laboratory runs it): fully methylated and fully unmethylated control DNA on every slide.

**Acceptance rerun** (box job 820698ac, `chain_tests/chain_acceptance_tare_median_2026-10-02.csv`):

| set | n | result |
|---|---|---|
| purified healthy neutrophils (reference arrays) | 6 | Normal 6/6, A 0.994-1.006; N 0.122-0.149, gate pass 6/6 |
| known mixtures, neutrophils >= 50 %, tared against each other | 6 | Normal 6/6, A_rel 0.987-1.021; untared A 0.943-0.968 (Normal 3/6) |
| AML second-remission bloods (other lab) | 10 | not run: IDATs not on the current box 1 |

**Withhold branch** (`chain_tests/test_noise_gate.py`, synthetic arrays): N 0.045 untared -> state kept; N 0.327 untared -> withheld; N 0.327 tared -> tared state. PASS.
The gate was not triggered by any real array in this set; it still has to be tested on real noisy arrays (second-lab neutrophils of DEV-NOISE-01).
