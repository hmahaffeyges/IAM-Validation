# DEV-FINGERPRINT-01 — cancer both ways: LNCaP against PrEC (GSE86833) (development; step 1 written 2026-10-10, nothing from GSE86833 read)

**Question.** If copy error is all that changed, Met-A can move only as far as curve B allows for the measured IAM-A (DEV-LINK-IAMA-METAA-01).
Prediction: cancer cells sit above that limit; the program was rewritten, not only copied badly. Design and power: DEV-SYNTH-LEVERS-01
(`development/sims/fingerprint_power_01.py`: excess 0.04 / 0.06 / 0.10 above curve B detected 58 / 87 / 100 %, false call 5 %).
References: Met-A prostate identity sites from atlas v2 (prostate epithelium IN); IAM-A position P = 1.210 (4 Loyfer donors, whole files,
`data/IAMA_POSITION_PROSTATE/position_prostate_v1.json`).

**Step 1 — synthetic checks on healthy prostate molecules (stand-in: Loyfer GSM5652341, first 1.5 M lines; PrEC's own molecules come next).**
1. Stage Q's response to scattered loss (`data/DEV_FINGERPRINT_01/insilico_standin_prostate_GSM5652341.csv`): ε_v 0.0404; IAM-A_rel 1.10 / 1.19 /
   1.34 / 1.46 / 1.65 at δ 0.01 / 0.02 / 0.04 / 0.06 / 0.10, against 1.17 / 1.33 / 1.63 / 1.89 / 2.35 by the simple form. Readable (≥ 70 % of
   molecules) to δ ≈ 0.12. As in HCT116, Stage Q reads about half the simple rise; every window is built on the measured response.
2. Run-type loss (`runloss_sim_prostate.txt`, run-loss reading of DEV-RUNLOSS-01 on two Loyfer prostate donors): planted run loss 0.10 / 0.22 read as
   excess 0.0997 / 0.2163; scattered loss 0.10 / 0.22 / 0.40 read 0.0000; mixture run 0.10 + scattered 0.10 read 0.0983.

**What it means for the test.** Cancer cells lose methylation in large blocks. Molecules lost in a block leave Stage Q's count, so LNCaP may fall
below 70 % read and give no IAM-A reading, as decitabine did. The design therefore carries two arms, both sealed before LNCaP is read:
- **Arm A (IAM-A readable):** Met-A_rel(LNCaP ÷ PrEC) against curve B at the measured IAM-A_rel; fingerprint = Met-A above curve B (power above).
- **Arm B (fewer than 70 % read):** the run-loss reading on LNCaP against PrEC; fingerprint = excess run loss above the PrEC replicate spread.
  Copy error alone cannot produce run loss (simulation: 0.0000 for scattered loss up to 0.40).

**Still to do, in order:** PrEC WGBS (4 runs) through the pinned pipeline → Stage Q's response on PrEC's own molecules → windows for both arms
sealed → LNCaP (5 runs) and the arrays read → scored as sealed.

**Power through Stage Q (2026-10-10; `development/sims/fingerprint_power_02.py`, output `fingerprint_power_02_output.txt`).** Curve B was written
against the simple-form IAM-A. In Stage Q's units (prostate response) its slope is 2.06, not 1.17, so IAM-A's own repeat error weighs about twice as
much. Detection of Met-A above curve B (2 arrays, 4 runs a side; false call 5 %): excess 0.04 → 0.50 (was 0.58), 0.06 → 0.79 (was 0.87),
0.10 → 0.99. Arm A of the window will be sealed on this curve, rebuilt on PrEC's own molecules.
