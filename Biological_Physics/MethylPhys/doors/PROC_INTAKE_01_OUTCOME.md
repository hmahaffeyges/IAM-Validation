# PROC-INTAKE-01 — outcome: ADOPTED. The intake gate runs on the array's own numbers; a deferred check never advances.

**Scored 2026-09-27** against [`PROC_INTAKE_01_PREREG.md`](PROC_INTAKE_01_PREREG.md) under the author's 0.93 line (recorded in the
pre-registration's decision section before these bars were scored). Runner kit/PROC_INTAKE_01.py (archived privately); results kit/results/PROC_INTAKE_01.json (archived privately);
the 48-array measurement behind the line kit/results/PROC_INTAKE_01_48_array_measurement.csv (archived privately). A first run of the runner was voided:
a stray copy kept writing into the same log (this sandbox cannot stop a process started in an earlier cell), so the run was repeated
on fresh paths and only that run is scored.

| bar | rule | measured | |
|---|---|---|---|
| B1 | 12 Munich panel arrays: call rate reported; < 0.93 QUARANTINE, no report | 12 of 12 quarantined (detected fraction 0.764–0.928), 0 reports written, exit 2 each | MET |
| B2 | first 100 GSE87571 IDAT pairs: ≥ 95 PROCEED | **100 of 100** advance (92 PASS, 8 PROCEED_WITH_PENALTY at 0.973–0.980); min 0.9728, median 0.9877 | MET |
| B3 | masking moves A on good arrays by < 0.002 per present cell | max |ΔA| **0.0066** (12 arrays; per-array max 0.0018–0.0066) | **FAILED as written** |
| B4 | betas-only input renders with `intake_verified: False` and one printed line | exit 0, `intake_verified` False, line printed | MET |
| B5 | Stage 1 return with the mask withheld must QUARANTINE | **PROCEED, advance True** — the decision gate recorded `deferred` and advanced | **FAILED** → fixed at source, re-tested: QUARANTINE, advance False, `intake_deferred:call_rate` |

## What B3 measured (diagnostic after the bar, on three Uppsala arrays)

ΔA = A_masked − A_unmasked is **negative on every present cell**: −0.0010 to −0.0046, in proportion to the fraction of the cell's
identity loci that were masked (0.5 % → −0.001 to −0.0025; 2 % → −0.0025 to −0.0046). The masked probes' unmasked β sits at
0.33–0.42 (both channels at background read toward 0.5) while identity loci sit near 0.74, so a probe at background pulls the mean
β down and H(mean β) up. **Every reading before 2026-09-27 carried that upward bias — about 2 mA per percent of loci at
background.** The bar of 0.002 was written without this measurement; the masked reading is the measurement and the unmasked one
was not. Adopted: probes at background are removed before any stage reads a β (Stage 1, [`stage_1_idat_calibration.py`](../chain/stage_1_idat_calibration.py)).

## What B5 found

`step_0_9_decision_gate` listed a deferred detection or call-rate check under `stage0_deferred_qc` and returned PROCEED — the hole
that let a low-signal laboratory be read for months (FINDING_GSE125105_LOW_SIGNAL.md). Now a deferred **detection or call rate**
is a hard failure (`QUARANTINE`, `intake_deferred:…`). Bead count (not extracted by Stage 1) and the provisional bisulfite
threshold remain recorded-deferred, as the SOP states for them.

## Constants

`Runtime Matrices/Intake/intake_thresholds_v1.json`: call rate PROCEED ≥ 0.98, QUARANTINE < 0.93 (PENALTY between); detection
PASS > 0.99, FAIL < 0.93. The author set 0.93 from the 48-array measurement: Uppsala 0.985 (min 0.979), Karolinska 0.975 (min
0.891), UCLA 0.953 (min 0.932), Munich 0.878 (max 0.928); gap 0.894 → 0.928 → 0.932. Signal-to-background alone does not
separate the laboratories. These describe the array, never a person.
