# IAM canon: constants and names

Generated from `iam_canon.json` v1.0 (2026-10-01). Do not edit by hand.

This file is the single source of truth for constants and names. To change a value or a name, change it here first; canon_check.py then lists every file that must follow. A push is blocked while a LIVE file disagrees.

## Constants

| name | value | units | derivation / source |
|---|---|---|---|
| k_B | 1.38065e-23 | J/K |  — SI 2019 exact |
| R | 8.31446 | J/(mol K) |  — SI 2019 exact (N_A k_B) |
| T_cell | 310.15 | K |  — 37 °C, mammalian body temperature |
| dG_ATP | 54000 | J/mol |  — cellular ATP hydrolysis free energy used by the author (54 kJ/mol) |
| M_cell | 20.94 | dimensionless (kT units) | M = dG_ATP/(R T_cell) = 54000/(8.314462618 x 310.15) — author's settled ruling 2026-09-19/27: no ln2 in the definition |
| M_cell_Landauer | 30.2 | Landauer units (k_B T ln2 per bit) | M_cell / ln 2 — conversion only; not a second definition |
| Landauer_floor_cell | 2.96811e-21 | J per bit | k_B T_cell ln 2  |
| E_hold_meth | 3.41 | kT per methylated site | ln((1-eps)/eps) of the measured copy error on single molecules (Loyfer read-level, uncorrected) — PROC-CHANNEL-01 |
| E_hold_meth_Landauer | 4.9 | Landauer units | E_hold_meth / ln 2  |
| phi | 0.1628 | fraction of one ATP per held bit | E_hold_meth / M_cell — PROC-CHANNEL-01 |
| eps0_meth | 0.032 | error per methylated site per copy | 1/(1+exp(phi*M_cell)) — PROC-CHANNEL-01 physics floor |
| Normal_band | [0.95, 1.05] | A |  — author: 'Healthy is A=1 +/-5' |
| H_min_450K_neutrophil | 0.784057 | bits | Met-A floor from purified 450K neutrophils (GSE88824, 8 donors, our Stage 1) — DIAG-450K-01 / floors_450k_v1.json |
| beta_m | 0.1575 | dimensionless | Omega_m / 2 with Omega_m = 0.315 — IAM's Law paper (cosmological expression) |
| E_of_a | exp(1 - 1/a) | activation function |  — IAM's Law paper |

## Names in use

| name | status | definition |
|---|---|---|
| IAM's law | current | Every irreversible transition from quantum potential to classical actuality pays k_B T ln 2 per bit to the nearest encoding surface at the local temperature. |
| Mahaffey number (M) | current | Drive energy over the thermal noise quantum: M = E_drive/(k_B T). Cell: 20.94 (30.2 in Landauer units). |
| Landauer metrology | current | Measuring how far above its thermal floor an information-writing process operates, against a fixed physical zero, in any substrate. |
| MethylPhys CPG | current | The instrument: Methylation Physics, Cellular Performance Gauge. |
| Met-A (Metrology A) | current | Program reading: Shannon entropy of the mean beta at a cell's identity loci over that cell type's reference floor, built from purified cells on the same platform, tared against same-run controls. Information theory only; no Landauer or energy content. |
| IAM-A | current | Error reading: the per-molecule copy error of a cell's held methylation pattern, on single-molecule sequencing, read against the physics floor eps0 = 1/(1+exp(phi*M)). Thermodynamic; the cell's instance of IAM's law. |
| Salmon-A | informal | Not a separate quantity: IAM-A applied to salmonids (body temperature = water temperature). |
| the quantum-processor report | current | Quantum computing report line. |
| the semiconductor report | current | Semiconductor report line. |

## Retired or pending names

| pattern | use instead | severity |
|---|---|---|
| `n_bio` | none (removed from the method) | block |
| `\bold A\b` | Met-A | warn |
| `\bnew A\b` | IAM-A | warn |
| `gate-error reading` | IAM-A | warn |
| `program reading` | Met-A | warn |
| `Genomic Analytical (&|and) Performance Engine` | MethylPhys CPG | confirm |
| `\bGAPE\b` | MethylPhys CPG (author to confirm) | confirm |
| `M\s*=\s*30\.2` | M = 20.94 (30.2 only as the Landauer-unit conversion) | warn |
