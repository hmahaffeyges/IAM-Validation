# IAM canon: constants and names

Generated from `iam_canon.json` v1.1 (2026-10-02). Do not edit by hand.

This file is the single source of truth for constants and names. To change a value or a name, change it here first; canon_check.py then lists every file that must follow. A push is blocked while a LIVE file disagrees.

## Constants

| name | value | units | derivation / source |
|---|---|---|---|
| k_B | 1.38065e-23 | J/K |  — SI 2019 exact |
| R | 8.31446 | J/(mol K) |  — SI 2019 exact (N_A k_B) |
| T_cell | 310.15 | K |  — 37 °C, mammalian body temperature |
| dG_ATP | 54000 | J/mol |  — cellular ATP hydrolysis free energy (54 kJ/mol) |
| M_cell | 20.94 | dimensionless (kT units) | M = dG_ATP/(R T_cell) = 54000/(8.314462618 x 310.15) — defined without ln2; the Landauer-unit value M/ln2 = 30.2 is given separately |
| M_cell_Landauer | 30.2 | Landauer units (k_B T ln2 per bit) | M_cell / ln 2 — conversion only; not a second definition |
| Landauer_floor_cell | 2.96811e-21 | J per bit | k_B T_cell ln 2  |
| E_hold_meth | 3.41 | kT per methylated site | ln((1-eps)/eps) of the measured copy error on single molecules (Loyfer read-level, uncorrected) — PROC-CHANNEL-01 |
| E_hold_meth_Landauer | 4.9 | Landauer units | E_hold_meth / ln 2  |
| phi | 0.1628 | fraction of one ATP per held bit | E_hold_meth / M_cell — PROC-CHANNEL-01 |
| eps0_meth | 0.032 | error per methylated site per copy | 1/(1+exp(phi*M_cell)) with phi*M_cell = E_hold_meth = 3.41 kT MEASURED from cells' copy error: the HEALTHY REFERENCE copy error (not the floor). The form is physics (Boltzmann), the height is measured; about half of healthy cell types hold better than it — PROC-CHANNEL-01; a physics-only height (a derived phi from the kinetics of maintenance) is open |
| Normal_band | [0.95, 1.05] | A |  — healthy is A = 1 within 5 % |
| beta_m | 0.15765 | dimensionless | beta_m = Omega_m/2 with Omega_m = 0.3153 (Planck 2018 TT,TE,EE+lowE+lensing 68 % limits, Aghanim et al. 2020 Table 2; the best fit is 0.3158); value used in every Level 2 chain — Cosmological_Physics/camb_validation/equations_iam_level2.f90; IAM's Law paper (cosmological expression) |
| E_of_a | exp(1 - 1/a) | activation function |  — IAM's Law paper |
| Met_A_floor_EPIC_neutrophil | 0.330263 | bits (mean per-site H(beta)) | mean over 6 purified healthy EPIC neutrophil arrays (Salas, GSE110554; GSE167998 re-deposits the same 6 arrays) of the mean H(beta) over 6000 identity sites, our Stage 1 — chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json (frozen 2026-10-01); held-out precision (sites re-chosen on the other 5 arrays): SD 0.020, 0.983-1.045 (metA_floors_v1_3_loo.csv) |
| Met_A_floor_450K_neutrophil | None | bits | pending: v1.2 rebuild from purified 450K neutrophils (GSE88824, 8 donors) — to be frozen |
| Met_A_site_rule | across-array SD <= 0.05; own mean beta 0.75-0.95 (methylated) or 0.05-0.25 (unmethylated); <= 3,000 sites per channel, most stable first | rule |  — Met-A reference floors v1.2 |
| P_neutrophil_IAM_A | 1.1492 | dimensionless (H(eps_healthy) / H_ref) | mean over 3 Loyfer granulocyte donors of H(eps of the other donors)/H(eps0), WHOLE files (1.1396-1.1554, CV 1.2 %) — chain/Runtime Matrices/IAM_A_Positions/iama_positions_v2.json (frozen 2026-10-08, whole files); valid for the loyfer_pat_v1 pipeline only |
| H_ref_cell | 0.2043 | bits | H(eps0_meth): the healthy reference height, where healthy cells actually hold their methylated sites (56 healthy cell types) — PROC-CHANNEL-01 (holding energy measured across 56 healthy cell types) |
| H_min_cell | 2.5e-08 | bits | H(1/(1+exp(M_cell))): the floor, where thermal kicks win against one full ATP per site (copy error 8.1e-10) — calculated from M_cell = 20.94 |

## Names in use

| name | status | definition |
|---|---|---|
| IAM's law | current | Every irreversible transition from quantum superposition to classical record pays k_B T ln 2 per bit to the nearest encoding surface at the local temperature. |
| Mahaffey number (M) | current | The drive ratio: drive energy over the thermal noise quantum, M = E_drive/(k_B T). Sets the operating regime; for a record held in a gap hf, M = hf/(k_B T) and its thermal floor is 1/(1+e^M). Cell: 20.94 (30.2 in Landauer units). Chips: M = E_sw/(k_B T_j) = A x ln 2 (chip gauge). The only quantity named 'Mahaffey'. |
| IAM floor | current | Synonym of H_min (the floor): the lowest error a system can hold at its temperature before thermal kicks win. |
| Informational fidelity ratio (A) | current | A = reading / the same system's healthy reading (the healthy reference), measured on the same instrument. Dimensionless; A = 1 is the system working as itself, in the middle of the gauge. Below it lies H_min, the lowest error the system can hold before thermal kicks win; nothing reads below H_min. Above 1 = more error than healthy. Cells: Met-A = mean H(beta) over identity sites / the healthy reference (purified healthy cells, same platform); IAM-A = H(eps)/(P_cell H(eps0)). Qubits: eps/eps(as built), eps = -ln(1-p). Chips: E_sw/E_sw(as built). A is never the temperature exponent n. |
| Landauer metrology | current | Measuring how far above its thermal floor an information-writing process operates, against a fixed physical zero, in any substrate. |
| the methylation report | current | The biological (genetic) application of IAM's law: the umbrella for every cellular and organismal chain. Expansion as in Issue 002: Genomic Analytical & Performance Engine. Chains under it: MethylPhys CPG (methylation, cells), the salmonid chain (in development). |
| MethylPhys CPG | current | The cell instrument: Methylation Physics, Cellular Performance Gauge. Reads cells' methylation (Met-A, IAM-A) and places them on the gauge. |
| Salmonid chain | current (in development) | The cell instrument applied to salmonids: IAM-A per fish and tissue at water temperature. Not a cellular performance gauge. |
| Met-A (Metrology A) | current | Array reading of ONE cell type in one specimen (chain v3, Stage M): Met-A = mean over the cell's identity sites of H(beta) / H_min(cell), H(b) = -b log2 b - (1-b) log2(1-b). H_min(cell) = the same quantity on purified healthy cells of the same type on the same platform, through the chain's own Stage 1 (constant Met_A_floor_EPIC_neutrophil; site rule Met_A_site_rule); >= 90 % of the identity sites must be measured. Whole blood: the denominator is the composition-matched healthy expectation mean_i H(sum_c f_c mu_c,i), with the specimen's own fractions f (Stage A, EPIC blood NNLS) and purified healthy profiles mu (no group comparison; PROC-WB-NEUT-01); read at any neutrophil fraction with its own detection limit (DEV-LOWFRAC-01; no fixed fraction cut). Stage T tare: against >= 3 healthy references run the same way (same slide, else same batch), A_rel = A / median(reference A); with >= 20 reference records the noise-corrected tare A = a + b f_neu + c N (N = noise index, mean H(beta) over 48,528 fixed sites; DEV-NOISE-02). The gauge state is printed on A_rel only; without references the reading is reported as untared. Each reading carries its detection limit (smallest loss of the cell's pattern the specimen could show). Scope: neutrophils on EPIC v1; 450K and EPIC v2 refused. Normal 0.95-1.05. Information theory only; no energy content. Expansion: Informational Metrology A score. |
| IAM-A | current | Error reading: the per-molecule copy error eps of a cell's held methylation pattern on single-molecule sequencing (isolated unmethylated call between two methylated calls on a qualifying molecule), read against the cell type's healthy reference (its position P_cell relative to H_ref): IAM-A = H(eps) / (P_cell * H(eps0)), eps0 = 1/(1+exp(phi*M)) = 0.032 (form derived, height measured), P_cell = H(eps_healthy)/H(eps0), the healthy cell type's position on the floor, measured once on healthy cells with the same read-level pipeline and frozen (chain/Runtime Matrices/IAM_A_Positions/iama_positions_v2.json; neutrophils P = 1.149). Instrument effects cancel because reference and reading share the pipeline. Methylated (copy) channel only: the unmethylated (de novo) channel is dominated by bisulfite conversion failure. Holding energy E = ln((1-eps)/eps) kT. Normal 0.95-1.05. Chain v3 Stage Q: P is valid only for the read-level pipeline it was measured on (neutrophils: loyfer_pat_v1, first 60 MB of each granulocyte .pat file); any other pipeline is refused until P is measured on healthy cells with it. A reading needs >= 100,000 opportunities; the two run-halves are printed separately above 50,000 each. |
| Salmon-A | informal | Not a separate quantity: IAM-A read by the salmonid chain (body temperature = water temperature). |
| the quantum-processor report | current | Quantum computing report line. |
| the semiconductor report | current | Semiconductor report line. |
| EPLG | current | Error Per Layered Gate (IBM): the average error per gate across a 100-qubit chain running layered circuits, including crosstalk and idle errors. It is NOT the same quantity as an isolated median two-qubit error p(2Q) and must not be compared with one. |
| C-score (sky-map C-score) | current | Third reading per specimen: clustering of a specimen's residual sky map (variance of 50-consecutive-site block means x 50, over the site variance), divided by the healthy baseline measured on the same platform and site set, so healthy = 1 on the same gauge as A. Two C-scores per cell, one per reading: Met-A C-score (arrays) and IAM-A C-score (sequencing; map of per-region copy error against the cell's physics floor). Catches damage concentrated in genomic regions that the averaged A dilutes. A cosmology map statistic (Methods from the Sky), not a physics floor. Development only; band not yet set. On first use in papers write 'sky-map C-score' (C-score is also an ecology co-occurrence statistic). IMPLEMENTED in chain v3: Met-A C-score only (Stage MC; clustering block from neutrophil_reference_v1_1, healthy baseline from leave-one-out neutrophil maps). The IAM-A C-score is defined but not yet built. On the one gauge: C = clustering / healthy median, so healthy reads 1. |
| H_min (cell) | retired | In the cell chain, the floor a reading is divided by. For Met-A: the cell type's reference floor on that platform (Met_A_floor_*), one per cell type, from purified healthy cells - never a class value. For IAM-A: P_cell * H(eps0), the physics floor's entropy at the cell's frozen architecture position P_cell (same read-level pipeline). |
| H_min (the floor) | current | The absolute lowest error a system can hold before thermal kicks win; below it the system no longer exists as itself. It is NOT the denominator of A and NOT 1: it sits below 1 at H_min / healthy reference. Cells: thermal kicks win against one full ATP per site, copy error 1/(1+e^M) = 8.1e-10, H_min = 2.5e-8 bits (1e-7 on the IAM-A gauge, 8e-8 on Met-A, neutrophils). It is NOT the healthy reference height H_ref = H(eps0) = 0.2043 bits. Qubits: the thermal floor p_eq t_g/T1, p_eq = 1/(1+exp(hf/(k_B T))) at the temperature the qubit's record sees. Chips: the Landauer floor k_B T_j ln 2. |
| eps (gate error in nats) | current | eps = -ln(1 - p(2Q)), the two-qubit gate error expressed in nats; eps ~ p for small p. It is the qubit reading; A = eps / eps(as built). |
| One gauge (convention) | current | Every score is an error score on one gauge: A = 1 is healthy in the middle, H_min below it is the floor nothing passes, above 1 is more error; a failure point may lie on the high side (qubits: error correction fails near p = 1 %). Each domain keeps its field's form. Always quote a value with its score's name. |
| Healthy reference | current | The denominator of A: the same system's healthy reading on the same instrument. Cells: Met-A, the purified healthy cells of that type on that platform (EPIC neutrophils 0.330263 bits; constant name Met_A_floor_* kept for the chain until the SOP rename); IAM-A, P_cell H_ref = P_cell H(eps0) (neutrophils 0.2348 bits). Devices: the device as built. |
| Thermal floor (qubit) | current | The qubit H_min and the left-hand mark of the qubit gauge: IAM's Law at the temperature the record sees. p_eq = 1/(1+exp(hf/(k_B T))) for the held record; p_eq t_g/T1 for one gate of duration t_g (detailed balance). |
| H_ref (healthy reference height) | current | Cells: where healthy cells actually hold their methylated sites, H(eps0) = 0.2043 bits (E_hold 3.41 kT, measured across 56 healthy cell types; about half hold better than it). Each cell type's own IAM-A healthy reference is P_cell H_ref (neutrophils 0.2348 bits). Measured, not a limit; the floor is H_min. The distance from H_min to H_ref in energy is phi = 0.16 of one ATP. |

## Retired or pending names

| pattern | use instead | severity |
|---|---|---|
| `n_bio` | none (removed from the method) | block |
| `\bold A\b` | Met-A | warn |
| `\bnew A\b` | IAM-A | warn |
| `gate-error reading` | IAM-A | warn |
| `program reading` | Met-A | warn |
| `M\s*=\s*30\.2` | M = 20.94 (30.2 only as the Landauer-unit conversion) | warn |
| `Mahaffey value` | Mahaffey number (M) for the drive ratio; informational fidelity ratio (A) for measured over floor | warn |
| `Mahaffey ratio` | Mahaffey number (M) | warn |
| `Aristotelian Principle` | the virial theorem (with IAM's thermodynamic completion); physics names only | block |
| `class H_min` | H_min (cell): the cell type's own reference floor (Met-A) or P_cell H(eps0) (IAM-A); H_min itself is current, only the class floor is retired | warn |
| `\bA[- ]score` | eps (qubit gate error in nats, -ln(1-p)) or A (reading / healthy or as-built reference) | warn |
| `the semiconductor report [Ii]ndex` | E_sw/(k_B T_j ln 2): distance above the Landauer floor; on the one gauge the chip as built reads 1 | warn |
| `[Ww]all ratio` | A = reading / device as built; the thermal floor lies to its left | warn |
| `\bA-floor` | H_min (the thermal floor) | warn |
| `zero adjustable parameters` | state which quantities are measured and which are assumed | warn |
