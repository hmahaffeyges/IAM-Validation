# VERIFY_BOOK_INVENTORY

Every displayed equation and every number the book prints that this inventory found, in `docs/book/main.tex` order, with its status label,
how `verify_book.py` checks it and the result of the run committed beside it (`verify_book_output.txt`).

How the inventory was built. Chapters from the front matter to the first half of *The cosmological constant* were read in full, line by
line, and every displayed equation and printed number was listed (source `read`). The remaining chapters were inventoried by a line scan of
every displayed equation, every table that carries a status label, and every number in a paragraph that carries a status label (source
`scan`). Numbers in sentences with no status label, and integers written in words or in running text, are not in the scan. Scan items were
then checked by hand chapter by chapter along the book's path (black holes, records, particles, devices, the cell, the status table, the
derivations appendix); a scan item that prints the same value, to five significant figures or more (four for a derived or calculated number
outside Part VI), as an item already checked reuses that check and says where it comes from.

Checked how: `sympy` = algebra, lhs - rhs simplifies to zero or a stated property holds; `numeric` = recomputed from first principles
(IAM's constants from `CANON/iam_canon.json`, published inputs written in the check); `file` = read from the committed file named;
`heavy` = read from a committed output that needs chains, CAMB or the methylation chain to regenerate (the command is in the check); `not run` = inventoried only (definition, input restated, conjecture, prediction, calibrated value, or a measured value whose source is listed in `SOURCES_NEEDED.md`).

Result: PASS, FAIL (each FAIL is listed in FOR_AUTHOR.md), or - (not run).

This file is written by `python3 docs/book/verify_book.py --inventory-md > docs/book/verification/VERIFY_BOOK_INVENTORY.md`.


Totals: 4664 PASS, 0 FAIL, 1341 inventoried and not run. Each run item carries the label of its check: `python3 docs/book/verify_book.py --label <label>` runs it alone.


## Part 0 - ch:p0_preface - `docs/book/front/p0_preface.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 53 |  | none | `72.26` | not run: H0_matter canon locked result, input | - |
| 54 |  | none | `67.16` | not run: H0_photon canon locked result, input | - |
| 55 |  | prediction | `-0.136` | not run: IAM prediction mu0, not reproducible | - |
| 59 | ch:p0_preface:L59 | observed | `67.36` | numeric: Planck 2018 H0 (published) | PASS |
| 59 | ch:p0_preface:L59:0.54 | observed | `0.54` | numeric: Planck 2018 H0 error (published) | PASS |
| 60 | ch:p0_preface:L60 | observed | `73.04` | numeric: SH0ES distance-ladder H0 (published) | PASS |
| 60 | ch:p0_preface:L60:1.04 | observed | `1.04` | numeric: SH0ES H0 error (published) | PASS |
| 60 | ch:p0_preface:L60:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0, Level 2 chain Run A posterior mean | PASS |
| 61 | ch:p0_preface:L61 | calc | `0.37` | numeric: sigma deviation of IAM photon H0 from Planck | PASS |
| 61 | ch:p0_preface:L61:0.75 | calc | `0.75` | numeric: sigma deviation of IAM matter H0 from Riess22 | PASS |
| 61 | ch:p0_preface:L61:72.26 | calc | `72.26` | numeric: matter-sector H0 = chain H0 x sqrt(1+beta_m) | PASS |
| 62 | ch:p0_preface:L62 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM-sector level-2 chain result | PASS |
| 62 | ch:p0_preface:L62:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 IAM matter-sector level-2 chain result | PASS |
| 65 | ch:p0_preface:L65 | calc | `-0.27` | numeric: half Delta-chi2 exponent for likelihood ratio | PASS |
| 65 | ch:p0_preface:L65:0.76 | calc | `0.76` | numeric: likelihood ratio exp(-Delta-chi2/2) | PASS |
| 65 |  | calc | `+0.54` | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 65 |  | calc | `+0.54` | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |

## Part 0 - ch:giants - `docs/book/front/p0_giants.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | ch:giants:L24 | observed | `153` | numeric: Timeline span, 2023 minus 1870 publication years | PASS |
| 41 | ch:giants:L41 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon sector matches Level2 chain value | PASS |
| 41 | ch:giants:L41:72.26 | calc | `72.26` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 55 | ch:giants:L55 | derived | `6.2\times10^{-8}` | numeric: Hawking temperature of one-solar-mass black hole | PASS |
| 58 | ch:giants:L58 | derived | `2.65\times10^{-30}` | numeric: de Sitter temperature today from current H0 | PASS |
| 79 | ch:giants:L79 | calc | `2.97\times10^{-21}` | numeric: Landauer bound energy per bit at 310.15K | PASS |
| 124 | ch:giants:L124 | calc | `7.5` | numeric: Diosi-Penrose time hbar/E_G, 1e-12 kg silica sphere | PASS |
| 125 | ch:giants:L125 | calc | `509` | numeric: IAM decoherence time, 1e-12 kg silica at 10 mK | PASS |
| 125 | ch:giants:L125:four | calc | `four` | numeric: doubling T multiplies tau_IAM by four, tau_DP unchanged | PASS |
| 133 | ch:giants:L133 | observed | `3.3` | numeric: Ohm 1961 excess system temperature (published) | PASS |
| 135 | ch:giants:L135 | observed | `3.5` | numeric: Penzias-Wilson excess antenna temperature (published) | PASS |
| 140 | ch:giants:L140 | observed | `10^5` | numeric: COBE DMR anisotropy, about one part in 10^5 | PASS |
| 140 | ch:giants:L140:2.7255 | observed | `2.7255` | numeric: CMB temperature (Fixsen 2009, published) | PASS |
| 176 | ch:giants:L176 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy from the PROC-CHANNEL-01 record | PASS |
| 177 | ch:giants:L177 | derived | `0.032` | numeric: Boltzmann floor computed from holding energy input | PASS |
| 205 |  | prediction |  | not run: statement, no numeric value | - |

## Part 0 - ch:p0_how_to_read - `docs/book/front/p0_how_to_read.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 50 | ch:p0_how_to_read:L50 | calc | `20.94` | numeric: Mahaffey number M_cell for one ATP at 37C | PASS |
| 50 | ch:p0_how_to_read:L50:30.2 | calc | `30.2` | numeric: M_cell converted to Landauer (bit) units | PASS |

## Part 1 - ch:iams_law - `docs/book/part1/p1_02_iams_law.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 72 | eq:law | none |  | not run: definition: IAM's Law cost per bit | - |
| 102 |  | none | `7/2` | not run: exponent value, derived in other section | - |
| 112 | eq:law_entangle | none |  | not run: definition: decoherence entangled state | - |
| 117 | eq:law_rhodiag | none |  | not run: definition: reduced density matrix via trace | - |
| 128 | eq:law_bits | none |  | not run: definition: information content formula | - |
| 133 | eq:law_landauer | none |  | not run: definition: Landauer's principle statement | - |
| 135 | ch:iams_law:L135 | derived | `0.0179` | numeric: Landauer cost per bit at 300K in eV | PASS |
| 150 | eq:epochcost | derived |  | sympy: substitute T_H(a) into Landauer formula | PASS |
| 152 | ch:iams_law:L152 | calc | `2.65\times10^{-30}` | numeric: Gibbons-Hawking horizon temperature today | PASS |
| 152 | ch:iams_law:L152:2.53\times10^{-53} | calc | `2.53\times10^{-53}` | numeric: Landauer cost per bit at horizon today | PASS |
| 152 |  | none | `67.16` | not run: input, H0_photon canon value restated | - |
| 172 | eq:law_virialchain | derived |  | sympy: virial+Landauer chain of equalities | PASS |
| 190 |  | conjecture | `1/2` | not run: claims partition necessity, not newly computed | - |
| 213 | eq:law_dQ | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 219 | eq:law_dS | derived |  | sympy: entropy change from horizon expansion theta | PASS |
| 228 | eq:law_jacobson | derived |  | sympy: 2pi/(hbar c eta) = 8 pi G/c^4 with G from the Bekenstein-Hawking eta = c^3/(4 hbar G) | PASS |
| 243 | eq:law_structure | derived |  | sympy: structural equation gives eta=c^3/4hbarG | PASS |
| 260 | eq:law_eta | derived |  | sympy: eta equals quarter inverse Planck area | PASS |
| 268 | eq:law_dAmin | derived |  | sympy: minimum horizon area equals 4 Planck areas | PASS |
| 270 | ch:iams_law:L270 | derived | `1` | sympy: eta from the structural identity times dA_min from the first law with one Unruh quantum is 1 nat | PASS |
| 270 | ch:iams_law:L270:2.77 | derived | `2.77` | numeric: one bit in Planck-area units (4 ln2) | PASS |
| 271 | ch:iams_law:L271 | derived | `4` | sympy: ratio of Euclidean and Einstein normalization factors | PASS |
| 288 | eq:law_Stotal | none |  | not run: definition: total entropy geometric plus informational | - |
| 314 | eq:law_F2 | derived |  | sympy: Friedmann acceleration eq from Cai-Kim first law | PASS |
| 321 | eq:firstlaw | none |  | not run: definition: first law with informational entropy added | - |
| 326 | eq:law_F2info | derived |  | sympy: modified Friedmann equation including record term | PASS |
| 329 | ch:iams_law:L329 | derived | `-1` | sympy: record term gives phantom equation of state w<-1 | PASS |
| 332 | eq:law_Sneed | derived |  | sympy: today's informational entropy rate per e-fold | PASS |
| 335 | ch:iams_law:L335 | calc | `5.2\times10^{121}` | numeric: informational bits produced per e-fold today | PASS |
| 344 | eq:Hm | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 353 | eq:rate | conjecture |  | not run: ansatz for information production rate, not derived | - |
| 358 | eq:Sdot | none |  | not run: text changed at HEAD; definition: entropy rate from information flux | - |
| 368 | ch:iams_law:L368 | derived | `1/2` | numeric: inflection point of activation function E(a) | PASS |
| 368 | ch:iams_law:L368:-1.67 | derived | `-1.67` | numeric: equation of state at a=0.5 | PASS |
| 368 | ch:iams_law:L368:-1.33 | derived | `-1.33` | numeric: equation of state at a=1 today | PASS |
| 368 | ch:iams_law:L368:-1.17 | derived | `-1.17` | numeric: equation of state at a=2 | PASS |
| 372 | eq:law_constraint | conjecture |  | not run: postulated decoherence constraint, a premise | - |
| 378 | eq:Ea | derived |  | sympy: integrates constraint to activation function E(a) | PASS |
| 382 | ch:iams_law:L382 | derived | `0` | sympy: activation function vanishes as a to 0 | PASS |
| 384 | ch:iams_law:L384 | derived |  | sympy: derivative of E(a) equals E/a^2, positive | PASS |
| 385 | ch:iams_law:L385 | derived | `2.718` | numeric: asymptotic value of activation function E(infinity) | PASS |
| 387 | ch:iams_law:L387 | calc | `2.30` | numeric: redshift where E reaches 10% of today | PASS |
| 387 | ch:iams_law:L387:0.69 | calc | `0.69` | numeric: redshift where E reaches 50% of today | PASS |
| 387 | ch:iams_law:L387:0.11 | calc | `0.11` | numeric: redshift where E reaches 90% of today | PASS |
| 387 | ch:iams_law:L387:-999 | calc | `-999` | numeric: exponent of E at recombination scale factor | PASS |
| 389 | ch:iams_law:L389 | derived | `1/2` | numeric: inflection point of E(a), repeated | PASS |
| 389 | ch:iams_law:L389:1 | derived | `1` | numeric: redshift at inflection point a=1/2 | PASS |
| 398 | eq:law_phi | none |  | not run: definition of scalar field phi from E(a) | - |
| 403 | eq:law_action | none |  | not run: definition of informational action functional | - |
| 409 | eq:law_L | none |  | not run: Lagrangian in FRW minisuperspace with record sector | - |
| 440 | eq:law_rhoinfo1 | none |  | not run: Definition of informational energy density at a=1 | - |
| 444 | eq:law_beta | none | `0.15765` | numeric: beta_m is half of Omega_m | PASS |
| 447 |  | prediction | `-0.13495` | not run: fixed chain input for mu0, not reproducible | - |
| 460 | eq:mu | none |  | sympy: mu(a) definition; mu<1 by positivity | PASS |
| 464 | eq:law_Sigma | none |  | not run: definition: Sigma=1 for photons | - |
| 466 | ch:iams_law:L466 | prediction | `0.8638` | numeric: mu at a=1 from beta_m | PASS |
| 466 | ch:iams_law:L466:-0.136 | prediction | `-0.136` | numeric: mu0 defined from beta_m | PASS |
| 466 | ch:iams_law:L466:-0.13618 | prediction | `-0.13618` | numeric: mu0 at beta_m=0.15765, precise | PASS |
| 467 | ch:iams_law:L467 | none | `13.62` | numeric: percent change of coupling today | PASS |
| 483 | eq:law_dphi | none |  | not run: definition: horizon-field perturbation vanishes | - |
| 496 | eq:law_poisson | none |  | not run: standard GR Poisson equation, restated | - |
| 497 | eq:law_noaniso | none |  | not run: standard GR no-anisotropic-stress condition | - |
| 498 | eq:law_growth | none |  | not run: matter growth equation with friction, definition | - |
| 508 | eq:law_muratio | none |  | sympy: mu as ratio of Hubble rates, mu<1 | PASS |
| 516 | ch:iams_law:L516 | calc | `0.8638` | numeric: mu(a) at z=0 | PASS |
| 516 | ch:iams_law:L516:0.9050 | calc | `0.9050` | numeric: mu(a) at z=0.2 | PASS |
| 516 | ch:iams_law:L516:0.9218 | calc | `0.9218` | numeric: mu(a) at z=0.3 | PASS |
| 516 | ch:iams_law:L516:0.9482 | calc | `0.9482` | numeric: mu(a) at z=0.5 | PASS |
| 516 | ch:iams_law:L516:0.9661 | calc | `0.9661` | numeric: mu(a) at z=0.7 | PASS |
| 516 | ch:iams_law:L516:0.9822 | calc | `0.9822` | numeric: mu(a) at z=1 | PASS |
| 516 | ch:iams_law:L516:0.9977 | calc | `0.9977` | numeric: mu(a) at z=2 | PASS |
| 516 | ch:iams_law:L516:0.9996 | calc | `0.9996` | numeric: mu(a) at z=3 | PASS |
| 517 | ch:iams_law:L517 | calc | `1.0759` | numeric: H_m/H at z=0 | PASS |
| 517 | ch:iams_law:L517:1.0512 | calc | `1.0512` | numeric: H_m/H at z=0.2 | PASS |
| 517 | ch:iams_law:L517:1.0415 | calc | `1.0415` | numeric: H_m/H at z=0.3 | PASS |
| 517 | ch:iams_law:L517:1.0270 | calc | `1.0270` | numeric: H_m/H at z=0.5 | PASS |
| 517 | ch:iams_law:L517:1.0174 | calc | `1.0174` | numeric: H_m/H at z=0.7 | PASS |
| 517 | ch:iams_law:L517:1.0090 | calc | `1.0090` | numeric: H_m/H at z=1 | PASS |
| 517 | ch:iams_law:L517:1.0012 | calc | `1.0012` | numeric: H_m/H at z=2 | PASS |
| 517 | ch:iams_law:L517:1.0002 | calc | `1.0002` | numeric: H_m/H at z=3 | PASS |
| 525 |  | none |  | not run: growth eq form (i), G_eff=mu*G, definition | - |
| 526 |  | none |  | not run: growth eq form (ii), friction on LCDM clock, definition | - |
| 527 |  | none |  | not run: growth eq form (iii), whole eq on H_m, definition | - |
| 529 | ch:iams_law:L529 | calc | `-0.78` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 529 | ch:iams_law:L529:-0.67 | calc | `-0.67` | numeric: Delta D/D today, form (ii): friction 2H_m, LambdaCDM clock | PASS |
| 529 | ch:iams_law:L529:-1.87 | calc | `-1.87` | numeric: Delta D/D today, form (iii): whole equation on H_m | PASS |
| 531 | ch:iams_law:L531 | calc | `4.25` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 531 | ch:iams_law:L531:2.17 | calc | `2.17` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 531 | ch:iams_law:L531:1.35 | calc | `1.35` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 531 | ch:iams_law:L531:0.41 | calc | `0.41` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 536 |  | calc | `0.13` | not run: not reproducible here: a CAMB TT-spectrum comparison (Cosmological_Physics/tests/iam_camb_full_boltzmann.py); the sentence itself says the spectra are not stored in the repository, so there is no committed output to read | - |
| 552 | ch:iams_law:L552 | none | `61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2b background-modification H0 chain output | PASS |
| 553 | ch:iams_law:L553 | measured | `\le0.010` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: largest final R-1 over all 18 chains | PASS |
| 561 | eq:law_dSdlna | none |  | sympy: exponent of a in dS/dlna scaling | PASS |
| 565 | eq:law_Sa | none |  | sympy: integral of a^(n-11/2), n != 9/2 | PASS |
| 569 | eq:law_n | calc | `7/2` | numeric: exponent n solving n-9/2=-1 | PASS |
| 573 | ch:iams_law:L573 | calc | `-1.02` | numeric: power of dS_info/dln a, n = 7/2, 0.01 <= a <= 0.1 | PASS |
| 573 | ch:iams_law:L573:-2.02 | calc | `-2.02` | numeric: power of dS_info/dln a, n = 5/2, 0.01 <= a <= 0.1 | PASS |
| 573 | ch:iams_law:L573:-1.52 | calc | `-1.52` | numeric: power of dS_info/dln a, n = 3, 0.01 <= a <= 0.1 | PASS |
| 573 | ch:iams_law:L573:-0.53 | calc | `-0.53` | numeric: power of dS_info/dln a, n = 4, 0.01 <= a <= 0.1 | PASS |
| 574 | ch:iams_law:L574 | calc | `-1.57` | numeric: power of dS_info/dln a, n = 7/2, Lambda era 0.25 <= a <= 1 | PASS |
| 581 | ch:iams_law:L581 | calc | `-2.02` | numeric: caption: fitted power, n = 5/2, 0.01 <= a <= 0.1 | PASS |
| 581 | ch:iams_law:L581:-1.52 | calc | `-1.52` | numeric: caption: fitted power, n = 3, 0.01 <= a <= 0.1 | PASS |
| 581 | ch:iams_law:L581:-1.02 | calc | `-1.02` | numeric: caption: fitted power, n = 7/2, 0.01 <= a <= 0.1 | PASS |
| 581 | ch:iams_law:L581:-0.53 | calc | `-0.53` | numeric: caption: fitted power, n = 4, 0.01 <= a <= 0.1 | PASS |
| 582 | ch:iams_law:L582 | calc | `0.864` | numeric: mu(z=0), fig caption rounded | PASS |
| 582 | ch:iams_law:L582:0.948 | calc | `0.948` | numeric: mu(z=0.5), fig caption rounded | PASS |
| 582 | ch:iams_law:L582:0.982 | calc | `0.982` | numeric: mu(z=1), fig caption rounded | PASS |
| 589 | ch:iams_law:L589 | derived |  | sympy: d ln nu/d ln D = -1 for nu = delta_c/(sigma_M D) | PASS |
| 590 | ch:iams_law:L590 | calc | `5.5` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up n_eff at z = 9, middle of the three mass functions | PASS |
| 590 | ch:iams_law:L590:7/2 | calc | `7/2` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up n_eff over z = 3-4, mean of three mass functions | PASS |
| 590 | ch:iams_law:L590:2 | calc | `2` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up n_eff at z = 1, mean of three mass functions | PASS |
| 591 | ch:iams_law:L591 | calc | `3.9` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: lowest model mean of n_eff over z = 2.3-9 | PASS |
| 591 | ch:iams_law:L591:4.3 | calc | `4.3` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: highest model mean of n_eff over z = 2.3-9 | PASS |
| 630 | eq:law_saturation | none |  | not run: holographic saturation condition, definition | - |
| 634 | eq:law_hoop | derived |  | sympy: S_BH/A at Schwarzschild radius equals holographic bound | PASS |
| 643 | eq:law_Meq | derived |  | sympy: equilibrium mass from T_BH=T_H | PASS |
| 645 | ch:iams_law:L645 | derived |  | sympy: ratio identity T_BH/T_H=Meq/M | PASS |
| 645 | ch:iams_law:L645:2.33\times10^{22} | calc | `2.33\times10^{22}` | numeric: equilibrium mass at H0=67.16 | PASS |
| 645 | ch:iams_law:L645:2.32\times10^{22} | calc | `2.32\times10^{22}` | numeric: equilibrium mass at H0=67.4 | PASS |
| 649 | eq:law_Gamma | none |  | sympy: Hawking bit-emission rate formula | PASS |
| 651 | ch:iams_law:L651 | calc | `152.5` | numeric: Hawking info rate for 1 solar mass | PASS |
| 669 |  | none | `1` | not run: trivial E(1)=exp(0) | - |
| 669 |  | none | `0` | not run: approx E(a) at a~1e-3, not exact | - |
| 670 | ch:iams_law:L670 | none |  | sympy: late-time limit E(a)->e | PASS |
| 681 | ch:iams_law:L681 | none | `0.15765` | numeric: beta_m defined as Omega_m/2 | PASS |
| 684 |  | none | `-0.13495` | not run: MGCAMB fixed amplitude, not derived here | - |
| 687 |  | none | `-0.13495` | not run: mu0 fixed amplitude repeated in caption | - |
| 692 | ch:iams_law:L692 | none | `0.15765` | numeric: beta_m repeated in table | PASS |
| 693 | ch:iams_law:L693 | calc | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 dchi2, IAM minus LCDM | PASS |
| 694 | ch:iams_law:L694 | calc | `+0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: min dchi2 across 4 L1 combos | PASS |
| 694 | ch:iams_law:L694:+1.73 | calc | `+1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: max dchi2 across 4 L1 combos | PASS |
| 695 | ch:iams_law:L695 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM Level2 chain value | PASS |
| 695 | ch:iams_law:L695:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 IAM Level2 chain value | PASS |
| 696 | ch:iams_law:L696 | calc | `-0.37` | numeric: sigma offset from Planck H0 | PASS |
| 696 | ch:iams_law:L696:67.36 | measured | `67.36` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 696 | ch:iams_law:L696:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 error, Level 2 Run A | PASS |
| 696 | ch:iams_law:L696:0.54 | observed | `0.54` | numeric: Planck 2018 H0 error (published) | PASS |
| 696 |  | none | `67.16` | not run: locked canon H0_photon, input | - |
| 697 | ch:iams_law:L697 | none | `72.26` | numeric: H0 matter-sector formula | PASS |
| 697 | ch:iams_law:L697:-0.75 | calc | `-0.75` | numeric: sigma offset from SH0ES | PASS |
| 697 | ch:iams_law:L697:73.04 | measured | `73.04` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 697 | ch:iams_law:L697:1.04 | measured | `1.04` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 698 |  | none | `-0.136` | not run: mu0 prediction restated in table | - |
| 701 | ch:iams_law:L701 | none | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 dchi2 repeated | PASS |
| 704 | ch:iams_law:L704 | derived | `1.076` | numeric: Hubble sector ratio sqrt(1+beta_m) | PASS |
| 705 |  | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 707 | ch:iams_law:L707 | calc | `72.26` | numeric: H0 photon to matter conversion | PASS |
| 712 | eq:law_sirens | prediction | `1.0759` | numeric: siren prediction intermediate factor | PASS |
| 712 | eq:law_sirens:72.26 | prediction | `72.26` | numeric: siren H0 prediction | PASS |
| 714 | ch:iams_law:L714 | observed | `70.0` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 714 | ch:iams_law:L714:12.0 | observed | `12.0` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 714 | ch:iams_law:L714:68.9 | observed | `68.9` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 714 | ch:iams_law:L714:8.0 | observed | `8.0` | numeric: GW170817 siren H0 lower error (published) | PASS |
| 714 |  | observed | `4.7` | not run: measured, source not named | - |
| 714 |  | observed | `4.6` | not run: measured, source not named | - |
| 715 | ch:iams_law:L715 | observed | `75.46` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: measured: printed value found in verify_iams_law_derivations_output.txt, a file the chapter names | PASS |
| 715 | ch:iams_law:L715:5.34 | observed | `5.34` | file `docs/verification/theory/IAM_LAW_CHECK.md`: measured: printed value found in IAM_LAW_CHECK.md, a file the chapter names | PASS |
| 715 | ch:iams_law:L715:5.39 | observed | `5.39` | file `docs/verification/theory/IAM_LAW_CHECK.md`: measured: printed value found in IAM_LAW_CHECK.md, a file the chapter names | PASS |
| 717 |  | none | `-0.136` | not run: mu0 prediction restated | - |
| 718 | ch:iams_law:L718 | observed | `0.11` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 (recorded value) | PASS |
| 718 | ch:iams_law:L718:0.45 | observed | `0.45` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 upper error (recorded value) | PASS |
| 718 | ch:iams_law:L718:0.54 | observed | `0.54` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 lower error (recorded value) | PASS |
| 730 | ch:iams_law:L730 | calc | `1.133\times10^{-123}` | numeric: Lambda/rho_vac identity at H0=67.4 | PASS |
| 733 | eq:law_ccbase | none |  | not run: baseline expression restated from ch:lambda | - |
| 735 | ch:iams_law:L735 | calc | `1.380\times10^{-123}` | numeric: baseline Lambda/rho_vac with Ob,Om | PASS |
| 735 | ch:iams_law:L735:1.22 | calc | `1.22` | numeric: ratio baseline to measured Lambda | PASS |
| 738 | eq:law_TdS | none | `0.8275` | numeric: de Sitter to Hubble temperature ratio | PASS |
| 740 | ch:iams_law:L740 | calc | `1.142\times10^{-123}` | numeric: corrected Lambda/rho_vac with sqrt(OmegaL) | PASS |
| 740 | ch:iams_law:L740:0.8\% | calc | `0.8\%` | numeric: percent offset corrected vs measured | PASS |
| 741 | ch:iams_law:L741 | derived |  | sympy: algebra: Ob/Om=(3/16)sqrt(OmegaL) | PASS |
| 742 | ch:iams_law:L742 | calc | `0.5\%` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: Ob/Om over (3/16)sqrt(OL) on the CMB-only chain, per cent | PASS |
| 742 | ch:iams_law:L742:0.7 | calc | `0.7` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same offset in units of its posterior error | PASS |
| 750 |  | none | `0.009273` | not run: chain convergence stat, no matching csv row | - |
| 752 | eq:law_eta_chain | fitted | `0.02232` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: baryon chain Ombh2 fitted value | PASS |
| 752 | eq:law_eta_chain:0.00014 | fitted | `0.00014` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: baryon chain Ombh2 uncertainty | PASS |
| 752 | eq:law_eta_chain:6.113\times10^{-10} | fitted | `6.113\times10^{-10}` | numeric: baryon-to-photon ratio from Ombh2 | PASS |
| 752 | eq:law_eta_chain:0.037\times10^{-10} | fitted | `0.037\times10^{-10}` | numeric: eta uncertainty from Ombh2 sd | PASS |
| 754 | ch:iams_law:L754 | measured | `6.117\times10^{-10}` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: min eta across 4 LCDM L1 chains | PASS |
| 754 | ch:iams_law:L754:6.137\times10^{-10} | none | `6.137\times10^{-10}` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: max eta across 4 LCDM L1 chains | PASS |
| 805 | ch:iams_law:L805 | derived |  | sympy: E(a) = exp(1 - 1/a) from integrating the record constraint | PASS |
| 806 |  | derived | `1` | not run: Sigma=1 sector rule, trivial restatement | - |
| 807 | ch:iams_law:L807 | derived | `7/2` | numeric: n = 7/2 from the horizon accounting | PASS |
| 808 | ch:iams_law:L808 | derived | `-0.136` | numeric: mu0 = mu(1) - 1 = -beta_m/(1+beta_m) | PASS |
| 840 | ch:iams_law:L840 | derived | `1/2` | sympy: virial ratio for converged 1/r potential | PASS |
| 841 | ch:iams_law:L841 | derived | `1.456` | numeric: Chandrasekhar mass, mu_e=2, m_u | PASS |
| 887 | ch:iams_law:L887 | derived | `2\pi/8\pi` | sympy: Jacobson eta=1/(4lP^2) ratio check | PASS |
| 921 |  | none |  | not run: restates law eq:law, definition | - |
| 948 |  | none | `1` | not run: trivial restatement E(1)=exp(0)=1 | - |
| 957 | ch:iams_law:L957 | calc | `20.94` | numeric: cell Mahaffey number | PASS |
| 957 | ch:iams_law:L957:30.2 | calc | `30.2` | numeric: cell Mahaffey number, Landauer units | PASS |
| 957 | ch:iams_law:L957:399 | calc | `399` | numeric: chip Mahaffey number | PASS |
| 957 | ch:iams_law:L957:2.1\times10^{77} | calc | `2.1\times10^{77}` | numeric: BH Mahaffey number, one solar mass | PASS |
| 957 | ch:iams_law:L957:3.9\times10^{90} | calc | `3.9\times10^{90}` | numeric: BH Mahaffey number, Sgr A* | PASS |
| 963 | eq:law_mahaffey | none |  | not run: defines Mahaffey number M | - |
| 979 | ch:iams_law:L979 | calc | `20.94` | numeric: cell Mahaffey number, repeat | PASS |
| 980 | ch:iams_law:L980 | calc | `30.2` | numeric: cell Mahaffey number Landauer, repeat | PASS |
| 996 |  | none | `7/2` | not run: definition, exponent value restated | - |

## Part 1 - ch:virial_law - `docs/book/part1/p1_03_virial_law.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 33 | eq:vl_euler | derived |  | sympy: Euler homogeneous-function relation, virial degree | PASS |
| 37 | eq:vl_virial | derived |  | sympy: virial relation 2K+V=0, E=-K for k=-1 | PASS |
| 44 | eq:vl_n | derived |  | sympy: virial relation T=n/2|V| for V~-r^-n | PASS |
| 69 | ch:virial_law:L69 | calc | `-13.6057` | numeric: hydrogen ground-state energy, Rydberg formula | PASS |
| 69 | ch:virial_law:L69:-27.2114 | calc | `-27.2114` | numeric: hydrogen potential energy V=2E1 | PASS |
| 69 |  | none | `13.6057` | not run: trivial sign restatement K=-E1 | - |
| 69 |  | none | `1/2` | not run: trivial ratio K/|V| by construction | - |
| 70 |  | none | `13.6057` | not run: trivial, photon energy equals kinetic half | - |
| 75 | ch:virial_law:L75 | calc | `1.0000000000` | sympy: virial ratio eta=-T/E identity | PASS |
| 77 |  | none | `-0.500000` | not run: input: H Hartree-Fock energy, cited | - |
| 77 |  | none | `0.500000` | not run: input: H Hartree-Fock |V|/T, cited | - |
| 77 |  | none | `1.000000` | not run: trivial ratio of equal cited numbers | - |
| 77 |  | none | `-128.547` | not run: input: Ne Hartree-Fock energy, cited | - |
| 77 |  | none | `128.547` | not run: input: Ne Hartree-Fock energy magnitude, cited | - |
| 77 |  | none | `-7232.138` | not run: input: Xe Hartree-Fock energy, cited | - |
| 77 |  | none | `7232.138` | not run: input: Xe Hartree-Fock energy magnitude, cited | - |
| 87 | ch:virial_law:L87 | calc | `-5.69\times10^{41}` | numeric: polytrope-3 gravitational binding energy, Sun | PASS |
| 88 | ch:virial_law:L88 | calc | `24` | numeric: Kelvin-Helmholtz timescale, rounded | PASS |
| 88 | ch:virial_law:L88:23.6 | calc | `23.6` | numeric: Kelvin-Helmholtz timescale, precise | PASS |
| 88 | ch:virial_law:L88:9.4 | calc | `9.4` | numeric: uniform-sphere contraction timescale | PASS |
| 94 | eq:vl_chandra | derived | `1.456` | numeric: Chandrasekhar mass formula | PASS |
| 94 |  | none | `2` | not run: input: mean molecular weight per electron mu_e | - |
| 96 |  | none | `2.01824` | not run: input: Chandrasekhar polytrope constant, cited | - |
| 97 |  | none | `1.33` | not run: measured: SDSS DR4 max white-dwarf mass, cited | - |
| 97 |  | none | `1.327` | not run: measured: ZTF J1901+1458 mass lower bound, cited | - |
| 97 |  | none | `1.365` | not run: measured: ZTF J1901+1458 mass upper bound, cited | - |
| 104 |  | none | `1.1` | not run: published N-body virial ratio lower bound, cited | - |
| 104 |  | none | `1.3` | not run: published N-body virial ratio upper bound, cited | - |
| 106 |  | none | `1.15` | not run: published virial ratio at 1e12 Msun, cited | - |
| 107 |  | none | `1.25` | not run: published virial ratio at 1e15 Msun, cited | - |
| 108 |  | none | `1.02` | not run: published surface-pressure virial ratio lower bound, cited | - |
| 108 |  | none | `1.17` | not run: published surface-pressure virial ratio upper bound, cited | - |
| 108 |  | none | `1.35` | not run: published relaxed-halo threshold, cited | - |
| 114 | eq:vl_bh | none |  | sympy: Bekenstein-Hawking entropy formula, definition | PASS |
| 118 | eq:vl_smarr | derived |  | sympy: Smarr relation T_H S = Mc^2/2 | PASS |
| 120 | ch:virial_law:L120 | calc | `1.51\times10^{77}` | numeric: information bits on 1-Msun horizon | PASS |
| 120 | ch:virial_law:L120:6.17\times10^{-8} | calc | `6.17\times10^{-8}` | numeric: Hawking temperature of 1-Msun black hole | PASS |
| 121 | ch:virial_law:L121 | calc | `0.5000000000` | sympy: Smarr ratio mass-independent, equals 1/2 | PASS |
| 121 |  | none | `4.3\times10^{6}` | not run: input: Sgr A* black-hole mass, cited | - |
| 121 |  | none | `6.5\times10^{9}` | not run: input: M87 black-hole mass, cited | - |
| 124 | ch:virial_law:L124 | calc | `0.433` | numeric: Kerr horizon energy share, chi=0.5 | PASS |
| 124 | ch:virial_law:L124:0.218 | calc | `0.218` | numeric: Kerr horizon energy share, chi=0.9 | PASS |
| 124 | ch:virial_law:L124:0.032 | calc | `0.032` | numeric: Kerr horizon energy share, chi=0.998 | PASS |
| 132 | eq:vl_beta | prediction | `0.15765` | numeric: cosmic coupling beta_m = Omega_m/2 | PASS |
| 132 |  | none | `0.3153` | not run: input: Planck 2018 Omega_m, cited | - |
| 135 | ch:virial_law:L135 | measured | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi2, IAM vs LCDM Level2 chains | PASS |
| 136 | ch:virial_law:L136 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 posterior mean, LCDM Level2 chain | PASS |
| 136 | ch:virial_law:L136:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 posterior mean, IAM Level2 chain | PASS |
| 137 | ch:virial_law:L137 | calc | `0.864` | numeric: mu(z=0) growth coupling from mu0 | PASS |
| 137 |  | none | `-0.136` | not run: input: IAM locked mu0 result, restated | - |
| 144 | ch:virial_law:L144 | derived | `13.606` | numeric: hydrogen |E| = alpha^2 m_e c^2/2 (infinite-mass Rydberg) | PASS |
| 144 | ch:virial_law:L144:27.211 | derived | `27.211` | numeric: hydrogen |V| = alpha^2 m_e c^2 | PASS |
| 144 | ch:virial_law:L144:0.5000000000 | calc | `0.5000000000` | sympy: Smarr ratio, caption repeat | PASS |
| 144 | ch:virial_law:L144:0.15765 | prediction | `0.15765` | numeric: beta_m prediction, caption repeat | PASS |
| 144 | ch:virial_law:L144:0.3166 | measured | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 Planck posterior Omega_m mean | PASS |
| 144 | ch:virial_law:L144:0.0065 | measured | `0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 Planck posterior Omega_m sd | PASS |
| 144 | ch:virial_law:L144:0.498 | calc | `0.498` | numeric: beta_m/Omega_m at Level2 posterior mean | PASS |
| 144 | ch:virial_law:L144:0.010 | calc | `0.010` | numeric: propagated sd of beta_m/Omega_m ratio | PASS |
| 144 |  | none | `1.1` | not run: published virial ratio lower bound, repeat | - |
| 144 |  | none | `1.3` | not run: published virial ratio upper bound, repeat | - |
| 144 |  | none | `1.02` | not run: surface-pressure ratio lower bound, repeat | - |
| 144 |  | none | `1.17` | not run: surface-pressure ratio upper bound, repeat | - |
| 144 |  | none | `1/2` | not run: trivial beta_m/Omega_m ratio by definition | - |
| 156 |  | conjecture | `0.3\%` | not run: conjecture: electron Compton-scale deviation | - |
| 157 | ch:virial_law:L157 | derived | `13.6057` | numeric: hydrogen |E| = alpha^2 m_e c^2/2 (infinite-mass Rydberg) | PASS |
| 157 | ch:virial_law:L157:27.2114 | derived | `27.2114` | numeric: hydrogen |V| = alpha^2 m_e c^2 | PASS |
| 157 |  | none | `1/2` | not run: trivial ratio restatement, table row | - |
| 158 | ch:virial_law:L158 | calc | `1.0000000000` | sympy: virial ratio identity, table repeat | PASS |
| 158 |  | none | `1/2` | not run: trivial T/|V| ratio restatement | - |
| 160 | ch:virial_law:L160 | derived | `1.456` | numeric: Chandrasekhar mass, table repeat | PASS |
| 160 |  | none | `2` | not run: input: mu_e, table repeat | - |
| 160 |  | none | `1.33` | not run: measured white-dwarf mass lower bound, repeat | - |
| 160 |  | none | `1.37` | not run: measured white-dwarf mass upper bound, rounded repeat | - |
| 161 | ch:virial_law:L161 | calc | `24` | numeric: Sun contraction timescale, table repeat | PASS |
| 163 |  | none | `1.1` | not run: published virial ratio lower bound, table repeat | - |
| 163 |  | none | `1.3` | not run: published virial ratio upper bound, table repeat | - |
| 163 |  | none | `1.02` | not run: surface-pressure ratio lower bound, table repeat | - |
| 163 |  | none | `1.17` | not run: surface-pressure ratio upper bound, table repeat | - |
| 164 | ch:virial_law:L164 | calc | `1/2` | sympy: Smarr relation, table repeat | PASS |
| 165 | ch:virial_law:L165 | measured | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi2, table repeat | PASS |
| 165 |  | none | `1/2` | not run: trivial beta_m/Omega_m ratio, table repeat | - |
| 173 | ch:virial_law:L173 | calc | `33` | numeric: orders of magnitude, atom to cluster | PASS |
| 173 | ch:virial_law:L173:thirty-seven | calc | `thirty-seven` | numeric: decades, atom to cosmic horizon | PASS |
| 173 | ch:virial_law:L173:36.4 | calc | `36.4` | numeric: decades, Bohr radius to c/H0 | PASS |

## Part 1 - ch:virial_identity - `docs/book/part1/p1_04_virial_identity.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 12 | ch:virial_identity:L12 | calc | `33` | numeric: orders of magnitude atom to cluster | PASS |
| 12 | ch:virial_identity:L12:37 | calc | `37` | numeric: decades 10^-11 m to 10^26 m (book inputs, ch:virial_law line 173) | PASS |
| 17 | ch:virial_identity:L17 | calc | `37` | numeric: decades 10^-11 m to 10^26 m (book inputs, ch:virial_law line 173) | PASS |
| 24 | eq:vi_virial | none |  | not run: virial theorem statement (Clausius 1870), definition | - |
| 44 | ch:virial_identity:L44 | calc | `-27.21` | numeric: hydrogen potential energy, 2x Rydberg | PASS |
| 44 | ch:virial_identity:L44:13.61 | calc | `13.61` | numeric: hydrogen kinetic energy (Rydberg) | PASS |
| 44 | ch:virial_identity:L44:-13.61 | calc | `-13.61` | numeric: hydrogen total energy | PASS |
| 44 | ch:virial_identity:L44:23.6 | calc | `23.6` | numeric: Sun Kelvin-Helmholtz contraction timescale | PASS |
| 54 | eq:vi_Q | derived |  | sympy: heat released, first law step | PASS |
| 61 | eq:vi_dS | none |  | not run: second law (Clausius inequality), premise | - |
| 67 | eq:vi_EL | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 77 | eq:vi_identity | derived |  | sympy: combines steps 1-4 into boxed identity | PASS |
| 89 | ch:virial_identity:L89 | calc | `13.6` | numeric: hydrogen binding energy, lower-precision repeat | PASS |
| 89 | ch:virial_identity:L89:-27.2 | calc | `-27.2` | numeric: hydrogen potential energy, repeat | PASS |
| 121 | ch:virial_identity:L121 | calc | `37` | numeric: decades 10^-11 m to 10^26 m (book inputs, ch:virial_law line 173) | PASS |
| 121 | ch:virial_identity:L121:33 | calc | `33` | numeric: decades spanned atom to cluster, repeat | PASS |
| 125 |  | fitted | `(2\pi)^{3/10}` | not run: fitted: the factor (2pi)^{3/10} was found by numerical search (the sentence says so); nothing to recompute here, the electron mass it gives is checked in ch:electronmass | - |
| 126 |  | conjecture | `0.3` | not run: electron-mass agreement precision, cited elsewhere | - |
| 129 | ch:virial_identity:L129 | calc | `1.0000000000` | numeric: virial ratio -T/E = 1 at the variational optimum (scaling argument) | PASS |
| 133 | ch:virial_identity:L133 | observed | `1.02` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 133 | ch:virial_identity:L133:1.17 | observed | `1.17` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 133 | ch:virial_identity:L133:1.1 | observed | `1.1` | file `docs/verification/virial/NBODY_TRACE.md`: lowest published 2T/|U| of simulated halos (Bett, Neto, Power) | PASS |
| 133 | ch:virial_identity:L133:1.3 | observed | `1.3` | file `docs/verification/virial/NBODY_TRACE.md`: highest published 2T/|U| of simulated halos (Bett, Neto, Power) | PASS |
| 139 | ch:virial_identity:L139 | calc | `0.15765` | numeric: beta_m from Omega_m/2 partition | PASS |
| 140 | ch:virial_identity:L140 | measured | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 posterior Omega_m mean | PASS |
| 140 | ch:virial_identity:L140:0.0065 | measured | `0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 posterior Omega_m std dev | PASS |
| 140 |  | none | `-0.13495` | not run: locked IAM mu0 input, fixed in chains | - |
| 141 | ch:virial_identity:L141 | calc | `0.498` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: beta_m over Omega_m ratio | PASS |
| 141 | ch:virial_identity:L141:0.010 | calc | `0.010` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: propagated uncertainty on beta_m/Omega_m | PASS |
| 141 | ch:virial_identity:L141:0.54 | measured | `0.54` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: growth chi2 difference vs LCDM baseline | PASS |
| 151 | ch:virial_identity:L151 | observed | `33` | numeric: orders of magnitude tested, repeat | PASS |
| 159 | ch:virial_identity:L159 | calc | `37` | numeric: decades 10^-11 m to 10^26 m (book inputs, ch:virial_law line 173) | PASS |

## Part 2 - ch:virial - `docs/book/part2/p2_02_virial.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 | ch:virial:L19 | none | `0.15765` | numeric: beta_m preview value, Om/2 | PASS |
| 39 | eq:vc_firstlaw | none |  | not run: definition: IAM horizon first law extension | - |
| 49 | eq:vc_beta | prediction | `0.15765` | numeric: beta_m=Om/2, stated prediction | PASS |
| 64 | ch:virial:L64 | measured | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 posterior Om_m mean | PASS |
| 64 | ch:virial:L64:0.498 | calc | `0.498` | numeric: beta_m/Om_m ratio for fixed beta_m | PASS |
| 65 | ch:virial:L65 | measured | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2_min excess of IAM vs LCDM, L2 chains | PASS |
| 70 | eq:vc_decompose | none |  | not run: definition: coupling decomposition Om*fcoll*etavir | - |
| 75 |  | none | `0.62` | not run: input f_coll value used in eq:vc_eta | - |
| 76 | eq:vc_eta | none | `0.81` | numeric: eta_vir = 1/(2 f_coll) | PASS |
| 85 | ch:virial:L85 | observed | `17\%` | file `docs/verification/virial/NBODY_TRACE.md`: largest surface-corrected excess of 2K/|W| over 1 (Klypin 2016) | PASS |
| 86 | ch:virial:L86 | calc | `0.77` | file `docs/verification/virial/NBODY_TRACE.md`: reciprocal |U|/2T, lower end, within r_vir | PASS |
| 86 | ch:virial:L86:0.89 | calc | `0.89` | file `docs/verification/virial/NBODY_TRACE.md`: reciprocal |U|/2T, upper end, within r_vir | PASS |
| 87 | ch:virial:L87 | calc | `0.85` | numeric: reciprocal of Klypin corrected ratio | PASS |
| 88 | ch:virial:L88 | calc | `0.98` | numeric: reciprocal of Klypin corrected ratio | PASS |
| 88 | ch:virial:L88:1.1 | calc | `1.1` | numeric: reciprocal of Power surface-corrected eta' | PASS |
| 88 |  | none | `0.9` | not run: restates Power et al. table value | - |
| 97 | ch:virial:L97 | observed | `-0.2` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007: ridge of 2T/U + 1, upper | PASS |
| 97 | ch:virial:L97:-0.3 | observed | `-0.3` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007: ridge of 2T/U + 1, lower | PASS |
| 97 | ch:virial:L97:1.2 | calc | `1.2` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007: 2T/|U| from the ridge -0.2 | PASS |
| 97 | ch:virial:L97:1.3 | calc | `1.3` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007: 2T/|U| from the ridge -0.3 | PASS |
| 97 | ch:virial:L97:0.5 | calc | `0.5` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007 quasi-equilibrium cut, lower edge of 2T/|U| | PASS |
| 97 | ch:virial:L97:1.5 | calc | `1.5` | file `docs/verification/virial/NBODY_TRACE.md`: Bett 2007 quasi-equilibrium cut, upper edge of 2T/|U| | PASS |
| 98 | ch:virial:L98 | observed | `1.12` | file `docs/verification/virial/NBODY_TRACE.md`: measured: printed value found in NBODY_TRACE.md, a file the chapter names | PASS |
| 98 | ch:virial:L98:1.26 | observed | `1.26` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 98 | ch:virial:L98:1.35 | observed | `1.35` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: Neto 2007 relaxation cut read from verify_virial_papers_output.txt, a file the chapter names | PASS |
| 99 | ch:virial:L99 | observed | `1.15` | file `docs/verification/virial/NBODY_TRACE.md`: measured: Power 2012 eta at 10^12 read from NBODY_TRACE.md, a file the chapter names | PASS |
| 99 | ch:virial:L99:1.25 | observed | `1.25` | file `docs/verification/virial/NBODY_TRACE.md`: measured: printed value found in NBODY_TRACE.md, a file the chapter names | PASS |
| 99 | ch:virial:L99:0.9 | observed | `0.9` | file `docs/verification/virial/NBODY_TRACE.md`: Power 2012: surface-corrected eta' centre | PASS |
| 100 | ch:virial:L100 | observed | `1.02` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 100 | ch:virial:L100:1.17 | observed | `1.17` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 100 | ch:virial:L100:1.1 | calc | `1.1` | file `docs/verification/virial/NBODY_TRACE.md`: Klypin 2016: 2K/|W| at 10^12 | PASS |
| 100 | ch:virial:L100:1.4 | calc | `1.4` | file `docs/verification/virial/NBODY_TRACE.md`: Klypin 2016: 2K/|W| at 10^15 | PASS |
| 101 | ch:virial:L101 | observed | `1.3` | file `docs/verification/virial/NBODY_TRACE.md`: Ludlow 2010 relaxation cut on 2K/|Phi| | PASS |
| 102 | ch:virial:L102 | observed | `0.82` | file `docs/verification/virial/NBODY_TRACE.md`: Bryan & Norman 1998: f_sigma, lower | PASS |
| 102 | ch:virial:L102:0.89 | observed | `0.89` | file `docs/verification/virial/NBODY_TRACE.md`: Bryan & Norman 1998: f_sigma, upper | PASS |
| 102 | ch:virial:L102:0.75 | observed | `0.75` | file `docs/verification/virial/NBODY_TRACE.md`: Bryan & Norman 1998: f_T, lower | PASS |
| 102 | ch:virial:L102:0.79 | observed | `0.79` | file `docs/verification/virial/NBODY_TRACE.md`: Bryan & Norman 1998: f_T, upper | PASS |
| 108 |  | calc | `0.5` | not run: rough statement: the book says the collapsed-fraction slope is about 0.5 at 10^10 h^-1 Msun; printed value 0.5 has one significant digit and the six mass functions of NBODY_TRACE_massfunction_slopes.csv give 0.48 to 0.59 (median 0.525), so a 5 % change (0.525) cannot be told from rounding or from the spread | - |
| 108 |  | calc | `3` | not run: rough statement: the book says the slope is about 3 at 10^14 h^-1 Msun; printed value 3 has one significant digit and the six mass functions of NBODY_TRACE_massfunction_slopes.csv give 3.02 to 4.01 (median 3.21), so a 5 % change (3.15) cannot be told from rounding or from the spread | - |
| 125 | ch:virial:L125 | calc | `0.486` | numeric: Tinker 2008 collapsed fraction above 10^10.5 h^-1 Msun | PASS |
| 126 | ch:virial:L126 | calc | `0.447` | numeric: Tinker 2008 collapsed fraction above 10^11 | PASS |
| 126 | ch:virial:L126:0.348 | calc | `0.348` | numeric: Tinker 2008 collapsed fraction above 10^12 | PASS |
| 126 | ch:virial:L126:8.2 | calc | `8.2` | numeric: log10 M_min where the Tinker collapsed fraction reaches 0.62 | PASS |
| 126 |  | calc | `0.62` | not run: input: f_coll = 0.62 restated from Eq. vc_eta (book line 75), the target value whose M_min is then computed (checked in ch:virial:L126:8.2) | - |
| 139 | eq:vc_n | derived |  | sympy: matter-domination exponent n=7/2 from n-9/2=-1 | PASS |
| 143 | ch:virial:L143 | calc | `7/2` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 143 | ch:virial:L143:2\% | calc | `2\%` | numeric: coefficient of 1/a in the fitted record, offset from 1 | PASS |
| 143 | ch:virial:L143:7\% | calc | `7\%` | numeric: constant of the fitted record, offset from 1 | PASS |
| 148 | ch:virial:L148:1 | calc | `1` | file `docs/verification/virial/NBODY_TRACE_massfunction_slopes.csv`: collapse-fraction log-slope above 10^12 | PASS |
| 148 |  | calc | `0.5` | not run: rough statement: about 0.5 above 10^10 h^-1 Msun; printed value 0.5 has one significant digit and the six mass functions of NBODY_TRACE_massfunction_slopes.csv give 0.48 to 0.59 (median 0.525), so a 5 % change (0.525) cannot be told from rounding or from the spread | - |
| 148 |  | calc | `3` | not run: rough statement: about 3 above 10^14 h^-1 Msun; printed value 3 has one significant digit and the six mass functions of NBODY_TRACE_massfunction_slopes.csv give 3.02 to 4.01 (median 3.21), so a 5 % change (3.15) cannot be told from rounding or from the spread | - |
| 151 | ch:virial:L151 | calc | `7/2` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up n_eff passes through 7/2 at z = 3-4 | PASS |
| 151 | ch:virial:L151:3 | calc | `3` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: lower edge of the z window of the 7/2 crossings | PASS |
| 151 | ch:virial:L151:4 | calc | `4` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: upper edge of the z window of the 7/2 crossings | PASS |
| 156 | eq:vc_E | derived |  | sympy: activation function E(a)=e^{-z} identity | PASS |
| 159 | ch:virial:L159 | derived | `0` | sympy: E(a) -> 0 as a -> 0 | PASS |
| 159 | ch:virial:L159:4.5\times10^{-5} | derived | `4.5\times10^{-5}` | numeric: E at z=10 | PASS |
| 159 | ch:virial:L159:1 | derived | `1` | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 160 | ch:virial:L160 | derived |  | sympy: dE/da positive for all a>0 | PASS |
| 160 | ch:virial:L160:3 | derived |  | sympy: E(infty)=e asymptotic limit | PASS |
| 161 | ch:virial:L161 | calc | `2.30` | numeric: redshift where E=10% of today | PASS |
| 161 | ch:virial:L161:0.69 | calc | `0.69` | numeric: redshift where E=50% of today | PASS |
| 161 | ch:virial:L161:0.11 | calc | `0.11` | numeric: redshift where E=90% of today | PASS |
| 162 | ch:virial:L162 | calc |  | sympy: dE/dlna=E/a peaks at a=1 | PASS |
| 162 | ch:virial:L162:3 | calc |  | sympy: E has inflection at a=1/2 | PASS |
| 165 | part2:eq:mu_virial | none |  | not run: definition: modified growth ODE and mu(a) function | - |
| 169 | ch:virial:L169 | none | `0.864` | numeric: mu(1)=1/(1+beta_m) today | PASS |
| 169 | ch:virial:L169:-0.136 | none | `-0.136` | numeric: mu0 = mu(1)-1 | PASS |
| 170 | ch:virial:L170 | derived | `1` | numeric: Sigma = 1 from Phi = Psi and the unmodified Poisson equation | PASS |
| 181 | ch:virial:L181 | derived |  | sympy: E inflection at a=1/2, fig caption repeat | PASS |
| 181 | ch:virial:L181:3 | derived |  | sympy: E tends to e, fig caption repeat | PASS |
| 181 | ch:virial:L181:13.6\% | calc | `13.6\%` | numeric: today's mu coupling deficit 1-mu(1) | PASS |
| 181 | ch:virial:L181:7.8\% | calc | `7.8\%` | numeric: mu coupling deficit at z=0.3 | PASS |
| 181 | ch:virial:L181:4.25\% | calc | `4.25\%` | numeric: fsigma8 deficit today vs LCDM growth ODE | PASS |
| 181 | ch:virial:L181:0 | derived | `0` | numeric: E(a) -> 0 at early times | PASS |
| 181 | ch:virial:L181:1 | derived | `1` | numeric: E(1) = 1 today | PASS |
| 190 | ch:virial:L190 | calc | `1.000` | numeric: E(a) at z=0 table row | PASS |
| 190 | ch:virial:L190:0.864 | calc | `0.864` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 190 | ch:virial:L190:13.6\% | calc | `13.6\%` | numeric: coupling deficit 1-mu at z=0 | PASS |
| 191 | ch:virial:L191 | calc | `0.741` | numeric: E(a) at z=0.3 | PASS |
| 191 | ch:virial:L191:0.922 | calc | `0.922` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 191 | ch:virial:L191:7.8\% | calc | `7.8\%` | numeric: coupling deficit 1-mu at z=0.3 | PASS |
| 192 | ch:virial:L192 | calc | `0.607` | numeric: E(a) at z=0.5 | PASS |
| 192 | ch:virial:L192:0.948 | calc | `0.948` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 192 | ch:virial:L192:5.2\% | calc | `5.2\%` | numeric: coupling deficit 1-mu at z=0.5 | PASS |
| 193 | ch:virial:L193 | calc | `0.497` | numeric: E(a) at z=0.7 | PASS |
| 193 | ch:virial:L193:0.966 | calc | `0.966` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 193 | ch:virial:L193:3.4\% | calc | `3.4\%` | numeric: coupling deficit 1-mu at z=0.7 | PASS |
| 194 | ch:virial:L194 | calc | `0.368` | numeric: E(a) at z=1.0 | PASS |
| 194 | ch:virial:L194:0.982 | calc | `0.982` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 194 | ch:virial:L194:1.8\% | calc | `1.8\%` | numeric: coupling deficit 1-mu at z=1.0 | PASS |
| 195 | ch:virial:L195 | calc | `0.223` | numeric: E(a) at z=1.5 | PASS |
| 195 | ch:virial:L195:0.994 | calc | `0.994` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 195 | ch:virial:L195:0.6\% | calc | `0.6\%` | numeric: coupling deficit 1-mu at z=1.5 | PASS |
| 196 | ch:virial:L196 | calc | `0.135` | numeric: E(a) at z=2.0 | PASS |
| 196 | ch:virial:L196:0.998 | calc | `0.998` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 196 | ch:virial:L196:0.2\% | calc | `0.2\%` | numeric: coupling deficit 1-mu at z=2.0 | PASS |
| 199 | ch:virial:L199 | calc | `0.864` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 203 | ch:virial:L203 | calc | `4.25\%` | numeric: f sigma8 deficit today, mu-Sigma form | PASS |
| 203 | ch:virial:L203:2.19\% | calc | `2.19\%` | numeric: f sigma8 deficit at z = 0.295 | PASS |
| 203 | ch:virial:L203:2.17\% | calc | `2.17\%` | numeric: f sigma8 deficit at z = 0.3 | PASS |
| 203 | ch:virial:L203:1.35\% | calc | `1.35\%` | numeric: f sigma8 deficit at z = 0.5 | PASS |
| 204 | ch:virial:L204 | calc | `0.41\%` | numeric: f sigma8 deficit at z = 1 | PASS |
| 204 | ch:virial:L204:0.13\% | calc | `0.13\%` | numeric: f sigma8 deficit at z = 1.491 | PASS |
| 204 | ch:virial:L204:0.4\% | calc | `0.4\%` | numeric: Level 2 (matter-rate) form against mu-Sigma form, largest f sigma8 gap | PASS |
| 206 | eq:vc_fs8 | prediction |  | not run: definition of predicted fsigma8 shape | - |
| 212 | ch:virial:L212 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM chain sigma8 | PASS |
| 212 | ch:virial:L212:0.0059 | measured | `0.0059` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM chain sigma8 sd | PASS |
| 212 | ch:virial:L212:0.7998 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 IAM chain sigma8 | PASS |
| 212 | ch:virial:L212:0.0058 | measured | `0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 IAM chain sigma8 sd | PASS |
| 212 | ch:virial:L212:0.814 | measured | `0.814` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 LCDM chain sigma8 | PASS |
| 212 | ch:virial:L212:0.802 | measured | `0.802` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 IAM chain sigma8 | PASS |
| 213 | ch:virial:L213 | calc | `-1.51\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 shift Level2 chains | PASS |
| 213 | ch:virial:L213:-0.78\sigma | calc | `-0.78\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 shift Level2 chains | PASS |
| 214 | ch:virial:L214 | calc | `-0.07\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: omega_b shift Level2 chains | PASS |
| 214 | ch:virial:L214:+0.05\sigma | calc | `+0.05\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Omega_m shift Level2 chains | PASS |
| 214 | ch:virial:L214:+0.09\sigma | measured | `+0.09\sigma` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: Level 2 shift of ln(10^10 A_s), IAM minus LambdaCDM | PASS |
| 224 | ch:virial:L224 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A sigma8 | PASS |
| 224 | ch:virial:L224:0.0058 | measured | `0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A sigma8 sd | PASS |
| 224 | ch:virial:L224:0.822 | measured | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A S8 | PASS |
| 224 | ch:virial:L224:0.011 | measured | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A S8 sd | PASS |
| 224 | ch:virial:L224:67.16 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A H0 | PASS |
| 224 | ch:virial:L224:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A H0 sd | PASS |
| 224 | ch:virial:L224:+0.54 | calc | `+0.54` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi2 Run A vs Run C | PASS |
| 224 | ch:virial:L224:-0.136 | derived | `-0.136` | numeric: mu0 of Run A, derived from beta_m | PASS |
| 225 | ch:virial:L225 | measured | `0.7995` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D sigma8 | PASS |
| 225 | ch:virial:L225:0.0058 | measured | `0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D sigma8 sd | PASS |
| 225 | ch:virial:L225:0.821 | measured | `0.821` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D S8 | PASS |
| 225 | ch:virial:L225:0.011 | measured | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D S8 sd | PASS |
| 225 | ch:virial:L225:67.19 | measured | `67.19` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D H0 | PASS |
| 225 | ch:virial:L225:0.46 | measured | `0.46` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D H0 sd | PASS |
| 225 | ch:virial:L225:-0.136 | derived | `-0.136` | numeric: mu0 of Run D, derived from beta_m | PASS |
| 226 | ch:virial:L226 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C sigma8 | PASS |
| 226 | ch:virial:L226:0.0059 | measured | `0.0059` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C sigma8 sd | PASS |
| 226 | ch:virial:L226:0.830 | measured | `0.830` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C S8 | PASS |
| 226 | ch:virial:L226:0.011 | measured | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C S8 sd | PASS |
| 226 | ch:virial:L226:67.19 | measured | `67.19` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C H0 | PASS |
| 226 | ch:virial:L226:0.46 | measured | `0.46` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C H0 sd | PASS |
| 227 | ch:virial:L227 | measured | `0.8015` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 IAM fixed sigma8 | PASS |
| 227 | ch:virial:L227:0.0058 | measured | `0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 IAM fixed sigma8 sd | PASS |
| 227 | ch:virial:L227:67.08 | measured | `67.08` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 IAM fixed H0 | PASS |
| 227 |  | none | `-0.135` | not run: coupling fixed value, MGCAMB restated | - |

## Part 2 - ch:virial_tests - `docs/book/part2/p2_02b_virial_tests.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 11 |  | none | `-0.136` | not run: mu0 locked IAM value restated | - |
| 11 |  | none | `0` | not run: Sigma0 locked value restated | - |
| 12 |  | none | `67.16` | not run: H0 photon-sector locked value restated | - |
| 12 |  | none | `72.26` | not run: H0 matter-sector locked value restated | - |
| 26 | ch:virial_tests:L26 | calc | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: IAM sigma8 prediction from Level2 chain | PASS |
| 26 | ch:virial_tests:L26:0.0058 | calc | `0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 chain standard deviation | PASS |
| 26 | ch:virial_tests:L26:0.802 | observed | `0.802` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 26 | ch:virial_tests:L26:0.022 | observed | `0.022` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: joint sigma8 upper error (KiDS-Legacy + DES Y3 + DESI + Pantheon+) | PASS |
| 26 | ch:virial_tests:L26:0.018 | observed | `0.018` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: joint sigma8 lower error | PASS |
| 26 |  | calc | `0.12` | not run: printed value 0.12 sigma has two significant digits: recomputed |0.79975 - 0.802|/0.018 = 0.1249 lies at the rounding edge, within 0.0011 of the value moved by 5 % (0.126), so a 5 % change cannot be told from rounding (the committed verify_entropic_gravity_output.txt prints the same -0.12 sigma) | - |
| 27 | ch:virial_tests:L27 | calc | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: IAM S8 prediction from Level2 chain | PASS |
| 27 | ch:virial_tests:L27:0.011 | calc | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 chain standard deviation | PASS |
| 27 | ch:virial_tests:L27:0.815 | observed | `0.815` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 27 | ch:virial_tests:L27:0.33 | calc | `0.33` | numeric: S8 Level 2 vs KiDS-Legacy: difference over the combined error (chain sd, KiDS upper error) | PASS |
| 27 | ch:virial_tests:L27:0.016 | observed | `0.016` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: KiDS-Legacy S8 upper error | PASS |
| 27 | ch:virial_tests:L27:0.021 | observed | `0.021` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: KiDS-Legacy S8 lower error | PASS |
| 28 | ch:virial_tests:L28 | calc | `72.26` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 28 | ch:virial_tests:L28:73.04 | observed | `73.04` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 28 | ch:virial_tests:L28:1.04 | observed | `1.04` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 28 | ch:virial_tests:L28:0.75 | calc | `0.75` | numeric: H0 matter-sector tension in sigma | PASS |
| 29 | ch:virial_tests:L29 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: IAM H0 photon prediction from Level2 chain | PASS |
| 29 | ch:virial_tests:L29:0.47 | calc | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon chain standard deviation | PASS |
| 29 | ch:virial_tests:L29:67.36 | observed | `67.36` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 29 | ch:virial_tests:L29:0.37 | calc | `0.37` | numeric: H0 photon-sector tension in sigma | PASS |
| 29 | ch:virial_tests:L29:0.54 | observed | `0.54` | numeric: Planck 2018 H0 error (published) | PASS |
| 30 | ch:virial_tests:L30 | calc | `0.299` | numeric: growth-only Omega_m = Omega_m mu(z) at z = 0.5 | PASS |
| 31 | ch:virial_tests:L31 | calc | `-2.2` | numeric: f sigma8 ramp over the six DESI DR1 bins, deepest | PASS |
| 31 | ch:virial_tests:L31:-0.1 | calc | `-0.1` | numeric: f sigma8 ramp over the six DESI DR1 bins, shallowest | PASS |
| 31 |  | observed | `8` | not run: measured, source not named | - |
| 31 |  | observed | `13` | not run: measured, source not named | - |
| 32 | ch:virial_tests:L32 | calc | `-0.136` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 32 | ch:virial_tests:L32:0 | calc | `0` | numeric: Sigma_0 = Sigma - 1 = 0 | PASS |
| 32 | ch:virial_tests:L32:0.11 | observed | `0.11` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 (recorded value) | PASS |
| 32 | ch:virial_tests:L32:0.45 | observed | `0.45` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 upper error (recorded value) | PASS |
| 32 | ch:virial_tests:L32:0.54 | observed | `0.54` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 lower error (recorded value) | PASS |
| 39 | ch:virial_tests:L39 | calc | `4.52` | numeric: diagonal chi2, LambdaCDM, six DESI DR1 ShapeFit bins | PASS |
| 39 | ch:virial_tests:L39:5.14 | calc | `5.14` | numeric: diagonal chi2, IAM (MGCAMB form), six DESI DR1 ShapeFit bins | PASS |
| 39 | ch:virial_tests:L39:6.20 | calc | `6.20` | numeric: diagonal chi2, LambdaCDM, SDSS DR16 six points | PASS |
| 39 | ch:virial_tests:L39:6.96 | calc | `6.96` | numeric: diagonal chi2, IAM (MGCAMB form), SDSS DR16 six points | PASS |
| 56 | eq:vt_eg | none |  | not run: definition of observational E_G statistic | - |
| 62 | eq:vt_egiam | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 66 |  | prediction | `+3.6` | not run: E_G enhancement today, growth curves elsewhere | - |
| 66 |  | prediction | `+1.8` | not run: E_G enhancement at z=0.3, growth curves elsewhere | - |
| 66 |  | prediction | `+1.1` | not run: E_G enhancement at z=0.5, growth curves elsewhere | - |
| 74 | eq:vt_R | none |  | not run: definitions of cluster ratios R1,R2,R3 | - |
| 87 | ch:virial_tests:L87 | prediction | `0.864` | numeric: mu value from 1+mu0 | PASS |
| 87 |  | prediction | `1.000` | not run: Sigma unmodified, trivial definition | - |
| 88 | ch:virial_tests:L88 | prediction | `14` | numeric: matter-to-photon discrepancy percent from mu0 | PASS |
| 96 |  | prediction | `4.25` | not run: fsigma8 ramp at z=0, eq:vc_fs8 elsewhere | - |
| 96 |  | prediction | `2.19` | not run: fsigma8 ramp at z=0.295, eq:vc_fs8 elsewhere | - |
| 96 |  | prediction | `1.35` | not run: fsigma8 ramp at z=0.5, eq:vc_fs8 elsewhere | - |
| 96 |  | prediction | `0.41` | not run: fsigma8 ramp at z=1, eq:vc_fs8 elsewhere | - |
| 97 |  | prediction | `0.13` | not run: fsigma8 ramp at z=1.491, eq:vc_fs8 elsewhere | - |
| 102 | ch:virial_tests:L102 | observed | `0.35` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 phantom-crossing redshift, lowest | PASS |
| 102 | ch:virial_tests:L102:0.5 | observed | `0.5` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 phantom-crossing redshift, highest | PASS |
| 111 | ch:virial_tests:L111 | calc | `5.1` | numeric: growth-only Omega_m below geometric at z = 0.51, per cent | PASS |
| 129 |  | none | `1/2` | not run: virial theorem product restated, input | - |
| 134 |  | interp | `-0.136` | not run: mu0 locked value, repeat | - |
| 134 |  | interp | `0` | not run: Sigma0 locked value, repeat | - |
| 137 |  | interp | `-0.136` | not run: mu0 locked value, repeat | - |
| 137 |  | interp | `0` | not run: Sigma0 locked value, repeat | - |
| 138 |  | interp | `0` | not run: mu0 falsification threshold, repeat | - |
| 147 |  | interp | `-0.136` | not run: mu0 locked value, repeat | - |
| 149 | ch:virial_tests:L149 | calc | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 restated rounded, chain repeat | PASS |
| 149 | ch:virial_tests:L149:0.1 | calc | `0.1` | numeric: sigma8 tension restated, repeat | PASS |
| 151 | ch:virial_tests:L151 | prediction | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 prediction restated, repeat | PASS |
| 151 | ch:virial_tests:L151:67.16 | prediction | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon prediction restated, repeat | PASS |
| 151 |  | prediction | `-0.136` | not run: mu0 locked prediction, repeat | - |
| 151 |  | prediction | `0` | not run: Sigma0 locked prediction, repeat | - |
| 152 |  | prediction | `72.26` | not run: H0 matter prediction restated, repeat | - |

## Part 2 - ch:theory - `docs/book/part2/p2_03_theory.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 23 | ch:theory:L23 | derived | `0.864` | numeric: mu(z=0) from locked mu0 | PASS |
| 23 |  | none | `-0.136` | not run: mu0 canon locked value restated | - |
| 29 | ch:theory:L29 | derived | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Δχ² IAM vs ΛCDM full Planck L2 chains | PASS |
| 53 |  | none | `-0.136` | not run: mu0 canon value repeated | - |
| 101 | ch:theory:L101 | interp | `2.718` | numeric: asymptote of E(a)=exp(1-1/a) as a->infty | PASS |
| 108 | eq:th:entangle | none |  | not run: defines system-environment entanglement (definition) | - |
| 110 |  | none |  | not run: defines reduced density matrix via partial trace (definition) | - |
| 113 | eq:th:diag | derived |  | sympy: diagonal limit from orthogonal env states | PASS |
| 120 |  | none |  | not run: defines information content I=log2N (definition) | - |
| 122 | eq:th:landauer | none |  | not run: states Landauer bound (cited definition) | - |
| 141 | eq:th:unruh | none |  | not run: states Unruh temperature formula (cited definition) | - |
| 143 | eq:th:dQ | derived |  | sympy: heat flux via chi^a, dSigma^b substitution | PASS |
| 147 | eq:th:dS | derived |  | sympy: entropy variation from S=etaA ansatz | PASS |
| 151 | eq:th:raych | none |  | not run: states Raychaudhuri equation (cited definition) | - |
| 154 | eq:th:dA | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 156 | eq:th:clausius | derived |  | sympy: imposes Clausius relation with prior substitutions | PASS |
| 159 | eq:th:Tab | derived |  | sympy: solves T_ab via null-vector generality | PASS |
| 162 | eq:th:einstein | derived |  | sympy: Einstein eq coefficient G=c^3/(4 hbar eta) | PASS |
| 183 |  | none | `2\pi/8\pi` | not run: trivial ratio simplifies to 1/4 | - |
| 190 | eq:th:period | derived |  | sympy: period fixed by removing conical deficit | PASS |
| 203 | eq:th:structural | derived |  | sympy: Clausius coefficient matched to Einstein coefficient | PASS |
| 206 | eq:th:eta | derived |  | sympy: solve structural eq for entropy density eta | PASS |
| 208 | ch:theory:L208 | calc | `9.570\times10^{68}` | numeric: numeric entropy density eta in m^-2 | PASS |
| 213 | eq:th:dAmin | derived |  | sympy: minimum horizon area per decoherence event | PASS |
| 215 | ch:theory:L215 | derived | `2.77` | numeric: one bit in units of Planck area | PASS |
| 221 | eq:th:rA | none |  | not run: definition: apparent horizon radius | - |
| 223 | eq:th:Sgeo | none |  | sympy: substitute area into entropy formula | PASS |
| 224 | eq:th:TH | none |  | not run: definition: apparent horizon temperature | - |
| 226 | eq:th:EMS | none |  | sympy: equivalent forms of Misner-Sharp energy | PASS |
| 229 | eq:th:dSgeo | none |  | sympy: differentiate geometric entropy wrt H | PASS |
| 232 | eq:th:dE | none |  | sympy: energy flux through apparent horizon | PASS |
| 234 | eq:th:firstlaw | none |  | sympy: simplify T_H dS_geo term | PASS |
| 236 | eq:th:Hdot | derived |  | sympy: solve first law for Hdot | PASS |
| 239 | eq:th:friedmann | derived |  | sympy: check d(H^2)/dt consistency with continuity | PASS |
| 244 | eq:th:bitcost | none |  | sympy: Landauer bound using horizon temperature | PASS |
| 246 | ch:theory:L246 | calc | `2.5\times10^{-53}` | numeric: min energy per bit today, H0=67.4 | PASS |
| 283 | eq:th:Stotal | conjecture |  | not run: definition: conjectured entropy decomposition | - |
| 291 | ch:theory:L291 | openprob | `0.158` | numeric: record-entropy fraction per e-fold, open | PASS |
| 294 | eq:Idot | conjecture |  | not run: conjectured functional form of info rate | - |
| 302 | eq:dSdt | conjecture |  | not run: text changed at HEAD; central claim: encoding rate, conjectured | - |
| 313 | eq:th:virial | none |  | not run: definition: virial theorem | - |
| 317 | eq:th:beta | derived |  | sympy: beta_m = Omega_m/2 from the virial partition and E(1) = 1 | PASS |
| 318 | ch:theory:L318 | calc | `0.15765` | numeric: beta_m numeric from Omega_m | PASS |
| 325 | eq:th:Hmd | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 326 | ch:theory:L326 | none |  | sympy: horizon area scaling via H(a) | PASS |
| 327 |  | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 328 | eq:th:Dmd | none |  | not run: standard matter-domination growth result | - |
| 332 | eq:th:perlna | none |  | sympy: combine Idot, dSdt per ln a | PASS |
| 334 | eq:th:power | none |  | sympy: substitute matter-domination scalings | PASS |
| 336 | eq:th:perda | none |  | sympy: convert d ln a to da | PASS |
| 338 | eq:Sn | derived |  | sympy: d/da of a^(n-9/2)/(n-9/2) = a^(n-11/2) | PASS |
| 345 | ch:theory:L345 | calc | `-2.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fitted slope n=2.5, matter era | PASS |
| 345 | ch:theory:L345:-1.52 | calc | `-1.52` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fitted slope n=3, matter era | PASS |
| 345 | ch:theory:L345:-1.02 | calc | `-1.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fitted slope n=3.5, matter era | PASS |
| 345 | ch:theory:L345:-0.53 | calc | `-0.53` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fitted slope n=4, matter era | PASS |
| 348 | eq:th:target | none |  | not run: statement of required target form | - |
| 350 | eq:th:n | derived |  | sympy: solve exponent condition for n | PASS |
| 351 | ch:theory:L351 | derived | `7/2` | numeric: n = 7/2 from matter-era power counting | PASS |
| 353 | ch:theory:L353 | calc | `-1.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: full LCDM slope, matter era, n=7/2 | PASS |
| 354 | ch:theory:L354 | calc | `-1.57` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: full LCDM slope, Lambda era, n=7/2 | PASS |
| 360 | eq:th:S72 | none |  | sympy: substitute n=7/2 into entropy formula | PASS |
| 363 | eq:th:dEinfo | none |  | not run: definition: informational term in first law | - |
| 366 | eq:th:rhodot | conjecture |  | not run: conjectured identification of growth rate | - |
| 369 | eq:th:rhoint | none |  | sympy: integrate separable ODE for rho_info | PASS |
| 371 | eq:th:dH2 | none |  | sympy: substitute rho_info into Delta H^2 | PASS |
| 377 | eq:th:C | none |  | sympy: solve normalisation constant C | PASS |
| 379 | part2:eq:Ea | derived |  | sympy: d rho/da = rho/a^2 with E(1) = 1 gives E(a) = exp(1 - 1/a) (C = 1) | PASS |
| 384 | eq:th:surfdens | derived |  | sympy: ratio of growth factor to horizon area | PASS |
| 391 | eq:th:Ez | derived |  | sympy: activation function in terms of redshift | PASS |
| 393 | ch:theory:L393 | calc | `4.5\times10^{-5}` | numeric: E at z=10 | PASS |
| 393 | ch:theory:L393:0.135 | calc | `0.135` | numeric: E at z=2 | PASS |
| 393 | ch:theory:L393:0.368 | calc | `0.368` | numeric: E at z=1 | PASS |
| 393 | ch:theory:L393:1 | calc | `1` | numeric: E at z=0 | PASS |
| 393 | ch:theory:L393:2.718 | calc | `2.718` | numeric: E limit as a to infinity | PASS |
| 399 | ch:theory:L399 | calc | `0.93` | numeric: D^{7/2} record fit: constant alpha | PASS |
| 399 | ch:theory:L399:1.02 | calc | `1.02` | numeric: D^{7/2} record fit: coefficient beta of 1/a | PASS |
| 400 | ch:theory:L400 | calc | `2\%` | numeric: pct deviation of fitted 1/a coeff from analytic 1 | PASS |
| 400 | ch:theory:L400:7\% | calc | `7\%` | numeric: pct deviation of fitted constant from analytic 1 | PASS |
| 405 | eq:th:firstlaw2 | none |  | not run: definition of modified first law | - |
| 408 | ch:theory:L408 | calc | `e^{-999}` | sympy: activation function at recombination | PASS |
| 411 | ch:theory:L411 | measured | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from Level2b chain A | PASS |
| 411 | ch:theory:L411:61.52 | measured | `61.52` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from Level2b chain D | PASS |
| 430 | eq:th:firstlaw3 | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 432 | eq:HIAM | none |  | not run: result of entropy first law, physics derivation not pure algebra | - |
| 435 | eq:th:rhoinfo | none |  | not run: definition of informational energy density | - |
| 438 | eq:th:cancel | derived |  | sympy: cancel 8piG/3 factors substituting rho_info | PASS |
| 486 | eq:th:mudef | none |  | not run: definition of mu via modified Poisson equation | - |
| 487 | eq:th:sigmadef | none |  | not run: definition of Sigma via lensing equation | - |
| 491 | eq:th:mu | none |  | not run: mapping definition of mu(a) | - |
| 493 | eq:th:mu0 | derived |  | sympy: mu0 algebraic identity | PASS |
| 493 | ch:theory:L493 | derived | `-0.136` | numeric: mu0 numeric value | PASS |
| 494 | ch:theory:L494 | calc | `0.8638` | numeric: mu(0) precise value | PASS |
| 494 | ch:theory:L494:-0.1362 | calc | `-0.1362` | numeric: mu0 precise value | PASS |
| 495 | eq:th:Sigma | derived |  | sympy: Sigma = 1 from the unmodified photon source | PASS |
| 503 | ch:theory:L503 | calc | `0.864` | numeric: mu at z=0 | PASS |
| 503 | ch:theory:L503:0.948 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 503 | ch:theory:L503:0.982 | calc | `0.982` | numeric: mu at z=1 | PASS |
| 503 | ch:theory:L503:2.8\% | calc | `2.8\%` | numeric: max deviation of MGCAMB mu approx from exact | PASS |
| 503 |  | calc | `0.65` | not run: rough statement: the book says the MGCAMB form departs from the exact mu by at most 2.8 % near z ~ 0.65; the location of a flat maximum is recomputed at z = 0.635 (relative departure, 2.82 %) or z = 0.669 (absolute departure, 2.71 %), so a 5 % change (0.68) cannot be told from what near means here | - |
| 504 | ch:theory:L504 | derived | `-0.136` | numeric: mu0 repeat, IAM point in mu0-Sigma0 plane | PASS |
| 504 | ch:theory:L504:0 | derived | `0` | numeric: Sigma_0 = Sigma(z=0) - 1 of the IAM point | PASS |
| 506 | ch:theory:L506 | calc | `2.718` | numeric: E saturation value, repeat | PASS |
| 507 |  | none | `0.3153` | not run: input Omega_m restated | - |
| 515 | ch:theory:L515 | calc | `-1.062` | numeric: tangent w0 value | PASS |
| 515 | ch:theory:L515:-1.065 | calc | `-1.065` | numeric: least-squares CPL fit intercept w0 | PASS |
| 515 | ch:theory:L515:0.017 | calc | `0.017` | numeric: least-squares CPL fit slope wa | PASS |
| 515 |  | none | `0.315` | not run: input Omega_m restated (rounded), fig caption | - |
| 515 |  | calc | `-0.012` | not run: printed value -0.012 has two significant digits: recomputed -Omega_m^2/3(2-Omega_m)^2 = -0.01165 (Omega_m 0.315) rounds to -0.012 and lies 0.00025 from the value moved by 5 % (-0.0114), inside half the last printed digit, so a 5 % change cannot be told from rounding | - |
| 520 | eq:th:phi | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 522 | eq:th:phidot | derived |  | sympy: constraint equation for phidot | PASS |
| 529 | eq:th:Stot | none |  | not run: definition of total gravitational action | - |
| 531 | eq:th:action | conjecture |  | not run: postulated informational action term | - |
| 538 | eq:th:Lmini | none |  | sympy: minisuperspace Lagrangian substitution | PASS |
| 544 | eq:th:lambdadot | derived |  | sympy: Euler-Lagrange equation for lambda | PASS |
| 551 | eq:th:Hvar | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 558 | eq:th:rhodot2 | none |  | sympy: rho_info time evolution equation | PASS |
| 560 | eq:th:winfo | derived |  | sympy: solve continuity equation for w_info | PASS |
| 562 | ch:theory:L562 | derived | `-4/3` | numeric: w_info at present epoch | PASS |
| 562 | ch:theory:L562:-1 | derived | `-1` | numeric: w_info -> -1 as a -> infinity | PASS |
| 574 | eq:th:weff | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 577 | ch:theory:L577 | none |  | sympy: simplify w_eff(1) formula | PASS |
| 577 | ch:theory:L577:-1.062 | none | `-1.062` | numeric: w_eff at a=1 | PASS |
| 579 | ch:theory:L579 | none | `-4/3` | numeric: w_info(1) repeat | PASS |
| 579 | ch:theory:L579:1/3 | none | `1/3` | numeric: w_info derivative at a=1 | PASS |
| 581 | ch:theory:L581 | derived |  | sympy: simplify wa formula | PASS |
| 581 | ch:theory:L581:-1.062 | calc | `-1.062` | numeric: w0 repeat value | PASS |
| 581 |  | calc | `-0.012` | not run: printed value -0.012 has two significant digits: recomputed -Omega_m^2/3(2-Omega_m)^2 = -0.01168 (Omega_m 0.3153) rounds to -0.012 and lies 0.00028 from the value moved by 5 % (-0.0114), inside half the last printed digit, so a 5 % change cannot be told from rounding | - |
| 582 | ch:theory:L582 | none | `-1.065` | numeric: least-squares w0 repeat | PASS |
| 582 | ch:theory:L582:0.017 | none | `0.017` | numeric: least-squares wa repeat | PASS |
| 600 | eq:th:virialavg | none |  | not run: standard virial theorem, cited physics | - |
| 605 | eq:th:rhoinfo1 | conjecture |  | not run: conjecture: equal share of grav. energy | - |
| 607 | eq:th:betam | derived |  | sympy: beta_m=Om/2 from rho_info identification | PASS |
| 608 | ch:theory:L608 | calc | `0.15765` | numeric: numeric beta_m prediction from Om | PASS |
| 615 | eq:th:fcoll | calc | `0.64` | numeric: collapsed fraction above 1e6 Msun, Sheth-Tormen | PASS |
| 615 | eq:th:fcoll:0.71 | calc | `0.71` | numeric: collapsed fraction above 1e6 Msun, Tinker 2008 | PASS |
| 617 | ch:theory:L617 | calc | `0.20` | numeric: naive beta_m=Om*f_coll lower bound | PASS |
| 617 | ch:theory:L617:0.22 | calc | `0.22` | numeric: naive beta_m=Om*f_coll upper bound | PASS |
| 617 | ch:theory:L617:27 | calc | `27` | numeric: percent naive beta_m above Om/2, low | PASS |
| 617 | ch:theory:L617:41 | calc | `41` | numeric: percent naive beta_m above Om/2, high | PASS |
| 621 | eq:th:betadecomp | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 622 | ch:theory:L622 | calc | `0.7` | numeric: virial efficiency lower bound | PASS |
| 622 | ch:theory:L622:0.8 | calc | `0.8` | numeric: virial efficiency upper bound | PASS |
| 627 | ch:theory:L627 | observed | `1.35` | file `docs/verification/theory/THEORY_CHECK.md`: Neto 2007 relaxed-halo cut 2T/|U| (source check record) | PASS |
| 628 | ch:theory:L628 | observed | `1.15` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: measured: printed value found in verify_theory_derivations_output.txt, a file the chapter names | PASS |
| 628 | ch:theory:L628:1.25 | observed | `1.25` | file `docs/verification/virial/NBODY_TRACE.md`: Power 2012 virial ratio fit at 1e15 Msun/h | PASS |
| 642 | ch:theory:L642 | calc | `1.02` | numeric: 1/a coefficient for D^{7/2} (parameter count) | PASS |
| 663 |  | none | `0.315` | not run: input Om for numerical verification | - |
| 663 |  | none | `9.1\times10^{-5}` | not run: input Omega_r for numerical verification | - |
| 663 |  | none | `67.4` | not run: input H0 for numerical verification | - |
| 669 | eq:th:Iint | none |  | not run: definition of cumulative decoherence integral | - |
| 671 | ch:theory:L671 | none |  | sympy: divergence threshold exponent n=9/2 | PASS |
| 676 |  | none | `1` | not run: target alpha value restated, trivial | - |
| 680 |  | none | `1` | not run: target alpha restated in caption, trivial | - |
| 683 | ch:theory:L683 | calc | `0.86` | numeric: D^{7/2} fit, points uniform in a: alpha | PASS |
| 683 | ch:theory:L683:0.94 | calc | `0.94` | numeric: D^{7/2} fit, points uniform in a: beta | PASS |
| 686 | ch:theory:L686 | calc | `0.85` | numeric: record table, D^2 Omega_m(a) f, no horizon factor: alpha | PASS |
| 686 | ch:theory:L686:0.96 | calc | `0.96` | numeric: record table, D^2 Omega_m(a) f, no horizon factor: beta | PASS |
| 686 | ch:theory:L686:0.993 | calc | `0.993` | numeric: record table, D^2 Omega_m(a) f, no horizon factor: r | PASS |
| 687 | ch:theory:L687 | calc | `0.53` | numeric: record table, D^2 Omega_m(a) f: alpha | PASS |
| 687 | ch:theory:L687:0.59 | calc | `0.59` | numeric: record table, D^2 Omega_m(a) f: beta | PASS |
| 687 | ch:theory:L687:0.995 | calc | `0.995` | numeric: record table, D^2 Omega_m(a) f: r | PASS |
| 688 | ch:theory:L688 | calc | `0.66` | numeric: record table, D^{5/2} Omega_m(a) f: alpha | PASS |
| 688 | ch:theory:L688:0.74 | calc | `0.74` | numeric: record table, D^{5/2} Omega_m(a) f: beta | PASS |
| 688 | ch:theory:L688:0.995 | calc | `0.995` | numeric: record table, D^{5/2} Omega_m(a) f: r | PASS |
| 689 | ch:theory:L689 | calc | `0.79` | numeric: record table, D^3 Omega_m(a) f: alpha | PASS |
| 689 | ch:theory:L689:0.88 | calc | `0.88` | numeric: record table, D^3 Omega_m(a) f: beta | PASS |
| 689 | ch:theory:L689:0.994 | calc | `0.994` | numeric: record table, D^3 Omega_m(a) f: r | PASS |
| 690 | ch:theory:L690 | calc | `0.93` | numeric: record table, D^{7/2} Omega_m(a) f: alpha | PASS |
| 690 | ch:theory:L690:1.02 | calc | `1.02` | numeric: record table, D^{7/2} Omega_m(a) f: beta | PASS |
| 690 | ch:theory:L690:0.994 | calc | `0.994` | numeric: record table, D^{7/2} Omega_m(a) f: r | PASS |
| 691 | ch:theory:L691 | calc | `1.06` | numeric: record table, D^4 Omega_m(a) f: alpha | PASS |
| 691 | ch:theory:L691:1.16 | calc | `1.16` | numeric: record table, D^4 Omega_m(a) f: beta | PASS |
| 691 | ch:theory:L691:0.994 | calc | `0.994` | numeric: record table, D^4 Omega_m(a) f: r | PASS |
| 692 | ch:theory:L692 | calc | `1.06` | numeric: record table, Press-Schechter sigma_* 1.0: alpha | PASS |
| 692 | ch:theory:L692:1.24 | calc | `1.24` | numeric: record table, Press-Schechter sigma_* 1.0: beta | PASS |
| 692 | ch:theory:L692:0.983 | calc | `0.983` | numeric: record table, Press-Schechter sigma_* 1.0: r | PASS |
| 693 | ch:theory:L693 | calc | `0.86` | numeric: record table, Press-Schechter sigma_* 1.2: alpha | PASS |
| 693 | ch:theory:L693:1.02 | calc | `1.02` | numeric: record table, Press-Schechter sigma_* 1.2: beta | PASS |
| 693 | ch:theory:L693:0.982 | calc | `0.982` | numeric: record table, Press-Schechter sigma_* 1.2: r | PASS |
| 694 | ch:theory:L694 | calc | `0.90` | numeric: record table, Sheth-Tormen sigma_* 1.0: alpha | PASS |
| 694 | ch:theory:L694:1.07 | calc | `1.07` | numeric: record table, Sheth-Tormen sigma_* 1.0: beta | PASS |
| 694 | ch:theory:L694:0.983 | calc | `0.983` | numeric: record table, Sheth-Tormen sigma_* 1.0: r | PASS |
| 695 | ch:theory:L695 | calc | `0.75` | numeric: record table, Sheth-Tormen sigma_* 1.2: alpha | PASS |
| 695 | ch:theory:L695:0.89 | calc | `0.89` | numeric: record table, Sheth-Tormen sigma_* 1.2: beta | PASS |
| 695 | ch:theory:L695:0.982 | calc | `0.982` | numeric: record table, Sheth-Tormen sigma_* 1.2: r | PASS |
| 696 |  | none | `1.00` | not run: target alpha, by construction | - |
| 696 |  | none | `1.000` | not run: trivial self-correlation of target function | - |
| 706 | ch:theory:L706 | calc | `0.93` | numeric: best power law D^{7/2}: alpha | PASS |
| 706 | ch:theory:L706:1.02 | calc | `1.02` | numeric: best power law D^{7/2}: beta | PASS |
| 707 | ch:theory:L707 | calc | `2` | numeric: percent deviation of beta from target | PASS |
| 707 | ch:theory:L707:7 | calc | `7` | numeric: percent deviation of alpha from target | PASS |
| 715 | ch:theory:L715 | calc | `3.8` | numeric: n where fitted alpha crosses 1 | PASS |
| 716 | ch:theory:L716 | calc | `3.4` | numeric: n where fitted beta crosses 1 | PASS |
| 724 | eq:th:fST | none |  | not run: definition of Sheth-Tormen multiplicity function | - |
| 725 |  | none | `0.3222` | not run: published ST constant A (input) | - |
| 725 |  | none | `0.707` | not run: published ST constant q (input) | - |
| 725 |  | none | `0.3` | not run: published ST constant p (input) | - |
| 727 | ch:theory:L727 | calc | `1.07` | numeric: Sheth-Tormen sigma_* 1.0: beta (text) | PASS |
| 727 | ch:theory:L727:0.89 | calc | `0.89` | numeric: Sheth-Tormen sigma_* 1.2: beta (text) | PASS |
| 728 | ch:theory:L728 | calc | `1.02` | numeric: Press-Schechter sigma_* 1.2: beta (text) | PASS |
| 752 | eq:th:dphi | derived |  | sympy: delta phi = 0 from the perturbed constraint | PASS |
| 764 | eq:th:poisson | derived |  | sympy: Fourier form of the comoving Poisson equation | PASS |
| 766 | eq:th:noaniso | derived |  | sympy: Psi = Phi from the traceless ij equation with no anisotropic stress | PASS |
| 768 | eq:th:growth | derived |  | sympy: growth equation from continuity, Euler and Poisson | PASS |
| 772 | ch:theory:L772 | calc | `0.67` | numeric: growth deficit today, friction form (Eq. th:growth) | PASS |
| 772 | ch:theory:L772:0.78 | calc | `0.78` | numeric: growth deficit today, G_eff = mu G | PASS |
| 778 | ch:theory:L778 | measured | `61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2b background-modified H0 from chains | PASS |
| 779 | ch:theory:L779 | measured | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 chi^2 diff, IAM vs LCDM | PASS |
| 780 | ch:theory:L780 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: LCDM sigma8 from Level2 chain | PASS |
| 780 | ch:theory:L780:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: IAM sigma8 from Level2 chain | PASS |
| 780 | ch:theory:L780:72.26 | measured | `72.26` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: measured: printed value found in verify_theory_derivations_output.txt, a file the chapter names | PASS |
| 780 | ch:theory:L780:0.75 | measured | `0.75` | numeric: matter-sector H0 against SH0ES, sigma | PASS |
| 785 | eq:th:mu2 | derived |  | sympy: mu<1 since E_IAM^2>E_LCDM^2 | PASS |
| 788 | ch:theory:L788 | calc | `0.864` | numeric: mu(z=0) from E^2 ratio | PASS |
| 788 | ch:theory:L788:0.948 | calc | `0.948` | numeric: mu(z=0.5) from E^2 ratio | PASS |
| 788 | ch:theory:L788:0.982 | calc | `0.982` | numeric: mu(z=1) from E^2 ratio | PASS |
| 789 | ch:theory:L789 | calc | `0.9996` | numeric: mu(z=3) from E^2 ratio | PASS |
| 812 | eq:th:D2 | derived |  | sympy: second-order growth eqn + Friedmann matter-density identity | PASS |
| 814 | ch:theory:L814 | derived | `-3/7` | sympy: EdS trial-solution coefficient for D2 initial condition | PASS |
| 815 | ch:theory:L815 | calc | `0.989` | numeric: D2 ratio IAM/LCDM at z = 0 | PASS |
| 815 | ch:theory:L815:0.995 | calc | `0.995` | numeric: D2 ratio IAM/LCDM at z = 0.3 | PASS |
| 815 | ch:theory:L815:0.997 | calc | `0.997` | numeric: D2 ratio IAM/LCDM at z = 0.5 | PASS |
| 815 | ch:theory:L815:0.999 | calc | `0.999` | numeric: D2 ratio IAM/LCDM at z = 1 | PASS |
| 818 | eq:th:F2 | none |  | not run: definition, cited second-order kernel (Bernardeau2002) | - |
| 819 | ch:theory:L819 | derived | `0` | sympy: F2(k,-k) vanishes by momentum conservation | PASS |
| 820 | ch:theory:L820 | derived | `2` | sympy: F2(k,k) value from kernel definition | PASS |
| 823 | ch:theory:L823 | calc | `0.998` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 823 | ch:theory:L823:0.974 | calc | `0.974` | numeric: bispectrum amplitude ratio (D^4), z = 0, same early amplitude | PASS |
| 823 | ch:theory:L823:0.989 | calc | `0.989` | numeric: bispectrum amplitude ratio (D^4), z = 0.3, same early amplitude | PASS |
| 823 | ch:theory:L823:0.993 | calc | `0.993` | numeric: bispectrum amplitude ratio (D^4), z = 0.5, same early amplitude | PASS |
| 824 | ch:theory:L824 | calc | `1.015` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 824 | ch:theory:L824:1.020 | calc | `1.020` | numeric: bispectrum ratio, same amplitude today, z = 0.5 | PASS |
| 824 | ch:theory:L824:1.025 | calc | `1.025` | numeric: bispectrum ratio, same amplitude today, z = 1 | PASS |
| 825 |  | none | `1.2%` | not run: sigma8 reduction in CAMB, restated from Chapter level2 | - |
| 834 | ch:theory:L834 | calc | `0.251` | numeric: nonlinear scale k_nl, LambdaCDM, z = 0 | PASS |
| 834 | ch:theory:L834:0.255 | calc | `0.255` | numeric: nonlinear scale k_nl, IAM, z = 0 | PASS |
| 834 |  | none | `0.811` | not run: input sigma8 value restated (Planck) | - |
| 835 | ch:theory:L835:+0.6% | calc | `+0.6%` | numeric: k_nl shift IAM vs LambdaCDM, z = 0.3 | PASS |
| 835 | ch:theory:L835:+0.1% | calc | `+0.1%` | numeric: k_nl shift IAM vs LambdaCDM, z = 1 | PASS |
| 835 |  | calc | `+1.2%` | not run: printed value +1.2 % has two significant digits: recomputed k_nl shift 1.231 % rounds to 1.2 but lies 0.029 from the value moved by 5 % (1.26), inside the control tolerance (0.03), so a 5 % change cannot be told from rounding | - |
| 836 | ch:theory:L836 | calc | `0.759` | numeric: nonlinear scale k_nl, LambdaCDM, z = 1 | PASS |
| 836 | ch:theory:L836:0.760 | calc | `0.760` | numeric: nonlinear scale k_nl, IAM, z = 1 | PASS |
| 845 |  | none | `-0.136` | not run: mu0 prediction, restated canon value | - |
| 849 | ch:theory:L849 | measured | `+0.96` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Delta chi2 Planck-only chain pair | PASS |
| 849 | ch:theory:L849:+0.56 | measured | `+0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Delta chi2 Planck+RSD chain pair | PASS |
| 850 | ch:theory:L850 | measured | `+0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: smallest Level 1 Delta chi2 over the four combinations | PASS |
| 850 | ch:theory:L850:+1.73 | measured | `+1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: upper bound of dchi2 range across combos | PASS |
| 851 | ch:theory:L851 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 chi2 difference IAM vs LCDM | PASS |
| 855 | ch:theory:L855 | measured | `0.814` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 LCDM sigma8 | PASS |
| 855 | ch:theory:L855:0.802 | measured | `0.802` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 IAM-fixed sigma8 | PASS |
| 855 | ch:theory:L855:-1.6% | measured | `-1.6%` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level1 sigma8 percent shift | PASS |
| 855 | ch:theory:L855:0.809 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM sigma8 | PASS |
| 855 | ch:theory:L855:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 IAM sigma8 | PASS |
| 856 | ch:theory:L856 | measured | `-1.1%` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 sigma8 percent shift | PASS |
| 861 |  | calc | `0.13%` | not run: measured, source not named: the TT residual (< 0.13 % at l > 30, Level 1 posterior means) needs CAMB spectra; no committed spectra or output holds it (CANON/predictions_triage_2026-10-02.json: 'a CMB TT number not in the record') | - |
| 867 | ch:theory:L867 | derived |  | sympy: continuity-equation identity for w_info(a) | PASS |
| 873 | eq:th:conservation | derived |  | sympy: total continuity: matter, Lambda and info each conserved | PASS |
| 879 | ch:theory:L879 | observed | `>-1` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 fits favour w0 > -1 (least w0 of the four fits) | PASS |
| 879 | ch:theory:L879:wa<0 | observed | `<0` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 fits favour wa < 0 (largest wa of the four fits) | PASS |
| 882 | eq:th:weff2 | none |  | not run: definition of effective dark-energy equation of state | - |
| 883 | ch:theory:L883 | calc | `-1.062` | numeric: w_eff at z=0 from weighted-average formula | PASS |
| 883 | ch:theory:L883:-1.061 | calc | `-1.061` | numeric: w_eff at z=0.5 | PASS |
| 883 | ch:theory:L883:-1.052 | calc | `-1.052` | numeric: w_eff at z=1 | PASS |
| 889 | eq:th:sirens | calc | `72.26` | numeric: matter-sector H0 from photon-sector H0 and beta_m | PASS |
| 889 |  | none | `67.16` | not run: input, photon-sector H0 restated (canon) | - |
| 889 |  | none | `1.15765` | not run: trivial arithmetic 1+beta_m | - |
| 890 | ch:theory:L890 | calc | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 posterior mean H0 | PASS |
| 890 | ch:theory:L890:0.467 | calc | `0.467` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 posterior H0 std dev | PASS |
| 890 | ch:theory:L890:70.0 | observed | `70.0` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 891 |  | observed | `68.9` | not run: measured, source not named: published H0 of Hotokezaka et al. 2019 (doi 10.1038/s41550-019-0820-1); no repository file records it and the value could not be confirmed offline | - |
| 891 |  | observed | `75.46` | not run: measured, source not named: published H0 of Palmese et al. 2024 (doi 10.1103/PhysRevD.109.063508); no repository file records it and the value could not be confirmed offline | - |
| 941 | eq:th:hoop | conjecture |  | not run: conjectured holographic black-hole formation criterion | - |
| 944 | eq:th:hoopcheck | derived |  | sympy: entropy ratio reduces to inverse Planck-length-squared | PASS |
| 951 | eq:th:Meq | derived |  | sympy: equilibrium mass from setting T_BH=T_GH | PASS |
| 952 | ch:theory:L952 | calc | `2.3e22` | numeric: thermal-equilibrium mass M_eq in solar masses | PASS |
| 952 |  | none | `67.4` | not run: input, present-epoch H0 (rounded Planck value) | - |
| 953 |  | none | `7e10` | not run: cited largest known black hole mass (Shemmer 2004) | - |
| 956 | eq:th:Gamma | derived |  | sympy: thermally limited bit-encoding rate from Hawking power | PASS |
| 1042 | ch:theory:L1042 | calc | `2.8\%` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 1042 | ch:theory:L1042:0.65 | calc | `0.65` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 1045 | ch:theory:L1045 | record | `61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from two background-level exploratory chains | PASS |
| 1047 | ch:theory:L1047 | calc | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 IAM minus LCDM best-fit chi2 | PASS |
| 1047 | ch:theory:L1047:0.809 | record | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 from Level2 LCDM chain | PASS |
| 1047 | ch:theory:L1047:0.800 | record | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 from Level2 IAM chain | PASS |
| 1047 | ch:theory:L1047:72.26 | derived | `72.26` | numeric: matter-sector H0 from photon H0 and beta_m | PASS |
| 1048 | ch:theory:L1048 | calc | `0.75` | numeric: tension of matter H0 vs SH0ES, sigma units | PASS |
| 1060 | ch:theory:L1060 | prediction | `0.864` | numeric: IAM mu prediction at z=0 | PASS |
| 1060 | ch:theory:L1060:0.948 | prediction | `0.948` | numeric: IAM mu prediction at z=0.5 | PASS |
| 1060 |  | prediction | `1` | not run: Sigma=1 part of headline prediction, trivial | - |
| 1063 | ch:theory:L1063 | measured | `0.35--0.5` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: phantom-crossing redshift, lowest of the DESI DR2 fits | PASS |
| 1065 |  | prediction | `10^{-4}` | not run: falsification threshold on beta_gamma | - |
| 1066 |  | prediction | `10^{-5}` | not run: CMB-S4 energy-injection sensitivity target | - |
| 1066 |  | prediction | `10^{-4}` | not run: repeat of beta_gamma falsification threshold | - |
| 1067 | ch:theory:L1067 | calc | `3.3\%` | numeric: beta_gamma 95 % bound (committed output) over beta_m, per cent | PASS |
| 1067 | ch:theory:L1067:0.0052 | measured | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma 95 % bound (committed output) | PASS |
| 1069 |  | prediction | `1/2` | not run: restated beta_m/Om=1/2 constancy prediction | - |
| 1078 |  | prediction | `-0.136` | not run: headline IAM mu0 prediction, locked canon value | - |
| 1078 |  | prediction | `1` | not run: Sigma=1 part of headline prediction, trivial | - |
| 1084 | ch:theory:L1084 | derived | `7/2` | numeric: n = 7/2 restated in the summary (power counting) | PASS |
| 1085 |  | calc | `2\%` | not run: printed value 2 % (within 2 %) has one significant digit: recomputed distance of the D^{7/2} 1/a coefficient from 1 is 2.12 %, which rounds to 2 %, so a 5 % change (2.1 %) cannot be told from rounding | - |

## Part 2 - ch:entropicgravity - `docs/book/part2/p2_03a_entropic_gravity.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 30 | eq:eg_flux | derived |  | sympy: horizon energy flux, algebra from A_H,r_A | PASS |
| 32 | eq:eg_dS | derived |  | sympy: horizon entropy time-derivative, algebra | PASS |
| 34 | eq:eg_firstlaw | none |  | not run: definition: Clausius-form first law postulate | - |
| 36 | eq:eg_Hdot | derived |  | sympy: combine flux+entropy+first law | PASS |
| 38 | eq:eg_friedmann | derived |  | sympy: integrate Hdot with continuity eq to Friedmann | PASS |
| 57 | eq:eg_barrow | none |  | not run: definition: Barrow fractal entropy | - |
| 60 | eq:eg_tsallis | none |  | not run: definition: Tsallis non-extensive entropy | - |
| 71 | eq:eg_stotal | conjecture |  | not run: conjecture: additional entropy source definition | - |
| 78 | eq:eg_bitcost | conjecture |  | sympy: Landauer bit cost at horizon temperature | PASS |
| 91 | ch:entropicgravity:L91 | observed | `72.2\pm0.9` | file `docs/book/verification/sources/eg_MANIFEST.md`: Barrow fit H0 in one DESI DR2 combination (Luciano 2025) | PASS |
| 96 | eq:eg_growth | none |  | sympy: friction-form growth eq, matter-density identity | PASS |
| 99 | eq:eg_growthN | derived |  | sympy: e-fold transform of growth equation | PASS |
| 106 | ch:entropicgravity:L106 | calc | `1.64` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: growth-factor deficit, friction form | PASS |
| 107 | ch:entropicgravity:L107 | calc | `0.78` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: growth-factor deficit, G_eff = mu G with the exact mu(a) | PASS |
| 109 | ch:entropicgravity:L109 | calc | `0.67` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: growth-factor deficit, Level-2 form | PASS |
| 111 | ch:entropicgravity:L111 | calc | `2.158` | numeric: friction coefficient today, friction form | PASS |
| 111 | ch:entropicgravity:L111:2.152 | calc | `2.152` | numeric: friction coefficient today, Level-2 form | PASS |
| 119 | ch:entropicgravity:L119 | calc | `-1.64` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: dD/D today, note form (committed output) | PASS |
| 119 | ch:entropicgravity:L119:-0.67 | calc | `-0.67` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: dD/D today, L2_fric form (committed output) | PASS |
| 119 | ch:entropicgravity:L119:-0.78 | calc | `-0.78` | heavy file `docs/verification/scripts/verify_entropic_gravity_output.txt`: dD/D today, L1_muG form (committed output) | PASS |
| 127 | ch:entropicgravity:L127 | calc | `13.62` | numeric: 1 - mu(1) = beta_m/(1+beta_m) | PASS |
| 128 | ch:entropicgravity:L128 | calc | `4.25` | numeric: f sigma8 deficit at z=0, growth ODE with mu(a) | PASS |
| 128 | ch:entropicgravity:L128:2.17 | calc | `2.17` | numeric: f sigma8 deficit at z=0.3, growth ODE with mu(a) | PASS |
| 128 | ch:entropicgravity:L128:1.35 | calc | `1.35` | numeric: f sigma8 deficit at z=0.5, growth ODE with mu(a) | PASS |
| 128 | ch:entropicgravity:L128:0.41 | calc | `0.41` | numeric: f sigma8 deficit at z=1.0, growth ODE with mu(a) | PASS |
| 128 |  | none | `0.7--0.8` | not run: restated range of 0.67/0.78 values | - |
| 139 | eq:eg_virial | derived |  | sympy: virial theorem, k=-1 gravity case | PASS |
| 143 | eq:eg_beta | calc | `0.15765` | numeric: coupling constant beta_m = Omega_m/2 | PASS |
| 144 |  | none | `0.3153` | not run: input, Planck2018VI Omega_m | - |
| 149 | ch:entropicgravity:L149 | measured | `1.15` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 149 | ch:entropicgravity:L149:1.25 | observed | `1.25` | file `docs/verification/virial/NBODY_TRACE.md`: Power 2012 virial ratio fit at 1e15 Msun/h | PASS |
| 151 |  | prediction | `37` | not run: prediction: scale range, atom to horizon | - |
| 151 |  | prediction | `33` | not run: prediction: scale range subset, atom to cluster | - |
| 159 | ch:entropicgravity:L159 | derived | `7/2` | sympy: exponent n from decoherence scaling condition | PASS |
| 162 | eq:eg_Ea | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 167 | ch:entropicgravity:L167 | derived | `C=k` | sympy: E(1)=exp(C-k)=1 gives C=k | PASS |
| 169 | ch:entropicgravity:L169 | derived | `C=1` | sympy: boundary condition E->e gives C=1 | PASS |
| 171 | ch:entropicgravity:L171 | derived | `k=1` | sympy: exponent matching gives k=1 independently | PASS |
| 171 |  | none | `C=k=1` | not run: restatement combining two prior results | - |
| 172 | ch:entropicgravity:L172 | fitted | `0.93` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fit exp(alpha - beta/a) to D^3.5 record (committed output, alpha 0.93 beta 1.02) | PASS |
| 172 | ch:entropicgravity:L172:1.02 | fitted | `1.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: fit exp(alpha - beta/a) to D^3.5 record (committed output, alpha 0.93 beta 1.02) | PASS |
| 174 | ch:entropicgravity:L174 | calc | `36.8` | numeric: E(1)/e percentage of ceiling today | PASS |
| 175 | ch:entropicgravity:L175 | calc | `a=1` | sympy: inflection of dE/dlna at a=1 today | PASS |
| 176 | ch:entropicgravity:L176 | calc | `1.26` | numeric: redshift of peak dE/dt rate | PASS |
| 176 | ch:entropicgravity:L176:a=1/2 | calc | `a=1/2` | sympy: scale factor of peak dE/da | PASS |
| 199 | ch:entropicgravity:L199 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0, Level-2 chain | PASS |
| 199 | ch:entropicgravity:L199:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 uncertainty, Level-2 chain | PASS |
| 199 | ch:entropicgravity:L199:72.2 | observed | `72.2\pm0.9` | file `docs/book/verification/sources/eg_MANIFEST.md`: Barrow fit H0 (table), Luciano 2025 | PASS |
| 200 | ch:entropicgravity:L200 | calc | `72.26` | numeric: matter-sector H0 from photon H0 and beta_m | PASS |
| 201 | ch:entropicgravity:L201 | calc | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Delta chi2 (lower end of the quoted range) | PASS |
| 201 | ch:entropicgravity:L201:1.73 | calc | `1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: maximum dchi2 across chain pairs | PASS |
| 210 | ch:entropicgravity:L210 | calc | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 Planck chi2 difference IAM vs LCDM | PASS |
| 210 | ch:entropicgravity:L210:0.56 | calc | `0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level1 min chi2 diff (Planck+RSD pair) | PASS |
| 210 | ch:entropicgravity:L210:1.73 | calc | `1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level1 max chi2 diff (Planck+BAO pair) | PASS |
| 232 | ch:entropicgravity:L232 | measured | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2b background-term H0 | PASS |
| 232 | ch:entropicgravity:L232:61.52 | measured | `61.52` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2b background-term H0, run D | PASS |
| 233 | ch:entropicgravity:L233 | measured | `0.010` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: max last-recorded Gelman-Rubin R-1 across chains | PASS |
| 237 | ch:entropicgravity:L237 | calc | `-0.80\sigma` | numeric: mu0 tension vs DESI in sigma | PASS |
| 237 | ch:entropicgravity:L237:-0.82\sigma | calc | `-0.82\sigma` | numeric: mu0 tension vs ACT combo in sigma | PASS |
| 238 | ch:entropicgravity:L238 | calc | `-0.94\sigma` | numeric: Sigma0 tension vs DESI in sigma | PASS |
| 238 | ch:entropicgravity:L238:-0.31\sigma | calc | `-0.31\sigma` | numeric: Sigma0 tension vs ACT combo in sigma | PASS |
| 238 | ch:entropicgravity:L238:-0.12\sigma | calc | `-0.12\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 tension vs KiDS/DES/DESI/Pantheon+ | PASS |
| 238 | ch:entropicgravity:L238:-0.75\sigma | calc | `-0.75\sigma` | numeric: matter-sector H0 tension vs SH0ES | PASS |
| 244 | ch:entropicgravity:L244 | derived | `-0.136` | numeric: mu0 = -beta_m/(1 + beta_m) | PASS |
| 246 | ch:entropicgravity:L246 | derived | `0` | numeric: Sigma_0 = 0 from the unmodified photon source | PASS |
| 248 | ch:entropicgravity:L248 | measured | `0.7998\pm0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 IAM chain sigma8 | PASS |
| 248 | ch:entropicgravity:L248:0.8087 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM chain sigma8 | PASS |
| 249 | ch:entropicgravity:L249 | calc | `72.26` | numeric: matter-sector H0 = H0_photon sqrt(1 + beta_m) | PASS |
| 250 | ch:entropicgravity:L250 | measured | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: table repeat: Level2 chi2 diff | PASS |
| 250 | ch:entropicgravity:L250:0.56 | measured | `0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: table repeat: Level1 min chi2 diff | PASS |
| 250 | ch:entropicgravity:L250:1.73 | measured | `1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: table repeat: Level1 max chi2 diff | PASS |
| 251 | eq:eg_stotal | derived | `0.15765` | numeric: beta_m defined as Omega_m/2 | PASS |
| 259 |  | prediction | `-0.136` | not run: input restated (Euclid prediction target) | - |
| 259 |  | prediction | `0` | not run: input restated (Euclid prediction target) | - |
| 266 | ch:entropicgravity:L266 | calc | `5.14` | heavy file `docs/verification/scripts/verify_shapefit_chi2_output.txt`: DESI ShapeFit+BAO chi2, IAM (MGCAMB form) | PASS |
| 266 | ch:entropicgravity:L266:4.52 | calc | `4.52` | heavy file `docs/verification/scripts/verify_shapefit_chi2_output.txt`: DESI ShapeFit+BAO chi2, LambdaCDM | PASS |
| 271 | ch:entropicgravity:L271 | calc | `1.018` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 271 | ch:entropicgravity:L271:1.002 | calc | `1.002` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 271 | ch:entropicgravity:L271:1.158 | calc | `1.158` | numeric: M_lens/M_dyn = 1/mu today | PASS |
| 271 | ch:entropicgravity:L271:1.105 | calc | `1.105` | numeric: M_lens/M_dyn = 1/mu at z = 0.2 | PASS |
| 271 | ch:entropicgravity:L271:1.055 | calc | `1.055` | numeric: M_lens/M_dyn = 1/mu at z = 0.5 | PASS |
| 274 |  | openprob | `1` | not run: Level2 form ratio statement, no computation | - |
| 288 | ch:entropicgravity:L288 | calc | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: summary repeat: Level2 chi2 diff | PASS |
| 288 | ch:entropicgravity:L288:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: summary repeat: Level2 IAM sigma8 rounded | PASS |
| 288 |  | calc | `0.12\sigma` | not run: printed value 0.12 sigma has two significant digits: a 5 % change (0.006) is about the rounding half-unit (0.005), so it cannot be told from rounding (recomputed 0.1249 from CHAIN_EXTRACTION_FINAL.csv, Run A sigma8 vs joint 0.802 +- 0.018) | - |
| 289 | ch:entropicgravity:L289 | calc | `-0.37\sigma` | numeric: photon-sector H0 tension vs Planck | PASS |
| 289 | ch:entropicgravity:L289:-0.75\sigma | calc | `-0.75\sigma` | numeric: summary repeat: matter-sector H0 tension | PASS |
| 292 |  | prediction | `-0.136` | not run: input restated (Euclid prediction target) | - |
| 292 |  | prediction | `0` | not run: input restated (Euclid prediction target) | - |
| 307 | ch:entropicgravity:L307 | derived | `1` | numeric: Sigma = 1 given eligibility (status table) | PASS |
| 308 | ch:entropicgravity:L308 | derived | `0` | numeric: virial theorem 2<T> + <V> = 0 for V ~ -1/r | PASS |
| 308 | ch:entropicgravity:L308:1/2 | derived | `1/2` | numeric: the 1/2 partition <T>/|<V>| | PASS |
| 310 | eq:eg_stotal:0.15765 | derived | `0.15765` | numeric: table-status repeat: beta_m=Omega_m/2 | PASS |
| 311 | ch:entropicgravity:L311 | derived | `7/2` | numeric: n = 7/2 (status table), power counting | PASS |
| 312 | ch:entropicgravity:L312 | calc | `1.26` | numeric: peak of the writing rate per unit time, redshift | PASS |

## Part 2 - ch:dual - `docs/book/part2/p2_04_dualsector_chains.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 252 | ch:dual:L252 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Planck chain photon-sector H0 | PASS |
| 252 | ch:dual:L252:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 chain H0 std dev, photon sector | PASS |
| 252 | ch:dual:L252:72.26 | calc | `72.26` | numeric: matter-sector H0 = photon H0 * sqrt(1+beta_m) | PASS |
| 252 | ch:dual:L252:0.50 | calc | `0.50` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: propagated uncertainty on matter-sector H0 | PASS |
| 252 | ch:dual:L252:67.36 | observed | `67.36` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 252 | ch:dual:L252:-0.37 | observed | `-0.37` | numeric: sigma tension photon H0 vs Planck 2018 | PASS |
| 252 | ch:dual:L252:73.04 | observed | `73.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 252 | ch:dual:L252:1.04 | observed | `1.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 252 | ch:dual:L252:-0.75 | observed | `-0.75` | numeric: sigma tension matter H0 vs SH0ES | PASS |
| 252 | ch:dual:L252:0.54 | observed | `0.54` | file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Planck 2018 H0 error, committed output | PASS |
| 255 | ch:dual:L255 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: restated photon-sector H0 (Level 2) | PASS |
| 256 | ch:dual:L256 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon H0 used as multiplicand | PASS |
| 256 | ch:dual:L256:1.0759 | derived | `1.0759` | numeric: sqrt(1+beta_m) factor | PASS |
| 256 | ch:dual:L256:72.26 | derived | `72.26` | numeric: matter-sector H0 derived result | PASS |
| 256 |  | none | `0.15765` | not run: restated canon beta_m value, input | - |
| 263 |  | prediction | `-0.136` | not run: restated canon mu0 prediction value | - |
| 263 |  | prediction | `1` | not run: Sigma=1, unmodified lensing slip, model statement | - |

## Part 2 - ch:latetime - `docs/book/part2/p2_07_late_time_growth.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 107 | ch:latetime:L107:0.04 | calc | `0.04` | heavy file `docs/verification/scripts/verify_limber_lensing_output.txt`: Limber: lensing power change, low end over 30 <= L <= 1000 (per cent) | PASS |
| 107 | ch:latetime:L107:0.24 | calc | `0.24` | heavy file `docs/verification/scripts/verify_limber_lensing_output.txt`: Limber: lensing power change, high end over 30 <= L <= 1000 (per cent) | PASS |

## Part 2 - ch:level2 - `docs/book/part2/p2_06_dual_sector_perturbation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 14 | ch:level2:L14 | derived | `0.15765` | numeric: beta_m = Omega_m/2 | PASS |
| 14 | ch:level2:L14:0.3153 | measured | `0.3153` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 19 | ch:level2:L19 | calc | `+0.54` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2_min diff, lowest points of two L2 chains | PASS |
| 19 | ch:level2:L19:-0.01 | calc | `-0.01` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: chain-average chi2, Run A minus Run C | PASS |
| 20 | ch:level2:L20 | calc | `0.76` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: likelihood ratio from dchi2 | PASS |
| 21 | ch:level2:L21 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM posterior, L2 chain | PASS |
| 21 | ch:level2:L21:0.7998 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 dual-sector posterior, L2 chain | PASS |
| 21 | ch:level2:L21:1.1 | calc | `1.1` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: percent reduction in sigma8 | PASS |
| 24 | ch:level2:L24 | measured | `67.16` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 24 | ch:level2:L24:0.37 | calc | `0.37` | numeric: H0 photon sector vs Planck LCDM | PASS |
| 25 | ch:level2:L25 | measured | `72.26` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 25 | ch:level2:L25:0.75 | calc | `0.75` | numeric: sigma of H0_matter from SH0ES | PASS |
| 26 | ch:level2:L26 | measured | `61.5` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from background-Friedmann exploratory chains | PASS |
| 26 | ch:level2:L26:10.9 | calc | `10.9` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2b H0 against Planck, in Planck sigma | PASS |
| 33 | ch:level2:L33 | measured | `67.4` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 33 | ch:level2:L33:0.5 | measured | `0.5` | file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Planck 2018 H0 error, rounded | PASS |
| 34 | ch:level2:L34 | measured | `73.04` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 34 | ch:level2:L34:1.04 | measured | `1.04` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: SH0ES H0 error as used in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 35 | ch:level2:L35 | calc | `4.9` | numeric: Hubble tension significance | PASS |
| 37 | ch:level2:L37 | observed | `70.39` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 37 | ch:level2:L37:1.22 | observed | `1.22` | numeric: TRGB H0 statistical error (Freedman 2025) | PASS |
| 37 | ch:level2:L37:1.33 | observed | `1.33` | numeric: TRGB H0 systematic error (Freedman 2025) | PASS |
| 37 | ch:level2:L37:0.70 | observed | `0.70` | numeric: TRGB H0 supernova error (Freedman 2025) | PASS |
| 38 | ch:level2:L38 | calc | `1.94` | numeric: quadrature sum of TRGB errors | PASS |
| 38 | ch:level2:L38:0.96 | calc | `0.96` | numeric: sigma of TRGB below H0_matter | PASS |
| 38 | ch:level2:L38:72.26 | measured | `72.26` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 39 | ch:level2:L39 | calc | `1.66` | numeric: sigma of TRGB above H0_photon | PASS |
| 39 | ch:level2:L39:67.16 | measured | `67.16` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 42 | ch:level2:L42 | observed | `0.759` | file `docs/verification/chains/DUAL_SECTOR_PERTURBATION_CHECK.md`: measured: printed value found in DUAL_SECTOR_PERTURBATION_CHECK.md, a file the chapter names | PASS |
| 43 | ch:level2:L43 | observed | `0.766` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 43 | ch:level2:L43:0.776 | observed | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DES Y3 3x2pt S8 (committed output) | PASS |
| 44 | ch:level2:L44:0.769 | observed | `0.769` | numeric: HSC Y3 cosmic shear S8 (Li et al. 2023) | PASS |
| 45 | ch:level2:L45 | observed | `0.815` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 52 | ch:level2:L52 | calc | `+0.56` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: min dchi2 across four chain pairs | PASS |
| 52 | ch:level2:L52:+1.73 | calc | `+1.73` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: max dchi2 across four chain pairs | PASS |
| 75 | eq:l2_friedmann | none |  | not run: definition: standard Friedmann equation | - |
| 79 | eq:l2_Hm | none |  | not run: definition: matter-sector expansion rate ansatz | - |
| 83 | eq:l2_beta | derived | `0.15765` | numeric: beta_m equation, Omega_m/2 | PASS |
| 97 | ch:level2:L97 | derived |  | sympy: identity defining mu(a) via Omega_m(a) | PASS |
| 101 | eq:l2_mu | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 102 | eq:l2_sigma | derived | `1` | numeric: Sigma = 1: lensing equation unchanged | PASS |
| 106 | eq:l2_mu0 | calc | `0.864` | numeric: mu at z=0 | PASS |
| 108 | ch:level2:L108 | calc | `13.6` | numeric: percent suppression of mu at z=0 | PASS |
| 115 | ch:level2:L115 | measured | `0.3153` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 118 | ch:level2:L118 | calc | `1.000` | numeric: scale factor a at z=0 | PASS |
| 118 | ch:level2:L118:1.0000 | calc | `1.0000` | numeric: E(a) at z=0 | PASS |
| 118 | ch:level2:L118:0.864 | calc | `0.864` | numeric: mu(a) at z=0 | PASS |
| 118 | ch:level2:L118:1.076 | calc | `1.076` | numeric: Hm/H at z=0 | PASS |
| 119 | ch:level2:L119 | calc | `0.833` | numeric: scale factor a at z=0.2 | PASS |
| 119 | ch:level2:L119:0.8187 | calc | `0.8187` | numeric: E(a) at z=0.2 | PASS |
| 119 | ch:level2:L119:0.905 | calc | `0.905` | numeric: mu(a) at z=0.2 | PASS |
| 119 | ch:level2:L119:1.051 | calc | `1.051` | numeric: Hm/H at z=0.2 | PASS |
| 120 | ch:level2:L120 | calc | `0.667` | numeric: scale factor a at z=0.5 | PASS |
| 120 | ch:level2:L120:0.6065 | calc | `0.6065` | numeric: E(a) at z=0.5 | PASS |
| 120 | ch:level2:L120:0.948 | calc | `0.948` | numeric: mu(a) at z=0.5 | PASS |
| 120 | ch:level2:L120:1.027 | calc | `1.027` | numeric: Hm/H at z=0.5 | PASS |
| 121 | ch:level2:L121 | calc | `0.500` | numeric: scale factor a at z=1 | PASS |
| 121 | ch:level2:L121:0.3679 | calc | `0.3679` | numeric: E(a) at z=1 | PASS |
| 121 | ch:level2:L121:0.982 | calc | `0.982` | numeric: mu(a) at z=1 | PASS |
| 121 | ch:level2:L121:1.009 | calc | `1.009` | numeric: Hm/H at z=1 | PASS |
| 122 | ch:level2:L122 | calc | `0.333` | numeric: scale factor a at z=2 | PASS |
| 122 | ch:level2:L122:0.1353 | calc | `0.1353` | numeric: E(a) at z=2 | PASS |
| 122 | ch:level2:L122:0.9977 | calc | `0.9977` | numeric: mu(a) at z=2 | PASS |
| 122 | ch:level2:L122:1.001 | calc | `1.001` | numeric: Hm/H at z=2 | PASS |
| 123 | ch:level2:L123 | calc | `0.250` | numeric: scale factor a at z=3 | PASS |
| 123 | ch:level2:L123:0.0498 | calc | `0.0498` | numeric: E(a) at z=3 | PASS |
| 123 | ch:level2:L123:0.9996 | calc | `0.9996` | numeric: mu(a) at z=3 | PASS |
| 123 | ch:level2:L123:1.0002 | calc | `1.0002` | numeric: Hm/H at z=3 | PASS |
| 124 | ch:level2:L124 | calc | `0.167` | numeric: scale factor a at z=5 | PASS |
| 124 | ch:level2:L124:0.0067 | calc | `0.0067` | numeric: E(a) at z=5 | PASS |
| 124 | ch:level2:L124:1.0000 | calc | `1.0000` | numeric: mu(a) at z=5 | PASS |
| 124 | ch:level2:L124:1.0000' | calc | `1.0000` | numeric: Hm/H at z=5 | PASS |
| 133 | ch:level2:L133 | derived | `0.15765` | numeric: iam_beta literal in Fortran source | PASS |
| 149 | ch:level2:L149 | calc | `0.0014` | numeric: Omega_nu from neutrino mass | PASS |
| 149 |  | none | `0.06` | not run: neutrino mass assumption, eV (input) | - |
| 150 | ch:level2:L150 | calc | `0.9986` | numeric: coded coupling fraction excl. neutrino | PASS |
| 150 | ch:level2:L150:0.14 | calc | `0.14` | numeric: percent difference from excl. neutrino | PASS |
| 150 |  | calc | `2` | not run: printed value 2 (per cent posterior error on Omega_m) has one significant digit: a 5 % change cannot be told from rounding (recomputed 2.05 % from CHAIN_EXTRACTION_FINAL.csv, Run A omegam_sd/omegam) | - |
| 178 | ch:level2:L178 | calc | `1.2` | numeric: sigma8 lowering, table row z=0 (book input) | PASS |
| 178 | ch:level2:L178:0.8 | calc | `0.8` | numeric: sigma8 lowering via Eq. l2_mu, row z=0 | PASS |
| 189 | ch:level2:L189 | measured | `0.9880` | heavy file `docs/verification/chains/data/growth_on.json`: sigma8 on/off at z=0.0, committed CAMB growth record | PASS |
| 189 | ch:level2:L189:0.9922 | calc | `0.9922` | numeric: D ratio, G_eff = mu G, same early amplitude, z=0.0 | PASS |
| 189 | ch:level2:L189:0.498 | measured | `0.498` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch on, z=0.0 | PASS |
| 189 | ch:level2:L189:0.524 | measured | `0.524` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch off, z=0.0 | PASS |
| 189 | ch:level2:L189:0.542 | measured | `0.542` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, on, z=0.0 | PASS |
| 189 | ch:level2:L189:0.529 | measured | `0.529` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, off, z=0.0 | PASS |
| 190 | ch:level2:L190 | measured | `0.9962` | heavy file `docs/verification/chains/data/growth_on.json`: sigma8 on/off at z=0.5, committed CAMB growth record | PASS |
| 190 | ch:level2:L190:0.9978 | calc | `0.9978` | numeric: D ratio, G_eff = mu G, same early amplitude, z=0.5 | PASS |
| 190 | ch:level2:L190:0.746 | measured | `0.746` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch on, z=0.5 | PASS |
| 190 | ch:level2:L190:0.760 | measured | `0.760` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch off, z=0.5 | PASS |
| 190 | ch:level2:L190:0.770 | measured | `0.770` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, on, z=0.5 | PASS |
| 190 | ch:level2:L190:0.763 | measured | `0.763` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, off, z=0.5 | PASS |
| 191 | ch:level2:L191 | measured | `0.9988` | heavy file `docs/verification/chains/data/growth_on.json`: sigma8 on/off at z=1.0, committed CAMB growth record | PASS |
| 191 | ch:level2:L191:0.9994 | calc | `0.9994` | numeric: D ratio, G_eff = mu G, same early amplitude, z=1.0 | PASS |
| 191 | ch:level2:L191:0.869 | measured | `0.869` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch on, z=1.0 | PASS |
| 191 | ch:level2:L191:0.875 | measured | `0.875` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch off, z=1.0 | PASS |
| 191 | ch:level2:L191:0.881 | measured | `0.881` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, on, z=1.0 | PASS |
| 191 | ch:level2:L191:0.879 | measured | `0.879` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, off, z=1.0 | PASS |
| 192 | ch:level2:L192 | measured | `0.9999` | heavy file `docs/verification/chains/data/growth_on.json`: sigma8 on/off at z=2.0, committed CAMB growth record | PASS |
| 192 | ch:level2:L192:0.955 | measured | `0.955` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch on, z=2.0 | PASS |
| 192 | ch:level2:L192:0.956 | measured | `0.956` | heavy file `docs/verification/chains/data/growth_on.json`: f = dln sigma8/dln a, switch off, z=2.0 | PASS |
| 192 | ch:level2:L192:0.960 | measured | `0.960` | heavy file `docs/verification/chains/data/growth_on.json`: f CAMB = fsigma8/sigma8, on and off, z=2.0 | PASS |
| 196 | ch:level2:L196:2.8 | calc | `2.8` | numeric: largest gap MGCAMB mu (mu0 -0.135) vs exact mu | PASS |
| 196 |  | none | `-0.135` | not run: MGCAMB mu0, input restated from Ch. latetime | - |
| 225 | ch:level2:L225 | observed | `0.423` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: 6dFGS fsigma8 | PASS |
| 225 | ch:level2:L225:0.055 | observed | `0.055` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: 6dFGS fsigma8 error | PASS |
| 226 | ch:level2:L226 | observed | `0.530` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: SDSS MGS fsigma8 | PASS |
| 226 | ch:level2:L226:0.160 | observed | `0.160` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: SDSS MGS fsigma8 error | PASS |
| 227 | ch:level2:L227 | observed | `0.497` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: BOSS DR12 z=0.38 fsigma8 | PASS |
| 227 | ch:level2:L227:0.045 | observed | `0.045` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: BOSS DR12 z=0.38 fsigma8 error | PASS |
| 228 | ch:level2:L228 | observed | `0.459` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: BOSS DR12 z=0.51 fsigma8 | PASS |
| 228 | ch:level2:L228:0.038 | observed | `0.038` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: BOSS DR12 z=0.51 fsigma8 error | PASS |
| 229 | ch:level2:L229 | observed | `0.473` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS LRG fsigma8 | PASS |
| 229 | ch:level2:L229:0.041 | observed | `0.041` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS LRG fsigma8 error | PASS |
| 230 | ch:level2:L230 | observed | `0.315` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS ELG fsigma8 | PASS |
| 230 | ch:level2:L230:0.095 | observed | `0.095` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS ELG fsigma8 error | PASS |
| 231 | ch:level2:L231 | observed | `0.462` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS quasars fsigma8 | PASS |
| 231 | ch:level2:L231:0.045 | observed | `0.045` | file `Cosmological_Physics/camb_validation/likelihood_rsd.py`: eBOSS quasars fsigma8 error | PASS |
| 245 | ch:level2:L245 | measured | `0.0099` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A final R-1 convergence stat | PASS |
| 246 | ch:level2:L246 | measured | `0.0081` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C final R-1 convergence stat | PASS |
| 247 | ch:level2:L247 | measured | `0.0080` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D final R-1 convergence stat | PASS |
| 248 | ch:level2:L248 | measured | `0.0100` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run Ab final R-1 convergence stat | PASS |
| 249 | ch:level2:L249 | measured | `0.0068` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run Db final R-1 convergence stat | PASS |
| 279 | ch:level2:L279 | measured | `67.188\pm0.465` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C posterior H0 mean | PASS |
| 279 | ch:level2:L279:67.161\pm0.467 | measured | `67.161\pm0.467` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior H0 mean | PASS |
| 279 |  | calc | `-0.06\sigma` | not run: printed value -0.06 sigma has one significant digit: a 5 % change (0.003) is below the rounding half-unit (0.005) and cannot be told from rounding (recomputed -0.0576 from CHAIN_EXTRACTION_FINAL.csv; -0.06 also in section D of verify_late_time_level2_output.txt) | - |
| 280 | ch:level2:L280 | measured | `0.8087\pm0.0059` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C posterior sigma8 mean | PASS |
| 280 | ch:level2:L280:0.7998\pm0.0058 | measured | `0.7998\pm0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior sigma8 mean | PASS |
| 280 | ch:level2:L280:-1.51\sigma | calc | `-1.51\sigma` | numeric: sigma8 shift A vs C in sigma | PASS |
| 281 | ch:level2:L281 | measured | `0.830\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C posterior S8 mean | PASS |
| 281 | ch:level2:L281:0.822\pm0.011 | measured | `0.822\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior S8 mean | PASS |
| 281 | ch:level2:L281:-0.78\sigma | calc | `-0.78\sigma` | numeric: S8 shift A vs C in sigma | PASS |
| 282 | ch:level2:L282 | measured | `0.02218\pm0.00013` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C posterior ombh2 mean | PASS |
| 282 | ch:level2:L282:0.02217\pm0.00013 | measured | `0.02217\pm0.00013` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior ombh2 mean | PASS |
| 282 | ch:level2:L282:-0.07\sigma | calc | `-0.07\sigma` | numeric: ombh2 shift A vs C in sigma | PASS |
| 283 | ch:level2:L283 | measured | `0.11989\pm0.00105` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 283 | ch:level2:L283:0.11994\pm0.00105 | measured | `0.11994\pm0.00105` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 283 | ch:level2:L283:+0.05\sigma | calc | `+0.05\sigma` | numeric: Omega_c h^2 shift in sigma | PASS |
| 284 | ch:level2:L284 | measured | `0.0532\pm0.0073` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: tau, Run C posterior mean, in verify_late_time_level2_output.txt section D | PASS |
| 284 | ch:level2:L284:0.0537\pm0.0073 | measured | `0.0537\pm0.0073` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 284 | ch:level2:L284:+0.07\sigma | calc | `+0.07\sigma` | numeric: tau shift in sigma | PASS |
| 285 | ch:level2:L285 | measured | `0.9630\pm0.0040` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 285 | ch:level2:L285:0.9630\pm0.0039 | measured | `0.9630\pm0.0039` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 285 | ch:level2:L285:0.00\sigma | calc | `0.00\sigma` | numeric: n_s shift in sigma, zero | PASS |
| 286 | ch:level2:L286 | measured | `3.0393\pm0.0146` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 286 | ch:level2:L286:3.0407\pm0.0145 | measured | `3.0407\pm0.0145` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 286 |  | calc | `+0.09\sigma` | not run: printed value +0.09 sigma has one significant digit: a 5 % change (0.0045) is below the rounding half-unit (0.005) and cannot be told from rounding (recomputed 0.0959 from the table inputs 3.0407, 3.0393, 0.0146; 0.091 from the chain means in verify_late_time_level2_output.txt section D) | - |
| 287 | ch:level2:L287 | measured | `0.3162\pm0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C posterior Omega_m mean | PASS |
| 287 | ch:level2:L287:0.3166\pm0.0065 | measured | `0.3166\pm0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior Omega_m mean | PASS |
| 287 | ch:level2:L287:+0.05\sigma | calc | `+0.05\sigma` | numeric: Omega_m shift A vs C in sigma | PASS |
| 288 | ch:level2:L288 | measured | `10972.07` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run C lowest chi2 | PASS |
| 288 | ch:level2:L288:10972.61 | measured | `10972.61` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A lowest chi2 | PASS |
| 288 | ch:level2:L288:+0.54 | calc | `+0.54` | numeric: chi2 difference A minus C | PASS |
| 289 | ch:level2:L289 | measured | `10985.08` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 289 | ch:level2:L289:10985.07 | measured | `10985.07` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 289 | ch:level2:L289:-0.01 | calc | `-0.01` | numeric: chain-average chi2 difference | PASS |
| 290 | ch:level2:L290 | measured | `0.0081` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: final R-1 Run C, repeated | PASS |
| 290 | ch:level2:L290:0.0099 | measured | `0.0099` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: final R-1 Run A, repeated | PASS |
| 293 | ch:level2:L293 | calc | `0.009` | numeric: sigma8 drop, rounded values | PASS |
| 293 | ch:level2:L293:0.809 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 Run C, rounded repeat | PASS |
| 293 | ch:level2:L293:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 Run A, rounded repeat | PASS |
| 294 | ch:level2:L294 | calc | `-1.1\%` | numeric: percent sigma8 drop | PASS |
| 294 | ch:level2:L294:-1.51\sigma | calc | `-1.51\sigma` | numeric: sigma8 shift, repeated | PASS |
| 295 | ch:level2:L295 | calc | `0.76` | numeric: likelihood ratio exp(-0.54/2) | PASS |
| 295 | ch:level2:L295:-0.01 | calc | `-0.01` | numeric: chain-average chi2 difference, repeat | PASS |
| 296 | ch:level2:L296 | calc | `13` | numeric: chi2 avg minus lowest, Run C | PASS |
| 299 |  | calc | `+0.09\sigma` | not run: restatement of ch:level2:L286 (+0.09 sigma); printed value has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.0959) | - |
| 300 | ch:level2:L300 | calc | `+0.05\sigma` | numeric: Omega_m shift, repeat | PASS |
| 300 | ch:level2:L300:-1.51\sigma | calc | `-1.51\sigma` | numeric: sigma8 shift, repeat | PASS |
| 304 | ch:level2:L304 | calc | `1.5\sigma` | numeric: sigma8 shift, rounded repeat | PASS |
| 308 | ch:level2:L308 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 Run A posterior mean, repeat | PASS |
| 308 | ch:level2:L308:0.8087 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 Run C posterior mean, repeat | PASS |
| 314 | ch:level2:L314 | calc | `-1.51\sigma` | numeric: sigma8 shift, repeat | PASS |
| 315 | ch:level2:L315 | calc | `-0.78\sigma` | numeric: S8 shift, repeat | PASS |
| 316 |  | calc | `+0.09\sigma` | not run: restatement of ch:level2:L286 in a caption (+0.09 sigma); printed value has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.0959) | - |
| 320 | ch:level2:L320 | calc | `0.06\sigma` | numeric: Run D vs A H0 agreement in sigma | PASS |
| 321 | ch:level2:L321 | measured | `0.7995\pm0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run D posterior sigma8 mean | PASS |
| 321 | ch:level2:L321:0.7998\pm0.0058 | measured | `0.7998\pm0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A posterior sigma8 mean, repeat | PASS |
| 322 | ch:level2:L322 | calc | `0.00\sigma` | numeric: Omega_m shift Run D vs C | PASS |
| 322 |  | calc | `+0.08\sigma` | not run: printed value +0.08 sigma has one significant digit: a 5 % change (0.004) is below the rounding half-unit (0.005) and cannot be told from rounding (recomputed 0.0836 from Run D and Run C ln10^10As in verify_late_time_level2_output.txt section D) | - |
| 325 | ch:level2:L325 | measured | `0.542` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 325 | ch:level2:L325:0.498 | measured | `0.498` | heavy file `docs/verification/chains/data/growth_on.json`: density growth rate f at z = 0, switch on (CAMB record) | PASS |
| 326 | ch:level2:L326 | calc | `8.1\%` | numeric: percent diff f velocity vs density | PASS |
| 491 | ch:level2:L491 | calc | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2_min difference between IAM and LCDM runs | PASS |
| 492 | ch:level2:L492 | record | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 before (LCDM-like run) | PASS |
| 492 | ch:level2:L492:0.800 | record | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 after (IAM run) | PASS |
| 493 | ch:level2:L493:0.1 | calc | `0.1` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: largest standard-parameter shift A vs C, in sigma | PASS |
| 495 | ch:level2:L495 | record | `67.16` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 495 | ch:level2:L495:0.37 | calc | `0.37` | numeric: sigma tension vs Planck 2018 H0 | PASS |
| 496 | ch:level2:L496 | record | `72.26` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: measured: printed value found in verify_late_time_level2_output.txt, a file the chapter names | PASS |
| 496 | ch:level2:L496:0.75 | calc | `0.75` | numeric: sigma tension vs SH0ES 2022 H0 | PASS |

## Part 2 - ch:dsnote - `docs/book/part2/p2_05_dual_sector_note.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 20 | eq:dsn_timelike | none |  | not run: definition: timelike geodesic normalization (GR) | - |
| 21 | eq:dsn_null | none |  | not run: definition: null geodesic condition (GR) | - |
| 72 | eq:dsn_firstlaw | none |  | not run: definition: IAM modified first law (postulated) | - |
| 80 | eq:dsn_iff | derived |  | sympy: d tau = 0 on null worldlines, > 0 on timelike ones | PASS |
| 88 | eq:dsn_mu | derived |  | sympy: mu = H^2/(H^2 + beta_m E H0^2) from the matter-sector Hubble time | PASS |
| 89 | eq:dsn_sigma | none |  | not run: trivial definition: photon sector unmodified | - |
| 92 | ch:dsnote:L92 | derived | `0.15765` | numeric: beta_m = Omega_m/2 virial coupling | PASS |
| 93 | ch:dsnote:L93 | derived | `0.864` | numeric: mu at z=0 from beta_m | PASS |
| 96 |  | none | `4/3` | not run: cited f(R) gravity bound on mu (external) | - |
| 99 | ch:dsnote:L99 | prediction | `0.864` | numeric: mu(0) restated as prediction | PASS |
| 111 | ch:dsnote:L111 | calc | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: photon coupling bound from theta_s fit | PASS |
| 111 | ch:dsnote:L111:0.033 | calc | `0.033` | numeric: sector ratio beta_gamma/beta_m from the committed 95 % bound | PASS |
| 112 | ch:dsnote:L112 | calc | `30` | numeric: 'at least 30x': beta_m/beta_gamma bound, whole multiples | PASS |
| 113 | ch:dsnote:L113 | measured | `0.1583` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Omega_m/2 from Planck posterior (beta_m-fixed chain) | PASS |
| 113 | ch:dsnote:L113:0.0032 | measured | `0.0032` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: uncertainty on Omega_m/2 posterior | PASS |
| 114 | ch:dsnote:L114 | measured | `0.2` | numeric: sigma deviation of Omega_m/2 from beta_m | PASS |
| 114 | ch:dsnote:L114:0.15765 | measured | `0.15765` | numeric: beta_m value restated for comparison | PASS |
| 117 | ch:dsnote:L117 | observed | `-0.035` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: best beta in the Pantheon+ distances (full covariance) | PASS |
| 117 | ch:dsnote:L117:-0.068 | observed | `-0.068` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: lower 68 % edge of beta in the Pantheon+ distances | PASS |
| 118 | ch:dsnote:L118 | observed | `73.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 126 | ch:dsnote:L126 | measured | `0.010` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: worst-case chain convergence R-1 | PASS |
| 127 | ch:dsnote:L127 | calc | `0.76` | numeric: likelihood ratio from Delta chi^2=0.54 | PASS |
| 127 | ch:dsnote:L127:0.54 | calc | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Delta chi2 lowest points, Run A minus Run C | PASS |
| 129 | ch:dsnote:L129 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM chain sigma8 | PASS |
| 129 | ch:dsnote:L129:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 beta_m-fixed chain sigma8 | PASS |
| 130 | ch:dsnote:L130 | measured | `67.16` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 130 | ch:dsnote:L130:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 posterior sd (Run A) | PASS |
| 130 | ch:dsnote:L130:0.37 | measured | `0.37` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 against Planck, sigma | PASS |
| 131 | ch:dsnote:L131 | calc | `72.26` | numeric: matter-sector H0 from photon H0 and beta_m | PASS |
| 131 | ch:dsnote:L131:0.50 | calc | `0.50` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 error, sd x sqrt(1 + beta_m) | PASS |
| 131 | ch:dsnote:L131:0.75 | calc | `0.75` | numeric: matter-sector H0 against SH0ES, sigma | PASS |
| 132 | ch:dsnote:L132 | measured | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: background-level chain H0 (falsification test) | PASS |
| 132 | ch:dsnote:L132:0.42 | measured | `0.42` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: uncertainty on background chain H0 (runA) | PASS |
| 132 | ch:dsnote:L132:61.52 | measured | `61.52` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: second background chain H0 | PASS |
| 132 | ch:dsnote:L132:0.43 | measured | `0.43` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: uncertainty on background chain H0 (runD) | PASS |
| 133 | ch:dsnote:L133 | measured | `10.9` | numeric: sigma below Planck H0 (background chains) | PASS |
| 133 | ch:dsnote:L133:8.6 | measured | `8.6` | numeric: sigma below Planck, chain error in quadrature | PASS |
| 135 | ch:dsnote:L135 | calc | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: photon coupling bound restated | PASS |
| 135 | ch:dsnote:L135:0.033 | calc | `0.033` | numeric: sector ratio beta_gamma/beta_m from the committed 95 % bound | PASS |
| 136 | ch:dsnote:L136 | observed | `23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 145 | ch:dsnote:L145 | calc | `1.4` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: sigma offset of the beta_gamma=0 point from the observed theta_s (committed output) | PASS |
| 145 | ch:dsnote:L145:0.0052 | calc | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: photon coupling bound (figure) | PASS |
| 145 | ch:dsnote:L145:0.033 | calc | `0.033` | numeric: sector ratio beta_gamma/beta_m from the committed 95 % bound | PASS |
| 145 | ch:dsnote:L145:0.90 | calc | `0.90` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: theta_s shift from full matter coupling | PASS |
| 145 | ch:dsnote:L145:30 | calc | `30` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: sigma significance of theta_s shift | PASS |
| 145 |  | none | `144.43` | not run: input: sound horizon fixed value | - |
| 145 |  | none | `0.0104110` | not run: input: Planck theta_s measurement | - |
| 145 |  | none | `0.0000031` | not run: input: theta_s uncertainty | - |
| 145 |  | none | `67.4` | not run: text changed at HEAD; input: background H0 for toy model | - |
| 145 |  | none | `0.315` | not run: text changed at HEAD; input: Omega_m for toy model | - |
| 145 |  | none | `9.24\times10^{-5}` | not run: text changed at HEAD; input: Omega_r for toy model | - |
| 150 | ch:dsnote:L150 | calc | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: photon coupling bound restated | PASS |
| 151 | ch:dsnote:L151 | calc | `3.8\times10^{-4}` | numeric: 1-mu at z=3 | PASS |
| 152 |  | calc | `1.5\times10^{-5}` | not run: printed value 1.5e-5 has two significant digits: a 5 % change (0.075e-5) is within rounding plus the control tolerance of the recomputed 1.544e-5 (1 - mu at z = 5), so it cannot be told from rounding; the z = 3 value is bounded at ch:dsvalidation:L515 | - |
| 154 | ch:dsnote:L154:0.04 | calc | `0.04` | heavy file `docs/verification/scripts/verify_limber_lensing_output.txt`: Limber: lensing power lowered, low end over 30 <= L <= 1000 (per cent) | PASS |
| 154 | ch:dsnote:L154:0.24 | calc | `0.24` | heavy file `docs/verification/scripts/verify_limber_lensing_output.txt`: Limber: lensing power lowered, high end over 30 <= L <= 1000 (per cent) | PASS |
| 158 | ch:dsnote:L158 | derived | `0.15765` | numeric: beta_m restated in figure caption | PASS |
| 158 | ch:dsnote:L158:67.16 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 chain H0 restated | PASS |
| 159 | ch:dsnote:L159 | measured | `67.16` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 159 | ch:dsnote:L159:72.26 | calc | `72.26` | numeric: matter-sector H0 restated | PASS |
| 160 | ch:dsnote:L160 | calc | `0.864` | numeric: mu at z=0 restated | PASS |
| 160 | ch:dsnote:L160:0.888 | calc | `0.888` | numeric: mu at z=0.11 | PASS |
| 160 | ch:dsnote:L160:0.965 | calc | `0.965` | numeric: mu at z=0.69 | PASS |
| 160 | ch:dsnote:L160:0.999 | calc | `0.999` | numeric: mu at z=2.3 | PASS |
| 161 | ch:dsnote:L161 | calc | `90` | numeric: activation E at z=0.11 | PASS |
| 161 | ch:dsnote:L161:50 | calc | `50` | numeric: activation E at z=0.69 | PASS |
| 161 | ch:dsnote:L161:10 | calc | `10` | numeric: activation E at z=2.3 | PASS |
| 162 | ch:dsnote:L162 | calc | `6.1` | numeric: sector gap H_m/H-1 at z=0.11 | PASS |
| 162 | ch:dsnote:L162:1.8 | calc | `1.8` | numeric: sector gap at z=0.69 | PASS |
| 162 | ch:dsnote:L162:0.1 | calc | `0.1` | numeric: sector gap at z=2.3 | PASS |
| 163 | ch:dsnote:L163 | calc | `-13.6` | numeric: matter density suppression at z=0 | PASS |
| 168 | ch:dsnote:L168 | calc | `-9.4` | numeric: f sigma8 rel. LambdaCDM, z = 0, term in the background only (B) | PASS |
| 169 | ch:dsnote:L169 | calc | `-4.25` | numeric: f sigma8 rel. LambdaCDM, z = 0, perturbations only (C) | PASS |
| 169 | ch:dsnote:L169:-1.35 | calc | `-1.35` | numeric: f sigma8 rel. LambdaCDM, z = 0.5, model C | PASS |
| 169 | ch:dsnote:L169:-0.41 | calc | `-0.41` | numeric: f sigma8 rel. LambdaCDM, z = 1, model C | PASS |
| 169 | ch:dsnote:L169:-13.2 | calc | `-13.2` | numeric: f sigma8 rel. LambdaCDM, z = 0, both placements (D) | PASS |
| 170 |  | none | `61.5` | not run: H0 background restated approx | - |
| 171 | ch:dsnote:L171 | measured | `0.830` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 LCDM chain S8 | PASS |
| 171 | ch:dsnote:L171:0.011 | measured | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: uncertainty on LCDM chain S8 | PASS |
| 171 | ch:dsnote:L171:0.822 | measured | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level2 beta_m-fixed chain S8 | PASS |
| 171 | ch:dsnote:L171:0.011' | measured | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: uncertainty on beta_m-fixed chain S8 | PASS |
| 171 |  | none | `0.832` | not run: Planck 2018 published S8 (external) | - |
| 171 |  | none | `0.013` | not run: uncertainty on Planck S8 | - |
| 172 | ch:dsnote:L172 | observed | `0.815` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 172 | ch:dsnote:L172:0.776 | observed | `0.776` | file `docs/verification/PAPER_ERRATA.md`: DES Y3 S8 as recorded in PAPER_ERRATA.md, a file the chapter names | PASS |
| 172 | ch:dsnote:L172:0.017 | observed | `0.017` | file `docs/verification/PAPER_ERRATA.md`: DES Y3 3x2pt S8 error (errata ledger) | PASS |
| 173 | ch:dsnote:L173 | observed | `0.759` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 176 | ch:dsnote:L176 | calc | `4.25` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 176 | ch:dsnote:L176:2.17 | calc | `2.17` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 176 | ch:dsnote:L176:1.35 | calc | `1.35` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 176 | ch:dsnote:L176:0.41 | calc | `0.41` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 177 | ch:dsnote:L177 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 restated (LCDM Level2) | PASS |
| 177 | ch:dsnote:L177:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 restated (beta_m-fixed Level2) | PASS |
| 177 | ch:dsnote:L177:0.830 | measured | `0.830` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 restated (LCDM Level2) | PASS |
| 177 | ch:dsnote:L177:0.822 | measured | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 restated (beta_m-fixed Level2) | PASS |
| 177 | ch:dsnote:L177:0.8 | measured | `0.8` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: S8 shift, Run C to Run A, in sigma | PASS |
| 180 | ch:dsnote:L180 | interp | `0.69` | numeric: redshift where E(a) is half present value | PASS |
| 180 | ch:dsnote:L180:0.11 | interp | `0.11` | numeric: redshift where E(a) reaches 90% of present value | PASS |
| 188 | ch:dsnote:L188 | calc | `1.14` | numeric: sigma distance of prediction from DES Y3 | PASS |
| 188 | ch:dsnote:L188:0.46 | calc | `0.46` | numeric: sigma distance of prediction from DESI BAO+BBN | PASS |
| 188 | ch:dsnote:L188:0.80 | calc | `0.80` | numeric: sigma distance of prediction from DESI CMB+DES Y3 | PASS |
| 188 | ch:dsnote:L188:0.82 | calc | `0.82` | numeric: sigma distance of prediction from ACT combo | PASS |
| 188 | ch:dsnote:L188:0.42 | calc | `0.42` | numeric: GR max sigma distance from published mu0 values | PASS |
| 188 | ch:dsnote:L188:0.08 | observed | `0.08^{+0.21}_{-0.19}` | file `docs/verification/PAPER_ERRATA.md`: published mu0, DES Y3 with external data | PASS |
| 188 | ch:dsnote:L188:0.11 | observed | `0.11^{+0.45}_{-0.54}` | file `docs/verification/scripts/verify_theory_derivations_output.txt`: published mu0, DESI 2024 full shape + BAO + BBN | PASS |
| 188 | ch:dsnote:L188:0.04 | observed | `0.04\pm0.22` | file `docs/verification/scripts/verify_entropic_gravity_output.txt`: published mu0, DESI full shape + CMB + DES Y3 | PASS |
| 188 | ch:dsnote:L188:0.02 | observed | `0.02\pm0.19` | file `docs/verification/scripts/verify_entropic_gravity_output.txt`: published mu0, ACT + WMAP + SDSS + SN | PASS |
| 188 |  | prediction | `-0.136` | not run: canon mu0 prediction value, restated in caption | - |
| 193 |  | prediction | `1` | not run: exact photon-sector prediction Sigma(a)=1 | - |
| 197 |  | none | `0.0039` | not run: text changed at HEAD; present beta_gamma bound, restated, no macro | - |
| 199 |  | prediction | `1/2` | not run: definition beta_m=Om/2, virial ratio restated | - |
| 212 |  | prediction | `1` | not run: table: exact prediction Sigma=1, restated | - |
| 213 | ch:dsnote:L213:0.0052 | calc | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma 95 % bound in the falsifier table (committed output) | PASS |
| 213 |  | calc | `0` | not run: prediction, nothing to recompute: beta_gamma = 0 is the photon exemption itself (eq:dsn_iff, checked there); the measured side of the row is the bound 0.0052 (ch:dsnote:L213:0.0052) | - |
| 214 |  | prediction | `1/2` | not run: table: definition beta_m=Om/2, restated | - |
| 215 |  | prediction | `-0.136` | not run: table Value column: canon mu0 prediction restated | - |
| 234 | ch:dsnote:L234 | record | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Planck-only chi2 diff IAM vs LCDM, level-2 chains | PASS |
| 235 | ch:dsnote:L235 | record | `0.2` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Planck posterior Omega_m/2 against fixed beta_m, sigma | PASS |
| 236 | ch:dsnote:L236 | record | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma 95 % bound in the summary (committed output) | PASS |

## Part 2 - ch:s8trend - `docs/book/part2/p2_08_s8_trend.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 15 |  | observed | `3` | not run: measured, source not named: approximate statement of the cited trend analysis (MNRAS 528, L20, 2024; arXiv 2303.06928), '~3 sigma below Planck at low redshift'; the source does not tabulate it and no repository file records it | - |
| 15 |  | observed | `1` | not run: measured, source not named: approximate statement of the cited trend analysis (MNRAS 528, L20, 2024; arXiv 2303.06928), 'within 1 sigma at high redshift'; the source does not tabulate it and no repository file records it | - |
| 20 |  | calc | `a tenth` | not run: rough statement: "about a tenth" of the low-redshift deficit; a 5 % change cannot be told from a word value (recomputed 0.104 = 0.78/7.5) | - |
| 25 | ch:s8trend:L25 | measured | `0.3111+/-0.0056` | file `docs/verification/scripts/verify_s8_trend.py`: Omega_m prior of the trend analysis (Planck + BAO) | PASS |
| 26 |  | observed | `3` | not run: measured, source not named: approximate statement of the cited trend analysis (MNRAS 528, L20, 2024; arXiv 2303.06928), '~3 sigma tension at lower redshifts'; the source does not tabulate it and no repository file records it | - |
| 26 |  | observed | `1` | not run: measured, source not named: approximate statement of the cited trend analysis (MNRAS 528, L20, 2024; arXiv 2303.06928), 'consistent within 1 sigma at high redshifts'; the source does not tabulate it and no repository file records it | - |
| 27 | ch:s8trend:L27:1.6 | observed | `1.6` | file `docs/book/verification/sources/ts_MANIFEST_sector_s8.md`: trend significance, 20-point sample (read ledger) | PASS |
| 28 | ch:s8trend:L28 | observed | `2.8` | file `docs/book/verification/sources/ts_MANIFEST_sector_s8.md`: trend significance, 66-point sample (read ledger) | PASS |
| 35 | ch:s8trend:L35 | observed | `0.832` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: measured: printed value found in verify_s8_trend_output.txt, a file the chapter names | PASS |
| 40 | ch:s8trend:L40 | observed | `0.633(+0.025/-0.024)` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: measured: printed value found in verify_s8_trend_output.txt, a file the chapter names | PASS |
| 40 | ch:s8trend:L40:3.7 | observed | `3.7` | file `docs/book/verification/sources/ts_MANIFEST_sector_s8.md`: gamma = 0.55 excluded at 3.7 sigma (Nguyen 2023, read ledger) | PASS |
| 40 |  | none | `0.55` | not run: GR growth-index prediction, input | - |
| 41 | ch:s8trend:L41:4.2 | observed | `4.2` | file `docs/book/verification/sources/ts_MANIFEST_sector_s8.md`: f sigma8 + Planck only: 4.2 sigma (Nguyen 2023, read ledger) | PASS |
| 41 |  | observed | `0.639(+0.024/-0.025)` | not run: measured, source not named: gamma = 0.639 +0.024 -0.025 (f sigma8 + Planck only) of Nguyen, Huterer and Wen 2023 (PRL 131, 111001); the read ledger records 0.633, 3.7 sigma and 4.2 sigma from the abstract but not 0.639, and no other repository file has it | - |
| 48 | eq:s8_mu | none |  | not run: definition of modified coupling μ(a) | - |
| 52 | eq:s8_Ea | none |  | not run: definition of activation function E(a) | - |
| 56 | eq:s8_beta | derived | `0.15765` | numeric: coupling constant from Om/2 | PASS |
| 56 |  | none | `0.3153` | not run: input, Planck Om | - |
| 63 | ch:s8trend:L63 | derived |  | sympy: E(a) at a=1 equals 1 | PASS |
| 64 | ch:s8trend:L64 | derived |  | sympy: flat universe Om+OL=1 | PASS |
| 66 | ch:s8trend:L66 | derived |  | sympy: mu(1) simplification | PASS |
| 69 | eq:s8_mu0 | derived | `-0.1362` | numeric: mu0 from beta_m | PASS |
| 73 |  | none | `-0.135` | not run: input, MGCAMB Level-1 chain amplitude | - |
| 74 | ch:s8trend:L74 | calc | `about one per cent` | numeric: mu0 diff vs free-mu0 posterior width | PASS |
| 77 | ch:s8trend:L77 | calc | `2.30` | numeric: z threshold where E(a)<0.1 | PASS |
| 78 | ch:s8trend:L78 | calc | `0.864` | numeric: mu at z=0 | PASS |
| 79 | ch:s8trend:L79 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 79 | ch:s8trend:L79:0.982 | calc | `0.982` | numeric: mu at z=1 | PASS |
| 79 | ch:s8trend:L79:0.998 | calc | `0.998` | numeric: mu at z=2 | PASS |
| 85 | ch:s8trend:L85 | calc | `0.8255` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 85 | ch:s8trend:L85:0.78 | calc | `0.78` | numeric: lensing deficit today from S8 ratio | PASS |
| 85 | ch:s8trend:L85:1.68 | calc | `1.68` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 85 | ch:s8trend:L85:0.633(+0.025/-0.024) | observed | `0.633(+0.025/-0.024)` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: measured: printed value found in verify_s8_trend_output.txt, a file the chapter names | PASS |
| 85 | ch:s8trend:L85:0.585 | calc | `0.585` | numeric: effective growth index today, IAM | PASS |
| 85 | ch:s8trend:L85:0.554 | calc | `0.554` | numeric: effective growth index today, LambdaCDM | PASS |
| 90 |  | none | `0.832` | not run: input, Planck S8 value | - |
| 95 |  | calc | `seventeenfold` | not run: rough statement: "overstates the effect about seventeenfold"; a 5 % change cannot be told from a word value (recomputed 17.53 = (1 - mu(1)) over the z = 0 amplitude deficit) | - |
| 98 | ch:s8trend:L98 | calc | `0.864` | numeric: mu at z=0 | PASS |
| 98 | ch:s8trend:L98:0.8255 | calc | `0.8255` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=0 | PASS |
| 98 | ch:s8trend:L98:0.78 | calc | `0.78` | numeric: D deficit, z=0 | PASS |
| 98 | ch:s8trend:L98:0.965 | calc | `0.965` | numeric: f_IAM/f_LCDM, z=0 | PASS |
| 98 | ch:s8trend:L98:0.7966 | calc | `0.7966` | numeric: growth-rate S8 = S8_Planck (f sigma8 ratio), z=0 | PASS |
| 98 | ch:s8trend:L98:0.914 | calc | `0.914` | numeric: mu at z=0.25 | PASS |
| 98 | ch:s8trend:L98:0.8285 | calc | `0.8285` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=0.25 | PASS |
| 98 | ch:s8trend:L98:0.42 | calc | `0.42` | numeric: D deficit, z=0.25 | PASS |
| 98 | ch:s8trend:L98:0.980 | calc | `0.980` | numeric: f_IAM/f_LCDM, z=0.25 | PASS |
| 98 | ch:s8trend:L98:0.922 | calc | `0.922` | numeric: mu at z=0.3 | PASS |
| 98 | ch:s8trend:L98:0.8289 | calc | `0.8289` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=0.3 | PASS |
| 98 | ch:s8trend:L98:0.37 | calc | `0.37` | numeric: D deficit, z=0.3 | PASS |
| 98 | ch:s8trend:L98:0.8140 | calc | `0.8140` | numeric: growth-rate S8 = S8_Planck (f sigma8 ratio), z=0.3 | PASS |
| 99 | ch:s8trend:L99 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 99 | ch:s8trend:L99:0.8302 | calc | `0.8302` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=0.5 | PASS |
| 99 | ch:s8trend:L99:0.22 | calc | `0.22` | numeric: D deficit, z=0.5 | PASS |
| 99 | ch:s8trend:L99:0.989 | calc | `0.989` | numeric: f_IAM/f_LCDM, z=0.5 | PASS |
| 99 | ch:s8trend:L99:0.8208 | calc | `0.8208` | numeric: growth-rate S8 = S8_Planck (f sigma8 ratio), z=0.5 | PASS |
| 99 | ch:s8trend:L99:0.982 | calc | `0.982` | numeric: mu at z=1.0 | PASS |
| 99 | ch:s8trend:L99:0.8315 | calc | `0.8315` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=1.0 | PASS |
| 99 | ch:s8trend:L99:0.06 | calc | `0.06` | numeric: D deficit, z=1.0 | PASS |
| 99 | ch:s8trend:L99:0.997 | calc | `0.997` | numeric: f_IAM/f_LCDM, z=1.0 | PASS |
| 99 | ch:s8trend:L99:0.8286 | calc | `0.8286` | numeric: growth-rate S8 = S8_Planck (f sigma8 ratio), z=1.0 | PASS |
| 99 | ch:s8trend:L99:0.998 | calc | `0.998` | numeric: mu at z=2.0 | PASS |
| 99 | ch:s8trend:L99:0.8320 | calc | `0.8320` | numeric: lensing S8 = S8_Planck D_IAM/D_LCDM, z=2.0 | PASS |
| 99 | ch:s8trend:L99:0.01 | calc | `0.01` | numeric: D deficit, z=2.0 | PASS |
| 99 | ch:s8trend:L99:1.000 | calc | `1.000` | numeric: f_IAM/f_LCDM, z=2.0 | PASS |
| 99 | ch:s8trend:L99:0.8316 | calc | `0.8316` | numeric: growth-rate S8 = S8_Planck (f sigma8 ratio), z=2.0 | PASS |
| 101 | ch:s8trend:L101 | calc | `0.8` | numeric: rounded lensing deficit today | PASS |
| 101 |  | measured | `6--9` | not run: range bracket of published survey values, not one rounded number: KiDS-1000 shear 0.759 (8.8 % below 0.832) and DES Y3 3x2pt 0.776 (6.7 % below), both inside 6-9 % (values in verify_sector_tension_output.txt section 11); see for_author | - |
| 102 | ch:s8trend:L102 | calc | `4.25` | numeric: fσ8 deficit today | PASS |
| 102 | ch:s8trend:L102:2.17 | calc | `2.17` | numeric: f sigma8 deficit z=0.3 | PASS |
| 102 | ch:s8trend:L102:1.35 | calc | `1.35` | numeric: fσ8 deficit at z=0.5 | PASS |
| 102 | ch:s8trend:L102:0.41 | calc | `0.41` | numeric: fσ8 deficit at z=1 | PASS |
| 106 |  | none | `0.3153` | not run: input, Om fixed for fit | - |
| 111 | ch:s8trend:L111 | calc | `0.8231` | numeric: trend analysis on the term: inferred S8, z_min 0.0 | PASS |
| 111 | ch:s8trend:L111:0.8249 | calc | `0.8249` | numeric: trend analysis on the term: inferred S8, z_min 0.2 | PASS |
| 111 | ch:s8trend:L111:0.8263 | calc | `0.8263` | numeric: trend analysis on the term: inferred S8, z_min 0.4 | PASS |
| 111 | ch:s8trend:L111:0.8282 | calc | `0.8282` | numeric: trend analysis on the term: inferred S8, z_min 0.6 | PASS |
| 111 | ch:s8trend:L111:0.8298 | calc | `0.8298` | numeric: trend analysis on the term: inferred S8, z_min 0.8 | PASS |
| 111 | ch:s8trend:L111:0.8306 | calc | `0.8306` | numeric: trend analysis on the term: inferred S8, z_min 1.0 | PASS |
| 112 | ch:s8trend:L112 | calc | `0.027` | numeric: trend analysis on the term: statistical sigma(S8), z_min 0.0 | PASS |
| 112 | ch:s8trend:L112:0.028 | calc | `0.028` | numeric: trend analysis on the term: statistical sigma(S8), z_min 0.2 | PASS |
| 112 | ch:s8trend:L112:0.030 | calc | `0.030` | numeric: trend analysis on the term: statistical sigma(S8), z_min 0.4 | PASS |
| 112 | ch:s8trend:L112:0.035 | calc | `0.035` | numeric: trend analysis on the term: statistical sigma(S8), z_min 0.6 | PASS |
| 112 | ch:s8trend:L112:0.043 | calc | `0.043` | numeric: trend analysis on the term: statistical sigma(S8), z_min 0.8 | PASS |
| 112 | ch:s8trend:L112:0.052 | calc | `0.052` | numeric: trend analysis on the term: statistical sigma(S8), z_min 1.0 | PASS |
| 114 | ch:s8trend:L114 | calc | `1.1` | numeric: inferred S8 deviation at zmin=0 | PASS |
| 114 | ch:s8trend:L114:0.2 | calc | `0.2` | numeric: inferred S8 deviation at zmin=1.0 | PASS |
| 115 | ch:s8trend:L115 | calc | `0.3` | numeric: rise in units of statistical sigma | PASS |
| 115 |  | observed | `3` | not run: measured, source not named: approximate statement of the cited trend analysis (MNRAS 528, L20, 2024; arXiv 2303.06928), '~3 sigma low-redshift offset of the measured trend'; the source does not tabulate it and no repository file records it | - |
| 118 | ch:s8trend:L118 | calc | `2.30` | numeric: repeat threshold where E(a)<0.1 | PASS |
| 122 | ch:s8trend:L122 | record | `0.8143` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: LCDM chain sigma8, Planck-only | PASS |
| 122 | ch:s8trend:L122:0.8015 | record | `0.8015` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: IAM chain sigma8, Planck-only | PASS |
| 122 | ch:s8trend:L122:1.68 | calc | `1.68` | numeric: amplitude deficit today, MGCAMB chain form | PASS |
| 123 | ch:s8trend:L123 | calc | `1.6` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: sigma8 shift percent, Planck chains | PASS |
| 123 | ch:s8trend:L123:half | calc | `half` | numeric: exact-form fraction of MGCAMB deficit | PASS |
| 128 | ch:s8trend:L128 | calc | `61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from background-coupled Level2b chains | PASS |
| 129 | ch:s8trend:L129 | fitted | `0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: delta chi2, perturbation-only vs LCDM | PASS |
| 132 | ch:s8trend:L132 | derived | `0.15765` | numeric: repeat beta_m from Om/2 | PASS |
| 133 | ch:s8trend:L133 | measured | `0.3166+/-0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level-2 posterior Omega_m | PASS |
| 133 | ch:s8trend:L133:0.1583+/-0.0032 | calc | `0.1583+/-0.0032` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Omega_m/2 from Level-2 posterior | PASS |
| 134 | ch:s8trend:L134 | calc | `0.2` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma distance from fixed beta_m | PASS |
| 143 | ch:s8trend:L143 | calc | `1.16` | numeric: M_lens/M_dyn today | PASS |
| 143 | ch:s8trend:L143:1.05 | calc | `1.05` | numeric: M_lens/M_dyn at z=0.5 | PASS |
| 190 |  | prediction | `-0.136` | not run: predicted coupling mu0, canon locked input | - |
| 190 |  | prediction | `0` | not run: predicted slip parameter Sigma_0, definition | - |
| 194 | ch:s8trend:L194 | calc | `3.1\%` | numeric: f sigma8 deficit at z = 0.15 | PASS |
| 194 | ch:s8trend:L194:1.9% | calc | `1.9\%` | numeric: f sigma8 deficit at z = 0.35 | PASS |
| 194 | ch:s8trend:L194:0.4% | calc | `0.4\%` | numeric: f sigma8 deficit at z = 1.05 | PASS |
| 194 |  | calc | `0.9\%` | not run: printed value 0.9 % has one significant digit: a 5 % change (0.045) is below the rounding half-unit (0.05) and cannot be told from rounding (recomputed 0.941 % f sigma8 deficit at z = 0.65) | - |
| 195 | ch:s8trend:L195 | calc | `0.1\%` | numeric: f sigma8 deficit at z = 1.55 | PASS |
| 196 | ch:s8trend:L196 | calc | `5.0\sigma` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 203 | ch:s8trend:L203 | calc | `0.8\%` | numeric: amplitude deficit today (status) | PASS |
| 203 | ch:s8trend:L203:4.25% | calc | `4.25\%` | numeric: growth-rate deficit today (status) | PASS |
| 203 |  | calc | `a tenth` | not run: ratio to low-z deficit, imprecise restatement | - |
| 204 | ch:s8trend:L204 | calc | `40\%` | numeric: growth index moved toward the measured value, per cent of the way | PASS |
| 204 | ch:s8trend:L204:0.3 | calc | `0.3\sigma` | numeric: rise of inferred S8 with z_min in statistical sigma | PASS |
| 206 | ch:s8trend:L206 | calc | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi2 IAM vs LCDM, Level2 chains | PASS |

## Part 2 - ch:sectortension - `docs/book/part2/p2_09_sector_tension.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 16 | ch:sectortension:L16 | calc | `0.35` | numeric: phantom-crossing z, DESI+CMB+Pantheon+ fit | PASS |
| 16 | ch:sectortension:L16:0.50 | calc | `0.50` | numeric: phantom-crossing z, DESI+CMB fit | PASS |
| 20 | ch:sectortension:L20 | calc | `0.15765` | numeric: beta_m=Omega_m/2 virial partition | PASS |
| 21 | ch:sectortension:L21 | calc | `0.2` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Planck posterior vs fixed beta_m sigma | PASS |
| 30 | ch:sectortension:L30 | measured | `+0.54` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2 difference term vs LCDM best points | PASS |
| 30 | ch:sectortension:L30:0.010 | measured | `0.010` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: max chain convergence R-1 across 18 chains | PASS |
| 30 | ch:sectortension:L30:0.7998 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: IAM term sigma8 posterior | PASS |
| 31 | ch:sectortension:L31 | observed | `0.802` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 31 | ch:sectortension:L31:0.1 | calc | `0.1` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma agreement IAM vs lensing sigma8 | PASS |
| 31 | ch:sectortension:L31:0.1' | calc | `0.1` | numeric: fsigma8 deficit at QSO redshift z=1.491 | PASS |
| 31 | ch:sectortension:L31:2.2 | calc | `2.2` | numeric: fsigma8 deficit at BGS redshift z=0.295 | PASS |
| 41 | ch:sectortension:L41 | observed | `67.4` | numeric: Planck 2018 H0 under LambdaCDM | PASS |
| 42 | ch:sectortension:L42 | calc | `4.9` | numeric: Hubble tension significance Planck vs SH0ES | PASS |
| 42 | ch:sectortension:L42:73.04 | observed | `73.04` | numeric: SH0ES H0 | PASS |
| 43 | ch:sectortension:L43 | observed | `70.39` | numeric: TRGB H0 (CCHP) | PASS |
| 47 | ch:sectortension:L47 | observed | `0.766` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 47 | ch:sectortension:L47:0.832 | observed | `0.832` | numeric: Planck LambdaCDM S8 = sigma8 (Om/0.3)^0.5 | PASS |
| 48 | ch:sectortension:L48 | observed | `0.759` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 48 | ch:sectortension:L48:0.776 | observed | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DES Y3 3x2pt S8 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 49 | ch:sectortension:L49 | observed | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: HSC Y3 S8 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 49 | ch:sectortension:L49:2 | observed | `2` | numeric: S8 tension, lower end over the four lensing surveys | PASS |
| 49 | ch:sectortension:L49:3 | observed | `3` | numeric: S8 tension, upper end over the four lensing surveys | PASS |
| 50 | ch:sectortension:L50 | observed | `0.815` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 50 | ch:sectortension:L50:0.814 | observed | `0.814` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 51 | ch:sectortension:L51 | observed | `0.802` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 53 | ch:sectortension:L53 | observed | `2.6` | numeric: DESI DR1 BAO+CMB w0wa preference | PASS |
| 54 | ch:sectortension:L54 | observed | `2.5` | numeric: DESI DR1 + Pantheon+ preference | PASS |
| 54 | ch:sectortension:L54:3.5 | observed | `3.5` | numeric: DESI DR1 + Union3 preference | PASS |
| 54 | ch:sectortension:L54:3.9 | observed | `3.9` | numeric: DESI DR1 + DES Y5 preference | PASS |
| 54 | ch:sectortension:L54:2.8 | observed | `2.8` | numeric: DESI DR2 preference, lowest over the supernova compilations | PASS |
| 54 | ch:sectortension:L54:4.2 | observed | `4.2` | numeric: DESI DR2 preference, highest over the supernova compilations | PASS |
| 60 | ch:sectortension:L60 | observed | `-1.75` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 60 | ch:sectortension:L60:0.50 | calc | `0.50` | numeric: phantom-crossing redshift, DESI+CMB | PASS |
| 60 | ch:sectortension:L60:-0.42 | observed | `-0.42` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB w0 | PASS |
| 60 | ch:sectortension:L60:3.1 | observed | `3.1` | numeric: DESI DR2 + CMB significance | PASS |
| 61 | ch:sectortension:L61 | observed | `-0.838` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI+CMB+Pantheon+ w0 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 61 | ch:sectortension:L61:0.35 | calc | `0.35` | numeric: phantom-crossing redshift, DESI+CMB+Pantheon+ | PASS |
| 61 | ch:sectortension:L61:-0.62 | observed | `-0.62` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + Pantheon+ wa | PASS |
| 61 | ch:sectortension:L61:2.8 | observed | `2.8` | numeric: DESI DR2 + CMB + Pantheon+ significance | PASS |
| 62 | ch:sectortension:L62 | observed | `-0.667` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI+CMB+Union3 w0 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 62 | ch:sectortension:L62:-1.09 | observed | `-1.09` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 62 | ch:sectortension:L62:0.44 | calc | `0.44` | numeric: phantom-crossing redshift, DESI+CMB+Union3 | PASS |
| 62 | ch:sectortension:L62:3.8 | observed | `3.8` | numeric: DESI DR2 + CMB + Union3 significance | PASS |
| 63 | ch:sectortension:L63 | observed | `-0.752` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 63 | ch:sectortension:L63:0.41 | calc | `0.41` | numeric: phantom-crossing redshift, DESI+CMB+DES Y5 | PASS |
| 63 | ch:sectortension:L63:-0.86 | observed | `-0.86` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + DES Y5 wa | PASS |
| 63 | ch:sectortension:L63:4.2 | observed | `4.2` | numeric: DESI DR2 + CMB + DES Y5 significance | PASS |
| 80 | ch:sectortension:L80 | calc | `13.6` | numeric: coupling deficit 1-mu today | PASS |
| 80 | ch:sectortension:L80:4.25 | calc | `4.25` | numeric: fsigma8 deficit today z=0 | PASS |
| 80 | ch:sectortension:L80:2.17 | calc | `2.17` | numeric: fsigma8 deficit at z=0.3 | PASS |
| 80 | ch:sectortension:L80:0.41 | calc | `0.41` | numeric: fsigma8 deficit at z=1 | PASS |
| 99 | eq:st_entropy | none |  | not run: definition: entropy budget geometric+informational split | - |
| 110 | eq:st_hm | none |  | not run: definition: modified matter effective expansion rate ansatz | - |
| 208 | eq:st_ode | none |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 213 | ch:sectortension:L213 | fitted | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 term, Level2 Run A chain | PASS |
| 213 | ch:sectortension:L213:0.8087 | fitted | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM, Level2 Run C chain | PASS |
| 216 | ch:sectortension:L216 | calc | `1.11\%` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 % diff between Level2 chains | PASS |
| 216 | ch:sectortension:L216:0.78\% | calc | `0.78\%` | numeric: sigma8 lowered by the growth equation, same early amplitude | PASS |
| 217 | ch:sectortension:L217 | fitted | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM repeated | PASS |
| 217 | ch:sectortension:L217:0.8024 | calc | `0.8024` | numeric: IAM sigma8 estimate from ODE ratio x LCDM sigma8 | PASS |
| 217 | ch:sectortension:L217:0.3\% | calc | `0.3\%` | numeric: offset of ODE estimate above Boltzmann value | PASS |
| 217 | ch:sectortension:L217:0.7998 | fitted | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Boltzmann sigma8 repeated | PASS |
| 223 | ch:sectortension:L223 | fitted | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 term repeated in caption | PASS |
| 223 | ch:sectortension:L223:0.8087 | fitted | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM repeated in caption | PASS |
| 227 | ch:sectortension:L227 | observed | `0.377` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 227 | ch:sectortension:L227:0.460 | calc | `0.460` | numeric: f sigma8 prediction, term, z=0.295, sigma8 0.7998 (Run A) | PASS |
| 227 | ch:sectortension:L227:0.472 | calc | `0.472` | numeric: f sigma8 prediction, LCDM, z=0.295, sigma8 0.8087 (Run C) | PASS |
| 227 | ch:sectortension:L227:-0.88 | calc | `-0.88` | numeric: pull (obs-pred)/sigma, term, z=0.295 | PASS |
| 227 | ch:sectortension:L227:-1.01 | calc | `-1.01` | numeric: pull (obs-pred)/sigma, LCDM, z=0.295 | PASS |
| 227 | ch:sectortension:L227:0.094 | observed | `0.094` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs BGS, sqrt of the DESI ShapeFit-only variance | PASS |
| 228 | ch:sectortension:L228 | observed | `0.514` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR1 LRG1 f sigma_s8 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 228 | ch:sectortension:L228:0.465 | calc | `0.465` | numeric: f sigma8 prediction, term, z=0.51, sigma8 0.7998 (Run A) | PASS |
| 228 | ch:sectortension:L228:0.473 | calc | `0.473` | numeric: f sigma8 prediction, LCDM, z=0.51, sigma8 0.8087 (Run C) | PASS |
| 228 | ch:sectortension:L228:+0.75 | calc | `+0.75` | numeric: pull (obs-pred)/sigma, term, z=0.51 | PASS |
| 228 | ch:sectortension:L228:+0.63 | calc | `+0.63` | numeric: pull (obs-pred)/sigma, LCDM, z=0.51 | PASS |
| 228 | ch:sectortension:L228:0.064 | observed | `0.064` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs LRG1, sqrt of the DESI ShapeFit-only variance | PASS |
| 229 | ch:sectortension:L229 | observed | `0.484` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 229 | ch:sectortension:L229:0.455 | calc | `0.455` | numeric: f sigma8 prediction, term, z=0.706, sigma8 0.7998 (Run A) | PASS |
| 229 | ch:sectortension:L229:0.460 | calc | `0.460` | numeric: f sigma8 prediction, LCDM, z=0.706, sigma8 0.8087 (Run C) | PASS |
| 229 | ch:sectortension:L229:+0.54 | calc | `+0.54` | numeric: pull (obs-pred)/sigma, term, z=0.706 | PASS |
| 229 | ch:sectortension:L229:+0.44 | calc | `+0.44` | numeric: pull (obs-pred)/sigma, LCDM, z=0.706 | PASS |
| 229 | ch:sectortension:L229:0.053 | observed | `0.053` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs LRG2, sqrt of the DESI ShapeFit-only variance | PASS |
| 230 | ch:sectortension:L230 | observed | `0.422` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR1 LRG3 f sigma_s8 in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 230 | ch:sectortension:L230:0.436 | calc | `0.436` | numeric: f sigma8 prediction, term, z=0.919, sigma8 0.7998 (Run A) | PASS |
| 230 | ch:sectortension:L230:0.439 | calc | `0.439` | numeric: f sigma8 prediction, LCDM, z=0.919, sigma8 0.8087 (Run C) | PASS |
| 230 | ch:sectortension:L230:-0.28 | calc | `-0.28` | numeric: pull (obs-pred)/sigma, term, z=0.919 | PASS |
| 230 | ch:sectortension:L230:-0.36 | calc | `-0.36` | numeric: pull (obs-pred)/sigma, LCDM, z=0.919 | PASS |
| 230 | ch:sectortension:L230:0.047 | observed | `0.047` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs LRG3, sqrt of the DESI ShapeFit-only variance | PASS |
| 231 | ch:sectortension:L231 | observed | `0.377` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 231 | ch:sectortension:L231:0.392 | calc | `0.392` | numeric: f sigma8 prediction, term, z=1.317, sigma8 0.7998 (Run A) | PASS |
| 231 | ch:sectortension:L231:0.394 | calc | `0.394` | numeric: f sigma8 prediction, LCDM, z=1.317, sigma8 0.8087 (Run C) | PASS |
| 231 | ch:sectortension:L231:-0.40 | calc | `-0.40` | numeric: pull (obs-pred)/sigma, term, z=1.317 | PASS |
| 231 | ch:sectortension:L231:-0.45 | calc | `-0.45` | numeric: pull (obs-pred)/sigma, LCDM, z=1.317 | PASS |
| 231 | ch:sectortension:L231:0.037 | observed | `0.037` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs ELG2, sqrt of the DESI ShapeFit-only variance | PASS |
| 232 | ch:sectortension:L232 | observed | `0.435` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 232 | ch:sectortension:L232:0.373 | calc | `0.373` | numeric: f sigma8 prediction, term, z=1.491, sigma8 0.7998 (Run A) | PASS |
| 232 | ch:sectortension:L232:0.374 | calc | `0.374` | numeric: f sigma8 prediction, LCDM, z=1.491, sigma8 0.8087 (Run C) | PASS |
| 232 | ch:sectortension:L232:+1.40 | calc | `+1.40` | numeric: pull (obs-pred)/sigma, term, z=1.491 | PASS |
| 232 | ch:sectortension:L232:+1.36 | calc | `+1.36` | numeric: pull (obs-pred)/sigma, LCDM, z=1.491 | PASS |
| 232 | ch:sectortension:L232:0.044 | observed | `0.044` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs QSO, sqrt of the DESI ShapeFit-only variance | PASS |
| 234 | ch:sectortension:L234 | observed | `0.423` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 234 | ch:sectortension:L234:0.425 | calc | `0.425` | numeric: f sigma8 prediction, term, z=0.067, sigma8 0.7998 (Run A) | PASS |
| 234 | ch:sectortension:L234:0.443 | calc | `0.443` | numeric: f sigma8 prediction, LCDM, z=0.067, sigma8 0.8087 (Run C) | PASS |
| 234 | ch:sectortension:L234:-0.04 | calc | `-0.04` | numeric: pull (obs-pred)/sigma, term, z=0.067 | PASS |
| 234 | ch:sectortension:L234:-0.36 | calc | `-0.36` | numeric: pull (obs-pred)/sigma, LCDM, z=0.067 | PASS |
| 234 | ch:sectortension:L234:0.055 | observed | `0.055` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs 6dFGS | PASS |
| 236 | ch:sectortension:L236 | observed | `0.530` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 236 | ch:sectortension:L236:0.160 | observed | `0.160` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 236 | ch:sectortension:L236:0.442 | calc | `0.442` | numeric: f sigma8 prediction, term, z=0.15, sigma8 0.7998 (Run A) | PASS |
| 236 | ch:sectortension:L236:0.458 | calc | `0.458` | numeric: f sigma8 prediction, LCDM, z=0.15, sigma8 0.8087 (Run C) | PASS |
| 236 | ch:sectortension:L236:+0.55 | calc | `+0.55` | numeric: pull (obs-pred)/sigma, term, z=0.15 | PASS |
| 236 | ch:sectortension:L236:+0.45 | calc | `+0.45` | numeric: pull (obs-pred)/sigma, LCDM, z=0.15 | PASS |
| 238 | ch:sectortension:L238 | observed | `0.500` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 238 | ch:sectortension:L238:0.465 | calc | `0.465` | numeric: f sigma8 prediction, term, z=0.38, sigma8 0.7998 (Run A) | PASS |
| 238 | ch:sectortension:L238:0.475 | calc | `0.475` | numeric: f sigma8 prediction, LCDM, z=0.38, sigma8 0.8087 (Run C) | PASS |
| 238 | ch:sectortension:L238:+0.75 | calc | `+0.75` | numeric: pull (obs-pred)/sigma, term, z=0.38 | PASS |
| 238 | ch:sectortension:L238:+0.54 | calc | `+0.54` | numeric: pull (obs-pred)/sigma, LCDM, z=0.38 | PASS |
| 238 | ch:sectortension:L238:0.047 | observed | `0.047` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs BOSS z 0.38 | PASS |
| 240 | ch:sectortension:L240 | observed | `0.455` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 240 | ch:sectortension:L240:0.465 | calc | `0.465` | numeric: f sigma8 prediction, term, z=0.51, sigma8 0.7998 (Run A) | PASS |
| 240 | ch:sectortension:L240:0.473 | calc | `0.473` | numeric: f sigma8 prediction, LCDM, z=0.51, sigma8 0.8087 (Run C) | PASS |
| 240 | ch:sectortension:L240:-0.26 | calc | `-0.26` | numeric: pull (obs-pred)/sigma, term, z=0.51 | PASS |
| 240 | ch:sectortension:L240:-0.46 | calc | `-0.46` | numeric: pull (obs-pred)/sigma, LCDM, z=0.51 | PASS |
| 240 | ch:sectortension:L240:0.039 | observed | `0.039` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs BOSS z 0.51 | PASS |
| 242 | ch:sectortension:L242 | observed | `0.448` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 242 | ch:sectortension:L242:0.455 | calc | `0.455` | numeric: f sigma8 prediction, term, z=0.7, sigma8 0.7998 (Run A) | PASS |
| 242 | ch:sectortension:L242:0.461 | calc | `0.461` | numeric: f sigma8 prediction, LCDM, z=0.7, sigma8 0.8087 (Run C) | PASS |
| 242 | ch:sectortension:L242:-0.17 | calc | `-0.17` | numeric: pull (obs-pred)/sigma, term, z=0.7 | PASS |
| 242 | ch:sectortension:L242:-0.30 | calc | `-0.30` | numeric: pull (obs-pred)/sigma, LCDM, z=0.7 | PASS |
| 242 | ch:sectortension:L242:0.043 | observed | `0.043` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs eBOSS LRG | PASS |
| 244 | ch:sectortension:L244 | observed | `0.315` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 244 | ch:sectortension:L244:0.442 | calc | `0.442` | numeric: f sigma8 prediction, term, z=0.85, sigma8 0.7998 (Run A) | PASS |
| 244 | ch:sectortension:L244:0.447 | calc | `0.447` | numeric: f sigma8 prediction, LCDM, z=0.85, sigma8 0.8087 (Run C) | PASS |
| 244 | ch:sectortension:L244:-1.34 | calc | `-1.34` | numeric: pull (obs-pred)/sigma, term, z=0.85 | PASS |
| 244 | ch:sectortension:L244:-1.38 | calc | `-1.38` | numeric: pull (obs-pred)/sigma, LCDM, z=0.85 | PASS |
| 244 | ch:sectortension:L244:0.095 | observed | `0.095` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs eBOSS ELG | PASS |
| 245 | ch:sectortension:L245 | observed | `0.462` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 245 | ch:sectortension:L245:0.374 | calc | `0.374` | numeric: f sigma8 prediction, term, z=1.48, sigma8 0.7998 (Run A) | PASS |
| 245 | ch:sectortension:L245:0.375 | calc | `0.375` | numeric: f sigma8 prediction, LCDM, z=1.48, sigma8 0.8087 (Run C) | PASS |
| 245 | ch:sectortension:L245:+1.96 | calc | `+1.96` | numeric: pull (obs-pred)/sigma, term, z=1.48 | PASS |
| 245 | ch:sectortension:L245:+1.92 | calc | `+1.92` | numeric: pull (obs-pred)/sigma, LCDM, z=1.48 | PASS |
| 245 | ch:sectortension:L245:0.045 | observed | `0.045` | file `docs/verification/scripts/verify_sector_tension.py`: sigma_obs eBOSS QSO | PASS |
| 247 | ch:sectortension:L247:0.426 | calc | `0.426` | numeric: f sigma8 prediction, term, z=0.07, sigma8 0.7998 (Run A) | PASS |
| 247 | ch:sectortension:L247:0.443 | calc | `0.443` | numeric: f sigma8 prediction, LCDM, z=0.07, sigma8 0.8087 (Run C) | PASS |
| 247 | ch:sectortension:L247:+0.44 | calc | `+0.44` | numeric: pull (obs-pred)/sigma, term, z=0.07 | PASS |
| 247 | ch:sectortension:L247:+0.12 | calc | `+0.12` | numeric: pull (obs-pred)/sigma, LCDM, z=0.07 | PASS |
| 247 |  | observed | `0.055` | not run: measured, source not named | - |
| 247 |  | observed | `0.450` | not run: measured, source not named: the DESI DR1 peculiar-velocity f sigma8 = 0.450 (Qin2026) is in no committed file; the old file_has matched a pull of -0.45/+0.45 in another row of verify_sector_tension_output.txt (see SOURCES_NEEDED.md, row of line 247) | - |
| 249 |  | observed | `0.055` | not run: measured, source not named | - |
| 249 |  | observed | `0.450` | not run: measured, source not named: running-text repeat of the DESI DR1 peculiar-velocity f sigma8 = 0.450 (Qin2026), in no committed file; the old file_has matched a pull of -0.45/+0.45 in another row (see SOURCES_NEEDED.md, row of line 249) | - |
| 250 | ch:sectortension:L250 | calc | `-0.88` | numeric: lowest pull over the six DESI bins, term | PASS |
| 250 | ch:sectortension:L250:+1.40 | calc | `+1.40` | numeric: highest pull over the six DESI bins, term | PASS |
| 250 | ch:sectortension:L250:-1.01 | calc | `-1.01` | numeric: lowest pull over the six DESI bins, LCDM | PASS |
| 250 | ch:sectortension:L250:+1.36 | calc | `+1.36` | numeric: highest pull over the six DESI bins, LCDM | PASS |
| 251 | ch:sectortension:L251 | calc | `3.84` | numeric: diagonal chi2, six DESI bins, term | PASS |
| 251 | ch:sectortension:L251:3.81 | calc | `3.81` | numeric: diagonal chi2, six DESI bins, LCDM | PASS |
| 251 | ch:sectortension:L251:6.61 | calc | `6.61` | numeric: diagonal chi2, seven legacy points, term | PASS |
| 251 | ch:sectortension:L251:6.53 | calc | `6.53` | numeric: diagonal chi2, seven legacy points, LCDM | PASS |
| 251 | ch:sectortension:L251:4.52 | calc | `4.52` | numeric: ShapeFit+BAO chi2, six DESI bins, LCDM (MGCAMB comparison) | PASS |
| 251 | ch:sectortension:L251:5.14 | calc | `5.14` | numeric: ShapeFit+BAO chi2, six DESI bins, IAM MGCAMB form | PASS |
| 251 | ch:sectortension:L251:6.20 | calc | `6.20` | numeric: chi2 on SDSS DR16, LCDM (MGCAMB comparison) | PASS |
| 251 | ch:sectortension:L251:6.96 | calc | `6.96` | numeric: chi2 on SDSS DR16, IAM MGCAMB form | PASS |
| 252 | ch:sectortension:L252 | calc | `2.5\%` | numeric: (LCDM-term)/LCDM, BGS; tol covers the 3-decimal predictions of the table | PASS |
| 252 | ch:sectortension:L252:1.7\% | calc | `1.7\%` | numeric: (LCDM-term)/LCDM, LRG1; tol covers the 3-decimal predictions of the table | PASS |
| 262 | ch:sectortension:L262 | observed | `0.759^{+0.024}_{-0.021}` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 262 | ch:sectortension:L262:0.766^{+0.020}_{-0.014} | observed | `0.766^{+0.020}_{-0.014}` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 264 | ch:sectortension:L264 | observed | `0.815^{+0.016}_{-0.021}` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 265 | ch:sectortension:L265 | observed | `0.814^{+0.011}_{-0.012}` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 265 | ch:sectortension:L265:0.802^{+0.022}_{-0.018} | observed | `0.802^{+0.022}_{-0.018}` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 266 | ch:sectortension:L266 | observed | `0.776\pm0.017` | heavy file `docs/verification/scripts/verify_sector_tension.py`: DES Y3 S8 and error in verify_sector_tension.py, the script of the chapter | PASS |
| 267 | ch:sectortension:L267 | observed | `0.776^{+0.032}_{-0.033}` | heavy file `docs/verification/scripts/verify_sector_tension.py`: HSC Y3 S8 and errors in verify_sector_tension.py, the script of the chapter | PASS |
| 268 | ch:sectortension:L268 | observed | `0.589\pm0.020` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: measured: printed value found in verify_sector_tension_output.txt, a file the chapter names | PASS |
| 273 | eq:st_sigma8 | fitted | `0.7998\pm0.0058` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 IAM Level2 posterior | PASS |
| 273 | eq:st_sigma8:0.8087\pm0.0059 | fitted | `0.8087\pm0.0059` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM Level2 posterior | PASS |
| 276 | eq:st_S8 | fitted | `0.822\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 IAM Level2 posterior | PASS |
| 276 | eq:st_S8:0.830\pm0.011 | fitted | `0.830\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 LCDM Level2 posterior | PASS |
| 278 | ch:sectortension:L278 | calc | `0.822` | numeric: S8 recomputed at chain's own Om | PASS |
| 280 | ch:sectortension:L280 | none | `+0.05\sigma` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Omega_m shift between chains, in sigma | PASS |
| 280 |  | none | `+0.09\sigma` | not run: ln(1e10 As) shift between chains; As not in committed CSV | - |
| 281 | ch:sectortension:L281 | calc | `0.600` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: term sigma8 Om^0.25 | PASS |
| 281 | ch:sectortension:L281:0.08\% | calc | `0.08\%` | numeric: CMB lensing power lowered, Limber estimate | PASS |

## Part 2 - ch:dsvalidation - `docs/book/part2/p2_10_dual_sector_validation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 17 | ch:dsvalidation:L17 | observed | `-0.035` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: best beta on SN distances, full covariance | PASS |
| 17 | ch:dsvalidation:L17:-0.068 | observed | `-0.068` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: lower 68 % end of beta on SN distances, full covariance | PASS |
| 17 | ch:dsvalidation:L17:0.000 | observed | `0.000` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: upper 68 % end of beta on SN distances, full covariance | PASS |
| 19 | ch:dsvalidation:L19 | calc | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Delta chi2 of beta_m on SN distances, full covariance | PASS |
| 27 | ch:dsvalidation:L27 | observed | `4.9` | numeric: Hubble tension significance from cited H0 values | PASS |
| 34 | ch:dsvalidation:L34 | prediction | `0.15765` | numeric: β_m = Ω_m/2 virial coupling value | PASS |
| 36 | ch:dsvalidation:L36 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Δχ² IAM vs ΛCDM Level-2 chains | PASS |
| 37 | ch:dsvalidation:L37 | measured | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: σ8 ΛCDM Level-2 chain value | PASS |
| 37 | ch:dsvalidation:L37:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: σ8 IAM suppressed value, avg IAM Level-2 runs | PASS |
| 37 | ch:dsvalidation:L37:67.16 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon-sector chain value | PASS |
| 38 | ch:dsvalidation:L38 | measured | `72.26` | numeric: H0(matter)=H0(photon)*sqrt(1+β_m) | PASS |
| 38 | ch:dsvalidation:L38:0.75 | measured | `0.75` | numeric: significance of H0(matter) vs SH0ES | PASS |
| 38 | ch:dsvalidation:L38:0.37 | measured | `0.37` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon (Level 2 Run A) from Planck 2018, in Planck sigma | PASS |
| 39 | ch:dsvalidation:L39 | calc | `0.033` | numeric: sector ratio from the committed 95 % bound | PASS |
| 39 | ch:dsvalidation:L39:30 | calc | `30` | numeric: 'at least 30x': beta_m/beta_gamma bound, whole multiples | PASS |
| 39 | ch:dsvalidation:L39:0.0052 | calc | `0.0052` | numeric: beta_gamma 95 % bound from the acoustic angle | PASS |
| 64 | eq:dsv_Hm | none |  | not run: definition of matter-sector Friedmann equation | - |
| 67 | ch:dsvalidation:L67 | derived |  | sympy: activation function E(1)=1 check | PASS |
| 67 | ch:dsvalidation:L67:3 | derived |  | sympy: inflection of E(a) at a=1/2 | PASS |
| 67 | ch:dsvalidation:L67:4 | derived |  | sympy: maximum of dE/dlna at a=1 | PASS |
| 71 | ch:dsvalidation:L71 | calc | `0.0052` | numeric: beta_gamma 95 % bound, Eq. dsv_bg | PASS |
| 72 | eq:dsv_bm | prediction | `0.15765` | numeric: β_m=Ω_m/2 definition | PASS |
| 75 | ch:dsvalidation:L75 | calc | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Ωm Level-2 MCMC posterior | PASS |
| 75 | ch:dsvalidation:L75:0.0065 | calc | `0.0065` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Ωm posterior standard deviation | PASS |
| 75 | ch:dsvalidation:L75:0.1583 | calc | `0.1583` | numeric: Ωm/2 implied value | PASS |
| 75 | ch:dsvalidation:L75:0.0032 | calc | `0.0032` | numeric: sd of Ωm/2 | PASS |
| 75 | ch:dsvalidation:L75:0.2 | calc | `0.2` | numeric: consistency of Ωm/2 with fixed β_m, sigma | PASS |
| 80 | eq:dsv_H0g | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon-sector Level-2 posterior | PASS |
| 80 | eq:dsv_H0g:0.47 | measured | `0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon-sector posterior uncertainty | PASS |
| 81 | eq:dsv_H0m | calc | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon value restated, extra digit | PASS |
| 81 | eq:dsv_H0m:1.0759 | calc | `1.0759` | numeric: sqrt(1+β_m) factor | PASS |
| 81 | eq:dsv_H0m:72.26 | calc | `72.26` | numeric: H0(matter)=H0(photon)*sqrt(1+β_m) | PASS |
| 81 | eq:dsv_H0m:0.50 | calc | `0.50` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Run A H0 sd x sqrt(1+beta_m) | PASS |
| 84 | ch:dsvalidation:L84 | derived |  | sympy: H_m^2(1) = H0^2 (1 + beta_m) in a flat universe | PASS |
| 90 | ch:dsvalidation:L90 | prediction | `72.26` | numeric: Prediction 1 H0(matter), repeat of eq:dsv_H0m | PASS |
| 110 | ch:dsvalidation:L110 | observed | `0.21` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: median diagonal m_b error of the 1588 SNe | PASS |
| 111 | ch:dsvalidation:L111 | observed | `0.212` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 115 | eq:dsv_mbcorr | none |  | not run: definition of distance modulus | - |
| 122 | eq:dsv_mu | none |  | sympy: magnitude formula, pc vs Mpc identity | PASS |
| 126 | eq:dsv_dL | none |  | not run: definition of luminosity distance integral | - |
| 130 | eq:dsv_Hz | none |  | not run: definition of sector-dependent Hubble rate | - |
| 240 | ch:dsvalidation:L240 | calc | `721.12` | numeric: Test C: chi2_min at H0 = 64.00 | PASS |
| 240 | ch:dsvalidation:L240:-28.935 | calc | `-28.935` | numeric: Test C: M - 5 log10 H0 at H0 = 64.00 | PASS |
| 241 | ch:dsvalidation:L241 | calc | `721.12` | numeric: Test C: chi2_min at H0 = 67.40 | PASS |
| 241 | ch:dsvalidation:L241:-28.935 | calc | `-28.935` | numeric: Test C: M - 5 log10 H0 at H0 = 67.40 | PASS |
| 242 | ch:dsvalidation:L242 | calc | `721.12` | numeric: Test C: chi2_min at H0 = 70.00 | PASS |
| 242 | ch:dsvalidation:L242:-28.935 | calc | `-28.935` | numeric: Test C: M - 5 log10 H0 at H0 = 70.00 | PASS |
| 243 | ch:dsvalidation:L243 | calc | `721.12` | numeric: Test C: chi2_min at H0 = 73.04 | PASS |
| 243 | ch:dsvalidation:L243:-28.935 | calc | `-28.935` | numeric: Test C: M - 5 log10 H0 at H0 = 73.04 | PASS |
| 246 | ch:dsvalidation:L246 | calc | `60.0` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Test C simplex ends at the H0 boundary | PASS |
| 247 |  | calc | `10^{-4}` | not run: precision statement: 'flat to 10^-4' bounds the numerical spread of the H0 profile, nothing to recompute; the flatness itself is checked at ch:dsvalidation:L240-L243 (same chi2_min and M - 5 log10 H0 at all four H0) | - |
| 259 | ch:dsvalidation:L259:-0.30 | calc | `-0.30` | numeric: Test A best-fit beta (boundary) | PASS |
| 259 | ch:dsvalidation:L259:67.40 | calc | `67.40` | numeric: Test A best-fit H0 (prior met) | PASS |
| 259 | ch:dsvalidation:L259:721.12 | calc | `721.12` | numeric: Test A chi2 | PASS |
| 260 | ch:dsvalidation:L260 | calc | `0.1745` | numeric: M offset shift between Test A/B | PASS |
| 260 | ch:dsvalidation:L260:-0.30 | calc | `-0.30` | numeric: Test B best-fit beta (boundary) | PASS |
| 260 | ch:dsvalidation:L260:73.04 | calc | `73.04` | numeric: Test B best-fit H0 (prior met) | PASS |
| 260 | ch:dsvalidation:L260:721.12 | calc | `721.12` | numeric: Test B chi2 | PASS |
| 261 | ch:dsvalidation:L261 | calc | `-0.30` | numeric: Test C best-fit beta (boundary) | PASS |
| 261 | ch:dsvalidation:L261:721.12 | calc | `721.12` | numeric: Test C chi2 | PASS |
| 266 | ch:dsvalidation:L266 | observed | `73.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 266 | ch:dsvalidation:L266:-28.935 | derived | `-28.935` | numeric: shared minimum M - 5 log10 H0 | PASS |
| 271 | ch:dsvalidation:L271 | calc | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: figure caption: Delta chi2 of beta_m on SN distances | PASS |
| 271 | ch:dsvalidation:L271:-0.035 | fitted | `-0.035` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: figure caption: best beta, full covariance | PASS |
| 271 | ch:dsvalidation:L271:721.12 | calc | `721.12` | numeric: figure caption: flat chi2_min | PASS |
| 271 | ch:dsvalidation:L271:-28.935 | calc | `-28.935` | numeric: figure caption: M - 5 log10 H0 at every H0 | PASS |
| 289 | eq:dsv_H0local | prediction | `72.26` | numeric: matter-sector local H0 prediction | PASS |
| 295 | eq:dsv_shape | none |  | not run: definition of luminosity distance integral | - |
| 298 | ch:dsvalidation:L298 | calc | `-6.5` | numeric: d_L change at z=0.1, fixed H0, beta_m=0.15765 (text says 0.157) | PASS |
| 298 | ch:dsvalidation:L298:-2.3 | calc | `-2.3` | numeric: dL shift at z=2, fixed H0 | PASS |
| 432 | ch:dsvalidation:L432 | measured | `0.822\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 from IAM Level2 chain | PASS |
| 433 | ch:dsvalidation:L433 | measured | `0.830\pm0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 from LCDM Level2 baseline chain | PASS |
| 433 | ch:dsvalidation:L433:0.8\sigma | calc | `0.8\sigma` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 shift significance between chains | PASS |
| 440 | ch:dsvalidation:L440 | calc | `+1.02\%` | numeric: theta_s shift, beta=0.18 on photon paths, from LCDM | PASS |
| 440 | ch:dsvalidation:L440:34\sigma | calc | `34\sigma` | numeric: theta_s shift / Planck error | PASS |
| 441 | ch:dsvalidation:L441 | calc | `+1.06\%` | numeric: theta_s shift, beta=0.18, from the observed value | PASS |
| 441 | ch:dsvalidation:L441:36\sigma | calc | `36\sigma` | numeric: from the observed value, in Planck errors | PASS |
| 441 | ch:dsvalidation:L441:+0.90\% | calc | `+0.90\%` | numeric: theta_s shift with beta_m | PASS |
| 441 | ch:dsvalidation:L441:30\sigma | calc | `30\sigma` | numeric: shift with beta_m in Planck errors | PASS |
| 442 | ch:dsvalidation:L442 | measured | `\approx61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from Level2b exploratory chain | PASS |
| 443 | ch:dsvalidation:L443 | calc | `<0.033` | numeric: sector ratio from the committed 95 % bound | PASS |
| 456 | ch:dsvalidation:L456 | calc | `0.0052` | numeric: Table dsv_observables: beta_gamma bound | PASS |
| 457 | ch:dsvalidation:L457 | measured | `67.16\pm0.47` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon sector from Level2 chain | PASS |
| 457 | ch:dsvalidation:L457:67.36\pm0.54 | measured | `67.36\pm0.54` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 458 | ch:dsvalidation:L458 | measured | `0.809` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 458 | ch:dsvalidation:L458:0.800 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 prediction from Level2 chain | PASS |
| 460 | ch:dsvalidation:L460 | observed | `73.04\pm1.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 460 |  | prediction | `72.26` | not run: locked IAM matter-sector H0 prediction | - |
| 462 | ch:dsvalidation:L462 | observed | `-0.035^{+0.035}_{-0.033}` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: Table dsv_observables: beta_distance | PASS |
| 463 | ch:dsvalidation:L463 | calc | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Table dsv_observables: beta_m on SN distances excluded | PASS |
| 466 |  | interp | `67.16` | not run: H0 photon sector restated, input | - |
| 467 |  | interp | `0.157` | not run: beta_m growth value restated, input | - |
| 467 |  | interp | `72.26` | not run: H0 matter sector restated, input | - |
| 470 |  | none | `\beta_\gamma<0.0039` | not run: text changed at HEAD; bound restated in figure caption | - |
| 471 |  | none | `67.16` | not run: H0 photon restated in figure caption | - |
| 471 |  | none | `0.15765` | not run: beta_m restated in figure caption | - |
| 471 |  | none | `72.26` | not run: H0 matter restated in figure caption | - |
| 479 |  | interp | `67.16` | not run: H0 photon restated, input | - |
| 480 |  | interp | `73.04` | not run: SH0ES H0 restated, input | - |
| 480 |  | interp | `72.26` | not run: predicted H0 matter restated, input | - |
| 489 |  | prediction | `\beta_\gamma<0.0039` | not run: text changed at HEAD; current bound restated as forecast baseline | - |
| 490 | ch:dsvalidation:L490 | prediction | `0.822` | numeric: S8 forecast from sigma8 and Omega_m | PASS |
| 490 |  | none | `0.7998` | not run: sigma8 input value stated in text | - |
| 490 |  | none | `0.3166` | not run: Omega_m input value stated in text | - |
| 490 |  | prediction | `0.800` | not run: predicted sigma8 restated, input | - |
| 492 |  | prediction | `0.74\%` | not run: published DESI forecast precision, input | - |
| 492 |  | prediction | `0.38\%` | not run: published DESI forecast precision, input | - |
| 493 |  | prediction | `1.35\%` | not run: fsigma8 deficit forecast, no data here | - |
| 493 |  | prediction | `2.17\%` | not run: fsigma8 deficit forecast, no data here | - |
| 494 | ch:dsvalidation:L494 | prediction | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chain posterior sigma8 restated | PASS |
| 496 |  | prediction | `72.26` | not run: predicted matter-sector H0 restated | - |
| 497 | ch:dsvalidation:L497 | observed | `70.0^{+12.0}_{-8.0}` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 497 | ch:dsvalidation:L497:75.5^{+5.3}_{-5.4} | observed | `75.5^{+5.3}_{-5.4}` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 498 |  | prediction | `-0.136` | not run: locked IAM mu0 prediction, input | - |
| 507 | eq:dsv_poisson | none |  | not run: definition of standard mu-Sigma parametrization | - |
| 511 | eq:dsv_mu_sigma | derived |  | sympy: mu(a) reduces correctly at a=1 | PASS |
| 513 | ch:dsvalidation:L513 | calc | `0.864` | numeric: mu(a=1) from mu(a) formula | PASS |
| 514 | ch:dsvalidation:L514 | calc | `13.6\%` | numeric: fractional Newton-constant suppression | PASS |
| 515 | ch:dsvalidation:L515 | calc | `0.982` | numeric: mu(a) at z=1 from formula | PASS |
| 515 | ch:dsvalidation:L515:1-\mu<4\times10^{-4} | calc | `<4\times10^{-4}` | numeric: mu(a) deviation at z=3 below bound | PASS |
| 515 | ch:dsvalidation:L515:4.25\% | calc | `4.25\%` | numeric: f sigma8 deficit at z = 0 | PASS |
| 517 |  | none | `\beta_\gamma<0.0039` | not run: text changed at HEAD; photon-sector bound restated | - |
| 520 | ch:dsvalidation:L520 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2 difference IAM vs LCDM Level2 | PASS |
| 525 | ch:dsvalidation:L525 | derived | `\Omega_m/2` | sympy: beta_m amplitude equals Omega_m/2 | PASS |
| 531 | ch:dsvalidation:L531 | observed | `-0.035` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: Conclusions: beta_distance | PASS |
| 533 | ch:dsvalidation:L533 | calc | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Conclusions: Delta chi2 of beta_m on SN distances | PASS |
| 533 | ch:dsvalidation:L533:0.41 | calc | `0.41` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Omega_m that offsets beta_m on SN distances | PASS |
| 536 | ch:dsvalidation:L536 | observed | `73.04` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 536 | ch:dsvalidation:L536:72.26 | observed | `72.26` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 536 | ch:dsvalidation:L536:721.12 | derived | `721.12` | numeric: Conclusions: shared chi2 minimum | PASS |
| 537 | ch:dsvalidation:L537 | observed | `-0.75\sigma` | numeric: sigma deviation of prediction from SH0ES | PASS |
| 543 |  | interp | `67.16` | not run: H0 photon restated, input | - |
| 543 |  | interp | `72.26` | not run: H0 matter predicted restated, input | - |
| 543 |  | interp | `73.04` | not run: H0 matter measured restated, input | - |
| 549 | ch:dsvalidation:L549 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: chi2 difference IAM vs LCDM, restated | PASS |
| 550 | ch:dsvalidation:L550 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 posterior restated from chain | PASS |
| 550 | ch:dsvalidation:L550:72.26 | measured | `72.26` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: measured: printed value found in verify_dual_sector_chapters_output.txt, a file the chapter names | PASS |
| 550 | ch:dsvalidation:L550:\approx61.5 | measured | `\approx61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 from second Level2b background run | PASS |

## Part 2 - ch:darkenergy - `docs/book/part2/p2_11_dark_energy.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 26 | ch:darkenergy:L26 | observed | `2.8` | numeric: DESI DR2 preference, lowest over the supernova sets | PASS |
| 26 | ch:darkenergy:L26:4.2 | observed | `4.2` | numeric: DESI DR2 preference, highest over the supernova sets | PASS |
| 50 | eq:de_sinfo | none |  | not run: definition of informational entropy S_info(a) | - |
| 54 | ch:darkenergy:L54 | prediction | `0.15765` | numeric: beta_m = Omega_m/2 virial coupling | PASS |
| 58 | part2:eq:Hm | none |  | not run: definition of matter-sector expansion rate | - |
| 60 | eq:de_rhoinfo | none |  | not run: definition of informational energy density | - |
| 67 | eq:de_cont | none |  | sympy: continuity eq gives w_info formula, step | PASS |
| 69 | eq:de_dlnE | none |  | sympy: d ln E/d ln a derivative step | PASS |
| 71 | eq:de_winfo | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 82 | ch:darkenergy:L82 | derived |  | sympy: w_info(a) < -1, always phantom | PASS |
| 85 | eq:de_wlim | derived |  | sympy: limit of w_info as a->infinity | PASS |
| 90 | eq:de_w1 | derived |  | sympy: w_info at a=1 today | PASS |
| 94 | ch:darkenergy:L94 | derived |  | sympy: derivative of w_info positive, no Big Rip | PASS |
| 99 | ch:darkenergy:L99 | calc | `-1.3333` | numeric: w_info decimal value at a=1 | PASS |
| 99 | ch:darkenergy:L99:-1.1667 | calc | `-1.1667` | numeric: w_info decimal value at a=2 | PASS |
| 99 | ch:darkenergy:L99:-1.0667 | calc | `-1.0667` | numeric: w_info decimal value at a=5 | PASS |
| 99 | ch:darkenergy:L99:-1.0033 | calc | `-1.0033` | numeric: w_info decimal value at a=100 | PASS |
| 99 | ch:darkenergy:L99:1 | calc | `1` | numeric: E(a) activation function at a=1 | PASS |
| 99 | ch:darkenergy:L99:1.6487 | calc | `1.6487` | numeric: E(a) activation function at a=2 | PASS |
| 99 | ch:darkenergy:L99:2.2255 | calc | `2.2255` | numeric: E(a) activation function at a=5 | PASS |
| 100 | ch:darkenergy:L100 | calc | `2.6912` | numeric: E(a) activation function at a=100 | PASS |
| 100 | ch:darkenergy:L100:2.71828 | calc | `2.71828` | numeric: Euler's number, saturation limit of E(a) | PASS |
| 104 | ch:darkenergy:L104 | calc | `1.15` | numeric: rho_info/rho_Lambda percentage at z=3 | PASS |
| 112 | ch:darkenergy:L112 | derived | `-4/3` | sympy: repeat of w_info(1) in figure caption | PASS |
| 113 | ch:darkenergy:L113 | derived | `36.8` | numeric: maturity fraction E(1)/e today, percent | PASS |
| 115 | ch:darkenergy:L115 | derived | `0.541` | numeric: peak maturation rate d(E/e)/da value | PASS |
| 115 | ch:darkenergy:L115:1/2 | derived | `1/2` | sympy: location of peak maturation rate | PASS |
| 116 | ch:darkenergy:L116 | derived | `1/e` | numeric: peak maturation rate per e-fold at a=1 | PASS |
| 123 | eq:wz_cpl | derived |  | sympy: CPL mapping w0,wa from w_info at a=1 | PASS |
| 123 | ch:darkenergy:L123 | derived | `-1.333` | numeric: CPL w0 decimal approximation | PASS |
| 123 | ch:darkenergy:L123:-0.333 | derived | `-0.333` | numeric: CPL wa decimal approximation | PASS |
| 124 | ch:darkenergy:L124 | derived | `-4/3` | sympy: repeat of CPL w0 point | PASS |
| 124 | ch:darkenergy:L124:-1/3 | derived | `-1/3` | sympy: repeat of CPL wa point | PASS |
| 132 | eq:de_Elimits | derived |  | sympy: E(a) limits: 0, 1, and e | PASS |
| 135 | ch:darkenergy:L135 | derived | `1/2` | sympy: inflection point of E(a) in a | PASS |
| 135 | ch:darkenergy:L135:1 | derived | `1` | sympy: inflection point of E in ln a | PASS |
| 142 | eq:de_friedmann | none |  | not run: definition, standard LCDM Friedmann equation | - |
| 145 | eq:de_Hinf | calc | `55.6` | numeric: asymptotic Hubble parameter H_infinity | PASS |
| 146 | ch:darkenergy:L146 | calc | `55.57` | numeric: precise H_infinity, Level 2 chains | PASS |
| 147 | ch:darkenergy:L147 | calc | `55.78` | numeric: H_infinity, Planck 2018 base values | PASS |
| 162 | eq:de_weff | derived |  | sympy: final effective equation of state limit | PASS |
| 172 | ch:darkenergy:L172 | calc | `72.26` | numeric: H_m(1) matter-sector rate today | PASS |
| 172 | ch:darkenergy:L172:55.57 | calc | `55.57` | numeric: repeat of H_infinity in caption | PASS |
| 172 |  | none | `67.16` | not run: H0_photon restated, trivial H(a=1)=H0 | - |
| 173 | ch:darkenergy:L173 | calc | `70.86` | numeric: H_m asymptote, matter-sector | PASS |
| 173 | ch:darkenergy:L173:1.076 | calc | `1.076` | numeric: H_m/H ratio at a=1 | PASS |
| 173 | ch:darkenergy:L173:1.166 | calc | `1.166` | numeric: H_m/H ratio at a=2 | PASS |
| 173 | ch:darkenergy:L173:1.251 | calc | `1.251` | numeric: H_m/H ratio at a=10 | PASS |
| 174 | ch:darkenergy:L174 | calc | `1.275` | numeric: H_m/H ratio limit a->infinity | PASS |
| 181 | ch:darkenergy:L181 | calc | `72.26` | numeric: table: H_m at a=1 | PASS |
| 181 | ch:darkenergy:L181:1.076 | calc | `1.076` | numeric: table: ratio at a=1 | PASS |
| 181 | ch:darkenergy:L181:57.15 | calc | `57.15` | numeric: table: photon rate H at a=2 | PASS |
| 181 | ch:darkenergy:L181:66.62 | calc | `66.62` | numeric: table: matter rate H_m at a=2 | PASS |
| 181 | ch:darkenergy:L181:1.166 | calc | `1.166` | numeric: table: ratio at a=2 | PASS |
| 181 | ch:darkenergy:L181:55.59 | calc | `55.59` | numeric: table: photon rate H at a=10 | PASS |
| 181 |  | none | `67.16` | not run: table: H(a=1)=H0, trivial restatement | - |
| 202 | eq:de_maturity_today | derived |  | sympy: E(1)/e equals 1/e identity | PASS |
| 202 | eq:de_maturity_today:0.36788 | derived | `0.36788` | numeric: maturity fraction today, 1/e | PASS |
| 202 | eq:de_maturity_today:36.8\% | derived | `36.8\%` | numeric: maturity fraction today as percent | PASS |
| 205 | eq:de_rate_a | derived |  | sympy: derivative of maturity fraction wrt a | PASS |
| 206 | ch:darkenergy:L206 | derived | `0.368` | numeric: rate da at a=1 equals 1/e | PASS |
| 206 | ch:darkenergy:L206:0.541 | derived | `0.541` | numeric: max of rate da value | PASS |
| 206 | ch:darkenergy:L206:4 | derived |  | sympy: a=1/2 maximizes rate da | PASS |
| 208 | eq:de_rate_lna | derived |  | sympy: chain rule to per e-fold rate | PASS |
| 209 | ch:darkenergy:L209 | derived |  | sympy: max of per e-fold rate = 1/e at a=1 | PASS |
| 214 | eq:de_af | derived |  | sympy: invert E(a)/e=f for a | PASS |
| 223 | ch:darkenergy:L223 | calc | `0.217` | numeric: scale factor at 1% maturity | PASS |
| 223 | ch:darkenergy:L223:3.61 | calc | `3.61` | numeric: redshift at 1% maturity | PASS |
| 223 | ch:darkenergy:L223:1.7 | calc | `1.7` | numeric: age of universe at 1% maturity | PASS |
| 223 | ch:darkenergy:L223:-12.1 | calc | `-12.1` | numeric: time from now at 1% maturity | PASS |
| 223 | ch:darkenergy:L223:0.334 | calc | `0.334` | numeric: scale factor at 5% maturity | PASS |
| 223 | ch:darkenergy:L223:2.00 | calc | `2.00` | numeric: redshift at 5% maturity | PASS |
| 223 | ch:darkenergy:L223:3.3 | calc | `3.3` | numeric: age at 5% maturity | PASS |
| 223 | ch:darkenergy:L223:-10.5 | calc | `-10.5` | numeric: time from now at 5% maturity | PASS |
| 223 | ch:darkenergy:L223:0.434 | calc | `0.434` | numeric: scale factor at 10% maturity | PASS |
| 223 | ch:darkenergy:L223:1.30 | calc | `1.30` | numeric: redshift at 10% maturity | PASS |
| 223 | ch:darkenergy:L223:4.8 | calc | `4.8` | numeric: age at 10% maturity | PASS |
| 223 | ch:darkenergy:L223:-9.0 | calc | `-9.0` | numeric: time from now at 10% maturity | PASS |
| 224 | ch:darkenergy:L224 | calc | `0.621` | numeric: scale factor at 20% maturity | PASS |
| 224 | ch:darkenergy:L224:0.61 | calc | `0.61` | numeric: redshift at 20% maturity | PASS |
| 224 | ch:darkenergy:L224:7.8 | calc | `7.8` | numeric: age at 20% maturity | PASS |
| 224 | ch:darkenergy:L224:-6.0 | calc | `-6.0` | numeric: time from now at 20% maturity | PASS |
| 224 | ch:darkenergy:L224:13.8 | calc | `13.8` | numeric: age of universe today | PASS |
| 224 | ch:darkenergy:L224:36.8\% | calc | `36.8\%` | numeric: repeat: maturity today as percent | PASS |
| 224 | ch:darkenergy:L224:1.000 | calc | `1.000` | numeric: scale factor at 36.8 % maturity (today) | PASS |
| 224 | ch:darkenergy:L224:0.00 | calc | `0.00` | numeric: redshift at 36.8 % maturity (today) | PASS |
| 224 |  | calc | `0` | not run: definition: the row is labelled 36.8 % (today), so its time from now is zero by definition; the check computed t(a(1/e)) - t(1) with a(1/e) = 1 exactly, i.e. t(1) - t(1), identically zero whatever the inputs. The one non-trivial step, a(1/e) = 1 from Eq. de_af, is checked at ch:darkenergy:L224:1.000 | - |
| 225 | ch:darkenergy:L225 | calc | `1.443` | numeric: scale factor at 50% maturity | PASS |
| 225 | ch:darkenergy:L225:-0.31 | calc | `-0.31` | numeric: redshift at 50% maturity | PASS |
| 225 | ch:darkenergy:L225:19.5 | calc | `19.5` | numeric: age at 50% maturity | PASS |
| 225 | ch:darkenergy:L225:5.7 | calc | `5.7` | numeric: time from now at 50% maturity | PASS |
| 225 | ch:darkenergy:L225:3.476 | calc | `3.476` | numeric: scale factor at 75% maturity | PASS |
| 225 | ch:darkenergy:L225:-0.71 | calc | `-0.71` | numeric: redshift at 75% maturity | PASS |
| 225 | ch:darkenergy:L225:34.5 | calc | `34.5` | numeric: age at 75% maturity | PASS |
| 225 | ch:darkenergy:L225:20.7 | calc | `20.7` | numeric: time from now at 75% maturity | PASS |
| 225 | ch:darkenergy:L225:9.491 | calc | `9.491` | numeric: scale factor at 90% maturity | PASS |
| 225 | ch:darkenergy:L225:-0.89 | calc | `-0.89` | numeric: redshift at 90% maturity | PASS |
| 225 | ch:darkenergy:L225:52.1 | calc | `52.1` | numeric: age at 90% maturity | PASS |
| 225 | ch:darkenergy:L225:38.3 | calc | `38.3` | numeric: time from now at 90% maturity | PASS |
| 226 | ch:darkenergy:L226 | calc | `19.50` | numeric: scale factor at 95% maturity | PASS |
| 226 | ch:darkenergy:L226:-0.95 | calc | `-0.95` | numeric: redshift at 95% maturity | PASS |
| 226 | ch:darkenergy:L226:64.7 | calc | `64.7` | numeric: age at 95% maturity | PASS |
| 226 | ch:darkenergy:L226:50.9 | calc | `50.9` | numeric: time from now at 95% maturity | PASS |
| 226 | ch:darkenergy:L226:99.50 | calc | `99.50` | numeric: scale factor at 99% maturity | PASS |
| 226 | ch:darkenergy:L226:-0.99 | calc | `-0.99` | numeric: redshift at 99% maturity | PASS |
| 226 | ch:darkenergy:L226:93.3 | calc | `93.3` | numeric: age at 99% maturity | PASS |
| 226 | ch:darkenergy:L226:79.5 | calc | `79.5` | numeric: time from now at 99% maturity | PASS |
| 228 | ch:darkenergy:L228 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 background H0 | PASS |
| 228 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153 (Planck 2018 VI Table 2, TT,TE,EE+lowE+lensing) restated as the Level 2 background | - |
| 229 | ch:darkenergy:L229 | calc | `0.3` | numeric: max age increase under Level2 params | PASS |

## Part 2 - ch:wzfuture - `docs/book/part2/p2_20_wz_far_future.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 23 | ch:wzfuture:L23 | calc | `-1.333` | numeric: w_info at z=0 | PASS |
| 23 | ch:wzfuture:L23:0.230 | calc | `0.230` | numeric: rho_info/rho_Lambda at z=0 | PASS |
| 23 | ch:wzfuture:L23:0.187 | calc | `0.187` | numeric: info share of vacuum total z=0 | PASS |
| 23 | ch:wzfuture:L23:-1.500 | calc | `-1.500` | numeric: w_info at z=0.5 | PASS |
| 23 | ch:wzfuture:L23:0.140 | calc | `0.140` | numeric: rho_info/rho_Lambda at z=0.5 | PASS |
| 23 | ch:wzfuture:L23:0.123 | calc | `0.123` | numeric: rho_info/(rho_L+rho_info) at z=0.5 | PASS |
| 23 | ch:wzfuture:L23:-1.667 | calc | `-1.667` | numeric: w_info at z=1 | PASS |
| 23 | ch:wzfuture:L23:0.085 | calc | `0.085` | numeric: rho_info/rho_Lambda at z=1 | PASS |
| 23 | ch:wzfuture:L23:0.078 | calc | `0.078` | numeric: info share of vacuum total z=1 | PASS |
| 23 | ch:wzfuture:L23:-2.000 | calc | `-2.000` | numeric: w_info at z=2 | PASS |
| 23 | ch:wzfuture:L23:0.031 | calc | `0.031` | numeric: rho_info/rho_Lambda at z=2 | PASS |
| 23 | ch:wzfuture:L23:0.030 | calc | `0.030` | numeric: info share of vacuum total z=2 | PASS |
| 24 | ch:wzfuture:L24 | calc | `-2.333` | numeric: w_info at z=3 | PASS |
| 24 | ch:wzfuture:L24:-1 | calc | `-1` | numeric: w_info limit as a to infinity | PASS |
| 24 | ch:wzfuture:L24:0.626 | calc | `0.626` | numeric: rho_info/rho_Lambda at saturation | PASS |
| 24 | ch:wzfuture:L24:0.385 | calc | `0.385` | numeric: info share of vacuum total saturation | PASS |
| 24 |  | calc | `0.011` | not run: printed value 0.011 has two significant digits: a 5 % change (0.00055) is about the rounding half-unit (0.0005), so it cannot be told from rounding (recomputed rho_info/rho_Lambda = 0.01146 at z = 3) | - |
| 24 |  | calc | `0.011` | not run: printed value 0.011 has two significant digits: a 5 % change (0.00055) is about the rounding half-unit (0.0005), so it cannot be told from rounding (recomputed share of the vacuum-like total 0.01133 at z = 3) | - |
| 26 | ch:wzfuture:L26 | calc | `1.15` | numeric: info density pct of rho_Lambda at z=3 | PASS |
| 27 | ch:wzfuture:L27 | calc | `18.7` | numeric: info term pct of vacuum total today | PASS |
| 27 | ch:wzfuture:L27:23 | calc | `23` | numeric: info term pct of rho_Lambda today | PASS |
| 28 | ch:wzfuture:L28 | calc | `0.626` | numeric: info/Lambda ratio at saturation (repeat) | PASS |
| 31 | ch:wzfuture:L31:0.230 | calc | `0.230` | numeric: caption repeat of today ratio | PASS |
| 31 | ch:wzfuture:L31:0.626 | calc | `0.626` | numeric: caption repeat of saturation ratio | PASS |
| 31 |  | calc | `0.011` | not run: caption restatement of ch:wzfuture:L24:0.011; printed value 0.011 has two significant digits: a 5 % change cannot be told from rounding (recomputed 0.01146) | - |
| 74 | eq:wz_norip | derived |  | sympy: monotonic w, bounded density statement | PASS |
| 79 | eq:wz_rho_from_w | derived |  | sympy: density recovered by integrating continuity eq | PASS |
| 83 | ch:wzfuture:L83 | derived | `1` | numeric: convergent integral bounding density | PASS |
| 88 | ch:wzfuture:L88 | derived | `1.195` | numeric: bound on matter-sector rate ratio | PASS |
| 96 | ch:wzfuture:L96 | observed | `-0.838` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + Pantheon+ w0 | PASS |
| 96 | ch:wzfuture:L96:-0.62 | observed | `-0.62` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + Pantheon+ wa | PASS |
| 97 | ch:wzfuture:L97 | observed | `-0.667` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + Union3 w0 | PASS |
| 97 | ch:wzfuture:L97:-1.09 | observed | `-1.09` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + Union3 wa | PASS |
| 98 | ch:wzfuture:L98 | observed | `-0.752` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + DES Y5 w0 | PASS |
| 98 | ch:wzfuture:L98:-0.86 | observed | `-0.86` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI DR2 + CMB + DES Y5 wa | PASS |
| 100 | ch:wzfuture:L100 | observed | `2.8` | numeric: DESI DR2 preference, lowest of the three fits | PASS |
| 100 | ch:wzfuture:L100:4.2 | observed | `4.2` | numeric: DESI DR2 preference, highest of the three fits | PASS |
| 101 | ch:wzfuture:L101 | calc | `0.739` | numeric: a where Pantheon+ fit crosses w=-1 | PASS |
| 101 | ch:wzfuture:L101:0.694 | calc | `0.694` | numeric: a where Union3 fit crosses w=-1 | PASS |
| 101 | ch:wzfuture:L101:0.712 | calc | `0.712` | numeric: a where DES Y5 fit crosses w=-1 | PASS |
| 101 | ch:wzfuture:L101:0.35 | calc | `0.35` | numeric: redshift of Pantheon+ crossing | PASS |
| 101 | ch:wzfuture:L101:0.44 | calc | `0.44` | numeric: redshift of Union3 crossing | PASS |
| 101 | ch:wzfuture:L101:0.41 | calc | `0.41` | numeric: redshift of DES Y5 crossing | PASS |
| 105 | eq:wz_cplpoint | calc | `(-4/3,-1/3)≈(-1.333,-0.333)` | sympy: IAM CPL point from mapping | PASS |
| 107 | ch:wzfuture:L107 | calc | `9.0` | numeric: w0 sigma distance, Pantheon+ | PASS |
| 107 | ch:wzfuture:L107:7.6 | calc | `7.6` | numeric: w0 sigma distance, Union3 | PASS |
| 107 | ch:wzfuture:L107:10.2 | calc | `10.2` | numeric: w0 sigma distance, DES Y5 | PASS |
| 107 | ch:wzfuture:L107:1.3 | calc | `1.3` | numeric: wa sigma distance, Pantheon+ | PASS |
| 107 | ch:wzfuture:L107:2.4 | calc | `2.4` | numeric: wa sigma distance, Union3 | PASS |
| 107 | ch:wzfuture:L107:2.3 | calc | `2.3` | numeric: wa sigma distance, DES Y5 | PASS |
| 115 | ch:wzfuture:L115 | calc | `0.5` | sympy: caption repeat: peak of scale-factor clock | PASS |
| 115 | ch:wzfuture:L115:1.26 | calc | `1.26` | numeric: caption repeat: z of cosmic-time clock peak | PASS |
| 124 | ch:wzfuture:L124 | calc | `-2.333` | numeric: w_info at a=0.25 | PASS |
| 124 | ch:wzfuture:L124:-1.667 | calc | `-1.667` | numeric: w_info at a=0.5 | PASS |
| 124 | ch:wzfuture:L124:-1.333 | calc | `-1.333` | numeric: w_info at a=1 | PASS |
| 124 | ch:wzfuture:L124:-1.167 | calc | `-1.167` | numeric: w_info at a=2 | PASS |
| 124 | ch:wzfuture:L124:-1.067 | calc | `-1.067` | numeric: w_info at a=5 | PASS |
| 125 | ch:wzfuture:L125 | calc | `-1.583` | numeric: CPL image at a=0.25 | PASS |
| 125 | ch:wzfuture:L125:-1.500 | calc | `-1.500` | numeric: CPL image at a=0.5 | PASS |
| 125 | ch:wzfuture:L125:-1.333 | calc | `-1.333` | numeric: CPL image at a=1 | PASS |
| 125 | ch:wzfuture:L125:-1.000 | calc | `-1.000` | numeric: CPL image at a=2 | PASS |
| 125 | ch:wzfuture:L125:0.000 | calc | `0.000` | numeric: CPL image at a=5 | PASS |
| 126 | ch:wzfuture:L126 | calc | `-0.750` | numeric: w_info minus CPL image at a=0.25 | PASS |
| 126 | ch:wzfuture:L126:-0.167 | calc | `-0.167` | numeric: difference at a=0.5 | PASS |
| 126 | ch:wzfuture:L126:0 | calc | `0` | numeric: difference at a=1 | PASS |
| 126 | ch:wzfuture:L126:-0.167' | calc | `-0.167` | numeric: difference at a=2 | PASS |
| 126 | ch:wzfuture:L126:-1.067 | calc | `-1.067` | numeric: difference at a=5 | PASS |
| 128 | ch:wzfuture:L128 | calc | `0.17` | numeric: max |CPL deviation| between a=0.5 and 2 | PASS |
| 142 | ch:wzfuture:L142 | derived | `1.076` | numeric: H_m/H today | PASS |
| 142 | ch:wzfuture:L142:1.275 | derived | `1.275` | numeric: H_m/H as a to infinity | PASS |
| 148 | ch:wzfuture:L148 | calc | `3.1` | numeric: info density pct of rho_Lambda at z=2 | PASS |
| 148 | ch:wzfuture:L148:1.15 | calc | `1.15` | numeric: repeat: info density pct at z=3 | PASS |
| 152 | ch:wzfuture:L152 | measured | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 from Level-2 chain | PASS |
| 153 | ch:wzfuture:L153 | derived | `4.25` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 162 | ch:wzfuture:L162 | calc | `1` | sympy: peak of e-fold clock dE/dlna | PASS |
| 162 | ch:wzfuture:L162:1.26 | calc | `1.26` | numeric: z of cosmic-time clock peak | PASS |
| 162 | ch:wzfuture:L162:8.9 | calc | `8.9` | numeric: lookback time to cosmic-time clock peak | PASS |
| 163 | ch:wzfuture:L163 | calc | `3.37` | numeric: peak dE/dt = H E/a over e, per Gyr, H0 = 67.16 | PASS |
| 163 | ch:wzfuture:L163:2.53 | calc | `2.53` | numeric: dE/dt today over e, per Gyr, H0 = 67.16 | PASS |
| 164 | ch:wzfuture:L164 | calc | `0.5` | sympy: peak of scale-factor clock d(E/e)/da | PASS |
| 164 | ch:wzfuture:L164:1 | calc | `1` | numeric: z equivalent of a=0.5 peak | PASS |
| 179 | ch:wzfuture:L179 | derived |  | sympy: E(1)/e equals 1/e identity | PASS |
| 186 | ch:wzfuture:L186 | calc | `1` | numeric: repeat: z of scale-factor clock peak | PASS |
| 186 | ch:wzfuture:L186:1.26 | calc | `1.26` | numeric: repeat: z of cosmic-time clock peak | PASS |
| 191 | ch:wzfuture:L191 | calc | `55.57` | numeric: H_infty, photon sector, H0=67.16 | PASS |
| 191 | ch:wzfuture:L191:55.78 | calc | `55.78` | numeric: H_inf = H0 sqrt(Omega_L) for H0=67.4 (Omega_m 0.315) | PASS |
| 192 | ch:wzfuture:L192 | calc | `70.86` | numeric: matter-sector rate settle, H0=67.16 | PASS |
| 192 | ch:wzfuture:L192:71.12 | calc | `71.12` | numeric: matter-sector late rate for H0=67.4 | PASS |
| 203 | ch:wzfuture:L203 | calc | `36.8` | numeric: 1/e as percent, info fraction written by today | PASS |
| 207 | ch:wzfuture:L207 | calc | `-4/3` | sympy: CPL w0 from w_info(a) at a=1 | PASS |
| 207 | ch:wzfuture:L207:-1/3 | calc | `-1/3` | sympy: CPL wa from slope of w_info at a=1 | PASS |
| 221 | ch:wzfuture:L221 | calc | `-1-(1+z)/3` | sympy: w_info(z) derived from E(a)=exp(1-1/a) | PASS |
| 222 | ch:wzfuture:L222 | derived | `E(a)` | sympy: integrating w_info reproduces density ratio E(a) | PASS |
| 222 | ch:wzfuture:L222:e | derived | `e` | sympy: sup of E(a) as a->inf equals e, bounds rho_info | PASS |
| 224 | ch:wzfuture:L224 | calc | `-4/3` | sympy: CPL image w0 repeated in status table | PASS |
| 224 | ch:wzfuture:L224:-1/3 | calc | `-1/3` | sympy: CPL image wa repeated in status table | PASS |
| 227 | ch:wzfuture:L227 | calc | `55.57` | numeric: asymptotic de Sitter rate, light/photon sector | PASS |
| 227 | ch:wzfuture:L227:67.16 | calc | `67.16` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 227 | ch:wzfuture:L227:70.86 | calc | `70.86` | numeric: asymptotic matter-sector rate | PASS |
| 228 |  | prediction | `-1` | not run: definition: light-ruler EoS fixed value | - |

## Part 2 - ch:lambda - `docs/book/part2/p2_12_lambda.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 35 | eq:lam_rhovac | calc | `4.633\times10^{113}` | numeric: Planck-cutoff vacuum energy density | PASS |
| 37 | ch:lambda:L37 | calc | `1.956\times10^9` | numeric: Planck energy E_P | PASS |
| 40 | eq:lam_rhoL | observed | `5.250\times10^{-10}` | numeric: observed dark-energy density | PASS |
| 42 | ch:lambda:L42 | observed | `67.4` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 42 | ch:lambda:L42:0.6846 | observed | `0.6846` | numeric: Omega_Lambda from 1-Om-Omega_r | PASS |
| 42 | ch:lambda:L42:0.6847 | observed | `0.6847\pm0.0073` | numeric: Planck 2018 Omega_Lambda as printed by Planck | PASS |
| 44 | eq:lam_ratio | calc | `1.133\times10^{-123}` | numeric: ratio of densities | PASS |
| 44 | eq:lam_ratio:-122.95 | calc | `-122.95` | numeric: log10 of density ratio | PASS |
| 87 | ch:lambda:L87 | calc | `2.655\times10^{-30}` | numeric: Gibbons-Hawking horizon temperature | PASS |
| 88 | eq:lam_ebit | calc | `2.541\times10^{-53}` | numeric: Landauer cost per horizon bit | PASS |
| 94 | eq:lam_nmax | calc | `2.265\times10^{122}` | numeric: max horizon info, nats | PASS |
| 94 | eq:lam_nmax:3.268\times10^{122} | calc | `3.268\times10^{122}` | numeric: max horizon info, bits | PASS |
| 120 | ch:lambda:L120 | calc |  | sympy: asymptotic limit of E(a) | PASS |
| 138 | ch:lambda:L138 | calc | `12` | numeric: rounded geometric-term residual factor | PASS |
| 146 | eq:lam_fgeo | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 152 | eq:lam_fb | conjecture |  | not run: assumed baryon writing fraction | - |
| 157 | eq:base | openprob |  | not run: formula, coefficient 2/pi unresolved | - |
| 164 | eq:lam_in1 | measured | `1.616\times10^{-35}` | numeric: Planck length | PASS |
| 165 | eq:lam_in2 | measured | `67.4` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 165 | eq:lam_in2:2.184\times10^{-18} | calc | `2.184\times10^{-18}` | numeric: H0 converted to SI | PASS |
| 166 | eq:lam_in3 | calc | `1.372\times10^{26}` | numeric: Hubble length today | PASS |
| 167 | eq:lam_in4 | measured | `0.0493` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 167 | eq:lam_in4:0.3153 | measured | `0.3153` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 171 | eq:lam_geo | calc | `1.387\times10^{-122}` | numeric: square of Planck/Hubble length ratio | PASS |
| 172 | eq:lam_geo:0.1564 | calc | `0.1564` | numeric: baryon fraction of matter | PASS |
| 176 | eq:lam_result | calc | `1.380\times10^{-123}` | numeric: IAM predicted density ratio | PASS |
| 180 | eq:lam_obs | calc | `1.133\times10^{-123}` | numeric: observed density ratio | PASS |
| 184 | eq:lam_factor | calc | `1.218` | numeric: ratio of prediction to observation | PASS |
| 194 | eq:lam_TH | derived |  | not run: definition: Gibbons-Hawking temperature of the Hubble rate, T_H = hbar H0/(2 pi k_B) (cited GibbonsHawking1977); the derived relation T_dS = T_H sqrt(Omega_L) follows on line 196 | - |
| 196 | eq:lam_TdS | derived |  | sympy: T_dS = hbar H_dS/(2 pi k_B) with H_dS = c sqrt(Lambda/3) equals T_H sqrt(Omega_L), Omega_L = Lambda c^2/(3 H0^2) | PASS |
| 198 | eq:lam_Eeff | derived |  | not run: restatement of the expression on the preceding line (substitution or rearrangement only); nothing independent to compute | - |
| 201 | eq:corr | derived | `1.142\times10^{-123}` | numeric: eq:base times T_dS/T_H, with T_dS from H_dS = c sqrt(Lambda/3) and Lambda = 3 Omega_L H0^2/c^2, evaluated with the book's Planck 2018 inputs (value printed at line 205) | PASS |
| 205 | eq:lam_corr_num | calc | `0.8274` | numeric: square root of Omega_Lambda | PASS |
| 205 | eq:lam_corr_num:1.142\times10^{-123} | calc | `1.142\times10^{-123}` | numeric: de-Sitter-corrected density ratio | PASS |
| 207 | ch:lambda:L207 | calc | `0.79` | numeric: percent excess over observed | PASS |
| 208 | ch:lambda:L208 | calc | `0.521` | numeric: exact-closing exponent of Omega_Lambda | PASS |
| 208 |  | none | `1/2` | not run: trivial restatement, sqrt exponent | - |
| 214 | ch:lambda:L214 | calc | `12.24` | numeric: geometric term alone vs observed | PASS |
| 214 | ch:lambda:L214:1.913 | calc | `1.913` | numeric: after baryon fraction applied | PASS |
| 214 | ch:lambda:L214:1.218 | calc | `1.218` | numeric: after 2/pi applied (repeat) | PASS |
| 214 | ch:lambda:L214:1.0079 | calc | `1.0079` | numeric: after sqrt(Omega_Lambda) applied | PASS |
| 214 | ch:lambda:L214:0.521 | calc | `0.521` | numeric: closing exponent p (repeat) | PASS |
| 214 | ch:lambda:L214:0.79 | calc | `0.79` | numeric: p=1/2 residual percent (repeat) | PASS |
| 223 | ch:lambda:L223 | measured | `1.616\times10^{-35}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 224 | ch:lambda:L224 | calc | `1.372\times10^{26}` | numeric: same value as p2_12_lambda:166 (Hubble length today) | PASS |
| 225 | ch:lambda:L225 | calc | `1.387\times10^{-122}` | numeric: same value as p2_12_lambda:171 (square of Planck/Hubble length ratio) | PASS |
| 226 | ch:lambda:L226 | openprob | `0.6366` | numeric: coefficient 2/pi | PASS |
| 227 | ch:lambda:L227 | measured | `0.1564` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 228 | ch:lambda:L228 | calc | `1.380\times10^{-123}` | numeric: same value as p1_02_iams_law:734 (baseline Lambda/rho_vac with Ob,Om) | PASS |
| 229 | ch:lambda:L229 | calc | `1.133\times10^{-123}` | numeric: same value as p1_02_iams_law:729 (Lambda/rho_vac identity at H0=67.4) | PASS |
| 230 | ch:lambda:L230 | calc | `1.218` | numeric: same value as p2_12_lambda:184 (ratio of prediction to observation) | PASS |
| 231 | ch:lambda:L231 | fitted | `0.8274` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 231 | ch:lambda:L231:1.142\times10^{-123} | fitted | `1.142\times10^{-123}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 231 | ch:lambda:L231:+0.79 | fitted | `+0.79` | numeric: Eq. corr above the observed ratio, per cent | PASS |
| 237 | eq:ident | derived |  | sympy: rho_L/rho_vac = (3 Omega_L/8pi)(l_P/l_H)^2 | PASS |
| 239 | ch:lambda:L239 | derived | `1.133\times10^{-123}` | numeric: same value as p1_02_iams_law:729 (Lambda/rho_vac identity at H0=67.4) | PASS |
| 242 | eq:lam_smarr | derived |  | sympy: T_GH S = c^5/(2GH) for any H | PASS |
| 247 | ch:lambda:L247 | calc | `0.0817` | numeric: 3 Omega_L/8 pi | PASS |
| 253 | ch:lambda:L253 | observed | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 253 | ch:lambda:L253:1.0056 | observed | `1.0056` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 253 | ch:lambda:L253:1.0106 | observed | `1.0106` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 253 | ch:lambda:L253:1.0079 | observed | `1.0079` | numeric: same value as p2_12_lambda:214 (after sqrt(Omega_Lambda) applied) | PASS |
| 256 | eq:rel | derived |  | sympy: Omega_b/Omega_m = (3/16) sqrt(Omega_L) | PASS |
| 259 | ch:lambda:L259 | calc | `0.1564` | numeric: same value as p2_12_lambda:172 (baryon fraction of matter) | PASS |
| 259 | ch:lambda:L259:0.1551 | calc | `0.1551` | numeric: (3/16) sqrt(Omega_L) | PASS |
| 259 | ch:lambda:L259:1.0079 | calc | `1.0079` | numeric: same value as p2_12_lambda:214 (after sqrt(Omega_Lambda) applied) | PASS |
| 259 | ch:lambda:L259:1.0079' | calc | `1.0079` | numeric: ratio of the two sides | PASS |
| 260 | ch:lambda:L260 | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 260 | ch:lambda:L260:1.0056 | measured | `1.0056` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 260 | ch:lambda:L260:1.0106 | measured | `1.0106` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 260 | ch:lambda:L260:0.7 | measured | `0.7` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: 18th chain: ratio of the two sides from 1, in sigma | PASS |
| 270 | ch:lambda:L270 | calc | `0.0817` | numeric: 3 Omega_L/8 pi | PASS |
| 270 |  | calc | `0.7` | not run: restates ch:lambda:L392:3 and ch:lambda:L392 (3 of 414 forms = 0.72 %); printed to one digit, so the 5 % shifted value (0.735) lies inside its own rounding and no check can carry a failing negative control | - |
| 275 | ch:lambda:L275 | calc | `2.2\times10^{68}` | numeric: (E_P/M)^4, M = 100 GeV | PASS |
| 276 | ch:lambda:L276 | calc | `1.4\times10^{79}` | numeric: (E_P/M)^4, M = 200 MeV | PASS |
| 276 | ch:lambda:L276:2.5\times10^{-55} | calc | `2.5\times10^{-55}` | numeric: rho_L/rho_vac(100 GeV) | PASS |
| 277 | ch:lambda:L277 | calc | `1.6\times10^{-44}` | numeric: rho_L/rho_vac(200 MeV) | PASS |
| 307 | eq:lam_aeff | none |  | sympy: half the horizon sphere: A_eff = 2 pi l_H^2 | PASS |
| 309 | eq:lam_nbits | derived |  | sympy: N_eff = A_eff/(4 l_P^2), A_eff = 2 pi l_H^2 | PASS |
| 311 | eq:lam_fbit | derived |  | sympy: f_bit = 2 (l_P/l_H)^2 | PASS |
| 318 | ch:lambda:L318 | openprob | `0.8274` | numeric: temperature factor sqrt(Omega_L) | PASS |
| 326 | ch:lambda:L326 | openprob | `1.2` | numeric: Eq. base against the observed ratio (factor of 1.2) | PASS |
| 340 | ch:lambda:L340 | openprob | `1.2` | numeric: factor left after Ob/Om and 2/pi (1.2) | PASS |
| 340 | ch:lambda:L340:12 | openprob | `12` | numeric: (l_P/l_H)^2 alone over the observed ratio (factor of 12) | PASS |
| 349 | ch:lambda:L349 | calc | `12.24` | numeric: same value as p2_12_lambda:214 (geometric term alone vs observed) | PASS |
| 349 | ch:lambda:L349:1.913 | calc | `1.913` | numeric: same value as p2_12_lambda:214 (after baryon fraction applied) | PASS |
| 349 | ch:lambda:L349:1.218 | calc | `1.218` | numeric: same value as p2_12_lambda:184 (ratio of prediction to observation) | PASS |
| 353 | ch:lambda:L353 | openprob | `1.2` | numeric: remaining factor of 1.2 (heading) | PASS |
| 354 | ch:lambda:L354 | openprob | `1.2` | numeric: remaining factor of 1.2 (objection) | PASS |
| 358 | ch:lambda:L358 | derived | `0.8274` | numeric: same value as p2_12_lambda:205 (square root of Omega_Lambda) | PASS |
| 359 | ch:lambda:L359 | derived | `1.142\times10^{-123}` | numeric: same value as p1_02_iams_law:739 (corrected Lambda/rho_vac with sqrt(OmegaL)) | PASS |
| 359 | ch:lambda:L359:1.133\times10^{-123} | derived | `1.133\times10^{-123}` | numeric: same value as p1_02_iams_law:729 (Lambda/rho_vac identity at H0=67.4) | PASS |
| 362 | ch:lambda:L362 | calc | `0.521` | numeric: exponent p of Omega_L that closes the expression | PASS |
| 363 | ch:lambda:L363 | calc | `+0.79` | numeric: p = 1/2 against the measured ratio, per cent | PASS |
| 363 | ch:lambda:L363:10^{-123} | calc | `10^{-123}` | numeric: the 10^-123 carried by the identity | PASS |
| 366 | ch:lambda:L366 | openprob | `1.22` | numeric: Eq. base within a factor of 1.22 | PASS |
| 367 | ch:lambda:L367 | openprob | `0.79` | numeric: Eq. corr within 0.79 % | PASS |
| 373 | ch:lambda:L373 | observed | `1.1` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: relation holds to 1.1 % at most on every chain | PASS |
| 373 | ch:lambda:L373:0.5 | observed | `0.5` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: relation holds to 0.5 % at least on every chain | PASS |
| 382 | ch:lambda:L382 | calc | `4.633\times10^{113}` | numeric: same value as p2_12_lambda:35 (Planck-cutoff vacuum energy density) | PASS |
| 383 | ch:lambda:L383 | observed | `5.250\times10^{-10}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 384 | ch:lambda:L384 | derived | `1.133\times10^{-123}` | numeric: same value as p1_02_iams_law:729 (Lambda/rho_vac identity at H0=67.4) | PASS |
| 385 | ch:lambda:L385 | calc | `2.265\times10^{122}` | numeric: same value as p2_12_lambda:94 (max horizon info, nats) | PASS |
| 385 | ch:lambda:L385:3.268\times10^{122} | calc | `3.268\times10^{122}` | numeric: same value as p2_12_lambda:94 (max horizon info, bits) | PASS |
| 386 | ch:lambda:L386 | calc | `1.380\times10^{-123}` | numeric: same value as p1_02_iams_law:734 (baseline Lambda/rho_vac with Ob,Om) | PASS |
| 387 | ch:lambda:L387 | fitted | `1.142\times10^{-123}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 387 | ch:lambda:L387:+0.79 | fitted | `+0.79` | numeric: Table lambda_numbers: Eq. corr offset, per cent | PASS |
| 388 | ch:lambda:L388 | calc | `0.521` | numeric: exponent p of Omega_L that closes the expression | PASS |
| 389 | ch:lambda:L389 | observed | `0.1564` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 389 | ch:lambda:L389:0.1551 | observed | `0.1551` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 390 | ch:lambda:L390 | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 390 |  | calc | `18` | not run: label: '18th chain' is the ordinal name of the chain (iam_baryon_test), not a number to recompute | - |
| 391 | ch:lambda:L391 | measured | `1.0056` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 391 | ch:lambda:L391:1.0106 | measured | `1.0106` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 392 | ch:lambda:L392 | calc | `414` | numeric: number of O(1) forms | PASS |
| 392 | ch:lambda:L392:3 | calc | `3` | numeric: O(1) forms within 1 % of 3 Omega_L/8 pi | PASS |
| 393 | ch:lambda:L393 | calc | `0.523` | numeric: required coefficient K = (3 OL/8 pi)/(Ob/Om) | PASS |
| 393 | ch:lambda:L393:3.1\times10^{30} | calc | `3.1\times10^{30}` | numeric: history integral as written, as coefficient K | PASS |
| 394 | ch:lambda:L394 | calc | `3.15\times10^{-8}` | numeric: accumulated virial heat of baryons over rho_L c^2 | PASS |

## Part 2 - ch:lambda_history - `docs/book/part2/p2_12b_lambda_history.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 21 | eq:lh_dE | derived |  | sympy: dE/da = E/a^2, 1 at a = 1 | PASS |
| 24 | eq:lh_w | derived |  | sympy: w = -1 - (1/3) dln rho/dln a from continuity | PASS |
| 26 | eq:lh_eps | derived |  | sympy: w = -1 - epsilon from the continuity equation | PASS |
| 29 | eq:lh_winfo | derived |  | sympy: w_info = -1 - 1/(3a), -4/3 today | PASS |
| 34 | ch:lambda_history:L34 | observed | `2.8` | numeric: DESI DR2 preference, lowest | PASS |
| 34 | ch:lambda_history:L34:4.2 | observed | `4.2` | numeric: DESI DR2 preference, highest | PASS |
| 45 | eq:lh_integral | conjecture |  | not run: conjecture: the history integral Eq. lh_integral is the conjectured form of the accumulation (a definition); its evaluation is checked at ch:lambda_history:L72:3.1\times10^{30}, ch:lambda_history:L64, L65 | - |
| 59 | ch:lambda_history:L59 | derived | `9.22\times10^{-5}` | numeric: Omega_r from T_CMB and N_eff 3.046 | PASS |
| 59 |  | derived | `2.3\times10^{-15}` | not run: input: a_EW = 2.3e-15 as printed in Chapter electroweak, used as the lower limit; the entropy-conserving value 7.8e-16 is checked at ch:lambda_history:L63 | - |
| 61 | eq:lh_K | calc |  | sympy: required coefficient K = (3 OL/8 pi)/(Ob/Om) | PASS |
| 63 | ch:lambda_history:L63 | calc | `7.8\times10^{-16}` | numeric: a at T = 100 GeV, entropy conservation | PASS |
| 63 | ch:lambda_history:L63:106.75 | calc | `106.75` | numeric: g_*s of the standard model above the electroweak scale | PASS |
| 64 | ch:lambda_history:L64 | calc | `2.7\times10^{31}` | numeric: K from a at 100 GeV (entropy conservation) | PASS |
| 64 |  | calc | `2.3\times10^{-15}` | not run: input restated: the printed a_EW = 2.3e-15 (see line 59); the corrected 7.8e-16 is checked at ch:lambda_history:L63 | - |
| 64 |  | calc | `159.5` | not run: input: electroweak crossover temperature 159.5 GeV (Chapter electroweak, lattice result), used as a lower-limit temperature | - |
| 65 | ch:lambda_history:L65 | calc | `6.9\times10^{31}` | numeric: K from a at the 159.5 GeV crossover | PASS |
| 65 | ch:lambda_history:L65:6.5\times10^6 | calc | `6.5\times10^6` | numeric: K from a = 1e-3 | PASS |
| 66 |  | calc | `10` | not run: input: the lower limit a = 10^-3 chosen for the comparison; the K it gives is checked at ch:lambda_history:L65:6.5\times10^6 | - |
| 72 | ch:lambda_history:L72 | calc | `0.523` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 72 | ch:lambda_history:L72:0.470 | calc | `0.470` | numeric: int (H/H0)^-2 dE, Omega_m 0.3198 | PASS |
| 72 | ch:lambda_history:L72:0.641 | calc | `0.641` | numeric: int (H/H0)^-1 dE, Omega_m 0.3198 | PASS |
| 72 | ch:lambda_history:L72:1.983 | calc | `1.983` | numeric: int (H/H0)^1 dE, Omega_m 0.3198 | PASS |
| 72 | ch:lambda_history:L72:5.797 | calc | `5.797` | numeric: int (H/H0)^2 dE, Omega_m 0.3198 | PASS |
| 72 | ch:lambda_history:L72:3.1\times10^{30} | calc | `3.1\times10^{30}` | numeric: figure caption: K from the printed a_EW | PASS |
| 72 | ch:lambda_history:L72:0.3198 | calc | `0.3198` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: 18th-chain Omega_m | PASS |
| 72 |  | calc | `2.3\times10^{-15}` | not run: input restated: the printed a_EW = 2.3e-15 in the figure caption | - |
| 77 | eq:lh_weights | calc |  | sympy: the five weights int (H/H0)^p dE | PASS |
| 81 | ch:lambda_history:L81 | calc | `0.470` | numeric: int (H0/H)^2 dE | PASS |
| 81 | ch:lambda_history:L81:0.523 | calc | `0.523` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 81 | ch:lambda_history:L81:10 | calc | `10` | numeric: per cent below the required 0.523 | PASS |
| 86 |  | calc | `10` | not run: input: halo mass cut 10^8 Msun of the Sheth-Tormen sum (the printed 10 is its base) | - |
| 87 | ch:lambda_history:L87 | calc | `0.811` | numeric: sigma8 normalisation of the power spectrum | PASS |
| 91 | ch:lambda_history:L91 | calc | `0.58` | numeric: baryon mass fraction in halos above 1e8 Msun | PASS |
| 91 | ch:lambda_history:L91:213 | calc | `213` | numeric: mean dispersion sigma_eff, km/s | PASS |
| 92 | ch:lambda_history:L92 | calc | `3.15\times10^{-8}` | numeric: accumulated virial heat of baryons over rho_L c^2 | PASS |
| 93 | ch:lambda_history:L93 | calc | `2.6\times10^{-44}` | numeric: virial heat priced at the horizon, over rho_L c^2 | PASS |
| 94 | ch:lambda_history:L94 | calc | `0.072` | numeric: Omega_b/Omega_L | PASS |
| 96 |  | interp | `3\times10^{-8}` | not run: restates ch:lambda_history:L92 and ch:lambda:L394 (3.15e-8) to one digit; the 5 % shifted value 3.15e-8 coincides with the computed value, so no check can carry a failing negative control | - |
| 129 | ch:lambda_history:L129 | conjecture | `13.8` | numeric: age of the universe, Gyr | PASS |
| 133 | eq:lh_base | none | `1.380\times10^{-123}` | numeric: (2/pi)(l_P/l_H)^2 Ob/Om | PASS |
| 137 | eq:lh_corr | calc | `1.142\times10^{-123}` | numeric: (2/pi)(l_P/l_H)^2 sqrt(OL) Ob/Om | PASS |
| 140 | ch:lambda_history:L140 | calc | `0.79` | numeric: Eq. lh_corr above the observed ratio, per cent | PASS |
| 141 | ch:lambda_history:L141 | observed | `0.5` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: relation on the CMB-only chain, per cent | PASS |
| 156 | ch:lambda_history:L156 | calc | `13.8` | numeric: age of the universe, Gyr (conclusions) | PASS |
| 157 | ch:lambda_history:L157 | calc | `8.5\times10^{60}` | numeric: Hubble radius in Planck lengths (H0 = 67.4) | PASS |
| 157 | ch:lambda_history:L157:15.6 | calc | `15.6` | numeric: baryon share of matter | PASS |
| 157 | ch:lambda_history:L157:68.5 | calc | `68.5` | numeric: Omega_L in per cent | PASS |
| 158 | ch:lambda_history:L158 | calc | `0.827` | numeric: sqrt Omega_L | PASS |
| 158 | ch:lambda_history:L158:1.142\times10^{-123} | calc | `1.142\times10^{-123}` | numeric: same value as p1_02_iams_law:739 (corrected Lambda/rho_vac with sqrt(OmegaL)) | PASS |
| 158 | ch:lambda_history:L158:1.133\times10^{-123} | calc | `1.133\times10^{-123}` | numeric: same value as p1_02_iams_law:729 (Lambda/rho_vac identity at H0=67.4) | PASS |
| 160 | ch:lambda_history:L160 | observed | `0.5` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: relation observed to 0.5 % (conclusions) | PASS |

## Part 2 - ch:baryon - `docs/book/part2/p2_13_baryon.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 29 | eq:bar_eta | none | `6.1\times10^{-10}` | numeric: baryon-to-photon ratio, about 6.1e-10 | PASS |
| 31 | ch:baryon:L31 | observed | `6.180` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 31 | ch:baryon:L31:6.108 | observed | `6.108` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 32 | ch:baryon:L32 | observed | `6.127` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 32 | ch:baryon:L32:0.02237 | observed | `0.02237` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 33 | ch:baryon:L33 | observed | `273.9\times10^{-10}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 33 |  | observed | `10` | not run: input: the unit 10^{-10} of Steigman's conversion eta = 273.9e-10 Omega_b h^2 (doi:10.1088/1475-7516/2006/10/016); the printed 10 is the base of that power, nothing to recompute | - |
| 83 | ch:baryon:L83 | calc | `3.1\times10^{30}` | numeric: history integral as written, as coefficient K | PASS |
| 83 | ch:baryon:L83:0.523 | calc | `0.523` | numeric: required coefficient K | PASS |
| 91 | ch:baryon:L91 | conjecture | `13.8` | numeric: age of the universe, Gyr | PASS |
| 97 | ch:baryon:L97 | observed | `0.1564` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 105 | ch:baryon:L105 | calc | `9.5\times10^{-13}` | numeric: a at T = 150 MeV, entropy conservation ('approx') | PASS |
| 105 | ch:baryon:L105:17.25 | calc | `17.25` | numeric: g_*s at T = 150 MeV | PASS |
| 106 | ch:baryon:L106 | calc | `0.6` | numeric: baryons per 1e9 photons at eta = 6.1e-10 | PASS |
| 106 |  | calc | `10` | not run: input: the printed 10 is the base of '10^9 photons', a unit of the sentence; the 0.6 baryons per 1e9 photons is checked at ch:baryon:L106 | - |
| 108 | eq:bar_nb | derived |  | not run: definition: n_b = eta n_gamma is Eq. bar_eta rearranged (the book labels it 'derived (definition)') | - |
| 129 | eq:bar_loop | none |  | not run: definition: Eq. bar_loop is a schematic causal chain (eta -> n_b -> W_info -> Lambda_acc -> {rho_dm, rho_de} -> H -> eta), no equation to verify | - |
| 150 | ch:baryon:L150 | calc | `1.93\times10^4` | numeric: Hubble rate at T = 150 MeV, g* = 17.25, s^-1 | PASS |
| 150 |  | calc | `150` | not run: input: QCD transition temperature T = 150 MeV used as the epoch; the quantities computed at it are checked at ch:baryon:L105, L150, L151 | - |
| 151 | ch:baryon:L151 | calc | `5.0\times10^{-13}` | numeric: Hubble length, pc | PASS |
| 151 | ch:baryon:L151:2.9\times10^{78} | calc | `2.9\times10^{78}` | numeric: A_H/(4 l_P^2), nats | PASS |
| 151 | ch:baryon:L151:4.2\times10^{78} | calc | `4.2\times10^{78}` | numeric: bits | PASS |
| 151 | ch:baryon:L151:15.5 | calc | `15.5` | numeric: Hubble length, km | PASS |
| 164 |  | calc | `273.9\times10^{-10}` | not run: input: Steigman's conversion eta = 273.9e-10 Omega_b h^2 (JCAP 10 (2006) 016, doi:10.1088/1475-7516/2006/10/016), quoted | - |
| 165 | ch:baryon:L165 | calc | `0.1431` | numeric: Omega_m h^2 = 0.3153 x 0.6736^2 | PASS |
| 165 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153 (Planck 2018 VI Table 2) restated; Omega_m h^2 = 0.1431 is checked at ch:baryon:L165 | - |
| 166 | ch:baryon:L166 | calc | `0.01837` | numeric: Omega_b h^2 from Omega_b/Omega_m = 3 Omega_L/16 | PASS |
| 166 | ch:baryon:L166:5.03\times10^{-10} | calc | `5.03\times10^{-10}` | numeric: eta from it | PASS |
| 167 | ch:baryon:L167 | calc | `0.02220` | numeric: Omega_b h^2 from (3/16) sqrt(Omega_L) | PASS |
| 168 | ch:baryon:L168 | calc | `6.08\times10^{-10}` | numeric: eta from it | PASS |
| 168 | ch:baryon:L168:0.5 | calc | `0.5` | numeric: below the 18th-chain 6.113 | PASS |
| 168 | ch:baryon:L168:6.113 | calc | `6.113` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_13b_baryon_chain:117 (eta = 2.739e-8 Omega_b h^2 from the 18th chain) | PASS |
| 168 | ch:baryon:L168:0.8 | calc | `0.8` | numeric: below Planck 6.127 | PASS |
| 168 | ch:baryon:L168:6.137 | calc | `6.137` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: largest eta of the CMB chains | PASS |
| 168 |  | calc | `0.9` | not run: printed value 0.9 (per cent) has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.936; 1.05 x 0.9 = 0.945 lies within the capped tolerance). The eta it rests on is checked at ch:baryon:L168 (6.08e-10) and the 6.137 at ch:baryon:L168:6.137 | - |
| 169 | ch:baryon:L169 | calc | `6.127` | numeric: Planck's eta x 1e10 | PASS |
| 192 | ch:baryon:L192 | measured | `0.009273` | file `docs/verification/cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md`: measured: printed value found in CC_AND_BARYON_CHECK.md, a file the chapter names | PASS |

## Part 2 - ch:baryon_chain - `docs/book/part2/p2_13b_baryon_chain.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 21 | eq:bc_eta | observed | `6.1\times10^{-10}` | numeric: eta from the Planck 2018 Omega_b h^2 and the n_b/n_gamma conversion | PASS |
| 33 | eq:bc_law | derived |  | sympy: Landauer bound: maximising the entropy of one binary record gives k_B ln 2, so the cost at T_H is k_B T_H ln 2 | PASS |
| 45 | eq:bc_cc | openprob |  | sympy: eq:bc_cc with rho_L and rho_vac written from their definitions, solved for Omega_b/Omega_m, gives (3/16) Omega_L | PASS |
| 49 | eq:bc_tratio | derived |  | sympy: H_dS = c sqrt(Lambda/3) for the horizon set by Lambda alone and Omega_L = Lambda c^2/(3 H0^2) give H_dS/H0 = sqrt(Omega_L); T_GH is proportional to H | PASS |
| 51 | eq:bc_cccorr | derived | `1.142\times10^{-123}` | numeric: eq:bc_cc times T_dS/T_H, with T_dS from H_dS = c sqrt(Lambda/3) and Lambda = 3 Omega_L H0^2/c^2, evaluated with the book's Planck 2018 inputs | PASS |
| 53 | ch:baryon_chain:L53 | fitted | `1.133\times10^{-123}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 53 | ch:baryon_chain:L53:0.79 | fitted | `0.79` | numeric: eq:bc_cccorr above the observed rho_L/rho_vac, per cent | PASS |
| 56 | eq:bc_etaob | derived | `2.739\times10^{-8}` | numeric: eta / Omega_b h^2 from rho_crit, the mean baryon mass and n_gamma (Steigman 2006 conversion) | PASS |
| 57 | ch:baryon_chain:L57 | derived | `0.1431` | numeric: Omega_m h^2 | PASS |
| 57 | ch:baryon_chain:L57:0.6847 | derived | `0.6847` | numeric: Omega_L = 1 - Omega_m (flat), Planck 2018 | PASS |
| 58 | ch:baryon_chain:L58 | calc | `0.01837` | numeric: Omega_b h^2 without sqrt | PASS |
| 58 | ch:baryon_chain:L58:5.03\times10^{-10} | calc | `5.03\times10^{-10}` | numeric: eta without sqrt | PASS |
| 58 | ch:baryon_chain:L58:0.02220 | calc | `0.02220` | numeric: Omega_b h^2 with sqrt | PASS |
| 59 | ch:baryon_chain:L59 | calc | `6.08\times10^{-10}` | numeric: eta with sqrt | PASS |
| 59 | ch:baryon_chain:L59:0.8 | calc | `0.8` | numeric: below Planck 6.127 | PASS |
| 59 | ch:baryon_chain:L59:6.127 | calc | `6.127` | numeric: Planck 2018 eta (1e-10) from its Omega_b h^2 | PASS |
| 69 | eq:bc_std | calc | `0.00014` | file `Cosmological_Physics/mgcamb_validation/chains/lcdm_baseline.updated.yaml`: other runs: Omega_b h^2 flat on [0.020, 0.025], start N(0.02242, 0.00014^2), from the four LambdaCDM YAML files | PASS |
| 73 | ch:baryon_chain:L73 | calc | `0.030` | numeric: 18th-chain prior width | PASS |
| 73 | ch:baryon_chain:L73:0.005 | calc | `0.005` | numeric: other runs' prior width | PASS |
| 115 | eq:bc_ob | none | `0.022320` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: 18th chain Omega_b h^2 and sd | PASS |
| 117 | eq:bc_etares | measured | `6.113` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: eta = 2.739e-8 Omega_b h^2 from the 18th chain | PASS |
| 118 | ch:baryon_chain:L118 | measured | `0.022319` | heavy file `docs/book/verification/sources/bl_MANIFEST.md`: the run record's Omega_b h^2 (30 % burn-in on the chain copy) | PASS |
| 118 | ch:baryon_chain:L118:6.1155\times10^{-10} | measured | `6.1155\times10^{-10}` | heavy file `docs/book/verification/sources/bl_MANIFEST.md`: the record's eta = 2.74e-8 x the record's Omega_b h^2 | PASS |
| 119 | ch:baryon_chain:L119 | calc | `2.74\times10^{-8}` | numeric: the record's conversion factor 2.74e-8 is the derived factor to three figures | PASS |
| 119 | ch:baryon_chain:L119:2.739\times10^{-8} | calc | `2.739\times10^{-8}` | numeric: the conversion factor used in the chapter, recomputed | PASS |
| 120 | ch:baryon_chain:L120 | measured | `0.0218` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 120 | ch:baryon_chain:L120:0.0228 | measured | `0.0228` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 121 | ch:baryon_chain:L121 | measured | `67.04` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 121 | ch:baryon_chain:L121:0.3198 | measured | `0.3198` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 121 | ch:baryon_chain:L121:0.010 | measured | `0.010` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: 18th chain: lower end of the flat Omega_b h^2 range | PASS |
| 121 | ch:baryon_chain:L121:0.040 | measured | `0.040` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: 18th chain: upper end of the flat Omega_b h^2 range | PASS |
| 122 | ch:baryon_chain:L122 | measured | `0.1554` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 123 | eq:bc_ratio | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: ratio on the 18th chain, committed output | PASS |
| 128 | ch:baryon_chain:L128 | observed | `273.9\times10^{-10}` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 128 | ch:baryon_chain:L128:6.113 | observed | `6.113` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 128 | ch:baryon_chain:L128:6.117 | observed | `6.117` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 128 | ch:baryon_chain:L128:6.137 | observed | `6.137` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 128 | ch:baryon_chain:L128:0.1431 | calc | `0.1431` | numeric: same value as p2_13_baryon:165 (Omega_m h^2 = 0.3153 x 0.6736^2) | PASS |
| 128 | ch:baryon_chain:L128:6.080 | calc | `6.080` | numeric: eta (1e-10) from the expression with sqrt | PASS |
| 128 | ch:baryon_chain:L128:5.031 | calc | `5.031` | numeric: eta (1e-10) without sqrt | PASS |
| 136 |  | measured | `10` | not run: unit: '$10^{-10}$' in the table caption, the unit of the eta column, nothing to recompute | - |
| 137 | ch:baryon_chain:L137 | measured | `6.180` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 138 | ch:baryon_chain:L138 | measured | `6.108` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 139 | ch:baryon_chain:L139 | measured | `6.098` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 140 | ch:baryon_chain:L140 | measured | `6.127` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 141 | ch:baryon_chain:L141 | measured | `6.141` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 142 | ch:baryon_chain:L142 | measured | `6.113` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 142 | ch:baryon_chain:L142:0.010 | measured | `0.010` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: Table tab:bc_compare, 18th chain: lower end of the flat range | PASS |
| 142 | ch:baryon_chain:L142:0.040 | measured | `0.040` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: Table tab:bc_compare, 18th chain: upper end of the flat range | PASS |
| 142 |  | measured | `18` | not run: label: '18th' in '18th chain' is the ordinal name of the chain, not a number to recompute | - |
| 143 | ch:baryon_chain:L143 | calc | `6.080` | numeric: eta (1e-10) from the expression with sqrt | PASS |
| 144 | ch:baryon_chain:L144 | calc | `5.031` | numeric: eta (1e-10) without sqrt | PASS |
| 148 | ch:baryon_chain:L148 | calc | `6.113` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: 18th chain eta | PASS |
| 148 | ch:baryon_chain:L148:0.3 | calc | `0.3` | numeric: chain vs nucleosynthesis, sigma | PASS |
| 148 | ch:baryon_chain:L148:0.2 | calc | `0.2` | numeric: below Planck 6.127, per cent | PASS |
| 148 | ch:baryon_chain:L148:6.180 | calc | `6.180` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: nucleosynthesis with deuterium, Cyburt et al. 2016 Table IV, as traced in the committed output | PASS |
| 148 | ch:baryon_chain:L148:6.127 | calc | `6.127` | numeric: Planck 2018 eta (1e-10) from its Omega_b h^2 | PASS |
| 159 | ch:baryon_chain:L159 | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 159 | ch:baryon_chain:L159:0.7 | measured | `0.7` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: ratio (Ob/Om)/[(3/16) sqrt(OL)] on the 18th chain: distance from 1 in sigma | PASS |
| 165 |  | measured | `10` | not run: unit: '$10^{10}\eta$' column header, nothing to recompute | - |
| 166 | ch:baryon_chain:L166 | measured | `0.02232` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 166 | ch:baryon_chain:L166:6.113 | measured | `6.113` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 166 | ch:baryon_chain:L166:1.0046 | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 166 | ch:baryon_chain:L166:957 | measured | `957` | file `docs/verification/cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md`: measured: printed value found in CC_AND_BARYON_CHECK.md, a file the chapter names | PASS |
| 166 | ch:baryon_chain:L166:0.010 | measured | `0.010` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: Table tab:baryon_chains, iam_baryon_test: lower end of the flat Omega_b h^2 range | PASS |
| 166 | ch:baryon_chain:L166:0.040 | measured | `0.040` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: Table tab:baryon_chains, iam_baryon_test: upper end of the flat Omega_b h^2 range | PASS |
| 166 | ch:baryon_chain:L166:14{,}957 | measured | `14{,}957` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Table tab:baryon_chains, iam_baryon_test: rows after 30 % burn-in | PASS |
| 166 |  | measured | `18` | not run: label: '18th' in '18th chain (CMB only)' is the ordinal name of the chain, not a number to recompute | - |
| 167 | ch:baryon_chain:L167 | measured | `0.02234` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 167 | ch:baryon_chain:L167:6.118 | measured | `6.118` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 167 | ch:baryon_chain:L167:1.0058 | measured | `1.0058` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 167 | ch:baryon_chain:L167:0.020 | measured | `0.020` | file `Cosmological_Physics/mgcamb_validation/chains/lcdm_baseline.updated.yaml`: Table tab:baryon_chains, lcdm_baseline: lower end of the flat Omega_b h^2 range | PASS |
| 167 | ch:baryon_chain:L167:0.025 | measured | `0.025` | file `Cosmological_Physics/mgcamb_validation/chains/lcdm_baseline.updated.yaml`: Table tab:baryon_chains, lcdm_baseline: upper end of the flat Omega_b h^2 range | PASS |
| 167 | ch:baryon_chain:L167:12{,}544 | measured | `12{,}544` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Table tab:baryon_chains, lcdm_baseline: rows after 30 % burn-in | PASS |
| 168 | ch:baryon_chain:L168 | measured | `0.02240` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 168 | ch:baryon_chain:L168:6.137 | measured | `6.137` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 168 | ch:baryon_chain:L168:1.0106 | measured | `1.0106` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 168 | ch:baryon_chain:L168:0.020 | measured | `0.020` | file `Cosmological_Physics/mgcamb_validation/chains/planck_bao_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_bao_lcdm_baseline: lower end of the flat Omega_b h^2 range | PASS |
| 168 | ch:baryon_chain:L168:0.025 | measured | `0.025` | file `Cosmological_Physics/mgcamb_validation/chains/planck_bao_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_bao_lcdm_baseline: upper end of the flat Omega_b h^2 range | PASS |
| 168 | ch:baryon_chain:L168:12{,}600 | measured | `12{,}600` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Table tab:baryon_chains, planck_bao_lcdm_baseline: rows after 30 % burn-in | PASS |
| 169 | ch:baryon_chain:L169 | measured | `0.02233` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 169 | ch:baryon_chain:L169:6.117 | measured | `6.117` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 169 | ch:baryon_chain:L169:1.0056 | measured | `1.0056` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 169 | ch:baryon_chain:L169:0.020 | measured | `0.020` | file `Cosmological_Physics/mgcamb_validation/chains/planck_pantheon_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_pantheon_lcdm_baseline: lower end of the flat Omega_b h^2 range | PASS |
| 169 | ch:baryon_chain:L169:0.025 | measured | `0.025` | file `Cosmological_Physics/mgcamb_validation/chains/planck_pantheon_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_pantheon_lcdm_baseline: upper end of the flat Omega_b h^2 range | PASS |
| 169 | ch:baryon_chain:L169:21{,}168 | measured | `21{,}168` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Table tab:baryon_chains, planck_pantheon_lcdm_baseline: rows after 30 % burn-in | PASS |
| 170 | ch:baryon_chain:L170 | measured | `0.02239` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 170 | ch:baryon_chain:L170:6.134 | measured | `6.134` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 170 | ch:baryon_chain:L170:1.0098 | measured | `1.0098` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 170 | ch:baryon_chain:L170:0.020 | measured | `0.020` | file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_rsd_lcdm_baseline: lower end of the flat Omega_b h^2 range | PASS |
| 170 | ch:baryon_chain:L170:0.025 | measured | `0.025` | file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_lcdm_baseline.updated.yaml`: Table tab:baryon_chains, planck_rsd_lcdm_baseline: upper end of the flat Omega_b h^2 range | PASS |
| 170 | ch:baryon_chain:L170:18{,}424 | measured | `18{,}424` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Table tab:baryon_chains, planck_rsd_lcdm_baseline: rows after 30 % burn-in | PASS |
| 171 | ch:baryon_chain:L171 | calc | `0.01837` | numeric: Omega_b h^2 without sqrt | PASS |
| 171 | ch:baryon_chain:L171:5.031 | calc | `5.031` | numeric: eta (1e-10) without sqrt | PASS |
| 172 | ch:baryon_chain:L172 | calc | `0.02220` | numeric: with sqrt | PASS |
| 172 | ch:baryon_chain:L172:6.080 | calc | `6.080` | numeric: eta (1e-10) from the expression with sqrt | PASS |
| 179 | ch:baryon_chain:L179 | measured | `0.02232` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 179 | ch:baryon_chain:L179:0.010 | measured | `0.010` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: fig:baryon_posterior caption: 18th chain, lower end of the flat Omega_b h^2 range | PASS |
| 179 | ch:baryon_chain:L179:0.040 | measured | `0.040` | file `Cosmological_Physics/mgcamb_validation/yaml_configs/iam_baryon_test.updated.yaml`: fig:baryon_posterior caption: 18th chain, upper end of the flat Omega_b h^2 range | PASS |
| 179 | ch:baryon_chain:L179:0.020 | measured | `0.020` | file `Cosmological_Physics/mgcamb_validation/chains/lcdm_baseline.updated.yaml`: fig:baryon_posterior caption: other runs, lower end of the flat Omega_b h^2 range | PASS |
| 179 | ch:baryon_chain:L179:0.025 | measured | `0.025` | file `Cosmological_Physics/mgcamb_validation/chains/lcdm_baseline.updated.yaml`: fig:baryon_posterior caption: other runs, upper end of the flat Omega_b h^2 range | PASS |
| 183 | ch:baryon_chain:L183 | openprob | `6\times10^{-10}` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: the eta every CMB fit returns, to one figure: the 18th chain | PASS |
| 188 | ch:baryon_chain:L188 | fitted | `0.827` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 188 | ch:baryon_chain:L188:+0.79 | fitted | `+0.79` | numeric: sqrt(Omega_L) brings eq:bc_cc to +0.79 % of the observed Lambda | PASS |
| 189 | ch:baryon_chain:L189 | fitted | `5.03` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: measured: printed value found in verify_lambda_baryon_book_output.txt, a file the chapter names | PASS |
| 189 | ch:baryon_chain:L189:6.08\times10^{-10} | fitted | `6.08\times10^{-10}` | numeric: eq:bc_cccorr inverted for eta at the Planck Omega_m h^2 | PASS |
| 196 | ch:baryon_chain:L196 | measured | `0.2` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: 18th chain eta below the Planck 2018 value, per cent | PASS |
| 196 | ch:baryon_chain:L196:0.3 | measured | `0.3` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: 18th chain eta from nucleosynthesis with deuterium, in sigma | PASS |

## Part 2 - ch:surveys - `docs/book/part2/p2_16_survey_predictions.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 29 | eq:sp_poisson | none |  | not run: definition: the mu-Sigma parametrisation of the two Poisson equations (Pogosian & Silvestri 2016); mu and Sigma are defined by it | - |
| 31 | eq:sp_mu | none |  | sympy: mu(a) of Eq. sp_mu: mu(a=1) = 1/(1+Omega_m/2) and mu -> 1 as a -> 0 | PASS |
| 36 | eq:sp_mu0 | prediction |  | sympy: mu0 = -beta_m/(1+beta_m) = -0.136 | PASS |
| 38 | ch:surveys:L38 | calc | `-0.13618` | numeric: same value as p1_02_iams_law:465 (mu0 at beta_m=0.15765, precise) | PASS |
| 38 | ch:surveys:L38:-0.13495 | calc | `-0.13495` | file `Cosmological_Physics/mgcamb_validation/chains/iam_fixed_mu0_r2.updated.yaml`: mu0 of the MGCAMB tracking form in the Level 1 chains, from the chain settings | PASS |
| 40 | ch:surveys:L40 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2: Delta chi2 of the IAM run against LambdaCDM (chain minima) | PASS |
| 41 | ch:surveys:L41 | measured | `0.800` | heavy file `docs/verification/scripts/verify_dark_energy_far_future_surveys_book_output.txt`: measured: printed value found in verify_dark_energy_far_future_surveys_book_output.txt, a file the chapter names | PASS |
| 53 | ch:surveys:L53 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 56 | ch:surveys:L56 | calc | `0.864` | numeric: mu at z=0.0 | PASS |
| 56 | ch:surveys:L56:13.62 | calc | `13.62` | numeric: 1 - mu at z=0.0, per cent | PASS |
| 56 | ch:surveys:L56:-0.78 | calc | `-0.78` | numeric: Delta D/D (= Delta Phi/Phi) at z=0.0 | PASS |
| 56 | ch:surveys:L56:-4.25 | calc | `-4.25` | numeric: Delta f sigma8 at z=0.0 | PASS |
| 57 | ch:surveys:L57 | calc | `0.886` | numeric: mu at z=0.1 | PASS |
| 57 | ch:surveys:L57:11.44 | calc | `11.44` | numeric: 1 - mu at z=0.1, per cent | PASS |
| 57 | ch:surveys:L57:-0.61 | calc | `-0.61` | numeric: Delta D/D (= Delta Phi/Phi) at z=0.1 | PASS |
| 57 | ch:surveys:L57:-3.42 | calc | `-3.42` | numeric: Delta f sigma8 at z=0.1 | PASS |
| 57 |  | calc | `0.1` | not run: label: redshift of a table row / bin (z = 0.1), an input of the computation, nothing to recompute | - |
| 58 | ch:surveys:L58 | calc | `0.922` | numeric: mu at z=0.3 | PASS |
| 58 | ch:surveys:L58:7.82 | calc | `7.82` | numeric: 1 - mu at z=0.3, per cent | PASS |
| 58 | ch:surveys:L58:-0.37 | calc | `-0.37` | numeric: Delta D/D (= Delta Phi/Phi) at z=0.3 | PASS |
| 58 | ch:surveys:L58:-2.17 | calc | `-2.17` | numeric: Delta f sigma8 at z=0.3 | PASS |
| 58 |  | calc | `0.3` | not run: label: redshift of a table row / bin (z = 0.3), an input of the computation, nothing to recompute | - |
| 59 | ch:surveys:L59 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 59 | ch:surveys:L59:5.18 | calc | `5.18` | numeric: 1 - mu at z=0.5, per cent | PASS |
| 59 | ch:surveys:L59:-0.22 | calc | `-0.22` | numeric: Delta D/D (= Delta Phi/Phi) at z=0.5 | PASS |
| 59 | ch:surveys:L59:-1.35 | calc | `-1.35` | numeric: Delta f sigma8 at z=0.5 | PASS |
| 59 |  | calc | `0.5` | not run: label: redshift of a table row / bin (z = 0.5), an input of the computation, nothing to recompute | - |
| 60 | ch:surveys:L60 | calc | `0.966` | numeric: mu at z=0.7 | PASS |
| 60 | ch:surveys:L60:3.39 | calc | `3.39` | numeric: 1 - mu at z=0.7, per cent | PASS |
| 60 | ch:surveys:L60:-0.13 | calc | `-0.13` | numeric: Delta D/D (= Delta Phi/Phi) at z=0.7 | PASS |
| 60 | ch:surveys:L60:-0.83 | calc | `-0.83` | numeric: Delta f sigma8 at z=0.7 | PASS |
| 60 |  | calc | `0.7` | not run: label: redshift of a table row / bin (z = 0.7), an input of the computation, nothing to recompute | - |
| 61 | ch:surveys:L61 | calc | `0.982` | numeric: mu at z=1.0 | PASS |
| 61 | ch:surveys:L61:1.78 | calc | `1.78` | numeric: 1 - mu at z=1.0, per cent | PASS |
| 61 | ch:surveys:L61:-0.06 | calc | `-0.06` | numeric: Delta D/D (= Delta Phi/Phi) at z=1.0 | PASS |
| 61 | ch:surveys:L61:-0.41 | calc | `-0.41` | numeric: Delta f sigma8 at z=1.0 | PASS |
| 61 |  | calc | `1.0` | not run: label: redshift of a table row / bin (z = 1.0), an input of the computation, nothing to recompute | - |
| 62 | ch:surveys:L62 | calc | `0.994` | numeric: mu at z=1.5 | PASS |
| 62 | ch:surveys:L62:0.62 | calc | `0.62` | numeric: 1 - mu at z=1.5, per cent | PASS |
| 62 | ch:surveys:L62:-0.02 | calc | `-0.02` | numeric: Delta D/D (= Delta Phi/Phi) at z=1.5 | PASS |
| 62 | ch:surveys:L62:-0.13 | calc | `-0.13` | numeric: Delta f sigma8 at z=1.5 | PASS |
| 62 |  | calc | `1.5` | not run: label: redshift of a table row / bin (z = 1.5), an input of the computation, nothing to recompute | - |
| 63 | ch:surveys:L63 | calc | `0.998` | numeric: mu at z=2.0 | PASS |
| 63 | ch:surveys:L63:0.23 | calc | `0.23` | numeric: 1 - mu at z=2.0, per cent | PASS |
| 63 | ch:surveys:L63:-0.01 | calc | `-0.01` | numeric: Delta D/D (= Delta Phi/Phi) at z=2.0 | PASS |
| 63 | ch:surveys:L63:-0.04 | calc | `-0.04` | numeric: Delta f sigma8 at z=2.0 | PASS |
| 63 |  | calc | `2.0` | not run: label: redshift of a table row / bin (z = 2.0), an input of the computation, nothing to recompute | - |
| 70 | ch:surveys:L70 | calc | `+1.8` | numeric: E_G = Omega_m0 Sigma/f: change at z = 0.3, per cent | PASS |
| 70 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153, Planck 2018 VI Table 2 (the global Om of the growth checks) | - |
| 70 |  | calc | `0.3` | not run: label: redshift of a table row / bin (z = 0.3), an input of the computation, nothing to recompute | - |
| 79 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153, Planck 2018 VI Table 2, restated in the table caption | - |
| 82 | ch:surveys:L82 | calc | `0.305` | numeric: a where 1% of 1-mu(0) is on | PASS |
| 82 | ch:surveys:L82:2.28 | calc | `2.28` | numeric: z where 1% is on | PASS |
| 82 | ch:surveys:L82:10.9 | calc | `10.9` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 82 | ch:surveys:L82:0.407 | calc | `0.407` | numeric: a where 5% of 1-mu(0) is on | PASS |
| 82 | ch:surveys:L82:1.46 | calc | `1.46` | numeric: z where 5% is on | PASS |
| 82 | ch:surveys:L82:9.4 | calc | `9.4` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 82 | ch:surveys:L82:0.471 | calc | `0.471` | numeric: a where 10% of 1-mu(0) is on | PASS |
| 82 | ch:surveys:L82:1.12 | calc | `1.12` | numeric: z where 10% is on | PASS |
| 82 | ch:surveys:L82:8.4 | calc | `8.4` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 82 | ch:surveys:L82:0.589 | calc | `0.589` | numeric: a where 25% of 1-mu(0) is on | PASS |
| 82 | ch:surveys:L82:0.70 | calc | `0.70` | numeric: z where 25% is on | PASS |
| 82 | ch:surveys:L82:6.5 | calc | `6.5` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 82 |  | calc | `10` | not run: label: the fraction (10 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 82 |  | calc | `25` | not run: label: the fraction (25 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 83 | ch:surveys:L83 | calc | `0.731` | numeric: a where 50% of 1-mu(0) is on | PASS |
| 83 | ch:surveys:L83:0.37 | calc | `0.37` | numeric: z where 50% is on | PASS |
| 83 | ch:surveys:L83:4.2 | calc | `4.2` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 83 | ch:surveys:L83:0.861 | calc | `0.861` | numeric: a where 75% of 1-mu(0) is on | PASS |
| 83 | ch:surveys:L83:0.16 | calc | `0.16` | numeric: z where 75% is on | PASS |
| 83 | ch:surveys:L83:2.1 | calc | `2.1` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 83 | ch:surveys:L83:0.942 | calc | `0.942` | numeric: a where 90% of 1-mu(0) is on | PASS |
| 83 | ch:surveys:L83:0.06 | calc | `0.06` | numeric: z where 90% is on | PASS |
| 83 | ch:surveys:L83:0.9 | calc | `0.9` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 83 | ch:surveys:L83:0.971 | calc | `0.971` | numeric: a where 95% of 1-mu(0) is on | PASS |
| 83 | ch:surveys:L83:0.03 | calc | `0.03` | numeric: z where 95% is on | PASS |
| 83 |  | calc | `50` | not run: label: the fraction (50 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 83 |  | calc | `75` | not run: label: the fraction (75 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 83 |  | calc | `90` | not run: label: the fraction (90 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 83 |  | calc | `95` | not run: label: the fraction (95 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 83 |  | calc | `0.4` | not run: printed value 0.4 Gyr has one significant digit: a 5 % change cannot be told from rounding (recomputed lookback 0.428 Gyr at the 95 % point; 1.05 x 0.4 = 0.42). The neighbouring table entries with more digits are checked at ch:surveys:L82:10.9 and ch:surveys:L82:9.4 | - |
| 84 | ch:surveys:L84 | calc | `0.994` | numeric: a where 99% of 1-mu(0) is on | PASS |
| 84 | ch:surveys:L84:0.01 | calc | `0.01` | numeric: z where 99% is on | PASS |
| 84 | ch:surveys:L84:0.1 | calc | `0.1` | numeric: lookback time, Gyr (H0 67.36, Om 0.3153) | PASS |
| 84 |  | calc | `99` | not run: label: the fraction (99 %) of today's 1 - mu that defines a row of Table tab:sp_activation, an input; the a, z and lookback of that row are checked (ch:surveys:L82..L84) | - |
| 88 | eq:sp_zone | calc |  | sympy: 10-90 % zone 0.06 < z < 1.12 | PASS |
| 89 | ch:surveys:L89 | calc | `4.2` | numeric: midpoint lookback, Gyr | PASS |
| 89 | ch:surveys:L89:0.37 | calc | `0.37` | numeric: midpoint z | PASS |
| 89 |  | calc | `50` | not run: definition: the midpoint is defined as 50 % of the total modification | - |
| 90 | ch:surveys:L90 | calc | `0.229` | numeric: |dmu/dz| today | PASS |
| 91 | ch:surveys:L91 | calc | `0.109` | numeric: |dmu/dz| at z=0.5 | PASS |
| 91 | ch:surveys:L91:0.128 | calc | `0.128` | numeric: |dmu/dz| MGCAMB form at z=0 | PASS |
| 91 |  | calc | `0.5` | not run: label: redshift of a table row / bin (z = 0.5), an input of the computation, nothing to recompute | - |
| 92 | ch:surveys:L92 | calc | `2.30` | numeric: z where E = 10% of today | PASS |
| 92 | ch:surveys:L92:0.69 | calc | `0.69` | numeric: z where E = 50% of today | PASS |
| 92 | ch:surveys:L92:0.11 | calc | `0.11` | numeric: z where E = 90% of today | PASS |
| 92 |  | calc | `10` | not run: label: the 10 % level of today's E(a) whose redshift is quoted (checked by ch:surveys:L92 and its siblings) | - |
| 92 |  | calc | `50` | not run: label: the 50 % level of today's E(a) whose redshift is quoted (checked by ch:surveys:L92 and its siblings) | - |
| 92 |  | calc | `90` | not run: label: the 90 % level of today's E(a) whose redshift is quoted (checked by ch:surveys:L92 and its siblings) | - |
| 95 | ch:surveys:L95 | calc | `2.30` | numeric: z at which E(a) reaches 10 % of today | PASS |
| 95 | ch:surveys:L95:0.69 | calc | `0.69` | numeric: z at which E(a) reaches 50 % of today | PASS |
| 95 |  | calc | `10` | not run: label: the 10 % level of today's E(a) in the figure caption; its redshift is checked by ch:surveys:L95 / L95:0.69 / L96 | - |
| 95 |  | calc | `50` | not run: label: the 50 % level of today's E(a) in the figure caption; its redshift is checked by ch:surveys:L95 / L95:0.69 / L96 | - |
| 95 |  | calc | `90` | not run: label: the 90 % level of today's E(a) in the figure caption; its redshift is checked by ch:surveys:L95 / L95:0.69 / L96 | - |
| 96 | ch:surveys:L96 | calc | `0.11` | numeric: z at which E(a) reaches 90 % of today | PASS |
| 96 | ch:surveys:L96:0.06 | calc | `0.06` | numeric: transition zone of mu, lower end: 90 % of 1 - mu(0) present | PASS |
| 96 | ch:surveys:L96:1.12 | calc | `1.12` | numeric: transition zone of mu, upper end: 10 % of 1 - mu(0) present | PASS |
| 98 | ch:surveys:L98 | calc | `0.229` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 99 | ch:surveys:L99 | calc | `2.66` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 100 | ch:surveys:L100 | calc | `0.68` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 100 | ch:surveys:L100:0.3153 | calc | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 105 | ch:surveys:L105 | calc | `0.864` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 105 | ch:surveys:L105:0.865 | calc | `0.865` | numeric: MGCAMB tracking form at z = 0 | PASS |
| 106 | ch:surveys:L106 | calc | `0.027` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 106 | ch:surveys:L106:2.8 | calc | `2.8` | numeric: largest suppression of the tracking form below the exact mu, per cent of mu | PASS |
| 106 | ch:surveys:L106:0.7 | calc | `0.7` | numeric: redshift of the largest difference between the exact and tracking mu | PASS |
| 121 | eq:sp_iswT | none |  | not run: definition: the integrated Sachs-Wolfe temperature shift along the photon path (standard, Sachs & Wolfe 1967), stated in the metric of Eq. eq:sp_poisson | - |
| 125 | eq:sp_iswsrc | derived |  | sympy: Phi+Psi ~ D/a from the Poisson equation; d/dtau = a^2 H d/da gives the ISW source H D (f - 1) | PASS |
| 128 | ch:surveys:L128 | calc | `0.22` | numeric: Phi+Psi normalised at z = 3: IAM below LambdaCDM at z = 0.5, per cent | PASS |
| 129 | ch:surveys:L129 | calc | `0.78` | numeric: Phi+Psi normalised at z = 3: IAM below LambdaCDM today, per cent | PASS |
| 129 |  | calc | `0.5` | not run: label: redshift of a table row / bin (z = 0.5), an input of the computation, nothing to recompute | - |
| 135 | eq:sp_isw | calc | `1.03` | numeric: ISW amplitude ratio, uniform weight over 0.05 < z < 1.5 | PASS |
| 136 | ch:surveys:L136 | calc | `1.054` | numeric: ISW-galaxy amplitude ratio, MGCAMB tracking form, window at z = 0.3 | PASS |
| 136 | ch:surveys:L136:1.064 | calc | `1.064` | numeric: ISW-galaxy amplitude ratio, MGCAMB tracking form, window at z = 0.5 | PASS |
| 136 | ch:surveys:L136:1.072 | calc | `1.072` | numeric: ISW-galaxy amplitude ratio, MGCAMB tracking form, window at z = 0.7 | PASS |
| 141 |  | observed | `20` | not run: measured, source not named | - |
| 141 |  | observed | `30` | not run: measured, source not named | - |
| 146 | ch:surveys:L146 | calc | `1.035` | numeric: same value as p2_17_lensing_dynamics:103 (1/mu at z=0.7) | PASS |
| 146 | ch:surveys:L146:1.031 | calc | `1.031` | numeric: ISW source ratio IAM/LambdaCDM today | PASS |
| 146 | ch:surveys:L146:0.3 | calc | `0.3` | numeric: redshift of the largest ISW source ratio (exact form) | PASS |
| 147 | ch:surveys:L147:0.22 | calc | `0.22` | numeric: Phi+Psi normalised at z = 3: IAM below LambdaCDM at z = 0.5 (caption) | PASS |
| 147 |  | calc | `0.5` | not run: label: redshift of a table row / bin (z = 0.5), an input of the computation, nothing to recompute | - |
| 148 | ch:surveys:L148 | calc | `1.035` | numeric: same value as p2_17_lensing_dynamics:103 (1/mu at z=0.7) | PASS |
| 148 | ch:surveys:L148:1.034 | calc | `1.034` | numeric: ISW-galaxy amplitude ratio, exact form, LRG window at z = 0.5 | PASS |
| 148 | ch:surveys:L148:1.031 | calc | `1.031` | numeric: ISW-galaxy amplitude ratio, exact form, LRG window at z = 0.7 | PASS |
| 148 |  | calc | `0.1` | not run: input: Gaussian redshift window width sigma_z = 0.1, a setting of the ISW-galaxy computation (used in ch:surveys:L148:1.034) | - |
| 149 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153, Planck 2018 VI Table 2, restated in the figure caption | - |
| 155 | ch:surveys:L155 | calc | `0.08` | numeric: CMB lensing power lower, Limber estimate, per cent | PASS |
| 156 | ch:surveys:L156 | observed | `40` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/validation.csv`: Planck 2018 CMB lensing detection significance | PASS |
| 156 | ch:surveys:L156:2.5 | observed | `2.5` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/validation.csv`: Planck lensing amplitude error from the 40 sigma detection, per cent | PASS |
| 156 |  | observed | `30` | not run: approximate ratio: 'about 30 times smaller' is 2.5 % / 0.08 % = 31.0 (computed by ch:surveys:L156:2.5 and ch:surveys:L155); a one-figure 'about' cannot carry the 5 % control (31.0 vs 31.5) | - |
| 159 | ch:surveys:L159 | calc | `+1.8` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 159 | ch:surveys:L159:+3.6 | calc | `+3.6` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 159 |  | calc | `0.3` | not run: label: redshift of a table row / bin (z = 0.3), an input of the computation, nothing to recompute | - |
| 160 | ch:surveys:L160 | calc | `0.455` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 160 | ch:surveys:L160:0.463 | calc | `0.463` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 160 | ch:surveys:L160:7.5 | calc | `7.5` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 160 | ch:surveys:L160:0.39 | observed | `0.39` | numeric: E_G measured from SDSS luminous red galaxies, Reyes et al. 2010 | PASS |
| 160 | ch:surveys:L160:1.1 | calc | `1.1` | numeric: E_G measurement below the GR value Omega_m0/f, in sigma | PASS |
| 160 | ch:surveys:L160:1.2 | calc | `1.2` | numeric: E_G measurement below the value with the informational term, in sigma | PASS |
| 160 | ch:surveys:L160:0.008 | calc | `0.008` | numeric: predicted E_G shift at z = 0.32 | PASS |
| 164 | ch:surveys:L164 | observed | `-0.136` | heavy file `docs/verification/scripts/verify_dark_energy_far_future_surveys_book_output.txt`: measured: printed value found in verify_dark_energy_far_future_surveys_book_output.txt, a file the chapter names | PASS |
| 164 | ch:surveys:L164:0.08 | observed | `0.08` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: DES Y3 + external mu0, central value | PASS |
| 164 | ch:surveys:L164:0.19 | observed | `0.19` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: DES Y3 + external mu0, lower error | PASS |
| 165 | ch:surveys:L165 | observed | `0.11` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: DESI DR1 full shape + BAO mu0, central value | PASS |
| 165 | ch:surveys:L165:0.54 | observed | `0.54` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: DESI DR1 full shape + BAO mu0, lower error | PASS |
| 166 | ch:surveys:L166 | observed | `0.02` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: ACT + WMAP + SDSS + supernovae mu0, central value (Andrade et al. 2024) | PASS |
| 166 | ch:surveys:L166:+0.2 | fitted | `+0.2` | file `Cosmological_Physics/mgcamb_validation/chains/iam_float_mu0_r2.updated.yaml`: upper prior edge of mu0 in the free-mu0 chains | PASS |
| 166 | ch:surveys:L166:+0.059 | fitted | `+0.059` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: free-mu0 chain, Planck: median of mu0 | PASS |
| 166 |  | fitted | `90` | not run: definition: the central 90 % interval whose lower end (5 % quantile) is quoted; the median and quantiles are checked by ch:surveys:L166:+0.059 and ch:surveys:L167:-0.304 | - |
| 167 | ch:surveys:L167 | fitted | `-0.136` | heavy file `docs/verification/scripts/verify_dark_energy_far_future_surveys_book_output.txt`: measured: printed value found in verify_dark_energy_far_future_surveys_book_output.txt, a file the chapter names | PASS |
| 167 | ch:surveys:L167:-0.304 | fitted | `-0.304` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: free-mu0 chain, Planck: 5 % quantile of mu0 | PASS |
| 167 | ch:surveys:L167:+0.064 | fitted | `+0.064` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: free-mu0 chain, Planck + RSD: median of mu0 | PASS |
| 167 | ch:surveys:L167:-0.204 | fitted | `-0.204` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: free-mu0 chain, Planck + RSD: 5 % quantile of mu0 | PASS |
| 174 | ch:surveys:L174 | observed | `23` | file `docs/verification/forecasts/euclid_fisher_iam_mu/out/template_validation.json`: Euclid's published error on 1+mu0 with conservative cuts, per cent | PASS |
| 190 | ch:surveys:L190 | calc | `2.532` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 190 | ch:surveys:L190:0.39 | calc | `0.39` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 190 | ch:surveys:L190:3.640 | calc | `3.640` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 190 | ch:surveys:L190:0.27 | calc | `0.27` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 191 | ch:surveys:L191 | calc | `1.472` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 191 | ch:surveys:L191:0.68 | calc | `0.68` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 191 | ch:surveys:L191:1.754 | calc | `1.754` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 191 | ch:surveys:L191:0.57 | calc | `0.57` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 192 | ch:surveys:L192 | calc | `0.901` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 192 | ch:surveys:L192:1.11 | calc | `1.11` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 192 | ch:surveys:L192:1.295 | calc | `1.295` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 192 | ch:surveys:L192:0.77 | calc | `0.77` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 193 | ch:surveys:L193 | calc | `0.524` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 193 | ch:surveys:L193:1.91 | calc | `1.91` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 193 | ch:surveys:L193:0.624 | calc | `0.624` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 193 | ch:surveys:L193:1.60 | calc | `1.60` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 194 | ch:surveys:L194 | calc | `0.897` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 194 | ch:surveys:L194:1.12 | calc | `1.12` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 194 | ch:surveys:L194:1.295 | calc | `1.295` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 194 | ch:surveys:L194:0.77 | calc | `0.77` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 195 | ch:surveys:L195 | calc | `0.523` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 195 | ch:surveys:L195:1.91 | calc | `1.91` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 195 | ch:surveys:L195:0.624 | calc | `0.624` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 195 | ch:surveys:L195:1.60 | calc | `1.60` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 196 | ch:surveys:L196 | calc | `0.759` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 196 | ch:surveys:L196:1.32 | calc | `1.32` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 196 | ch:surveys:L196:0.866 | calc | `0.866` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 196 | ch:surveys:L196:1.15 | calc | `1.15` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 197 | ch:surveys:L197 | calc | `0.482` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 197 | ch:surveys:L197:2.07 | calc | `2.07` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 197 | ch:surveys:L197:0.550 | calc | `0.550` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 197 | ch:surveys:L197:1.82 | calc | `1.82` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 198 | ch:surveys:L198 | calc | `0.392` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma = 1 (committed forecast) | PASS |
| 198 | ch:surveys:L198:2.55 | calc | `2.55` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma = 1) | PASS |
| 198 | ch:surveys:L198:0.413 | calc | `0.413` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), Sigma_0 free (committed forecast) | PASS |
| 198 | ch:surveys:L198:2.42 | calc | `2.42` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: 1/sigma(A) from the committed sigma (Sigma_0 free) | PASS |
| 198 | ch:surveys:L198:0.2 | calc | `0.2` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: k_max of the DESI sensitivity row of the forecast | PASS |
| 205 | ch:surveys:L205 | calc | `0.76` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), full Euclid + Planck lensing + DESI, pessimistic, Sigma = 1 | PASS |
| 206 | ch:surveys:L206 | calc | `0.48` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: sigma(A), full Euclid + Planck lensing + DESI, optimistic, Sigma = 1 | PASS |
| 206 | ch:surveys:L206:0.1 | calc | `0.1` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: k_max of the DESI f sigma8 errors used in the forecast | PASS |
| 208 | ch:surveys:L208 | calc | `0.3` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: 3x2pt spectra of IAM below LambdaCDM: smallest lowering, per cent | PASS |
| 208 | ch:surveys:L208:1.1 | calc | `1.1` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: 3x2pt spectra of IAM below LambdaCDM: largest lowering, per cent | PASS |
| 214 | ch:surveys:L214 | calc | `40` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/validation.csv`: Planck lensing reconstruction noise of the forecast reproduces the 40 sigma detection | PASS |
| 216 | ch:surveys:L216 | calc | `10` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/validation.csv`: largest departure of the pipeline's weak-lensing errors from Euclid's published errors, per cent | PASS |
| 217 | ch:surveys:L217 | calc | `0.536` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/template_validation.json`: template sigma(mu0), spectroscopic clustering alone, from the pipeline | PASS |
| 217 | ch:surveys:L217:0.530 | observed | `0.530` | file `docs/verification/forecasts/euclid_fisher_iam_mu/out/template_validation.json`: published template sigma(mu0), spectroscopic clustering alone | PASS |
| 221 | ch:surveys:L221 | calc | `4.25` | numeric: f sigma8 deficit at z = 0, per cent | PASS |
| 221 | ch:surveys:L221:2.17 | calc | `2.17` | numeric: f sigma8 deficit at z = 0.3, per cent | PASS |
| 221 |  | calc | `0.3` | not run: label: redshift of a table row / bin (z = 0.3), an input of the computation, nothing to recompute | - |
| 222 | ch:surveys:L222 | calc | `1.35` | numeric: f sigma8 deficit at z = 0.5, per cent | PASS |
| 222 | ch:surveys:L222:0.41 | calc | `0.41` | numeric: f sigma8 deficit at z = 1, per cent | PASS |
| 222 | ch:surveys:L222:0.9 | calc | `0.9` | file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: Euclid spectroscopic range, lower edge (forecast binning) | PASS |
| 222 | ch:surveys:L222:1.8 | calc | `1.8` | file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: Euclid spectroscopic range, upper edge (forecast binning) | PASS |
| 222 |  | calc | `0.5` | not run: label: redshift of a table row / bin (z = 0.5), an input of the computation, nothing to recompute | - |
| 223 | ch:surveys:L223:0.3 | calc | `0.3` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: 3x2pt spectra lowered: smallest, per cent (text) | PASS |
| 223 | ch:surveys:L223:1.1 | calc | `1.1` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/METHODS.md`: 3x2pt spectra lowered: largest, per cent (text) | PASS |
| 223 |  | calc | `0.5` | not run: restates the f sigma8 deficit at the start of Euclid's spectroscopic range, z = 0.9: fs8_deficit(0.9) = 0.517 %, which rounds to the printed 0.5; a one-figure value whose 5 % control (0.525) cannot be told from 0.517, so no check is registered (the deficits around it are checked by ch:surveys:L222 and ch:surveys:L222:0.41) | - |
| 224 | ch:surveys:L224 | calc | `0.27` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: weakest significance in Table tab:sp_euclid (DR1 pessimistic, Sigma0 free) | PASS |
| 224 | ch:surveys:L224:2.55 | calc | `2.55` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: strongest significance in Table tab:sp_euclid | PASS |
| 228 | ch:surveys:L228 | calc | `0.41` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 228 | ch:surveys:L228:2.17 | calc | `2.17` | numeric: growth deficit at z = 0.3, per cent (the ramp) | PASS |
| 228 | ch:surveys:L228:4.25 | calc | `4.25` | numeric: growth deficit today, per cent (the ramp) | PASS |
| 228 |  | calc | `0.3` | not run: label: redshift of a table row / bin (z = 0.3), an input of the computation, nothing to recompute | - |
| 232 |  | observed | `0.2` | not run: measured, source not named (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 232 |  | observed | `1.5` | not run: measured, source not named (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 232 |  | observed | `2.5` | not run: measured, source not named (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 233 |  | observed | `2.6` | not run: measured, source not named (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 233 |  | observed | `2.8` | not run: measured, source not named (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 234 |  | observed | `0.3` | not run: definition: z<0.3, the redshift range where per-cent growth data are needed for the switch-on (a target range, nothing to recompute) | - |
| 234 |  | observed | `0.1` | not run: input: redshift range of the DESI DR2 full-shape growth measurement (about 0.1<z<2), a survey specification | - |
| 235 |  | observed | `0.07` | not run: input: z_eff = 0.07 of the DESI DR1 peculiar-velocity point (Qin2026), listed in Table tab:st_fsig8 (row checked at ch:sectortension:L247) | - |
| 237 |  | observed | `1.2` | not run: measured, source not named (agreement of the peculiar-velocity module with the published DESI-LSST forecast of Howlett et al.: no committed output holds it) | - |
| 239 | ch:surveys:L239 | observed | `150` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 239 | ch:surveys:L239:4.2 | calc | `4.2` | numeric: bulk-flow (f sigma8) deficit at z=0.01, per cent | PASS |
| 239 | ch:surveys:L239:3.8 | calc | `3.8` | numeric: bulk-flow (f sigma8) deficit at z=0.05, per cent | PASS |
| 239 |  | calc | `0.01` | not run: input: z=0.01, lower edge of the redshift range at which the bulk-flow deficit is evaluated | - |
| 239 |  | calc | `0.05` | not run: input: z=0.05, upper edge of the redshift range at which the bulk-flow deficit is evaluated | - |
| 239 |  | observed | `395` | not run: measured, source not named in the repository: CosmicFlows-4 bulk flow 395 +- 29 km/s (Watkins et al. 2023, doi 10.1093/mnras/stad1984); no repository file holds the value | - |
| 239 |  | observed | `139` | not run: measured, source not named in the repository: LambdaCDM bulk-flow expectation 139 km/s (Watkins et al. 2023, doi 10.1093/mnras/stad1984); no repository file holds the value | - |
| 244 | eq:sp_siren | calc | `72.26` | numeric: H0 matter = H0 photon sqrt(1+beta_m) | PASS |
| 245 | ch:surveys:L245 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 246 | ch:surveys:L246 | observed | `70.0` | heavy file `docs/verification/scripts/verify_dark_energy_far_future_surveys_book_output.txt`: measured: printed value found in verify_dark_energy_far_future_surveys_book_output.txt, a file the chapter names | PASS |
| 246 | ch:surveys:L246:8.0 | observed | `8.0` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 (Abbott 2017) lower error on H0 | PASS |
| 246 | ch:surveys:L246:68.9 | observed | `68.9` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 H0 (Hotokezaka 2019) | PASS |
| 246 | ch:surveys:L246:4.6 | observed | `4.6` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 (Hotokezaka 2019) lower error on H0 | PASS |
| 246 | ch:surveys:L246:75.46 | observed | `75.46` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 H0 (Palmese 2024) | PASS |
| 246 | ch:surveys:L246:5.39 | observed | `5.39` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 (Palmese 2024) lower error on H0 | PASS |
| 248 | ch:surveys:L248 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 248 | ch:surveys:L248:2.4 | calc | `2.4` | numeric: 3 sigma siren error as per cent of H0 matter | PASS |
| 250 | ch:surveys:L250:3.5 | calc | `3.5` | numeric: separation of the two rates at a 2 % siren H0 | PASS |
| 250 | ch:surveys:L250:7.1 | calc | `7.1` | numeric: separation of the two rates at a 1 % siren H0 | PASS |
| 250 | ch:surveys:L250:73.04 | observed | `73.04` | numeric: SH0ES H0 (Riess 2022) | PASS |
| 251 | ch:surveys:L251:0.75 | prediction | `0.75` | numeric: SH0ES offset from H0 matter, sigma | PASS |
| 251 | ch:surveys:L251:72.26 | prediction | `72.26` | numeric: H0 matter, recomputed (SH0ES comparison) | PASS |
| 254 | ch:surveys:L254:70.0 | prediction | `70.0` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 H0 (Abbott 2017), figure caption | PASS |
| 254 | ch:surveys:L254:8.0 | prediction | `8.0` | file `docs/verification/scripts/verify_iams_law_derivations.py`: GW170817 (Abbott 2017) lower error, figure caption | PASS |
| 254 | ch:surveys:L254:72.26 | prediction | `72.26` | numeric: H0 matter, recomputed (figure caption) | PASS |
| 254 | ch:surveys:L254:67.16 | prediction | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon sector, Level 2 Run A chain (figure caption) | PASS |
| 259 | ch:surveys:L259 | calc | `-1.052` | numeric: same value as p2_03_theory:882 (w_eff at z=1) | PASS |
| 259 | ch:surveys:L259:-1.015 | calc | `-1.015` | numeric: w_eff at z=3 | PASS |
| 260 | ch:surveys:L260 | calc | `-1.062` | numeric: same value as p2_03_theory:514 (tangent w0 value) | PASS |
| 260 | ch:surveys:L260:-1.063 | calc | `-1.063` | numeric: minimum of w_eff | PASS |
| 260 | ch:surveys:L260:0.19 | calc | `0.19` | numeric: redshift of the w_eff minimum | PASS |
| 260 | ch:surveys:L260:-1.012 | calc | `-1.012` | numeric: w_eff at a=10 | PASS |
| 260 |  | calc | `10` | not run: input: a=10, the evaluation point of w_eff (its value -1.012 is checked at ch:surveys:L260:-1.012) | - |
| 293 | ch:surveys:L293:-0.136 | prediction | `-0.136` | numeric: mu0 = -beta_m/(1+beta_m), falsification list | PASS |
| 293 |  | prediction | `-0.05` | not run: prediction, nothing to recompute: falsification threshold mu0 > -0.05 | - |
| 293 |  | prediction | `95` | not run: definition: 95 % confidence level of the falsification threshold | - |
| 296 |  | prediction | `0.3` | not run: definition: z<0.3, the redshift range of the falsification criterion (the 2-4 % deficit there is checked at ch:surveys:L56:-4.25 and ch:surveys:L58:-2.17) | - |
| 300 | ch:surveys:L300:72.26 | prediction | `72.26` | numeric: H0 matter, recomputed (falsification list) | PASS |
| 321 | ch:surveys:L321:-0.136 | prediction | `-0.136` | numeric: mu0 = -beta_m/(1+beta_m), status table | PASS |
| 322 | ch:surveys:L322:0.06 | calc | `0.06` | numeric: transition zone lower edge (90 % of 1-mu(0) on) | PASS |
| 326 | ch:surveys:L326:0.27 | calc | `0.27` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: Euclid DR1 lowest significance (pessimistic, Sigma0 free) | PASS |
| 326 | ch:surveys:L326:0.68 | calc | `0.68` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: Euclid DR1 highest significance (optimistic, Sigma=1) | PASS |
| 326 | ch:surveys:L326:0.77 | calc | `0.77` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: Euclid full survey lowest significance (pessimistic, Sigma0 free) | PASS |
| 326 | ch:surveys:L326:1.91 | calc | `1.91` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: Euclid full survey highest significance (optimistic, Sigma=1) | PASS |
| 326 | ch:surveys:L326:1.15 | calc | `1.15` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: with Planck lensing and DESI, lowest significance | PASS |
| 326 | ch:surveys:L326:2.55 | calc | `2.55` | heavy file `docs/verification/forecasts/euclid_fisher_iam_mu/out/results.csv`: with Planck lensing and DESI, highest significance | PASS |
| 327 |  | calc | `2.5` | not run: measured, source not named: 'about 2.5-3 sigma by the mid-2030s' (low-redshift growth-survey Fisher forecast with the peculiar-velocity module: its outputs are not committed; docs/verification/forecasts/euclid_fisher_iam_mu holds only the Euclid/Planck/DESI forecast) | - |
| 328 |  | prediction | `1000` | not run: definition: survey name KiDS-1000 (not a number) | - |
| 330 | ch:surveys:L330:72.26 | prediction | `72.26` | numeric: siren H0 matter, recomputed (status table) | PASS |
| 330 | ch:surveys:L330:3.5 | prediction | `3.5` | numeric: separation at a 2 % siren H0 (status table) | PASS |

## Part 2 - ch:lensdyn - `docs/book/part2/p2_17_lensing_dynamics.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 31 | eq:ld_b | observed |  | not run: definition: hydrostatic mass-bias parametrisation M_true = M_X/(1-b) | - |
| 34 |  | observed | `0.1` | not run: measured, source not named (approximate literature range of the hydrostatic bias b; cited papers' values are not held in any repository file) | - |
| 34 |  | observed | `0.4` | not run: measured, source not named (approximate literature range of the hydrostatic bias b; cited papers' values are not held in any repository file) | - |
| 37 |  | observed | `0.15` | not run: measured, source not named (approximate literature range of the hydrostatic bias b from simulations (Lau2009, Nelson2014); cited papers' values are not held in any repository file) | - |
| 38 | ch:lensdyn:L38:0.58 | observed | `0.58` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: 1-b needed by Planck SZ counts + primary CMB | PASS |
| 46 | ch:lensdyn:L46 | fitted | `0.7998` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 46 | ch:lensdyn:L46:+0.54 | fitted | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 chi2_min IAM (Run A) minus LCDM (Run C) | PASS |
| 53 | eq:ld_poisson | derived |  | not run: definition: mu-Sigma parametrisation of the Poisson equation (the chapter marks it '(definitions)') | - |
| 62 | eq:ld_mu | prediction |  | sympy: mu(a) = H_L^2/(H_L^2 + beta_m E H0^2) from the matter-sector rate | PASS |
| 65 | ch:lensdyn:L65 | prediction | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 77 | eq:ld_mdyn | none |  | sympy: M_dyn = mu M_true (Gauss law on the mu Poisson equation) | PASS |
| 82 | eq:ld_mlens | none |  | sympy: M_lens = Sigma M_true = M_true (Gauss law on the lensing potential) | PASS |
| 86 | eq:ld_ratio_mu | derived |  | sympy: M_lens/M_dyn = Sigma/mu = 1/mu | PASS |
| 98 | ch:lensdyn:L98 | calc | `0.864` | numeric: mu at z=0.0 | PASS |
| 98 | ch:lensdyn:L98:1.158 | calc | `1.158` | numeric: 1/mu at z=0.0 | PASS |
| 98 | ch:lensdyn:L98:15.8 | calc | `15.8` | numeric: lensing excess at z=0.0, per cent | PASS |
| 98 |  | calc | `0.0` | not run: input: redshift z=0.0 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L98) | - |
| 99 | ch:lensdyn:L99 | calc | `0.886` | numeric: mu at z=0.1 | PASS |
| 99 | ch:lensdyn:L99:1.129 | calc | `1.129` | numeric: 1/mu at z=0.1 | PASS |
| 99 | ch:lensdyn:L99:12.9 | calc | `12.9` | numeric: lensing excess at z=0.1, per cent | PASS |
| 99 |  | calc | `0.1` | not run: input: redshift z=0.1 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L99) | - |
| 100 | ch:lensdyn:L100 | calc | `0.905` | numeric: mu at z=0.2 | PASS |
| 100 | ch:lensdyn:L100:1.105 | calc | `1.105` | numeric: 1/mu at z=0.2 | PASS |
| 100 | ch:lensdyn:L100:10.5 | calc | `10.5` | numeric: lensing excess at z=0.2, per cent | PASS |
| 100 |  | calc | `0.2` | not run: input: redshift z=0.2 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L100) | - |
| 101 | ch:lensdyn:L101 | calc | `0.922` | numeric: mu at z=0.3 | PASS |
| 101 | ch:lensdyn:L101:1.085 | calc | `1.085` | numeric: 1/mu at z=0.3 | PASS |
| 101 | ch:lensdyn:L101:8.5 | calc | `8.5` | numeric: lensing excess at z=0.3, per cent | PASS |
| 101 |  | calc | `0.3` | not run: input: redshift z=0.3 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L101) | - |
| 102 | ch:lensdyn:L102 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 102 | ch:lensdyn:L102:1.055 | calc | `1.055` | numeric: 1/mu at z=0.5 | PASS |
| 102 | ch:lensdyn:L102:5.5 | calc | `5.5` | numeric: lensing excess at z=0.5, per cent | PASS |
| 102 |  | calc | `0.5` | not run: input: redshift z=0.5 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L102) | - |
| 103 | ch:lensdyn:L103 | calc | `0.966` | numeric: mu at z=0.7 | PASS |
| 103 | ch:lensdyn:L103:1.035 | calc | `1.035` | numeric: 1/mu at z=0.7 | PASS |
| 103 | ch:lensdyn:L103:3.5 | calc | `3.5` | numeric: lensing excess at z=0.7, per cent | PASS |
| 103 |  | calc | `0.7` | not run: input: redshift z=0.7 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L103) | - |
| 104 | ch:lensdyn:L104 | calc | `0.982` | numeric: mu at z=1.0 | PASS |
| 104 | ch:lensdyn:L104:1.018 | calc | `1.018` | numeric: 1/mu at z=1.0 | PASS |
| 104 | ch:lensdyn:L104:1.8 | calc | `1.8` | numeric: lensing excess at z=1.0, per cent | PASS |
| 104 |  | calc | `1.0` | not run: input: redshift z=1.0 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L104) | - |
| 105 | ch:lensdyn:L105 | calc | `0.994` | numeric: mu at z=1.5 | PASS |
| 105 | ch:lensdyn:L105:1.006 | calc | `1.006` | numeric: 1/mu at z=1.5 | PASS |
| 105 |  | calc | `1.5` | not run: input: redshift z=1.5 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L105) | - |
| 105 |  | calc | `0.6` | not run: printed value 0.6 (per cent) has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.627; 1.05 x 0.6 = 0.63). The precise value is checked at ch:lensdyn:L105:1.006 (1/mu at z = 1.5) | - |
| 106 | ch:lensdyn:L106 | calc | `0.998` | numeric: mu at z=2.0 | PASS |
| 106 | ch:lensdyn:L106:1.002 | calc | `1.002` | numeric: 1/mu at z=2.0 | PASS |
| 106 | ch:lensdyn:L106:0.2 | calc | `0.2` | numeric: lensing excess at z=2.0, per cent | PASS |
| 106 |  | calc | `2.0` | not run: input: redshift z=2.0 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L106) | - |
| 107 | ch:lensdyn:L107 | calc | `1.000` | numeric: 1/mu at z=3.0 | PASS |
| 107 | ch:lensdyn:L107:0.0 | calc | `0.0` | numeric: lensing excess at z=3.0, per cent | PASS |
| 107 |  | calc | `3.0` | not run: input: redshift z=3.0 of the row of Table tab:ld_ratio (the row's values are checked at ch:lensdyn:L107) | - |
| 112 | eq:ld_ratio | derived |  | sympy: 1/mu = 1 + beta_m E H0^2/H^2 | PASS |
| 116 | ch:lensdyn:L116 | calc | `-0.31` | numeric: dR/dz at z=0.0 | PASS |
| 116 | ch:lensdyn:L116:-0.18 | calc | `-0.18` | numeric: dR/dz at z=0.3 | PASS |
| 116 | ch:lensdyn:L116:-0.12 | calc | `-0.12` | numeric: dR/dz at z=0.5 | PASS |
| 116 |  | calc | `0.3` | not run: input: redshift z=0.3 at which the slope dR/dz is evaluated (slopes checked at ch:lensdyn:L116 ff.) | - |
| 117 | ch:lensdyn:L117 | calc | `-0.04` | numeric: dR/dz at z=1.0 | PASS |
| 117 |  | calc | `0.5` | not run: input: redshift z=0.5 at which the slope dR/dz is evaluated (slopes checked at ch:lensdyn:L116 ff.) | - |
| 121 | ch:lensdyn:L121 | measured | `-1.57` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 121 | ch:lensdyn:L121:-1.11 | measured | `-1.11` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 121 | ch:lensdyn:L121:1.105 | calc | `1.105` | numeric: same value as p2_17_lensing_dynamics:100 (1/mu at z=0.2) | PASS |
| 121 | ch:lensdyn:L121:1.055 | calc | `1.055` | numeric: same value as p2_17_lensing_dynamics:102 (1/mu at z=0.5) | PASS |
| 121 | ch:lensdyn:L121:1.018 | calc | `1.018` | numeric: same value as p2_17_lensing_dynamics:104 (1/mu at z=1.0) | PASS |
| 121 | ch:lensdyn:L121:1.002 | calc | `1.002` | numeric: same value as p2_17_lensing_dynamics:106 (1/mu at z=2.0) | PASS |
| 121 | ch:lensdyn:L121:0.8143 | measured | `0.8143` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 121 | ch:lensdyn:L121:0.8015 | measured | `0.8015` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 121 | ch:lensdyn:L121:0.8087 | measured | `0.8087` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 121 | ch:lensdyn:L121:0.7998 | measured | `0.7998` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 127 | eq:ld_slip | derived |  | sympy: Phi/Psi = (2-mu)/mu with Psi = mu Psi_GR and (Phi+Psi)/2 = Psi_GR | PASS |
| 130 | ch:lensdyn:L130 | calc | `1.315` | numeric: Phi/Psi at z=0.0 | PASS |
| 130 | ch:lensdyn:L130:1.170 | calc | `1.170` | numeric: Phi/Psi at z=0.3 | PASS |
| 130 | ch:lensdyn:L130:1.036 | calc | `1.036` | numeric: Phi/Psi at z=1.0 | PASS |
| 130 |  | calc | `0.3` | not run: input: redshift z=0.3 at which Phi/Psi is evaluated (checked at ch:lensdyn:L130:1.170) | - |
| 143 | ch:lensdyn:L143:0.08 | derived | `0.08` | numeric: CMB lensing power lower, Limber estimate, per cent | PASS |
| 149 | ch:lensdyn:L149 | calc | `1.205` | numeric: 1/(1-b), b = 0.17 | PASS |
| 149 |  | calc | `0.17` | not run: input: b = 0.17, the illustrative bias value (1/(1-b) = 1.205 is checked at ch:lensdyn:L149) | - |
| 178 | ch:lensdyn:L178 | observed | `1.158` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 183 | ch:lensdyn:L183:0.58 | observed | `0.58` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: 1-b needed by Planck SZ counts + primary CMB (data section) | PASS |
| 184 | ch:lensdyn:L184 | observed | `1.57` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 184 | ch:lensdyn:L184:0.8 | observed | `0.8` | numeric: baseline 1-b of the 2013 Planck SZ analysis | PASS |
| 185 | ch:lensdyn:L185 | calc | `0.8143` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: same value as p2_08_s8_trend:122 (LCDM chain sigma8, Planck-only) | PASS |
| 185 | ch:lensdyn:L185:1.11 | calc | `1.11` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 lower, Level 2 | PASS |
| 185 | ch:lensdyn:L185:0.8087 | calc | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_02_virial:212 (Level2 LCDM chain sigma8) | PASS |
| 186 |  | openprob | `0.15` | not run: restates the simulation range b about 0.1-0.15 of line 37 (Lau2009, Nelson2014); measured, source not named (listed in sources_needed at line 37) | - |
| 200 | ch:lensdyn:L200 | observed | `0.688` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 200 | ch:lensdyn:L200:1.45 | observed | `1.45` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 201 | ch:lensdyn:L201 | observed | `0.780` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 201 | ch:lensdyn:L201:1.28 | observed | `1.28` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 202 | ch:lensdyn:L202:0.99 | observed | `0.99` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CMB-lensing calibration 1/(1-b) | PASS |
| 203 | ch:lensdyn:L203 | observed | `1.72` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 203 | ch:lensdyn:L203:0.58 | observed | `0.58` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: 1-b needed by counts + CMB (table) | PASS |
| 204 | ch:lensdyn:L204 | observed | `1.05` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 204 | ch:lensdyn:L204:0.909 | observed | `0.909` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 204 | ch:lensdyn:L204:0.225 | observed | `0.225` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 204 | ch:lensdyn:L204:0.95 | observed | `0.95` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: LoCuSS beta_X = M_X/M_WL | PASS |
| 204 | ch:lensdyn:L204:+0.8 | observed | `+0.8` | numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: LoCuSS offset (beta_X - mu(0.225))/sigma | PASS |
| 204 |  | observed | `0.15` | not run: input: redshift bound z=0.15 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 204 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 205 | ch:lensdyn:L205 | observed | `0.909` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 205 | ch:lensdyn:L205:0.225 | observed | `0.225` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 205 | ch:lensdyn:L205:0.90 | observed | `0.90` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG reanalysed z<0.3, beta_P | PASS |
| 205 | ch:lensdyn:L205:-0.1 | observed | `-0.1` | numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG z<0.3 offset (beta_P - mu(0.225))/sigma | PASS |
| 205 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 206 | ch:lensdyn:L206 | observed | `0.936` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 206 | ch:lensdyn:L206:0.71 | observed | `0.71` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG reanalysed z>0.3, beta_P | PASS |
| 206 | ch:lensdyn:L206:-3.2 | observed | `-3.2` | numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG z>0.3 offset (beta_P - mu(0.4))/sigma | PASS |
| 206 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 206 |  | observed | `0.4` | not run: input: representative redshift z=0.4 at which mu is evaluated for the z>0.3 samples (mu(0.4) enters ch:lensdyn:L206:-3.2 and L208:-3.6) | - |
| 207 | ch:lensdyn:L207 | observed | `0.909` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 207 | ch:lensdyn:L207:0.225 | observed | `0.225` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 207 | ch:lensdyn:L207:0.96 | observed | `0.96` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP reanalysed z<0.3, beta_P | PASS |
| 207 | ch:lensdyn:L207:+0.6 | observed | `+0.6` | numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP z<0.3 offset (beta_P - mu(0.225))/sigma | PASS |
| 207 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 208 | ch:lensdyn:L208 | observed | `0.936` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 208 | ch:lensdyn:L208:0.61 | observed | `0.61` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP reanalysed z>0.3, beta_P | PASS |
| 208 | ch:lensdyn:L208:-3.6 | observed | `-3.6` | numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP z>0.3 offset (beta_P - mu(0.4))/sigma | PASS |
| 208 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 208 |  | observed | `0.4` | not run: input: representative redshift z=0.4 at which mu is evaluated for the z>0.3 samples (mu(0.4) enters ch:lensdyn:L206:-3.2 and L208:-3.6) | - |
| 209 | ch:lensdyn:L209 | observed | `1.19` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 209 | ch:lensdyn:L209:0.84 | observed | `0.84` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP + MENeaCS 1-b (Herbonnet 2020) | PASS |
| 212 | ch:lensdyn:L212 | calc | `1.07` | numeric: 1/mu at z=0.4 ('approx') | PASS |
| 212 | ch:lensdyn:L212:1.10 | calc | `1.10` | numeric: 1/mu at z=0.2 ('approx') | PASS |
| 212 |  | calc | `0.4` | not run: input: upper end z=0.4 of the sample redshifts z about 0.2-0.4 (1/mu there is checked at ch:lensdyn:L212) | - |
| 213 |  | calc | `0.15` | not run: input: redshift bound z=0.15 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 213 |  | calc | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 214 | ch:lensdyn:L214:0.96 | calc | `0.96` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: upper end of beta_P at 0.15<z<0.3 (CCCP) | PASS |
| 214 | ch:lensdyn:L214:0.95 | calc | `0.95` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: LoCuSS beta_X (text) | PASS |
| 214 | ch:lensdyn:L214:0.909 | calc | `0.909` | numeric: mu at z=0.225 (Level 1 form) | PASS |
| 215 | ch:lensdyn:L215:0.71 | observed | `0.71` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG reanalysed z>0.3 beta_P (text) | PASS |
| 215 | ch:lensdyn:L215:0.61 | observed | `0.61` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP reanalysed z>0.3 beta_P (text) | PASS |
| 215 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 223 |  | prediction | `10` | not run: prediction, nothing to recompute: 'of order 10^5' clusters from the mission planning (a count) | - |
| 231 | ch:lensdyn:L231 | calc | `10.5` | numeric: excess at z=0.2 | PASS |
| 231 |  | calc | `0.2` | not run: input: lower redshift 0.2 of the Euclid range in the caption (the 10.5 % there is checked at ch:lensdyn:L234:10.5) | - |
| 234 | ch:lensdyn:L234 | calc | `1.105` | numeric: 1/mu at z=0.2 | PASS |
| 234 | ch:lensdyn:L234:10.5 | calc | `10.5` | numeric: excess at z=0.2 | PASS |
| 234 |  | calc | `0.2` | not run: input: redshift z=0.2 of the row of Table tab:ld_euclid (its values are checked at ch:lensdyn:L234) | - |
| 235 | ch:lensdyn:L235 | calc | `1.055` | numeric: 1/mu at z=0.5 | PASS |
| 235 | ch:lensdyn:L235:5.5 | calc | `5.5` | numeric: excess at z=0.5 | PASS |
| 235 |  | calc | `0.5` | not run: input: redshift z=0.5 of the row of Table tab:ld_euclid (its values are checked at ch:lensdyn:L235) | - |
| 236 | ch:lensdyn:L236 | calc | `1.018` | numeric: 1/mu at z=1.0 | PASS |
| 236 | ch:lensdyn:L236:1.8 | calc | `1.8` | numeric: excess at z=1.0 | PASS |
| 236 |  | calc | `1.0` | not run: input: redshift z=1.0 of the row of Table tab:ld_euclid (its values are checked at ch:lensdyn:L236) | - |
| 237 | ch:lensdyn:L237 | calc | `1.006` | numeric: 1/mu at z=1.5 | PASS |
| 237 |  | calc | `1.5` | not run: input: redshift z=1.5 of the row of Table tab:ld_euclid (its values are checked at ch:lensdyn:L237) | - |
| 237 |  | calc | `0.6` | not run: printed value 0.6 (per cent) has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.627; 1.05 x 0.6 = 0.63). The precise value is checked at ch:lensdyn:L237 (1/mu = 1.006 at z = 1.5) | - |
| 238 | ch:lensdyn:L238 | calc | `1.002` | numeric: 1/mu at z=2.0 | PASS |
| 238 | ch:lensdyn:L238:0.2 | calc | `0.2` | numeric: excess at z=2.0 | PASS |
| 238 |  | calc | `2.0` | not run: input: redshift z=2.0 of the row of Table tab:ld_euclid (its values are checked at ch:lensdyn:L238) | - |
| 242 |  | prediction | `10` | not run: prediction, nothing to recompute: 'of order 10^5' clusters (a count) | - |
| 246 |  | prediction | `10` | not run: input: 'of order 10^5' clusters in the eROSITA survey planning (a count, Merloni2012) | - |
| 247 |  | prediction | `1.5` | not run: input: redshift range 0<z<1.5 of the cross-matched sample (a survey specification) | - |
| 250 |  | calc | `0.1` | not run: input: redshift range 0.1<z<2 of the test design | - |
| 258 |  | calc | `0.2` | not run: input: bin centre z=0.2 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 258 |  | calc | `0.5` | not run: input: bin centre z=0.5 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 258 |  | calc | `0.8` | not run: input: bin centre z=0.8 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 258 |  | calc | `1.2` | not run: input: bin centre z=1.2 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 259 | ch:lensdyn:L259 | calc | `2.7` | numeric: absolute error per bin (in per cent of the ratio) for chi2 = 9 against the best constant | PASS |
| 259 |  | calc | `1.8` | not run: input: bin centre z=1.8 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 263 | ch:lensdyn:L263 | calc | `-0.31` | numeric: dR/dz at z=0.0 | PASS |
| 263 | ch:lensdyn:L263:-0.18 | calc | `-0.18` | numeric: dR/dz at z=0.3 | PASS |
| 263 | ch:lensdyn:L263:2.7 | calc | `2.7` | numeric: absolute error per bin (in per cent of the ratio) for chi2 = 9 against the best constant | PASS |
| 263 |  | calc | `0.3` | not run: input: redshift z=0.3 at which the slope is quoted in the caption (dR/dz there is checked at ch:lensdyn:L263:-0.18) | - |
| 263 |  | calc | `0.2` | not run: input: bin centre z=0.2 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 263 |  | calc | `0.5` | not run: input: bin centre z=0.5 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 263 |  | calc | `0.8` | not run: input: bin centre z=0.8 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 263 |  | calc | `1.2` | not run: input: bin centre z=1.2 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 263 |  | calc | `1.8` | not run: input: bin centre z=1.8 of the five-bin test design (the 2.7 % result is checked at ch:lensdyn:L259 and L263:2.7) | - |
| 268 | ch:lensdyn:L268 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 268 | ch:lensdyn:L268:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 269 | ch:lensdyn:L269 | calc | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_02_virial:212 (Level2 LCDM chain sigma8) | PASS |
| 269 | ch:lensdyn:L269:4.25 | calc | `4.25` | numeric: f sigma8 deficit z=0 | PASS |
| 270 | ch:lensdyn:L270 | calc | `2.17` | numeric: f sigma8 deficit z=0.3 | PASS |
| 270 | ch:lensdyn:L270:1.35 | calc | `1.35` | numeric: f sigma8 deficit z=0.5 | PASS |
| 270 | ch:lensdyn:L270:0.41 | calc | `0.41` | numeric: f sigma8 deficit z=1.0 | PASS |
| 270 |  | calc | `0.3` | not run: input: redshift z=0.3 of the f sigma8 deficit list (2.17 % checked at ch:lensdyn:L270) | - |
| 270 |  | calc | `0.5` | not run: input: redshift z=0.5 of the f sigma8 deficit list (1.35 % checked at ch:lensdyn:L270:1.35) | - |
| 285 | ch:lensdyn:L285 | observed | `0.864` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 291 | ch:lensdyn:L291:1.158 | prediction | `1.158` | numeric: M_lens/M_dyn = 1/mu at z=0 (falsification list) | PASS |

## Part 2 - ch:threeway - `docs/book/part2/p2_18_three_way_clusters.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 37 | ch:threeway:L37 | observed | `20` | numeric: low end of the lensing excess over Planck masses, per cent | PASS |
| 37 | ch:threeway:L37:45 | observed | `45` | numeric: high end of the lensing excess over Planck masses, per cent | PASS |
| 38 |  | observed | `0.15` | not run: input: redshift range 0.15<z<0.3 of the LoCuSS sample (Smith2016LoCuSS, doi 10.1093/mnrasl/slv175), as in tab:ld_published; a sample boundary, nothing to recompute | - |
| 38 |  | observed | `0.3` | not run: input: redshift range 0.15<z<0.3 of the LoCuSS sample (Smith2016LoCuSS, doi 10.1093/mnrasl/slv175), as in tab:ld_published; a sample boundary, nothing to recompute | - |
| 48 | eq:tw_adot | derived |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 53 | eq:tw_E | derived |  | sympy: E(1) = 1, E -> e | PASS |
| 58 | eq:tw_beta | prediction | `0.15765` | numeric: beta_m = Omega_m/2 | PASS |
| 66 | eq:tw_mu | interp |  | sympy: mu(a): 1/(1+beta_m) today, 1 at early times | PASS |
| 70 | ch:threeway:L70 | calc | `0.8638` | numeric: same value as p1_02_iams_law:465 (mu at a=1 from beta_m) | PASS |
| 71 | ch:threeway:L71 | calc | `13.62` | numeric: same value as p1_02_iams_law:466 (percent change of coupling today) | PASS |
| 71 | ch:threeway:L71:0.78 | calc | `0.78` | numeric: linear growth deficit at z=0, per cent | PASS |
| 79 | eq:tw_mhydro | derived |  | sympy: M_hydro = mu M_true from hydrostatic balance | PASS |
| 85 | eq:tw_msz | derived |  | sympy: M_SZ = mu M_true through the Y-M calibration | PASS |
| 91 | eq:tw_mlens | derived |  | sympy: M_lens = Sigma M_true = M_true | PASS |
| 98 | eq:tw_R | derived |  | sympy: M_lens/M_hydro = 1/mu with Sigma = 1 | PASS |
| 103 | eq:tw_szx | none |  | sympy: M_SZ/M_hydro = mu/mu = 1 | PASS |
| 107 | eq:tw_order | calc |  | sympy: M_lens > M_SZ = M_hydro for mu(z) < 1, 0 <= z <= 2 | PASS |
| 116 | ch:threeway:L116 | calc | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 116 | ch:threeway:L116:0.6847 | calc | `0.6847` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 116 | ch:threeway:L116:67.36 | calc | `67.36` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 117 | ch:threeway:L117 | calc | `1.158` | numeric: same value as p2_17_lensing_dynamics:98 (1/mu at z=0.0) | PASS |
| 117 | ch:threeway:L117:1.002 | calc | `1.002` | numeric: R at z=2 | PASS |
| 117 | ch:threeway:L117:1.10 | calc | `1.10` | numeric: R at z=0.2 ('approx') | PASS |
| 117 | ch:threeway:L117:1.07 | calc | `1.07` | numeric: R at z=0.4 ('approx') | PASS |
| 117 |  | calc | `0.4` | not run: input: redshift range z ~ 0.2-0.4 of eROSITA cluster samples; R at 0.2 and 0.4 is checked by ch:threeway:L117:1.10 and ch:threeway:L117:1.07 | - |
| 118 | ch:threeway:L118 | calc | `10` | sympy: the 7-10 % range is 100(R-1) at z = 0.4 and 0.2, rounded | PASS |
| 122 | ch:threeway:L122 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 125 | ch:threeway:L125 | calc | `1.000` | numeric: a at z=0.0 | PASS |
| 125 | ch:threeway:L125:0.8638 | calc | `0.8638` | numeric: mu at z=0.0 | PASS |
| 125 | ch:threeway:L125:1.158 | calc | `1.158` | numeric: R = 1/mu at z=0.0 | PASS |
| 125 |  | calc | `0.0` | not run: input: redshift z = 0.0 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 126 | ch:threeway:L126 | calc | `0.909` | numeric: a at z=0.1 | PASS |
| 126 | ch:threeway:L126:0.8856 | calc | `0.8856` | numeric: mu at z=0.1 | PASS |
| 126 | ch:threeway:L126:1.129 | calc | `1.129` | numeric: R = 1/mu at z=0.1 | PASS |
| 126 |  | calc | `0.1` | not run: input: redshift z = 0.1 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 127 | ch:threeway:L127 | calc | `0.833` | numeric: a at z=0.2 | PASS |
| 127 | ch:threeway:L127:0.9050 | calc | `0.9050` | numeric: mu at z=0.2 | PASS |
| 127 | ch:threeway:L127:1.105 | calc | `1.105` | numeric: R = 1/mu at z=0.2 | PASS |
| 127 |  | calc | `0.2` | not run: input: redshift z = 0.2 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 128 | ch:threeway:L128 | calc | `0.769` | numeric: a at z=0.3 | PASS |
| 128 | ch:threeway:L128:0.9218 | calc | `0.9218` | numeric: mu at z=0.3 | PASS |
| 128 | ch:threeway:L128:1.085 | calc | `1.085` | numeric: R = 1/mu at z=0.3 | PASS |
| 128 |  | calc | `0.3` | not run: input: redshift z = 0.3 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 129 | ch:threeway:L129 | calc | `0.714` | numeric: a at z=0.4 | PASS |
| 129 | ch:threeway:L129:0.9362 | calc | `0.9362` | numeric: mu at z=0.4 | PASS |
| 129 | ch:threeway:L129:1.068 | calc | `1.068` | numeric: R = 1/mu at z=0.4 | PASS |
| 129 |  | calc | `0.4` | not run: input: redshift z = 0.4 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 130 | ch:threeway:L130 | calc | `0.667` | numeric: a at z=0.5 | PASS |
| 130 | ch:threeway:L130:0.9482 | calc | `0.9482` | numeric: mu at z=0.5 | PASS |
| 130 | ch:threeway:L130:1.055 | calc | `1.055` | numeric: R = 1/mu at z=0.5 | PASS |
| 130 |  | calc | `0.5` | not run: input: redshift z = 0.5 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 131 | ch:threeway:L131 | calc | `0.588` | numeric: a at z=0.7 | PASS |
| 131 | ch:threeway:L131:0.9661 | calc | `0.9661` | numeric: mu at z=0.7 | PASS |
| 131 | ch:threeway:L131:1.035 | calc | `1.035` | numeric: R = 1/mu at z=0.7 | PASS |
| 131 |  | calc | `0.7` | not run: input: redshift z = 0.7 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 132 | ch:threeway:L132 | calc | `0.500` | numeric: a at z=1.0 | PASS |
| 132 | ch:threeway:L132:0.9822 | calc | `0.9822` | numeric: mu at z=1.0 | PASS |
| 132 | ch:threeway:L132:1.018 | calc | `1.018` | numeric: R = 1/mu at z=1.0 | PASS |
| 132 |  | calc | `1.0` | not run: input: redshift z = 1.0 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 133 | ch:threeway:L133 | calc | `0.400` | numeric: a at z=1.5 | PASS |
| 133 | ch:threeway:L133:0.9938 | calc | `0.9938` | numeric: mu at z=1.5 | PASS |
| 133 | ch:threeway:L133:1.006 | calc | `1.006` | numeric: R = 1/mu at z=1.5 | PASS |
| 133 |  | calc | `1.5` | not run: input: redshift z = 1.5 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 134 | ch:threeway:L134 | calc | `0.333` | numeric: a at z=2.0 | PASS |
| 134 | ch:threeway:L134:0.9977 | calc | `0.9977` | numeric: mu at z=2.0 | PASS |
| 134 | ch:threeway:L134:1.002 | calc | `1.002` | numeric: R = 1/mu at z=2.0 | PASS |
| 134 |  | calc | `2.0` | not run: input: redshift z = 2.0 of the tab:tw_R row; the row's a, mu and R are checked (ch:threeway:L125 ff.) | - |
| 139 | ch:threeway:L139 | calc | `-0.152` | numeric: straight-line slope over the four bin centres | PASS |
| 139 | ch:threeway:L139:1.5 | calc | `1.5` | numeric: fractional error per bin for a 3 sigma slope | PASS |
| 145 | ch:threeway:L145 | calc | `-0.183` | numeric: dR/dz at z=0.3 | PASS |
| 145 | ch:threeway:L145:-0.31 | calc | `-0.31` | numeric: dR/dz at z=0 | PASS |
| 145 | ch:threeway:L145:-0.12 | calc | `-0.12` | numeric: dR/dz at z=0.5 | PASS |
| 145 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which dR/dz is evaluated (slope checked by ch:threeway:L145) | - |
| 145 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which dR/dz is evaluated (slope checked by ch:threeway:L145:-0.12) | - |
| 152 | eq:tw_bias | derived |  | sympy: 1 - b_hydro = (1 - b_NT)(1 - b_IAM); biases add at first order | PASS |
| 155 | ch:threeway:L155 | derived | `0.014` | numeric: cross term b_NT b_IAM at z=0.3 | PASS |
| 155 |  | derived | `0.3` | not run: input: redshift z = 0.3 at which the cross term is evaluated (0.014 checked by ch:threeway:L155) | - |
| 159 | ch:threeway:L159 | calc | `+0.027` | numeric: dC_NT/dz at z=0.65 | PASS |
| 159 | ch:threeway:L159:+0.036 | calc | `+0.036` | numeric: dC_NT/dz at z=0.15 | PASS |
| 159 |  | calc | `0.20` | not run: input: assumed normalisation 0.20 of the illustrative non-thermal factor C_NT = 1+0.20(1+z)^0.2 (the chapter states the form is assumed) | - |
| 159 |  | calc | `0.15` | not run: input: lowest bin centre z = 0.15 of the range over which dC_NT/dz is quoted (slope checked by ch:threeway:L159:+0.036) | - |
| 159 |  | calc | `0.65` | not run: input: highest bin centre z = 0.65 of the range over which dC_NT/dz is quoted (slope checked by ch:threeway:L159) | - |
| 160 |  | openprob | `+0.02` | not run: input: lower end of the non-thermal slope range suggested by the cited simulations (Nelson2014, ShiKomatsu2014), quoted, nothing to recompute | - |
| 160 |  | openprob | `+0.04` | not run: input: upper end of the non-thermal slope range suggested by the cited simulations (Nelson2014, ShiKomatsu2014), quoted, nothing to recompute | - |
| 161 | ch:threeway:L161 | calc | `-0.187` | numeric: slope of R C_NT at z=0.3 | PASS |
| 161 | ch:threeway:L161:-0.025 | calc | `-0.025` | numeric: slope at z=1 | PASS |
| 161 | ch:threeway:L161:1.40 | calc | `1.40` | numeric: turnover of R C_NT | PASS |
| 161 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the slope of R x C_NT is evaluated (-0.187 checked by ch:threeway:L161) | - |
| 170 | ch:threeway:L170 | calc | `1.117` | numeric: R at bin centre 0.15 | PASS |
| 170 | ch:threeway:L170:1.206 | calc | `1.206` | numeric: C_NT at 0.15 | PASS |
| 170 | ch:threeway:L170:1.346 | calc | `1.346` | numeric: R C_NT at 0.15 | PASS |
| 170 |  | calc | `0.1` | not run: input: lower edge z = 0.1 of the first bin of tab:tw_bins | - |
| 170 |  | calc | `0.2` | not run: input: upper edge z = 0.2 of the first bin of tab:tw_bins | - |
| 170 |  | calc | `0.150` | not run: definition: bin centre 0.150 = midpoint of the bin edges 0.1 and 0.2 (R, C_NT and R x C_NT at this centre are checked by ch:threeway:L170 ff.) | - |
| 171 | ch:threeway:L171 | calc | `1.094` | numeric: R at bin centre 0.25 | PASS |
| 171 | ch:threeway:L171:1.209 | calc | `1.209` | numeric: C_NT at 0.25 | PASS |
| 171 | ch:threeway:L171:1.323 | calc | `1.323` | numeric: R C_NT at 0.25 | PASS |
| 171 |  | calc | `0.2` | not run: input: lower edge z = 0.2 of the second bin of tab:tw_bins | - |
| 171 |  | calc | `0.3` | not run: input: upper edge z = 0.3 of the second bin of tab:tw_bins | - |
| 171 |  | calc | `0.250` | not run: definition: bin centre 0.250 = midpoint of the bin edges 0.2 and 0.3 (values at this centre checked by ch:threeway:L171 ff.) | - |
| 172 | ch:threeway:L172 | calc | `1.068` | numeric: R at bin centre 0.4 | PASS |
| 172 | ch:threeway:L172:1.214 | calc | `1.214` | numeric: C_NT at 0.4 | PASS |
| 172 | ch:threeway:L172:1.297 | calc | `1.297` | numeric: R C_NT at 0.4 | PASS |
| 172 |  | calc | `0.3` | not run: input: lower edge z = 0.3 of the third bin of tab:tw_bins | - |
| 172 |  | calc | `0.5` | not run: input: upper edge z = 0.5 of the third bin of tab:tw_bins | - |
| 172 |  | calc | `0.400` | not run: definition: bin centre 0.400 = midpoint of the bin edges 0.3 and 0.5 (values at this centre checked by ch:threeway:L172 ff.) | - |
| 173 | ch:threeway:L173 | calc | `1.039` | numeric: R at bin centre 0.65 | PASS |
| 173 | ch:threeway:L173:1.221 | calc | `1.221` | numeric: C_NT at 0.65 | PASS |
| 173 | ch:threeway:L173:1.269 | calc | `1.269` | numeric: R C_NT at 0.65 | PASS |
| 173 |  | calc | `0.5` | not run: input: lower edge z = 0.5 of the fourth bin of tab:tw_bins | - |
| 173 |  | calc | `0.8` | not run: input: upper edge z = 0.8 of the fourth bin of tab:tw_bins | - |
| 173 |  | calc | `0.650` | not run: definition: bin centre 0.650 = midpoint of the bin edges 0.5 and 0.8 (values at this centre checked by ch:threeway:L173 ff.) | - |
| 178 | ch:threeway:L178 | calc | `-0.183` | numeric: dR/dz at z=0.3 | PASS |
| 178 | ch:threeway:L178:+0.027 | calc | `+0.027` | numeric: dC_NT/dz at z=0.65 | PASS |
| 178 | ch:threeway:L178:+0.036 | calc | `+0.036` | numeric: dC_NT/dz at z=0.15 | PASS |
| 178 |  | calc | `0.20` | not run: input: assumed normalisation 0.20 of the illustrative C_NT (figure caption restates the form of line 159) | - |
| 178 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which dR/dz is quoted (-0.183 checked by ch:threeway:L178) | - |
| 182 | ch:threeway:L182 | calc | `-0.152` | numeric: straight-line slope over the four bin centres | PASS |
| 182 | ch:threeway:L182:1.5 | calc | `1.5` | numeric: fractional error per bin for a 3 sigma slope | PASS |
| 183 |  | calc | `0.1` | not run: input: lower end z = 0.1 of the redshift range of the proposed cross-matched sample | - |
| 183 |  | calc | `0.6` | not run: input: upper end z = 0.6 of the redshift range of the proposed cross-matched sample | - |
| 184 | ch:threeway:L184 | calc | `-0.18` | numeric: slope -0.18 at z = 0.3, restated | PASS |
| 185 |  | calc | `+0.02` | not run: input: lower end of the non-thermal slope range of the cited simulations (Nelson2014, ShiKomatsu2014), restated from line 160 | - |
| 185 |  | calc | `+0.04` | not run: input: upper end of the non-thermal slope range of the cited simulations (Nelson2014, ShiKomatsu2014), restated from line 160 | - |
| 189 | ch:threeway:L189 | calc | `1.07` | numeric: R at z=0.4 ('approx') | PASS |
| 189 | ch:threeway:L189:1.10 | calc | `1.10` | numeric: R at z=0.2 ('approx') | PASS |
| 189 | ch:threeway:L189:1.27 | calc | `1.27` | numeric: total at the last bin | PASS |
| 189 | ch:threeway:L189:1.35 | calc | `1.35` | numeric: total at the first bin | PASS |
| 190 | ch:threeway:L190 | calc | `1.28` | numeric: CCCP Planck-prior calibration 1/(1-b) | PASS |
| 190 | ch:threeway:L190:1.45 | calc | `1.45` | numeric: WtG Planck-prior calibration 1/(1-b) | PASS |
| 191 | ch:threeway:L191 | calc | `1.05` | numeric: LoCuSS like-for-like M_WL/M_X = 1/beta_X | PASS |
| 191 |  | calc | `0.15` | not run: input: redshift range 0.15<z<0.3 of the LoCuSS sample (Smith2016LoCuSS), a sample boundary | - |
| 191 |  | calc | `0.3` | not run: input: redshift range 0.15<z<0.3 of the LoCuSS sample (Smith2016LoCuSS), a sample boundary | - |
| 192 |  | openprob | `0.3` | not run: input: redshift z = 0.3 at which the cited reanalysis splits its sample (Smith2016LoCuSS), nothing to recompute | - |
| 201 | eq:tw_test1 | calibrated |  | sympy: three conditions: SZ/hydro = 1, lens/hydro > 1, slope < 0 (Level 1) | PASS |
| 211 | ch:threeway:L211 | interp | `10` | sympy: the 7-10 % range is 100(R-1) at z = 0.4 and 0.2, rounded | PASS |
| 211 |  | interp | `0.2` | not run: input: redshift range z = 0.2-0.4 of the samples (the signal there is checked by ch:threeway:L211) | - |
| 211 |  | interp | `0.4` | not run: input: redshift range z = 0.2-0.4 of the samples (the signal there is checked by ch:threeway:L211) | - |
| 212 |  | interp | `0.02` | not run: input: published shape-measurement bound |m| < 0.02 for DES Y3 quoted from the cited MacCrann2022, nothing to recompute | - |
| 220 | ch:threeway:L220 | interp | `-0.18` | numeric: Level 1 slope -0.18, restated | PASS |
| 237 | ch:threeway:L237 | prediction |  | sympy: R = M_lens/M_hydro = 1/mu (what would test it) | PASS |

## Part 2 - ch:satellites - `docs/book/part2/p2_19_missing_satellites.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 | ch:satellites:L19 | calc | `0.864` | numeric: mu today | PASS |
| 19 | ch:satellites:L19:13.62 | calc | `13.62` | numeric: same value as p1_02_iams_law:466 (percent change of coupling today) | PASS |
| 21 | ch:satellites:L21 | calc | `0.78` | numeric: D deficit today, form (i) | PASS |
| 21 | ch:satellites:L21:0.67 | calc | `0.67` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: D deficit, form (ii) (committed output) | PASS |
| 21 | ch:satellites:L21:1.87 | calc | `1.87` | heavy file `docs/verification/scripts/verify_iams_law_derivations_output.txt`: D deficit, form (iii) (committed output) | PASS |
| 21 | ch:satellites:L21:1.1 | calc | `1.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 lower, Level 2 chains | PASS |
| 21 | ch:satellites:L21:1.6 | calc | `1.6` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 lower, Level 1 chains | PASS |
| 27 |  | observed | `10` | not run: input: circular-velocity cut v_c > 10 km/s of the cited simulations (Klypin1999, Moore1999), a selection threshold, nothing to recompute | - |
| 42 | eq:ms_virial | derived |  | sympy: 2K + V = 0 gives K = |V|/2 (the display's K+V=0 reads the bound-state energy E = -K) | PASS |
| 51 | eq:ms_beta | prediction | `0.15765` | numeric: beta_m | PASS |
| 55 | ch:satellites:L55 | calc | `0.1583` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_05_dual_sector_note:113 (Omega_m/2 from Planck posterior (beta_m-fixed chain)) | PASS |
| 56 | ch:satellites:L56 | calc | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p1_03_virial_law:144 (Level2 Planck posterior Omega_m mean) | PASS |
| 61 | eq:ms_E | interp |  | sympy: E(a): E(1) = 1, E -> 0 as a -> 0, dE/da > 0 | PASS |
| 69 | eq:ms_mu | interp |  | sympy: mu(a): 1/(1+beta_m) today, 1 at early times | PASS |
| 74 | eq:ms_mu0 | interp | `0.864` | numeric: mu(0) = 1/(1+Omega_m/2) | PASS |
| 77 | ch:satellites:L77 | calc | `13.62` | numeric: same value as p1_02_iams_law:466 (percent change of coupling today) | PASS |
| 77 | ch:satellites:L77:-0.136 | calc | `-0.136` | numeric: mu0 | PASS |
| 78 | ch:satellites:L78 | calc | `0.135` | numeric: E at z=2 | PASS |
| 78 | ch:satellites:L78:0.998 | calc | `0.998` | numeric: mu at z=2 | PASS |
| 82 | eq:ms_growth | derived |  | sympy: growth equation: delta = a is the growing mode in matter domination | PASS |
| 91 | eq:ms_dD | calc | `-0.78` | numeric: Delta D/D today, form (i) | PASS |
| 101 | eq:ms_psmf | derived |  | sympy: Press-Schechter dn/dM from F = erfc(nu/sqrt2) | PASS |
| 107 | eq:ms_ps | derived |  | sympy: d ln n/d ln D = nu^2 - 1, so Delta ln n = (nu^2-1) eps | PASS |
| 113 |  | calc | `10` | not run: input: halo mass range 10^7-10^9 M_sun at which sigma_M and nu are evaluated (checked by ch:satellites:L114:7.0 ff.) | - |
| 114 | ch:satellites:L114 | calc | `0.811` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 114 | ch:satellites:L114:-0.78 | calc | `-0.78` | numeric: epsilon, form (i) | PASS |
| 114 | ch:satellites:L114:7.0 | calc | `7.0` | numeric: sigma_M at 1e7 M_sun (Eisenstein-Hu, sigma8 0.811) | PASS |
| 114 | ch:satellites:L114:4.8 | calc | `4.8` | numeric: sigma_M at 1e9 M_sun (Eisenstein-Hu, sigma8 0.811) | PASS |
| 114 | ch:satellites:L114:0.24 | calc | `0.24` | numeric: nu = delta_c/sigma_M at 1e7 M_sun | PASS |
| 114 | ch:satellites:L114:0.35 | calc | `0.35` | numeric: nu = delta_c/sigma_M at 1e9 M_sun | PASS |
| 115 | ch:satellites:L115 | calc | `+0.68` | numeric: Delta ln n at nu = 0.35, per cent | PASS |
| 115 | ch:satellites:L115:+0.73 | calc | `+0.73` | numeric: Delta ln n at nu = 0.24, per cent | PASS |
| 115 | ch:satellites:L115:0.78 | calc | `0.78` | numeric: |Delta ln n| < |eps| = 0.78 % for nu < sqrt 2 | PASS |
| 118 | ch:satellites:L118 | calc | `2.30` | numeric: ln 10 | PASS |
| 123 | ch:satellites:L123 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 123 | ch:satellites:L123:0.3153 | calc | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 123 | ch:satellites:L123:0.864 | calc | `0.864` | numeric: mu today | PASS |
| 123 | ch:satellites:L123:-0.78 | calc | `-0.78` | numeric: form (i) | PASS |
| 123 | ch:satellites:L123:-0.67 | calc | `-0.67` | numeric: Delta D/D today, form (ii) friction | PASS |
| 123 | ch:satellites:L123:-1.87 | calc | `-1.87` | numeric: Delta D/D today, form (iii) all on H_m | PASS |
| 123 | ch:satellites:L123:0.24 | calc | `0.24` | numeric: nu at 1e7 M_sun (fig caption) | PASS |
| 123 | ch:satellites:L123:0.35 | calc | `0.35` | numeric: nu at 1e9 M_sun (fig caption) | PASS |
| 123 |  | calc | `10` | not run: input: the tenfold (order-of-magnitude) satellite deficit the figure compares against; its logarithm ln 10 = 2.30 is checked by ch:satellites:L118 | - |
| 136 | ch:satellites:L136 | calc | `0.1583` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_05_dual_sector_note:113 (Omega_m/2 from Planck posterior (beta_m-fixed chain)) | PASS |
| 137 | ch:satellites:L137 | fitted | `0.7998` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 137 | ch:satellites:L137:0.802 | fitted | `0.802` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 137 | ch:satellites:L137:0.018 | fitted | `0.018` | file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: lower error of the joint weak-lensing sigma8 | PASS |
| 137 | ch:satellites:L137:0.1 | fitted | `0.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 sigma8 vs joint weak lensing, in sigma | PASS |
| 138 | ch:satellites:L138 | fitted | `67.16` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 138 | ch:satellites:L138:67.36 | fitted | `67.36` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: measured: printed value found in verify_cluster_mass_satellites_output.txt, a file the chapter names | PASS |
| 138 | ch:satellites:L138:0.37 | fitted | `0.37` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 vs Planck 2018, in sigma | PASS |
| 139 | ch:satellites:L139 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 139 | ch:satellites:L139:67.161 | calc | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_03_theory:889 (Level2 posterior mean H0) | PASS |
| 139 | ch:satellites:L139:0.75 | calc | `0.75` | numeric: H0 matter vs SH0ES | PASS |
| 139 |  | calc | `73.04` | not run: input: SH0ES H0 = 73.04 +- 1.04 (Riess2022, published), used as input by ch:satellites:L139:0.75; nothing to recompute | - |
| 140 | ch:satellites:L140 | fitted | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi^2 of Level 2 against the LCDM best fit | PASS |
| 143 | ch:satellites:L143 | calc | `0.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 0.1 sigma from joint weak lensing, restated | PASS |
| 144 | ch:satellites:L144 | interp | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Planck LCDM sigma8 in the same code (Level 2 LCDM chain) | PASS |
| 157 | ch:satellites:L157 | prediction | `-0.136` | numeric: mu0 = mu(z=0) - 1 | PASS |
| 157 |  | prediction | `0.90` | not run: prediction, nothing to recompute: tension threshold mu(z=0) > 0.90 at 2 sigma for a future measurement | - |
| 160 | ch:satellites:L160 | prediction | `4.25` | numeric: f sigma8 deficit at z = 0, form (i) | PASS |
| 160 | ch:satellites:L160:2.17 | prediction | `2.17` | numeric: f sigma8 deficit at z = 0.3, form (i) | PASS |
| 160 | ch:satellites:L160:1.35 | prediction | `1.35` | numeric: f sigma8 deficit at z = 0.5, form (i) | PASS |
| 160 | ch:satellites:L160:0.41 | prediction | `0.41` | numeric: f sigma8 deficit at z = 1, form (i) | PASS |
| 161 | ch:satellites:L161:0.7998 | prediction | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 sigma8 (chain) | PASS |
| 161 | ch:satellites:L161:0.8087 | prediction | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: LCDM sigma8 in the same code (chain) | PASS |
| 161 |  | prediction | `0.3` | not run: input: redshift z = 0.3 at which the f sigma8 deficit is quoted (checked by ch:satellites:L160:2.17) | - |
| 161 |  | prediction | `0.5` | not run: input: redshift z = 0.5 at which the f sigma8 deficit is quoted (checked by ch:satellites:L160:1.35) | - |
| 165 | ch:satellites:L165 | prediction | `-0.78` | numeric: growth deficit in D today, form (i) | PASS |
| 165 | ch:satellites:L165:-1.55 | prediction | `-1.55` | numeric: linear power deficit today, form (i) | PASS |
| 183 | ch:satellites:L183 | calc | `0.864` | numeric: mu(0) | PASS |
| 184 | ch:satellites:L184 | calc | `-0.78` | numeric: growth today (i) | PASS |
| 184 | ch:satellites:L184:-1.55 | calc | `-1.55` | numeric: linear power today (i) | PASS |
| 184 | ch:satellites:L184:-0.67 | calc | `-0.67` | numeric: growth today, form (ii) (status table) | PASS |
| 184 | ch:satellites:L184:-1.87 | calc | `-1.87` | numeric: growth today, form (iii) (status table) | PASS |
| 186 | ch:satellites:L186 | calc | `+0.68` | numeric: Delta ln n at nu=0.35 | PASS |
| 186 | ch:satellites:L186:+0.73 | calc | `+0.73` | numeric: Delta ln n at nu=0.24 | PASS |
| 186 | ch:satellites:L186:0.24 | calc | `0.24` | numeric: nu at 1e7 M_sun (status table) | PASS |
| 186 | ch:satellites:L186:0.35 | calc | `0.35` | numeric: nu at 1e9 M_sun (status table) | PASS |
| 188 | ch:satellites:L188 | prediction | `-0.136` | numeric: mu0 (status table) | PASS |

## Part 3 - ch:blackholes - `docs/book/part3/p3_01_blackholes.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 38 | ch:blackholes:L38 | none |  | sympy: S_BH from A = 16 pi G^2 M^2/c^4 | PASS |
| 42 | eq:bh_Tuniv | none |  | sympy: T = hbar kappa/(2 pi k_B c) with kappa of Schwarzschild gives T_BH | PASS |
| 61 | eq:bh_gamma_def | none |  | not run: definition: thermally limited encoding rate Gamma = P/(k_B T_BH ln2); its evaluation is checked by eq:bh_gamma | - |
| 65 | eq:bh_gamma | derived |  | sympy: Gamma = P/(k_B T ln2) = c^3/(1920 G M ln2) | PASS |
| 69 | ch:blackholes:L69 | derived | `152.5` | numeric: Gamma for 1 M_sun (CODATA 2018) | PASS |
| 73 | ch:blackholes:L73 | calc | `5120` | numeric: 5120 in tau_evap from integrating dM/dt | PASS |
| 76 |  | calc | `1.98847\times10^{30}` | not run: input: solar mass M_sun = 1.98847e30 kg as stated in the table caption (verify_book Msun); see for_author note on CODATA 2018 | - |
| 76 |  | calc | `3.15576\times10^7` | not run: definition: Julian year 3.15576e7 s (365.25 d x 86400 s), the unit of the table (verify_book yr) | - |
| 79 | ch:blackholes:L79 | calc | `6.17\times10^{-8}` | numeric: T_BH (K) for 1 M_sun (CODATA 2018) | PASS |
| 79 | ch:blackholes:L79:1.525\times10^{2} | calc | `1.525\times10^{2}` | numeric: Gamma (bits/s) for 1 M_sun (CODATA 2018) | PASS |
| 79 | ch:blackholes:L79:1.049\times10^{77} | calc | `1.049\times10^{77}` | numeric: S_BH (nats) for 1 M_sun (CODATA 2018) | PASS |
| 79 | ch:blackholes:L79:1.51\times10^{77} | calc | `1.51\times10^{77}` | numeric: S_BH (bits) for 1 M_sun (CODATA 2018) | PASS |
| 79 | ch:blackholes:L79:2.10\times10^{67} | calc | `2.10\times10^{67}` | numeric: tau_evap (yr) for 1 M_sun (CODATA 2018) | PASS |
| 80 | ch:blackholes:L80 | calc | `6.17\times10^{-9}` | numeric: T_BH (K) for 10 M_sun (CODATA 2018) | PASS |
| 80 | ch:blackholes:L80:1.525\times10^{1} | calc | `1.525\times10^{1}` | numeric: Gamma (bits/s) for 10 M_sun (CODATA 2018) | PASS |
| 80 | ch:blackholes:L80:1.049\times10^{79} | calc | `1.049\times10^{79}` | numeric: S_BH (nats) for 10 M_sun (CODATA 2018) | PASS |
| 80 | ch:blackholes:L80:1.51\times10^{79} | calc | `1.51\times10^{79}` | numeric: S_BH (bits) for 10 M_sun (CODATA 2018) | PASS |
| 80 | ch:blackholes:L80:2.10\times10^{70} | calc | `2.10\times10^{70}` | numeric: tau_evap (yr) for 10 M_sun (CODATA 2018) | PASS |
| 80 |  | calc | `10` | not run: input: table mass 10 M_sun (row label) | - |
| 81 | ch:blackholes:L81 | calc | `6.17\times10^{-14}` | numeric: T_BH (K) for 1e+06 M_sun (CODATA 2018) | PASS |
| 81 | ch:blackholes:L81:1.525\times10^{-4} | calc | `1.525\times10^{-4}` | numeric: Gamma (bits/s) for 1e+06 M_sun (CODATA 2018) | PASS |
| 81 | ch:blackholes:L81:1.049\times10^{89} | calc | `1.049\times10^{89}` | numeric: S_BH (nats) for 1e+06 M_sun (CODATA 2018) | PASS |
| 81 | ch:blackholes:L81:1.51\times10^{89} | calc | `1.51\times10^{89}` | numeric: S_BH (bits) for 1e+06 M_sun (CODATA 2018) | PASS |
| 81 | ch:blackholes:L81:2.10\times10^{85} | calc | `2.10\times10^{85}` | numeric: tau_evap (yr) for 1e+06 M_sun (CODATA 2018) | PASS |
| 81 |  | calc | `10` | not run: input: table mass 10^6 M_sun (row label) | - |
| 82 | ch:blackholes:L82 | calc | `6.17\times10^{-17}` | numeric: T_BH (K) for 1e+09 M_sun (CODATA 2018) | PASS |
| 82 | ch:blackholes:L82:1.525\times10^{-7} | calc | `1.525\times10^{-7}` | numeric: Gamma (bits/s) for 1e+09 M_sun (CODATA 2018) | PASS |
| 82 | ch:blackholes:L82:1.049\times10^{95} | calc | `1.049\times10^{95}` | numeric: S_BH (nats) for 1e+09 M_sun (CODATA 2018) | PASS |
| 82 | ch:blackholes:L82:1.51\times10^{95} | calc | `1.51\times10^{95}` | numeric: S_BH (bits) for 1e+09 M_sun (CODATA 2018) | PASS |
| 82 | ch:blackholes:L82:2.10\times10^{94} | calc | `2.10\times10^{94}` | numeric: tau_evap (yr) for 1e+09 M_sun (CODATA 2018) | PASS |
| 82 |  | calc | `10` | not run: input: table mass 10^9 M_sun (row label) | - |
| 96 | ch:blackholes:L96 | none |  | sympy: sigma_SB = pi^2 k_B^4/(60 hbar^3 c^2) from the Planck spectrum | PASS |
| 100 | eq:bh_PSB | none |  | sympy: Stefan-Boltzmann power of the horizon = Hawking power | PASS |
| 106 | ch:blackholes:L106 | derived |  | sympy: P_SB/P_Hawking = 1 | PASS |
| 122 | eq:bh_transfer_rate | none |  | sympy: transfer rate = Gamma | PASS |
| 126 | ch:blackholes:L126 | derived |  | sympy: loss of S_BH in bits per second = Gamma | PASS |
| 136 | ch:blackholes:L136:0.433 | calc | `0.433` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 136 | ch:blackholes:L136:0.218 | calc | `0.218` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 136 | ch:blackholes:L136:0.032 | calc | `0.032` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 136 | ch:blackholes:L136:67.4 | calc | `67.4` | numeric: H0 = 67.4 for the cosmic horizon (Planck 2018) | PASS |
| 136 |  | calc | `10` | not run: input: plotted ranges (1 to 10^11 M_sun; H0 up to 10^4 km/s/Mpc) in the fig:smarr caption | - |
| 136 |  | calc | `0.5` | not run: input: Kerr spin chi = 0.5 at which 0.433 is evaluated (0.433 checked by ch:blackholes:L136:0.433) | - |
| 136 |  | calc | `0.9` | not run: input: Kerr spin chi = 0.9 at which 0.218 is evaluated (0.218 checked by ch:blackholes:L136:0.218) | - |
| 136 |  | calc | `0.998` | not run: input: Kerr spin chi = 0.998 at which 0.032 is evaluated (0.032 checked by ch:blackholes:L136:0.032) | - |
| 136 |  | calc | `<2\times10^{-16}` | not run: a floating-point round-off bound, not a quantity of the equation: the Smarr ratio is exactly 1/2 (checked symbolically at eq:vl_smarr and ch:virial_law:L121); the recomputed max |Delta| = 2.22e-16 is machine epsilon (2^-52), set by the platform arithmetic, so a 5 % change of the printed 2e-16 cannot be told from round-off (see FOR_AUTHOR: the printed bound is below the recomputed value) | - |
| 139 | eq:bh_smarr | none |  | sympy: Smarr: N k_B T ln2 = T S = Mc^2/2 | PASS |
| 142 | ch:blackholes:L142 | derived | `0.5000000000` | numeric: same value as p7_11_status_all:27 (Smarr share at 6.5e9 M_sun) | PASS |
| 142 | ch:blackholes:L142:4.3\times10^6 | derived | `4.3\times10^6` | numeric: Sgr A* mass (Gillessen 2009) | PASS |
| 143 | ch:blackholes:L143 | derived | `6.5\times10^9` | numeric: M87* mass (EHT 2019) | PASS |
| 148 | ch:blackholes:L148 | derived |  | sympy: Kerr T S/(Mc^2) = sqrt(1-chi^2)/2 and its three values (G=c=hbar=k_B=1) | PASS |
| 155 | eq:bh_dMdt | none |  | sympy: dM/dt = -sigma A T^4/c^2 = -hbar c^4/(15360 pi G^2 M^2) | PASS |
| 159 | eq:bh_Mt | none |  | sympy: M(t)^3 solves dM/dt = -hbar c^4/(15360 pi G^2 M^2) | PASS |
| 163 | eq:bh_Str | none |  | sympy: Int Gamma(M(t)) dt = (S_BH,0 - S_BH(t))/(k_B ln2) | PASS |
| 168 | eq:bh_Str_closed | none |  | sympy: S_tr = S_BH,0 - S_BH(t) with S proportional to M^2 | PASS |
| 172 | ch:blackholes:L172 | none |  | sympy: half the entropy transferred at t = (1-2^-3/2) tau = 0.646 tau | PASS |
| 176 | ch:blackholes:L176 | derived | `1.51\times10^{77}` | numeric: S_BH,0 in bits for one solar mass | PASS |
| 177 | ch:blackholes:L177 | derived | `152.5` | numeric: same value as p1_02_iams_law:650 (Hawking info rate for 1 solar mass) | PASS |
| 182 | ch:blackholes:L182 | derived | `0.646` | numeric: S_tr and S_BH cross at t = 0.646 tau (fig caption) | PASS |
| 192 | ch:blackholes:L192 | derived | `0.646` | numeric: min(S_tr, S_BH) turns over at 0.646 tau | PASS |
| 194 | ch:blackholes:L194:64.6 | derived | `64.6` | numeric: black-body crossing at 64.6 % of tau | PASS |
| 194 |  | derived | `53.81` | not run: input: published value cited (Page 2013, doi 10.1088/1475-7516/2013/09/028: maximum of the fine-grained radiation entropy at 53.81 % of the evaporation time for photon and graviton emission), nothing to recompute in the book | - |
| 201 | ch:blackholes:L201 | calc | `2.6\times10^{-30}` | numeric: T_GH = hbar H0/(2 pi k_B) at the photon-sector H0 = 67.16 the caption states | PASS |
| 201 | ch:blackholes:L201:4.5\times10^{22} | calc | `4.5\times10^{22}` | numeric: mass with T_BH = T_CMB, kg | PASS |
| 201 | ch:blackholes:L201:2.3\times10^{22} | calc | `2.3\times10^{22}` | numeric: M_eq = c^3/(4 G H0), H0=67.4 | PASS |
| 201 | ch:blackholes:L201:2.1\times10^{67} | calc | `2.1\times10^{67}` | numeric: evaporation time, 1 M_sun | PASS |
| 201 | ch:blackholes:L201:5120 | calc | `5120` | numeric: 5120 in tau_evap (fig caption) | PASS |
| 201 |  | calc | `67.4` | not run: not printed at this line in the current text: the fig:bh_temperature caption gives H0 = 67.16 (checked by ch:blackholes:L201); the Planck 2018 input 67.4 is at line 209 (ch:blackholes:L209:67.4) | - |
| 201 |  | calc | `6.6\times10^{10}` | not run: input: TON 618 mass 6.6e10 M_sun, an observed value from the cited Shemmer2004 (doi 10.1086/423607), nothing to recompute | - |
| 206 | eq:bh_TGH | calc |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 209 | ch:blackholes:L209 | calc | `2.66\times10^{-30}` | numeric: T_GH, H0 = 67.4 | PASS |
| 209 | ch:blackholes:L209:2.65\times10^{-30} | calc | `2.65\times10^{-30}` | numeric: T_GH, H0 = 67.16 | PASS |
| 209 | ch:blackholes:L209:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 209 | ch:blackholes:L209:67.4 | calc | `67.4` | numeric: H0 = 67.4 (Planck 2018) | PASS |
| 213 | eq:bh_Pnet | derived |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 221 | eq:bh_Meq | derived |  | sympy: M_eq from T_BH = T_GH | PASS |
| 224 | ch:blackholes:L224 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 224 | ch:blackholes:L224:2.33\times10^{22} | calc | `2.33\times10^{22}` | numeric: M_eq at H0 = 67.16 | PASS |
| 228 |  | calc | `6.6\times10^{10}` | not run: input: TON 618 mass 6.6e10 M_sun, an observed value from the cited Shemmer2004 (doi 10.1086/423607), nothing to recompute | - |
| 228 |  | calc | `10` | not run: input: epoch z = 10^10 of the table row | - |
| 231 | ch:blackholes:L231 | calc | `2.18\times10^{-18}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 231 | ch:blackholes:L231:2.66\times10^{-30} | calc | `2.66\times10^{-30}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 231 | ch:blackholes:L231:2.32\times10^{22} | calc | `2.32\times10^{22}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 232 | ch:blackholes:L232 | calc | `3.91\times10^{-18}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 232 | ch:blackholes:L232:1.30\times10^{22} | calc | `1.30\times10^{22}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 232 | ch:blackholes:L232:4.75\times10^{-30} | calc | `4.75\times10^{-30}` | numeric: T_GH at z = 1 | PASS |
| 233 | ch:blackholes:L233 | calc | `4.41\times10^{-14}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 233 | ch:blackholes:L233:1.15\times10^{18} | calc | `1.15\times10^{18}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 233 | ch:blackholes:L233:5.37\times10^{-26} | calc | `5.37\times10^{-26}` | numeric: T_GH at z = 10^3 | PASS |
| 233 |  | calc | `10` | not run: input: epoch z = 10^3 of the table row | - |
| 234 | ch:blackholes:L234 | calc | `2.10\times10^{-8}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 234 | ch:blackholes:L234:2.42\times10^{12} | calc | `2.42\times10^{12}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 234 | ch:blackholes:L234:2.55\times10^{-20} | calc | `2.55\times10^{-20}` | numeric: T_GH at z = 10^6 | PASS |
| 234 |  | calc | `10` | not run: input: epoch z = 10^6 of the table row | - |
| 235 | ch:blackholes:L235 | calc | `2.10\times10^{0}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 235 | ch:blackholes:L235:2.42\times10^{4} | calc | `2.42\times10^{4}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 235 | ch:blackholes:L235:2.55\times10^{-12} | calc | `2.55\times10^{-12}` | numeric: T_GH at z = 10^10 | PASS |
| 235 |  | calc | `10` | not run: input: epoch z = 10^10 of the table row | - |
| 239 | ch:blackholes:L239 | calc | `3.4\times10^{-90}` | numeric: (T_GH/T_BH)^4 for one solar mass today | PASS |
| 246 | ch:blackholes:L246 | none |  | sympy: M_CMB = 4.5e22 kg = 0.6 lunar masses | PASS |
| 248 | ch:blackholes:L248 | calc | `4.4\times10^{7}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 249 | ch:blackholes:L249 | calc | `10^{67}` | numeric: evaporation times of 10^67 yr and longer | PASS |
| 262 | eq:bh_saturation | conjecture |  | not run: conjecture: saturation criterion for black-hole formation, nothing to recompute (its equivalence to the hoop conjecture is checked by eq:bh_hoop) | - |
| 271 | eq:bh_hoop | none |  | sympy: S/(k_B A) at the hoop radius = k_B/(4 l_P^2) | PASS |
| 295 | eq:bh_seed | derived |  | sympy: M from S = 4 pi G M^2/(hbar c), in units of m_Pl | PASS |
| 305 | ch:blackholes:L305 | calc | `0.98` | numeric: seed mass for S = 1e77 nats | PASS |
| 305 |  | calc | `10` | not run: input: S_collapse = 10^77 nats (table row label) | - |
| 306 | ch:blackholes:L306 | calc | `9.8\times10^{2}` | numeric: seed mass for S = 1e83 nats | PASS |
| 306 |  | calc | `10` | not run: input: S_collapse = 10^83 nats (table row label) | - |
| 307 | ch:blackholes:L307 | calc | `9.8\times10^{5}` | numeric: seed mass for S = 1e89 nats | PASS |
| 307 |  | calc | `10` | not run: input: S_collapse = 10^89 nats (table row label) | - |
| 308 | ch:blackholes:L308 | calc | `9.8\times10^{8}` | numeric: seed mass for S = 1e95 nats | PASS |
| 308 |  | calc | `10` | not run: input: S_collapse = 10^95 nats (table row label) | - |
| 316 | ch:blackholes:L316 | observed | `1.456` | heavy file `docs/verification/scripts/verify_virial_atoms_to_horizon_output.txt`: measured: printed value found in verify_virial_atoms_to_horizon_output.txt, a file the chapter names | PASS |
| 316 |  | observed | `1.44` | not run: input: Chandrasekhar mass 1.44 M_sun as conventionally quoted (Chandrasekhar1931, doi 10.1086/143324); the constants-only value 1.456 for mu_e = 2 printed beside it is checked by ch:blackholes:L316 | - |
| 318 | ch:blackholes:L318 | observed | `2.25` | numeric: TOV maximum mass (Fan et al. 2024) | PASS |
| 318 | ch:blackholes:L318:0.07 | observed | `0.07` | numeric: TOV maximum mass, lower error (Fan et al. 2024) | PASS |
| 319 | ch:blackholes:L319 | observed | `0.6` | numeric: typical white dwarf 0.6 M_sun from the DA mean 0.593 | PASS |
| 319 | ch:blackholes:L319:0.593 | observed | `0.593` | numeric: mean DA white dwarf mass (Kepler et al. 2007) | PASS |
| 319 |  | observed | `1.4` | not run: measured, source not named | - |
| 333 | ch:blackholes:L333 | observed | `2.25` | numeric: TOV maximum mass (Fan et al. 2024), restated | PASS |
| 333 | ch:blackholes:L333:0.07 | observed | `0.07` | numeric: TOV maximum mass, lower error, restated | PASS |
| 335 | ch:blackholes:L335 | calc | `0.6` | numeric: typical white dwarf 0.6 M_sun (fig caption) | PASS |
| 335 |  | calc | `1.4` | not run: measured, source not named | - |
| 336 | ch:blackholes:L336:2.40 | calc | `2.40` | numeric: A at the Chandrasekhar mass = 1.44/0.6 | PASS |
| 336 |  | calc | `1.44` | not run: input: Chandrasekhar mass 1.44 M_sun as conventionally quoted (Chandrasekhar1931), restated from line 316; the constants-only 1.456 is checked by ch:blackholes:L316 | - |
| 358 | ch:blackholes:L358 | derived | `2.1\times10^{77}` | numeric: Mahaffey number M c^2/(k_B T_BH) = 8 pi (M/m_P)^2, 1 M_sun | PASS |
| 358 | ch:blackholes:L358:3.0\times10^{77} | derived | `3.0\times10^{77}` | numeric: the same in Landauer units | PASS |
| 369 | ch:blackholes:L369 | calc | `10^{22}` | numeric: M_eq > 10^22 M_sun at z = 0 | PASS |
| 372 | ch:blackholes:L372 | prediction | `1.158` | numeric: 1/mu(z=0) = 1 + beta_m | PASS |
| 393 | ch:blackholes:L393 | none |  | sympy: eta = c^3/(4 hbar G) | PASS |

## Part 3 - ch:bekenstein - `docs/book/part3/p3_01a_bekenstein.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | ch:bekenstein:L24 | none |  | sympy: 1/4 = 2pi/8pi | PASS |
| 46 | eq:bk_unruh | none |  | sympy: Unruh T = hbar kappa/(2 pi k_B c); with kappa = c^4/4GM it lands on the Hawking temperature | PASS |
| 51 | eq:bk_SetaA | none |  | sympy: S = k_B eta A: the Clausius integral of a Schwarzschild hole gives k_B eta A with eta = c^3/(4 hbar G) | PASS |
| 57 | eq:bk_Geta | derived |  | sympy: G from matching 2pi/(hbar c eta) to the Newtonian-limit 8 pi G/c^4 equals c^3/(4 hbar eta), and agrees with inverting the Bekenstein-Hawking eta; the c-less match is shown to fail | PASS |
| 79 | eq:bk_decoherence | none |  | not run: definition: the system-environment entangling evolution of a decoherence event (schematic, no coefficient); its consequence, the diagonal reduced density matrix, is checked at eq:bk_diagonal | - |
| 84 | eq:bk_diagonal | none |  | sympy: partial trace over orthogonal environment states leaves rho_S diagonal with |c_i|^2 | PASS |
| 109 | eq:bk_SpropA | none |  | not run: definition: the area-law proportionality S propto A, carried as interpretation (no coefficient); the coefficient is checked at eq:bk_SetaA and eq:bk_structure | - |
| 122 | eq:bk_four | none |  | sympy: 4 = 8pi/2pi | PASS |
| 129 | eq:bk_rindler | none |  | sympy: Rindler metric from Minkowski by x = rho cosh(kappa t/c), cT = rho sinh(kappa t/c) | PASS |
| 134 | eq:bk_euclid | none |  | sympy: Euclidean Rindler metric from t -> -i tau; the (rho, tau) plane is flat | PASS |
| 142 | eq:bk_period | none |  | sympy: no conical deficit: circumference/(2 pi rho) = 1 fixes the period 2 pi c/kappa | PASS |
| 148 | ch:bekenstein:L148 | derived |  | sympy: Euclidean period hbar/(k_B T) = 2pi c/kappa gives the Unruh temperature | PASS |
| 161 | ch:bekenstein:L161 | derived | `2.77` | numeric: 4 ln 2 l_P^2 per bit, in units of l_P^2 | PASS |
| 162 | ch:bekenstein:L162 | derived | `4.3\times10^6` | file `docs/book/figscripts/fig_p2_bekenstein.py`: Sgr A* mass used by the figure script (solar masses) | PASS |
| 162 | ch:bekenstein:L162:6.5\times10^9 | derived | `6.5\times10^9` | file `docs/book/figscripts/fig_p2_bekenstein.py`: M87* mass used by the figure script (solar masses) | PASS |
| 163 | ch:bekenstein:L163 | derived | `67.4` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 168 | eq:bk_einstein | none |  | sympy: Einstein equation (Lambda = 0) and its trace-reversed form R_ab = kappa (T_ab - T g_ab/2) are equivalent | PASS |
| 174 | eq:bk_poisson | none |  | sympy: Poisson equation from Gauss flux: grad Phi = G M(r)/r^2 gives Laplacian Phi = 4 pi G rho_m | PASS |
| 178 | eq:bk_solid | none |  | sympy: solid angle 4 pi | PASS |
| 188 | ch:bekenstein:L188 | derived |  | sympy: R_00 for dust, Newtonian limit | PASS |
| 204 | ch:bekenstein:L204 | none |  | sympy: boost Killing vector on the past horizon is -kappa_g lambda k | PASS |
| 211 | ch:bekenstein:L211 | none |  | sympy: heat flux T_ab chi^a k^b = -kappa_g lambda T_ab k^a k^b, and its units are an energy | PASS |
| 220 | ch:bekenstein:L220 | none |  | sympy: Raychaudhuri -theta^2/2 term: the light cone from a point, theta = 2/lambda, satisfies it | PASS |
| 225 | ch:bekenstein:L225 | none |  | sympy: delta A = -int lambda R_kk: Raychaudhuri solved to first order with theta(0) = sigma = 0 | PASS |
| 230 | ch:bekenstein:L230 | none |  | sympy: Clausius: T delta S with the Unruh T at kappa = c^2 kappa_g gives hbar c kappa_g/(2 pi k_B) k_B eta delta A | PASS |
| 235 | ch:bekenstein:L235 | none |  | sympy: kappa_g cancels: T_kk = (hbar c eta/2 pi) R_kk, and g_ab k^a k^b = 0 for null k | PASS |
| 242 | ch:bekenstein:L242 | none |  | sympy: f from 0 = k grad R/2 + grad f, solved as an ODE along any path | PASS |
| 246 | ch:bekenstein:L246 | derived |  | sympy: f from the Bianchi step equals -(hbar c eta/2pi)(R/2 - Lambda); the coefficient 2pi/(hbar c eta), with eta = c^3/(4 hbar G) from the Bekenstein-Hawking area law, equals 8 pi G/c^4 from the Newtonian limit; the c-less form 2pi/(hbar eta) is shown to fail | PASS |
| 251 | eq:bk_core | derived |  | sympy: the two forms of the core identity agree | PASS |
| 261 | eq:bk_eta | derived |  | sympy: eta = 1/(4 l_P^2), l_P^2 = hbar G/c^3 | PASS |
| 271 | eq:bk_etasolve | derived |  | sympy: eta = (c^3/8piG)(2pi/hbar) = (1/4) c^3/(hbar G) | PASS |
| 278 | ch:bekenstein:L278 | calc | `1.61626\times10^{-35}` | numeric: Planck length sqrt(hbar G/c^3) in m (CODATA 2018) | PASS |
| 279 | ch:bekenstein:L279 | calc | `9.570\times10^{68}` | numeric: eta = 1/(4 l_P^2) in m^-2 (CODATA 2018) | PASS |
| 315 | eq:bk_Enat | none |  | sympy: E_nat = k_B T_H = hbar kappa/(2 pi c); for a Schwarzschild hole it is the Hawking k_B T | PASS |
| 321 | eq:bk_firstlaw | none |  | sympy: first law inverted for dA | PASS |
| 327 | eq:bk_dAmin | none |  | sympy: dA_min = 4 hbar G/c^3 = 4 l_P^2 (kappa cancels) | PASS |
| 331 | ch:bekenstein:L331 | derived | `5.56\times10^{51}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 332 | ch:bekenstein:L332 | derived | `6.5\times10^{-10}` | numeric: kappa = c H0, H0 = 67.4 (book input line 163) | PASS |
| 335 | eq:bk_onenat | calc |  | sympy: eta from the structural identity times dA_min from the first law with one Unruh quantum is 1 nat | PASS |
| 338 | ch:bekenstein:L338 | calc | `2.77` | numeric: 4 ln 2 | PASS |
| 338 | ch:bekenstein:L338:7.24\times10^{-70} | calc | `7.24\times10^{-70}` | numeric: one bit = 4 ln2 l_P^2 in m^2 | PASS |
| 345 | eq:bk_structure | none |  | sympy: hbar eta/2 pi = c^3/8 pi G has the single solution eta = 1/(4 l_P^2) | PASS |

## Part 3 - ch:bhinformation - `docs/book/part3/p3_01b_bh_information.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 161 | ch:bhinformation:L161 | openprob | `0.646` | numeric: time at which half the horizon entropy has been transferred, from the evaporation ODE | PASS |
| 225 | ch:bhinformation:L225 | derived | `0.646` | numeric: half the horizon entropy transferred at (1-2^-3/2) tau_evap | PASS |

## Part 3 - ch:saturation - `docs/book/part3/p3_07_saturation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 35 | ch:saturation:L35 | calc | `6.2\times10^{-8}` | numeric: T_H for 1 M_sun | PASS |
| 35 | ch:saturation:L35:2.1\times10^{77} | calc | `2.1\times10^{77}` | numeric: M = Mc^2/(k_B T_H) for 1 M_sun | PASS |
| 37 | ch:saturation:L37 | calc | `4.5\times10^{22}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 44 | ch:saturation:L44 | derived | `6.86` | numeric: M = hf/(k_B T), 5 GHz at 35 mK | PASS |
| 44 | ch:saturation:L44:1.05\times10^{-3} | derived | `1.05\times10^{-3}` | numeric: thermal floor 1/(1+e^M), 35 mK | PASS |
| 44 | ch:saturation:L44:16.0 | derived | `16.0` | numeric: M at 15 mK | PASS |
| 44 | ch:saturation:L44:3.41 | derived | `3.41` | file `CANON/iam_canon.json`: E_hold from the canon record | PASS |
| 44 | ch:saturation:L44:59.5 | derived | `59.5` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 44 | ch:saturation:L44:69.1 | derived | `69.1` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 44 | ch:saturation:L44:0.032 | derived | `0.032` | numeric: copy-error floor 1/(1+exp(E_hold/k_BT)) at E_hold = 3.41 k_BT | PASS |
| 44 |  | derived | `1.1\times10^{-7}` | not run: printed value 1.1e-7 has two significant digits: a 5 % change cannot be told from rounding (recomputed 1.128e-7 lies 2.6 % above 1.1e-7, so 1.05 x 1.1e-7 = 1.155e-7 is within the 2.5 % control tolerance). The precise input is checked at ch:saturation:L44:16.0 (M = hf/kT = 16.0) | - |
| 66 | ch:saturation:L66 | measured | `1.099` | file `CANON/iam_canon.json`: P_neutrophil from the canon record | PASS |
| 66 | ch:saturation:L66:0.032 | measured | `0.032` | numeric: eps0 = 1/(1+e^E_hold), E_hold = 3.41 kT (canon, measured) | PASS |
| 66 | ch:saturation:L66:3.41 | measured | `3.41` | file `CANON/iam_canon.json`: E_hold from the canon record | PASS |
| 66 | ch:saturation:L66:0.910 | derived | `0.910` | file `CANON/iam_canon.json`: 1/P | PASS |
| 66 | ch:saturation:L66:0.0362 | derived | `0.0362` | file `CANON/iam_canon.json`: eps where IAM-A = H(eps)/(P H(eps0)) = 1 | PASS |
| 66 | ch:saturation:L66:4.45 | derived | `4.45` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 66 |  | derived | `0.95` | not run: definition: lower edge 0.95 of the shaded Normal band of the gauge (a chosen band, nothing to recompute) | - |
| 66 |  | derived | `1.05` | not run: definition: upper edge 1.05 of the shaded Normal band of the gauge (a chosen band, nothing to recompute) | - |
| 71 | ch:saturation:L71 | measured | `3.41` | file `CANON/iam_canon.json`: E_hold from the canon record | PASS |

## Part 4 - ch:quantumrecords - `docs/book/part4/p4_14_quantum_records.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 33 | eq:qr_SE | derived |  | sympy: a controlled unitary takes (sum c_i s_i) x e_0 to sum c_i s_i x e_i | PASS |
| 40 | eq:qr_tauD | derived |  | sympy: tau_D from the Caldeira-Leggett decoherence rate equals tau_R (lambda_th/Delta x)^2 | PASS |
| 44 | ch:quantumrecords:L44 | calc | `3.7\times 10^{-23}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 71 | ch:quantumrecords:L71 | calc | `2.5\times10^{-87}` | numeric: hbar R/(G M^2) for a Milky Way halo, 1e12 M_sun and 200 kpc (figure caption) | PASS |
| 88 | eq:qr_virial_n | none |  | sympy: circular orbit in V = -k r^-n gives 2K/|V| = n | PASS |
| 92 | eq:qr_virial_1 | derived |  | sympy: n=1 in 2K = n|V| gives K = |V|/2 | PASS |
| 104 | eq:qr_betam | derived | `0.15765` | numeric: beta_m = Omega_m/2 = 0.15765 | PASS |
| 107 | ch:quantumrecords:L107 | derived | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 127 | eq:qr_tauhalo | none |  | sympy: hbar/|U_self| of a uniform sphere scales as hbar R/(G M^2) (order-one factor 5/3) | PASS |
| 131 | eq:qr_tauMW | calc | `2.5\times10^{-87}` | numeric: tau_D = hbar R_vir/(G M^2), 1e12 M_sun, 200 kpc (book inputs line 129) | PASS |
| 134 | ch:quantumrecords:L134 | calc | `10^{104}` | numeric: Hubble time 1/H0 over the Milky Way halo decoherence time | PASS |
| 143 | eq:qr_Idot_particle | none |  | not run: definition: the one-record-per-particle writing rate (Press-Schechter integral, no number); its exact consequence n_eff = nu_min^2 - 1 is checked at eq:qr_neff_nu | - |
| 149 | eq:qr_neff_nu | derived |  | sympy: dF/dlnD and n_eff = nu^2 - 1 with nu = delta_c/(sigma D) | PASS |
| 153 | ch:quantumrecords:L153 | derived | `2.12` | numeric: nu_min for n_eff = 7/2 | PASS |
| 154 | ch:quantumrecords:L154 | calc | `1.87` | numeric: nu_min for n_eff = 5/2 | PASS |
| 168 | eq:qr_Stot | none |  | not run: definition: the split of the horizon entropy into S_geo and S_info (interpretation, no number) | - |
| 176 | eq:qr_Sinfo | none |  | sympy: matter-era scalings give dS_info/da propto a^(n-11/2), S_info propto a^(n-9/2)/(n-9/2) | PASS |
| 181 | eq:qr_n72 | derived |  | sympy: S_info propto -1/a requires n - 9/2 = -1, n = 7/2 | PASS |
| 183 | ch:quantumrecords:L183 | calc | `-2.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope p, n=5/2, a=0.01-0.1, full LCDM growth (committed output) | PASS |
| 183 | ch:quantumrecords:L183:-1.52 | calc | `-1.52` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope p, n=3, a=0.01-0.1 (committed output) | PASS |
| 183 | ch:quantumrecords:L183:-1.02 | calc | `-1.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope p, n=7/2, a=0.01-0.1 (committed output) | PASS |
| 183 |  | calc | `0.01` | not run: input: lower end a = 0.01 of the matter-era window over which the slope p is measured (the slopes themselves are checked at ch:quantumrecords:L183) | - |
| 183 |  | calc | `0.1` | not run: input: upper end a = 0.1 of the matter-era window over which the slope p is measured (the slopes are checked at ch:quantumrecords:L183) | - |
| 183 |  | calc | `0.25` | not run: input: lower end a = 0.25 of the late window over which the slope p is measured (the slopes are checked at ch:quantumrecords:L184) | - |
| 184 | ch:quantumrecords:L184 | calc | `-2.42` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope, n=5/2, a=0.25-1 (committed output) | PASS |
| 184 | ch:quantumrecords:L184:-1.99 | calc | `-1.99` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope, n=3, a=0.25-1 (committed output) | PASS |
| 184 | ch:quantumrecords:L184:-1.57 | calc | `-1.57` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope, n=7/2, a=0.25-1 (committed output) | PASS |
| 186 | eq:qr_Idot | none |  | sympy: I_dot propto rho_m D^(7/2) f H accumulates to S_info propto -1/a in matter domination | PASS |
| 193 | ch:quantumrecords:L193 | calc | `3.3` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: z where the bottom-up exponent equals 7/2, Press-Schechter (committed output [3.29]) | PASS |
| 193 | ch:quantumrecords:L193:4.0 | calc | `4.0` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: z where the bottom-up exponent equals 7/2, Sheth-Tormen (committed output) | PASS |
| 193 | ch:quantumrecords:L193:3.2 | calc | `3.2` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: z where the bottom-up exponent equals 7/2, Tinker et al. 2008 (committed output) | PASS |
| 203 |  | calc | `0.02` | not run: measured, source not named: shift of n_eff by the halo definition (at most 0.02 at z = 4) comes from a sensitivity run of verify_bottom_up_exponent.py whose output is not committed | - |
| 203 |  | calc | `0.09` | not run: measured, source not named: shift of n_eff by the halo definition (up to 0.09 at z = 2) comes from a sensitivity run of verify_bottom_up_exponent.py whose output is not committed | - |
| 203 |  | calc | `0.77` | not run: input: lower end sigma8 = 0.77 of the range used in the sensitivity test | - |
| 203 |  | calc | `0.85` | not run: input: upper end sigma8 = 0.85 of the range used in the sensitivity test | - |
| 204 |  | calc | `5.5` | not run: about 5.5 summarises the spread of the three committed values at z = 9 (5.84, 5.31, 5.78 in verify_bottom_up_exponent_output.txt); no single value to compare, and a 5 % control cannot be told from the spread | - |
| 224 | eq:qr_Ea | interp |  | sympy: E(a) = exp(1-1/a): E(1) = 1, increasing, E -> e as a -> infinity, E -> 0 as a -> 0 | PASS |
| 235 | eq:qr_mu | derived |  | sympy: mu < 1 for beta E > 0 | PASS |
| 246 | eq:qr_Scompton | conjecture |  | sympy: S = 4 pi lambdabar_C^2/(4 l_P^2) = pi (m_P/m)^2 | PASS |
| 251 | eq:qr_ELcompton | calc |  | sympy: E_L = S k_B T_GH ln 2 | PASS |
| 254 | ch:quantumrecords:L254 | calc | `1.79\times10^{45}` | numeric: S for the electron | PASS |
| 254 | ch:quantumrecords:L254:2.54\times10^{-53} | calc | `2.54\times10^{-53}` | numeric: k_B T_GH ln2, H0 = 67.4 | PASS |
| 254 | ch:quantumrecords:L254:4.6\times10^{-8} | calc | `4.6\times10^{-8}` | numeric: E_L for the electron, J | PASS |
| 254 | ch:quantumrecords:L254:5.6\times10^5 | calc | `5.6\times10^5` | numeric: E_L / m_e c^2 | PASS |
| 254 | ch:quantumrecords:L254:67.4 | calc | `67.4` | numeric: H0 of Planck 2018 used for the bit price, 100 h | PASS |
| 257 |  | conjecture | `0.3` | not run: restates ch:electronmass:L188 (0.4 sigma(H0)/H0 = 0.32 %), printed here to one digit, too coarse for the 5 % negative control | - |
| 264 | eq:qr_GammaBH | interp |  | sympy: Hawking power over k_B T ln 2 is c^3/(1920 G M ln 2) | PASS |
| 267 | ch:quantumrecords:L267 | derived | `152.5` | numeric: same value as p1_02_iams_law:650 (Hawking info rate for 1 solar mass) | PASS |
| 268 | ch:quantumrecords:L268 | calc | `203.4` | numeric: radiation entropy rate (4/3) P/T in bits per second, one solar mass | PASS |
| 273 | eq:qr_smarr | derived |  | sympy: T_BH S_BH = Mc^2/2 | PASS |
| 283 | ch:quantumrecords:L283 | derived | `0.8638` | numeric: mu(0) = 1/(1+beta_m) | PASS |
| 284 | ch:quantumrecords:L284 | calc | `4.25` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 284 | ch:quantumrecords:L284:2.17 | calc | `2.17` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 284 | ch:quantumrecords:L284:1.35 | calc | `1.35` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 284 | ch:quantumrecords:L284:0.41 | calc | `0.41` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 285 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the f sigma8 deficit is evaluated (deficit checked at ch:quantumrecords:L284:2.17) | - |
| 285 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which the f sigma8 deficit is evaluated (deficit checked at ch:quantumrecords:L284:1.35) | - |
| 292 | ch:quantumrecords:L292 | derived | `-0.136` | numeric: mu0 = mu(1) - 1 from beta_m | PASS |
| 298 | ch:quantumrecords:L298 | calc | `4.25` | numeric: f sigma8 deficit at z=0 | PASS |
| 298 | ch:quantumrecords:L298:2.17 | calc | `2.17` | numeric: f sigma8 deficit at z=0.3 | PASS |
| 298 | ch:quantumrecords:L298:1.35 | calc | `1.35` | numeric: f sigma8 deficit at z=0.5 | PASS |
| 298 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the f sigma8 deficit is evaluated (deficit checked at ch:quantumrecords:L298:2.17) | - |
| 299 | ch:quantumrecords:L299 | calc | `0.41` | numeric: f sigma8 deficit at z=1.0 | PASS |
| 299 | ch:quantumrecords:L299:0.800 | calc | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8, Level 2 Run A | PASS |
| 299 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which the f sigma8 deficit is evaluated (deficit checked at ch:quantumrecords:L298:1.35) | - |
| 303 | eq:qr_H0 | calc | `72.26` | numeric: H0 matter = 67.16 sqrt(1.15765) | PASS |
| 306 | ch:quantumrecords:L306 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 307 | ch:quantumrecords:L307 | interp | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 from the Level 2 chain H0 times sqrt(1+beta_m) | PASS |
| 311 | ch:quantumrecords:L311 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 chi2_min IAM (runA) minus LCDM (runC) | PASS |
| 330 | ch:quantumrecords:L330 | observed | `1.3` | numeric: separation of the electron spins in the loophole-free Bell test, km | PASS |
| 339 | ch:quantumrecords:L339 | prediction | `2.2\times10^{-10}` | numeric: mass where tau_IAM = tau_PD at 10 mK, kg | PASS |
| 339 | ch:quantumrecords:L339:7.5 | prediction | `7.5` | numeric: tau_PD = hbar/E_G at 1e-12 kg, microseconds | PASS |
| 339 | ch:quantumrecords:L339:509 | prediction | `509` | numeric: tau_IAM at 1e-12 kg, 10 mK, s | PASS |
| 339 |  | prediction | `2200` | not run: input: silica density 2200 kg/m^3 of the sphere (figure caption) | - |
| 339 |  | prediction | `10` | not run: input: bath temperature 10 mK of the figure (caption) | - |
| 344 | eq:qr_tauIAM | none |  | sympy: at fixed density tau_IAM propto T^2 m^-5 and tau_PD propto m^-5/3 | PASS |
| 345 | ch:quantumrecords:L345 | calc | `509` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 345 | ch:quantumrecords:L345:7.5 | calc | `7.5` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 345 |  | calc | `10` | not run: input: bath temperature 10 mK of the worked example | - |
| 345 |  | calc | `2200` | not run: input: silica density 2200 kg/m^3 of the worked example | - |
| 347 |  | calc | `10` | not run: input: mass 10^-12 kg (a nanogram) of the worked example; the times at that mass are checked at ch:quantumrecords:L345 and L345:7.5 | - |
| 369 | ch:quantumrecords:L369 | calc | `2.5\times10^{-87}` | numeric: tau_D, Milky Way halo | PASS |
| 373 | ch:quantumrecords:L373 | prediction | `-0.136` | numeric: mu0 = 1/(1+beta_m) - 1 (status list) | PASS |
| 373 | ch:quantumrecords:L373:0.800 | prediction | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8, Level 2 Run A (status list) | PASS |
| 373 | ch:quantumrecords:L373:72.26 | prediction | `72.26` | numeric: H0 matter = H0 photon sqrt(1+beta_m) (status list) | PASS |
| 373 | ch:quantumrecords:L373:67.16 | prediction | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0, Level 2 Run A chain (status list) | PASS |

## Part 4 - ch:entanglement - `docs/book/part4/p4_21_entanglement_records.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 44 | eq:ent:smax | derived |  | sympy: Horodecki: dephased Bell state, S_max = 2 sqrt(1+c^2) | PASS |
| 52 | ch:entanglement:L52 | calc | `0.414` | numeric: sqrt2 (1+c) < 2 for c < sqrt2 - 1 | PASS |
| 53 | ch:entanglement:L53 | calc | `0.293` | numeric: isotropic: mixed fraction above 1 - 1/sqrt2 | PASS |
| 70 |  | analogy | `0.5` | not run: input: t/tau_IAM = 0.5, the first time at which the caption's points are drawn (the CHSH values there are in the table at line 60) | - |
| 83 |  | calc | `10` | not run: input: 10 mK bath temperature of the picogram row (and 10^-15 kg mass), stated inputs of the table | - |
| 85 |  | prediction | `10` | not run: input: mass 10^-12 kg named for the discriminating experiment (prediction, nothing to recompute) | - |
| 89 | ch:entanglement:L89 | calc | `2.2\times10^{-10}` | numeric: tau_IAM = tau_PD at 10 mK, silica | PASS |
| 89 |  | calc | `2200` | not run: input: silica density 2200 kg/m^3 (figure caption) | - |
| 89 |  | calc | `10` | not run: input: 10 mK bath temperature of the caption's points (and 10^-15 kg mass); the times at that temperature are checked in ch:quantumrecords | - |
| 94 | ch:entanglement:L94 | observed | `1.42` | heavy file `docs/verification/scripts/verify_entanglement_electroweak_output.txt`: measured: printed value found in verify_entanglement_electroweak_output.txt, a file the chapter names | PASS |
| 94 | ch:entanglement:L94:4.6\times10^{-25} | observed | `4.6\times10^{-25}` | numeric: top-quark lifetime hbar/Gamma_t, Gamma_t = 1.42 GeV | PASS |
| 119 |  | prediction | `10` | not run: prediction, nothing to recompute: mass 10^-12 kg near which the two times separate (the crossover is checked at ch:entanglement:L89) | - |

## Part 4 - ch:measurement - `docs/book/part4/p4_04_measurement.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 28 | ch:measurement:L28 | measured | `0.800` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 28 | ch:measurement:L28:72.26 | measured | `72.26` | heavy file `docs/verification/scripts/verify_records_measurement_time_output.txt`: measured: printed value found in verify_records_measurement_time_output.txt, a file the chapter names | PASS |
| 42 | ch:measurement:L42 | calc | `16.7` | numeric: flight time over 5 m, ns | PASS |
| 49 | eq:mp_QL | calc |  | sympy: Landauer threshold: T times the entropy of one equiprobable bit is k_B T ln 2 | PASS |
| 52 | ch:measurement:L52 | calc | `0.0179` | numeric: Q_L = k_B T ln2 at 300 K, eV | PASS |
| 70 | eq:mp_tauIAM | conjecture |  | sympy: tau_IAM = hbar (k_B T)^2 ln2/E_G^3 diverges as E_G -> 0 (photon) and is in seconds | PASS |
| 76 |  | derived | `10` | not run: input: 10^-12 kg mass of the dust grain (its density is checked at ch:measurement:L77) | - |
| 77 | ch:measurement:L77 | derived | `1.9\times10^3` | numeric: density of a 1e-12 kg sphere of R = 5 um | PASS |
| 84 |  | calc | `300` | not run: input: T = 300 K at which the table's tau_IAM is evaluated | - |
| 85 | ch:measurement:L85 | calc | `5.5\times10^{-61}` | numeric: E_G = G m^2/R, m=9.1e-31 kg, R=1e-10 m (table inputs) | PASS |
| 85 | ch:measurement:L85:1.9\times10^{26} | calc | `1.9\times10^{26}` | numeric: tau_PD = hbar/E_G, m=9.1e-31 kg | PASS |
| 85 | ch:measurement:L85:7.4\times10^{105} | calc | `7.4\times10^{105}` | numeric: tau_IAM at 300 K, m=9.1e-31 kg | PASS |
| 85 | ch:measurement:L85:9.1\times10^{-31} | calc | `9.1\times10^{-31}` | numeric: electron mass, kg (CODATA 2018) | PASS |
| 85 |  | calc | `10` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L85 | - |
| 86 | ch:measurement:L86 | calc | `1.9\times10^{-49}` | numeric: E_G = G m^2/R, m=1.2e-24 kg, R=5e-10 m (table inputs) | PASS |
| 86 | ch:measurement:L86:5.5\times10^{14} | calc | `5.5\times10^{14}` | numeric: tau_PD = hbar/E_G, m=1.2e-24 kg | PASS |
| 86 | ch:measurement:L86:1.8\times10^{71} | calc | `1.8\times10^{71}` | numeric: tau_IAM at 300 K, m=1.2e-24 kg | PASS |
| 86 | ch:measurement:L86:1.2\times10^{-24} | calc | `1.2\times10^{-24}` | numeric: C60 mass, 60 x 12 u, kg | PASS |
| 86 |  | calc | `5\times10^{-10}` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L86 | - |
| 87 | ch:measurement:L87:7.9\times10^{4} | calc | `7.9\times10^{4}` | numeric: tau_PD = hbar/E_G, m=1e-18 kg | PASS |
| 87 | ch:measurement:L87:5.3\times10^{41} | calc | `5.3\times10^{41}` | numeric: tau_IAM at 300 K, m=1e-18 kg | PASS |
| 87 |  | calc | `10` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L87 | - |
| 87 |  | calc | `5\times10^{-8}` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L87 | - |
| 87 |  | calc | `1.3\times10^{-39}` | not run: printed value 1.3e-39 J has two significant digits: a 5 % change cannot be told from rounding (recomputed G m^2/R = 1.335e-39; 1.05 x 1.3e-39 = 1.365e-39 is within the 2.5 % control tolerance). No more precise value is printed | - |
| 88 | ch:measurement:L88:0.79 | calc | `0.79` | numeric: tau_PD = hbar/E_G, m=1e-15 kg | PASS |
| 88 | ch:measurement:L88:5.3\times10^{26} | calc | `5.3\times10^{26}` | numeric: tau_IAM at 300 K, m=1e-15 kg | PASS |
| 88 |  | calc | `10` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L88 | - |
| 88 |  | calc | `5\times10^{-7}` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L88 | - |
| 88 |  | calc | `1.3\times10^{-34}` | not run: printed value 1.3e-34 J has two significant digits: a 5 % change cannot be told from rounding (recomputed G m^2/R = 1.335e-34; 1.05 x 1.3e-34 = 1.365e-34 is within the 2.5 % control tolerance). No more precise value is printed | - |
| 89 | ch:measurement:L89:7.9\times10^{-6} | calc | `7.9\times10^{-6}` | numeric: tau_PD = hbar/E_G, m=1e-12 kg | PASS |
| 89 | ch:measurement:L89:5.3\times10^{11} | calc | `5.3\times10^{11}` | numeric: tau_IAM at 300 K, m=1e-12 kg | PASS |
| 89 |  | calc | `10` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L89 | - |
| 89 |  | calc | `5\times10^{-6}` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L89 | - |
| 89 |  | calc | `1.3\times10^{-29}` | not run: printed value 1.3e-29 J has two significant digits: a 5 % change cannot be told from rounding (recomputed G m^2/R = 1.335e-29; 1.05 x 1.3e-29 = 1.365e-29 is within the 2.5 % control tolerance). No more precise value is printed | - |
| 90 | ch:measurement:L90:7.9\times10^{-16} | calc | `7.9\times10^{-16}` | numeric: tau_PD = hbar/E_G, m=1e-06 kg | PASS |
| 90 | ch:measurement:L90:5.3\times10^{-19} | calc | `5.3\times10^{-19}` | numeric: tau_IAM at 300 K, m=1e-06 kg | PASS |
| 90 |  | calc | `10` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L90 | - |
| 90 |  | calc | `5\times10^{-4}` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L90 | - |
| 90 |  | calc | `1.3\times10^{-19}` | not run: printed value 1.3e-19 J has two significant digits: a 5 % change cannot be told from rounding (recomputed G m^2/R = 1.335e-19; 1.05 x 1.3e-19 = 1.365e-19 is within the 2.5 % control tolerance). No more precise value is printed | - |
| 91 | ch:measurement:L91 | calc | `7.1\times10^{-9}` | numeric: E_G = G m^2/R, m=4 kg, R=0.15 m (table inputs) | PASS |
| 91 | ch:measurement:L91:1.5\times10^{-26} | calc | `1.5\times10^{-26}` | numeric: tau_PD = hbar/E_G, m=4 kg | PASS |
| 91 | ch:measurement:L91:3.5\times10^{-51} | calc | `3.5\times10^{-51}` | numeric: tau_IAM at 300 K, m=4 kg | PASS |
| 91 |  | calc | `4.0` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L91 | - |
| 91 |  | calc | `0.15` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L91 | - |
| 92 | ch:measurement:L92 | calc | `1.1\times10^{-6}` | numeric: E_G = G m^2/R, m=70 kg, R=0.3 m (table inputs) | PASS |
| 92 | ch:measurement:L92:9.7\times10^{-29} | calc | `9.7\times10^{-29}` | numeric: tau_PD = hbar/E_G, m=70 kg | PASS |
| 92 | ch:measurement:L92:9.7\times10^{-58} | calc | `9.7\times10^{-58}` | numeric: tau_IAM at 300 K, m=70 kg | PASS |
| 92 |  | calc | `70` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L92 | - |
| 92 |  | calc | `0.30` | not run: input: table entry of the system list (mass or radius of the object, chosen size), nothing to recompute; the E_G and times of the row are checked at ch:measurement:L92 | - |
| 99 |  | calc | `10` | not run: definition: the mesoscopic range 10^-15 to 10^-10 kg shaded in the figure | - |
| 100 | ch:measurement:L100 | calc | `2.2\times10^{-10}` | numeric: crossing mass at 10 mK, silica | PASS |
| 117 | ch:measurement:L117 | calc | `16.7` | numeric: flight time over 5 m, ns | PASS |
| 133 | ch:measurement:L133 | observed | `1.3` | numeric: separation of the NV-centre electron spins entangled through emitted photons, km | PASS |
| 139 | eq:F | calc |  | sympy: F = 1 - e^-1 exp(1 - Q_L/Q) = 1 - exp(-Q_L/Q) | PASS |
| 144 |  | conjecture | `0.04` | not run: input: threshold Q/Q_L < 0.04 chosen to define reversible markers (1/0.04 = 25 is the exponent in e^-25) | - |
| 145 | ch:measurement:L145 | calc | `<0.01` | numeric: F at Q/Q_L = 100 | PASS |
| 145 | ch:measurement:L145:0.632 | calc | `0.632` | numeric: F at Q = Q_L | PASS |
| 145 |  | calc | `100` | not run: input: threshold Q/Q_L > 100 chosen to define irreversible markers; F there is checked at ch:measurement:L145 | - |
| 150 | ch:measurement:L150 | conjecture | `0.0179` | numeric: Q_L = k_B T ln 2 at 300 K, eV (caption) | PASS |
| 150 | ch:measurement:L150:<0.01 | conjecture | `<0.01` | numeric: F = 1 - exp(-Q_L/Q) for a retinal rod (140) and a CCD pixel (167): below 0.01 | PASS |
| 150 | ch:measurement:L150:0.632 | conjecture | `0.632` | numeric: F at Q = Q_L (caption) | PASS |
| 150 |  | conjecture | `0.04` | not run: input: threshold Q/Q_L < 0.04 for reversible markers (caption) | - |
| 164 | ch:measurement:L164 | prediction | `0.632` | numeric: F at Q = Q_L (checkbox) | PASS |
| 169 | ch:measurement:L169 | calc | `0.000` | numeric: F, Q = 0.1 eV, 10 mK | PASS |
| 169 | ch:measurement:L169:0.0024 | calc | `0.0024` | numeric: F, Q = 0.1 eV, 4 K | PASS |
| 169 | ch:measurement:L169:0.164 | calc | `0.164` | numeric: F, Q = 0.1 eV, 300 K | PASS |
| 169 |  | calc | `0.1` | not run: input: Q = 0.1 eV dissipated in the which-path interaction of the temperature test | - |
| 175 | ch:measurement:L175 | conjecture | `6.0\times10^{-7}` | numeric: Q_L = k_B T ln 2 at 10 mK, eV | PASS |
| 175 | ch:measurement:L175:0.0179 | conjecture | `0.0179` | numeric: Q_L = k_B T ln 2 at 300 K, eV (caption, panel a) | PASS |
| 175 | ch:measurement:L175:0.000 | conjecture | `0.000` | numeric: F at Q = 0.1 eV, 10 mK | PASS |
| 175 | ch:measurement:L175:0.002 | conjecture | `0.002` | numeric: F at Q = 0.1 eV, 4 K | PASS |
| 175 | ch:measurement:L175:0.164 | conjecture | `0.164` | numeric: F at Q = 0.1 eV, 300 K | PASS |
| 175 |  | conjecture | `0.1` | not run: input: Q = 0.1 eV of panel b | - |
| 180 | ch:measurement:L180 | calc | `1.6\times10^7` | numeric: Planck time over tau_IAM(cat) | PASS |
| 181 | ch:measurement:L181 | calc | `5.4\times10^{-44}` | numeric: Planck time | PASS |
| 195 | eq:mp_smax | derived |  | sympy: Horodecki: dephased Bell state, S_max = 2 sqrt(1+c^2) | PASS |
| 199 | ch:measurement:L199 | calc | `0.586` | numeric: sqrt2 (1+c) = 2 at c = 1-D: D = 2 - sqrt2 | PASS |
| 199 | ch:measurement:L199:0.293 | calc | `0.293` | numeric: 2 sqrt2 (1-D) = 2: D = 1 - 1/sqrt2 | PASS |
| 201 | ch:measurement:L201 | observed | `2.42` | heavy file `docs/verification/scripts/verify_records_measurement_time_output.txt`: measured: printed value found in verify_records_measurement_time_output.txt, a file the chapter names | PASS |
| 201 | ch:measurement:L201:1.3 | observed | `1.3` | numeric: separation of the electron spins in the loophole-free Bell test, km | PASS |
| 209 | ch:measurement:L209 | calc | `2\times10^3` | numeric: 1,000 photons of 2 eV | PASS |
| 209 | ch:measurement:L209:1.1\times10^5 | calc | `1.1\times10^5` | numeric: 2e3 eV over Q_L at 300 K | PASS |
| 210 | ch:measurement:L210 | calc | `111612` | numeric: bits written: 2e3 eV / (k_B T ln2) at 300 K | PASS |
| 210 | ch:measurement:L210:33599 | calc | `33599` | numeric: log10 of 2^111612 | PASS |
| 224 | ch:measurement:L224 | interp | `13.8` | numeric: age of the universe, flat LCDM with Planck 2018 parameters, Gyr | PASS |
| 224 |  | interp | `10` | not run: measured, source not named: about 10^11 galaxies, an order-of-magnitude count stated without a citation | - |
| 256 | ch:measurement:L256 | observed | `0.0179` | heavy file `docs/verification/scripts/verify_measurement_problem_output.txt`: measured: printed value found in verify_measurement_problem_output.txt, a file the chapter names | PASS |

## Part 4 - ch:gravdec - `docs/book/part4/p4_05_gravdec.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 26 | ch:gravdec:L26 | derived | `0.15765` | numeric: beta_m | PASS |
| 27 | ch:gravdec:L27 | derived | `-0.136` | numeric: mu0 = mu(1) - 1 | PASS |
| 33 | ch:gravdec:L33 | measured | `0.8087` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 33 | ch:gravdec:L33:0.7998 | measured | `0.7998` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 33 | ch:gravdec:L33:0.822 | measured | `0.822` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 33 | ch:gravdec:L33:72.26 | measured | `72.26` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 33 | ch:gravdec:L33:+0.54 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 chi2_min IAM (runA) minus LCDM (runC) | PASS |
| 34 | ch:gravdec:L34 | calc | `72.26` | numeric: H0 matter sector | PASS |
| 34 | ch:gravdec:L34:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 34 | ch:gravdec:L34:0.75 | calc | `0.75` | numeric: H0 matter vs SH0ES, sigma | PASS |
| 34 | ch:gravdec:L34:0.37 | calc | `0.37` | numeric: H0 photon vs Planck, sigma | PASS |
| 39 | ch:gravdec:L39 | measured | `0.0100` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 39 | ch:gravdec:L39:0.0068 | measured | `0.0068` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: final R-1 of the second Level 2b chain (runD) | PASS |
| 40 | ch:gravdec:L40 | measured | `61.45` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 40 | ch:gravdec:L40:61.52 | measured | `61.52` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 40 | ch:gravdec:L40:67.4 | measured | `67.4` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: measured: printed value found in verify_virial_papers_output.txt, a file the chapter names | PASS |
| 53 | eq:gd_dp | none |  | not run: definition: the Diosi-Penrose proposal tau_DP = hbar/E_G with E_G = G m^2/R up to an order-one factor (published form, no coefficient to recompute); its numbers are checked at ch:gravdec:L83:7.5 and L83:1.4e-29 | - |
| 63 | eq:gd_bitrate | conjecture |  | not run: definition: assumed bit rate E_G^2/(hbar k_B T ln 2) (conjecture, taken as given in the text); its consequence tau_IAM is checked at eq:tauIAM | - |
| 69 | eq:tauIAM | derived |  | sympy: integral of Gamma_info/S_boundary is linear in t; its time constant is hbar k_B^2 T^2 ln2/E_G^3, in seconds | PASS |
| 76 | eq:gd_ramp | conjecture |  | sympy: ramp C = 1 - E_q(eta)/e: C(0+) = 1 and the rate -dC/d eta peaks at eta = 1/2 | PASS |
| 83 | ch:gravdec:L83:4.8 | prediction | `4.8` | numeric: radius of a 1e-12 kg silica sphere, um | PASS |
| 83 | ch:gravdec:L83:1.4\times10^{-29} | prediction | `1.4\times10^{-29}` | numeric: E_G = G m^2/R of a 1e-12 kg silica sphere, J | PASS |
| 83 | ch:gravdec:L83:7.5 | prediction | `7.5` | numeric: tau_DP = hbar/E_G of a 1e-12 kg silica sphere, us | PASS |
| 83 |  | prediction | `10` | not run: input: 10 mK bath temperature of the caption | - |
| 83 |  | prediction | `2200` | not run: input: silica density 2200 kg/m^3 (caption) | - |
| 85 | ch:gravdec:L85 | calc | `509` | numeric: tau_IAM, 1e-12 kg silica, 10 mK | PASS |
| 85 |  | calc | `10` | not run: input: 10^-12 kg nanosphere at 10 mK (worked example; tau_IAM = 509 s checked at ch:gravdec:L85) | - |
| 93 | ch:gravdec:L93 | prediction | `2.2\times10^{-10}` | numeric: mass at which tau_IAM = tau_DP at 10 mK, silica | PASS |
| 93 |  | prediction | `2200` | not run: input: silica density 2200 kg/m^3 (caption) | - |
| 97 | ch:gravdec:L97 | calc | `5\times10^{17}` | numeric: tau_IAM, 1e-15 kg, 10 mK | PASS |
| 97 |  | calc | `10` | not run: input: masses 10^-15 kg and temperature 10 mK at which tau_IAM is quoted (the times are checked at ch:gravdec:L97 and L85) | - |
| 98 | ch:gravdec:L98 | calc | `5\times10^{-8}` | numeric: tau_IAM, 1e-10 kg, 10 mK | PASS |
| 98 | ch:gravdec:L98:3.5\times10^{-12} | calc | `3.5\times10^{-12}` | numeric: mass where tau_IAM = 1 s at 10 mK | PASS |
| 98 |  | calc | `10` | not run: input: masses 10^-12 and 10^-10 kg at which tau_IAM is quoted (times checked at ch:gravdec:L98) | - |
| 99 | ch:gravdec:L99 | calc | `1.4\times10^{-11}` | numeric: mass where tau_IAM = 1 ms at 10 mK | PASS |
| 100 |  | calc | `10` | not run: restates the testable mass range 10^-12-10^-11 kg bounded by ch:gravdec:L98:3.5e-12 (1 s) and ch:gravdec:L99 (1.4e-11 kg, 1 ms) | - |
| 108 | ch:gravdec:L108 | calc | `3.33` | numeric: exponent difference m^-5 vs m^-5/3 | PASS |
| 115 | eq:gd_heating | calc |  | sympy: P_IAM = k_B T ln2 / tau_IAM = E_G^3/(hbar k_B T) | PASS |
| 119 | ch:gravdec:L119 | calc | `1.9\times10^{-28}` | numeric: P_IAM, 1e-12 kg, 10 mK, W | PASS |
| 119 | ch:gravdec:L119:2.8 | calc | `2.8` | numeric: phonon rate E_G^3/(hbar^2 k_B T omega0), 1e-12 kg, 10 mK, 100 kHz | PASS |
| 119 |  | calc | `10` | not run: input: 10^-12 kg sphere at 10 mK for which the heating numbers are computed (checked at ch:gravdec:L119 and ch:gravdec:L119:2.8) | - |
| 120 | ch:gravdec:L120 | calc | `0.8\times10^{-12}` | numeric: mass at which the phonon rate is 1 per second at 10 mK, 100 kHz | PASS |
| 121 | ch:gravdec:L121 | openprob | `1.9\times10^{-24}` | numeric: bit rate priced at k_B T ln2: E_G^2/hbar, 1e-12 kg silica, W | PASS |
| 130 | eq:gd_gamma | none |  | sympy: Gamma = dE_q/d eta = eta^-2 exp(1 - 1/eta), and its integral from 0 is E_q | PASS |
| 134 | eq:gd_lindblad | conjecture |  | sympy: position dephasing L = x heats: d<n>/d eta = Gamma/2 for any state | PASS |
| 143 | ch:gravdec:L143 | calc | `0.225` | numeric: eta of the largest purity difference (ramp vs constant rate) | PASS |
| 143 | ch:gravdec:L143:0.139 | calc | `0.139` | numeric: largest purity difference | PASS |
| 147 | ch:gravdec:L147 | calc | `0.5` | numeric: edge of the protected regime: the ramp rate peaks at eta = 0.5 (numerical maximum) | PASS |
| 148 | ch:gravdec:L148 | calc | `0.5` | numeric: peak of the ramp rate Gamma = e^(1-1/eta)/eta^2 | PASS |
| 149 | ch:gravdec:L149 | calc | `0.759` | numeric: purity 1/sqrt(1+2 x), ramp, eta=0.5 | PASS |
| 149 | ch:gravdec:L149:0.707 | calc | `0.707` | numeric: purity 1/sqrt(1+2 x), constant, eta=0.5 | PASS |
| 149 | ch:gravdec:L149:0.577 | calc | `0.577` | numeric: purity 1/sqrt(1+2 x), eta=1 (both) | PASS |
| 149 | ch:gravdec:L149:0.482 | calc | `0.482` | numeric: purity 1/sqrt(1+2 x), ramp, eta=2 | PASS |
| 149 | ch:gravdec:L149:0.447 | calc | `0.447` | numeric: purity 1/sqrt(1+2 x), constant, eta=2 | PASS |
| 149 |  | calc | `0.5` | not run: input: eta = 0.5 at which the two purities are compared (purities checked at ch:gravdec:L149) | - |
| 150 | ch:gravdec:L150 | calc | `0.43` | numeric: purity, ramp, eta=5 | PASS |
| 150 | ch:gravdec:L150:0.30 | calc | `0.30` | numeric: purity, constant, eta=5 | PASS |
| 154 | ch:gravdec:L154 | derived | `0.5` | numeric: peak of the ramp rate | PASS |
| 157 |  | calc | `50` | not run: input: N = 50 time points of the proposed experiment | - |
| 159 |  | calc | `10` | not run: input: 10^-12 kg (509 s) named as the lower mass of the profile test (509 s checked at ch:gravdec:L85) | - |
| 160 | ch:gravdec:L160 | calc | `3.5\times10^{-12}` | numeric: mass where tau_IAM = 1 s at 10 mK | PASS |
| 160 | ch:gravdec:L160:2.2\times10^{-10} | calc | `2.2\times10^{-10}` | numeric: crossover tau_IAM = tau_DP at 10 mK | PASS |
| 164 | ch:gravdec:L164 | calc | `9.8` | numeric: significance of the factor 16 (10 to 40 mK) against constant tau, 20 % precision on each tau | PASS |
| 165 |  | calc | `10` | not run: restates the measurable mass range 10^-12-10^-11 kg at 10 mK (bounds checked at ch:gravdec:L98:3.5e-12 and ch:gravdec:L99) | - |
| 168 |  | prediction | `10` | not run: input: 10^-12 kg (a nanogram) near which the heating rate passes 1 phonon/s (the crossing mass is checked at ch:gravdec:L120) | - |
| 172 | ch:gravdec:L172 | calc | `3.5\times10^{-12}` | numeric: mass where tau_IAM = 1 s at 10 mK | PASS |
| 172 |  | calc | `10` | not run: input: 10^-19 kg, the mass of present nanoparticle quantum control (cited Rossi2025, Neumeier2024) | - |
| 207 | ch:gravdec:L207 | calc | `7.1\times10^{-9}` | numeric: E_G cat, m 4 kg, R 0.15 m | PASS |
| 207 |  | calc | `300` | not run: input: T = 300 K of the cat example | - |
| 208 | ch:gravdec:L208 | calc | `3.5\times10^{-51}` | numeric: tau_IAM cat at 300 K | PASS |
| 211 |  | calc | `10` | not run: input: 10^-12 kg mass of the nanosphere example | - |
| 211 |  | calc | `2200` | not run: input: silica density 2200 kg/m^3 | - |
| 212 | ch:gravdec:L212 | calc | `4.8` | numeric: radius of 1e-12 kg silica sphere, um | PASS |
| 212 | ch:gravdec:L212:1.4\times10^{-29} | calc | `1.4\times10^{-29}` | numeric: E_G of the nanosphere | PASS |
| 212 | ch:gravdec:L212:509 | calc | `509` | numeric: tau_IAM at 10 mK | PASS |
| 213 | ch:gravdec:L213 | calc | `7.5` | numeric: tau_DP, us | PASS |
| 214 | ch:gravdec:L214 | calc | `7\times10^{7}` | numeric: tau_IAM / tau_DP | PASS |
| 214 |  | calc | `10` | not run: input: 10^-12 kg mass of the nanosphere example (the ratio 7e7 is checked at ch:gravdec:L214) | - |
| 231 | ch:gravdec:L231 | conjecture | `0.5` | numeric: t < 0.5 tau_IAM: the ramp rate is largest at eta = 1/2 (sympy) | PASS |
| 248 | ch:gravdec:L248 | conjecture | `13.8` | numeric: age of the universe, flat LCDM with Planck 2018 parameters, Gyr | PASS |

## Part 4 - ch:nonlocal - `docs/book/part4/p4_06_nonlocality.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 20 | ch:nonlocal:L20 | conjecture | `0.414` | numeric: coherence at which sqrt2 (1+c) = 2 | PASS |
| 20 | ch:nonlocal:L20:1.87 | conjecture | `1.87` | numeric: t/tau_IAM at which the assumed ramp c = 1 - E(t/tau)/e reaches the crossing | PASS |
| 31 | ch:nonlocal:L31 | derived |  | sympy: Horodecki: dephased Bell state, S_max = 2 sqrt(1+c^2) | PASS |
| 35 | ch:nonlocal:L35 | calc | `0.414` | numeric: sqrt2 (1+c) = 2 at c = sqrt2 - 1 | PASS |
| 36 | ch:nonlocal:L36 | calc | `0.586` | numeric: D = 1 - c = 2 - sqrt2 | PASS |
| 41 | ch:nonlocal:L41 | observed | `1.3` | numeric: separation of the NV-centre electron spins in the loophole-free Bell test, km | PASS |
| 41 | ch:nonlocal:L41:10^3 | observed | `10^3` | numeric: ratio of the satellite photon distance to the NV-spin separation | PASS |

## Part 4 - ch:electroweak - `docs/book/part4/p4_22_electroweak.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 18 | eq:ew:virial | derived |  | sympy: virial theorem for a potential homogeneous of degree k | PASS |
| 25 | ch:electroweak:L25 | calc | `0.91` | numeric: string tension 0.18 GeV^2 in GeV/fm | PASS |
| 32 | ch:electroweak:L32 | calc | `2.5\times10^{-18}` | numeric: range hbar/(M_W c), m | PASS |
| 32 |  | calc | `80.369` | not run: input: M_W = 80.369 GeV (PDG 2024, doi:10.1103/PhysRevD.110.030001), used by ch:electroweak:L32; nothing to recompute | - |
| 36 | ch:electroweak:L36 | observed | `80.4` | numeric: W mass, PDG 2022 | PASS |
| 36 | ch:electroweak:L36:91.2 | observed | `91.2` | numeric: Z mass, PDG 2022 | PASS |
| 53 | ch:electroweak:L53 | observed | `159.5` | heavy file `docs/verification/scripts/verify_entanglement_electroweak_output.txt`: measured: printed value found in verify_entanglement_electroweak_output.txt, a file the chapter names | PASS |
| 53 | ch:electroweak:L53:0.301 | observed | `0.301` | file `docs/verification/particle/ELECTROWEAK_CHECK.md`: measured: printed value found in ELECTROWEAK_CHECK.md, a file the chapter names | PASS |
| 54 | ch:electroweak:L54 | calc | `4.9\times10^{-16}` | numeric: scale factor at the crossover, entropy conservation | PASS |
| 54 | ch:electroweak:L54:106.75 | calc | `106.75` | numeric: g_* of the Standard Model above the top mass | PASS |
| 55 | ch:electroweak:L55 | calc | `-2.0\times10^{15}` | numeric: ln E(a_EW) = 1 - 1/a_EW | PASS |
| 62 | ch:electroweak:L62 | calc | `246.22` | numeric: v = (sqrt2 G_F)^(-1/2), GeV | PASS |
| 69 | ch:electroweak:L69 | derived | `10^{-18}` | numeric: photon mass bound, PDG 2024, eV | PASS |
| 90 | eq:ew:betam | calc | `0.1577` | numeric: beta_m = Omega_b/2 + Omega_dm/2 (Planck 2018 Omega_b 0.0493) | PASS |
| 90 | ch:electroweak:L90 | calc | `0.0247` | numeric: Omega_b/2 | PASS |
| 90 | ch:electroweak:L90:0.1330 | calc | `0.1330` | numeric: Omega_dm/2 | PASS |
| 93 | ch:electroweak:L93 | calc | `15.6` | numeric: baryonic share of beta_m, per cent | PASS |
| 93 | ch:electroweak:L93:84.4 | calc | `84.4` | numeric: dark share of beta_m, per cent | PASS |
| 109 | ch:electroweak:L109 | prediction | `0.864` | numeric: mu(z=0) = 1/(1+beta_m) | PASS |
| 109 |  | prediction | `-0.136` | not run: locked value mu0 restated (prediction) | - |
| 115 |  | derived | `159.5` | not run: restates ch:electroweak:L53 (T_c = 159.5 GeV, D'Onofrio and Rummukainen 2016, input) | - |
| 119 |  | prediction | `-0.136` | not run: locked value mu0 restated (prediction) | - |

## Part 4 - ch:higgsrecord - `docs/book/part4/p4_22b_higgs_record.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 17 | ch:higgsrecord:L17 | observed | `159.5` | heavy file `docs/verification/scripts/verify_entanglement_electroweak_output.txt`: measured: printed value found in verify_entanglement_electroweak_output.txt, a file the chapter names | PASS |
| 17 | ch:higgsrecord:L17:106.75 | observed | `106.75` | heavy file `docs/verification/scripts/verify_entanglement_electroweak_output.txt`: measured: printed value found in verify_entanglement_electroweak_output.txt, a file the chapter names | PASS |
| 17 | ch:higgsrecord:L17:0.301 | observed | `0.301` | numeric: coefficient 0.301 of t = 0.301 g*^(-1/2) m_P/T^2 | PASS |
| 18 | ch:higgsrecord:L18 | calc | `9.2\times10^{-12}` | numeric: t = 0.301 g*^(-1/2) m_P/T^2 at T_c, s | PASS |
| 18 | ch:higgsrecord:L18:9.0 | calc | `9.0` | numeric: crossover time at T_c + 1.5 GeV, units of 1e-12 s | PASS |
| 18 | ch:higgsrecord:L18:9.4 | calc | `9.4\times10^{-12}` | numeric: crossover time at T_c - 1.5 GeV, s | PASS |
| 19 | ch:higgsrecord:L19 | calc | `246.22` | numeric: v = (sqrt2 G_F)^(-1/2) | PASS |
| 20 | ch:higgsrecord:L20 | calc | `0.129` | numeric: lambda = m_H^2/(2 v^2) | PASS |
| 20 |  | calc | `125.20` | not run: input: m_H = 125.20 +- 0.11 GeV (PDG 2024, doi:10.1103/PhysRevD.110.030001), used by ch:higgsrecord:L20; also read from file by ch:higgsrecord:L125:125.20 | - |
| 22 | ch:higgsrecord:L22 | observed | `80.3692` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 22 | ch:higgsrecord:L22:91.1880 | observed | `91.1880` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 23 | ch:higgsrecord:L23 | calc | `2.9\times10^{-6}` | numeric: y_e = sqrt2 m_e/v | PASS |
| 23 | ch:higgsrecord:L23:0.991 | calc | `0.991` | numeric: y_t = sqrt2 m_t/v | PASS |
| 25 | ch:higgsrecord:L25 | observed | `10^{-18}` | numeric: photon mass bound, PDG 2024, eV | PASS |
| 65 | ch:higgsrecord:L65 | calc | `4.9\times10^{-16}` | numeric: a at the crossover | PASS |
| 66 | ch:higgsrecord:L66 | calc | `-2.04\times10^{15}` | numeric: ln E at the crossover | PASS |
| 66 | ch:higgsrecord:L66:0.17 | calc | `0.17` | numeric: E = e^-z at z = 1.77 | PASS |
| 67 | ch:higgsrecord:L67 | calc | `1.77` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 67 | ch:higgsrecord:L67:3.67 | calc | `3.67` | numeric: cosmic age at z=1.77, Gyr (Planck 2018 LCDM) | PASS |
| 67 | ch:higgsrecord:L67:0.44 | calc | `0.44` | numeric: z at a cosmic age of 9 Gyr | PASS |
| 67 | ch:higgsrecord:L67:0.64 | calc | `0.64` | numeric: E at z = 0.44 | PASS |
| 72 | ch:higgsrecord:L72 | observed | `0.120` | file `docs/book/verification/sources/MANIFEST_particle.md`: measured: printed value found in MANIFEST_particle.md, a file the chapter names | PASS |
| 88 | ch:higgsrecord:L88 | calc | `9.4\times10^{-14}` | numeric: E(z=30) = e^-30 | PASS |
| 88 | ch:higgsrecord:L88:4.5\times10^{-5} | calc | `4.5\times10^{-5}` | numeric: E(z=10) = e^-10 | PASS |
| 92 | ch:higgsrecord:L92 | calc | `110.6` | numeric: k_B T_c ln2 in GeV | PASS |
| 98 |  | calc | `159.5` | not run: restates ch:higgsrecord:L17 (T_c = 159.5 GeV, input from DOnofrio2016) in the figure caption | - |
| 99 |  | calc | `10` | not run: restates ch:higgsrecord:L25 (photon mass bound 10^-18 eV) in the figure caption | - |
| 100 | ch:higgsrecord:L100 | calc | `2.0\times10^{15}` | numeric: z at the crossover | PASS |
| 100 | ch:higgsrecord:L100:1.1e3 | calc | `1.1\times10^3` | numeric: -ln E = z at recombination | PASS |
| 100 |  | calc | `30` | not run: input: z = 30, the end of the figure axis (first haloes), nothing to recompute | - |
| 101 | ch:higgsrecord:L101:3.938 | calc | `3.938` | numeric: g*s today, photons plus neutrinos | PASS |
| 110 | ch:higgsrecord:L110 | calc | `2.7\times10^{-5}` | numeric: Hubble rate at T_c redshifted to today, Hz | PASS |
| 111 | ch:higgsrecord:L111 | calc | `0.23` | numeric: bubble-collision peak / Hubble frequency per beta/H | PASS |
| 111 | ch:higgsrecord:L111:1.15 | calc | `1.15` | numeric: sound-wave peak / Hubble frequency per beta/H | PASS |
| 111 | ch:higgsrecord:L111:10^{-4} | calc | `10^{-4}` | numeric: low end of the signal band, beta/H = 10, Hz | PASS |
| 111 | ch:higgsrecord:L111:10^{-2} | calc | `10^{-2}` | numeric: high end of the signal band, beta/H = 1000, Hz | PASS |
| 111 |  | calc | `1000` | not run: input: beta/H = 10-1000, the range of transition rates considered (Caprini2016); the band it gives is checked by ch:higgsrecord:L111:10^{-4} and ch:higgsrecord:L111:10^{-2} | - |
| 125 | ch:higgsrecord:L125 | observed | `159.5` | heavy file `docs/verification/scripts/verify_entanglement_electroweak_output.txt`: measured: printed value found in verify_entanglement_electroweak_output.txt, a file the chapter names | PASS |
| 125 | ch:higgsrecord:L125:246.22 | observed | `246.22` | numeric: same value as p4_22_electroweak:62 (v = (sqrt2 G_F)^(-1/2), GeV) | PASS |
| 125 | ch:higgsrecord:L125:125.20 | observed | `125.20` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 125 | ch:higgsrecord:L125:9.2 | observed | `9.2\times10^{-12}` | numeric: crossover time at T_c (status table), s | PASS |
| 130 | ch:higgsrecord:L130 | calc | `-2.04\times10^{15}` | numeric: ln E at the crossover | PASS |
| 134 |  | prediction | `-0.136` | not run: locked value mu0 restated (prediction) | - |

## Part 4 - ch:koide - `docs/book/part4/p4_15a_lepton_koide.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 18 | eq:ko:Q | derived |  | sympy: Q = 1/3 for equal masses, Q < 1 (sqrt-mass form) | PASS |
| 29 | ch:koide:L29 | observed | `0.51099895000` | numeric: electron mass, MeV (CODATA 2018) | PASS |
| 30 | ch:koide:L30 | observed | `105.6583755` | numeric: muon mass, MeV (CODATA 2018 ratio) | PASS |
| 31 | ch:koide:L31 | observed | `1776.93` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 32 | ch:koide:L32 | observed | `1776.86` | heavy file `docs/verification/scripts/verify_koide_output.txt`: measured: printed value found in verify_koide_output.txt, a file the chapter names | PASS |
| 33 | ch:koide:L33 | calc | `0.66666446` | numeric: Koide Q, PDG 2024 masses | PASS |
| 34 | ch:koide:L34 | calc | `2.2\times10^{-6}` | numeric: 2/3 - Q, 2024 | PASS |
| 34 | ch:koide:L34:0.43 | calc | `0.43` | numeric: (2/3 - Q)/sigma(Q) from sigma(m_tau) | PASS |
| 35 | ch:koide:L35 | calc | `0.66666051` | numeric: Koide Q with the 2022 m_tau | PASS |
| 35 | ch:koide:L35:6.2\times10^{-6} | calc | `6.2\times10^{-6}` | numeric: 2/3 - Q, 2022 | PASS |
| 39 | ch:koide:L39 | observed | `2.2\times10^{-6}` | numeric: 2/3 - Q, PDG 2024 masses (text restatement) | PASS |
| 45 | ch:koide:L45:0.667824 | calc | `0.667824` | numeric: Q of the running masses at mu = m_tau | PASS |
| 45 | ch:koide:L45:0.667840 | calc | `0.667840` | numeric: Q of the running masses at mu = M_Z | PASS |
| 48 | ch:koide:L48 | calc | `1.2\times10^{-3}` | numeric: departure of running-mass Q from 2/3 | PASS |
| 48 | ch:koide:L48:10^{-6} | calc | `10^{-6}` | numeric: pole-mass agreement is at the 1e-6 level | PASS |
| 54 | ch:koide:L54 | calc |  | sympy: square-root mass vector, MeV^1/2 | PASS |
| 58 | eq:ko:angle | calc |  | sympy: cos^2 theta = 1/(3Q) | PASS |
| 61 | ch:koide:L61 | derived | `45` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 62 | ch:koide:L62 | calc | `44.99991` | numeric: angle of the sqrt-mass vector to (1,1,1) | PASS |
| 63 | ch:koide:L63 | calc | `45` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 67 | ch:koide:L67 | calc | `10.2790` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 67 | ch:koide:L67:42 | calc | `42` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 68 | ch:koide:L68 | calc | `45` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 69 | ch:koide:L69 | calc | `313.85` | numeric: x^2, MeV | PASS |
| 77 | eq:ko:param | derived |  | sympy: any three square roots fit x[1 + r cos(delta + 2 pi k/3)] | PASS |
| 82 | eq:ko:Z3 | derived |  | sympy: Z3 sums of cos and cos^2 | PASS |
| 86 | eq:ko:Qr | derived |  | sympy: Q = (1 + r^2/2)/3 from the parametrisation | PASS |
| 91 | ch:koide:L91 | calc | `17.71584` | numeric: x = sum sqrt(m)/3 | PASS |
| 91 | ch:koide:L91:313.851 | calc | `313.851` | numeric: x^2, MeV | PASS |
| 91 | ch:koide:L91:1.414209 | calc | `1.414209` | numeric: r from Q = (1 + r^2/2)/3 | PASS |
| 91 | ch:koide:L91:1.414214 | calc | `1.414214` | numeric: sqrt 2 | PASS |
| 92 | ch:koide:L92 | calc | `0.222225` | numeric: offset delta, rad (tau at k=0) | PASS |
| 92 | ch:koide:L92:0.222222 | calc | `0.222222` | numeric: 2/9 | PASS |
| 93 | ch:koide:L93 | calc | `2.5\times10^{-6}` | numeric: delta - 2/9 | PASS |
| 93 | ch:koide:L93:0.41 | calc | `0.41` | numeric: (delta - 2/9)/sigma(delta) | PASS |
| 120 | eq:ko:TU | none |  | sympy: Unruh temperature from regularity of the Euclidean Rindler plane | PASS |
| 125 | eq:ko:eta | derived |  | sympy: eta = c^3/(4 hbar G) = 1/(4 l_P^2) | PASS |
| 133 | eq:ko:onebit | derived |  | sympy: eta dA_min = 1, dA_min = 4 l_P^2 | PASS |
| 143 | eq:ko:maxent | derived |  | sympy: maximum entropy over K states gives p_k = 1/K (K=3) | PASS |
| 152 | eq:ko:fourier | none |  | not run: definition: the general Fourier series of sqrt(m) on the orbit S^1 | - |
| 156 | eq:ko:grad | derived |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 161 | eq:ko:boltz | conjecture |  | sympy: Boltzmann suppression of mode n relative to mode 1 | PASS |
| 166 | eq:ko:two | conjecture |  | sympy: reflection symmetry removes sin(phi): sqrt m = x + y cos(phi) | PASS |
| 174 | ch:koide:L174 | calc | `0.67` | numeric: largest shift of Q per unit a_2/y from a cos(2 phi) admixture | PASS |
| 181 | eq:ko:weights | conjecture |  | sympy: Parseval weights of the constant and first-harmonic channels | PASS |
| 186 | eq:ko:ratio | conjecture |  | sympy: equal weights x^2 = y^2/2 give y/x = sqrt2 | PASS |
| 196 | eq:ko:pos | derived |  | sympy: 1 + sqrt2 cos(phi) > 0 iff |phi - pi| > pi/4 | PASS |
| 206 | ch:koide:L206 | derived | `0.2222` | numeric: measured offset delta | PASS |
| 212 | ch:koide:L212 | derived | `0.50` | numeric: fraction of offsets admitting n=2 | PASS |
| 212 | ch:koide:L212:0.25 | derived | `0.25` | numeric: fraction of offsets admitting n=3 | PASS |
| 213 | ch:koide:L213 | derived | `0.2222` | numeric: measured offset delta | PASS |
| 213 | ch:koide:L213:0.975 | derived | `0.975` | numeric: cos delta | PASS |
| 228 | eq:ko:thm2 | derived |  | sympy: Q = 2/3 at y/x = sqrt2 | PASS |
| 240 | eq:ko:delta0 | calc |  | sympy: masses at delta = 0 with x^2 = 313.851 MeV | PASS |
| 243 | ch:koide:L243 | openprob | `0.2222` | numeric: measured offset delta (restated) | PASS |
| 246 | ch:koide:L246 | calc | `0.2618` | numeric: electron amplitude zero at delta = pi/12 | PASS |
| 247 | ch:koide:L247 | calc | `0.0396` | numeric: measured delta inside the edge | PASS |
| 253 | ch:koide:L253 | calc | `0.2222` | numeric: measured offset delta | PASS |
| 254 | ch:koide:L254 | calc | `2.379` | numeric: tau amplitude | PASS |
| 254 | ch:koide:L254:0.040 | calc | `0.040` | numeric: electron amplitude | PASS |
| 255 | ch:koide:L255 | calc | `0.580` | numeric: muon amplitude | PASS |
| 255 | ch:koide:L255:0.293 | calc | `0.293` | numeric: e and mu amplitude at delta = 0 | PASS |
| 260 | ch:koide:L260 | calc | `313.85` | numeric: x^2, MeV | PASS |
| 260 | ch:koide:L260:0.2222 | calc | `0.2222` | numeric: measured offset delta | PASS |
| 260 |  | calc | `0.10` | not run: input: delta = 0.10, an offset chosen for the sweep figure, nothing to recompute | - |
| 261 |  | calc | `0.40` | not run: input: delta = 0.40, an offset chosen for the sweep figure, nothing to recompute | - |
| 263 | ch:koide:L263 | calc | `26.92` | numeric: m_e = m_mu at delta = 0, MeV | PASS |
| 272 | ch:koide:L272 | calc | `17.71584` | numeric: x = sum sqrt(m)/3 | PASS |
| 272 | ch:koide:L272:313.851 | calc | `313.851` | numeric: x^2, MeV | PASS |
| 273 | ch:koide:L273 | calc | `1.414209` | numeric: r from Q = (1 + r^2/2)/3 | PASS |
| 273 | ch:koide:L273:1.414214 | calc | `1.414214` | numeric: sqrt 2 | PASS |
| 274 | ch:koide:L274 | calc | `0.222225` | numeric: offset delta, rad (tau at k=0) | PASS |
| 275 | ch:koide:L275 | calc | `44.99991` | numeric: angle of the sqrt-mass vector to (1,1,1) | PASS |
| 276 | ch:koide:L276 | calc | `26.92` | numeric: m_e = m_mu at delta = 0, MeV | PASS |
| 276 | ch:koide:L276:1829.26 | calc | `1829.26` | numeric: m_tau at delta = 0, MeV | PASS |
| 277 | ch:koide:L277 | calc | `0.2618` | numeric: electron amplitude zero at delta = pi/12 | PASS |
| 277 | ch:koide:L277:0.0396 | calc | `0.0396` | numeric: measured delta inside the edge | PASS |
| 278 | ch:koide:L278 | calc | `0.222222047` | numeric: delta fixed by m_e, m_mu at Q = 2/3 | PASS |
| 279 | ch:koide:L279 | calc | `1.75\times10^{-7}` | numeric: 2/9 - delta at Q = 2/3 | PASS |
| 283 | ch:koide:L283 | calc | `0.222222047` | numeric: delta fixed by m_e, m_mu at Q = 2/3 | PASS |
| 284 | ch:koide:L284 | calc | `4\times10^{-10}` | numeric: uncertainty of delta from m_mu | PASS |
| 284 | ch:koide:L284:1.75\times10^{-7} | calc | `1.75\times10^{-7}` | numeric: 2/9 - delta at Q = 2/3 | PASS |
| 285 | ch:koide:L285 | calc | `0.4` | numeric: delta - 2/9 in sigma | PASS |
| 295 | ch:koide:L295 | calc | `1776.9690` | numeric: m_tau fixed by Q = 2/3, MeV | PASS |
| 295 |  | calc | `1776.93` | not run: input: m_tau = 1776.93 MeV (PDG 2024, doi:10.1103/PhysRevD.110.030001), restates the table value read by ch:koide:L31 | - |
| 296 | ch:koide:L296 | calc | `-0.43` | numeric: pull of the 2024 average | PASS |
| 296 | ch:koide:L296:-0.9 | calc | `-0.9` | numeric: pull of the 2022 average | PASS |
| 296 |  | calc | `1776.86` | not run: input: m_tau = 1776.86 MeV (PDG 2022, doi:10.1093/ptep/ptac097), restates the table value read by ch:koide:L32 | - |
| 297 | ch:koide:L297 | prediction | `3.9` | numeric: separation of 1776.93 and the Q = 2/3 tau mass at +-0.01 MeV | PASS |
| 297 |  | prediction | `1777.09` | not run: input: single measurement m_tau = 1777.09 +- 0.08 +- 0.11 MeV quoted from BelleII2023tau, nothing to recompute | - |
| 299 | ch:koide:L299 | calc | `1776.969` | numeric: m_tau fixed by Q = 2/3, MeV | PASS |
| 299 | ch:koide:L299:0.2222220 | calc | `0.2222220` | numeric: delta at m_tau = 1776.969 | PASS |
| 315 | ch:koide:L315 | observed | `0.66666446` | numeric: same value as p4_15a_lepton_koide:33 (Koide Q, PDG 2024 masses) | PASS |
| 315 | ch:koide:L315:0.43 | observed | `0.43` | numeric: 2/3 - Q in standard deviations (status table) | PASS |
| 316 | ch:koide:L316 | derived | `45` | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 322 | ch:koide:L322 | openprob | `0.2222` | numeric: offset delta in the status table | PASS |
| 323 | ch:koide:L323 | prediction | `1776.969` | numeric: same value as p4_15a_lepton_koide:299 (m_tau fixed by Q = 2/3, MeV) | PASS |

## Part 4 - ch:electronmass - `docs/book/part4/p4_15b_electron_mass.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 17 | eq:em:T | derived |  | sympy: Hawking temperature from T = hbar kappa/(2 pi k_B) | PASS |
| 22 | ch:electronmass:L22 | calc | `2.66\times10^{-30}` | numeric: T_GH, H0 = 67.4 | PASS |
| 22 | ch:electronmass:L22:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 22 | ch:electronmass:L22:2.65\times10^{-30} | calc | `2.65\times10^{-30}` | numeric: T_GH, H0 = 67.16 | PASS |
| 22 |  | calc | `67.4` | not run: input: H0 = 67.4 km/s/Mpc (Planck 2018, Aghanim et al. 2020, doi:10.1051/0004-6361/201833910), nothing to recompute | - |
| 27 | eq:em:S | conjecture | `1.79\times10^{45}` | numeric: S = pi (m_P/m_e)^2 | PASS |
| 32 | ch:electronmass:L32 | calc | `2.9\times10^{44}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 35 | eq:em:Ebit | derived | `2.54\times10^{-53}` | numeric: E_bit = hbar H0 ln2/2pi, H0 = 67.4 | PASS |
| 42 | eq:em:N | conjecture | `3.69\times10^{33}` | numeric: N = (m_P/m_e)^(3/2) | PASS |
| 53 | eq:em:fs | none | `4.55\times10^{-6}` | numeric: alpha^(5/2) | PASS |
| 58 | eq:em:ft | conjecture |  | sympy: temporal reading of the electromagnetic factor, alpha^(5/2) | PASS |
| 67 | eq:em:fp | conjecture |  | sympy: m c^2 = E_bit N(m)/f has exactly one positive root | PASS |
| 72 | eq:em:m52 | derived |  | sympy: m^(5/2) from m c^2 = E_bit N/f | PASS |
| 77 | eq:em:mstar | derived |  | sympy: m* = (2 pi)^(-2/5) B | PASS |
| 80 | ch:electronmass:L80 | derived | `1.2018` | numeric: B/m_e at H0 = 67.4 | PASS |
| 80 | ch:electronmass:L80:0.4794 | derived | `0.4794` | numeric: (2 pi)^(-2/5) | PASS |
| 80 |  | derived | `67.4` | not run: input: H0 = 67.4 restated (the value at which the factor was identified) | - |
| 81 | eq:em:short | derived | `0.5762` | numeric: fixed point as derived, units of m_e | PASS |
| 84 | ch:electronmass:L84 | calc | `0.252` | numeric: right side at m_e over m_e c^2 | PASS |
| 84 | ch:electronmass:L84:3.97 | calc | `3.97` | numeric: factor short | PASS |
| 87 | eq:em:result | calc | `1.0000066` | numeric: (2pi)^(-1/10) B / m_e at H0 = 67.4 | PASS |
| 90 | ch:electronmass:L90 | calc | `0.832107` | numeric: needed prefactor m_e/B | PASS |
| 90 | ch:electronmass:L90:0.832112 | calc | `0.832112` | numeric: (2 pi)^(-1/10) | PASS |
| 91 | ch:electronmass:L91 | fitted | `1.7356` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 92 | ch:electronmass:L92 | calc | `3.969` | numeric: (2 pi)^(3/4) | PASS |
| 100 | ch:electronmass:L100 | derived | `2.54\times10^{-53}` | numeric: bit price, H0 = 67.4 | PASS |
| 101 | ch:electronmass:L101 | conjecture | `1.79\times10^{45}` | numeric: area count of the Compton sphere (table) | PASS |
| 102 | ch:electronmass:L102 | derived | `6.28` | numeric: 2 pi | PASS |
| 103 | ch:electronmass:L103 | conjecture | `3.69\times10^{33}` | numeric: cell count (m_P/m_e)^(3/2) (table) | PASS |
| 104 | ch:electronmass:L104 | conjecture | `4.55\times10^{-6}` | numeric: electromagnetic factor alpha^(5/2) (table) | PASS |
| 105 | ch:electronmass:L105 | calc | `1.2018` | numeric: B/m_e at H0 = 67.4 | PASS |
| 106 | ch:electronmass:L106 | calc | `0.5762` | numeric: fixed point as derived | PASS |
| 107 | ch:electronmass:L107 | fitted | `1.7356` | heavy file `docs/verification/scripts/verify_particle_book_output.txt`: measured: printed value found in verify_particle_book_output.txt, a file the chapter names | PASS |
| 108 | ch:electronmass:L108 | calc | `1.0000066` | numeric: with the identified factor | PASS |
| 115 | ch:electronmass:L115 | calc | `0.576` | numeric: crossing as derived | PASS |
| 116 | ch:electronmass:L116 | calc | `1.00001` | numeric: crossing with the identified factor | PASS |
| 116 | ch:electronmass:L116:-0.14 | calc | `-0.14` | numeric: fixed point offset at H0 = 67.16, per cent | PASS |
| 116 | ch:electronmass:L116:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 116 | ch:electronmass:L116:-0.02 | calc | `-0.02` | numeric: fixed point offset at H0 = 67.36, per cent | PASS |
| 116 |  | calc | `67.4` | not run: input: H0 = 67.4 restated in the figure caption | - |
| 117 | ch:electronmass:L117 | calc | `+2.82` | numeric: fixed point offset at H0 = 72.26, per cent | PASS |
| 117 | ch:electronmass:L117:72.26 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 117 | ch:electronmass:L117:+3.27 | calc | `+3.27` | numeric: fixed point offset at H0 = 73.04, per cent | PASS |
| 117 |  | calc | `67.36` | not run: input: H0 = 67.36, Planck 2018 best fit (Aghanim et al. 2020 Table 2, doi:10.1051/0004-6361/201833910); the offset there is checked by ch:electronmass:L116:-0.02 | - |
| 117 |  | calc | `73.04` | not run: input: H0 = 73.04, SH0ES (Riess2022); the offset there is checked by ch:electronmass:L117:+3.27 | - |
| 119 | ch:electronmass:L119 | calc | `7.16` | numeric: p = 7/2 | PASS |
| 123 |  | calc | `0.54` | not run: input: sigma(H0) = 0.54, Planck 2018 (Aghanim et al. 2020 Table 2, doi:10.1051/0004-6361/201833910) | - |
| 124 | ch:electronmass:L124 | calc | `67.399` | numeric: H0 at which the fixed point is exact | PASS |
| 124 |  | calc | `67.4` | not run: input: H0 = 67.4 restated | - |
| 125 | ch:electronmass:L125 | calc | `6.6` | numeric: ppm offset at 67.4 | PASS |
| 125 |  | calc | `67.4` | not run: input: H0 = 67.4 restated | - |
| 126 |  | calc | `0.3` | not run: restates ch:electronmass:L137 (0.32 per cent) rounded to one digit; a one-digit 0.3 cannot pass a 5 % negative control (0.315 lies within half its last digit of 0.320) | - |
| 136 | ch:electronmass:L136 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 136 | ch:electronmass:L136:0.14 | calc | `0.14` | numeric: per cent low at 67.16 | PASS |
| 137 | ch:electronmass:L137 | openprob | `0.32` | numeric: spread from sigma(H0), per cent | PASS |
| 137 | ch:electronmass:L137:2.8 | openprob | `2.8` | numeric: fixed point at the matter-sector H0, per cent high | PASS |
| 137 |  | openprob | `72.26` | not run: locked value H0 = 72.26 (matter sector) restated | - |
| 148 | ch:electronmass:L148 | calc | `0.14` | numeric: fixed point with alpha^1.5, units of m_e | PASS |
| 148 | ch:electronmass:L148:0.37 | calc | `0.37` | numeric: fixed point with alpha^2, units of m_e | PASS |
| 148 | ch:electronmass:L148:1.00 | calc | `1.00` | numeric: fixed point with alpha^2.5, units of m_e | PASS |
| 148 | ch:electronmass:L148:2.68 | calc | `2.68` | numeric: fixed point with alpha^3, units of m_e | PASS |
| 148 | ch:electronmass:L148:7.16 | calc | `7.16` | numeric: fixed point with alpha^3.5, units of m_e | PASS |
| 149 | ch:electronmass:L149 | calc | `2.68` | numeric: alpha^(-1/5) | PASS |
| 152 | ch:electronmass:L152 | calc | `9.44\times10^8` | numeric: T_C = m_e c^2/(2 pi k_B) | PASS |
| 153 | ch:electronmass:L153 | calc | `2.66\times10^{-30}` | numeric: T_GH at 67.4 | PASS |
| 153 | ch:electronmass:L153:3.55\times10^{38} | calc | `3.55\times10^{38}` | numeric: T_C/T_GH | PASS |
| 153 | ch:electronmass:L153:38.6 | calc | `38.6` | numeric: orders of magnitude | PASS |
| 153 |  | calc | `67.4` | not run: input: H0 = 67.4 restated | - |
| 158 | ch:electronmass:L158 | calc | `1.56` | numeric: (H(z=2)/H0)^(2/5) | PASS |
| 159 | ch:electronmass:L159 | observed | `5\times10^{-6}` | numeric: bound on the drift of m_p/m_e from H2, 3 sigma | PASS |
| 159 | ch:electronmass:L159:2.0 | observed | `2.0` | numeric: lowest redshift of the H2 systems | PASS |
| 159 | ch:electronmass:L159:4.2 | observed | `4.2` | numeric: highest redshift of the H2 systems | PASS |
| 159 |  | observed | `10` | not run: measured, source not named | - |
| 171 | ch:electronmass:L171 | openprob | `67.40` | numeric: H0 at which the fixed point gives m_e | PASS |
| 186 | ch:electronmass:L186 | calc | `0.5762` | numeric: fixed point as derived | PASS |
| 188 | ch:electronmass:L188 | calc | `0.32` | numeric: 0.4 sigma(H0)/H0, per cent | PASS |
| 188 | ch:electronmass:L188:67.399 | calc | `67.399` | numeric: H0 at which the fixed point is exact | PASS |
| 189 | ch:electronmass:L189 | calc | `+2.8` | numeric: matter-sector H0 | PASS |

## Part 5 - ch:scprimer - `docs/book/part5/p5_01_sc_primer.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | ch:scprimer:L15 | calc | `16.0` | numeric: hf/k_B T, 5 GHz, 15 mK | PASS |
| 24 | ch:scprimer:L15:8.2\times10^{-3} | calc | `8.2\times10^{-3}` | numeric: e^(-hf/kT) at 50 mK | PASS |
| 24 |  | calc | `1.1\times10^{-7}` | not run: printed value 1.1e-7 has two significant digits: a 5 % change cannot be told from rounding (recomputed exp(-hf/kT) = 1.128e-7; 1.05 x 1.1e-7 = 1.155e-7 is within the 2.5 % control tolerance). The precise exponent is checked at ch:scprimer:L15 (hf/kT = 16.0) | - |
| 25 | ch:scprimer:L16 | calc | `1.20` | numeric: Delta_Al/(1.764 k_B), K | PASS |
| 25 | ch:scprimer:L16:182 | calc | `182` | numeric: BCS gap of aluminium, ueV | PASS |
| 26 | ch:scprimer:L17 | calc | `1.44\times10^{-25}` | numeric: k_B T ln2 at 15 mK, J | PASS |
| 26 | ch:scprimer:L17:5\times10^{-5} | calc | `5\times10^{-5}` | numeric: temperature ratio | PASS |
| 31 |  | observed | `0.1` | not run: measured, source not named | - |
| 32 | ch:scprimer:L23 | calc | `6.86` | numeric: M = hf/kT, 5 GHz, 35 mK | PASS |
| 32 | ch:scprimer:L23:1.1\times10^{-3} | calc | `1.1\times10^{-3}` | numeric: equilibrium occupation e^-M at 35 mK | PASS |
| 32 |  | calc | `1.1\times10^{-7}` | not run: printed value 1.1e-7 has two significant digits: a 5 % change cannot be told from rounding (recomputed 1.128e-7; 1.05 x 1.1e-7 = 1.155e-7 is within the 2.5 % control tolerance). The precise exponent is checked at ch:scprimer:L15 (hf/kT = 16.0) | - |
| 45 | ch:scprimer:L36 | calc |  | sympy: thermal x_qp vanishes as T -> 0 and rises with T | PASS |
| 49 | ch:scprimer:L40 | calc | `141` | numeric: Delta/k_B T, Al, 15 mK | PASS |
| 51 |  | observed | `10` | not run: measured, source not named | - |
| 53 |  | observed | `10` | not run: measured, source not named | - |
| 56 | eq:catelani | none |  | not run: definition: the published quasiparticle relaxation rate of Catelani2011 quoted as the model; used numerically by ch:scprimer:L126 and ch:scprimer:L126:2.4e-8 | - |
| 76 |  | observed | `592` | not run: measured, source not named | - |
| 76 |  | observed | `41` | not run: measured, source not named | - |
| 76 |  | observed | `17.1` | not run: measured, source not named | - |
| 78 |  | observed | `10` | not run: measured, source not named | - |
| 83 | ch:scprimer:L74 | calc | `88` | numeric: nu > 2 Delta/h, GHz | PASS |
| 83 |  | calc | `182` | not run: input: Delta_Al = 182 ueV restated (recomputed from BCS by ch:scprimer:L16:182) | - |
| 85 | ch:scprimer:L76 | calc | `8\times10^{16}` | numeric: black-body photons above 2 Delta_Al, 4 K, per s per m^2 | PASS |
| 85 | ch:scprimer:L76:2\times10^{20} | calc | `2\times10^{20}` | numeric: the same at 50 K | PASS |
| 85 | ch:scprimer:L76:10^{-108} | calc | `10^{-108}` | numeric: the same at 15 mK (order) | PASS |
| 85 |  | calc | `10` | not run: restates ch:scprimer:L76:10^{-108} (the '10' is the base of a printed power; every number on this line is checked by ch:scprimer:L76, ch:scprimer:L76:2\times10^{20}, ch:scprimer:L76:10^{-108}) | - |
| 86 | ch:scprimer:L77:2.7255 | calc | `2.7255` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 86 |  | calc | `4\times10^{22}` | not run: printed value 4e22 s^-1 m^-2 has one significant digit: a 5 % change cannot be told from rounding (recomputed 4.105e22; 1.05 x 4e22 = 4.2e22). No more precise value is printed | - |
| 87 | ch:scprimer:L78 | calc | `2\times10^{16}` | numeric: the same for the CMB | PASS |
| 93 | ch:scprimer:L84 | calc | `2.7\times10^{8}` | numeric: pairs broken at most by a 100 keV deposit | PASS |
| 103 |  | calc | `2.7255` | not run: input: T_CMB = 2.7255 K (Fixsen 2009) restated; also read by ch:scprimer:L77:2.7255 | - |
| 105 | ch:scprimer:L96 | calc | `61.7` | numeric: T_H of 1 M_sun, nK | PASS |
| 105 | ch:scprimer:L96:4.4\times10^{7} | calc | `4.4\times10^{7}` | numeric: T_CMB / T_H(1 M_sun) | PASS |
| 105 | ch:scprimer:L96:182 | calc | `182` | numeric: T_CMB / 15 mK | PASS |
| 106 | ch:scprimer:L97 | calc | `4.5\times10^{22}` | numeric: mass in balance with the CMB, kg | PASS |
| 108 | ch:scprimer:L99 | calc | `3.13\times10^{-6}` | numeric: CMB energy flux, W/m^2 | PASS |
| 109 | ch:scprimer:L100 | calc | `2.81\times10^{-6}` | numeric: CMB energy flux above 2 Delta_Al | PASS |
| 109 | ch:scprimer:L100:2.19\times10^{16} | calc | `2.19\times10^{16}` | numeric: CMB photons above 2 Delta_Al | PASS |
| 110 | ch:scprimer:L101 | calc | `2.2` | numeric: broken pairs per photon at most | PASS |
| 110 | ch:scprimer:L101:9.31 | calc | `9.31` | numeric: mean photon energy above threshold, in k_B K | PASS |
| 120 | ch:scprimer:L111 | calc | `2.11` | numeric: Delta_Al/k_B, K | PASS |
| 127 | ch:scprimer:L118 | calc | `1.764` | numeric: BCS ratio Delta/(k_B T_c) = pi e^(-gamma_E) | PASS |
| 127 |  | calc | `1.2` | not run: input: T_c(Al) = 1.2 K (material constant), restated from line 16; Delta_Al/(1.764 k_B) = 1.20 K is checked by ch:scprimer:L16 | - |
| 127 |  | calc | `4.47` | not run: input: T_c(Ta) = 4.47 K, material constant used for the figure, nothing to recompute | - |
| 127 |  | calc | `9.25` | not run: input: T_c(Nb) = 9.25 K, material constant used for the figure, nothing to recompute | - |
| 134 | ch:scprimer:L125 | calc | `182` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 135 | ch:scprimer:L126 | calc | `0.24` | numeric: T1 from Catelani, x_qp = 1e-7, ms | PASS |
| 135 | ch:scprimer:L126:2.4e-8 | calc | `2.4\times10^{-8}` | numeric: x_qp for T1 = 1 ms from Catelani, Al, 5 GHz | PASS |
| 136 | ch:scprimer:L127 | calc | `0.32` | numeric: T1 from Gamma = x_qp omega_q, ms | PASS |
| 137 | ch:scprimer:L128 | calc | `0.5` | file `docs/book/iam.bib`: best transmon T1, ms, from the cited title | PASS |
| 143 |  | observed | `10` | not run: input: surface-code threshold of about 1e-2 quoted from Fowler2012 (doi:10.1103/physreva.86.032324), an order of magnitude, nothing to recompute | - |
| 144 |  | observed | `68` | not run: measured, source not named | - |
| 144 |  | observed | `89` | not run: measured, source not named | - |
| 146 | ch:scprimer:L137 | calc | `1.31` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 169 | ch:scprimer:L160 | calc | `3.6\times10^{-12}` | numeric: hbar/Delta_Al, s | PASS |
| 169 | ch:scprimer:L160:1.055e-34 | calc | `1.055\times10^{-34}` | numeric: hbar = h/2pi, J s | PASS |
| 171 | ch:scprimer:L162 | calc | `4\times10^{-6}` | numeric: tau_phi/tau_TLS at 1 us | PASS |
| 171 | ch:scprimer:L162:4\times10^{-8} | calc | `4\times10^{-8}` | numeric: tau_phi/tau_TLS at 100 us | PASS |

## Part 5 - ch:xqp - `docs/book/part5/p5_02_xqp.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 14 | ch:xqp:L14 | observed | `1.5\times10^{-62}` | numeric: equilibrium x_qp in Al at 15 mK | PASS |
| 14 | ch:xqp:L14:141 | observed | `141` | numeric: Delta/k_B T, Al, 15 mK | PASS |
| 14 |  | calc | `182` | not run: input: Delta_Al = 182 ueV restated (recomputed from BCS by ch:scprimer:L16:182) | - |
| 14 |  | observed | `10` | not run: measured, source not named | - |
| 17 |  | observed |  | not run: measured, source not named | - |
| 19 |  | observed | `0.04` | not run: measured, source not named | - |
| 20 | ch:xqp:L20 | calc | `1\times10^{-8}` | numeric: x_qp from n_qp = 0.04 per um^3 and n_cp = 4e6 per um^3 | PASS |
| 55 | ch:xqp:L55 | conjecture | `2\times10^{-3}` | numeric: phase kick g/omega_q at g/2pi = 10 MHz, 5 GHz | PASS |
| 56 |  | conjecture | `100` | not run: input: tau_TLS ~ 1-100 us, the range of TLS switching times assumed, nothing to recompute | - |
| 67 | ch:xqp:L67 | derived | `364` | numeric: 2 Delta_Al, ueV | PASS |
| 68 | ch:xqp:L68 | derived | `126` | numeric: Delta ln2, ueV | PASS |
| 71 | ch:xqp:L71 | calc | `800` | numeric: diffusion length sqrt(D tau_qp), upper end, um | PASS |
| 72 |  | calc | `6.4` | not run: input: diffusion constant D = 0.6-6.4 um^2/ns of quasiparticles in Al (docs/verification/scripts/verify_xqp.py: normal-state value ~ 6 um^2/ns); used by ch:xqp:L71 | - |
| 77 | ch:xqp:L77 | calc | `9\times10^{10}` | numeric: n_e/2 for Al, per um^3 | PASS |
| 77 | ch:xqp:L77:22600 | calc | `22600` | numeric: n_e/2 over n_cp = 4e6 per um^3 | PASS |
| 79 | eq:xqp | derived |  | sympy: steady state x_qp = 2 N tau_qp/(tau_TLS n_cp V) | PASS |
| 86 | eq:xqp_veff | derived |  | sympy: V_eff = 4 pi int e^(-2r/lambda) r^2 dr = pi lambda^3 | PASS |
| 90 | ch:xqp:L90 | calc | `3.9\times10^{-4}` | numeric: V_eff, um^3 | PASS |
| 90 | ch:xqp:L90:1.6\times10^{3} | calc | `1.6\times10^{3}` | numeric: pairs in V_eff at n_cp = 4e6 per um^3 | PASS |
| 90 |  | calc | `4\times10^{6}` | not run: input: n_cp = 4e6 per um^3 restated from line 17 (see sources_needed for idx 1479) | - |
| 101 | eq:xqp_feedback | derived |  | sympy: fixed point of x = x0 + g x | PASS |
| 103 | ch:xqp:L103 | calc | `<0.15` | numeric: g = 2 phi tau_qp/tau_TLS < 1 gives phi < tau_TLS/(2 tau_qp) | PASS |
| 103 |  | calc | `100` | not run: input: tau_qp = 100 us, assumed lifetime for the worked example | - |
| 103 |  | calc | `30` | not run: input: tau_TLS = 30 us, assumed switching time for the worked example | - |
| 111 |  | derived | `4\times10^{6}` | not run: input: n_cp = 4e6 per um^3 restated in the figure caption | - |
| 111 |  | derived | `30` | not run: input: tau_TLS = 30 us restated in the figure caption | - |
| 111 |  | derived | `100` | not run: input: tau_qp = 100 us restated in the figure caption | - |
| 111 |  | derived | `10` | not run: input: island volumes 10^3-10^5 um^3 and target x_qp = 10^-7 of the figure (the '10' is the base of a printed power), nothing to recompute | - |
| 117 |  | calc | `30` | not run: input: tau_TLS = 30 us restated under the table | - |
| 117 |  | calc | `100` | not run: input: tau_qp = 100 us restated under the table | - |
| 122 |  | observed | `0.25` | not run: measured, source not named | - |
| 122 |  | observed | `5.4` | not run: measured, source not named | - |
| 124 |  | observed | `0.1` | not run: measured, source not named | - |
| 129 | ch:xqp:L129 | prediction |  | sympy: the predicted ratio x n_cp V tau_TLS/(N tau_qp) = 2 | PASS |
| 134 | ch:xqp:L134 | prediction | `<10^{-9}` | numeric: thermal x_qp at 100 mK below 1e-9 | PASS |
| 148 | ch:xqp:L148 | calc | `0.24` | numeric: T1 cap at x_qp = 1e-7, ms | PASS |
| 148 | ch:xqp:L148:0.02 | calc | `0.02` | numeric: T1 cap at x_qp = 1e-6, ms | PASS |
| 148 |  | calc | `182` | not run: input: Delta_Al = 182 ueV restated (recomputed from BCS by ch:scprimer:L16:182) | - |
| 148 |  | calc | `10` | not run: input: x_qp = 1e-7 and 1e-6, the densities at which the T1 cap is evaluated (the '10' is the base of a printed power); the caps are checked by ch:xqp:L148 and ch:xqp:L148:0.02 | - |
| 149 | ch:xqp:L149 | calc | `0.5` | file `docs/book/iam.bib`: best transmon T1, ms, from the cited title | PASS |
| 149 | ch:xqp:L149:10^{-7} | calc | `<10^{-7}` | numeric: x_qp allowed at T1 = 0.3 ms is below 1e-7 | PASS |

## Part 5 - ch:ascoreqc - `docs/book/part5/p5_03_a_for_processors.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 23 | eq:Agate | calc |  | sympy: -ln(1-p) = p + p^2/2 + ... | PASS |
| 27 |  | calc | `10` | not run: input: p < 1e-2 for current platforms (the '10' is the base of 10^-2); the 0.503 % at p = 1e-2 is checked by ch:ascoreqc:L28 | - |
| 28 | ch:ascoreqc:L28 | calc | `0.503` | numeric: (eps - p)/p at p = 1e-2, per cent | PASS |
| 28 |  | calc | `10` | not run: input: p = 1e-2 at which the 0.503 % is evaluated (the '10' is the base of 10^-2); checked by ch:ascoreqc:L28 | - |
| 51 |  | observed | `2.8\times10^{-3}` | not run: measured, source not named | - |
| 51 |  | observed | `5.5\times10^{-4}` | not run: measured, source not named | - |
| 51 |  | observed | `2.0\times10^{-4}` | not run: measured, source not named | - |
| 56 |  | observed | `99.922` | not run: measured, source not named | - |
| 56 |  | observed | `99.5` | not run: measured, source not named | - |
| 57 |  | observed | `99.93` | not run: measured, source not named | - |
| 57 |  | observed | `99.5` | not run: measured, source not named | - |
| 74 |  | calc | `68` | not run: measured, source not named | - |
| 75 | ch:ascoreqc:L75 | calc | `6.2\times10^{-7}` | numeric: thermal floor p_eq t_g/T1: 5 GHz, 35 mK, 40 ns, T1 = 68 us (book inputs) | PASS |
| 75 |  | calc | `10` | not run: input: a two-qubit error near 1e-3 (illustrative, the '10' is the base of 10^-3), nothing to recompute | - |
| 76 |  | interp | `6\times10^{-4}` | not run: restates ch:ascoreqc:L86:6.2\times10^{-4} rounded to one digit (6e-4); a one-digit value cannot pass the 5 % negative control (6.3e-4 lies within half its last digit of 6.18e-4) | - |
| 86 | ch:ascoreqc:L86 | calc | `6.2\times10^{-7}` | numeric: thermal floor p_eq t_g/T1: 5 GHz, 35 mK, 40 ns, T1 = 68 us (book inputs) | PASS |
| 86 | ch:ascoreqc:L86:6.2\times10^{-4} | calc | `6.2\times10^{-4}` | numeric: floor on the gauge of a 1e-3 two-qubit error | PASS |
| 86 | ch:ascoreqc:L86:0.503 | derived | `0.503` | numeric: (eps - p)/p at p = 1e-2, per cent | PASS |
| 86 | ch:ascoreqc:L86:10 | calc | `10` | numeric: threshold mark on the gauge, 1e-2/eps | PASS |
| 86 |  | calc | `68` | not run: measured, source not named | - |
| 89 |  | calc | `10` | not run: input: surface-code threshold about 1e-2 quoted from Fowler2012 (doi:10.1103/physreva.86.032324), an order of magnitude | - |
| 90 |  | calc | `10` | not run: input: fault-tolerance target about 1e-3 quoted from Fowler2012 and GoogleWillow2025, an order of magnitude | - |
| 93 |  | calc | `6\times10^{-4}` | not run: printed value 6e-4 has one significant digit: a 5 % change cannot be told from rounding (recomputed 6.188e-4; 1.05 x 6e-4 = 6.3e-4). Its input p_eq at 35 mK is checked to three digits at ch:qplatforms:L52 (1.05e-3) | - |

## Part 5 - ch:thermaln - `docs/book/part5/p5_04_thermal_n.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 9 | eq:thermal_slope | derived |  | sympy: d ln p/d ln T = M (1 - p), M = hf/kT | PASS |
| 15 | ch:thermaln:L15 | calc | `6.85` | numeric: slope at 35 mK | PASS |
| 15 | ch:thermaln:L15:16.0 | calc | `16.0` | numeric: slope at 15 mK | PASS |
| 27 | ch:thermaln:L27 | calc | `0.47` | numeric: Doppler limit hbar Gamma/2k_B, Yb+ 369 nm, mK | PASS |
| 27 | ch:thermaln:L27:8.12 | calc | `8.12` | numeric: tau = 1/Gamma, ns | PASS |
| 27 | ch:thermaln:L27:19.6 | calc | `19.6` | numeric: Yb+ 369 nm linewidth Gamma/2pi from the lifetime, MHz | PASS |
| 36 | ch:thermaln:L36 | calc | `17.5` | numeric: temperature at which M = hf/k_BT doubles from 35 mK, mK | PASS |
| 37 | ch:thermaln:L37 | calc | `1.05\times10^{-3}` | numeric: p_eq at 35 mK | PASS |
| 37 | ch:thermaln:L37:1.1\times10^{-6} | calc | `1.1\times10^{-6}` | numeric: p_eq at 17.5 mK | PASS |
| 37 | ch:thermaln:L37:6.86 | calc | `6.86` | numeric: M at 35 mK | PASS |
| 37 | ch:thermaln:L37:13.7 | calc | `13.7` | numeric: M at 17.5 mK | PASS |

## Part 5 - ch:walls - `docs/book/part5/p5_05_coherence_optimum.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 7 | eq:pdecomp | derived |  | sympy: independent error channels add to first order | PASS |
| 22 | eq:t1star | derived |  | sympy: T1* = T1,free r/(1+r) solves a/T1^2 = b/(T1,free - T1)^2 | PASS |
| 36 | ch:walls:L36 | derived | `0.50` | numeric: T1*/T1,free at a/b = 1 | PASS |
| 36 | ch:walls:L36:0.63 | derived | `0.63` | numeric: T1*/T1,free at a/b = 3 | PASS |
| 36 | ch:walls:L36:0.76 | derived | `0.76` | numeric: T1*/T1,free at a/b = 10 | PASS |
| 44 | ch:walls:L44 | observed | `0.3` | file `docs/book/iam.bib`: T1 of tantalum transmons, lower end, ms (cited title) | PASS |
| 44 | ch:walls:L44:0.5 | observed | `0.5` | file `docs/book/iam.bib`: T1 of tantalum transmons, upper end, ms (cited title) | PASS |
| 59 |  | observed | `68` | not run: measured, source not named | - |
| 59 |  | observed | `89` | not run: measured, source not named | - |
| 79 | eq:walls_mhi | conjecture |  | sympy: minimum of A/P + BP + p_ctrl at P* = sqrt(A/B), value 2 sqrt(AB) | PASS |
| 85 | ch:walls:L85 | calc | `0.0121` | numeric: eta^2 (pi/2)^2 at eta = 0.07 | PASS |
| 85 |  | calc | `0.07` | not run: input: Lamb-Dicke parameter eta = 0.07 (illustrative); the prefactor eta^2 (pi/2)^2 is checked by ch:walls:L85 | - |
| 91 |  | observed | `29` | not run: measured, source not named | - |
| 92 |  | observed | `3.0` | not run: measured, source not named | - |
| 101 | ch:walls:L101 | calc | `\le4\times10^{-6}` | numeric: largest gap between the exact and linear forms, p <= 2e-3, alpha C <= 1 | PASS |
| 101 | ch:walls:L101:10^{-3} | calc | `10^{-3}` | numeric: relative gap between exact and linear crosstalk forms | PASS |
| 106 | ch:walls:L106 | calc | `1.15` | numeric: ln200/ln100 | PASS |

## Part 5 - ch:qplatforms - `docs/book/part5/p5_10_qubit_platforms.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 22 |  | calc | `10` | not run: input: surface-code threshold of about 10^-2 per operation (Fowler2012, doi 10.1103/physreva.86.032324) | - |
| 25 |  | observed | `0.2` | not run: measured, source not named | - |
| 32 | eq:qp_peq | none |  | sympy: p_eq = 1/(1+e^M) from detailed balance | PASS |
| 37 | eq:qp_pth | derived |  | sympy: p_th(t) = Gamma_up t = p_eq t/T1 | PASS |
| 51 | ch:qplatforms:L51 | calc | `6.86` | numeric: M transmon 35 mK | PASS |
| 52 | ch:qplatforms:L52 | calc | `1.05\times10^{-3}` | numeric: p_eq 35 mK | PASS |
| 52 |  | calc | `68` | not run: input: mean T1 = 68 us of the 105-qubit processor (GoogleWillow2025), restated; checked at ch:qplatforms:L110 | - |
| 53 | ch:qplatforms:L53 | calc | `6.2\times10^{-7}` | numeric: p_th(t_g) | PASS |
| 53 |  | calc | `40` | not run: input: illustrative gate time t_g = 40 ns (book's choice) | - |
| 53 |  | calc | `10` | not run: input: illustrative two-qubit error near 10^-3 (book's round figure) | - |
| 54 |  | calc | `6\times10^{-4}` | not run: printed value 6e-4 (the book says near 6e-4) has one significant digit: a 5 % change cannot be told from rounding (recomputed 6.188e-4; 1.05 x 6e-4 = 6.3e-4). Its input p_eq at 35 mK is checked to three digits at ch:qplatforms:L52 (1.05e-3) | - |
| 55 | ch:qplatforms:L55 | calc | `16.0` | numeric: M at 15 mK | PASS |
| 55 |  | calc | `1.1\times10^{-7}` | not run: printed value 1.1x10^-7 has two significant digits: a 5 % change cannot be told from rounding (recomputed 1.128e-7); the precise value is checked at ch:qplatforms:L55 (M = 16.0) | - |
| 69 | ch:qplatforms:L69 | calc | `6.86` | numeric: M transmon | PASS |
| 69 | ch:qplatforms:L69:1.1\times10^{-3} | calc | `1.1\times10^{-3}` | numeric: p_eq transmon | PASS |
| 69 |  | calc | `35` | not run: input: transmon effective temperature 35 mK (Jin2015), table operating point | - |
| 69 |  | calc | `15` | not run: input: mixing-chamber temperature 15 mK, table operating point | - |
| 69 |  | calc | `1.1\times10^{-7}` | not run: printed value 1.1x10^-7 has two significant digits: a 5 % change cannot be told from rounding (recomputed 1.128e-7); the precise value is checked at ch:qplatforms:L55 (M = 16.0) | - |
| 70 | ch:qplatforms:L70 | calc | `0.48` | numeric: M fluxonium 0.2 GHz 20 mK | PASS |
| 70 | ch:qplatforms:L70:2.4 | calc | `2.4` | numeric: M fluxonium 1 GHz | PASS |
| 70 | ch:qplatforms:L70:0.38 | calc | `0.38` | numeric: p_eq fluxonium 0.2 GHz | PASS |
| 70 | ch:qplatforms:L70:0.083 | calc | `0.083` | numeric: p_eq fluxonium 1 GHz | PASS |
| 70 |  | calc | `0.2` | not run: input: fluxonium gap, lower end 0.2 GHz of the stated range | - |
| 70 |  | calc | `20` | not run: input: fluxonium bath temperature 20 mK, table operating point | - |
| 71 | ch:qplatforms:L71 | calc | `5\times10^{-4}` | numeric: M, 43Ca+ hyperfine at 300 K | PASS |
| 71 | ch:qplatforms:L71:2\times10^{-3} | calc | `2\times10^{-3}` | numeric: M, 171Yb+ at 300 K | PASS |
| 71 |  | calc | `3.2` | not run: input: hyperfine splitting of 43Ca+, 3.2 GHz (published atomic constant, rounded); used in ch:qplatforms:L71 | - |
| 71 |  | calc | `8.0` | not run: input: hyperfine splitting of 137Ba+, 8.0 GHz (published atomic constant, rounded) | - |
| 71 |  | calc | `12.6` | not run: input: hyperfine splitting of 171Yb+, 12.6 GHz (published atomic constant, rounded); used in ch:qplatforms:L71:2\times10^{-3} | - |
| 71 |  | calc | `300` | not run: input: room temperature 300 K, table operating point | - |
| 72 | ch:qplatforms:L72 | calc | `0.47` | numeric: Doppler limit Yb+, mK | PASS |
| 72 | ch:qplatforms:L72:0.10 | calc | `0.10` | numeric: M axial 1 MHz at Doppler | PASS |
| 72 | ch:qplatforms:L72:0.20 | calc | `0.20` | numeric: M axial 2 MHz | PASS |
| 72 | ch:qplatforms:L72:9.3 | calc | `9.3` | numeric: n-bar 1 MHz | PASS |
| 72 | ch:qplatforms:L72:4.4 | calc | `4.4` | numeric: n-bar 2 MHz | PASS |
| 73 | ch:qplatforms:L73 | calc | `0.0016` | numeric: M Rydberg 10 GHz 300 K | PASS |
| 73 | ch:qplatforms:L73:0.016 | calc | `0.016` | numeric: M Rydberg 100 GHz | PASS |
| 73 | ch:qplatforms:L73:620 | calc | `620` | numeric: n-bar 10 GHz at 300 K (two significant figures) | PASS |
| 73 | ch:qplatforms:L73:62 | calc | `62` | numeric: n-bar 100 GHz 300 K | PASS |
| 73 | ch:qplatforms:L73:0.12 | calc | `0.12` | numeric: M Rydberg 10 GHz at 4 K | PASS |
| 73 | ch:qplatforms:L73:1.2 | calc | `1.2` | numeric: M 100 GHz at 4 K | PASS |
| 73 |  | calc | `10` | not run: input: Rydberg transition range, lower end 10 GHz; used in ch:qplatforms:L73 | - |
| 73 |  | calc | `-100` | not run: input: Rydberg transition range, upper end 100 GHz (the scan read '10--100' as -100); used in ch:qplatforms:L73:0.016 | - |
| 73 |  | calc | `300` | not run: input: radiation-field temperature 300 K, table operating point | - |
| 74 | ch:qplatforms:L74 | calc | `4.6\times10^{-4}` | numeric: M NV at 300 K | PASS |
| 74 | ch:qplatforms:L74:0.034 | calc | `0.034` | numeric: M NV at 4 K | PASS |
| 74 | ch:qplatforms:L74:10^{-3} | calc | `10^{-3}` | numeric: NV sublevels equal to 1e-3 at 300 K | PASS |
| 74 |  | calc | `2.87` | not run: input: NV zero-field splitting D = 2.87 GHz (published constant); used in ch:qplatforms:L74 | - |
| 74 |  | calc | `300` | not run: input: lattice temperature 300 K, table operating point | - |
| 75 | ch:qplatforms:L75 | calc | `7.2` | numeric: M spin 15 GHz 0.1 K | PASS |
| 75 | ch:qplatforms:L75:0.11 | calc | `0.11` | numeric: M spin 3.5 GHz 1.5 K | PASS |
| 75 | ch:qplatforms:L75:7.5\times10^{-4} | calc | `7.5\times10^{-4}` | numeric: p_eq 15 GHz 0.1 K | PASS |
| 75 | ch:qplatforms:L75:0.47 | calc | `0.47` | numeric: p_eq 3.5 GHz 1.5 K | PASS |
| 75 |  | calc | `3.5` | not run: input: silicon Zeeman gap, lower end 3.5 GHz of the stated range; used in ch:qplatforms:L75:0.11 | - |
| 75 |  | calc | `-15` | not run: input: silicon Zeeman gap, upper end 15 GHz (the scan read '3.5--15' as -15); used in ch:qplatforms:L75 | - |
| 75 |  | calc | `0.1` | not run: input: silicon electron temperature 0.1 K, table operating point | - |
| 75 |  | calc | `1.5` | not run: input: hot-operation temperature 1.5 K (Yang2020hot); checked as a published value at ch:qplatforms:L260 | - |
| 76 | ch:qplatforms:L76 | calc | `30.9` | numeric: M photon 1550 nm 300 K | PASS |
| 76 | ch:qplatforms:L76:3.7\times10^{-14} | calc | `3.7\times10^{-14}` | numeric: thermal occupation | PASS |
| 76 |  | calc | `1550` | not run: input: photon wavelength 1550 nm; used in ch:qplatforms:L76 | - |
| 76 |  | calc | `300` | not run: input: waveguide temperature 300 K, table operating point | - |
| 82 | ch:qplatforms:L82 | calc | `10^{-28}` | numeric: k_B T ln2 for atoms at 10 uK | PASS |
| 83 | ch:qplatforms:L83 | calc | `3\times10^{-21}` | numeric: k_B T ln2 at 300 K | PASS |
| 93 | ch:qplatforms:L93 | interp | `30.9` | numeric: M photon 1550 nm at 300 K (lesson three) | PASS |
| 102 | ch:qplatforms:L102 | observed | `0.1` | numeric: residual excited-state occupation, 3D transmon (Jin 2015) | PASS |
| 110 | ch:qplatforms:L110 | observed | `68` | numeric: mean T1 of the 105-qubit processor | PASS |
| 110 | ch:qplatforms:L110:89 | observed | `89` | numeric: mean T2,CPMG of the 105-qubit processor | PASS |
| 116 | ch:qplatforms:L116 | observed | `0.3` | numeric: tantalum transmon T1 above 0.3 ms (Place 2021) | PASS |
| 116 | ch:qplatforms:L116:0.5 | observed | `0.5` | numeric: tantalum transmon T1 approaching 0.5 ms (Wang 2022) | PASS |
| 132 |  | observed | `1.0` | not run: measured, source not named | - |
| 132 |  | observed | `1.4` | not run: measured, source not named | - |
| 148 | ch:qplatforms:L148 | observed | `1.48` | numeric: fluxonium Ramsey T2* (Somoroff 2023) | PASS |
| 148 | ch:qplatforms:L148:0.9999 | observed | `0.9999` | numeric: fluxonium single-qubit gate fidelity (Somoroff 2023) | PASS |
| 149 | ch:qplatforms:L149 | observed | `99.922` | numeric: fluxonium CZ fidelity via transmon coupler (Ding 2023) | PASS |
| 149 | ch:qplatforms:L149:7.8\times10^{-4} | observed | `7.8\times10^{-4}` | numeric: eps = -ln(1-p) at 99.922 % | PASS |
| 150 | ch:qplatforms:L150 | calc | `12.8` | numeric: threshold 1e-2 over eps 7.8e-4 | PASS |
| 152 | ch:qplatforms:L152 | calc | `2.4` | numeric: M fluxonium 1 GHz 20 mK | PASS |
| 152 |  | calc | `8` | not run: printed value 8 (per cent) has one significant digit: a 5 % change cannot be told from rounding (recomputed 8.32); the precise value is checked at ch:qplatforms:L152 (M = 2.4) | - |
| 153 | ch:qplatforms:L153 | calc | `0.48` | numeric: M fluxonium 0.2 GHz | PASS |
| 153 | ch:qplatforms:L153:38 | calc | `38` | numeric: p_eq 0.2 GHz, per cent | PASS |
| 153 |  | calc | `0.2` | not run: input: fluxonium gap 0.2 GHz (lower end of the range of line 70); used in ch:qplatforms:L153 | - |
| 162 | ch:qplatforms:L162 | calc | `10^{-3}` | numeric: ion hyperfine M of order 1e-3 at 300 K | PASS |
| 164 | ch:qplatforms:L164 | calc | `0.47` | numeric: Doppler limit | PASS |
| 165 | ch:qplatforms:L165 | calc | `9.3` | numeric: n-bar at the Doppler limit, 1 MHz | PASS |
| 176 |  | observed | `29` | not run: measured, source not named | - |
| 177 |  | observed | `3.0` | not run: measured, source not named | - |
| 178 |  | observed | `7.9` | not run: measured, source not named | - |
| 179 |  | observed | `1.57\times10^{-3}` | not run: measured, source not named | - |
| 179 |  | observed | `4.64\times10^{-3}` | not run: measured, source not named | - |
| 181 |  | observed | `12000` | not run: measured, source not named | - |
| 181 |  | observed | `4200` | not run: measured, source not named | - |
| 193 |  | observed | `3.3` | not run: measured, source not named | - |
| 194 |  | observed | `10` | not run: measured, source not named | - |
| 204 |  | observed | `8.4` | not run: measured, source not named | - |
| 205 |  | observed | `9.4` | not run: measured, source not named | - |
| 214 | ch:qplatforms:L214 | calc | `620` | numeric: n-bar 10 GHz at 300 K (two significant figures) | PASS |
| 216 | ch:qplatforms:L216 | calc | `7.8` | numeric: n-bar 10 GHz at 4 K | PASS |
| 216 | ch:qplatforms:L216:0.43 | calc | `0.43` | numeric: n-bar 100 GHz at 4 K | PASS |
| 222 | eq:qp_rydberg | derived |  | sympy: Omega* = (a/2b)^(1/3) minimises a/Omega + b Omega^2 | PASS |
| 227 | ch:qplatforms:L227 | observed | `99.5` | numeric: neutral-atom parallel CZ fidelity (Evered 2023) | PASS |
| 228 | ch:qplatforms:L228 | calc | `2.0` | numeric: threshold over eps | PASS |
| 228 | ch:qplatforms:L228:5.0\times10^{-3} | observed | `5.0\times10^{-3}` | numeric: eps = -ln(1-p) at 99.5 % (neutral atoms) | PASS |
| 240 |  | calc | `2.87` | not run: input: NV zero-field splitting D = 2.87 GHz (published constant), restated from Table tab:qp_which; used in ch:qplatforms:L241 | - |
| 241 | ch:qplatforms:L241 | calc | `4.6\times10^{-4}` | numeric: M NV 300 K | PASS |
| 244 |  | observed | `73` | not run: measured, source not named | - |
| 249 | ch:qplatforms:L249 | calc | `14.3` | numeric: threshold over eps | PASS |
| 249 | ch:qplatforms:L249:7.0\times10^{-4} | observed | `7.0\times10^{-4}` | numeric: eps = -ln(1-p) at 99.93 % (NV gate) | PASS |
| 249 |  | observed | `99.93` | not run: measured, source not named | - |
| 259 |  | observed | `0.1` | not run: measured, source not named | - |
| 260 | ch:qplatforms:L260 | observed | `1.5` | numeric: hot silicon unit cell at 1.5 K (Yang 2020) | PASS |
| 260 |  | observed | `3.5` | not run: measured, source not named | - |
| 262 | ch:qplatforms:L262 | calc | `0.11` | numeric: M at 3.5 GHz 1.5 K | PASS |
| 262 | ch:qplatforms:L262:0.47 | calc | `0.47` | numeric: p_eq at 3.5 GHz 1.5 K | PASS |
| 262 |  | calc | `3.5` | not run: input: 3.5 GHz control frequency of the hot unit cell (Yang2020hot), restated; used in ch:qplatforms:L262 | - |
| 262 |  | calc | `1.5` | not run: input: 1.5 K hot-operation temperature, restated; checked at ch:qplatforms:L260 | - |
| 267 | ch:qplatforms:L267 | observed | `99.5` | numeric: silicon two-qubit fidelity 99.5 % (Xue 2022, Noiri 2022) | PASS |
| 268 | ch:qplatforms:L268 | calc | `5.0\times10^{-3}` | numeric: eps = -ln(1 - p) at 99.5 % fidelity | PASS |
| 268 | ch:qplatforms:L268:5.0\times10^{-3} | observed | `5.0\times10^{-3}` | numeric: eps = -ln(1-p) at 99.5 % (silicon) | PASS |
| 268 |  | calc | `99.5` | not run: restates ch:qplatforms:L267 (published silicon two-qubit fidelity 99.5 %) | - |
| 269 | ch:qplatforms:L269 | calc | `2.0` | numeric: threshold over eps at 99.5 % | PASS |
| 277 | ch:qplatforms:L277 | calc | `30.9` | numeric: M optical photon | PASS |
| 277 | ch:qplatforms:L277:3.7\times10^{-14} | calc | `3.7\times10^{-14}` | numeric: thermal occupation | PASS |
| 281 | ch:qplatforms:L281 | observed | `99.98` | numeric: photonic SPAM fidelity (PsiQuantum 2025) | PASS |
| 282 | ch:qplatforms:L282 | observed | `99.50` | numeric: photonic HOM visibility (PsiQuantum 2025) | PASS |
| 282 | ch:qplatforms:L282:99.22 | observed | `99.22` | numeric: photonic two-qubit fusion fidelity (PsiQuantum 2025) | PASS |
| 283 | ch:qplatforms:L283 | observed | `99.72` | numeric: photonic chip-to-chip interconnect fidelity (PsiQuantum 2025) | PASS |
| 296 |  | observed | `14.5` | not run: measured, source not named | - |
| 296 |  | observed | `12.4` | not run: measured, source not named | - |
| 297 |  | observed | `0.5` | not run: measured, source not named | - |

## Part 5 - ch:cmos - `docs/book/part5/p5_06_cmos.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 7 | ch:cmos:L7 | none |  | sympy: k_B T_j ln2 at 75 C = 3.33e-21 J = 0.021 eV | PASS |
| 11 |  | calc | `4.3` | not run: input: base clock 4.3 GHz (AMD9950X, maker specification); used in ch:cmos:L12:576 | - |
| 12 | ch:cmos:L12 | calc | `20` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 12 | ch:cmos:L12:576 | calc | `576` | numeric: E_sw/(k_B T_j ln2), upper transistor count | PASS |
| 12 | ch:cmos:L12:593 | calc | `593` | numeric: E_sw/(k_B T_j ln2), lower count | PASS |
| 13 | ch:cmos:L13 | calc | `399` | numeric: M = E_sw/k_B T_j, upper count | PASS |
| 13 | ch:cmos:L13:411 | calc | `411` | numeric: M, lower count | PASS |
| 14 | ch:cmos:L14 | openprob | `0.0017` | numeric: Landauer floor on the chip gauge, k_B T_j ln2 / E_sw | PASS |
| 17 |  | openprob | `20.0\times10^9` | not run: input: transistor count 20.0e9 from die-level reports (no maker figure; the book marks it openprob); used in ch:cmos:L12:593 | - |
| 17 |  | openprob | `20.6\times10^9` | not run: input: transistor count 20.6e9 from die-level reports (no maker figure; the book marks it openprob); used in ch:cmos:L12:576 | - |
| 26 | ch:cmos:L26 | derived | `60` | numeric: (k_B T/q) ln 10 at 300 K, mV | PASS |
| 31 | ch:cmos:L31 | observed | `1.57` | numeric: Koomey doubling time 1.57 years | PASS |
| 31 |  | observed | `2.7` | not run: measured, source not named | - |
| 48 | eq:reliableswitch | calc |  | sympy: E_min = k_B T ln(1/p): Landauer at p = 1/2 | PASS |
| 52 |  | calc | `10` | not run: input: error probability p = 10^-15 (book's choice); the floor at it is checked at ch:cmos:L53 | - |
| 53 | ch:cmos:L53 | calc | `34.5` | numeric: ln(1/p) at p = 1e-15 | PASS |
| 53 | ch:cmos:L53:49.8 | calc | `49.8` | numeric: in Landauer units | PASS |
| 53 | ch:cmos:L53:83.0 | calc | `83.0` | numeric: p = 1e-25, Landauer units | PASS |
| 53 |  | calc | `10` | not run: input: error probability p = 10^-25 (book's choice); the floor at it is checked at ch:cmos:L53:83.0 | - |
| 58 | ch:cmos:L58 | derived | `64.6` | numeric: 1 - kappa^-3, kappa = sqrt2 | PASS |
| 59 | ch:cmos:L59 | derived | `29.3` | numeric: 1 - kappa^-1 | PASS |
| 65 | ch:cmos:L65 | derived | `64.6` | numeric: 1 - kappa^-3, kappa = sqrt2 | PASS |
| 66 | ch:cmos:L66 | derived | `29.3` | numeric: 1 - kappa^-1 | PASS |
| 82 | ch:cmos:L82 | calc | `8.6` | numeric: k_B T_j ln2, 105 C over 75 C | PASS |
| 102 | ch:cmos:L102 | calc | `7.6` | numeric: 1 - 1.83/1.98 | PASS |
| 102 |  | calc | `1.83` | not run: input: quoted clock 1.83 GHz of one die (book's worked case); used in ch:cmos:L102 | - |
| 102 |  | calc | `1.98` | not run: input: quoted clock 1.98 GHz of one die (book's worked case); used in ch:cmos:L102 | - |
| 108 | ch:cmos:L108 | calc | `40.7` | numeric: 1 - (1 - 0.707) 253/125 | PASS |
| 108 |  | calc | `70.7` | not run: input: the step of an unnamed generation pair read on mixed power definitions; the chips' power, count and clock are not stated in the book, so nothing to recompute (the 40.7 % derived from it is checked at ch:cmos:L108) | - |
| 109 | ch:cmos:L109 | calc | `8.6` | numeric: k_B T_j ln2, 105 C over 75 C | PASS |

## Part 5 - ch:chipgen - `docs/book/part5/p5_11_chip_generations.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 18 | ch:chipgen:L18 | calc | `3.33\times10^{-21}` | numeric: Landauer floor at 75 C | PASS |
| 18 | ch:chipgen:L18:3.62\times10^{-21} | calc | `3.62\times10^{-21}` | numeric: Landauer floor at 105 C | PASS |
| 18 | ch:chipgen:L18:8.6 | calc | `8.6` | numeric: ratio | PASS |
| 20 | ch:chipgen:L20 | calc | `1.92` | numeric: E_sw, upper transistor count, 1e-18 J | PASS |
| 20 | ch:chipgen:L20:1.98\times10^{-18} | calc | `1.98\times10^{-18}` | numeric: E_sw, lower count | PASS |
| 21 | ch:chipgen:L21 | calc | `0.0017` | numeric: floor on the gauge at 75 C (both counts) | PASS |
| 21 | ch:chipgen:L21:0.0018 | calc | `0.0018` | numeric: floor at 105 C, lower count | PASS |
| 21 | ch:chipgen:L21:0.0019 | calc | `0.0019` | numeric: floor at 105 C, upper count | PASS |
| 27 |  | calc | `10` | not run: input: error probability p = 10^-15 of the reliability-floor curve in the caption (book's choice; Chapter ch:cmos) | - |
| 28 | ch:chipgen:L28 | calc | `1.92` | numeric: E_sw | PASS |
| 28 | ch:chipgen:L28:1.98\times10^{-18} | calc | `1.98\times10^{-18}` | numeric: E_sw | PASS |
| 48 | ch:chipgen:L48 | calc | `40.7` | numeric: 1 - (1 - 0.707) 253/125 | PASS |
| 48 |  | calc | `70.7` | not run: input: the step of an unnamed generation pair read on mixed power definitions; chip inputs not stated in the book (restates ch:cmos line 108; the 40.7 % derived from it is checked at ch:chipgen:L48) | - |
| 49 | ch:chipgen:L49 | calc | `7.6` | numeric: 1 - 1.83/1.98 | PASS |
| 49 |  | calc | `1.83` | not run: input: quoted clock 1.83 GHz (book's worked case); used in ch:chipgen:L49 | - |
| 49 |  | calc | `1.98` | not run: input: quoted clock 1.98 GHz (book's worked case); used in ch:chipgen:L49 | - |
| 54 | ch:chipgen:L54 | derived | `64.6` | numeric: constant-field node step | PASS |
| 54 | ch:chipgen:L54:29.3 | derived | `29.3` | numeric: fixed-voltage node step | PASS |
| 55 | ch:chipgen:L55 | interp | `64.6` | numeric: constant-field step per halving of area, 1 - kappa^-3 | PASS |
| 56 | ch:chipgen:L56 | interp | `29.3` | numeric: fixed-voltage step per halving of area, 1 - kappa^-1 | PASS |
| 57 | ch:chipgen:L57 | interp | `64.6` | numeric: constant-field step 64.6 % (upper reference) | PASS |
| 68 | eq:cg_nfloor | derived |  | sympy: generations to the floor: (1-s)^n = 1/R | PASS |
| 72 | ch:chipgen:L72 | calc | `6.2` | numeric: R = 600, constant-field rate | PASS |
| 72 | ch:chipgen:L72:18.5 | calc | `18.5` | numeric: R = 600, fixed-voltage rate | PASS |
| 72 |  | calc | `600` | not run: input: illustrative R = 600, a round figure for the 576-593 floors of ch:cmos:L12:576; used in ch:chipgen:L72 | - |

## Part 6 - ch:bridge - `docs/book/part6/p6_01_bridge.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 26 | ch:bridge:L26 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy per maintained site | PASS |
| 52 | eq:landauer | observed |  | sympy: E_bit = k_B T ln 2 from erasing one bit | PASS |
| 60 | ch:bridge:L60 | observed | `310.15` | file `CANON/iam_canon.json`: cell nucleus temperature 310.15 K | PASS |
| 66 |  | calc | `67.4` | not run: locked value H0 = 67.16 (photon sector) restated in the caption; the 67.4 of the inventory row is no longer printed at line 66 | - |
| 175 | ch:bridge:L175 | conjecture | `310` | file `CANON/iam_canon.json`: cell surface temperature, 310 K | PASS |
| 222 | ch:bridge:L222 | calc | `36.1` | numeric: decades from 1e-10 m to c/H0 (H0 = 67.4, ch:bridge line 66) | PASS |
| 235 | ch:bridge:L235 | calc | `6.2\times10^{-8}` | numeric: T_BH, 1 M_sun | PASS |
| 235 | ch:bridge:L235:5.9\times10^{-31} | calc | `5.9\times10^{-31}` | numeric: k_B T ln2 at the horizon, J | PASS |
| 235 | ch:bridge:L235:1.5\times10^{77} | calc | `1.5\times10^{77}` | numeric: bits on the horizon | PASS |
| 235 | ch:bridge:L235:8.9\times10^{46} | calc | `8.9\times10^{46}` | numeric: N k_B T ln2 = Mc^2/2, J | PASS |
| 236 | ch:bridge:L236 | calc | `6.2\times10^{-14}` | numeric: T_BH, 1e+06 M_sun | PASS |
| 236 | ch:bridge:L236:5.9\times10^{-37} | calc | `5.9\times10^{-37}` | numeric: k_B T ln2 at the horizon, J | PASS |
| 236 | ch:bridge:L236:1.5\times10^{89} | calc | `1.5\times10^{89}` | numeric: bits on the horizon | PASS |
| 236 | ch:bridge:L236:8.9\times10^{52} | calc | `8.9\times10^{52}` | numeric: N k_B T ln2 = Mc^2/2, J | PASS |
| 236 |  | calc | `10` | not run: input: horizon of 10^6 solar masses (table row label, book's choice); its entries are checked at ch:bridge:L236 | - |
| 237 | ch:bridge:L237 | calc | `2.0\times10^{-2}` | numeric: 20 mK in K | PASS |
| 237 | ch:bridge:L237:1.9\times10^{-25} | calc | `1.9\times10^{-25}` | numeric: k_B T ln2 at 20 mK | PASS |
| 237 |  | calc | `20` | not run: input: qubit temperature 20 mK (table row label); its entries are checked at ch:bridge:L237 | - |
| 238 | ch:bridge:L238 | calc | `3.1\times10^{2}` | numeric: 310.15 K | PASS |
| 238 | ch:bridge:L238:3.0\times10^{-21} | calc | `3.0\times10^{-21}` | numeric: k_B T ln2 at 310.15 K | PASS |
| 238 | ch:bridge:L238:2.8\times10^{7} | calc | `2.8\times10^{7}` | numeric: CpG sites (ch:landauer) | PASS |
| 238 | ch:bridge:L238:8.4\times10^{-14} | calc | `8.4\times10^{-14}` | numeric: N k_B T ln2 | PASS |
| 238 | ch:bridge:L238:310.15 | calc | `310.15` | file `CANON/iam_canon.json`: cell nucleus row, T = 310.15 K | PASS |

## Part 6 - ch:astrogenetics - `docs/book/part6/p6_00b_astrogenetics.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 40 | ch:astrogenetics:L40 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy | PASS |
| 40 | ch:astrogenetics:L40:4.9 | measured | `4.9` | numeric: Landauer units | PASS |
| 63 | ch:astrogenetics:L63 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A at a coin flip: 1/0.330263 (canon floor) | PASS |
| 63 | ch:astrogenetics:L63:4.45 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A at a coin flip: 1/(P H(eps0)) | PASS |
| 63 | ch:astrogenetics:L63:0.2043 | calc | `0.2043` | numeric: H(eps0), bits | PASS |
| 88 | ch:astrogenetics:L88 | derived | `0.910` | file `CANON/iam_canon.json`: 1/P | PASS |
| 88 | ch:astrogenetics:L88:310 | derived | `310` | file `CANON/iam_canon.json`: the floor is set at 310 K (cell temperature) | PASS |
| 89 | ch:astrogenetics:L89 | calibrated | `0.330263` | heavy file `docs/verification/scripts/verify_astrogenetics_book_output.txt`: measured: printed value found in verify_astrogenetics_book_output.txt, a file the chapter names | PASS |
| 90 | ch:astrogenetics:L90 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A at a coin flip | PASS |
| 90 | ch:astrogenetics:L90:4.45 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A at a coin flip | PASS |
| 98 | ch:astrogenetics:L98 | derived |  | sympy: H(beta) is symmetric: H(beta) = H(1 - beta) | PASS |
| 116 | ch:astrogenetics:L116 | calc | `2.40` | numeric: white-dwarf gauge full at the Chandrasekhar mass, A = 1.44/0.6 | PASS |
| 139 | ch:astrogenetics:L139 | measured | `1.016` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 senescent cultures, methylated-channel Met-A, highest | PASS |
| 140 | ch:astrogenetics:L140 | measured | `0.695` | heavy file `docs/verification/scripts/verify_astrogenetics_book_output.txt`: measured: printed value found in verify_astrogenetics_book_output.txt, a file the chapter names | PASS |
| 140 | ch:astrogenetics:L140:1.120 | measured | `1.120` | heavy file `docs/verification/scripts/verify_astrogenetics_book_output.txt`: measured: printed value found in verify_astrogenetics_book_output.txt, a file the chapter names | PASS |
| 141 | ch:astrogenetics:L141 | measured | `0.664` | heavy file `docs/verification/scripts/verify_astrogenetics_book_output.txt`: measured: printed value found in verify_astrogenetics_book_output.txt, a file the chapter names | PASS |
| 141 | ch:astrogenetics:L141:0.975 | measured | `0.975` | heavy file `docs/verification/scripts/verify_astrogenetics_book_output.txt`: measured: printed value found in verify_astrogenetics_book_output.txt, a file the chapter names | PASS |
| 280 | ch:astrogenetics:L280 | derived | `0.15765` | numeric: beta_m | PASS |
| 283 | ch:astrogenetics:L283 | measured | `0.2` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: 18th chain eta below Planck 2018, per cent | PASS |
| 283 | ch:astrogenetics:L283:0.3 | measured | `0.3` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: 18th chain eta from nucleosynthesis with deuterium, sigma | PASS |
| 285 | ch:astrogenetics:L285 | openprob | `1.142\times10^{-123}` | numeric: rho_L/rho_vac from (2/pi)(l_P/l_H)^2 (Ob/Om) sqrt(OL) | PASS |
| 286 | ch:astrogenetics:L286 | fitted | `1.133\times10^{-123}` | numeric: measured rho_L/rho_vac | PASS |
| 286 | ch:astrogenetics:L286:0.79 | fitted | `0.79` | numeric: expression above the measured ratio, per cent | PASS |
| 288 | ch:astrogenetics:L288 | observed | `0.5` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Ob/Om = (3/16) sqrt(OL) holds to 0.5 % on the CMB-only chain | PASS |

## Part 6 - ch:landauer - `docs/book/part6/p6_02_landauer.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 12 | eq:ebit | calc | `2.968\times10^{-21}` | numeric: k_B T_body ln2, J | PASS |
| 17 | ch:landauer:L17 | calc | `1.787` | numeric: per mole of bits, kJ | PASS |
| 27 | eq:M | calc | `20.94` | numeric: M = dG_ATP/(R T_body) | PASS |
| 48 | eq:bitsperATP | calc | `30.21` | numeric: M/ln2 | PASS |
| 65 | ch:landauer:L65 | measured | `0.024` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy copy error, lowest of 56 cell types | PASS |
| 65 | ch:landauer:L65:0.042 | measured | `0.042` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: highest | PASS |
| 70 | ch:landauer:L70 | calc | `0.69` | numeric: ln 2 | PASS |
| 71 | ch:landauer:L71 | calc | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy per site 3.41 k_B T (figure caption) | PASS |
| 71 | ch:landauer:L71:20.94 | calc | `20.94` | numeric: M = dG_ATP/(R T) per ATP (figure caption) | PASS |
| 72 | ch:landauer:L72 | calc | `1.00` | numeric: Landauer floor in Landauer units | PASS |
| 72 | ch:landauer:L72:4.92 | calc | `4.92` | numeric: E_hold/ln2 (E_hold canon) | PASS |
| 72 | ch:landauer:L72:30.21 | calc | `30.21` | numeric: M/ln2 | PASS |
| 88 | ch:landauer:L88 | calc | `399` | numeric: M = E_sw/k_B T_j, upper count | PASS |
| 88 | ch:landauer:L88:411 | calc | `411` | numeric: 9950X upper end of the range | PASS |
| 88 | ch:landauer:L88:576 | calc | `576` | numeric: E_sw/(k_B T_j ln2), upper count | PASS |
| 88 | ch:landauer:L88:593 | calc | `593` | numeric: 9950X upper end of the range | PASS |
| 88 |  | calc | `9950` | not run: not a number: part of the processor name (Ryzen 9 9950X) | - |
| 88 |  | calc | `348.15` | not run: input: junction temperature 75 C = 348.15 K (Chapter ch:cmos, book line 6 there) | - |
| 89 | ch:landauer:L89 | calc | `20.94` | numeric: M | PASS |
| 89 | ch:landauer:L89:30.21 | calc | `30.21` | numeric: M/ln2 | PASS |
| 89 | ch:landauer:L89:310 | calc | `310` | file `CANON/iam_canon.json`: cell nucleus at 310 K (table tab:p4operating) | PASS |
| 90 | ch:landauer:L90 | calc | `0.693` | numeric: ln 2 | PASS |
| 95 | eq:Mtransmon | derived |  | sympy: M_transmon = ln 2 | PASS |
| 100 | ch:landauer:L100 | calc | `20.94` | numeric: M | PASS |
| 100 | ch:landauer:L100:30.2 | calc | `30.2` | numeric: M/ln2 | PASS |
| 123 | eq:markmargin | calc | `30` | numeric: M/ln2, about 30 | PASS |
| 138 | eq:efloor | calc | `8.38\times10^{-14}` | numeric: N k_B T ln2, N = 28,217,448 | PASS |
| 143 | ch:landauer:L143 | calc | `9.3\times10^5` | numeric: floor in ATP at 54 kJ/mol | PASS |
| 143 | ch:landauer:L143:1.0\times10^6 | calc | `1.0\times10^6` | numeric: floor in ATP at 50 kJ/mol | PASS |
| 149 | ch:landauer:L149 | calc | `54` | file `CANON/iam_canon.json`: dG_ATP = 54 kJ/mol in the figure caption | PASS |
| 150 | ch:landauer:L150 | calc | `9.34\times10^5` | numeric: floor in ATP | PASS |
| 155 |  | calc | `3000` | not run: input: cell volume of about 3000 um^3 (Milo2015) | - |
| 155 |  | calc | `10` | not run: input: ATP turnover of order 1e9 per second (Milo2015) | - |
| 156 | ch:landauer:L156 | calc | `10^{14}` | numeric: ATP over a 24-hour cycle, order 1e14 | PASS |
| 157 | ch:landauer:L157 | calc | `2.0\times10^7` | numeric: 70 % of CpGs methylated | PASS |
| 157 | ch:landauer:L157:10^{-7} | calc | `10^{-7}` | numeric: methyl writing as a fraction of the daily ATP budget, order 1e-7 | PASS |
| 158 | ch:landauer:L158 | calc | `9.3\times10^5` | numeric: floor in ATP at 54 kJ/mol | PASS |
| 173 |  | observed | `0.90` | not run: measured, source not named | - |
| 173 |  | observed | `0.98` | not run: measured, source not named | - |
| 176 | ch:landauer:L176 | calc | `2.3` | numeric: ln(1/0.10) | PASS |
| 176 | ch:landauer:L176:3.9 | calc | `3.9` | numeric: ln(1/0.02) | PASS |
| 176 |  | calc | `0.10` | not run: input: failure rate 10 % = 1 - 0.90, the lower maintenance efficiency of line 173 restated; ln(1/0.10) is checked at ch:landauer:L176 | - |
| 178 | ch:landauer:L178 | calc | `21` | numeric: kT per ATP | PASS |
| 186 | ch:landauer:L186 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy across 56 cell types (Loyfer read-level) | PASS |
| 186 | ch:landauer:L186:3.13 | measured | `3.13` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy across 56 cell types (Loyfer read-level) | PASS |
| 186 | ch:landauer:L186:3.72 | measured | `3.72` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy across 56 cell types (Loyfer read-level) | PASS |
| 187 | ch:landauer:L187 | calc | `4.92` | numeric: 3.41/ln2 | PASS |
| 188 | eq:phi | measured | `0.163` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: phi = E_hold/(M k T): committed record and E_hold/M (canon) | PASS |
| 194 | ch:landauer:L194 | calc | `1.9` | numeric: Hopfield discrimination ln(7) from the enzyme selectivity | PASS |
| 194 | ch:landauer:L194:4.4 | calc | `4.4` | numeric: Hopfield discrimination ln(80) from the enzyme selectivity | PASS |
| 205 | eq:sanchezH | none |  | not run: definition: per-site Shannon entropy of methylation status (Sanchez2016) | - |
| 211 | eq:sanchezER | derived |  | sympy: E_R = I_R k_B T ln2 from Landauer, with H in bits | PASS |
| 249 | ch:landauer:L249 | measured | `0.075` | file `Biological_Physics/MethylPhys/doors/PHASE1_OUTCOME.md`: pipeline offset in beta on the immune identity sites, 450K | PASS |
| 259 | ch:landauer:L259 | calc | `2.968\times10^{-21}` | numeric: k_B T_body ln2 | PASS |
| 260 | ch:landauer:L260 | calc | `20.94` | numeric: M | PASS |
| 261 | ch:landauer:L261 | calc | `30.21` | numeric: M/ln2 | PASS |
| 262 | ch:landauer:L262 | derived | `0.693` | numeric: ln 2 | PASS |
| 263 | ch:landauer:L263 | calc | `399` | numeric: M = E_sw/k_B T_j, upper count | PASS |
| 263 | ch:landauer:L263:411 | calc | `411` | numeric: M, lower count | PASS |
| 263 |  | calc | `9950` | not run: not a number: part of the processor name (AMD 9950X) | - |
| 264 | ch:landauer:L264 | calc | `8.38\times10^{-14}` | numeric: floor for all CpGs, J | PASS |
| 264 | ch:landauer:L264:9.3\times10^5 | calc | `9.3\times10^5` | numeric: floor in ATP at 54 kJ/mol | PASS |
| 267 | ch:landauer:L267 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy, 56 cell types | PASS |
| 267 | ch:landauer:L267:56 | calc | `56` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy read in 56 cell types | PASS |
| 268 | ch:landauer:L268 | measured | `0.163` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: phi, committed record | PASS |
| 269 |  | calc | `450` | not run: not a number: array platform name (450K) | - |

## Part 6 - ch:surface - `docs/book/part6/p6_03_surface.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 13 | eq:H | derived |  | sympy: binary entropy: 1 bit at 1/2, symmetric | PASS |
| 50 | eq:jensen | derived |  | sympy: H(mean beta) >= mean H(beta) (concavity), 200 random sets | PASS |
| 55 |  | derived | `0.9` | not run: input: illustrative site at beta = 0.9 in the Jensen example (book's choice) | - |
| 55 |  | derived | `0.1` | not run: input: illustrative site at beta = 0.1 in the Jensen example (book's choice) | - |
| 57 | eq:meanH | none |  | not run: definition: mean of the per-site entropies, the Met-A statistic | - |
| 66 | ch:surface:L66 | calc | `0.330` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean of the per-site entropies, all 6,000 neutrophil identity sites (bits) | PASS |
| 67 | ch:surface:L67 | calc | `1.000` | numeric: entropy of the mean beta over both channels (beta-bar 0.502, table line 83) | PASS |
| 68 | ch:surface:L68 | calc | `0.325` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: entropy of the mean beta on the methylated channel (bits) | PASS |
| 81 | ch:surface:L81 | calc | `3000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: number of methylated-channel identity sites | PASS |
| 81 | ch:surface:L81:0.941 | calc | `0.941` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean beta, methylated channel | PASS |
| 81 | ch:surface:L81:0.325 | calc | `0.325` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: H(mean beta), methylated channel (bits) | PASS |
| 81 | ch:surface:L81:0.323 | calc | `0.323` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean of per-site H, methylated channel (bits) | PASS |
| 82 | ch:surface:L82 | calc | `3000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: number of unmethylated-channel identity sites | PASS |
| 82 | ch:surface:L82:0.063 | calc | `0.063` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean beta, unmethylated channel | PASS |
| 82 | ch:surface:L82:0.340 | calc | `0.340` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: H(mean beta), unmethylated channel (bits) | PASS |
| 82 | ch:surface:L82:0.337 | calc | `0.337` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean of per-site H, unmethylated channel (bits) | PASS |
| 83 | ch:surface:L83 | calc | `1.000` | numeric: H(beta-bar = 0.502) | PASS |
| 83 | ch:surface:L83:6000 | calc | `6000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: number of identity sites, both channels | PASS |
| 83 | ch:surface:L83:0.502 | calc | `0.502` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean beta over both channels | PASS |
| 83 | ch:surface:L83:0.330 | calc | `0.330` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: mean of per-site H over both channels (bits) | PASS |
| 111 | eq:steady | derived |  | sympy: steady state of gain d(1-beta) and loss u beta | PASS |
| 115 | ch:surface:L115 | calc | `0.727` | numeric: beta_ss at u = 0.03, d = 0.08 | PASS |
| 115 | ch:surface:L115:0.845 | calc | `0.845` | numeric: H(beta_ss), bits | PASS |
| 115 |  | calc | `0.03` | not run: input: illustrative loss rate u = 0.03 of the two-state model (book's choice); beta_ss and H are checked at ch:surface:L115 | - |
| 115 |  | calc | `0.08` | not run: input: illustrative gain rate d = 0.08 of the two-state model (book's choice); beta_ss and H are checked at ch:surface:L115 | - |
| 125 | ch:surface:L125 | calc | `2.8\times10^7` | numeric: CpG sites, one bit each | PASS |

## Part 6 - ch:ledgers - `docs/book/part6/p6_04_ledgers.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 14 | eq:virial | derived |  | sympy: virial theorem 2<K> = k<U> for U homogeneous of degree k | PASS |
| 44 | eq:smarr | calc |  | sympy: Mc^2 = 2 T_H S | PASS |
| 49 | ch:ledgers:L49 | calc | `8.94\times10^{46}` | numeric: T_H S = Mc^2/2 for 1 M_sun, J | PASS |
| 54 | ch:ledgers:L54 | derived | `13.6` | numeric: <K> hydrogen, eV | PASS |
| 55 | ch:ledgers:L55 | derived | `-27.2` | numeric: <V> hydrogen, eV | PASS |
| 91 | ch:ledgers:L91 | openprob | `10^{-7}` | numeric: methylation maintenance share of the cell budget, order 1e-7 | PASS |
| 104 | ch:ledgers:L104 | calc | `8.4\times10^{-14}` | numeric: cell entry N k_B T ln2, J | PASS |
| 109 | ch:ledgers:L109 | calc | `8.4\times10^{-14}` | numeric: cell entry N k_B T ln2, J | PASS |
| 114 | ch:ledgers:L114 | calc | `8.94\times10^{46}` | numeric: T_H S = Mc^2/2 for 1 M_sun, J | PASS |
| 114 | ch:ledgers:L114:2.8\times10^7 | calc | `2.8\times10^7` | numeric: CpG bits | PASS |
| 114 | ch:ledgers:L114:310.15 | calc | `310.15` | file `CANON/iam_canon.json`: cell at 310.15 K (figure caption) | PASS |
| 122 | ch:ledgers:L122 | derived | `13.61` | numeric: <K> hydrogen, eV | PASS |
| 122 | ch:ledgers:L122:-27.21 | derived | `-27.21` | numeric: <V> hydrogen, eV | PASS |
| 125 | ch:ledgers:L125 | calc | `8.94\times10^{46}` | numeric: T_H S = Mc^2/2 for 1 M_sun, J | PASS |
| 126 | ch:ledgers:L126 | derived | `2.14\times10^{-21}` | numeric: k_B T/2 at 310.15 K | PASS |
| 126 | ch:ledgers:L126:310.15 | calc | `310.15` | file `CANON/iam_canon.json`: equipartition row at 310.15 K | PASS |

## Part 6 - ch:floorbreach - `docs/book/part6/p6_05_floorbreach.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 25 | ch:floorbreach:L25 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy 3.41 k_B T per site on molecules | PASS |
| 42 | ch:floorbreach:L42 | calc | `5.4\times10^{69}` | numeric: bits of 1 M_sun horizon over CpG bits | PASS |
| 50 | ch:floorbreach:L50 | calc | `310.15` | file `CANON/iam_canon.json`: cell methylome at 310.15 K (figure caption) | PASS |
| 59 | eq:Amax | calc | `3.03` | file `CANON/iam_canon.json`: Met-A full surface 1/0.330263 (canon floor) | PASS |
| 63 | ch:floorbreach:L63 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A full surface | PASS |
| 72 | ch:floorbreach:L72 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A full surface | PASS |
| 72 | ch:floorbreach:L72:1.099 | calc | `1.099` | file `CANON/iam_canon.json`: P = 1.099 of IAM-A (figure caption) | PASS |
| 73 | ch:floorbreach:L73 | calc | `0.910` | file `CANON/iam_canon.json`: 1/P | PASS |
| 73 | ch:floorbreach:L73:0.0362 | calc | `0.0362` | file `CANON/iam_canon.json`: eps at IAM-A = 1 | PASS |
| 73 | ch:floorbreach:L73:0.032 | calc | `0.032` | numeric: eps0 = 1/(1 + e^(E_hold/k_B T)) | PASS |
| 74 | ch:floorbreach:L74 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A at eps = 1/2 | PASS |
| 80 | ch:floorbreach:L80 | conjecture | `10^{-7}` | numeric: methylation maintenance share of the cell ATP, order 1e-7 | PASS |
| 96 | ch:floorbreach:L96 | measured | `0.695` | heavy file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: senescent IMR90 unmethylated channel, upper end | PASS |

## Part 6 - ch:gauge - `docs/book/part6/p6_06_gauge.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 7 | eq:A | none |  | not run: definition: A = reading / healthy reference (the gauge) | - |
| 22 |  | calc | `0.95` | not run: definition: Normal band 0.95-1.05, a design tolerance A = 1 +- 5 % (stated at line 35) | - |
| 22 |  | calc | `1.05` | not run: definition: Normal band 0.95-1.05, a design tolerance A = 1 +- 5 % (stated at line 35) | - |
| 23 | ch:gauge:L23 | calc | `0.910` | file `CANON/iam_canon.json`: H_min for IAM-A, 1/P | PASS |
| 24 | ch:gauge:L24 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A full surface | PASS |
| 24 | ch:gauge:L24:4.45 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A full surface | PASS |
| 34 | ch:gauge:L34 | derived | `0.910` | file `CANON/iam_canon.json`: H_min for IAM-A, 1/P | PASS |
| 35 |  | derived | `0.95` | not run: definition: Normal band, design tolerance, healthy is A = 1 +- 5 % | - |
| 35 |  | derived | `1.05` | not run: definition: Normal band, design tolerance, healthy is A = 1 +- 5 % | - |
| 38 | ch:gauge:L38 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A full surface | PASS |
| 38 | ch:gauge:L38:4.45 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A full surface | PASS |
| 39 | ch:gauge:L39 | calc | `45` | numeric: C-score far end: var ratio 50 over the baseline | PASS |
| 39 | ch:gauge:L39:50 | derived | `50` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score block size, read from the frozen reference | PASS |
| 39 | ch:gauge:L39:1.1104 | derived | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy baseline: median of the six leave-one-out clustering values | PASS |
| 53 | ch:gauge:L53 | derived |  | sympy: H(beta) at a site held methylated equals H of the error rate | PASS |
| 101 | ch:gauge:L101 | measured | `0.685` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 senescent, unmethylated channel, lowest culture | PASS |
| 101 | ch:gauge:L101:0.695 | measured | `0.695` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 senescent, unmethylated channel, highest culture | PASS |
| 102 | ch:gauge:L102 | measured | `1.077` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, methylated channel, lowest culture | PASS |
| 102 | ch:gauge:L102:1.120 | measured | `1.120` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, methylated channel, highest culture | PASS |
| 102 | ch:gauge:L102:0.965 | measured | `0.965` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, both channels, lowest culture | PASS |
| 102 | ch:gauge:L102:0.975 | measured | `0.975` | file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, both channels, highest culture | PASS |
| 112 | ch:gauge:L112 | measured | `+0.06` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: tared Met-A shift of a simulated 2 % neutrophil pattern loss | PASS |
| 112 | ch:gauge:L112:1.090 | measured | `1.090` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: tared Met-A of the damaged mixtures, highest | PASS |
| 113 | ch:gauge:L113 | measured | `0.033` | file `Biological_Physics/MethylPhys/doors/data/lowfrac_readings.csv`: 2 % blur shift at 40-50 % neutrophils | PASS |
| 113 | ch:gauge:L113:0.064 | measured | `0.064` | file `Biological_Physics/MethylPhys/doors/data/lowfrac_readings.csv`: 2 % blur shift above 70 % neutrophils | PASS |
| 114 | ch:gauge:L114 | measured | `1.00` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: IAM-A of healthy granulocyte donors, mean | PASS |
| 114 | ch:gauge:L114:1.29 | measured | `1.29` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: IAM-A after a simulated 2 % rise in copy error, lowest donor | PASS |
| 114 | ch:gauge:L114:1.35 | measured | `1.35` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: IAM-A after a simulated 2 % rise in copy error, highest donor | PASS |
| 120 | ch:gauge:L120 | measured | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: SD of the six held-out reference arrays | PASS |

## Part 6 - ch:meta - `docs/book/part6/p6_07_meta.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 8 | eq:meta | calibrated | `0.330263` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: H_ref of EPIC neutrophils: mean of the six arrays' mean H on the identity sites | PASS |
| 23 |  | measured | `0.75` | not run: definition: identity-site selection band, methylated channel beta 0.75-0.95 (a rule of the chain) | - |
| 23 |  | measured | `0.95` | not run: definition: identity-site selection band, methylated channel beta 0.75-0.95 (a rule of the chain) | - |
| 23 |  | measured | `0.05` | not run: definition: identity-site selection band, unmethylated channel beta 0.05-0.25 (a rule of the chain) | - |
| 23 |  | measured | `0.25` | not run: definition: identity-site selection band, unmethylated channel beta 0.05-0.25 (a rule of the chain) | - |
| 25 | ch:meta:L25 | measured | `0.330263` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: the frozen Met-A floor of EPIC neutrophils | PASS |
| 27 | ch:meta:L27 | measured | `0.983` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A of the six reference arrays, lowest | PASS |
| 27 | ch:meta:L27:1.045 | measured | `1.045` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A of the six reference arrays, highest | PASS |
| 27 | ch:meta:L27:0.020 | measured | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A of the six reference arrays, SD | PASS |
| 28 | ch:meta:L28 | measured | `0.993` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A on the frozen sites, lowest | PASS |
| 28 | ch:meta:L28:1.008 | measured | `1.008` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A on the frozen sites, highest | PASS |
| 41 | ch:meta:L41 | measured | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: SD of the held-out readings (figure caption) | PASS |
| 50 | ch:meta:L50 | measured | `201868500150` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998021 | PASS |
| 50 | ch:meta:L50:1.0032 | measured | `1.0032` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998021 read in the floor (acceptance run) | PASS |
| 50 | ch:meta:L50:1.011 | measured | `1.011` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998021 held out, sites re-chosen | PASS |
| 50 | ch:meta:L50:1.004 | measured | `1.004` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998021 held out, frozen sites | PASS |
| 50 | ch:meta:L50:0.1418 | measured | `0.1418` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998021 | PASS |
| 50 | ch:meta:L50:0.83 | measured | `0.83` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998021 (acceptance run) | PASS |
| 51 | ch:meta:L51 | measured | `201868590243` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998057 | PASS |
| 51 | ch:meta:L51:0.9945 | measured | `0.9945` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998057 read in the floor (acceptance run) | PASS |
| 51 | ch:meta:L51:1.010 | measured | `1.010` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998057 held out, sites re-chosen | PASS |
| 51 | ch:meta:L51:0.993 | measured | `0.993` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998057 held out, frozen sites | PASS |
| 51 | ch:meta:L51:0.1223 | measured | `0.1223` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998057 | PASS |
| 51 | ch:meta:L51:0.95 | measured | `0.95` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998057 (acceptance run) | PASS |
| 52 | ch:meta:L52 | measured | `201870610056` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998116 | PASS |
| 52 | ch:meta:L52:0.9989 | measured | `0.9989` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998116 read in the floor (acceptance run) | PASS |
| 52 | ch:meta:L52:1.012 | measured | `1.012` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998116 held out, sites re-chosen | PASS |
| 52 | ch:meta:L52:0.999 | measured | `0.999` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998116 held out, frozen sites | PASS |
| 52 | ch:meta:L52:0.1244 | measured | `0.1244` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998116 | PASS |
| 52 | ch:meta:L52:1.21 | measured | `1.21` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998116 (acceptance run) | PASS |
| 53 | ch:meta:L53 | measured | `201868500150` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998023 | PASS |
| 53 | ch:meta:L53:0.9942 | measured | `0.9942` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998023 read in the floor (acceptance run) | PASS |
| 53 | ch:meta:L53:0.983 | measured | `0.983` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998023 held out, sites re-chosen | PASS |
| 53 | ch:meta:L53:0.993 | measured | `0.993` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998023 held out, frozen sites | PASS |
| 53 | ch:meta:L53:0.1286 | measured | `0.1286` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998023 | PASS |
| 53 | ch:meta:L53:1.05 | measured | `1.05` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998023 (acceptance run) | PASS |
| 54 | ch:meta:L54 | measured | `201870610111` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998143 | PASS |
| 54 | ch:meta:L54:1.0028 | measured | `1.0028` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998143 read in the floor (acceptance run) | PASS |
| 54 | ch:meta:L54:1.016 | measured | `1.016` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998143 held out, sites re-chosen | PASS |
| 54 | ch:meta:L54:1.003 | measured | `1.003` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998143 held out, frozen sites | PASS |
| 54 | ch:meta:L54:0.1284 | measured | `0.1284` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998143 | PASS |
| 54 | ch:meta:L54:0.69 | measured | `0.69` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998143 (acceptance run) | PASS |
| 55 | ch:meta:L55 | measured | `201868590206` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Sentrix chip of reference array GSM2998030 | PASS |
| 55 | ch:meta:L55:1.0064 | measured | `1.0064` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: GSM2998030 read in the floor (acceptance run) | PASS |
| 55 | ch:meta:L55:1.045 | measured | `1.045` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998030 held out, sites re-chosen | PASS |
| 55 | ch:meta:L55:1.008 | measured | `1.008` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: GSM2998030 held out, frozen sites | PASS |
| 55 | ch:meta:L55:0.1489 | measured | `0.1489` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index N of GSM2998030 | PASS |
| 55 | ch:meta:L55:1.08 | measured | `1.08` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of GSM2998030 (acceptance run) | PASS |
| 70 | ch:meta:L70 | measured | `0.932` | file `Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md`: 450K purified neutrophils on EPIC references, median A | PASS |
| 70 | ch:meta:L70:0.916 | measured | `0.916` | file `Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md`: 450K purified monocytes on EPIC references, median A | PASS |
| 70 | ch:meta:L70:0.904 | measured | `0.904` | file `Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md`: 450K purified NK cells on EPIC references, median A | PASS |
| 79 | eq:metawb | none |  | not run: definition: whole-blood Met-A with the specimen's own expectation e_i = sum_g f_g mu_g,i | - |
| 86 | ch:meta:L86 | measured | `0.982` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: known-fraction expectation on six DNA mixtures, lowest | PASS |
| 86 | ch:meta:L86:1.016 | measured | `1.016` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: known-fraction expectation on six DNA mixtures, highest | PASS |
| 86 | ch:meta:L86:1.062 | measured | `1.062` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: same mixtures on the neutrophil reference alone, lowest | PASS |
| 86 | ch:meta:L86:1.118 | measured | `1.118` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: same mixtures on the neutrophil reference alone, highest | PASS |
| 91 | ch:meta:L91 | measured | `0.86` | file `Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_T2_OUTCOME.md`: second-laboratory isolated neutrophils on the reference, lowest | PASS |
| 91 | ch:meta:L91:1.26 | measured | `1.26` | file `Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_T2_OUTCOME.md`: second-laboratory isolated neutrophils on the reference, highest | PASS |
| 94 | ch:meta:L94 | measured | `0.122` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index of the reference arrays, lowest | PASS |
| 94 | ch:meta:L94:0.149 | measured | `0.149` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index of the reference arrays, highest | PASS |
| 95 | ch:meta:L95 | measured | `0.243` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: noise index of second-laboratory arrays, highest | PASS |
| 95 | ch:meta:L95:0.79 | measured | `0.79` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: Spearman rho of Met-A with N, second laboratory (30 y donor) | PASS |
| 95 | ch:meta:L95:0.83 | measured | `0.83` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: Spearman rho of Met-A with N, second laboratory (54 y donor) | PASS |
| 111 | ch:meta:L111:1.002 | measured | `1.002` | file `Biological_Physics/MethylPhys/doors/data/dnmt_arrays_readings.csv`: DNMT-inhibitor series: inactive analogue, lowest | PASS |
| 112 |  | measured | `2.8` | not run: measured, source not named | - |

## Part 6 - ch:iama - `docs/book/part6/p6_08_iama.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 14 | eq:eps | derived |  | sympy: E_hold = kT ln((1-eps)/eps) solves the two-state Boltzmann error | PASS |
| 27 | eq:eps0 | derived |  | sympy: eps0 = 1/(1+e^(phi M)) inverts E_hold = ln((1-eps)/eps) = phi M | PASS |
| 31 | ch:iama:L31 | derived | `20.94` | numeric: M | PASS |
| 31 | ch:iama:L31:0.1628 | derived | `0.1628` | numeric: phi = E_hold/M | PASS |
| 32 | ch:iama:L32 | measured | `0.032` | numeric: eps0 = 1/(1+e^(phi M)) | PASS |
| 32 | ch:iama:L32:0.2043 | measured | `0.2043` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: H_min = H(eps0) at the frozen eps0 | PASS |
| 40 | eq:iama | measured | `1.099` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: P of neutrophils from the three granulocyte donors | PASS |
| 44 | ch:iama:L44 | measured | `1.084` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: P range over the donors, lowest | PASS |
| 44 | ch:iama:L44:1.108 | measured | `1.108` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: P range over the donors, highest | PASS |
| 44 | ch:iama:L44:1.2 | measured | `1.2` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: coefficient of variation of P across donors, per cent | PASS |
| 46 | ch:iama:L46 | measured | `0.910` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: the floor at A = 1/P | PASS |
| 52 | ch:iama:L52 | measured | `1.084` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: healthy donors on eps0 alone, lowest | PASS |
| 52 | ch:iama:L52:1.127 | measured | `1.127` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: healthy donors on eps0 alone, highest | PASS |
| 53 | ch:iama:L53 | measured | `0.978` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: healthy donors with P, leave-one-donor-out, lowest | PASS |
| 53 | ch:iama:L53:1.040 | measured | `1.040` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: healthy donors with P, leave-one-donor-out, highest | PASS |
| 55 | ch:iama:L55 | measured | `1.285` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: simulated 2 % rise in copy error, lowest | PASS |
| 55 | ch:iama:L55:1.346 | measured | `1.346` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: simulated 2 % rise in copy error, highest | PASS |
| 56 | ch:iama:L56 | measured | `0.70` | file `Biological_Physics/MethylPhys/chain_tests/IAMA_FLOOR_COMPARISON.md`: same cells on a second read pipeline, bare floor, lowest | PASS |
| 56 | ch:iama:L56:0.79 | measured | `0.79` | file `Biological_Physics/MethylPhys/chain_tests/IAMA_FLOOR_COMPARISON.md`: same cells on a second read pipeline, bare floor, highest | PASS |
| 74 | ch:iama:L74 | calc | `1.099` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: P in the curve of the figure | PASS |
| 75 | ch:iama:L75 | calc | `0.3` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: a 2 % rise in copy error moves the reading by about 0.3 | PASS |
| 90 | ch:iama:L90 | measured | `0.024` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy error across 56 healthy cell types, lowest | PASS |
| 90 | ch:iama:L90:0.042 | measured | `0.042` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy error across 56 healthy cell types, highest | PASS |
| 90 | ch:iama:L90:3.41 | measured | `3.41` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy across 56 cell types, mean | PASS |
| 91 | ch:iama:L91 | measured | `0.79` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy cell types on one physics floor, lowest | PASS |
| 91 | ch:iama:L91:1.23 | measured | `1.23` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy cell types on one physics floor, highest | PASS |
| 91 | ch:iama:L91:1.01 | measured | `1.01` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy cell types on one physics floor, median | PASS |
| 94 | ch:iama:L94 | measured | `1.66` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy error on the common sites, highest over lowest | PASS |
| 95 | ch:iama:L95 | measured | `0.80` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: intraclass correlation of the common-site copy error across donors | PASS |
| 96 | ch:iama:L96 | measured | `3.4` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: colon epithelium lifespan (days) | PASS |
| 97 | ch:iama:L97 | measured | `0.031` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy error of colon epithelium and cardiomyocytes | PASS |

## Part 6 - ch:cscore - `docs/book/part6/p6_09_cscore.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 14 | eq:z | none |  | not run: definition: residual z_i = (H(beta_i) - H(ref_i)) / s_i (the construction of the map) | - |
| 26 | eq:C | calibrated | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: c_healthy: median of the six leave-one-out clustering values | PASS |
| 31 | ch:cscore:L31 | calibrated | `0.70` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C of the healthy reference arrays, lowest | PASS |
| 31 | ch:cscore:L31:1.23 | calibrated | `1.23` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C of the healthy reference arrays, highest | PASS |
| 38 |  | calc | `+0.5` | not run: input: shift of +0.5 healthy SD given to every site of the simulated illustration map (no specimen), nothing to recompute | - |
| 39 | ch:cscore:L39 | calc | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy baseline used in the construction figure | PASS |
| 46 | ch:cscore:L46 | measured | `0.69` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: acceptance run: isolated neutrophils, lowest C | PASS |
| 46 | ch:cscore:L46:1.21 | measured | `1.21` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: acceptance run: isolated neutrophils, highest C | PASS |
| 46 | ch:cscore:L46:0.78 | measured | `0.78` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: acceptance run: whole bloods, lowest C | PASS |
| 46 | ch:cscore:L46:1.49 | measured | `1.49` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: acceptance run: whole bloods, highest C | PASS |
| 63 | ch:cscore:L63 | calibrated | `0.70` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: table: healthy reference arrays, lowest C | PASS |
| 63 | ch:cscore:L63:1.23 | calibrated | `1.23` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: table: healthy reference arrays, highest C | PASS |
| 64 | ch:cscore:L64 | measured | `0.69` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: isolated reference neutrophils, lowest C | PASS |
| 64 | ch:cscore:L64:1.21 | measured | `1.21` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: isolated reference neutrophils, highest C | PASS |
| 65 | ch:cscore:L65 | measured | `0.91` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: DNA mixtures, lowest C | PASS |
| 65 | ch:cscore:L65:1.49 | measured | `1.49` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: DNA mixtures, highest C | PASS |
| 66 | ch:cscore:L66 | measured | `0.78` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: remission bloods, lowest C | PASS |
| 66 | ch:cscore:L66:1.32 | measured | `1.32` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: table: remission bloods, highest C | PASS |
| 67 | ch:cscore:L67 | calc | `45` | numeric: C far end | PASS |
| 67 | ch:cscore:L67:50 | calibrated | `50` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score block size | PASS |

## Part 6 - ch:temperature - `docs/book/part6/p6_10_temperature.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 5 | ch:temperature:L5 | calc | `22.94` | numeric: M at 10 C | PASS |
| 6 | ch:temperature:L6 | calc | `20.94` | numeric: M at 37 C | PASS |
| 6 | ch:temperature:L6:20.84 | calc | `20.84` | numeric: M at 38.5 C | PASS |
| 6 | ch:temperature:L6:20.74 | calc | `20.74` | numeric: M at 40 C | PASS |
| 6 |  | calc | `38.5` | not run: input: dog body temperature 38.5 C (M at 38.5 C is checked by ch:temperature:L6:20.84) | - |
| 12 | eq:eps0T | derived |  | sympy: eps0 rises with T at fixed holding energy | PASS |
| 16 | ch:temperature:L16 | derived | `0.0233` | numeric: eps0 at 10 C | PASS |
| 17 | ch:temperature:L17 | calc | `0.78` | numeric: floor H(eps0) at 10 C over the human | PASS |
| 17 | ch:temperature:L17:1.012 | calc | `1.012` | numeric: floor at 38.5 C | PASS |
| 17 |  | calc | `38.5` | not run: input: dog body temperature 38.5 C (the floor at 38.5 C is checked by ch:temperature:L17:1.012) | - |
| 23 | ch:temperature:L23 | calc | `3.41` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy of human cells at 37 C | PASS |
| 30 | ch:temperature:L30 | calc | `1.025` | numeric: floor at 40 C | PASS |
| 30 | ch:temperature:L30:1.041 | calc | `1.041` | numeric: floor at 42 C | PASS |
| 31 | ch:temperature:L31 | calc | `0.902` | numeric: floor at 25 C | PASS |
| 31 | ch:temperature:L31:0.959 | calc | `0.959` | numeric: floor at 32 C | PASS |
| 32 | ch:temperature:L32 | calc | `0.821` | numeric: floor at 15 C | PASS |
| 36 | ch:temperature:L36 | calc | `8.1` | numeric: 7-fold selectivity at 15 C | PASS |
| 36 | ch:temperature:L36:6.8 | calc | `6.8` | numeric: at 42 C | PASS |
| 76 | ch:temperature:L76 | measured | `3.41` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: human cells at 37 C (dashed line) | PASS |
| 87 | ch:temperature:L111 | prediction | `20.84` | numeric: M for a dog at 38.5 C | PASS |
| 98 |  | observed | `0.96` | not run: measured, source not named | - |
| 113 |  | prediction | `1.00` | not run: prediction, nothing to recompute (held-out canine cells should read 1.00 on a canine reference) | - |

## Part 6 - ch:translation - `docs/book/part6/p6_11_translation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 33 |  | derived | `10` | not run: input restated: a plasma draw carries of order 10^4 genome equivalents, the upper end of the 10^3-10^4 copies of ch:sky L72-73 (about 10^3 genome equivalents per millilitre, Sender2024); the printed '10' is the base of 10^4, nothing to recompute here | - |
| 110 | ch:translation:L110 | calc | `2.3` | numeric: neutron-star saturation mass at the upper edge of the TOV bound | PASS |
| 111 | ch:translation:L111:0.2 | calc | `0.2` | numeric: TOV mass not fixed to better than about 0.2 M_sun | PASS |
| 111 |  | calc | `0.15` | not run: input: lower error -0.15 M_sun of the TOV bound 2.16 (+0.17, -0.15) M_sun (Rezzolla2018, doi 10.3847/2041-8213/aaa401) | - |

## Part 6 - ch:instrument - `docs/book/part6/p6_12_instrument.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 31 | ch:instrument:L31:0.979 | measured | `0.979` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, first laboratory minimum | PASS |
| 31 | ch:instrument:L31:0.975 | measured | `0.975` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, second laboratory median | PASS |
| 31 | ch:instrument:L31:0.891 | measured | `0.891` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, second laboratory minimum | PASS |
| 31 | ch:instrument:L31:0.953 | measured | `0.953` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, third laboratory median | PASS |
| 32 | ch:instrument:L32:0.932 | measured | `0.932` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, third laboratory minimum | PASS |
| 32 | ch:instrument:L32:0.878 | measured | `0.878` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, fourth laboratory median | PASS |
| 32 | ch:instrument:L32:0.928 | measured | `0.928` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: intake call rate, fourth laboratory maximum | PASS |
| 50 | ch:instrument:L50:0.15 | measured | `0.15` | file `Biological_Physics/MethylPhys/doors/FINDING_GSE125105_LOW_SIGNAL.md`: low-signal laboratory: lowest control-signal ratio | PASS |
| 50 | ch:instrument:L50:0.31 | measured | `0.31` | file `Biological_Physics/MethylPhys/doors/FINDING_GSE125105_LOW_SIGNAL.md`: low-signal laboratory: highest control-signal ratio | PASS |
| 51 | ch:instrument:L51 | measured | `12.5` | file `Biological_Physics/MethylPhys/doors/FINDING_GSE125105_LOW_SIGNAL.md`: low-signal laboratory: probes at background | PASS |
| 55 | ch:instrument:L55:0.33 | measured | `0.33` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: beta of probes at background, lower end | PASS |
| 55 | ch:instrument:L55:0.42 | measured | `0.42` | file `Biological_Physics/MethylPhys/doors/PROC_INTAKE_01_OUTCOME.md`: beta of probes at background, upper end | PASS |
| 56 |  | measured | `0.75` | not run: definition: identity-site window 0.75-0.95 restated from the site rule (ch:identity L20, calibrated) | - |
| 56 |  | measured | `0.95` | not run: definition: identity-site window 0.75-0.95 restated from the site rule (ch:identity L20, calibrated); the site set's upper extreme is checked at ch:identity:L20:0.95 | - |
| 56 |  | measured | `0.05` | not run: definition: identity-site window 0.05-0.25 restated from the site rule (ch:identity L20, calibrated); the site set's lower extreme is checked at ch:identity:L20:0.05 | - |
| 56 |  | measured | `0.25` | not run: definition: identity-site window 0.05-0.25 restated from the site rule (ch:identity L20, calibrated) | - |
| 82 | ch:instrument:L82:0.79 | measured | `0.79` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: Met-A follows N, second laboratory donor 2 | PASS |
| 82 | ch:instrument:L82:0.83 | measured | `0.83` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: Met-A follows N, second laboratory donor 1 | PASS |
| 83 | ch:instrument:L83 | measured | `1.07` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: second-laboratory arrays at reference N still read high | PASS |
| 101 | ch:instrument:L101:0.943 | measured | `0.943` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: untared Met-A of six DNA mixtures, lowest | PASS |
| 101 | ch:instrument:L101:0.968 | measured | `0.968` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: untared Met-A of six DNA mixtures, highest | PASS |
| 101 | ch:instrument:L101:1.073 | measured | `1.073` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: untared Met-A of five remission bloods, lowest | PASS |
| 101 | ch:instrument:L101:1.115 | measured | `1.115` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: untared Met-A of five remission bloods, highest | PASS |
| 105 | ch:instrument:L105:0.86 | measured | `0.86` | file `Biological_Physics/MethylPhys/doors/data/t2_diag.csv`: second-laboratory purified neutrophils, lowest untared A | PASS |
| 105 | ch:instrument:L105:1.26 | measured | `1.26` | file `Biological_Physics/MethylPhys/doors/PROC_NEUT_TEST_01_T2_OUTCOME.md`: second-laboratory purified neutrophils, highest untared A | PASS |
| 112 | ch:instrument:L112:0.205 | fitted | `0.205` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit: slope on neutrophil fraction | PASS |
| 112 | ch:instrument:L112:+2.51 | fitted | `+2.51` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit: slope on noise index | PASS |
| 112 | ch:instrument:L112:0.85 | fitted | `0.85` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit: R^2 | PASS |
| 120 | ch:instrument:L120:0.60 | fitted | `0.60` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit (figure): intercept | PASS |
| 120 | ch:instrument:L120:0.205 | fitted | `0.205` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit (figure): slope on neutrophil fraction | PASS |
| 120 | ch:instrument:L120:+2.51 | fitted | `+2.51` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit (figure): slope on noise index | PASS |
| 120 | ch:instrument:L120:0.85 | fitted | `0.85` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: noise fit (figure): R^2 | PASS |
| 126 | ch:instrument:L126:-0.14 | measured | `-0.14` | file `Biological_Physics/MethylPhys/doors/PROC_TARE_01_OUTCOME.md`: SNP tare scale does not predict the reading | PASS |
| 126 | ch:instrument:L126:-0.07 | measured | `-0.07` | file `Biological_Physics/MethylPhys/doors/PROC_TARE_01_OUTCOME.md`: SNP tare offset does not predict the reading | PASS |

## Part 6 - ch:separation - `docs/book/part6/p6_13_separation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 6 | eq:mix | none |  | not run: definition: linear mixing model of a specimen, beta_i = sum_g f_g mu_{g,i} + eps_i with f_g >= 0 and sum f_g = 1 (the equation Stage A solves; Houseman2012, Salas2022) | - |
| 45 |  | measured | `0.05` | not run: measured, source not named | - |
| 49 |  | measured | `0.05` | not run: definition: pre-registered bar of test T1b (fraction within 0.05 of the flow counts), not a measurement | - |
| 68 |  | openprob | `0.40` | not run: input: the read line 0.40 restated from the table (line 56); the count of five healthy arrays below it is not the printed value | - |
| 88 | ch:separation:L88 | measured | `+0.12` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: tared Met-A rises with neutrophil fraction | PASS |
| 91 | ch:separation:L91 | fitted | `-0.02` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: fraction dependence after the development fit | PASS |
| 99 |  | measured | `1.05` | not run: definition: upper edge of the Normal band (0.95-1.05) restated, the line the tared reading is compared with | - |
| 100 |  | measured | `1.05` | not run: definition: upper edge of the Normal band (0.95-1.05) restated | - |

## Part 6 - ch:atlas - `docs/book/part6/p6_14_atlas.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 34 | ch:atlas:L34:0.983 | measured | `0.983` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out neutrophil reference, lowest A | PASS |
| 34 | ch:atlas:L34:1.045 | measured | `1.045` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out neutrophil reference, highest A | PASS |
| 34 | ch:atlas:L34:0.020 | measured | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out neutrophil reference, SD | PASS |
| 42 | ch:atlas:L42:0.904 | measured | `0.904` | file `Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md`: 450K purified cells on another platform's reference, lowest median | PASS |
| 42 | ch:atlas:L42:0.932 | measured | `0.932` | file `Biological_Physics/MethylPhys/doors/DIAG_450K_01_OUTCOME.md`: 450K purified cells on another platform's reference, highest median | PASS |
| 62 |  | calc | `0.05` | not run: definition: identity-site window edge 0.05 (0.05-0.25 and 0.75-0.95) restated from the site rule (ch:identity L20); nothing computed at this line | - |
| 62 |  | measured | `0.25` | not run: definition: identity-site window edge 0.25 restated from the site rule (ch:identity L20) | - |
| 62 |  | measured | `0.75` | not run: definition: identity-site window edge 0.75 restated from the site rule (ch:identity L20) | - |
| 62 |  | measured | `0.95` | not run: definition: identity-site window edge 0.95 restated from the site rule (ch:identity L20) | - |
| 64 | ch:atlas:L64:1.062 | measured | `1.062` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the neutrophil floor alone, lowest | PASS |
| 64 | ch:atlas:L64:1.118 | measured | `1.118` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the neutrophil floor alone, highest | PASS |
| 64 | ch:atlas:L64:0.982 | measured | `0.982` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the expectation from their own composition, lowest | PASS |
| 64 | ch:atlas:L64:1.016 | measured | `1.016` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the expectation from their own composition, highest | PASS |
| 73 |  | calc | `0.05` | not run: definition: the 0.05 threshold in |Delta beta| of the table column 'sites within 0.05' (dotted line of the figure); the shares themselves are checked at ch:identity:L57 and L66 | - |
| 101 |  | none |  | not run: definition: the hierarchical model of atlas v2 (likelihood and prior of beta_obs, mu_ic = m_i + e_ic); a model statement, nothing to derive | - |
| 140 | ch:atlas:L140:1.03 | measured | `1.03` | file `Biological_Physics/MethylPhys/atlas/v2/records/03b_crossplatform_check.csv`: array-sequencing transfer, lowest slope | PASS |
| 140 | ch:atlas:L140:1.09 | measured | `1.09` | file `Biological_Physics/MethylPhys/atlas/v2/records/03b_crossplatform_check.csv`: array-sequencing transfer, highest slope | PASS |
| 140 | ch:atlas:L140:-0.04 | measured | `-0.04` | file `Biological_Physics/MethylPhys/atlas/v2/records/03b_crossplatform_check.csv`: array-sequencing transfer, intercept nearest zero | PASS |
| 140 | ch:atlas:L140:-0.07 | measured | `-0.07` | file `Biological_Physics/MethylPhys/atlas/v2/records/03b_crossplatform_check.csv`: array-sequencing transfer, most negative intercept | PASS |
| 144 |  | measured | `0.015` | not run: measured, source not named | - |
| 148 | ch:atlas:L148 | measured | `4.35` | file `Biological_Physics/MethylPhys/atlas/v2/records/11c_stageA_report_NOT_CONVERGED.json`: joint fit did not converge: largest R-hat | PASS |
| 150 | ch:atlas:L150:1.005 | measured | `1.005` | file `Biological_Physics/MethylPhys/atlas/v2/records/14_stageB_timing_block.log`: test block converged: R-hat 99th percentile | PASS |
| 150 | ch:atlas:L150:0.83 | measured | `0.83` | file `Biological_Physics/MethylPhys/atlas/v2/records/14_stageB_timing_block.log`: test block: seconds per locus | PASS |
| 153 | ch:atlas:L153 | measured | `0.19` | file `Biological_Physics/MethylPhys/atlas/v2/postbuild/records/atlas_v2_gate.json`: values flagged for per-block convergence | PASS |
| 163 | ch:atlas:L163:92.3 | measured | `92.3` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out coverage, lowest block | PASS |
| 163 | ch:atlas:L163:93.1 | measured | `93.1` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out coverage, highest block | PASS |
| 167 | ch:atlas:L167 | measured | `92.7` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out coverage overall | PASS |
| 169 | ch:atlas:L169:96.9 | measured | `96.9` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out coverage, highest cells, lower end | PASS |
| 169 | ch:atlas:L169:97.2 | measured | `97.2` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out coverage, highest cells, upper end | PASS |
| 169 | ch:atlas:L169:0.051 | measured | `0.051` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: held-out mean absolute prediction error | PASS |
| 173 |  | derived | `36` | not run: input: illustrative number of independent samples n = 36 (the derived factor six is sqrt(36) applied to it) | - |
| 177 | ch:atlas:L177 | derived | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: precision of the neutrophil reference restated | PASS |
| 181 | ch:atlas:L181 | measured | `87.7` | file `Biological_Physics/MethylPhys/atlas/v2/postbuild/README.md`: atlas v2 identity sites, array to array in Normal | PASS |
| 182 | ch:atlas:L182 | measured | `73.1` | file `Biological_Physics/MethylPhys/atlas/v2/postbuild/README.md`: atlas v2 identity sites, sequencing to array in Normal | PASS |
| 223 | ch:atlas:L223:1.062 | measured | `1.062` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the neutrophil floor alone, lowest | PASS |
| 223 | ch:atlas:L223:1.118 | measured | `1.118` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the neutrophil floor alone, highest | PASS |
| 223 | ch:atlas:L223:0.982 | measured | `0.982` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the expectation from their own composition, lowest | PASS |
| 223 | ch:atlas:L223:1.016 | measured | `1.016` | file `Biological_Physics/MethylPhys/doors/PROC_WB_NEUT_01_OUTCOME.md`: six DNA mixtures against the expectation from their own composition, highest | PASS |

## Part 6 - ch:identity - `docs/book/part6/p6_15_identity.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 |  | calibrated | `0.05` | not run: definition: site-selection rule, across-array SD of beta at most 0.05 (calibrated rule; the per-array betas behind it are not in a committed file) | - |
| 20 | ch:identity:L20:0.95 | calibrated | `0.95` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: identity sites: upper edge of the methylated window | PASS |
| 20 | ch:identity:L20:0.05 | calibrated | `0.05` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: identity sites: lower edge of the unmethylated window | PASS |
| 20 |  | calibrated | `0.75` | not run: definition: site-selection rule, inner edge 0.75 of the methylated window (calibrated); the frozen set's sites sit inside it (lowest methylated-side mean beta 0.775) | - |
| 20 |  | calibrated | `0.25` | not run: definition: site-selection rule, inner edge 0.25 of the unmethylated window (calibrated); the frozen set's sites sit inside it (highest unmethylated-side mean beta 0.244) | - |
| 25 | ch:identity:L25 | calc | `1.58` | numeric: dH/dbeta at beta = 0.25 | PASS |
| 26 | ch:identity:L26:4.25 | calc | `4.25` | numeric: dH/dbeta at beta = 0.05 | PASS |
| 26 | ch:identity:L26:0.75 | calc | `0.75` | numeric: same steepness at 0.75 as at 0.25 | PASS |
| 26 | ch:identity:L26:0.95 | calc | `0.95` | numeric: same steepness at 0.95 as at 0.05 | PASS |
| 26 |  | calc | `0.25` | not run: input: beta = 0.25, the point at which dH/dbeta is evaluated (the window edge); the slope is checked at ch:identity:L25 | - |
| 26 |  | calc | `0.05` | not run: input: beta = 0.05, the point at which dH/dbeta is evaluated (the window edge); the slope is checked at ch:identity:L26:4.25 | - |
| 27 |  | calc | `0.95` | not run: restates ch:identity:L26:0.95 (a site near 0.95, where the slope magnitude equals that at 0.05) | - |
| 34 | ch:identity:L34:0.0114 | calibrated | `0.0114` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: shrunk SD of H at the identity sites, median | PASS |
| 34 | ch:identity:L34:0.0094 | calibrated | `0.0094` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: shrunk SD of H at the identity sites, smallest | PASS |
| 34 | ch:identity:L34:0.0137 | calibrated | `0.0137` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: shrunk SD of H at the identity sites, largest | PASS |
| 35 |  | calibrated | `0.05` | not run: definition: window edge beta = 0.05 restated in the caption (where the spread is largest); restates ch:identity:L20:0.05 | - |
| 35 |  | calibrated | `0.95` | not run: definition: window edge beta = 0.95 restated in the caption; restates ch:identity:L20:0.95 | - |
| 52 |  | measured | `1.00` | not run: definition: healthy is A = 1.00 (the reference arrays read 1.00 by construction); the held-out readings are checked at ch:atlas:L34 | - |
| 57 | ch:identity:L57:98.7 | calc | `98.7` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`: identity sites shared by monocytes | PASS |
| 57 | ch:identity:L57:75.6 | calc | `75.6` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`: identity sites shared by CD4 T cells | PASS |
| 57 |  | calc | `0.05` | not run: definition: the 0.05 threshold in |Delta beta| that defines a shared site; the shares are checked at ch:identity:L57:98.7 and L57:75.6 | - |
| 66 | ch:identity:L66:98.7 | calc | `98.7` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`: monocytes within 0.05 (figure) | PASS |
| 66 | ch:identity:L66:75.6 | calc | `75.6` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json`: CD4 T cells within 0.05 (figure) | PASS |
| 66 |  | calc | `0.05` | not run: definition: the 0.05 threshold in |Delta beta| (grey band of the figure); the shares are checked at ch:identity:L66 | - |
| 76 | ch:identity:L76 | openprob | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: held-out SD of the neutrophil set (status box) | PASS |

## Part 6 - ch:skytools - `docs/book/part6/p6_16a_skytools.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 23 | ch:skytools:L23:3.5 | observed | `3.5` | numeric: Penzias and Wilson excess antenna temperature | PASS |
| 23 |  | observed | `3.3` | not run: measured, source not named | - |
| 58 |  | calc | `12` | not run: definition: HEALPix divides the sphere into 12 N_side^2 pixels (Gorski2005) | - |
| 60 | ch:skytools:L60:64 | calc | `64` | numeric: HEALPix N_side for 49,152 pixels | PASS |
| 60 | ch:skytools:L60:17.6 | calc | `17.6` | numeric: EPIC CpGs per HEALPix pixel, N_side 64 | PASS |
| 61 | ch:skytools:L61:2.5 | calc | `2.5` | numeric: atlas CpGs per HEALPix pixel, N_side 128 | PASS |
| 61 |  | calc | `128` | not run: input: N_side = 128 chosen for the atlas map (a display choice); the CpGs per pixel it gives are checked at ch:skytools:L61:2.5 | - |
| 83 |  | none |  | not run: definition: residual z_i = (H(beta_i) - mean H_i)/s_i at each site (same construction as eq:sky) | - |
| 90 | ch:skytools:L90:1.1104 | derived | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy clustering baseline | PASS |
| 90 | ch:skytools:L90:0.70 | derived | `0.70` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy C-score, lowest | PASS |
| 90 | ch:skytools:L90:1.23 | derived | `1.23` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy C-score, highest | PASS |
| 94 | ch:skytools:L94:0.71 | measured | `0.71` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv`: sky-map healthy C-score, lowest (two digits) | PASS |
| 94 | ch:skytools:L94:1.23 | measured | `1.23` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv`: sky-map healthy C-score, highest | PASS |
| 94 | ch:skytools:L94:0.705 | measured | `0.705` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv`: sky-map healthy C-score, lowest (three digits) | PASS |
| 95 | ch:skytools:L95:0.70 | measured | `0.70` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: chain healthy C-score, lowest | PASS |
| 95 | ch:skytools:L95:1.23 | measured | `1.23` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: chain healthy C-score, highest | PASS |
| 95 | ch:skytools:L95:0.699 | measured | `0.699` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: chain healthy C-score, lowest (three digits) | PASS |
| 95 |  | measured | `0.1` | not run: printed value 0.1 (about 0.1) has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.1027) | - |
| 96 | ch:skytools:L96:12.5 | measured | `12.5` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv`: regional damage C-score, lowest | PASS |
| 96 | ch:skytools:L96:15.7 | measured | `15.7` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv`: regional damage C-score, highest | PASS |
| 96 |  | measured | `1.05` | not run: definition: upper edge of the Normal band 0.95-1.05 restated | - |
| 114 |  | measured | `0.3` | not run: definition: beta window 0.3-0.7 used to count heterozygous-looking chrX sites (the classification threshold, not a measurement) | - |
| 114 |  | measured | `0.7` | not run: definition: beta window 0.3-0.7 used to count heterozygous-looking chrX sites | - |
| 125 | ch:skytools:L125:0.32 | measured | `0.32` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d) of beta at 1-1.8 kb | PASS |
| 125 | ch:skytools:L125:1.8 | measured | `1.8` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: upper edge of the 1 kb distance bin | PASS |
| 125 |  | measured | `0.05` | not run: printed value 0.05 has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.0526) | - |
| 126 | ch:skytools:L126 | measured | `0.327` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d) at 1 kb, highest of the six arrays | PASS |
| 138 | ch:skytools:L138 | measured | `0.02` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: healthy residual uncorrelated beyond 1 kb | PASS |
| 139 | ch:skytools:L139:0.08 | measured | `0.08` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: regional-damage plateau, upper end | PASS |
| 139 |  | measured | `0.05` | not run: printed value 0.05 has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.0517); the upper end is checked at ch:skytools:L139:0.08 | - |
| 165 | ch:skytools:L165 | calc | `0.00047` | file `Biological_Physics/MethylPhys/doors/PROC_AGE_01_OUTCOME.md`: age-ladder slope per year | PASS |

## Part 6 - ch:sky - `docs/book/part6/p6_16_sky.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 | eq:sky | none |  | not run: definition: the sky residual z_i = (H(beta_i) - H(ref_i))/s_i at each identity site (construction; s_i calibrated) | - |
| 30 | ch:sky:L30:0.69 | measured | `0.69` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of isolated neutrophils, lowest | PASS |
| 30 | ch:sky:L30:1.21 | measured | `1.21` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of isolated neutrophils, highest | PASS |
| 30 | ch:sky:L30:0.78 | measured | `0.78` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of whole bloods, lowest | PASS |
| 30 | ch:sky:L30:1.49 | measured | `1.49` | file `Biological_Physics/MethylPhys/chain_tests/chain_acceptance.csv`: C-score of whole bloods, highest | PASS |
| 46 |  | calibrated | `000` | not run: count: 6,000 identity sites in genome order (printed value split at the thousands comma; the count is the length of sites_ordered in neutrophil_reference_v1_1.json and is used by ch:sky:L47:120) | - |
| 47 | ch:sky:L47:50 | calibrated | `50` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: sky block size | PASS |
| 47 | ch:sky:L47:120 | calibrated | `120` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: blocks per sky map | PASS |
| 48 | ch:sky:L48 | calibrated | `0.3133` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy mean H per site, median | PASS |
| 49 | ch:sky:L49 | calibrated | `0.0114` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: shrunk SD s_i, median | PASS |
| 50 | ch:sky:L50 | calibrated | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: healthy clustering baseline (table) | PASS |
| 67 | eq:cellcount | derived |  | sympy: binomial copy-number floor sigma = sqrt(beta(1-beta)/2N) | PASS |
| 72 |  | calc | `10` | not run: input: plasma cell-free DNA carries about 10^3 genome equivalents per millilitre in health (Sender2024); the printed '10' is the base of 10^3 | - |
| 73 | ch:sky:L73 | calc | `5\times10^3` | numeric: diploid genomes for 10^4 copies | PASS |
| 73 |  | calc | `10` | not run: input: a draw yields of order 10^3-10^4 copies of a site (from about 10^3 genome equivalents per millilitre, Sender2024, and the draw volume); the printed '10' is the base of the power | - |
| 74 | ch:sky:L74 | calc | `0.0046` | numeric: copy-number floor at 10^4 copies, beta = 0.7 | PASS |
| 74 |  | calc | `0.7` | not run: input: illustrative beta = 0.7 at which the copy-number floor is evaluated | - |
| 80 |  | calc | `0.7` | not run: input: beta = 0.7 of the figure's curve (same illustrative value as line 74) | - |
| 81 |  | calc | `10` | not run: input: vertical lines of the figure at 10^3 and 10^4 copies (the plasma-draw range of line 73); the printed '10' is the base of the power | - |
| 81 |  | calc | `0.01` | not run: input: illustrative array measurement noise 0.01 (figure reference line) | - |
| 81 |  | calc | `0.02` | not run: input: illustrative array measurement noise 0.02 (figure reference line) | - |
| 90 | eq:outspan | derived |  | sympy: out-of-span residual is blind to composition error | PASS |

## Part 6 - ch:serial - `docs/book/part6/p6_17_serial.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 31 | ch:serial:L31 | measured | `0.045` | file `Biological_Physics/MethylPhys/doors/data/t2_diag.csv`: within-person SD, donor 1 (GSE247195), untared isolated neutrophils | PASS |
| 35 | ch:serial:L71 | measured | `0.894` | file `Biological_Physics/MethylPhys/doors/PLAN.md`: E-MTAB-7309 Stage 1: median call rate | PASS |

## Part 6 - ch:discipline - `docs/book/part6/p6_18_discipline.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | ch:discipline:L24 | measured | `0.015` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: severe vs healthy above Normal: one-sided Fisher p | PASS |
| 24 | ch:discipline:L24:0.79 | measured | `0.79` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: median neutrophil fraction, severe infection | PASS |
| 24 | ch:discipline:L24:0.65 | measured | `0.65` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: median neutrophil fraction, healthy (NEGATIVE) | PASS |
| 25 | ch:discipline:L25 | measured | `+0.0075` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: severity term with fraction in the model | PASS |
| 25 | ch:discipline:L25:0.42 | measured | `0.42` | file `Biological_Physics/MethylPhys/doors/data/neut_test_T1T3T4_readings.csv`: p of the severity term with fraction in the model | PASS |
| 57 | ch:discipline:L57 | measured | `+0.061` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: planted 2 % loss: shift of the tared reading | PASS |
| 58 |  | measured | `1.05` | not run: definition: 1.05 is the upper edge of the Normal band (0.95-1.05), the line the planted readings are counted against; the reading itself is checked in ch:discipline:L57 | - |
| 68 | ch:discipline:L68 | measured | `0.053` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: planted loss, fraction from markers: smallest rise | PASS |
| 68 | ch:discipline:L68:0.067 | measured | `0.067` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: planted loss, fraction from markers: largest rise | PASS |
| 68 | ch:discipline:L68:0.044 | measured | `0.044` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: planted loss, fraction re-fitted: smallest rise | PASS |
| 68 | ch:discipline:L68:0.056 | measured | `0.056` | file `Biological_Physics/MethylPhys/doors/data/selfconsist.csv`: planted loss, fraction re-fitted: largest rise | PASS |

## Part 6 - ch:chain - `docs/book/part6/p6_19_chain.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 23 | ch:chain:L23 | calibrated | `50` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: Stage MC block size | PASS |
| 23 | ch:chain:L23:1.1104 | calibrated | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: Stage MC healthy C-score baseline (median of leave-one-out) | PASS |
| 25 | ch:chain:L25 | calibrated | `1.099` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: Stage Q neutrophil position P | PASS |
| 72 | ch:chain:L72 | openprob | `1.1104` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy baseline (Stage MC text) | PASS |

## Part 6 - ch:leukocyte - `docs/book/part6/p6_22_leukocyte.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 16 |  | conjecture | `0.95` | not run: definition: 0.95 is the lower edge of the Normal band (0.95-1.05) that defines 'Below Normal' | - |
| 19 |  | conjecture | `1.05` | not run: definition: 1.05 is the upper edge of the Normal band (0.95-1.05) that defines 'Above Normal' | - |
| 45 | ch:leukocyte:L45 | observed | `0.23` | file `Biological_Physics/MethylPhys/doors/PROC_PREDX_NEUT_01_OUTCOME.md`: 450K development reading: rank correlation with age among controls | PASS |

## Part 6 - part4:ch:reach - `docs/book/part6/p6_23_reach.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 30 |  | prediction | `1.00` | not run: prediction, nothing to recompute: each lineage of a blood-cancer specimen read against its own healthy floor, against 1.00 (the healthy reference value of Met-A) | - |
| 41 | part4:ch:reach:L41 | measured | `0.005` | file `Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_PREREG.md`: instrument bar on the conversion-failure difference | PASS |
| 41 | part4:ch:reach:L41:1.183 | measured | `1.183` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: colorectal pairs under the bar: median ratio | PASS |
| 51 | part4:ch:reach:L51 | measured | `0.005` | file `Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_PREREG.md`: instrument bar (figure caption) | PASS |
| 59 | part4:ch:reach:L59 | measured | `0.03514` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC1 WGBS: eps normal | PASS |
| 59 | part4:ch:reach:L59:0.04668 | measured | `0.04668` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC1 WGBS: eps tumour | PASS |
| 59 | part4:ch:reach:L59:1.328 | measured | `1.328` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC1 WGBS: ratio | PASS |
| 59 | part4:ch:reach:L59:0.0027 | measured | `0.0027` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC1 WGBS: conversion difference | PASS |
| 60 | part4:ch:reach:L60 | measured | `0.03550` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC2 WGBS: eps normal | PASS |
| 60 | part4:ch:reach:L60:0.04269 | measured | `0.04269` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC2 WGBS: eps tumour | PASS |
| 60 | part4:ch:reach:L60:1.203 | measured | `1.203` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC2 WGBS: ratio | PASS |
| 60 | part4:ch:reach:L60:0.0009 | measured | `0.0009` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC2 WGBS: conversion difference | PASS |
| 61 | part4:ch:reach:L61 | measured | `0.03268` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC3 WGBS: eps normal | PASS |
| 61 | part4:ch:reach:L61:0.03865 | measured | `0.03865` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC3 WGBS: eps tumour | PASS |
| 61 | part4:ch:reach:L61:1.183 | measured | `1.183` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC3 WGBS: ratio | PASS |
| 61 | part4:ch:reach:L61:0.0032 | measured | `0.0032` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC3 WGBS: conversion difference | PASS |
| 62 | part4:ch:reach:L62 | measured | `0.03275` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC4 WGBS: eps normal | PASS |
| 62 | part4:ch:reach:L62:0.03646 | measured | `0.03646` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC4 WGBS: eps tumour | PASS |
| 62 | part4:ch:reach:L62:1.113 | measured | `1.113` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC4 WGBS: ratio | PASS |
| 62 | part4:ch:reach:L62:0.0001 | measured | `0.0001` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC4 WGBS: conversion difference | PASS |
| 63 | part4:ch:reach:L63 | measured | `0.03715` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC5 WGBS: eps normal | PASS |
| 63 | part4:ch:reach:L63:0.03963 | measured | `0.03963` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC5 WGBS: eps tumour | PASS |
| 63 | part4:ch:reach:L63:1.067 | measured | `1.067` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC5 WGBS: ratio | PASS |
| 63 | part4:ch:reach:L63:0.0051 | measured | `0.0051` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC5 WGBS: conversion difference | PASS |
| 64 | part4:ch:reach:L64 | measured | `0.03866` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC6 WGBS: eps normal | PASS |
| 64 | part4:ch:reach:L64:0.04216 | measured | `0.04216` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC6 WGBS: eps tumour | PASS |
| 64 | part4:ch:reach:L64:1.090 | measured | `1.090` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC6 WGBS: ratio | PASS |
| 64 | part4:ch:reach:L64:0.0001 | measured | `0.0001` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour CRC6 WGBS: conversion difference | PASS |
| 65 | part4:ch:reach:L65 | measured | `0.03692` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 WGBS: eps normal | PASS |
| 65 | part4:ch:reach:L65:0.04079 | measured | `0.04079` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 WGBS: eps tumour | PASS |
| 65 | part4:ch:reach:L65:1.105 | measured | `1.105` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 WGBS: ratio | PASS |
| 65 | part4:ch:reach:L65:0.0010 | measured | `0.0010` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 WGBS: conversion difference | PASS |
| 66 | part4:ch:reach:L66 | measured | `0.03888` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 oxWGBS: eps normal | PASS |
| 66 | part4:ch:reach:L66:0.04160 | measured | `0.04160` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 oxWGBS: eps tumour | PASS |
| 66 | part4:ch:reach:L66:1.070 | measured | `1.070` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC1 oxWGBS: ratio | PASS |
| 66 |  | measured | `0.0004` | not run: measured: OSCC1 oxWGBS conversion-failure difference |0.004662 - 0.005079| = 0.000417 from tumour_readings.csv (PROC-TUMOUR-01) rounds to the printed 0.0004; a one-digit value cannot carry the 5 % negative control (0.00042 lies within half its last digit of 0.000417) | - |
| 67 | part4:ch:reach:L67 | measured | `0.03557` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 WGBS: eps normal | PASS |
| 67 | part4:ch:reach:L67:0.03668 | measured | `0.03668` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 WGBS: eps tumour | PASS |
| 67 | part4:ch:reach:L67:1.031 | measured | `1.031` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 WGBS: ratio | PASS |
| 67 | part4:ch:reach:L67:0.0008 | measured | `0.0008` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 WGBS: conversion difference | PASS |
| 68 | part4:ch:reach:L68 | measured | `0.03915` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 oxWGBS: eps normal | PASS |
| 68 | part4:ch:reach:L68:0.03334 | measured | `0.03334` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 oxWGBS: eps tumour | PASS |
| 68 | part4:ch:reach:L68:0.851 | measured | `0.851` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 oxWGBS: ratio | PASS |
| 68 | part4:ch:reach:L68:0.0005 | measured | `0.0005` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC2 oxWGBS: conversion difference | PASS |
| 69 | part4:ch:reach:L69 | measured | `0.03893` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 WGBS: eps normal | PASS |
| 69 | part4:ch:reach:L69:0.04678 | measured | `0.04678` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 WGBS: eps tumour | PASS |
| 69 | part4:ch:reach:L69:1.202 | measured | `1.202` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 WGBS: ratio | PASS |
| 69 | part4:ch:reach:L69:0.0004 | measured | `0.0004` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 WGBS: conversion difference | PASS |
| 70 | part4:ch:reach:L70 | measured | `0.04064` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 oxWGBS: eps normal | PASS |
| 70 | part4:ch:reach:L70:0.04567 | measured | `0.04567` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 oxWGBS: eps tumour | PASS |
| 70 | part4:ch:reach:L70:1.124 | measured | `1.124` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 oxWGBS: ratio | PASS |
| 70 | part4:ch:reach:L70:0.0004 | measured | `0.0004` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC3 oxWGBS: conversion difference | PASS |
| 71 | part4:ch:reach:L71 | measured | `0.03623` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 WGBS: eps normal | PASS |
| 71 | part4:ch:reach:L71:0.03899 | measured | `0.03899` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 WGBS: eps tumour | PASS |
| 71 | part4:ch:reach:L71:1.076 | measured | `1.076` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 WGBS: ratio | PASS |
| 71 | part4:ch:reach:L71:0.0005 | measured | `0.0005` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 WGBS: conversion difference | PASS |
| 72 | part4:ch:reach:L72 | measured | `0.03917` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 oxWGBS: eps normal | PASS |
| 72 | part4:ch:reach:L72:0.04107 | measured | `0.04107` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 oxWGBS: eps tumour | PASS |
| 72 | part4:ch:reach:L72:1.049 | measured | `1.049` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 oxWGBS: ratio | PASS |
| 72 | part4:ch:reach:L72:0.0005 | measured | `0.0005` | heavy file `Biological_Physics/MethylPhys/doors/data/tumour_readings.csv`: Table tab:p4_tumour OSCC4 oxWGBS: conversion difference | PASS |
| 79 |  | measured | `10` | not run: restates Chapter ch:sky (p6_16_sky.tex L72-73): about 10^3 genome equivalents per millilitre of plasma (Sender2024), so a draw yields of order 10^3-10^4 copies of a site; an order of magnitude carried over, nothing to recompute here | - |
| 82 | part4:ch:reach:L82 | measured | `0.5` | heavy file `Biological_Physics/MethylPhys/doors/PROC_MOLECULE_01_OUTCOME.md`: constructed mixtures: fewest molecules | PASS |
| 109 |  | prediction | `1.00` | not run: prediction, nothing to recompute: sorted healthy canine cells held out of a canine reference read 1.00 within tolerance | - |

## Part 6 - ch:status - `docs/book/part6/p6_24_status.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 9 | ch:status:L9 | calc | `2.97\times10^{-21}` | numeric: Landauer cost per bit at 37 C | PASS |
| 9 |  | calc | `37` | not run: input: T_cell = 310.15 K, 37 C (CANON T_cell) | - |
| 10 | ch:status:L10 | calc | `20.94` | numeric: Mahaffey number of the cell M = dG_ATP/(R T) | PASS |
| 10 | ch:status:L10:30.2 | calc | `30.2` | numeric: Landauer bits per ATP, M/ln 2 | PASS |
| 19 | ch:status:L19 | calc | `0.330263` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: EPIC neutrophil floor | PASS |
| 19 |  | calc | `000` | not run: count: the neutrophil reference's 6,000 identity sites (the inventory read '000' from '6,000'); a design choice of the frozen reference (metA_floors_v1_3.json n_sites), checked by ch:status:L19 reading the same file | - |
| 20 | ch:status:L20 | calc | `0.020` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out Met-A of the six reference arrays, SD | PASS |
| 22 | ch:status:L22 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy E_hold | PASS |
| 22 | ch:status:L22:0.032 | measured | `0.032` | numeric: eps0 = 1/(1+e^(E_hold/k_B T)) | PASS |
| 22 | ch:status:L22:0.163 | measured | `0.163` | numeric: phi = E_hold/M | PASS |
| 23 | ch:status:L23 | measured | `1.099` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: neutrophil position P | PASS |
| 23 | ch:status:L23:1.084 | calc | `1.084` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: neutrophil position P, lowest donor | PASS |
| 24 | ch:status:L24 | calc | `1.1104` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy baseline | PASS |
| 24 | ch:status:L24:0.70 | calc | `0.70` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score of the healthy arrays, lowest | PASS |
| 24 | ch:status:L24:1.23 | calc | `1.23` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score of the healthy arrays, highest | PASS |
| 26 | ch:status:L26 | calc | `0.910` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: IAM-A floor 1/P | PASS |
| 26 | ch:status:L26:3.03 | calc | `3.03` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Met-A at the full surface | PASS |
| 26 | ch:status:L26:4.45 | calc | `4.45` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: IAM-A at the full surface | PASS |
| 26 | ch:status:L26:45 | calc | `45` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score far end | PASS |
| 29 | ch:status:L29 | calc | `0.78` | numeric: copy-error floor at 10 C, fixed holding energy | PASS |
| 29 | ch:status:L29:1.012 | calc | `1.012` | numeric: copy-error floor at 38.5 C, fixed holding energy | PASS |
| 29 |  | calc | `10` | not run: input: the temperature 10 C at which the floor ratio is evaluated (ch:status:L29 computes the ratio) | - |
| 29 |  | calc | `38.5` | not run: input: the temperature 38.5 C at which the floor ratio is evaluated (ch:status:L29:1.012 computes the ratio) | - |
| 33 | ch:status:L33 | calc | `92.7` | heavy file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: atlas v2 held-out coverage | PASS |
| 34 |  | calc | `95` | not run: definition: the pre-registered bar, 95 % of held-out readings in Normal (Chapter ch:atlas) | - |
| 34 |  | calc | `87.7` | not run: measured: recorded only in the _provenance.tests field of Biological_Physics/MethylPhys/atlas/v2/postbuild/runtime/iamatlas_v2_identity_loci_v1_1.json ('array->array 87.7% of held-out readings in NORMAL, 28/29 cell medians'), a 15 MB file above the DATA_FILES size limit; restates Chapter ch:atlas L181 | - |
| 34 |  | calc | `73.1` | not run: measured: recorded only in the _provenance.tests field of Biological_Physics/MethylPhys/atlas/v2/postbuild/runtime/iamatlas_v2_identity_loci_v1_1.json ('Loyfer->array with this correction 73.1%, 14/17 cell medians'), a 15 MB file above the DATA_FILES size limit; restates Chapter ch:atlas L182 | - |
| 44 |  | calc | `100` | not run: input: the 100 nM dose of the single-molecule DNMT1 inhibitor libraries (Part B design) | - |

## Part 7 - ch:theoryinterp - `docs/book/part7/p7_01_interpretation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 30 | ch:theoryinterp:L30 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 30 | ch:theoryinterp:L30:2.3\times10^{22} | calc | `2.3\times10^{22}` | numeric: M_eq = c^3/4GH0 today, solar masses | PASS |
| 30 | ch:theoryinterp:L30:1.3\times10^{22} | calc | `1.3\times10^{22}` | numeric: M_eq = c^3/4GH(z) at z = 1 | PASS |
| 30 | ch:theoryinterp:L30:2.4\times10^{12} | calc | `2.4\times10^{12}` | numeric: M_eq = c^3/4GH(z) at z = 10^6 | PASS |
| 30 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153, Planck 2018 (Aghanim et al. 2020, doi 10.1051/0004-6361/201833910), the figure's input | - |
| 30 |  | calc | `9.1\times10^{-5}` | not run: input: Omega_r = 9.1e-5, the figure's stated radiation density input (docs/book/figscripts/fig_p5_extra.py line 22); used by ch:theoryinterp:L30:1.3\times10^{22} and L30:2.4\times10^{12} | - |
| 30 |  | calc | `10` | not run: input: the base of a power of ten (z = 10^6, a redshift chosen for the figure; ~7x10^10), not a computed number; the 7x10^10 is checked as ch:theoryinterp:L41:7\times10^{10} | - |
| 36 | ch:theoryinterp:L36 | none |  | sympy: S_BH/A at the Schwarzschild radius = 1/4 l_P^2 | PASS |
| 40 | ch:theoryinterp:L40 | none |  | sympy: T_BH = T_GH solved for M gives M_eq = c^3/4GH | PASS |
| 41 | ch:theoryinterp:L41 | observed | `10.82` | numeric: TON 618 black-hole mass, log M/Msun (published) | PASS |
| 41 | ch:theoryinterp:L41:7\times10^{10} | calc | `7\times10^{10}` | numeric: TON 618 mass about 7e10 Msun from log M = 10.82 | PASS |
| 54 | ch:theoryinterp:L54 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 54 | ch:theoryinterp:L54:2.65\times10^{-30} | calc | `2.65\times10^{-30}` | numeric: cosmic-horizon temperature T_GH = hbar H0/2 pi k_B | PASS |
| 54 | ch:theoryinterp:L54:2.3\times10^{22} | calc | `2.3\times10^{22}` | numeric: M_eq where T_BH = T_GH today, solar masses | PASS |

## Part 7 - ch:time - `docs/book/part7/p7_03_time.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | eq:time_tau | none |  | not run: definition: proper time as the length of a worldline, tau = (1/c) int sqrt(-g dx dx) (Eq. eq:time_tau) | - |
| 43 | ch:time:L43 | calc | `4.5\times10^{-5}` | numeric: accumulated record E(z=10) = exp(-z) | PASS |
| 60 | ch:time:L60 | derived |  | sympy: properties of E(a): limits, E(1) = 1, dE/da > 0 | PASS |
| 78 | eq:time_Ephoton | derived |  | not run: definition: E(a)|photon = 0 (Eq. eq:time_Ephoton), the sector split stated by construction (no record on a null worldline); nothing to recompute | - |
| 92 | ch:time:L92 | measured | `13.8` | heavy file `docs/verification/scripts/verify_records_measurement_time_output.txt`: measured: printed value found in verify_records_measurement_time_output.txt, a file the chapter names | PASS |
| 93 | ch:time:L93 | measured | `67.16` | heavy file `docs/verification/scripts/verify_records_measurement_time_output.txt`: measured: printed value found in verify_records_measurement_time_output.txt, a file the chapter names | PASS |
| 94 | ch:time:L94 | measured | `67.36` | file `docs/verification/PAPER_ERRATA.md`: measured: printed value found in PAPER_ERRATA.md, a file the chapter names | PASS |
| 96 | ch:time:L96 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 97 | ch:time:L97 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 97 | ch:time:L97:72.26 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 97 | ch:time:L97:73.04 | observed | `73.04` | numeric: SH0ES H0 (Riess et al. 2022, published) | PASS |
| 102 | ch:time:L102 | derived | `0.15765` | numeric: beta_m = Omega_m/2 (virial partition), Planck Omega_m | PASS |
| 113 | ch:time:L113 | derived |  | sympy: growth source term 4 pi G rho_m mu = (3/2) Omega_m H0^2 a^-3 mu | PASS |
| 133 | ch:time:L133 | calc | `10^{-15}` | numeric: scale factor at electroweak breaking a_EW ~ 10^-15 | PASS |

## Part 7 - ch:virial_partners - `docs/book/part7/p7_05b_virial_partners.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 17 | ch:virial_partners:L17 | conjecture | `2.0` | numeric: virial ratio R(a=1) = Omega_m/[beta_m E(1)] | PASS |
| 29 | ch:virial_partners:L29 | calc | `0.38` | numeric: dark matter to dark energy ratio today, 0.26/0.69 | PASS |
| 36 | ch:virial_partners:L36 | observed | `10^{-11}` | numeric: atomic scale: the Bohr radius lies in the 10^-11 m decade | PASS |
| 37 | ch:virial_partners:L37 | observed | `10^{26}` | numeric: cosmic horizon scale c/H0 ~ 10^26 m | PASS |
| 68 | ch:virial_partners:L68 | calc | `399` | numeric: virial ratio R at z = 2 | PASS |
| 68 | ch:virial_partners:L68:20 | calc | `20` | numeric: virial ratio R at z = 0.7 | PASS |
| 68 |  | calc | `0.7` | not run: input: the redshift z = 0.7 at which R is evaluated (R(z=0.7) = 20 is checked as ch:virial_partners:L68:20) | - |
| 69 | ch:virial_partners:L69 | calc | `6` | numeric: virial ratio R at z = 0.3 | PASS |
| 87 | ch:virial_partners:L87 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 87 | ch:virial_partners:L87:0.361 | calc | `0.361` | numeric: matter equals vacuum plus record term at z = 0.361 | PASS |
| 87 | ch:virial_partners:L87:0.295 | calc | `0.295` | numeric: LCDM matter-vacuum equality redshift | PASS |
| 87 | ch:virial_partners:L87:4.9 | calc | `4.9` | numeric: record share of dark energy at z = 1.5 | PASS |
| 87 | ch:virial_partners:L87:18.7 | calc | `18.7` | numeric: record share of dark energy today | PASS |
| 87 | ch:virial_partners:L87:7.59 | calc | `7.59` | numeric: sector gap H_m/H - 1 = mu^-1/2 - 1 today | PASS |
| 87 | ch:virial_partners:L87:7.3 | calc | `7.3` | numeric: kinetic-half growth over matter dilution at z = 0.3 | PASS |
| 87 | ch:virial_partners:L87:2.9 | calc | `2.9` | numeric: kinetic-half growth over matter dilution at z = 0.7 | PASS |
| 87 | ch:virial_partners:L87:0.6 | calc | `0.6` | numeric: kinetic-half growth over matter dilution at z = 1.5 | PASS |
| 87 | ch:virial_partners:L87:Ea2/6 | calc |  | sympy: rate ratio closed form E(a) a^2/6 | PASS |
| 87 |  | calc | `0.7` | not run: input: z = 0.7, upper edge of the transition zone and an evaluation redshift of panel (f) (its value 2.9 % is checked as ch:virial_partners:L87:2.9) | - |
| 87 |  | calc | `1.5` | not run: input: the redshift z = 1.5 at which panels (c) and (f) are read (checked as ch:virial_partners:L87:4.9 and L87:0.6) | - |
| 87 |  | calc | `0.3` | not run: input: z = 0.3, lower edge of the transition zone and an evaluation redshift of panel (f) (checked as ch:virial_partners:L87:7.3) | - |
| 90 |  | calc | `0.7` | not run: definition: the transition zone z = 0.3-0.7 (its edge 0.7), a range named for the figure, nothing to recompute | - |
| 93 | ch:virial_partners:L93 | calc | `18.7` | numeric: record share today beta_m/(Omega_L + beta_m) | PASS |
| 97 | ch:virial_partners:L97 | calc | `0.078825` | numeric: three channels: geometric = beta_m/2, sum = Omega_m/2 | PASS |
| 102 | ch:virial_partners:L102 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 143 | ch:virial_partners:L143 | interp | `2.0` | numeric: R(a=1) = 2.0 (repeat) | PASS |
| 151 | ch:virial_partners:L151 | prediction | `-0.136` | numeric: mu0 = mu(a=1) - 1 from beta_m = Omega_m/2 | PASS |
| 155 | ch:virial_partners:L155 | interp | `-0.136` | numeric: mu0 = mu(a=1) - 1 (repeat) | PASS |

## Part 7 - ch:virial_decoherence - `docs/book/part7/p7_05c_virial_decoherence.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 | ch:virial_decoherence:L19 | interp | `13.8` | numeric: age of the universe, flat LCDM, Planck 2018 | PASS |
| 23 | ch:virial_decoherence:L23 | interp | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Delta chi2, IAM Run A minus LCDM Run C | PASS |
| 73 | ch:virial_decoherence:L73 | calc | `2.65\times10^{-30}` | numeric: Gibbons-Hawking temperature today, H0 = 67.16 | PASS |
| 74 | ch:virial_decoherence:L74 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 81 | ch:virial_decoherence:L81 | calc | `4.5\times10^{-5}` | numeric: E(z = 10) = exp(1 - 1/a) | PASS |
| 92 | ch:virial_decoherence:L92 | derived | `13.8` | numeric: age of the universe (repeat) | PASS |
| 100 | ch:virial_decoherence:L100 | openprob | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 photon sector, Level 2 chain value | PASS |
| 106 | ch:virial_decoherence:L106 | conjecture |  | sympy: saturation bound S/A = k_B/4 l_P^2 at R_s | PASS |
| 111 | ch:virial_decoherence:L111 | derived |  | sympy: Hawking power and bit rate c^3/(1920 G M ln2) | PASS |
| 115 | ch:virial_decoherence:L115 | calc | `2.1\times10^{67}` | numeric: Hawking evaporation time of a solar-mass black hole | PASS |
| 160 | ch:virial_decoherence:L160 | interp | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 IAM Level 2 chain value | PASS |
| 175 | ch:virial_decoherence:L175 | prediction | `13.6` | numeric: coupling reduction today 1 - mu(1) | PASS |
| 176 | ch:virial_decoherence:L176 | prediction | `4.25` | numeric: f sigma8 deficit at z = 0, same early amplitude | PASS |

## Part 7 - ch:onegauge - `docs/book/part7/p7_08_one_gauge.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 21 |  | calc | `20` | not run: input: transistor count (20.0-20.6)x10^9 of the 9950X from die-level reports (\cite{AMD9950X}); E_sw and M from it are checked as ch:onegauge:L22 and L22:411 | - |
| 21 |  | calc | `4.3` | not run: input: clock 4.3 GHz of the 9950X, manufacturer specification (\cite{AMD9950X}); used by ch:onegauge:L22 | - |
| 21 |  | calc | `75` | not run: input: junction temperature T_j = 75 C, stated operating point of the chip; used by ch:onegauge:L22 and L22:0.0017 | - |
| 22 | ch:onegauge:L22 | calc | `399` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 22 | ch:onegauge:L22:411 | calc | `411` | numeric: drafted check, screened (runs; negative control fails) (tolerance: E_sw is printed to 3 figures) | PASS |
| 22 | ch:onegauge:L22:0.0017 | calc | `0.0017` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 41 | ch:onegauge:L41 | calc | `10` | numeric: transmon worked example: error-correction threshold at A = 10 | PASS |
| 42 | ch:onegauge:L42 | calc | `0.2043` | numeric: same value as p6_00b_astrogenetics:63 (H(eps0), bits) | PASS |
| 42 | ch:onegauge:L42:3.03 | calc | `3.03` | numeric: full surface on Met-A: 1/(healthy reference) | PASS |
| 42 | ch:onegauge:L42:4.45 | calc | `4.45` | numeric: full surface on IAM-A: 1/(P H(eps0)), neutrophils | PASS |
| 42 | ch:onegauge:L42:0.032 | calc | `0.032` | numeric: eps0 = 1/(1 + exp(phi M)) | PASS |
| 63 | ch:onegauge:L63 | openprob | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy E_hold from the PROC-CHANNEL-01 record | PASS |
| 64 | ch:onegauge:L64 | openprob | `0.032` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: eps0 from the measured holding energy | PASS |
| 66 | ch:onegauge:L66 | openprob | `1.9` | numeric: Hopfield energy gap ln 7 (low end) | PASS |
| 66 | ch:onegauge:L66:4.4 | openprob | `4.4` | numeric: Hopfield energy gap ln 80 (high end) | PASS |
| 67 | ch:onegauge:L67 | openprob | `20.94` | numeric: Mahaffey number M = dG_ATP/(R T_cell) | PASS |
| 79 | ch:onegauge:L79 | derived |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 83 |  | none |  | not run: definition: Met-A = <H(beta)> at identity sites over <H(beta)> of the healthy reference | - |
| 94 | ch:onegauge:L94 | derived | `0.032` | numeric: IAM-A floor eps0 = 1/(1 + e^{phi M}) | PASS |
| 110 | ch:onegauge:L110 | measured | `1.05` | file `CANON/iam_canon.json`: Normal band, upper edge (canon Normal_band) | PASS |
| 110 | ch:onegauge:L110:0.695 | measured | `0.695` | heavy file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 senescent, unmethylated channel, upper end | PASS |
| 110 | ch:onegauge:L110:1.120 | measured | `1.120` | heavy file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, methylated channel, upper end | PASS |
| 110 | ch:onegauge:L110:0.664 | measured | `0.664` | heavy file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, unmethylated channel, upper end | PASS |
| 110 | ch:onegauge:L110:0.975 | measured | `0.975` | heavy file `Biological_Physics/MethylPhys/doors/PROC_LINES_02_channels/imr90_channels.csv`: IMR90 SV40, both channels, upper end | PASS |
| 110 | ch:onegauge:L110:0.95 | measured | `0.95` | file `CANON/iam_canon.json`: Normal band lower edge (CANON) | PASS |
| 143 | ch:onegauge:L143 | measured | `3.41` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 143 | ch:onegauge:L143:3.77 | measured | `3.77` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 143 | ch:onegauge:L143:1.9 | observed | `1.9` | numeric: Hopfield gap ln 7 (figure caption) | PASS |
| 143 | ch:onegauge:L143:4.4 | observed | `4.4` | numeric: Hopfield gap ln 80 (figure caption) | PASS |
| 154 |  | observed | `21` | not run: measured, source not named | - |
| 154 |  | observed | `30` | not run: measured, source not named | - |
| 154 |  | observed | `40` | not run: measured, source not named | - |
| 154 |  | observed | `80` | not run: measured, source not named | - |
| 156 | ch:onegauge:L156 | observed | `3.41` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 156 | ch:onegauge:L156:3.77 | observed | `3.77` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 156 | ch:onegauge:L156:1.9 | observed | `1.9` | numeric: Delta E = ln 7 (Hopfield) | PASS |
| 156 | ch:onegauge:L156:4.4 | observed | `4.4` | numeric: Delta E = ln 80 (Hopfield) | PASS |

## Part 7 - ch:synthesis - `docs/book/part7/p7_08_synthesis.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 29 | ch:synthesis:L29 | derived | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 29 | ch:synthesis:L29:2.65\times10^{-30} | derived | `2.65\times10^{-30}` | numeric: cosmic-horizon temperature T_GH, H0 = 67.16 | PASS |
| 29 | ch:synthesis:L29:-0.136 | calc | `-0.136` | numeric: mu0 = mu(a=1) - 1 | PASS |
| 30 | ch:synthesis:L30 | derived | `6.2\times10^{-8}` | numeric: Hawking temperature of a solar-mass black hole | PASS |
| 31 |  | derived | `35` | not run: input: the transmon's own temperature 35 mK, a published measurement (\cite{Jin2015} in ch:onegauge); used by ch:synthesis:L87 | - |
| 31 |  | derived | `15` | not run: input: the 15 mK refrigerator stage, an operating condition stated with the 35 mK reading; nothing to recompute | - |
| 32 | ch:synthesis:L32 | calc | `576` | numeric: 9950X switching energy over the Landauer floor, lower end | PASS |
| 32 | ch:synthesis:L32:593 | calc | `593` | numeric: 9950X switching energy over the Landauer floor, upper end | PASS |
| 32 |  | derived | `-350` | not run: input: the junction temperature range 300-350 K of a transistor (the scan read '300--350' as -350), an operating range, nothing to recompute | - |
| 32 |  | derived | `9950` | not run: definition: '9950' is part of the processor's model name (Ryzen 9 9950X), not a quantity | - |
| 33 | ch:synthesis:L33 | derived | `0.032` | numeric: cell floor eps0 = 1/(1 + exp(phi M)) | PASS |
| 33 | ch:synthesis:L33:0.910 | derived | `0.910` | numeric: IAM-A floor of neutrophils, 1/P_cell | PASS |
| 33 | ch:synthesis:L33:3.03 | derived | `3.03` | numeric: full surface on Met-A, 1/(healthy reference) | PASS |
| 33 | ch:synthesis:L33:4.45 | derived | `4.45` | numeric: full surface on IAM-A, 1/(P H(eps0)) | PASS |
| 51 | ch:synthesis:L51 | derived | `2.112` | numeric: Al superconducting gap expressed as temperature | PASS |
| 51 | ch:synthesis:L51:310.15 | derived | `310.15` | numeric: body temperature 37 C in kelvin | PASS |
| 53 | ch:synthesis:L53 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 53 | ch:synthesis:L53:0.032 | measured | `0.032` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: eps0 from the measured holding energy | PASS |
| 53 | ch:synthesis:L53:3.41 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy E_hold, PROC-CHANNEL-01 | PASS |
| 54 | ch:synthesis:L54 | calc | `-0.136` | numeric: mu0 = mu(a=1) - 1 (slot table) | PASS |
| 54 | ch:synthesis:L54:0.7998 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 IAM Level 2 chain | PASS |
| 64 | ch:synthesis:L64 | derived | `0.500000` | numeric: capacitor charging: dissipated over drawn energy, any R | PASS |
| 64 |  | derived | `10` | not run: input: the resistance range 10^-2 to 10^2 ohm over which the capacitor ratio is computed (checked as ch:synthesis:L64) | - |
| 69 | ch:synthesis:L69 | derived | `-0.136` | numeric: mu0 from beta_m = Omega_m/2 | PASS |
| 77 |  | measured | `1.8` | not run: measured, source not named | - |
| 79 | ch:synthesis:L79 | fitted | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Delta chi2, IAM minus LCDM | PASS |
| 80 | ch:synthesis:L80 | fitted | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 LCDM Level 2 chain (Run C) | PASS |
| 80 | ch:synthesis:L80:4.25 | fitted | `4.25` | numeric: f sigma8 deficit at z = 0 | PASS |
| 80 | ch:synthesis:L80:0.41 | fitted | `0.41` | numeric: f sigma8 deficit at z = 1 | PASS |
| 87 | ch:synthesis:L87 | derived | `6.2\times10^{-7}` | numeric: transmon per-gate thermal floor p_eq t_g/T1 | PASS |
| 87 |  | derived | `68` | not run: input: T_1 = 68 microseconds, a published device value (\cite{GoogleWillow2025} in ch:onegauge); used by ch:synthesis:L87 | - |
| 88 |  | derived | `10` | not run: input: the base 10 of a two-qubit error near 10^-3, an illustrative device value stated in the sentence; nothing to recompute | - |
| 93 | ch:synthesis:L93 | calibrated | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out SD of the six reference arrays | PASS |

## Part 7 - ch:reach - `docs/book/part7/p7_09_reach.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 68 | ch:reach:L68 | calc | `1.012` | numeric: same value as p6_10_temperature:17 (floor at 38.5 C) | PASS |
| 68 | ch:reach:L68:0.78 | calc | `0.78` | numeric: copy-error floor at 10 C relative to 37 C, fixed holding energy | PASS |
| 68 |  | calc | `38.5` | not run: input: a dog's body temperature 38.5 C, the temperature at which the floor ratio 1.012 is evaluated (checked as ch:reach:L68) | - |
| 90 | ch:reach:L90 | calc | `6.2\times10^{-7}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 90 |  | calc | `68` | not run: input: T_1 = 68 microseconds, a published device value (\cite{GoogleWillow2025}); used by ch:reach:L90 | - |
| 92 | ch:reach:L92 | calc | `1.05\times10^{-3}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 92 | ch:reach:L92:1.1\times10^{-6} | calc | `1.1\times10^{-6}` | numeric: held-record floor of a 5 GHz transmon at half of 35 mK | PASS |
| 94 |  | calc | `10` | not run: input: the excess quasiparticle fraction 10^-7, a stated condition of the example (the base 10 of 10^-7), nothing to recompute | - |
| 104 |  | calc | `170` | not run: input: TDP 170 W of the 9950X, manufacturer specification (\cite{AMD9950X}); E_sw from it is checked as ch:onegauge:L22 | - |
| 104 |  | calc | `20` | not run: input: transistor count (20.0-20.6)x10^9 from die-level reports (\cite{AMD9950X}); E_sw from it is checked as ch:onegauge:L22 | - |

## Part 7 - ch:predictions - `docs/book/part7/p7_07_predictions.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 24 | ch:predictions:L24 | derived | `-0.136` | numeric: mu0 = mu(a=1) - 1 (fig:cosmo_tests panel a) | PASS |
| 24 | ch:predictions:L24:+0.064 | measured | `+0.064` | heavy file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_mu0_float.1.txt`: free mu0, Planck + RSD chain posterior median | PASS |
| 24 | ch:predictions:L24:-0.087 | measured | `-0.087` | heavy file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_mu0_float.1.txt`: free mu0, Planck + RSD, lower end of central 68 % | PASS |
| 24 | ch:predictions:L24:+0.163 | measured | `+0.163` | heavy file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_mu0_float.1.txt`: free mu0, Planck + RSD, upper end of central 68 % | PASS |
| 24 | ch:predictions:L24:0.11 | measured | `0.11` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 (recorded value) | PASS |
| 24 | ch:predictions:L24:0.54 | measured | `0.54` | file `docs/verification/theory/IAM_LAW_CHECK.md`: DESI 2024 full-shape mu0 lower error (recorded value) | PASS |
| 24 | ch:predictions:L24:1.8 | calc | `1.8` | numeric: E_G above LCDM at z = 0.3 (C8) | PASS |
| 24 | ch:predictions:L24:0.78 | calc | `0.78` | numeric: growth factor D below LCDM today (C4) | PASS |
| 24 |  | calc | `0.3` | not run: input: the redshift z = 0.3 at which the E_G change is read (checked as ch:predictions:L24:1.8) | - |
| 24 |  | calc | `0.5` | not run: input: the redshift z = 0.5 at which the E_G change is read (checked as ch:predictions:L24:1.1) | - |
| 24 |  | calc | `1.1` | not run: printed value 1.1 % has two significant digits and the recomputed E_G change, 1.144 %, lies within 2.5 % of 1.1 x 1.05: a 5 % change cannot be told from rounding | - |
| 86 | ch:predictions:L86 | observed | `70.0` | numeric: GW170817 standard-siren H0 (published) | PASS |
| 86 | ch:predictions:L86:8.0 | observed | `8.0` | numeric: GW170817 siren H0 lower error (published) | PASS |
| 87 | ch:predictions:L87 | openprob | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 = H0_photon sqrt(1 + beta_m) | PASS |
| 154 | ch:predictions:L154 | observed | `5\times10^{-6}` | numeric: bound on drift of m_p/m_e from H2 quasar absorbers (published) | PASS |
| 154 | ch:predictions:L154:2.0 | observed | `2.0` | numeric: lowest absorber redshift of the m_p/m_e bound (published) | PASS |
| 174 | ch:predictions:L174 | derived |  | sympy: internal-source quasiparticle density, steady state | PASS |
| 191 | ch:predictions:L191 | openprob | `0.24` | numeric: quasiparticle-limited T1 of a 5 GHz Al transmon at x_qp = 1e-7, ms | PASS |

## Part 7 - ch:exploratory - `docs/book/part7/p7_02_exploratory.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 16 | eq:ge_landauer | none |  | not run: definition: Landauer cost k_B T ln 2 per bit at the Gibbons-Hawking horizon temperature T_H = hbar H/(2 pi k_B) (published results, cited); its values are checked at ch:exploratory:L20 to L22 | - |
| 20 | ch:exploratory:L20 | calc | `2.65\times10^{-30}` | numeric: horizon temperature T_H = hbar H/(2 pi k_B) at H0_photon | PASS |
| 21 | ch:exploratory:L21 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 21 | ch:exploratory:L21:2.53\times10^{-53} | calc | `2.53\times10^{-53}` | numeric: Landauer cost k_B T_H ln 2 per bit at H0_photon | PASS |
| 21 | ch:exploratory:L21:2.85\times10^{-30} | calc | `2.85\times10^{-30}` | numeric: horizon temperature T_H = hbar H/(2 pi k_B) at H0_matter | PASS |
| 22 | ch:exploratory:L22 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 22 | ch:exploratory:L22:2.72\times10^{-53} | calc | `2.72\times10^{-53}` | numeric: Landauer cost k_B T_H ln 2 per bit at H0_matter | PASS |
| 27 | ch:exploratory:L27 | prediction | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 28 | ch:exploratory:L28 | prediction |  | sympy: E(1)=1, E->e, rate dE/dlna = E/a peaks at a=1 | PASS |
| 33 | ch:exploratory:L33 | derived | `72.26` | numeric: H_eff,m at a=1 gives the matter-sector H0 | PASS |
| 44 | ch:exploratory:L44 | derived |  | sympy: g = -grad Phi recovers Newtonian gravity | PASS |
| 55 | ch:exploratory:L55 | none |  | sympy: equation of motion from the weak-field Lagrangian | PASS |
| 61 | ch:exploratory:L61 | derived |  | sympy: felt force vanishes on the free-fall trajectory | PASS |
| 67 | ch:exploratory:L67 | derived |  | sympy: tidal residual g(r)-g(r+L) = 2GML/r^3 = L |Phi_rr| | PASS |
| 72 | ch:exploratory:L72 | derived | `3.1\times10^{-6}` | numeric: tidal residual 2GML/R^3 across 10 m at the Earth surface, in g | PASS |
| 73 | ch:exploratory:L73 | calc | `3.1\times10^{-5}` | numeric: tidal residual 2GML/R^3 across 100 m at the Earth surface, in g | PASS |
| 76 |  | derived | `90` | not run: input: illustrative turn angle of the example (90 degrees), nothing to recompute | - |
| 83 | ch:exploratory:L83 | derived |  | sympy: hover: grad Phi_IAM = g_amb solves zero net field | PASS |
| 95 | ch:exploratory:L95 | derived |  | sympy: plumb hang angle tan(theta) = a_h/g_v | PASS |
| 99 | ch:exploratory:L99 | derived | `5.7` | numeric: hang angle at a_h = 0.1 g, degrees | PASS |
| 99 |  | derived | `0.1` | not run: input: horizontal acceleration 0.1 g of the plumb example | - |
| 108 | ch:exploratory:L108 | derived |  | sympy: gravity-gradient torque on a dumbbell | PASS |
| 118 | ch:exploratory:L118 | none |  | sympy: Hubble law from proper distance D = a chi | PASS |
| 123 | eq:ge_DH | calc | `1.46\times10^{10}` | numeric: Hubble radius c/H0 at 67.16, light-years | PASS |
| 123 | eq:ge_DH:1.35\times10^{10} | calc | `1.35\times10^{10}` | numeric: Hubble radius c/H0 at 72.26, light-years | PASS |
| 127 |  | calc | `70` | not run: input: round value H0 = 70 km/s/Mpc used for illustration | - |
| 136 | ch:exploratory:L136 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 136 | ch:exploratory:L136:72.26 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 137 | ch:exploratory:L137 | calc | `1.46` | numeric: Hubble radius at 67.16 in 1e10 ly (figure caption) | PASS |
| 137 | ch:exploratory:L137:1.35\times10^{10} | calc | `1.35\times10^{10}` | numeric: Hubble radius at 72.26 in ly (figure caption) | PASS |
| 137 | ch:exploratory:L137:3\times10^{-8} | calc | `3\times10^{-8}` | numeric: Pleiades distance as a fraction of the Hubble radius | PASS |
| 145 | eq:ge_alc | none |  | not run: definition: Alcubierre metric (published, Alcubierre1994); its centre-line consequence is checked at eq:ge_tau and its energy density at eq:ge_rho | - |
| 153 | ch:exploratory:L153 | derived |  | sympy: centre worldline of the displaced region: ds^2 = -c^2 dt^2 | PASS |
| 161 | ch:exploratory:L161 | derived |  | sympy: clock difference set by the potential difference over c^2 | PASS |
| 176 | ch:exploratory:L176 | derived |  | sympy: Eulerian energy density of the displaced region | PASS |
| 187 |  | calc | `0.99` | not run: input: through-space speed 0.99 c chosen for the figure example | - |
| 188 | ch:exploratory:L188 | calc | `2.3\times10^4` | numeric: rate xi for a seven-day one-way transit to the Pleiades | PASS |
| 196 | eq:ge_tone | none |  | not run: definition: one-way time D/v_eff with v_eff = xi c; 1 ly = c x 1 Julian year, so 444 ly/(xi c) = 444/xi yr | - |
| 202 | eq:ge_xi | calc | `2.3\times10^4` | numeric: xi = 444 yr / 0.0192 yr | PASS |
| 208 | ch:exploratory:L208 | calc | `3\times10^{-8}` | numeric: Pleiades distance as a fraction of the Hubble radius | PASS |
| 214 | ch:exploratory:L214 | calc | `2.3\times10^4` | numeric: transit figure: xi for seven days | PASS |
| 215 | ch:exploratory:L215 | derived | `3.1\times10^{-6}` | numeric: transit figure: tidal residual across 10 m, in g | PASS |
| 215 | ch:exploratory:L215:3.1\times10^{-5} | derived | `3.1\times10^{-5}` | numeric: transit figure: tidal residual across 100 m, in g | PASS |
| 230 | eq:ge_kernel | conjecture |  | not run: conjecture: form of the response kernel of Conjecture C1, nothing to recompute | - |
| 262 | ch:exploratory:L262 | observed | `-1.5` | numeric: MICROSCOPE eta(Ti,Pt) central value | PASS |
| 262 | ch:exploratory:L262:2.7\times10^{-15} | observed | `2.7\times10^{-15}` | numeric: MICROSCOPE combined uncertainty, stat and syst in quadrature | PASS |
| 266 |  | prediction | `10` | not run: prediction, nothing to recompute (target sensitivity 10^-15 of a weighing; the MICROSCOPE level is checked at ch:exploratory:L262) | - |
| 273 | eq:ge_drive | derived |  | not run: definition: superposition of phased sources through the unknown kernel G (conditional form); the steering numbers are checked at ch:exploratory:L298 and L299 | - |
| 298 | ch:exploratory:L298 | derived | `1.00` | numeric: three coplanar sources: mirror/target ratio | PASS |
| 299 | ch:exploratory:L299 | derived | `0.44` | numeric: four non-coplanar sources: mirror/target ratio | PASS |
| 331 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 332 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 333 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 334 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 335 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 336 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 337 | ch:exploratory:L337 | derived | `3.1\times10^{-6}` | numeric: status table: tidal residual across 10 m, in g | PASS |
| 337 | ch:exploratory:L337:3.1\times10^{-5} | derived | `3.1\times10^{-5}` | numeric: status table: tidal residual across 100 m, in g | PASS |
| 337 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 338 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 339 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 340 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 341 | ch:exploratory:L341 | derived | `1.46` | numeric: status table: Hubble radius at 67.16 in 1e10 ly | PASS |
| 341 | ch:exploratory:L341:1.35\times10^{10} | derived | `1.35\times10^{10}` | numeric: status table: Hubble radius at 72.26 in ly | PASS |
| 341 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 342 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 343 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 344 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 345 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 346 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 347 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 348 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 349 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 350 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 351 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 352 |  | calc | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 373 | ch:exploratory:L373 | none |  | sympy: mutual static forces cancel for an even kernel | PASS |
| 395 | ch:exploratory:L395 | derived |  | sympy: radiated power for a given thrust, P = m a c | PASS |
| 399 | ch:exploratory:L399 | derived | `3.34\times10^{-9}` | numeric: thrust per watt 1/c | PASS |
| 399 | ch:exploratory:L399:2.94\times10^{12} | derived | `2.94\times10^{12}` | numeric: power to hold one tonne at 1 g | PASS |
| 399 |  | derived | `100` | not run: input: departure acceleration 100 g of the example | - |
| 400 | ch:exploratory:L400 | calc | `2.94\times10^{14}` | numeric: power for a 100 g departure of one tonne | PASS |
| 415 |  | observed | `10` | not run: order-of-magnitude restatement (about 10^-15) of the MICROSCOPE result checked at ch:exploratory:L262 and ch:exploratory:L262:2.7\times10^{-15} | - |
| 417 | ch:exploratory:L417 | derived |  | sympy: felt load m eta a and the bound on eta | PASS |
| 421 | ch:exploratory:L421 | calc | `1000` | numeric: a 1000 g departure felt below 0.1 g needs the same eta | PASS |
| 421 |  | calc | `100` | not run: input: departure acceleration 100 g of the example | - |
| 421 |  | calc | `0.01` | not run: input: allowed felt load 0.01 g of the example | - |
| 421 |  | calc | `0.1` | not run: input: allowed felt load 0.1 g of the second example | - |
| 431 | ch:exploratory:L431 | derived |  | sympy: tide of a monopole focus 2GML/r^3 = 2aL/r | PASS |
| 435 | ch:exploratory:L435 | derived | `20` | numeric: tide 2aL/r at 100 g, 10 m, focus at 100 m | PASS |
| 435 |  | derived | `100` | not run: input: focus acceleration 100 g (and the focus distance 100 m) of the example | - |
| 435 |  | derived | `10` | not run: input: craft length 10 m of the example | - |
| 436 | ch:exploratory:L436 | calc | `5.9\times10^{23}` | numeric: strength of a 100 g focus at 200 km | PASS |
| 436 |  | calc | `0.01` | not run: input: tide limit 0.01 g of the example | - |
| 436 |  | calc | `100` | not run: input: focus pull 100 g of the example | - |
| 445 | ch:exploratory:L445 | derived | `2.3\times10^4` | numeric: Pleiades rate xi for seven days | PASS |
| 450 | ch:exploratory:L450 | derived |  | sympy: total negative energy -(v^2/12) int r^2 f^2 dr | PASS |
| 455 | ch:exploratory:L455 | derived |  | sympy: linear wall: int r^2 f^2 dr = R^2/Delta + Delta/12 | PASS |
| 461 | ch:exploratory:L461 | calc | `6.9\times10^{62}` | numeric: negative energy, 100 m bubble at c, wall 100 L_P | PASS |
| 461 |  | calc | `6.2\times10^{62}` | not run: input: published value of Pfenning and Ford, |E| = 6.2e62 v_s kg (PfenningFord1997, doi 10.1088/0264-9381/14/7/011), quoted for comparison with the recomputed 6.9e62 (ch:exploratory:L461) | - |
| 462 | ch:exploratory:L462 | calc | `3\times10^{20}` | numeric: negative energy of a 100 m bubble in galaxies of 2e42 kg | PASS |
| 462 | ch:exploratory:L462:2.3\times10^4 | calc | `2.3\times10^4` | numeric: Pleiades rate xi (bubble speed) | PASS |
| 462 |  | calc | `2\times10^{42}` | not run: input: galaxy mass 2e42 kg used for the comparison (PfenningFord1997) | - |
| 463 | ch:exploratory:L463 | calc | `1.6\times10^{67}` | numeric: negative energy at the Pleiades rate | PASS |
| 463 | ch:exploratory:L463:0.56 | calc | `0.56` | numeric: negative energy with a 1 m wall, solar masses | PASS |
| 474 | ch:exploratory:L474 | derived |  | sympy: pair self-force from unequal active/passive mass ratios | PASS |
| 479 | ch:exploratory:L479 | observed | `4\times10^{-12}` | numeric: lunar ranging: m_a/m_p equal for Al and Fe (1986) | PASS |
| 480 | ch:exploratory:L480 | observed | `3.9\times10^{-14}` | numeric: lunar laser ranging: m_a/m_p equal for Al and Fe (2023) | PASS |
| 486 | ch:exploratory:L486 | derived |  | sympy: hover by weight reduction: delta = 1 + m_pay/m_d | PASS |
| 496 | ch:exploratory:L496 | derived | `2.94\times10^{12}` | numeric: summary: power per tonne at 1 g | PASS |
| 498 | ch:exploratory:L498 | derived | `20` | numeric: summary: tide 20 g at 100 m from a 100 g focus | PASS |
| 498 | ch:exploratory:L498:6.9\times10^{62} | derived | `6.9\times10^{62}` | numeric: summary: negative energy, 100 m bubble at c | PASS |
| 498 |  | derived | `100` | not run: input: focus acceleration 100 g (focus distance 100 m) of the example | - |
| 499 | ch:exploratory:L499 | derived | `1.6\times10^{67}` | numeric: summary: negative energy at the Pleiades rate | PASS |
| 502 | ch:exploratory:L502 | conjecture | `3.9\times10^{-14}` | numeric: summary: lunar laser ranging Al/Fe ratio (2023) | PASS |
| 512 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 513 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 514 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 515 | ch:exploratory:L515 | derived | `2.94\times10^{12}` | numeric: status table: power per tonne at 1 g | PASS |
| 515 | ch:exploratory:L515:2.94\times10^{14} | derived | `2.94\times10^{14}` | numeric: status table: power per tonne at 100 g | PASS |
| 515 |  | derived | `100` | not run: input: departure acceleration 100 g of the example | - |
| 515 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 516 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 517 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 518 | ch:exploratory:L518 | derived | `100` | numeric: status table: 100 g felt below 0.01 g needs eta = 1e-4 | PASS |
| 518 |  | derived | `0.01` | not run: input: allowed felt load 0.01 g of the example | - |
| 518 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 519 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 520 | ch:exploratory:L520 | derived | `20` | numeric: status table: tide 20 g | PASS |
| 520 | ch:exploratory:L520:5.9\times10^{23} | derived | `5.9\times10^{23}` | numeric: status table: focus strength for a 0.01 g tide | PASS |
| 520 | ch:exploratory:L520:200 | derived | `200` | numeric: status table: focus distance for a 0.01 g tide, km | PASS |
| 520 |  | derived | `0.01` | not run: input: tide limit 0.01 g of the example | - |
| 520 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 521 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 522 | ch:exploratory:L522 | observed | `4\times10^{-12}` | numeric: status table: Al/Fe ratio (1986) | PASS |
| 522 | ch:exploratory:L522:3.9\times10^{-14} | observed | `3.9\times10^{-14}` | numeric: status table: Al/Fe ratio (2023) | PASS |
| 522 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 523 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 524 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 525 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 526 |  | derived | `0.2` | not run: table formatting: the 0.2 is the width of a \cmidrule[0.2pt] rule in the status table, not a book number | - |
| 527 | ch:exploratory:L527 | calc | `6.9\times10^{62}` | numeric: status table: negative energy, 100 m bubble at c | PASS |
| 527 | ch:exploratory:L527:1.6\times10^{67} | calc | `1.6\times10^{67}` | numeric: status table: negative energy at 2.3e4 c | PASS |
| 527 | ch:exploratory:L527:2.3\times10^4 | calc | `2.3\times10^4` | numeric: status table: Pleiades rate xi | PASS |
| 527 | ch:exploratory:L527:0.56 | calc | `0.56` | numeric: status table: 1 m wall in solar masses | PASS |

## Part 7 - ch:statusall - `docs/book/part7/p7_11_status_all.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 12 | ch:statusall:L12 | calc | `-0.28` | numeric: photon-sector H0 vs Planck 67.36 +- 0.54, errors in quadrature | PASS |
| 12 | ch:statusall:L12:-0.68 | calc | `-0.68` | numeric: matter-sector H0 vs SH0ES 73.04 +- 1.04 | PASS |
| 12 | ch:statusall:L12:8.6 | calc | `8.6` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2b H0 vs Planck | PASS |
| 12 |  | calc | `0.54` | not run: input: Planck 2018 H0 error 0.54 km/s/Mpc (Aghanim et al. 2020, doi:10.1051/0004-6361/201833910), the reference error a distance in sigma divides by; used in ch:statusall:L45:-0.37 | - |
| 12 |  | calc | `1.04` | not run: input: SH0ES H0 error 1.04 km/s/Mpc (Riess et al. 2022, doi:10.3847/2041-8213/ac5c5b), the reference error a distance in sigma divides by; used in ch:statusall:L46:-0.75 | - |
| 21 | ch:statusall:L21 | calc | `1.2\times10^{32}` | numeric: cost per bit cell / cosmic horizon | PASS |
| 23 | ch:statusall:L23 | calc | `2.82\times10^{7}` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpG sites of the hg19 index | PASS |
| 23 | ch:statusall:L23:1.6\times10^{59} | calc | `1.6\times10^{59}` | numeric: holographic capacity of a nucleus-sized area, bits | PASS |
| 23 | ch:statusall:L23:52 | calc | `52` | numeric: orders of magnitude between CpG sites and the capacity | PASS |
| 24 | ch:statusall:L24 | derived | `-1.33` | numeric: w_info today | PASS |
| 25 | ch:statusall:L25 | calc | `2.1\times10^{77}` | numeric: Mc^2/k_B T_BH, 1 M_sun | PASS |
| 26 | ch:statusall:L26 | derived | `13.606` | numeric: hydrogen kinetic energy (Rydberg), eV | PASS |
| 27 | ch:statusall:L27 | calc | `0.5000000000` | numeric: Smarr share at 6.5e9 M_sun | PASS |
| 27 |  | calc | `6.5\times10^9` | not run: input: upper end of the black-hole mass range of the row (6.5e9 solar masses); the Smarr share at that mass is checked in ch:statusall:L27 | - |
| 28 |  | observed | `1.1` | not run: observed: published approximate range 2T/|U| = 1.1-1.3 of simulated halos, cited in ch:virial_law (Bett2007, Neto2007, Power2012); no committed file holds it, nothing to recompute | - |
| 28 |  | observed | `1.3` | not run: observed: published approximate range 2T/|U| = 1.1-1.3 of simulated halos, cited in ch:virial_law (Bett2007, Neto2007, Power2012); no committed file holds it, nothing to recompute | - |
| 28 |  | calc | `-1.17` | not run: observed: published approximate range 1.02-1.17 with the surface-pressure term, cited in ch:virial_law (Klypin2016); no committed file holds it, nothing to recompute | - |
| 32 | ch:statusall:L32 | prediction | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 33 | ch:statusall:L33 | calc | `0.155` | numeric: free coupling = local rate restated | PASS |
| 33 | ch:statusall:L33:-0.28 | calc | `-0.28` | numeric: f sigma8 alone: lower 68 % end of beta | PASS |
| 33 | ch:statusall:L33:0.47 | calc | `0.47` | numeric: f sigma8 alone: upper 68 % end of beta | PASS |
| 35 | ch:statusall:L35 | derived | `-1.062` | numeric: same value as p2_03_theory:514 (tangent w0 value) | PASS |
| 35 |  | derived | `-0.012` | not run: restates ch:theory:L515:-0.012 (w_a = -Omega_m^2/3(2-Omega_m)^2 = -0.0116; a fresh check cannot carry a working negative control at two printed digits, the value sits 0.00025 from the 5 % shifted number) | - |
| 37 | ch:statusall:L37 | calc | `-0.136` | numeric: mu0 | PASS |
| 37 | ch:statusall:L37:13.62 | calc | `13.62` | numeric: same value as p1_02_iams_law:466 (percent change of coupling today) | PASS |
| 38 | ch:statusall:L38 | measured | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 Delta chi2, IAM minus LambdaCDM | PASS |
| 39 | ch:statusall:L39 | measured | `+0.96` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level 1 Delta chi2, Planck | PASS |
| 39 | ch:statusall:L39:+0.56 | measured | `+0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level 1 Delta chi2, Planck + RSD | PASS |
| 39 | ch:statusall:L39:+1.73 | measured | `+1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level 1 Delta chi2, Planck + BAO | PASS |
| 39 | ch:statusall:L39:+1.58 | measured | `+1.58` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: Level 1 Delta chi2, Planck + Pantheon+ | PASS |
| 40 | ch:statusall:L40 | measured | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8, Level 2 LambdaCDM (run C) | PASS |
| 40 | ch:statusall:L40:-1.1 | measured | `-1.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 change run C to run A, per cent | PASS |
| 40 | ch:statusall:L40:-1.51 | measured | `-1.51` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 shift in run C sigma | PASS |
| 41 | ch:statusall:L41 | measured | `0.830` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8, Level 2 LambdaCDM (run C) | PASS |
| 41 | ch:statusall:L41:-0.78 | measured | `-0.78` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 shift in run C sigma | PASS |
| 42 | ch:statusall:L42 | measured | `0.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: largest shift of the other Level 2 parameters, sigma | PASS |
| 43 | ch:statusall:L43 | measured | `-1.6` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: sigma8 change, Level 1, each data combination, per cent | PASS |
| 44 | ch:statusall:L44 | calc | `+0.030` | heavy file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: free mu0: smallest posterior median | PASS |
| 44 | ch:statusall:L44:+0.064 | calc | `+0.064` | heavy file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: free mu0: largest posterior median | PASS |
| 44 | ch:statusall:L44:+0.2 | calc | `+0.2` | file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_mu0_float.input.yaml`: free mu0: upper prior edge | PASS |
| 44 |  | calc | `90` | not run: definition: the central 90 % credible interval used to report the free-mu0 posteriors | - |
| 45 | ch:statusall:L45 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0, Level 2 chain | PASS |
| 45 | ch:statusall:L45:-0.37 | measured | `-0.37` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 from Planck, Planck error alone | PASS |
| 46 | ch:statusall:L46 | derived | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 46 | ch:statusall:L46:-0.75 | derived | `-0.75` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 from SH0ES, SH0ES error alone | PASS |
| 47 | ch:statusall:L47 | measured | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 with the term in the background (Level 2b) | PASS |
| 47 | ch:statusall:L47:10.9 | measured | `10.9` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2b H0 below Planck, Planck error alone | PASS |
| 48 | ch:statusall:L48 | measured | `-0.035` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: best beta on supernova distances | PASS |
| 48 | ch:statusall:L48:-0.068 | measured | `-0.068` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: lower 68 % end of beta on supernova distances | PASS |
| 48 | ch:statusall:L48:0.000 | measured | `0.000` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: upper 68 % end of beta on supernova distances | PASS |
| 48 | ch:statusall:L48:1590 | calc | `1590` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: supernovae in the Hubble-flow sample | PASS |
| 49 | ch:statusall:L49 | calc | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Delta chi2 of beta_m on supernova distances | PASS |
| 50 | ch:statusall:L50 | calc | `0.585` | numeric: effective growth index today, IAM | PASS |
| 50 | ch:statusall:L50:0.554 | calc | `0.554` | numeric: effective growth index today, LambdaCDM | PASS |
| 50 | ch:statusall:L50:0.633 | calc | `0.633` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: measured growth index (Nguyen, Huterer, Wen 2023) | PASS |
| 50 | ch:statusall:L50:0.024 | calc | `0.024` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: lower error of the measured growth index | PASS |
| 51 | ch:statusall:L51 | calc | `4.25` | numeric: f sigma8 deficit z=0 | PASS |
| 51 | ch:statusall:L51:2.17 | calc | `2.17` | numeric: z=0.3 | PASS |
| 51 | ch:statusall:L51:1.35 | calc | `1.35` | numeric: z=0.5 | PASS |
| 51 | ch:statusall:L51:0.41 | calc | `0.41` | numeric: z=1 | PASS |
| 52 | ch:statusall:L52 | calc | `+1.8` | numeric: E_G change at z = 0.3, per cent | PASS |
| 52 | ch:statusall:L52:+3.6 | calc | `+3.6` | numeric: E_G change today, per cent | PASS |
| 52 |  | calc | `0.3` | not run: input: the redshift z = 0.3 at which the E_G change is quoted | - |
| 53 | ch:statusall:L53 | calc | `0.08` | numeric: CMB lensing power lower, per cent | PASS |
| 55 |  | prediction | `0.3` | not run: prediction, nothing to recompute: the redshift range 0.3 < z < 1 of the growth-ramp turn-on that the survey test targets | - |
| 57 | ch:statusall:L57 | observed | `0.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 sigma8 against the 2025 joint lensing value, sigma | PASS |
| 57 | ch:statusall:L57:0.815 | observed | `0.815` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: KiDS-Legacy S8 | PASS |
| 57 | ch:statusall:L57:0.3 | observed | `0.3` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 S8 against KiDS-Legacy, sigma | PASS |
| 57 | ch:statusall:L57:0.776 | calc | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DES Y3 3x2pt S8 | PASS |
| 57 | ch:statusall:L57:2.3 | observed | `2.3` | numeric: Level 2 S8 against DES Y3 3x2pt, sigma | PASS |
| 58 | ch:statusall:L58 | calc | `7.6` | numeric: w0 distance of the CPL image from DESI, Union3 | PASS |
| 58 | ch:statusall:L58:10.2 | calc | `10.2` | numeric: w0 distance of the CPL image from DESI, DES Y5 | PASS |
| 59 | ch:statusall:L59 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 59 | ch:statusall:L59:55.57 | calc | `55.57` | numeric: same value as p2_11_dark_energy:146 (precise H_infinity, Level 2 chains) | PASS |
| 59 | ch:statusall:L59:70.86 | calc | `70.86` | numeric: same value as p2_11_dark_energy:173 (H_m asymptote, matter-sector) | PASS |
| 60 | ch:statusall:L60 | calc | `1.076` | numeric: same value as p1_02_iams_law:703 (Hubble sector ratio sqrt(1+beta_m)) | PASS |
| 60 | ch:statusall:L60:1.275 | calc | `1.275` | numeric: same value as p2_11_dark_energy:174 (H_m/H ratio limit a->infinity) | PASS |
| 61 | ch:statusall:L61 | calc | `1.26` | numeric: z of the peak | PASS |
| 61 | ch:statusall:L61:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 61 | ch:statusall:L61:3.37 | calc | `3.37` | numeric: peak writing rate, per cent per Gyr, H0 67.16 | PASS |
| 61 | ch:statusall:L61:2.53 | calc | `2.53` | numeric: writing rate now, per cent per Gyr | PASS |
| 62 | ch:statusall:L62 | observed | `0.79` | numeric: baryon relation (3/16) sqrt(Omega_L), per cent off | PASS |
| 68 | ch:statusall:L68 | calc | `2.1\times10^{67}` | numeric: evaporation time 1 M_sun | PASS |
| 71 | ch:statusall:L71 | observed | `0.66666446` | numeric: same value as p4_15a_lepton_koide:33 (Koide Q, PDG 2024 masses) | PASS |
| 71 | ch:statusall:L71:0.43 | observed | `0.43` | numeric: Koide Q from 2/3, sigma | PASS |
| 72 | ch:statusall:L72 | derived | `0.2222` | numeric: same value as p4_15a_lepton_koide:206 (measured offset delta) | PASS |
| 73 | ch:statusall:L73 | calc | `-0.02` | numeric: electron fixed point at H0 = 67.36, per cent | PASS |
| 73 |  | calc | `67.36` | not run: input: Planck 2018 H0 = 67.36 (Aghanim et al. 2020), the rate at which the electron-mass relation is evaluated; the -0.02 % is checked in ch:statusall:L73 | - |
| 76 | ch:statusall:L76:60 | calc | `60` | numeric: active fluctuators for x_qp = 1e-7, 1e3 um^3 island | PASS |
| 76 | ch:statusall:L76:600 | calc | `600` | numeric: active fluctuators, 1e4 um^3 island | PASS |
| 76 | ch:statusall:L76:6000 | calc | `6000` | numeric: active fluctuators, 1e5 um^3 island | PASS |
| 76 |  | derived | `10` | not run: input: base of the island volumes 10^3-10^5 cubic micrometres; the fluctuator counts are checked in ch:statusall:L76:60/600/6000 | - |
| 77 |  | observed | `10` | not run: observed: measured quasiparticle background band 10^-8 to 10^-6 cited in ch:xqp; the inventoried 10 is the base of the power of ten, no committed file holds the band | - |
| 78 |  | calc | `10` | not run: input: x_qp = 10^-7, the quasiparticle fraction at which T_1 is evaluated (the inventoried 10 is the base of the power of ten) | - |
| 79 | ch:statusall:L79 | calc | `6.2\times10^{-4}` | numeric: transmon floor on its gauge | PASS |
| 79 | ch:statusall:L79:10 | calc | `10` | numeric: surface-code threshold on the qubit gauge | PASS |
| 81 | ch:statusall:L81 | calc | `6.2\times10^{-7}` | numeric: per-gate thermal floor | PASS |
| 81 |  | calc | `68` | not run: input: T_1 = 68 microseconds (GoogleWillow2025, as cited in ch:ascoreqc); the floor is checked in ch:statusall:L81 | - |
| 81 |  | calc | `10` | not run: input: illustrative two-qubit gate error 10^-3 (the inventoried 10 is the base of the power of ten) | - |
| 81 |  | calc | `35` | not run: input: device temperature 35 mK (Jin2015, as cited in ch:ascoreqc) | - |
| 81 |  | calc | `40` | not run: input: illustrative 40 ns gate time | - |
| 81 |  | calc | `6\times10^{-4}` | not run: printed value 6x10^-4 has one significant digit: a 5 % change cannot be told from rounding (recomputed 6.19e-4); the precise per-gate floor is checked at ch:statusall:L81 (6.2x10^-7) | - |
| 82 | ch:statusall:L82 | calc | `6.85` | numeric: slope at 35 mK | PASS |
| 82 | ch:statusall:L82:16.0 | calc | `16.0` | numeric: slope at 15 mK | PASS |
| 82 |  | calc | `35` | not run: input: temperature 35 mK at which the slope is evaluated; the slope is checked in ch:statusall:L82 | - |
| 82 |  | calc | `15` | not run: input: temperature 15 mK at which the slope is evaluated; the slope is checked in ch:statusall:L82:16.0 | - |
| 85 | ch:statusall:L85 | calc | `576` | numeric: E_sw/(k_B T_j ln2) | PASS |
| 85 | ch:statusall:L85:593 | calc | `593` | numeric: E_sw/(k_B T_j ln2) | PASS |
| 85 |  | calc | `9950` | not run: input: processor model designation (a name, not a quantity); the switching energy is checked in ch:statusall:L85 and L85:593 | - |
| 86 | ch:statusall:L86 | calc | `3.33\times10^{-21}` | numeric: Landauer floor at 75 C | PASS |
| 86 | ch:statusall:L86:8.6 | calc | `8.6` | numeric: 105 C vs 75 C | PASS |
| 86 |  | calc | `75` | not run: input: junction temperature 75 C; the Landauer floor is checked in ch:statusall:L86 | - |
| 86 |  | calc | `105` | not run: input: junction temperature 105 C; the 8.6 % is checked in ch:statusall:L86:8.6 | - |
| 88 | ch:statusall:L88:1.9 | calc | `1.9` | numeric: Hopfield gap, lowest published DNMT1 preference | PASS |
| 88 | ch:statusall:L88:4.4 | calc | `4.4` | numeric: Hopfield gap, highest published DNMT1 preference | PASS |
| 88 | ch:statusall:L88:3.41 | calc | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: measured holding energy, k_B T | PASS |
| 90 | ch:statusall:L90 | calc | `2.968\times10^{-21}` | numeric: Landauer bit-cost energy at body temperature | PASS |
| 90 |  | calc | `37` | not run: input: body temperature 37 C (T_cell = 310.15 K, canon); the Landauer cost is checked in ch:statusall:L90 | - |
| 91 | ch:statusall:L91 | calc | `20.94` | numeric: same value as p0_how_to_read:50 (Mahaffey number M_cell for one ATP at 37C) | PASS |
| 91 | ch:statusall:L91:30.2 | calc | `30.2` | numeric: M/ln2 | PASS |
| 92 | ch:statusall:L92 | measured | `3.41` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: holding energy of a methylated site | PASS |
| 92 | ch:statusall:L92:0.032 | measured | `0.032` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy-error floor eps0 | PASS |
| 92 | ch:statusall:L92:0.163 | measured | `0.163` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: phi = E_hold / M | PASS |
| 93 | ch:statusall:L93 | measured | `1.099` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: neutrophil position P on IAM-A | PASS |
| 93 | ch:statusall:L93:1.084 | calc | `1.084` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: lowest donor P | PASS |
| 94 | ch:statusall:L94 | calibrated | `0.330263` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: Met-A neutrophil reference floor, bits | PASS |
| 94 | ch:statusall:L94:6000 | calibrated | `6000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: identity sites of the neutrophil reference | PASS |
| 95 | ch:statusall:L95 | measured | `0.020` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out SD of the six reference arrays | PASS |
| 96 | ch:statusall:L96 | calc | `0.910` | file `CANON/iam_canon.json`: 1/P | PASS |
| 96 | ch:statusall:L96:3.03 | calc | `3.03` | file `CANON/iam_canon.json`: Met-A full | PASS |
| 96 | ch:statusall:L96:4.45 | calc | `4.45` | file `CANON/iam_canon.json`: IAM-A full | PASS |
| 98 | ch:statusall:L98 | calc | `0.78` | numeric: floor at 10 C | PASS |
| 98 | ch:statusall:L98:1.012 | calc | `1.012` | numeric: same value as p6_10_temperature:17 (floor at 38.5 C) | PASS |
| 98 | ch:statusall:L98:1.012' | calc | `1.012` | numeric: floor at 38.5 C | PASS |
| 98 |  | calc | `10` | not run: input: temperature 10 C at which the floor is evaluated; checked in ch:statusall:L98 | - |
| 98 |  | calc | `38.5` | not run: input: temperature 38.5 C at which the floor is evaluated; checked in ch:statusall:L98:1.012 | - |
| 99 | ch:statusall:L99 | measured | `92.7` | heavy file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: atlas v2 held-out interval coverage, per cent | PASS |
| 106 | ch:statusall:L106 | calc | `2.3\times10^{22}` | numeric: M_eq today | PASS |
| 108 | ch:statusall:L108 | conjecture | `0.0179` | numeric: Q_L = k_B T ln 2 at room temperature, eV | PASS |
| 110 | ch:statusall:L110:509 | prediction | `509` | numeric: tau_IAM of a 1e-12 kg silica sphere at 10 mK, s | PASS |
| 110 | ch:statusall:L110:7.5 | prediction | `7.5` | numeric: tau_DP of the same sphere, microseconds | PASS |
| 110 |  | prediction | `10` | not run: input: temperature 10 mK of the gravitational-decoherence prediction; tau values checked in ch:statusall:L110:509 and L110:7.5 | - |

## Part 7 - ch:conclusion - `docs/book/part7/p7_10_conclusion.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 16 | ch:conclusion:L16 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 17 | ch:conclusion:L17 | derived | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 17 | ch:conclusion:L17:4.25 | derived | `4.25` | numeric: f sigma8 deficit today, per cent | PASS |
| 18 | ch:conclusion:L18 | derived | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 23 | ch:conclusion:L23 | observed | `0.66666446` | numeric: same value as p4_15a_lepton_koide:33 (Koide Q, PDG 2024 masses) | PASS |
| 23 | ch:conclusion:L23:0.43 | observed | `0.43` | numeric: Koide Q from 2/3, sigma | PASS |
| 25 | ch:conclusion:L25 | calc | `576` | numeric: switching energy over k_B T_j ln 2, 20.6e9 transistors | PASS |
| 25 | ch:conclusion:L25:593 | calc | `593` | numeric: switching energy over k_B T_j ln 2, 20.0e9 transistors | PASS |
| 33 | ch:conclusion:L33 | prediction | `-0.136` | numeric: mu0 = mu(a=1) - 1 | PASS |
| 34 | ch:conclusion:L34 | prediction | `4.25` | numeric: growth deficit today, per cent | PASS |
| 36 | ch:conclusion:L36 | prediction | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 from the Level 2 chain | PASS |

## Part 8 - app:constants - `docs/book/appendices/app_A2_frozen_values.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 6 | app:constants:L6 | observed | `1.380649\times10^{-23}` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 7 | app:constants:L7 | observed | `6.02214076\times10^{23}` | file `CANON/iam_canon.json`: Avogadro constant as R / k_B | PASS |
| 8 | app:constants:L8 | observed | `8.314462618` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 10 | app:constants:L10 | observed | `310.15` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 10 | app:constants:L10:37 | observed | `37` | file `CANON/iam_canon.json`: body temperature in Celsius | PASS |
| 11 | app:constants:L11 | observed | `54` | file `CANON/iam_canon.json`: free energy of ATP hydrolysis, kJ/mol | PASS |
| 11 |  | observed | `50` | not run: observed: textbook range 50-65 kJ/mol of the free energy of ATP hydrolysis in cells, cited (Nelson2017); no committed file holds it, the canon value 54 is checked in app:constants:L11 | - |
| 11 |  | observed | `-65` | not run: observed: textbook range 50-65 kJ/mol of the free energy of ATP hydrolysis in cells, cited (Nelson2017); no committed file holds it, the canon value 54 is checked in app:constants:L11 | - |
| 12 | app:constants:L12 | calc | `2.968\times10^{-21}` | numeric: Landauer bit-cost energy at body temperature | PASS |
| 13 | app:constants:L13 | calc | `20.94` | numeric: same value as p0_how_to_read:50 (Mahaffey number M_cell for one ATP at 37C) | PASS |
| 14 | app:constants:L14 | observed | `67.36` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Planck 2018 H0, standard LambdaCDM fit | PASS |
| 15 | app:constants:L15 | observed | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0, Level 2 chain | PASS |
| 15 | app:constants:L15:72.26 | observed | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 from the Level 2 chain | PASS |
| 16 | app:constants:L16 | observed | `3.41` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 16 | app:constants:L16:0.032 | observed | `0.032` | file `CANON/iam_canon.json`: copy-error floor eps0 from E_hold | PASS |
| 17 | app:constants:L17 | calibrated | `0.330263` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 17 | app:constants:L17:6000 | calibrated | `6000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: identity sites of the Met-A neutrophil reference | PASS |
| 18 | app:constants:L18 | measured | `1.099` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 18 | app:constants:L18:1.084 | measured | `1.084` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 19 | app:constants:L19 | calibrated | `1.1104` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy clustering baseline | PASS |
| 20 | app:constants:L20 | calibrated | `0.93` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Intake/intake_thresholds_v1.json`: intake quarantine call-rate line | PASS |

## Part 8 - app:notation - `docs/book/appendices/app_N_notation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 103 | app:notation:L103 | observed | `310.15` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 103 | app:notation:L103:37 | observed | `37` | file `CANON/iam_canon.json`: body temperature in Celsius | PASS |
| 104 | app:notation:L104 | observed | `54` | file `CANON/iam_canon.json`: free energy of ATP hydrolysis, kJ/mol | PASS |
| 104 |  | observed | `50` | not run: observed: textbook range 50-65 kJ/mol of the free energy of ATP hydrolysis in cells, cited (Nelson2017); no committed file holds it, the canon value 54 is checked in app:notation:L104 | - |
| 104 |  | observed | `-65` | not run: observed: textbook range 50-65 kJ/mol of the free energy of ATP hydrolysis in cells, cited (Nelson2017); no committed file holds it, the canon value 54 is checked in app:notation:L104 | - |
| 105 | app:notation:L105 | calc | `2.968\times10^{-21}` | numeric: Landauer bit-cost energy at body temperature | PASS |
| 105 | app:notation:L105:1.787 | calc | `1.787` | numeric: same value as p6_02_landauer:17 (per mole of bits, kJ) | PASS |
| 106 | app:notation:L106 | calc | `20.94` | numeric: same value as p0_how_to_read:50 (Mahaffey number M_cell for one ATP at 37C) | PASS |
| 107 | app:notation:L107 | observed | `30.2` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 108 | app:notation:L108 | measured | `3.41` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 109 | app:notation:L109 | observed | `4.9` | file `CANON/iam_canon.json`: E_hold in Landauer units | PASS |
| 110 | app:notation:L110 | measured | `0.1628` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 111 | app:notation:L111 | derived | `0.032` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 112 | app:notation:L112 | calc | `0.2043` | numeric: same value as p6_00b_astrogenetics:63 (H(eps0), bits) | PASS |
| 113 | app:notation:L113 | measured | `1.099` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 113 | app:notation:L113:1.084 | measured | `1.084` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 113 | app:notation:L113:-1.108 | measured | `-1.108` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 114 | app:notation:L114 | calc | `0.2246` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 115 | app:notation:L115 | calc | `0.910` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: 1/P, position of H_min on the IAM-A gauge | PASS |
| 116 | app:notation:L116 | calibrated | `0.330263` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 116 | app:notation:L116:6000 | calibrated | `6000` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: identity sites of the Met-A neutrophil reference | PASS |
| 117 | app:notation:L117 | observed | `1.05` | file `CANON/iam_canon.json`: Normal band, upper edge (canon Normal_band) | PASS |
| 117 | app:notation:L117:0.95 | observed | `0.95` | file `CANON/iam_canon.json`: lower edge of the Normal band | PASS |
| 118 | app:notation:L118 | calc | `3.03` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: full surface on Met-A | PASS |
| 118 | app:notation:L118:4.45 | calc | `4.45` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: full surface on IAM-A | PASS |
| 119 | app:notation:L119 | calibrated | `1.1104` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy clustering baseline | PASS |
| 120 | app:notation:L120 | observed | `528` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 120 | app:notation:L120:48528 | observed | `48528` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json`: fixed noise sites of the noise index | PASS |

## Part 8 - app:formulas - `docs/book/appendices/app_E_formulas.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 6 |  | none |  | not run: definition: line 6 is the body of the \iamfsentry macro (the equation* wrapper of every formula-sheet entry), not an equation of the book | - |
| 47 | app:formulas:L56:-1.67 | derived | `-1.67` | numeric: w_info at a = 0.5 | PASS |
| 47 | app:formulas:L56:-1.33 | derived | `-1.33` | numeric: w_info at a = 1 | PASS |
| 47 | app:formulas:L56:-1.17 | derived | `-1.17` | numeric: w_info at a = 2 | PASS |
| 47 |  | derived | `0.5` | not run: input: the scale factor a = 0.5 at which w_info is evaluated | - |
| 49 | app:formulas:L58 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 49 | app:formulas:L58:0.3153 | derived | `0.3153` | file `CANON/iam_canon.json`: Omega_m behind beta_m (canon) | PASS |
| 91 | app:formulas:L100 | derived | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 91 | app:formulas:L100:0.15765 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 99 | app:formulas:L108 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 99 | app:formulas:L108:72.26 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 117 | app:formulas:L126 | derived |  | sympy: area change of a local Rindler horizon from Raychaudhuri | PASS |
| 130 | app:formulas:L139 | derived |  | sympy: energy flux across the apparent horizon | PASS |
| 168 | app:formulas:L177 | derived | `0.136` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 180 | app:formulas:L189 | derived | `1.062` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 180 | app:formulas:L189:1.062 | calc | `1.062` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 181 | app:formulas:L190 | calc | `0.012` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 217 | app:formulas:L226 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 231 | app:formulas:L240 | calc | `-0.13495` | file `Cosmological_Physics/mgcamb_validation/chains/iam_fixed_mu0_r2.input.yaml`: MGCAMB amplitude mu0 set in the runs | PASS |
| 234 | app:formulas:L243 | derived | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 239 | app:formulas:L248 | derived | `0.864` | numeric: mu today | PASS |
| 241 | app:formulas:L250 | derived | `1.0759` | numeric: same value as p1_02_iams_law:516 (H_m/H at z=0) | PASS |
| 321 | app:formulas:L330 | calc | `1.133\times10^{-123}` | numeric: measured rho_Lambda / rho_vac | PASS |
| 326 | app:formulas:L335 | fitted | `1.218` | numeric: base expression over the measured ratio | PASS |
| 326 |  | openprob | `0.7` | not run: no 0.7 is printed at line 326 in the current text (text changed at HEAD; the entry is the base cosmological-constant expression, whose 1.218 is checked in app:formulas:L335); nothing to check | - |
| 333 | app:formulas:L342 | fitted | `1.380\times10^{-123}` | numeric: base expression | PASS |
| 339 | app:formulas:L348 | fitted | `0.79` | numeric: corrected expression above the measured value, per cent | PASS |
| 340 | app:formulas:L349 | fitted | `1.142\times10^{-123}` | numeric: corrected expression | PASS |
| 363 |  | measured | `0.02232` | not run: definition: line 363 is n_b = eta n_gamma (labelled a definition); it prints no 0.02232 (the scan line drifted; Omega_b h^2 = 0.02232 appears at lines 79 and 385, outside this batch) | - |
| 378 | app:formulas:L387 | measured | `1.0046` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: same value as p2_13b_baryon_chain:123 (ratio on the 18th chain, committed output) | PASS |
| 387 | app:formulas:L396 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 387 | app:formulas:L396:67.16 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 438 | app:formulas:L447 | derived | `0.433` | numeric: Kerr T S / M c^2 at chi = 0.5 | PASS |
| 438 | app:formulas:L447:0.218 | derived | `0.218` | numeric: Kerr T S / M c^2 at chi = 0.9 | PASS |
| 438 | app:formulas:L447:0.032 | derived | `0.032` | numeric: Kerr T S / M c^2 at chi = 0.998 | PASS |
| 438 |  | derived | `0.5` | not run: input: spin chi = 0.5 at which the Kerr fraction is evaluated | - |
| 438 |  | derived | `0.9` | not run: input: spin chi = 0.9 | - |
| 438 |  | derived | `0.998` | not run: input: spin chi = 0.998 | - |
| 447 | app:formulas:L456 | derived | `2.32\times10^{22}` | numeric: M_eq = c^3/(4GH) at H0 67.4, solar masses | PASS |
| 447 |  | derived | `67.4` | not run: input: Planck 2018 H0 = 67.4 km/s/Mpc (rounded 67.36, Aghanim et al. 2020) at which M_eq is evaluated | - |
| 448 | app:formulas:L457 | derived | `4.5\times10^{22}` | numeric: black hole in balance with the CMB, kg | PASS |
| 531 | app:formulas:L540 | calc | `0.91` | numeric: string tension in GeV/fm | PASS |
| 534 | app:formulas:L543 | observed | `0.66666051` | numeric: same value as p4_15a_lepton_koide:35 (Koide Q with the 2022 m_tau) | PASS |
| 580 | app:formulas:L589 | derived | `16.0` | numeric: slope of the held-record floor at 15 mK, 5 GHz | PASS |
| 587 | app:formulas:L596 | derived | `6.2\times10^{-7}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 587 |  | derived | `68` | not run: input: T_1 = 68 microseconds of the worked transmon example (GoogleWillow2025, as cited in ch:qplatforms) | - |
| 587 |  | derived | `40` | not run: input: gate duration t = 40 ns of the worked example | - |
| 590 | app:formulas:L599 | calc | `0.021` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 590 | app:formulas:L599:3.33\times10^{-21} | calc | `3.33\times10^{-21}` | numeric: Landauer floor at the junction temperature | PASS |
| 590 |  | calc | `348` | not run: input: junction temperature T_j = 348 K (75 C) at which the floor is evaluated | - |
| 593 | app:formulas:L602 | derived | `6.2` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 593 | app:formulas:L602:18.5 | derived | `18.5` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 593 | app:formulas:L602:0.646 | derived | `0.646` | numeric: constant-field step per halving of area | PASS |
| 593 |  | derived | `600` | not run: input: chip reading R = 600 times its floor (round value of the 576-593 checked in ch:conclusion:L25 and ch:statusall:L85); n_floor at R = 600 is checked in app:formulas:L602 | - |
| 601 | app:formulas:L610 | calc | `310.15` | file `CANON/iam_canon.json`: body temperature, K (canon) | PASS |
| 601 | app:formulas:L610:2.968 | calc | `2.968\times10^{-21}` | numeric: Landauer cost of one bit at body temperature | PASS |
| 602 | app:formulas:L611 | calc | `20.94` | numeric: same value as p0_how_to_read:50 (Mahaffey number M_cell for one ATP at 37C) | PASS |
| 604 | app:formulas:L613 | calc | `30.21` | numeric: same value as p6_02_landauer:48 (M/ln2) | PASS |
| 607 | app:formulas:L616 | calc | `2.822\times10^{7}` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpG sites of the hg19 index | PASS |
| 608 | app:formulas:L617 | calc | `8.38\times10^{-14}` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: Landauer floor of one copy of the methylome | PASS |
| 609 | app:formulas:L618:2.3 | calc | `2.3` | numeric: discrimination energy at 10 % maintenance error | PASS |
| 609 | app:formulas:L618:3.9 | calc | `3.9` | numeric: discrimination energy at 2 % maintenance error | PASS |
| 609 |  | calc | `0.10` | not run: input: maintenance error rate 10 % (upper end of the 2-10 % of Genereux et al. 2005) | - |
| 610 | app:formulas:L619 | measured | `0.163` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: phi = E_hold / M | PASS |
| 622 | app:formulas:L631 | calc | `3.03` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: full surface on Met-A | PASS |
| 628 | app:formulas:L637 | calibrated | `0.330263` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`: frozen EPIC neutrophil reference, bits | PASS |
| 633 | app:formulas:L642 | measured | `0.032` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: copy-error floor eps0 | PASS |
| 633 | app:formulas:L642:0.2043 | measured | `0.2043` | file `CANON/iam_canon.json`: H_min = H(eps0), bits | PASS |
| 634 | app:formulas:L643 | measured | `1.099` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json`: neutrophil position P | PASS |
| 638 | app:formulas:L647 | calibrated | `1.1104` | heavy file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json`: C-score healthy baseline | PASS |
| 655 | app:formulas:L664 | calc | `2.3\times10^{22}` | numeric: M_eq at H0 = 67.16, solar masses | PASS |
| 673 | app:formulas:L682 | derived | `0.032` | file `CANON/iam_canon.json`: eps0 from phi and M | PASS |

## Part 8 - app:derivations - `docs/book/appendices/app_C3_derivations.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 33 | app:derivations:L33 | derived |  | sympy: dG/dt = sum p^2/m + sum F.r (one degree of freedom, F = m r'') | PASS |
| 49 | app:derivations:L49 | calc | `13.606` | numeric: same value as p1_03_virial_law:144 (hydrogen |E| = alpha^2 m_e c^2/2 (infinite-mass Rydberg)) | PASS |
| 49 | app:derivations:L49:-27.211 | calc | `-27.211` | numeric: <V> hydrogen, eV | PASS |
| 50 | app:derivations:L50 | calc | `23.6` | numeric: Kelvin-Helmholtz |U|/2L, n=3 polytrope, Myr | PASS |
| 51 | app:derivations:L51 | calc | `1.456` | numeric: same value as p1_02_iams_law:840 (Chandrasekhar mass, mu_e=2, m_u) | PASS |
| 51 | app:derivations:L51:2.01824 | calc | `2.01824` | numeric: omega_3^0 of the n = 3 Lane-Emden equation | PASS |
| 56 | app:derivations:L56 | derived |  | sympy: drafted check, screened (runs; negative control fails) | PASS |
| 65 | app:derivations:L65 | derived | `0.500` | numeric: Kerr T S/Mc^2 at chi=0.0 | PASS |
| 65 | app:derivations:L65:0.433 | derived | `0.433` | numeric: Kerr T S/Mc^2 at chi=0.5 | PASS |
| 65 | app:derivations:L65:0.218 | derived | `0.218` | numeric: Kerr T S/Mc^2 at chi=0.9 | PASS |
| 65 | app:derivations:L65:0.032 | derived | `0.032` | numeric: Kerr T S/Mc^2 at chi=0.998 | PASS |
| 65 |  | derived | `0.5` | not run: input: Kerr spin chi = 0.5 at which TS/M is evaluated (TS/M checked at app:derivations:L65:0.433) | - |
| 65 |  | derived | `0.9` | not run: input: Kerr spin chi = 0.9 at which TS/M is evaluated (checked at app:derivations:L65:0.218) | - |
| 65 |  | derived | `0.998` | not run: input: Kerr spin chi = 0.998 at which TS/M is evaluated (checked at app:derivations:L65:0.032) | - |
| 68 | app:derivations:L68 | derived |  | sympy: T_H S = c^5/(2GH) = M_H c^2 with M_H the critical-density mass of the Hubble sphere | PASS |
| 71 | app:derivations:L71 | calc | `2.654\times10^{-30}` | numeric: T_H at H0 = 67.36 | PASS |
| 71 | app:derivations:L71:3.272\times10^{122} | calc | `3.272\times10^{122}` | numeric: horizon bits at H0 = 67.36 | PASS |
| 71 | app:derivations:L71:2.655\times10^{-30} | calc | `2.655\times10^{-30}` | numeric: same value as p2_12_lambda:87 (Gibbons-Hawking horizon temperature) | PASS |
| 71 |  | calc | `67.36` | not run: input: H0 = 67.36 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 71 |  | calc | `67.4` | not run: input: H0 = 67.4 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 77 | app:derivations:L77 | derived | `0.646` | numeric: half the entropy gone at 0.646 tau | PASS |
| 77 | app:derivations:L77:5120 | derived | `5120` | numeric: evaporation time coefficient 5120 pi | PASS |
| 78 | app:derivations:L78 | derived | `6.170\times10^{-8}` | numeric: T_BH, 1 M_sun | PASS |
| 78 | app:derivations:L78:1.513\times10^{77} | derived | `1.513\times10^{77}` | numeric: bits, 1 M_sun | PASS |
| 79 | app:derivations:L79 | calc | `152.5` | numeric: same value as p1_02_iams_law:650 (Hawking info rate for 1 solar mass) | PASS |
| 79 | app:derivations:L79:2.10\times10^{67} | calc | `2.10\times10^{67}` | numeric: tau_evap, 1 M_sun | PASS |
| 79 | app:derivations:L79:1.43\times10^{-14} | calc | `1.43\times10^{-14}` | numeric: T_BH, 4.3e6 M_sun | PASS |
| 79 | app:derivations:L79:2.80\times10^{90} | calc | `2.80\times10^{90}` | numeric: bits, 4.3e6 M_sun | PASS |
| 79 |  | calc | `4.3\times10^6` | not run: input: black-hole mass 4.3e6 M_sun at which T and the bit count are evaluated (results checked at app:derivations:L79:1.43\times10^{-14} and L79:2.80\times10^{90}) | - |
| 81 | app:derivations:L81 | calc | `2.32\times10^{22}` | numeric: M_eq today (H0 67.36; tol = half a unit plus the rounding of the 4-figure H0) | PASS |
| 81 | app:derivations:L81:1.30\times10^{22} | calc | `1.30\times10^{22}` | numeric: M_eq at z=1 | PASS |
| 82 | app:derivations:L82 | calc | `2.4\times10^{12}` | numeric: M_eq at z=1e6 (Planck 2018, radiation 9.1e-5) | PASS |
| 82 | app:derivations:L82:4.50\times10^{22} | calc | `4.50\times10^{22}` | numeric: M_CMB, kg | PASS |
| 82 |  | calc | `10` | not run: input: the printed 10 is the base of the redshift z = 10^6 at which M_eq is evaluated (M_eq checked at app:derivations:L82) | - |
| 83 | app:derivations:L83 | calc | `4.42\times10^7` | numeric: T_CMB / T_BH(1 M_sun) | PASS |
| 83 | app:derivations:L83:17.6 | calc | `17.6` | numeric: ln of the ratio | PASS |
| 87 | app:derivations:L87 | derived | `2.77` | numeric: 4 ln2 | PASS |
| 102 | app:derivations:L102 | derived |  | sympy: 2 pi/(hbar eta) = 8 pi G with G = 1/(4 hbar eta) (c = 1) | PASS |
| 118 | der:F2 | derived |  | sympy: Hdot = -4 pi G (rho+P) from -dE = T_H dS_geo on the apparent horizon | PASS |
| 129 | der:F2info | derived |  | sympy: Hdot with the record term T_H Sdot_info added to the first law | PASS |
| 139 | der:Sneed | calc |  | sympy: Sdot_info needed for the record term, and dS_info/dln a = beta_m S_geo today | PASS |
| 142 | app:derivations:L142 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 142 | app:derivations:L142:5.2\times10^{121} | calc | `5.2\times10^{121}` | numeric: bits per e-fold today, beta_m S_geo/(k_B ln2) at H0 = 67.36 | PASS |
| 142 |  | calc | `67.36` | not run: input: H0 = 67.36 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 145 |  | openprob | `5\times10^{121}` | not run: restates app:derivations:L142:5.2\times10^{121} (5.16e121) rounded to one figure; a 5 % control cannot fail at one figure | - |
| 154 | app:derivations:L154 | calc |  | sympy: dS/dln a ~ a^(n-9/2), dS/da ~ a^(n-11/2), S ~ a^(n-9/2)/(n-9/2) in matter domination | PASS |
| 163 | app:derivations:L163 | calc | `-2.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope of dS/dln a, full LambdaCDM growth (committed output; same value as ch:quantumrecords:L183) | PASS |
| 163 | app:derivations:L163:-1.52 | calc | `-1.52` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope of dS/dln a, full LambdaCDM growth (committed output; same value as ch:quantumrecords:L183) | PASS |
| 163 | app:derivations:L163:-1.02 | calc | `-1.02` | heavy file `docs/verification/scripts/verify_theory_derivations_output.txt`: slope of dS/dln a, full LambdaCDM growth (committed output; same value as ch:quantumrecords:L183) | PASS |
| 163 |  | calc | `0.315` | not run: input: Omega_m = 0.315 used for the growth numerics (Planck 2018, rounded) | - |
| 163 |  | calc | `9.1\times10^{-5}` | not run: input: Omega_r = 9.1e-5 used for the growth numerics (Planck 2018 radiation density) | - |
| 163 |  | calc | `0.01` | not run: input: lower end a = 0.01 of the fitting interval | - |
| 164 | app:derivations:L164 | calc | `-0.53` | numeric: power of dS/dln a, n = 4, 0.01 <= a <= 0.1, full LambdaCDM | PASS |
| 164 | app:derivations:L164:-2.42 | calc | `-2.42` | numeric: power of dS/dln a, n = 2.5, 0.25 <= a <= 1 | PASS |
| 164 | app:derivations:L164:-1.99 | calc | `-1.99` | numeric: power of dS/dln a, n = 3, 0.25 <= a <= 1 | PASS |
| 164 | app:derivations:L164:-1.57 | calc | `-1.57` | numeric: power of dS/dln a, n = 3.5, 0.25 <= a <= 1 | PASS |
| 164 | app:derivations:L164:-1.14 | calc | `-1.14` | numeric: power of dS/dln a, n = 4, 0.25 <= a <= 1 | PASS |
| 164 |  | calc | `2.5` | not run: input: exponent n = 2.5 at which the slope is quoted (slope checked at app:derivations:L163 and L164:-2.42) | - |
| 164 |  | calc | `3.5` | not run: input: exponent n = 3.5 at which the slope is quoted (slope checked at app:derivations:L163:-1.02 and L164:-1.57) | - |
| 164 |  | calc | `0.25` | not run: input: lower end a = 0.25 of the late fitting interval | - |
| 165 | app:derivations:L165 | calc | `0.7` | numeric: n = 7/2 shape residual against k(1-1/a)+const, 0.01 <= a <= 0.1 (per cent of range) | PASS |
| 165 | app:derivations:L165:3.8 | calc | `3.8` | numeric: n = 7/2 shape residual, 0.15 <= a <= 1 (per cent of range) | PASS |
| 165 |  | calc | `0.01` | not run: input: lower end a = 0.01 of the shape-fit interval | - |
| 166 | app:derivations:L166 | calc | `6.3` | numeric: n = 7/2 shape residual, 0.15 <= a <= 2 (per cent of range) | PASS |
| 166 |  | calc | `0.15` | not run: input: lower end a = 0.15 of the shape-fit interval | - |
| 173 | app:derivations:L173 | openprob | `5.5` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up n_eff at z = 9, middle of the three mass functions | PASS |
| 181 | app:derivations:L181 | calc | `2.30` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 181 | app:derivations:L181:0.69 | calc | `0.69` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 181 | app:derivations:L181:0.11 | calc | `0.11` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 182 | app:derivations:L182 | calc | `4.5\times10^{-5}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 182 | app:derivations:L182:9.4\times10^{-14} | calc | `9.4\times10^{-14}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 185 | der:winfo | derived |  | sympy: w_info = -1 - 1/(3a) at a = 0.5, 1, 2 | PASS |
| 190 | app:derivations:L190 | derived |  | sympy: w_eff = -1 - (1/3) dln rho/dln a for rho = Omega_L + (Omega_m/2) E(a): w_eff(1) = 2(3-Omega_m)/(3(Omega_m-2)), w_a = -Omega_m^2/(3(2-Omega_m)^2) | PASS |
| 193 | app:derivations:L193 | derived | `-1.0588` | numeric: w_eff(1) at Omega_m=0.3 | PASS |
| 193 | app:derivations:L193:-1.0623 | derived | `-1.0623` | numeric: w_eff(1) at Omega_m=0.315 | PASS |
| 193 | app:derivations:L193:-1.0624 | derived | `-1.0624` | numeric: w_eff(1) at Omega_m=0.3153 | PASS |
| 193 | app:derivations:L193:-1.0635 | derived | `-1.0635` | numeric: w_eff(1) at Omega_m=0.32 | PASS |
| 193 | app:derivations:L193:-0.0104 | derived | `-0.0104` | numeric: w_a at Omega_m=0.3 | PASS |
| 193 | app:derivations:L193:-0.0116 | derived | `-0.0116` | numeric: w_a at Omega_m=0.315 | PASS |
| 193 | app:derivations:L193:-0.0117 | derived | `-0.0117` | numeric: w_a at Omega_m=0.3153 | PASS |
| 193 | app:derivations:L193:-0.0121 | derived | `-0.0121` | numeric: w_a at Omega_m=0.32 | PASS |
| 193 |  | derived | `0.300` | not run: input: Omega_m = 0.300 at which w0 and w_a are evaluated (checked at app:derivations:L193) | - |
| 193 |  | derived | `0.315` | not run: input: Omega_m = 0.315 at which w0 and w_a are evaluated (checked at app:derivations:L193:-1.0623) | - |
| 193 |  | derived | `0.3153` | not run: input: Omega_m = 0.3153 (Planck 2018) at which w0 and w_a are evaluated (checked at app:derivations:L193:-1.0624) | - |
| 193 |  | derived | `0.320` | not run: input: Omega_m = 0.320 at which w0 and w_a are evaluated (checked at app:derivations:L193:-1.0635) | - |
| 194 | app:derivations:L194 | calc | `-1.065` | numeric: same value as p2_03_theory:514 (least-squares CPL fit intercept w0) | PASS |
| 194 | app:derivations:L194:+0.017 | calc | `+0.017` | numeric: least-squares CPL slope w_a over 0.5 <= a <= 1 (Omega_m 0.315) | PASS |
| 194 |  | calc | `0.5` | not run: input: lower end a = 0.5 of the CPL fitting interval | - |
| 194 |  | calc | `0.315` | not run: input: Omega_m = 0.315 for the CPL fit (fit checked at app:derivations:L194 and L194:+0.017) | - |
| 199 | app:derivations:L199 | derived |  | sympy: minisuperspace Lagrangian: the lapse drops out of the constraint and delta N gives the Friedmann equation | PASS |
| 243 | app:derivations:L243 | interp |  | sympy: mu(a) = H^2/(H^2 + beta_m E H0^2), mu(1) = 1/(1+beta_m), mu0 = -beta_m/(1+beta_m) | PASS |
| 246 | app:derivations:L246 | interp | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 246 | app:derivations:L246:0.8638 | interp | `0.8638` | numeric: mu(1) = 1/(1+beta_m) | PASS |
| 246 | app:derivations:L246:13.62 | interp | `13.62` | numeric: 1 - mu(1) in per cent | PASS |
| 247 | app:derivations:L247 | calc | `-0.13618` | numeric: same value as p1_02_iams_law:465 (mu0 at beta_m=0.15765, precise) | PASS |
| 247 | app:derivations:L247:-0.13607 | calc | `-0.13607` | numeric: mu0 for beta_m = 0.1575 | PASS |
| 247 | app:derivations:L247:0.905 | calc | `0.905` | numeric: mu at z=0.2 | PASS |
| 247 | app:derivations:L247:0.922 | calc | `0.922` | numeric: mu at z=0.3 | PASS |
| 247 | app:derivations:L247:0.948 | calc | `0.948` | numeric: mu at z=0.5 | PASS |
| 247 | app:derivations:L247:0.966 | calc | `0.966` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 247 | app:derivations:L247:0.982 | calc | `0.982` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 247 |  | calc | `0.1575` | not run: input: beta_m = 0.1575 (= 0.315/2) at which mu0 is evaluated (mu0 checked at app:derivations:L247:-0.13607) | - |
| 247 |  | calc | `0.2` | not run: input: redshift z = 0.2 at which mu and H_m/H are evaluated (checked at app:derivations:L247:0.905) | - |
| 247 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which mu and H_m/H are evaluated (checked at app:derivations:L247:0.922) | - |
| 247 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which mu and H_m/H are evaluated (checked at app:derivations:L247:0.948) | - |
| 247 |  | calc | `0.7` | not run: input: redshift z = 0.7 at which mu and H_m/H are evaluated (checked at app:derivations:L247:0.966) | - |
| 248 | app:derivations:L248 | calc | `1.000` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 248 | app:derivations:L248:1.051 | calc | `1.051` | numeric: same value as p2_06_dual_sector_perturbation:119 (Hm/H at z=0.2) | PASS |
| 248 | app:derivations:L248:1.042 | calc | `1.042` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 248 | app:derivations:L248:1.027 | calc | `1.027` | numeric: same value as p2_06_dual_sector_perturbation:120 (Hm/H at z=0.5) | PASS |
| 248 | app:derivations:L248:1.017 | calc | `1.017` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 248 | app:derivations:L248:1.009 | calc | `1.009` | numeric: same value as p2_06_dual_sector_perturbation:121 (Hm/H at z=1) | PASS |
| 248 | app:derivations:L248:1.001 | calc | `1.001` | numeric: same value as p2_06_dual_sector_perturbation:122 (Hm/H at z=2) | PASS |
| 248 | app:derivations:L248:0.998 | calc | `0.998` | numeric: mu at z = 2 | PASS |
| 252 | app:derivations:L252 | calc |  | sympy: growth forms (i)-(iii) in ln a from the time-domain equations | PASS |
| 258 | app:derivations:L258 | calc | `-0.78` | numeric: Delta D/D today, form (i) G_eff = mu G | PASS |
| 258 | app:derivations:L258:-0.67 | calc | `-0.67` | numeric: Delta D/D today, form (ii) friction 2 H_m | PASS |
| 258 | app:derivations:L258:-1.87 | calc | `-1.87` | numeric: Delta D/D today, form (iii) whole equation on H_m | PASS |
| 261 | app:derivations:L261 | calc | `4.25` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 261 | app:derivations:L261:2.17 | calc | `2.17` | numeric: f sigma8 deficit at z = 0.3, form (i) | PASS |
| 261 | app:derivations:L261:1.35 | calc | `1.35` | numeric: f sigma8 deficit at z = 0.5, form (i) | PASS |
| 262 | app:derivations:L262 | calc | `+3.63` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 262 | app:derivations:L262:0.41 | calc | `0.41` | numeric: f sigma8 deficit at z = 1, form (i) | PASS |
| 262 | app:derivations:L262:+1.86 | calc | `+1.86` | numeric: E_G change f_LCDM/f_IAM - 1 at z = 0.295 | PASS |
| 262 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the f sigma8 deficit is quoted (checked at app:derivations:L261:2.17) | - |
| 262 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which the f sigma8 deficit is quoted (checked at app:derivations:L261:1.35) | - |
| 263 | app:derivations:L263:+1.84 | calc | `+1.84` | numeric: E_G change at z = 0.3 | PASS |
| 263 | app:derivations:L263:+1.14 | calc | `+1.14` | numeric: E_G change at z = 0.5 | PASS |
| 263 | app:derivations:L263:0.299 | calc | `0.299` | numeric: Omega_m mu(z) inferred by a growth-only fit at z = 0.5 | PASS |
| 263 |  | calc | `0.295` | not run: input: redshift z = 0.295 at which the E_G change is quoted (the LambdaCDM matter-Lambda equality, checked at app:derivations:L284) | - |
| 263 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the E_G change is quoted (checked at app:derivations:L263:+1.84) | - |
| 263 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which the E_G change and Omega_m mu are quoted (checked at app:derivations:L263:+1.14, L263:0.299) | - |
| 266 | app:derivations:L266 | calc | `0.8638` | numeric: same value as p1_02_iams_law:465 (mu at a=1 from beta_m) | PASS |
| 266 | app:derivations:L266:0.8650 | calc | `0.8650` | numeric: MGCAMB-form mu today with the runs mu0 | PASS |
| 266 | app:derivations:L266:-2.76 | calc | `-2.76` | numeric: largest gap MGCAMB form vs exact mu, 0 <= z <= 3 | PASS |
| 266 | app:derivations:L266:0.65 | calc | `0.65` | numeric: redshift of the largest MGCAMB gap | PASS |
| 266 | app:derivations:L266:-2.48 | calc | `-2.48` | numeric: MGCAMB gap at z = 1 | PASS |
| 266 |  | calc | `-0.13495` | not run: input: mu0 = -0.13495, the MGCAMB amplitude of the Level 1 runs (MU0_MGCAMB in verify_book, ch:dual) | - |
| 267 | app:derivations:L267 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 267 | app:derivations:L267:-0.13618 | calc | `-0.13618` | numeric: same value as p1_02_iams_law:465 (mu0 at beta_m=0.15765, precise) | PASS |
| 267 | app:derivations:L267:0.1560 | calc | `0.1560` | numeric: beta with -beta/(1+beta) = mu0 of the runs | PASS |
| 267 |  | calc | `-0.13495` | not run: input: mu0 = -0.13495 of the Level 1 runs restated (MU0_MGCAMB); beta from it checked at app:derivations:L267:0.1560 | - |
| 267 |  | calc | `0.0012` | not run: difference of two checked values (MU0_MGCAMB and app:derivations:L267:-0.13618), printed to two figures: recomputed 0.00123, but a 5 % negative control cannot fail at two figures (1.05 x 0.0012 = 0.00126 rounds the same) | - |
| 274 | app:derivations:L274 | prediction | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 274 | app:derivations:L274:0.15750 | prediction | `0.15750` | numeric: beta_m = Omega_m/2 at Omega_m = 0.315 | PASS |
| 274 |  | prediction | `0.3153` | not run: input: Omega_m = 0.3153 (Planck 2018) from which beta_m is computed (checked at app:derivations:L274) | - |
| 274 |  | prediction | `0.315` | not run: input: Omega_m = 0.315 from which beta_m = 0.15750 is computed (checked at app:derivations:L274:0.15750) | - |
| 274 |  | prediction | `0.62` | not run: input: f_coll = 0.62, the collapsed fraction the decomposition uses (stated input; its source discussion is in ch:theory) | - |
| 275 | app:derivations:L275 | calc | `0.81` | numeric: eta_vir = 1/(2 f_coll) at f_coll = 0.62 | PASS |
| 275 | app:derivations:L275:0.195 | calc | `0.195` | numeric: Omega_m f_coll | PASS |
| 277 | app:derivations:L277 | calc | `1.0759` | numeric: same value as p1_02_iams_law:516 (H_m/H at z=0) | PASS |
| 277 | app:derivations:L277:72.26 | calc | `72.26` | numeric: same value as p1_02_iams_law:696 (H0 matter-sector formula) | PASS |
| 277 | app:derivations:L277:67.161 | calc | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_03_theory:889 (Level2 posterior mean H0) | PASS |
| 278 | app:derivations:L278 | calc | `72.51` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 278 | app:derivations:L278:72.48 | calc | `72.48` | numeric: H0_m = H0 sqrt(1+beta_m) from 67.36 | PASS |
| 278 | app:derivations:L278:-0.75 | calc | `-0.75` | numeric: H0_m 72.26 against SH0ES 73.04 +- 1.04, in sigma | PASS |
| 278 |  | calc | `67.36` | not run: input: H0 = 67.36 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 278 |  | calc | `67.4` | not run: input: H0 = 67.4 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 278 |  | calc | `0.1575` | not run: input: beta_m = 0.1575 (= 0.315/2) used with H0 = 67.4 (result checked at app:derivations:L278) | - |
| 278 |  | calc | `73.04` | not run: input: SH0ES H0 = 73.04 +- 1.04 (Riess et al. 2022), the comparison value (sigma distance checked at app:derivations:L278:-0.75) | - |
| 279 | app:derivations:L279 | calc | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_03_theory:889 (Level2 posterior mean H0) | PASS |
| 279 | app:derivations:L279:0.3166 | calc | `0.3166` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p1_03_virial_law:144 (Level2 Planck posterior Omega_m mean) | PASS |
| 279 | app:derivations:L279:88.89 | calc | `88.89` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 279 | app:derivations:L279:91.29 | calc | `91.29` | numeric: drafted check, screened (runs; negative control fails) (tolerance: the posterior Omega_m is printed to 4 figures) | PASS |
| 279 | app:derivations:L279:-0.37 | calc | `-0.37` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2 chain H0 against Planck 67.36 +- 0.54, in sigma | PASS |
| 279 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which H and H_m are evaluated (checked at app:derivations:L279:88.89, L279:91.29) | - |
| 280 | app:derivations:L280 | calc | `120.44` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 280 | app:derivations:L280:121.53 | calc | `121.53` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 280 | app:derivations:L280:204.06 | calc | `204.06` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 280 | app:derivations:L280:204.29 | calc | `204.29` | numeric: drafted check, screened (runs; negative control fails) (tolerance: the posterior Omega_m is printed to 4 figures) | PASS |
| 280 | app:derivations:L280:307.37 | calc | `307.37` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 280 | app:derivations:L280:307.43 | calc | `307.43` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 282 | app:derivations:L282 | calc | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 282 | app:derivations:L282:399 | calc | `399` | numeric: R(a) = Omega_m a^-3/(beta_m E) at z = 2 | PASS |
| 282 | app:derivations:L282:19.8 | calc | `19.8` | numeric: R(a) at z = 0.7 | PASS |
| 282 | app:derivations:L282:5.9 | calc | `5.9` | numeric: R(a) at z = 0.3 | PASS |
| 282 | app:derivations:L282:2.0 | calc | `2.0` | numeric: R(a) at z = 0 | PASS |
| 282 |  | calc | `0.3153` | not run: input: Omega_m = 0.3153 (Planck 2018) for the record history (ratios checked at app:derivations:L282:399 ...) | - |
| 282 |  | calc | `0.7` | not run: input: redshift z = 0.7 at which R is evaluated (checked at app:derivations:L282:19.8) | - |
| 282 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which R is evaluated (checked at app:derivations:L282:5.9) | - |
| 283 | app:derivations:L283 | calc | `18.7` | numeric: record share of Omega_L + beta_m E at z = 0 (per cent) | PASS |
| 283 | app:derivations:L283:14.6 | calc | `14.6` | numeric: record share at z = 0.3 (per cent) | PASS |
| 283 | app:derivations:L283:10.3 | calc | `10.3` | numeric: record share at z = 0.7 (per cent) | PASS |
| 283 | app:derivations:L283:4.9 | calc | `4.9` | numeric: record share at z = 1.5 (per cent) | PASS |
| 283 |  | calc | `0.3` | not run: input: redshift z = 0.3 at which the record share is evaluated (checked at app:derivations:L283:14.6) | - |
| 283 |  | calc | `0.7` | not run: input: redshift z = 0.7 at which the record share is evaluated (checked at app:derivations:L283:10.3) | - |
| 283 |  | calc | `1.5` | not run: input: redshift z = 1.5 at which the record share is evaluated (checked at app:derivations:L283:4.9) | - |
| 284 | app:derivations:L284 | calc | `0.295` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 284 | app:derivations:L284:0.361 | calc | `0.361` | numeric: redshift where matter equals vacuum plus record | PASS |
| 288 | app:derivations:L288 | derived |  | sympy: rho_Lambda/rho_vac = (3 Omega_L/8 pi) hbar G H0^2/c^5 = (3 Omega_L/8 pi)(l_P/l_H)^2 | PASS |
| 291 | app:derivations:L291 | derived | `0.6847` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 291 | app:derivations:L291:4.633\times10^{113} | derived | `4.633\times10^{113}` | numeric: same value as p2_12_lambda:35 (Planck-cutoff vacuum energy density) | PASS |
| 291 | app:derivations:L291:5.251\times10^{-10} | derived | `5.251\times10^{-10}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 291 |  | derived | `67.4` | not run: input: H0 = 67.4 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 292 | app:derivations:L292 | calc | `1.1334\times10^{-123}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 296 | app:derivations:L296 | derived | `0.0493` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 296 | app:derivations:L296:0.3153 | derived | `0.3153` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 297 | app:derivations:L297 | calc | `1.380\times10^{-123}` | numeric: same value as p1_02_iams_law:734 (baseline Lambda/rho_vac with Ob,Om) | PASS |
| 297 | app:derivations:L297:1.218 | calc | `1.218` | numeric: same value as p2_12_lambda:184 (ratio of prediction to observation) | PASS |
| 297 | app:derivations:L297:1.142\times10^{-123} | calc | `1.142\times10^{-123}` | numeric: same value as p1_02_iams_law:739 (corrected Lambda/rho_vac with sqrt(OmegaL)) | PASS |
| 297 | app:derivations:L297:+0.79 | calc | `+0.79` | numeric: expression with sqrt(Omega_L) over the identity, per cent | PASS |
| 298 | app:derivations:L298 | calc | `0.1564` | numeric: same value as p2_12_lambda:172 (baryon fraction of matter) | PASS |
| 298 | app:derivations:L298:0.1551 | calc | `0.1551` | numeric: same value as p2_12_lambda:259 ((3/16) sqrt(Omega_L)) | PASS |
| 298 | app:derivations:L298:0.521 | calc | `0.521` | numeric: power of Omega_L that closes the first form exactly | PASS |
| 302 | app:derivations:L302:0.1430 | calc | `0.1430` | numeric: Omega_m h^2 = Omega_c h^2 + Omega_b h^2 + Omega_nu h^2 (Planck 2018) | PASS |
| 302 |  | calc | `273.9\times10^{-10}` | not run: input: the standard conversion eta = 273.9e-10 Omega_b h^2 (big-bang nucleosynthesis literature); its value depends on the mean mass per baryon and T_CMB conventions (m_p and T_CMB = 2.7255 K give 273.4), so it is a stated input, not a result of the chapter; eta values computed from it are checked at app:derivations:L303 | - |
| 303 | app:derivations:L303 | calc | `5.03\times10^{-10}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 303 | app:derivations:L303:6.08\times10^{-10} | calc | `6.08\times10^{-10}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 308 | app:derivations:L308 | derived |  | sympy: m c^2 = E_bit N/f solved: m = (2 pi)^(-2/5) B^(2/5) | PASS |
| 311 | app:derivations:L311 | derived | `1.2018` | numeric: same value as p4_15b_electron_mass:80 (B/m_e at H0 = 67.4) | PASS |
| 311 | app:derivations:L311:0.832112 | derived | `0.832112` | numeric: same value as p4_15b_electron_mass:90 ((2 pi)^(-1/10)) | PASS |
| 311 |  | derived | `67.4` | not run: input: H0 = 67.4 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 312 | app:derivations:L312 | calc | `1.7356` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 312 | app:derivations:L312:+6.6\times10^{-6} | calc | `+6.6\times10^{-6}` | numeric: m/m_e - 1 with (2 pi)^(-1/10) at H0 = 67.4 | PASS |
| 313 | app:derivations:L313:-2.31\times10^{-4} | calc | `-2.31\times10^{-4}` | numeric: m/m_e - 1 at H0 = 67.36 | PASS |
| 313 | app:derivations:L313:+3.24\times10^{-2} | calc | `+3.24\times10^{-2}` | numeric: m/m_e - 1 at H0 = 73.0 | PASS |
| 313 | app:derivations:L313:+3.27\times10^{-2} | calc | `+3.27\times10^{-2}` | numeric: m/m_e - 1 at H0 = 73.04 | PASS |
| 313 |  | calc | `67.4` | not run: input: H0 = 67.4 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 313 |  | calc | `67.36` | not run: input: H0 = 67.36 km/s/Mpc, Planck 2018 (Aghanim et al. 2020, bib key Planck2018VI), the value at which the numbers on this line are evaluated | - |
| 313 |  | calc | `73.0` | not run: input: H0 = 73.0 km/s/Mpc, a local-distance-ladder value at which m/m_e is evaluated (checked at app:derivations:L313:+3.24\times10^{-2}) | - |
| 313 |  | calc | `73.04` | not run: input: SH0ES H0 = 73.04 (Riess et al. 2022) at which m/m_e is evaluated (checked at app:derivations:L313:+3.27\times10^{-2}) | - |
| 314 | app:derivations:L314:0.32 | calc | `0.32` | numeric: sigma(H0) = 0.54 as a per cent in m (m ~ H0^(2/5)) | PASS |
| 314 |  | calc | `0.54` | not run: input: sigma(H0) = 0.54, the Planck 2018 uncertainty (propagated at app:derivations:L314:0.32) | - |
| 318 | app:derivations:L318 | derived |  | sympy: Koide Q = (1/3)(1 + y^2/2x^2) = 2/3 at y = sqrt2 x | PASS |
| 321 | app:derivations:L321 | conjecture | `0.66666051` | numeric: same value as p4_15a_lepton_koide:35 (Koide Q with the 2022 m_tau) | PASS |
| 321 | app:derivations:L321:313.84 | conjecture | `313.84` | numeric: Koide scale x^2 from the measured masses, MeV | PASS |
| 322 | app:derivations:L322:0.22227 | calc | `0.22227` | numeric: Koide offset delta, rad | PASS |
| 322 | app:derivations:L322:0.510 | calc | `0.510` | numeric: m_e returned by x and delta, MeV | PASS |
| 322 | app:derivations:L322:105.68 | calc | `105.68` | numeric: m_mu returned by x and delta, MeV | PASS |
| 323 | app:derivations:L323 | calc | `26.9` | numeric: the two lighter masses at delta = 0, MeV | PASS |
| 327 | app:derivations:L327:0.50 | derived | `0.50` | numeric: allowed fraction of offsets, n = 2 | PASS |
| 327 | app:derivations:L327:0.25 | derived | `0.25` | numeric: allowed fraction of offsets, n = 3 | PASS |
| 330 | app:derivations:L330 | calc | `0.1330` | numeric: same value as p4_22_electroweak:90 (Omega_dm/2) | PASS |
| 330 | app:derivations:L330:0.1577 | calc | `0.1577` | numeric: same value as p4_22_electroweak:90 (beta_m = Omega_b/2 + Omega_dm/2 (Planck 2018 Omega_b 0.0493)) | PASS |
| 330 | app:derivations:L330:0.0247 | calc | `0.0247` | numeric: Omega_b/2 | PASS |
| 330 | app:derivations:L330:15.6 | calc | `15.6` | numeric: baryon share of beta_m, per cent | PASS |
| 333 | app:derivations:L333 | calc | `2.5\times10^{-87}` | numeric: hbar R/(G M^2) for 1e12 M_sun within 200 kpc, s | PASS |
| 333 |  | calc | `10` | not run: input: the printed 10 is the base of the halo mass 10^12 M_sun (collapse time checked at app:derivations:L333) | - |
| 333 |  | calc | `200` | not run: input: halo radius 200 kpc (collapse time checked at app:derivations:L333) | - |
| 338 | app:derivations:L338 | calc | `-1.667` | numeric: same value as p2_20_wz_far_future:23 (w_info at z=1) | PASS |
| 338 | app:derivations:L338:4.77 | calc | `4.77` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 338 | app:derivations:L338:-5.000 | calc | `-5.000` | numeric: d ln tau_IAM/d ln m at fixed density | PASS |
| 338 |  | calc | `2200` | not run: input: silica density 2200 kg/m^3 | - |
| 338 |  | calc | `10` | not run: input: the printed 10 is the base of the mass 10^-12 kg | - |
| 339 | app:derivations:L339 | calc | `1.40\times10^{-29}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 339 | app:derivations:L339:509 | calc | `509` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 339 | app:derivations:L339:4.0 | calc | `4.0` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 339 | app:derivations:L339:16.0 | calc | `16.0` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 339 | app:derivations:L339:7.54 | calc | `7.54` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 340 | app:derivations:L340 | calc | `2.2\times10^{-10}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 340 | app:derivations:L340:49 | calc | `49` | numeric: R needed for tau_IAM = 560 microseconds at 1e-12 kg, nm | PASS |
| 340 | app:derivations:L340:2.0\times10^9 | calc | `2.0\times10^9` | numeric: density for that radius, kg/m^3 | PASS |
| 340 |  | calc | `560` | not run: input: the 560 microseconds coherence time the sentence posits; R and density it requires are checked at app:derivations:L340:49 and L340:2.0\times10^9 | - |
| 341 | app:derivations:L341 | calc | `7.1\times10^{-9}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 341 | app:derivations:L341:3.50\times10^{-51} | calc | `3.50\times10^{-51}` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 341 | app:derivations:L341:0.150 | calc | `0.150` | numeric: R = G m^2/E_G for 4 kg and E_G = 7.1e-9 J, m | PASS |
| 343 | app:derivations:L343 | derived | `0.0179` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 343 | app:derivations:L343:0.632 | derived | `0.632` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 346 |  | derived | `00` | not run: not a number: ket label |00> of the Bell state | - |
| 346 |  | derived | `11` | not run: not a number: ket label |11> of the Bell state | - |
| 348 | app:derivations:L348 | derived | `0.414` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 348 | app:derivations:L348:0.586 | derived | `0.586` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 349 | app:derivations:L349 | derived | `1.87` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 349 | app:derivations:L349:0.414 | derived | `0.414` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 349 |  | derived | `0.7` | not run: input: dephasing value c = 0.7 at which the Bell bound is checked numerically | - |
| 349 |  | derived | `0.2` | not run: input: dephasing value c = 0.2 at which the Bell bound is checked numerically | - |
| 355 | app:derivations:L355 | openprob | `5.2\times10^{121}` | numeric: bits per e-fold today restated in the open list | PASS |
| 363 | app:derivations:L363 | openprob | `-0.78` | numeric: Delta D/D today, form (i), restated | PASS |
| 363 | app:derivations:L363:-0.67 | openprob | `-0.67` | numeric: Delta D/D today, form (ii), restated | PASS |
| 363 | app:derivations:L363:-1.87 | openprob | `-1.87` | numeric: Delta D/D today, form (iii), restated | PASS |

## Part 8 - app:saturation - `docs/book/appendices/app_S_saturation.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 43 | app:saturation:L43 | derived | `2.77` | numeric: one bit per 4 ln2 l_P^2, in Planck areas | PASS |
| 45 | eq:sat_eta | derived |  | sympy: 1/4 = 2 pi / 8 pi and eta = c^3/(4 hbar G) = 1/(4 l_P^2) | PASS |
| 76 |  | derived | `310.15` | not run: input: T_cell (CANON), body temperature | - |
| 117 | app:saturation:L117 | prediction | `0.15765` | numeric: beta_m = Omega_m/2 with Omega_m = 0.3153 | PASS |
| 117 | eq:sat_beta | prediction | `0.3153` | not run: input: Planck 2018 Omega_m (Aghanim et al. 2020) | - |
| 120 | app:saturation:L120 | measured | `0.2` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Planck posterior vs fixed beta_m sigma | PASS |
| 140 |  | calc | `310.15` | not run: input: T_cell (CANON), body temperature | - |
| 142 | eq:sat_ebit | calc | `2.97\times10^{-21}` | numeric: k_B T ln2 at T_body = 310.15 K | PASS |
| 146 | app:saturation:L146 | calc | `28{,}217{,}448` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpG sites of the hg19 index | PASS |
| 169 | app:saturation:L169 | calc | `5.0\times10^{9}` | numeric: T_body / T_BH, 1 M_sun | PASS |
| 169 | app:saturation:L169:5.4\times10^{69} | calc | `5.4\times10^{69}` | numeric `Biological_Physics/MethylPhys/atlas/v2/README.md`: N_BH / N_CpG, 1 M_sun | PASS |
| 177 | app:saturation:L177 | calc | `6.17\times10^{-8}` | numeric: T_BH, 1 M_sun | PASS |
| 177 |  | calc | `310.15` | not run: input: T_cell (CANON), body temperature | - |
| 178 | app:saturation:L178 | calc | `1.05\times10^{77}` | numeric: A/4 l_P^2 of one solar mass, nats | PASS |
| 178 | app:saturation:L178:1.51\times10^{77} | calc | `1.51\times10^{77}` | numeric: bits, 1 M_sun | PASS |
| 178 | app:saturation:L178:2.82\times10^{7} | calc | `2.82\times10^{7}` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpG sites of the hg19 index, rounded | PASS |
| 179 | app:saturation:L179 | calc | `5.90\times10^{-31}` | numeric: k_B T_BH ln2, 1 M_sun | PASS |
| 179 | app:saturation:L179:2.97\times10^{-21} | calc | `2.97\times10^{-21}` | numeric: k_B T_body ln2 in the table | PASS |

## Part 8 - app:glossary - `docs/book/appendices/app_F_glossary.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 18 | app:glossary:L18:0.067 | observed | `0.067` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: 6dFGS f sigma8 redshift | PASS |
| 18 | app:glossary:L18:0.15 | observed | `0.15` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: SDSS MGS f sigma8 redshift | PASS |
| 24 | app:glossary:L24:0.0052 | observed | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma 95 % bound from the acoustic scale | PASS |
| 26 |  | observed | `0.05` | not run: definition: identity-site rule, across-array SD at most 0.05 (canon Met_A_site_rule) | - |
| 28 | app:glossary:L28:0.02 | observed | `0.02` | file `docs/verification/chains/LATE_TIME_GROWTH_CHECK.md`: ACT + WMAP + SDSS + SN mu0 (Andrade et al. 2024) | PASS |
| 46 | app:glossary:L46:3.03 | observed | `3.03` | file `CANON/iam_canon.json`: Met-A at the full surface, neutrophils | PASS |
| 46 | app:glossary:L46:4.45 | observed | `4.45` | file `CANON/iam_canon.json`: IAM-A at the full surface, neutrophils | PASS |
| 48 | app:glossary:L48:576 | observed | `576` | numeric: 9950X switching energy over k_B T_j ln 2, upper transistor count | PASS |
| 48 | app:glossary:L48:593 | observed | `593` | numeric: 9950X switching energy over k_B T_j ln 2, lower transistor count | PASS |
| 48 |  | observed | `9950` | not run: not a number: part of the processor name (Ryzen 9 9950X) | - |
| 48 |  | observed | `20` | not run: input: transistor count (20.0-20.6) x 10^9 from die-level reports (no maker figure); the readings it gives are checked at app:glossary:L48:576 and L48:593 | - |
| 50 | app:glossary:L50:10 | observed | `10` | file `Biological_Physics/MethylPhys/doors/PROC_AML_SERIAL_01_OUTCOME.md`: AML diagnosis blood: patients read | PASS |
| 58 | app:glossary:L58 | observed | `1.099` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 58 | app:glossary:L58:1.084 | observed | `1.084` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 60 | app:glossary:L60 | observed | `450` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 68 | app:glossary:L68 | calc | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p0_giants:41 (H0 photon sector matches Level2 chain value) | PASS |
| 68 | app:glossary:L68:55.57 | calc | `55.57` | numeric: same value as p2_11_dark_energy:146 (precise H_infinity, Level 2 chains) | PASS |
| 68 | app:glossary:L68:70.86 | calc | `70.86` | numeric: same value as p2_11_dark_energy:173 (H_m asymptote, matter-sector) | PASS |
| 68 | app:glossary:L68:67.4 | calc | `67.4` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 68 | app:glossary:L68:0.315 | calc | `0.315` | numeric: drafted check, screened (runs; negative control fails) | PASS |
| 68 | app:glossary:L68:71.12 | observed | `71.12` | numeric: matter-sector late rate, Planck 2018 base values | PASS |
| 70 | app:glossary:L70:74 | observed | `74` | file `Biological_Physics/MethylPhys/atlas/v2/postbuild/records/atlas_v2_gate.json`: atlas v2: cell types | PASS |
| 70 | app:glossary:L70:814 | observed | `814` | file `Biological_Physics/MethylPhys/atlas/v2/postbuild/records/atlas_v2_gate.json`: atlas v2: CpGs fitted, thousands | PASS |
| 70 |  | observed | `000` | not run: not a separate number: thousands group of "814,000", checked at app:glossary:L70:814 | - |
| 72 | app:glossary:L72 | observed | `20.94` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 72 | app:glossary:L72:54 | observed | `54` | file `CANON/iam_canon.json`: ATP free energy of hydrolysis, kJ/mol | PASS |
| 74 | app:glossary:L74:61.5 | observed | `61.5` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2b H0 (term in the background) | PASS |
| 74 | app:glossary:L74:10.9 | observed | `10.9` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Level 2b H0 from Planck, in Planck errors | PASS |
| 88 | app:glossary:L88:0.0052 | observed | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma 95 % bound | PASS |
| 88 | app:glossary:L88:3.3 | observed | `3.3` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: beta_gamma bound over beta_m, per cent | PASS |
| 89 | app:glossary:L89 | observed | `0.15765` | numeric: same value as p1_02_iams_law:443 (beta_m is half of Omega_m) | PASS |
| 89 | app:glossary:L89:0.3153 | observed | `0.3153` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 89 | app:glossary:L89:0.1575 | observed | `0.1575` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 89 | app:glossary:L89:0.315 | observed | `0.315` | file `CANON/iam_canon.json`: Omega_m written 0.315 (= 2 beta_m to three decimals) | PASS |
| 90 | app:glossary:L90:0.295 | observed | `0.295` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DESI BGS effective redshift | PASS |
| 103 |  | observed | `50` | not run: definition: block size of the C-score, 50 consecutive identity sites | - |
| 109 | app:glossary:L109:5.5 | observed | `5.5` | heavy file `docs/verification/scripts/verify_bottom_up_exponent_output.txt`: bottom-up exponent at z = 9 | PASS |
| 116 |  | observed | `30` | not run: definition: 30 % burn-in, the setting of the book extractions | - |
| 124 | app:glossary:L124:0.97 | observed | `0.97` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d) of beta at 10-18 bp | PASS |
| 124 | app:glossary:L124:10 | observed | `10` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d): lower edge of the first distance bin | PASS |
| 124 | app:glossary:L124:18 | observed | `18` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d): upper edge of the first distance bin | PASS |
| 124 | app:glossary:L124:0.32 | observed | `0.32` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d) of beta at 1-1.8 kb | PASS |
| 124 | app:glossary:L124:1.8 | observed | `1.8` | file `Biological_Physics/MethylPhys/reference_floors_v1/sky/cd_neut.csv`: C(d): upper edge of the 1 kb bin, kb | PASS |
| 124 |  | observed | `0.05` | not run: printed value 0.05 has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.0526; the same quantity is printed to one digit at ch:skytools:L125) | - |
| 126 |  | observed | `10` | not run: input: about 10^4 genome equivalents in a millilitre-scale draw, an order of magnitude (used in ch:sky L73-74) | - |
| 133 | app:glossary:L133:1.456 | observed | `1.456` | numeric: Chandrasekhar mass for mu_e = 2 from the constants | PASS |
| 133 |  | observed | `1.44` | not run: input: Chandrasekhar mass 1.44 M_sun as conventionally quoted (Chandrasekhar1931); the constants-only 1.456 beside it is checked at app:glossary:L133:1.456 | - |
| 138 | app:glossary:L138:+0.54 | observed | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: Delta chi2 IAM minus LambdaCDM, Level 2 | PASS |
| 144 | app:glossary:L144:2.7255 | observed | `2.7255` | numeric: CMB temperature today (Fixsen 2009) | PASS |
| 149 | app:glossary:L149:1.2 | observed | `1.2` | file `Biological_Physics/MethylPhys/chain_tests/iama_floor_granulocytes.csv`: coefficient of variation of P across three donors, per cent | PASS |
| 153 | app:glossary:L153:1.68 | observed | `1.68` | numeric: best 2D transmon T1 (Bland et al. 2025) | PASS |
| 156 |  | observed | `0.62` | not run: input: f_coll = 0.62 restated from Eq. vc_eta (ch:virial line 75); the mass it implies is checked at ch:virial:L126:8.2 | - |
| 156 |  | observed | `10` | not run: input: halo mass threshold 10^6 M_sun of the collapsed fraction (ch:virial), an order of magnitude | - |
| 171 | app:glossary:L171:0.024 | observed | `0.024` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy copy error, lowest of 56 cell types | PASS |
| 171 | app:glossary:L171:0.042 | observed | `0.042` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: healthy copy error, highest of 56 cell types | PASS |
| 178 | app:glossary:L178:0.7 | observed | `0.7` | heavy file `docs/verification/scripts/verify_lambda_baryon_book_output.txt`: Omega_b/Omega_m against (3/16) sqrt(Omega_Lambda), in sigma | PASS |
| 180 |  | observed | `10` | not run: not a separate number: base of 10^{123}; the 123 orders are checked at app:glossary:L355:123 | - |
| 183 | app:glossary:L183:2.8e7 | observed | `2.8\times10^7` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpGs per haploid human genome | PASS |
| 183 |  | observed | `70` | not run: measured, source not named | - |
| 185 | app:glossary:L185:+0.2 | observed | `+0.2` | file `Cosmological_Physics/mgcamb_validation/chains/iam_float_mu0_r2.updated.yaml`: upper prior edge of mu0 | PASS |
| 185 |  | observed | `90` | not run: definition: central 90 % credible interval whose lower end (5 % quantile) is quoted | - |
| 187 |  | observed | `20` | not run: definition: a level of comparison ("rankings at the 20 % level"), not a computed number | - |
| 188 | app:glossary:L188:2.2e-10 | calc | `2.2\times10^{-10}` | numeric: crossover mass, tau_IAM = tau_DP at 10 mK | PASS |
| 188 |  | observed | `10` | not run: input: temperature 10 mK at which the crossover mass is evaluated (used in app:glossary:L188:2.2e-10) | - |
| 189 | app:glossary:L189 | observed | `1.1104` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 189 |  | observed | `50` | not run: definition: block size of the C-score, 50 consecutive sites | - |
| 193 | app:glossary:L193:84.4 | observed | `84.4` | numeric: dark-matter share of beta_m, per cent | PASS |
| 198 | app:glossary:L198:55.57 | observed | `55.57` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector de Sitter rate H_infinity | PASS |
| 201 | app:glossary:L201 | observed | `963` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 211 | app:glossary:L211:0.776 | observed | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: DES Y3 S8 | PASS |
| 214 |  | observed | `0.93` | not run: definition: Stage 0 call-rate quarantine threshold 0.93 (chain setting) | - |
| 214 |  | observed | `-0.98` | not run: definition: upper edge 0.98 of the Stage 0 call-rate flag band (chain setting) | - |
| 223 | app:glossary:L223:1.9 | observed | `1.9` | numeric: DNMT1 selectivity 7-fold in k_B T | PASS |
| 223 | app:glossary:L223:4.4 | observed | `4.4` | numeric: DNMT1 selectivity 80-fold in k_B T | PASS |
| 223 |  | observed | `-80` | not run: input: 80-fold average selectivity across flanking sequences (Adam2023, ch:landauer line 194); its k_B T ln 80 is checked at app:glossary:L223:4.4 | - |
| 224 | app:glossary:L224:1.16 | observed | `1.16` | file `Biological_Physics/MethylPhys/doors/data/dnmt_arrays_readings.csv`: DNMT1 inhibitor arrays: lowest Met-A at >= 80 nM | PASS |
| 230 | app:glossary:L230 | observed | `3.41` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 230 | app:glossary:L230:-3.77 | observed | `-3.77` | file `CANON/iam_canon.json`: measured: printed value found in iam_canon.json, a file the chapter names | PASS |
| 230 | app:glossary:L230:0.163 | observed | `0.163` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 230 | app:glossary:L230:4.9 | observed | `4.9` | numeric: holding energy in Landauer units | PASS |
| 238 | app:glossary:L238:+1.8 | observed | `+1.8` | numeric: E_G above GR at z = 0.3, per cent | PASS |
| 238 | app:glossary:L238:+3.6 | observed | `+3.6` | numeric: E_G above GR today, per cent | PASS |
| 238 |  | observed | `0.3` | not run: input: redshift z = 0.3 at which E_G is evaluated (used in app:glossary:L238:+1.8) | - |
| 243 | app:glossary:L243:0.576 | observed | `0.576` | numeric: electron fixed point without the (2 pi)^(3/10) prefactor, in m_e | PASS |
| 243 |  | observed | `0.3` | not run: printed value 0.3 % has one significant digit: a 5 % change cannot be told from rounding (recomputed 0.320 %; the precise 0.32 % is checked at ch:electronmass:L188) | - |
| 245 | app:glossary:L245:159.5 | observed | `159.5` | numeric: electroweak crossover temperature (lattice) | PASS |
| 245 | app:glossary:L245:9.2e-12 | observed | `9.2\times10^{-12}` | numeric: time of the electroweak crossover, s | PASS |
| 246 | app:glossary:L246 | observed | `246.22` | numeric: same value as p4_22_electroweak:62 (v = (sqrt2 G_F)^(-1/2), GeV) | PASS |
| 246 | app:glossary:L246:159.5 | observed | `159.5` | numeric: electroweak crossover temperature (lattice), restated | PASS |
| 247 |  | conjecture | `1.05` | not run: definition: upper edge of the Normal band 0.95-1.05 (canon Normal_band) | - |
| 247 |  | conjecture | `0.95` | not run: definition: lower edge of the Normal band 0.95-1.05 (canon Normal_band) | - |
| 251 | app:glossary:L251 | observed | `7309` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 251 | app:glossary:L251:738 | observed | `738` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 251 | app:glossary:L251:1056 | observed | `1{,}056` | file `Biological_Physics/MethylPhys/doors/PLAN.md`: E-MTAB-7309 arrays calibrated | PASS |
| 254 | app:glossary:L254 | calc | `152.5` | numeric: same value as p1_02_iams_law:650 (Hawking info rate for 1 solar mass) | PASS |
| 258 |  | observed | `1.00` | not run: definition: an atlas cell reads 1.00 on its own profile by construction (entry rule); nothing to compute, and the number is not in MethylPhys_CPG_SOP_v3.md (the earlier match was an unrelated number) | - |
| 260 |  | observed | `865` | not run: measured, source not named | - |
| 260 |  | observed | `000` | not run: not a separate number: thousands group of "865,000" (the EPIC CpG count, listed in sources_needed) | - |
| 262 | app:glossary:L262 | observed | `3.41` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 262 | app:glossary:L262:0.032 | observed | `0.032` | numeric: copy-error floor eps0 | PASS |
| 263 | app:glossary:L263 | observed | `-1.062` | file: w_eff(1) for vacuum plus record | PASS |
| 270 |  | observed | `10` | not run: input: surface-code threshold about 10^{-2}, an order of magnitude quoted in ch:ascoreqc | - |
| 274 | app:glossary:L274:5120 | observed | `5120` | numeric: 5120 in the evaporation time | PASS |
| 274 | app:glossary:L274:2.1e67 | observed | `2.1\times10^{67}` | numeric: evaporation time of one solar mass, years | PASS |
| 280 |  | observed | `10` | not run: input: fault-tolerance target about 10^{-3}, an order of magnitude quoted in ch:ascoreqc (Fowler2012, GoogleWillow2025) | - |
| 286 |  | observed | `0.93` | not run: definition: lower edge of the call-rate flag band 0.93-0.98 (chain setting) | - |
| 286 |  | observed | `0.98` | not run: definition: upper edge of the call-rate flag band 0.93-0.98 (chain setting) | - |
| 287 | app:glossary:L287:45 | observed | `45` | numeric: Koide angle to the democratic direction | PASS |
| 295 | app:glossary:L295:4.25 | observed | `4.25` | numeric: f sigma8 deficit today, per cent | PASS |
| 295 | app:glossary:L295:1.35 | observed | `1.35` | numeric: f sigma8 deficit at z = 0.5, per cent | PASS |
| 295 | app:glossary:L295:0.04 | observed | `0.04` | numeric: f sigma8 deficit at z = 2, per cent | PASS |
| 295 |  | observed | `0.5` | not run: input: redshift z = 0.5 at which the deficit is evaluated (used in app:glossary:L295:1.35) | - |
| 304 |  | observed | `0.01` | not run: definition: convergence criterion R - 1 < 0.01 of the chains | - |
| 311 | app:glossary:L311:2.65e-30 | observed | `2.65\times10^{-30}` | numeric: Gibbons-Hawking temperature today | PASS |
| 312 | app:glossary:L312:105 | observed | `105` | numeric: qubits of the Willow processor | PASS |
| 312 |  | observed | `68` | not run: measured, source not named | - |
| 316 | app:glossary:L316:1.315 | observed | `1.315` | numeric: gravitational slip today | PASS |
| 318 | app:glossary:L318:0.78 | observed | `0.78` | numeric: growth factor deficit today, per cent | PASS |
| 319 | app:glossary:L319:0.55 | observed | `0.55` | numeric: growth index of general relativity (LambdaCDM) | PASS |
| 319 | app:glossary:L319:0.585 | observed | `0.585` | numeric: growth index under IAM today | PASS |
| 319 | app:glossary:L319:0.633 | observed | `0.633` | heavy file `docs/verification/scripts/verify_s8_trend_output.txt`: measured growth index (Nguyen et al. 2023) | PASS |
| 320 | app:glossary:L320 | observed | `100` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 320 | app:glossary:L320:106.75 | observed | `106.75` | numeric: g_*s of the standard model | PASS |
| 320 | app:glossary:L320:7.8e-16 | observed | `7.8\times10^{-16}` | numeric: scale factor at 100 GeV | PASS |
| 323 | app:glossary:L323:64 | observed | `64` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: GSE250556 arrays | PASS |
| 323 | app:glossary:L323:0.30 | observed | `0.30` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: GSE250556: lowest neutrophil fraction | PASS |
| 323 | app:glossary:L323:0.56 | observed | `0.56` | file `Biological_Physics/MethylPhys/doors/data/chain_v3_dev3_readings.csv`: GSE250556: highest neutrophil fraction | PASS |
| 324 | app:glossary:L324 | measured | `100` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 325 |  | observed | `-70` | not run: measured, source not named | - |
| 325 |  | observed | `75` | not run: measured, source not named | - |
| 327 | app:glossary:L327:67.16 | observed | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: photon-sector H0 (Level 2 chain) | PASS |
| 327 | app:glossary:L327:72.26 | observed | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: matter-sector H0 = 67.16 sqrt(1 + beta_m) | PASS |
| 327 | app:glossary:L327:67.36 | observed | `67.36` | numeric: Planck 2018 H0 | PASS |
| 327 | app:glossary:L327:73.04 | observed | `73.04` | numeric: SH0ES H0 | PASS |
| 328 |  | observed | `0.90` | not run: measured, source not named | - |
| 328 |  | observed | `-0.98` | not run: measured, source not named | - |
| 334 | app:glossary:L334:6.17e-8 | observed | `6.17\times10^{-8}` | numeric: Hawking temperature of one solar mass | PASS |
| 335 |  | observed | `12` | not run: definition: HEALPix divides the sphere into 12 N_side^2 pixels (Gorski2005) | - |
| 336 | app:glossary:L336 | observed | `0.330263` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 336 | app:glossary:L336:0.2246 | observed | `0.2246` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 338 | app:glossary:L338 | observed | `0.983` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: held-out neutrophil reference arrays, lowest Met-A | PASS |
| 338 | app:glossary:L338:-1.045 | observed | `-1.045` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 341 | app:glossary:L341:28217448 | observed | `28{,}217{,}448` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpGs of the hg19 index | PASS |
| 341 |  | observed | `217` | not run: not a separate number: thousands group of "28,217,448", checked at app:glossary:L341:28217448 | - |
| 341 |  | observed | `448` | not run: not a separate number: last group of "28,217,448", checked at app:glossary:L341:28217448 | - |
| 342 | app:glossary:L342:2.8e7 | observed | `2.8\times10^7` | file `Biological_Physics/MethylPhys/atlas/v2/README.md`: CpGs per haploid genome (hg19 count) | PASS |
| 342 |  | observed | `217` | not run: not a separate number: thousands group of "28,217,448", checked at app:glossary:L341:28217448 | - |
| 344 | app:glossary:L344 | observed | `246.22` | numeric: same value as p4_22_electroweak:62 (v = (sqrt2 G_F)^(-1/2), GeV) | PASS |
| 344 | app:glossary:L344:125.20 | observed | `125.20` | numeric: Higgs boson mass (PDG 2024) | PASS |
| 344 | app:glossary:L344:0.129 | observed | `0.129` | numeric: Higgs self-coupling m_H^2/(2 v^2) | PASS |
| 348 | app:glossary:L348 | observed | `0.2043` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 348 | app:glossary:L348:0.910 | observed | `0.910` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 354 | app:glossary:L354:2.9e78 | calc | `2.9\times10^{78}` | numeric: horizon capacity at T = 150 MeV, nats | PASS |
| 354 |  | calc | `150` | not run: input: temperature T = 150 MeV (QCD scale) at which the horizon capacity is evaluated (used in app:glossary:L354:2.9e78) | - |
| 355 | app:glossary:L355:123 | observed | `123` | numeric: orders of magnitude of the cosmological-constant problem | PASS |
| 359 | app:glossary:L359:0.776 | observed | `0.776` | heavy file `docs/verification/scripts/verify_sector_tension_output.txt`: HSC Y3 S8 | PASS |
| 360 | app:glossary:L360:1588 | observed | `1588` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_data.json`: Pantheon+ supernovae in the Hubble flow | PASS |
| 361 | app:glossary:L361:1.4e26 | observed | `1.4\times10^{26}` | numeric: Hubble radius c/H0, m | PASS |
| 368 | app:glossary:L368 | observed | `1.099` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 368 | app:glossary:L368:100 | observed | `100` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 368 | app:glossary:L368:-1.05 | observed | `1.05` | file `CANON/iam_canon.json`: IAM-A: Normal band, upper edge (canon Normal_band) | PASS |
| 368 | app:glossary:L368:0.032 | observed | `0.032` | numeric: copy-error floor eps0 (IAM-A entry) | PASS |
| 368 |  | observed | `000` | not run: definition: at least 100,000 opportunities per IAM-A reading (chain setting); thousands group | - |
| 368 |  | observed | `0.95` | not run: definition: lower edge of the Normal band 0.95-1.05 (canon Normal_band) | - |
| 370 | app:glossary:L370:0.80 | observed | `0.80` | file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: ICC of cell-type copy-error differences across donors | PASS |
| 372 |  | observed | `0.75` | not run: definition: identity-site window 0.75-0.95 (canon Met_A_site_rule) | - |
| 372 |  | observed | `-0.95` | not run: definition: identity-site window 0.75-0.95 (canon Met_A_site_rule) | - |
| 372 |  | observed | `0.05` | not run: definition: identity-site window 0.05-0.25 (canon Met_A_site_rule) | - |
| 372 |  | observed | `-0.25` | not run: definition: identity-site window 0.05-0.25 (canon Met_A_site_rule) | - |
| 372 |  | observed | `000` | not run: definition: at most 3,000 sites per channel (canon Met_A_site_rule); thousands group | - |
| 372 |  | observed | `90` | not run: definition: at least 90 % of identity sites measured in a reading (chain setting) | - |
| 379 |  | observed | `0.93` | not run: definition: intake call-rate threshold 0.93 (chain setting) | - |
| 395 |  | observed | `75` | not run: input: nominal junction temperature 75 C of published TDP figures (348.15 K, used in app:glossary:L48:576) | - |
| 400 | app:glossary:L400:0.815 | observed | `0.815` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: KiDS-Legacy S8 | PASS |
| 400 | app:glossary:L400:0.021 | observed | `0.021` | heavy file `docs/verification/scripts/verify_virial_papers_output.txt`: KiDS-Legacy S8, lower error | PASS |
| 400 |  | observed | `1000` | not run: not a number: part of the survey name KiDS-1000 | - |
| 406 | app:glossary:L406:2.2e-6 | observed | `2.2\times10^{-6}` | numeric: Koide: 2/3 - Q, PDG 2024 | PASS |
| 406 | app:glossary:L406:0.43 | observed | `0.43` | numeric: Koide: (2/3 - Q) in units of sigma(Q) | PASS |
| 406 | app:glossary:L406:0.2222 | observed | `0.2222` | numeric: Koide offset delta | PASS |
| 407 | app:glossary:L407:1.57 | observed | `1.57` | numeric: Koomey doubling time, years | PASS |
| 407 |  | observed | `2.7` | not run: measured, source not named | - |
| 411 | app:glossary:L411 | observed | `2.968\times10^{-21}` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 411 | app:glossary:L411:310.15 | observed | `310.15` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 411 | app:glossary:L411:348 | observed | `348` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 411 | app:glossary:L411:3.33e-21 | observed | `3.33\times10^{-21}` | numeric: Landauer floor at 348 K | PASS |
| 413 | app:glossary:L413 | observed | `20.94` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 413 | app:glossary:L413:30.2 | observed | `30.2` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 413 | app:glossary:L413:3.41 | observed | `3.41` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 413 | app:glossary:L413:4.9 | observed | `4.9` | numeric: holding energy in Landauer units | PASS |
| 420 | app:glossary:L420 | observed | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 of the Level 2b background chain A | PASS |
| 420 | app:glossary:L420:61.52 | observed | `61.52` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 of the Level 2b background chain D | PASS |
| 422 | app:glossary:L422 | observed | `0.76` | heavy numeric `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: likelihood ratio of the Level 2 Planck chain | PASS |
| 426 | app:glossary:L426 | observed | `68` | file `docs/verification/observations/MISSING_SATELLITES_CHECK.md`: Milky Way satellites in the Local Volume Database | PASS |
| 429 | app:glossary:L429 | observed | `1.3` | numeric: separation of the spins in the 2015 loophole-free Bell test, km | PASS |
| 430 | app:glossary:L430 | observed | `56` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: cell types of the Loyfer atlas read for the holding energy | PASS |
| 434 | app:glossary:L434 | observed | `1.05` | file: M_lens/M_dyn = 1/mu(z) at z = 0.5 (Level 1) | PASS |
| 434 | app:glossary:L434:1.02 | observed | `1.02` | file: M_lens/M_dyn = 1/mu(z) at z = 1 (Level 1) | PASS |
| 434 | app:glossary:L434:1.16 | observed | `1.16` | numeric: M_lens/M_dyn = 1/mu today, Level 1 form | PASS |
| 434 |  | calc | `0.5` | not run: input: redshift z = 0.5 at which 1/mu is quoted (the value 1.05 there is checked at app:glossary:L434) | - |
| 435 |  | calc | `6.5\times10^9` | not run: input: M87* black-hole mass 6.5e9 M_sun, Event Horizon Telescope 2019 (doi 10.3847/2041-8213/ab0ec7); the Smarr share at this mass is checked at ch:blackholes:L143 | - |
| 436 | app:glossary:L436 | observed | `20.94` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 438 |  | observed | `0.95` | not run: measured, source not named | - |
| 438 |  | observed | `-0.98` | not run: measured, source not named | - |
| 445 |  | observed | `30` | not run: definition: masking threshold of the copy-error statistic (more than 30 % of qualifying molecules in error), a chain rule | - |
| 446 | app:glossary:L446 | observed | `0.58` | heavy file `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: 1-b needed by Planck SZ counts with the primary-CMB cosmology | PASS |
| 446 | app:glossary:L446:0.80 | observed | `0.80` | numeric: baseline 1-b of about 0.80 | PASS |
| 451 | app:glossary:L451 | observed | `36.8` | numeric: maturity E(1)/e today, per cent | PASS |
| 457 | app:glossary:L457 | observed | `0.330263` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 457 | app:glossary:L457:-1.05 | observed | `1.05` | file `CANON/iam_canon.json`: Met-A: Normal band, upper edge (canon Normal_band) | PASS |
| 457 | app:glossary:L457:0.95 | observed | `0.95` | file `CANON/iam_canon.json`: lower edge of the Normal band (CANON) | PASS |
| 457 |  | observed | `20` | not run: definition: smallest neutrophil fraction (20 %) at which whole-blood Met-A is read, a chain rule | - |
| 463 | app:glossary:L463 | observed | `1{,}200` | numeric: Micius entanglement distribution distance, km | PASS |
| 464 | app:glossary:L464 | observed | `-1.5` | numeric: MICROSCOPE Eotvos parameter Ti-Pt, central value | PASS |
| 464 | app:glossary:L464:2.7e-15 | observed | `2.7\times10^{-15}` | numeric: MICROSCOPE total uncertainty, stat and syst in quadrature | PASS |
| 474 | app:glossary:L474 | observed | `-0.136` | numeric: mu0 = -beta_m/(1+beta_m) | PASS |
| 474 | app:glossary:L474:-0.13495 | observed | `-0.13495` | file `Cosmological_Physics/mgcamb_validation/chains/iam_fixed_mu0_r2.updated.yaml`: mu0 fixed in the MGCAMB runs | PASS |
| 474 | app:glossary:L474:4.25 | observed | `4.25` | numeric: f sigma8 deficit today, per cent | PASS |
| 479 |  | observed | `10` | not run: measured, source not named | - |
| 480 |  | observed | `1.4` | not run: input: canonical neutron-star mass 1.4 M_sun, nothing to recompute | - |
| 484 | app:glossary:L484 | observed | `528` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 484 | app:glossary:L484:-0.149 | observed | `-0.149` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 484 | app:glossary:L484:48528 | observed | `48{,}528` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: EPIC noise sites of the noise index | PASS |
| 486 | app:glossary:L486:0.95 | observed | `0.95` | file `CANON/iam_canon.json`: Normal band lower edge (CANON) | PASS |
| 498 | app:glossary:L498 | observed | `0.3153` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 501 | app:glossary:L501 | observed | `100` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 501 |  | observed | `80` | not run: definition: qualifying-molecule rule, at least 80 % of calls methylated (chain rule) | - |
| 501 |  | observed | `000` | not run: definition: at least 100,000 opportunities for an IAM-A reading (chain rule; the 000 is the tail of 100,000) | - |
| 507 |  | observed | `364` | not run: restates ch:xqp:L67 (2 Delta_Al with Delta_Al = 182 ueV, recomputed from BCS at ch:scprimer:L16:182) | - |
| 508 | app:glossary:L508:1701 | observed | `1701` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Pantheon+ light curves | PASS |
| 508 | app:glossary:L508:0.001 | observed | `0.001` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Pantheon+ lowest redshift | PASS |
| 508 | app:glossary:L508:2.26 | observed | `2.26` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Pantheon+ highest redshift | PASS |
| 514 |  | observed | `39` | not run: measured, source not named | - |
| 514 |  | observed | `-67` | not run: measured, source not named | - |
| 520 | app:glossary:L520 | observed | `30.9` | numeric: M = hc/(lambda k_B T) of a 1550 nm photon at 300 K | PASS |
| 520 |  | observed | `1550` | not run: input: photon wavelength 1550 nm; used in app:glossary:L520 | - |
| 521 | app:glossary:L521 | observed | `0.075` | file `Biological_Physics/MethylPhys/doors/PHASE1_OUTCOME.md`: pipeline offset in beta on immune identity sites | PASS |
| 525 | app:glossary:L525 | observed | `1.956\times10^9` | numeric: Planck energy sqrt(hbar c^5/G), J | PASS |
| 526 | app:glossary:L526 | observed | `1.616\times10^{-35}` | numeric: Planck length sqrt(hbar G/c^3), m | PASS |
| 526 | app:glossary:L526:2.176e-8 | observed | `2.176\times10^{-8}` | numeric: Planck mass sqrt(hbar c/G), kg | PASS |
| 528 | app:glossary:L528 | observed | `5.4\times10^{-44}` | numeric: Planck time sqrt(hbar G/c^5), s | PASS |
| 530 | app:glossary:L530 | observed | `450` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 534 | app:glossary:L534 | observed | `-5.69\times10^{41}` | numeric: index-3 polytrope binding energy of the Sun, J | PASS |
| 534 | app:glossary:L534:23.6 | observed | `23.6` | numeric: Kelvin-Helmholtz time of the Sun, Myr | PASS |
| 536 |  | observed | `20` | not run: definition: atlas v2 stores 20 posterior draws per value (a design choice), nothing to recompute | - |
| 539 | app:glossary:L539 | observed | `92.7` | file `Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md`: atlas v2 held-out 90 % interval coverage, per cent | PASS |
| 539 |  | observed | `90` | not run: definition: nominal 90 % coverage of the predictive interval (the interval level itself) | - |
| 542 | app:glossary:L542 | observed | `-0.5` | file `Cosmological_Physics/mgcamb_validation/chains/iam_float_mu0_r2.updated.yaml`: lower bound of the flat prior on free mu0 | PASS |
| 542 | app:glossary:L542:+0.2 | observed | `+0.2` | file `Cosmological_Physics/mgcamb_validation/chains/iam_float_mu0_r2.updated.yaml`: upper bound of the flat prior on free mu0 | PASS |
| 544 | app:glossary:L544:56 | observed | `56` | heavy file `Biological_Physics/MethylPhys/doors/PROC_CHANNEL_01_OUTCOME.md`: cell types of PROC-CHANNEL-01 | PASS |
| 544 |  | observed | `01` | not run: not a number: part of the identifier PROC-CHANNEL-01 | - |
| 553 | app:glossary:L553 | observed | `150` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 553 | app:glossary:L553:1e-5 | observed | `10^{-5}` | numeric: age of the radiation era at QCD confinement, s | PASS |
| 555 |  | observed | `80` | not run: definition: qualifying-molecule rule, at least 80 % of calls methylated (chain rule) | - |
| 556 | app:glossary:L556:12.7 | observed | `12.7` | numeric: surface-code threshold on the Helios gauge | PASS |
| 556 |  | observed | `98` | not run: measured, source not named | - |
| 556 |  | observed | `7.9\times10^{-4}` | not run: measured, source not named | - |
| 560 |  | observed | `0.93` | not run: definition: intake call-rate line 0.93 (quarantine rule of the chain) | - |
| 562 |  | observed | `10` | not run: measured, source not named | - |
| 567 |  | observed | `1.01` | not run: convention: R-hat < 1.01 is the acceptance threshold of the atlas fits, not a computed value; it is not in MethylPhys_CPG_SOP_v3.md (the earlier match was an unrelated number) | - |
| 570 |  | observed | `0.20` | not run: definition: read line 0.20, the smallest neutrophil fraction chain v3 reads (chain rule) | - |
| 583 | app:glossary:L583:0.79 | observed | `0.79` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: rho of Met-A and noise index, second laboratory, 30 y | PASS |
| 583 | app:glossary:L583:0.83 | observed | `0.83` | file `Biological_Physics/MethylPhys/doors/data/noise_index.csv`: rho of Met-A and noise index, second laboratory, 54 y | PASS |
| 587 | app:glossary:L587 | observed | `1.2\times10^{-3}` | numeric: departure of running-mass Koide Q from 2/3 | PASS |
| 588 | app:glossary:L588 | measured | `0.830` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 588 | app:glossary:L588:67.16 | measured | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: H0 of Level 2 run A | PASS |
| 588 | app:glossary:L588:0.7998 | measured | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 of Level 2 run A | PASS |
| 588 | app:glossary:L588:0.822 | measured | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 of Level 2 run A | PASS |
| 588 | app:glossary:L588:0.821 | measured | `0.821` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 of Level 2 run D | PASS |
| 593 | app:glossary:L593 | observed | `0.830` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 593 | app:glossary:L593:0.822 | observed | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: S8 of the Level 2 IAM chain | PASS |
| 599 | app:glossary:L599 | observed | `738` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 599 | app:glossary:L599:1056 | observed | `1{,}056` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: SATSA raw arrays | PASS |
| 599 |  | observed | `0.93` | not run: definition: intake call-rate line 0.93 restated (quarantine rule); the count 1,056 is checked at app:glossary:L599:1056 | - |
| 607 | app:glossary:L607:0.020 | observed | `0.020` | file `Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`: SD of the held-out reference readings, sites re-chosen | PASS |
| 607 |  | observed | `1.00` | not run: definition: 1.00 is the reading a reference sample should give by construction; nothing to compute; the selection-noise SDs (0.020, 0.006) are the measured part, and the earlier match in MethylPhys_CPG_SOP_v3.md was an unrelated number | - |
| 613 |  | observed | `4.3\times10^6` | not run: input: Sgr A* mass 4.3e6 M_sun (Gillessen 2009), as taken in Chapter ch:iams_law; nothing to recompute | - |
| 614 | app:glossary:L614 | observed | `73.04` | numeric: SH0ES H0 | PASS |
| 619 | app:glossary:L619 | observed | `0.91` | numeric: Cornell sigma = 0.18 GeV^2 in GeV/fm | PASS |
| 620 | app:glossary:L620 | observed | `0.809` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 of the Level 2 LambdaCDM chain | PASS |
| 620 | app:glossary:L620:0.800 | observed | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 of the Level 2 IAM chain | PASS |
| 620 | app:glossary:L620:0.813 | observed | `0.813` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: sigma8 of the Level 1 LambdaCDM baseline (Planck + RSD) | PASS |
| 622 | app:glossary:L622:509 | calc | `509` | numeric: tau_IAM of a 1e-12 kg silica sphere at 10 mK, s | PASS |
| 622 | app:glossary:L622:7.5 | calc | `7.5` | numeric: tau_DP = hbar/E_G of a 1e-12 kg silica sphere, us | PASS |
| 622 |  | calc | `10` | not run: input: test mass 1e-12 kg of the worked example (the 10 is the base of a printed power); tau values checked at app:glossary:L622:509 and L622:7.5 | - |
| 622 |  | observed | `2200` | not run: input: fused-silica density 2200 kg/m^3 (published material constant), used in app:glossary:L622:509 | - |
| 623 |  | observed | `000` | not run: definition: 6,000 neutrophil identity sites of chain v3 (design of the site set; the 000 is the tail of 6,000) | - |
| 624 | app:glossary:L624 | observed | `450` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 637 | app:glossary:L637 | observed | `963` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 637 | app:glossary:L637:867 | observed | `867` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 638 |  | observed | `000` | not run: definition: 6,000 identity sites of Stage M restated (the 000 is the tail of 6,000) | - |
| 638 |  | observed | `90` | not run: definition: Stage M needs at least 90 % of identity sites measured (chain rule) | - |
| 640 | app:glossary:L640 | observed | `100` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: measured: printed value found in MethylPhys_CPG_SOP_v3.md, a file the chapter names | PASS |
| 640 |  | observed | `000` | not run: definition: at least 100,000 opportunities for Stage Q (chain rule; the 000 is the tail of 100,000) | - |
| 648 |  | observed | `10` | not run: input: approximate frequency band 1e-4 to 1e-2 Hz of an electroweak-transition gravitational-wave peak from the literature (the 10 is the base of a printed power), nothing to recompute | - |
| 650 | app:glossary:L650 | observed | `60` | numeric: (k_B T/q) ln 10 at 300 K, mV/decade | PASS |
| 650 |  | observed | `300` | not run: input: temperature 300 K at which the swing bound is evaluated; used in app:glossary:L650 | - |
| 653 | app:glossary:L653 | observed | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: Delta chi2 of the matter-sector rate on SN distances | PASS |
| 655 |  | observed | `10` | not run: input: surface-code threshold of about 1e-2 per operation (Fowler2012, doi 10.1103/PhysRevA.86.032324); the 10 is the base of a printed power | - |
| 661 | app:glossary:L661 | observed | `310.15` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 662 |  | observed | `89` | not run: measured, source not named | - |
| 662 |  | observed | `105` | not run: measured, source not named | - |
| 663 |  | observed | `0.3` | not run: measured, source not named | - |
| 663 |  | observed | `-0.5` | not run: measured, source not named | - |
| 668 | app:glossary:L668 | observed | `64` | file `Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v3.md`: whole-blood technical-replicate arrays (GSE250556) | PASS |
| 668 |  | observed | `0.20` | not run: definition: read line 0.20 restated (chain rule) | - |
| 671 | app:glossary:L671 | observed | `3.7\times10^{-23}` | numeric: thermal de Broglie wavelength of 1 g at 300 K, m | PASS |
| 671 |  | observed | `300` | not run: input: temperature 300 K of the worked example; used in app:glossary:L671 | - |
| 675 | app:glossary:L675 | observed | `1.26` | file: three clocks: redshift of the peak per unit time | PASS |
| 675 | app:glossary:L675:8.9 | observed | `8.9` | numeric: lookback time to the peak of the per-time clock, Gyr | PASS |
| 681 | app:glossary:L681 | observed | `4.6\times10^{-25}` | numeric: top-quark lifetime hbar/Gamma_t, s | PASS |
| 682 |  | observed | `2.3` | not run: input: TOV limit of about 2.3 M_sun (published approximate value), nothing to recompute | - |
| 693 |  | observed | `0.20` | not run: definition: twin-test separation of 0.20 in beta (atlas entry rule) | - |
| 707 | app:glossary:L707 | observed | `4.63\times10^{113}` | numeric: Planck-cutoff vacuum energy density E_P^4/(hbar c)^3, J/m^3 | PASS |
| 707 | app:glossary:L707:1e123 | observed | `10^{123}` | numeric: rho_vac over the measured dark-energy density | PASS |
| 708 | app:glossary:L708 | observed | `0.1628` | file `CANON/GLOSSARY.md`: measured: printed value found in GLOSSARY.md, a file the chapter names | PASS |
| 714 | app:glossary:L714 | observed | `1.02` | file `docs/verification/scripts/verify_virial_papers_output.txt`: virial ratio with surface pressure, lower end (Klypin 2016) | PASS |
| 714 | app:glossary:L714:1.1 | observed | `1.1` | file `docs/verification/virial/NBODY_TRACE.md`: lowest 2T/|U| of simulated halos within the virial radius | PASS |
| 714 | app:glossary:L714:1.3 | observed | `1.3` | file `docs/verification/virial/NBODY_TRACE.md`: highest 2T/|U| of simulated halos within the virial radius | PASS |
| 714 | app:glossary:L714:1.17 | observed | `1.17` | file `docs/verification/virial/NBODY_TRACE.md`: surface-pressure corrected 2K/|W|, upper end | PASS |
| 719 | app:glossary:L719 | observed | `2.5\times10^{-18}` | numeric: weak-interaction range hbar/(M_W c), m | PASS |
| 725 | app:glossary:L725 | observed | `0.593` | numeric: mean DA white dwarf mass (Kepler et al. 2007) | PASS |
| 725 | app:glossary:L735 | observed | `2.9\times10^{-6}` | numeric: electron Yukawa y_e = sqrt2 m_e/v | PASS |
| 725 | app:glossary:L735:0.991 | observed | `0.991` | numeric: top Yukawa y_t = sqrt2 m_t/v | PASS |
| 728 | app:glossary:L728 | measured | `0.045` | file `Biological_Physics/MethylPhys/doors/data/t2_diag.csv`: within-person SD, untared, 54 y repeat arrays | PASS |
| 728 | app:glossary:L728:0.044 | measured | `0.044` | file `Biological_Physics/MethylPhys/doors/data/t2_diag.csv`: within-person SD, untared, 30 y repeat arrays | PASS |
| 729 | app:glossary:L729 | observed | `1.05` | file: LoCuSS M_WL/M_X = 1/beta_X | PASS |
| 729 | app:glossary:L729:1.45 | observed | `1.45` | heavy numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: WtG Planck-prior 1/(1-b) | PASS |
| 729 | app:glossary:L729:1.28 | observed | `1.28` | heavy numeric `docs/verification/scripts/verify_cluster_mass_satellites_output.txt`: CCCP Planck-prior 1/(1-b) | PASS |
| 729 |  | observed | `0.15` | not run: input: redshift bound z=0.15 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |
| 729 |  | observed | `0.3` | not run: input: redshift bound z=0.3 of the published cluster sample split (Smith2016LoCuSS), a sample definition | - |

## Part 8 - app:register - `docs/book/appendices/app_G_predictions_register.tex`

| line | label | status | printed | checked how | result |
|---:|---|---|---|---|---|
| 19 | app:register:L19 | observed | `126` | file `CANON/predictions_triage_2026-10-02.json`: register, cosmology and gravitation: merged entries | PASS |
| 19 | app:register:L19:133 | observed | `133` | file `CANON/predictions_triage_2026-10-02.json`: register, cosmology and gravitation: entries not listed | PASS |
| 19 | app:register:L19:357 | observed | `357` | file `CANON/predictions_triage_2026-10-02.json`: measured: printed value found in predictions_triage_2026-10-02.json, a file the chapter names | PASS |
| 19 | app:register:L19:45 | observed | `45` | file `CANON/predictions_triage_2026-10-02.json`: register counts: cosmology entries restated | PASS |
| 19 | app:register:L19:40 | observed | `40` | file `CANON/predictions_triage_2026-10-02.json`: register counts: cosmology rows listed here | PASS |
| 20 | app:register:L20:25 | observed | `25` | file `CANON/predictions_triage_2026-10-02.json`: register counts: particle-physics total | PASS |
| 21 | app:register:L21:24 | observed | `24` | file `CANON/predictions_triage_2026-10-02.json`: register counts: qubit total | PASS |
| 23 | app:register:L23:12 | observed | `12` | file `CANON/predictions_triage_2026-10-02.json`: register counts: foundations not listed | PASS |
| 23 | app:register:L23:15 | observed | `15` | file `CANON/predictions_triage_2026-10-02.json`: register counts: foundations total | PASS |
| 24 | app:register:L24 | observed | `143` | file `CANON/predictions_triage_2026-10-02.json`: register, all domains: merged entries | PASS |
| 24 | app:register:L24:157 | observed | `157` | file `CANON/predictions_triage_2026-10-02.json`: register, all domains: entries not listed | PASS |
| 24 | app:register:L24:10 | observed | `10` | file `CANON/predictions_triage_2026-10-02.json`: register counts: all kept | PASS |
| 24 | app:register:L24:57 | observed | `57` | file `CANON/predictions_triage_2026-10-02.json`: register counts: all restated | PASS |
| 24 | app:register:L24:54 | observed | `54` | file `CANON/predictions_triage_2026-10-02.json`: register counts: all compared with data | PASS |
| 24 | app:register:L24:421 | observed | `421` | file `CANON/predictions_triage_2026-10-02.json`: register counts: all entries | PASS |
| 35 | app:register:L35 | openprob | `67.161` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: same value as p2_03_theory:889 (Level2 posterior mean H0) | PASS |
| 35 | app:register:L35:67.16 | openprob | `67.16` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-001: photon-sector H0, Level 2 posterior | PASS |
| 35 | app:register:L35:72.26 | openprob | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-001: matter-sector H0 = H0 sqrt(1 + beta_m) | PASS |
| 35 |  | openprob | `001` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 35 |  | openprob | `108` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 35 |  | openprob | `110` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 35 |  | openprob | `70.0` | not run: input: GW170817 standard-siren H0 70.0 +12.0/-8.0 (Abbott2017Siren, doi:10.1038/nature24471, cited at ch:theory line 890) | - |
| 36 | app:register:L36:72.26 | openprob | `72.26` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-007: matter-sector H0 = H0 sqrt(1 + beta_m) | PASS |
| 36 | app:register:L36:0.75 | openprob | `0.75` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-007: matter-sector H0 from SH0ES, in sigma | PASS |
| 36 |  | openprob | `007` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 36 |  | openprob | `73.04` | not run: input: SH0ES H0 73.04 +/- 1.04 (Riess2022, doi:10.3847/2041-8213/ac5c5b, cited in the preface line 60) | - |
| 37 | app:register:L37:1.062 | openprob | `1.062` | numeric: COS-010: effective w0 | PASS |
| 37 | app:register:L37:0.012 | openprob | `0.012` | numeric: COS-010: effective wa | PASS |
| 37 |  | openprob | `010` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 37 |  | openprob | `087` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 38 | app:register:L38:1.062 | openprob | `1.062` | numeric: COS-013: effective w0 | PASS |
| 38 | app:register:L38:0.012 | openprob | `0.012` | numeric: COS-013: effective wa | PASS |
| 38 | app:register:L38:7.6 | openprob | `7.6` | numeric: COS-013: DESI DR2 w0 from -4/3, lower end | PASS |
| 38 | app:register:L38:10.2 | openprob | `10.2` | numeric: COS-013: DESI DR2 w0 from -4/3, upper end | PASS |
| 38 |  | openprob | `013` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 39 | app:register:L39:0.0052 | openprob | `0.0052` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: COS-016: photon-coupling bound beta_gamma, 95 % | PASS |
| 39 | app:register:L39:0.033 | openprob | `0.033` | heavy file `docs/verification/scripts/verify_beta_gamma_output.txt`: COS-016: beta_gamma bound over beta_m | PASS |
| 39 |  | openprob | `016` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 39 |  | openprob | `017` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 39 |  | openprob | `229` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 39 |  | openprob | `003` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 | app:register:L40:+0.96 | openprob | `+0.96` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: COS-018: Level 1 Delta chi2, Planck only | PASS |
| 40 | app:register:L40:+0.56 | openprob | `+0.56` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: COS-018: Level 1 Delta chi2, Planck + RSD | PASS |
| 40 | app:register:L40:+1.73 | openprob | `+1.73` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: COS-018: Level 1 Delta chi2, Planck + BAO | PASS |
| 40 | app:register:L40:+1.58 | openprob | `+1.58` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_PAIRS_FINAL.csv`: COS-018: Level 1 Delta chi2, Planck + Pantheon+ | PASS |
| 40 | app:register:L40:+0.54 | openprob | `+0.54` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-018: Level 2 Delta chi2 | PASS |
| 40 | app:register:L40:18 | openprob | `18` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-018: number of MCMC chains | PASS |
| 40 |  | openprob | `018` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `008` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `050` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `116` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `117` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `214` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 40 |  | openprob | `049` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 41 | app:register:L41:0.7 | openprob | `0.7` | heavy file `docs/verification/scripts/verify_cc_and_baryon_output.txt`: COS-046: Ob/Om over (3/16) sqrt(OL), in sigma | PASS |
| 41 | app:register:L41:+0.79 \% | openprob | `+0.79 \%` | numeric: COS-046: Lambda ratio, per cent | PASS |
| 41 |  | openprob | `046` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 41 |  | openprob | `215` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 | app:register:L42:0.864 | openprob | `0.864` | numeric: COS-058: mu at z = 0 | PASS |
| 42 | app:register:L42:0.136 | openprob | `0.136` | numeric: COS-058: deviation of mu from GR today | PASS |
| 42 | app:register:L42:0.948 | openprob | `0.948` | numeric: COS-058: mu at z = 0.5 | PASS |
| 42 | app:register:L42:0.982 | openprob | `0.982` | numeric: COS-058: mu at z = 1 | PASS |
| 42 |  | openprob | `058` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `004` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `072` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `124` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `127` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `332` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 42 |  | openprob | `0.5` | not run: input: redshift z = 0.5 at which mu is quoted | - |
| 42 |  | openprob | `0.3` | not run: input: lower end of the redshift range z ~ 0.3-1 of the growth ramp, a range named for the test, nothing to recompute | - |
| 43 | app:register:L43:0.800 | openprob | `0.800` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-062: Level 2 sigma8 | PASS |
| 43 | app:register:L43:0.1 | openprob | `0.1` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-062: Level 2 sigma8 from the joint lensing value, in sigma | PASS |
| 43 | app:register:L43:0.020 | openprob | `0.020` | numeric: COS-062: symmetrised error of the joint lensing sigma8 | PASS |
| 43 |  | openprob | `062` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 43 |  | openprob | `0.802` | not run: input: joint lensing sigma8 0.802 +0.022/-0.018 (Stolzner2025, cited in ch:virial line 237 and ch:sectortension line 51) | - |
| 44 | app:register:L44:+23.6 | openprob | `+23.6` | heavy file `docs/verification/scripts/verify_dual_sector_chapters_output.txt`: COS-075: Delta chi2 of beta on SN distances | PASS |
| 44 |  | openprob | `075` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 44 |  | openprob | `107` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 45 |  | openprob | `083` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 46 | app:register:L46:1.158 | openprob | `1.158` | numeric: COS-091: M_lens/M_dyn = 1/mu at z = 0 | PASS |
| 46 | app:register:L46:1.055 | openprob | `1.055` | numeric: COS-091: M_lens/M_dyn = 1/mu at z = 0.5 | PASS |
| 46 | app:register:L46:1.018 | openprob | `1.018` | numeric: COS-091: M_lens/M_dyn = 1/mu at z = 1 | PASS |
| 46 | app:register:L46:1.002 | openprob | `1.002` | numeric: COS-091: M_lens/M_dyn = 1/mu at z = 2 | PASS |
| 46 |  | openprob | `091` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 46 |  | openprob | `167` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 46 |  | openprob | `037` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 46 |  | openprob | `0.5` | not run: input: redshift z = 0.5 at which 1/mu is quoted | - |
| 47 | app:register:L47:+0.064 | openprob | `+0.064` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: COS-123: free-mu0 Planck + RSD median | PASS |
| 47 | app:register:L47:0.204 | openprob | `0.204` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: COS-123: free-mu0 Planck + RSD 5 % quantile | PASS |
| 47 | app:register:L47:+0.2 | openprob | `+0.2` | heavy file `Cosmological_Physics/mgcamb_validation/chains/planck_rsd_mu0_float.updated.yaml`: COS-123: upper prior edge of free mu0 | PASS |
| 47 | app:register:L47:0.10 | openprob | `0.10` | heavy file `docs/verification/scripts/verify_late_time_level2_output.txt`: COS-123: P(mu0 < -0.135) in the free-mu0 chain | PASS |
| 47 |  | openprob | `123` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 47 |  | openprob | `057` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 47 |  | openprob | `90 \%` | not run: definition: the central 90 % interval of the posterior | - |
| 47 |  | openprob | `5 \%` | not run: definition: the 5 % quantile, lower end of the central 90 % interval | - |
| 47 |  | openprob | `0.136` | not run: locked value mu0 = -0.136 restated | - |
| 48 | app:register:L48:0.8087 | openprob | `0.8087` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-131: Level 2 LambdaCDM sigma8 (Run C) | PASS |
| 48 | app:register:L48:0.7998 | openprob | `0.7998` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-131: Level 2 IAM sigma8 (Run A) | PASS |
| 48 | app:register:L48:0.009 | openprob | `0.009` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-131: Level 2 sigma8 drop | PASS |
| 48 | app:register:L48:1.1 \% | openprob | `1.1 \%` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-131: Level 2 sigma8 drop, per cent | PASS |
| 48 | app:register:L48:0.822 | openprob | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-131: Level 2 IAM S8 (Run A) | PASS |
| 48 | app:register:L48:0.832 | openprob | `0.832` | numeric: COS-131: LambdaCDM S8 (Planck 2018) | PASS |
| 48 |  | openprob | `131` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 48 |  | openprob | `012` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 48 |  | openprob | `063` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 48 |  | openprob | `064` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 48 |  | openprob | `314` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 48 |  | openprob | `061` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 49 | app:register:L49:13.62 \% | openprob | `13.62 \%` | numeric: COS-134: 1 - mu at z = 0, per cent | PASS |
| 49 | app:register:L49:5.18 \% | openprob | `5.18 \%` | numeric: COS-134: 1 - mu at z = 0.5, per cent | PASS |
| 49 |  | openprob | `134` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 49 |  | openprob | `0.5` | not run: input: redshift z = 0.5 at which the deficit is quoted | - |
| 50 | app:register:L50:0.08 \% | openprob | `0.08 \%` | numeric: COS-140: CMB lensing power lowered, per cent | PASS |
| 50 | app:register:L50:+1.8 \% | openprob | `+1.8 \%` | numeric: COS-140: E_G change at z = 0.3, per cent | PASS |
| 50 |  | openprob | `140` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 50 |  | openprob | `142` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 50 |  | openprob | `0.3` | not run: input: redshift z = 0.3 at which E_G is quoted | - |
| 51 | app:register:L51:0.2990 | openprob | `0.2990` | numeric: COS-148: growth Omega_m at z = 0.5 | PASS |
| 51 | app:register:L51:0.3 | openprob | `0.3` | numeric: COS-148: growth Omega_m from DESI DR1, in sigma | PASS |
| 51 |  | openprob | `148` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 51 |  | openprob | `0.5` | not run: input: redshift z = 0.5 at which the growth Omega_m is quoted | - |
| 51 |  | openprob | `0.2962` | not run: input: DESI DR1 full-shape plus BAO Omega_m 0.2962 +/- 0.0095 (DESI2024VII, cited in ch:virial line 300) | - |
| 51 |  | openprob | `0.0095` | not run: input: error of the DESI DR1 Omega_m 0.2962 +/- 0.0095 (DESI2024VII, cited in ch:virial line 300) | - |
| 52 |  | openprob | `159` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 53 | app:register:L53:1.05 | openprob | `1.05` | numeric: COS-172: lower end of 1/mu over 0 < z < 0.5 | PASS |
| 53 | app:register:L53:1.16 | openprob | `1.16` | numeric: COS-172: upper end of 1/mu over 0 < z < 0.5 | PASS |
| 53 |  | openprob | `172` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 54 |  | openprob | `179` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 55 | app:register:L55:1.094 | openprob | `1.094` | numeric: COS-217: 1/mu at z = 0.25 | PASS |
| 55 | app:register:L55:2.4 | openprob | `2.4` | numeric: COS-217: IAM below the WtG value, in sigma | PASS |
| 55 |  | openprob | `217` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 55 |  | openprob | `0.25` | not run: input: median redshift z ~ 0.25 of the Weighing the Giants sample | - |
| 55 |  | openprob | `1.31` | not run: measured, source not named: observed lensing-to-hydrostatic ratio 1.31 attributed to Weighing the Giants (von der Linden et al. 2014); the value is carried only by the register files and is marked untraced (PAPER_ERRATA.md LD4) | - |
| 55 |  | openprob | `0.11` | not run: measured, source not named: error 0.11 of the Weighing the Giants ratio 1.31; marked untraced (PAPER_ERRATA.md LD4) | - |
| 56 |  | openprob | `219` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 56 |  | openprob | `100` | not run: input: oscillator frequency omega_0 = 2 pi x 100 kHz of the proposed test | - |
| 57 | app:register:L57:61.45 | openprob | `61.45` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-240: Level 2b H0 | PASS |
| 57 | app:register:L57:0.42 | openprob | `0.42` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-240: Level 2b H0 error | PASS |
| 57 | app:register:L57:10.9 | openprob | `10.9` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-240: Level 2b H0 from Planck LambdaCDM, in sigma | PASS |
| 57 |  | openprob | `240` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 57 |  | openprob | `005` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 57 |  | openprob | `241` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 57 |  | openprob | `67.36` | not run: input: Planck 2018 LambdaCDM H0 67.36 (Aghanim et al. 2020, doi:10.1051/0004-6361/201833910) | - |
| 57 |  | openprob | `0.54` | not run: input: error of the Planck 2018 H0 67.36 +/- 0.54 (Aghanim et al. 2020, doi:10.1051/0004-6361/201833910) | - |
| 58 | app:register:L58:0.08 \% | openprob | `0.08 \%` | numeric: COS-260: CMB lensing power lowered, per cent | PASS |
| 58 |  | openprob | `260` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 58 |  | openprob | `0.05` | not run: calc, method not committed: lower end of the L-dependent Limber estimate 0.05-0.3 % (restates ch:dsnote line 154 and ch:latetime line 107, kept there for the same reason: no committed script or output) | - |
| 58 |  | openprob | `0.3` | not run: calc, method not committed: upper end of the L-dependent Limber estimate 0.05-0.3 % (restates ch:dsnote line 154, kept there for the same reason) | - |
| 59 |  | openprob | `265` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 59 |  | openprob | `10 \%` | not run: measured, source not named: stated accuracy of the virial 1/2 partition for the Sun; the register names no data (triage: no data named) | - |
| 60 |  | openprob | `266` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 60 |  | openprob | `20` | not run: measured, source not named: number of galaxy clusters (~20) behind the virial 1/2 partition entry; the register names no data | - |
| 60 |  | openprob | `20 \%` | not run: measured, source not named: stated accuracy (~20 %) of the virial partition for galaxy clusters; the register names no data | - |
| 61 | app:register:L61:1.085 | openprob | `1.085` | numeric: COS-272: 1/mu at z = 0.3 | PASS |
| 61 | app:register:L61:1.3 | openprob | `1.3` | numeric: COS-272: IAM below the CCCP value, in sigma | PASS |
| 61 |  | openprob | `272` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 61 |  | openprob | `50` | not run: input: CCCP sample size, 50 clusters (Hoekstra2015) | - |
| 61 |  | openprob | `0.15` | not run: input: lower end of the CCCP redshift range 0.15 < z < 0.55 (Hoekstra2015) | - |
| 61 |  | openprob | `0.3` | not run: input: median redshift z ~ 0.3 of the CCCP sample | - |
| 61 |  | openprob | `1.20` | not run: measured, source not named: observed lensing-to-hydrostatic ratio 1.20 attributed to CCCP (Hoekstra et al. 2015); the value is carried only by the register files and is marked untraced (PAPER_ERRATA.md LD4) | - |
| 61 |  | openprob | `0.12` | not run: measured, source not named: error 0.12 of the CCCP ratio 1.20; marked untraced (PAPER_ERRATA.md LD4) | - |
| 62 |  | openprob | `290` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 63 |  | openprob | `301` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 63 |  | openprob | `021` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 63 |  | openprob | `345` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 64 | app:register:L64:0.11 | observed | `0.11` | file `docs/verification/theory/IAM_LAW_CHECK.md`: COS-309: DESI full-shape mu0 (recorded value) | PASS |
| 64 |  | openprob | `309` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 64 |  | openprob | `0.136` | not run: locked value mu0 = -0.136 restated | - |
| 65 |  | openprob | `317` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 66 | app:register:L66:0.822 | openprob | `0.822` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-319: Level 2 IAM S8 (Run A) | PASS |
| 66 | app:register:L66:0.011 | openprob | `0.011` | heavy file `Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`: COS-319: Level 2 IAM S8 error (Run A) | PASS |
| 66 | app:register:L66:0.832 | openprob | `0.832` | numeric: COS-319: LambdaCDM S8 (Planck 2018) | PASS |
| 66 |  | openprob | `319` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 66 |  | openprob | `067` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 66 |  | openprob | `259` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 66 |  | openprob | `318` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 67 | app:register:L67:4.25 \% | openprob | `4.25 \%` | numeric: COS-321: f sigma8 deficit at z = 0, per cent | PASS |
| 67 | app:register:L67:1.35 \% | openprob | `1.35 \%` | numeric: COS-321: f sigma8 deficit at z = 0.5, per cent | PASS |
| 67 |  | openprob | `321` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 | app:register:L68:4.25 \% | openprob | `4.25 \%` | numeric: COS-324: f sigma8 deficit at z = 0, per cent | PASS |
| 68 | app:register:L68:3.42 \% | openprob | `3.42 \%` | numeric: COS-324: f sigma8 deficit at z = 0.1, per cent | PASS |
| 68 | app:register:L68:2.17 \% | openprob | `2.17 \%` | numeric: COS-324: f sigma8 deficit at z = 0.3, per cent | PASS |
| 68 | app:register:L68:1.35 \% | openprob | `1.35 \%` | numeric: COS-324: f sigma8 deficit at z = 0.5, per cent | PASS |
| 68 | app:register:L68:0.41 \% | openprob | `0.41 \%` | numeric: COS-324: f sigma8 deficit at z = 1, per cent | PASS |
| 68 | app:register:L68:0.04 \% | openprob | `0.04 \%` | numeric: COS-324: f sigma8 deficit at z = 2, per cent | PASS |
| 68 |  | openprob | `324` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `036` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `069` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `070` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `325` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `331` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 68 |  | openprob | `0.1` | not run: input: redshift z = 0.1 at which the f sigma8 deficit is quoted | - |
| 68 |  | openprob | `0.3` | not run: input: redshift z = 0.3 at which the f sigma8 deficit is quoted | - |
| 68 |  | openprob | `0.5` | not run: input: redshift z = 0.5 at which the f sigma8 deficit is quoted | - |
| 69 | app:register:L69:13.62 \% | openprob | `13.62 \%` | numeric: COS-336: 1 - mu at z = 0, per cent | PASS |
| 69 | app:register:L69:1.78 \% | openprob | `1.78 \%` | numeric: COS-336: 1 - mu at z = 1, per cent | PASS |
| 69 | app:register:L69:0.23 \% | openprob | `0.23 \%` | numeric: COS-336: 1 - mu at z = 2, per cent | PASS |
| 69 | app:register:L69:0.04 \% | openprob | `0.04 \%` | numeric: COS-336: 1 - mu at z = 3, per cent | PASS |
| 69 | app:register:L69:0.04 \%:b20:fs8 | openprob | `0.04 \%` | numeric: COS-336: f sigma8 drop at z = 2, per cent | PASS |
| 69 |  | openprob | `336` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 70 | app:register:L70:0.78 \% | openprob | `0.78 \%` | numeric: COS-338: linear growth factor D lowered today, per cent | PASS |
| 70 | app:register:L70:+0.68 | openprob | `+0.68` | numeric: COS-338: Press-Schechter count change at 1e9 solar masses | PASS |
| 70 | app:register:L70:+0.73 \% | openprob | `+0.73 \%` | numeric: COS-338: Press-Schechter count change at 1e7 solar masses | PASS |
| 70 |  | openprob | `338` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 70 |  | openprob | `10` | not run: input: base 10 of the satellite-mass range 10^7 to 10^9 solar masses, the mass range of the satellites chapter (ch:satellites line 114) | - |
| 71 |  | openprob | `342` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 71 |  | openprob | `343` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 71 |  | openprob | `294` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 72 |  | openprob | `351` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 72 |  | openprob | `156` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 72 |  | openprob | `1.7` | not run: input: lower end of the redshift range z ~ 1.7-2.4 where the competing Lambda_s CDM model shifts BAO; a property of another model, nothing to recompute | - |
| 72 |  | openprob | `2.4` | not run: input: upper end of the redshift range z ~ 1.7-2.4 where the competing Lambda_s CDM model shifts BAO; a property of another model, nothing to recompute | - |
| 73 | app:register:L73:1.55 \% | openprob | `1.55 \%` | numeric: COS-357: P_IAM/P_LCDM lowered at z = 0, per cent | PASS |
| 73 | app:register:L73:0.78 \% | openprob | `0.78 \%` | numeric: COS-357: D lowered at z = 0, per cent | PASS |
| 73 |  | openprob | `357` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 73 |  | openprob | `056` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 73 |  | openprob | `122` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 74 | app:register:L74:1.03 | openprob | `1.03` | numeric: COS-365: ISW amplitude ratio A_ISW | PASS |
| 74 |  | openprob | `365` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
| 74 |  | openprob | `364` | not run: definition: register entry identifier (COS-/PAR- number), not a quantity | - |
