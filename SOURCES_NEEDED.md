# SOURCES_NEEDED

Measured or observed values the book prints whose source could not be found: not in a reference the chapter cites, not in any
file of this repository. Each stays "not run" in `VERIFY_BOOK_INVENTORY.md` with the reason "measured, source not named" until a
source is named. No source was guessed.

Entries: 13

| chapter | file:line | printed | what is needed |
|---|---|---:|---|
| ch:iams_law | `docs/book/part1/p1_02_iams_law.tex:536` | `0.13` | IAM vs LambdaCDM CMB TT spectra differ by less than 0.13 % at l > 30, from tests/iam_camb_full_boltzmann.py; the chapter says the spectra are not stored. Searched docs/verification, tests/, camb_validation/ for a committed output of that comparison: none. A committed output (or the two Cl tables) is needed to check it. |
| ch:iams_law | `docs/book/part1/p1_02_iams_law.tex:714` | `4.7` | Upper error of the GW170817 siren H0 68.9 (+4.7 -4.6) attributed to Hotokezaka et al. 2019 (doi 10.1038/s41550-019-0820-1). No committed file holds the errors (verify_iams_law_derivations_output.txt has only the central 68.9). The errors, and which posterior of the paper they belong to, need to be confirmed against the paper and recorded in a committed file. |
| ch:iams_law | `docs/book/part1/p1_02_iams_law.tex:714` | `4.6` | Lower error of the same Hotokezaka et al. 2019 siren H0; same search and same open point as the upper error. |
| ch:virial_tests | `docs/book/part2/p2_02b_virial_tests.tex:31` | `8` | Lower end of 'errors 8--13 % per bin' on the six DESI DR1 f sigma8 bins (cite DESI2024V). The committed DESI DR1 numbers do not give 8-13 %: the ShapeFit+BAO ratio errors of verify_shapefit_chi2.py (symmetrised, relative to the fiducial) are 19, 13, 10.1, 9.2, 8.7, 12 % (BGS..QSO), and relative to the measured value 22.6, 11.2, 9.7, 9.2, 9.2, 10.3 %; verify_sector_tension.py's values give 9.8-25 %. Searched docs/verification/scripts (shapefit, sector_tension outputs), camb_validation/likelihood_rsd.py. The source of 8-13 % (perhaps the direct full-modelling f sigma8 of DESI 2024 V) needs to be named and recorded. |
| ch:virial_tests | `docs/book/part2/p2_02b_virial_tests.tex:31` | `13` | Upper end of the same 'errors 8--13 % per bin'; same search. Every committed version of the BGS bin (z = 0.295) has a larger error (19-25 %). |
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:34` | `0.1` | Lower end of the hydrostatic-bias estimates 'b from about 0.1 to about 0.4 depending on the method', cited to Nagai2007ICM, Rasia2012, Biffi2016. No repository file holds the values. |
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:34` | `0.4` | Upper end of the same range (Nagai2007ICM, Rasia2012, Biffi2016). No repository file holds it. |
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:37` | `0.15` | Simulation range b about 0.1-0.15 (Lau2009, Nelson2014); also restated at line 186. No repository file holds it; the committed cluster script does not list it. |
| ch:blackholes | `docs/book/part2/p2_01_blackholes.tex:319` | `1.4` | typical neutron-star mass 1.4 M_sun used as the gauge reference (A = 1); no citation in the chapter; searched the repo (only docs/book/figscripts/fig_p2_star_gauge.py restates 1.4 without a source) |
| ch:blackholes | `docs/book/part2/p2_01_blackholes.tex:335` | `1.4` | same typical neutron-star mass 1.4 M_sun in the fig:stargauge caption; no citation; same search as line 319 |
| ch:quantumrecords | `docs/book/part2/p2_14_quantum_records.tex:203` | `0.02` | largest shift of the bottom-up n_eff at z = 4 when the halo overdensity definition in K is changed; the chapter attributes it to the method of verify_bottom_up_exponent.py, but the committed output (verify_bottom_up_exponent_output.txt) has only the default definition. Searched docs/verification/, PAPER_ERRATA.md, results/. |
| ch:quantumrecords | `docs/book/part2/p2_14_quantum_records.tex:203` | `0.09` | largest shift of n_eff at z = 2 from the halo definition; same sensitivity run as 0.02, output not committed. Searched as above. |
| ch:measurement | `docs/book/part5/p5_04_measurement.tex:224` | `10` | 'about 10^{11} galaxies' summed over with 13.8 Gyr; an order-of-magnitude galaxy count in the observable universe given without a citation (the same count appears in ch:gravdec line 247). No file in the repository holds it; a published estimate would need a reference in iam.bib. |
