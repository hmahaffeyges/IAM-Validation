# SOURCES_NEEDED

Measured or observed values the book prints whose source could not be found: not in a reference the chapter cites, not in any
file of this repository. Each stays "not run" in `VERIFY_BOOK_INVENTORY.md` with the reason "measured, source not named" until a
source is named. No source was guessed.

Entries: 8

| chapter | file:line | printed | what is needed |
|---|---|---:|---|
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:34` | `0.1` | Lower end of the hydrostatic-bias estimates 'b from about 0.1 to about 0.4 depending on the method', cited to Nagai2007ICM, Rasia2012, Biffi2016. No repository file holds the values. |
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:34` | `0.4` | Upper end of the same range (Nagai2007ICM, Rasia2012, Biffi2016). No repository file holds it. |
| ch:lensdyn | `docs/book/part2/p2_17_lensing_dynamics.tex:37` | `0.15` | Simulation range b about 0.1-0.15 (Lau2009, Nelson2014); also restated at line 186. No repository file holds it; the committed cluster script does not list it. |
| ch:blackholes | `docs/book/part2/p2_01_blackholes.tex:319` | `1.4` | typical neutron-star mass 1.4 M_sun used as the gauge reference (A = 1); no citation in the chapter; searched the repo (only docs/book/figscripts/fig_p2_star_gauge.py restates 1.4 without a source) |
| ch:blackholes | `docs/book/part2/p2_01_blackholes.tex:335` | `1.4` | same typical neutron-star mass 1.4 M_sun in the fig:stargauge caption; no citation; same search as line 319 |
| ch:quantumrecords | `docs/book/part2/p2_14_quantum_records.tex:203` | `0.02` | largest shift of the bottom-up n_eff at z = 4 when the halo overdensity definition in K is changed; the chapter attributes it to the method of verify_bottom_up_exponent.py, but the committed output (verify_bottom_up_exponent_output.txt) has only the default definition. Searched docs/verification/, PAPER_ERRATA.md, results/. |
| ch:quantumrecords | `docs/book/part2/p2_14_quantum_records.tex:203` | `0.09` | largest shift of n_eff at z = 2 from the halo definition; same sensitivity run as 0.02, output not committed. Searched as above. |
| ch:measurement | `docs/book/part5/p5_04_measurement.tex:224` | `10` | 'about 10^{11} galaxies' summed over with 13.8 Gyr; an order-of-magnitude galaxy count in the observable universe given without a citation (the same count appears in ch:gravdec line 247). No file in the repository holds it; a published estimate would need a reference in iam.bib. |
