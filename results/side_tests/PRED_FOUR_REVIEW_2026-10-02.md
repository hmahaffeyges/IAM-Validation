# Four dated predictions (16 March 2026): review for the book, 2026-10-02

Read: the four scripts in `tests/` (computation sections in full: age 166 of 309 lines, axis of evil 226 of 436, small-scale 227 of 456;
plotting code not read line by line), `results/side_tests/PROC_CUSPCORE_01_*`, register entries COS-146, 147, 198, 368, 369, PAR-019.
Recomputed with the scripts' own background (H0 67.4, Omega_m 0.3153).

| prediction | verdict for the book | why |
|---|---|---|
| Globular-cluster age excess (COS-146) | listed as **not a test** | The "observed excess" is the measured age minus the LCDM time since an *assumed* z_form = 3.5 (script comment: "Use z=3.5 as representative midpoint"). With z_form free, 13.39 +/- 0.25 Gyr means formation at t = 0.40 Gyr, z = 11.3 (7.9 to 22.8): no excess. The script's own formula, t_act = T0 (1 - E(a_form)), gives 13.78 Gyr for any formation above z = 8, i.e. 1.6 sigma above the measured age. If local clocks ran at the rate of E(a), neutron decay would stop during nucleosynthesis (E ~ 0); n/p frozen near 1/6 would give helium near 0.29 by mass against the measured 0.245 (estimate, not run). |
| Hemispherical power asymmetry / axis of evil (COS-147) | listed as **not a test** | 0.0628 = (1.134 - 1)/(1.134 + 1) is an asymmetry of the late-ISW amplitude alone; the observed 0.066 +/- 0.021 is a dipole modulation of the whole temperature map, of which late ISW is a small part (the script's own caveat 1). Planck 2015 XVI (A&A 594, A16) finds the asymmetry persists to l ~ 600, where late ISW contributes nothing. The hemisphere split mu_+ = 2 mu - 1, mu_- = 1 is not derived, no direction is predicted, and the data predate the script. The script cites the number to Planck 2018 A6 (parameters); the dipole-modulation results are in the isotropy-and-statistics papers. |
| Cusp-core r_core ~ sigma^2 (COS-368, PAR-019) | listed as **rejected** | Pre-registered test PROC-CUSPCORE-01 on the published LITTLE THINGS table (Oh et al. 2015, 25 galaxies): slope 0.71 (95 % CI -0.18 to 1.56) against 2; scatter 0.41 dex against <= 0.15. The scripts' seven "observed" points are typed in and do not match the published tables; the "constant ~130x" came from them. |
| Unified small-scale structure (COS-369) | **not listed** | Its two numbers do not stand: 13.6 % is 1 - mu(1), not growth suppression (D falls 0.78 %, f sigma8 4.25 % today); the sigma^2 core law is rejected (above). The black-hole threshold for satellites is qualitative with M_min open, and the dispersion floor is rejected by the census (25 of 54 satellites below 4 km/s). Missing satellites has its own chapter. |

Book: Chapter "Falsifiable predictions", section "Already tested" (three rows) and Appendix G (overrides in
`docs/book/figscripts/app_G_overrides.json`). The four scripts carry a header pointing here; their dated content is unchanged.
