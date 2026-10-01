# PROC-CUSPCORE-01 — outcome (2026-10-01; pre-registration sha 96efa7e206519a95, unchanged)

Data: Oh et al. 2015, LITTLE THINGS Table 2, parsed from the published PDF (oh2015_table2_parsed.csv); 25 of 26 galaxies used
(Haro 36 excluded by the pre-registered rule: R_C 8.40 ± 12.17). σ = V_max/√2.

| test | result | |
|---|---|---|
| P1 slope of log R_C on log σ, 95 % CI contains 2.0 | 0.71 (bootstrap 95 % CI −0.18 to 1.56); bisector 1.57; r² 0.13 | **FAIL** |
| P2 same σ → same core: scatter ≤ 0.15 dex | 0.41 dex (measurement error ~0.08 dex) | **FAIL** |
| normalisation r_obs / r_IAM | median 206, range 11–1412 (not constant) | — |

**Why the dated figure looked right.** The script's seven "observed" points are typed in, credited to Oh 2015 and de Blok 2008, and do not match
the published tables: Oh 2015 gives DDO 154 R_C = 0.95 ± 0.03 kpc (script 0.35), DDO 168 2.81 ± 0.83 (script 0.42), NGC 2366 1.21 ± 0.04
(script 0.80); NGC 3741, IC 2574 are not in Oh 2015 Table 2; NGC 2976 and NGC 7793 are not core-radius entries in de Blok 2008 as quoted.
Their σ values (22–90 km/s) are also not the published V_max/√2 (DDO 154: 33.8). The "constant ~130× ratio" came from those values.

**For the book.** r_core ∝ σ² is a dated hypothesis tested against the published LITTLE THINGS core radii and not supported: the slope is
0.7 (CI −0.2 to 1.6) and same-σ galaxies differ in core radius by a factor ~2.6 rms. The mechanism (cores set by a central black hole) is
also not testable on these dwarfs, which have no measured black holes. The hand-entered data in the prediction script must be removed or
replaced with the published values; the repo figure IAM_cusp_core_sigma2_PREDICTION_Mar2026 should carry the outcome.
