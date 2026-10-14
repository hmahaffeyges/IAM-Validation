# Atlas v2.1 readiness L8: the rules for the five open conditions (committed 2026-10-14 BEFORE any of the arrays below is read for them)

Decisions 1 and 10 say their cuts were "fixed now" in the plan, but no number was ever written down. They are fixed here, before reading,
and the methods will say they were fixed on 2026-10-14 rather than when the plan was written. Purity is measured with a published standard
tool, EpiDISH 2.26.0 (Zheng et al. 2018, robust partial correlations, RPC), not with our own atlas, whose solver stays withheld until
commissioned.

**D1, GSE224807 CD15+ fractions as neutrophils.** Enter only if (i) v2's sample-level twin test (`10_twin_test_sample_level.py` logic, run10 in
`v21_twin_terms_box.py`, thresholds `twin_family_thresholds_v1.json`) between the GSE224807 neutrophil arrays and the Salas2018 + Salas2022
neutrophil arrays reads TWIN (r > twin_r and fewer than min_separating_loci non-overlapping sites), and (ii) per array, the EpiDISH
centBloodSub eosinophil fraction is below **0.10** (CD15 selects neutrophils and eosinophils; healthy blood granulocytes are about 2-8 %
eosinophils, so 0.10 admits a normal preparation and refuses an eosinophil-rich one). Arrays at or above 0.10 leave; at least 2 must remain.
If (i) fails, the set is research and none enters.

**D7, GSE306226 astrocyte fraction.** As written in the decision: kept only if its mean profile is closer to Tian astrocytes than to any other
cell, at cross-laboratory r >= 0.986, using the twin job's own profiles and correlation (run09: profiles, and the array-to-WGBS linear map for
cross-platform pairs). Otherwise research.

**D10, GSE146552 ovary and fallopian tube epithelium (scraping).** Per array, EpiDISH centEpiFibIC fibroblast + immune fraction below **0.20**
(at least 80 % epithelial). Arrays at or above 0.20 leave; a cell enters only if at least 2 of its arrays remain (v2's n >= 2 rule). The one
GSE146552 build array called male by the identity check is reported; by L5 rule c sex alone excludes nothing.

**D13, D14 (held-out sets GSE132547, GSE131989).** They do not change the fit. They are carried to the commissioning gate (L15) as
preconditions for scoring those two sets, and are recorded as deferred there, not dropped.

Code: `box/v21_l8_box.py` (twin test, profiles, export of the 904 EpiDISH sites) and `l8_epidish_v21.R` (the RPC fractions on the laptop).

## Addendum, 2026-10-14 (committed before any GSE137716 array is read for purity)

Decision 10 covers GSE137716 (lung bronchus epithelium, "enriched, not sorted"; 12 conditional build arrays) as well as GSE146552. The rule
above named only GSE146552, and the first outcome record said GSE137716 was unaffected; that was wrong, written without checking the build
list's reason column. GSE137716 gets the identical rule: EpiDISH 2.26.0 RPC, centEpiFibIC, fibroblast + immune below 0.20 per array, the
cell's GSE137716 arrays enter only if at least 2 pass. Export by `box/v21_l8b_box.py`; shares by `l8_epidish_v21.R` unchanged.
