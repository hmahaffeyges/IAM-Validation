# DEV-TISSUE-LUNG-02 — Met-A on early lung tumors against the same patient's normal lung (development; NOT SUITABLE at 12 pairs, 2026-10-12; tumors unread)

**Why.** DEV-TISSUE-LUNG-01 (GSE94785) read NO FINGERPRINT: most tumors leaned up, by less than its bar could see (LESSONS T15). This is a second,
independent lung set, never read by any version of the instrument, tested with the T15 design.

**Data.** GSE83842 (Kajiura, Masuda et al., Tokushima University; PMID 27926516): Illumina 450K, raw IDATs, 12 patients with early lung
adenocarcinoma (stage 1a 10, 1b 2), smokers and never-smokers, each tumor and the same patient's normal lung. **Layout** (`data/DEV_TISSUE_LUNG_02/
samples_lung_02.py`, output `layout_lung_02_output.txt`): all 12 tumors share a slide with their own normal.

**Reading:** as DEV-TISSUE-LUNG-01 (both lung epithelia agree with every panel cell at the blind sites). **Inputs:** `fetch_lung_02.py` (tumor
arrays refused before any request until `sealed_lung_02.json` is committed), `calib_lung_02.py`.

**Before any seal (LESSONS T15):** normal spread first; plants down to 1 % and the size lung 1 showed; the every-pair rule beside the count rule.
With 12 pairs, power is the first question: if the synthetic test says 12 pairs cannot decide a change the size lung 1 showed, the test is not run.

**Synthetic tests on the 12 real normal lungs, 2026-10-12** (`data/DEV_TISSUE_LUNG_02/lung2_synthetic.py`, outputs `lung2_synthetic_output.csv`,
`lung2_direction_output.csv`, `lung2_synthetic_summary.txt`). Normal lungs differ between people far more here than in lung 1 (stand-in spread SD
0.1391, against 0.0860 and colon 0.0502), so the pair bar sits at 1.2928. Composition, power and direction PASSED; **the planted 5 % change FAILED**
(above the bar in 0.400 of pairs; 0.80 required). The chosen rule would see a change the size of lung 1's lean in none of the draws.

**Decision, by the rule written before running: not suitable; not sealed; the 12 tumors stay unread.** A lung test needs many more patients, or
normals that differ less between people.
**Not a milestone:** not suitable (development).

**Test data:** GSE83842
