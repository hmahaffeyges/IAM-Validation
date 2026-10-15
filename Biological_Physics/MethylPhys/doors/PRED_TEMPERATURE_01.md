# PRED-TEMPERATURE-01: the temperature law (PHY-057), numbers sealed 2026-10-15 before any data is searched

**In plain words.** If IAM is right that a held DNA mark is a balance of energy against temperature, then a colder body holds its marks
more tightly by an exact amount, and a hotter body less tightly, with nothing adjusted. A hibernating animal's tagging mistakes should
drop by about a third when it cools to 5 C; a bird's, at 41 C, should sit slightly above ours.

**The rule.** ln[(1 - eps)/eps] = E/kT with E fixed, so the log-odds scale as 1/T. Basis: the measured 37 C line (firm 0.0212, loose 0.0497,
island 0.0124, development/loose_end_2026-10-15/loose_end_line.json). Code: `data/PRED_TEMPERATURE_01/temperature_law.py`; the numbers it
wrote are `data/PRED_TEMPERATURE_01/temperature_law.json` and are the prediction.

**The cleanest test: one species at two temperatures** (the within-species log-odds ratio is fixed by the two temperatures alone: 1.0343
for 18 C against 28 C; 1.115 for torpor at 5 C against active at 37 C). Across species the enzymes differ, so cross-species numbers are
read as a guide, not a test.

**What a miss means.** If the error does not move with temperature, or moves by a clearly different amount on the same species, the
Boltzmann premise does not hold for held marks there, and the gauge's use of kT has to be rethought.

**Data:** none searched yet. Salmon (PROC-SALMON-01) and every other fish set already scored cannot be the test.

**Synthetic test:** PRED_TEMPERATURE_01/temperature_synthetic.py, PRED_TEMPERATURE_01/temperature_synthetic.txt

The measurement synthetic (molecules planted at two temperatures, counted with the committed PROC-CHANNEL-01 counter) ran twice on
2026-10-15. Run 1 read NOT MET (kept as temperature_synthetic_run1_NOTMET.txt): converting counts back to an error rate with the formula
eps(1 - eps)^2 read the 5 C against 37 C ratio low (1.105 against 1.115) even at 200,000 molecules, a systematic bias in the conversion, not
in the law. Run 2 converts through the counter's own response curve, measured on planted molecules with its own seed, and read MET: at
200,000 molecules per temperature the sealed ratios are recovered within 0.51 percent (18 C against 28 C) and 0.31 percent (5 C against
37 C), told apart in every repeat. At 2,000 molecules the 18 C against 28 C pair is not reliably told apart (4 of 5), so a real test needs
at least about 20,000 qualifying molecules per temperature. The predictions themselves did not change.

The formula self-check (temperature_law_synthetic.txt) also read MET: at 37 C the rule returns the measured values exactly, and the error rises with temperature.

What it would take to succeed: single-molecule methylation of one species read at two body or water temperatures, with the log-odds ratio landing on the sealed value; finish line, not a promise.

# NO CHAIN: a prediction note; no chain stage reads temperature
