# DEV-GROUPA-COMMISSION-01: commissioning Group-A on whole blood (sealed 2026-10-20, before any array of GSE318173 is opened)

**The instrument's job:** two groups of healthy people read as no different after the tare, and a group shift planted into healthy arrays is found.

**What Group-A is.** The chain's whole-blood Met-A reading (run_sample.py, unchanged), tared against healthy arrays run in the same series (same
slide, else same series; the C-score band test's tare, checked word for word), then two groups compared by the difference of their means against
two standard errors (Welch). Scope if commissioned: whole blood, EPIC v1, groups compared within one laboratory series.

**Test data:** GSE318173 (the 278 healthy controls in GROUPA_COMMISSIONING/groupa_samples.csv), read once by this test and spent by it (CANON/data_spent.csv); line added after scoring, 2026-10-20, naming the set the seal already named

**Synthetic test:** GROUPA_COMMISSIONING/read_groupa.py, GROUPA_COMMISSIONING/rehearsal_groupa_summary.txt

**Dress rehearsal (the synthetic test, through the chain):** the same workflow, read_groupa.py, run_sample.py and score_groupa.py on the spent
healthy sets GSE315366 and GSE184269 (rehearsal_samples.csv; GitHub run 39498047678; readings in rehearsal_groupa_rows_idat.csv and
rehearsal_groupa_rows_values.csv): 46 of the 47 arrays Stage 0 admitted read (12 of 59 quarantined, two slides; controls in
rehearsal_diag_ctrl.csv); women vs men +0.12 % (0.12 SE); planted 1 % loss +2.07 % (2.9 SE); the GSE184269 readings equal the C-score band
test's on every array both read.

**Prior reading:** the synthetic test drew from tared readings of GSE315366 and GSE184269 (spent); the smoke test re-read the spent example-report array GSE235717 GSM7508429; GSE110554 and GSE167998 are the chain's references, read when it was built, not test data; GSE318173 is declared in CANON/data_spent.csv for this test alone and no array of it has been read.

**Reference sets:** GSE110554 GSE167998 (the chain's blood composition references); GSE318173 shares no contributor with either.

**Data.** GSE318173 (GEO; whole blood, EPIC v1, raw intensity files): the 278 healthy controls only, listed with their slide, sex and half in
data/GROUPA_COMMISSIONING/groupa_samples.csv. The 281 Parkinson's cases and the processed-values file are not opened. Planted half: within each
slide, the controls in slide-position order at odd ranks (143 planted, 135 not; the 8 slides with three controls put two in the planted half).

**Bars (scored once by data/GROUPA_COMMISSIONING/score_groupa.py; script hashes in sealed_groupa_01.json):**
1. **The instrument reads healthy whole blood:** Stage 0 admits at least half of the 278, and at least 90 % of the arrays it admits receive a
   tared reading. Arrays Stage 0 quarantines are reported with their reason and slide, not counted against the reading: the dress rehearsal
   showed its control-probe gate (limits marked provisional in stage_0_intake.py) quarantining two whole slides of one healthy series whose
   hybridization controls read high below low, arrays the C-score band test read normally from their calibrated values. Setting those limits
   from healthy arrays is its own step (PROC-STAGE0-04); the chain is not changed during this test.
2. **Sex null:** healthy women against healthy men, tared readings: the difference is under two standard errors and under 2 %.
3. **A planted shift is found:** a 1 % loss of the neutrophils' pattern (the chain's own rule, beta + f_neu x 0.01 x (0.5 - mu_NEU) at the
   neutrophil sites) planted in the saved values of the planted half; both halves read by the same values route and tared against the unplanted
   half: the planted half reads above the unplanted half by more than two standard errors.

**Reported, not scored:** the untared difference between the sexes; the spread of slide and amplification-plate medians before and after the tare
(the tare centers each slide by construction, so a tared slide or plate comparison is not a test); the IAM-Atlas v2.1 cell shares (IAM-Deconvolver)
of women and men; the C-Score's tared readings against its commissioned upper limit 1.152; and the C-Score of the planted half: the planted loss is scattered, so Met-A should see it and the C-Score, which flags clustered loss, should not.

**Where it runs:** GitHub's free machines (.github/workflows/groupa.yml, mode read), from the commit that carries this seal.
