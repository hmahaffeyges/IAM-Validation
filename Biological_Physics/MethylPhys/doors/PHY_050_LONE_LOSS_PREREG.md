# PHY-050, second test: the lone-loss rate against the M number, on a laboratory and tissue not used to find it (sealed before reading)

**The instrument's job:** IAM-A's counter, read by the one command, gives each sample's lone-loss rate (the chance that a CpG is a lone
lost mark between two held calls, with the counter's qualifying-rule bias removed; Stage Q2 `held_err_length`) and its loose share.

**Why this test (found while exploring, stated plainly).** PHY-050 (not met) recorded the lone-loss rate beside its verdict: its intercept
on 153 Loyfer samples was 0.04558 (0.04465 to 0.04651), equal to 1/(1+M) = 0.04558. On all 171 spent counts from four laboratories, with each
library's molecule-end damage added to the fit, the intercept read 0.04566 +/- 0.00052. Both were seen after the PHY-050 verdict, so they are
a lead. This test asks it of data on which the lone-loss rate has never been computed.

**Reference:** neither IAM-Atlas v2.1 nor the IAM Sequencing Atlas is used: IAM-A's prediction comes from the M number (canon) and
each sample's own loose share, with the committed slope.

**Data:** the six tissue samples labeled normal in GSE86340 (Johns Hopkins; three liver, three lung; `data/PHY_050_PERCALL/lone_loss_jhu_samples.csv`),
whose merged molecule files are already in the build bucket; each counted by `data/PHY_050_PERCALL/lone_loss_jhu.sh`, one sample first as a canary.

**Prediction (written before any of the six is counted):** each sample's lone-loss rate equals 0.04558 + 0.0284 x its loose share, where
0.04558 is 1/(1+M) (M 20.9405, canon) and 0.0284 is the lone-loss slope committed in `data/PHY_050_PERCALL/percall_summary.txt` (Loyfer).

**Bars, written before any sample is counted:** 1. at least 5 of the 6 samples read within 10 % of their own predicted lone-loss rate.
2. the median of observed over predicted across the 6 samples lies between 0.97 and 1.03.
- **Bar 1:** at least 5 of the 6 samples within 10 % of their own prediction.
- **Bar 2:** the median of observed over predicted between 0.97 and 1.03.

**Reported, not scored:** each sample's molecule-end damage (held error near the molecule ends minus farther in), and the six tumors of the
same people, counted afterwards.

**Known risk, stated before reading:** this laboratory read high on IAM-A before (its healthy tissues 1.02 to 1.18), and its loose shares
(about 0.55 to 0.69) lie above most of the Loyfer samples, so the prediction leans on the committed slope. A miss above with large end
damage will be read as the laboratory's copier; a miss with no end damage as against the lead.

**Sources:** GSE86340 (GEO, public); canon `CANON/iam_canon.json`; `data/PHY_050_PERCALL/percall_summary.txt`.

**Prior reading:** GSE86340's tissues were read before by the full chain (`CANON/data_spent.csv`; DEV-ATLAS-V21-01); their lone-loss rate was
never computed.
