# PHY-050, third test: the lone-loss rate against the M number, on healthy sorted samples never counted before (sealed before reading)

**The instrument's job:** IAM-A's counter, read by the one command, gives each sample's lone-loss rate (the chance that a CpG is a lone lost
mark between two held calls, with the counter's qualifying-rule bias removed; Stage Q2 `held_err_length`) and its loose share.

**Why this test.** The second test (PHY_050_LONE_LOSS_OUTCOME.md, not met) was read on tissue beside tumors in cancer patients, a caveat
recorded after its verdict; its record says the prediction must be asked of truly healthy samples before it counts for or against anything.
This is that test. The prediction and bars are the second test's, unchanged.

**Reference:** neither IAM-Atlas v2.1 nor the IAM Sequencing Atlas is used: IAM-A's prediction comes from the M number (canon) and each
sample's own loose share, with the committed slope.

**Data:** the 54 healthy sorted samples of GSE186458 (Loyfer 2023) that were not among the 153 counted for PHY-050 and the line
(`data/PHY_050_PERCALL/lone_loss_loyfer_unseen_samples.csv`). The 153 were chosen by rule (quality flag passed, cell type in the atlas roster,
at most the first three per cell type), so these are the samples beyond three per cell type, cell types outside the roster, and any whose
quality flag failed; all are counted and scored as they come. The 46 whole-blood and plasma mixtures of the series are not
in this test. Each sample is counted by `data/PHY_050_PERCALL/lone_loss_loyfer_count.py` (the one command, `run_sample.py --geo-pat GSM
--dev-count-only`) on GitHub's free machines, one sample first as a canary.

**Prediction (written before any of the 54 is counted):** each sample's lone-loss rate equals 0.04558 + 0.0284 x its loose share, where
0.04558 is 1/(1+M) (M 20.9405, canon) and 0.0284 is the lone-loss slope committed in `data/PHY_050_PERCALL/percall_summary.txt`.

**Bars, written before any sample is counted:**
- **Bar 1:** at least 83 percent of the samples read within 10 percent of their own prediction (the second test's 5 of 6).
- **Bar 2:** the median of observed over predicted between 0.97 and 1.03.
A sample whose count does not come back is listed as not read and does not enter the bars. Scored by `data/PHY_050_PERCALL/lone_loss_loyfer_score.py`.

**Reported, not scored:** each sample's molecule-end damage (held error near the molecule ends minus farther in), and its cell and tissue.

**Known risks, stated before reading:** these come from the same sequencing laboratory as the 153 on which the lead was seen, so a meeting
of the bars is not yet a second laboratory; the slope was measured on the 153, so samples with loose shares far outside theirs lean on it;
seven are neurons, whose marks outside CpGs the counter does not count.

**Sources:** GSE186458 (GEO, public; sample sources Had-HU 34, KI 8, Lonza 7, OHSU 2, UA 2, Sigma 1); canon `CANON/iam_canon.json`;
`data/PHY_050_PERCALL/percall_summary.txt`.

**Prior reading:** GSE186458 was read before (`CANON/data_spent.csv`; DEV-CHAIN-Q2-01, DEV-METAA-HEALTHY-01); none of these 54 samples'
lone-loss rate has ever been computed.
