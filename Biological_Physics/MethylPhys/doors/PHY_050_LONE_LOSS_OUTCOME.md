# PHY-050, second test: the lone-loss rate against the M number on Johns Hopkins tissue: outcome NOT MET

Sealed in `PHY_050_LONE_LOSS_PREREG.md` (5e5a7230) before any sample was counted; scorer committed (c9802ff0) before five of the six were
read (the canary, GSM2103008, was opened to check the job). Each sample counted by the one command (`data/PHY_050_PERCALL/lone_loss_jhu.sh`);
scored by `data/PHY_050_PERCALL/lone_loss_score.py` (`data/PHY_050_PERCALL/lone_loss_summary.txt`; bundles in `lone_loss_jhu_out/`).

**Not a milestone:** the sealed prediction was not met.

| Bar | Result | Met |
|---|---|---|
| 1. at least 5 of the 6 samples within 10 % of their own prediction | 0 of 6 | no |
| 2. the median of observed over predicted between 0.97 and 1.03 | 1.3380 | no |

**In plain words.** The six healthy liver and lung tissues lose lone marks 24 to 45 % more often than the M-based prediction, every one of
them. Their molecule-end damage is small (0.0002 to 0.0016), so by the reading fixed before the counts, the miss counts against the lead
that the lone-loss rate is set by 1/(1+M); it is not explained by this laboratory's end damage.

**What it does not settle (stated before reading as known risks):** these are bulk tissues at loose shares of 0.54 to 0.69, far above the
Loyfer samples the slope came from, and this laboratory read high on IAM-A before; neither was separated here. The lead now stands as:
found on sorted cells and Loyfer's tissues (171 samples, four laboratories), not confirmed on an independent laboratory's tissues.

**What it would take to succeed.** A prediction for the healthy middle derived from physics and sealed before reading that holds on
healthy samples from at least two laboratories not used to find it.

**Prior reading:** GSE86340's tissues were read before by the full chain (`CANON/data_spent.csv`); their lone-loss rate was first computed here.
