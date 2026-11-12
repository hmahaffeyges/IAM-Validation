# PHY-050, third test: the lone-loss rate against the M number on 54 healthy sorted samples: NOT MET

Sealed in `PHY_050_LONE_LOSS_LOYFER_PREREG.md` (commit 11a03096) before any of the 54 was counted; counted by the one command
(`run_sample.py --geo-pat --dev-count-only`) on GitHub's machines, all 54 read; scored once by `data/PHY_050_PERCALL/lone_loss_loyfer_score.py`
(`data/PHY_050_LONE_LOSS_LOYFER/lone_loss_loyfer_summary.txt`; numbers computed after the verdict in `data/PHY_050_LONE_LOSS_LOYFER/outcome_numbers.txt`; slim counts in `data/PHY_050_PERCALL/lone_loss_loyfer_out/`).

| Bar | Result | Met |
|---|---|---|
| 1. at least 83 % of samples within 10 % of their own prediction (0.04558 + 0.0284 x loose share) | 44 of 54 (81.5 %) | no |
| 2. median observed/predicted between 0.97 and 1.03 | 0.9870 | yes |

**Verdict: not met** (bar 1 short by one sample).

**Not a milestone:** a development test of the IAM-A middle's derivation; not met.

**In plain words.** On 54 healthy sorted samples whose rate had never been computed, the healthy middle predicted from the M number (1/(1+M) plus the measured share for the cell's loose regions) matched at a median of 0.9870 of the prediction. Without the neurons, the one cell type named as a risk before reading (their marks outside CpGs are not counted), 42 of 47 samples read within 10 % of their own prediction. As scored, the test was not met by one sample, because the neurons were named but not set aside before reading; that was a choice in the test's design, recorded as it happened.

**Stated plainly.** The canary sample (GSM5652190, row 0) was opened before the other 53 were counted, to check the job (ratio 1.013); nothing
was changed after it. The 44 samples inside 10 % read a mean ratio of 0.9989 (SD 0.0523). All ten outside read BELOW the
prediction (0.746 to 0.897): five of the seven neuron samples (0.746 to 0.760), two hepatocyte samples (0.847, 0.869), naive CD8 T cells (0.825),
prostate epithelium (0.861) and granulocytes (0.897).

**Exploration after the verdict (not scored).** The prereg named neurons as a known risk before reading (their marks outside CpGs are not
counted) but did not set them aside, so the verdict stands as scored. Without the seven neuron samples, 42 of 47 read
within 10 %. The ratio rises with the loose share (rank 0.62) and with each sample's molecule-end damage (rank 0.48), so the slope and the meter
both enter the misses; neither was corrected here.

**What it would take to succeed.** One final sealed test, on fresh healthy data from another laboratory, with the form settled by exploration first. **Next.** Exploration continues on samples already read (these 54 and the other Loyfer samples): each library's interior counted beyond its own measured molecule-end reach, neurons set aside for their stated reason, and copier lag separated from permanent loss. Fresh healthy data from another laboratory is reserved for one final sealed test, once exploration has settled the form.

**Prior reading:** GSE186458 was read before (`CANON/data_spent.csv`); these 54 samples' lone-loss rate was never computed before this test.
