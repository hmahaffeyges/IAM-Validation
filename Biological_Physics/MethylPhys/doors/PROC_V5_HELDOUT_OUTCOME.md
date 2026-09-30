# PROC-V5-HELDOUT — outcome (2026-09-30): PASS

Run as pre-registered (PROC_V5_HELDOUT_PREREG.md): 20 blocks drawn with default_rng(2026), 5 % of observations masked, refit with
the build model and seeds, masked values predicted from the posterior. First run lost to a spot reclaim before any block finished;
re-run on the replacement box, every block copied to S3 as it landed.

- **B1 PASS:** 90 % interval covers **92.68 %** of 342,716 held-out observations (bar 85–95 %).
- **B2 PASS:** by kind — WGBS 91.47 % (209,889), array 94.63 % (131,656), pooled 90.44 % (1,171).
- **B3 PASS:** every block 92.31–93.12 % (bar 80–98 %). 0 divergences.
- Reported: 50 % interval coverage 60.6 %; mean absolute prediction error 0.051.
- By cell, lowest: smooth muscle 82.9 %, aorta endothelium 88.8 %, lung alveolar endothelium 88.9 %; highest: CMP, L-MPP and MPP (bone marrow) and
  kidney glomerular epithelium, 96.9–97.2 %.

Reading: the atlas's stated uncertainty is honest for new data of the same kinds — slightly conservative overall (92.7 % against a
nominal 90 %), so intervals carried onto readings (V8) will not overstate certainty.
