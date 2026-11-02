# Outside healthy data for the IAM Sequencing Atlas (inventory; nothing here has been read)

Public data only. Each file lists candidate healthy samples from laboratories other than the atlas's, matched to an atlas cell, with
whether the sample (or its donor) is already spent (`CANON/data_spent.csv`). They are for the healthy tests across laboratories, and
are kept unread until a test names them.

- `blueprint_healthy_candidates.csv`: BLUEPRINT (release 20160816, EBI public FTP, GRCh38 bisulfite call and coverage bigWig, read by
  `run_sample.py --dev-seq-beta CALL.bw,COV.bw`): 59 unspent healthy libraries matched to 12 atlas cells (49 adult venous blood,
  10 cord blood); the adult neutrophils are spent. Index: `20160816.data.index`, healthy = no disease recorded.
- `gse186458_plasma_candidates.csv`: 23 healthy plasma cell-free DNA bisulfite samples in the atlas's own GEO series (GSE186458), unread.
  The same laboratory as the atlas, so a same-laboratory plasma check, not a test across laboratories.
- Plasma already in the inventory (`CANON/data_spent.csv`): PRJNA929650, 278 healthy plasmas and 239 breast cancer plasmas; 30 healthy spent
  for tuning, the rest unread. For plasma once the cell-count floor (checklist D7) is built.
