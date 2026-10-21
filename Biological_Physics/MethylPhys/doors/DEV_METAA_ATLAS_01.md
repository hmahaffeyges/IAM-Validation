# DEV-METAA-ATLAS-01: Met-A and the C-Score against IAM-Atlas v2.1 (development, written 2026-10-21 before any reading; not sealed)

**The instrument's job:** a healthy sample of any blood cell, read against that cell in IAM-Atlas v2.1 and tared against healthy samples of the
same cell run the same way, reads Normal; a planted loss of the cell's pattern moves it.

**Why (author, 2026-10-21).** Met-A is commissioned on neutrophils against purified arrays of that cell (6 healthy arrays). Purified arrays are
not available for most cell types, while IAM-Atlas v2.1 holds a healthy reference, with its uncertainty, for all 79. The atlas sits slightly
softer than one array (DEV-REFERENCES-01: at the neutrophil identity sites its entropy is -5 % to +12 % from the purified arrays, by cell), so no
single correction factor carries over; the same-run tare cancels any reference offset, whatever its size or sign. Readings made this way are
printed and flagged "read against IAM-Atlas v2.1; not commissioned by our standards", and the purified-cell route stays the one to test against
as purified arrays become available.

**Built from what, tested on what.** IAM-Atlas v2.1 was fitted on, among 21 sources, both purified-blood array studies (GSE110554, 35 arrays;
GSE167998, 56 arrays), including the 6 neutrophil arrays of Met-A's own reference. Those arrays are therefore not test data here. Test data:
- isolated cells: GSE184269, an independent laboratory not in the atlas; its sorted monocytes, B, CD4, CD8 and NK arrays are unread
  (CANON/data_spent.csv: partly spent, only its whole bloods and granulocytes read);
- whole blood: the Group-A healthy controls (GSE318173, not in the atlas), read both ways.

**Design, in order.**
1. Identity sites per cell, from the atlas alone, by one fixed rule written before reading (the neutrophil rule of Met-A, applied to each cell).
2. Synthetic first, through the chain: atlas-healthy samples of each cell must read 1; a planted 1 % loss must move the reading by the shift
   the chain predicts.
3. Isolated cells (GSE184269): each sorted array read against its atlas cell, tared against the other arrays of the same cell in that series.
4. Whole blood (Group-A): Met-A read against the atlas cells weighted by the person's own shares, and against the commissioned purified
   reference; reported side by side person by person (share in Normal, agreement). The C-Score likewise, with a planted clustered loss.

**What this is not.** Not a commissioning: the bars of a sealed test are written only when the exploration shows the route works ("We explore
WITHOUT sealing until we are certain"). Not a replacement of the commissioned neutrophil reading.
