"""DEV-IAMA-READACROSS-01 diagnostic 2 (written 2026-10-13, BEFORE it was run): after the chemistry fingerprint failed to separate the
suspect EMSeq heads from the controls (DEV-IAMA-TRIM-01 calibration), do the other quality measures the cause run already saved?

# NO CHAIN: arithmetic over head_diagnostics_01.csv (the cause run's alignment quality files, committed). No physics reading.

For each library and each measure m in (duplicate_fraction, primary_mapped / reads_scanned): head shift = m(part 0) - mean m(deeper parts).
BAR, per measure: the measure SEPARATES if every suspect's head shift lies beyond every control's, in one direction.
Declared weakness, before running: 3 suspects and 2 controls. A measure with no relation to the bias separates them perfectly by luck
with probability 2/10 (either direction), and one measure is tried (a second was dropped as malformed, see below). A separating measure is therefore a lead to test on unspent
libraries, never a finding. Suspects and controls from DEV-IAMA-CAUSE-01.
"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(HERE, "head_diagnostics_01.csv"))
# primary_mapped / reads_scanned is NOT a mapped fraction: reads_scanned counts only the reads sampled for the conversion check
# (alignment_qc.py), and every part maps about 24.3 million reads, so that measure was dropped (2026-10-13, after the first run printed
# ratios in the thousands; the duplicate measure and its bar are unchanged).
SUSPECT, CONTROL = ["SRR13051256", "SRR13051268", "SRR13051128"], ["SRR13051263", "SRR13051136"]
L = []
for m in ("duplicate_fraction",):
    sh = {r: float(g[g.part == 0][m].iloc[0] - g[g.part > 0][m].mean()) for r, g in T.groupby("run")}
    s = [sh[r] for r in SUSPECT]; c = [sh[r] for r in CONTROL]
    sep = min(s) > max(c) or max(s) < min(c)
    L.append(f"{m}: head shift suspects " + ", ".join(f"{x:+.5f}" for x in s) + " | controls " + ", ".join(f"{x:+.5f}" for x in c)
             + f" -> {'SEPARATES (a lead only: 2/10 by luck)' if sep else 'does not separate'}")
open(os.path.join(HERE, "head_quality_02_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
