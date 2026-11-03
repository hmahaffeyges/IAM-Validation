"""plasma_seq_02: each plasma's library floor against its islands (development; written and committed after the first plasma was opened and
before the other 29: its floor 0.022906 against the atlas inputs' median 0.0062855, and its islands' level change 0.01623). From each one-command
bundle: the library floor (stage_seq_atlas.library_floor, unmarked side), the islands' level change against the sample's own mix of the IAM
Sequencing Atlas, the islands and domains readings, and the whole-sample Met-A. Reported: the medians; the slope of the islands' level change
on (floor minus the atlas inputs' median floor), through the origin (1 if every false call at the floor sites also falls at the islands;
the atlas's own inputs carried 0.488); and the rank correlation of the islands reading with the floor. Nothing scored.
Usage: python3 plasma_floor_check.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
CH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"))
F0 = float(pd.read_parquet(os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "input_floors_v1.parquet")).floor.median()); rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "SRR*_bundle.json"), recursive=True)):
    B = json.load(open(f)); s = ((B.get("territories") or {}).get("sets") or {}); i = s.get("islands") or {}; d = s.get("domains") or {}
    lf = ((B.get("development") or {}).get("library_floor") or {}).get("unmarked") or {}
    rows.append(dict(run=os.path.basename(f).split("_")[0], floor=lf.get("rate"), islands_level_change=i.get("level_change"), islands=i.get("A"), islands_plugin=i.get("A_plugin"),
                     domains=d.get("A"), metaa_whole=(B.get("met_a_whole") or {}).get("A")))
D = pd.DataFrame(rows).dropna(subset=["floor", "islands_level_change"]); D.to_csv(sys.argv[2], index=False); x = D.floor - F0
L = [f"{len(D)} plasmas; atlas inputs' median floor {F0:.7g}"] + [f"{r.run}: floor {r.floor:.6f}; islands {r.islands:.5f} (plug-in {r.islands_plugin:.5f}), level change {r.islands_level_change:.5f}" for r in D.head(1).itertuples()] + [ f"floor median {D.floor.median():.6f} (range {D.floor.min():.6f} to {D.floor.max():.6f})",
     f"islands level change median {D.islands_level_change.median():.5f}; islands median {D.islands.median():.4f}; domains median {D.domains.median():.4f}; Met-A whole median {D.metaa_whole.median():.4f}",
     f"slope of the islands' level change on (floor - {F0:.5f}), through the origin: {float((x * D.islands_level_change).sum() / (x * x).sum()):.3f}",
     f"islands against the floor: Spearman rho {spearmanr(D.floor, D.islands).correlation:+.3f}; level change against the floor: rho {spearmanr(D.floor, D.islands_level_change).correlation:+.3f}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
