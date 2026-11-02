"""Each library's floor (the share of methylated calls at the atlas's floor sites, unmarked in every cell: stage_seq_atlas.library_floor, in
every sequenced bundle of the one command) against its islands reading, for BLUEPRINT's healthy libraries and the atlas's own inputs
(diagnostic, added after the outside-laboratory test was scored; nothing scored). Usage: python3 library_floors.py LABEL=BUNDLE_DIR[:IDS.txt] ... OUT.csv OUT.txt"""
import os, sys, json, pandas as pd
from scipy.stats import spearmanr
rows = []
for arg in sys.argv[1:-2]:
    lab, rest = arg.split("=", 1); d, ids = (rest.split(":", 1) + [None])[:2]; keep = set(open(ids).read().split()) if ids else None
    for f in sorted(os.listdir(d)):
        if not f.endswith("_bundle.json"): continue
        g = f[:-12]
        if keep is not None and g not in keep: continue
        B = json.load(open(os.path.join(d, f))); lf = (B.get("development") or {}).get("library_floor") or {}; i = ((B.get("territories") or {}).get("sets") or {}).get("islands") or {}
        rows.append(dict(gsm=g, source=lab, floor_unmarked=(lf.get("unmarked") or {}).get("rate"), floor_marked=(lf.get("marked") or {}).get("rate"), islands_A=i.get("A"), islands_level_change=i.get("level_change")))
D = pd.DataFrame(rows); D.to_csv(sys.argv[-2], index=False); L = []
for s, g in D.groupby("source"):
    L.append(f"{s}: {len(g)} libraries; floor median {g.floor_unmarked.median():.5f} (range {g.floor_unmarked.min():.5f} to {g.floor_unmarked.max():.5f}); islands median {g.islands_A.median():.4f}; "
             f"islands against own floor rho {spearmanr(g.floor_unmarked, g.islands_A, nan_policy='omit').correlation:+.3f}")
open(sys.argv[-1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
