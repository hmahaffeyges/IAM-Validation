"""DEV-METAA-ATLAS-01, exploration 8 (2026-10-22; exploratory, not sealed): explorations 6 and 7 again, the chain's way. Every reading is the
one command (run_sample.py, through chain/MethylPhys_Interface/run_series.py on GitHub, series.yml): Stage 0, Stage 1, Stage T step 1 (the
self-tare) and Met-A against IAM-Atlas v2.1 monocytes (development.metaa_atlas), tared by run_sample.py's own --metaa-atlas-ref-A against the
other healthy arrays of the series or of the same slide. This file only reads the table of bundles (rows.csv); it reads and tares nothing.
Usage: python3 chain_08_score.py ROWS_CSV SAMPLES_CSV OUTDIR   (writes OUTDIR/chain_08_summary.txt)"""
import os, sys, numpy as np, pandas as pd
R = pd.read_csv(sys.argv[1]); S = pd.read_csv(sys.argv[2], dtype={"slide": str}); out = sys.argv[3]; name = os.path.basename(sys.argv[2])
L = [f"DEV-METAA-ATLAS-01 exploration 8 (exploratory, not sealed), {name}: {len(S)} arrays listed"]
p1 = R[R.pass_ == "pass1"]
L.append(f"pass 1: {len(p1)} bundles | Stage 0 verdicts {p1.intake.value_counts(dropna=False).to_dict()} | atlas Met-A read {int(p1.A_atlas.notna().sum())}")
for sc in ("series", "slide"):
    g = R[R.pass_ == f"pass2_{sc}"]
    if g.empty: continue
    u = g[~g.planted].set_index("gsm"); p = g[g.planted].set_index("gsm"); a = u.A_atlas_rel.dropna()
    if len(a) < 3: L.append(f"tare {sc}: {len(a)} tared ({u.reason.dropna().head(2).tolist()})"); continue
    sp = float(np.std(a, ddof=1)); mv = (p.A_atlas_rel - u.A_atlas_rel).dropna()
    L.append(f"tare {sc:<6}: tared {len(a)} of {len(u)} | Normal {a.between(0.95, 1.05).mean():.3f} | median {a.median():.4f} | spread {sp:.4f} | "
             f"planted 1 % moves {mv.mean():+.4f} (n {len(mv)}) = {mv.mean() / sp:.2f} spreads")
open(os.path.join(out, "chain_08_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
