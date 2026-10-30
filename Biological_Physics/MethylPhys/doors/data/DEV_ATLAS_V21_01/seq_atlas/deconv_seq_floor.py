"""The IAM-Deconvolver on sequencing: the share floor by depth (measured, not scored; written and committed before it runs). A pure cell of the
atlas, read at a few reads per site, gives some share to other cells from counting alone; this measures how much, at 5, 10, 20 and 40 reads
per marker site, for 6 pure cells (seed 20261031), through the one command (run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-seq-ref
auto --specimen unknown), so a sorted sample's share outside its own cell is read against the floor at its own depth (the purity check).
Usage: python3 deconv_seq_floor.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV
R, M = Q.deconv_reference_of(); mu = R["mu"].loc[M]; cells = R["cells"]; k = SV.key(); idx = pd.Series(k["idx"], index=k["probe"])
idx = idx[~idx.index.duplicated()]; rng = np.random.default_rng(20261031)
pick = list(rng.choice(cells, 6, replace=False)); runs = [(c, n) for n in (5, 10, 20, 40) for c in pick]
L = [f"Deconvolver on sequencing, share floor by depth: {len(runs)} pure cells read, {len(M)} marker sites, {len(cells)} cells",
     "reads per site | cell | own share | given to other cells"]; F = {}
for j, (c0, N) in enumerate(runs):
    p = mu[c0].fillna(0).clip(0, 1); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    ii = idx.reindex(p.index).dropna().astype(np.int64); kk = rng.binomial(N, p.loc[ii.index].to_numpy())
    a[ii.to_numpy() - 1, 0] = kk; a[ii.to_numpy() - 1, 1] = N; f = os.path.join(W, f"pure{j}.lbeta"); a.tofile(f); b = os.path.join(W, f"pure{j}.json")
    subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--specimen", "unknown",
                    "--id", f"pure{j}", "--out", os.path.join(W, f"pure{j}.html"), "--bundle", b], cwd=os.path.join(C, "MethylPhys_Interface"),
                   capture_output=True, text=True)
    cs = json.load(open(b)).get("cell_shares") or {}; sh = {x["cell"]: x["share"] for x in cs.get("cells") or []}
    own = sh.get(c0) if sh else None; F.setdefault(N, []).append(None if own is None else 1 - own)
    L.append(f"{N} | {c0} | {own} | {None if own is None else round(1 - own, 4)} ({cs.get('reference')})"); os.remove(f)
for N, v in sorted(F.items()):
    v = [x for x in v if x is not None]; L.append(f"floor at {N} reads per site: median {np.median(v):.4f}, largest {max(v):.4f} (of {len(v)})" if v else f"floor at {N}: no shares")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
