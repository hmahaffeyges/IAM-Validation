"""The IAM-Deconvolver on sequencing, against the IAM Sequencing Atlas (plan 35.30): synthetic, written and committed before it runs.
The instrument's job: a mix of the atlas's own cells reads its planted shares.

Known answer: 9 mixes planted from the chain's sequencing reference's own levels at its marker sites (3 pure cells, 3 of 90/10, 3 of 60/30/10,
cells drawn with seed 20261030), reads drawn k of n at each marker site (20 reads), written as .lbeta files in the hg19 numbering and read by
the one command (run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-seq-ref auto --specimen unknown). Bars (operational, set before
the run): every planted share within 0.05 of its planted value; the cells not planted together at most 0.05.
Usage: python3 deconv_seq_synthetic.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV
R, M = Q.deconv_reference_of(); mu = R["mu"].loc[M]; cells = R["cells"]; k = SV.key(); idx = pd.Series(k["idx"], index=k["probe"])
idx = idx[~idx.index.duplicated()]; rng = np.random.default_rng(20261030); N = 20
pick = list(rng.choice(cells, 18, replace=False)); mixes = [{pick[i]: 1.0} for i in range(3)] + [{pick[3 + 2 * i]: 0.9, pick[4 + 2 * i]: 0.1} for i in range(3)] + \
       [{pick[9 + 3 * i]: 0.6, pick[10 + 3 * i]: 0.3, pick[11 + 3 * i]: 0.1} for i in range(3)]
L = [f"Deconvolver on sequencing, synthetic: {len(mixes)} mixes, {len(M)} marker sites, {N} reads per site, {len(cells)} cells",
     "mix | planted | read | largest error | not planted | bars"]; ok = []
for j, mx in enumerate(mixes):
    p = sum(f * mu[c].fillna(0) for c, f in mx.items()).clip(0, 1); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    ii = idx.reindex(p.index).dropna().astype(np.int64); kk = rng.binomial(N, p.loc[ii.index].to_numpy())
    a[ii.to_numpy() - 1, 0] = kk; a[ii.to_numpy() - 1, 1] = N; f = os.path.join(W, f"mix{j}.lbeta"); a.tofile(f)
    b = os.path.join(W, f"mix{j}.json")
    subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--specimen", "unknown",
                    "--id", f"mix{j}", "--out", os.path.join(W, f"mix{j}.html"), "--bundle", b], cwd=os.path.join(C, "MethylPhys_Interface"),
                   capture_output=True, text=True)
    cs = json.load(open(b)).get("cell_shares") or {}; sh = {x["cell"]: x["share"] for x in cs.get("cells") or []}
    err = max(abs(sh.get(c, 0) - v) for c, v in mx.items()) if sh else None; rest = sum(v for c, v in sh.items() if c not in mx) if sh else None
    w = sh != {} and err <= 0.05 and rest <= 0.05 and "Sequencing" in str(cs.get("reference")); ok.append(w)
    L.append(f"{j} | " + ", ".join(f"{c} {v:.2f}" for c, v in mx.items()) + " | " + (", ".join(f"{c} {sh.get(c, 0):.3f}" for c in mx) if sh else "no shares") +
             f" | {err if err is None else round(err, 4)} | {rest if rest is None else round(rest, 4)} | {'met' if w else 'not met'} ({cs.get('reference')})")
    os.remove(f)
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
