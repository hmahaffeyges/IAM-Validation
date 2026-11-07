"""The Deconvolver decides what is in a sample (author 2026-11-07: "the decovolver is ALWAYS the first things we use to determine what is in the
sample ... I dont care what the label says"): synthetic, written and committed before it runs, through the one command (run_sample.py --betas).
Four constructed arrays, every one labeled "sorted cd4 t cells", built from IAM-Atlas v2.1's blood columns (stage_deconv_v21.reference) plus
noise: 'cd4' (all CD4 T cells), 'half' (50 % CD4, 50 % monocytes), 'quarter' (24 % CD4, 40 % monocytes, 36 % NK cells; the lowest share of the
labeled cell the Deconvolver found in a real outside 'sorted' sample) and 'mono' (all monocytes, labeled CD4; as one Johns Hopkins sample was).
Bars: 1. 'mono' is read as monocytes: the label check names monocytes as the chain's cell and records the disagreement, and the sorted-cell
Met-A reads monocytes, not CD4; 2. 'cd4' is read as CD4 and the label check agrees; 3. in 'half' and 'quarter' every cell the Deconvolver finds
at 1 % or more has its own Met-A for each cell, and the sorted-cell Met-A reads the cell found most of.
Usage: python3 label_synthetic.py OUTDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_deconv_v21 as D
out, txt = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
R = D.reference(list(D.BLOOD6)); mu = R["mu"]; mu = mu[mu.notna().all(1)]
rng = np.random.default_rng(20261107)
CASES = {"cd4": {"cd4 t cells": 1.0}, "half": {"cd4 t cells": 0.5, "monocytes": 0.5},
         "quarter": {"cd4 t cells": 0.24, "monocytes": 0.40, "nk cells": 0.36}, "mono": {"monocytes": 1.0}}
res, L = {}, ["The Deconvolver decides: four arrays labeled 'sorted cd4 t cells', read by the one command"]
for k, mix in CASES.items():
    b = sum(w * mu[c] for c, w in mix.items()) + rng.normal(0, 0.02, len(mu)); bp = os.path.join(out, k + ".csv")
    pd.DataFrame({"beta": b.clip(0.001, 0.999)}).to_csv(bp); rp = os.path.join(out, k + ".html")
    r = subprocess.run([sys.executable, os.path.join(CH, "MethylPhys_Interface", "run_sample.py"), "--betas", bp, "--specimen", "sorted cd4 t cells",
                        "--id", "label_" + k, "--out", rp, "--full-chain"], capture_output=True, text=True)
    B = json.load(open(rp[:-5] + "_bundle.json")) if os.path.exists(rp[:-5] + "_bundle.json") else {}
    lc = B.get("label_check") or {}; ma = (B.get("development") or {}).get("metaa_atlas") or {}
    each = {c["cell"]: c.get("A") for c in ((B.get("met_a_each_cell") or {}).get("cells") or []) if c.get("share", 0) >= 0.01}
    res[k] = dict(rc=r.returncode, chain_cell=lc.get("chain_cell"), agrees=lc.get("agrees"), found=[(f["cell"], f["share"]) for f in lc.get("found", [])],
                  metaa_cell=ma.get("cell"), each=each, ref=B.get("whole_sample_reference"), tail=None if r.returncode == 0 else (r.stdout + r.stderr)[-400:])
    L.append(f"{k}: planted {mix}; found {res[k]['found'][:4]}; chain cell {res[k]['chain_cell']}; label agrees {res[k]['agrees']}; "
             f"sorted-cell Met-A reads {res[k]['metaa_cell']}; Met-A for each cell found: {sorted(each)}; whole-sample reference: {res[k]['ref']}"
             + ("" if r.returncode == 0 else f" | run failed: {res[k]['tail']}"))
m, c, h, q = res["mono"], res["cd4"], res["half"], res["quarter"]
b1 = m["chain_cell"] == "monocytes" and m["agrees"] is False and m["metaa_cell"] == "monocytes"
b2 = c["chain_cell"] == "cd4 t cells" and c["agrees"] is True and c["metaa_cell"] == "cd4 t cells"
b3 = all(x["found"] and set(f for f, s in x["found"] if s >= 0.01) <= {k for k, v in x["each"].items() if v is not None} and x["metaa_cell"] == x["found"][0][0] for x in (h, q))
L += [f"bar 1 (the monocyte sample labeled CD4 is read as monocytes): {'met' if b1 else 'not met'}", f"bar 2 (the CD4 sample is read as CD4): {'met' if b2 else 'not met'}",
      f"bar 3 (every cell found in the mixes has its own Met-A; the sorted-cell reading takes the cell found most of): {'met' if b3 else 'not met'}",
      "synthetic: " + ("met" if b1 and b2 and b3 else "not met")]
open(txt, "w").write("\n".join(L) + "\n"); print("\n".join(L))
