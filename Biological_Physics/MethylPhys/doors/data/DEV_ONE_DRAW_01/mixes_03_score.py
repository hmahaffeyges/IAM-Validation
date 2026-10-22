"""DEV-ONE-DRAW-01, exploration 3 (2026-10-22; not sealed): does the IAM-Deconvolver's full atlas panel find tissue DNA in the amounts put in?
GSE122126's 9 laboratory mixtures of genomic DNA (spent): hepatocyte, lung, colon and neuron DNA mixed into blood DNA in known amounts (3.5 to 10 %).
Each array read by the one command (series.yml; run_sample.py --specimen "constructed DNA mixture" --cell-panel atlas). This file only reads
the bundles. Reported per mixture: each tissue's known amount against the share found (the lung sum over the atlas's lung epithelia, the colon
over colon epithelium and enteroendocrine), and whether its 95 % interval holds the known amount; overall: mean absolute error, the share of
intervals that hold the truth, and the largest share given to a tissue that was not put in.
Usage: python3 mixes_03_score.py ROWS SAMPLES OUT"""
import os, sys, json, re, numpy as np, pandas as pd
rows, samples, out = sys.argv[1], sys.argv[2], sys.argv[3]; S = pd.read_csv(samples)
MAP = {"Hepatocytes": ["hepatocyte"], "Lung cells": ["lung alveolar epithelium", "lung bronchus epithelium"],
       "Colon cells": ["colon epithelium", "colon enteroendocrine"], "Neurons": ["cortical neurons"]}
L = ["DEV-ONE-DRAW-01 exploration 3: GSE122126's 9 known mixtures, read by run_sample.py on the full atlas panel"]; err = []; held = []; stray = []
for r in S.itertuples():
    f = os.path.join(out, "pass1", r.gsm + ".json")
    if not os.path.exists(f): L.append(f"{r.gsm}: no bundle"); continue
    C = {x["cell"]: x for x in (json.load(open(f)).get("cell_shares") or {}).get("cells", [])}
    if not C: L.append(f"{r.gsm}: no shares"); continue
    truth = {t.strip(): float(p) / 100 for p, t in re.findall(r"([\d.]+)% ([A-Za-z ]+?)(?:,|$)", r.truth)}
    parts = []
    for t, cells in MAP.items():
        s = sum(C[c]["share"] for c in cells if c in C); lo = sum(C[c]["interval_95"][0] for c in cells if c in C); hi = sum(C[c]["interval_95"][1] for c in cells if c in C)
        k = truth.get(t, 0.0)
        if k > 0: err.append(abs(s - k)); held.append(lo <= k <= hi); parts.append(f"{t} put in {k:.3f}, found {s:.3f} [{lo:.3f}, {hi:.3f}]")
        else: stray.append(s); parts.append(f"{t} not put in, found {s:.3f}")
    L.append(f"{r.gsm} ({r.truth}): " + "; ".join(parts))
if err: L.append(f"mean absolute error over the tissues put in: {np.mean(err):.4f} (n {len(err)}); intervals holding the known amount: {sum(held)} of {len(held)}; "
                 f"largest share found for a tissue not put in: {max(stray):.4f}")
open(os.path.join(out, "mixes_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
