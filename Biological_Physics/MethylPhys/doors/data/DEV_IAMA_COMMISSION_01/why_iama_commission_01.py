"""DEV-IAMA-COMMISSION-01 review (descriptive, after scoring; not a test): per line and library, the quantities behind the across-labs bar (C3
labs: EMSeq LAB01 vs LAB02, each tared within its own run) and EMSeq LAB02's repeat (C2), with depth, conversion and duplicate fraction, to look
for the cause the stopping rule asks for. Reads epiqc_libraries_scored.csv. Writes why_iama_commission_01_output.txt."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); L = pd.read_csv(os.path.join(HERE, "epiqc_libraries_scored.csv"))
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e))
E = L[L.kit == "EMSeq"].copy(); E["A"] = H(E.eps)
E["A_rel"] = E.groupby(["lab", "rep"]).A.transform(lambda a: [x / np.median(np.delete(a.values, i)) for i, x in enumerate(a.values)])
cols = ["lab", "rep", "line", "eps", "A_rel", "opp", "conversion", "duplicate_fraction", "q0", "run"]
out = ["EMSeq libraries (tare within kit x lab x rep):", E.sort_values(["line", "lab", "rep"])[cols].round(5).to_string(index=False)]
P = E[E.rep == "REP01"].pivot(index="line", columns="lab", values="A_rel"); P["LAB02_minus_LAB01"] = P.LAB02 - P.LAB01
out += ["", "across labs, replicate 1, tared IAM-A:", P.round(4).to_string()]
out += ["", "copy error by lab and line (raw eps), replicate 1:", E[E.rep == "REP01"].pivot(index="line", columns="lab", values="eps").round(5).to_string()]
open(os.path.join(HERE, "why_iama_commission_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
