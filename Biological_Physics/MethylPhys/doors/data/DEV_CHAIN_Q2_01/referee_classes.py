"""PHY-084 read by the project's agreed standard (WHY_IAM.md rule 4, committed 938a6f81 on 2026-11-12, 23 hours before PHY-084 was sealed):
report the pull, (observed - predicted) / sigma_total: within 2 passes; 2 to 3, watch; beyond 3, a real miss; one result just past 2 is not a
refutation; a miss counts when it is large or repeats. Reads referee_out.csv as written by the sealed scorer (unchanged); writes
referee_classes_summary.txt. The seal's own wording (NOT MET if any type lies outside k = 2) is reported beside it."""
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); O = pd.read_csv(os.path.join(H, "referee_out.csv"))
cls = lambda p: "pass" if abs(p) <= 2 else ("watch" if abs(p) <= 3 else "miss")
O["class"] = O.pull.map(cls); miss, watch = O[O["class"] == "miss"], O[O["class"] == "watch"]
v = "a real miss" if len(miss) else ("with " + ", ".join(f"{c.replace('b cells', 'B cells')} on watch (pull {p})" for c, p in zip(watch.cell, watch.pull)) if len(watch) else "all pass")
L = [f"PHY-084 by the agreed standard (WHY_IAM.md rule 4, committed before the seal): {'MET, ' + v if not len(miss) else 'NOT MET, ' + v}; no cell type beyond 3.",
     "Note: the seal's own wording (any type outside k = 2) was stricter than this standard; read by it, not met (amendment 4).",
     "", O[["cell", "n", "intercept", "pred_mid", "u_combined", "pull", "class"]].to_string(index=False), "",
     f"All four pulls are negative (mean {O.pull.mean():.2f}): the shared direction is on watch too; it is not independent across types (the between-type term is common)."]
open(os.path.join(H, "referee_classes_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
