"""DEV-METAA-HEALTHY-01, added after synthetic test 2 (disclosed): how far the IAM-Deconvolver's shares on each synthetic healthy array moved
from the shares the array was built from (those the chain found on the real blood GSM7508429). Total variation, half the sum of absolute
share differences, and the three largest moves. Reads only run_sample.py's bundles from synthetic_02.py's output folder.
Usage: python3 share_drift.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "share_drift_summary.txt")
def shares(tag): return {c["cell"]: c["share"] for c in json.load(open(glob.glob(os.path.join(src, "**", tag + "_bundle.json"), recursive=True)[0]))["cell_shares"]["cells"]}
S0 = shares("real_GSM7508429"); L = ["DEV-METAA-HEALTHY-01 synthetic test 2: the IAM-Deconvolver's shares on each healthy synthetic array against the shares built in"]
for t in ("mild_healthy", "typical_healthy", "weak_healthy"):
    S = shares(t); k = set(S) | set(S0); tv = 0.5 * sum(abs(S.get(c, 0) - S0.get(c, 0)) for c in k)
    big = sorted(((S.get(c, 0) - S0.get(c, 0), c) for c in k), key=lambda x: -abs(x[0]))[:3]
    L.append(f"{t:16} total variation {tv:.4f} | largest: " + ", ".join(f"{c} {d:+.3f}" for d, c in big))
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
