"""Summary of the bridge from the IAM Sequencing Atlas to IAM-Atlas v2.1 (bridge_v21.csv, run_sample.py --build seq-bridge).
Usage: python3 bridge_summary.py OUT.txt"""
import os, sys, pandas as pd
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge_v21.csv")); m = D.median(numeric_only=True)
L = [f"{len(D)} cells; median r {m.r:.4f} (lowest {D.r.min():.4f}, {D.loc[D.r.idxmin(), 'cell']}); median mean difference (sequencing minus v2.1) {m.mean_diff:.4f}, "
     f"cells with sequencing lower: {int((D.mean_diff < 0).sum())}; median absolute difference {m.median_abs_diff:.4f}; v2.1 inside the sequencing range {m.inside_range:.4f}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
