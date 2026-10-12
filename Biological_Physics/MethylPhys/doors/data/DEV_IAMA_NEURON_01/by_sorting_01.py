"""DEV-IAMA-NEURON-01, descriptive only (after the reference was built; not a test): copy error of every Loyfer neuron sample grouped by Loyfer's
'not sorted' comment (Table S1), region and donor. Reads neuron_eps_01.csv; writes by_sorting_01_output.txt. Usage: python3 by_sorting_01.py"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, "neuron_eps_01.csv"))
out = []
for k, g in R.groupby("sorted"):
    out.append(f"sorted={k}: {len(g)} samples, eps {g.eps.min():.4f}-{g.eps.max():.4f}; " + "; ".join(f"{r.donor} {r.region} {r.eps:.4f} (purity {r.purity})" for r in g.itertuples()))
nd = R[R.donor == "ND264"]; out.append(f"donor ND264 (age 19): cortex {nd[nd.region == 'Cortex'].eps.iloc[0]:.4f}, cerebellum {nd[nd.region == 'Cerebellum'].eps.iloc[0]:.4f}, both not sorted")
open(os.path.join(HERE, "by_sorting_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
