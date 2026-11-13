"""The C-score's healthy value and band from independence (derivation register C1; 2026-11-13). STATISTICS, NOT IAM's LAW: the C-score is
c = var(sqrt(B) x the means of B consecutive sites in genome order) / var(z) over the Met-A identity sites (CANON/iam_canon.json). If
healthy residuals are independent from site to site, c = 1 by construction, and its spread is set only by how many blocks there are.
This script draws independent z on the reference's own site count and block size and reports c's distribution, beside the six reference
arrays' leave-one-out values and the commissioned band (0.877-1.152, whole blood, conductor_v3.py). What independence leaves unexplained
is real clustering in healthy specimens, which stays measured."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "Met_A_Floors", "neutrophil_reference_v1_2.json")
J = json.load(open(R)); N, B = len(J["sites_ordered"]), J["clustering_block"]; ch = J["healthy_clustering_median"]
rng = np.random.default_rng(7); C = np.empty(20000)
for i in range(len(C)):
    z = rng.standard_normal(N); m = z[: (N // B) * B].reshape(-1, B).mean(1); C[i] = np.var(np.sqrt(B) * m, ddof=1) / np.var(z, ddof=1)
lo, hi, p95 = np.percentile(C, [2.5, 97.5, 95]); L = np.array(J["healthy_clustering_LOO"])
T = [f"sites {N}, block {B}, blocks {N // B}; 20,000 independent draws",
     f"c under independence: mean {C.mean():.4f}, sd {C.std():.4f}, 2.5-97.5 % {lo:.3f}-{hi:.3f}, one-sided 95 % {p95:.3f}",
     f"in C units (c / c_healthy, c_healthy {ch}): {lo / ch:.3f}-{hi / ch:.3f}, one-sided 95 % {p95 / ch:.3f}",
     f"six reference arrays (leave-one-out): mean {L.mean():.4f}, sd {L.std(ddof=1):.4f}",
     "commissioned band (whole blood): 0.877-1.152",
     "Reading: the lower edge is statistics; the upper edge is wider than independence allows by whole blood's own clustering (measured)."]
open(os.path.join(H, "cscore_band_summary.txt"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
