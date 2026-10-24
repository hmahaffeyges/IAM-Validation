"""DEV-METAA-HEALTHY-01 synthetic test 1 (bar 1; written before it ran). Arrays built from IAM-Atlas v2.1 mixed in one real healthy blood's
shares (shares_example.json: the IAM-Deconvolver's shares for a spent GSE87571 blood), passed through a KNOWN line and noise per probe
design, x = clip(a + b e + noise); then read by the chain's own stage_metaa_atlas.read_whole, which now measures the line and noise from the
array's anchors and predicts the healthy reading. Bar: A / A_pred = 1.000 within 0.005 across the grid; a planted 5 % loss of the held
pattern (t = e + 0.05 (0.5 - e) before the array's blur) raises A / A_pred by at least half as much as it raises A. Run 2 (written before it ran): the line through the anchors' means, and the
loss planted at every site except the anchors.
Usage: python3 synthetic_01.py [OUT.txt]"""
import sys, os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "synthetic_01_summary.txt")
S = json.load(open(os.path.join(H, "shares_example.json"))); F = pd.Series(S)
des, _, _ = SA._ruler(); idx = des.index
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in F.index], axis=1); have = MU.notna()
cover = have.mul(F, axis=1).sum(1) / F.sum(); e = (MU.fillna(0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1)).where(cover >= SA.MIX_COVER).dropna()
d = des.reindex(e.index).to_numpy(); anc = e.index.isin(sum(SA._ruler()[1].values(), [])); g = np.random.default_rng(20261024)
GRID = [("ideal", {"I": (0, 1, 0.0), "II": (0, 1, 0.0)}), ("mild", {"I": (0.02, 0.95, 0.02), "II": (0.04, 0.90, 0.03)}),
        ("typical", {"I": (0.03, 0.93, 0.03), "II": (0.06, 0.85, 0.04)}), ("weak", {"I": (0.05, 0.88, 0.05), "II": (0.09, 0.78, 0.06)})]
def array(t, P):
    x = np.empty(len(t))
    for k, (a, b, s) in P.items():
        m = d == k; x[m] = a + b * t[m] + s * g.standard_normal(m.sum())
    return pd.Series(np.clip(x, 0, 1), index=e.index)
L = [f"DEV-METAA-HEALTHY-01 synthetic test 1: {len(e)} sites with an expectation; seed 20261024"]
for name, P in GRID:
    R0 = SA.read_whole(array(e.to_numpy(), P), S); t1 = e + 0.05 * (0.5 - e); t1[anc] = e[anc]; R1 = SA.read_whole(array(t1.to_numpy(), P), S)   # run 2: the loss off the anchors
    h0, h1 = R0.get("healthy_prediction") or {}, R1.get("healthy_prediction") or {}
    bl = ", ".join(f"{k} a {v['a']} b {v['b']} s {v['s']}" for k, v in (h0.get("blur") or {}).items())
    L.append(f"{name:8} healthy: A {R0['A']} | A_pred {h0.get('A_pred')} | A/A_pred {h0.get('A_over_pred')} | measured line {bl}")
    if h0.get("A_over_pred") and h1.get("A_over_pred"):
        L.append(f"{'':8} 5 % loss: A {R1['A']} (+{R1['A'] - R0['A']:.4f}) | A/A_pred {h1['A_over_pred']} (+{h1['A_over_pred'] - h0['A_over_pred']:.4f})")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
