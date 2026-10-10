"""DEV-FINGERPRINT-02 identity sites per normal tissue, by the canon site rule used for prostate in DEV-FINGERPRINT-01 (atlas v2 posterior of the
cell: mean 0.75-0.95 or 0.05-0.25, posterior SD <= 0.05, up to 3,000 per channel, smallest SD). Posteriors from atlas/tools/extract_posterior.py.
Usage: python3 identity_sites.py POSTERIOR_DIR      -> identity_sites_<cell>.json"""
import os, sys, re, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PD = sys.argv[1]
for cell in ("prostate epithelium", "hepatocyte", "lung alveolar epithelium", "lung bronchus epithelium", "breast luminal epithelium", "breast basal epithelium"):
    tag = re.sub(r"\W+", "_", cell).strip("_"); P = pd.read_parquet(os.path.join(PD, f"atlas_{tag}_posterior.parquet")).set_index("cpg"); ok = P["sd"] <= 0.05
    hi = P[ok & P["mean"].between(0.75, 0.95)].sort_values("sd").index[:3000]; lo = P[ok & P["mean"].between(0.05, 0.25)].sort_values("sd").index[:3000]
    S = sorted(hi.union(lo))
    json.dump({"cell": cell, "rule": "atlas v2 posterior: mean 0.75-0.95 or 0.05-0.25, posterior SD <= 0.05, <= 3,000 per channel, smallest SD", "n": len(S),
               "n_hi": len(hi), "n_lo": len(lo), "sites": S, "posterior_mean": [round(float(P.loc[s, "mean"]), 5) for s in S]}, open(os.path.join(HERE, f"identity_sites_{tag}.json"), "w"))
    print(f"{cell:28s} sites {len(S)} (methylated {len(hi)}, unmethylated {len(lo)})")
P = json.load(open(os.path.join(HERE, "identity_sites_prostate_epithelium.json")))["sites"]
Q = json.load(open(os.path.join(HERE, "../DEV_FINGERPRINT_01/prostate_identity_sites_v1.json")))["sites"]
assert P == Q, "prostate sites differ from DEV-FINGERPRINT-01"; print("prostate identical to DEV-FINGERPRINT-01: yes")
