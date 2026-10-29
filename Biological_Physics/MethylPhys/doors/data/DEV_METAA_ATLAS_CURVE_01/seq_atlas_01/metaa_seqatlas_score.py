"""Met-A against the IAM Sequencing Atlas, the healthy test (written and committed before any healthy sample is read against it). Each of the
137 healthy inputs (metaa_seqatlas_healthy.csv) read by the one command against the other sealed donor half of its own cell. Depth: the same
original depth the earlier versions were scored against (seq_term_01/band_01/ratio_depth_readings.csv), so the bars compare like with like.
Bars: 1. median Met-A within 0.05 of 1; 2. |Spearman rho| of Met-A with depth at most 0.2. Also reported: 2.5th to 97.5th percentile, and the
readings that did not use the sequencing reference. Usage: python3 metaa_seqatlas_score.py BUNDLE_DIR OUT.txt"""
import sys, os, glob, json, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "metaa_seqatlas_healthy.csv"))
R0 = pd.read_csv(os.path.join(H, "..", "seq_term_01", "band_01", "ratio_depth_readings.csv"))[["gsm", "depth"]]
rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True))
    if not f: continue
    m = (json.load(open(f[-1])).get("development") or {}).get("metaa_atlas") or {}
    rows.append(dict(gsm=r.gsm, cell=r.cell, half=r.half, A=m.get("A"), reading=m.get("reading"), reason=m.get("reason"), n_sites=m.get("n_sites")))
d = pd.DataFrame(rows).merge(R0, on="gsm", how="left"); seq = d.reading.astype(str).str.contains("IAM Sequencing Atlas"); t = d[seq & d.A.notna()]
rho, p = spearmanr(t.A, t.depth) if t.depth.notna().sum() > 2 else (float("nan"), float("nan")); med = float(t.A.median())
L = ["Met-A against the IAM Sequencing Atlas, healthy test: each healthy input against the other donor half of its own cell (run_sample.py --dev-seq-ref)",
     f"read {len(d)} of {len(S)}; against the sequencing reference {len(t)}; not: {d[~(seq & d.A.notna())][['gsm', 'reason']].values.tolist()[:10]}",
     f"Met-A: median {med:.4f}, 2.5th to 97.5th percentile {t.A.quantile(.025):.4f} to {t.A.quantile(.975):.4f}",
     f"bar 1 median within 0.05 of 1: {'met' if abs(med - 1) <= 0.05 else 'not met'}",
     f"bar 2 Met-A against depth: rho {rho:+.3f} (p {p:.2g}); |rho| at most 0.2: {'met' if abs(rho) <= 0.2 else 'not met'}"]
d.to_csv(os.path.join(H, "metaa_seqatlas_readings.csv"), index=False); open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
