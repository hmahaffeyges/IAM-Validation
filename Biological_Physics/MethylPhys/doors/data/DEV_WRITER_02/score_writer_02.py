"""DEV-WRITER-02 scoring, by the rule sealed 2026-10-10 16:16 PDT and the strand amendment (doors/DEV_WRITER_02.md).
Per sample: contexts pooled with their reverse complements into 136 classes (counts summed; the 'NA' bucket of CpGs without a clean context is
excluded); eps per class = errors/opportunities; x = log of ½[1/(1+D_c) + 1/(1+D_rc)] from enzyme_D_256.csv. OLS slope of log eps on x for the
copy-error channel and for the control (de novo) channel; classes with zero errors are dropped. Statistic: median over samples of (copy-error
slope − control slope). Bars: 0.5-1.5 met; < 0.2 not met; between undecided. Then, reported only: k = exp(mean(log eps − log bracket)).
First, the built-in check: per-sample sums must reproduce PROC-CHANNEL-01's copy_err and denovo.
Usage: python3 score_writer_02.py context_counts.csv"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(sys.argv[1]); Z = pd.read_csv(os.path.join(HERE, "enzyme_D_256.csv")).set_index("ctx")
ref = pd.read_csv(os.path.join(HERE, "../PROC_CHANNEL_01/rerun_2026-10-10_channel_samples.csv")).set_index("sample")
S = C.groupby("sample")[["meth_err", "meth_opp", "ctrl_err", "ctrl_opp"]].sum()
chk = pd.DataFrame({"copy_err": S.meth_err / S.meth_opp, "ref_copy": ref.copy_err.reindex(S.index), "denovo": S.ctrl_err / S.ctrl_opp, "ref_denovo": ref.denovo.reindex(S.index)})
d1 = (chk.copy_err - chk.ref_copy).abs().max(); d2 = (chk.denovo - chk.ref_denovo).abs().max()
print(f"check vs PROC-CHANNEL-01: samples {len(S)}, matched {chk.ref_copy.notna().sum()}, max |diff| copy_err {d1:.2e}, denovo {d2:.2e}")
assert len(S) == 153 and chk.ref_copy.notna().all() and d1 < 1e-9 and d2 < 1e-9, "reader does not reproduce PROC-CHANNEL-01"
C = C[C.ctx != "NA"].copy(); C["cls"] = [min(c, Z.loc[c, "rc"]) for c in C.ctx]
G = C.groupby(["sample", "cls"])[["meth_err", "meth_opp", "ctrl_err", "ctrl_opp"]].sum().reset_index()
G["x"] = np.log(G.cls.map(Z.pred1))
rows = []
for g, d in G.groupby("sample"):
    out = dict(sample=g, classes=len(d))
    for ch in ("meth", "ctrl"):
        e = d[f"{ch}_err"] / d[f"{ch}_opp"]; ok = d[f"{ch}_err"] > 0
        out[f"slope_{ch}"] = np.polyfit(d.x[ok], np.log(e[ok]), 1)[0]; out[f"n_{ch}"] = int(ok.sum())
    e = d.meth_err / d.meth_opp; ok = d.meth_err > 0; out["k"] = float(np.exp(np.mean(np.log(e[ok]) - d.x[ok])))
    out["diff"] = out["slope_meth"] - out["slope_ctrl"]; rows.append(out)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "writer_02_rows.csv"), index=False)
m = R["diff"].median(); verdict = "MET" if 0.5 <= m <= 1.5 else ("NOT MET" if m < 0.2 else "UNDECIDED")
print(f"samples {len(R)} | copy-error slope median {R.slope_meth.median():.3f} | control slope median {R.slope_ctrl.median():.3f} | "
      f"difference median {m:.3f} (IQR {R['diff'].quantile(.25):.3f}-{R['diff'].quantile(.75):.3f}) -> {verdict}")
print(f"k (reported only): median {R.k.median():.3f} (IQR {R.k.quantile(.25):.3f}-{R.k.quantile(.75):.3f})")
