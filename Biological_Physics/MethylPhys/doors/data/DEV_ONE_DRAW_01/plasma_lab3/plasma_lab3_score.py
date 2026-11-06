"""Plasma from one laboratory, healthy and cancer (GSE186007; written and committed before any of its plasmas is read). The instrument's job: a
healthy plasma from a laboratory the atlas has never seen, tared on that laboratory's other healthy plasmas, reads Normal on the gauge.
The tare (stage_metaa_atlas.tare's rule): each healthy plasma's whole-sample Met-A divided by the median of the laboratory's other healthy plasmas
(leave-one-out); each cancer plasma divided by the median of all the laboratory's healthy plasmas. Bars (healthy plasmas only; at least 20 read):
 1. the tared whole-sample Met-A median within 0.95 to 1.05;
 2. at least 80 % of the healthy plasmas' tared whole-sample Met-A within 0.95 to 1.05.
Recorded, not scored: the untared readings (the laboratory's offset), the islands and domains tared the same way, the C-Score, the Deconvolver's
shares, the library floor; the cancer plasmas tared on the healthy ones, by cancer type (the lines between healthy and cancer are for experts to
set; these go to the A-score database). A plasma the chain refuses is listed with its reason.
Usage: python3 plasma_lab3_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "plasma_lab3_samples.csv")); rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)); o = dict(gsm=r.gsm, group=r.group, age=r.age, sex=r.sex)
    if f:
        B = json.load(open(f[-1])); D = B.get("development") or {}; T = (B.get("territories") or {}).get("sets") or {}
        o.update({"status": B.get("status", "read"), "Met-A whole sample": (B.get("met_a_whole") or {}).get("A"), "islands": (T.get("islands") or {}).get("A"),
                  "domains": (T.get("domains") or {}).get("A"), "C-Score": ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"),
                  "library floor": (D.get("library_floor") or {}).get("floor"), "not read": B.get("reason")})
    else: o["not read"] = "no bundle"
    rows.append(o)
d = pd.DataFrame(rows); hea = d.group == "HEA"
for c in ("Met-A whole sample", "islands", "domains", "C-Score", "library floor", "not read"):
    if c not in d: d[c] = np.nan
for c in ("Met-A whole sample", "islands", "domains"):
    v = pd.to_numeric(d.get(c), errors="coerce"); h = v[hea]
    d[c + ", tared"] = [ (v.iloc[i] / np.nanmedian(h.drop(d.index[i])) if hea.iloc[i] else v.iloc[i] / np.nanmedian(h)) if pd.notna(v.iloc[i]) and h.notna().sum() >= 4 else np.nan for i in range(len(d))]
d.to_csv(sys.argv[2], index=False)
t = d.loc[hea, "Met-A whole sample, tared"].dropna(); ok = len(t) >= 20
b1 = ok and 0.95 <= t.median() <= 1.05; b2 = ok and ((t >= 0.95) & (t <= 1.05)).mean() >= 0.80
L = [f"healthy plasmas read: {len(t)} of {int(hea.sum())} (bars need at least 20)",
     f"bar 1, tared whole-sample Met-A median: {t.median():.4f}" + (" (met)" if b1 else " (not met)") if len(t) else "bar 1: nothing read",
     f"bar 2, share within 0.95 to 1.05: {((t >= 0.95) & (t <= 1.05)).mean():.3f}" + (" (met)" if b2 else " (not met)") if len(t) else "bar 2: nothing read",
     "test: " + ("met" if b1 and b2 else "not met"), "recorded, not scored (median by group):"]
for g, x in d.groupby("group"):
    L.append(f"  {g} ({x['Met-A whole sample'].notna().sum()} read): " + ", ".join(f"{c} {pd.to_numeric(x[c], errors='coerce').median():.4f}" for c in ("Met-A whole sample", "Met-A whole sample, tared", "islands, tared", "domains, tared", "C-Score", "library floor")))
nr = d[d["not read"].notna()]; L += [f"not read: {len(nr)}"] + [f"  {g} ({gr}): {str(w)[:120]}" for g, gr, w in zip(nr.gsm, nr.group, nr['not read'])][:40]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
