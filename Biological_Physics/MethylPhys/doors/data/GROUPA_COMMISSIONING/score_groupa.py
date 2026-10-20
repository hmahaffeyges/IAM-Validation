"""Group-A commissioning, scoring (2026-10-20). Reads every rows_*.csv once, applies the same-run tare of the C-score band test (tare(), checked
below against doors/data/CSCORE_COMMISSIONING/read_band_test.py, which carries the verbatim copy of boxruns/run3/run3.py), and scores the three
sealed bars of sealed_groupa_01.json. Reported, not scored: what the tare does to the slide, plate and laboratory offsets, and the IAM-Atlas v2.1
cell shares (IAM-Deconvolver) by sex. Usage: python3 score_groupa.py ROWS_DIR [SAMPLES_CSV NAME]   -> NAME.json and NAME_summary.txt here
(defaults: groupa_samples.csv, scored_groupa_01). A samples file with a "set" column (the rehearsal on spent sets) is tared within each set."""
import os, sys, glob, json, ast, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..")
def tare(r, healthy):
    import conductor_v3 as C
    hh = r[healthy & r.A.notna()]; out = []
    for _, x in r.iterrows():
        if pd.isna(x.A): out.append((None, None, 0)); continue
        res = None
        for scope in ("slide", "series"):
            ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)]
            if len(ref) >= 3: res = (float(x.A / ref.A.median()), scope, len(ref)); break
        out.append(res or (None, None, 0))
    r["A_rel"], r["ref_scope"], r["n_refs"] = zip(*out)
    cr = []
    for _, x in r.iterrows():
        v = None
        if pd.notna(x.C):
            for scope in ("slide", "series"):
                ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)].C.dropna()
                if len(ref) >= 3: v = float(x.C / ref.median()); break
        cr.append(v)
    r["C_rel"] = cr
    r["state"] = r.A_rel.map(lambda a: None if pd.isna(a) else ("Normal" if 0.95 <= a <= 1.05 else ("above Normal" if a > 1.05 else "below Normal")))
    return r
def _body(src, name):
    f = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(l.strip() for l in ast.get_source_segment(src, f).split("\n") if l.strip() and not l.strip().startswith("#"))
assert _body(open(__file__).read(), "tare") == _body(open(os.path.join(D, "CSCORE_COMMISSIONING", "read_band_test.py")).read(), "tare"), "tare drifted"
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    d = float(a.mean() - b.mean()); return dict(n1=len(a), n2=len(b), mean1=round(float(a.mean()), 5), mean2=round(float(b.mean()), 5), difference=round(d, 5),
                                                 rel_difference=round(d / float(np.concatenate([a, b]).mean()), 5), se=round(se, 5), t=round(d / se, 3))
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(H, "..", "..", "..", "chain")))
    SF = sys.argv[2] if len(sys.argv) > 2 else "groupa_samples.csv"; NAME = sys.argv[3] if len(sys.argv) > 3 else "scored_groupa_01"
    S = pd.read_csv(os.path.join(H, SF)); SEAL = json.load(open(os.path.join(H, "sealed_groupa_01.json"))) if os.path.exists(os.path.join(H, "sealed_groupa_01.json")) else dict(sealed=None, data=SF, bars=dict(B1=dict(min_read=int(0.9 * len(S)))))
    if "set" not in S: S["set"] = "one series"
    TS = lambda d, h: pd.concat([tare(d[d.set == k].copy(), h[d.set == k]) for k in d.set.unique()])
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sys.argv[1], "rows_*.csv")))], ignore_index=True).merge(S, on="gsm", how="left")
    I = R[R.route == "idat"].drop_duplicates("gsm").copy(); I = TS(I, pd.Series(True, index=I.index))
    V = R[R.route == "values"].drop_duplicates("gsm").copy(); V = TS(V, ~V.planted.astype(bool))
    n_read = int(I.A_rel.notna().sum()); b1 = n_read >= SEAL["bars"]["B1"]["min_read"]
    w = I[I.sex == "female"].A_rel.dropna(); m = I[I.sex == "male"].A_rel.dropna(); s2 = welch(w, m)
    b2 = abs(s2["t"]) < 2 and abs(s2["rel_difference"]) < 0.02
    p = V[V.planted.astype(bool)].A_rel.dropna(); u = V[~V.planted.astype(bool)].A_rel.dropna(); s3 = welch(p, u); b3 = s3["t"] > 2
    rep = dict(untared_set_means={k: round(float(v), 4) for k, v in I.groupby("set").A.mean().items()}, tared_set_means={k: round(float(v), 4) for k, v in I.groupby("set").A_rel.mean().items()},
               untared_by_sex=welch(I[I.sex == "female"].A.dropna(), I[I.sex == "male"].A.dropna()),
               untared_slide_spread=round(float(I.groupby("slide").A.median().std() / I.A.median()), 5), tared_slide_spread=round(float(I.groupby("slide").A_rel.median().std()), 5),
               untared_plate_means={k: round(float(v), 4) for k, v in I.groupby("amp_plate").A.mean().items()},
               tared_plate_means={k: round(float(v), 4) for k, v in I.groupby("amp_plate").A_rel.mean().items()},
               ref_scope=I.ref_scope.value_counts().to_dict(), healthy_relative_sd_tared=round(float(I.A_rel.std()), 5),
               cscore_above_band=int((I.C_rel > 1.152).sum()), cscore_read=int(I.C_rel.notna().sum()),
               atlas_shares_by_sex={c: welch(I[I.sex == "female"][c].dropna(), I[I.sex == "male"][c].dropna()) for c in I.columns if c.startswith("share_") and I[c].notna().sum() > 10},
               withheld=I[I.A.isna()][["gsm", "withheld", "error"]].astype(str).values.tolist())
    out = dict(sealed=SEAL["sealed"], read_on=SEAL["data"], B1=dict(n_read=n_read, of=len(S), met=b1), B2=dict(**s2, met=b2), B3=dict(**s3, met=b3),
               all_met=bool(b1 and b2 and b3), reported=rep)
    json.dump(out, open(os.path.join(H, NAME + ".json"), "w"), indent=1, default=str)
    I.to_csv(os.path.join(H, NAME + "_rows_idat.csv"), index=False); V.to_csv(os.path.join(H, NAME + "_rows_values.csv"), index=False)
    L = [f"Group-A {'commissioning, scored once' if SEAL.get('sealed') else 'REHEARSAL on spent sets (no bar, no verdict)'}: {n_read} of {len(S)} healthy whole bloods read (B1 {'MET' if b1 else 'NOT MET'})",
         f"B2 sex null, tared: women {s2['mean1']} (n {s2['n1']}) men {s2['mean2']} (n {s2['n2']}), difference {100*s2['rel_difference']:+.2f} %, {s2['t']} SE -> {'MET' if b2 else 'NOT MET'}",
         f"B3 planted 1 % loss: planted {s3['mean1']} (n {s3['n1']}) unplanted {s3['mean2']} (n {s3['n2']}), difference {100*s3['rel_difference']:+.2f} %, {s3['t']} SE -> {'MET' if b3 else 'NOT MET'}",
         f"ALL BARS {'MET' if out['all_met'] else 'NOT MET'}"]
    open(os.path.join(H, NAME + "_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
