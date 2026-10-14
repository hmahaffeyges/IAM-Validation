"""DEV-ERASE-01 verdict (committed before any read). Reads every <gsm>_stageq.json; pools per unit and genotype (sum of errors over sum of
opportunities); ratio = eraser-removed over control (KO/WT; side arms AG120/DMSO and vitC/untreated, reported only). Primary-unit verdicts by
the sealed rule (MET 0.40-0.60; NOT MET >= 0.85; else INCONCLUSIVE; UNREADABLE under 100,000 opportunities per genotype). DEV-ERASE-01 MET if
>= 3 of U1-U5 MET and none NOT MET; NOT MET if >= 3 NOT MET; else INCONCLUSIVE. Writes erase_verdict.txt and erase_verdict_units.csv."""
import sys, os, json, glob, pandas as pd
d = sys.argv[1]; S = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "erase_samples.csv")).drop_duplicates("gsm")
J = {os.path.basename(f).split("_")[0]: json.load(open(f)) for f in glob.glob(os.path.join(d, "*_stageq.json"))}
S["opp"] = S.gsm.map(lambda g: J.get(g, {}).get("opp")); S["err"] = S.gsm.map(lambda g: J.get(g, {}).get("err")); S["share"] = S.gsm.map(lambda g: J.get(g, {}).get("qualifying_share"))
S["eps"] = S.err / S.opp
rows = []
for u, g in S.groupby("unit", sort=True):
    gs = list(dict.fromkeys(g.genotype)); ctrl = [x for x in gs if x in ("WT", "DMSO", "untreated")][0]; trt = [x for x in gs if x != ctrl][0]
    c, t = g[g.genotype == ctrl], g[g.genotype == trt]
    oc, ot = c.opp.sum(skipna=True), t.opp.sum(skipna=True); missing = int(g.opp.isna().sum())
    ec, et = (c.err.sum() / oc if oc else float("nan")), (t.err.sum() / ot if ot else float("nan")); q = et / ec if oc and ot else float("nan")
    if u.startswith("U"):
        v = "UNREADABLE" if (oc < 100_000 or ot < 100_000 or missing) else ("MET" if 0.40 <= q <= 0.60 else ("NOT MET" if q >= 0.85 else "INCONCLUSIVE"))
    else: v = "reported"
    rows.append(dict(unit=u, set=",".join(dict.fromkeys(g.gse)), control=ctrl, treated=trt, n_ctrl=len(c), n_trt=len(t), missing=missing, eps_ctrl=ec, eps_trt=et,
                     ratio=q, share_ctrl=c.share.mean(), share_trt=t.share.mean(), opp_ctrl=int(oc), opp_trt=int(ot), verdict=v))
U = pd.DataFrame(rows); U.to_csv("erase_verdict_units.csv", index=False)
P = U[U.unit.str.startswith("U")]; m, n = int((P.verdict == "MET").sum()), int((P.verdict == "NOT MET").sum())
V = "MET" if (m >= 3 and n == 0) else ("NOT MET" if n >= 3 else "INCONCLUSIVE")
L = ["DEV-ERASE-01, read once (erase_verdict.py)", U.round(5).to_string(index=False), f"primary units MET {m}, NOT MET {n} of {len(P)}", f"VERDICT: DEV-ERASE-01 {V}"]
open("erase_verdict.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
