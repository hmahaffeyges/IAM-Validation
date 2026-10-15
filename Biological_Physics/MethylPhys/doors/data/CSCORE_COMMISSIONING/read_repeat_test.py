"""C-score test (b), same-person repeats, and test (a) on a third laboratory: GSE147740 (AIRWAVE, Imperial College London; EPIC v1,
healthy whole blood). Pre-registered in CSCORE_COMMISSIONING_PLAN.md before any array was fetched (2026-10-15).
Stage 1 (chain/stage_1_idat_calibration.py, unchanged) on each array's IDATs, then conductor_v3.run_neutrophil and the same-run tare
used for the first two laboratories (verbatim from boxruns/run3/run3.py; refuses to run if it has drifted).
Bars (definitions adopted 2026-10-09): (b) within-person SD of C_rel <= 0.125 over the complete pairs, SD = sqrt(sum d^2 / 2n); at least
10 complete pairs or INCONCLUSIVE. (a) at least 95 % of all arrays read not above 1.152 (one-sided).
Usage: python3 read_repeat_test.py IDAT_DIR OUT_DIR WORKERS
"""
import ast, hashlib, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")); sys.path.insert(0, CH)
RUN3 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "boxruns", "run3", "run3.py"))
BAND_HI, BAR = 1.152, 0.95

def tare(r, healthy):                      # verbatim from boxruns/run3/run3.py (checked below)
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
mine = _body(open(__file__, encoding="utf-8").read(), "tare")
theirs = _body(open(RUN3, encoding="utf-8").read(), "tare")
assert mine.split("\n", 1)[1] == theirs.split("\n", 1)[1], "the tare has drifted from boxruns/run3/run3.py - the two laboratories would not be scored by the same code"

idat, out, NW = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(HERE, "repeat_GSE147740_samples.csv"), dtype=str)
def one(g):
    import glob, stage_1_idat_calibration as S1, conductor_v3 as C
    try:
        gr = glob.glob(os.path.join(idat, f"{g}_*Grn.idat*")); rd = glob.glob(os.path.join(idat, f"{g}_*Red.idat*"))
        if not gr or not rd: return dict(gsm=g, status="no idat")
        b, _ = S1.calibrate_idat_to_beta(gr[0], rd[0], verbose=False); b = b.astype(float); b.index = b.index.astype(str)
        o = C.run_neutrophil(b, specimen="whole blood", sample_id=g); m = o.get("met_a") or {}; cs = o.get("met_a_cscore") or {}
        return dict(gsm=g, A=m.get("A"), f_neu=m.get("fraction"), N=m.get("noise_index"), C=cs.get("C"),
                    refusal=(o.get("refusal") or o.get("refusal_code") or m.get("refusal") or (m.get("reason") if m.get("A") is None else None)
                             or (cs.get("reason") if cs.get("C") is None else None)),
                    status=("ok" if m.get("A") is not None else "withheld by the chain"))
    except Exception as e: return dict(gsm=g, status=f"error: {type(e).__name__}: {str(e)[:100]}")
import multiprocessing as mp
with mp.Pool(NW) as P: rows = list(P.imap(one, S.gsm))
T = S.merge(pd.DataFrame(rows), on="gsm", how="left")
T = tare(T.reset_index(drop=True), T.A.notna())
T.to_csv(os.path.join(out, "repeat_GSE147740_rows.csv"), index=False)
sc = T[T.C_rel.notna()]; ab = int((sc.C_rel > BAND_HI).sum()); fa = 1 - ab / len(sc) if len(sc) else float("nan")
pairs = [g.sort_values("gsm") for _, g in T[T["repeat"].astype(str) == "True"].groupby("person")]
def sd_pairs(col):
    d = [float(p[col].iloc[0] - p[col].iloc[1]) for p in pairs if len(p) == 2 and p[col].notna().all()]
    return (float(np.sqrt(np.sum(np.square(d)) / (2 * len(d)))) if d else None), len(d)
sd_rel, npair = sd_pairs("C_rel"); sd_raw, _ = sd_pairs("C")
ser = T.C / T[T.C.notna()].C.median(); T["C_rel_series"] = ser; sd_ser, _ = sd_pairs("C_rel_series")
res = {"set": "GSE147740", "laboratory": "AIRWAVE, Imperial College London", "platform": "EPIC v1", "specimen": "whole blood, healthy",
       "test_b": {"bar": "within-person SD of C_rel <= 0.125 (chain tare, slide first then series)", "complete_pairs": npair,
                  "within_sd_C_rel": None if sd_rel is None else round(sd_rel, 4),
                  "verdict": "INCONCLUSIVE (fewer than 10 complete pairs)" if npair < 10 else ("MET" if sd_rel <= 0.125 else "NOT MET"),
                  "reported_not_scored": {"within_sd_untared_C": None if sd_raw is None else round(sd_raw, 4),
                                          "within_sd_series_tared": None if sd_ser is None else round(sd_ser, 4)}},
       "test_a": {"bar": "at least 95 % of arrays not above 1.152", "arrays_scored": len(sc), "above": ab,
                  "not_above_fraction": None if not len(sc) else round(fa, 4), "verdict": None if not len(sc) else ("MET" if fa >= BAR else "NOT MET"),
                  "C_rel_median": None if not len(sc) else round(float(sc.C_rel.median()), 4)},
       "arrays_listed": len(T), "withheld_or_failed": int(T.C_rel.isna().sum()), "reasons": T[T.C_rel.isna()].fillna("").groupby(["status", "refusal"]).size().astype(int).to_dict() if T.C_rel.isna().any() else {},
       "tare_scopes": sc.ref_scope.value_counts().to_dict()}
res["reasons"] = {f"{k[0]} | {k[1]}"[:120]: v for k, v in res["reasons"].items()}
json.dump(res, open(os.path.join(out, "repeat_GSE147740.json"), "w"), indent=1); print(json.dumps(res, indent=1))
