"""C-score test (a), second laboratory: read the calibrated GSE184269 arrays through the chain and score them against the bar
pre-registered in CSCORE_COMMISSIONING_PLAN.md (and its amendment) before any array was read.

The chain unchanged: conductor_v3.run_neutrophil on each beta table, then the same-run tare used for the first laboratory -
slide first, then series, at least 3 healthy references of the same specimen, the reference's median as the divisor. That tare
is copied verbatim from boxruns/run3/run3.py (the code that scored the first laboratory) and this script refuses to run if the
copy has drifted from it, so the two laboratories cannot be scored by different code.

Bar: of the 24 whole bloods, at least 95 % not above the band's upper limit 1.152 (so at most 1). Granulocytes are refused at
intake (no reference of their own) and are reported with their refusal, excluded from the proportion.
Usage: python3 read_band_test.py BETA_DIR OUT_DIR
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

beta_dir, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(HERE, "band_test_GSE184269_samples.csv"))
import conductor_v3 as C
rows = []
for r in S.itertuples():
    p = os.path.join(beta_dir, r.gsm + "_beta.parquet")
    if not os.path.isfile(p): rows.append(dict(gsm=r.gsm, specimen=r.src, status="no beta")); continue
    _d = pd.read_parquet(p); b = _d.beta.astype(float); b.index = _d.index.astype(str)
    assert b.index.str.startswith("cg").any(), f"{r.gsm}: beta table has no probe names"
    o = C.run_neutrophil(b, specimen=r.src, sample_id=r.gsm)
    m = o.get("met_a") or {}
    rows.append(dict(gsm=r.gsm, specimen=r.src, slide=str(r.title).split()[-1], A=m.get("A"), f_neu=m.get("fraction"),
                     N=m.get("noise_index"), C=(o.get("met_a_cscore") or {}).get("C"),
                     refusal=(o.get("refusal") or o.get("refusal_code") or m.get("refusal")
                              # the chain states WHY it withholds a reading (coverage rule, noise gate, no tare); the first
                              # version of this script dropped those reasons and logged status "ok" with a blank, which made a
                              # withheld array look like a silent hole in the table (2026-10-13)
                              or (m.get("reason") if m.get("A") is None else None)
                              or ((o.get("met_a_cscore") or {}).get("reason") if (o.get("met_a_cscore") or {}).get("C") is None else None)
                              or ((o.get("tare") or {}).get("reason") if (o.get("tare") or {}).get("A_rel") is None else None)),
                     status=("ok" if m.get("A") is not None else "withheld by the chain")))
    print(r.gsm, r.src, "A", rows[-1]["A"], "C", rows[-1]["C"], "|", str(rows[-1]["refusal"])[:60], flush=True)
T = pd.DataFrame(rows)
WB = T[T.specimen.str.contains("blood", case=False)].copy()
if WB.A.notna().any():
    WB = tare(WB.reset_index(drop=True), WB.A.notna())
    T = pd.concat([WB, T[~T.specimen.str.contains("blood", case=False)]], ignore_index=True)
T.to_csv(os.path.join(out, "band_test_GSE184269_rows.csv"), index=False)
scored = WB[WB.C_rel.notna()] if "C_rel" in WB else WB.iloc[0:0]
above = int((scored.C_rel > BAND_HI).sum()) if len(scored) else 0
frac = 1 - above / len(scored) if len(scored) else float("nan")
res = {"set": "GSE184269", "laboratory": "NIA-IRP, NIH (GESTALT healthy donors)", "platform": "EPIC v1",
       "bar": f"at least {BAR:.0%} of the whole bloods not above {BAND_HI}", "n_whole_blood_scored": len(scored),
       "above_limit": above, "not_above_fraction": None if len(scored) == 0 else round(frac, 4),
       "met": None if len(scored) == 0 else bool(frac >= BAR),
       "C_rel_median": None if not len(scored) else round(float(scored.C_rel.median()), 4),
       "C_rel_min": None if not len(scored) else round(float(scored.C_rel.min()), 4),
       "C_rel_max": None if not len(scored) else round(float(scored.C_rel.max()), 4),
       "granulocytes_refused": int((T.specimen.str.contains("ranulocyt", case=False) & T.refusal.notna()).sum()),
       "other_refusals": T[T.specimen.str.contains("blood", case=False) & T.refusal.notna()].refusal.astype(str).tolist(),
       "whole_bloods_listed": int(T.specimen.str.contains("blood", case=False).sum()),
       "whole_bloods_withheld": int((T.specimen.str.contains("blood", case=False) & T.C_rel.isna()).sum()),
       "verdict_if_every_withheld_array_were_above": (
           lambda n_all, n_ab: ("would still be met" if (n_all - n_ab) / n_all >= 0.95 else f"would NOT be met ({(n_all - n_ab) / n_all:.4f})")
       )(int(T.specimen.str.contains("blood", case=False).sum()),
         int(((T.specimen.str.contains("blood", case=False)) & ((T.C_rel > 1.152) | (T.C_rel.isna()))).sum())),
       "tare_scopes": (scored.ref_scope.value_counts().to_dict() if len(scored) else {})}
json.dump(res, open(os.path.join(out, "band_test_GSE184269.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
