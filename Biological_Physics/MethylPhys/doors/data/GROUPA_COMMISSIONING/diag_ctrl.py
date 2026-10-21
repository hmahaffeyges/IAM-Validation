"""Stage 0 control-probe check on the rehearsal arrays (2026-10-20, diagnostic, no score). Rehearsal run 3: 12 of the 24 GSE184269 healthy
whole bloods were quarantined by Stage 0's control-probe gate (ctrl_qc) on the raw-file route; the C-score band test read the same arrays from
their calibrated values, which does not pass through that gate. For every rehearsal array this prints the control-class medians and the
three metrics the gate tests (bisulfite conversion efficiency, hybridization high/low ratio, extension meth/unmeth ratio) with the verdict,
using stage_0_intake unchanged. Usage: python3 diag_ctrl.py SAMPLES_CSV OUT_CSV"""
import os, sys, time, urllib.request, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_0_intake as S0
S = pd.read_csv(sys.argv[1]); W = "/tmp/dc"; os.makedirs(W, exist_ok=True); rows = []
def get(gsm, name):
    out = os.path.join(W, name)
    for t in range(6):
        try:
            urllib.request.urlretrieve(f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={name}", out)
            if open(out, "rb").read(2) == b"\x1f\x8b": return out
        except Exception: pass
        time.sleep(10 * (t + 1))
    raise RuntimeError(name)
for _, s in S.iterrows():
    try:
        g, r = get(s.gsm, s.grn), get(s.gsm, s.red); cs = S0.extract_control_probes(g, r, "EPIC_v1"); v = S0.validate_control_probes(cs)
        rows.append(dict(gsm=s.gsm, set=s.set, slide=s.get("slide"), bs_pairs=";".join(f"{x['pair']}:{x['channel'][0]}:{x['efficiency']}" for x in v["metrics"].get("bisulfite_pairs", [])), verdict=v["ctrl_qc"], **v["metrics"], **{k: round(float(x), 1) for k, x in cs.items() if isinstance(x, (int, float)) and k not in v["metrics"]}))   # 2026-10-20: the summary now carries the efficiency too
        os.remove(g); os.remove(r)
    except Exception as e: rows.append(dict(gsm=s.gsm, set=s.set, verdict="ERROR " + str(e)[:200]))
    print(rows[-1]["gsm"], rows[-1]["verdict"], {k: rows[-1].get(k) for k in ("bisulfite_conversion_efficiency", "hyb_signal", "extension_ratio")}, flush=True)
D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False)
print(D.groupby(["set", "verdict"]).size().to_string())
for c in ("bisulfite_conversion_efficiency", "extension_ratio"):
    if c in D: print(c, D.groupby("set")[c].describe().round(3).to_string())
if "slide" in D:   # 2026-10-20: per slide, so the summary alone shows where low values sit
    print(D.groupby(["set", "slide"]).agg(n=("gsm", "size"), bs_median=("bisulfite_conversion_efficiency", "median"), bs_min=("bisulfite_conversion_efficiency", "min"),
                                          verdicts=("verdict", lambda v: "; ".join(f"{k} {n}" for k, n in v.value_counts().items()))).round(3).to_string())
    lo = D.sort_values("bisulfite_conversion_efficiency").head(4)
    for _, r in lo.iterrows(): print("lowest:", r.gsm, r.slide, r.bisulfite_conversion_efficiency, r.bs_pairs)
