"""Group-A commissioning, reading (2026-10-20). Runs the one command a researcher runs, run_sample.py, unchanged, on each array.
  read SAMPLES_CSV OUTDIR SHARD NSHARDS   for every array of this shard: fetch its two IDATs from GEO (per-sample download), run_sample.py on the
       IDAT pair (bundle and saved values); for an array in the planted half, plant a 1 % loss of the neutrophils' pattern in its saved values by
       the chain's own rule (conductor_v3.stage_m_blood: beta + f_neu x 0.01 x (0.5 - mu_NEU) at the neutrophil sites, f_neu this array's own
       fraction from its run) and run run_sample.py on the planted values; every array in the unplanted half is also run on its own saved values,
       so both halves of bar 3 are read by the same values route. Writes OUTDIR/rows_SHARD.csv. The IDATs are deleted after each array.
  smoke OUTDIR   the same path on the four spent GSE235717 arrays of the example reports: GSM7508429's untared reading must equal its committed
       report (doors/data/REPORT_EXAMPLES_V4/final/GSE235717_GSM7508429_bundle.json), or the run stops (environment or script not the chain's)."""
import os, sys, json, csv, subprocess, urllib.request, time, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
CH = os.path.join(MP, "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); PY = sys.executable
sys.path.insert(0, CH)

def fetch(gsm, name, out):
    u = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={name}"
    for t in range(6):
        try: urllib.request.urlretrieve(u, out); return out
        except Exception: time.sleep(20 * (t + 1))
    raise RuntimeError(f"could not fetch {name}")

def run(args, tag, out):
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([PY, RS] + args + ["--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", b,
                       "--ledger", os.path.join(out, "ledger.jsonl")], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): return None, (r.stderr or r.stdout)[-300:]
    return json.load(open(b)), ""

def row(b):
    m = (b or {}).get("met_a") or {}; c = (b or {}).get("met_a_cscore") or {}; d = (b or {}).get("cell_shares") or {}
    s = {x["cell"]: x["share"] for x in d.get("cells", [])}
    return dict(A=m.get("A"), f_neu=m.get("fraction"), shift_1pct=m.get("shift_per_1pct_loss"), N=m.get("noise_index"), C=c.get("C"),
                withheld=m.get("reason") if m.get("A") is None else "", **{"share_" + k.replace(" ", "_"): v for k, v in s.items()})

def plant(betas_csv, f_neu, out_csv):
    import conductor_v3 as C
    B = C._bc(); S = pd.Index(B["neutrophil_sites"]); mu = pd.Series(B["profiles_at_neutrophil_sites"]["NEU"], index=S, dtype="float64")
    x = pd.read_csv(betas_csv, index_col=0).iloc[:, 0]; x.index = x.index.astype(str)
    k = S.intersection(x.index); x.loc[k] = (x.loc[k] + f_neu * 0.01 * (0.5 - mu.loc[k])).clip(0, 1)
    x.to_frame().to_csv(out_csv); return len(k)

def one(gsm, grn, red, half, out, work):
    g = fetch(gsm, grn, os.path.join(work, grn)); r = fetch(gsm, red, os.path.join(work, red)); bv = os.path.join(work, gsm + "_betas.csv")
    b, err = run(["--grn", g, "--red", r, "--save-betas", bv], gsm, out); rec = dict(gsm=gsm, route="idat", planted=False, error=err, **row(b))
    recs = [rec]
    if b and rec["A"] is not None and os.path.exists(bv):
        if half == "planted":
            pv = os.path.join(work, gsm + "_planted.csv"); n = plant(bv, rec["f_neu"], pv)
            bp, e2 = run(["--betas", pv], gsm + "_PLANTED", out); recs.append(dict(gsm=gsm, route="values", planted=True, error=e2, sites_planted=n, **row(bp)))
        else:
            bu, e2 = run(["--betas", bv], gsm + "_VALUES", out); recs.append(dict(gsm=gsm, route="values", planted=False, error=e2, **row(bu)))
    for f in os.listdir(work): os.remove(os.path.join(work, f))
    for f in os.listdir(out):
        if f.endswith((".html", ".pdf", ".png")): os.remove(os.path.join(out, f))
    return recs

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "read":
        S = pd.read_csv(sys.argv[2]); out = sys.argv[3]; k, n = int(sys.argv[4]), int(sys.argv[5]); os.makedirs(out, exist_ok=True)
        work = os.path.join(out, "_work"); os.makedirs(work, exist_ok=True); R = []
        for i, s in S.iloc[k::n].iterrows():
            R += one(s.gsm, s.grn, s.red, s.half, out, work); print(s.gsm, R[-1].get("A"), R[-1].get("error", "")[:80], flush=True)
            pd.DataFrame(R).to_csv(os.path.join(out, f"rows_{k}.csv"), index=False)
    elif mode == "smoke":
        out = sys.argv[2]; os.makedirs(out, exist_ok=True); work = os.path.join(out, "_work"); os.makedirs(work, exist_ok=True)
        u = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7508nnn/GSM7508429/suppl/"; L = urllib.request.urlopen(u).read().decode()
        import re; f = sorted(set(re.findall(r"GSM7508429_[^\"]*_(?:Grn|Red)\.idat\.gz", L)))
        g = os.path.join(work, f[0]); r = os.path.join(work, f[1]); urllib.request.urlretrieve(u + f[0], g); urllib.request.urlretrieve(u + f[1], r)
        b, err = run(["--grn", g, "--red", r], "SMOKE_GSM7508429", out)
        ref = json.load(open(os.path.join(MP, "doors", "data", "REPORT_EXAMPLES_V4", "final", "GSE235717_GSM7508429_bundle.json")))
        got, want = row(b), row(ref); keys = ["A", "f_neu", "shift_1pct", "C"] + [k for k in want if k.startswith("share_")]
        diff = {k: (got.get(k), want.get(k)) for k in keys if got.get(k) != want.get(k)}
        json.dump(dict(got=got, want=want, differs=diff, error=err), open(os.path.join(out, "smoke.json"), "w"), indent=1)
        print("SMOKE", "PASS" if not diff and not err else f"FAIL {diff} {err[:200]}"); sys.exit(0 if not diff and not err else 1)
