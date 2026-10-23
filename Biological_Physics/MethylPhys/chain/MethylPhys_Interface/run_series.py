"""run_series.py - the one command over a series of arrays (chain v4, 2026-10-22). Every reading is chain/MethylPhys_Interface/run_sample.py,
run unchanged on the array's two IDATs from GEO; this file only fetches the files, chooses each array's same-run references and collects the
bundles. Nothing is read, tared or scored here: the tare is run_sample.py's own (--slide-ref-table for Met-A and the C-Score, --metaa-atlas-ref-A
for Met-A against the atlas), and a scorer reads only the bundles.

  read  SAMPLES OUT SHARD NSHARDS [--specimen S] [--pass 1|2] [--scope series|slide] [--plant metaa_atlas] [-- extra run_sample flags]
        SAMPLES: CSV with gsm and slide (other columns are kept). Pass 1: each array once, no references; its bundle goes to OUT/pass1/.
        Pass 2: each array again with its references, the other arrays of SCOPE (the series, or the same slide) taken from the pass-1 bundles
        in OUT/pass1 (never the array itself; at least 3 or the chain prints the array untared, with the reason); its bundle goes to OUT/pass2_SCOPE/.
        --plant metaa_atlas: in pass 2 the array's own values (run_sample.py --save-betas) also get the constructed 1 % loss of the specimen's
        atlas cell (stage_metaa_atlas.plant) and run_sample.py reads them, with the same references and a PLANTED banner -> OUT/pass2_SCOPE/<gsm>_planted.json
        (the saved values are float32, so the planted reading differs from the IDAT reading by the plant and by rounding below 1e-6).
        --plant neutrophil: Group-A's sealed design (conductor_v3.plant_neutrophil_loss): with a 'half' column, the planted half is read again
        with the loss planted in its own saved values and the unplanted half is read again from its own saved values unchanged, so both halves
        are read by the same values route, and tared against the unplanted half read the same way (pass 1 reads that half from its saved
        values too); without one, every array is planted. --scope slide_else_series: the same slide when it holds at least
        3 others, else the series (Group-A's tare).
  table OUT   one row per bundle (pass, gsm, planted, intake verdict, Met-A A / A_rel / state, C / C_rel, atlas Met-A A / A_rel / state, reason) -> OUT/rows.csv
"""
import os, sys, re, json, time, glob, argparse, subprocess, urllib.request
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "run_sample.py"); PY = sys.executable

def fetch(gsm, d):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    for t in range(6):
        try:
            got = {}
            for n in sorted(set(re.findall(rf"{gsm}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
                p = os.path.join(d, n); urllib.request.urlretrieve(u + n, p); got["grn" if "_Grn" in n else "red"] = p
            if len(got) == 2: return got
        except Exception as e: print(gsm, "fetch retry", type(e).__name__, flush=True)
        time.sleep(20 * (t + 1))
    raise RuntimeError(f"{gsm}: IDATs not fetched from GEO")

def run(args, log):
    r = subprocess.run([PY, RS] + args, capture_output=True, text=True)
    open(log, "w").write(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:]); return r.returncode

def refs_for(gsm, slide, P1, scope, out):
    """The pass-1 readings of the other arrays of the scope, as run_sample.py's reference inputs (a table for Met-A and the C-Score; a list for atlas Met-A)."""
    if "ref" in P1.columns: P1 = P1[P1["ref"].astype(str).str.lower().isin(("1", "yes", "true"))]   # 2026-10-23: a ref column names the references (DEV-FIRES-01: only those who stayed cancer-free)
    same = P1[(P1.gsm != gsm) & (P1.slide == slide)]
    R = same if scope == "slide" or (scope == "slide_else_series" and same.A.notna().sum() >= 3) else P1[P1.gsm != gsm]
    t = os.path.join(out, f"refs_{gsm}.csv"); R[["gsm", "A", "C"]].rename(columns={"gsm": "id"}).dropna(subset=["A"]).to_csv(t, index=False)
    at = [str(x) for x in R.A_atlas.dropna()]
    return t, len(R.A.dropna()), at

def bundle_row(p):
    b = json.load(open(p)); m = b.get("met_a") or {}; t = b.get("tare") or {}; c = b.get("met_a_cscore") or {}; d = (b.get("development") or {}).get("metaa_atlas") or {}
    dt = d.get("tare") or {}
    return dict(A=m.get("A"), A_rel=t.get("A_rel"), f_neu=m.get("fraction"), n_refs=t.get("n_refs"), state=m.get("state"), C=c.get("C"), C_rel=c.get("C_rel"),
                A_atlas=d.get("A"), A_atlas_rel=dt.get("A_rel"), state_atlas=dt.get("state") or d.get("state"), n_refs_atlas=dt.get("n_refs"),
                intake=(b.get("intake") or {}).get("stage0_verdict"), reason=d.get("reason") or m.get("reason") or b.get("refusal"))

def read(a, extra):
    S = pd.read_csv(a.samples, dtype={"slide": str}); S = S.iloc[a.shard::a.nshards]
    if S.empty: print(f"shard {a.shard} of {a.nshards}: no arrays in this shard"); return
    out = os.path.abspath(a.out); d1, d2, W = (os.path.join(out, x) for x in ("pass1", f"pass2_{a.scope}", f"work_{a.scope}_{a.shard}")); [os.makedirs(x, exist_ok=True) for x in (d1, d2, W)]
    P1 = P1v = None
    if a.pass_ == 2:
        sl = pd.read_csv(a.samples, dtype={"slide": str, "ref": str}); sl = sl[[k for k in ("gsm", "slide", "ref") if k in sl.columns]]
        P1 = pd.DataFrame([dict(gsm=os.path.basename(p)[:-5], **bundle_row(p)) for p in glob.glob(os.path.join(d1, "*.json")) if not p.endswith(("_planted.json", "_values.json"))]).merge(sl, on="gsm")
        PV = [dict(gsm=os.path.basename(p)[:-12], **bundle_row(p)) for p in glob.glob(os.path.join(d1, "*_values.json"))]
        P1v = pd.DataFrame(PV).merge(sl, on="gsm") if PV else None   # the values route's references: the unplanted half read from its saved values
    for k, r in enumerate(S.itertuples()):
        t0 = time.time()
        try:
            f = fetch(r.gsm, W); base = ["--grn", f["grn"], "--red", f["red"], "--specimen", a.specimen, "--id", r.gsm]
            if a.pass_ == 1:
                bv1 = os.path.join(W, r.gsm + "_betas.parquet")
                rc = run(base + ["--out", os.path.join(W, r.gsm + ".html"), "--bundle", os.path.join(d1, r.gsm + ".json"), "--save-betas", bv1] + extra, os.path.join(d1, r.gsm + ".log"))
                if a.plant == "neutrophil" and "half" in S.columns and getattr(r, "half") == "unplanted" and rc == 0 and os.path.exists(bv1):
                    uv = os.path.join(W, r.gsm + "_values.csv"); pd.read_parquet(bv1).iloc[:, 0].rename("beta").to_frame().to_csv(uv)
                    run(["--betas", uv, "--specimen", a.specimen, "--id", r.gsm + "_values", "--out", os.path.join(W, r.gsm + "_values.html"),
                         "--bundle", os.path.join(d1, r.gsm + "_values.json")] + extra, os.path.join(d1, r.gsm + "_values.log"))
            else:
                tab, n, at = refs_for(r.gsm, r.slide, P1, a.scope, W)
                ref = (["--slide-ref-table", tab] if n else []) + (["--metaa-atlas-ref-A", ",".join(at)] if at else [])
                bv = os.path.join(W, r.gsm + "_betas.parquet")
                rc = run(base + ref + ["--out", os.path.join(W, r.gsm + ".html"), "--bundle", os.path.join(d2, r.gsm + ".json"), "--save-betas", bv] + extra, os.path.join(d2, r.gsm + ".log"))
                half = getattr(r, "half", None) if "half" in S.columns else None
                if a.plant == "neutrophil" and rc == 0 and os.path.exists(bv):   # Group-A's sealed design: planted half against unplanted half, both by saved values
                    sys.path.insert(0, os.path.dirname(HERE)); import conductor_v3 as C3
                    b = pd.read_parquet(bv).iloc[:, 0]; f_neu = bundle_row(os.path.join(d2, r.gsm + ".json")).get("f_neu")
                    vref = ref
                    if P1v is not None:
                        vd = os.path.join(W, "v"); os.makedirs(vd, exist_ok=True)
                        tv, nv, _ = refs_for(r.gsm, r.slide, P1v, a.scope, vd); vref = ["--slide-ref-table", tv] if nv else []
                    if f_neu is not None and (half in (None, "planted")):
                        x, n = C3.plant_neutrophil_loss(b, f_neu); pv = os.path.join(W, r.gsm + "_planted.csv"); x.rename("beta").to_frame().to_csv(pv)
                        run(["--betas", pv, "--specimen", a.specimen, "--id", r.gsm + "_planted", "--banner", "PLANTED: a constructed 1 % loss of the neutrophil pattern, not a person."]
                            + vref + ["--out", os.path.join(W, r.gsm + "_planted.html"), "--bundle", os.path.join(d2, r.gsm + "_planted.json")] + extra, os.path.join(d2, r.gsm + "_planted.log"))
                    if half == "unplanted":
                        uv = os.path.join(W, r.gsm + "_values.csv"); b.rename("beta").to_frame().to_csv(uv)
                        run(["--betas", uv, "--specimen", a.specimen, "--id", r.gsm + "_values"] + vref
                            + ["--out", os.path.join(W, r.gsm + "_values.html"), "--bundle", os.path.join(d2, r.gsm + "_values.json")] + extra, os.path.join(d2, r.gsm + "_values.log"))
                if a.plant == "metaa_atlas" and rc == 0 and os.path.exists(bv):
                    sys.path.insert(0, os.path.dirname(HERE)); import stage_metaa_atlas as SA, stage_0_intake as S0
                    cell = S0.atlas_cell_of(a.specimen); b = pd.read_parquet(bv).iloc[:, 0]; pv = os.path.join(W, r.gsm + "_planted.csv")
                    SA.plant(b, cell).rename("beta").to_frame().to_csv(pv)
                    run(["--betas", pv, "--specimen", a.specimen, "--id", r.gsm + "_planted", "--banner", f"PLANTED: a constructed 1 % loss of the {cell} pattern, not a person."]
                        + ref + ["--out", os.path.join(W, r.gsm + "_planted.html"), "--bundle", os.path.join(d2, r.gsm + "_planted.json")] + extra, os.path.join(d2, r.gsm + "_planted.log"))
            print(k + 1, "of", len(S), r.gsm, "pass", a.pass_, "exit", rc, f"{time.time() - t0:.0f} s", flush=True)
        except Exception as e: print(k + 1, "of", len(S), r.gsm, "ERROR", f"{type(e).__name__}: {e}"[:200], flush=True)
        import shutil; shutil.rmtree(W, ignore_errors=True); os.makedirs(W, exist_ok=True)

def table(a):
    rows = []
    for ps in ("pass1", "pass2_series", "pass2_slide", "pass2_slide_else_series"):
        for p in sorted(glob.glob(os.path.join(a.out, ps, "*.json"))):
            g = os.path.basename(p)[:-5]; route = "values" if g.endswith(("_planted", "_values")) else "idat"
            rows.append(dict(pass_=ps, gsm=g.replace("_planted", "").replace("_values", ""), planted=g.endswith("_planted"), route=route, **bundle_row(p)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rows.csv"), index=False); print(len(rows), "bundles ->", os.path.join(a.out, "rows.csv"))

if __name__ == "__main__":
    argv = sys.argv[1:]; extra = argv[argv.index("--") + 1:] if "--" in argv else []; argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r_ = sp.add_parser("read"); r_.add_argument("samples"); r_.add_argument("out"); r_.add_argument("shard", type=int); r_.add_argument("nshards", type=int)
    r_.add_argument("--specimen", default="whole blood"); r_.add_argument("--pass", dest="pass_", type=int, default=1, choices=(1, 2))
    r_.add_argument("--scope", default="series", choices=("series", "slide", "slide_else_series")); r_.add_argument("--plant", default=None, choices=("metaa_atlas", "neutrophil"))
    t_ = sp.add_parser("table"); t_.add_argument("out")
    a = ap.parse_args(argv); read(a, extra) if a.cmd == "read" else table(a)
