#!/usr/bin/env python3
"""DEVELOPMENT - not commissioned. Chain v3 development round 2, box driver for the IDAT work (2026-10-04):
  DEV-INTAKE-02     the 1,569 arrays that stopped on a missing age / sex in DEV-BASE-CHAIN-01, re-run end to end with the round-2 chain
                    (pass 1 + pass 2 same-run median tare, the DEV-BASE-CHAIN-01 reference rule)
  DEV-DETECTION-01  every EPIC v1 / 450K array of the 56 series: Stage 1 once without the mask, then poobah (P, p <= 0.05) and the Gaussian
                    negative-control test (G, p <= 0.01) masks; noise index, identity-site coverage and Met-A under each; scanner from the IDAT
Resumable: one json per array and task. Copies results to S3 (presigned POST) as produced."""
import os, sys, json, glob, subprocess, time, re, tarfile, shutil, traceback, gzip
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

W = os.getcwd(); D = "/home/ubuntu/data/round2"; MP = f"{W}/repo/Biological_Physics/MethylPhys"; CH = f"{MP}/chain"
RS = f"{CH}/MethylPhys_Interface/run_sample.py"; PY = "/home/ubuntu/env/bin/python"
IDAT = f"{D}/idat"; BET = f"{D}/betas"; OUT = f"{D}/out"; DET = f"{D}/det"; ST = f"{OUT}/status"
for d in (IDAT, BET, OUT, DET, ST): os.makedirs(d, exist_ok=True)
URLS = json.load(open(f"{W}/urls.json")); POST = json.load(open(f"{W}/post.json"))
MAN = pd.read_csv(f"{W}/manifest.csv", dtype={"slide": str, "sentrix": str}).drop_duplicates("gsm")
R1 = pd.read_csv(f"{W}/readings_all.csv")
RERUN = set(R1[(R1.arm == "pass1") & R1.stage0_reason.astype(str).str.contains("manifest", case=False)].gsm)
LOCAL = {"GSE250556": "/home/ubuntu/data/G_chain_tests/GSE250556/idat", "GSE247193": "/home/ubuntu/data/G_chain_tests/GSE247193/idat",
         "GSE247195": "/home/ubuntu/data/G_chain_tests/GSE247195/idat"}
NW = int(os.environ.get("NW", "110")); PREFIX = "results/DEV_ROUND2/idat/"
ONLY = os.environ.get("ONLY_SERIES"); ONLY = set(ONLY.split(",")) if ONLY else None
LOG = open(f"{OUT}/driver.log", "a")
sys.path[:0] = [CH]


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def upload(path, key):
    args = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-F", f"key={PREFIX}{key}"]
    for k, v in POST["fields"].items():
        if k != "key": args += ["-F", f"{k}={v}"]
    args += ["-F", f"file=@{path}", POST["url"]]
    for _ in range(3):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.stdout.strip() == "204": return True
        time.sleep(5)
    log("UPLOAD FAILED", key, r.stdout, r.stderr[-200:]); return False


def download(ser):
    if ser in LOCAL and os.path.isdir(LOCAL[ser]): return LOCAL[ser]
    dest = f"{IDAT}/{ser}"; ok = f"{dest}/.done"
    if os.path.exists(ok): return dest
    os.makedirs(dest, exist_ok=True)
    cmd = "set -o pipefail; ( " + "; ".join(f"curl -sSf --retry 5 '{u}'" for u in URLS[ser]) + f" ) | tar -x -C {dest} --wildcards '*.idat*'"
    for k in range(3):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode == 0: open(ok, "w").write("ok"); return dest
        log("download retry", ser, k, r.stderr[-300:])
    raise RuntimeError(f"download failed {ser}")


def idat_pair(gsm, idir):
    g = sorted(glob.glob(f"{idir}/{gsm}_*Grn.idat*")) or sorted(glob.glob(f"{idir}/**/{gsm}_*Grn.idat*", recursive=True))
    if not g: return None, None
    red = g[0].replace("_Grn", "_Red")
    return g[0], (red if os.path.exists(red) else None)


def scanner_text(path):
    """Printable strings in the IDAT that name the scanner (the run record names the instrument software); [] when none."""
    try:
        raw = gzip.open(path, "rb").read() if path.endswith(".gz") else open(path, "rb").read()
    except Exception as e:
        return [f"unreadable: {type(e).__name__}"]
    hits = set(m.group(0).decode("ascii", "ignore") for m in re.finditer(rb"[\x20-\x7e]{4,80}", raw[-400000:])
               if re.search(rb"(?i)iscan|hiscan|nextseq|beadarray reader|autoloader|scanner", m.group(0)))
    return sorted(hits)[:12]


def instrument(txt):
    t = " ".join(txt).lower()
    return "NextSeq 550" if "nextseq" in t else "HiScan" if "hiscan" in t else "iScan" if "iscan" in t else ("other: " + txt[0][:40] if txt else "not stated")


H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))


def detect_one(job):
    """DEV-DETECTION-01 for one array. Writes DET/<gsm>.json."""
    sp = f"{DET}/{job['gsm']}.json"
    if os.path.exists(sp): return json.load(open(sp))
    row = {k: job[k] for k in ("series", "gsm", "plat", "specimen", "healthy", "slide")}
    try:
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        import stage_1_idat_calibration as S1, stage_0_1_qc_handoff as QH, stage_0_intake as S0, conductor_v3 as C3
        t0 = time.time()
        b, meta = S1.calibrate_idat_to_beta(job["grn"], job["red"], mask_detection=False, return_mask=True, verbose=False)
        b = (b.iloc[:, 0] if hasattr(b, "columns") else b).dropna().astype("float64"); b.index = b.index.astype(str)
        dm = meta.get("_detected_mask"); dm = dm[~dm.index.duplicated()] if dm is not None else None
        at = "EPIC_v1" if job["plat"] == "EPIC_v1" else "HM450K"
        q = QH.decode_qc_inputs(job["grn"], job["red"], at)
        p = np.asarray(S0.compute_detection_p(q["probe_intensities"], q["neg_control_stats"]["mu_bg"], q["neg_control_stats"]["sigma_bg"]))
        gm = pd.Series(p <= S0.DETECTION_P_THRESHOLD, index=pd.Index(q["probe_ids"]).astype(str)); gm = gm[~gm.index.duplicated()]
        txt = scanner_text(job["grn"]); row.update(scanner_text=" | ".join(txt), instrument=instrument(txt), n_cg=int(len(b)), seconds_stage1=round(time.time() - t0, 1))
        NS = pd.Index(C3.noise_sites()["sites"]); IS = pd.Index(C3._bc()["neutrophil_sites"])
        for nm, mk in (("P", dm), ("G", gm)):
            if mk is None: row[f"{nm}_available"] = False; continue
            keep = mk.reindex(b.index).fillna(False).astype(bool); bm = b[keep.values]
            xn = bm.reindex(NS).dropna(); xi = bm.reindex(IS).dropna()
            row.update({f"{nm}_kept": int(len(bm)), f"{nm}_noise_sites": int(len(xn)), f"{nm}_N_raw": float(H(xn.values).mean()) if len(xn) else None,
                        f"{nm}_identity_sites": int(len(xi)), f"{nm}_identity_cov": len(xi) / len(IS)})
            if at == "EPIC_v1" and S0.specimen_refusal(job["specimen"]) is None:
                o = C3.run_neutrophil(bm, specimen=job["specimen"], array_type="EPIC_v1")
                m = o.get("met_a") or {}
                row.update({f"{nm}_A": m.get("A"), f"{nm}_N": m.get("noise_index"), f"{nm}_f_neu": m.get("fraction"), f"{nm}_reason": m.get("reason") or o.get("refusal")})
        row["ok"] = True
    except Exception as e:
        row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-800:])
    json.dump(row, open(sp, "w"), default=str); return row


def _stage0_reason(txt):
    m = re.search(r"Stage 0 verdict: QUARANTINE\s+hard failures: (\[.*?\])", txt)
    if m: return m.group(1)
    m = re.search(r"(QUARANTINE_[A-Z_]+)", txt)
    return m.group(1) if m else None


def run_one(job):
    sp = f"{ST}/{job['arm']}/{job['gsm']}.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        if s.get("class") in ("ok", "stage0_stop", "no_idat"): return s
    os.makedirs(os.path.dirname(sp), exist_ok=True); os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    t0 = time.time(); s = {k: job[k] for k in ("series", "gsm", "arm")}
    if job.get("cmd") is None:
        s.update(**{"class": "no_idat", "reason": "IDAT pair not found", "exit": None}); json.dump(s, open(sp, "w")); return s
    try:
        r = subprocess.run(job["cmd"], capture_output=True, text=True, timeout=3600, cwd=os.path.dirname(RS),
                           env=dict(os.environ, PYTHONPATH=CH, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        txt = r.stdout + "\n" + r.stderr; code = r.returncode
    except subprocess.TimeoutExpired as e:
        txt = f"TIMEOUT {e}"; code = -9
    bp = os.path.splitext(job["out"])[0] + "_bundle.json"; has = os.path.exists(job["out"]) and os.path.exists(bp)
    if code == 0 and has: cls = "ok"
    elif code == 2 and "QUARANTINE" in txt and "Traceback" not in txt and _stage0_reason(txt): cls = "stage0_stop"
    elif code == 3 and "ENVIRONMENT_" in txt: cls = "environment"
    else: cls = "crash"
    s.update(exit=code, **{"class": cls}, reason=_stage0_reason(txt) if cls == "stage0_stop" else None, seconds=round(time.time() - t0, 1),
             report=job["out"] if has else None, bundle=bp if has else None, ledger=job["ledger"], tail=txt[-2500:] if cls != "ok" else txt[-300:])
    json.dump(s, open(sp, "w")); return s


def make_job(r, arm, idir, refs_csv=None):
    out = f"{OUT}/{arm}/{r.series}/{r.gsm}.html"; led = os.path.splitext(out)[0] + "_ledger.jsonl"
    grn, red = idat_pair(r.gsm, idir); job = dict(series=r.series, gsm=r.gsm, arm=arm, out=out, ledger=led, cmd=None)
    if not grn or not red: return job
    c = [PY, RS, "--grn", grn, "--red", red, "--specimen", r.specimen, "--array-type", r.plat, "--id", r.gsm, "--out", out, "--ledger", led,
         "--covariate", f"series={r.series}"]
    if arm == "pass1": c += ["--save-betas", f"{BET}/{r.gsm}.parquet"]
    if isinstance(r.sex, str) and r.sex: c += ["--sex", r.sex]
    if pd.notna(r.age): c += ["--age", str(r.age)]
    if refs_csv: c += ["--slide-ref-table", refs_csv]
    job["cmd"] = c; return job


def bundle_A(s):
    if s.get("class") != "ok": return None
    try: return (json.load(open(s["bundle"])).get("met_a") or {}).get("A")
    except Exception: return None


R1A = R1[(R1.arm == "pass1") & (R1.cls == "ok")].drop_duplicates("gsm").set_index("gsm").A.to_dict()


def tare_jobs(rows, sts, idir):
    """pass 2: >= 3 healthy-reference arrays of the same series and specimen (round-2 pass 1, else round-1 pass 1), same slide, else series."""
    A = dict(R1A); A.update({s["gsm"]: bundle_A(s) for s in sts})
    allr = MAN[MAN.series == rows.series.iloc[0]].assign(A=lambda d: d.gsm.map(A)); jobs = []
    for r in rows.assign(A=rows.gsm.map(A)).itertuples():
        if r.A is None or pd.isna(r.A): continue
        pool = allr[(allr.gsm != r.gsm) & (allr.specimen == r.specimen) & allr.healthy & allr.A.notna()]
        ref = pool[pool.slide == r.slide]
        if len(ref) < 3: ref = pool
        if len(ref) < 3: continue
        p = f"{OUT}/refs/pass2/{r.series}/{r.gsm}.csv"; os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame({"A": ref.A.astype(float), "id": ref.gsm}).to_csv(p, index=False)
        jobs.append(make_job(r, "pass2", idir, p))
    return jobs


def series_pipeline(ser, ex):
    try:
        rows = MAN[MAN.series == ser]; t0 = time.time(); idir = download(ser); log("downloaded", ser, len(rows), f"{time.time()-t0:.0f}s")
        dj = []
        for r in rows.itertuples():
            if r.plat not in ("EPIC_v1", "HM450K"): continue
            g, rr = idat_pair(r.gsm, idir)
            if g and rr: dj.append(dict(series=ser, gsm=r.gsm, plat=r.plat, specimen=r.specimen, healthy=bool(r.healthy), slide=r.slide, grn=g, red=rr))
        rer = rows[rows.gsm.isin(RERUN)]
        f1 = [ex.submit(run_one, make_job(r, "pass1", idir)) for r in rer.itertuples()]
        fd = [ex.submit(detect_one, j) for j in dj]
        st1 = [f.result() for f in f1]
        st2 = [f.result() for f in [ex.submit(run_one, j) for j in tare_jobs(rer, st1, idir)]] if len(rer) else []
        det = [f.result() for f in fd]
        log("done", ser, "rerun", len(st1), pd.Series([s["class"] for s in st1]).value_counts().to_dict() if st1 else {}, "pass2", len(st2),
            "detection", len(det), sum(1 for d in det if d.get("ok")), f"{time.time()-t0:.0f}s")
        tg = f"{OUT}/chunk_{ser}.tgz"
        with tarfile.open(tg, "w:gz") as T:
            for arm in ("pass1", "pass2"):
                if os.path.isdir(f"{OUT}/{arm}/{ser}"): T.add(f"{OUT}/{arm}/{ser}", arcname=f"{arm}/{ser}")
        upload(tg, f"chunks/chunk_{ser}.tgz"); os.remove(tg)
        if ser not in LOCAL: shutil.rmtree(f"{IDAT}/{ser}", ignore_errors=True)
        return ser, "ok"
    except Exception as e:
        log("SERIES FAILED", ser, type(e).__name__, str(e)[:300], traceback.format_exc()[-800:]); return ser, f"failed: {e}"


GSM_RX = re.compile(r"GSM\d{5,9}")


def flat(s):
    d = dict(series=s["series"], gsm=s["gsm"], arm=s["arm"], cls=s["class"], exit=s.get("exit"), stage0_reason=s.get("reason"), seconds=s.get("seconds"))
    if s["class"] == "ok":
        txt = open(s["bundle"]).read(); b = json.loads(txt); m = b.get("met_a") or {}; t = b.get("tare") or {}; it = b.get("intake") or {}
        led = open(s["ledger"]).read() if os.path.exists(s["ledger"]) else ""; h = open(s["report"], encoding="utf-8").read()
        d.update(stage0_verdict=it.get("stage0_verdict"), sex_check=it.get("sex_check"), predicted_sex=it.get("predicted_sex"), declared_sex=it.get("declared_sex"),
                 declared_age=it.get("declared_chronological_age"), refusal=b.get("refusal"), refusal_code=b.get("refusal_code"), A=m.get("A"),
                 state=m.get("state", m.get("reason")), f_neu=m.get("fraction"), n_sites=m.get("n_sites"), N=m.get("noise_index"),
                 noise_measured=m.get("noise_sites_measured"), gate=m.get("noise_gate"), A_rel=t.get("A_rel"), tare_state=t.get("state", t.get("reason")),
                 n_refs=t.get("n_refs"), C=(b.get("met_a_cscore") or {}).get("C"), sample_id=b.get("sample_id"),
                 typed_id_in_bundle=s["gsm"] in txt, typed_id_in_ledger=s["gsm"] in led, any_gsm_in_bundle=bool(GSM_RX.search(txt)),
                 typed_id_in_report_title=f"Cellular Performance Gauge - {s['gsm']}" in h, gauge_drawn="<svg" in h.split("id='sec-met-a'")[-1].split("<h2")[0],
                 explanation_in_report=("noise sites were measured on this array" in h), label_in_report="DEVELOPMENT - not commissioned" in h,
                 red_flags=";".join(f["code"] for f in b.get("red_flags", [])), chain_commit=(b.get("versions") or {}).get("chain_commit"))
    return d


def main():
    t0 = time.time(); sers = [s for s in MAN.series.unique() if (ONLY is None or s in ONLY)]
    log("chain", open(f"{W}/CHAIN_VERSION.txt").read().strip(), "rerun arrays", len(RERUN), "series", len(sers))
    first = MAN[MAN.series == "GSE250556"].iloc[0]; g, r = idat_pair(first.gsm, LOCAL["GSE250556"])
    log("warm-up", first.gsm, detect_one(dict(series=first.series, gsm=first.gsm, plat=first.plat, specimen=first.specimen, healthy=bool(first.healthy),
                                               slide=first.slide, grn=g, red=r)).get("ok"))
    with ProcessPoolExecutor(NW) as ex, ThreadPoolExecutor(6) as tp:
        res = list(tp.map(lambda s: series_pipeline(s, ex), sers))
    log("series results", [x for x in res if x[1] != "ok"])
    X = pd.DataFrame([flat(json.load(open(p))) for p in glob.glob(f"{ST}/*/*.json")]); X.to_csv(f"{W}/intake02_readings.csv", index=False)
    DT = pd.DataFrame([json.load(open(p)) for p in glob.glob(f"{DET}/*.json")]); DT.drop(columns=[c for c in ("tb",) if c in DT], errors="ignore").to_csv(f"{W}/detection01_rows.csv", index=False)
    with open(f"{W}/crash_tails.txt", "w") as f:
        for p in glob.glob(f"{ST}/*/*.json"):
            s = json.load(open(p))
            if s["class"] not in ("ok", "stage0_stop"): f.write(f"===== {s['arm']} {s['series']} {s['gsm']} {s['class']} exit {s.get('exit')}\n{s.get('tail','')[-1500:]}\n")
        for p in glob.glob(f"{DET}/*.json"):
            s = json.load(open(p))
            if not s.get("ok"): f.write(f"===== detection {s['series']} {s['gsm']}\n{s.get('tb','')}\n")
    # a few reports of each kind for reading
    os.makedirs(f"{W}/sample_reports", exist_ok=True)
    for k, g in X[X.cls == "ok"].groupby(X.refusal_code.fillna("read")):
        for gsm, arm, ser in g[["gsm", "arm", "series"]].head(3).values:
            for ext in (".html", "_bundle.json", "_ledger.jsonl"):
                p = f"{OUT}/{arm}/{ser}/{gsm}{ext}"
                if os.path.exists(p): shutil.copy(p, f"{W}/sample_reports/{arm}_{gsm}{ext}")
    for f in ("intake02_readings.csv", "detection01_rows.csv", "crash_tails.txt"): upload(f"{W}/{f}", f)
    upload(f"{OUT}/driver.log", "driver.log")
    log("ALL DONE", f"{time.time()-t0:.0f}s", X.groupby(["arm", "cls"]).size().to_dict(), "detection ok", int(DT.ok.sum()) if len(DT) else 0)


if __name__ == "__main__":
    main()
