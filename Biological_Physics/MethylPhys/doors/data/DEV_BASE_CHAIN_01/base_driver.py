#!/usr/bin/env python3
"""DEV-BASE-CHAIN-01 box driver: every array of every series through run_sample.py --engine v3 (pass 1), the diagnostic
--no-intake arm for arrays stopped at Stage 0.2 for a missing age/sex, pass 2 (same-run median tare), the report-section check,
and copy-to-S3 per series as produced. Resumable: one status file per array and arm."""
import os, sys, json, glob, subprocess, time, re, tarfile, shutil, traceback
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

W = os.getcwd(); D = "/home/ubuntu/data/base_chain_01"; REPO = f"{D}/repo"; CH = f"{REPO}/Biological_Physics/MethylPhys/chain"
RS = f"{CH}/MethylPhys_Interface/run_sample.py"; PY = "/home/ubuntu/env/bin/python"
OUT = f"{D}/out"; IDAT = f"{D}/idat"; BET = f"{D}/betas"; ST = f"{OUT}/status"
for d in (OUT, IDAT, BET, ST): os.makedirs(d, exist_ok=True)
URLS = json.load(open(f"{W}/urls.json")); POST = json.load(open(f"{W}/post.json"))
MAN = pd.read_csv(f"{W}/manifest.csv", dtype={"slide": str, "sentrix": str})
ONLY = os.environ.get("ONLY_SERIES"); ONLY = set(ONLY.split(",")) if ONLY else None
LOCAL = {"GSE250556": "/home/ubuntu/data/G_chain_tests/GSE250556/idat", "GSE247193": "/home/ubuntu/data/G_chain_tests/GSE247193/idat",
         "GSE247195": "/home/ubuntu/data/G_chain_tests/GSE247195/idat"}
NW = int(os.environ.get("NW", "96")); PREFIX = "results/DEV_BASE_CHAIN_01/"
LOG = open(f"{OUT}/driver.log", "a")


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
    parts = URLS[ser]
    cmd = "set -o pipefail; ( " + "; ".join(f"curl -sSf --retry 5 '{u}'" for u in parts) + f" ) | tar -x -C {dest} --wildcards '*.idat*'"
    for k in range(3):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode == 0: open(ok, "w").write("ok"); return dest
        log("download retry", ser, k, r.stderr[-300:])
    raise RuntimeError(f"download failed {ser}")


def idat_pair(r, idir):
    g = sorted(glob.glob(f"{idir}/{r.gsm}_*Grn.idat*")) or sorted(glob.glob(f"{idir}/**/{r.gsm}_*Grn.idat*", recursive=True))
    if not g: return None, None
    red = g[0].replace("_Grn", "_Red")
    return g[0], (red if os.path.exists(red) else None)


def _stage0_reason(txt):
    m = re.search(r"Stage 0 verdict: QUARANTINE\s+hard failures: (\[.*?\])", txt)
    if m: return m.group(1)
    m = re.search(r"(QUARANTINE_[A-Z_]+)", txt)
    return m.group(1) if m else None


def run_one(job):
    """job: dict(series, gsm, arm, cmd, out). Writes ST/<arm>/<gsm>.json; returns it."""
    sp = f"{ST}/{job['arm']}/{job['gsm']}.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        if s.get("class") in ("ok", "stage0_stop", "no_idat"): return s
    os.makedirs(os.path.dirname(sp), exist_ok=True); os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    t0 = time.time()
    s = {k: job[k] for k in ("series", "gsm", "arm")}
    if job.get("cmd") is None:
        s.update(**{"class": "no_idat", "reason": "IDAT pair not found in the series archive", "exit": None})
        json.dump(s, open(sp, "w")); return s
    try:
        r = subprocess.run(job["cmd"], capture_output=True, text=True, timeout=3600, cwd=os.path.dirname(RS),
                           env=dict(os.environ, PYTHONPATH=CH, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        txt = r.stdout + "\n" + r.stderr; code = r.returncode
    except subprocess.TimeoutExpired as e:
        txt = f"TIMEOUT {e}"; code = -9
    bp = os.path.splitext(job["out"])[0] + "_bundle.json"
    has = os.path.exists(job["out"]) and os.path.exists(bp)
    if code == 0 and has: cls = "ok"
    elif code == 2 and "QUARANTINE" in txt and "Traceback" not in txt and _stage0_reason(txt): cls = "stage0_stop"
    elif code == 3 and "ENVIRONMENT_" in txt: cls = "environment"
    else: cls = "crash"
    s.update(exit=code, **{"class": cls}, reason=_stage0_reason(txt) if cls == "stage0_stop" else None, seconds=round(time.time() - t0, 1),
             report=job["out"] if has else None, bundle=bp if has else None, tail=txt[-2500:] if cls != "ok" else txt[-400:])
    open(os.path.splitext(job["out"])[0] + ".log", "w").write(txt)
    json.dump(s, open(sp, "w")); return s


def make_job(r, arm, idir, refs_csv=None):
    out = f"{OUT}/{arm}/{r.series}/{r.gsm}.html"
    grn, red = idat_pair(r, idir)
    job = dict(series=r.series, gsm=r.gsm, arm=arm, out=out, cmd=None)
    if not grn or not red: return job
    c = [PY, RS, "--grn", grn, "--red", red, "--engine", "v3", "--specimen", r.specimen, "--array-type", r.plat, "--id", r.gsm,
         "--out", out, "--ledger", os.path.splitext(out)[0] + "_ledger.jsonl", "--covariate", f"series={r.series}"]
    if arm in ("pass1", "diag1"): c += ["--save-betas", f"{BET}/{r.gsm}.parquet"]
    if bool(r.no_intake) or arm.startswith("diag"): c += ["--no-intake"]
    else:
        if isinstance(r.sex, str) and r.sex: c += ["--sex", r.sex]
        if pd.notna(r.age): c += ["--age", str(r.age)]
    if refs_csv: c += ["--slide-ref-table", refs_csv]
    job["cmd"] = c; return job


def bundle_A(s):
    if s.get("class") != "ok": return None
    try: return (json.load(open(s["bundle"])).get("met_a") or {}).get("A")
    except Exception: return None


def tare_jobs(rows, sts, arm2, idir):
    """pass 2: >= 3 other healthy-reference arrays of the same series and specimen read in pass 1, same slide, else same series."""
    A = {s["gsm"]: bundle_A(s) for s in sts}
    R = rows.assign(A=rows.gsm.map(A)); jobs = []
    for r in R.itertuples():
        if r.A is None or pd.isna(r.A): continue
        pool = R[(R.gsm != r.gsm) & (R.specimen == r.specimen) & R.healthy & R.A.notna()]
        ref = pool[pool.slide == r.slide]; where = "same slide"
        if len(ref) < 3: ref = pool; where = "same series (batch)"
        if len(ref) < 3: continue
        p = f"{OUT}/refs/{arm2}/{r.series}/{r.gsm}.csv"; os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame({"A": ref.A.astype(float), "id": ref.gsm}).to_csv(p, index=False)
        j = make_job(r, arm2, idir, p); j["ref_scope"] = where; jobs.append(j)
    return jobs


def series_pipeline(ser, ex):
    try:
        rows = MAN[MAN.series == ser]
        t0 = time.time(); idir = download(ser); log("downloaded", ser, len(rows), f"{time.time()-t0:.0f}s")
        st1 = list(ex.map(run_one, [make_job(r, "pass1", idir) for r in rows.itertuples()]))
        dg = rows[[s["class"] == "stage0_stop" and any(k in str(s.get("reason", "")).lower() for k in ("manifest_invalid", "incomplete_manifest")) for s in st1]]
        std = list(ex.map(run_one, [make_job(r, "diag1", idir) for r in dg.itertuples()])) if len(dg) else []
        j2 = tare_jobs(rows, st1, "pass2", idir); jd2 = tare_jobs(dg, std, "diag2", idir) if len(dg) else []
        st2 = list(ex.map(run_one, j2 + jd2))
        c = pd.Series([s["class"] for s in st1]).value_counts().to_dict()
        log("done", ser, "pass1", c, "diag", len(std), "pass2", len(st2), f"{time.time()-t0:.0f}s")
        # copy as produced: reports + statuses of this series, then its beta vectors
        tg = f"{OUT}/chunk_{ser}.tgz"
        with tarfile.open(tg, "w:gz") as T:
            for arm in ("pass1", "diag1", "pass2", "diag2"):
                if os.path.isdir(f"{OUT}/{arm}/{ser}"): T.add(f"{OUT}/{arm}/{ser}", arcname=f"{arm}/{ser}")
        upload(tg, f"chunks/chunk_{ser}.tgz"); os.remove(tg)
        bt = f"{OUT}/betas_{ser}.tar"
        with tarfile.open(bt, "w") as T:
            for g in rows.gsm:
                p = f"{BET}/{g}.parquet"
                if os.path.exists(p): T.add(p, arcname=f"{g}.parquet")
        upload(bt, f"betas/betas_{ser}.tar"); os.remove(bt)
        if ser not in LOCAL: shutil.rmtree(f"{IDAT}/{ser}", ignore_errors=True)
        return ser, "ok"
    except Exception as e:
        log("SERIES FAILED", ser, type(e).__name__, str(e)[:300], traceback.format_exc()[-800:]); return ser, f"failed: {e}"


SEC = ["sec-reading-intake", "sec-composition", "sec-met-a", "sec-tare", "sec-noise-gate", "sec-cscore", "sec-red-flags", "sec-safeguards",
       "sec-troubleshooting", "sec-integrity", "sec-inventory", "sec-run-yourself", "sec-stages", "sec-toolkit"]
SAFE = ["rendered-claim scan", "formula self-test", "anchors", "deconvolver conformance", "atlas separability"]


def section_check(s):
    h = open(s["report"], encoding="utf-8").read(); b = json.load(open(s["bundle"]))
    miss = [x for x in SEC if f"id='{x}'" not in h] + [x for x in SAFE if f"<td>{x}</td>" not in h]
    tk = h.split("id='sec-toolkit'")[-1] if "id='sec-toolkit'" in h else ""
    if "NOT_BUILT" not in tk or not ("NOT_RUN" in tk or "PASS" in tk): miss.append("toolkit statuses")
    if b.get("intake") and "Stage 1 (recorded, not gated)" not in h: miss.append("Stage 1 line")
    if "red_flags" not in b: miss.append("bundle red_flags")
    sg = {x["check"]: x["result"] for x in b.get("safeguards", [])}
    return dict(sections_ok=not miss, sections_missing=";".join(miss), **{f"sg_{k.split()[0]}": v for k, v in sg.items()})


def flat(s, man_row):
    d = dict(series=s["series"], gsm=s["gsm"], arm=s["arm"], cls=s["class"], exit=s.get("exit"), stage0_reason=s.get("reason"), seconds=s.get("seconds"))
    if s["class"] == "ok":
        b = json.load(open(s["bundle"])); m = b.get("met_a") or {}; t = b.get("tare") or {}; it = b.get("intake") or {}; c = b.get("met_a_cscore") or {}
        d.update(stage0_verdict=it.get("stage0_verdict"), refusal=b.get("refusal"), platform=b.get("platform"), A=m.get("A"), state=m.get("state", m.get("reason")),
                 state_own_floor=m.get("state_own_floor"), f_neu=m.get("fraction"), n_sites=m.get("n_sites"), N=m.get("noise_index"), gate=m.get("noise_gate"),
                 A_rel=t.get("A_rel"), tare_state=t.get("state", t.get("reason")), n_refs=t.get("n_refs"), ref_median=t.get("reference_median"),
                 ref_sd=t.get("reference_spread_sd"), det_limit=t.get("detection_limit_pct_loss"), shift1=m.get("shift_per_1pct_loss"), C=c.get("C"),
                 ceiling=m.get("past_entropy_ceiling"), n_red_flags=len(b.get("red_flags", [])), chain_commit=(b.get("versions") or {}).get("chain_commit"),
                 chain_dirty=(b.get("versions") or {}).get("chain_dirty"))
        try: d.update(section_check(s))
        except Exception as e: d.update(sections_ok=False, sections_missing=f"check failed {e}")
    return d


def main():
    t0 = time.time()
    sers = [s for s in MAN.series.unique() if (ONLY is None or s in ONLY)]
    first = MAN[(MAN.series == "GSE250556")].iloc[0]
    log("warm-up (methylprep manifest) on", first.gsm)
    log(run_one(make_job(first, "pass1", download("GSE250556")))["class"])
    with ProcessPoolExecutor(NW) as ex, ThreadPoolExecutor(8) as tp:
        res = list(tp.map(lambda s: series_pipeline(s, ex), sers))
    log("series results", res)
    rows = []
    for p in glob.glob(f"{ST}/*/*.json"):
        s = json.load(open(p)); rows.append(flat(s, None))
    X = pd.DataFrame(rows); X.to_csv(f"{W}/readings_all.csv", index=False); shutil.copy(f"{W}/readings_all.csv", f"{OUT}/readings_all.csv")
    cr = [json.load(open(p)) for p in glob.glob(f"{ST}/*/*.json")]
    with open(f"{W}/crash_tails.txt", "w") as f:
        for s in cr:
            if s["class"] in ("crash", "no_idat"): f.write(f"===== {s['arm']} {s['series']} {s['gsm']} exit {s.get('exit')}\n{s.get('tail','')[-1500:]}\n")
    upload(f"{W}/readings_all.csv", "readings_all.csv"); upload(f"{W}/crash_tails.txt", "crash_tails.txt"); upload(f"{OUT}/driver.log", "driver.log")
    log("ALL DONE", f"{time.time()-t0:.0f}s", X.groupby(["arm", "cls"]).size().to_dict())


if __name__ == "__main__":
    main()
