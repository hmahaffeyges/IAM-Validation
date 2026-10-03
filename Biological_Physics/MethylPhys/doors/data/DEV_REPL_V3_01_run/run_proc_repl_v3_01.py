#!/usr/bin/env python3
"""PROC-REPL-V3-01 box runner. GSE250556 (64 EPIC whole-blood technical replicates), chain v3 at commit 87cfa65, unchanged.
Per array: IDAT pair -> run_sample.py --engine v3 (intake -> calibration -> composition -> Met-A -> Stage T -> noise gate -> report).
Pass 1: no references (untared A). Pass 2: --slide-ref-table = pass-1 records of the other GSE250556 arrays (the array itself excluded):
same slide when >= 3 such arrays have an A, else the whole GSE250556 batch (the pre-registered rule). Stage T = median tare (nothing fitted).
Harness = chain_tests/chain_batch.py (RUN3) with these differences, all set before any array was read:
  * data: GSE250556 only, RAW tar + series matrix fetched here; manifest built from the series matrix
  * reference choice: same slide (>=3) else batch, as PROC_REPL_V3_01_PREREG.md states; chain_batch.py's '>= 20 records -> whole batch'
    branch existed for the removed fitted tare and is not used
  * S3 copies of IDATs and every per-array output as they are produced (presigned POST policy in s3post.json; no credentials on the box)
"""
import os, sys, re, json, gzip, subprocess, tarfile, time, hashlib, pandas as pd, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
W = os.getcwd(); PY = "/home/ubuntu/env/bin/python"
tarfile.open("chain_v3_87cfa65.tgz").extractall("bio/MethylPhys"); CH = f"{W}/bio/MethylPhys/chain"
ROOT = "/home/ubuntu/data/G_chain_tests/GSE250556"; D = f"{ROOT}/idat"; os.makedirs(D, exist_ok=True)
OUT = f"{ROOT}/proc_repl_v3_01"; os.makedirs(f"{OUT}/reports", exist_ok=True); os.makedirs(f"{OUT}/logs", exist_ok=True)
POL = json.load(open("s3post.json")); NP = int(os.environ.get("NP", "32"))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def s3put(path, kind="res", sub=""):
    """Upload one file under the policy's prefix (POST form upload). Returns True on HTTP 204/201."""
    p = POL[kind]; key = p["prefix"] + sub + os.path.basename(path)
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--retry", "3", "-F", f"key={key}"]
    for k, v in p["fields"].items():
        if k != "key": cmd += ["-F", f"{k}={v}"]
    cmd += ["-F", f"file=@{path}", p["url"]]
    try: code = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout.strip()
    except Exception as e: code = f"err {e}"
    ok = code in ("204", "201", "200")
    with open(f"{OUT}/s3_uploads.log", "a") as fh: fh.write(f"{code}\t{key}\n")
    return ok

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

# ---- 1. data: RAW tar and series matrix (GEO) -------------------------------------------------------------------------------------
RAW = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250556/suppl/GSE250556_RAW.tar"
SM = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250556/matrix/GSE250556_series_matrix.txt.gz"
tp = f"{ROOT}/GSE250556_RAW.tar"
if not os.path.exists(f"{ROOT}/RAW_extracted.ok"):
    for a in range(6):
        rc = os.system(f"curl -s -f -L -C - --retry 3 -o {tp} {RAW}")
        if rc == 0 and tarfile.is_tarfile(tp): break
        time.sleep(20 * (a + 1))
    with tarfile.open(tp) as t: names = t.getnames(); t.extractall(D)
    json.dump({"url": RAW, "bytes": os.path.getsize(tp), "sha256": sha(tp), "members": names}, open(f"{ROOT}/RAW_tar_record.json", "w"), indent=1)
    open(f"{ROOT}/RAW_extracted.ok", "w").write("ok")
log("RAW tar", json.load(open(f"{ROOT}/RAW_tar_record.json"))["bytes"], "bytes")
smp = f"{ROOT}/GSE250556_series_matrix.txt.gz"
for a in range(6):
    if os.path.exists(smp) and os.path.getsize(smp) > 1000: break
    os.system(f"curl -s -f -L --retry 3 -o {smp} {SM}"); time.sleep(5)
idats = sorted(f for f in os.listdir(D) if re.search(r"_(Grn|Red)\.idat(\.gz)?$", f))
with ThreadPoolExecutor(8) as ex: up = list(ex.map(lambda f: s3put(f"{D}/{f}", "idat", "idat/"), idats))
for f in (smp, f"{ROOT}/RAW_tar_record.json"): up.append(s3put(f, "idat"))
log("IDATs on box", len(idats), "| S3 uploads ok", sum(up), "of", len(up))

# ---- 2. manifest from the series matrix -------------------------------------------------------------------------------------------
rows = {}
with gzip.open(smp, "rt", errors="replace") as fh:
    for line in fh:
        if not line.startswith("!Sample_"): continue
        k, *v = line.rstrip("\n").split("\t"); v = [x.strip('"') for x in v]
        rows.setdefault(k, []).append(v)
gsms = rows["!Sample_geo_accession"][0]; n = len(gsms)
M = pd.DataFrame({"gsm": gsms, "title": rows["!Sample_title"][0], "source": rows.get("!Sample_source_name_ch1", [[""] * n])[0]})
for ch in rows.get("!Sample_characteristics_ch1", []):
    for i, x in enumerate(ch):
        if ":" in x: kk, vv = x.split(":", 1); M.loc[i, "ch_" + kk.strip().lower().replace(" ", "_")] = vv.strip()
def gfile(g):
    f = [x for x in idats if x.startswith(g + "_") and "_Grn.idat" in x]; return f[0] if f else None
M["grn"] = M.gsm.map(gfile)
M["sentrix"] = M.grn.str.extract(r"_(\d{10,12}_R\d\dC\d\d)_Grn")[0]; M["slide"] = M.sentrix.str.split("_").str[0]
def pick(r, keys):
    for c in M.columns:
        if c.startswith("ch_") and any(k in c for k in keys) and isinstance(r[c], str) and r[c]: return r[c]
    return None
M["sex_d"] = M.apply(lambda r: pick(r, ("sex", "gender")), axis=1)
M["age_d"] = M.apply(lambda r: pick(r, ("age",)), axis=1)
M["specimen"] = "whole blood"
M.to_csv(f"{OUT}/manifest_from_series_matrix.csv", index=False); s3put(f"{OUT}/manifest_from_series_matrix.csv")
log("manifest", len(M), "samples | with IDAT", M.grn.notna().sum(), "| columns", list(M.columns))

# ---- 3. chain ---------------------------------------------------------------------------------------------------------------------
def run(args):
    gsm, refs = args; r = M[M.gsm == gsm].iloc[0]
    tag = "_tared" if refs is not None else ""; out = f"{OUT}/reports/{gsm}{tag}.html"; bp = out.replace(".html", "_bundle.json")
    if not isinstance(r.grn, str): return dict(gsm=gsm, status="no IDAT pair in RAW tar")
    g = f"{D}/{r.grn}"; rd = g.replace("_Grn", "_Red")
    if not os.path.exists(rd): return dict(gsm=gsm, status="Red IDAT missing")
    cmd = [PY, f"{CH}/MethylPhys_Interface/run_sample.py", "--grn", g, "--red", rd, "--engine", "v3", "--specimen", r.specimen,
           "--array-type", "EPIC_v1", "--out", out, "--id", gsm, "--ledger", f"{OUT}/reports/ledger{tag}.jsonl"]
    if isinstance(r.sex_d, str) and r.sex_d[:1].upper() in "MF": cmd += ["--sex", r.sex_d[:1].upper()]
    try: cmd += ["--age", str(float(re.findall(r"[\d.]+", str(r.age_d))[0]))]
    except Exception: pass
    if refs is not None:
        rp = f"{OUT}/reports/{gsm}_refs.csv"; refs.to_csv(rp, index=False); cmd += ["--slide-ref-table", rp]; s3put(rp, sub="reports/")
    if not os.path.exists(bp):
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=f"{CH}/MethylPhys_Interface", env=dict(os.environ, PYTHONPATH=CH, OMP_NUM_THREADS="1"))
        lp = f"{OUT}/logs/{gsm}{tag}.log"; open(lp, "w").write(" ".join(cmd) + f"\n# exit {p.returncode}\n" + p.stdout + "\n--- stderr ---\n" + p.stderr)
        s3put(lp, sub="logs/"); tail = (p.stdout + p.stderr)[-600:]; rc = p.returncode
    else:
        tail = "(bundle already present; not rerun)"; rc = 0
    if not os.path.exists(bp): return dict(gsm=gsm, status="no bundle", exit=rc, tail=tail)
    for f in (bp, out): s3put(f, sub="reports/")
    o = json.load(open(bp)); m = o.get("met_a") or {}; c = o.get("met_a_cscore") or {}; t = o.get("tare") or {}; it = o.get("intake") or {}
    fr = (o.get("composition") or {}).get("fractions") or {}
    return dict(gsm=gsm, status="ok", exit=rc, refusal=o.get("refusal"), intake=it.get("stage0_verdict"), call_rate=it.get("call_rate_status"),
                A=m.get("A"), reason=m.get("reason"), state=m.get("state"), f_neu=fr.get("NEU"), n_sites=m.get("n_sites"), C=c.get("C"),
                N=m.get("noise_index"), noise_gate=m.get("noise_gate"), N_max=m.get("noise_gate_N_max"),
                A_rel=t.get("A_rel"), tare=t.get("state", t.get("reason")), n_refs=t.get("n_refs"), n_self_excluded=t.get("n_self_excluded"),
                ref_median=t.get("reference_median"), ref_sd=t.get("reference_spread_sd"), det_limit=t.get("detection_limit_pct_loss"),
                tare_method=t.get("method"), past_ceiling=m.get("past_entropy_ceiling"))

todo = M.gsm.tolist()
log("serial first array (methylprep manifest fetch):", todo[0]); first = run((todo[0], None)); log(first.get("status"), first.get("A"))
with mp.get_context("fork").Pool(NP) as P: R1 = pd.DataFrame([first] + P.map(run, [(g, None) for g in todo[1:]]))
bad = R1[R1.status != "ok"].gsm.tolist()
if bad:
    log("pass 1 serial retry", bad); R1 = pd.concat([R1[R1.status == "ok"], pd.DataFrame([run((g, None)) for g in bad])], ignore_index=True)
R1.to_csv(f"{OUT}/pass1.csv", index=False); s3put(f"{OUT}/pass1.csv"); log("pass1", R1.status.value_counts().to_dict(), "| A present", R1.A.notna().sum())

Y = M.merge(R1, on="gsm", how="left"); jobs = []; refrule = {}
for _, r in Y.iterrows():
    if pd.isna(r.A): continue
    ref = Y[(Y.gsm != r.gsm) & Y.A.notna()]; s = ref[ref.slide == r.slide]
    if len(s) >= 3: ref, rule = s, "same slide"
    else: rule = "same batch (GSE250556)"
    refrule[r.gsm] = (rule, len(ref))
    if len(ref) >= 3: jobs.append((r.gsm, ref[["gsm", "A", "f_neu", "N"]].rename(columns={"gsm": "id"}).reset_index(drop=True)))
with mp.get_context("fork").Pool(NP) as P: R2 = pd.DataFrame(P.map(run, jobs))
bad2 = R2[R2.status != "ok"].gsm.tolist() if len(R2) else []
if bad2:
    jd = dict(jobs); log("pass 2 serial retry", bad2)
    R2 = pd.concat([R2[R2.status == "ok"], pd.DataFrame([run((g, jd[g])) for g in bad2])], ignore_index=True)
R2.to_csv(f"{OUT}/pass2.csv", index=False); s3put(f"{OUT}/pass2.csv")
Y["ref_rule"] = Y.gsm.map(lambda g: refrule.get(g, (None, 0))[0]); Y["ref_pool_size"] = Y.gsm.map(lambda g: refrule.get(g, (None, 0))[1])
k = ["gsm", "status", "exit", "state", "noise_gate", "A_rel", "tare", "n_refs", "n_self_excluded", "ref_median", "ref_sd", "det_limit", "tare_method", "tail"]
Y = Y.merge(R2[[c for c in k if c in R2.columns]].add_suffix("_p2").rename(columns={"gsm_p2": "gsm"}), on="gsm", how="left")
Y.to_csv(f"{OUT}/proc_repl_v3_01_box_readings.csv", index=False); s3put(f"{OUT}/proc_repl_v3_01_box_readings.csv")
os.system(f"cd {OUT} && tar czf {W}/reports.tgz reports logs && cp {OUT}/*.csv {OUT}/s3_uploads.log {W}/ && cp {ROOT}/RAW_tar_record.json {smp} {W}/")
s3put(f"{W}/reports.tgz"); s3put(f"{OUT}/s3_uploads.log")
log("pass2", R2.status.value_counts().to_dict() if len(R2) else {}, "| tared", Y.A_rel_p2.notna().sum(), "| S3 failures",
    sum(1 for l in open(f"{OUT}/s3_uploads.log") if not l.startswith(("204", "201", "200")))); log("DONE")
