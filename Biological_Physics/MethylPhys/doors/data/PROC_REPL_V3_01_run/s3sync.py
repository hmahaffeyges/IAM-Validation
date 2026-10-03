#!/usr/bin/env python3
"""Copy GSE250556 IDATs and PROC-REPL-V3-01 outputs from the box to S3 as they appear (presigned POST policy; no credentials on the box).
Loops every 45 s until the main job's run.log says DONE, then one last pass."""
import os, sys, json, time, subprocess
POL = json.load(open("s3post.json")); ROOT = "/home/ubuntu/data/G_chain_tests/GSE250556"; OUT = f"{ROOT}/proc_repl_v3_01"; MAINW = sys.argv[1]
seen = {}; LOG = f"{ROOT}/s3_sync.log"
def put(path, kind, key):
    p = POL[kind]; cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--retry", "3", "-F", f"key={p['prefix']}{key}"]
    for k, v in p["fields"].items():
        if k != "key": cmd += ["-F", f"{k}={v}"]
    code = subprocess.run(cmd + ["-F", f"file=@{path}", p["url"]], capture_output=True, text=True).stdout.strip()
    open(LOG, "a").write(f"{code}\t{p['prefix']}{key}\n"); return code == "204"
def files():
    for f in sorted(os.listdir(f"{ROOT}/idat")): yield f"{ROOT}/idat/{f}", "idat", f"idat/{f}"
    for f in ("GSE250556_series_matrix.txt.gz", "RAW_tar_record.json"):
        if os.path.exists(f"{ROOT}/{f}"): yield f"{ROOT}/{f}", "idat", f
    for sub in ("", "reports/", "logs/"):
        d = f"{OUT}/{sub}"
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if os.path.isfile(d + f) and f != "s3_uploads.log": yield d + f, "res", sub + f
    for f in ("reports.tgz", "run.log"):
        if os.path.exists(f"{MAINW}/{f}"): yield f"{MAINW}/{f}", "res", f
def sweep():
    n = 0
    for path, kind, key in files():
        try: st = os.stat(path); sig = (st.st_size, st.st_mtime)
        except FileNotFoundError: continue
        if seen.get(path) == sig or time.time() - st.st_mtime < 5: continue
        if put(path, kind, key): seen[path] = sig; n += 1
    return n
while True:
    done = os.path.exists(f"{MAINW}/run.log") and "DONE" in open(f"{MAINW}/run.log").read()
    n = sweep(); print(time.strftime("%H:%M:%S"), "uploaded", n, "| total", len(seen), flush=True)
    if done: time.sleep(10); sweep(); break
    if os.path.exists(f"{MAINW}/run.log") and time.time() - os.path.getmtime(f"{MAINW}/run.log") > 3 * 3600: break
    time.sleep(45)
print("synced", len(seen), "| failures", sum(1 for l in open(LOG) if not l.startswith("204")), flush=True)
