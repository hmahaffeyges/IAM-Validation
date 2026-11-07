"""The Sequencing Atlas Globe at every CpG, deposited on Zenodo (DEV-SEQ-GLOBE-WG-01): run on a Google machine, never the laptop.
For each of the published atlas's 12 piece sets (Zenodo 10.5281/zenodo.23934666): fetch the set, check every piece against the deposit's
SHA256 manifest, build that set's stretch of the globe with seq_globe_wg.py slice (NSIDE 2048, one CpG to a pixel), pack the slices as
globe57_set<NN>_of_12.tar and upload it to the draft's file bucket, checking the MD5 Zenodo returns. Saves as it goes: a set already in the
draft at its full size is skipped ("already read and deposited"), so a reclaimed machine's relaunch resumes at the next set. Then the Globe
Viewer textures and the check report (committed), a SHA256 manifest of every file, and a README. The Zenodo key is read from Google's secret
store (secret zenodo-upload) at run time and never written anywhere. The record stays a draft: the author publishes.
Usage: python3 zenodo_globe_upload.py ZENODO_FILE_BUCKET_URL WORKDIR"""
import os, sys, glob, hashlib, subprocess, tarfile, shutil, time, requests
ZB, W = sys.argv[1].rstrip("/"), sys.argv[2]; os.makedirs(W, exist_ok=True)
MP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL = os.path.join(MP, "atlas", "tools", "seq_globe_wg.py")
INPUTS = os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv")
CHECK = os.path.join(MP, "doors", "data", "DEV_SEQ_GLOBE_WG_01")
ATLAS = "https://zenodo.org/records/23934666/files"
TOK = subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret=zenodo-upload"], capture_output=True, text=True, check=True).stdout.strip()
H = {"Authorization": "Bearer " + TOK}; del TOK

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()
def have():
    for t in range(6):
        try: r = requests.get(ZB, headers=H, timeout=120); r.raise_for_status(); return {f["key"]: f for f in r.json().get("contents", [])}
        except Exception as e: log("bucket list retry", e); time.sleep(30)
    raise SystemExit("cannot list the draft")
def get(url, path):
    for t in range(8):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for c in r.iter_content(8 << 20): f.write(c)
            return
        except Exception as e: log("fetch retry", os.path.basename(path), e); time.sleep(30 * (t + 1))
    raise SystemExit("fetch failed: " + url)
def put(path, name):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): md5.update(c)
    for t in range(8):
        try:
            with open(path, "rb") as f: r = requests.put(f"{ZB}/{name}", data=f, headers=H, timeout=7200)
            r.raise_for_status(); got = r.json().get("checksum", "").replace("md5:", "")
            if got == md5.hexdigest(): log("deposited", name, os.path.getsize(path)); return
            log("checksum mismatch, again", name)
        except Exception as e: log("upload retry", name, e); time.sleep(60 * (t + 1))
    raise SystemExit("upload failed: " + name)

man = os.path.join(W, "atlas_manifest.txt"); get(f"{ATLAS}/SHA256_MANIFEST.txt?download=1", man)
want = {}
for l in open(man):
    p = l.split()
    if len(p) >= 2: want[p[1]] = p[0]
for s in range(1, 13):
    name = f"globe57_set{s:02d}_of_12.tar"; done = have()
    if name in done and done[name]["size"] > 0: log(name, "already read and deposited"); continue
    T = f"pieces57_set{s:02d}_of_12.tar"; d = os.path.join(W, f"set{s:02d}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    get(f"{ATLAS}/{T}?download=1", os.path.join(d, T))
    with tarfile.open(os.path.join(d, T)) as tf: tf.extractall(os.path.join(d, "p"))
    os.remove(os.path.join(d, T)); P = sorted(glob.glob(os.path.join(d, "p", "**", "piece_*.parquet"), recursive=True))
    for f in P:
        k = [v for key, v in want.items() if key.startswith(T) and key.endswith("/" + os.path.basename(f))]
        if not k or k[0] != sha256(f): raise SystemExit("checksum refused: " + os.path.basename(f))
    log(name, "pieces checked", len(P))
    subprocess.run([sys.executable, TOOL, "slice", INPUTS, os.path.join(d, "out")] + P, check=True)
    tp = os.path.join(W, name)
    with tarfile.open(tp, "w") as tf:
        for f in sorted(glob.glob(os.path.join(d, "out", "*"))): tf.add(f, arcname=os.path.basename(f))
    put(tp, name); os.remove(tp); shutil.rmtree(os.path.join(d, "p"), ignore_errors=True)   # the slices stay for the textures below
# the textures and the check, rebuilt here from every set's slices (a set built by an earlier machine is taken back from the draft)
for s in range(1, 13):
    o = os.path.join(W, f"set{s:02d}", "out")
    if glob.glob(os.path.join(o, "*_slice_n2048.npz")): continue
    os.makedirs(o, exist_ok=True); tp = os.path.join(W, "back.tar"); name = f"globe57_set{s:02d}_of_12.tar"
    for t_ in range(8):
        try:
            with requests.get(f"{ZB}/{name}", headers=H, stream=True, timeout=600) as r:
                r.raise_for_status(); open(tp, "wb").writelines(r.iter_content(8 << 20))
            break
        except Exception as e: log("take-back retry", name, e); time.sleep(60)
    with tarfile.open(tp) as tf: tf.extractall(o)
    os.remove(tp); log(name, "taken back from the draft")
A = os.path.join(W, "assembled")
subprocess.run([sys.executable, TOOL, "assemble", A] + [os.path.join(W, f"set{s:02d}", "out") for s in range(1, 13)], check=True)
import json as _json
chk = _json.load(open(os.path.join(A, "globe_wg_check.json")))["check"]
if not chk.get("met"): raise SystemExit("the rebuilt globe does not match the current globe at the array sites: " + _json.dumps(chk))
done = have()
for f, p in (("globe_wg_textures_n256.npz", os.path.join(A, "globe_wg_textures_n256.npz")), ("globe_wg_check.json", os.path.join(A, "globe_wg_check.json")),
             ("seq_globe_wg_synthetic_run2.txt", os.path.join(CHECK, "seq_globe_wg_synthetic_run2.txt")), ("seq_globe_wg.py", TOOL)):
    if f not in done: put(p, f)
rd = os.path.join(W, "README.md"); open(rd, "w").write(open(os.path.join(os.path.dirname(__file__), "zenodo_globe_README.md")).read()); put(rd, "README.md")
done = have(); ms = os.path.join(W, "SHA256_MANIFEST.txt")
# the manifest lists each file's MD5 as Zenodo returns it and its size; the SHA256 of every slice is inside each set's own tar listing below
with open(ms, "w") as f:
    for k, v in sorted(done.items()): f.write(f"{v.get('checksum', '')}  {k}  {v['size']}\n")
put(ms, "CHECKSUMS.txt"); log("all 12 sets deposited, with the textures, the check, the README and the checksums; the record stays a draft")
