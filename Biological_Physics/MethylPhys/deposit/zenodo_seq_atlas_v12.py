"""The IAM Sequencing Atlas, version 1.2 (2026-11-08, author: 'anything related to the atlas should be with THAT atlas DOI'). Into the atlas's
new-version draft (made from 10.5281/zenodo.24029328; every atlas file carried over by Zenodo), this adds:
  1. the Sequencing Atlas Globe, every file of its record 10.5281/zenodo.24029332, fetched from Zenodo and checked against that record's own md5
     before it is sent (names that would clash with the atlas's get the prefix in RENAME);
  2. the 57 cells' noise maps at every CpG (globe_wg_noise_maps_n256.npz, the output of .github/workflows/globe_noise.yml, GitHub run 42339899387,
     taken from the build bucket and checked by SHA256 NOISE_SHA256; its maps must be exactly the cells and maps of the committed noise record);
  3. the records that read them, as committed (doors/data/DEV_SEQ_GLOBE_WG_01): the noise record, the null test's, both synthetics' summaries;
  4. README_noise.md, FILES_v1.2.md (what was added and every renamed file) and CHECKSUMS_v1.2.txt (SHA256 and size of every added file).
Saves as it goes: a file already in the draft with the right md5 is skipped ('already read (in the draft)'); downloads are re-fetched from Zenodo.
Nothing is published here; the version is published after the draft's file list is checked. Usage: python3 zenodo_seq_atlas_v12.py GB DRAFT_BUCKET W"""
import os, sys, json, hashlib, subprocess, time
import numpy as np, requests
GB, ZA, W = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "doors", "data", "DEV_SEQ_GLOBE_WG_01")
GLOBE = 24029332; NOISE_SHA256 = "6226e6a464293e13ce48a23a4ec8f732d4088c47ff2886eabd87aeae6cbdfdd7"
RENAME = {"README.md": "README_globe.md", "README_megakaryocytes.md": "README_globe_megakaryocytes.md", "CHECKSUMS.txt": "CHECKSUMS_globe.txt",
          "CHECKSUMS_megakaryocytes.txt": "CHECKSUMS_globe_megakaryocytes.txt"}
TOK = os.environ.get("ZENODO_TOKEN") or subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret=zenodo-upload"], capture_output=True, text=True, check=True).stdout.strip()
H = {"Authorization": "Bearer " + TOK}; del TOK
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def digest(p, k):
    h = hashlib.new(k)
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()
def have():
    r = requests.get(ZA, headers=H, timeout=120); r.raise_for_status(); return {f["key"]: f.get("checksum", "") for f in r.json().get("contents", [])}
def put(p, name, done):
    md5 = digest(p, "md5")
    if done.get(name) == "md5:" + md5: log("already read (in the draft):", name); return
    for t in range(8):
        try:
            with open(p, "rb") as f: r = requests.put(f"{ZA}/{name}", data=f, headers=H, timeout=7200)
            r.raise_for_status(); assert r.json().get("checksum") == "md5:" + md5, "md5 differs"; log("sent, md5 checked:", name, os.path.getsize(p)); return
        except Exception as e: log(f"{name}: try {t + 1} failed: {str(e)[:200]}"); time.sleep(30 * (t + 1))
    raise SystemExit(f"{name}: not sent after 8 tries")
done = have(); added = []
# 1. the globe, file by file, each checked against its own record's md5 before it is sent
G = requests.get(f"https://zenodo.org/api/records/{GLOBE}", timeout=120).json()
for f in sorted(G["files"], key=lambda x: x["key"]):
    name = RENAME.get(f["key"], f["key"]); want = f["checksum"].split(":")[-1]
    if done.get(name) == "md5:" + want: log("already read (in the draft):", name); added.append((name, f["key"], None)); continue
    p = os.path.join(W, f["key"])
    for t in range(6):
        try:
            with requests.get(f["links"]["self"], stream=True, timeout=7200) as r:
                r.raise_for_status()
                with open(p, "wb") as o:
                    for c in r.iter_content(8 << 20): o.write(c)
            if digest(p, "md5") == want: break
            raise RuntimeError("md5 differs from the globe record's")
        except Exception as e: log(f"{f['key']}: fetch {t + 1} failed: {str(e)[:200]}"); time.sleep(30 * (t + 1))
    else: raise SystemExit(f"{f['key']}: not fetched")
    put(p, name, done); added.append((name, f["key"], p)); os.remove(p)
# 2. the noise maps, checked before they are sent
p = os.path.join(W, "globe_wg_noise_maps_n256.npz")
subprocess.run(["gcloud", "storage", "cp", "-q", f"gs://{GB}/seq_atlas/atlas_v12/globe_wg_noise_maps_n256.npz", p], check=True)
assert digest(p, "sha256") == NOISE_SHA256, "the noise maps are not the run's output"
Z = np.load(p); R = json.load(open(os.path.join(D, "globe_wg_noise.json"), encoding="utf-8"))["cells"]
want = {f"{c}|{m}" for c, v in R.items() for m in v if m.startswith("map")}   # map 1 only where a cell has two or more donors
assert set(Z.files) == want, sorted(set(Z.files) ^ want)[:6]
bad = [k for k in Z.files if Z[k].shape != (786432,) or not np.isfinite(Z[k]).any()]; assert not bad, bad[:4]
log("noise maps checked:", len(Z.files), "maps,", len(R), "cells"); put(p, "globe_wg_noise_maps_n256.npz", done); added.append(("globe_wg_noise_maps_n256.npz", None, p))
# 3. the records, as committed
for f in ("globe_wg_noise.json", "globe_wg_noise_summary.txt", "globe_wg_flip_summary.txt", "seq_globe_wg_noise_synthetic.txt", "seq_globe_wg_flip_synthetic.txt"):
    put(os.path.join(D, f), f, done); added.append((f, None, os.path.join(D, f)))
put(os.path.join(HERE, "zenodo_seq_atlas_v12_README_noise.md"), "README_noise.md", done); added.append(("README_noise.md", None, os.path.join(HERE, "zenodo_seq_atlas_v12_README_noise.md")))
# 4. the file list and the checksums of everything added
fl = os.path.join(W, "FILES_v1.2.md")
open(fl, "w").write("# Version 1.2: what was added\n\nThe atlas's files are unchanged from version 1.1. Added: the Sequencing Atlas Globe (every file of its record, "
    "10.5281/zenodo.24029332; see README_globe.md), the 57 cells' noise maps at every CpG and the records that read them (see README_noise.md).\n\n"
    "Renamed so they do not clash with the atlas's own files (contents unchanged, so each matches the globe record's checksum):\n\n"
    + "".join(f"- `{RENAME[k]}` is the globe record's `{k}`\n" for k in sorted(RENAME)) + "\n`CHECKSUMS_v1.2.txt` gives the SHA256 and size of every added file.\n")
put(fl, "FILES_v1.2.md", done)
cs = []
for name, orig, path in added:
    if path is None:   # sent by an earlier launch: fetched back from the globe record to take its SHA256
        path = os.path.join(W, orig); r = requests.get(f"https://zenodo.org/api/records/{GLOBE}/files/{orig}/content", stream=True, timeout=7200)
        with open(path, "wb") as o:
            for c in r.iter_content(8 << 20): o.write(c)
    cs.append(f"{digest(path, 'sha256')}  {name}  {os.path.getsize(path)}\n")
    if path.startswith(W): os.remove(path)
ck = os.path.join(W, "CHECKSUMS_v1.2.txt"); open(ck, "w").write("".join(sorted(cs, key=lambda l: l.split()[1]))); put(ck, "CHECKSUMS_v1.2.txt", done)
names = set(have()); need = {n for n, _, _ in added} | {"FILES_v1.2.md", "CHECKSUMS_v1.2.txt"}
miss = sorted(need - names); assert not miss, f"not in the draft: {miss}"
log("draft complete:", len(names), "files")
