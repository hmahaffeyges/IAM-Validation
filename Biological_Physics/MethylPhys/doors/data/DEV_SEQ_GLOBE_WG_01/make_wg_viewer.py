"""The Globe Viewer's pictures of the Sequencing Atlas Globe at every CpG (2026-11-07). Fetches the published textures from Zenodo
(10.5281/zenodo.24029332: globe_wg_textures_n256.npz, globe_wg_textures_megakaryocytes_n256.npz), checks each against the record's own
checksums, and draws them by the one command (run_sample.py --build seq-wg-sphere). Usage: python3 make_wg_viewer.py OUTDIR"""
import os, sys, hashlib, subprocess, urllib.request, json, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "_tex"); os.makedirs(W, exist_ok=True)
R = json.load(urllib.request.urlopen("https://zenodo.org/api/records/24029332", timeout=120)); F = {f["key"]: f for f in R["files"]}
paths = []
for k in ("globe_wg_textures_n256.npz", "globe_wg_textures_megakaryocytes_n256.npz"):
    p = os.path.join(W, k)
    for t in range(5):
        try: urllib.request.urlretrieve(F[k]["links"]["self"], p); break
        except Exception as e: print("retry", k, type(e).__name__, flush=True); time.sleep(30 * (t + 1))
    h = hashlib.md5(open(p, "rb").read()).hexdigest(); want = F[k]["checksum"].split(":")[-1]
    assert h == want, f"{k}: checksum {h} is not the record's {want}"; print(k, "checked", flush=True); paths.append(p)
r = subprocess.run([sys.executable, RS, "--build", "seq-wg-sphere", "--build-input", ",".join(paths), "--build-out", os.path.join(out, "seq_wg")], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1500:]); [os.remove(p) for p in paths]
if r.returncode == 0:
    import tarfile, shutil
    with tarfile.open(os.path.join(out, "seq_wg.tar.gz"), "w:gz") as t: t.add(os.path.join(out, "seq_wg"), arcname="seq_wg")
    shutil.copy(os.path.join(out, "seq_wg", "index.json"), os.path.join(out, "seq_wg_index.json")); shutil.rmtree(os.path.join(out, "seq_wg"))
sys.exit(r.returncode)
