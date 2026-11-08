"""The Globe Viewer's layers of the Sequencing Atlas Globe (2026-11-08, author: one layer per tool, 'Imagine its the CMB'). Fetches from the
IAM Sequencing Atlas v1.2 record (Zenodo 10.5281/zenodo.24051846) the textures and the noise maps of all 58 cells, checks each against the
record's own md5, and draws the layers by the one command (run_sample.py --build seq-wg-layers): the slider codes, noise maps 1 and 2, the
null and the mask. Packs them into seq_wg_layers.tar.gz (the workflow uploads top-level files). Usage: python3 make_wg_layers.py OUTDIR"""
import os, sys, hashlib, subprocess, urllib.request, json, time, tarfile, shutil
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "_in"); os.makedirs(W, exist_ok=True)
R = json.load(urllib.request.urlopen("https://zenodo.org/api/records/24051846", timeout=120)); F = {f["key"]: f for f in R["files"]}
def get(k):
    p = os.path.join(W, k)
    for t in range(5):
        try: urllib.request.urlretrieve(F[k]["links"]["self"], p); break
        except Exception as e: print("retry", k, type(e).__name__, flush=True); time.sleep(30 * (t + 1))
    h = hashlib.md5(open(p, "rb").read()).hexdigest(); want = F[k]["checksum"].split(":")[-1]
    assert h == want, f"{k}: checksum {h} is not the record's {want}"; print(k, "checked", flush=True); return p
tex = [get(k) for k in ("globe_wg_textures_n256.npz", "globe_wg_textures_megakaryocytes_n256.npz")]
noi = [get(k) for k in ("globe_wg_noise_maps_n256.npz", "globe_wg_noise_maps_megakaryocytes_n256.npz")]
o = os.path.join(out, "seq_wg_layers")
r = subprocess.run([sys.executable, RS, "--build", "seq-wg-layers", "--build-input", ",".join(tex), "--build-samples", ",".join(noi), "--build-out", o], capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1500:]); shutil.rmtree(W)
if r.returncode == 0:
    with tarfile.open(os.path.join(out, "seq_wg_layers.tar.gz"), "w:gz") as t: t.add(o, arcname="seq_wg_layers")
    shutil.rmtree(o)
sys.exit(r.returncode)
