"""The megakaryocyte column into the IAM Sequencing Atlas (version 1.1) and the Sequencing Atlas Globe (its new version), on a Google machine.
The 57 published cells are not touched: both new versions carry every file of the version before, unchanged, and add the 58th cell.

Atlas (draft of 10.5281/zenodo.24029328): the column's 29 pieces (1,000,000 CpGs each, built by run_sample.py --build seq-atlas from GSE87197
donors 1 and 2, donor 1 in half A, donor 2 in half B) and their build reports, packed in one tar with its own SHA256 manifest; the column's
inputs (mk_inputs.csv), its build file list (52 GEO files), the two donors' GRCh38-to-hg19 conversion reports, and its fresh test on platelet
DNA (GSE206818) with that test's synthetic.
Globe (draft of 10.5281/zenodo.24029332): from those same pieces, checked against the manifest just written, the column's globe at NSIDE 2048
(seq_globe_wg.py slice; one CpG to a pixel), its Globe Viewer texture at NSIDE 256 (assemble), and both its noise maps with their spectrum,
shuffle and sign-flip nulls (noise-part, noise-join; the X and Y chromosomes masked, from the hg19 CpG dictionary CPG_BED).

Saves as it goes: every output is put in the bucket (seq_atlas/mk_v11/) the moment it exists and a relaunch takes it back ('already read');
every Zenodo upload is skipped when the draft already holds that file at that size. The Zenodo key is read from Google's secret store
(secret zenodo-upload) at run time and never written. The drafts stay drafts: the laptop publishes after checking every file is there.
Usage: python3 zenodo_mk_v11.py GCS_BUCKET ATLAS_FILES_URL GLOBE_FILES_URL CPG_BED WORKDIR"""
import os, sys, glob, json, hashlib, subprocess, tarfile, time, requests
GB, ZA, ZG, CPG_BED, W = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/"), sys.argv[4], sys.argv[5]
MP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SA = os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas"); TOOLS = os.path.join(MP, "atlas", "tools")
sys.path.insert(0, TOOLS); sys.path.insert(0, os.path.join(MP, "chain")); import seq_globe_wg as G
TOK = subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret=zenodo-upload"], capture_output=True, text=True, check=True).stdout.strip()
H = {"Authorization": "Bearer " + TOK}; del TOK
B = f"gs://{GB}/seq_atlas/mk_v11"; os.makedirs(W, exist_ok=True)

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()
def saved(name):   # take back an output a reclaimed machine already saved
    p = os.path.join(W, name)
    if os.path.exists(p): return p
    if sh(f"gcloud storage cp -q {B}/{name} {p}").returncode == 0: log("already read:", name); return p
    return None
def save(p): sh(f"gcloud storage cp -q {p} {B}/")
def have(Z):
    r = requests.get(Z, headers=H, timeout=120); r.raise_for_status(); return {f["key"]: f["size"] for f in r.json().get("contents", [])}
def put(Z, path, name, done):
    n = os.path.getsize(path)
    if done.get(name) == n: log("already read (in the draft):", name); return
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): md5.update(c)
    for t in range(8):
        try:
            with open(path, "rb") as f: r = requests.put(f"{Z}/{name}", data=f, headers=H, timeout=3600)
            r.raise_for_status()
            if r.json().get("checksum", "") == "md5:" + md5.hexdigest(): log("sent, md5 checked:", name, n); return
            raise RuntimeError("md5 differs")
        except Exception as e:
            log(f"{name}: try {t + 1} failed: {str(e)[:200]}"); time.sleep(30 * (t + 1))
    raise SystemExit(f"{name}: not sent after 8 tries")
def det_tar(path, files, arc):   # same bytes every time it is built (sorted, mtime 0), so a relaunch's tar matches the draft's
    def f0(ti): ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""; return ti
    with tarfile.open(path, "w") as z:
        for f in sorted(files): z.add(f, arcname=f"{arc}/{os.path.basename(f)}", filter=f0)

# 1. the column's pieces, from the build bucket, packed with a SHA256 manifest
PT = "pieces_megakaryocytes_29.tar"; MAN = "SHA256_MANIFEST_megakaryocytes.txt"; pd_ = os.path.join(W, "pmk"); os.makedirs(pd_, exist_ok=True)
if sh(f"gcloud storage cp -q 'gs://{GB}/seq_atlas/pieces_mk/piece_*' {pd_}/").returncode != 0: raise SystemExit("pieces download failed")
pieces = sorted(glob.glob(os.path.join(pd_, "piece_*.parquet"))); reps = sorted(glob.glob(os.path.join(pd_, "piece_*.json")))
if len(pieces) != 29 or len(reps) != 29: raise SystemExit(f"pieces: {len(pieces)} parquet, {len(reps)} reports, not 29 each")
tp = os.path.join(W, PT); det_tar(tp, pieces + reps, "pieces_megakaryocytes")
mp = os.path.join(W, MAN)
with open(mp, "w") as f:
    for p in sorted(pieces + reps): f.write(f"{sha256(p)}  {PT}/pieces_megakaryocytes/{os.path.basename(p)}  {os.path.getsize(p)}\n")
    f.write(f"{sha256(tp)}  {PT}  {os.path.getsize(tp)}\n")
log("pieces packed:", len(pieces), round(os.path.getsize(tp) / 1e6, 1), "MB")
side = [os.path.join(SA, x) for x in ("mk_inputs.csv", "mk_build_files.csv", "mk_donor1_lift.json", "mk_donor2_lift.json", "mk_platelet_compare.json", "mk_platelet_synthetic.txt")]
for p in side:
    if not os.path.exists(p): raise SystemExit(f"missing committed file {p}")
with open(mp, "a") as f:
    for p in side: f.write(f"{sha256(p)}  {os.path.basename(p)}  {os.path.getsize(p)}\n")
done = have(ZA)
for p in [tp, mp] + side + [os.path.join(MP, "deposit", "zenodo_mk_v11_README_atlas.md")]:
    put(ZA, p, "README_megakaryocytes.md" if p.endswith("_README_atlas.md") else os.path.basename(p), done)
log("atlas files in the draft")

# 2. the globe's 58th cell, from the same pieces (checked against the manifest just written)
want = {l.split()[1].rsplit("/", 1)[-1]: l.split()[0] for l in open(mp) if "/pieces_megakaryocytes/" in l}
for p in pieces + reps:
    if want[os.path.basename(p)] != sha256(p): raise SystemExit(f"checksum refused: {p}")
log("pieces checked against the manifest:", len(want))
ST = "globe_megakaryocytes_slices.tar"; stp = saved(ST)
if not stp:
    sd = os.path.join(W, "slices"); rep = G.slice_(os.path.join(SA, "mk_inputs.csv"), sd, pieces)
    det_tar(os.path.join(W, ST), glob.glob(os.path.join(sd, "*")), "globe_megakaryocytes"); stp = os.path.join(W, ST); save(stp)
    json.dump(rep, open(os.path.join(W, "slice_report_megakaryocytes.json"), "w"), indent=1); save(os.path.join(W, "slice_report_megakaryocytes.json"))
    log("slices built:", sum(r["cpgs_placed"]["megakaryocytes"] for r in rep), "CpGs placed")
TX = "globe_wg_textures_megakaryocytes_n256.npz"
if not saved(TX):
    sd = os.path.join(W, "slices_x"); os.makedirs(sd, exist_ok=True)
    with tarfile.open(stp) as z: z.extractall(sd)
    td = os.path.join(W, "tex"); G.assemble([sd], td)
    os.replace(os.path.join(td, "globe_wg_textures_n256.npz"), os.path.join(W, TX)); save(os.path.join(W, TX)); log("texture built")
NJ = "globe_wg_noise_megakaryocytes.json"
if not saved(NJ):
    nd = os.path.join(W, "noise_part"); G.noise_part(nd, pieces, cpg_bed=CPG_BED)
    jd = os.path.join(W, "noise_join"); G.noise_join([nd], jd)
    os.replace(os.path.join(jd, "globe_wg_noise.json"), os.path.join(W, NJ)); save(os.path.join(W, NJ))
    os.replace(os.path.join(jd, "globe_wg_noise_maps_n256.npz"), os.path.join(W, "globe_wg_noise_maps_megakaryocytes_n256.npz")); save(os.path.join(W, "globe_wg_noise_maps_megakaryocytes_n256.npz"))
    log("noise maps built")
for n in ("slice_report_megakaryocytes.json", "globe_wg_noise_maps_megakaryocytes_n256.npz"): saved(n)
gf = [os.path.join(W, x) for x in (ST, TX, NJ, "globe_wg_noise_maps_megakaryocytes_n256.npz", "slice_report_megakaryocytes.json")]
cs = os.path.join(W, "CHECKSUMS_megakaryocytes.txt")
with open(cs, "w") as f:
    for p in gf: f.write(f"{sha256(p)}  {os.path.basename(p)}  {os.path.getsize(p)}\n")
done = have(ZG)
for p in gf + [cs, os.path.join(TOOLS, "seq_globe_wg.py"), os.path.join(MP, "deposit", "zenodo_mk_v11_README_globe.md")]:
    put(ZG, p, "README_megakaryocytes.md" if p.endswith("_README_globe.md") else os.path.basename(p), done)
log("globe files in the draft; both drafts complete")
