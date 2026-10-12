"""Fetch IAM-Atlas v2 from its public Zenodo record and verify every file (no account, no cloud access needed).
Downloads the 7 block archives listed in the record, checks each archive against ARCHIVES_SHA256.txt in the record, unpacks them, and checks
all 2,100 files against ../v2/postbuild/records/ATLAS_V2_OUTPUT_MANIFEST.json (SHA-256 and size). Prints 'atlas v2 verified: 2100 files'.
Usage: python3 fetch_atlas.py OUTDIR 23325865 [ARCHIVE ...]   (record of DOI 10.5281/zenodo.23325865; naming archives fetches and checks
only those, e.g. iam_atlas_v2_blocks_00600-00699.tar)"""
import os, sys, json, hashlib, tarfile, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(HERE, "..", "v2", "postbuild", "records", "ATLAS_V2_OUTPUT_MANIFEST.json")))
OUT = sys.argv[1]; REC = sys.argv[2] if len(sys.argv) > 2 else None; ONLY = sys.argv[3:]; os.makedirs(OUT, exist_ok=True)
if REC is None: raise SystemExit("give the Zenodo record id (README, section 'Get it'); the deposit is not yet published if none is listed")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{REC}"))
files = {f["key"]: f["links"]["self"] for f in meta["files"]}
def get(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p): urllib.request.urlretrieve(files[name], p)
    return p
want = dict(l.split()[::-1] for l in open(get("ARCHIVES_SHA256.txt")).read().strip().split("\n"))
if ONLY: want = {k: v for k, v in want.items() if k in ONLY}; assert want, "no such archive in the record"
for name, h in sorted(want.items()):
    p = get(name); assert sha(p) == h, f"{name}: checksum mismatch"
    with tarfile.open(p) as t: t.extractall(OUT)
    print(name, "ok", flush=True)
def inside(k):
    b = int(k.split("_")[1].split(".")[0]); return any(int(a.split("_")[-1][:5]) <= b <= int(a.split("-")[-1][:5]) for a in want)
MAN = {k: v for k, v in MAN.items() if inside(k)}
bad = [k for k, v in MAN.items() if not os.path.exists(os.path.join(OUT, "blocks_v2", k)) or os.path.getsize(os.path.join(OUT, "blocks_v2", k)) != v["bytes"]
       or sha(os.path.join(OUT, "blocks_v2", k)) != v["sha256"]]
if bad: raise SystemExit(f"{len(bad)} files do not match the manifest, e.g. {bad[:3]}")
print(f"atlas v2 verified: {len(MAN)} files" + (" (archives named)" if ONLY else ""))
