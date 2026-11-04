"""The IAM Sequencing Atlas Zenodo deposit (draft 23934666, reserved DOI 10.5281/zenodo.23934666): run on a Google machine, never the laptop.
Streams the 113 merged 57-cell pieces from the project's bucket (seq_atlas/pieces57/) and the runtime tables the chain reads (committed in
chain/Runtime Matrices/Seq_Atlas, packed here) to the draft's file bucket one file at a time, with retries, and checks the MD5 Zenodo returns
against the bytes sent. The Zenodo key is read from Google's secret store (secret zenodo-upload) at run time and never written anywhere. The
record stays a draft: the author publishes. Usage: python3 zenodo_seq_atlas_upload.py BUCKET ZENODO_FILE_BUCKET_URL WORKDIR"""
import os, sys, json, glob, hashlib, subprocess, tarfile, time, requests
GBK, ZB, W = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]; os.makedirs(W, exist_ok=True)
MP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOK = subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret=zenodo-upload"], capture_output=True, text=True, check=True).stdout.strip()
H = {"Authorization": "Bearer " + TOK}; del TOK

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): h.update(c)
    return h.hexdigest()

def have():
    r = requests.get(ZB, headers=H, timeout=120); r.raise_for_status()
    return {f["key"]: f for f in r.json().get("contents", [])}

def put(path, name, done):
    n = os.path.getsize(path)
    if name in done and done[name]["size"] == n: return "already there"
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""): md5.update(c)
    for t in range(8):
        try:
            with open(path, "rb") as f: r = requests.put(f"{ZB}/{name}", data=f, headers=H, timeout=3600)
            r.raise_for_status(); got = r.json().get("checksum", "")
            if got == "md5:" + md5.hexdigest(): return "sent, md5 checked"
            raise RuntimeError(f"md5 differs: {got}")
        except Exception as e:
            print(f"{name}: try {t + 1} failed: {str(e)[:200]}", flush=True); time.sleep(30 * (t + 1))
    raise SystemExit(f"{name}: not sent after 8 tries")

done = have(); man = []
tb = os.path.join(W, "seq_atlas_runtime_tables.tgz")
with tarfile.open(tb, "w:gz") as z: z.add(os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas"), arcname="Seq_Atlas")
small = [tb] + [os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", n) for n in ("seq_atlas_inputs.csv", "gran_inputs.csv", "atlas57_merge.json", "atlas57_summary.txt", "donor_traits.csv")]
for p in small:
    if os.path.exists(p): man.append((os.path.basename(p), os.path.getsize(p), sha256(p))); print(os.path.basename(p), put(p, os.path.basename(p), done), flush=True)
keys = sorted(l.strip() for l in subprocess.run(["gcloud", "storage", "ls", f"gs://{GBK}/seq_atlas/pieces57/"], capture_output=True, text=True, check=True).stdout.splitlines() if l.strip().endswith(".parquet"))
print(len(keys), "pieces", flush=True)
for k in keys:
    name = os.path.basename(k); p = os.path.join(W, name)
    subprocess.run(["gcloud", "storage", "cp", "-q", k, p], check=True, capture_output=True)
    man.append((name, os.path.getsize(p), sha256(p))); print(name, put(p, name, done), flush=True); os.remove(p)
mf = os.path.join(W, "SHA256_MANIFEST.txt")
open(mf, "w").write("".join(f"{s}  {n}  {b}\n" for n, b, s in man))
print("SHA256_MANIFEST.txt", put(mf, "SHA256_MANIFEST.txt", have()), flush=True)
fin = have(); print(f"done: {len(fin)} files, {sum(f['size'] for f in fin.values()) / 1e9:.2f} GB in the draft", flush=True)
