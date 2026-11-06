"""The whole IAM-Validation repository on Zenodo (author 2026-11-06: "if something ever happened to the github repo I have none of this to
start over with"): run on a Google machine, never the laptop. Clones main fresh from GitHub, packs every tracked file at that commit
(git archive, the repository exactly as GitHub holds it: the chain, every test record, the canon, the book, the website, the A-score
database, the logs), adds the full git history as a bundle (git bundle --all, so every past commit can be restored), and the two alignment
indexes from the repository's releases (hg19 and mm10 bwa-meth, in their release parts with SHA256SUMS). Each file goes to the Zenodo
draft one at a time with retries, and the MD5 Zenodo returns is checked against the bytes sent; a SHA-256 manifest goes up last. The Zenodo
key is read from Google's secret store (secret zenodo-upload) at run time and never written. The record stays a draft: the author publishes.
Usage: python3 zenodo_repo_snapshot.py ZENODO_FILE_BUCKET_URL WORKDIR"""
import os, sys, hashlib, subprocess, time, requests
ZB, W = sys.argv[1].rstrip("/"), sys.argv[2]; os.makedirs(W, exist_ok=True)
REPO = "https://github.com/hmahaffeyges/IAM-Validation"; API = "https://api.github.com/repos/hmahaffeyges/IAM-Validation"
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

def run(*a, cwd=None): return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

done = have(); man = []
src = os.path.join(W, "IAM-Validation")
if not os.path.isdir(src): run("git", "clone", "--quiet", REPO, src)
sha = run("git", "rev-parse", "HEAD", cwd=src); day = run("git", "show", "-s", "--format=%cs", "HEAD", cwd=src)
files = [(f"IAM-Validation_{day}_{sha[:8]}.tar.gz", ["git", "archive", "--format=tar.gz", f"--prefix=IAM-Validation/", "-o"]),
         (f"IAM-Validation_{day}_{sha[:8]}_history.bundle", ["git", "bundle", "create"])]
for name, cmd in files:
    p = os.path.join(W, name)
    if not os.path.exists(p): run(*(cmd + [p] + (["HEAD"] if cmd[1] == "archive" else ["--all"])), cwd=src)
    man.append((name, os.path.getsize(p), sha256(p))); print(name, put(p, name, done), flush=True); os.remove(p)
for tag in ("hg19-bwameth-index-v1", "mm10-bwameth-index-v1"):
    rel = requests.get(f"{API}/releases/tags/{tag}", timeout=60).json()
    for a in rel["assets"]:
        name = f"{tag}__{a['name']}"; p = os.path.join(W, name)
        if name in done and done[name]["size"] == a["size"]: man.append((name, a["size"], "(already there)")); print(name, "already there", flush=True); continue
        run("curl", "-sfL", "--retry", "5", "-o", p, a["browser_download_url"])
        man.append((name, os.path.getsize(p), sha256(p))); print(name, put(p, name, done), flush=True); os.remove(p)
mp = os.path.join(W, "SHA256_MANIFEST.txt")
open(mp, "w").write(f"# IAM-Validation repository snapshot: commit {sha} ({day}), from {REPO}\n# file\tbytes\tsha256\n"
                    + "".join(f"{n}\t{s}\t{h}\n" for n, s, h in man))
print("SHA256_MANIFEST.txt", put(mp, "SHA256_MANIFEST.txt", have()), flush=True)
print(f"done: commit {sha}, {len(man)} files", flush=True)
