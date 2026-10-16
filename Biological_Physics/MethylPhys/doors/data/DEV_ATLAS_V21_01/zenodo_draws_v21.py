"""Atlas v2.1 fit files to a PRIVATE Zenodo draft (author 2026-10-16: raw files go to Zenodo and stay in S3). Never publishes.
Zenodo holds at most 100 files per record, so the 2,800 block files go as 10 tar bundles of 70 blocks each (all four files of each block),
plus output_manifest_v21.json (every file's SHA-256) and a README. Each bundle is built, uploaded, checked against Zenodo's MD5, then deleted.
The token is read from ./zt and the file is deleted at once; it is never printed or written anywhere else.
Usage: python zenodo_draws_v21.py <blocks dir> <output_manifest_v21.json>"""
import os, sys, json, hashlib, tarfile, requests
src, man = sys.argv[1:3]; tok = open("zt").read().strip(); os.remove("zt"); H = {"Authorization": "Bearer " + tok}; API = "https://zenodo.org/api/deposit/depositions"
meta = {"metadata": {"title": "IAM-Atlas v2.1: posterior draws and per-block outputs of the Bayesian fit (79 cell types, 813,951 sites)", "upload_type": "dataset",
        "description": "Every file the IAM-Atlas v2.1 fit wrote (700 blocks: per-site summaries, 100 thinned posterior draws of each cell's mean and donor spread, priors, sampler warnings), in 10 tar bundles, with each file's SHA-256 in output_manifest_v21.json. Draft deposited 2026-10-16; to be published with the v2.1 release.",
        "creators": [{"name": "Mahaffey, Heath W.", "affiliation": "IAMPerformance"}], "access_right": "open", "license": "cc-by-4.0"}}
d = requests.post(API, headers=H, json=meta, timeout=120); d.raise_for_status(); d = d.json(); bucket = d["links"]["bucket"]; out = dict(deposition=d["id"], state="draft (unpublished)", files=[])
def put(path, name):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): md5.update(b)
    with open(path, "rb") as f: r = requests.put(f"{bucket}/{name}", data=f, headers=H, timeout=7200)
    r.raise_for_status(); z = r.json(); ok = z.get("checksum", "").replace("md5:", "") == md5.hexdigest()
    out["files"].append(dict(name=name, bytes=os.path.getsize(path), md5=md5.hexdigest(), zenodo_md5_match=ok)); print(name, os.path.getsize(path), "match" if ok else "MISMATCH", flush=True)
open("README_draws.md", "w").write("# IAM-Atlas v2.1 fit files\n\nBundles block_00000-00069.tar ... block_00630-00699.tar each hold every file of 70 blocks: block_NNNNN.parquet (per-site summary), "
    "block_NNNNN_draws.npz (100 posterior draws of mean and donor spread, float16), block_NNNNN_prior.parquet, block_NNNNN_warnings.json. Check any file against output_manifest_v21.json (SHA-256).\n")
put("README_draws.md", "README.md"); put(man, "output_manifest_v21.json")
F = sorted(os.listdir(src))
for g in range(10):
    lo, hi = 70 * g, 70 * g + 69; name = f"block_{lo:05d}-{hi:05d}.tar"
    with tarfile.open(name, "w") as t:
        for f in F:
            if lo <= int(f[6:11]) <= hi: t.add(os.path.join(src, f), arcname=f)
    put(name, name); os.remove(name)
out["all_match"] = all(f["zenodo_md5_match"] for f in out["files"]); out["n_files"] = len(out["files"])
json.dump(out, open("zenodo_draws_v21.json", "w"), indent=1); print("DONE", out["deposition"], out["n_files"], "files, all match:", out["all_match"])
