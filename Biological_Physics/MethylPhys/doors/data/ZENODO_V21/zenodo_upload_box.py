"""IAM-Atlas v2.1 Zenodo deposit, large files (2026-10-17): run on an AWS machine, never the laptop. Streams each file from S3 to its Zenodo
draft bucket, one at a time, with retries, and checks the MD5 Zenodo returns against the MD5 of the bytes sent. The Zenodo key is read from
the AWS secret store (/methylphys/zenodo-upload) and never written anywhere. Both records stay drafts: the author publishes.
  record A (IAM-Atlas v2.1, reserved 10.5281/zenodo.23441999): the assembled atlas
  record B (raw fit, reserved 10.5281/zenodo.23422688): the 10 fit bundles, their manifest, the per-file manifest, the run log and the fit code"""
import boto3, hashlib, requests, json, time
from publish_code import publish_code
publish_code("results/ATLAS_V21/zenodo_upload/code/run1/", ["zenodo_upload_box.py", "publish_code.py"], job="IAM-Atlas v2.1 Zenodo large-file upload")
B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3")
TOK = boto3.client("ssm", region_name="us-west-2").get_parameter(Name="/methylphys/zenodo-upload", WithDecryption=True)["Parameter"]["Value"]
A = {"access_token": TOK}
JOBS = [("https://zenodo.org/api/files/d8b5ce4e-5481-4085-9956-96fec36ed04e", "results/ATLAS_V21/assembly/atlas_v21_summary.parquet", "atlas_v21_summary.parquet")]
JOBS += [("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", "results/ATLAS_V21/zenodo_bundles/" + n, n) for n in sorted(json.loads(s3.get_object(Bucket=B, Key="results/ATLAS_V21/zenodo_bundles/bundles_manifest.json")["Body"].read()))]
JOBS += [("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", "results/ATLAS_V21/zenodo_bundles/bundles_manifest.json", "bundles_manifest.json"), ("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", "results/ATLAS_V21/assembly/output_manifest_v21.json", "output_manifest_v21.json"),
         ("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", "results/ATLAS_V21/fit/run_v21.log", "run_v21.log"), ("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", "results/ATLAS_V21/fit/stageB_v21_summary.json", "stageB_v21_summary.json")]
JOBS += [("https://zenodo.org/api/files/817e45e4-5a4d-48a8-b8bd-46eed9176339", o["Key"], "fit_code_" + o["Key"].split("/")[-1]) for o in s3.list_objects_v2(Bucket=B, Prefix="results/ATLAS_V21/fit/code/")["Contents"]]
def st(m): s3.put_object(Bucket=B, Key="results/ATLAS_V21/zenodo_upload/status.txt", Body=m.encode()); print(m, flush=True)
class R:
    def __init__(s, key, n): s.b = s3.get_object(Bucket=B, Key=key)["Body"]; s.n = n; s.h = hashlib.md5()
    def read(s, k=-1):
        c = s.b.read(1 << 20 if k is None or k < 0 else min(k, 1 << 20)); s.h.update(c); return c
    def __len__(s): return s.n
log = []
for bk, key, name in JOBS:
    have = {f["key"]: f for f in requests.get(bk, params=A, timeout=120).json().get("contents", [])}
    n = s3.head_object(Bucket=B, Key=key)["ContentLength"]
    if name in have and have[name]["size"] == n: log.append(f"{name}: already there"); continue
    ok = False
    for t in range(8):
        try:
            r_ = R(key, n); r = requests.put(f"{bk}/{name}", data=r_, params=A, headers={"Content-Length": str(n)}, timeout=7200)
            if r.status_code < 300 and r.json().get("checksum", "").endswith(r_.h.hexdigest()): ok = True; break
            log.append(f"{name}: try {t + 1} HTTP {r.status_code}")
        except Exception as e: log.append(f"{name}: try {t + 1} {type(e).__name__}")
        time.sleep(60 * (t + 1))
    log.append(f"{name}: {'uploaded, md5 ok' if ok else 'FAILED'} ({n} bytes)"); st("\n".join(log[-40:]))
st("\n".join(log) + "\nALL DONE" + (" with failures" if any("FAILED" in l for l in log) else ""))
