"""DEV-TISSUE-COLON-01: pull IDATs of GSE131013 (Barault/CLX colon 450K: healthy donors' mucosa, adjacent normal, tumour) from the S3 copy of
GSE131013_RAW.tar by byte range, member by member. 'normal' pulls Mucosa and Normal arrays only; 'tumour' refuses unless sealed_colon_tissue.json
is committed, and pulls only the sealed pairs' tumour arrays. Samples and plates: colon_samples.csv (GEO annotations + the series' Sentrix sheet).
Usage: python3 fetch_colon.py WORKDIR normal|tumour"""
import os, sys, json, subprocess, boto3, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W, which = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "idat"), exist_ok=True)
s3 = boto3.client("s3"); B = "methylphys-data-945451304272-us-west-2-an"; KEY = "downloads/C_tissue_tests/GSE131013/GSE131013_RAW.tar"
S = pd.read_csv(os.path.join(HERE, "colon_samples.csv"))
if which == "tumour":
    SEAL = os.path.join(HERE, "sealed_colon_tissue.json")
    assert os.path.exists(SEAL) and subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.basename(SEAL)], capture_output=True).returncode == 0, "tumour arrays are fetched only after the seal is committed"
    want = {p["tumour_gsm"] for p in json.load(open(SEAL))["pairs"]}
else: want = set(S[S.tissue.isin(["Mucosa", "Normal"])].gsm)
off = 0; n = 0
while True:
    h = s3.get_object(Bucket=B, Key=KEY, Range=f"bytes={off}-{off + 511}")["Body"].read(); nm = h[:100].split(b"\0")[0].decode(errors="replace")
    if not nm: break
    sz = int(h[124:136].split(b"\0")[0].strip() or b"0", 8); g = nm.split("_")[0]
    if g in want and ".idat" in nm and not os.path.exists(os.path.join(W, "idat", nm)):
        body = s3.get_object(Bucket=B, Key=KEY, Range=f"bytes={off + 512}-{off + 512 + sz - 1}")["Body"].read(); assert len(body) == sz
        open(os.path.join(W, "idat", nm), "wb").write(body); n += 1
    off += 512 + ((sz + 511) // 512) * 512
print(which, "fetched", n, "| present", len([f for f in os.listdir(os.path.join(W, "idat")) if f.split("_")[0] in want]), "of", 2 * len(want))
