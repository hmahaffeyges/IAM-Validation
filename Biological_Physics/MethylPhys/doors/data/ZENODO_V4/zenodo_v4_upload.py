"""Uploads the files built by zenodo_v4_box.sh to the two Zenodo drafts and checks the MD5 Zenodo returns against the bytes sent (2026-10-19).
The key is read from the AWS secret store (/methylphys/zenodo-upload) and never written anywhere. The book uploads only if every book check
passed (FAIL 0 in the SUMMARY line). Usage: python zenodo_v4_upload.py zenodo_v4_drafts.json <up dir> "<book SUMMARY line>"
"""
import boto3, hashlib, json, os, re, sys, time, requests
D = json.load(open(sys.argv[1])); UP = sys.argv[2]; BS = sys.argv[3]
TOK = boto3.client("ssm", region_name="us-west-2").get_parameter(Name="/methylphys/zenodo-upload", WithDecryption=True)["Parameter"]["Value"]
class F:
    def __init__(s, p): s.f = open(p, "rb"); s.h = hashlib.md5(); s.n = os.path.getsize(p)
    def read(s, k=-1):
        b = s.f.read(k if k and k > 0 else 8 << 20); s.h.update(b); return b
    def __len__(s): return s.n
ok_all = True
book_ok = bool(re.search(r"FAIL 0\b", BS))
for rec in ("package", "book"):
    if rec == "book" and not book_ok: print("book NOT uploaded: book checks did not all pass:", BS); ok_all = False; continue
    for name in sorted(os.listdir(os.path.join(UP, rec))):
        p = os.path.join(UP, rec, name); ok = False
        for t in range(5):
            f = F(p); r = requests.put(D[rec]["bucket"] + "/" + name, data=f, params={"access_token": TOK}, timeout=7200)
            if r.status_code < 300 and r.json().get("checksum", "").endswith(f.h.hexdigest()): ok = True; break
            time.sleep(60 * (t + 1))
        print(f"{rec}/{name}: {'uploaded, md5 ok' if ok else 'FAILED'} ({os.path.getsize(p)} bytes)", flush=True); ok_all &= ok
print("ALL DONE" if ok_all else "SOME FAILED"); sys.exit(0 if ok_all else 1)
