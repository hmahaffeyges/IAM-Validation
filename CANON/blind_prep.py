"""Blind studies, step 1 (2026-10-23; author: 'find as many studies as we can that have already been analyzed and blindly test them').
For each study in CANON/blind_studies.json with status 'chosen': its array accessions from the GEO series record ONLY (no titles, no
characteristics, no labels), written as doors/data/BLIND_01/<GSE>_samples.csv (gsm, slide), with the list's SHA-256 recorded in the ledger.
The labels stay in GEO; they are fetched only by blind_unblind.py, after the chain's readings of every array are committed.
Usage: python3 CANON/blind_prep.py [GSE ...]"""
import json, sys, os, time, hashlib, urllib.request, urllib.parse
L = "CANON/blind_studies.json"; D = "Biological_Physics/MethylPhys/doors/data/BLIND_01"
J = json.load(open(L, encoding="utf-8")); want = set(sys.argv[1:])
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
def get(u):
    for k in range(5):
        try: return json.loads(urllib.request.urlopen(u, timeout=120).read().decode())
        except Exception: time.sleep(5 * (k + 1))
    raise RuntimeError(u)
for s in J["studies"]:
    if s["status"] != "chosen" or (want and s["gse"] not in want): continue
    ids = get(E + "esearch.fcgi?" + urllib.parse.urlencode(dict(db="gds", term=s["gse"] + "[ACCN] AND gse[ETYP]", retmode="json")))["esearchresult"]["idlist"]
    x = get(E + "esummary.fcgi?" + urllib.parse.urlencode(dict(db="gds", id=ids[0], retmode="json")))["result"][ids[0]]
    gsm = sorted({m["accession"] for m in x.get("samples", []) if str(m.get("accession", "")).startswith("GSM")})   # accessions only; titles not kept
    p = os.path.join(D, s["gse"] + "_samples.csv"); open(p, "w").write("gsm,slide\n" + "".join(f"{g},NA\n" for g in gsm))
    s.update(status="blinded", arrays=len(gsm), samples=p, samples_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()); print(s["gse"], len(gsm)); time.sleep(0.5)
json.dump(J, open(L, "w", encoding="utf-8"), indent=1, ensure_ascii=False); open(L, "a").write("\n")
