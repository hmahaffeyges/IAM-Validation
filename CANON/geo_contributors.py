"""GEO contributor register (2026-10-11, rule 9). For every GSE accession named in a doors note that cites a laboratory or is a commissioning note,
the series' contributors and contact institute from GEO (acc.cgi brief text). Writes CANON/geo_contributors.csv; rows already present are kept.
Two series share a laboratory for rule 9 when they share >= 2 contributors. Usage: python3 CANON/geo_contributors.py"""
import os, re, glob, csv, time, requests
OUT = "CANON/geo_contributors.csv"; have = {r["accession"]: r for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else {}
acc = set()
for f in glob.glob("Biological_Physics/MethylPhys/doors/*.md"):
    t = open(f, encoding="utf-8").read()
    if "COMMISSIONING_NOTE" in f or re.search(r"(?i)laborator", t): acc |= set(re.findall(r"\bGSE\d{4,}\b", t))
for a in sorted(acc - set(have)):
    for i in range(4):
        try:
            t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": a, "targ": "self", "form": "text", "view": "brief"}, timeout=60).text; break
        except Exception: time.sleep(5 * (i + 1))
    get = lambda k: [l.split("=", 1)[1].strip() for l in t.split("\n") if l.startswith(k)]
    have[a] = dict(accession=a, contributors=";".join(get("!Series_contributor")), institute=(get("!Series_contact_institute") or [""])[0]); time.sleep(0.35)
with open(OUT, "w", newline="") as g:
    w = csv.DictWriter(g, fieldnames=["accession", "contributors", "institute"]); w.writeheader(); [w.writerow(have[a]) for a in sorted(have)]
print(len(have), "series;", sum(1 for r in have.values() if not r["contributors"]), "without contributors")
