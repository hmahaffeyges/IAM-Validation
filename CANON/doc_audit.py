#!/usr/bin/env python3
"""doc_audit.py (2026-10-19, author: 'double check that every cell of the testing manifest and the equipment document (table) and ANY of our
referenced docs are accurate'). Mechanical part of the audit; a person reads every finding. Checks:
 A. every repository path a manifest row, the instrument catalogue or a public document names exists in the repository (all of it, not the
    sparse checkout) at that path, or at the path it moved to;
 B. statuses agree with the commissioning record: an instrument the record commissions is not described as not commissioned / development /
    chain v3 / position form as current;
 C. 'running' rows and next steps naming a machine job agree with the job register (a finished job is not 'running now');
 D. every number in an instrument's catalogue entry appears in a record that instrument cites (its commissioning note or runtime matrices).
Usage: python3 CANON/doc_audit.py [--json out.json]. Prints counts; writes the findings."""
import json, os, re, subprocess, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
TREE = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True).stdout.split("\n"))
BASE = {}
for p in TREE: BASE.setdefault(os.path.basename(p), []).append(p)
MP = "Biological_Physics/MethylPhys/"
PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:md|py|json|csv|tsv|tex|sh|parquet|txt|yml|pdf|html|ipynb))(?![\w/])")
def resolve(p):
    p = p.strip("./`*()[]")
    for c in (p, MP + p, MP + "doors/" + p, MP + "doors/data/" + p, "docs/book/" + p, MP + "chain/" + p, MP + "boxruns/" + p):
        if c in TREE: return c
    return None
F = []
def add(kind, where, what): F.append(dict(kind=kind, where=where, what=what))
def text_of(o): return json.dumps(o, ensure_ascii=False) if not isinstance(o, str) else o
M = json.load(open("CANON/manifest.json", encoding="utf-8"))["rows"]
J = {j["job_id"]: j for j in json.load(open("CANON/box_jobs.json", encoding="utf-8"))["jobs"]}
COMM = json.load(open("CANON/commissioning.json", encoding="utf-8"))["instruments"]
COMMISSIONED = {"Met-A": "neutrophils", "C-Score": "whole blood", "IAM-A": "CD8", "IAM-Atlas v2.1": "", "Deconvolver": ""}
HONEST = re.compile(r"not in the repository|not found|never (committed|written)|no (separate )?repo record|not in the repo|pasted|private draft|search file|replaced the same day|deleted 20[0-9-]*|in our S3|exists in doors/; status|No [A-Z_0-9]+\\.md exists", re.I)
REVIEWED = {   # read 2026-10-19 against the commissioning record: the wording is right
 "COM-METAA-009": "Met-A's whole-blood reading is preliminary; only its neutrophil reading is commissioned",
 "COM-IAMA-042": "IAM-A's clustering reading is not commissioned", "ATL-017": "the atlas's identity-site use (U4) is not in COMMISSIONING_NOTE_ATLAS_V21.md",
 "TIS-026": "'do not use in development' is a data-hold instruction", "DAT-055": "a data-hold list"}
NOTCOMM = re.compile(r"(not (yet )?commissioned|stays? (in )?development|in development|still development|chain v3 (reads|is the)|position form (is|reads) (the )?(current|commissioned))", re.I)
for r in M:
    t = text_of({k: r.get(k) for k in ("title", "purpose", "record", "sources", "next_step", "notes", "data_status")})
    HIST = text_of(r.get("runs"))   # dated run notes are history: a file they name must have existed in the repository at some commit
    for m in PATH.finditer(HIST):
        p = m.group(1)
        if resolve(p) is None and not subprocess.run(["git", "log", "--all", "--format=%h", "-1", "--", "*" + os.path.basename(p)], capture_output=True, text=True).stdout.strip() and not re.match(r"^(results|downloads|atlas_v21|s3:|GSE|GSM)", p):
            add("A missing path (run note, never in the repository)", r["id"], p)
    for m in PATH.finditer(t):
        _c = t[max(0, m.start() - 200): m.end() + 200]
        if HONEST.search(_c): continue   # the text itself says the file is not in the repository
        p = m.group(1)
        if "/" not in p and p.count(".") == 1 and len(BASE.get(p, [])) == 1: continue   # a bare file name that names one file
        if resolve(p) is None and not (len(BASE.get(os.path.basename(p), [])) >= 1 and "/" not in p):
            if not re.match(r"^(results|downloads|atlas_v21|s3:|GSE|GSM)", p): add("A missing path", r["id"], p)
    if r["status"] == "running":
        ids = re.findall(r"i-[0-9a-f]{8,17}", t); st = [J[i]["state"] for i in ids if i in J]
        if ids and st and all(s != "running" for s in st): add("C running row, jobs finished", r["id"], f"{ids} -> {st}")
    for i in set(re.findall(r"[Rr]unning now: job (i-[0-9a-f]+)", str(r.get("next_step")))):
        if i in J and J[i].get("state") != "running": add("C next step says running", r["id"], f"{i} is {J[i].get('state')}")
    if r["status"] not in ("retired",) and r["id"] not in REVIEWED and NOTCOMM.search(str(r.get("next_step")) + " " + str(r.get("purpose"))):
        add("B status wording to read", r["id"], NOTCOMM.search(str(r.get("next_step")) + " " + str(r.get("purpose"))).group(0) + " | " + r["title"][:90])
CAT = json.load(open("CANON/catalogue.json", encoding="utf-8"))
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{2,}(?:,\d{3})*)(?![\w.])")
for ins in CAT["instruments"]:
    t = text_of(ins); cites = [resolve(p) for p in (m.group(1) for m in PATH.finditer(t))] + [ins.get("record")]; cites = [c for c in cites if c]
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Z]{3}-[A-Z0-9-]+\b|\b(GSE|GSM|SRR|ENCFF)\w+|\bch:\w+", "", t)   # dates, row ids, accessions, labels are not measurements
    for m in PATH.finditer(t):
        if resolve(m.group(1)) is None and len(BASE.get(os.path.basename(m.group(1)), [])) == 0: add("A missing path (catalogue)", ins["name"], m.group(1))
    corpus = ""
    for c in cites:
        try: corpus += open(c, encoding="utf-8", errors="replace").read() if os.path.exists(c) else subprocess.run(["git", "show", "HEAD:" + c], capture_output=True, text=True).stdout
        except Exception: pass
    # only the instrument's own records: a large matrix holds almost every short decimal, so it proves nothing (tested 2026-10-19)
    nums = sorted(set(m.group(1) for m in NUM.finditer(t)) - {"2026", "2025", "2024", "2023", "2018"})
    for n in nums:
        if n not in corpus.replace(",", "") and n.replace(",", "") not in corpus.replace(",", ""): add("D catalogue number not in its cited records", ins["name"], n)
if __name__ == "__main__":
    out = os.path.join(ROOT, "..", "doc_audit.json") if "--json" not in sys.argv else sys.argv[sys.argv.index("--json") + 1]
    if "--check" in sys.argv:
        print("doc_audit: " + ("ok (manifest paths, statuses, running jobs, catalogue numbers)" if not F else f"REFUSED: {len(F)} finding(s): " + "; ".join(f"{f['where']} {f['kind']}: {f['what'][:60]}" for f in F[:8])))
        sys.exit(1 if F else 0)
    json.dump(F, open(out, "w"), indent=1, ensure_ascii=False)
    print(collections.Counter(f["kind"] for f in F).most_common())
