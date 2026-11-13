#!/usr/bin/env python3
"""doc_inventory.py (2026-11-13; author: "make an inventory of active items that require updating ... Less is more, KISS, and code is the
answer!"). Lists every DOCUMENT (not data set: that is master_inventory.py) and how it stays current:
  generated   written by a generator (a CANON/build_all.py step names it, or its head says it is generated / never edited by hand): it
              stays current by itself;
  hand        written by hand: it goes stale when what it states changes. For each, what it depends on: canon constants it prints (a value
              of CANON/iam_canon.json written in the text), test rows it names (PHY/COM/DEV/OTH/ATL/GRP-nnn) and their manifest status,
              and its last change (git, or file time outside a repository).
A hand document is FLAGGED when a test row it names was changed in the manifest after the document last was, or when it prints a canon
constant. Writes DOCUMENT_INVENTORY.csv and DOCUMENT_INVENTORY.md beside --out (default CANON/).
Usage: python3 CANON/doc_inventory.py [--root DIR] [--label NAME] [--out DIR]   (--root defaults to the repository)."""
import os, re, sys, csv, json, subprocess, datetime as dt, collections

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT = os.path.abspath(arg("--root", REPO)); LABEL = arg("--label", "repository"); OUT = os.path.abspath(arg("--out", HERE))
EXT = (".md", ".tex", ".txt", ".html", ".rst")
SKIP_DIR = re.compile(r"(^|/)(\.git|node_modules|__pycache__|xout|chains?|boxruns|Runtime Matrices|out|runs?|raw|logs?|_site|site)(/|$)|/doors/data/|/data/")
SKIP_FILE = re.compile(r"(_log|_summary|\.log|run\.log|requirements|LICENSE|_count|flagstat|_out)\b", re.I)
GEN_HEAD = re.compile(r"generated|never edited by hand|do not edit|written by (CANON|build)|built by|auto-?generated|rebuilt by", re.I)
DATA = re.compile(r"(^|/)(mgcamb_validation|chains?_[\w-]*|results?)/|\.(progress|chain)\.|_output\.txt$|_summary\.txt$")
GEN_NAME = re.compile(r"^(VERIFY_BOOK_INVENTORY\.md|verify_book_output\.txt|DOCUMENT_INVENTORY\.md|MASTER_INVENTORY\.md)$")
FROZEN = re.compile(r"(^|/)(archive|archived|retired|superseded)(/|$)|_20\d\d-\d\d-\d\d\.(md|txt|tex)$|(^|/)development/archive/")
ROW = re.compile(r"\b(PHY|COM|DEV|OTH|ATL|GRP|PROC)-[A-Z]*-?\d{2,3}\b")

def gen_targets():
    """Basenames that a build_all step writes (string literals in each generator's source)."""
    t = set(); s = open(os.path.join(HERE, "build_all.py")).read()
    for scr in re.findall(r'\("([^"]+\.py)"', s) + [m.replace('MP + "', "Biological_Physics/MethylPhys/") for m in re.findall(r'MP \+ "([^"]+\.py)"', s)]:
        p = os.path.join(REPO, scr if not scr.startswith("manual") and not scr.startswith("sop") else "Biological_Physics/MethylPhys/" + scr)
        if os.path.exists(p):
            for lit in re.findall(r'["\']([\w./ -]+\.(?:md|tex|html|txt|csv|json))["\']', open(p, errors="ignore").read()):
                t.add(os.path.basename(lit))
    return t

def canon_values():
    K = json.load(open(os.path.join(HERE, "iam_canon.json")))["constants"]; V = {}
    for k, c in K.items():
        v = c.get("value") if isinstance(c, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1, 2) and abs(v) > 1e-12:
            s = repr(float(v)).rstrip("0").rstrip(".")
            if len(s.replace("-", "").replace(".", "").lstrip("0")) >= 3: V[k] = s   # only values specific enough to recognise
    return V

def last_change(p):
    if os.path.isdir(os.path.join(ROOT, ".git")) or ROOT == REPO:
        r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", p], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if r: return r
    return dt.date.fromtimestamp(os.path.getmtime(p)).isoformat()

def manifest_dates():
    M = json.load(open(os.path.join(HERE, "manifest.json"))); D = {}
    for r in M["rows"]:
        d = [x for x in re.findall(r"20\d\d-\d\d-\d\d", json.dumps(r))]
        D[r["id"]] = (r.get("status", ""), max(d) if d else "")
    return D

if __name__ == "__main__":
    GT, CV, MD = gen_targets(), canon_values(), manifest_dates()
    rows = []
    for d, ds, fs in os.walk(ROOT):
        rel_d = os.path.relpath(d, ROOT).replace(os.sep, "/")
        if SKIP_DIR.search("/" + rel_d + "/"): ds[:] = []; continue
        for f in fs:
            if not f.endswith(EXT) or SKIP_FILE.search(f): continue
            p = os.path.join(d, f); rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
            try: t = open(p, encoding="utf-8", errors="ignore").read()
            except Exception: continue
            if len(t) < 200: continue
            if DATA.search(rel) or (f.endswith(".txt") and sum(ch.isdigit() for ch in t[:4000]) > 0.25 * min(len(t), 4000)): continue   # results and data, not documents
            if f in GT or GEN_HEAD.search(t[:800]) or GEN_NAME.search(f): kind = "generated"
            elif rel.startswith("docs/book/") and f.endswith(".tex"): kind = "checked"       # every printed number has a verify_book check
            elif FROZEN.search(rel): kind = "frozen"                                           # archives and dated snapshots: kept as they were
            else: kind = "hand"
            consts = sorted(k for k, s in CV.items() if s in t)
            ids = sorted(set(m.group(0) for m in ROW.finditer(t)))
            when = last_change(p)
            newer = sorted(i for i in ids if i in MD and MD[i][1] > when)
            flag = kind == "hand" and bool(newer or consts)
            rows.append(dict(where=LABEL, path=rel, kind=kind, words=len(t.split()), last_change=when, canon_constants=" ".join(consts),
                             rows_named=len(ids), rows_changed_since=" ".join(newer[:12]) + (" ..." if len(newer) > 12 else ""), flagged=flag))
    rows.sort(key=lambda r: (not r["flagged"], r["kind"], r["path"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "DOCUMENT_INVENTORY.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    C = collections.Counter((r["kind"], r["flagged"]) for r in rows); K = collections.Counter(r["kind"] for r in rows)
    L = [f"# Document inventory: {LABEL} (generated by CANON/doc_inventory.py; never edited by hand)", "",
         f"{len(rows)} documents: {K['generated']} generated (current by themselves); {K['checked']} book chapters (every number checked by code); {K['frozen']} frozen (archives, dated snapshots); "
         f"{C[('hand', True)]} hand-written and FLAGGED; {C[('hand', False)]} hand-written, not flagged.", "",
         "Flagged: a test row it names changed after the document did, or it prints a canon constant by hand.", "",
         "| Document | Words | Last change | Canon constants printed | Rows changed since |", "|---|---|---|---|---|"]
    for r in rows:
        if r["flagged"]: L.append(f"| {r['path']} | {r['words']} | {r['last_change']} | {r['canon_constants'] or '-'} | {r['rows_changed_since'] or '-'} |")
    L += ["", "## Hand-written, not flagged", ""] + [f"- {r['path']} ({r['words']} words, {r['last_change']})" for r in rows if r["kind"] == "hand" and not r["flagged"]]
    open(os.path.join(OUT, "DOCUMENT_INVENTORY.md"), "w").write("\n".join(L) + "\n")
    print(f"doc_inventory: {len(rows)} documents: {K['generated']} generated, {K['checked']} checked by code, {K['frozen']} frozen, {C[('hand', True)]} hand FLAGGED, {C[('hand', False)]} hand not flagged")
