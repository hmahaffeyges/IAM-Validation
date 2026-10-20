#!/usr/bin/env python3
"""number_trace.py (2026-10-19, to-do step 8): every number with three or more significant figures that a public document prints must appear,
at the precision printed, in a measurement record (doors/data outputs, the chain's runtime matrices, the verification outputs, the reproduction
verdicts, the commissioning notes and outcome records, the atlas v2.1 tables). The documents' own generator sources (CANON/*.json) do not count:
they would prove a number only against itself. Numbers that are not measurements (identifiers, unit conversions, vendor specifications,
encodings) are listed in EXPLAINED with the reason. Small counts ('29 of 29') are not covered; they are read by hand in the table audit.
Usage: python3 CANON/number_trace.py [--check]   (writes CANON/number_trace.json)"""
import json, re, os, subprocess, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); MP = "Biological_Physics/MethylPhys/"
DOCS = ["README.md", "REFERENCE.md", "FOUR_DOMAINS.md", "Biological_Physics/README.md", MP + "README.md", MP + "START_HERE.md", MP + "QUESTIONS_AND_ANSWERS.md",
        MP + "RESEARCHER_PACKAGE.md", MP + "INTRODUCTION.md", MP + "COMMISSIONING_REPORT.md", MP + "INSTRUMENTS.md", MP + "sop/MethylPhys_CPG_SOP_v4.md", MP + "reproduce/README.md",
        MP + "atlas/v2.1/README.md", MP + "doors/data/SHOWCASE_V4/README.md"]
EXPLAINED = {"348.15": "75 degrees C, the chip junction temperature, in kelvin (a unit conversion)", "1206.4916": "an arXiv identifier", "2507.05237": "an arXiv identifier",
             "237901": "a journal article number (Phys. Rev. Lett. 88)", "095206": "a journal article number (AIP Adv. 9)", "085014": "a journal article number (AIP Adv. 10)",
             "935,000": "the vendor's probe count for EPIC v2", "65535": "the atlas file's 'not measured' code (largest unsigned 16-bit integer)"}
def is_rec(p):
    if not p.endswith((".json", ".csv", ".txt", ".tsv", ".md")): return False
    if p.startswith((MP + "doors/data/", MP + "chain/Runtime Matrices/", "docs/verification/", MP + "reproduce/verified")): return True
    if p.startswith(MP + "doors/") and re.search(r"(COMMISSIONING_NOTE|OUTCOME|_NOTE_)", p): return True
    return p.startswith(MP + "atlas/v2.1/") and p.endswith((".json", ".csv"))
TREE = [p for p in subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True).stdout.split("\n") if p]
vals = collections.defaultdict(set)
for p in (p for p in TREE if is_rec(p) and os.path.exists(p) and os.path.getsize(p) < 3_000_000):
    s = open(p, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+(?:e-?\d+)?)(?![\w])", s):
        try: f = float(m.group(1))
        except ValueError: continue
        for k in range(1, 7):
            vals[k].add(f"{round(f, k):.{k}f}")
            if f < 1: vals["pct%d" % k].add(f"{round(100 * f, k):.{k}f}")
    for m in re.finditer(r"(?<![\w.])(\d{2,}(?:,\d{3})*)(?![\w.])", s): vals[0].add(m.group(1).replace(",", ""))
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d{3,})(?![\w.])")
STRIP = re.compile(r"https?://\S+|\]\([^)]*\)|10\.\d{4,}/\S+|\b\d{4}-\d{2}-\d{2}\b|\b(GSE|GSM|SRR|PRJNA|ENCFF|E-MTAB-|GPL)\w*|\bi-[0-9a-f]+\b|\b[0-9a-f]{7,40}\b|"
                   r"\b(OTH|PHY|ATL|TIS|COM|DAT|SCO|FLD|BRE|GRP|DEV|PROC|PRED)-[\w-]+|\b(19|20)\d\d\b|v\d+(\.\d+)*|`[^`]*`|\b\d+(\.\d+)? ?(GB|MB|KB|kB|min|minutes|h|hours|cores|USD)\b|\$\s?\d[\d.,]*")
out = []
for d in (d for d in DOCS if os.path.exists(d)):
    for i, l in enumerate(open(d, encoding="utf-8").read().split("\n"), 1):
        for m in NUM.finditer(STRIP.sub("", l)):
            x = m.group(1).replace(",", "")
            if m.group(1) in EXPLAINED or x in EXPLAINED: continue
            k = len(x.split(".")[1]) if "." in x else 0
            ok = (k > 6 or x in vals[k] or x in vals.get("pct%d" % k, set())) if k else x in vals[0]
            if not ok: out.append(dict(doc=d, line=i, num=m.group(1), text=l.strip()[:200]))
json.dump(dict(untraced=out, explained=EXPLAINED), open("CANON/number_trace.json", "w"), indent=1, ensure_ascii=False)
print("number_trace: " + ("ok (every number traced or explained)" if not out else f"REFUSED: {len(out)} untraced: " + "; ".join(f"{o['doc'].split('/')[-1]}:{o['line']} {o['num']}" for o in out[:10])))
sys.exit(1 if out and "--check" in sys.argv else 0)
