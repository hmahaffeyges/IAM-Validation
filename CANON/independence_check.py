"""Rule 9 (2026-10-11, author: "make sure this sort of thing NEVER happens again"). Laboratory independence, checked from GEO contributors
(CANON/geo_contributors.csv, built by CANON/geo_contributors.py). Two series share a laboratory when they share >= 2 contributors.
  a. A line that calls a series another / other / independent / second / different / outside laboratory is refused when that series shares a
     laboratory with a reference set of the note ('**Reference sets:**' line) or with a series named before the phrase on the same line.
  b. A commissioning note must carry a '**Reference sets:**' line; a bar row (| n | ...) that cites a series sharing a laboratory with a
     reference set must say 'in-sample'.
  c. A series named in such a line or row and missing from the register is refused (run CANON/geo_contributors.py).
check(path, text) -> list of failures."""
import re, csv, os
REG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geo_contributors.csv")
C = {r["accession"]: {x.strip().lower() for x in r["contributors"].split(";") if x.strip()} for r in csv.DictReader(open(REG, encoding="utf-8"))}
LAB = re.compile(r"(?i)\b(another|other|independent|second|different|outside|a new)[\s-]+(laborator|lab\b|labs\b)|\b(laborator(y|ies)|labs?) other than"); GSE = re.compile(r"\bGSE\d{4,}\b")
GLOBAL = [l.split()[0] for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_sets.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]
def same_lab(a, b): return a != b and len(C.get(a, set()) & C.get(b, set())) >= 2
def check(path, text):
    fail = []; m = re.search(r"^\*\*Reference sets:\*\*(.*)$", text, re.M); refs = GSE.findall(m.group(1)) if m else []
    commissioning = "COMMISSIONING_NOTE" in os.path.basename(path)
    if commissioning and not refs: fail.append(f"{path}: commissioning note without a '**Reference sets:**' line (rule 9)")
    # a. laboratory claims, sentence by sentence (paragraph lines joined), against the note's and the chain's reference sets
    for para in re.split(r"\n\s*\n", text):
        for sent in re.split(r"(?<=[.;])\s+", para.replace("\n", " ")):
            mm = LAB.search(sent)
            if not mm: continue
            before = GSE.findall(sent[:mm.start()])
            for a in GSE.findall(sent):
                if a in before or a in GLOBAL or a in refs: continue   # a reference set called independent of the set under test
                if a not in C: fail.append(f"{path}: {a} not in CANON/geo_contributors.csv (run CANON/geo_contributors.py) (rule 9)"); continue
                for b in set(refs + GLOBAL + before):
                    if same_lab(a, b): fail.append(f"{path}: calls {a} another laboratory, but it shares >= 2 contributors with {b} (rule 9)")
    for n, line in enumerate(text.split("\n"), 1):
        g = GSE.findall(line); bar_row = commissioning and re.match(r"^\|\s*\d+[a-z]?\s*\|", line)
        if not g or not bar_row: continue
        for a in g:
            if a not in C: fail.append(f"{path}:{n}: {a} not in CANON/geo_contributors.csv (run CANON/geo_contributors.py) (rule 9)"); continue
            if bar_row and not re.search(r"(?i)in-sample", line):
                for b in refs:
                    if same_lab(a, b) or a == b: fail.append(f"{path}:{n}: bar row cites {a}, from the reference laboratory ({b}), without 'in-sample' (rule 9)")
    return sorted(set(fail))
