#!/usr/bin/env python3
"""book_count.py (2026-11-13; consolidation plan, group 3): the book's check count is GENERATED, never typed. Reads the committed check
record (status_facts.json 'book_check_count': output) and writes '(N checks, F failures)' into every place listed there (REFERENCE.md ...).
--check: report places that differ and exit 1 (build_all.py --check). Today's stale hand-typed count (4,692 for 4,701) was refused by the
gate twice; this removes the hand step."""
import os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bc = json.load(open(os.path.join(ROOT, "CANON", "status_facts.json"), encoding="utf-8"))["book_check_count"]
m = re.search(r"SUMMARY\s+PASS (\d+)\s+FAIL (\d+)", open(os.path.join(ROOT, bc["output"]), encoding="utf-8").read())
n, f = int(m.group(1)) + int(m.group(2)), int(m.group(2)); want = f"({n:,} checks, {f} failures)"
check = "--check" in sys.argv; differ = []
for p in bc["places"]:
    fp = os.path.join(ROOT, p)
    if not os.path.isfile(fp): continue
    t = open(fp, encoding="utf-8").read(); u = re.sub(r"\(([\d,]+) checks, (\d+) failures\)", want, t)
    if u != t:
        differ.append(p)
        if not check: open(fp, "w", encoding="utf-8").write(u)
print(f"book_count: {want} " + (("differs in " + ", ".join(differ)) if check and differ else ("written to " + ", ".join(differ) if differ else "current everywhere")))
sys.exit(1 if check and differ else 0)
