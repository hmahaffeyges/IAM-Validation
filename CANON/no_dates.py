#!/usr/bin/env python3
"""CANON/no_dates.py - no dates in public prose (2026-10-23; author: "dates belong nowhere except the logs. Day or commissioning or
whatever, NEVER in the book or webpage or anywhere in prose"). Reads every public document and the canon text that is generated into them,
and lists every date: 2026-10-15, October 15, Oct 15, 15 October. Dates live in the day logs, the development log, the manifest, the records
in doors/, the plan and the archives, which are not read. Allowed
(author agreed 2026-10-23): the book's generated provenance appendix, the website's generated "as of" line, a year alone in a citation. Usage: python3 CANON/no_dates.py [--check] [--files]"""
import os, re, subprocess, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = re.compile(r"^(README\.md|FOUR_DOMAINS\.md|Biological_Physics/MethylPhys/(INSTRUMENTS|INTRODUCTION|QUESTIONS_AND_ANSWERS|REPLICATE)\.md|"
                    r"docs/book/.+\.tex)$")
# The canon's JSON files are data and may hold dates; every generator prints them through CANON/undate.py, and the website cleans its pages
# at their last write (website/build/build_home.py _nodates), so what a reader sees is what this check reads (2026-10-23).
M = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
DATE = re.compile(rf"\b20\d\d-[01]\d-[0-3]\d\b|\b{M}\s+[0-3]?\d(st|nd|rd|th)?\b(,?\s+20\d\d)?|\b[0-3]?\d\s+{M}\b(\s+20\d\d)?")
ALLOWED = {"docs/book/appendices/app_I_provenance.tex"}   # generated provenance (author 2026-10-23: the seal-before-reading order is evidence)
AS_OF = re.compile(r"[Aa]s of\b")   # the website's one generated "as of" line
def scan():
    F = [f for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n") if PUBLIC.match(f) and f not in ALLOWED]
    hits = []
    for f in F:
        for k, l in enumerate(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace"), 1):
            if f.endswith(".tex"): l = re.split(r"(?<!\\)%", l)[0]   # a LaTeX comment is not printed
            if f.startswith("website/build/") and AS_OF.search(l): continue
            for m in DATE.finditer(l): hits.append((f, k, m.group(0)))
    return F, hits
if __name__ == "__main__":
    F, hits = scan(); c = collections.Counter(h[0] for h in hits)
    if "--files" in sys.argv:
        for f, n in c.most_common(): print(f"  {n:5} {f}")
    print(f"no_dates: {len(hits)} date(s) in {len(c)} of {len(F)} public documents" + ("" if hits else " - ok"))
    sys.exit(1 if hits and "--check" in sys.argv else 0)
