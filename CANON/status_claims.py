#!/usr/bin/env python3
"""CANON/status_claims.py - refuse a public sentence that contradicts an instrument's commissioned status (2026-10-23, plan 39.14; author:
"we are not hand updating things like this because that is how we end up with all kinds of stale information all over"). Built from
CANON/catalogue.json itself: for every instrument the catalogue lists as commissioned, a sentence in a public document saying it is not
commissioned, or waiting for it to be ("once / until / before / after X is commissioned", "commission X" as a next step), is refused with
the file, line and words. As an instrument's status changes in the catalogue, the check changes with it. Dated records (day logs, the
development log, doors/ records, archives) are history and are not read. Usage: python3 CANON/status_claims.py [--check]"""
import json, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"atlas": ["IAM-Atlas v2.1", "IAM-Atlas", "the atlas", "atlas"], "deconv": ["IAM-Deconvolver", "the Deconvolver", "Deconvolver"],
         "meta": ["Met-A"], "cscore": ["C-Score", "C-score", "clustering meter"], "iama": ["IAM-A"], "groupa": ["Group-A"]}
PUBLIC = re.compile(r"^(README\.md|FOUR_DOMAINS\.md|Biological_Physics/MethylPhys/(INSTRUMENTS|INTRODUCTION|QUESTIONS_AND_ANSWERS|REPLICATE)\.md|"
                    r"docs/book/.+\.tex|CANON/(catalogue|instruments|atlas_story|researcher_page|four_domains)\.json|website/build/.+\.(py|json))$")
def rules():
    C = json.load(open(os.path.join(ROOT, "CANON", "catalogue.json"), encoding="utf-8")); R = []
    for x in C["instruments"]:
        if not x.get("commissioned"): continue
        for n in NAMES.get(x["id"], [x["name"]]):
            e = re.escape(n)
            R.append((x["id"], re.compile(rf"(?i)\b{e}\b[^.;]{{0,40}}\b(is|are|was)\s+not\s+(yet\s+)?commissioned"), n))
            R.append((x["id"], re.compile(rf"(?i)\b(once|until|before|after)\s+(the\s+)?{e}\s+(is|has been)\s+commissioned"), n))
            R.append((x["id"], re.compile(rf"(?i)\bcommission(ing)?\s+(the\s+)?{e}\b(?![-\w])"), n))
    return R
def scan():
    F = [f for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n") if PUBLIC.match(f)]
    R = rules(); bad = []
    for f in F:
        for k, l in enumerate(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace"), 1):
            seen = set()
            for i, p, n in R:
                m = p.search(l)
                if m and i not in seen: seen.add(i); bad.append(f"{f}:{k}: '{m.group(0)}' ({n} is commissioned in CANON/catalogue.json)")
    return F, bad
if __name__ == "__main__":
    F, bad = scan()
    for b in bad: print("  " + b[:240])
    print(f"status_claims: {'REFUSED (%d)' % len(bad) if bad else 'ok'} ({len(F)} public documents read)")
    sys.exit(1 if bad and "--check" in sys.argv else 0)
