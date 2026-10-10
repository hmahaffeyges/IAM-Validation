#!/usr/bin/env python3
"""Status gate (2026-10-10). Reads CANON/status_facts.json and refuses a push when
  1. a living document still carries a phrase a fact forbids (the old status),
  2. a document a fact requires no longer states it,
  3. the book's check count printed in README.md differs from the committed verify_book output, or that output no longer counts every
     check verify_book.py defines (new checks added without regenerating the output).
Run: python3 CANON/status_check.py   (called by CANON/checked_push.sh)"""
import json, os, re, subprocess, sys
F = json.load(open("CANON/status_facts.json", encoding="utf-8")); ex = re.compile("|".join(F["exclude"]))
files = [f for f in subprocess.run(["git", "ls-files", "*.md", "*.tex", "*.py", "*.json", "*.html", "*.yml"], capture_output=True, text=True).stdout.split("\n")
         if f and not ex.search(f) and os.path.isfile(f) and os.path.getsize(f) < 3_000_000]
fail = []
for fact in F["facts"]:
    pats = [re.compile(p) for p in fact.get("forbid", [])]
    for f in files:
        s = open(f, encoding="utf-8", errors="replace").read()
        for p in pats:
            for m in p.finditer(s):
                fail.append(f"[{fact['id']}] {f}:{s.count(chr(10), 0, m.start()) + 1}: '{m.group(0)}' contradicts: {fact['fact']}")
    for f, p in fact.get("require", []):
        if os.path.isfile(f) and not re.search(p, open(f, encoding="utf-8").read()): fail.append(f"[{fact['id']}] {f} no longer states: {fact['fact']}")
bc = F.get("book_check_count")
if bc and os.path.isfile(bc["output"]):
    m = re.search(r"SUMMARY\s+PASS (\d+)\s+FAIL (\d+)", open(bc["output"], encoding="utf-8").read()); n_out = int(m.group(1)) + int(m.group(2)); fails = int(m.group(2))
    for p in bc["places"]:
        if os.path.isfile(p):
            for x in re.findall(r"\(([\d,]+) checks, (\d+) failures\)", open(p, encoding="utf-8").read()):
                if int(x[0].replace(",", "")) != n_out or int(x[1]) != fails: fail.append(f"[book_check_count] {p} prints {x[0]} checks, {x[1]} failures; the committed output has {n_out:,}, {fails}")
    if os.path.isfile("docs/book/verify_book.py"):
        n_def = len(re.findall(r"^@check\(", open("docs/book/verify_book.py", encoding="utf-8").read(), re.M))
        if n_def != n_out: fail.append(f"[book_check_count] verify_book.py defines {n_def} checks but {bc['output']} counts {n_out}: regenerate the output")
if fail:
    print(f"status_check: REFUSED ({len(fail)})\n  " + "\n  ".join(fail[:40])); sys.exit(1)
print(f"status_check: ok ({len(F['facts'])} facts, {len(files)} documents)")
