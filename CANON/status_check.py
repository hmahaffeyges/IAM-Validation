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
# ---- results register (2026-10-10): book development chapter and README Advancements are generated from CANON/results_register.json
import importlib.util as _u
_s = _u.spec_from_file_location("r2t", "CANON/results_to_tex.py"); _m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
try:
    tex, rd, REG, _V = _m.build()
    if not os.path.isfile(_m.TEX) or open(_m.TEX, encoding="utf-8").read() != tex: fail.append(f"[results_register] {os.path.relpath(_m.TEX)} differs from the register: run python3 CANON/results_to_tex.py")
    rs = open(_m.README, encoding="utf-8").read()
    if _m.B not in rs or _m.E not in rs or rs[rs.index(_m.B) + len(_m.B):rs.index(_m.E)].strip() != rd.strip(): fail.append("[results_register] Biological_Physics/README.md Advancements differs from the register: run python3 CANON/results_to_tex.py")
    if "\\input{part6/p6_25_development}" not in open("docs/book/main.tex", encoding="utf-8").read(): fail.append("[results_register] docs/book/main.tex does not input the development chapter")
    recs = {e["record"]: e for e in REG}
    for e in REG:
        if not os.path.exists(e["record"]): fail.append(f"[results_register] {e['id']}: record {e['record']} does not exist"); continue
        if os.path.isfile(e["record"]):
            said = re.search(r"\*\*Status: COMMISSIONED", open(e["record"], encoding="utf-8").read()) is not None
            if e["kind"] == "commissioning" and (e["status"] == "commissioned") != said: fail.append(f"[results_register] {e['id']}: register says {e['status']}, record says {'COMMISSIONED' if said else 'not commissioned'}")
            if e["status"] == "development" and said: fail.append(f"[results_register] {e['id']}: record says COMMISSIONED, register still says development")
        for x in e.get("compute") and [k for k, v in _V[e["id"]].items() if k == "inside" and v is False] or []: fail.append(f"[results_register] {e['id']}: listed as a passed result but its sealed rule is not met")
    ids = {e["id"]: e for e in REG}
    for e in REG:
        if e["kind"] == "commissioning":
            if "covers" not in e: fail.append(f"[results_register] {e['id']}: a commissioning entry must list 'covers' (the development results it closes; [] if none)")
            for c in e.get("covers", []):
                if c not in ids: fail.append(f"[results_register] {e['id']} covers {c}, which is not in the register")
                elif ids[c]["status"] == "development": fail.append(f"[results_register] {c} is covered by commissioning {e['id']} but is still status development: change its status and rerun results_to_tex.py")
    for f in [t for t in subprocess.run(["git", "ls-files", "Biological_Physics/MethylPhys/doors/*.md"], capture_output=True, text=True).stdout.split("\n") if t]:
        if not os.path.isfile(f): continue
        txt = open(f, encoding="utf-8").read()
        if re.search(r"\*\*Status: COMMISSIONED", txt) and f not in recs: fail.append(f"[results_register] {f} says COMMISSIONED but is not in the register")
        if "**Milestone:**" in txt and f not in recs and not any(f in e.get("also", []) for e in REG): fail.append(f"[results_register] {f} is marked **Milestone:** but is not in the register")
except Exception as ex: fail.append(f"[results_register] generator failed: {ex}")
if fail:
    print(f"status_check: REFUSED ({len(fail)})\n  " + "\n  ".join(fail[:40])); sys.exit(1)
print(f"status_check: ok ({len(F['facts'])} facts, {len(files)} documents)")
