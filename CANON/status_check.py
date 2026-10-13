#!/usr/bin/env python3
"""Status gate (2026-10-10). Reads CANON/status_facts.json and refuses a push when
  1. a living document still carries a phrase a fact forbids (the old status),
  2. a document a fact requires no longer states it,
  3. the book's check count printed in README.md differs from the committed verify_book output, or that output no longer counts every
     check verify_book.py defines (new checks added without regenerating the output).
Run: python3 CANON/status_check.py   (called by CANON/checked_push.sh)"""
import subprocess, json as _js, glob, json, os, re, subprocess, sys
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
    tex, rd, REG, _V, _rr, _fr = _m.build()
    for _e in REG:   # rule 14 (author 2026-10-12): every result explained in plain words for a donor following along
        if _e.get("status") != "withdrawn" and not (_e.get("plain") or "").strip(): fail.append(f"[results_register] {_e['id']} has no plain-language explanation ('plain')")
    for _rec, _txt in _m.plain_blocks(REG).items():
        _n = open(_rec, encoding="utf-8").read() if os.path.isfile(_rec) else ""
        if _m.PB not in _n or _n[_n.index(_m.PB) + len(_m.PB):_n.index(_m.PE)].strip() != _txt.strip(): fail.append(f"[results_register] {_rec}: In plain words block differs from the register: run python3 CANON/results_to_tex.py")
    _f = open(_m.RFOLLOW, encoding="utf-8").read()
    if _m.FB not in _f or _m.FE not in _f or _f[_f.index(_m.FB) + len(_m.FB):_f.index(_m.FE)].strip() != _fr.strip(): fail.append("[results_register] FOLLOW.md results differ from the register: run python3 CANON/results_to_tex.py")
    _t = open(_m.RROOT, encoding="utf-8").read()
    if _m.RB not in _t or _m.RE not in _t or _t[_t.index(_m.RB) + len(_m.RB):_t.index(_m.RE)].strip() != _rr.strip(): fail.append("[results_register] README.md cell results differ from the register: run python3 CANON/results_to_tex.py")
    _mi = __import__("subprocess").run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_inventory.py"), "--check"], capture_output=True, text=True)
    if _mi.returncode != 0: fail.append("[master_inventory] " + (_mi.stdout.strip() or _mi.stderr.strip()[-200:]))
    _nr = __import__("subprocess").run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "now_running.py"), "--check"], capture_output=True, text=True)
    if _nr.returncode != 0: fail.append("[now_running] " + (_nr.stdout.strip() or _nr.stderr.strip()[-200:]))
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
# GitHub rejects a workflow file with a repeated trigger key, and PyYAML accepts it silently, so three pushes once built
# nothing (2026-10-13). Check the keys directly.
for _w in sorted(glob.glob(".github/workflows/*.yml")):
    _b = open(_w, encoding="utf-8").read()
    if "on:" not in _b: continue
    _top = [l.strip().rstrip(":") for l in _b[_b.index("on:"):].split("\n") if re.fullmatch(r"  [a-z_]+:", l)]
    _dup = sorted({k for k in _top if _top.count(k) > 1})
    if _dup: fail.append(f"{_w}: repeated key(s) {', '.join(_dup)} - GitHub will reject this workflow")

# Every item of the author's to-do list must be a job, a track, an idea, housekeeping or closed (2026-10-13)
_tc = subprocess.run([sys.executable, "CANON/todo_coverage.py", "--check"], capture_output=True, text=True)
if _tc.returncode: fail.append("[todo_coverage] " + (_tc.stdout or _tc.stderr).strip().split("\n")[-1])

# Every losses and decisions entry must point at a record that exists and carries plain words (2026-10-13)
_R = _js.load(open("CANON/results_register.json", encoding="utf-8")) if os.path.exists("CANON/results_register.json") else {}
for _k in ("losses", "decisions"):
    for _x in _R.get(_k, []):
        if not os.path.exists(_x.get("record", "")): fail.append(f"[results_register] {_k} entry {_x.get('date')} names a record that does not exist: {_x.get('record')}")
        elif not _x.get("plain"): fail.append(f"[results_register] {_k} entry {_x.get('date')} has no plain-words text")

# The foundation must stay sound: anchors present, no retired phrasing in a file this push changed (2026-10-13)
_fc = subprocess.run([sys.executable, "CANON/foundation_check.py"], capture_output=True, text=True)
if _fc.returncode:
    for _l in (_fc.stdout or _fc.stderr).strip().split("\n")[1:]: fail.append("[foundation] " + _l.strip())

# The lever register is the single source for the lever chapter: falsifiers present, terms known to the foundation,
# records and synthetic tests committed, generated section in step (2026-10-13)
_lv = subprocess.run([sys.executable, "CANON/levers_check.py"], capture_output=True, text=True)
if _lv.returncode:
    for _l in (_lv.stdout or _lv.stderr).strip().split("\n")[1:]: fail.append("[levers] " + _l.strip())

# The chain manifest: every chain file is live, a tool, or class-era, and the computed live set matches (2026-10-13)
_cm = subprocess.run([sys.executable, "CANON/chain_manifest.py"], capture_output=True, text=True)
if _cm.returncode:
    for _l in (_cm.stdout or _cm.stderr).strip().split("\n")[1:]: fail.append("[chain] " + _l.strip())

# The traced case: every step must point at a record that exists, and every commit it cites must be a real commit (2026-10-13).
# The whole value of that section is that a reader can check the order from outside, so a dangling reference voids it.
_tr = _js.load(open("CANON/trace.json", encoding="utf-8"))
for _s in _tr.get("steps", []):
    if not os.path.isfile(_s["rec"]): fail.append(f"[trace] step {_s['n']}: record {_s['rec']} does not exist")
    for _h in re.findall(r"\b[0-9a-f]{8}\b", _s.get("c", "")):
        if subprocess.run(["git", "cat-file", "-e", _h + "^{commit}"], capture_output=True).returncode:
            fail.append(f"[trace] step {_s['n']}: commit {_h} is not in this repository")

# The candidate-instrument board is generated from CANON/instruments.json (2026-10-13): every entry keeps a falsifier and a
# next step, every record named exists, and the board does not drift from the register.
_ib = subprocess.run([sys.executable, "CANON/instruments_board.py", "--check"], capture_output=True, text=True)
if _ib.returncode:
    _lines = [l.strip() for l in _ib.stdout.split("\n") if l.strip().startswith("[instruments]")]
    # a checker that CRASHES must not pass silently (the levers checker had this same bug, 2026-10-13)
    fail += _lines or ["[instruments] instruments_board failed to run: " + (_ib.stderr.strip().split("\n")[-1] if _ib.stderr.strip() else "no output")]

if fail:
    print(f"status_check: REFUSED ({len(fail)})\n  " + "\n  ".join(fail[:40])); sys.exit(1)
print(f"status_check: ok ({len(F['facts'])} facts, {len(files)} documents)")
