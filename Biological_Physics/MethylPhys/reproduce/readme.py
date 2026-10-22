#!/usr/bin/env python3
"""Writes Biological_Physics/MethylPhys/reproduce/README.md from tests.json (the tests) and verified.json (each test's last fresh-clone verdict, copied from its
verdict.json when the run lands). Usage: python3 Biological_Physics/MethylPhys/reproduce/readme.py [--check]"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); T = json.load(open(os.path.join(H, "tests.json"), encoding="utf-8"))["tests"]
V = json.load(open(os.path.join(H, "verified.json"), encoding="utf-8")) if os.path.exists(os.path.join(H, "verified.json")) else {}
L = ["# Reproduce IAM's results from public data", "",
 "Each test below starts from public accessions only (GEO, ArrayExpress, SRA, ENA, ENCODE, Zenodo), runs the committed code of the chain, and",
 "compares what it makes with the result committed in this repository. Nothing is read from our own storage.", "", "```bash",
 "git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation",
 "bash Biological_Physics/MethylPhys/reproduce/setup.sh                       # once: Python 3.11 with the chain's pinned packages (Biological_Physics/MethylPhys/reproduce/.venv)",
 "python3 Biological_Physics/MethylPhys/reproduce/reproduce.py --list         # the tests",
 "python3 Biological_Physics/MethylPhys/reproduce/reproduce.py cscore_band    # one test: prints each compared number and PASS or DIFF",
 "python3 Biological_Physics/MethylPhys/reproduce/reproduce.py --all          # every test", "```", "",
 "**On GitHub's free machines, the same ones we used.** Fork this repository, open the fork's Actions tab, turn workflows on, then run",
 "**reproduce_all** (`.github/workflows/reproduce_all.yml`). Every test runs on GitHub's fixed `ubuntu-24.04` machine image, the image our",
 "verified runs used; each job prints the image's exact version. Tests too big for one machine are split across many: IAM-A's line over 13",
 "machines, the mouse test over 19 and Group-A over 10. The run's summary page ends with a table of every test's verdict. A free account runs",
 "20 machines at once, so the whole set takes about 8 to 10 hours. One test alone: the **reproduce** workflow with the test's name.", "",
 "Each run writes `Biological_Physics/MethylPhys/reproduce/out/<test>/verdict.json` (every comparison, the commit, the time) and a log per step. The last column is the",
 "test's own run from a fresh public clone on a clean machine.", "",
 "## What each test shows, in plain words", ""]
for t in T:
    L += [f"**{t['title']}.** {t['plain']} *Why it matters:* {t['why']}", ""]
L += ["## The tests", "", "| test | what it shows | public inputs | time, disk | record | from a fresh clone |", "|---|---|---|---|---|---|"]
for t in T:
    v = V.get(t["id"]); ok = (f"**{v['verdict']}** {v['date']}, " + (f"{v['checks_ok']} of {v['checks']} numbers identical" if v.get("checks") else "compared numbers to be recorded")) if v else "being run"
    L.append(f"| `{t['id']}` | **{t['title']}.** {t['shows']} | {'; '.join(t['public_inputs'])} | {t['time']}, {t['disk']} | "
             f"[record](../{t['record']}) | {ok} |")
L += ["", "Tolerances: numbers are compared exactly (to the stored precision) except where a test's `tolerance_note` in [tests.json](tests.json)",
      "says why not (the sequencing aligner can place a few reads differently with a different thread count).", ""]
OUT = os.path.join(H, "README.md"); sys.path.insert(0, os.path.join(H, "..", "..", "..", "CANON"))
try: import relink as _rl; txt = _rl.fix("\n".join(L), OUT)
except ImportError: txt = "\n".join(L)   # a Biological_Physics-only clone has no CANON/; the committed README is already relinked
if "--check" in sys.argv:
    ok = open(OUT, encoding="utf-8").read() == txt; print("reproduce readme: " + ("ok" if ok else "REFUSED: run python3 Biological_Physics/MethylPhys/reproduce/readme.py")); sys.exit(0 if ok else 1)
open(OUT, "w", encoding="utf-8").write(txt); print("reproduce readme: written,", len(T), "tests")
