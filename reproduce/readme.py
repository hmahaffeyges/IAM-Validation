#!/usr/bin/env python3
"""Writes reproduce/README.md from tests.json (the tests) and verified.json (each test's last fresh-clone verdict, copied from its
verdict.json when the run lands). Usage: python3 reproduce/readme.py [--check]"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); T = json.load(open(os.path.join(H, "tests.json"), encoding="utf-8"))["tests"]
V = json.load(open(os.path.join(H, "verified.json"), encoding="utf-8")) if os.path.exists(os.path.join(H, "verified.json")) else {}
L = ["# Reproduce IAM's results from public data", "",
 "Each test below starts from public accessions only (GEO, ArrayExpress, SRA, ENA, ENCODE, Zenodo), runs the committed code of the chain, and",
 "compares what it makes with the result committed in this repository. Nothing is read from our own storage.", "", "```bash",
 "git clone https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation",
 "bash reproduce/setup.sh                       # once: Python 3.11 with the chain's pinned packages (reproduce/.venv)",
 "python3 reproduce/reproduce.py --list         # the tests",
 "python3 reproduce/reproduce.py cscore_band    # one test: prints each compared number and PASS or DIFF",
 "python3 reproduce/reproduce.py --all          # every test", "```", "",
 "Each run writes `reproduce/out/<test>/verdict.json` (every comparison, the commit, the time) and a log per step. The last column is the",
 "test's own run from a fresh public clone on a clean machine.", "",
 "| test | what it shows | public inputs | time, disk | record | from a fresh clone |", "|---|---|---|---|---|---|"]
for t in T:
    v = V.get(t["id"]); ok = f"**{v['verdict']}** {v['date']}, {v['checks_ok']} of {v['checks']} numbers identical" if v else "being run"
    L.append(f"| `{t['id']}` | **{t['title']}.** {t['shows']} | {'; '.join(t['public_inputs'])} | {t['time']}, {t['disk']} | "
             f"[record](../{t['record']}) | {ok} |")
L += ["", "Tolerances: numbers are compared exactly (to the stored precision) except where a test's `tolerance_note` in [tests.json](tests.json)",
      "says why not (the sequencing aligner can place a few reads differently with a different thread count).", ""]
txt = "\n".join(L); OUT = os.path.join(H, "README.md")
if "--check" in sys.argv:
    ok = open(OUT, encoding="utf-8").read() == txt; print("reproduce readme: " + ("ok" if ok else "REFUSED: run python3 reproduce/readme.py")); sys.exit(0 if ok else 1)
open(OUT, "w", encoding="utf-8").write(txt); print("reproduce readme: written,", len(T), "tests")
