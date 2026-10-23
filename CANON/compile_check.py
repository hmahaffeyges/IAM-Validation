#!/usr/bin/env python3
"""CANON/compile_check.py - refuse a push when a Python file in the repository does not compile (2026-10-22, lesson C36: a test script that
did not compile was pushed and dispatched, because the path check skips a file it cannot parse). Compiles every tracked .py file outside
archive folders, without running it, and names the file, line and error of each that fails."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = [f for f in subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n") if f and "/archive/" not in f and "RETIRED" not in f]
bad = []
for f in F:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p): continue
    try: compile(open(p, "rb").read(), f, "exec")
    except SyntaxError as e: bad.append(f"{f}: line {e.lineno}: {e.msg}")
if bad: print("compile_check: REFUSED (%d)" % len(bad)); print("\n".join("  " + b for b in bad)); sys.exit(1)
print(f"compile_check: ok ({len(F)} files compile)")
