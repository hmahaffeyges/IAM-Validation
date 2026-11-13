#!/usr/bin/env python3
"""The book's full check record on a GitHub runner (2026-11-13; OTH-040). The full checkout has every committed data file. Installs the
book's math packages, runs verify_book.py three ways and writes verify_book_output.txt, verify_book_output.json and
VERIFY_BOOK_INVENTORY.md into xout/ (the explore workflow uploads them). The same run as the website's 'Run every check' button."""
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy", "mpmath", "scipy", "numpy", "pandas"], check=True)
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
O = os.path.abspath("xout")
os.makedirs(O, exist_ok=True)
RUNS = (([], "verify_book_output.txt"), (["--json"], "verify_book_output.json"), (["--inventory-md"], "VERIFY_BOOK_INVENTORY.md"))
for args, name in RUNS:
    with open(os.path.join(O, name), "w") as f:
        err = subprocess.STDOUT if not args else subprocess.DEVNULL
        r = subprocess.run([sys.executable, "verify_book.py"] + args, cwd=B, stdout=f, stderr=err)
    print(name, "rc", r.returncode)
print(open(os.path.join(O, "verify_book_output.txt")).read().splitlines()[-1][:300])
