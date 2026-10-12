"""Ledger guard (2026-10-12; author: 'we must use code always to demand accuracy'). The fundraiser page promises that every rented-computer run is
listed in the public ledger, with its cost estimate, BEFORE it starts. require(TAG) enforces that: it reads RUNS.md as published on GitHub
(origin/main, fetched now) and refuses unless the Upcoming table has a row whose run tag is TAG and whose estimate states a dollar amount.
Every launcher (spot_from_box.py, workbox.py, start_box.py) calls it before any machine starts. Usage: python3 ledger_guard.py TAG"""
import os, re, subprocess, sys
ROOT = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
def published_runs_md():
    subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", "main"], capture_output=True)
    r = subprocess.run(["git", "-C", ROOT, "show", "origin/main:Biological_Physics/MethylPhys/RUNS.md"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""
def check(tag, md):
    up = md.split("## Upcoming", 1)[1].split("\n## ", 1)[0] if "## Upcoming" in md else ""
    for line in up.split("\n"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 5 and c[0] == f"`{tag}`":
            return (True, "") if re.search(r"\$\s?\d", c[-1]) else (False, f"ledger row for {tag} has no dollar estimate")
    return False, f"{tag} is not in the Upcoming table of the published ledger (RUNS.md on GitHub)"
def require(tag):
    ok, why = check(tag, published_runs_md())
    if not ok: raise SystemExit(f"LEDGER GUARD: refused to start a machine: {why}. Add it to UPCOMING in make_runs_ledger.py, regenerate RUNS.md and push first.")
    print(f"ledger guard: {tag} is listed with its estimate", flush=True)
if __name__ == "__main__": require(sys.argv[1])
