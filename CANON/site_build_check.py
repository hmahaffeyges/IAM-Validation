#!/usr/bin/env python3
"""Website build guard (2026-10-16). From d1dd1c1e (2026-10-15) every website build failed on one unescaped _ in the book, and the page and
book stopped updating for a day while pushes kept landing. This reads the latest finished 'website' run on main from GitHub: if it FAILED,
a push is refused unless it changes docs/book/ or website/ (a push that may fix the build). No network or token: a warning, not a refusal.
Usage: python3 CANON/site_build_check.py [base]   (base: the remote ref being pushed onto, default origin/main)"""
import os, sys, json, subprocess, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
tok = os.environ.get("GITHUB_TOKEN", "")
try:
    rq = urllib.request.Request("https://api.github.com/repos/hmahaffeyges/IAM-Validation/actions/runs?branch=main&per_page=30&status=completed",
                                headers={"Accept": "application/vnd.github+json", **({"Authorization": "Bearer " + tok} if tok else {})})
    R = [r for r in json.load(urllib.request.urlopen(rq, timeout=30))["workflow_runs"] if r["name"] == "website" and r["conclusion"] in ("success", "failure")]
except Exception as e:
    print(f"site_build_check: WARNING could not read the website build status ({type(e).__name__}); not checked"); sys.exit(0)
if not R or R[0]["conclusion"] == "success": print("site_build_check: ok (last website build " + (R[0]["conclusion"] + " at " + R[0]["head_sha"][:8] if R else "unknown") + ")"); sys.exit(0)
try:   # a newer build already running after the failure may be the fix: let work through, the next check will judge it
    rq2 = urllib.request.Request("https://api.github.com/repos/hmahaffeyges/IAM-Validation/actions/runs?branch=main&per_page=10",
                                 headers={"Accept": "application/vnd.github+json", **({"Authorization": "Bearer " + tok} if tok else {})})
    run = [r for r in json.load(urllib.request.urlopen(rq2, timeout=30))["workflow_runs"] if r["name"] == "website" and r["status"] != "completed" and r["created_at"] > R[0]["created_at"]]
    if run: print(f"site_build_check: last website build FAILED at {R[0]['head_sha'][:8]}, but a newer build ({run[0]['head_sha'][:8]}) is running - allowed"); sys.exit(0)
except Exception: pass
ch = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], capture_output=True, text=True).stdout.split()
if any(f.startswith(("docs/book/", "website/")) for f in ch):
    print(f"site_build_check: last website build FAILED at {R[0]['head_sha'][:8]}; this push changes the book or the site, so it may fix it - allowed. Confirm the next build passes."); sys.exit(0)
print(f"site_build_check: REFUSED: the last website build FAILED at {R[0]['head_sha'][:8]} ({R[0]['html_url']}); the page and the book are not updating. Fix the build first."); sys.exit(1)
