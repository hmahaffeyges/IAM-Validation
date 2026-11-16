#!/usr/bin/env python3
"""Replicate a runner's HEAD commit into the repository through GitHub's git database API (blobs, tree, commit) without creating or moving any
branch: the repository ruleset lets only the owner create or update refs (2026-11-15). Prints '::notice::<label> commit <sha> base <sha> tree <sha>';
the owner's session then points a branch (or main, after the gate) at that commit with one small request. Used by .github/workflows/edit.yml
and gated_merge.yml. Usage: python3 CANON/api_commit.py <label> [base, default HEAD~1]"""
import os, sys, base64, subprocess, requests
lab = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "HEAD~1"
g = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True).stdout
H = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}
R = "https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
def post(u, j):
    r = requests.post(R + u, headers=H, json=j, timeout=300)
    if r.status_code >= 300: print(f"::error::{lab}: POST {u} {r.status_code} {r.text[:200]}"); sys.exit(1)
    return r.json()["sha"]
b = g("rev-parse", base).decode().strip(); bt = g("rev-parse", b + "^{tree}").decode().strip(); ht = g("rev-parse", "HEAD^{tree}").decode().strip()
items = []
for l in g("diff", "--raw", "--no-renames", "--no-abbrev", b, "HEAD").decode().splitlines():
    meta, p = l.split("\t", 1); f = meta.split()
    if f[4] == "D": items.append({"path": p, "mode": f[0].lstrip(":") if f[0].lstrip(":") != "000000" else "100644", "type": "blob", "sha": None}); continue
    items.append({"path": p, "mode": f[1], "type": "blob", "sha": post("/git/blobs", {"content": base64.b64encode(g("show", "HEAD:" + p)).decode(), "encoding": "base64"})})
if not items: print(f"::error::{lab}: no change between {b[:8]} and HEAD"); sys.exit(1)
tr = post("/git/trees", {"base_tree": bt, "tree": items})
if tr != ht: print(f"::error::{lab}: the API tree {tr[:12]} differs from the runner's tree {ht[:12]}"); sys.exit(1)
c = post("/git/commits", {"message": g("log", "-1", "--format=%B").decode(), "tree": tr, "parents": [b],
                          "author": {"name": "Heath W. Mahaffey", "email": "hmahaffeyges@users.noreply.github.com"}})
print(f"::notice::{lab} commit {c} base {b} tree {tr} files {len(items)}")
