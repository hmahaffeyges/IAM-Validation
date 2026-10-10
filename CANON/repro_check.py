#!/usr/bin/env python3
"""Push gate for the reproducibility rule (SOP, 2026-10-09). Refuses a push when
  1. the working tree has untracked files (anything not committed would be lost or left out), or
  2. a development/outcome note in Biological_Physics/MethylPhys/doors/ that is new or changed in this push carries computed numbers but
     has no committed script: one under doors/data/<NOTE>/ (or <NOTE> without _OUTCOME), or a script the note names that exists in the repo.
Notes still listed as 'open' in development/REPRODUCIBILITY_BACKLOG.md are reported but not refused (the backlog is being closed).
Pre-registrations (*_PREREG.md) are exempt: their numbers are bars set in advance."""
import re, subprocess, sys
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
fail = []
junk = re.compile(r"(__pycache__|\.pyc$|\.DS_Store|/_data/|/_cache/|/_atlas_cache/)")
untracked = [f for f in git("ls-files", "--others", "--exclude-standard").split("\n") if f and not junk.search(f)]
if untracked: fail.append("untracked files (commit them or add them to .gitignore):\n    " + "\n    ".join(untracked[:20]))
base = git("merge-base", "HEAD", "origin/main").strip() or "origin/main"
changed = [f for f in git("diff", "--name-only", base, "HEAD").split("\n") if re.match(r"Biological_Physics/MethylPhys/doors/[^/]+\.md$", f)]
tree = git("ls-tree", "-r", "--name-only", "HEAD").split("\n"); scripts = [t for t in tree if t.endswith((".py", ".sh", ".R"))]
backlog = git("show", "HEAD:development/REPRODUCIBILITY_BACKLOG.md")
for f in changed:
    stem = f.rsplit("/", 1)[1][:-3]
    if stem.endswith("_PREREG") or not re.match(r"(DEV|PROC|COMMISSIONING|BOXRUN|Q0)", stem): continue
    try: text = open(f, encoding="utf-8").read()
    except FileNotFoundError: continue
    if not re.search(r"\b\d+\.\d{2,}\b", text): continue
    k = re.sub(r"_OUTCOME$", "", stem)
    near = [s for s in scripts if f"/doors/data/{k}/" in s or f"/doors/data/{stem}/" in s]
    named = [s for s in scripts if s.rsplit("/", 1)[1] in text]
    if near or named: continue
    if re.search(rf"\| {re.escape(stem)} \|[^\n]*\| open \|", backlog): print(f"repro_check: {stem} is still on the backlog (no script yet)"); continue
    fail.append(f"{f}: numbers but no committed script (doors/data/{k}/ or a script the note names)")
if fail:
    print("repro_check: REFUSED\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"repro_check: ok ({len(changed)} note(s) changed in this push)")
