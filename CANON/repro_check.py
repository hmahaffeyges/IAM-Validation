#!/usr/bin/env python3
"""Push gate for the reproducibility rule (SOP, 2026-10-09). Refuses a push when
  1. the working tree has untracked files (anything not committed would be lost or left out), or
  2. a development/outcome note in Biological_Physics/MethylPhys/doors/ that is new or changed in this push carries computed numbers but
     has no committed script: one under doors/data/<NOTE>/ (or <NOTE> without _OUTCOME), or a script the note names that exists in the repo.
Notes still listed as 'open' in development/REPRODUCIBILITY_BACKLOG.md are reported but not refused (the backlog is being closed).
Pre-registrations (*_PREREG.md) are exempt: their numbers are bars set in advance.
Rule 7 (2026-10-10): a note that adds a prediction, bar or seal must name its committed synthetic test (script + output).\nRule 6 (2026-10-10): a number in a new or changed note that no committed output of its data folder carries (see below).\nRecord rule (2026-10-10). Also refuses a push when
  3. an outcome note (doors/*_OUTCOME.md) new or changed in this push is not named in development/METHYLPHYS_DEVELOPMENT_LOG.md,
     or the log did not change in the same push;
  4. a note that carries a "**Milestone:**" line is not linked from the Advancements table of Biological_Physics/README.md,
     or a link in that table points at a file that does not exist;
  5. a day with log entries (from 2026-10-10) has no "### <date> · Day summary" once a later day has begun (Pacific time)."""
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
# ---- record rule (checks 3-5)
import os, datetime
from zoneinfo import ZoneInfo
LOGF = "development/METHYLPHYS_DEVELOPMENT_LOG.md"; README = "Biological_Physics/README.md"
alld = git("diff", "--name-only", base, "HEAD").split("\n")
log = open(LOGF, encoding="utf-8").read() if os.path.exists(LOGF) else ""
for f in changed:
    stem = f.rsplit("/", 1)[1][:-3]
    if not stem.endswith("_OUTCOME"): continue
    k = stem[:-8]; ids = {stem, k, k.replace("_", "-"), stem + ".md"}
    if LOGF not in alld: fail.append(f"{f}: outcome changed but the development log was not updated in this push"); continue
    if not any(i in log for i in ids): fail.append(f"{f}: outcome not named in {LOGF} (name {k.replace('_', '-')} or {stem}.md)")
rd = open(README, encoding="utf-8").read() if os.path.exists(README) else ""
adv = rd.split("## Advancements", 1)[1].split("\n## ", 1)[0] if "## Advancements" in rd else ""
for t in re.findall(r"\]\(([^)#]+)\)", adv):
    if not t.startswith("http") and not os.path.exists(os.path.normpath(os.path.join("Biological_Physics", t))): fail.append(f"{README} Advancements: broken link {t}")
for f in [t for t in tree if re.match(r"Biological_Physics/MethylPhys/doors/[^/]+\.md$", t)]:
    try: txt = open(f, encoding="utf-8").read()
    except FileNotFoundError: continue
    if "**Milestone:**" in txt and f.rsplit("/", 1)[1] not in adv: fail.append(f"{f}: marked **Milestone:** but not linked in {README} Advancements")
days = sorted({d for d in re.findall(r"^### (\d{4}-\d{2}-\d{2})", log, re.M) if d >= "2026-10-10"})
today = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
for d in days:
    if (d < today or d < days[-1]) and not re.search(rf"^### {d} · Day summary", log, re.M): fail.append(f"{LOGF}: no '### {d} · Day summary' (a later day has begun)")
# 6. Number traceability (2026-10-10, author: "make sure that every bit of code and data necessary for replication makes it to the repo").
# Every decimal number with >= 2 decimals in a note NEW or CHANGED in this push must appear, at the note's precision, in a committed
# file of the note's data folder (doors/data/<NOTE>/, <NOTE>_PLANT/, rows/outputs/json/scripts) or in development/sims. DOIs and `code spans`
# are skipped. Numbers quoted from another record are traced by putting that record's file in the data folder or naming it in a code span.
NUM6 = re.compile(r"(?<![\w./])[+\u2212-]?\d+\.\d{2,}(?![\w/])")
def _pool6(dirs):
    P = []
    for d in dirs:
        for t in [x for x in tree if x.startswith(d)]:
            if t.endswith((".txt", ".csv", ".json", ".py", ".md")) and os.path.isfile(t) and os.path.getsize(t) < 50_000_000:
                P += [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", open(t, encoding="utf-8", errors="replace").read())]
    return P
for f in changed:
    if not os.path.isfile(f) or f.endswith("_PREREG.md"): continue
    k = os.path.basename(f)[:-3]; stem = k.replace("_OUTCOME", "")
    txt = re.sub(r"(?i)doi[: ]\S+|10\.\d{4,}/\S+|`[^`]*`", "", open(f, encoding="utf-8").read())
    N = set(m.group(0).lstrip("+").replace("\u2212", "-") for m in NUM6.finditer(txt))
    if not N: continue
    P = _pool6([f"Biological_Physics/MethylPhys/doors/data/{stem}/", f"Biological_Physics/MethylPhys/doors/data/{stem}_PLANT/", "development/sims/"])
    miss = sorted(n for n in N if not any(abs(round(v, len(n.split(".")[1])) - float(n)) < 1e-12 or abs(round(-v, len(n.split(".")[1])) - float(n)) < 1e-12 for v in P))
    if miss: fail.append(f"{f}: {len(miss)} number(s) not found in any committed output of doors/data/{stem}/: {', '.join(miss[:10])}")
# 7. Synthetic test before sealing (author 2026-10-10: "EVERY test with a sealed prediction MUST be first tested thoroughly with synthetic
# data so we dont waste the few tests available to us"). A doors/*.md note whose lines ADDED in this push carry a prediction, bar or seal
# (**Prediction..., **Bar..., **Bars..., **Pass bar..., **Sealed..., "sealed" with a date) must carry a line
#   **Synthetic test:** path, path, ...
# naming committed files (repo-relative, or relative to doors/data/), at least one script (.py/.sh) and one output (.txt/.csv/.json),
# every one present in HEAD. Outcome notes are exempt (they read a seal, they do not make one).
SEAL7 = re.compile(r"\*\*(Prediction|Bars?\b|Pass bar|Sealed|Window sealed|Rule sealed)|\bsealed (on |before |\d{4}-\d\d-\d\d)", re.I)
SYN7 = re.compile(r"^\*\*Synthetic test:\*\*\s*(.+)$", re.M)
treeset = set(tree)
for f in changed:
    if not os.path.isfile(f) or f.endswith("_OUTCOME.md") or f.endswith("_PREREG.md"): continue
    added = "\n".join(l[1:] for l in git("diff", base, "HEAD", "--", f).split("\n") if l.startswith("+") and not l.startswith("+++"))
    if not SEAL7.search(added): continue
    m = SYN7.search(open(f, encoding="utf-8").read())
    if not m: fail.append(f"{f}: adds a prediction/bar/seal but has no '**Synthetic test:**' line (rule 7)"); continue
    paths = [x.strip().strip("`") for x in re.split(r"[,;]", m.group(1)) if x.strip()]
    res = []
    for x in paths:
        cands = [x, "Biological_Physics/MethylPhys/doors/data/" + x, "Biological_Physics/MethylPhys/" + x]
        hit = next((c for c in cands if c in treeset), None)
        if hit is None: fail.append(f"{f}: Synthetic test file not committed: {x} (rule 7)")
        else: res.append(hit)
    if res and not any(r.endswith((".py", ".sh")) for r in res): fail.append(f"{f}: Synthetic test names no committed script (rule 7)")
    if res and not any(r.endswith((".txt", ".csv", ".json")) for r in res): fail.append(f"{f}: Synthetic test names no committed output (rule 7)")
if fail:
    print("repro_check: REFUSED\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"repro_check: ok ({len(changed)} note(s) changed in this push)")
