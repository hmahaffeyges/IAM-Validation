#!/usr/bin/env python3
"""Push gate for the reproducibility rule (SOP, 2026-10-09). Refuses a push when
  1. the working tree has untracked files (anything not committed would be lost or left out), or
  2. a development/outcome note in Biological_Physics/MethylPhys/doors/ that is new or changed in this push carries computed numbers but
     has no committed script: one under doors/data/<NOTE>/ (or <NOTE> without _OUTCOME), or a script the note names that exists in the repo.
Notes still listed as 'open' in development/REPRODUCIBILITY_BACKLOG.md are reported but not refused (the backlog is being closed).
Pre-registrations (*_PREREG.md) are exempt: their numbers are bars set in advance.
Rule 9 (2026-10-11): laboratory independence from GEO contributors (CANON/independence_check.py).\nRule 8 (2026-10-11): a note that adds a prediction, bar or seal and names a dataset analysed earlier by another record (CANON/data_spent.csv) must\ncarry a one-line '**Prior reading:**' disclosure naming each such dataset (what analysed it, and that it was a different quantity where so).\nDisclosure only: it does not change a result's status.\nRule 7 (2026-10-10): a note that adds a prediction, bar or seal must name its committed synthetic test (script + output).\nRule 6 (2026-10-10): a number in a new or changed note that no committed output of its data folder carries (see below).\nRecord rule (2026-10-10). Also refuses a push when
  3. an outcome note (doors/*_OUTCOME.md) new or changed in this push is not named in development/METHYLPHYS_DEVELOPMENT_LOG.md,
     or the log did not change in the same push;
  4. a note that carries a "**Milestone:**" line is not linked from the Advancements table of Biological_Physics/README.md,
     or a link in that table points at a file that does not exist;
  5. a day with log entries (from 2026-10-10) has no "### <date> · Day summary" once a later day has begun (Pacific time);
  10. a box-run folder (boxruns/runN*) is not in the run ledger (boxruns/make_runs_ledger.py), or RUNS.md is missing;
  11. the ledger guard's controls (boxruns/ledger_guard_test.py) fail;
  12. an outcome note new or changed in this push does not declare **Milestone:** or **Not a milestone:** <reason>;
  13. a **Milestone:** note has no results-register entry, or a register entry's record lacks the line or does not exist;
  15. a number on the live fundraiser page (fundraiser/GOFUNDME_STORY_LIVE.txt) no longer matches its record;
  16. donor-facing text (READMEs, FOLLOW.md, fundraiser text, register plain/title/headline) has a British spelling;
  17. a lesson named in the log is not a row of LESSONS.md, a frozen older lessons file changes, or a new lessons file appears;
  18. the funding statement in README.md, FOLLOW.md or the book preface differs from CANON/funding.json;
  19. the atlas README's last search for new purified cells is more than 6 months old;
  20. a new or changed outcome note's test note has no '**Test data:**' line."""
import os, re, subprocess, sys
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
# are skipped. A number may instead sit in the data folder of a test cited on the same line (DEV-X-NN / PROC-X-NN), and must then be in it.
NUM6 = re.compile(r"(?<![\w./])[+\u2212-]?\d+\.\d{2,}(?![\w/])")
def _pool6(dirs, note=None):
    """Outputs of the named folders, plus every doors/data/<DIR>/ the note itself points at (2026-10-13: a note's data folder
    need not be named after the note - the C-score plan's folder is doors/data/CSCORE_COMMISSIONING/)."""
    dirs = list(dirs)
    if note and os.path.isfile(note):
        for d in set(re.findall(r"data/([A-Za-z0-9_]+)/", open(note, encoding="utf-8", errors="replace").read())):
            dirs.append(f"Biological_Physics/MethylPhys/doors/data/{d}/")
    P = []
    for d in dirs:
        for t in [x for x in tree if x.startswith(d)]:
            if t.endswith((".txt", ".csv", ".json", ".py", ".md")) and os.path.isfile(t) and os.path.getsize(t) < 50_000_000:
                P += [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", open(t, encoding="utf-8", errors="replace").read())]
    return P
def _strip_generated(t):
    """A note's text without the blocks the register generator writes and without its **Milestone:** line (rules 12-14 add these)."""
    t = re.sub(r"<!-- PLAIN:BEGIN.*?<!-- PLAIN:END -->", "", t, flags=re.S)
    t = re.sub(r"^\*\*Test data:\*\*.*$", "", t, flags=re.M)   # rule 20 line (accessions only)
    return "\n".join(l for l in t.split("\n") if not l.startswith("**Milestone:**")).strip()
def generated_only(f):
    old = git("show", f"{base}:{f}")
    return bool(old) and os.path.isfile(f) and _strip_generated(old) == _strip_generated(open(f, encoding="utf-8").read())
for f in [x for x in changed if not generated_only(x)]:   # a change only inside generated blocks adds no number (2026-10-12)
    if not os.path.isfile(f) or f.endswith("_PREREG.md"): continue
    k = os.path.basename(f)[:-3]; stem = k.replace("_OUTCOME", "")
    txt = re.sub(r"(?i)doi[: ]\S+|10\.\d{4,}/\S+|`[^`]*`", "", open(f, encoding="utf-8").read())
    N = set(m.group(0).lstrip("+").replace("\u2212", "-") for m in NUM6.finditer(txt))
    if not N: continue
    P = _pool6([f"Biological_Physics/MethylPhys/doors/data/{stem}/", f"Biological_Physics/MethylPhys/doors/data/{stem}_PLANT/", "development/sims/"], note=f)
    def _hit(n, pool): k_ = len(n.split(".")[1]); v0 = float(n); return any(abs(round(v, k_) - v0) < 1e-12 or abs(round(-v, k_) - v0) < 1e-12 for v in pool)
    # A number may also come from a test CITED ON ITS OWN LINE (DEV-X-NN / PROC-X-NN): it must then be in that test's data folder (2026-10-10).
    cited = {}
    for line in txt.split("\n"):
        tags = {t.replace("-", "_") for t in re.findall(r"\b(?:DEV|PROC)-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d\d\b", line)}
        tags |= set(re.findall(r"\b((?:DEV|PROC)_[A-Z0-9_]+?_\d\d)(?:_OUTCOME)?\.md\b", line))   # cited by file name
        for m in NUM6.finditer(line): cited.setdefault(m.group(0).lstrip("+").replace("\u2212", "-"), set()).update(tags)
    _cache = {}
    def _pool_of(t):
        if t not in _cache: _cache[t] = _pool6([f"Biological_Physics/MethylPhys/doors/data/{t}/", f"Biological_Physics/MethylPhys/doors/data/{t}_PLANT/"], note=f)
        return _cache[t]
    miss = sorted(n for n in N if not _hit(n, P) and not any(_hit(n, _pool_of(t)) for t in cited.get(n, ())))
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
    dl = git("diff", base, "HEAD", "--", f).split("\n")
    removed = [l[1:] for l in dl if l.startswith("-") and not l.startswith("---")]
    # an edited line that was already a seal/prediction line (same opening) is a correction, not a new seal (2026-10-11)
    old_heads = {r.strip()[:40] for r in removed if SEAL7.search(r)}
    added = "\n".join(l[1:] for l in dl if l.startswith("+") and not l.startswith("+++") and not (SEAL7.search(l[1:]) and l[1:].strip()[:40] in old_heads))
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
import csv
SPENT = {r["dataset"].strip(): (r["spent_status"], r.get("spent_record", "")) for r in csv.DictReader(open("CANON/data_spent.csv", encoding="utf-8")) if r["spent_status"].startswith(("spent", "partly"))}
ACC8 = re.compile(r"\b(GSE\d{4,}|PRJNA\d+|PRJDB\d+|TCGA-[A-Z]{2,5}|ENCSR[0-9A-Z]{6})\b")
def spent_of(a): return a if a in SPENT else ("TCGA" if a.startswith("TCGA-") and "TCGA" in SPENT else None)
IDRX = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,4}\b")
def prior_by_other(f, a):
    own = os.path.basename(f).replace("_OUTCOME", "").replace("_PREREG", "").replace(".md", "").replace("_", "-")
    rec = SPENT[spent_of(a)][1]; ids = {i for i in IDRX.findall(rec) if not i.startswith(("GSE", "PRJ", "TCGA", "SRR", "GSM"))}
    return (not ids) or bool(ids - {own})       # a reading by any other record (or an unnamed one) counts
def rule8(f, txt):
    hits = sorted({a for a in ACC8.findall(txt) if spent_of(a) and prior_by_other(f, a)}); m = re.search(r"^\*\*Prior reading:\*\*(.+(?:\n(?!\*\*|#).+)*)", txt, re.M)
    return [a for a in hits if not (m and a in m.group(1))]
for f in changed:
    try: txt = open(f, encoding="utf-8").read()
    except FileNotFoundError: continue
    if not SEAL7.search(txt): continue
    miss = rule8(f, txt)
    if miss: fail.append(f"{f}: seals on data analysed earlier ({', '.join(miss)}; CANON/data_spent.csv) without a one-line '**Prior reading:**' disclosure (rule 8)")
sys.path.insert(0, "CANON"); from independence_check import check as rule9
for f in changed:
    try: fail += rule9(f, open(f, encoding="utf-8").read())
    except FileNotFoundError: pass
if "--audit9" in sys.argv:
    import glob
    for f in sorted(glob.glob("Biological_Physics/MethylPhys/doors/*.md")):
        for x in rule9(f, open(f, encoding="utf-8").read()): print("AUDIT9", x)
    sys.exit(0)
if "--audit8" in sys.argv:
    import glob
    for f in sorted(glob.glob("Biological_Physics/MethylPhys/doors/*.md")):
        txt = open(f, encoding="utf-8").read()
        if SEAL7.search(txt) and rule8(f, txt): print("AUDIT8", f, rule8(f, txt))
    sys.exit(0)
# 10. Run ledger (author 2026-10-11: donors follow every job and its cost). Every box-run folder must be named in the ledger generator
#     (boxruns/make_runs_ledger.py, RUNS_PURPOSE or UPCOMING), and RUNS.md must exist.
import os as _os, re as _re
_BR = "Biological_Physics/MethylPhys/boxruns"; _LG = _os.path.join(_BR, "make_runs_ledger.py")
if _os.path.isdir(_BR):
    if not _os.path.exists("Biological_Physics/MethylPhys/RUNS.md"): fail.append("Biological_Physics/MethylPhys/RUNS.md missing (rule 10)")
    _src = open(_LG, encoding="utf-8").read() if _os.path.exists(_LG) else ""
    _named = {m.upper() for m in _re.findall(r"BOXRUN(\d+)", _src)} | {m.upper() for m in _re.findall(r"Box Run (\d+)", _src)}
    for _d in sorted(_os.listdir(_BR)):
        _m = _re.match(r"run(\d+)", _d)
        if _m and _os.path.isdir(_os.path.join(_BR, _d)) and _m.group(1) not in _named:
            fail.append(f"{_BR}/{_d}: box run not in the run ledger ({_LG}, rule 10)")

# 11. Ledger guard (author 2026-10-12: code must enforce the page's promise that every run is listed with its cost before it starts).
import subprocess as _sp
_T = "Biological_Physics/MethylPhys/boxruns/ledger_guard_test.py"
if _os.path.exists(_T):
    _r = _sp.run([sys.executable, _T], capture_output=True, text=True)
    if _r.returncode != 0: fail.append(f"{_T}: ledger guard controls failed: " + (_r.stdout.strip().split(chr(10)) or [""])[-1])

# 12. Every outcome note new or changed in this push declares whether it is a milestone (author 2026-10-12: "the code needs to update these
#     places so it never gets missed again"): a "**Milestone:**" line, or "**Not a milestone:** <reason>".
# 13. Milestones and the results register match both ways: a note marked **Milestone:** is the record of a register entry, and every register
#     entry's record exists and carries the **Milestone:** line. The register generates the READMEs, FOLLOW.md and the book chapter.
import json as _js, glob
for f in changed:
    if not f.endswith("_OUTCOME.md") or not os.path.isfile(f): continue
    _x = open(f, encoding="utf-8").read()
    if "**Milestone:**" not in _x and "**Not a milestone:**" not in _x: fail.append(f"{f}: declare '**Milestone:**' or '**Not a milestone:** <reason>' (rule 12)")
_REG = _js.load(open("CANON/results_register.json", encoding="utf-8"))["entries"]; _recs = {e["record"] for e in _REG if e.get("status") != "withdrawn"}
for f in sorted(glob.glob("Biological_Physics/MethylPhys/doors/*.md")):
    if "**Milestone:**" in open(f, encoding="utf-8").read() and f not in _recs: fail.append(f"{f}: marked **Milestone:** but has no entry in CANON/results_register.json (rule 13)")
for r in sorted(_recs):
    if not r.startswith("Biological_Physics/MethylPhys/doors/"): continue
    if not os.path.isfile(r): fail.append(f"CANON/results_register.json: record {r} does not exist (rule 13)")
    elif "**Milestone:**" not in open(r, encoding="utf-8").read(): fail.append(f"{r}: in the results register but has no **Milestone:** line (rule 13)")

# 15. The live fundraiser page's numbers match the records (fundraiser/check_story.py; author 2026-10-12).
if _os.path.exists("fundraiser/check_story.py"):
    _r = _sp.run([sys.executable, "fundraiser/check_story.py"], capture_output=True, text=True)
    if _r.returncode != 0: fail.append("fundraiser/check_story.py: the live fundraiser page no longer matches the records: " + "; ".join(l for l in _r.stdout.split(chr(10)) if l.startswith("FAIL")))

# 16. Donor-facing text in American spelling (author 2026-10-12; CANON/us_spelling.py).
_r = _sp.run([sys.executable, "CANON/us_spelling.py", "check"], capture_output=True, text=True)
if _r.returncode != 0: fail.append("British spelling in donor-facing text (run python3 CANON/us_spelling.py fix): " + "; ".join(_r.stdout.split(chr(10))[:8]))

# 17. One lessons file (author 2026-10-12). Every lesson the log names (e.g. "LESSONS T15", "Lesson C6") is a row of
#     Biological_Physics/MethylPhys/LESSONS.md; the older topic lists it indexes are frozen records; no other lessons file may be added.
_LF = "Biological_Physics/MethylPhys/LESSONS.md"
_FROZEN = {"Biological_Physics/MethylPhys/atlas/IAMAtlas_FLATNESS_LESSON.md", "Biological_Physics/MethylPhys/atlas/v2/IAMAtlas_FLATNESS_LESSON_v1.md",
           "Biological_Physics/MethylPhys/atlas/v2/LESSONS.md", "Biological_Physics/MethylPhys/chain/CPG_Lessons_Learned_2026-06-29.md"}
_rows = set(re.findall(r"^\| ([A-Z]{1,2}\d{1,2}) \|", git("show", "HEAD:" + _LF), re.M))
_ids = set(re.findall(r"\b(?:LESSONS|[Ll]esson) ([A-Z]{1,2}\d{1,2})\b", git("show", "HEAD:development/METHYLPHYS_DEVELOPMENT_LOG.md")))
if _ids - _rows: fail.append(f"{_LF}: lessons named in the development log have no row: {sorted(_ids - _rows)} (rule 17)")
_all = [f for f in git("diff", "--name-only", base, "HEAD").split("\n") if f]
for _f in _all:
    if _f in _FROZEN: fail.append(f"{_f}: a frozen lessons record was changed; add the lesson to {_LF} instead (rule 17)")
    elif re.search(r"lesson", _f.rsplit("/", 1)[-1], re.I) and _f != _LF and _f not in _FROZEN and git("cat-file", "-t", f"HEAD:{_f}").strip():
        fail.append(f"{_f}: a new lessons file; lessons go in {_LF} only (rule 17)")

# 18. One funding statement (author 2026-10-12): README.md, FOLLOW.md and the book preface carry the block written from CANON/funding.json.
_r = _sp.run([sys.executable, "CANON/funding_sync.py", "check"], capture_output=True, text=True)
if _r.returncode != 0: fail.append("funding statement out of step (run python3 CANON/funding_sync.py write): " + "; ".join(_r.stdout.split(chr(10))[:4]) + " (rule 18)")

# 19. Atlas search every 6 months (author 2026-10-12): the atlas README's "Last atlas search for new purified cells: YYYY-MM-DD" is <= 183 days old.
import datetime as _dt19
_m19 = re.search(r"Last atlas search for new purified cells: (\d{4}-\d\d-\d\d)", open("Biological_Physics/MethylPhys/atlas/README.md", encoding="utf-8").read())
if not _m19: fail.append("atlas README has no 'Last atlas search for new purified cells: <date>' line (rule 19)")
elif (_dt19.date.today() - _dt19.date.fromisoformat(_m19.group(1))).days > 183: fail.append(f"atlas search overdue: last {_m19.group(1)}, more than 6 months ago (rule 19)")

# 20. Master Inventory (author 2026-10-12): a new or changed outcome note's test note names its data on a "**Test data:**" line, so the
#     inventory (CANON/master_inventory.py, checked by status_check) records which sets each test read.
for f in changed:
    if f.startswith("Biological_Physics/MethylPhys/doors/") and f.endswith("_OUTCOME.md"):
        dn = f[:-len("_OUTCOME.md")] + ".md"
        try: dt = open(dn, encoding="utf-8").read()
        except FileNotFoundError: continue
        if "**Test data:**" not in dt: fail.append(f"{dn}: no '**Test data:** <sets>' line for its outcome (rule 20)")

# 21. A held-out test set must never be one we have already read (2026-10-13, LESSONS C11): in a note new or changed in this push, a line
# naming a set as the test / held-out set is refused when that accession is in CANON/data_spent.csv. Build-only use is unaffected.
_SPENT = set()
for _l in open("CANON/data_spent.csv", encoding="utf-8", errors="replace").read().split("\n")[1:]:
    _SPENT.update(re.findall(r"\b(?:GSE\d{4,6}|E-MTAB-\d{3,5}|PRJ[A-Z]{2}\d{3,8}|TCGA-[A-Z]{2,4})\b", _l.split(",")[0]))
for f in [x for x in changed if os.path.isfile(x)]:
    for _l in open(f, encoding="utf-8", errors="replace").read().split("\n"):
        if not re.search(r"(?i)\b(held[- ]out|test set|as the test|test data)\b", _l): continue
        if re.search(r"(?i)builds only|build set|build only|not valid as|not a valid|removed|spent|already read|unread", _l): continue
        for _a in re.findall(r"\b(?:GSE\d{4,6}|E-MTAB-\d{3,5}|PRJ[A-Z]{2}\d{3,8}|TCGA-[A-Z]{2,4})\b", _l):
            # the rule catches borrowing a set someone else has read; a note that itself names the set as its OWN test
            # and records the outcome (unsuitable, unread, no fingerprint) is the reason the set is spent, not a violation
            _own = re.search(r"(?i)not suitable|unread|no fingerprint|stays sealed|was read by this", open(f, encoding="utf-8", errors="replace").read()[:4000])
            if _a in _SPENT and not _own:
                fail.append(f"{f}: {_a} is named as a held-out test set but is in CANON/data_spent.csv (rule 21)")

# 22. Plain words follow VISIBILITY, not milestone status (2026-10-13). The rule used to apply only to notes declaring a
# **Milestone:**, so the lung notes, the colorectal replication and the atlas plan - all reachable from the public page -
# carried none and the author had to ask. A note changed in this push that records an outcome, or is named by the results
# register (result, loss or decision), must carry an "In plain words" paragraph AND a "What it would take" sentence.
_REGJ = _js.load(open("CANON/results_register.json", encoding="utf-8"))
_pub = {e.get("record") for e in _REGJ.get("entries", [])} | {x.get("record") for x in _REGJ.get("losses", [])} | {x.get("record") for x in _REGJ.get("decisions", [])}
for f in [x for x in changed if x.startswith("Biological_Physics/MethylPhys/doors/") and x.endswith(".md") and os.path.isfile(x)]:
    _t = open(f, encoding="utf-8", errors="replace").read()
    if not (re.search(r"(?i)\b(not suitable|no fingerprint|NOT MET|BAR MET|: MET\b|FINGERPRINT\b)", _t) or f in _pub): continue
    if "In plain words" not in _t:
        fail.append(f"{f}: records an outcome or is shown on the page but has no 'In plain words' paragraph (rule 22)")
    elif "What it would take" not in _t:
        fail.append(f"{f}: has plain words but does not say what it would take to succeed or finish (rule 22)")

# 23. A synthetic test must read through the CHAIN, not through a second implementation of the same physics (2026-10-13).
# DEV-LEVERS-01 test 3 v1 reimplemented the held error and the clustering reading in the test script, got the erase channel's
# direction backwards, and contradicted the derivation it was testing. A script named on a **Synthetic test:** line must either
# import from Biological_Physics/MethylPhys/chain, or carry a line beginning "# NO CHAIN:" saying why it cannot.
for f in [x for x in changed if x.startswith("Biological_Physics/MethylPhys/doors/") and x.endswith(".md") and os.path.isfile(x)]:
    for _l in open(f, encoding="utf-8", errors="replace").read().split("\n"):
        if not _l.startswith("**Synthetic test:**"): continue
        for _x in re.findall(r"[A-Za-z0-9_/]+\.(?:py|sh)", _l):
            _p = "Biological_Physics/MethylPhys/doors/data/" + _x
            if not os.path.isfile(_p): continue
            _src = open(_p, encoding="utf-8", errors="replace").read()
            if "chain" in _src and re.search(r"(?m)^\s*(import|from)\s", _src) and re.search(r"chain|conductor|stage_", _src):
                if re.search(r"(?m)^\s*(from|import)\s+(conductor_v3|stage_\w+|dev_stages|sky_map|cmb_tools)", _src) or "sys.path.insert" in _src and "chain" in _src:
                    continue
            if "# NO CHAIN:" in _src: continue
            fail.append(f"{f}: synthetic test {_x} neither imports the chain nor declares '# NO CHAIN: <why>' (rule 23)")

# 24. The day log is the page's live status (2026-10-13, author: one stop shop; the hand-kept "running now" board went stale and
# advertised a job that had not run for a day). If this push changes an outcome note or a doors/ record, there must be a day log
# whose date is the newest date in this push, and the page's status line comes from its "Open at turnover".
_doors = [f for f in changed if f.startswith("Biological_Physics/MethylPhys/doors/") and f.endswith(".md")]
if _doors:
    import datetime as _dt, subprocess as _sp
    _d = _sp.run(["git", "log", "-1", "--format=%ad", "--date=format:%Y-%m-%d"], capture_output=True, text=True).stdout.strip()
    _p = os.path.join("CANON", "daylogs", _d + ".json")
    if not os.path.isfile(_p):
        fail.append(f"{_d}: this push changes {len(_doors)} record(s) but CANON/daylogs/{_d}.json does not exist - the day log is the "
                    f"page's status, so work without one leaves the page stale (rule 24)")
    else:
        _j = _js.load(open(_p, encoding="utf-8"))
        if not _j.get("open_at_turnover"): fail.append(f"{_p}: no 'open_at_turnover' - that paragraph is what the page shows (rule 24)")

# 25. The held balance is independent of division (author 2026-10-14: "It is independent of fucking division damnit!!! Stop reverting
# back to this"; LEV-002; foundation copy_error_term). A line ADDED in this push may not make division a term of the held error or the
# gauge. Allowed: the author's own tumor-phase hypothesis (a line naming falsifier 7, the two-phase picture or an established tumor),
# and lines marked as a dated or retired record.
_DIV25 = re.compile(r"(?i)\bper (cell )?divisions?\b|\bper[- ]copy\b|\bat copying\b|\bdivision adds\b|missed restores at copying|"
                    r"written only when (the )?DNA is copied|third culprit")
_OK25 = re.compile(r"(?i)falsifier 7|two-phase|established tumou?r|retired wording|dated record|RULE 25 EXEMPT")
_all25 = [f for f in git("diff", "--name-only", base, "HEAD").split("\n") if f and f.endswith((".md", ".tex", ".json", ".py"))
          and f.startswith(("docs/book/", "Biological_Physics/MethylPhys/doors/", "CANON/", "website/", "development/"))
          and f != "CANON/repro_check.py" and os.path.isfile(f)]
for f in _all25:
    for l in git("diff", base, "HEAD", "--", f).split("\n"):
        if l.startswith("+") and not l.startswith("+++") and _DIV25.search(l) and not _OK25.search(l):
            fail.append(f"{f}: makes division a term of the held error ('{_DIV25.search(l).group(0)}'); the balance is independent of division (rule 25): {l[1:120].strip()}")

# 26. "Copy error" is a retired term (author 2026-10-14: "amend the glossary definition of copy error, to say that it is NOT what we measure
# in IAM or what determines the A score ... so you dont keep forgetting"; glossary 'Copy error (retired term)', foundation copy_error_term).
# What is read is the HELD error: the error of the write-erase balance at the held sites. A line ADDED in this push may not use the old
# name, except a line that says it is retired. Dated records already written are not touched.
_CE26 = re.compile(r"(?i)\bcopy[- ]errors?\b"); _OK26 = re.compile(r"(?i)retired|formerly called|older name")
for f in _all25:
    for l in git("diff", base, "HEAD", "--", f).split("\n"):
        if l.startswith("+") and not l.startswith("+++") and _CE26.search(l) and not _OK26.search(l):
            fail.append(f"{f}: uses the retired term 'copy error'; write 'held error' (rule 26): {l[1:120].strip()}")

# 27. Nothing that runs or is planned stays off the Master Testing Manifest (author 2026-10-15: "we are gonna lose ideas and plans this way").
#  a. every box job dated 2026-10-15 or later names its manifest rows ("manifest": [ids]) and every id exists;
#  b. a line ADDED in this push to a whiteboard or ideas file that states a plan, a next step, an idea or a test must carry a manifest id.
import json
_M27 = json.load(open("CANON/manifest.json", encoding="utf-8")); _ids27 = {r["id"] for r in _M27["rows"]}
for _j in json.load(open("CANON/box_jobs.json", encoding="utf-8"))["jobs"]:
    if str(_j.get("date", "")) >= "2026-10-15":
        _m = _j.get("manifest") or []
        if not _m: fail.append(f"box job {_j['job_id'][:8]} ({_j.get('name','')[:60]}): no manifest row named (rule 27)")
        for _x in _m:
            if _x not in _ids27: fail.append(f"box job {_j['job_id'][:8]}: manifest row {_x} does not exist (rule 27)")
_ID27 = re.compile(r"\b[A-Z]{3}-\d{3}\b|\bDEV-[A-Z0-9-]+\b|\bPROC-[A-Z0-9-]+\b")
_PLAN27 = re.compile(r"(?i)^\s*(?:[-*]\s+|\d+\.\s+)?(?:\*\*)?(?:next|plan|idea|to test|test next|queued|still to come|where the next session starts)\b")
_f27 = [f for f in git("diff", "--name-only", base, "HEAD").split()
        if re.match(r"(development/(WHITEBOARD|IDEAS)_[^/]*\.md$|Biological_Physics/MethylPhys/whiteboard/.*\.md$)", f) and os.path.isfile(f)]
for f in _f27:
    for l in git("diff", base, "HEAD", "--", f).split("\n"):
        if l.startswith("+") and not l.startswith("+++") and _PLAN27.search(l[1:]) and not _ID27.search(l):
            fail.append(f"{f}: a plan, idea or test with no manifest id (rule 27): {l[1:110].strip()}")

if fail:
    print("repro_check: REFUSED\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"repro_check: ok ({len(changed)} note(s) changed in this push)")
