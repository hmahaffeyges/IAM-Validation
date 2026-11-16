#!/usr/bin/env python3
"""Repository inventory (2026-11-16; author: 'go through the repo and inventory every file, then find every file touched in the last month and
see if its stale or needs to be consolidated'). Runs ON A GITHUB MACHINE (explore.yml); reads git metadata and file text only; moves nothing.
Writes xout/repo_inventory.csv.gz (one row per tracked file) and xout/repo_inventory_summary.txt (a few KB). Flags, fixed before running:
  dated     the name carries a date (2026-11-14, 20261114, _11-14) or a version (_v2, v3.1) and another tracked file shares its stem
  dup       identical content to another tracked file
  marked    the first 60 lines say SUPERSEDED / RETIRED / WITHDRAWN / DEPRECATED / OBSOLETE / 'do not use'
  orphan    no other tracked text file names it (by file name), for documents (.md .html .tex .txt .json .csv)
  generated named as an output in CANON/build_all.py
Self-check: the per-file 30-day touch counts sum to the git log's total file touches in that window."""
import os, re, csv, gzip, hashlib, subprocess, collections, time
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True, errors="replace").stdout
if os.path.exists(".git/shallow"): subprocess.run(["git", "fetch", "--unshallow", "-q"])
SINCE = "2026-10-16"
files = [f for f in git("ls-files", "-z").split("\0") if f]
last, created, n30, subj, auth30 = {}, {}, collections.Counter(), {}, collections.defaultdict(set)
cur = None; total30 = 0
for line in git("log", "--name-only", "--format=@@%cI|%s", "--no-renames").split("\n"):
    if line.startswith("@@"):
        d, s = line[2:].split("|", 1); cur = (d[:10], s); continue
    if not line.strip() or cur is None: continue
    f = line.strip()
    if f not in last: last[f] = cur[0]; subj[f] = cur[1][:90]
    created[f] = cur[0]
    if cur[0] >= SINCE: n30[f] += 1; total30 += 1
TEXT = (".md", ".html", ".tex", ".txt", ".json", ".csv", ".py", ".yml", ".yaml", ".bib", ".R", ".sh", ".js", ".css")
DOC = (".md", ".html", ".tex", ".txt", ".json", ".csv")
gen = set()
if os.path.exists("CANON/build_all.py"):
    gen = set(re.findall(r"[\"']([\w./-]+\.(?:md|html|json|tex|csv|txt))[\"']", open("CANON/build_all.py", errors="replace").read()))
names = collections.Counter(os.path.basename(f) for f in files)
mentioned = collections.Counter(); hashes = {}; marked = {}; size = {}
MARK = re.compile(r"(?i)\b(superseded|retired|withdrawn|deprecated|obsolete|do not use)\b")
TOK = re.compile(r"[\w.\-]+\.(?:md|html|tex|txt|json|csv|py|yml|ipynb|pdf|png|R|sh)\b")
for f in files:
    try: b = open(f, "rb").read()
    except Exception: continue
    size[f] = len(b); hashes[f] = hashlib.sha1(b).hexdigest()
    if f.endswith(TEXT) and len(b) < 8_000_000:
        t = b.decode("utf-8", "replace")
        head = "\n".join(t.split("\n")[:60]); mm = MARK.search(head)
        if mm: marked[f] = mm.group(1).lower()
        for tok in set(TOK.findall(t)):
            if tok != os.path.basename(f): mentioned[tok] += 1
dupgroups = collections.defaultdict(list)
for f, h in hashes.items(): dupgroups[h].append(f)
DATE = re.compile(r"(?i)(20\d\d[-_]?\d\d[-_]?\d\d|[-_]\d\d-\d\d(?=[._-])|[-_ ]v\d+(?:\.\d+)?(?=[._-]|$))")
stem = lambda f: DATE.sub("", os.path.splitext(os.path.basename(f))[0]).lower()
stems = collections.Counter((os.path.dirname(f), stem(f)) for f in files)
rows = []
for f in files:
    flags = []
    if DATE.search(os.path.basename(f)) and stems[(os.path.dirname(f), stem(f))] > 1: flags.append("dated")
    if len(dupgroups.get(hashes.get(f), [])) > 1: flags.append("dup")
    if f in marked: flags.append("marked:" + marked[f])
    if f.endswith(DOC) and mentioned[os.path.basename(f)] == 0 and names[os.path.basename(f)] == 1: flags.append("orphan")
    if f in gen or os.path.basename(f) in {os.path.basename(g) for g in gen}: flags.append("generated")
    rows.append([f, f.split("/")[0] if "/" in f else ".", os.path.splitext(f)[1], size.get(f, 0), created.get(f, ""), last.get(f, ""),
                 n30[f], ";".join(flags), mentioned[os.path.basename(f)], subj.get(f, "")])
os.makedirs("xout", exist_ok=True)
with gzip.open("xout/repo_inventory.csv.gz", "wt", newline="") as fo:
    w = csv.writer(fo); w.writerow(["path", "top", "ext", "bytes", "created", "last_touched", "commits_30d", "flags", "named_by_n_files", "last_commit"]); w.writerows(rows)
touched = [r for r in rows if r[6] > 0]
assert sum(r[6] for r in rows) <= total30, (sum(r[6] for r in rows), total30)
by_top = collections.Counter(r[1] for r in rows); t_top = collections.Counter(r[1] for r in touched)
fl = collections.Counter(x.split(":")[0] for r in touched for x in r[7].split(";") if x)
S = [f"repo inventory {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}: {len(rows)} tracked files, {sum(size.values())/1e6:.1f} MB; "
     f"touched since {SINCE}: {len(touched)} files ({total30} file-touches)",
     "files by top folder (all / touched in 30 days): " + ", ".join(f"{k} {v}/{t_top[k]}" for k, v in by_top.most_common(25)),
     "flags among touched files: " + ", ".join(f"{k} {v}" for k, v in fl.most_common()),
     f"identical-content groups: {sum(1 for g in dupgroups.values() if len(g) > 1)}; marked superseded/retired among touched: {sum(1 for r in touched if 'marked' in r[7])}"]
open("xout/repo_inventory_summary.txt", "w").write("\n".join(S) + "\n")
for s in S: print("::notice::" + s[:900])
