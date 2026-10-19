#!/usr/bin/env python3
"""Manifest freshness gate (author 2026-10-16: "It should demand that every place from the log to every doc that we maintain should be updated.
The manifest is one of the most important."). Compares the push (HEAD) with what is already on origin/main and refuses when:
 a. a development-log heading added in this push names no manifest row ID (PHY-047, COM-IAMA-053, ...), or names rows none of which changed
 b. a machine-ledger job (CANON/box_jobs.json) is added or changes state, and none of its manifest rows changed
 c. a doors record changes and a manifest row points to it (its 'record'), and none of those rows changed
 d. a chain-stamped result added or changed in this push (stamp written from 2026-10-16) names no manifest row
 e. CANON/manifest.json is behind its chain results (CANON/chain_events.py --check)
A row 'changed' means its JSON differs from origin/main (new rows count as changed). CANON/manifest_fresh_controls.py must pass first.
Usage: python3 CANON/manifest_fresh.py check [base]"""
import os, sys, json, re, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = "Biological_Physics/MethylPhys/METHYLPHYS_DEVELOPMENT_LOG.md"; ID = re.compile(r"\b(?:[A-Z]{2,5}(?:-[A-Z0-9]+)*-\d{2,3})\b")
CUT = "2026-10-16"
def findings(old_rows, new_rows, old_log, new_log, old_jobs, new_jobs, changed, stamps):
    """old/new_rows: {id: row}; *_log: text; *_jobs: {job_id: job}; changed: repo paths changed; stamps: [(path, stamp)] added or changed."""
    ch = {i for i, r in new_rows.items() if old_rows.get(i) != r}; F = []
    oh = set(l for l in old_log.split("\n") if l.startswith("### ")); new_h = [l for l in new_log.split("\n") if l.startswith("### ") and l not in oh]
    for h in new_h:
        if h[4:14] < CUT or "Day summary" in h: continue
        ids = [i for i in ID.findall(h) if i in new_rows]
        if not ids: F.append(f"[a] log heading names no manifest row: {h[:90]}")
        elif not set(ids) & ch: F.append(f"[a] log heading names {', '.join(ids)}, none changed in this push: {h[:70]}")
    for j, x in new_jobs.items():
        o = old_jobs.get(j)
        if o is not None and o.get("state") == x.get("state"): continue
        rows = [i for i in x.get("manifest", []) if i in new_rows]
        if not rows: F.append(f"[b] machine job {j} {'added' if o is None else 'changed state'} and names no manifest row")
        elif not set(rows) & ch: F.append(f"[b] machine job {j} {'added' if o is None else 'changed state'}; none of {', '.join(rows)} changed")
    for p in changed:
        if "/doors/" not in p or not p.endswith(".md"): continue
        rows = [i for i, r in new_rows.items() if str(r.get("record", "")).endswith(p.split("/doors/")[-1]) and "/doors/" in str(r.get("record", ""))]
        if rows and not set(rows) & ch: F.append(f"[c] record {p.split('/')[-1]} changed; none of its rows ({', '.join(rows[:4])}) changed")
    for p, st in stamps:
        if str(st.get("utc", "")) >= CUT and not st.get("manifest"): F.append(f"[d] chain result names no manifest row: {p}")
    return F
def git(*a): return subprocess.run(["git", "-C", R] + list(a), capture_output=True, text=True).stdout
def at(rev, path):
    t = git("show", f"{rev}:{path}"); return t
def rows_of(t):
    try: return {r["id"]: r for r in json.loads(t)["rows"]}
    except Exception: return {}
def jobs_of(t):
    try: return {j["job_id"]: j for j in json.loads(t)["jobs"]}
    except Exception: return {}
if __name__ == "__main__" and sys.argv[1:2] == ["check"]:
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
    changed = [x for x in git("diff", "--name-only", base, "HEAD").split("\n") if x]
    stamps = []
    for p in changed:
        if p.endswith(".json") and os.path.isfile(os.path.join(R, p)):
            try: o = json.load(open(os.path.join(R, p), encoding="utf-8"))
            except Exception: continue
            if isinstance(o, dict) and isinstance(o.get("chain_stamp"), dict): stamps.append((p, o["chain_stamp"]))
    F = findings(rows_of(at(base, "CANON/manifest.json")), rows_of(at("HEAD", "CANON/manifest.json")), (at(base, LOG) or at(base, "development/METHYLPHYS_DEVELOPMENT_LOG.md")), at("HEAD", LOG),
                 jobs_of(at(base, "CANON/box_jobs.json")), jobs_of(at("HEAD", "CANON/box_jobs.json")), changed, stamps)
    ce = subprocess.run([sys.executable, os.path.join(R, "CANON/chain_events.py"), "--check"], capture_output=True, text=True)
    if ce.returncode: F.append("[e] " + ce.stdout.strip().replace("\n", " | ")[:200])
    print("manifest_fresh: " + ("ok" if not F else "REFUSED (%d)" % len(F))); [print("  " + f) for f in F]; sys.exit(1 if F else 0)
