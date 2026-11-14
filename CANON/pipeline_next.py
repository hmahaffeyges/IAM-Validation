#!/usr/bin/env python3
"""pipeline_next.py (2026-11-14; the author: 'when one set of computers finishes with one derivation it starts the next one so we dont have
downtime'). Runs on GitHub (.github/workflows/pipeline.yml) whenever an explore or erase_test run finishes, and every 15 minutes. Reads the
queue (CANON/pipeline_queue.json), finds each item's runs by their run name 'pq:<id>', and starts every item that is ready:
  - every item it waits on ('after') finished successfully;
  - if it names a canary ('canary': id, 'canary_rule': script), the canary finished and the rule script, run on the canary's downloaded
    artifact, exits 0 (the numeric canary rule, as every canary so far);
  - its design sheet (design/<row>.json, or the item's own 'design' sheet for a later step of a row) passes CANON/design_check.py, the same check gh.py applies before any test job;
  - fewer than MAX_LIVE of this repository's runs are in progress (GitHub's free tier runs 20 jobs at once).
It only starts compute. It never scores a test or writes a result: scoring stays with a person and the sealed record. A failed item stops
everything that waits on it, and the run log says why. GitHub machines only; Google is never started from here.
  python3 CANON/pipeline_next.py          start what is ready
  python3 CANON/pipeline_next.py status   print each item's state
  python3 CANON/pipeline_next.py controls planted queues and run states must give the expected decisions"""
import os, sys, json, io, zipfile, subprocess, tempfile, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "CANON", "pipeline_queue.json")
MAX_LIVE = 18
REPO = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")

def api(path, method="GET", body=None, raw=False):
    rq = urllib.request.Request("https://api.github.com/repos/" + REPO + path, method=method, data=json.dumps(body).encode() if body else None,
                                headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(rq, timeout=300) as r:
        b = r.read(); return b if raw else (json.loads(b) if b else {})

def runs_by_tag():
    out = {}
    for page in (1, 2, 3):
        for w in api(f"/actions/runs?per_page=100&page={page}").get("workflow_runs", []):
            n = w.get("display_title") or w.get("name") or ""
            if n.startswith("pq:") and n[3:] not in out: out[n[3:]] = w   # newest first, so the first seen is the latest attempt
    return out

def live_count():
    return sum(api(f"/actions/runs?status={s}&per_page=1").get("total_count", 0) for s in ("in_progress", "queued"))

def state(item, R):
    w = R.get(item["id"])
    if not w: return "pending"
    if w["status"] != "completed": return "running"
    return "done" if w["conclusion"] == "success" else "failed"

def canary_ok(item, R):
    c = item.get("canary")
    if not c: return True, ""
    w = R.get(c)
    if not w or w["status"] != "completed": return False, "canary not finished"
    if w["conclusion"] != "success": return False, "canary failed"
    A = api(f"/actions/runs/{w['id']}/artifacts").get("artifacts", [])
    if not A: return False, "canary has no artifact"
    d = tempfile.mkdtemp(); zipfile.ZipFile(io.BytesIO(api(A[0]["archive_download_url"].split(REPO)[1], raw=True))).extractall(d)
    rc = subprocess.run([sys.executable, os.path.join(ROOT, item["canary_rule"]), d], capture_output=True, text=True)
    return rc.returncode == 0, (rc.stdout + rc.stderr).strip()[-300:]

def design_ok(item):
    row = item["row"]
    if row.startswith("OTH-"): return True, ""
    p = os.path.join(ROOT, item.get("design") or f"design/{row}.json")   # a step of a row may carry its own sheet
    if not os.path.exists(p): return False, f"design/{row}.json missing"
    sys.path.insert(0, os.path.join(ROOT, "CANON")); import design_check
    S = json.load(open(p)); F = design_check.sheet_findings(S, row)
    if F: return False, "design sheet: " + "; ".join(F)[:300]
    if str((S.get("machines") or {}).get("platform", "")).lower() != "github": return False, "design sheet does not name github"
    return True, ""

def decide(queue, R, live, design=design_ok, canary=canary_ok):
    """Pure decision: which items to start now, and why the others wait. Used by the runner and by the controls."""
    st = {i["id"]: state(i, R) for i in queue}; start, why = [], {}
    for i in queue:
        if st[i["id"]] != "pending": continue
        deps = list(i.get("after", [])) + ([i["canary"]] if i.get("canary") else [])
        bad = [d for d in deps if st.get(d) == "failed"]
        if bad: why[i["id"]] = f"blocked: {', '.join(bad)} failed"; continue
        if any(st.get(d) != "done" for d in deps): why[i["id"]] = "waiting on " + ", ".join(d for d in deps if st.get(d) != "done"); continue
        ok, msg = canary(i, R)
        if not ok: why[i["id"]] = "canary rule: " + msg; continue
        ok, msg = design(i)
        if not ok: why[i["id"]] = msg; continue
        if live + len(start) >= MAX_LIVE: why[i["id"]] = "machines full"; continue
        start.append(i)
    return st, start, why

def main(mode):
    queue = json.load(open(Q))["items"]
    R = runs_by_tag(); live = live_count()
    st, start, why = decide(queue, R, live)
    for i in queue: print(f"{i['id']:24} {st[i['id']]:8} {why.get(i['id'], '')}")
    if mode == "status": return
    for i in start:
        inp = dict(i.get("inputs", {})); inp["tag"] = "pq:" + i["id"]
        api(f"/actions/workflows/{i['workflow']}/dispatches", "POST", {"ref": "main", "inputs": inp})
        print("started", i["id"], "on", i["workflow"])

def controls():
    Rr = lambda s, c=None: {"status": s, "conclusion": c, "id": 1}
    q = [dict(id="a", row="PHY-1"), dict(id="b", row="PHY-1", after=["a"]), dict(id="c", row="PHY-1", canary="a", canary_rule="x"),
         dict(id="d", row="PHY-1", after=["z"]), dict(id="z", row="PHY-1")]
    yes = lambda *a: (True, ""); no = lambda *a: (False, "planted")
    cno = lambda i, R: (False, "planted") if i.get("canary") else (True, "")   # like canary_ok: no canary named, nothing to fail
    cases = [  # (runs, live, design, canary, expected started ids)
        ({}, 0, yes, yes, {"a", "z"}),
        ({"a": Rr("in_progress")}, 0, yes, yes, {"z"}),
        ({"a": Rr("completed", "success"), "z": Rr("completed", "failure")}, 0, yes, yes, {"b", "c"}),
        ({"a": Rr("completed", "success")}, 0, yes, cno, {"b", "z"}),
        ({"a": Rr("completed", "success")}, 0, no, yes, set()),
        ({}, MAX_LIVE - 1, yes, yes, {"a"}),
    ]
    ok = True
    for n, (R, live, d, c, want) in enumerate(cases):
        got = {i["id"] for i in decide(q, R, live, d, c)[1]}
        if got != want: ok = False; print(f"case {n}: started {sorted(got)}, expected {sorted(want)}")
    _, _, why = decide(q, {"z": Rr("completed", "failure")}, 0, yes, yes)
    ok &= why.get("d", "").startswith("blocked")
    print("pipeline_next controls:", "ok" if ok else "FAILED"); return ok

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "run"
    if m == "controls": sys.exit(0 if controls() else 1)
    main(m)
