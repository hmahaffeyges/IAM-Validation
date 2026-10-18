#!/usr/bin/env python3
"""Reproduce IAM's results from public data with one command (2026-10-18).

Every test in reproduce/tests.json starts from public accessions only (GEO, SRA/ENA), runs the committed code of the chain on them, and
compares what it makes with the result committed in this repository. Nothing is read from our own storage.

  bash reproduce/setup.sh                      # once: the chain's pinned environment in reproduce/.venv
  python3 reproduce/reproduce.py --list        # the tests, with time, disk and what each shows
  python3 reproduce/reproduce.py cscore_band   # one test; writes reproduce/out/<test>/verdict.json and prints PASS or DIFF
  python3 reproduce/reproduce.py --all         # every test, in order

A test passes when every compared number equals the committed one within the stated tolerance (most are exact).
"""
import json, os, re, subprocess, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
T = json.load(open(os.path.join(HERE, "tests.json"), encoding="utf-8"))
PY = os.environ.get("REPRO_PY") or os.path.join(HERE, ".venv", "bin", "python")

def get(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d

def compare(t, work):
    rows = []
    for c in t["compare"]:
        if c["kind"] == "json":
            P = json.load(open(os.path.join(work, c["produced"]))); C = json.load(open(os.path.join(REPO, c["committed"])))
            for f, tol in c["fields"].items():
                a, b = get(P, f), get(C, f)
                ok = (a == b) if (tol is None or not isinstance(b, (int, float)) or isinstance(b, bool)) else abs(float(a) - float(b)) <= tol
                rows.append(dict(check=f"{os.path.basename(c['committed'])}: {f}", committed=b, reproduced=a, tolerance=tol, ok=bool(ok)))
        elif c["kind"] == "log":
            txt = open(os.path.join(work, "logs", c["step"] + ".log"), encoding="utf-8", errors="replace").read()
            for pat in c["must_contain"]:
                rows.append(dict(check=f"{c['step']} log: {pat}", committed=pat, reproduced=("found" if re.search(pat, txt) else "not found"), tolerance=None, ok=bool(re.search(pat, txt))))
    return rows

def run(tid):
    t = next((x for x in T["tests"] if x["id"] == tid), None)
    if t is None: sys.exit(f"no test {tid!r}; see --list")
    work = os.path.join(HERE, "out", tid); os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    env = dict(os.environ, REPO=REPO, WORK=work, PY=PY, MP=os.path.join(REPO, "Biological_Physics", "MethylPhys"))
    t0 = time.time(); print(f"== {tid}: {t['title']}")
    for s in t["steps"]:
        print(f"   {s['name']} ...", flush=True); lp = os.path.join(work, "logs", s["name"] + ".log")
        with open(lp, "w") as L: rc = subprocess.run(s["run"], shell=True, cwd=REPO, env=env, stdout=L, stderr=subprocess.STDOUT, executable="/bin/bash").returncode
        if rc != 0:
            v = dict(test=tid, verdict="FAILED", step=s["name"], exit_code=rc, log=os.path.relpath(lp, REPO)); json.dump(v, open(os.path.join(work, "verdict.json"), "w"), indent=1)
            print(f"   FAILED at {s['name']} (exit {rc}); see {os.path.relpath(lp, REPO)}"); return v
    rows = compare(t, work); ok = all(r["ok"] for r in rows)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    v = dict(test=tid, title=t["title"], verdict="PASS" if ok else "DIFF", commit=commit, utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
             minutes=round((time.time() - t0) / 60, 1), checks=rows)
    json.dump(v, open(os.path.join(work, "verdict.json"), "w"), indent=1, default=str)
    for r in rows: print(f"   {'ok  ' if r['ok'] else 'DIFF'} {r['check']}: committed {r['committed']} | reproduced {r['reproduced']}")
    print(f"   {v['verdict']} ({v['minutes']} min)"); return v

if __name__ == "__main__":
    a = sys.argv[1:] or ["--list"]
    if a[0] == "--list":
        for t in T["tests"]: print(f"{t['id']:<20} {t['time']:<12} {t['disk']:<8} {t['title']}")
    elif a[0] == "--all":
        V = [run(t["id"]) for t in T["tests"]]; print("\n".join(f"{v['test']:<20} {v['verdict']}" for v in V)); sys.exit(0 if all(v["verdict"] == "PASS" for v in V) else 1)
    else:
        V = [run(x) for x in a]; sys.exit(0 if all(v["verdict"] == "PASS" for v in V) else 1)
