#!/usr/bin/env python3
"""Chain events -> Master Testing Manifest (author 2026-10-16: "when the chain runs, it should have a certain amount of updates that happen
automatically from the chain script"). Reads every committed chain-stamped result (a .json with a chain_stamp naming manifest rows) and writes,
on each named row of CANON/manifest.json:
  chain_outputs : one line per output folder: folder, files, stage, latest chain version and time (generated; never typed)
  runs          : one run per stamped result carrying a 'verdict' (MET, NOT MET, INCONCLUSIVE, NOT ASSESSED, REPRODUCED, DIFFERS), added once,
                  keyed by the result's payload hash
Idempotent: running it twice changes nothing. checkpoint.sh runs it; --check refuses a manifest that is behind its chain results.
Usage: python3 CANON/chain_events.py [--check]"""
import os, sys, json, subprocess, collections
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MF = os.path.join(R, "CANON/manifest.json")
OUTCOME = {"MET": "success", "REPRODUCED": "success", "NOT MET": "fail", "DIFFERS": "fail", "INCONCLUSIVE": "progressed", "NOT ASSESSED": "progressed"}
def stamped():
    fs = subprocess.run(["git", "-C", R, "ls-files", "*.json"], capture_output=True, text=True).stdout.split("\n")
    for f in fs:
        if not f or not (f.startswith("Biological_Physics/") or f.startswith("development/")): continue
        p = os.path.join(R, f)
        try: o = json.load(open(p, encoding="utf-8"))
        except Exception: continue
        st = o.get("chain_stamp") if isinstance(o, dict) else None
        if isinstance(st, dict) and st.get("manifest"): yield f, o, st
def build(M):
    rows = {r["id"]: r for r in M["rows"]}; out = collections.defaultdict(dict); runs = collections.defaultdict(list); missing = []
    for f, o, st in stamped():
        for rid in st["manifest"]:
            if rid not in rows: missing.append(f"{f}: names manifest row {rid}, which does not exist"); continue
            d = os.path.dirname(f); e = out[rid].setdefault(d, dict(folder=d, files=0, stage=st.get("stage"), chain=str(st.get("chain_sha", ""))[:12], latest_utc=""))
            e["files"] += 1
            if st.get("utc", "") > e["latest_utc"]: e.update(latest_utc=st.get("utc", ""), chain=str(st.get("chain_sha", ""))[:12], stage=st.get("stage"))
            v = o.get("verdict")
            if isinstance(v, str) and v.upper() in OUTCOME:
                runs[rid].append(dict(date=st.get("utc", "")[:10], outcome=OUTCOME[v.upper()], note=f"{v.upper()} (chain result {f})",
                                      chain_result=f, payload_sha256=st.get("payload_sha256"), page="not headline: recorded automatically from a chain result; any front-page call is made in its record"))
    for rid, r in rows.items():
        co = sorted(out.get(rid, {}).values(), key=lambda x: x["folder"])
        if co: r["chain_outputs"] = co
        elif "chain_outputs" in r: del r["chain_outputs"]
        have = {x.get("payload_sha256") for x in r.get("runs", []) if isinstance(x, dict)}
        for x in runs.get(rid, []):
            if x["payload_sha256"] not in have: r.setdefault("runs", []).append(x); have.add(x["payload_sha256"])
    return M, missing
if __name__ == "__main__":
    raw = open(MF, encoding="utf-8").read(); M, missing = build(json.loads(raw))
    new = json.dumps(M, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        bad = missing + ([] if new == raw else ["CANON/manifest.json is behind its chain results (run python3 CANON/chain_events.py)"])
        print("chain_events: " + ("ok" if not bad else "REFUSED (%d)" % len(bad))); [print("  " + b) for b in bad]; sys.exit(1 if bad else 0)
    if missing: print("chain_events: REFUSED"); [print("  " + b) for b in missing]; sys.exit(1)
    if new != raw: open(MF, "w", encoding="utf-8").write(new)
    print("chain_events: ok (" + ("manifest updated" if new != raw else "nothing new") + ")")
