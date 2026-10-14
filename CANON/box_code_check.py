"""Box jobs' code in S3 must match the repo (author, 2026-10-13). Registry: CANON/box_jobs.json, one entry per box job: its S3 code prefix, the
repo commit its code was taken from, and each code file's repo path. Run by status_check (--check, no network):
  1. every registered job has an entry in CANON/box_code_verified.json whose checksums equal the repo files at the job's commit;
  2. every *_box.py in the repo either calls publish_code(, or is a registered job marked backfilled (code copied to S3 after the run),
     or is declared in not_jobs with its reason (a launcher that runs no work on the box);
  3. every registered repo file exists at its commit;
  4. every job has a log, machine, purpose, record and state; a finished job's log and outputs are verified in S3 after it finished;
  5. every job has a row in RUNS.md and is named (by its first 8 characters) in the development log.
--verify (needs AWS): reads each job's code_index.json from S3, compares with the repo at the commit, writes CANON/box_code_verified.json.
"""
import os, sys, json, glob, hashlib, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BUCKET = "methylphys-data-945451304272-us-west-2-an"
REG = os.path.join(ROOT, "CANON", "box_jobs.json"); VER = os.path.join(ROOT, "CANON", "box_code_verified.json")
def at_commit(commit, path):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, cwd=ROOT)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
def expected(job): return {os.path.basename(p): at_commit(job["commit"], p) for p in job["files"]}
def verify():
    import boto3; s3 = boto3.client("s3"); out = {}
    for j in json.load(open(REG))["jobs"]:
        idx = json.loads(s3.get_object(Bucket=BUCKET, Key=j["s3_code_prefix"].rstrip("/") + "/code_index.json")["Body"].read())
        got = idx.get("files", {k: v for k, v in idx.items() if not k.startswith("_")})
        def _has(k):
            try: s3.head_object(Bucket=BUCKET, Key=k); return True
            except Exception: return False
        out[j["job_id"]] = {"s3": got, "matches_repo": got == expected(j), "log_in_s3": _has(j["log"]) if j.get("log") else None,
                            "outputs_in_s3": {k: _has(k) for k in j.get("outputs", [])}, "state_at_verify": j.get("state")}
    json.dump(out, open(VER, "w"), indent=1); open(VER, "a").write("\n"); return out
def problems():
    P = []; J = json.load(open(REG))["jobs"]; V = json.load(open(VER)) if os.path.exists(VER) else {}
    for j in J:
        e = expected(j)
        for k, h in e.items():
            if h is None: P.append(f"{j['job_id']}: {k} is not in the repo at {j['commit']}")
        v = V.get(j["job_id"])
        if not v: P.append(f"{j['job_id']}: its S3 code copy has not been verified (python3 CANON/box_code_check.py --verify)")
        elif v["s3"] != e: P.append(f"{j['job_id']}: the code in S3 differs from the repo at {j['commit']}")
        sid = j["job_id"][:8]
        for f in ("log", "machine", "purpose", "record", "state"):
            if not j.get(f): P.append(f"{sid}: box job has no {f!r} in CANON/box_jobs.json")
        if j.get("state") == "done":
            if not v or v.get("state_at_verify") != "done": P.append(f"{sid}: finished, but not verified since it finished (python3 CANON/box_code_check.py --verify)")
            else:
                if not v.get("log_in_s3"): P.append(f"{sid}: its log is not in S3 ({j.get('log')})")
                miss = [k for k, ok in v.get("outputs_in_s3", {}).items() if not ok]
                if miss or not j.get("outputs"): P.append(f"{sid}: outputs missing from S3: {miss or 'none listed'}")
        if sid not in open(os.path.join(ROOT, "Biological_Physics", "MethylPhys", "RUNS.md"), encoding="utf-8").read():
            P.append(f"{sid}: no row in RUNS.md (python3 Biological_Physics/MethylPhys/boxruns/make_runs_ledger.py)")
        if sid not in open(os.path.join(ROOT, "development", "METHYLPHYS_DEVELOPMENT_LOG.md"), encoding="utf-8").read():
            P.append(f"{sid}: not named in the development log")
    back = {os.path.normpath(p) for j in J if j.get("backfilled") for p in j["files"]}
    back |= {os.path.normpath(p) for p in json.load(open(REG)).get("not_jobs", {})}
    for f in glob.glob(os.path.join(ROOT, "**", "*_box.py"), recursive=True):
        r = os.path.relpath(f, ROOT)
        if "publish_code(" not in open(f, encoding="utf-8").read() and os.path.normpath(r) not in back:
            P.append(f"{r}: a box job must call publish_code() before any work, so its code reaches S3")
    return P
if __name__ == "__main__":
    if "--verify" in sys.argv: print(json.dumps(verify(), indent=1)[:800])
    P = problems(); print("box_code_check: " + ("ok" if not P else "REFUSED:\n  " + "\n  ".join(P))); sys.exit(1 if P else 0)
