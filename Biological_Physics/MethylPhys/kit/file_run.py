#!/usr/bin/env python3
"""File one run into the repository - the loop's return path (author, 2026-09-26: "the report interface should declare
what should be added to the repo ... once the test has been run and files are updated with the new info, it should be
listed to go to the repo in the correct file").

    python3 file_run.py --report path/to/REPORT.html [--run-id RUN-YYYYMMDD-NN] [--procedure PROC-XXX-NN] [--commit]

What goes where - the same table the report's Run tab prints for itself:
  chain/example_runs/<run_id>/<specimen>.html          the report            (RUN.md written beside it)
  chain/example_runs/<run_id>/<specimen>_bundle.json   the bundle
  chain/example_runs/evidence_ledger_example.jsonl     this run's ledger row appended (one row per run, never rewritten)
  chain/example_runs/RUN_INDEX.csv                     rebuilt by kit/build_run_index.py from the folders
  kit/results/<PROCEDURE>*.json                        when --procedure is given: the run is evidence for a sealed procedure
                                                       and the bundle is copied there under the procedure's name
Then: propagate.py must pass, and --commit runs guarded_push.sh with a message that names the run. Without --commit it
prints the git status and stops - filing is reversible until it is pushed.
"""
import argparse, csv, datetime, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); CH = os.path.join(MP, "chain")
EX = os.path.join(CH, "example_runs"); R = os.path.dirname(os.path.dirname(MP))


def plan(report, run_id=None, procedure=None, sid=None):
    """The filing plan - pure, so the report can print it without filing anything."""
    base = os.path.splitext(report)[0]; bundle = base + "_bundle.json"
    b = json.load(open(bundle)) if os.path.exists(bundle) else {}
    rid = run_id or b.get("run_id") or ("RUN-%s-XX" % datetime.date.today().strftime("%Y%m%d"))
    sid = sid or (b.get("context") or {}).get("sample_id") or os.path.basename(base)
    led = os.path.join(os.path.dirname(os.path.abspath(report)) or ".", "evidence_ledger.jsonl")
    rows = [(report, f"chain/example_runs/{rid}/{sid}.html", "the report"),
            (bundle, f"chain/example_runs/{rid}/{sid}_bundle.json", "the machine-readable bundle - every stage's output")]
    if os.path.exists(led): rows.append((led, "chain/example_runs/evidence_ledger_example.jsonl", "this run's ledger row, appended"))
    for extra, why in ((base + "_intake.jsonl", "intake and integrity records"), (os.path.join(os.path.dirname(report), "manifests"), "the immutable per-sample manifest")):
        if os.path.exists(extra): rows.append((extra, f"chain/example_runs/{rid}/{os.path.basename(extra)}", why))
    if procedure:
        rows.append((bundle, f"kit/results/{procedure.replace('-', '_')}_{sid}_bundle.json", f"evidence for {procedure} - the procedure's outcome document must link it"))
    out = []
    for src, dst, why in rows:
        full = os.path.join(MP, dst); tracked = subprocess.run(["git", "-C", R, "ls-files", "--error-unmatch", os.path.relpath(full, R)], capture_output=True).returncode == 0
        out.append({"source": src, "destination": "Biological_Physics/MethylPhys/" + dst, "why": why,
                    "state": "committed" if tracked else ("on disk, not committed" if os.path.exists(full) else "not filed")})
    return rid, sid, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", required=True); ap.add_argument("--run-id"); ap.add_argument("--procedure"); ap.add_argument("--commit", action="store_true")
    a = ap.parse_args(); rid, sid, rows = plan(a.report, a.run_id, a.procedure)
    print(f"run {rid} - specimen {sid}")
    for r in rows: print(f"  {r['state']:<24} {r['destination']:<70} {r['why']}")
    for r in rows:
        dst = os.path.join(R, r["destination"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if r["destination"].endswith("evidence_ledger_example.jsonl"):
            last = open(r["source"]).read().strip().split("\n")[-1]
            if not (os.path.exists(dst) and last in open(dst).read()):
                open(dst, "a").write(last + "\n")
        elif os.path.isdir(r["source"]): shutil.copytree(r["source"], dst, dirs_exist_ok=True)
        else: shutil.copy2(r["source"], dst)
    b = json.load(open(os.path.splitext(a.report)[0] + "_bundle.json")) if os.path.exists(os.path.splitext(a.report)[0] + "_bundle.json") else {}
    open(os.path.join(EX, rid, "RUN.md"), "w").write(
        f"# {rid} - {sid}\n\nOne execution of the commissioned chain on one specimen, filed by kit/file_run.py on {datetime.date.today()}. "
        f"**Not a test**: it makes no claim and passes no bar. Chain commit {(b.get('versions') or {}).get('chain_commit', '?')}. "
        f"See [`../README.md`](../README.md) for what each kind of report is.\n")
    subprocess.run([sys.executable, os.path.join(R, "Biological_Physics/MethylPhys/chain/build_run_index.py")], check=False)   # the run index is rebuilt from the folders
    r = subprocess.run([sys.executable, os.path.join(CH, "propagate.py")], capture_output=True, text=True)
    print("propagate:", "PASS" if r.returncode == 0 else "FAILED - fix before pushing"); 
    if r.returncode: print("\n".join(l for l in r.stdout.split("\n") if "FAIL" in l)[:1500]); sys.exit(1)
    if a.commit:
        subprocess.run(["git", "-C", R, "add", "-A", "Biological_Physics"], check=True)
        subprocess.run(["sh", os.path.join(CH, "guarded_push.sh"), f"File run {rid} ({sid}) into example_runs" + (f"; evidence for {a.procedure}" if a.procedure else "")], cwd=R)
    else:
        print("filed on disk, not committed - rerun with --commit, or `git status` to review")


if __name__ == "__main__":
    main()
