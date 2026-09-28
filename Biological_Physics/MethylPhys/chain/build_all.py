#!/usr/bin/env python3
"""build_all.py - THE ONE COMMAND. Everything that reports the chain is regenerated from the chain, in order, then gated.

Author, 2026-09-27: "once the official chain is updated or amended it automatically updates every place else that reports
the chain itself, like the SOP, OM, Report, REPO_INVENTORY.md, REVIEWER_MANIFEST.md, RUNBOOK.md ... I think we already have
too many files to keep updated as there is."

The source of truth is the chain folder: cpg_conductor.py and the stage modules, run_sample.py, the Runtime Matrices, the
kit tests. Nothing below is edited by hand; each is written by a generator that reads the chain, and each carries a header
naming its generator. guarded_push.sh runs this script before propagate.py on every push, so a hand edit to a generated
document is overwritten before it can be committed, and a chain change is reflected everywhere before the push lands.

ORDER (each step's output is the next step's input):
   1 chain/chain_sequence.json + doors/CHAIN_SEQUENCE.md      build_chain_sequence.py   what run_full calls, in order, from the code
   2 chain_inventory_v1.json + doors/COMPONENT_MAP.md   build_chain_inventory.py  every file under chain/, its role, described
   3 Runtime Matrices/Cell_Descriptions                  build_cell_descriptions   biology per cell from the author's drafts (skipped if no drafts)
   4 doors/RUN_INDEX.csv                                build_run_index.py        every filed run
   5 sop/MethylPhys_CPG_SOP.md                          build_sop_mirror.py       STATUS banner per section + Part II-A from step 1; then sop_repoint.py (references, file table)
   6 manual/MethylPhys_CPG_Operations_Manual.pdf        build_om.sh               two-pass, under THIS interpreter (reportlab lives in the chain environment)
   7 doors/REVIEWER_MANIFEST.md                         build_reviewer_manifest.py
   8 doors/REPORT_TAB_REFERENCE.md                      build_report_tab_reference.py   from the last filed report
   9 doors/REPO_INVENTORY.md                            build_repo_inventory.py   measured from git ls-files + the generator table below
  10 doors/RUNBOOK.md, README.md (marked blocks only)   build_marked_blocks.py    the chain-as-it-runs table between <!-- GENERATED --> markers
  11 folder READMEs                                     build_folder_readmes.py
  12 GENERATED_MANIFEST.json                            (here)                    sha256 of every generated file and of every chain input
GATES (any failure = exit 1; guarded_push refuses):
  vocab_scan.py (the report's own banned-word and population-vocabulary guards over SOP + OM), class_guard.py (a class is only the floor a cell is divided by), build_sop_mirror.py --check,
  link_check.py, propagate.py (run by guarded_push after this script).

Usage:  python3 build_all.py            build everything, then gate
        python3 build_all.py --check    build nothing; fail if any generated file differs from what the chain would produce now
"""
import os, sys, json, hashlib, subprocess, shutil, tempfile, time, glob

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(MP))
PY = sys.executable
CHECK = "--check" in sys.argv

GENERATED = [  # (path relative to MP, generator) - the register of every document that reports the chain
    ("chain/chain_sequence.json", "chain/build_chain_sequence.py"),
    ("doors/CHAIN_SEQUENCE.md", "chain/build_chain_sequence.py"),
    ("chain/Runtime Matrices/chain_inventory_v1.json", "chain/MethylPhys_Interface/build_chain_inventory.py"),
    ("doors/COMPONENT_MAP.md", "chain/MethylPhys_Interface/build_chain_inventory.py"),
    ("chain/Runtime Matrices/Cell_Descriptions/cell_descriptions_v1.json", "kit/build_cell_descriptions.py"),
    ("chain/example_runs/RUN_INDEX.csv", "chain/build_run_index.py"),
    ("sop/MethylPhys_CPG_SOP.md", "sop/build_sop_mirror.py + sop/sop_repoint.py"),
    ("manual/MethylPhys_CPG_Operations_Manual.pdf", "manual/build_om.sh (build_operations_manual.py + om_data.py)"),
    ("doors/REVIEWER_MANIFEST.md", "kit/build_reviewer_manifest.py"),
    ("doors/REPORT_TAB_REFERENCE.md", "kit/build_report_tab_reference.py"),
    ("doors/REPO_INVENTORY.md", "kit/build_repo_inventory.py"),
    ("doors/RUNBOOK.md (marked block)", "kit/build_marked_blocks.py"),
    ("README.md (marked block)", "kit/build_marked_blocks.py"),
    ("chain/GENERATED_MANIFEST.json", "chain/build_all.py"),
]
INPUT_GLOBS = ["chain/*.py", "chain/MethylPhys_Interface/*.py", "chain/Runtime Matrices/**/*.json", "chain/Runtime Matrices/**/*.py",
               "kit/*.py", "sop/sop_repoint.py", "sop/build_sop_mirror.py", "manual/*.py", "doors/*_PREREG.md", "doors/*_OUTCOME.md",
               "doors/CHAIN_COMMISSIONING.md", "doors/PLAN.md", "doors/ENHANCEMENTS.md"]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def run(label, cmd, cwd, env=None, must=True, tail=2):
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    lines = [l for l in (r.stdout + r.stderr).strip().split("\n") if l and "Deprecat" not in l and not l.startswith("INFO")]
    ok = r.returncode == 0
    print(f"  {'ok ' if ok else 'FAIL'} {label:<44} {time.time()-t:5.1f}s  {' | '.join(x[:90] for x in lines[-tail:])}")
    if not ok and must:
        print("\n".join(lines[-15:])); print(f"\nbuild_all: {label} failed - nothing further is built"); sys.exit(1)
    return ok

def main():
    print(f"build_all {'--check' if CHECK else ''} @ {subprocess.run(['git','rev-parse','--short','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()}  python {PY}")
    inputs = sorted({p for g in INPUT_GLOBS for p in glob.glob(os.path.join(MP, g), recursive=True) if os.path.isfile(p)})
    in_hash = {os.path.relpath(p, MP): sha(p) for p in inputs}
    man_path = os.path.join(HERE, "GENERATED_MANIFEST.json")
    if CHECK:
        if not os.path.exists(man_path): print("GENERATED_MANIFEST.json missing - run build_all.py"); sys.exit(1)
        man = json.load(open(man_path)); bad = []
        if man["inputs"] != in_hash:
            ch = sorted(set(k for k in set(man["inputs"]) | set(in_hash) if man["inputs"].get(k) != in_hash.get(k)))
            bad.append(f"chain inputs changed since the last build: {ch[:8]}{' ...' if len(ch) > 8 else ''}")
        for rel, h in man["outputs"].items():
            p = os.path.join(MP, rel)
            if not os.path.exists(p): bad.append(f"generated file missing: {rel}")
            elif sha(p) != h: bad.append(f"generated file edited by hand or stale: {rel}")
        print("BUILD CHECK:", "PASS" if not bad else "FAIL\n  " + "\n  ".join(bad)); sys.exit(1 if bad else 0)

    K, CH, MI, SOP, MAN, DOORS = (os.path.join(MP, "kit"), HERE, os.path.join(HERE, "MethylPhys_Interface"), os.path.join(MP, "sop"), os.path.join(MP, "manual"), os.path.join(MP, "doors"))
    run("1 chain sequence (from the code)", [PY, "build_chain_sequence.py"], CH)
    run("2 chain inventory", [PY, "build_chain_inventory.py"], MI)
    drafts = os.environ.get("IAM_WEBDRAFTS") or next((d for d in (os.path.join(ROOT, "..", "webdrafts"), os.path.join(os.getcwd(), "webdrafts")) if os.path.isdir(d)), None)
    if drafts and os.path.exists(os.path.join(K, "build_cell_descriptions.py")):
        run("3 cell descriptions (author's drafts)", [PY, "build_cell_descriptions.py", drafts], K)
    else:
        print("  --  3 cell descriptions                       skipped (no drafts folder; the last built file stands)")
    run("4 run index", [PY, "build_run_index.py"], CH, must=False)
    run("5a SOP mirror (banners, Part II-A)", [PY, "build_sop_mirror.py"], SOP)
    run("5b SOP repoint (references, file table)", [PY, "sop_repoint.py"], SOP)
    run("6 Operations Manual (two-pass PDF)", ["sh", "build_om.sh", "MethylPhys_CPG_Operations_Manual.pdf"], MAN, env={"PYTHON": PY})
    run("7 reviewer manifest", [PY, "build_reviewer_manifest.py"], K)
    # 8a. a FRESH report from repo-shipped data (one calibrated GSE87571 array from reference_data), so the tab reference
    #     describes the interface as it is now - never the last filed report, which is the previous commit's interface.
    ref_dir = os.path.join(K, "results", "reference_report"); os.makedirs(ref_dir, exist_ok=True)
    csv = os.path.join(ref_dir, "reference_array.csv")
    if not os.path.exists(csv):
        import lzma, pickle
        src = sorted(glob.glob(os.path.join(MP, "reference_data", "stage1_betas_GSE87571*.pkl.xz")))
        if src:
            df = pickle.load(lzma.open(src[0], "rb")); df.iloc[:, [0]].dropna().to_csv(csv)
    rep = None
    if os.path.exists(csv):
        ok8 = run("8a fresh reference report (repo data)", [PY, os.path.join(MI, "run_sample.py"), "--betas", csv, "--age", "60", "--sex", "F", "--specimen", "whole blood",
                  "--pipeline", "stage1_noob_450K", "--no-intake", "--lab", "GSE87571", "--id", "REFERENCE", "--out", os.path.join(ref_dir, "reference.html")], K, must=False, tail=1)
        if ok8: rep = os.path.join(ref_dir, "reference.html")
    if rep: run("8b report tab reference (fresh render)", [PY, "build_report_tab_reference.py", rep], K, must=False)
    else:
        print("  --  8b report tab reference: NOT rebuilt - no fresh render (reference_data absent or render failed); the previous file stands and is STALE")
    run("9 repo inventory (measured)", [PY, "build_repo_inventory.py"], K)
    run("10 RUNBOOK / README marked blocks", [PY, "build_marked_blocks.py"], K)
    run("11 folder READMEs", [PY, "build_folder_readmes.py"], K, must=False)
    # gates
    print("gates:")
    g1 = run("vocab scan (report guards over SOP + OM)", [PY, "vocab_scan.py"], K, must=False)
    g2 = run("SOP mirror reconciliation", [PY, "build_sop_mirror.py", "--check"], SOP, must=False)
    g3 = run("link check", [PY, "link_check.py"], K, must=False)
    g4 = run("class guard (a class is only the floor a cell is divided by)", [PY, "class_guard.py"], K, must=False)
    # manifest
    outputs = {}
    for rel, gen in GENERATED:
        p = os.path.join(MP, rel.split(" (")[0])
        if os.path.exists(p) and not rel.endswith("GENERATED_MANIFEST.json"): outputs[rel.split(" (")[0]] = sha(p)
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "python": PY, "generated": [{"file": a, "generator": b} for a, b in GENERATED],
               "inputs": in_hash, "outputs": outputs, "gates": {"vocab_scan": g1, "sop_mirror": g2, "link_check": g3, "class_guard": g4}}, open(man_path, "w"), indent=1)
    print(f"manifest: {len(in_hash)} chain inputs, {len(outputs)} generated outputs")
    ok = g1 and g2 and g3 and g4
    print("BUILD_ALL:", "PASS" if ok else "FAIL - a gate failed; fix the source it names (never the generated document)")
    sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
