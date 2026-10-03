#!/usr/bin/env python3
"""build_all.py - THE ONE COMMAND for chain v3. Everything that reports the chain is regenerated from the chain, in order, then gated.

Author, 2026-09-27: "once the official chain is updated or amended it automatically updates every place else that reports
the chain itself". The source of truth is the chain folder: run_sample.py, conductor_v3.py and the stage modules, the frozen inputs
under Runtime Matrices, and TOOLKIT.md. Each generated document carries a header naming its generator. guarded_push.sh runs this
script and then chain/release_check_v3.py before every push. Chain v3 is the only engine: the class-floor engine (v2), its SOP,
manual, inventory and report builders were retired on 2026-10-03 and are archived privately.

ORDER (each step's output is the next step's input):
   1 chain/chain_sequence.json + doors/CHAIN_SEQUENCE.md   chain/build_chain_sequence.py   the v3 live path, from the code
   2 manual/MethylPhys_CPG_Operations_Manual.pdf           manual/build_manual_v3.py       OM chapter + chain sequence + toolkit
   3 doors/REPO_INVENTORY.md                               kit/build_repo_inventory.py     measured from git ls-files
   4 doors/RUNBOOK.md (marked block only)                  kit/build_marked_blocks.py      the live path between <!-- GENERATED --> markers
   5 folder READMEs                                        kit/build_folder_readmes.py
   6 chain/GENERATED_MANIFEST.json                         (here)                          sha256 of every generated file and every chain input
GATES (any failure = exit 1; guarded_push refuses): kit/link_check.py; chain/build_chain_sequence.py --check.

Usage:  python3 build_all.py            build everything, then gate
        python3 build_all.py --check    build nothing; fail if any generated file differs from what the chain would produce now
"""
import os, sys, json, hashlib, subprocess, time, glob

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(MP))
PY = sys.executable
CHECK = "--check" in sys.argv

GENERATED = [  # (path relative to MP, generator) - the register of every document that reports the chain
    ("chain/chain_sequence.json", "chain/build_chain_sequence.py"),
    ("doors/CHAIN_SEQUENCE.md", "chain/build_chain_sequence.py"),
    ("manual/MethylPhys_CPG_Operations_Manual.pdf", "manual/build_manual_v3.py"),
    ("doors/REPO_INVENTORY.md", "kit/build_repo_inventory.py"),
    ("doors/RUNBOOK.md (marked block)", "kit/build_marked_blocks.py"),
    ("chain/GENERATED_MANIFEST.json", "chain/build_all.py"),
]
INPUT_GLOBS = ["chain/*.py", "chain/TOOLKIT.md", "chain/MethylPhys_Interface/*.py", "chain/Runtime Matrices/**/*.json",
               "chain/Runtime Matrices/**/*.csv", "chain/Runtime Matrices/**/*.py", "manual/*.md", "manual/*.py", "kit/*.py"]


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
            elif sha(p) != h and not rel.endswith(".pdf"): bad.append(f"generated file edited by hand or stale: {rel}")
        print("BUILD CHECK:", "PASS" if not bad else "FAIL\n  " + "\n  ".join(bad)); sys.exit(1 if bad else 0)

    K, MAN = os.path.join(MP, "kit"), os.path.join(MP, "manual")
    run("1 chain sequence (from the code)", [PY, "build_chain_sequence.py"], HERE)
    run("2 operations manual (PDF)", [PY, "build_manual_v3.py"], MAN)
    run("3 repo inventory (measured)", [PY, "build_repo_inventory.py"], K)
    run("4 RUNBOOK marked block", [PY, "build_marked_blocks.py"], K)
    run("5 folder READMEs", [PY, "build_folder_readmes.py"], K, must=False)
    print("gates:")
    g1 = run("link check", [PY, "link_check.py"], K, must=False)
    g2 = run("chain sequence matches the code", [PY, "build_chain_sequence.py", "--check"], HERE, must=False)
    outputs = {}
    for rel, gen in GENERATED:
        p = os.path.join(MP, rel.split(" (")[0])
        if os.path.exists(p) and not rel.endswith("GENERATED_MANIFEST.json"): outputs[rel.split(" (")[0]] = sha(p)
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": "v3", "generated": [{"file": a, "generator": b} for a, b in GENERATED],
               "inputs": in_hash, "outputs": outputs, "gates": {"link_check": g1, "chain_sequence": g2}}, open(man_path, "w"), indent=1)
    print(f"manifest: {len(in_hash)} chain inputs, {len(outputs)} generated outputs")
    ok = g1 and g2
    print("BUILD_ALL:", "PASS" if ok else "FAIL - a gate failed; fix the source it names (never the generated document)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__": main()
