import os as _os
#!/usr/bin/env python3
"""One command to run after ANY change to the chain, and before ANY push.

The author's problem, stated 2026-09-25: "I feel like we have a LOT of files in the repo with the same info
and that is going to make it more difficult to keep them all updated with each new change or addition."
He is right, and the answer is not discipline - it is that a fact should live in ONE place and every other
place should be generated from it or checked against it.

This script does both halves:

  REGENERATE  everything that is derived - the step order from the code's own AST, the procedure index, the
              reviewer manifest, the SOP's implemented-in lines and per-step detail, the run index, the
              code-name links in every document.

  CHECK       the things that cannot be generated because a human wrote them, against the things that can.
              Every rule below is a claim the repository makes about itself; a rule that fails is a document
              that has drifted from the tree. Exit code 1 if any rule fails, so this can gate a push.

Usage:
    python3 chain/propagate.py            # regenerate, then check
    python3 chain/propagate.py --check    # check only (what a pre-push hook runs)
"""
import argparse
import glob
import json
import os
import glob as _glob
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.dirname(HERE)                      # .../MethylPhys
ROOT = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip() or os.path.dirname(os.path.dirname(MP))

DOORS = os.path.join(MP, "doors")
SOP = os.path.join(MP, "sop", "MethylPhys_CPG_SOP.md")
MANUAL_DATA = os.path.join(MP, "manual", "data003.py")
REGISTER = os.path.join(DOORS, "CHAIN_COMMISSIONING.md")
RUNBOOK = os.path.join(DOORS, "RUNBOOK.md")
MANIFEST = os.path.join(DOORS, "REVIEWER_MANIFEST.md")
SEQ = os.path.join(HERE, "chain_sequence.json")

GENERATORS = [
    ("step order from the code", [sys.executable, os.path.join(HERE, "build_chain_sequence.py")]),
    ("run index", [sys.executable, os.path.join(HERE, "build_run_index.py")]),
    ("SOP paths", [sys.executable, os.path.join(MP, "sop", "sop_repoint.py")]),
    ("SOP implemented-in lines", [sys.executable, os.path.join(MP, "sop", "sop_stage_links.py")]),
    ("SOP per-step detail", [sys.executable, os.path.join(MP, "sop", "sop_step_detail.py")]),
    ("reviewer manifest", [sys.executable, os.path.join(MP, "kit", "build_reviewer_manifest.py")]),
    ("code-name links", [sys.executable, os.path.join(MP, "kit", "add_doc_links.py")]),
]
VERIFIERS = [
    ("every relative reference resolves", [sys.executable, os.path.join(MP, "kit", "link_check.py")]),
]


def read(p):
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def live_chain_modules():
    """Every .py the chain actually runs, from the derived sequence - not from a list someone typed."""
    try:
        seq = json.load(open(SEQ, encoding="utf-8"))
    except (OSError, ValueError):
        return []
    mods = set()
    for step in seq.get("live_path", []) + seq.get("batch_path", []):
        w = step.get("where") or ""
        if w.endswith(".py"):
            mods.add(os.path.basename(w))
    return sorted(mods)


def procedures():
    """Every sealed procedure, from its outcome document."""
    out = {}
    for p in glob.glob(os.path.join(DOORS, "PROC_*_OUTCOME.md")):
        m = re.search(r"PROC[-_]([A-Z0-9]+)[-_](\d+)_OUTCOME", os.path.basename(p))
        if m:
            out["PROC-%s-%s" % (m.group(1), m.group(2))] = p
    return out


def rules():
    """Each rule: (name, ok, detail). A rule that fails names the document that drifted."""
    R = []
    seq_mods = live_chain_modules()
    sop, manual, reg, runbook, manifest = (read(SOP), read(MANUAL_DATA), read(REGISTER), read(RUNBOOK),
                                           read(MANIFEST))

    # 1. every module the chain runs is named in the SOP
    missing = [m for m in seq_mods if m not in sop]
    R.append(("every live chain module is named in the SOP",
              not missing, "missing: %s" % ", ".join(missing) if missing else "%d modules" % len(seq_mods)))

    # 2. every module the chain runs is in the reviewer manifest
    missing = [m for m in seq_mods if m not in manifest]
    R.append(("every live chain module is in the reviewer manifest",
              not missing, "missing: %s" % ", ".join(missing) if missing else "%d modules" % len(seq_mods)))

    # 3. every sealed procedure has a row in the register
    procs = procedures()
    missing = [p for p in procs if p not in reg]
    R.append(("every sealed procedure has a row in CHAIN_COMMISSIONING",
              not missing, "missing: %s" % ", ".join(sorted(missing)) if missing else
              "%d procedures" % len(procs)))

    # 4. every sealed procedure is reachable from the reviewer manifest
    missing = [p for p, path in procs.items() if os.path.basename(path) not in manifest]
    R.append(("every sealed procedure is in the reviewer manifest",
              not missing, "missing: %s" % ", ".join(sorted(missing)) if missing else
              "%d procedures" % len(procs)))

    # 5. a procedure that changed what the chain REPORTS must be in the manual's data module
    reporting = [p for p in procs if p.split("-")[1] in ("SMALL", "STAGE0", "E2E", "MAHA")]
    missing = [p for p in reporting if p not in manual]
    R.append(("procedures that changed a reported number are in the Issue 003 data module",
              not missing, "missing: %s" % ", ".join(sorted(missing)) if missing else
              "%d procedures" % len(reporting)))

    # 6. every runtime file the chain reads is named in the SOP
    rt = sorted({os.path.basename(p) for p in glob.glob(os.path.join(HERE, "Runtime Matrices", "**", "*.json"),
                                                        recursive=True)})
    missing = [f for f in rt if f not in sop and f not in manifest]
    R.append(("every runtime matrix is named in the SOP or the manifest",
              not missing, "missing: %s" % ", ".join(missing) if missing else "%d files" % len(rt)))

    # 7. the run index covers every ledger in the tree
    ledgers = glob.glob(os.path.join(HERE, "**", "evidence_ledger*.jsonl"), recursive=True)
    idx = read(os.path.join(HERE, "example_runs", "RUN_INDEX.csv"))
    n_rows = max(len(idx.strip().split("\n")) - 1, 0)
    n_runs = sum(1 for p in ledgers for line in open(p, encoding="utf-8") if line.strip())
    R.append(("the run index covers every ledger row",
              n_rows == n_runs, "%d rows for %d ledger lines" % (n_rows, n_runs)))

    # 8. the retired v1 conductor: a live DOCUMENT OR SCRIPT may name it only in a sentence that says it
    # is retired. Historical provenance records (the disease-card JSONs) are exempt: they state what
    # actually ran at the time, and editing them would be falsifying a record. Added 2026-09-25 - after
    # the retirement, every unannotated mention is by definition stale information.
    import subprocess as _sp
    bad_lines = []
    for ln in _sp.run(["grep", "-rn", "walther_clinical", MP], capture_output=True,
                      text=True).stdout.split("\n"):
        if not ln or ":" not in ln:
            continue
        fn = ln.split(":")[0]
        if any(k in fn for k in ("__pycache__", ".zip", ".html", "RETIRED", "disease_matching.py",
                                 "Disease Cards", "chain_sequence.json", "chain_inventory_v1.json",
                                 "propagate.py", "propagate_status.json")):   # this gate itself names the module it checks for
            continue
        if not re.search(r"retired|RETIRED", ln):
            bad_lines.append(fn.replace(MP + "/", ""))
    R.append(("every live mention of the retired v1 conductor says so",
              not bad_lines, ("unannotated in: %s" % ", ".join(sorted(set(bad_lines))[:4]))
              if bad_lines else "checked, exempting historical provenance records"))

    # 8. nothing claims a dependency the environment does not state
    req = read(os.path.join(HERE, "requirements.txt"))
    R.append(("matplotlib is a stated dependency (the plate needs it)",
              "matplotlib" in req, "requirements.txt" if "matplotlib" in req else "ABSENT"))
    # 10. a procedure must invoke the chain, not reimplement it (author's ruling 2026-09-26).
    #     Reaching into stage functions or the deconvolver skips every guard the chain owns - the presence
    #     floor, the composition guard, the intake log, the ceiling check - and each of those exists because
    #     something went wrong once. Four defects in one day came from exactly this: an epithelial fraction
    #     summed over the wrong classes, pooled class fractions consumed without noticing there were two
    #     solves, a cohort scored with no report rendered, and an age truncated where the chain rounds.
    #     The scripts below PREDATE the ruling and are the record of what was actually run, so rewriting them
    #     would falsify that record - the same reason a sealed pre-registration is never edited. They are
    #     grandfathered by name. THE LIST MAY NOT GROW: any new procedure script must go through run_sample.
    GRANDFATHERED = frozenset({
        "PROC_BAND_01_measure.py", "PROC_BAND_01_analyse.py", "PROC_CLS_01_measure.py",
        "PROC_CLS_01_analyse.py", "PROC_CLS_01_b6.py", "PROC_LABBAND_01.py", "PROC_FOREIGN_01.py",
        "PROC_FOREIGN_01_analyse.py", "PROC_EPIC_01_score.py", "PROC_EPIC_01_analyse.py",
        "PROC_PARTIAL_01.py", "PROC_PARTIAL_01_analyse.py", "PROC_TISSUE_01_score.py",
        "PROC_TISSUE_01_analyse.py", "PROC_COV_01.py",
        # found by this rule on its first run - they predate the ruling for the same reason as the rest
        "PROC_DECON_01.py", "PROC_PLASMA_MIX_01.py", "PROC_SEP_03.py", "PROC_SMALL_01_prepare.py",
    })
    INTERNALS = ("stage_a_cells", "stage_b_identity", "stage_1s_scale_map", "run_full",
                 "WaltherIAMDeconvolver")
    bypass = []
    for fn in sorted(_glob.glob(_os.path.join(MP, "kit", "PROC_*.py"))):
        base = _os.path.basename(fn)
        if base in GRANDFATHERED:
            continue
        txt = read(fn)
        if any(k in txt for k in INTERNALS) and "run_sample" not in txt:
            bypass.append(base)
    R.append(("every new procedure script invokes the chain rather than its internals",
              not bypass,
              "bypassing: %s" % ", ".join(bypass) if bypass
              else "%d grandfathered, the rest clean" % len(GRANDFATHERED)))

    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check only; do not regenerate")
    a = ap.parse_args()

    if not a.check:
        print("regenerating what is derived")
        for name, cmd in GENERATORS:
            if not os.path.exists(cmd[1]):
                print("  SKIP %-34s (%s not present)" % (name, os.path.basename(cmd[1])))
                continue
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(cmd[1]))
            tail = [l for l in (r.stdout or "").strip().split("\n") if l.strip()]
            print("  %-34s %s" % (name, tail[-1][:90] if tail else "(no output)"))

    print("\nverifying")
    bad = 0
    for name, cmd in VERIFIERS:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(cmd[1]))
        ok = r.returncode == 0
        bad += 0 if ok else 1
        print("  %-4s %-40s %s" % ("PASS" if ok else "FAIL", name,
                                   (r.stdout or "").strip().split("\n")[-1][:70]))

    print("\nchecking that no document has drifted from the tree")
    for name, ok, detail in rules():
        bad += 0 if ok else 1
        print("  %-4s %-58s %s" % ("PASS" if ok else "FAIL", name, detail[:80]))

    # The report prints this, so a reader of a reading can see whether the documents were current when it
    # was produced (author, 2026-09-25: the record of how we got a result matters as much as the result).
    import datetime
    commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    json.dump({"when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "commit": commit, "pass": bad == 0,
               "rules": [{"rule": n, "ok": bool(o), "detail": d} for n, o, d in rules()]},
              open(os.path.join(HERE, "propagate_status.json"), "w"), indent=1)
    print("\n%s" % ("propagate: PASS - every derived document is current and no rule failed" if not bad else
                    "propagate: %d FAILURE(S) - fix these before pushing" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
