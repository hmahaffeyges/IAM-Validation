#!/usr/bin/env python3
"""Is this file necessary? Answer it from the tree, for one name or for everything.

Author's standing instruction, 2026-09-25, given when the v1 conductor (walther_clinical.py, retired) was retired: "Anywhere that used that
name, it shouldnt just be noted that the file was retired, it should be evaluated whether or not that file is
even necessary. We really dont need that many files ... it leaves the potential for data to be updated in one
file and not in another and then we end up not know what is accurate anymore."

So this is not a one-off script. Run it after any retirement, and before adding a file.

    python3 evaluate_necessity.py                    every live file under MethylPhys
    python3 evaluate_necessity.py walther_clinical   only files naming that string (here: the retired
                                                     v1 conductor, whose one live function is now
                                                     chain/disease_matching.py)
    python3 evaluate_necessity.py --orphans          only the files nothing points at

WHAT IT MEASURES, and why each test is there.
  runs          the file is in the live path of chain/chain_sequence.json, which is derived from the code by
                AST - not from prose. A file that runs on every sample is necessary by definition.
  imported      another live module imports it. Counted by pattern, so a dynamic _load_module() counts too.
  in code       a document or data file NAMED inside live code (the report's Files tab names documents this
                way). My first pass on 2026-09-25 missed this and called README_CPG_Plates.md an orphan when
                the report itself names it - a checker that only reads documents cannot see a document's real
                readers.
  in docs       a live document links or names it.
  generator     it regenerates something, and propagate.py runs it.
It prints a verdict per file. ORPHAN means nothing in the tree points at it - which is a question for the
author, not a licence to delete: the pre-atlas record is full of files nothing links to and they are record.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.dirname(HERE)
ROOT = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip()
REL = os.path.relpath(MP, ROOT)

GENERATORS = {"propagate.py", "build_chain_sequence.py", "build_reviewer_manifest.py",
              "build_chain_inventory.py", "build_run_index.py", "build_report_tab_reference.py",
              "sop_repoint.py", "sop_stage_links.py", "sop_step_detail.py", "link_check.py",
              "add_doc_links.py", "evaluate_necessity.py", "guarded_push.sh"}
# Files whose whole purpose is to be a record of something that happened. Nothing links to a sealed outcome's
# raw evidence, and nothing should have to.
RECORD_DIRS = ("evidence_pre_atlas", "hmin_calibration", "kit/results", "example_runs", "Record/",
               "author_copies", "reference_data",
               # The commissioning arrays are named by GSM accession in prose and on the
               # command line, never by filename, so a name scan cannot see their readers.
               "TEST_DATA")


def main(argv):
    only = None
    orphans_only = False
    for a in argv[1:]:
        if a == "--orphans":
            orphans_only = True
        else:
            only = a
    live = [f for f in subprocess.run(["git", "-C", ROOT, "ls-files"],
                                      capture_output=True, text=True).stdout.split("\n")
            if f and "RETIRED" not in f and "__pycache__" not in f]
    mine = [f for f in live if f.startswith(REL + "/")]
    text = {}
    for f in live:
        if f.endswith((".md", ".py", ".sh", ".json", ".txt", ".csv")):
            try:
                text[f] = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
            except OSError:
                pass
    seq_p = os.path.join(MP, "chain", "chain_sequence.json")
    runs = set()
    if os.path.exists(seq_p):
        seq = json.load(open(seq_p, encoding="utf-8"))
        for e in seq.get("live_path") or []:
            w = e.get("where") or ""
            if w:
                runs.add(os.path.basename(w))

    if only:
        targets = [f for f in mine if only in (text.get(f) or "") or only in f]
        print("%d file(s) name or match %r\n" % (len(targets), only))
    else:
        targets = mine

    counts = {"KEEP": 0, "ORPHAN": 0, "RECORD": 0}
    out = []
    for f in sorted(targets):
        base = os.path.basename(f)
        stem = base[:-3] if base.endswith(".py") else base
        in_docs = [k for k, s in text.items()
                   if k != f and k.endswith(".md") and base in s]
        in_code = [k for k, s in text.items()
                   if k != f and k.endswith((".py", ".sh")) and base in s]
        imported = [k for k, s in text.items()
                    if k != f and k.endswith(".py")
                    and re.search(r"import\s+%s\b|from\s+%s\s+import|_load_module\(\s*[\"']%s"
                                  % (re.escape(stem), re.escape(stem), re.escape(stem)), s)]
        why = []
        if base in runs:
            why.append("runs on every sample")
        if imported:
            why.append("imported by %d module(s)" % len(imported))
        if base in GENERATORS:
            why.append("generator run by the gate")
        if in_code and not imported:
            why.append("named in %d live code file(s)" % len(in_code))
        if in_docs:
            why.append("named in %d document(s)" % len(in_docs))
        if why:
            verdict, counts["KEEP"] = "KEEP", counts["KEEP"] + 1
        elif any(d in f for d in RECORD_DIRS):
            verdict, counts["RECORD"] = "RECORD", counts["RECORD"] + 1
            why = ["record - nothing links to a sealed outcome's evidence, and nothing should have to"]
        else:
            verdict, counts["ORPHAN"] = "ORPHAN", counts["ORPHAN"] + 1
            why = ["nothing in the tree points at it - a question for the author, not a licence to delete"]
        out.append((f[len(REL) + 1:], verdict, "; ".join(why)))

    show = [r for r in out if r[1] == "ORPHAN"] if orphans_only else out
    for f, v, why in show:
        print("  %-8s %-58s %s" % (v, f[:58], why))
    print("\n%d file(s): %d KEEP, %d RECORD, %d ORPHAN"
          % (len(out), counts["KEEP"], counts["RECORD"], counts["ORPHAN"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
