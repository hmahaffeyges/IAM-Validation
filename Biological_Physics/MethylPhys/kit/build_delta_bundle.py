#!/usr/bin/env python3
"""Build the delta between the author's offline copy and now.  Usage: build_delta_bundle.py <base-commit>

The author keeps an offline copy of the chain in case anything happens to the repository, and asks for "the
new files" when it falls behind. This writes a bundle in the SAME LAYOUT as that copy - what is
MethylPhys/chain/ here is chain/ there, with Record/ beside it - so the files can be dropped straight in.

Three things it must get right, each learned from getting it wrong:
  * THE ATLAS IS EXCLUDED. He adds the 101 MB archive and the class archives himself.
  * DELETIONS AND RENAMES ARE STATED FIRST. A file deleted here but surviving in his copy is exactly the
    drift we spend our time undoing - the first delta of 2026-09-23 shipped a file that had been deleted and
    the next one had to tell him to remove it.
  * THE BASE COMMIT MUST EXIST. The working clone is shallow, so a base older than its depth silently
    produces an EMPTY diff rather than an error. This refuses instead, and says how to fix it.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip()
EXCL_PREFIX = ("Biological_Physics/MethylPhys/atlas/iamatlas_class_archives/",)
EXCL_EXACT = ("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv.xz",)
KEY = [("manual/MethylPhys_CPG_Operations_Manual.pdf", "the operations manual"),
       ("sop/MethylPhys_CPG_SOP.md", "the chain-of-custody procedure"),
       ("chain/guarded_push.sh", "the only sanctioned push: refuses if the gate fails"),
       ("chain/propagate.py", "regenerates every derived document and checks the rules; non-zero on drift"),
       ("kit/evaluate_necessity.py", "is this file necessary - answered from the tree"),
       ("kit/build_folder_readmes.py", "the folder READMEs are generated, not typed"),
       ("kit/build_report_tab_reference.py", "the tab reference and one figure per tab, from a real report"),
       ("kit/build_delta_bundle.py", "this script"),
       ("chain/cmb_tools.py", "the register of borrowed CMB methods, with a per-run state"),
       ("chain/disease_matching.py", "Stage 8, future work for Issue 004")]


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True).stdout


def in_scope(f):
    if f in EXCL_EXACT or f.startswith(EXCL_PREFIX):
        return False
    if f.startswith(("Biological_Physics/MethylPhys/", "Biological_Physics/Record/")):
        return True
    if f.startswith("Biological_Physics/RETIRED_2026-09/"):
        return False
    return f in ("README.md", "Biological_Physics/README.md", "docs/README.md")


def dest(f):
    for p in ("Biological_Physics/MethylPhys/", "Biological_Physics/"):
        if f.startswith(p):
            return f[len(p):]
    return f


def main(base):
    if subprocess.run(["git", "-C", ROOT, "cat-file", "-t", base],
                      capture_output=True, text=True).returncode != 0:
        print("REFUSED: %s is not in this clone. It is shallow, and a missing base gives an EMPTY diff\n"
              "rather than an error - which would hand over a bundle with nothing in it.\n"
              "Fix:  git fetch --deepen=200 origin   (or --unshallow)" % base)
        return 2
    out = os.path.join(os.path.dirname(ROOT), "chain_delta", "MethylPhys_chain_delta")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    changed, deleted, renamed = [], [], []
    for line in git("diff", "--name-status", base, "HEAD").split("\n"):
        r = line.split("\t")
        if len(r) < 2:
            continue
        if r[0].startswith("R") and len(r) >= 3:
            if in_scope(r[2]):
                renamed.append((dest(r[1]), dest(r[2])))
                changed.append(r[2])
            continue
        if not in_scope(r[1]):
            continue
        (deleted if r[0] == "D" else changed).append(dest(r[1]) if r[0] == "D" else r[1])
    man = []
    for f in sorted(set(changed)):
        src = os.path.join(ROOT, f)
        if not os.path.exists(src):
            continue
        d = os.path.join(out, dest(f))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src, d)
        man.append({"path": dest(f), "bytes": os.path.getsize(src),
                    "sha256": hashlib.sha256(open(src, "rb").read()).hexdigest()})
    head = git("rev-parse", "--short", "HEAD").strip()
    L = ["# What to copy into your offline chain copy", "",
         "**Built %s**, from `%s` to `%s`. Same layout as your bundle: what is `MethylPhys/chain/` in the "
         "repository is `chain/` here, with `Record/` beside it. The atlas is excluded as before - add "
         "`IAMAtlasREBUILD.csv.xz` and `iamatlas_class_archives/` yourself." % (time.strftime("%Y-%m-%d"),
                                                                               base, head), ""]
    if renamed:
        L += ["## Renamed - delete the old name, then copy the new one", "",
              "| delete | copy in its place |", "|---|---|"]
        L += ["| `%s` | `%s` |" % (o, n) for o, n in sorted(set(renamed))]
        L += [""]
    if deleted:
        L += ["## Deleted since your copy - remove these", "",
              "Gone deliberately. A file that no longer exists in the repository but survives in your copy "
              "is the drift all of this is for.", ""]
        L += ["- `%s`" % d for d in sorted(set(deleted))] + [""]
    have = {m["path"] for m in man}
    L += ["## The files that matter most", "", "| file | why |", "|---|---|"]
    L += ["| `%s` | %s |" % (p, w) for p, w in KEY if p in have]
    L += ["", "## Everything in this bundle", "", "| file | bytes |", "|---|---|"]
    L += ["| `%s` | %d |" % (m["path"], m["bytes"]) for m in man]
    open(os.path.join(out, "DELTA_INDEX.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"from": base, "to": git("rev-parse", "HEAD").strip(), "built": time.strftime("%Y-%m-%d %H:%M"),
               "files": man, "deleted": sorted(set(deleted)),
               "renamed": [{"from": o, "to": n} for o, n in sorted(set(renamed))]},
              open(os.path.join(out, "DELTA_MANIFEST.json"), "w"), indent=1)
    open(os.path.join(out, "MANIFEST.sha256"), "w").write(
        "".join("%s  %s\n" % (m["sha256"], m["path"]) for m in man))
    print("delta: %d files, %.1f MB | renamed %d | deleted %d | %s -> %s"
          % (len(man), sum(m["bytes"] for m in man) / 1e6, len(set(renamed)), len(set(deleted)), base, head))
    print("  at", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"))
