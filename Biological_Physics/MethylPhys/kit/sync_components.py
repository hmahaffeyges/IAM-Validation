#!/usr/bin/env python3
"""Make every list of the chain's components agree with what the code actually reads - 2026-09-26.

The author: "The new files you found that were not called, but are now, they need to be not only added to the
chain folder but also included in every list where the chain components are described and laid out."

Root of the drift found today: sop_repoint.py labelled a runtime file "present in the tree, read by nothing"
when the INVENTORY did not name it - not when no code read it. percell_reference_identity_v1_0.json is loaded
by cpg_conductor and was labelled unread. A status derived from a list is only as current as the list; a
status derived from the code is current by construction.

So:
  1. READERSHIP is measured: for every runtime file, which live module names it (grep over the chain's .py).
  2. The inventory gets an entry for every runtime file the code reads, with its readers, so the SOP's
     generated table (sop_repoint.py) and the reviewer manifest (build_reviewer_manifest.py) pick it up.
  3. sop_repoint.py's "not read" rule is replaced by the measured one.
  4. propagate.py gets rule 12: a runtime file read by the chain must appear in the inventory, the SOP's
     reference table, the reviewer manifest and COMPONENT_MAP; a file read by nothing must be labelled so.
     A document that disagrees with the code fails the push.
  5. The failsafe and the builders that produced today's artifacts are registered as kit components.
"""
import ast
import glob
import json
import os
import re
import subprocess

MP = "iamrepo/Biological_Physics/MethylPhys"
CH = f"{MP}/chain"
INV = f"{CH}/Runtime Matrices/chain_inventory_v1.json"


def live_modules():
    out = {}
    for p in glob.glob(f"{CH}/**/*.py", recursive=True):
        if "RETIRED" in p or "__pycache__" in p:
            continue
        out[os.path.relpath(p, MP)] = open(p, encoding="utf-8", errors="replace").read()
    return out


def readership():
    """runtime file -> sorted list of live modules whose source names it."""
    mods = live_modules()
    rt = sorted(glob.glob(f"{CH}/Runtime Matrices/**/*.json", recursive=True))
    R = {}
    for f in rt:
        b = os.path.basename(f)
        readers = sorted(m for m, src in mods.items() if b in src and not m.endswith(("build_" + b, )))
        # a file named only by the script that BUILDS it is not read by the chain
        readers = [m for m in readers if not os.path.basename(m).startswith("build_")]
        R[b] = {"path": os.path.relpath(f, "iamrepo"), "readers": readers, "bytes": os.path.getsize(f)}
    return R


DESCRIBE = {
    "percell_reference_identity_v1_0.json":
        "THE PER-CELL CALIBRATION RECORD. For each of 102 cells: p10/p50/p90 of A on the cell's identity loci, "
        "per laboratory (four labs, ~970 arrays, held-out split, coverage 0.797), one laboratory offset per lab, "
        "and each cell's centre after the offset. NORMAL for a cell is judged against ITS OWN centre. Built by "
        "kit/build_percell_reference_identity.py on the commissioned H(beta_mean)/H_min form.",
    "iamatlas_percell_identity_loci_v1_0.json":
        "PER-CELL IDENTITY LOCI - the class panels' own criterion (|mean - H_min_beta| <= 0.05, one entropy "
        "branch) applied to each of 102 cells. The surface the per-cell A is computed on. Every cell's own atlas "
        "mean reads 0.936-1.020 on its panel. Built by kit/build_percell_identity.py.",
    "percell_reference_v0_3.json":
        "Per-cell per-laboratory bands measured 2026-09-22 on the MARKER surface. Superseded for judging readings "
        "by percell_reference_identity_v1_0.json (identity surface); kept as the record of the marker-surface "
        "measurement. Must not judge identity-surface readings (bands 0.47-0.95 vs readings near 1.0).",
}

KIT_NEW = [
    ("kit/test_percell_physics.py", "kit", "gate",
     "THE FAILSAFE for the per-cell A (propagate rule 11). Checks the four root causes of 2026-09-26 - wrong "
     "surface, unmapped betas, crossed formula, unreachable reference - plus A <= 1/H_min and the fraction-0 "
     "gate, by scoring a real healthy array through the chain. Exit 1 refuses the push."),
    ("kit/build_percell_identity.py", "kit", "builder",
     "Builds iamatlas_percell_identity_loci_v1_0.json from the atlas by the class panels' criterion."),
    ("kit/build_percell_reference_identity.py", "kit", "builder",
     "Builds percell_reference_identity_v1_0.json: per-cell per-lab A bands on the identity surface, "
     "commissioned form, mapped betas, disjoint held-out split, laboratory offsets."),
]


def main():
    R = readership()
    inv = json.load(open(INV, encoding="utf-8"))
    files = inv["files"]
    by = {f["file"]: f for f in files}
    added = updated = 0
    for b, r in R.items():
        if not r["readers"]:
            continue
        e = by.get(b)
        if e is None:
            e = {"file": b, "path": r["path"], "role": "reference", "stage": "stage A",
                 "description": DESCRIBE.get(b, "runtime matrix read by " + ", ".join(r["readers"])),
                 "bytes": r["bytes"], "sha256_12": ""}
            files.append(e); by[b] = e; added += 1
        if e.get("readers") != r["readers"] or (b in DESCRIBE and e.get("description") != DESCRIBE[b]):
            e["readers"] = r["readers"]
            if b in DESCRIBE:
                e["description"] = DESCRIBE[b]
            updated += 1
    for rel, role, stage, desc in KIT_NEW:
        b = os.path.basename(rel)
        if b not in by:
            files.append({"file": b, "path": f"Biological_Physics/MethylPhys/{rel}", "role": role, "stage": stage,
                          "description": desc, "bytes": os.path.getsize(f"{MP}/{rel}"), "sha256_12": ""})
            added += 1
    inv["_meta"]["readership_measured"] = "2026-09-26: 'readers' per runtime file is derived from the live modules' source, not typed"
    json.dump(inv, open(INV, "w", encoding="utf-8"), indent=1)
    print(f"inventory: +{added} entries, {updated} updated with measured readers")

    # 3. sop_repoint's not-read rule: derive from readership
    p = f"{MP}/sop/sop_repoint.py"; s = open(p, encoding="utf-8").read()
    old = '''        if _b not in _known and _b not in s:
            _extra.append(_b)'''
    new = '''        # 2026-09-26: READ is decided by the CODE, not by the inventory. percell_reference_identity_v1_0.json was
        # loaded by cpg_conductor and this rule called it "read by nothing" because the inventory lagged. A status
        # derived from a list is only as current as the list.
        _srcs = [open(_q, encoding="utf-8", errors="replace").read() for _q in
                 _glob.glob(_os.path.join(BIO, "MethylPhys", "chain", "**", "*.py"), recursive=True)
                 if "RETIRED" not in _q and not _os.path.basename(_q).startswith("build_")]
        _read = any(_b in _t for _t in _srcs)
        if not _read and _b not in _known and _b not in s:
            _extra.append(_b)'''
    if old in s:
        s = s.replace(old, new, 1); ast.parse(s); open(p, "w", encoding="utf-8").write(s)
        print("sop_repoint: 'not read' derived from the code")
    else:
        print("sop_repoint: rule already derived" if "_read = any" in s else "sop_repoint: ANCHOR NOT FOUND")

    # 4. propagate rule 12
    p = f"{CH}/propagate.py"; s = open(p, encoding="utf-8").read()
    if "RULE 12" not in s:
        i = s.index("\n    return R", s.index("def rules():"))
        rule = '''
    # RULE 12 (2026-09-26, author: components the chain reads must be "included in every list where the chain
    # components are described and laid out"). Readership is MEASURED from the live modules' source. A runtime
    # file the code reads must be in the inventory, the SOP's reference table, the reviewer manifest and
    # COMPONENT_MAP; a runtime file nothing reads must not be described as live.
    _srcs = {_q: open(_q, encoding="utf-8", errors="replace").read() for _q in
             glob.glob(os.path.join(HERE, "**", "*.py"), recursive=True)
             if "RETIRED" not in _q and not os.path.basename(_q).startswith("build_")}
    _inv = json.load(open(os.path.join(HERE, "Runtime Matrices", "chain_inventory_v1.json"), encoding="utf-8"))
    _inv_names = {f["file"] for f in _inv["files"]}
    _cmap = read(os.path.join(os.path.dirname(HERE), "doors", "COMPONENT_MAP.md"))
    _bad = []
    for _f in glob.glob(os.path.join(HERE, "Runtime Matrices", "**", "*.json"), recursive=True):
        _b = os.path.basename(_f)
        if _b.endswith("chain_inventory_v1.json"):
            continue
        _readers = [os.path.basename(q) for q, t in _srcs.items() if _b in t]
        if _readers:
            for _doc, _txt in (("inventory", " ".join(_inv_names)), ("SOP", sop), ("manifest", manifest), ("COMPONENT_MAP", _cmap)):
                if _b not in _txt:
                    _bad.append("%s read by %s but absent from %s" % (_b, _readers[0], _doc))
            if ("`%s` - present in the tree, read by nothing" % _b) in sop:
                _bad.append("%s is read by %s but the SOP says 'read by nothing'" % (_b, _readers[0]))
    R.append(("every runtime file the code reads is in every component list; none is mislabelled unread",
              not _bad, "; ".join(_bad[:4]) if _bad else "%d runtime files checked against measured readership"
              % len(glob.glob(os.path.join(HERE, "Runtime Matrices", "**", "*.json"), recursive=True))))
'''
        s = s[:i] + rule + s[i:]
        if "import glob" not in s.split("def ")[0]:
            s = "import glob\n" + s
        ast.parse(s); open(p, "w", encoding="utf-8").write(s); print("propagate: rule 12 added")
    else:
        print("propagate: rule 12 present")


if __name__ == "__main__":
    main()
