#!/usr/bin/env python3
"""Derive the chain's step sequence from the code and write it out, so no document has to be trusted.

Every document that states the order of steps - the SOP, the chain README, the report's Chain tab, the manual -
was written by hand and can drift from what the code does. Three drifts had accumulated by 2026-09-23: the SOP
said the conductor runs Stage 0 and Stage 1 (it runs neither), and a stage defined in the conductor was listed as
part of the chain although run_full never calls it.

This reads the AST and emits:
  doors/CHAIN_SEQUENCE.md    the ordered live path, then what is implemented but NOT in it
  chain/chain_sequence.json  the same, for the report to render instead of a hand-written table

Run it after any change to run_sample.py or cpg_conductor.py; release_check calls it and fails on a mismatch.
"""
import ast, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOORS = os.path.normpath(os.path.join(HERE, "..", "doors"))


def _doc(node):
    d = ast.get_docstring(node) or ""
    return d.strip().split("\n")[0].rstrip(".") if d else ""


def _module_doc(path):
    try:
        return _doc(ast.parse(open(path, encoding="utf-8").read()))
    except Exception:
        return ""


def derive():
    cond_src = open(os.path.join(HERE, "cpg_conductor.py"), encoding="utf-8").read()
    cond = ast.parse(cond_src)
    defs = {n.name: n for n in cond.body if isinstance(n, ast.FunctionDef)}
    run_full = defs["run_full"]

    # ordered, de-duplicated calls to stage_* inside run_full, with the line they sit on
    order, seen = [], set()
    for c in ast.walk(run_full):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id.startswith("stage_"):
            if c.func.id not in seen:
                seen.add(c.func.id)
                order.append((c.lineno, c.func.id))
    order.sort()
    inside = [{"step": n, "implements": _doc(defs[n]) if n in defs else "", "where": "cpg_conductor.py"}
              for _, n in order]

    # what the runner does before and after the conductor
    rs_path = os.path.join(HERE, "MethylPhys_Interface", "run_sample.py")
    rs = open(rs_path, encoding="utf-8").read()
    before = []
    if "calibrate_idat_to_beta" in rs:
        before.append({"step": "Stage 1 - IDAT calibration", "where": "stage_1_idat_calibration.py",
                       "implements": _module_doc(os.path.join(HERE, "stage_1_idat_calibration.py")),
                       "note": "called by run_sample.py when given --grn/--red; skipped when given --betas"})
    after = [{"step": "Report", "where": "MethylPhys_Interface/build_methylphys.py",
              "implements": "one self-contained HTML from the bundle"}] if "B.build(" in rs else []

    # every stage_* function in the conductor, and every stage_*.py module in chain/, accounted for
    not_wired = []
    for name, node in defs.items():
        if name.startswith("stage_") and name not in seen:
            not_wired.append({"step": name, "where": "cpg_conductor.py", "implements": _doc(node),
                              "status": "defined in the conductor; run_full does not call it"})
    for f in sorted(os.listdir(HERE)):
        if f.startswith("stage_") and f.endswith(".py"):
            mod = f[:-3]
            called_by_runner = mod in rs or mod.replace("stage_1_idat_calibration", "calibrate_idat_to_beta") in rs
            called_by_cond = mod in cond_src
            if not (called_by_runner or called_by_cond):
                not_wired.append({"step": f, "where": f, "implements": _module_doc(os.path.join(HERE, f)),
                                  "status": "module present; neither run_sample.py nor cpg_conductor.py calls it"})
    # the second interface: run_batch.py drives walther_clinical.py, which runs its own stages
    batch = []
    wc_path = os.path.join(HERE, "walther_clinical.py")
    rb_path = os.path.join(HERE, "run_batch.py")
    if os.path.exists(rb_path) and os.path.exists(wc_path):
        rb = open(rb_path, encoding="utf-8").read()
        wc_src = open(wc_path, encoding="utf-8").read()
        wc = ast.parse(wc_src)
        wdefs = {n.name: n for n in wc.body if isinstance(n, ast.FunctionDef)}
        driver = "walther_clinical" if "walther_clinical" in rb else None
        if driver:
            def _ordered(fn):
                out = []
                for node in ast.walk(wdefs[fn]):
                    if isinstance(node, ast.Call):
                        nm = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                        if nm.startswith(("stage_", "calibrate_", "run_second", "build_report")):
                            out.append((node.lineno, nm))
                out.sort()
                seq, seen = [], set()
                for _, nm in out:
                    if nm not in seen:
                        seen.add(nm); seq.append(nm)
                return seq
            for step in _ordered("run_from_folder"):
                if step == "run_pipeline":
                    continue
                batch.append({"step": step, "where": "walther_clinical.py", "implements": ""})
                if step == "calibrate_idat_to_beta":
                    batch[-1]["where"] = "stage_1_idat_calibration.py"
            for step in _ordered("run_pipeline"):
                batch.append({"step": step, "where": "walther_clinical.py",
                              "implements": _doc(wdefs[step]) if step in wdefs else ""})

    # the record's own role for each file, and anything role=chain that no interface calls
    roles, gaps = {}, []
    inv = os.path.join(HERE, "Runtime Matrices", "chain_inventory_v1.json")
    if os.path.exists(inv):
        import json as _json
        d_inv = _json.load(open(inv))
        key = "files" if "files" in d_inv else list(d_inv)[0]
        for r in d_inv[key]:
            roles[os.path.basename(r.get("file", ""))] = r.get("role")
        # A stage function loads its worker by path (importlib), so the worker's name appears only as a string
        # literal inside it. Collect those too, or a file the chain genuinely runs looks uncalled: the first
        # version of this check reported 16 gaps of which 15 were dynamic loads.
        called = {st["where"] for st in before + inside + after + batch} | {st["step"] for st in before + inside + after + batch}
        called |= {"cpg_conductor.py", "run_sample.py", "run_batch.py"}
        # Scan literals ONLY inside functions that are actually on a path. Scanning whole modules counts a file
        # whose path sits in a config constant as called - which is exactly the case of the intake module: its
        # path is in walther_clinical's DEFAULT_CONFIG, while nothing on either path invokes it.
        import re as _re
        onpath = []
        for fname in [st["step"] for st in inside]:
            if fname in defs:
                onpath.append(defs[fname])
        for mod_src, names in ((rs, ["main"]), ):
            try:
                mtree = ast.parse(mod_src)
                onpath += [n for n in ast.walk(mtree) if isinstance(n, ast.FunctionDef) and n.name in names]
            except SyntaxError:
                pass
        if os.path.exists(wc_path):
            wtree = ast.parse(open(wc_path, encoding="utf-8").read())
            wd = {n.name: n for n in wtree.body if isinstance(n, ast.FunctionDef)}
            for fname in ["run_from_folder", "run_pipeline"] + [st["step"] for st in batch]:
                if fname in wd:
                    onpath.append(wd[fname])
        onpath.append(defs["run_full"])
        for node in onpath:
            for c in ast.walk(node):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    for lit in _re.findall(r"[\w./-]+\.(?:py|json|csv|npy)", c.value):
                        called.add(os.path.basename(lit))
        # run_sample.py's module level is its main
        for lit in _re.findall(r"[\w./-]+\.(?:py|json|csv|npy)", rs):
            called.add(os.path.basename(lit))
        BUILD_TIME = ("build_", "generate_")   # tools that make a runtime file once, not per-sample steps
        for f, role in roles.items():
            if role == "chain" and f.endswith(".py") and f not in called:
                if f.startswith(BUILD_TIME):
                    gaps.append({"file": f, "role": role,
                                 "status": "build-time tool: makes a runtime file once, not a per-sample step"})
                else:
                    gaps.append({"file": f, "role": role,
                                 "status": "role=chain in the inventory, and NO path calls it - a step the chain "
                                           "is documented as performing does not run"})
    return {"live_path": before + inside + after, "batch_path": batch, "not_in_live_path": not_wired,
            "roles": roles, "role_gaps": gaps,
            "counts": {"live": len(before + inside + after), "batch": len(batch),
                       "not_wired": len(not_wired), "role_gaps": len(gaps)}}


def write(d):
    json.dump(d, open(os.path.join(HERE, "chain_sequence.json"), "w"), indent=1)
    L = ["# The chain, step by step — derived from the code",
         "",
         "Generated by `chain/build_chain_sequence.py` from the AST of `run_sample.py` and `cpg_conductor.py`.",
         "Do not edit by hand: re-run the generator. Every step below is a call the code actually makes, in the",
         "order it makes it, and every stage module or function that is **not** in that path is listed underneath",
         "rather than left out.",
         "",
         f"## The live path — {len(d['live_path'])} steps, in order", "",
         "| # | step | implemented in | what it does |", "|---|---|---|---|"]
    for i, s in enumerate(d["live_path"], 1):
        note = f" _{s['note']}_" if s.get("note") else ""
        L.append(f"| {i} | `{s['step']}` | `{s['where']}` | {s['implements']}{note} |")
    if d.get("batch_path"):
        L += ["", f"## The batch path — {len(d['batch_path'])} steps, in order", "",
              "`run_batch.py` processes a folder of patient visits. It does **not** call `cpg_conductor`: it drives",
              "`walther_clinical.py`, which runs its own stage functions. The numbers in the commissioning record and",
              "in Issue 003 come from the path above, not from this one.", "",
              "| # | step | implemented in | what it does |", "|---|---|---|---|"]
        for i, st in enumerate(d["batch_path"], 1):
            L.append(f"| {i} | `{st['step']}` | `{st['where']}` | {st['implements']} |")
    if d.get("role_gaps"):
        L += ["", "## Named as chain, called by nothing", "",
              "The inventory gives these files `role=chain`, and no interface in this tree calls them. Every document",
              "that presents them as a step the chain performs is wrong until they are wired.", "",
              "| file | status |", "|---|---|"]
        for g in d["role_gaps"]:
            L.append(f"| `{g['file']}` | {g['status']} |")
    L += ["", "## Implemented, but not in the live path", ""]
    if not d["not_in_live_path"]:
        L.append("_Nothing: every stage module and every stage function in the conductor is called._")
    else:
        L.append("These exist in the tree and are **not** run by `run_sample.py`. A document that lists them as")
        L.append("steps of the chain is wrong; a reader who needs them must call them deliberately.")
        L.append("")
        L.append("| step | in | what it implements | status |")
        L.append("|---|---|---|---|")
        for s in d["not_in_live_path"]:
            L.append(f"| `{s['step']}` | `{s['where']}` | {s['implements']} | {s['status']} |")
    L.append("")
    open(os.path.join(DOORS, "CHAIN_SEQUENCE.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    d = derive()
    write(d)
    print(f"live path: {d['counts']['live']} steps | batch path: {d['counts']['batch']} | "
          f"not in the live path: {d['counts']['not_wired']} | role=chain but uncalled: {d['counts']['role_gaps']}")
    for s in d["live_path"]:
        print(f"   {s['step']}")
    for s in d.get("batch_path", []):
        print(f"   [batch] {s['step']}")
    for g in d.get("role_gaps", []):
        print(f"   GAP: {g['file']} - {g['status']}")
    for s in d["not_in_live_path"]:
        print(f"   NOT WIRED: {s['step']} - {s['status']}")
