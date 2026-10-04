#!/usr/bin/env python3
"""Derive chain v3's step sequence from the code and write it out, so no document has to be trusted.

Every document that states the order of steps (the SOP, the chain README, the manual) can drift from what the code does. This reads
the AST of MethylPhys_Interface/run_sample.py, conductor_v3.py and stage_q_iam_a.py and emits:
  doors/CHAIN_SEQUENCE.md    the ordered live path, then every stage module in chain/ that is NOT in it (the toolkit)
  chain/chain_sequence.json  the same, as data

Run it after any change to run_sample.py, conductor_v3.py or stage_q_iam_a.py; build_all.py runs it, and release_check_v3.py fails
when the committed chain_sequence.json differs from what the code gives now. Chain v3 is the only engine (the class-floor engine was
retired 2026-10-03 and is archived privately).
"""
import ast, json, os, re, sys

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


def _intake_steps(rs):
    """Stage 0's steps, in SOP step order, as run_sample.py calls them (attribute calls on the imported module)."""
    def _key(nm):
        t = nm.split("step_0_")[1].split("_")[0]
        return (int(t[0]), t)
    names = sorted({n.func.attr for n in ast.walk(ast.parse(rs))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("step_0_")}, key=_key)
    return [{"step": nm, "where": "stage_0_intake.py", "implements": "SOP " + nm.replace("step_0_", "section 0."),
             "status": "runs in the live path, before calibration"} for nm in names]


def _toolkit():
    """The toolkit as chain/TOOLKIT.md lists it: (stage, name, first module named)."""
    p = os.path.join(HERE, "TOOLKIT.md")
    rows = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            m = re.match(r"\|\s*(\d+[a-z]?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|", l)
            if m:
                mods = re.findall(r"`([^`]+\.py)`", m.group(3))
                rows.append({"stage": m.group(1), "name": m.group(2), "where": mods[0] if mods else "none (not built)",
                             "status": ("wired into chain v3 (optional input; see chain/TOOLKIT.md)" if l.rstrip().rstrip("|").split("|")[-1].strip().startswith("**wired")
                                        else "toolkit: not yet wired into chain v3")})
    return rows


def derive():
    rs = open(os.path.join(HERE, "MethylPhys_Interface", "run_sample.py"), encoding="utf-8").read()
    intake = _intake_steps(rs)
    stage1 = []
    if "calibrate_idat_to_beta" in rs:
        stage1.append({"step": "Stage 1 - IDAT calibration", "where": "stage_1_idat_calibration.py",
                       "implements": _module_doc(os.path.join(HERE, "stage_1_idat_calibration.py")),
                       "note": "called by run_sample.py when given --grn/--red; skipped when given --betas"})
    live = derive_v3(rs, intake, stage1)
    called = {st["where"].split(":")[0] for st in live} | {"conductor_v3.py", "stage_m_met_a.py"}
    v3_src = open(os.path.join(HERE, "conductor_v3.py"), encoding="utf-8").read()
    not_wired = []
    for f in sorted(os.listdir(HERE)):
        if f.startswith("stage_") and f.endswith(".py") and f not in called and f[:-3] not in rs and f[:-3] not in v3_src:
            not_wired.append({"step": f, "where": f, "implements": _module_doc(os.path.join(HERE, f)),
                              "status": "module present; neither run_sample.py nor conductor_v3.py calls it (toolkit, chain/TOOLKIT.md)"})
    tk = _toolkit()
    dev = []
    if "_dev_run" in rs and os.path.exists(os.path.join(HERE, "dev_stages.py")):   # development flags (2026-10-04, doors/DEV_FLAGS_01.md)
        dt = ast.parse(open(os.path.join(HERE, "dev_stages.py"), encoding="utf-8").read()); dd = {n.name: n for n in dt.body if isinstance(n, ast.FunctionDef)}
        for flag, fn in (("--dev-selftare-ii", "selftare_ii"), ("--dev-direction", "direction_record"), ("--dev-trace", "trace_cell"), ("--dev-foreign", "foreign_cell"),
                         ("--dev-brightness", "brightness"), ("--dev-nilc", "nilc_e"), ("--dev-atlas-e", "atlas_e"), ("--dev-percell-b", "percell_b"),
                         ("--dev-sky", "sky"), ("--dev-epic-v2", "epicv2_calibrate")):
            if fn in dd and flag in rs:
                dev.append({"flag": flag, "step": f"dev_stages.{fn}", "where": "dev_stages.py", "implements": _doc(dd[fn]),
                            "status": "DEVELOPMENT - not commissioned: runs only with the flag; written under bundle['development']; not part of the reading"})
    return {"engine": "v3", "live_path": live, "not_in_live_path": not_wired, "toolkit": tk, "development_flags": dev,
            "counts": {"live": len(live), "not_wired": len(not_wired), "toolkit_rows": len(tk), "development_flags": len(dev)}}


# the noise index and the noise gate are steps of the live path although their functions are not named stage_* (added 2026-10-03)
NOISE_STEPS = {"noise_index": "stage 9: the array's noise index N over the noise sites",
               "noise_gate": "stage 9: N_max read here; run_neutrophil withholds the gauge state when N > N_max and the reading is untared"}


def derive_v3(rs, intake, stage1):
    """The default engine's path: Stage 0 steps (as run_sample.py calls them), Stage 1, then the stage_* calls inside
    conductor_v3.run_neutrophil in line order, Stage Q when run_sample.py calls stage_q_iam_a, then report_v3."""
    src = open(os.path.join(HERE, "conductor_v3.py"), encoding="utf-8").read(); t = ast.parse(src)
    defs = {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
    path = []
    if "specimen_refusal" in rs:   # author decision L (2026-10-04): the specimen rule runs first, before any intake step reads the IDATs
        s0 = ast.parse(open(os.path.join(HERE, "stage_0_intake.py"), encoding="utf-8").read()); d0 = {n.name: n for n in s0.body if isinstance(n, ast.FunctionDef)}
        path.append({"step": "specimen_refusal", "where": "stage_0_intake.py", "implements": _doc(d0["specimen_refusal"]),
                     "status": "runs first; a refused specimen gets a report and nothing is read"})
    path += [dict(x, status="runs in the live path, before calibration") for x in intake] + list(stage1)
    if "platform_refusal" in defs:
        path.append({"step": "platform_refusal", "where": "conductor_v3.py", "implements": _doc(defs["platform_refusal"])})
    order, seen = [], set()
    for c in ast.walk(defs["run_neutrophil"]):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and (c.func.id.startswith("stage_") or c.func.id in NOISE_STEPS) and c.func.id not in seen:
            seen.add(c.func.id); order.append((c.lineno, c.func.id))
    for _, n in sorted(order):
        st = {"step": n, "where": "conductor_v3.py", "implements": _doc(defs[n]) if n in defs else ""}
        if n in NOISE_STEPS: st["note"] = NOISE_STEPS[n]
        path.append(st)
    if "stage_q_iam_a" in rs:
        qs = open(os.path.join(HERE, "stage_q_iam_a.py"), encoding="utf-8").read(); qt = ast.parse(qs)
        qd = {n.name: n for n in qt.body if isinstance(n, ast.FunctionDef)}
        for n in ("pat_site_table", "read", "cscore"):
            if n in qd:
                path.append({"step": f"stage_q_iam_a.{n}", "where": "stage_q_iam_a.py", "implements": _doc(qd[n]),
                             "note": "runs when run_sample.py is given --pat or --site-table"})
    if "SMd.delta_sky" in rs:   # stage 12b, wired 2026-10-03 (DEV-TOOLKIT-ADDED-01)
        sm = ast.parse(open(os.path.join(HERE, "serial_mode.py"), encoding="utf-8").read()); sd = {n.name: n for n in sm.body if isinstance(n, ast.FunctionDef)}
        for n in ("check_same_person", "delta_sky"):
            path.append({"step": f"serial_mode.{n}", "where": "serial_mode.py", "implements": _doc(sd[n]), "note": "stage 12b: runs when run_sample.py is given --prior-betas and --prior-bundle"})
    if "R3.build(" in rs:
        path.append({"step": "Report", "where": "MethylPhys_Interface/report_v3.py", "implements": "one self-contained HTML page plus the JSON bundle"})
    return path


def write(d):
    json.dump(d, open(os.path.join(HERE, "chain_sequence.json"), "w"), indent=1)
    L = ["# The chain, step by step - derived from the code", "",
         "Generated by `chain/build_chain_sequence.py` from the AST of `run_sample.py`, `conductor_v3.py` and `stage_q_iam_a.py`.",
         "Do not edit by hand: re-run the generator. Every step below is a call the code actually makes, in the order it makes it.",
         "Chain v3 is the only engine; the class-floor engine (v2) was retired on 2026-10-03 and is archived privately.", "",
         f"## The live path (chain v3) - {len(d['live_path'])} steps, in order", "",
         "| # | step | implemented in | what it does |", "|---|---|---|---|"]
    for i, s in enumerate(d["live_path"], 1):
        note = f" _{s['note']}_" if s.get("note") else ""
        L.append(f"| {i} | `{s['step']}` | `{s['where']}` | {s['implements']}{note} |")
    L += ["", "## Stage modules in chain/ that the live path does not call", ""]
    if not d["not_in_live_path"]:
        L.append("_None._")
    else:
        L += ["| module | what it implements | status |", "|---|---|---|"]
        for s in d["not_in_live_path"]:
            L.append(f"| `{s['step']}` | {s['implements']} | {s['status']} |")
    if d.get("development_flags"):
        L += ["", "## Behind development flags (DEVELOPMENT - not commissioned; never part of a reading)", "",
              "| flag | step | implemented in | what it does |", "|---|---|---|---|"]
        for s in d["development_flags"]:
            L.append(f"| `{s['flag']}` | `{s['step']}` | `{s['where']}` | {s['implements']} |")
    L += ["", "## The toolkit (chain/TOOLKIT.md) - built; each stage wired only after its check passed", "",
          "| stage (SOP 2b) | name | module | status |", "|---|---|---|---|"]
    for t in d["toolkit"]:
        L.append(f"| {t['stage']} | {t['name']} | `{t['where']}` | {t['status']} |")
    L.append("")
    open(os.path.join(DOORS, "CHAIN_SEQUENCE.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    d = derive()
    if "--check" in sys.argv:
        cur = json.load(open(os.path.join(HERE, "chain_sequence.json")))
        same = cur == json.loads(json.dumps(d))
        print("chain_sequence.json", "matches the code" if same else "DIFFERS from the code - run build_chain_sequence.py")
        sys.exit(0 if same else 1)
    write(d)
    print(f"live path: {d['counts']['live']} steps | stage modules not called: {d['counts']['not_wired']} | toolkit rows: {d['counts']['toolkit_rows']}")
    for s in d["live_path"]:
        print(f"   {s['step']}")
