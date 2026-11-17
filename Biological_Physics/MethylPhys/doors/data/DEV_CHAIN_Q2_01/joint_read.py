#!/usr/bin/env python3
"""JOINT run (private/plans/JOINT_PLAN.md; 2026-11-16; author: 'ALWAYS THE FULL CHAIN, ONE BUTTON'). ONE spent atlas library per GitHub job:
run_sample.py --geo-pat GSM with every default on (information sphere included, no development flags, no cell label: the Deconvolver picks
the cell). Prints one notice line with every sequencing instrument's reading from the bundle (IAM-A and its count: coupling, run lengths, the
tile-by-tile line, the meter; Met-A, C-Score, territories, the information sphere: their scalar fields), and uploads the bundle as this run's
GitHub artifact (read later on GitHub machines, never downloaded over the hotspot). A derivation run and rehearsal; never commissioning.
Usage: joint_read.py GSM [GSM ...]."""
import os, sys, json, time, tempfile, subprocess
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"
os.makedirs("xout", exist_ok=True)
def scal(d, depth=0):
    """the scalar fields of a reading (numbers, short strings, small lists), two levels deep"""
    if not isinstance(d, dict): return d if isinstance(d, (int, float, str, type(None))) and len(str(d)) < 120 else None
    o = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) or v is None: o[k] = v
        elif isinstance(v, str) and len(v) < 120: o[k] = v
        elif isinstance(v, list) and len(v) <= 8 and all(isinstance(x, (int, float)) for x in v): o[k] = v
        elif isinstance(v, dict) and depth < 1:
            s = scal(v, depth + 1)
            if s: o[k] = s
    return o
for g in [a for a in sys.argv[1:] if a.startswith("GSM")]:
    w = tempfile.mkdtemp(prefix=g); b = os.path.join(w, "b.json"); t0 = time.time()
    with open(os.path.join("xout", f"joint_{g}_log.txt"), "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--id", g,
                             "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    B = json.load(open(b)) if rc == 0 and os.path.exists(b) else {}
    ia = B.get("iam_a") or {}
    cnt = ia.get("count") or {}
    dc = B.get("deconvolver", B.get("cell_shares", {}))
    top = sorted(dc.items(), key=lambda kv: -kv[1])[:2] if isinstance(dc, dict) and dc and all(isinstance(v, (int, float)) for v in dc.values()) else str(dc)[:80]
    r = {"gsm": g, "rc": rc, "minutes": round((time.time() - t0) / 60, 1),
         **{k: ia.get(k) for k in ("A", "A_derived", "predicted", "predicted_derived", "firm_derived", "held_err", "loose_share", "cell", "refusal", "refusal_code", "flag", "sites")},
         "count": {k: cnt.get(k) for k in ("held_opp", "coupling", "coupling_pairs", "loss_runs", "loss_run_ratio", "held_err_firm_tiles",
                                         "held_err_loose_tiles", "tile_line_n", "held_err_by_tile_call", "meter", "end_damage", "mol_n", "loss_dispersion", "loss_dispersion_molecules")},
         "deconvolver_top": top, "bundle_keys": sorted(B.keys())[:60] if B else [],
         **{k: scal(B.get(k)) for k in ("met_a_each_cell", "met_a_whole", "cscore_each_cell", "territories", "info_sphere", "iam_a_intake") if k in B}}
    json.dump(r, open(os.path.join("xout", f"joint_{g}.json"), "w"), indent=1, default=str)
    if os.path.exists(b): subprocess.run(["cp", b, os.path.join("xout", f"joint_{g}_bundle.json")])
    s = json.dumps(r, default=str)
    for i in range(0, min(len(s), 9500), 1900): print(f"::notice::JOINT {g} part {i // 1900 + 1}: " + s[i:i + 1900])
    subprocess.run(["rm", "-rf", w])
