#!/usr/bin/env python3
"""Blind-lab joint read (register A8, A9; PHY-084's spent libraries, GSE173787, hg19; a cosmologist's second probe: does each type's position repeat in a
second laboratory, and how wide is the between-laboratory term). Runs ON A GITHUB MACHINE. For each GSM: downloads its kept molecule file (artifact
erase-<GSM> of an earlier erase_test run; no re-alignment), then runs the ONE BUTTON with every default, exactly as the joint run's reader does except
for the input: run_sample.py --pat <file> --specimen "sorted cells" --id <GSM> (no cell label; the Deconvolver picks the cell; no development flags).
Prints the same JOINT notice parts as joint_read.py. Spent data only; never commissioning. Usage: referee_read.py GSM:RUN [GSM:RUN ...]."""
import os, sys, json, time, tempfile, subprocess, glob
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"
os.makedirs("xout", exist_ok=True)
def scal(d, depth=0):
    if not isinstance(d, dict): return d if isinstance(d, (int, float, str, type(None))) and len(str(d)) < 120 else None
    o = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool)) or v is None: o[k] = v
        elif isinstance(v, str) and len(v) < 120: o[k] = v
        elif isinstance(v, list) and len(v) <= 8 and all(isinstance(x, (int, float)) for x in v): o[k] = v
        elif isinstance(v, dict) and depth < 1:
            s = scal(v, depth + 1)
            if s: o[k] = s
    return o
for arg in [a for a in sys.argv[1:] if a.startswith("GSM")]:
    g, run = arg.split(":")
    w = tempfile.mkdtemp(prefix=g); b = os.path.join(w, "b.json"); t0 = time.time()
    d = subprocess.run(["gh", "run", "download", run, "-R", os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation"), "-n", f"erase-{g}", "-D", os.path.join(w, "prev")], capture_output=True, text=True)
    pats = glob.glob(os.path.join(w, "prev", "**", f"{g}.pat.gz"), recursive=True)
    if d.returncode != 0 or not pats:
        r = {"gsm": g, "rc": None, "refusal": f"kept molecule file not available: {(d.stderr or '')[-160:]}"}
    else:
        with open(os.path.join("xout", f"joint_{g}_log.txt"), "w") as fh:
            rc = subprocess.run(["python3", "run_sample.py", "--pat", pats[0], "--specimen", "sorted cells", "--id", g,
                                 "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
        B = json.load(open(b)) if rc == 0 and os.path.exists(b) else {}
        ia = B.get("iam_a") or {}; cnt = ia.get("count") or {}
        dc = B.get("deconvolver", B.get("cell_shares", {}))
        top = sorted(dc.items(), key=lambda kv: -kv[1])[:2] if isinstance(dc, dict) and dc and all(isinstance(v, (int, float)) for v in dc.values()) else str(dc)[:80]
        r = {"gsm": g, "lab": "GSE173787", "rc": rc, "minutes": round((time.time() - t0) / 60, 1),
             **{k: ia.get(k) for k in ("A", "A_derived", "predicted", "predicted_derived", "firm_derived", "held_err", "loose_share", "cell", "refusal", "refusal_code", "flag", "sites")},
             "count": {k: cnt.get(k) for k in ("held_opp", "coupling", "loss_run_ratio", "held_err_firm_tiles", "held_err_loose_tiles", "meter", "mol_n", "loss_dispersion")},
             "deconvolver_top": top}
        if os.path.exists(b): subprocess.run(["cp", b, os.path.join("xout", f"joint_{g}_bundle.json")])
    json.dump(r, open(os.path.join("xout", f"joint_{g}.json"), "w"), indent=1, default=str)
    s = json.dumps(r, default=str)
    for i in range(0, min(len(s), 7600), 1900): print(f"::notice::JOINT {g} part {i // 1900 + 1}: " + s[i:i + 1900])
    subprocess.run(["rm", "-rf", w])
