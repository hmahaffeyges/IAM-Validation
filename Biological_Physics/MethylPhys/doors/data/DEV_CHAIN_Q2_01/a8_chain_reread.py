#!/usr/bin/env python3
"""A8 / A8N chain reading (2026-11-16; author: 'Our rule is to ALWAYS USE THE CHAIN'; 'working smarter not harder'). ONE library per job, run
in parallel: the one command (run_sample.py --geo-pat, the information sphere skipped); the Deconvolver picks the cell (no label; rule 23b-h);
IAM-A's reading copied from the bundle. Usage: a8_chain_reread.py GSM [GSM ...]. Runs ON GITHUB."""
import os, sys, json, tempfile, subprocess
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"
os.makedirs("xout", exist_ok=True)
for g in [a for a in sys.argv[1:] if a.startswith("GSM")]:
    w = tempfile.mkdtemp(prefix=g); b = os.path.join(w, "b.json")
    with open(os.path.join("xout", f"a8_chain_{g}_log.txt"), "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--no-info-sphere", "--id", g,
                             "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    B = json.load(open(b)) if rc == 0 and os.path.exists(b) else {}
    ia = B.get("iam_a", {}) if B else {"refusal": f"run failed rc {rc}"}
    dc = B.get("deconvolver", B.get("cell_shares", {})) if B else {}
    top = sorted(dc.items(), key=lambda kv: -kv[1])[:2] if isinstance(dc, dict) and all(isinstance(v, (int, float)) for v in dc.values()) else str(dc)[:80]
    r = {"gsm": g, **{k: ia.get(k) for k in ("A", "predicted", "held_err", "loose_share", "cell", "refusal", "refusal_code", "flag", "label")}, "deconvolver_top": top}
    json.dump(r, open(os.path.join("xout", f"a8_chain_{g}.json"), "w"), indent=1, default=str)
    print("::notice::" + json.dumps(r, default=str)[:600])
    subprocess.run(["rm", "-rf", w])
