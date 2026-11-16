#!/usr/bin/env python3
"""A8 chain re-read (2026-11-16, author: 'Our rule is to ALWAYS USE THE CHAIN'). The 11 same-donor atlas libraries (GSE186458, spent) through the
FULL one command (run_sample.py, no --dev-count-only), so the chain's own IAM-A reading (Stage Q2, V4.stage_q2 / stage_q2_selfref.reading)
produces A for each; nothing computed here beyond copying the bundle's iam_a block. Runs ON GITHUB."""
import os, json, tempfile, subprocess
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"
LIBS = {"GSM5652296": "naive cd4 t cells", "GSM5652298": "naive cd8 t cells", "GSM5652286": "t central memory cd4", "GSM5652289": "t effector cell cd8", "GSM5652292": "t effector memory cd4", "GSM5652295": "effector memory cd8 t cells", "GSM5652297": "naive cd8 t cells", "GSM5652285": "t central memory cd4", "GSM5652288": "t effector cell cd8", "GSM5652291": "t effector memory cd4", "GSM5652294": "effector memory cd8 t cells"}
os.makedirs("xout", exist_ok=True); rows = []
for g, cell in LIBS.items():
    w = tempfile.mkdtemp(prefix=g); b = os.path.join(w, "b.json")
    with open(os.path.join("xout", f"a8_chain_{g}_log.txt"), "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--seq-cell", cell, "--id", g,
                             "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    ia = json.load(open(os.path.join(IF, b) if not os.path.isabs(b) else b)).get("iam_a", {}) if rc == 0 and os.path.exists(b) else {"refusal": f"run failed rc {rc}"}
    r = {"gsm": g, "cell": cell, **{k: ia.get(k) for k in ("A", "predicted", "held_err", "loose_share", "refusal", "refusal_code", "flag", "label")}}
    rows.append(r); print(f"::notice::{g} {cell}: A {r['A']} (held {r['held_err']}, predicted {r['predicted']}, loose {r['loose_share']}) {r.get('refusal') or ''}"[:300])
    subprocess.run(["rm", "-rf", w])
json.dump(rows, open("xout/a8_chain_reread.json", "w"), indent=1)
