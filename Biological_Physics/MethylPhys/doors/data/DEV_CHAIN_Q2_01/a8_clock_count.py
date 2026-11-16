#!/usr/bin/env python3
"""Register A8 (exploration on spent atlas data; 2026-11-15). Is the same-donor memory/naive rise the cell's DIVISIONS? The mitotic clock
(epiTOC2 sites, Teschendorff 2020) gains marks with divisions, so its stray-mark error counts them; the islands give each library's meter
(PHY-093: islands carry only the meter). Counts the eleven atlas libraries of donors 29 F and 30 F (naive, central/effector memory, effector;
CD4 and CD8) at both site sets, with the same chain command as every atlas count (run_sample.py --geo-pat --dev-count-only, chain Stage Q2)
plus --dev-count-sites. Writes xout/a8_clock_<GSM>.json and xout/a8_clock_summary.txt (few KB). Usage (explore.yml): a8_clock_count.py [GSM ...] (a subset for the canary)"""
import os, sys, json, tempfile, subprocess
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface")
os.makedirs("xout", exist_ok=True)
LIBS = {"GSM5652285": "T central memory CD4 | 30 F", "GSM5652286": "T central memory CD4 | 29 F", "GSM5652288": "T effector cell CD8 | 30 F", "GSM5652289": "T effector cell CD8 | 29 F", "GSM5652291": "T effector memory CD4 | 30 F", "GSM5652292": "T effector memory CD4 | 29 F", "GSM5652294": "T effector memory CD8 | 30 F", "GSM5652295": "T effector memory CD8 | 29 F", "GSM5652296": "Naive T cells CD4 | 29 F", "GSM5652297": "Naive T cells CD8 | 30 F", "GSM5652298": "Naive T cells CD8 | 29 F"}
if sys.argv[1:]: LIBS = {g: v for g, v in LIBS.items() if g in sys.argv[1:]}   # canary: a subset first
rows = []
for g, what in sorted(LIBS.items()):
    w = tempfile.mkdtemp(prefix=g); res = {"gsm": g, "what": what}
    for sites in ("epitoc2", "islands"):
        b = os.path.join(w, f"b_{sites}.json"); lg = os.path.join("xout", f"a8_clock_{g}_{sites}_log.txt")
        with open(lg, "w") as fh:
            rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--seq-cell", g, "--id", g,
                                 "--dev-count-only", "--dev-count-sites", sites, "--out", os.path.join(w, "r.html"), "--bundle", b],
                                cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and os.path.exists(b):
            o = json.load(open(b))["iam_a_count_only"]; cs = o.get("count_sites", {})
            res[sites] = {k: cs.get(k) for k in ("gain_iso", "gain_opp", "gain_err", "n_with_cpg_index")}
            res["genome"] = {k: o["count"].get(k) for k in ("held_err_length", "loose_share", "coupling", "gain_err")}
        else:
            res[sites] = f"FAILED rc {rc}"
    subprocess.run(["rm", "-rf", w]); rows.append(res); json.dump(res, open(os.path.join("xout", f"a8_clock_{g}.json"), "w"), indent=1)
    c, i = res.get("epitoc2"), res.get("islands")
    line = f"{g} {what}: " + (f"clock {c['gain_err']:.4f} ({c['gain_iso']}/{c['gain_opp']}), islands (meter) {i['gain_err']:.4f}, clock minus meter {c['gain_err'] - i['gain_err']:.4f}" if isinstance(c, dict) and isinstance(i, dict) and c.get("gain_err") is not None and i.get("gain_err") is not None else f"clock {c}, islands {i}")
    print("::notice::" + line[:290])
open(os.path.join("xout", "a8_clock_summary.txt"), "w").write(json.dumps(rows, indent=1))
