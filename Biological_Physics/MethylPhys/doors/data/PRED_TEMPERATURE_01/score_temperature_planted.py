"""PRED-TEMPERATURE-01 stage T3 scorer test on planted counts (2026-10-17): law true must read MET, no temperature effect NOT MET, a planted gate
failure must be left out. Output: score_temperature_planted.json (chain-stamped)."""
import json, os, sys, subprocess, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")))
import stage_temperature as TL
rng = np.random.default_rng(3); pr = TL.predicted_ratio(10, 16); OUT = {}
for name, law in (("law_true", pr), ("no_temperature_effect", 1.0)):
    D = tempfile.mkdtemp()
    for k in range(111):
        lw = TL.logodds(0.03) * np.exp(rng.normal(0, 0.05)); lc = lw * law * np.exp(rng.normal(0, 0.01))
        for t, lo in (("10C", lc), ("16C", lw)):
            bad = k == 5 and t == "10C"
            json.dump(dict(gsm=f"G{k}{t}", accession=str(k), temp=t, usable=not bad, reason="planted gate failure" if bad else "",
                           eps_corr=float(1 / (1 + np.exp(lo))), sub_err=0.004, conv_fail=0.006), open(f"{D}/G{k}{t}.json", "w"))
    subprocess.run([sys.executable, os.path.join(HERE, "score_temperature.py"), D, D + "/scored.json"], check=True, capture_output=True)
    S = json.load(open(D + "/scored.json")); OUT[name] = dict(verdict=S["verdict"], median_ratio=round(S["reading"]["median_ratio"], 4), pairs=S["reading"]["pairs"], left_out=len(S["left_out"]))
OUT["met"] = OUT["law_true"]["verdict"] == "MET" and OUT["no_temperature_effect"]["verdict"] == "NOT MET" and OUT["law_true"]["left_out"] == 1
TL.Q2.write_json(OUT, os.path.join(HERE, "score_temperature_planted.json"), manifest=["PHY-057"]); print(json.dumps(OUT))
