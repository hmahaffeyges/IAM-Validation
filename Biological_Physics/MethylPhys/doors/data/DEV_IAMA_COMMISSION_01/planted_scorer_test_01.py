"""DEV-IAMA-COMMISSION-01: planted test of score_iama_commission_01.evaluate on synthetic library tables with known answers (no data read).
Base table: the 78 sealed libraries; line copy error 0.030-0.048 (as the EpiQC lines may differ), relative noise 0.2 %, eps_planted = 1.09 eps,
c1_iama 0.9927 (test 5) on replicate 1 of TruSeq, conversion 0.99. Each case changes one thing and states the verdict it must give.
Writes planted_scorer_test_01_output.txt. Usage: python3 planted_scorer_test_01.py"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import score_iama_commission_01 as S
LIB = pd.read_csv(os.path.join(HERE, "epiqc_design_03_04_output.csv"))[["kit", "lab", "line", "rep"]]
def base(seed=1):
    rg = np.random.default_rng(seed); L = LIB.copy(); lv = {f"HG00{i}": 0.030 + 0.003 * (i - 1) for i in range(1, 8)}
    L["eps"] = [lv[x] * (1 + rg.normal(0, 0.002)) for x in L.line]; L["opp"] = 1_000_000; L["conversion"] = 0.99; L["q0"] = "PROCEED"
    L["eps_planted"] = L.eps * 1.09; L["c1_iama"] = np.where((L.kit == "TruSeq") & (L.rep == "REP01"), 0.9927, np.nan); return L
def shift(L, m, f):                      # move a library's copy error and its planted reading together (the library itself differs)
    L.loc[m, "eps"] *= f; L.loc[m, "eps_planted"] *= f
def sel(L, **kw):
    m = np.ones(len(L), bool)
    for k, v in kw.items(): m &= (L[k].isin(v) if isinstance(v, list) else L[k] == v).values
    return m
cases = []
L = base(); cases.append(("clean", L, lambda o: o["C1"]["verdict"] == "MET" and o["C5"]["verdict"] == "MET" and o["C3_labs"]["verdict"] == "MET" and sorted(o["scope"]) == sorted(LIB.kit.unique())))
L = base(); L.loc[sel(L, kit="MethylSeq"), "eps"] *= 1.10; L.loc[sel(L, kit="MethylSeq"), "eps_planted"] *= 1.10
cases.append(("kit-wide offset 10 % (tare must remove)", L, lambda o: o["C3_kits"]["MethylSeq"]["verdict"] == "MET" and "MethylSeq" in o["scope"]))
L = base(); shift(L, sel(L, kit="TruSeq", rep="REP02"), 0.95)
cases.append(("TruSeq repeat gap", L, lambda o: o["C2"]["TruSeq LAB01"]["verdict"] == "NOT MET" and "TruSeq" not in o["scope"] and "SPLAT" in o["scope"]))
L = base(); shift(L, sel(L, kit="SPLAT", rep="REP01", line=["HG002", "HG005"]), 1.12)
cases.append(("SPLAT two lines off", L, lambda o: o["C3_kits"]["SPLAT"]["verdict"] == "NOT MET" and "SPLAT" not in o["scope"] and "TruSeq" in o["scope"]))
L = base(); shift(L, sel(L, kit="EMSeq", lab="LAB02", rep="REP01", line="HG003"), 1.06)
cases.append(("EMSeq one line off between laboratories", L, lambda o: o["C3_labs"]["verdict"] == "MET" and "EMSeq" in o["scope"]))
L = base(); shift(L, sel(L, kit="EMSeq", lab="LAB02", rep="REP01", line=["HG003", "HG006"]), 1.12)
cases.append(("EMSeq two lines off between laboratories", L, lambda o: o["C3_labs"]["verdict"] == "NOT MET" and "EMSeq" not in o["scope"]))
L = base(); L.loc[sel(L, kit="TruSeq", rep="REP01", line="HG004"), "c1_iama"] = 0.985
cases.append(("C1 bias too large", L, lambda o: o["C1"]["verdict"] == "NOT MET" and str(o["C2"]).startswith("NOT SCORED") and o["scope"] == []))
L = base(); L.loc[sel(L, kit="SPLAT", rep="REP02", line="HG007"), "conversion"] = 0.975
cases.append(("one library under the conversion limit", L, lambda o: len(o["excluded"]) == 1 and o["C2"]["SPLAT LAB01"]["n"] == 6 and "SPLAT" in o["scope"]))
L = base(); L.loc[sel(L, kit="EMSeq", lab="LAB01", rep="REP01", line="HG001"), "eps_planted"] = L.loc[sel(L, kit="EMSeq", lab="LAB01", rep="REP01", line="HG001"), "eps"] * 1.02
cases.append(("planted loss not seen in one library", L, lambda o: o["C5"]["verdict"] == "NOT MET" and o["scope"] == []))
out = []; allok = True
for name, L, want in cases:
    o = S.evaluate(L); ok = bool(want(o)); allok &= ok; out.append(f"{'PASS' if ok else 'FAIL'}  {name}: scope {o['scope']}")
out.append("PLANTED SCORER TEST " + ("PASSED" if allok else "FAILED"))
open(os.path.join(HERE, "planted_scorer_test_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
