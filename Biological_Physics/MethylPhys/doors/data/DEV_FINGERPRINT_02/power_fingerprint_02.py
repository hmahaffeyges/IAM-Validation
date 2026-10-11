"""DEV-FINGERPRINT-02 power per pair, NORMAL side only (written 2026-10-10 before any seal or cancer file). Arm A as the sealed scorer computes it:
se = sqrt(0.020^2 (1/nA_c + 1/nA_n) + slope^2 sd_rep^2 (1/nW_c + 1/nW_n)), slope from the normal's own planted-loss response, sd_rep =
max(0.009, spread of the normal files), nW_c = cancer files listed for that cancer cell (count only; no file read). Power to call a fingerprint
(z > 1.645) for a Met-A excess of 0.04 / 0.06 / 0.10 above curve B: 1 - Phi(1.645 - excess/se). Arm B power is the run-loss recovery in
normal_<cell>/runloss_sim.txt. Usage: python3 power_fingerprint_02.py PATDIR PAIR [PAIR ...]"""
import os, sys, json, math, numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../..")); R9 = os.path.join(MP, "boxruns/run9_fp2")
sys.argv = [sys.argv[0], "power"] + sys.argv[1:]   # reuse PAIRS from the scorer without running it
src = open(os.path.join(HERE, "score_fingerprint_02.py")).read(); PAIRS = eval(src[src.index("PAIRS = {") + 8: src.index("STEP, PAIR")].strip().rstrip(";"))
PATD = sys.argv[2]
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
nr = pd.read_csv(os.path.join(R9, "normal_rrbs_files.csv")); cr = pd.read_csv(os.path.join(R9, "cancer_rrbs_files.csv")) if os.path.exists(os.path.join(R9, "cancer_rrbs_files.csv")) else None
for pair in sys.argv[3:]:
    NC, CC, SC, AN, AC = PAIRS[pair]; tag = NC.replace(" ", "_"); E = json.load(open(os.path.join(HERE, f"normal_{tag}", "eps_per_file.json")))
    ee = np.array([v["errors"] for v in E.values()]); oo = np.array([v["opportunities"] for v in E.values()]); eN = ee.sum() / oo.sum()
    sd_rep = max(0.009, float(np.std([H(e / o) / H(eN) for e, o in zip(ee, oo)], ddof=1)))
    ins = sorted(x for x in os.listdir(os.path.join(HERE, f"normal_{tag}")) if x.startswith("insilico_"))[0]; RS = pd.read_csv(os.path.join(HERE, f"normal_{tag}", ins))
    XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188]); slope = float(np.polyfit(np.interp(XS, RS.IAMA_rel_simple, RS.IAMA_rel), YB, 1)[0])
    nWn = len(E); nWc = int((cr.cell == CC).sum()) if cr is not None else 4
    se = math.sqrt(0.020 ** 2 * (1 / len(AC) + 1 / len(AN)) + slope ** 2 * sd_rep ** 2 * (1 / nWc + 1 / nWn))
    pw = {ex: round(float(1 - norm.cdf(1.645 - ex / se)), 3) for ex in (0.04, 0.06, 0.10)}
    print(f"{pair:15s} normal files {nWn} cancer files {nWc}{'' if cr is not None else ' (assumed)'} | arrays {len(AN)}+{len(AC)} | sd_rep {sd_rep:.4f} slope {slope:.3f} se {se:.4f} | power at excess 0.04/0.06/0.10: {pw[0.04]}/{pw[0.06]}/{pw[0.10]}")
