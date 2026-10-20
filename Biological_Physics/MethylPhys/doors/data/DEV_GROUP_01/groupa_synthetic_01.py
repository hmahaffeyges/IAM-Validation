"""Group-A, synthetic first (2026-10-20, written before it was run). The instrument's job: two healthy groups read as no different after the
tare, and a planted group shift is found. Healthy people are drawn (with replacement) from the 58 real tared readings of the sham arm
(CSCORE_COMMISSIONING/band_test_GSE315366_rows.csv, clonal-haematopoiesis-negative, and band_test_GSE184269_rows.csv, whole blood:
35 + 23 healthy whole bloods, two laboratories, A_rel as read and tared by the chain). A planted pattern loss x % moves a reading by
x times the chain's leverage (2.12 % of the reading per 1 %, DEV_BLOOD_SENSITIVITY_01/blood_spread_02_chain_summary.txt), given to every member of
the case group. The group test is the chain's own comparison of tared readings: difference of means against two standard errors (Welch).
Reports, per group size, the false-positive rate on healthy-versus-healthy draws and the smallest planted loss found in at least 80 % of draws.
No new reading. Writes groupa_synthetic_01_summary.txt. Usage: python3 groupa_synthetic_01.py"""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261020)
C = os.path.join(H, "..", "CSCORE_COMMISSIONING")
g1 = pd.read_csv(os.path.join(C, "band_test_GSE315366_rows.csv")); g1 = g1[g1["clonal hematopoiesis"].astype(str) == "False"]
g2 = pd.read_csv(os.path.join(C, "band_test_GSE184269_rows.csv")); g2 = g2[g2.specimen.str.contains("blood", case=False)]
v = pd.concat([g1.A_rel, g2.A_rel]).dropna().astype(float).values; assert len(v) == 58, len(v)
LEV = 0.0212; DRAWS = 4000
def found(a, b):
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); return abs(a.mean() - b.mean()) > 2 * se
L = [f"Group-A synthetic: {len(v)} real tared healthy readings (relative SD {100*v.std(ddof=1)/np.median(v):.2f} %), leverage {100*LEV:.2f} % per 1 % loss, {DRAWS} draws per point", ""]
for n in (20, 35, 50, 100):
    fp = np.mean([found(rng.choice(v, n), rng.choice(v, n)) for _ in range(DRAWS)])
    lim = None
    for x in np.arange(0.05, 3.01, 0.05):
        p = np.mean([found(rng.choice(v, n) * (1 + LEV * x), rng.choice(v, n)) for _ in range(DRAWS // 4)])
        if p >= 0.80: lim = x; break
    L.append(f"n {n:3d} per group: healthy vs healthy flagged {100*fp:.1f} % of draws | smallest planted loss found in >= 80 % of draws: {lim:.2f} %")
open(os.path.join(H, "groupa_synthetic_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
