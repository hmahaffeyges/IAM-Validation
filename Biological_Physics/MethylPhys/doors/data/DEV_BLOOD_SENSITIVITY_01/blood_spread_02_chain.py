"""DEV-BLOOD-SENSITIVITY-01 part 2, corrected (2026-10-20). Part 2 (blood_spread_01.py) called conductor_v3.stage_a_composition and stage_m_blood
directly and skipped Stage T step 1 (self-tare II, probe-design mapping), which conductor_v3.run_neutrophil applies before reading whole blood. Its
spread (relative SD 6.36 %) is therefore not the chain's reading. The chain read the same 23 healthy whole bloods (GSE184269) in the C-score band
test (CSCORE_COMMISSIONING/band_test_GSE184269_rows.csv, run_neutrophil unchanged). This script recomputes the spread and the limits from that
reading, with the chain's own leverage (shift_per_1pct_loss) from the chain v4 report of a healthy whole blood (REPORT_EXAMPLES_V4).
No new reading; arithmetic over committed chain outputs. Writes blood_spread_02_chain_summary.txt. Usage: python3 blood_spread_02_chain.py"""
import os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..")
R = pd.read_csv(os.path.join(D, "CSCORE_COMMISSIONING", "band_test_GSE184269_rows.csv")); R = R[R.specimen.str.contains("blood", case=False) & R.A.notna()]
B = json.load(open(os.path.join(D, "REPORT_EXAMPLES_V4", "final", "GSE235717_GSM7508429_bundle.json"))); m = B["met_a"]
lev = m["shift_per_1pct_loss"] / m["A"]                       # fraction of the reading per 1 % pattern loss, this array
O = pd.read_csv(os.path.join(H, "blood_spread_01_output.csv"))
lev_old = float((O.shift_per_1pct_loss / O.A).median())
A = R.A.astype(float); med = A.median(); rsd = A.std() / med; p95 = A.quantile(0.95) / med - 1
L = ["DEV-BLOOD-SENSITIVITY-01 part 2, corrected: the spread of the CHAIN's whole-blood reading between real healthy people", "",
     f"arrays {len(A)} (GSE184269 whole blood, read by conductor_v3.run_neutrophil in the C-score band test)",
     f"median {med:.4f} | relative SD {100*rsd:.2f} % | 95th percentile {100*p95:+.2f} % above the median",
     f"part 2 (blood_spread_01, Stage T step 1 skipped) gave relative SD 6.36 % on the same arrays; correlation of the two readings {np.corrcoef(O.set_index('gsm').reindex(R.gsm).A.values, A.values)[0,1]:.2f}", "",
     f"leverage, chain v4 (GSE235717 GSM7508429, healthy whole blood, neutrophil share {m['fraction']}): {100*lev:.2f} % of the reading per 1 % pattern loss",
     f"  (part 2's leverage, median over its 23 arrays: {100*lev_old:.2f} %; the two agree)", "",
     f"one person: a loss of about {100*p95/lev/100:.2f} % clears the healthy 95th percentile"]
for n in (20, 50, 100):
    se2 = 2 * rsd / np.sqrt(n); L.append(f"a group of {n:3d}: two standard errors is {100*se2:.2f} % of the median, so a mean loss of about {100*se2/lev/100:.2f} % separates that group from healthy")
L += ["", "Development numbers on arrays already spent (no bar, no verdict): one laboratory, one leverage array. They replace part 2's limits;",
      "part 2's conclusion that one person cannot be read in whole blood is withdrawn and the question is OPEN (a sealed per-person test decides it)."]
open(os.path.join(H, "blood_spread_02_chain_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
