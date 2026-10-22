"""DEV-EACH-CELL-01, synthetic test 1 (written and committed 2026-10-22 before it runs; development, not sealed).
The instrument's job: in a whole blood, Met-A for each cell (run_sample.py, record met_a_each_cell: each cell against IAM-Atlas v2.1 mixed
in the person's own cell shares) moves for the cell whose pattern was lost, and hardly for the others.
Every reading is the one command, chain/MethylPhys_Interface/run_sample.py, on saved values: the healthy example blood (GSE235717
GSM7508429, spent) and the same blood with a CONSTRUCTED loss planted: at one cell's identity sites (stage_metaa_atlas.identity_sites),
beta + f_c x 0.05 x (0.5 - mu_c), f_c that cell's share in the healthy reading (a 5 % loss of its pattern, at its share).
Reported for each plant (not scored; exploration): each cell's A untared, healthy and planted; the planted cell's move against the shift the
reading predicts for a 5 % loss (5 x shift_per_1pct_loss); the largest move of any other cell; the planted cell's share before and after.
Usage: python3 synthetic_each_cell_01.py BETAS_CSV OUTDIR"""
import os, sys, json, subprocess, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
RS = os.path.join(MP, "chain", "MethylPhys_Interface", "run_sample.py"); sys.path.insert(0, os.path.join(MP, "chain"))
import stage_metaa_atlas as SA
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); LOSS = 0.05; PLANT = ["monocytes", "cd4 t cells"]
def one(betas, tag):
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", betas, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"),
                        "--bundle", b], capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py failed: {r.stdout[-600:]} {r.stderr[-900:]}")
    B = json.load(open(b)); E = {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}
    S = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells", [])}; return E, S
E0, S0 = one(src, "healthy"); L = ["DEV-EACH-CELL-01 synthetic test 1 (every reading by run_sample.py; a 5 % loss planted at one cell's sites, at its share)",
                                  "healthy: " + ", ".join(f"{c} A {E0[c]['A']} (share {S0[c]})" for c in E0)]
beta = pd.read_csv(src, index_col=0).iloc[:, 0]; beta.index = beta.index.astype(str)
for c in PLANT:
    S = SA.identity_sites(c); mu = SA._col("mu", c).reindex(S); x = beta.reindex(S); ok = x.notna() & mu.notna(); b = beta.copy()
    b.loc[x[ok].index] = (x[ok] + S0[c] * LOSS * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
    f = os.path.join(out, f"planted_{c.replace(' ', '_')}_betas.csv"); b.to_frame("beta").to_csv(f)
    E1, S1 = one(f, "planted_" + c.replace(" ", "_"))
    mv = {k: round(E1[k]["A"] - E0[k]["A"], 5) for k in E0 if E0[k].get("A") is not None and E1.get(k, {}).get("A") is not None}
    pred = round(5 * E0[c]["shift_per_1pct_loss"], 5); oth = max((abs(v), k) for k, v in mv.items() if k != c)
    L.append(f"planted {c}: moved {mv[c]:+.5f} (predicted {pred:+.5f} for 5 %, {mv[c] / pred:.0%} of it) | largest other move {oth[1]} {oth[0]:.5f} | "
             f"{c} share {S0[c]} -> {S1[c]}")
    L.append("   every cell's move: " + ", ".join(f"{k} {v:+.5f}" for k, v in mv.items()))
open(os.path.join(out, "synthetic_each_cell_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
