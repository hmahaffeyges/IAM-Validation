"""Each cell's expected disorder at the domains, from the one command (run_sample.py --build seq-domain-disorder: the pooled level, its
binary entropy and the donor spread, from the committed 57-cell array columns), and the erythrocyte progenitor inputs' own domains in the
57-cell healthy pass (diagnostic, added after the plasma share diagnostic; nothing scored). Usage: python3 domains_by_cell.py OUT.txt"""
import os, sys, subprocess, tempfile, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); t = tempfile.mkdtemp()
subprocess.run([sys.executable, "run_sample.py", "--build", "seq-domain-disorder", "--build-out", os.path.join(t, "d.csv"), "--bundle", os.path.join(t, "d.json")],
               cwd=os.path.join(CH, "MethylPhys_Interface"), check=True, capture_output=True)
D = pd.read_csv(os.path.join(t, "d.csv")).sort_values("disorder", ascending=False); g = D.set_index("cell")
R = pd.read_csv(os.path.join(CH, "..", "doors", "data", "DEV_METAA_ATLAS_CURVE_01", "seq_atlas_01", "healthy_fullchain7_readings.csv"))
E = R[(R.cell == "erythrocyte progenitors") & (R.reading == "territory domains")]
L = [f"expected disorder at the domains ({int(D.sites.max())} sites), median over the {len(D)} cells {D.disorder.median():.4f}; highest:"]
L += [f"{r.cell}: level {r.level:.4f}, disorder {r.disorder:.4f}, donor spread {r.donor_sd:.4f}" for r in D.head(4).itertuples()]
L += [f"granulocytes: level {g.level['granulocytes']:.4f}, disorder {g.disorder['granulocytes']:.4f}",
      "erythrocyte progenitor inputs' own domains, 57-cell healthy pass: " + ", ".join(f"{a} {b:.4f}" for a, b in zip(E.gsm, E.A))]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
