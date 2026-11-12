"""The cost of one SAM in ATP coins (PHY-083 desk check; standard textbook values, sources stated; nothing fitted).
SAM synthesis (MAT): ATP + Met -> SAM + PPi + Pi, the pyrophosphate then destroyed by pyrophosphatase (PPi -> 2 Pi). Energetically the cell
spends ATP -> AMP + 2 Pi (two phosphoanhydride bonds); the coin is one ATP -> ADP + Pi. Ratio = cost of one SAM / one coin, in the cell.
Standard free energies (pH 7, 25 C, Mg-adjusted textbook values; Lehninger, Principles of Biochemistry, Table 13-6):
  ATP -> ADP + Pi -30.5; ATP -> AMP + PPi -45.6; PPi -> 2 Pi -19.2 kJ/mol.
Concentrations (mM; Lehninger Table 13-5, total adenine nucleotides): rat hepatocyte, myocyte, neuron; human erythrocyte."""
import math, os
H = os.path.dirname(os.path.abspath(__file__)); OUT = []
def out(x): OUT.append(x); print(x)
R, T = 8.314462618e-3, 310.15
G_ADP, G_AMP_PPi, G_PPi = -30.5, -45.6, -19.2
CELLS = {"rat hepatocyte": (3.38, 1.32, 0.29, 4.8), "rat myocyte": (8.05, 0.93, 0.04, 8.05), "rat neuron": (2.59, 0.73, 0.06, 2.72),
         "human erythrocyte": (2.25, 0.25, 0.02, 1.65)}   # ATP, ADP, AMP, Pi
out("standard state: one SAM = %.2f coins" % ((G_AMP_PPi + G_PPi) / G_ADP))
for k, (atp, adp, amp, pi) in CELLS.items():
    c = lambda x: x * 1e-3
    coin = G_ADP + R * T * math.log(c(adp) * c(pi) / c(atp))
    sam = G_AMP_PPi + G_PPi + R * T * math.log(c(amp) * c(pi) ** 2 / c(atp))
    out(f"{k:18}: coin {coin:6.1f} kJ/mol | one SAM {sam:6.1f} kJ/mol | ratio {sam / coin:.2f}")
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w); M = 20.94
for r_ in (2.0, 2.09, 2.14):
    o = odds(r_ * M); out(f"{r_} coins per methyl transfer: held error {1/(1+o):.5f}, E_hold {math.log(o):.3f} kT")
out("measured: pooled healthy intercept 0.04526 (95 % 0.04375-0.04657); canon E_hold 3.084")
out("the two steps' shares, standard state: %.0f %% (ATP -> AMP + PPi) / %.0f %% (PPi -> 2 Pi)" % (100 * 45.6 / 64.8, 100 * 19.2 / 64.8))
open(os.path.join(H, "sam_cost_summary.txt"), "w").write("\n".join(OUT) + "\n")
