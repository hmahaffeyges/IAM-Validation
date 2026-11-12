"""The SAM price in the cell, from the adenylate kinase balance (PHY-083 desk derivation; nothing fitted; exploration).
Making one SAM spends ATP -> AMP + 2 Pi = (ATP -> ADP + Pi) + (ADP -> AMP + Pi). Adenylate kinase, ATP + AMP <-> 2 ADP, links the two bonds:
dG_AK = dG(ATP->ADP+Pi) - dG(ADP->AMP+Pi). So, exactly, in any cell:  SAM price / coin = 2 - dG_AK / coin (signed free energies; the coin is negative).
Where adenylate kinase is at equilibrium in the cytosol (dG_AK = 0; the assumption by which free AMP is computed from free ADP, Veech et al.
1979, J Biol Chem 254:6538), one SAM costs exactly two coins, whatever the concentrations. Off equilibrium by a mass-action factor
Gamma/K, dG_AK = RT ln(Gamma/K). The textbook 2.09-2.14 came from tabulated standard free energies (ATP -> ADP + Pi -30.5, ATP -> AMP + 2 Pi
-64.8 kJ/mol) whose difference implies dG0_AK = +3.8 kJ/mol, K_AK about 0.23, against a measured K_AK near 1 at physiological Mg
(Lawson and Veech 1979), with total rather than free nucleotides. Writes sam_cost_ak_summary.txt beside itself."""
import math, os, json
import sympy as sp
H = os.path.dirname(os.path.abspath(__file__)); OUT = []
def out(x): OUT.append(x); print(x)
# 1. the identity, symbolically
a, b, ak = sp.symbols("dG_ATP_ADP dG_ADP_AMP dG_AK")
sam = a + b; ident = sp.simplify((sam / a).subs(b, a - ak) - (2 - ak / a)) == 0
out(f"identity SAM/coin = 2 - dG_AK/coin: {'holds' if ident else 'FAILS'}")
# 2. with the canon's coin, off equilibrium by a factor Gamma/K
CANON = os.path.join(H, "..", "..", "..", "CANON", "iam_canon.json")
K = json.load(open(CANON))["constants"]; R = K["R"]["value"] / 1000; T = K["T_cell"]["value"]; coin = K["dG_ATP"]["value"] / 1000
M = K["dG_ATP"]["value"] / (K["R"]["value"] * T)
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
out(f"coin (canon) {coin:.1f} kJ/mol, M {M:.3f}")
for g in (0.25, 0.5, 1.0, 2.0, 4.0):
    r = 2 + (R * T * math.log(g)) / coin   # signed: dG_AK = RT ln(Gamma/K) and the coin is -|coin|, so the ratio is 2 + dG_AK/|coin|
    o = odds(r * M); out(f"adenylate kinase mass action {g:>4} x K: SAM = {r:.4f} coins, held error {1/(1+o):.5f}, E_hold {math.log(o):.3f} kT")
# 3. the textbook tables' own inconsistency
dG0_AK = -30.5 - (-64.8 + 30.5)
out(f"textbook standard values imply dG0_AK = {dG0_AK:+.1f} kJ/mol (K_AK = {math.exp(-dG0_AK / (R * 298.15)):.2f} at 25 C); measured K_AK is near 1")
out("measured: pooled healthy intercept 0.04526 (95 % 0.04375-0.04657); canon E_hold 3.084 kT")
open(os.path.join(H, "sam_cost_ak_summary.txt"), "w").write("\n".join(OUT) + "\n")
