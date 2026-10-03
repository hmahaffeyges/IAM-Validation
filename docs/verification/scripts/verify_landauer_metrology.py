"""verify_landauer_metrology.py -- every number and algebraic step carried from the Landauer-metrology paper into
Part 4 (ch:landauer, ch:fixedorigin), recomputed.

Run from the repository root:  python docs/verification/scripts/verify_landauer_metrology.py
Writes docs/verification/scripts/verify_landauer_metrology_output.txt
Inputs (read, never typed): docs/book/figscripts/cell_data/PROC_TARE_01_per_array.parquet,
docs/book/figscripts/cell_data/FINDING_GSE125105_controls.csv (both copied from the retired kit/results/).
Values quoted from sealed outcome records are marked [record] with the file name.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "docs/book/figscripts/cell_data"   # copied from Biological_Physics/MethylPhys/kit/results/ (retired 2026-10-03)
OUT = Path(__file__).with_name("verify_landauer_metrology_output.txt")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def check(name, value, target, tol):
    ok = abs(value - target) <= tol
    say(f"  [{'OK ' if ok else 'BAD'}] {name}: {value:.6g} (book {target:.6g}, tol {tol:g})")
    return ok


allok = True
kB = 1.380649e-23
NA = 6.02214076e23
R = kB * NA
Tb = 310.15
ln2 = math.log(2)

say("1. Landauer cost at body temperature")
ebit = kB * Tb * ln2
allok &= check("k_B T ln2 (J)", ebit, 2.968e-21, 0.0005e-21)
allok &= check("per mole of bits (kJ/mol)", ebit * NA / 1e3, 1.787, 0.001)
M = 54000 / (R * Tb)
allok &= check("M = dG_ATP/RT", M, 20.94, 0.005)
allok &= check("M/ln2 (Landauer units per ATP)", M / ln2, 30.21, 0.01)

say("2. Sanchez-Mackenzie information and energy (symbolic)")
p = sp.symbols("p", positive=True)
H = -p * sp.log(p, 2) - (1 - p) * sp.log(1 - p, 2)
say("  H(1/2) =", sp.nsimplify(H.subs(p, sp.Rational(1, 2))), "bit; H(p) = H(1-p):",
    sp.simplify(H - H.subs(p, 1 - p)) == 0)
I, T = sp.symbols("I_R T", positive=True)
ER = I * sp.Symbol("k_B") * T * sp.log(2)
say("  E_R = I_R k_B T ln2; for I_R = 1 bit at T_body:", float(ER.subs({I: 1, sp.Symbol("k_B"): kB, T: Tb})), "J")

say("3. The operating ratio M = E_drive/k_B T for three substrates")
D = sp.symbols("Delta", positive=True)
Mtr = sp.simplify((D * sp.log(2)) / D)
say("  transmon: E_drive = Delta ln2, k_B T_gap = Delta  ->  M =", Mtr, "=", float(Mtr),
    "; in Landauer units M/ln2 =", sp.simplify(Mtr / sp.log(2)))
allok &= check("transmon M", float(Mtr), 0.693, 0.0005)
# CMOS value as computed in ch:cmos (AMD Ryzen 9 9950X, TDP 170 W, 4.3 GHz, 20.0-20.6e9 transistors, T_j = 348 K)
for N in (20.0e9, 20.6e9):
    Esw = 170 / (N * 4.3e9)
    say(f"  CMOS 9950X N={N:.3g}: E_sw={Esw:.3e} J, M=E_sw/k_B T_j={Esw/(kB*348):.0f}, M/ln2={Esw/(kB*348*ln2):.0f}")

say("4. The floor for one copy of the methylome (errata FB1)")
N_cpg = 28_217_448
Ef = N_cpg * ebit
allok &= check("E_floor all CpGs (J)", Ef, 8.37e-14, 0.01e-14)
eATP54 = 54000 / NA
eATP50 = 50000 / NA
say(f"  E_floor = {Ef:.4e} J = {Ef/eATP54:.3e} ATP at 54 kJ/mol = {Ef/eATP50:.3e} ATP at 50 kJ/mol")
n_meth = 0.70 * N_cpg
say(f"  methylated CpGs at 70 %: {n_meth:.3e}; chemical cost >= 1 ATP each -> {n_meth:.2e} ATP")
say(f"  chemical cost of the copy / its Landauer floor = {n_meth/(Ef/eATP54):.1f}")
say(f"  per mark: M/ln2 = {M/ln2:.1f}; per copy the floor counts all {N_cpg:,} decisions, chemistry pays the methylated ones: "
    f"0.70*M/ln2 = {0.70*M/ln2:.1f}")
budget = 1e9 * 86400
say(f"  ATP over 24 h at 1e9/s: {budget:.2e}; copy cost fraction {n_meth/budget:.1e}")

say("5. Fidelity per write")
for f in (0.02, 0.10):
    say(f"  ln(1/{f}) = {math.log(1/f):.2f} k_B T")
for s in (7, 21, 80):
    say(f"  Hopfield ln({s}) = {math.log(s):.2f} k_B T")
allok &= check("phi = 3.41/M", 3.41 / M, 0.163, 0.0005)
say(f"  sigma(phi) = 0.12/M = {0.12/M:.4f}; E_hold in Landauer units 3.41/ln2 = {3.41/ln2:.2f}")

say("6. The pipeline scale map [record PHASE1_OUTCOME.md addendum]: beta_S1 = 1.0127 beta_ref + 0.0662")
a, b = 1.0127, 0.0662


def Hb(x):
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


Href = 0.8389   # immune reference entropy used in the test [record]
say(f"  H(0.7318) = {Hb(0.7318):.4f} bits (reference beta of the test; reference entropy {Href})")
for bS1 in (0.815,):
    bref = (bS1 - b) / a
    say(f"  noob beta-bar {bS1} -> mapped {bref:.4f}; A unmapped {Hb(bS1)/Href:.3f}; A mapped {Hb(bref)/Href:.3f}")
say(f"  offset at beta_ref = 0.737: beta_S1 - beta_ref = {a*0.737+b-0.737:.4f}")

say("7. Three laboratories on one scale (Table 3) from PROC_TARE_01_per_array.parquet, column A_raw = mapped A")
d = pd.read_parquet(KIT / "PROC_TARE_01_per_array.parquet")
u = d.loc[d.lab == "GSE87571", "A_raw"].median()
tab = {}
for lab, nm in (("GSE87571", "Uppsala"), ("GSE42861", "Karolinska"), ("GSE111629", "UCLA"), ("GSE125105", "Munich")):
    x = d.loc[d.lab == lab, "A_raw"]
    tab[lab] = (len(x), x.median(), x.quantile(.1), x.quantile(.9), x.median() - u)
    say(f"  {nm:11s} n={len(x):3d} median {x.median():.3f}  p10-p90 {x.quantile(.1):.3f}-{x.quantile(.9):.3f}  offset {x.median()-u:+.3f}")
allok &= check("Uppsala median", tab["GSE87571"][1], 0.992, 0.0005)
allok &= check("Karolinska median", tab["GSE42861"][1], 1.016, 0.0005)
allok &= check("UCLA median", tab["GSE111629"][1], 0.961, 0.0005)
three = d[d.lab.isin(["GSE87571", "GSE42861", "GSE111629"])]
frac = ((three.A_raw >= 0.95) & (three.A_raw < 1.05)).mean()
allok &= check("fraction of 756 arrays in [0.95,1.05)", frac, 0.946, 0.0005)
say(f"  arrays counted: {len(three)}")

say("8. The low-signal laboratory (FINDING_GSE125105_controls.csv; three arrays per laboratory)")
c = pd.read_csv(KIT / "FINDING_GSE125105_controls.csv").groupby("lab").median(numeric_only=True)
say(c[["nonpoly_G", "nonpoly_R", "neg_G", "neg_R", "bsII_R", "hyb_G", "poobah_fail%"]].round(1).to_string())
rG = c.loc["GSE87571", "nonpoly_G"] / c.loc["GSE125105", "nonpoly_G"]
rR = c.loc["GSE87571", "nonpoly_R"] / c.loc["GSE125105", "nonpoly_R"]
say(f"  Uppsala / Munich non-polymorphic signal: G {rG:.2f}x, R {rR:.2f}x  (one-sixth to one-seventh)")
say("  SNP homozygous-cluster SD [record FINDING_GSE125105_LOW_SIGNAL.md]: Munich 0.069/0.072 vs Uppsala 0.017/0.016 -> "
    f"{0.069/0.017:.1f}x, {0.072/0.016:.1f}x")

say("9. Control-probe ridge, leave-one-laboratory-out [record LABZERO02_OUTCOME.md P4b]")
for lab, err in (("Karolinska", 0.0041), ("Uppsala", 0.0180), ("UCLA", 0.0256), ("Munich", 0.0517)):
    say(f"  {lab:11s} |error| {err:.4f}  within 0.010: {err <= 0.010}")

say("10. Intake line 0.93 on the 48-array measurement [record PROC_INTAKE_01_OUTCOME.md]")
say("  medians (min/max): Uppsala 0.985 (min 0.979), Karolinska 0.975 (min 0.891), UCLA 0.953 (min 0.932), Munich 0.878 (max 0.928)")
say("  Munich 12/12 quarantined (0.764-0.928); Uppsala first 100: 100/100 advance")

say("\nALL CHECKS PASS" if allok else "\nSOME CHECKS FAILED")
OUT.write_text("\n".join(lines) + "\n")
