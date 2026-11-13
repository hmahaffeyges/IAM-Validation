"""PHY-088 step 2 (exploration, desk; 2026-11-13): turn the erasers' measured share of the held error into a time, using an independent
measured erasure rate. Inputs:
  (a) DEV_ERASE_01/recount_01/recount_units.csv: held error with the erasers (held_err_wt) and without (held_err_ko), commissioned counter,
      mouse ES and neural stem cells. The marks absent because of erasure at any moment: held_err_wt - held_err_ko.
  (b) Spada et al. 2020, Nat Chem Biol, doi 10.1038/s41589-020-0621-y (bioRxiv 846584): in primed mouse pluripotent cells the global mdC
      turnover is 3-6 % faster than the passive dilution of a labelled dT reference by DNA synthesis, mostly TET-dependent. The dT
      reference falls by ln 2 per doubling time T_d, so the active (erasure) rate per mark is 0.03-0.06 x ln 2 / T_d per hour; T_d of
      primed pluripotent cultures is taken as 12-16 h (a bracket, to be replaced by the paper's own dT dilution).
Steady state: (marks absent through erasure) = (erasure rate) x (time an erased mark stays unmarked), so that time = (wt - ko) / rate.
Independent check: an erased strand is restored like a hemimethylated site, by DNMT1/UHRF1, in minutes to hours (Repli-BS).
Not a sealed test."""
import os, csv, math
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "DEV_ERASE_01", "recount_01", "recount_units.csv")
U = [r for r in csv.DictReader(open(D))]; L = []
f, Td = (0.03, 0.06), (12.0, 16.0)
klo, khi = f[0] * math.log(2) / Td[1], f[1] * math.log(2) / Td[0]
L.append(f"erasure rate per mark (Spada et al. 2020, T_d 12-16 h): {klo:.5f} to {khi:.5f} per hour")
for r in U:
    a, b = float(r["held_err_wt"]), float(r["held_err_ko"]); x = a - b
    if x <= 0: L.append(f"{r['unit']}: wt {a:.5f} ko {b:.5f}: no measurable erased share"); continue
    L.append(f"{r['unit']}: wt {a:.5f} ko {b:.5f}; marks absent through erasure {x:.5f}; an erased mark stays unmarked {x / khi:.1f} to {x / klo:.1f} h")
open(os.path.join(H, "eraser_rate_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
