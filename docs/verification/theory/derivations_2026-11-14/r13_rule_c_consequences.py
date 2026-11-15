#!/usr/bin/env python3
"""EXPLORATION (register R13 rule C; GitHub explore.yml), 2026-11-14 night. If each sector's record is priced at the temperature of the horizon that
sector alone would make, beta_m = Omega_m^(3/2) (r13_temperature_rules.py). Consequences in the book's own formulas, beside the canon Omega_m/2:
matter-sector H0 (p2_02 eq:vc_h0: H0_m = H0_gamma sqrt(1 + beta_m)) against the book's table (SH0ES, megamasers, SBF, H0LiCOW), and mu_0 = 1/(1+beta_m).
Writes xout/r13_rule_c_summary.txt."""
import os, math
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
Om = 0.3153
B = {"canon Omega_m/2": Om / 2, "rule C Omega_m^(3/2)": Om ** 1.5, "free fit (X6 row)": 0.155}
data = {"SH0ES Cepheids (matter)": (73.04, 1.04), "Megamasers MCP (matter)": (73.9, 3.0), "SBF (matter)": (73.3, 2.5), "H0LiCOW (photon by the book's rule)": (73.3, 1.8)}
L = []
for Hg in (67.16, 67.36):
    L.append(f"photon-sector H0 = {Hg}:")
    for k, b in B.items():
        Hm = Hg * math.sqrt(1 + b)
        pulls = ", ".join(f"{n.split(' (')[0]} {(v - Hm) / e:+.2f} sigma" for n, (v, e) in data.items() if "matter" in n)
        L.append(f"  {k:24s} beta_m {b:.5f}: H0_m {Hm:.2f}; mu_0 = {1 / (1 + b):.4f} (mu_0 - 1 = {1 / (1 + b) - 1:+.4f}); pulls: {pulls}")
L.append("Reading guide (exploration): the matter-sector rate and mu_0 each rule implies, set beside the published matter-sector H0 values the book already uses.")
open(os.path.join(OUT, "r13_rule_c_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
