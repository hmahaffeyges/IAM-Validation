import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt', 'scipy', 'sympy'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8D2 (register '#### A8D2 DELIVERY BY TYPE' and its IDENTITY note, written before this ran). (0) sympy: the cell's driven-passage odds equal the
SURFACES slow-ramp held/lost ratio. (1) cell simulator on main, law_card with baseline_candidate on: the push w = SAM_cost x M each type needs for its
open-floor held error (A8S T1b). (2) x_type / 2M against the type level."""
import sys, math, json
import sympy as sp
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell"); import minicell as mc
from scipy.optimize import brentq
from scipy.stats import spearmanr
x = sp.symbols("x", positive=True)
odds = (x / 2) / (1 - (1 - sp.exp(-x)) / x); ramp = (2 / x) * (1 - (1 - sp.exp(-x)) / x)
print(f"::notice::A8D2 identity odds(w) == 1/(ramp loss/held at x=w): {sp.simplify(odds * ramp - 1) == 0}; sudden limit odds(0+) = {sp.limit(odds, x, 0, '+')}", flush=True)
EO = {"adipocytes": [0.048114, 1.1201], "aorta endothelium": [0.045269, 1.0622], "breast basal epithelium": [0.041118, 1.0012], "breast luminal epithelium": [0.040373, 0.9937], "cardiomyocyte": [0.043433, 1.0541], "cd4 t cells": [0.036021, 1.0563], "cd8 t cells": [0.041364, 1.0491], "colon enteroendocrine": [0.041447, 1.0314], "colon epithelium": [0.042185, 1.0302], "colon fibroblasts": [0.048022, 1.0982], "colon macrophages": [0.03763, 1.0087], "cortical neurons": [0.034682, 0.8954], "effector memory cd8 t cells": [0.046711, 1.0014], "gastric antrum epithelium": [0.04535, 1.0766], "gastric body epithelium": [0.046213, 1.0643], "gastric fundus epithelium": [0.046584, 1.0551], "heart fibroblasts": [0.052294, 0.9713], "hepatocyte": [0.038999, 0.9986], "lung alveolar endothelium": [0.044037, 1.0806], "lung alveolar epithelium": [0.044522, 1.0672], "lung alveolar macrophages": [0.035105, 0.9464], "lung bronchus epithelium": [0.049995, 1.1187], "lung interstitial macrophages": [0.03483, 0.9597], "memory b cells": [0.044279, 1.0957], "naive cd4 t cells": [0.032569, 0.9323], "naive cd8 t cells": [0.031523, 0.8979], "oligodendrocytes": [0.041391, 1.0266], "pancreas endothelium": [0.045852, 1.1498], "pancreatic acinar": [0.049718, 0.951], "pancreatic alpha": [0.043348, 1.1459], "pancreatic beta": [0.04313, 1.1281], "pancreatic delta": [0.045877, 1.1651], "pancreatic duct": [0.042041, 1.0698], "small intestine epithelium": [0.044396, 1.0621], "smooth muscle": [0.053419, 0.9583], "striated muscle": [0.047452, 1.1709], "t central memory cd4": [0.041957, 1.1713], "t effector cell cd8": [0.043631, 1.1056], "t effector memory cd4": [0.045344, 1.0451], "thyroid epithelium": [0.042713, 1.1164], "vascular saphenous endothelium": [0.045513, 1.1272]}
card = mc.law_card(); card["baseline_candidate"].value = True
M = float(mc.ledger(card)["M"]); c0 = card.v("SAM_cost")
def eps_at(c): card["SAM_cost"].value = c; return float(mc.ledger(card)["eps_hold"])
print(f"::notice::A8D2 M {M:.3f}; two coins: w {2*M:.2f}, eps {eps_at(2.0):.5f} (canon 0.04454)", flush=True)
need = {t: brentq(lambda c: eps_at(c) - e, 0.2, 20.0) for t, (e, lv) in EO.items()}
lo, hi = min(need.values()), max(need.values())
print(f"::notice::A8D2 S1 push needed per restore, in coins: {lo:.3f} to {hi:.3f} (two coins = the derived level; ratio {hi/lo:.3f})", flush=True)
rho = spearmanr([need[t] for t in EO], [EO[t][1] for t in EO]); print(f"::notice::A8D2 S2 x/2M vs type level rho {rho[0]:+.3f} (p {rho[1]:.3f}); types above two coins {sum(v > 2 for v in need.values())} of {len(need)}", flush=True)
for t in sorted(need, key=need.get): print(f"A8D2 {t[:30]:30s} eps_open {EO[t][0]:.5f} coins {need[t]:.3f} x/2M {need[t]/2:.3f} level {EO[t][1]:.3f}")
