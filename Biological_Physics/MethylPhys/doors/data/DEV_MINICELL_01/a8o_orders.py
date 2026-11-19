import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8O item 18 (author's duty cycle: orders vs holding), exploration on GitHub with the cell simulator on main. Prediction written before this ran.
An ORDER retargets a fraction of the record: sites meant to be marked (TM) become meant-unmarked (TU) and the same number the other way; each then relaxes
under its new class's kinetics (minicell._cycle_matrix). Operating point CALIBRATED first to the measured healthy levels: held error 0.0440 at TM sites
(canon eps0 0.04376), islands level 0.0040 at TU sites (kept-bundle median), by solving the writer d of each class. Steady orders at hazard h per cycle
(share of a class retargeted per cycle) give each site an age since its last retarget (geometric). Read: (a) true held error over all TM sites; (b) the
meter-filtered held error over TM sites whose level is >= 0.80 (sites mid-rewrite read as loose and drop out). Also one order then rest (residue)."""
import sys, json, math, os
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell"); import minicell as mc
card = mc.law_card(); lag = card.v("lag")
P = {c: [card.v(f"{c}.{x}") for x in ("m", "d", "e")] for c in ("held_marked", "held_unmarked")}
def T_of(m, d, e): return mc._cycle_matrix(m, d, e, lag)[0]
def ss(T):
    w, v = np.linalg.eig(T.T); p = np.real(v[:, np.argmin(abs(w - 1))]); return p / p.sum()
lev = lambda p: p[2] + p[1] / 2
mM, dM, eM = P["held_marked"]; mU, dU, eU = P["held_unmarked"]
fM = lambda d: (1 - lev(ss(T_of(mM, d, eM)))) - 0.0440
try: dM = brentq(fM, 1e-6, 0.999); calM = "writer d solved"
except ValueError:
    eM = brentq(lambda e: (1 - lev(ss(T_of(mM, dM, e)))) - 0.0440, 1e-7, 0.999); calM = "eraser e solved (d alone could not reach 0.044)"
fU = lambda d: lev(ss(T_of(mU, d, eU))) - 0.0040
try: dU = brentq(fU, 1e-8, 0.999); calU = "writer d solved"
except ValueError:
    eU = brentq(lambda e: lev(ss(T_of(mU, dU, e))) - 0.0040, 1e-7, 0.999); calU = "eraser e solved"
TM, TU = T_of(mM, dM, eM), T_of(mU, dU, eU); pM, pU = ss(TM), ss(TU)
e0 = 1 - lev(pM); u0 = lev(pU)
lamM = sorted(np.abs(np.linalg.eigvals(TM)))[-2]; relax = -1 / math.log(lamM)
A = 4000
LM = []; p = pU.copy()
for a in range(A): LM.append(lev(p)); p = p @ TM
LU = []; p = pM.copy()
for a in range(A): LU.append(lev(p)); p = p @ TU
LM, LU = np.array(LM), np.array(LU)
def reads(h):
    w = h * (1 - h) ** np.arange(A); w[-1] += (1 - h) ** A
    true = 1 - float(np.sum(w * LM)); keep = LM >= 0.80
    filt = 1 - float(np.sum(w[keep] * LM[keep]) / np.sum(w[keep])) if keep.any() else float("nan")
    gain = float(np.sum(w * LU)); share_mid = float(np.sum(w[LM < 0.80]))
    return dict(h=h, true_held_err=true, true_factor=true / e0, filtered_held_err=filt, filtered_factor=filt / e0, excluded_share=share_mid,
                islands_level=gain, islands_factor=gain / u0, duty_cycle_D=h * relax)
H = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
rows = [reads(h) for h in H]
def need(key):
    try: return brentq(lambda h: reads(h)[key] - 1.39, 1e-6, 0.5)
    except ValueError: return None
nt, nf = need("true_factor"), need("filtered_factor")
resid = {k: float((1 - LM[k]) / e0) for k in (1, 5, 10, 20, 50, 100, 200)}
out = dict(calibration=dict(held=calM, islands=calU, held_marked=[mM, dM, eM], held_unmarked=[mU, dU, eU], lag=lag, eps0=e0, islands0=u0, relax_cycles=relax),
           sweep=rows, h_for_x139_true=nt, h_for_x139_filtered=nf, residue_after_one_order=resid)
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8o_orders.json", "w"), indent=1)
print("::notice::A8ORD CAL " + json.dumps(out["calibration"]))
for r in rows: print("::notice::A8ORD " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
print("::notice::A8ORD NEED " + json.dumps(dict(h_true=nt, h_filtered=nf, D_true=(nt * relax if nt else None), D_filtered=(nf * relax if nf else None), residue=resid)))
