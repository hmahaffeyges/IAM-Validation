import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy'], check=True)
#!/usr/bin/env python3
"""SURFACES item 5, the QUANTUM case (register '### SURFACES plan' item 5; long form link 1). Exploration; runs ON A GITHUB MACHINE.
One field mode whose frequency switches from w1 to w2 over a time tau (a mass turning on), starting in its vacuum: how many quanta does the switch create
(the spent part), and how does that depend on how sudden it is? Then the electroweak crossover's adiabaticity. Prediction written before the run."""
import json, os, math
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
R = {}
def chk(k, ok, val): R[k] = {"pass": bool(ok), "value": val}

# (a) sudden limit by matching: in-mode u = e^{-i w1 t}/sqrt(2 w1) for t < 0; for t > 0, u = alpha e^{-i w2 t}/sqrt(2 w2) + beta e^{+i w2 t}/sqrt(2 w2)
w1, w2 = sp.symbols("omega1 omega2", positive=True); al, be = sp.symbols("alpha beta")
u0 = 1 / sp.sqrt(2 * w1); du0 = -sp.I * w1 / sp.sqrt(2 * w1)
sol = sp.solve([sp.Eq(al / sp.sqrt(2 * w2) + be / sp.sqrt(2 * w2), u0), sp.Eq(-sp.I * w2 * al / sp.sqrt(2 * w2) + sp.I * w2 * be / sp.sqrt(2 * w2), du0)], [al, be], dict=True)[0]
b2 = sp.simplify(sp.Abs(sol[be])**2); a2 = sp.simplify(sp.Abs(sol[al])**2)
chk("Q1 sudden switch: |beta|^2 = (w2 - w1)^2/(4 w1 w2)", sp.simplify(b2 - (w2 - w1)**2 / (4 * w1 * w2)) == 0, str(b2))
chk("Q2 normalisation |alpha|^2 - |beta|^2 = 1", sp.simplify(a2 - b2 - 1) == 0, str(sp.simplify(a2 - b2)))
# energy bookkeeping: the mode's energy after the switch, above the new vacuum, is (|beta|^2) hbar w2 per mode; the oscillator analogue of the spent part
chk("Q3 sudden from w1 to 2 w1: |beta|^2 = 1/8", sp.simplify(b2.subs(w2, 2 * w1) - sp.Rational(1, 8)) == 0, str(b2.subs(w2, 2 * w1)))

# (b) numerical: u'' + w(t)^2 u = 0, w(t)^2 = w1^2 + (w2^2 - w1^2) (1 + tanh(t/tau))/2, from vacuum; read beta at late times
def beta2(W1, W2, tau):
    w = lambda t: math.sqrt(W1**2 + (W2**2 - W1**2) * (1 + math.tanh(t / tau)) / 2)
    T0 = -max(40 * tau, 40 / W1); T1 = -T0
    y0 = [1 / math.sqrt(2 * W1), 0.0, 0.0, -W1 / math.sqrt(2 * W1)]       # Re u, Im u, Re u', Im u'  (u = e^{-i w1 t}/sqrt(2 w1) at t = T0, phase dropped)
    f = lambda t, y: [y[2], y[3], -w(t)**2 * y[0], -w(t)**2 * y[1]]
    s_ = solve_ivp(f, (T0, T1), y0, rtol=1e-10, atol=1e-12, max_step=min(tau, 1 / W2) / 4)
    u = s_.y[0, -1] + 1j * s_.y[1, -1]; du = s_.y[2, -1] + 1j * s_.y[3, -1]; Wf = w(T1)
    beta = math.sqrt(Wf / 2) * (u - 1j * du / Wf)   # u = (alpha e^{-iWt} + beta e^{+iWt})/sqrt(2W): u - i u'/W = 2 beta e^{iWt}/sqrt(2W)
    return abs(beta)**2
W1, W2 = 1.0, 2.0
tab = {round(t, 3): beta2(W1, W2, t) for t in (0.01, 0.1, 0.3, 1.0, 2.0, 4.0)}
sudden = float((W2 - W1)**2 / (4 * W1 * W2))
chk("Q4 numeric |beta|^2 at w*tau = 0.01 matches the sudden limit 1/8 within 2 %", abs(tab[0.01] - sudden) / sudden < 0.02, tab[0.01])
chk("Q5 adiabatic suppression: |beta|^2 falls by more than 1e4 from w*tau = 0.1 to 4", tab[0.1] / max(tab[4.0], 1e-300) > 1e4, tab)
chk("Q6 the fall is exponential in tau, not a power law: the drop from tau 2 to 4 exceeds 1.5x the drop from 1 to 2 (in log |beta|^2)",
    (math.log(tab[2.0]) - math.log(tab[4.0])) > 1.5 * (math.log(tab[1.0]) - math.log(tab[2.0])), {k: math.log(v) for k, v in tab.items() if v > 0})

# (c) the electroweak crossover: rate of change of the W mass over the crossover against m_W^2 (natural units, GeV)
hbar = 6.582119569e-25; MPl = 1.220890e19; gs = 106.75; Tc = 159.5; dT = 5.0; mW = 80.369
t_c = 0.3012 * MPl / (math.sqrt(gs) * Tc**2)        # in GeV^-1
dt = 2 * dT / Tc * t_c
adiab = (mW / dt) / mW**2                              # (dm/dt)/m^2 with m rising from ~0 to m_W over dt
chk("Q7 electroweak crossover adiabaticity (dm/dt)/m^2 < 1e-15: essentially no quanta created, no entropy produced (reversible)", adiab < 1e-15, adiab)
chk("Q8 equivalently the crossover lasts > 1e15 W periods", dt * mW > 1e15, dt * mW)

out = {"checks": R, "passed": sum(v["pass"] for v in R.values()), "failed": sum(not v["pass"] for v in R.values())}
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/surfaces2_checks.json", "w"), indent=1, default=str)
print(f"::notice::SURFACES2 {out['passed']} pass {out['failed']} fail")
for k, v in R.items(): print(f"::notice::SURFACES2 {k}: {'PASS' if v['pass'] else 'FAIL'} | {str(v['value'])[:300]}")
print(f"{out['passed']} passed, {out['failed']} failed")
