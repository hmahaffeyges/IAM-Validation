import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy'], check=True)
#!/usr/bin/env python3
"""SURFACES checks (register '### SURFACES plan'; long form open item 3, link 1). Exploration. Runs ON A GITHUB MACHINE.
Which real writes meet the 'sudden' condition behind the exact half (held = spent)?
(a) a dipole switched on in the free Maxwell field; the stiff-source circuit for comparison; (b) the electroweak crossover (the first write);
(c) Oppenheimer-Snyder collapse (black-hole formation). Prediction written in the register before this run."""
import json, os, math
import sympy as sp
R = {}
def chk(k, ok, val): R[k] = {"pass": bool(ok), "value": val}

# (a) dipole p(t) = p0 f(t/tau), f a smooth switch-on from 0 to 1
p0, eps0, c, a, tau, r, th, s = sp.symbols("p0 epsilon0 c a tau r theta s", positive=True)
E2 = (p0 / (4 * sp.pi * eps0 * r**3))**2 * (3 * sp.cos(th)**2 + 1)
U = sp.simplify(eps0 / 2 * sp.integrate(sp.integrate(E2 * r**2 * 2 * sp.pi * sp.sin(th), (th, 0, sp.pi)), (r, a, sp.oo)))
chk("A1 held near-field energy outside a: U = p0^2/(12 pi eps0 a^3)", sp.simplify(U - p0**2 / (12 * sp.pi * eps0 * a**3)) == 0, str(U))
# a smooth switch: f(s) = (1 - cos(pi s))/2 on [0,1]  ->  I_f = integral f''^2 ds
f = (1 - sp.cos(sp.pi * s)) / 2
If = sp.integrate(sp.diff(f, s, 2)**2, (s, 0, 1))
W = p0**2 * If / (6 * sp.pi * eps0 * c**3 * tau**3)          # Larmor: P = pddot^2/(6 pi eps0 c^3); pddot = p0 f''/tau^2, dt = tau ds
ratio = sp.simplify(W / U)
chk("A2 radiated/held = 2 I_f (a/(c tau))^3", sp.simplify(ratio - 2 * If * (a / (c * tau))**3) == 0, str(ratio))
tstar = sp.solve(sp.Eq(ratio, 1), tau)[0]
chk("A3 equal split only at tau* = (a/c) (2 I_f)^(1/3)", sp.simplify(tstar - (a / c) * (2 * If)**sp.Rational(1, 3)) == 0, f"tau* = {sp.nsimplify(tstar)} = {float(tstar.subs({a: 1, c: 1})):.4f} a/c (I_f = pi^4/8 = {float(If):.4f})")
chk("A4 slow writes radiate as tau^-3 (tau = 10 tau*: ratio 1e-3)", abs(float(ratio.subs(tau, 10 * tstar)) - 1e-3) < 1e-12, float(ratio.subs(tau, 10 * tstar)))
# circuit: stiff source V stepped onto uncharged C through R: loss = held for any R; ramp over T: loss -> R C/T times held
Rr, C, V, T, t = sp.symbols("R C V T t", positive=True)
held = C * V**2 / 2
loss_step = sp.integrate((V / Rr * sp.exp(-t / (Rr * C)))**2 * Rr, (t, 0, sp.oo))
chk("A5 stiff source, sudden step: loss = held for ANY R (the exact half)", sp.simplify(loss_step - held) == 0, str(sp.simplify(loss_step)))
# ramp V(t) = V t/T (T >> RC): current ~ C V/T, loss ~ (C V/T)^2 R T = (R C/T) * 2 * held
loss_ramp = (C * V / T)**2 * Rr * T
chk("A6 stiff source, slow ramp: loss/held = 2 R C/T -> 0 (adiabatic)", sp.simplify(loss_ramp / held - 2 * Rr * C / T) == 0, str(sp.simplify(loss_ramp / held)))

# (b) the electroweak crossover (D'Onofrio and Rummukainen, PRD 93, 025003, 2016: T_c = 159.5 GeV, width ~5 GeV); radiation era, g* = 106.75
hbar_GeV_s = 6.582119569e-25; MPl = 1.220890e19; gs = 106.75; Tc = 159.5; dT = 5.0; mW = 80.369
t_c = 0.3012 * MPl / (math.sqrt(gs) * Tc**2) * hbar_GeV_s        # t = 0.301 g*^-1/2 M_Pl / T^2
dt = 2 * dT / Tc * t_c                                            # t ~ T^-2: dt/t = 2 dT/T
tw_m, tw_T = hbar_GeV_s / mW, hbar_GeV_s / Tc
chk("B1 age of the universe at T_c (s)", 1e-12 < t_c < 1e-10, t_c)
chk("B2 crossover duration over the write time hbar/(m_W c^2) > 1e10 (adiabatic)", dt / tw_m > 1e10, dt / tw_m)
chk("B3 crossover duration over hbar/(k T_c) > 1e10 (adiabatic)", dt / tw_T > 1e10, dt / tw_T)

# (c) Oppenheimer-Snyder collapse from rest at R0: proper time to r = 0, tau_c = (pi/2) sqrt(R0^3/(2 G M)); in units of r_s/c with r_s = 2GM/c^2:
#     tau_c / (r_s/c) = (pi/2) (R0/r_s)^(3/2)
x = sp.symbols("x", positive=True)
tc_units = sp.pi / 2 * x**sp.Rational(3, 2)
vals = {k: float(tc_units.subs(x, k)) for k in (2, 3, 5, 10)}
chk("C1 collapse time from rest at 2-10 r_s is 4.4-50 r_s/c (sudden: of order the surface's own light-crossing time)", all(1 < v < 60 for v in vals.values()), vals)
chk("C2 collapse is >1e8 times closer to 'sudden' than the electroweak crossover (both against their own write time)", (dt / tw_m) / max(vals.values()) > 1e8, (dt / tw_m) / max(vals.values()))

out = {"checks": R, "passed": sum(v["pass"] for v in R.values()), "failed": sum(not v["pass"] for v in R.values())}
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/surfaces_checks.json", "w"), indent=1, default=str)
print(f"::notice::SURFACES {out['passed']} pass {out['failed']} fail")
for k, v in R.items(): print(f"::notice::SURFACES {k}: {'PASS' if v['pass'] else 'FAIL'} | {str(v['value'])[:300]}")
print(f"{out['passed']} passed, {out['failed']} failed")
