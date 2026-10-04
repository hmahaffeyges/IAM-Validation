"""Checks for Part 5, Chapter 'Exploratory: propulsion and the conservation laws' (part5/p5_02b_propulsion.tex).
Every number and algebraic step printed in the chapter is recomputed here, and every number of the source text
(the same text carried in part7/p7_02_exploratory.tex) is re-checked in section 1.
Run: python docs/verification/scripts/verify_propulsion.py
Sections follow the chapter: source numbers, momentum of a closed craft, field-momentum thrust, negative mass,
active and passive mass, hover by weight reduction, universality, tides of a steep focus, energy of a displaced region."""
import numpy as np, sympy as sp, scipy.constants as C
from scipy.integrate import quad
c, G, hbar = C.c, C.G, C.hbar
ly, pc, yr, day = 9.4607304725808e15, 3.0856775814913673e16, 365.25 * 86400, 86400.0
Mpc = 1e6 * pc
gstd = 9.80665
M_earth, M_sun = 5.9722e24, 1.98847e30
L_P = np.sqrt(hbar * G / c**3)
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

p("== 1. Numbers of the source text (re-check) ==")
D = 136 * pc / ly
p(f"Pleiades 136 pc = {D:.1f} ly (source: 444 ly)")
p(f"Hubble radius c/H0 at 70 km/s/Mpc = {c/(70e3/Mpc)/ly:.4e} ly (source: 1.4e10); at 67.16: {c/(67.16e3/Mpc)/ly:.4e}; at 72.26: {c/(72.26e3/Mpc)/ly:.4e}")
p(f"7 days = {7*day/yr:.5f} yr (source: 0.0192)")
p(f"xi = 444/0.0192 = {444/0.0192:.4e}; with 443.6 ly and 7 d exactly: {D/(7*day/yr):.4e} (source: 2.3e4)")
p(f"Pleiades / Hubble radius: {D/(c/(67.16e3/Mpc)/ly):.3e} (67.16), {D/(c/(72.26e3/Mpc)/ly):.3e} (72.26)")

p("== 2. Momentum of a closed craft: its own field exerts no net force (Eq. pr_noforce) ==")
# Two bodies coupled through a static kernel G(u). Force on A from B's field plus force on B from A's field.
u = sp.symbols('u', real=True)
def pair_sum(kern):
    # F_A = -d/dx_A G(x_A - x_B) = -G'(u);  F_B = -d/dx_B G(x_B - x_A) = -G'(-u);  u = x_A - x_B
    Gp = sp.diff(kern(u), u)
    return sp.simplify(-Gp - Gp.subs(u, -u))
p("F_A + F_B = -(G'(u) + G'(-u)): zero for every u iff G' is odd, i.e. G even")
for name, kern in (("1/|u|", lambda w: 1/sp.sqrt(w**2)), ("Yukawa exp(-2|u|)/|u|", lambda w: sp.exp(-2*sp.sqrt(w**2))/sp.sqrt(w**2)),
                   ("Gaussian exp(-u^2)", lambda w: sp.exp(-w**2))):
    p(f"even kernel {name}: F_A + F_B =", pair_sum(kern))
p("kernel with an odd part exp(-u^2)(1+u): F_A + F_B =", sp.factor(pair_sum(lambda w: sp.exp(-w**2)*(1 + w))),
  "(nonzero: a static kernel with an odd part breaks the third law)")
rng = np.random.default_rng(1)
X = rng.normal(size=(40, 3)); q = rng.uniform(0.5, 2.0, 40)   # 40 point sources aboard, any positions and strengths
def net_force(kfun):
    F = np.zeros(3)
    for i in range(40):
        for j in range(40):
            if i == j: continue
            d = X[i] - X[j]; r = np.linalg.norm(d)
            F += -q[i] * q[j] * kfun(r) * d / r   # -q_i q_j dG/dr r_hat
    return F
p("40 sources aboard, kernel 1/r: |sum of all forces| =", f"{np.linalg.norm(net_force(lambda r: -1/r**2)):.2e}",
  "; Yukawa: ", f"{np.linalg.norm(net_force(lambda r: -(1+2*r)*np.exp(-2*r)/r**2)):.2e}")

p("== 3. Thrust from emitted field momentum: F <= P/c (Eq. pr_thrust) ==")
for m_kg, a_g in ((1000, 1), (1000, 100)):
    p(f"m = {m_kg} kg at {a_g} g: P = m a c = {m_kg*a_g*gstd*c:.3e} W")
p(f"thrust per watt 1/c = {1/c:.3e} N/W")

p("== 4. Negative mass pair (Bondi): self-acceleration with momentum and energy conserved ==")
m, Gs, r = sp.symbols('m G r', positive=True)
mp, mn = m, -m   # inertial = passive = active for each body, as in Bondi's case
# put - at x=0 and + at x=r; acceleration of + along +x: -G m_n / r^2 ; of -: +G m_p / r^2
acc_plus = -Gs * mn / r**2
acc_minus = Gs * mp / r**2
p("acceleration of + body along +x:", acc_plus, "; of - body along +x:", acc_minus, "; equal:", sp.simplify(acc_plus - acc_minus) == 0)
v = sp.symbols('v', real=True)
p("total momentum m_+ v + m_- v =", sp.simplify(mp*v + mn*v), "; total kinetic energy =", sp.simplify(mp*v**2/2 + mn*v**2/2))

p("== 5. Active and passive gravitational mass: self-force of a pair (Eq. pr_selfforce) ==")
a1, a2, p1, p2 = sp.symbols('m_a1 m_a2 m_p1 m_p2', positive=True)
Fnet = Gs * p1 * a2 / r**2 - Gs * p2 * a1 / r**2
p("net force on the pair =", sp.factor(Fnet), "= G m_p1 m_p2 (m_a2/m_p2 - m_a1/m_p1)/r^2 :",
  sp.simplify(Fnet - Gs*p1*p2*(a2/p2 - a1/p1)/r**2) == 0)
p("LLR bound on S(Al,Fe): 4e-12 (Bartlett & Van Buren 1986), 3.9e-14 (Singh et al. 2023)")

p("== 6. Hover by weight reduction (Eq. pr_hover) ==")
md, mpay, dl = sp.symbols('m_d m_pay delta', positive=True)
sol = sp.solve(sp.Eq((1 - dl)*md + mpay, 0), dl)[0]
p("zero total weight needs delta =", sol, "; for m_pay = m_d: delta =", sol.subs(mpay, md), "; passive mass of drive =",
  sp.simplify((1 - sol)*md))

p("== 7. Universality: felt acceleration eta * a (Eq. pr_eta) ==")
for a_g, eps in ((100, 0.01), (10, 0.01), (1000, 0.1)):
    p(f"a = {a_g} g, felt <= {eps} g: |eta| <= {eps/a_g:.1e}")
p("MICROSCOPE ordinary matter, Earth's field: |eta| ~ 1e-15 (Touboul et al. 2022)")

p("== 8. Tides of a steep focus (Eq. pr_focus) ==")
Mf, rr, Lb = sp.symbols('M r L', positive=True)
tid = sp.series(Gs*Mf/rr**2 - Gs*Mf/(rr+Lb)**2, Lb, 0, 2).removeO()
acc = Gs*Mf/rr**2
p("tidal / acceleration for a monopole focus at distance r =", sp.simplify(tid/acc), "(i.e. tide = 2 a L / r)")
a_dep, Lc = 100*gstd, 10.0
for d in (100.0, 1e3, 2e5):
    p(f"a = 100 g, L = 10 m, focus at {d:.0e} m: tide = {2*a_dep*Lc/d/gstd:.3g} g")
dmin = 2*a_dep*Lc/(0.01*gstd)
Mreq = a_dep*dmin**2/G
p(f"tide <= 0.01 g needs r >= {dmin:.3e} m; source strength a r^2/G = {Mreq:.3e} kg = {Mreq/M_earth:.4f} Earth masses")

p("== 9. Energy of a displaced region (Eqs. pr_Etot, pr_Ewall) ==")
x, y, z, rs, th, ph, vs = sp.symbols('x y z r_s theta phi v_s', positive=True)
ang = sp.integrate(sp.integrate(sp.sin(th)**2 * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2*sp.pi))
p("angular integral of (y^2+z^2)/r^2 over the sphere =", ang)
coef = sp.Rational(-1, 32) / sp.pi * ang
p("E = -(v_s^2/32 pi) * (8 pi/3) * int r^2 f'^2 dr  => coefficient", sp.nsimplify(coef), "(-1/12)")
Rb, De = sp.symbols('R Delta', positive=True)
wall = sp.integrate(rs**2 / De**2, (rs, Rb - De/2, Rb + De/2))
p("piecewise-linear wall: int r^2 f'^2 dr =", sp.expand(wall), "(= R^2/Delta + Delta/12)")
# numeric check with the tanh shape function
def tanh_int(R, sig):
    sech2 = lambda w: 4*np.exp(-2*abs(w))/(1+np.exp(-2*abs(w)))**2
    fp = lambda r: (sig*sech2(sig*(r+R)) - sig*sech2(sig*(r-R))) / (2*np.tanh(sig*R))
    return quad(lambda r: r**2 * fp(r)**2, 0, R + 60/sig, points=[R], limit=400)[0]
R0, s0 = 100.0, 8.0
p(f"tanh shape, R = 100, sigma = 8: int r^2 f'^2 dr = {tanh_int(R0, s0):.4f}; R^2 sigma/3 = {R0**2*s0/3:.4f}")
c2G = c**2 / G   # kg per metre of geometric length
def E_kg(R, Dl, vb):
    return (vb**2 / 12) * (R**2/Dl + Dl/12) * c2G
Dqi = lambda vb: 1e2 * vb * L_P   # Pfenning-Ford Eq. (23), alpha = 1/10
p(f"L_P = {L_P:.4e} m; c^2/G = {c2G:.4e} kg/m")
E1 = E_kg(100, Dqi(1), 1)
p(f"R = 100 m, Delta = 100 v_b L_P, v_b = 1: |E| = {E1:.3e} kg (Pfenning-Ford Eq. 29: 6.2e65 v_b g = 6.2e62 v_b kg)")
xi = D/(7*day/yr)
Exi = E_kg(100, Dqi(xi), xi)
p(f"v_b = xi = {xi:.4e}: |E| = {Exi:.3e} kg (scales as v_b at the QI wall thickness)")
M_gal = 2e42   # Pfenning-Ford Eq. (30): 2e45 g
p(f"in Milky Way masses (2e42 kg, Pfenning-Ford Eq. 30): v_b = 1: {E1/M_gal:.2e}; v_b = xi: {Exi/M_gal:.2e}")
E1m = E_kg(100, 1.0, 1)
p(f"R = 100 m, Delta = 1 m, v_b = 1: |E| = {E1m:.3e} kg = {E1m/M_sun:.3f} solar masses")
p(f"wall thickness at the QI bound: v_b = 1: {Dqi(1):.2e} m; v_b = xi: {Dqi(xi):.2e} m")

open(__file__.replace('.py', '_output.txt'), 'w').write("\n".join(out) + "\n")
