"""Checks for Part 5, Chapter 'Exploratory: what the law permits for gravitational engineering' (part7/p7_02_exploratory.tex).
Every number and algebraic step printed in the chapter is recomputed here. Run: python docs/verification/scripts/verify_exploratory.py
Sections follow the chapter: premise, free fall, tides, hover sign, orientation, recession, displaced region, transit, steering geometry,
weight test."""
import numpy as np, sympy as sp, scipy.constants as C
c, G, hbar, kB = C.c, C.G, C.hbar, C.k
Mpc, ly, pc, yr = 3.0856775814913673e22, 9.4607304725808e15, 3.0856775814913673e16, 365.25 * 86400
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

p("== 1. Premise: horizon temperature and Landauer cost per bit (Eq. ge_landauer) ==")
for H0 in (67.16, 72.26):
    H = H0 * 1e3 / Mpc
    TH = hbar * H / (2 * np.pi * kB)
    p(f"H0 = {H0}: H = {H:.4e} s^-1, T_H = {TH:.4e} K, k_B T_H ln2 = {kB*TH*np.log(2):.4e} J")
a = sp.symbols('a', positive=True)
E = sp.exp(1 - 1/a)
p("E(1) =", E.subs(a, 1), "; lim a->oo E =", sp.limit(E, a, sp.oo), "; dE/dlna =", sp.simplify(a*sp.diff(E, a)),
  "; d/da (dE/dlna) = 0 at a =", sp.solve(sp.diff(sp.simplify(a*sp.diff(E, a)), a), a))
p("beta_m = Omega_m/2 with Omega_m = 0.3153:", 0.3153/2)

p("== 2. Free fall: felt force zero for a uniform gradient (Eqs. ge_eom, ge_felt) ==")
x, y, z, t, m = sp.symbols('x y z t m', real=True)
g0 = sp.Matrix(sp.symbols('g_x g_y g_z', real=True))
Phi_uniform = -(g0[0]*x + g0[1]*y + g0[2]*z)
grad = sp.Matrix([sp.diff(Phi_uniform, v) for v in (x, y, z)])
xdd = -grad
p("x'' = -grad Phi =", list(xdd), "; f_felt = m(x'' + grad Phi) =", list(sp.simplify(m*(xdd + grad))))

p("== 3. Tidal residual across a body of length L (Eq. ge_tidal) ==")
M, r, L = sp.symbols('M r L', positive=True)
gr = lambda R: G*0 + sp.Symbol('G')*M/R**2
Gs = sp.Symbol('G', positive=True)
diff_g = sp.series(Gs*M/r**2 - Gs*M/(r + L)**2, L, 0, 2).removeO()
p("g(r) - g(r+L) to first order in L =", sp.simplify(diff_g), " (= L * d^2Phi/dr^2 with Phi=-GM/r:",
  sp.simplify(L*sp.diff(-Gs*M/r, r, 2)), ")")
GMe, Re, gstd = 3.986004418e14, 6.371e6, 9.80665
for LL in (10, 100):
    p(f"L = {LL} m at Earth's surface: 2GML/r^3 = {2*GMe*LL/Re**3:.3e} m s^-2 = {2*GMe*LL/Re**3/gstd:.2e} g")

p("== 4. Hover: sign of the cancelling gradient (Eq. ge_hover) ==")
Pa = sp.Function('Phi_amb')(x, y, z); Pi = sp.Function('Phi_I')(x, y, z)
g_amb = -sp.Matrix([sp.diff(Pa, v) for v in (x, y, z)])
g_tot = -sp.Matrix([sp.diff(Pa + Pi, v) for v in (x, y, z)])
sol = sp.solve([sp.Eq(gi, 0) for gi in g_tot], [sp.diff(Pi, v) for v in (x, y, z)], dict=True)[0]
p("g_tot = 0  =>  grad Phi_I =", [sol[sp.diff(Pi, v)] for v in (x, y, z)], " = +g_amb =", list(g_amb))
p("   (so grad Phi_I = +g_amb; the form grad Phi_I = -g_amb doubles the field: g_tot =", list(sp.simplify(g_amb - (-g_amb)) ), ")")

p("== 5. Orientation ==")
ah, gv = sp.symbols('a_h g_v', positive=True)
p("supported body (plumb line) in a frame with horizontal acceleration a_h: tan(theta) = a_h/g_v; theta at a_h = 0.1 g:",
  f"{np.degrees(np.arctan(0.1)):.2f} deg")
p("free fall in a uniform field: torque about the centre of mass = sum r_i x m_i g = (sum m_i r_i) x g = 0 (r_i from CM).")
th, l, mm = sp.symbols('theta ell m', positive=True)
# dumbbell: two masses mm at +/- l/2 along a unit vector at angle th from the radial direction, centre at distance r from point mass M
def force(px, py):
    R = sp.sqrt(px**2 + py**2); return -Gs*M*mm*sp.Matrix([px, py]) / R**3
r1 = sp.Matrix([r + l/2*sp.cos(th), l/2*sp.sin(th)]); r2 = sp.Matrix([r - l/2*sp.cos(th), -l/2*sp.sin(th)])
F1, F2 = force(*r1), force(*r2)
d1, d2 = r1 - sp.Matrix([r, 0]), r2 - sp.Matrix([r, 0])
tau = d1[0]*F1[1] - d1[1]*F1[0] + d2[0]*F2[1] - d2[1]*F2[0]
tau_lead = sp.simplify(sp.series(tau, l, 0, 3).removeO())
p("gravity-gradient torque on a dumbbell (leading order) =", tau_lead,
  "; check vs -(3GM/(2r^3)) (I_perp - I_axis) sin(2 theta), I_perp - I_axis = m l^2/2:",
  sp.simplify(tau_lead + 3*Gs*M/(2*r**3)*(mm*l**2/2)*sp.sin(2*th)))
p("   restoring toward theta = 0: long (minimum-inertia) axis along the line to the source; set by d^2Phi (3GM/r^3), not by g.")

p("== 6. Recession and the Hubble radius (Eqs. ge_vrec, ge_DH) ==")
for H0 in (67.16, 70.0, 72.26):
    DH = c / (H0*1e3/Mpc) / ly
    p(f"H0 = {H0}: D_H = c/H0 = {DH:.4e} ly")
Dp_ly = 136*pc/ly
p(f"Pleiades 136 pc = {Dp_ly:.1f} ly; fraction of D_H: {Dp_ly/(c/(67.16e3/Mpc)/ly):.2e} (67.16), {Dp_ly/(c/(72.26e3/Mpc)/ly):.2e} (72.26)")

p("== 7. Flat displaced region (Alcubierre form): proper time and energy density (Eqs. ge_alc, ge_tau, ge_dtau, ge_rho) ==")
v, cs = sp.symbols('v_s c', positive=True)
f = sp.Function('f')
dt, dx = sp.symbols('dt dx')
ds2 = -cs**2*dt**2 + (dx - v*f(0)*dt)**2
p("ds^2 along the centre worldline dx = v_s dt, f = 1, dy = dz = 0:", sp.simplify(ds2.subs(f(0), 1).subs(dx, v*dt)), " -> d tau = dt")
# weak field: d tau = (1 + Phi/c^2) dt; fractional difference between two clocks at potentials Phi1, Phi2
P1, P2 = sp.symbols('Phi_1 Phi_2', real=True)
p("weak-field clock rate difference (1+P1/c^2)-(1+P2/c^2) =", sp.simplify((1+P1/cs**2)-(1+P2/cs**2)))
X, Y, Z, T = sp.symbols('X Y Z T', real=True)
rs = sp.sqrt((X - v*T)**2 + Y**2 + Z**2)
F = sp.Function('f')(rs)
beta = [-v*F, 0, 0]                                   # shift vector, lapse 1, flat 3-metric (G = c = 1)
co = (X, Y, Z)
K = sp.Matrix(3, 3, lambda i, j: sp.Rational(1, 2)*(sp.diff(beta[i], co[j]) + sp.diff(beta[j], co[i])))
Ktr = K.trace(); KK = sum(K[i, j]**2 for i in range(3) for j in range(3))
rhoE = sp.simplify((Ktr**2 - KK)/(16*sp.pi))       # Hamiltonian constraint with 3R = 0
rr, rho = sp.symbols('r_s rho', positive=True)
fp = sp.Symbol("f'")
target = -(1/(8*sp.pi))*v**2*(Y**2 + Z**2)/(4*rs**2)*sp.diff(F, X)**2/((X - v*T)/rs)**2
p("Eulerian energy density rho_E = (K^2 - K_ij K^ij)/16pi; minus Alcubierre's -(1/8pi) v^2 rho^2 (f')^2/(4 r_s^2):",
  sp.simplify(rhoE - target))
p("   rho_E <= 0 everywhere f' != 0 off the axis: the weak energy condition fails in the wall.")

p("== 8. Transit to the Pleiades (Eqs. ge_tone, ge_xi) ==")
t7 = 7/365.25
p(f"7 days = {t7:.5f} yr; xi = {Dp_ly:.1f}/{t7:.5f} = {Dp_ly/t7:.4e}; with 444 ly and 0.0192 yr: {444/0.0192:.4e}")
for beta_v in (0.9, 0.99, 0.999):
    gam = 1/np.sqrt(1 - beta_v**2)
    p(f"through-space at v = {beta_v}c: Earth {Dp_ly/beta_v:.1f} yr, ship {Dp_ly/beta_v/gam:.1f} yr (gamma {gam:.2f})")

p("== 9. Steering geometry (Eq. ge_drive) ==")
rv = sp.symbols('r', positive=True)
p("Laplacian of 1/r (r>0) =", sp.simplify(sp.diff(rv**2*sp.diff(1/rv, rv), rv)/rv**2),
  " -> a static kernel obeying Laplace's equation has no local extremum away from its sources (maximum principle).")
def field(pts, src, amp, ph, k):
    tot = np.zeros(len(pts), complex)
    for s, A, f0 in zip(src, amp, ph):
        R = np.linalg.norm(pts - s, axis=1); tot += A*np.exp(1j*(k*R + f0))/R
    return np.abs(tot)**2
k = 2*np.pi/1.0
target_pt = np.array([0.4, 0.3, 1.2])
mirror = target_pt*np.array([1, 1, -1])
S3 = np.array([[3., 0, 0], [-1.5, 2.6, 0], [-1.5, -2.6, 0]])
S4 = np.vstack([S3, [0, 0, 3.5]])
for name, S_ in (("3 coplanar", S3), ("4 non-coplanar", S4)):
    ph = [-k*np.linalg.norm(target_pt - s) for s in S_]
    amp = [np.linalg.norm(target_pt - s) for s in S_]
    I_t, I_m = field(np.array([target_pt, mirror]), S_, amp, ph, k)
    p(f"{name}: cycle-averaged |Phi|^2 at target {I_t:.3f}, at mirror image {I_m:.3f}, ratio mirror/target {I_m/I_t:.3f}")
p("   three coplanar isotropic sources are mirror-symmetric about their plane: an off-plane focus always has an equal twin.")

p("== 10. Weight test ==")
p("MICROSCOPE final: eta(Ti,Pt) = [-1.5 +/- 2.3(stat) +/- 1.5(syst)] x 1e-15; combined 1-sigma",
  f"{np.hypot(2.3, 1.5):.2f}e-15")
open(__file__.replace('.py', '_output.txt'), 'w').write("\n".join(out) + "\n")
