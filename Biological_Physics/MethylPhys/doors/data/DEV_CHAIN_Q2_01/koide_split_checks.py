import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""koide_split_checks.py  (2026-11-18, derivation spider, IAM web; private)

Gap 4 of IAM_LAW_FORWARD_2026-11-18.md (Step 4): why |s_perp| = |s_par| for the
charged leptons. s = (sqrt m_e, sqrt m_mu, sqrt m_tau); s_par along (1,1,1),
s_perp in the plane orthogonal to it.  Q = (|s_par|^2+|s_perp|^2)/(3|s_par|^2)
(forward D2), so Koide <=> |s_perp|^2/|s_par|^2 = 1.

Sections
  A  the forward derivation's calculation: three-component pure-dephasing pointer,
     H_int invariant under rotation about (1,1,1); classical, then quantum (spin-1)
  B  candidate (i)   maximum entropy
  C  candidate (ii)  sudden-write held = spent between axis and plane
  D  candidate (iii) equipartition per quadrature
  E  neighbours: Koide 1990 potential; Hubner 2026
  F  measured masses (book tab:ko:masses values)

Runs in a few seconds, < 200 MB.  Output: koide_split_checks_output.json
"""
import json, math, itertools
import numpy as np
import sympy as sp

R = []
def check(cid, desc, ok, value=None):
    R.append({"id": cid, "desc": desc, "pass": bool(ok),
              "value": None if value is None else str(value)})
    print(("PASS " if ok else "FAIL ") + cid + "  " + desc + ("" if value is None else "  -> " + str(value)))

def Qratio(rho2):            # Q from rho2 = |s_perp|^2/|s_par|^2
    return sp.Rational(1, 3) * (1 + rho2)

# ---------------------------------------------------------------------------
# A. the pointer calculation
# ---------------------------------------------------------------------------
n = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
Ppar = n * n.T
Pperp = sp.eye(3) - Ppar
th = sp.symbols('theta', real=True)
K = sp.Matrix([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
Rot = sp.simplify(sp.eye(3) + sp.sin(th) * K + (1 - sp.cos(th)) * K * K)   # Rodrigues
L = sp.Matrix(3, 3, sp.symbols('l0:9'))
Lsym = (L + L.T) / 2
eqs = list((Rot * Lsym - Lsym * Rot).subs(th, sp.Rational(7, 10)))
eqs += list((Rot * Lsym - Lsym * Rot).subs(th, sp.Rational(13, 10)))
sol = sp.solve([sp.nsimplify(sp.expand(e)) for e in eqs], list(L), dict=True)
Lgen = sp.simplify(Lsym.subs(sol[0]))
a_, b_ = sp.symbols('a b', real=True)
# show Lgen = a Ppar + b Pperp for some a, b
coefs = [sp.Matrix(3, 3, lambda i, j: sp.diff(Lgen[i, j], sy)) for sy in sorted(Lgen.free_symbols, key=str)]
span = sp.Matrix([list(cm) for cm in coefs])
inspan = all(sp.Matrix([list(Ppar), list(Pperp), list(cm)]).rank() == 2 for cm in coefs)
check("A1", "symmetric coupling tensor commuting with every rotation about (1,1,1) is a*P_par + b*P_perp (solution space has dimension 2)",
      span.rank() == 2 and inspan, "dimension %d" % span.rank())

# A2: sudden write of a fixed pointer value s into a Gaussian bath with coupling Lambda
#     (one bath oscillator per eigen-sector; the Step-1 result held = spent = U per sector)
m_, w_, c_, s_ = sp.symbols('m omega c s', positive=True)
q, t = sp.symbols('q t', real=True)
# bath mode coupled as -c*s*q: equilibrium shift q* = c s/(m w^2); held U = c^2 s^2/(2 m w^2)
qstar = c_ * s_ / (m_ * w_**2)
held = sp.Rational(1, 2) * m_ * w_**2 * qstar**2
qt = qstar * (1 - sp.cos(w_ * t))                       # sudden switch-on from rest at q = 0
pt = m_ * sp.diff(qt, t)
spent = sp.simplify(pt**2 / (2 * m_) + sp.Rational(1, 2) * m_ * w_**2 * (qt - qstar)**2)
check("A2a", "sudden write, one sector: ringing energy about the new equilibrium = held = c^2 s^2/(2 m w^2)",
      sp.simplify(spent - held) == 0, sp.simplify(held))
spar, sperp = sp.symbols('s_par s_perp', positive=True)
U_tot = sp.Rational(1, 2) * (a_ * spar**2 + b_ * sperp**2)
held_sec = [held.subs({c_: sp.sqrt(a_) * w_ * sp.sqrt(m_), s_: spar}), held.subs({c_: sp.sqrt(b_) * w_ * sp.sqrt(m_), s_: sperp})]
spent_sec = [spent.subs({c_: sp.sqrt(a_) * w_ * sp.sqrt(m_), s_: spar}), spent.subs({c_: sp.sqrt(b_) * w_ * sp.sqrt(m_), s_: sperp})]
ident = all(sp.simplify(h - p_) == 0 for h, p_ in zip(held_sec, spent_sec))
jac = sp.Matrix([[sp.diff(sum(held_sec), spar), sp.diff(sum(held_sec), sperp)]])
check("A2b", "rotation-invariant coupling: held = spent = (a|s_par|^2 + b|s_perp|^2)/2 holds identically in each sector for ANY (|s_par|, |s_perp|): the write imposes no relation between them",
      ident and sp.simplify(sum(held_sec) - U_tot) == 0 and jac.rank() == 1 and all(sp.simplify(x) != 0 for x in jac), U_tot)
# extremize held energy over directions at fixed |s|
phi = sp.symbols('phi', real=True)
Uphi = sp.Rational(1, 2) * (a_ * sp.cos(phi)**2 + b_ * sp.sin(phi)**2)
crit = sp.solve(sp.diff(Uphi, phi), phi)
check("A2c", "extrema of the held energy over directions at fixed |s| sit at 0 or 90 deg (eigen-sectors), never at 45 deg unless a = b (then every angle)",
      all(sp.simplify(sp.sin(2 * cc)) == 0 for cc in crit), crit)

# A3: the Step-3 Euler half applied between axis and plane, three role assignments
W = sp.symbols('W', positive=True)                     # source energy
heldE, spentE = W / 2, W / 2                            # degree-1 source into degree-2 store
assign = {
    "axis = source W, plane = held W/2": (W, heldE),
    "axis = held W/2, plane = source W": (heldE, W),
    "axis = held W/2, plane = spent W/2": (heldE, spentE),
}
out = {}
for k_, (ax, pl) in assign.items():
    out[k_] = Qratio(sp.simplify(pl / ax))
check("A3", "Euler 1:2 half between axis and plane: Q = 1/2, 1, 2/3 for the three role assignments; only 'axis held, plane spent' gives Koide",
      list(out.values()) == [sp.Rational(1, 2), 1, sp.Rational(2, 3)], out)

# A4: quantum version. Spin-1 in the Cartesian (vector) basis: (J_k)_ij = -i eps_kij
eps = lambda i, j, k: sp.LeviCivita(i, j, k)
J = [sp.Matrix(3, 3, lambda i, j: -sp.I * eps(k, i, j)) for k in range(3)]
comm_ok = all(sp.simplify(J[0] * J[1] - J[1] * J[0] - sp.I * J[2]) == sp.zeros(3)
              for _ in [0])
Jn = sp.simplify(sum((J[k] * n[k] for k in range(3)), sp.zeros(3)))
J2 = sp.simplify(sum((Jk * Jk for Jk in J), sp.zeros(3)))
check("A4a", "Cartesian spin-1: [Jx,Jy] = iJz and J^2 = 2 (j = 1) on the three-component space",
      comm_ok and sp.simplify(J2 - 2 * sp.eye(3)) == sp.zeros(3))
check("A4b", "J_n generates the rotation about (1,1,1): exp(-i theta J_n) = Rodrigues rotation",
      np.allclose(np.array(sp.N((-sp.I * sp.Rational(3, 5) * Jn).exp()), dtype=complex),
                     np.array(sp.N(Rot.subs(th, sp.Rational(3, 5))), dtype=float), atol=1e-12))
ev = Jn.eigenvects()
vals = sorted([sp.nsimplify(e[0]) for e in ev])
v0 = [e[2][0] for e in ev if sp.simplify(e[0]) == 0][0]
check("A4c", "J_n eigenvalues -1, 0, +1; the m = 0 eigenvector of the Cartesian rep is the axis (1,1,1) itself",
      vals == [-1, 0, 1] and sp.simplify(v0.cross(sp.Matrix([1, 1, 1]))) == sp.zeros(3, 1), vals)
# pure dephasing H_int = J_n (x) B commutes with the rotation; decoherence factor between m, m'
check("A4d", "[J_n (x) B, exp(-i theta J_n) (x) 1] = 0: H_int invariant under rotation about (1,1,1); pointer states = J_n eigenstates",
      np.allclose(np.array(sp.N(Jn * (-sp.I * sp.Rational(1, 3) * Jn).exp() - (-sp.I * sp.Rational(1, 3) * Jn).exp() * Jn), dtype=complex), 0, atol=1e-12))
Gam = sp.symbols('Gamma', positive=True)
mm = [-1, 0, 1]
deph = {(a, b): sp.exp(-Gam * (a - b)**2) for a in mm for b in mm}
check("A4e", "pure dephasing multiplies rho_{m m'} by exp(-Gamma (m-m')^2): diagonal kept, phase delta lost; <J_n> and <J_perp^2> = J^2 - J_n^2 conserved",
      deph[(1, 1)] == 1 and deph[(1, 0)] != 1 and sp.simplify((J2 - Jn * Jn) * Jn - Jn * (J2 - Jn * Jn)) == sp.zeros(3))
# second moments in each pointer state
Jperp2 = sp.simplify(J2 - Jn * Jn)
split = {}
for e in ev:
    vec = e[2][0] / sp.sqrt((e[2][0].H * e[2][0])[0])
    mval = sp.nsimplify(e[0])
    ax = sp.simplify((vec.H * Jn * vec)[0])**2
    pl = sp.simplify((vec.H * Jperp2 * vec)[0])
    split[int(mval)] = (sp.nsimplify(ax), sp.nsimplify(pl))
check("A4f", "spin-1 pointer states: (axis <J_n>^2, plane <J_perp^2>) = (1,1) for m = +-1 (stretched), (0,2) for m = 0",
      split[1] == (1, 1) and split[-1] == (1, 1) and split[0] == (0, 2), split)

# A5: general j, stretched state m = j: axis j^2, plane j; Q_vm(j) = j(j+1)/(3 j^2)
def spin_mats(j):
    d = int(round(2 * j + 1)); ms = [j - k for k in range(d)]
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((d, d), complex)
    for k in range(1, d):
        mval = ms[k]; Jp[k - 1, k] = math.sqrt(j * (j + 1) - mval * (mval + 1))
    Jx = (Jp + Jp.conj().T) / 2; Jy = (Jp - Jp.conj().T) / (2j)
    return Jx, Jy, Jz
tab = {}
okA5 = True
for j in [0.5, 1, 1.5, 2, 3, 5]:
    Jx, Jy, Jz = spin_mats(j)
    psi = np.zeros(int(round(2 * j + 1)), complex); psi[0] = 1
    ax = (psi.conj() @ Jz @ psi).real**2
    pl = (psi.conj() @ (Jx @ Jx + Jy @ Jy) @ psi).real
    dx2 = (psi.conj() @ Jx @ Jx @ psi).real; dy2 = (psi.conj() @ Jy @ Jy @ psi).real
    Qvm = (ax + pl) / (3 * ax); Qnu = (2 * j + 2) / (3 * 2 * j)
    tab[j] = (round(ax, 12), round(pl, 12), round(Qvm, 12))
    okA5 &= abs(ax - j * j) < 1e-12 and abs(pl - j) < 1e-12 and abs(Qvm - Qnu) < 1e-12
    okA5 &= abs(math.sqrt(dx2 * dy2) - abs(math.sqrt(ax)) / 2) < 1e-12      # Robertson saturated
check("A5a", "stretched state of spin j: axis j^2, plane j (Robertson bound saturated); Q_vm(j) = (j+1)/(3j) = Q_nu(nu = 2j)", okA5, tab)
jj = sp.symbols('j', positive=True)
check("A5b", "equal split for the stretched (coherent) pointer, j^2 = j, has the single solution j = 1",
      sp.solve(sp.Eq(jj**2, jj), jj) == [1])
pell = [(j, m) for j in range(1, 300) for m in range(0, j + 1) if j * (j + 1) == 2 * m * m]
check("A5c", "non-stretched pointer states with equal split need j(j+1) = 2 m^2: (1,1), (8,6), (49,35), (288,204); smallest is j = 1, m = 1",
      pell[:4] == [(1, 1), (8, 6), (49, 35), (288, 204)], pell[:4])
# Step-3 Euler balance applied to the Casimir j^2 (degree 2) + j (degree 1)
check("A5d", "Euler balance n_store*E_held = |n_source|*E_source with axis j^2 (degree 2) and plane j (degree 1): 2 j^2 = j -> j = 1/2, Q = 1 (NOT the Koide point)",
      sp.solve(sp.Eq(2 * jj**2, jj), jj) == [sp.Rational(1, 2)] and sp.simplify((jj + 1) / (3 * jj)).subs(jj, sp.Rational(1, 2)) == 1)
# Holstein-Primakoff: plane of the stretched state is one canonical pair at its zero point
jl = 400.0
Jx, Jy, Jz = spin_mats(jl)
psi = np.zeros(int(2 * jl + 1), complex); psi[0] = 1
qx = (psi.conj() @ Jx @ Jx @ psi).real / jl; py = (psi.conj() @ Jy @ Jy @ psi).real / jl
check("A5e", "Holstein-Primakoff: J_x/sqrt(j), J_y/sqrt(j) act as one canonical pair (q,p); stretched state has <q^2> = <p^2> = 1/2 (zero point)",
      abs(qx - 0.5) < 1e-12 and abs(py - 0.5) < 1e-12, (qx, py))
# A6: Reading B, the real Cartesian vector as a spin-1 state (NOT the same as A4f)
thB = sp.symbols('thB', real=True)
e1 = Pperp * sp.Matrix([1, 0, 0]); e1 = e1 / sp.sqrt((e1.T * e1)[0])
vB = sp.cos(thB) * n + sp.sin(thB) * e1
JexpB = [sp.simplify((vB.T * Jk * vB)[0]) for Jk in J]
pm = {}
for e in ev:
    vec = e[2][0] / sp.sqrt((e[2][0].H * e[2][0])[0])
    pm[int(sp.nsimplify(e[0]))] = sp.simplify(sp.Abs((vec.H * vB)[0])**2)
pmK = {k: sp.nsimplify(sp.simplify(v.subs(thB, sp.pi / 4))) for k, v in pm.items()}
Jn2B = sp.simplify((vB.T * Jn * Jn * vB)[0]).subs(thB, sp.pi / 4)
check("A6", "real s read as a spin-1 state: <J> = 0; Koide gives J_n distribution (1/4, 1/2, 1/4) and <J_n^2> = 1/2, <J_perp^2> = 3/2 (not the A4f split)",
      all(x == 0 for x in JexpB) and pmK == {-1: sp.Rational(1, 4), 0: sp.Rational(1, 2), 1: sp.Rational(1, 4)} and sp.nsimplify(Jn2B) == sp.Rational(1, 2),
      pmK)

# ---------------------------------------------------------------------------
# B. candidate (i): maximum entropy
# ---------------------------------------------------------------------------
p = sp.symbols('p0:3', positive=True)
lam = sp.symbols('lam')
Lg = -sum(pi * sp.log(pi) for pi in p) - lam * (sum(p) - 1)
solB = sp.solve([sp.diff(Lg, pi) for pi in p] + [sum(p) - 1], list(p) + [lam], dict=True)
pB = [solB[0][pi] for pi in p]
check("B1", "MaxEnt of the mass shares over the full simplex -> uniform -> Q = 1/3 (NO)",
      all(sp.simplify(x - sp.Rational(1, 3)) == 0 for x in pB), pB)
c2 = sp.symbols('c2', positive=True)                  # cos^2 theta = 1/(3Q)
hbin = -(c2 * sp.log(c2, 2) + (1 - c2) * sp.log(1 - c2, 2))
cstar = sp.solve(sp.diff(hbin, c2), c2)
check("B2", "binary which-sector entropy h(cos^2 theta), 2 sectors (axis, plane): maximum 1 bit at cos^2 theta = 1/2 <=> Q = 2/3 (formal YES; needs K = 2)",
      cstar == [sp.Rational(1, 2)] and sp.simplify(hbin.subs(c2, sp.Rational(1, 2))) == 1 and sp.Rational(1, 3) / cstar[0] == sp.Rational(2, 3))
check("B3", "3 sectors (axis, alpha, beta) at MaxEnt: each 1/3 -> |s_perp|^2 = 2|s_par|^2 -> Q = 1 (NO; the book's K = 3 case, p4_15a:172-175)",
      Qratio(2) == 1)
# B4: MaxEnt over the plane (over delta) at fixed Q = 2/3
dl = np.linspace(0, 2 * np.pi / 3, 200001)
r = math.sqrt(2)
sk = np.array([1 + r * np.cos(dl + 2 * np.pi * k / 3) for k in range(3)])
pos = np.all(sk > 0, axis=0)
mk = sk**2; pk = mk / mk.sum(0)
H = -(pk * np.log(np.where(pk > 0, pk, 1))).sum(0)
Hpos = np.where(pos, H, -1)
dmax = dl[np.argmax(Hpos)]
Hmeas = float(np.interp(0.222225, dl, H))
check("B4", "MaxEnt of the mass shares over delta at fixed Q = 2/3 (positive arcs): fixes delta, not Q, and picks the arc edge (one sqrt m -> 0), not the measured delta (NO)",
      min(abs(dmax - x) for x in [np.pi / 12, 2 * np.pi / 3 - np.pi / 12]) < 1e-3 and Hpos.max() > Hmeas,
      "delta_max = %.6f rad (pi/12 = %.6f); H_max = %.6f nats; H at measured delta 0.222225 = %.6f nats" % (dmax, np.pi / 12, Hpos.max(), Hmeas))
# B5: MaxEnt amplitudes (exponential at fixed mean): population vs three-sample
x_ = sp.symbols('x', positive=True); mu = sp.symbols('mu', positive=True)
f = sp.exp(-x_ / mu) / mu
Es = sp.integrate(x_ * f, (x_, 0, sp.oo)); Es2 = sp.integrate(x_**2 * f, (x_, 0, sp.oo))
Qpop = sp.simplify(Es2 / (3 * Es**2))
# three iid exponential draws: shares ~ Dirichlet(1,1,1); E[sum q^2] = 3*E[q1^2] = 3*(1*2)/(3*4)
EQ3 = 3 * sp.Rational(1 * 2, 3 * 4)
rng = np.random.default_rng(20261118)
sx = rng.exponential(size=(400000, 3)); Qmc = ((sx**2).sum(1) / sx.sum(1)**2)
check("B5", "MaxEnt positive amplitude (exponential): population Q = E[s^2]/(3E[s]^2) = 2/3, but three draws give E[Q] = 1/2 exactly (partial: ensemble only)",
      Qpop == sp.Rational(2, 3) and EQ3 == sp.Rational(1, 2) and abs(Qmc.mean() - 0.5) < 2e-3,
      "pop %s, E[Q_3] %s, MC %.4f" % (Qpop, EQ3, Qmc.mean()))
# B6: typicality
v = rng.normal(size=(400000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
c3 = ((v @ np.ones(3) / math.sqrt(3))**2).mean()
check("B6", "typicality: Liouville (isotropic) measure in 3D gives E[cos^2 theta] = 1/3 (Q-point 1); flat measure on the 2D record (theta uniform) gives 1/2",
      abs(c3 - 1 / 3) < 3e-3 and sp.integrate(sp.cos(th)**2, (th, 0, sp.pi / 2)) / (sp.pi / 2) == sp.Rational(1, 2), "%.4f" % c3)

# ---------------------------------------------------------------------------
# C. candidate (ii): sudden write, axis = held static displacement, plane = ringing pair
# ---------------------------------------------------------------------------
k_ = m_ * w_**2
alpha = qt - qstar; beta = pt / (m_ * w_)
ring = sp.simplify(alpha**2 + beta**2)
check("C1", "sudden write from a blank start: phase-space orbit is a circle about (q*,0) of radius q*: the ring passes through the blank point",
      sp.simplify(ring - qstar**2) == 0, ring)
sv = qstar * n + alpha * e1 + beta * (n.cross(e1))     # axis = DC displacement, plane = ringing pair
sv = sp.simplify(sv)
spar2 = sp.simplify((sv.T * n)[0]**2); sperp2 = sp.simplify((sv.T * Pperp * sv)[0])
Qsv = sp.simplify((sv.T * sv)[0] / (sum(sv))**2)
check("C2", "axis = held displacement q*, plane = (alpha, beta) of the ringing: |s_perp|^2 = |s_par|^2 = q*^2 and Q = 2/3 at every t (no fitted input)",
      sp.simplify(spar2 - sperp2) == 0 and sp.simplify(Qsv - sp.Rational(2, 3)) == 0, Qsv)
# Brannen form: the ringing phase is delta
comps = [sp.simplify(sv[k] / (qstar / sp.sqrt(3))) for k in range(3)]
rB = sp.sqrt(2)
okC3 = True
for tt in [0.1, 0.7, 2.3]:
    vals_ = [float(c.subs({w_: 1, t: tt, c_: 1, s_: 1, m_: 1})) for c in comps]
    S0 = sum(vals_) / 3; S1 = sum(vv * np.exp(2j * np.pi * kk / 3) for kk, vv in enumerate(vals_)) / 3
    okC3 &= abs(S0 - 1) < 1e-12 and abs(2 * abs(S1) - rB) < 1e-12
check("C3", "read on three phases, the written vector is Brannen's s_k = x[1 + sqrt2 cos(delta + 2 pi k/3)] with delta advancing with the ringing (omega t)",
      okC3)
# C4: time-domain sampling of position only
samp = [qstar * (1 - sp.cos(ph + 2 * sp.pi * kk / 3)) for kk, ph in zip(range(3), [0] * 3)]
ph0 = sp.symbols('ph0', real=True)
samp = [qstar * (1 - sp.cos(ph0 + 2 * sp.pi * kk / 3)) for kk in range(3)]
Qs = sp.simplify(sp.trigsimp(sum(x**2 for x in samp) / sum(samp)**2))
check("C4", "position-only sampling of the same ringing at three phases (one quadrature): r = 1, Q = 1/2 (NO)",
      sp.simplify(Qs - sp.Rational(1, 2)) == 0, Qs)
# C5: finite write time (linear ramp of duration tau)
tau = sp.symbols('tau', positive=True)
u = sp.symbols('u', real=True)
# residual ringing amplitude after a linear ramp, relative to q*: |sin(w tau/2)/(w tau/2)|
f_ = sp.Function('f')
y = sp.Function('y')
ode = sp.Eq(y(t).diff(t, 2) + w_**2 * y(t), w_**2 * qstar * t / tau)
ysol = sp.dsolve(ode, ics={y(0): 0, y(t).diff(t).subs(t, 0): 0}).rhs
yT = ysol.subs(t, tau); vT = sp.diff(ysol, t).subs(t, tau)
ring_after = sp.simplify((yT - qstar)**2 + (vT / w_)**2)
pts = [(1.3, 0.9), (0.7, 4.0), (2.0, 0.05)]
devs = [abs(float((ring_after / qstar**2).subs({w_: W_, tau: T_})) - (math.sin(W_ * T_ / 2) / (W_ * T_ / 2))**2) for W_, T_ in pts]
ratio_num = float((ring_after / qstar**2).subs({w_: 1.3, tau: 0.9}))
sinc2 = (math.sin(1.3 * 0.9 / 2) / (1.3 * 0.9 / 2))**2
check("C5", "finite ramp of duration tau: ring^2/q*^2 = sinc^2(omega tau/2) <= 1, so Q = (1 + sinc^2)/3 <= 2/3; the sudden write is the unique maximum",
      max(devs) < 1e-12, "3 (omega, tau) points; e.g. ratio %.12f vs sinc^2 %.12f" % (ratio_num, sinc2))
# C6: thermal (non-blank) start adds 2 k_B T/k to the plane only
kT = sp.symbols('kT', positive=True)
ring_T = qstar**2 + 2 * kT / k_
QT = sp.simplify(Qratio(ring_T / qstar**2))
check("C6", "thermal start (classical): <ring^2> = q*^2 + 2kT/k, so Q_T = 2/3 + 2kT/(3 k q*^2) > 2/3; equality needs the blank start (forward Step 1 condition)",
      sp.simplify(QT - sp.Rational(2, 3) - 2 * kT / (3 * k_ * qstar**2)) == 0, QT)
# C7: per-quadrature energies of the sudden write: axis carries a full pair
Uaxis = sp.Rational(1, 2) * k_ * qstar**2
Kavg = sp.simplify(sp.integrate(pt**2 / (2 * m_), (t, 0, 2 * sp.pi / w_)) * w_ / (2 * sp.pi))
Vavg = sp.simplify(sp.integrate(sp.Rational(1, 2) * k_ * (qt - qstar)**2, (t, 0, 2 * sp.pi / w_)) * w_ / (2 * sp.pi))
check("C7", "sudden write: held (axis) = U; ringing quadratures carry U/2 each on average: the axis equals one full canonical pair, i.e. twice one quadrature",
      sp.simplify(Kavg - Uaxis / 2) == 0 and sp.simplify(Vavg - Uaxis / 2) == 0)
# C8: n phases: equality gives r = sqrt2 for any n, Q_n = 2/n (book K11)
nn = sp.symbols('n', positive=True, integer=True)
check("C8", "same write read on n phases: |s_par|^2 = n x^2, |s_perp|^2 = (n/2) x^2 r^2; equality -> r = sqrt2, Q_n = 2/n (book K11)",
      all(abs(sum((1 + math.sqrt(2) * math.cos(0.3 + 2 * math.pi * k / N))**2 for k in range(N)) /
              sum(1 + math.sqrt(2) * math.cos(0.3 + 2 * math.pi * k / N) for k in range(N))**2 - 2 / N) < 1e-12
          for N in [3, 4, 5, 7]))

# C9: the one-bit form of Koide is the sudden write's held fraction
Fw = sp.simplify(held / (held + spent))
check("C9", "in candidate (ii) the fidelity F = cos^2 theta = |s_par|^2/|s|^2 = held/(held+spent) = 1/2, so D_1/2(p||u) = -ln F = ln 2: the one bit is held = spent",
      Fw == sp.Rational(1, 2) and sp.simplify(-sp.log(Fw) - sp.log(2)) == 0, Fw)

# ---------------------------------------------------------------------------
# D. candidate (iii): equipartition per quadrature
# ---------------------------------------------------------------------------
check("D1", "equipartition axis : alpha : beta = 1 : 1 : 1 -> |s_perp|^2 = 2|s_par|^2 -> Q = 1 (NO)", Qratio(2) == 1)
check("D2", "classical Gibbs: axis coordinate kT/2, plane pair kT -> same Q = 1 (NO); the sudden write violates per-quadrature equipartition (C7)",
      Qratio(sp.Rational(1, 1) / sp.Rational(1, 2)) == 1)

# ---------------------------------------------------------------------------
# E. neighbours
# ---------------------------------------------------------------------------
P, O, N_, mu2, lmb, eta, kap = sp.symbols('P O N mu2 lambda eta kappa', real=True)
V = mu2 * (P + O) + sp.Rational(1, 2) * lmb * (P + O)**2 + eta * P * O
statE = sp.solve([sp.diff(V, P), sp.diff(V, O)], [P, O], dict=True)
check("E1", "Koide 1990 / hep-ph/0506247 eq. 3.1: V = mu^2 N + lambda N^2/2 + eta P O (P = |v_sigma|^2, O = |v_pi|^2+|v_eta|^2): stationary -> P = O",
      len(statE) == 1 and sp.simplify(statE[0][P] - statE[0][O]) == 0, statE)
Hs = sp.hessian(V, (P, O)); evH = list(Hs.eigenvals().keys())
check("E2", "Hessian in (P,O): eigenvalues 2 lambda + eta (along N) and -eta (along the split): the equal split is a minimum only for eta < 0, i.e. maximum of P*O at fixed N",
      set(sp.simplify(e) for e in evH) == {2 * lmb + eta, -eta}, evH)
vv = sp.symbols('v0:3', positive=True)
Pv = sum(vv)**2 / 3; Ov = sum(x**2 for x in vv) - Pv
QE = sp.simplify(sum(x**2 for x in vv) / sum(vv)**2)
check("E3", "with v_i = sqrt m_i, P = |s_par|^2 and O = |s_perp|^2; P = O means sum v^2 = 2P, so Q = 2P/(3P) = 2/3",
      sp.simplify(Pv + Ov - sum(x**2 for x in vv)) == 0 and sp.simplify(2 * Pv / sum(vv)**2) == sp.Rational(2, 3))
V2 = V + kap * P**2
s2 = sp.solve([sp.diff(V2, P), sp.diff(V2, O)], [P, O], dict=True)
check("E4", "any extra singlet-only term kappa P^2 moves the stationary split to O/P = 1 - 2 kappa/eta: Koide's form must be exact ('given by hand', hep-ph/0506247 p.8)",
      sp.simplify(s2[0][O] / s2[0][P] - (1 - 2 * kap / eta)) == 0)
# Hubner 2026: N_eff = 1/Q ; extension minimum Q0/(1+Q0)
Q0 = sp.Rational(2, 3)
rr = sp.symbols('r', positive=True)
R1, R2 = sp.symbols('R1 R2', positive=True)
Qext = (R2 + rr**2) / (R1 + rr)**2
rstar = sp.solve(sp.diff(Qext, rr), rr)
Qmin = sp.simplify(Qext.subs(rr, rstar[0]))
check("E5", "Hubner 2026 Theorem 1 reproduced: r* = R2/R1, Q_min = Q0/(1+Q0); for Q0 = 2/3, N_eff = 3/2 -> 5/2, Q_min = 2/5",
      sp.simplify(rstar[0] - R2 / R1) == 0 and sp.simplify(Qmin - (R2 / R1**2) / (1 + R2 / R1**2)) == 0 and Q0 / (1 + Q0) == sp.Rational(2, 5))
check("E6", "N_eff = 1/Q in each YES candidate: sudden write 3|s_par|^2/|s|^2 = 3/2; stretched spin j: 3j/(j+1) = 3/2 at j = 1",
      sp.Rational(3, 2) == 1 / sp.Rational(2, 3) and sp.simplify(3 * jj / (jj + 1)).subs(jj, 1) == sp.Rational(3, 2))

# ---------------------------------------------------------------------------
# F. measured (book tab:ko:masses, p4_15a:28-35)
# ---------------------------------------------------------------------------
me, mmu = 0.51099895000, 105.6583755
out_F = {}
for lab, mt, dmt in [("2024", 1776.93, 0.09), ("2022", 1776.86, 0.12)]:
    sm = np.sqrt([me, mmu, mt]); Qm = (sm**2).sum() / sm.sum()**2
    h = 1e-4; sm2 = np.sqrt([me, mmu, mt + h]); dQ = ((sm2**2).sum() / sm2.sum()**2 - Qm) / h
    sp2 = sm.sum()**2 / 3; sq2 = (sm**2).sum() - sp2
    out_F[lab] = {"Q": Qm, "sigma_Q": abs(dQ) * dmt, "perp2/par2": sq2 / sp2, "pull_sigma": (Qm - 2 / 3) / (abs(dQ) * dmt)}
check("F1", "measured |s_perp|^2/|s_par|^2 = 3Q - 1 (2024 tau): Q = 0.66666446 +- 0.0000051 as in the book table; 2/3 within 0.43 sigma",
      abs(out_F["2024"]["Q"] - 0.66666446) < 2e-8 and abs(out_F["2024"]["sigma_Q"] - 5.08e-6) < 2e-7, out_F)
# implied ramp parameter if the deficit were read as a finite write time (C5) - consistent with zero
defic = 1 - out_F["2024"]["perp2/par2"]
uimp = math.sqrt(3 * max(defic, 0))
INFO = {"F2_not_a_check": "if the 2024 deficit 1-(3Q-1) = %.3e were read through C5, omega*tau/2 ~ %.4f; the deficit is 0.43 sigma from zero, so no write time is resolved" % (defic, uimp)}
print("INFO", INFO)

npass = sum(r_["pass"] for r_ in R); nfail = len(R) - npass
json.dump({"checks": R, "passed": npass, "failed": nfail, "measured": out_F,
           "B4_delta_max": float(dmax), "info": INFO}, open("koide_split_checks_output.json", "w"), indent=1, default=str)
print("passed %d failed %d" % (npass, nfail))
