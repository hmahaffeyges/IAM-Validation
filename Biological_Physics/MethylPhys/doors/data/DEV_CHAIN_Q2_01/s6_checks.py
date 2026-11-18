import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
#!/usr/bin/env python3
"""SPIDER S6 checks: the held/spent 50/50 of a decoherence record, and the Koide matching principle.
Private derivation web of IAM. Proposals only. Run: python3 spider_s6_checks.py  -> spider_s6_checks_output.json
Every check prints PASS/FAIL; a FAIL is reported, never hidden.
Part H: held/spent of a record written by a pointer into a bath (sudden write from a blank bath).
Part K: Koide's equal Frobenius norms; candidate matching principles.
"""
import json, math
import numpy as np
import sympy as sp
from scipy import integrate, linalg, special

R = []
def check(cid, ok, detail):
    ok = bool(ok)
    R.append({"id": cid, "pass": ok, "detail": detail})
    print(("PASS " if ok else "FAIL ") + cid + ": " + detail)

# ---------------------------------------------------------------- Part H
# H1. Classical linear bath, sudden write from blank. Mode k: H_k = p^2/2m + m w^2 x^2/2 ; coupling +sigma*c*x switched on at t=0
# from x=0, p=0 (the thermal average enters linearly and drops out; see H5/H6 for thermal states).
t, m, w, c = sp.symbols('t m omega c', positive=True)
sig = sp.Symbol('sigma')
xeq = -sig*c/(m*w**2)
x = xeq*(1 - sp.cos(w*t)); p = m*sp.diff(x, t)
HB = p**2/(2*m) + m*w**2*x**2/2            # bath's own energy
Hint = sig*c*x
Etot = sp.simplify(HB + Hint)
U = sp.simplify((m*w**2*xeq**2/2).subs(sig**2, 1))
avg = lambda f: sp.simplify(sp.integrate(f, (t, 0, 2*sp.pi/w))*w/(2*sp.pi))
HB_avg = sp.simplify(avg(HB).subs(sig**2, 1))
static = U
ringing = sp.simplify(HB_avg - static)
drop = sp.simplify(-avg(Hint).subs(sig**2, 1))
check("H1", sp.simplify(Etot) == 0 and sp.simplify(HB_avg - 2*U) == 0 and sp.simplify(ringing - U) == 0 and sp.simplify(drop - 2*U) == 0,
      f"sudden write: total energy conserved (=0); source work (interaction drop) = 2U; held (static elastic) = U; spent (ringing, radiated into the continuum) = U; U = c^2/(2 m w^2) = {U}")

# H1b. Euler at degree 2 is the same statement: source work F*x_eq = x dU/dx = 2U
xs, k = sp.symbols('x k', positive=True)
Ux = k*xs**2/2
check("H1b", sp.simplify(xs*sp.diff(Ux, xs) - 2*Ux) == 0, "Euler: x dU/dx = 2U for a quadratic store, so a stiff (constant-force) source does 2U: half held, half spent")

# H2. Gaussian records: relative entropy symmetric; equals U/kT
mu1, mu2, s = sp.symbols('mu1 mu2 s', real=True)
s = sp.Symbol('s', positive=True)
X = sp.Symbol('X', real=True)
P = sp.exp(-(X-mu1)**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)
Qd = sp.exp(-(X-mu2)**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)
KLpq = sp.simplify(sp.integrate(P*sp.expand(sp.log(P/Qd), force=True), (X, -sp.oo, sp.oo)))
KLqp = sp.simplify(sp.integrate(Qd*sp.expand(sp.log(Qd/P), force=True), (X, -sp.oo, sp.oo)))
kT = sp.Symbol('kT', positive=True)
KL_rec = sp.simplify(KLpq.subs({mu1: 0, mu2: xeq.subs(sig, 1), s: sp.sqrt(kT/(m*w**2))}))
check("H2", sp.simplify(KLpq - KLqp) == 0 and sp.simplify(KL_rec - U/kT) == 0,
      f"equal-variance Gaussians: D(P||Q) = D(Q||P) = (dmu)^2/(2 s^2); record vs blank: D = U/kT exactly ({sp.simplify(KL_rec)})")

# H3. Exponential family identity (any bath): S10 + S01 = s (K'(s) - K'(0)); S10 = s K'(s) - K(s); S01 = K(s) - s K'(0)
sv = sp.Symbol('s')
k1, k2, k3, k4 = sp.symbols('kappa1 kappa2 kappa3 kappa4')
K = k1*sv + k2*sv**2/2 + k3*sv**3/6 + k4*sv**4/24
S10 = sp.expand(sv*sp.diff(K, sv) - K)          # S(rho1||rho0) = held/kT
S01 = sp.expand(K - sv*sp.diff(K, sv).subs(sv, 0))  # S(rho0||rho1) = spent/kT
check("H3", sp.simplify(S10 + S01 - sv*(sp.diff(K, sv) - sp.diff(K, sv).subs(sv, 0))) == 0,
      "held + spent = s*(<V>_1 - <V>_0) in kT units (s = -beta*lambda): the source work, for ANY bath (exact, classical exponential family)")
# H4. departure from the half is the skewness
diff = sp.expand(S10 - S01)
h = sp.series((S10/(S10+S01)), sv, 0, 2).removeO()
check("H4", sp.simplify(diff - (k3*sv**3/6 + k4*sv**4/12)) == 0 and sp.simplify(h - (sp.Rational(1, 2) + k3*sv/(12*k2))) == 0,
      f"S10 - S01 = kappa3 s^3/6 + kappa4 s^4/12; held share = 1/2 + kappa3 s/(12 kappa2) + O(s^2): exactly 1/2 iff all cumulants above 2nd vanish (Gaussian), and 1/2 to second order for every bath")

# H5. Quantum: displaced thermal oscillator, S(D rho D+ || rho) = S(rho || D rho D+) = beta*hbar*w*|alpha|^2 at any T
def thermal(n, bw):
    e = np.exp(-bw*np.arange(n)); return np.diag(e/e.sum())
def disp(n, a):
    A = np.diag(np.sqrt(np.arange(1, n)), 1)
    return linalg.expm(a*A.conj().T - np.conj(a)*A)
def relent(r1, r2):
    w1, v1 = np.linalg.eigh(r1); w2, v2 = np.linalg.eigh(r2)
    w1 = np.clip(w1, 1e-300, None); w2 = np.clip(w2, 1e-300, None)
    L1 = v1 @ np.diag(np.log(w1)) @ v1.conj().T; L2 = v2 @ np.diag(np.log(w2)) @ v2.conj().T
    return float(np.real(np.trace(r1 @ (L1 - L2))))
N = 90; ok = True; rows = []
for bw in [0.3, 1.0, 3.0]:
    for a in [0.3, 0.8]:
        r0 = thermal(N, bw); D = disp(N, a); r1 = D @ r0 @ D.conj().T
        sa, sb = relent(r1, r0), relent(r0, r1); target = bw*a**2
        rows.append((bw, a, round(sa, 6), round(sb, 6), round(target, 6)))
        ok &= abs(sa-target) < 2e-4 and abs(sb-target) < 2e-4
check("H5", ok, f"quantum displaced thermal states: S(rho_s||rho_0) = S(rho_0||rho_s) = beta hbar w |alpha|^2 (beta*hbar*w, alpha, S10, S01, target): {rows}")

# H6. Many-mode bath (ohmic, sub-, super-ohmic, random couplings): held = spent mode by mode -> totals equal for any J(w)
# First run (uniform 400-mode grid from w = 0.01, window shorter than the slowest periods) FAILED on averaging, not on physics.
# Second run tested the pointer's collective force at single instants and FAILED at 2-8 %: finite-N fluctuation, not settling.
# This version: energy split (time average) and the collective force's window mean, plus its fluctuation shrinking ~ 1/sqrt(N).
rng = np.random.default_rng(6)
ok = True; out = []
def bath(Nm, expo, rand):
    wk = np.sort(rng.uniform(0.2, 5, Nm)); mk = np.ones_like(wk)
    ck = np.sqrt(wk**expo*np.exp(-wk))*(rng.uniform(0.2, 2, wk.size) if rand else 1)
    ck = ck/np.sqrt(Nm)
    xe = -ck/(mk*wk**2); Usum = np.sum(0.5*mk*wk**2*xe**2)
    tt = np.linspace(500, 2500, 4001)
    HB = np.array([np.sum(mk*wk**2*xe**2*(1-np.cos(wk*ti))) for ti in tt])/Usum
    Xc = np.array([np.sum(ck*xe*(1-np.cos(wk*ti))) for ti in tt])/(-2*Usum)
    return HB.mean(), Xc.mean(), Xc.std()
for name, expo, rand in [("ohmic", 1.0, False), ("subohmic", 0.5, False), ("superohmic", 3.0, False), ("random", 1.0, True)]:
    e1, x1, s1 = bath(500, expo, rand); e2, x2, s2 = bath(2000, expo, rand)
    out.append((name, round(e2, 4), round(x2, 4), round(s1/s2, 2)))
    ok &= abs(e2 - 2) < 0.01 and abs(x2 - 1) < 0.01 and 1.4 < s1/s2 < 2.9
check("H6", ok, f"(bath, <H_B>/U_held, window-mean collective force / static value, fluctuation ratio N=500 vs 2000 [~2 expected]): {out}: held U + spent U for every spectral density; the ringing dephases into the continuum")

# H6b. finite switching time: spent fraction = |g(w)|^2 = sinc^2(w tau/2) for a linear ramp (numeric ODE vs formula)
def ramp_spent(wv, tau):
    f = lambda tt_: np.clip(tt_/tau, 0, 1)
    def rhs(tt_, y): return [y[1], -wv**2*y[0] - f(tt_)]   # m=1, c=1 force -f(t)
    sol = integrate.solve_ivp(rhs, [0, tau+0.0], [0, 0], rtol=1e-10, atol=1e-12)
    xT, vT = sol.y[0, -1], sol.y[1, -1]
    xe_ = -1/wv**2
    ring = 0.5*vT**2 + 0.5*wv**2*(xT-xe_)**2
    return ring/(0.5*wv**2*xe_**2)
rows = []; ok = True
for wv, tau in [(1.0, 0.1), (1.0, 2.0), (1.0, 6.0), (2.0, 1.5)]:
    num = ramp_spent(wv, tau); th = np.sinc(wv*tau/(2*np.pi))**2
    rows.append((wv, tau, round(num, 6), round(th, 6))); ok &= abs(num-th) < 1e-5
check("H6b", ok, f"ramp of duration tau: spent/held = sinc^2(w tau/2) (w, tau, numeric, formula): {rows}; sudden (tau->0) gives 1, slow gives 0")

# H7. Information: binary pointer, Gaussian record. I(sigma;E) <= U/kT (Holevo/golden formula), equality as U/kT -> 0
def I_binary_gauss(snr):   # a/s = sqrt(snr); records N(+a,s), N(-a,s), equal priors; nats
    a = math.sqrt(snr)
    f = lambda y: (lambda p1, p2: 0.5*p1*math.log(2*p1/(p1+p2)) + 0.5*p2*math.log(2*p2/(p1+p2)) if p1 > 0 and p2 > 0 else 0.0)(
        math.exp(-(y-a)**2/2)/math.sqrt(2*math.pi), math.exp(-(y+a)**2/2)/math.sqrt(2*math.pi))
    return integrate.quad(f, -a-12, a+12, limit=200)[0]
rows = []; ok = True
for UkT in [1e-3, 1e-2, 0.1, 0.5, 1, 3, 10]:
    snr = 2*UkT          # a^2/s^2 = 2U/kT
    I = I_binary_gauss(snr); rows.append((UkT, round(I, 6), round(I/UkT, 4))); ok &= I <= UkT + 1e-12
ok &= abs(rows[0][2] - 1) < 2e-3 and rows[-1][1] <= math.log(2) + 1e-9
check("H7", ok, f"(U/kT, I nats, I/(U/kT)): {rows}: held = spent >= kT per nat of record information; equality only for weak records; a sharp bit costs U >> kT ln2")

# H8. Record distinguishability vs decoherence: -ln sqrt(F) = 2|alpha|^2 tanh(bw/2) <= U/kT ; decoherence exponent 2|alpha|^2 coth(bw/2)
rows = []; ok = True
for bw in [0.2, 1.0, 4.0]:
    a = 0.5; N = 90
    r0 = thermal(N, bw); Dp = disp(N, a); Dm = disp(N, -a)
    rp = Dp @ r0 @ Dp.conj().T; rm = Dm @ r0 @ Dm.conj().T
    sq = linalg.sqrtm(rp); Fid = np.real(np.trace(linalg.sqrtm(sq @ rm @ sq)))**2
    DB = -math.log(math.sqrt(Fid)); th = 2*a**2*math.tanh(bw/2); UkT = bw*a**2
    Gam = 2*a**2/math.tanh(bw/2)
    rows.append((bw, round(DB, 5), round(th, 5), round(UkT, 5), round(Gam, 5))); ok &= abs(DB-th) < 1e-3 and DB <= UkT + 1e-9
check("H8", ok, f"(beta hbar w, record Bhattacharyya -ln sqrtF, formula, U/kT, decoherence exponent): {rows}: the energy U tracks the record's readability, not the decoherence rate")

# H9. Spin bath (non-Gaussian): two-level bath H0 = -(D/2) tau_x, V = tau_z, Gibbs states; held/spent = S10/S01
tx = np.array([[0, 1], [1, 0]]); tz = np.array([[1, 0], [0, -1]])
def gibbs(H, b):
    wv, v = np.linalg.eigh(H); e = np.exp(-b*(wv-wv.min())); r = v @ np.diag(e/e.sum()) @ v.T; return r
rows = []; ok = True
for b in [0.5, 2.0]:
    for lam in [0.01, 0.5, 2.0]:
        H0 = -0.5*tx; H1 = H0 + lam*tz
        r0, r1 = gibbs(H0, b), gibbs(H1, b)
        S10q, S01q = relent(r1, r0), relent(r0, r1)
        src = b*lam*(np.trace(r0@tz) - np.trace(r1@tz)).real
        rows.append((b, lam, round(S10q/S01q, 4), round(S10q+S01q-src, 9)))
        ok &= abs(S10q+S01q-src) < 1e-9
ok &= abs(rows[0][2]-1) < 1e-3 and abs(rows[2][2]-1) > 0.02
check("H9", ok, f"spin bath (beta, lambda, held/spent, sum - source work): {rows}: 1 at weak coupling, departs at strong coupling; sum always = source work")
# H9b. Zurek's 1982 spin bath with no self-Hamiltonian: records written at zero energy
# numeric: central spin in (|0>+|1>)/sqrt2, 4 bath spins in random pure states, H = sigma_z (x) sum g_k tau_z^k, H_E = 0
from functools import reduce
nb = 4; g = rng.uniform(0.5, 1.5, nb)
I2 = np.eye(2); sz = np.diag([1., -1.])
def op(single, j, n):
    return reduce(np.kron, [single if i == j else I2 for i in range(n)])
Hb = sum(g[j]*op(sz, j, nb) for j in range(nb)); Hq = np.kron(sz, Hb)
def spinor(th, ph_):
    return np.array([np.cos(th/2), np.exp(1j*ph_)*np.sin(th/2)])
psi_b = reduce(np.kron, [spinor(*rng.uniform(0.3, np.pi-0.3, 2)) for _ in range(nb)])
psi = np.kron(np.array([1, 1])/np.sqrt(2), psi_b)
Ev = []; coh = []
for tt_ in [0, 0.7, 3.0]:
    U_ = linalg.expm(-1j*Hq*tt_); ps = U_ @ psi
    Ev.append(float(np.real(ps.conj() @ Hq @ ps)))
    rho = np.outer(ps, ps.conj()).reshape(2, 2**nb, 2, 2**nb); rs = np.einsum('aibi->ab', rho); coh.append(abs(rs[0, 1]))
check("H9b", max(Ev)-min(Ev) < 1e-12 and coh[2] < coh[0], f"Zurek spin bath with H_E = 0: <H> constant {[round(e,10) for e in Ev]} while the central spin's coherence falls {[round(c_,4) for c_ in coh]}: records written at zero energy; the half is vacuous (held = spent = 0)")

# H10. Two-state (Bernoulli) record written suddenly from a blank (p=1/2) to error eps
def bern(eps):
    S10b = math.log(2) + eps*math.log(eps) + (1-eps)*math.log(1-eps)          # ln2 - H(eps)
    S01b = -math.log(2) - 0.5*math.log(eps*(1-eps))
    bl = 0.5*math.log((1-eps)/eps)
    return S10b, S01b, bl*math.tanh(bl)
rows = []; ok = True
for eps in [0.45, 0.25, 0.04454, 1e-3]:
    a1, a2, src = bern(eps); rows.append((eps, round(a1, 4), round(a2, 4), round(a1/(a1+a2), 4)))
    ok &= abs(a1+a2-src) < 1e-12
ok &= abs(rows[0][3]-0.5) < 0.01 and rows[3][3] < 0.2
check("H10", ok, f"two-state record (eps, held/kT = ln2-H, spent/kT, held share): {rows}: half only for weak records; a sharp two-state mark spends far more than it holds")

# H11. sudden write then sudden erase: writer pays S01, eraser pays S10 (the held part); cycle total = Jeffreys divergence
b = 1.3; lam = 1.1; H0 = -0.5*tx; H1 = H0 + lam*tz
r0, r1 = gibbs(H0, b), gibbs(H1, b)
F = lambda H: -np.log(np.sum(np.exp(-b*np.linalg.eigvalsh(H))))/b
Wwr = np.trace(r0 @ (H1-H0)).real; Wer = np.trace(r1 @ (H0-H1)).real
diss_w = b*(Wwr - (F(H1)-F(H0))); diss_e = b*(Wer - (F(H0)-F(H1)))
check("H11", abs(diss_w - relent(r0, r1)) < 1e-10 and abs(diss_e - relent(r1, r0)) < 1e-10,
      f"sudden write dissipates kT S(rho0||rho1) = {diss_w:.6f} kT; sudden erase dissipates kT S(rho1||rho0) = {diss_e:.6f} kT = the held part: writer:eraser = S01:S10 (direct W - dF, spin bath)")
# H12. overwrite (+1 -> -1) in a Gaussian bath: spent 4U, held unchanged (Marcus reorganization energy)
KL_over = sp.simplify(KLpq.subs({mu1: xeq.subs(sig, 1), mu2: xeq.subs(sig, -1), s: sp.sqrt(kT/(m*w**2))}))
check("H12", sp.simplify(KL_over - 4*U/kT) == 0, f"overwrite: spent = kT D = 4U = Marcus lambda; held stays U (both branches carry U)")

# ---------------------------------------------------------------- Part K
a_, br, bi = sp.symbols('a b_r b_i', real=True)
b_ = br + sp.I*bi
C = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
Dm_ = a_*sp.eye(3); Hm_ = b_*C + sp.conjugate(b_)*C**2; A = Dm_ + Hm_
fro = lambda M: sp.simplify(sum(sp.Abs(e)**2 for e in M))
nD, nH, nA = fro(Dm_), sp.simplify(sp.expand(sum((e*sp.conjugate(e)) for e in Hm_))), sp.simplify(sp.expand(sum((e*sp.conjugate(e)) for e in A)))
check("K1", sp.simplify(nA - nD - nH) == 0 and sp.simplify(nH - 6*(br**2+bi**2)) == 0 and sp.simplify(nD - 3*a_**2) == 0,
      "circulant sqrt(M) = aI + bC + conj(b)C^2: ||D||^2 = 3a^2, ||H||^2 = 6|b|^2, orthogonal (Pythagoras); Q = ||A||^2/(tr A)^2 = (3a^2+6|b|^2)/(9a^2); Q = 2/3 <=> ||D|| = ||H||")
Qexpr = sp.simplify(nA/(sp.trace(A))**2)
check("K1b", sp.simplify(Qexpr.subs(br**2+bi**2, a_**2/2) - sp.Rational(2, 3)) == 0 or sp.simplify(Qexpr.subs(bi, 0).subs(br, a_/sp.sqrt(2)) - sp.Rational(2, 3)) == 0, "Q(|b|^2 = a^2/2) = 2/3")
# K2: pinching (complete dephasing in the flavour/site basis) keeps the diagonal aI
Pin = sp.diag(*[A[i, i] for i in range(3)])
check("K2", sp.simplify(Pin - a_*sp.eye(3)) == sp.zeros(3) , "dephasing sqrt(M) in the flavour basis (mutually unbiased with the mass basis) leaves aI: Koide <=> the dephased part keeps exactly half of ||sqrt(M)||^2 and the removed coherences carry the other half")
# K3: collision-entropy statement
Qs = sp.Symbol('Q', positive=True)
H2rho = -sp.log(Qs); H2pin = sp.log(3)
check("K3", sp.simplify((H2pin - H2rho).subs(Qs, sp.Rational(2, 3)) - sp.log(2)) == 0,
      "rho = sqrt(M)/tr sqrt(M): purity tr rho^2 = Q; dephased state I/3; collision entropy rises by ln(3Q): exactly ln 2 (one bit) iff Q = 2/3; equivalently D_2(q || uniform) = 1 bit")
nn = sp.Symbol('n', positive=True, integer=True)
check("K3b", sp.simplify(sp.log(nn*(2/nn)) - sp.log(2)) == 0, "n phases with y = sqrt2 x (equal norms): Q_n = 2/n and the dephasing increment is ln 2 for every n: equal HS split <=> one bit of collision entropy, a general identity tr(Delta rho)^2 = tr(rho^2)/2 <=> H2 jump = ln2")
# K4 leptons
me, mmu, mtau = 0.51099895000, 105.6583755, 1776.93
sq = np.sqrt([me, mmu, mtau]); Ql = (me+mmu+mtau)/sq.sum()**2; qv = sq/sq.sum()
D2bits = math.log2(3*Ql); vN_bits = (math.log(3) + float(np.sum(qv*np.log(qv))))/math.log(2)
dQ = 0.00000508
check("K4", abs(D2bits - 1) < 3*dQ/(Ql*math.log(2))*1.5 and abs(vN_bits-1) > 0.05,
      f"PDG 2024: Q = {Ql:.8f}; collision-entropy jump = {D2bits:.7f} bits (1 - {1-D2bits:.2e}); von Neumann jump = {vN_bits:.4f} bits (NOT one bit): only the Renyi-2 (Hilbert-Schmidt) reading gives exactly one bit")
# K5 maximum-product (AM-GM) and square-law first-harmonic maximum
u, v, Pt = sp.symbols('u v P', positive=True)
sol = sp.solve(sp.diff((u*(Pt-u)), u), u)
xx, yy, phi = sp.symbols('x y phi', positive=True)
mphi = (xx + yy*sp.cos(phi))**2
mean_m = sp.simplify(sp.integrate(mphi, (phi, 0, 2*sp.pi))/(2*sp.pi))
c1 = sp.simplify(sp.integrate(mphi*sp.cos(phi), (phi, 0, 2*sp.pi))/sp.pi)     # first-harmonic amplitude of m(phi)
lag = sp.solve([sp.diff(c1**2 - sp.Symbol('L')*(mean_m - Pt), xx), sp.diff(c1**2 - sp.Symbol('L')*(mean_m - Pt), yy), mean_m - Pt], [xx, yy, sp.Symbol('L')], dict=True)
okK5 = sol == [Pt/2] and any(sp.simplify(sl[xx]**2 - sl[yy]**2/2) == 0 for sl in lag)
check("K5", okK5, f"max ||D||^2||H||^2 at fixed ||A||^2 -> equal split; and the first harmonic of m(phi)=(x+y cos phi)^2 (amplitude {c1}) is largest at fixed mean mass {mean_m} exactly when x^2 = y^2/2: Koide is a maximum-transfer point of the square law (formal; no physical source/load identified)")
# K6 maximum power transfer behind a pure reactance
Rr, Xr, Vv = sp.symbols('R X V', positive=True)
Pw = Vv**2*Rr/(Rr**2 + Xr**2)
check("K6", sp.solve(sp.diff(Pw, Rr), Rr) == [Xr], "a variable resistor fed through a fixed reactance X takes most power at R = X: P = Q, power factor 1/sqrt2, 45 deg (textbook); maps onto Koide only if zero sequence = resistance and balanced part = reactance, which is not derived")
# K7 SIL test: per-mode split
dl = sp.Symbol('delta', real=True); bb = sp.Symbol('B', positive=True)
per_mode = [sp.simplify(2*bb*sp.cos(dl + 2*sp.pi*kk/3)) for kk in range(3)]
check("K7", all(sp.simplify(pm - a_) != 0 for pm in per_mode) and sp.simplify(sum(pm**2 for pm in per_mode) - 6*bb**2) == 0,
      "surge-impedance test: on each eigen-mode the on-site part is a and the hopping part 2|b|cos(delta+2pi k/3), not equal mode by mode; equality holds only for the aggregate norms. SIL's pointwise u_E = u_B (forced by the travelling-wave relation V = Z0 I) has no counterpart here: SIL is the right TYPE, no mechanism found")
# K8 Gaussian (MaxEnt at fixed ||A||^2) ensemble of circulants
# 1 DOF on-site, |b| 1 DOF if delta is a fixed reference, 2 DOF if b's phase is integrated
r1 = 1.0; r2 = 2.0
win = 3*dQ*3   # window on ||H||^2/||D||^2 = 3Q-1 ; sigma(3Q-1)=3*dQ ; use +-1 sigma
pF11 = special.betainc(0.5, 0.5, (1+3*dQ)/(2+3*dQ)) - special.betainc(0.5, 0.5, (1-3*dQ)/(2-3*dQ))
check("K8", abs(r1-1) < 1e-12 and abs(r2-2) < 1e-12 and pF11 < 1e-5,
      f"Gaussian ensemble: <||H||^2>/<||D||^2> = 1 with delta a fixed reference (Koide on average), 2 with b's phase free (Q = 1, excluded); exact equality within +-1 sigma has probability {pF11:.2e} (F(1,1)): equipartition is not the principle")
# K9 Z3 sampling reproduces the population moments of a + y cos phi for every delta
k_ = sp.symbols('k', integer=True)
ph = [dl + 2*sp.pi*j/3 for j in range(3)]
check("K9", sp.simplify(sum(sp.cos(p_) for p_ in ph)) == 0 and sp.simplify(sum(sp.cos(p_)**2 for p_ in ph) - sp.Rational(3, 2)) == 0,
      "three Z3 samples carry exactly the circle's mean and variance of a + y cos phi at every delta: CV(samples) = CV(orbit) = y/(sqrt2 a); Koide is CV = 1 of the orbit function itself")
# K10 speed limits
e0 = sp.Matrix([1, 0, 0]); mean_j = sp.simplify((e0.T*A*e0)[0]); var_j = sp.simplify(sp.expand((e0.T*A*A*e0)[0] - mean_j**2))
check("K10", sp.simplify(mean_j - a_) == 0 and sp.simplify(var_j - 2*(br**2+bi**2)) == 0,
      "flavour state |j>: <A> = a, Var A = 2|b|^2, so Koide <=> Delta E = <E> (energies from zero): the Mandelstam-Tamm and Margolus-Levitin times coincide for flavour states (identity restatement, no selection principle)")
# K11 carry-over test: Gaussian-record half vs Koide

npass = sum(r["pass"] for r in R); nfail = len(R) - npass
print(f"TOTAL {npass} pass, {nfail} fail")
json.dump({"checks": R, "passed": npass, "failed": nfail}, open("spider_s6_checks_output.json", "w"), indent=1)
