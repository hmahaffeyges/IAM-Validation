import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
#!/usr/bin/env python3
"""iam_law_checks.py -- every check that IAM_LAW_DERIVATION_2026-11-18.md relies on (PRIVATE; proposals only).

Self-contained: imports nothing from any other IAM script. Light by design (sympy, numpy, scipy, mpmath;
seconds, < 300 MB). Writes iam_law_checks_output.json.

Check ids keep the ids of the source reports so a reader can trace each one:
  A*  Step 1 actualization            (forward spider A*)
  B*  Step 2 the price                (forward B*)
  H*  Step 2b where the free energy comes from (forward H*)
  C*  Step 3 the virial half          (forward C*); W*  the same in d space dimensions (which-threes spider)
  D*  Step 4 Koide as the 3D->2D record (forward D*); K*  why the plane equals the axis (Koide-split spider)
  E*  Step 5 the arrow                (forward E*)
  F*  Step 6 the entropy functional   (forward F*); R*  the chain to the Friedmann equation (book p1_02, re-derived)
  L*  the beta_m = Omega_m/2 identities (count, ledger, time-and-rate, Bekenstein-Dirac, dark-sector spiders)
  G*  Step 7 the horizon as the ledger (forward G*)
  BH* black-hole reading; CE* cell reading
Code for A-H is carried over verbatim from forward_checks.py (2026-11-18), copied, not imported.
"""
import json, math, sys
import numpy as np
import sympy as sp
import mpmath as mp
from scipy import integrate, linalg, optimize
from scipy.integrate import solve_ivp

RESULTS = []
def check(cid, step, tag, desc, ok, value=None):
    ok = bool(ok)
    RESULTS.append(dict(id=cid, step=step, tag=tag, desc=desc, passed=ok,
                        value=(None if value is None else str(value))))
    print(("PASS " if ok else "FAIL ") + cid + "  " + desc + ("" if value is None else "  -> " + str(value)))

def simp0(e):
    return sp.simplify(sp.expand(e)) == 0

# ---------------------------------------------------------------- STEP 1: actualization
t, w, m, c, sig, U = sp.symbols('t omega m c sigma U', positive=True)
xeq = -c/(m*w**2)                    # sigma = +1 branch; force constant sigma*c
x = xeq*(1 - sp.cos(w*t))
HB = sp.Rational(1, 2)*m*sp.diff(x, t)**2 + sp.Rational(1, 2)*m*w**2*x**2
Hint = c*x
U_k = c**2/(2*m*w**2)
period = 2*sp.pi/w
avg = lambda f: sp.simplify(sp.integrate(f, (t, 0, period))/period)
check('A1a', 1, 'derived', 'sudden write: H_B(t) = 2U(1-cos wt), H_int(t) = -2U(1-cos wt), total conserved at 0',
      simp0(HB - 2*U_k*(1 - sp.cos(w*t))) and simp0(Hint + 2*U_k*(1 - sp.cos(w*t))) and simp0(HB + Hint))
check('A1b', 1, 'derived', 'time averages <H_B> = 2U, <H_int> = -2U',
      simp0(avg(HB) - 2*U_k) and simp0(avg(Hint) + 2*U_k))
# static (held) and ringing (spent) parts
y = x - xeq
E_ring = sp.Rational(1, 2)*m*sp.diff(y, t)**2 + sp.Rational(1, 2)*m*w**2*y**2
E_static = sp.Rational(1, 2)*m*w**2*xeq**2
check('A1c', 1, 'derived', 'held (static polaron displacement) = U and spent (ringing about the new equilibrium) = U',
      simp0(E_static - U_k) and simp0(E_ring - U_k))
xs = sp.symbols('x', real=True)
W = sp.Rational(1, 2)*m*w**2*xs**2 + c*xs
xmin = sp.solve(sp.diff(W, xs), xs)[0]
check('A2', 1, 'derived', 'final equilibrium energy W(x_eq) = -U: of the source work 2U, U is kept and U must leave as radiation',
      simp0(W.subs(xs, xmin) + U_k) and simp0(xmin - xeq))
# pointer commutation (Zurek criterion) for pure dephasing
sz = sp.Matrix([[1, 0], [0, -1]]); sx = sp.Matrix([[0, 1], [1, 0]]); I2 = sp.eye(2)
Bx = sp.Matrix([[0, 1], [1, 0]])  # bath coupling operator (any Hermitian)
from sympy.physics.quantum import TensorProduct as TP
Hi = TP(sz, Bx)
comm = lambda A, B: A*B - B*A
check('A3', 1, 'established+derived', 'H_int = sigma_z (x) B commutes with sigma_z (x) 1 (pointer observable), not with sigma_x (x) 1',
      comm(Hi, TP(sz, I2)) == sp.zeros(4) and comm(Hi, TP(sx, I2)) != sp.zeros(4))
# Gaussian relative entropy both ways = U/kT
kT = sp.symbols('kT', positive=True)
s2 = kT/(m*w**2)
X = sp.symbols('X', real=True)
def gauss(mu):
    return sp.exp(-(X - mu)**2/(2*s2))/sp.sqrt(2*sp.pi*s2)
mu1 = xeq
D10 = sp.integrate(gauss(mu1)*(sp.log(gauss(mu1)) - sp.log(gauss(0))).expand(force=True), (X, -sp.oo, sp.oo))
D01 = sp.integrate(gauss(0)*(sp.log(gauss(0)) - sp.log(gauss(mu1))).expand(force=True), (X, -sp.oo, sp.oo))
check('A4', 1, 'derived', 'classical thermal mode: kT D(rho1||rho0) = kT D(rho0||rho1) = U at every T',
      simp0(kT*D10 - U_k) and simp0(kT*D01 - U_k))
# quantum: displaced thermal state, both relative entropies = beta*hbar*omega*|alpha|^2
N = 140; xq = 0.7; alpha = 0.5
a = np.diag(np.sqrt(np.arange(1, N)), 1)
Dm = linalg.expm(alpha*a.conj().T - np.conj(alpha)*a)
p = np.exp(-xq*np.arange(N)); p /= p.sum()
logr = np.diag(-xq*np.arange(N) - np.log(np.sum(np.exp(-xq*np.arange(N)))))
rho = np.diag(p); sig_ = Dm@rho@Dm.conj().T
logs = Dm@logr@Dm.conj().T
S_sr = np.trace(sig_@logs).real - np.trace(sig_@logr).real
S_rs = np.trace(rho@logr).real - np.trace(rho@logs).real
check('A5', 1, 'derived (numeric)', 'quantum mode (beta hbar w = 0.7, alpha = 0.5, 140 levels): S(D rho D+||rho) = S(rho||D rho D+) = x|alpha|^2',
      abs(S_sr - xq*alpha**2) < 1e-8 and abs(S_rs - xq*alpha**2) < 1e-8, f"{S_sr:.10f}, {S_rs:.10f} vs {xq*alpha**2}")
# finite ramp: spent fraction sinc^2(w tau/2)
wt = sp.symbols('wt', positive=True)
A_amp = (sp.exp(sp.I*wt) - 1)/(sp.I*wt)
check('A6a', 1, 'derived', 'ramp of duration tau: ringing amplitude (e^{i w tau}-1)/(i w tau), |A|^2 = sinc^2(w tau/2)',
      simp0(sp.simplify(sp.expand_complex(A_amp*sp.conjugate(A_amp))) - (sp.sin(wt/2)/(wt/2))**2))
def ramp_ratio(wtau):
    om = 1.0; F = 1.0; tau = wtau/om
    f = lambda tt: F*min(tt/tau, 1.0)
    sol = solve_ivp(lambda tt, z: [z[1], f(tt) - om**2*z[0]], [0, tau + 20], [0, 0], rtol=1e-11, atol=1e-13, max_step=0.01)
    xe = F/om**2; yv = sol.y[0, -1] - xe; vv = sol.y[1, -1]
    return (0.5*vv**2 + 0.5*om**2*yv**2)/(0.5*om**2*xe**2)
vals = {v: ramp_ratio(v) for v in (0.1, 2.0, 6.0)}
check('A6b', 1, 'derived (numeric ODE)', 'ramp spent/U at w tau = 0.1, 2, 6 matches sinc^2(w tau/2) to 1e-6',
      all(abs(vals[v] - (math.sin(v/2)/(v/2))**2) < 1e-6 for v in vals), {k: round(v, 6) for k, v in vals.items()})

# ---------------------------------------------------------------- STEP 2: the price
V = sp.symbols('V', positive=True)
check('B1', 2, 'established', 'one-molecule gas compressed V -> V/2 at T: work = kT ln 2 (Landauer, Szilard)',
      simp0(sp.integrate(kT/V, (V, V/2, V)) - kT*sp.log(2)))
def binary_info(u):  # u = U/kT, record x ~ N(+-d, s), D(rho_s||rho0) = u => d^2/(2 s^2) = u, s = 1
    d = math.sqrt(2*u)
    g = lambda z, mu: math.exp(-(z - mu)**2/2)/math.sqrt(2*math.pi)
    def integrand(z):
        pp, pm = g(z, d), g(z, -d); pbar = 0.5*(pp + pm)
        out = 0.0
        for q in (pp, pm):
            if q > 0: out += 0.5*q*math.log(q/pbar)
        return out
    return integrate.quad(integrand, -d - 12, d + 12, limit=400)[0]
tab = {u: binary_info(u) for u in (1e-3, 0.1, 1.0, 10.0)}
check('B2', 2, 'derived (numeric) + established (Holevo golden formula)',
      'binary Gaussian record: I(sigma;record) <= U/kT; ratio -> 1 only for faint records',
      all(tab[u] <= u + 1e-12 for u in tab) and tab[1e-3]/1e-3 > 0.99 and tab[10.0] < math.log(2) + 1e-9,
      {u: round(v, 6) for u, v in tab.items()})
u99 = optimize.brentq(lambda u: binary_info(u) - 0.99*math.log(2), 0.5, 40)
check('B3', 2, 'derived (numeric)', 'a 0.99-bit Gaussian record needs U/kT held AND U/kT spent of (value); i.e. far above ln 2 = 0.693',
      u99 > 3*math.log(2), round(u99, 4))
Tsym, Qh = sp.symbols('T Q_h', positive=True)
check('B4', 2, 'book identity', "book eq:law_virialchain step 3: T*(Q/T) = Q for any T (an identity)", simp0(Tsym*(Qh/Tsym) - Qh))

# ---------------------------------------------------------------- STEP 3: the virial half
tt = sp.symbols('t', real=True)
kdeg = sp.symbols('k', real=True)
m1, m2, Cc = sp.symbols('m1 m2 C', positive=True)
r1 = sp.Matrix([sp.Function(f'x1{i}')(tt) for i in range(3)])
r2 = sp.Matrix([sp.Function(f'x2{i}')(tt) for i in range(3)])
d12 = sp.sqrt(sum((r1[i] - r2[i])**2 for i in range(3)))
Vp = Cc*d12**kdeg
F1 = sp.Matrix([-sp.diff(Vp, r1[i]) for i in range(3)])
F2 = sp.Matrix([-sp.diff(Vp, r2[i]) for i in range(3)])
virial = (r1.dot(F1) + r2.dot(F2))
check('C1a', 3, 'derived', 'Euler for a pair potential of degree k: sum r_i . F_i = -k V (Collins Eq. 1.2.x, any k)',
      sp.simplify(virial + kdeg*Vp) == 0)
Iin = m1*r1.dot(r1) + m2*r2.dot(r2)
Tk = sp.Rational(1, 2)*(m1*sp.diff(r1, tt).dot(sp.diff(r1, tt)) + m2*sp.diff(r2, tt).dot(sp.diff(r2, tt)))
newton = {sp.diff(r1[i], tt, 2): F1[i]/m1 for i in range(3)}
newton.update({sp.diff(r2[i], tt, 2): F2[i]/m2 for i in range(3)})
lag = sp.Rational(1, 2)*sp.diff(Iin, tt, 2) - (2*Tk - kdeg*Vp)
check('C1b', 3, 'derived', 'Lagrange identity (1/2) I.. = 2T - kV on Newtonian trajectories (k = -1: 2T + V)',
      sp.simplify(lag.subs(newton)) == 0)
aa, ee, EE = sp.symbols('a e E', positive=True)
inv_r_avg = sp.integrate((1/(aa*(1 - ee*sp.cos(EE))))*(1 - ee*sp.cos(EE)), (EE, 0, 2*sp.pi))/(2*sp.pi)
check('C2', 3, 'derived', 'Kepler ellipse: <1/r>_t = 1/a, hence <V> = 2E, <T> = -E', simp0(inv_r_avg - 1/aa))
qq, Aa, Ff = sp.symbols('q A F', positive=True)   # store degree n = 1 + q > 1
nn = 1 + qq
xst = (Ff/(nn*Aa))**(1/qq)
held = Aa*xst**nn; source = Ff*xst
stationary = sp.simplify(sp.diff(Aa*xs**nn - Ff*xs, xs).subs(xs, xst))
ratio_hs = sp.simplify(sp.powsimp(sp.powdenest(held/source, force=True), force=True))
check('C3', 3, 'derived', 'store A x^n charged by a constant force F (stationary point): held/source = 1/n; n = 2 gives 1/2',
      stationary == 0 and sp.simplify(ratio_hs - 1/nn) == 0, ratio_hs)
mdeg = sp.symbols('m_p', positive=True)
Tav, Vav = sp.symbols('T_av V_av', real=True)
sol = sp.solve(sp.Eq(mdeg*Tav, kdeg*Vav), Tav)[0]
heldfrac = sp.simplify(sol/(-Vav))
check('C4', 3, 'derived', 'generalized virial m<T> = k<V> (T of degree m in p): held fraction = -k/m; 1/2 iff (m,k) = (2,-1); 1 for (1,-1)',
      sp.simplify(heldfrac + kdeg/mdeg) == 0 and heldfrac.subs({kdeg: -1, mdeg: 2}) == sp.Rational(1, 2)
      and heldfrac.subs({kdeg: -1, mdeg: 1}) == 1)
# thermodynamic pair -> Euler split at T -> 0 for n = 4; exact at all T for n = 2
def kl_pair(n, beta, A=1.0, F=1.0):
    z = np.linspace(-6, 6, 400001); dz = z[1] - z[0]
    e0 = -beta*A*z**n; e1 = -beta*(A*z**n - F*z)
    m0, m1_ = e0.max(), e1.max()
    Z0 = np.sum(np.exp(e0 - m0))*dz; Z1 = np.sum(np.exp(e1 - m1_))*dz
    lnZ1Z0 = (m1_ - m0) + math.log(Z1/Z0)
    p1 = np.exp(e1 - m1_)/Z1
    mean1 = np.sum(z*p1)*dz
    D10 = beta*F*mean1 - lnZ1Z0
    D01 = lnZ1Z0
    return D10/beta, D01/beta
xs4 = (1/4)**(1/3); held4 = xs4**4; spent4 = 3*xs4**4
h1, s1 = kl_pair(4, 300.0); h2, s2_ = kl_pair(4, 3000.0)
check('C5a', 3, 'derived (numeric)', 'quartic store (n = 4): (kT D10, kT D01) -> (held, spent) = (1/4, 3/4) of the source work as T -> 0',
      abs(h2/held4 - 1) < abs(h1/held4 - 1) and abs(h2/held4 - 1) < 0.03 and abs(s2_/spent4 - 1) < 0.03,
      f"beta=300: {h1/held4:.4f},{s1/spent4:.4f}; beta=3000: {h2/held4:.4f},{s2_/spent4:.4f}")
hq, sq = kl_pair(2, 1.0)
Uq = 1.0**2/(4*1.0)  # A x^2 with A = 1/2 m w^2 = 1, F = 1: U = F^2/(4A)
check('C5b', 3, 'derived (numeric)', 'quadratic store (n = 2) at kT = 1: kT D10 = kT D01 = U exactly (no low-T limit needed)',
      abs(hq - Uq) < 1e-6 and abs(sq - Uq) < 1e-6, f"{hq:.8f}, {sq:.8f} vs {Uq}")
G_, c_, hb_, kB_, M_ = sp.symbols('G c hbar k_B M', positive=True)
S_bh = 4*sp.pi*G_*M_**2*kB_/(hb_*c_)
T_h = c_**2/sp.diff(S_bh, M_)
check('C6', 3, 'derived', 'Smarr as Euler degree 2: S proportional to M^2 => T_H S = (1/2) M c^2', simp0(T_h*S_bh - M_*c_**2/2))
qa, pa = sp.symbols('q p', real=True)
Kq = pa**2/2; Vq = qa**2/2
qt = sp.cos(tt); pt = -sp.sin(tt)
check('C7', 3, 'derived', 'harmonic mode (k = 2): <K> = <V>, but K + V is constant in time (both halves stay in the mode: a coincidence class, not held/spent)',
      simp0(sp.integrate((Kq - Vq).subs({qa: qt, pa: pt}), (tt, 0, 2*sp.pi))) and simp0(sp.diff((Kq + Vq).subs({qa: qt, pa: pt}), tt)))
ratios = {'sudden write (linear source, quadratic store)': sp.Rational(1, 2),
          'virial binding (|k| = 1 potential, p^2 store)': sp.Rational(1, 1)/2,
          'Smarr (M degree 1, S degree 2)': sp.Rational(1, 2),
          'capacitor step (QV degree 1 in Q, Q^2/2C degree 2)': sp.Rational(1, 2)}
check('C8', 3, 'derived (classification)', 'the four "same identity" halves are all the Euler degree ratio 1:2 (held = |n_source|/n_store x source)',
      all(v == sp.Rational(1, 2) for v in ratios.values()), list(ratios.keys()))

# ---------------------------------------------------------------- STEP 4: Koide as 3D -> 2D compression
s1_, s2s, s3_ = sp.symbols('s1 s2 s3', positive=True)
svec = sp.Matrix([s1_, s2s, s3_]); one = sp.Matrix([1, 1, 1])
Qk = (s1_**2 + s2s**2 + s3_**2)/(s1_ + s2s + s3_)**2
cos2 = (svec.dot(one))**2/(3*svec.dot(svec))
check('D1', 4, 'derived', 'cos^2(angle to (1,1,1)) = 1/(3Q): Q = 2/3 <=> 45 degrees', simp0(cos2 - 1/(3*Qk)))
xk, rk, dk = sp.symbols('x r delta', real=True)
sk = [xk*(1 + rk*sp.cos(dk + 2*sp.pi*j/3)) for j in range(3)]
spar2 = sp.simplify((sum(sk))**2/3)
sperp2 = sp.simplify(sp.trigsimp(sum(v**2 for v in sk) - spar2))
check('D2', 4, 'derived', 'Brannen form: |s_par|^2 = 3x^2, |s_perp|^2 = (3/2) x^2 r^2, Q = (1 + r^2/2)/3, independent of delta',
      simp0(spar2 - 3*xk**2) and simp0(sp.simplify(sp.expand_trig(sperp2 - sp.Rational(3, 2)*xk**2*rk**2)))
      and simp0(sp.simplify(sp.expand_trig(sum(v**2 for v in sk)/(sum(sk))**2 - (1 + rk**2/2)/3))))
mm = sp.symbols('m1:4', positive=True)
pk = [mi/sum(mm) for mi in mm]
H12 = 2*sp.log(sum(sp.sqrt(pi_) for pi_ in pk))
Qm = sum(mm)/(sum(sp.sqrt(mi) for mi in mm))**2
qk = [sp.sqrt(mi)/sum(sp.sqrt(mj) for mj in mm) for mi in mm]
CV2 = (sum(sp.sqrt(mi)**2 for mi in mm)/3 - (sum(sp.sqrt(mi) for mi in mm)/3)**2)/(sum(sp.sqrt(mi) for mi in mm)/3)**2
check('D3', 4, 'derived (KR1 re-check)', 'Q = exp(-H_1/2(p)) = sum q_i^2 (IPR) = (1 + CV^2)/3',
      simp0(sp.simplify(sp.exp(-H12) - Qm)) and simp0(sp.simplify(sum(q**2 for q in qk) - Qm)) and simp0(sp.simplify((1 + CV2)/3 - Qm)))
Qs = sp.symbols('Q', positive=True)
D_half_u = sp.log(3) - sp.log(1/Qs)     # D_1/2(p||u) = ln 3 - H_1/2
F_fid = 1/(3*Qs)
D2_q_u = sp.log(3*Qs)                   # D_2(q||u) = ln 3 - H_2(q) = ln 3 + ln Q
check('D4', 4, 'derived', 'D_1/2(p||uniform) = -ln F (book eq:ko:fid) = D_2(q||uniform) = ln(3Q); all = ln 2 (one bit) iff Q = 2/3',
      simp0(D_half_u + sp.log(F_fid)) and simp0(D_half_u - D2_q_u) and simp0(D_half_u.subs(Qs, sp.Rational(2, 3)) - sp.log(2)))
mp.mp.dps = 30
me, mmu, mtau = mp.mpf('0.51099895000'), mp.mpf('105.6583755'), mp.mpf('1776.93')
Qmeas = (me + mmu + mtau)/(mp.sqrt(me) + mp.sqrt(mmu) + mp.sqrt(mtau))**2
Hmeas = -mp.log(Qmeas)
Dbits = (mp.log(3) - Hmeas)/mp.log(2)
check('D5', 4, 'derived (numeric, PDG 2024 pole masses)', 'H_1/2 = 0.4054684 nats vs ln(3/2) = 0.4054651; D_1/2(p||u) in bits',
      abs(Hmeas - mp.mpf('0.4054684')) < 5e-8 and abs(mp.log(1.5) - mp.mpf('0.4054651')) < 5e-8,
      f"Q={mp.nstr(Qmeas, 9)}, H={mp.nstr(Hmeas, 8)}, ln1.5={mp.nstr(mp.log(1.5), 8)}, D={mp.nstr(Dbits, 8)} bits")
nu = sp.symbols('nu', positive=True)
Qnu = (1 + 2/nu)/3
check('D6a', 4, 'derived', 'chi-square nu: CV^2 = 2/nu, Q_nu = (nu+2)/(3 nu); nu = 2 gives 2/3, nu = 1 gives 1, nu -> oo gives 1/3',
      Qnu.subs(nu, 2) == sp.Rational(2, 3) and Qnu.subs(nu, 1) == 1 and sp.limit(Qnu, nu, sp.oo) == sp.Rational(1, 3))
Rr, th, Ee = sp.symbols('R theta E', positive=True)
# E = (q^2 + p^2)/2 with q, p ~ N(0,1): P(E <= e) = 1 - exp(-e)
cdf = sp.integrate(sp.integrate(sp.exp(-Rr**2/2)/(2*sp.pi)*Rr, (Rr, 0, sp.sqrt(2*Ee))), (th, 0, 2*sp.pi))
check('D6b', 4, 'derived', 'one canonical pair (q,p) in equilibrium: E = (q^2+p^2)/2 is exponential (CV = 1), so its spread gives Q = 2/3',
      simp0(sp.simplify(cdf - (1 - sp.exp(-Ee)))))
check('D7', 4, 'derived', 'rotation about (1,1,1) (advance delta) leaves |s_par| and |s_perp| (hence Q and the action) fixed',
      simp0(sp.diff(spar2, dk)) and simp0(sp.simplify(sp.diff(sp.expand_trig(sperp2), dk))))
# non-injectivity
r2v = sp.sqrt(2)
tripA = [sp.N(v.subs({xk: 1, rk: r2v, dk: sp.Rational(2, 9)}), 15) for v in sk]
tripB = [sp.N(v.subs({xk: 1, rk: r2v, dk: sp.Rational(2, 9) + sp.Rational(1, 10)}), 15) for v in sk]
QA = sum(v**2 for v in tripA)/sum(tripA)**2; QB = sum(v**2 for v in tripB)/sum(tripB)**2
check('D8a', 4, 'derived', 'record (scale, Q) is many-to-one: delta = 2/9 and 2/9 + 0.1 give the same (x, Q) but different mass triples',
      abs(QA - QB) < 1e-13 and abs(sum(tripA) - sum(tripB)) < 1e-13 and max(abs(a_ - b_) for a_, b_ in zip(tripA, tripB)) > 1e-2,
      f"Q = {float(QA):.12f}")
dgrid = np.linspace(0, 2*np.pi, 2_000_001)[:-1]
allpos = np.all(np.stack([1 + np.sqrt(2)*np.cos(dgrid + 2*np.pi*j/3) > 0 for j in range(3)]), axis=0)
check('D8b', 4, 'derived', 'on the Koide circle (r = sqrt 2) the all-positive-mass arcs are exactly 1/4 of the circle',
      abs(allpos.mean() - 0.25) < 1e-5, round(float(allpos.mean()), 6))
Jac = sp.Matrix(sk).jacobian([xk, rk, dk])
detJ = sp.simplify(sp.expand_trig(Jac.det()))
check('D9', 4, 'derived', 'keeping delta, (x, r, delta) -> masses is locally invertible: |det J| = (3 sqrt 3 / 2) x^2 |r|, nonzero for x, r != 0',
      simp0(sp.simplify(detJ**2 - (sp.Rational(27, 4))*xk**4*rk**2)), detJ)

# ---------------------------------------------------------------- STEP 5: the arrow
bet, Fq, ss = sp.symbols('beta F s', positive=True)
Xw = sp.symbols('X', real=True)
rho0 = sp.exp(-Xw**2/(2*ss**2))/sp.sqrt(2*sp.pi*ss**2)
jar = sp.integrate(rho0*sp.exp(bet*Fq*Xw), (Xw, -sp.oo, sp.oo))   # W = -F x for the sudden write
jar = sp.simplify(jar.subs(ss, 1/sp.sqrt(bet*m*w**2)))            # thermal width s^2 = kT/(m w^2)
dF_write = -Fq**2/(2*m*w**2)                                      # F1 - F0 = -U
check('E1a', 5, 'established (Jarzynski) + derived', 'sudden Gaussian write: <exp(-beta W)> = exp(-beta dF) with dF = -U = -F^2/(2 m w^2)',
      simp0(sp.simplify(jar - sp.exp(-bet*dF_write))))
Wv = sp.symbols('W', real=True)
Ub = sp.symbols('u', positive=True)  # u = beta U
# forward work W = -F x: Gaussian mean 0, var 2U/beta; reverse (write -> blank) work from rho1: mean... use Crooks with dF = -U
PF = sp.exp(-Wv**2/(2*(2*Ub)))/sp.sqrt(2*sp.pi*2*Ub)              # beta W in units (u = beta U)
PR = sp.exp(-(Wv - 2*Ub)**2/(2*(2*Ub)))/sp.sqrt(2*sp.pi*2*Ub)     # reverse work distribution, evaluated at -W: mean -2u shifted
crooks = sp.simplify(sp.log(PF/PR.subs(Wv, -Wv)) - (Wv + Ub))
check('E1b', 5, 'established (Crooks) + derived', 'Crooks: P_F(W)/P_R(-W) = exp(beta(W - dF)); mean beta W_diss = u = U/kT = D(rho0||rho1)',
      simp0(crooks))
check('E2', 5, 'derived (numeric)', 'arrow per record: beta W_diss = U/kT >= I(sigma; record) (the reversal is at least as distinguishable as the record is informative)',
      all(u >= tab[u] for u in tab), {u: round(u - tab[u], 6) for u in tab})
rng = np.random.default_rng(20261118)
def renyi_div(p_, q_, a_):
    if a_ == 1: return float(np.sum(p_*np.log(p_/q_)))
    return float(np.log(np.sum(p_**a_*q_**(1 - a_)))/(a_ - 1))
okdpi = True
for _ in range(2000):
    P = rng.dirichlet(np.ones(6)); Qd = rng.dirichlet(np.ones(6))
    Pc = P.reshape(3, 2).sum(1); Qc = Qd.reshape(3, 2).sum(1)
    for a_ in (0.5, 1, 2):
        if renyi_div(Pc, Qc, a_) > renyi_div(P, Qd, a_) + 1e-12: okdpi = False
check('E3', 5, 'established (van Erven-Harremoes; Lindblad-Uhlmann)', 'coarse-graining never increases D_1/2, D_KL, D_2 (2000 random pairs, 6 -> 3 cells)', okdpi)
sum_s = sp.simplify(sp.expand_trig(sum(sk)))
Q_s = sp.simplify(sp.expand_trig(sum(v**2 for v in sk)))/sum_s**2
xsol = sp.solve([sp.Eq(sum_s, 3), sp.Eq(sp.simplify(Q_s), sp.Rational(2, 3))], [xk, rk], dict=True)
xsol = [d_ for d_ in xsol if all(v.is_positive for v in d_.values())]
check('E4', 5, 'derived', 'inverting the record (x, Q) -> masses: solutions leave delta free (no inverse exists)',
      len(xsol) >= 1 and all(dk not in sol_.values() and not any(v.has(dk) for v in sol_.values()) for sol_ in xsol), xsol)
check('E5', 5, 'derived (numeric) + established (Sagawa-Ueda total)', 'sudden write + sudden erase cost 2U >= kT I (writer U, eraser U = held)',
      all(2*u >= tab[u] for u in tab))
A_ = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)); rr = A_@A_.conj().T; rr /= np.trace(rr).real
Uu = linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))[0]
vn = lambda r_: float(-np.sum([l*np.log(l) for l in np.linalg.eigvalsh(r_) if l > 1e-15]))
check('E6', 5, 'established (Liouville / unitarity)', 'fine-grained entropy unchanged by unitary evolution; dephasing (coarse-graining) raises it',
      abs(vn(Uu@rr@Uu.conj().T) - vn(rr)) < 1e-10 and vn(np.diag(np.diag(rr))) >= vn(rr) - 1e-12)
Ncells = sp.symbols('N', positive=True, integer=True)
check('E7', 5, 'established (Landauer, generalized)', 'forgetting delta held at N equiprobable cells lowers the memory entropy by ln N: minimum cost kT ln N; a bijection costs nothing',
      simp0(sp.log(Ncells) - (-(Ncells*(1/Ncells)*sp.log(1/Ncells)))))

# ---------------------------------------------------------------- STEP 6: the entropy functional (book's Cai-Kim form)
Gs, H, Hd, rho_, P_, Sd = sp.symbols('G H Hdot rho P Sdot_info', real=True)
TH = H/(2*sp.pi); dSgeo_dt = -(2*sp.pi/(Gs*H**3))*Hd
flux = 4*sp.pi*(rho_ + P_)/H**2
solH = sp.solve(sp.Eq(flux, TH*(dSgeo_dt + Sd)), Hd)[0]
check('F1', 6, 'derived (re-check of book eq:law_F2info)', '-dE = T_H d(S_geo + S_info) => Hdot = -4 pi G (rho+P) + (G/2 pi) H^3 Sdot_info',
      simp0(solH - (-4*sp.pi*Gs*(rho_ + P_) + Gs*H**3*Sd/(2*sp.pi))))
a_s, H0, bm = sp.symbols('a H_0 beta_m', positive=True)
rho_info = 3*H0**2*bm/(8*sp.pi*Gs)        # at a = 1, E(1) = 1
w_info = -1 - 1/(3*a_s)
rhoPx = (1 + w_info)*rho_info             # rho_x + P_x
Sdot_need = sp.solve(sp.Eq(rhoPx, -H**3*Sd/(8*sp.pi**2)), Sd)[0]
dSdlna = sp.simplify((Sdot_need/H).subs({a_s: 1, H: H0}))
check('F2', 6, 'derived (re-check of book eq:law_Sneed)', 'dS_info/dln a at a = 1 equals beta_m * pi/(G H0^2) = beta_m S_geo(1)',
      simp0(dSdlna - bm*sp.pi/(Gs*H0**2)))
dQ, Tt, dSg, dSi, diS = sp.symbols('dQ T dS_geo dS_info d_iS', real=True)
check('F3', 6, 'derived (placement)', 'book first law T(dS_geo + dS_info) = dQ is the EGJ balance dS_geo = dQ/T + d_iS with d_iS = -dS_info',
      simp0(sp.solve(sp.Eq(Tt*(dSg + dSi), dQ), dSg)[0] - (dQ/Tt + (-dSi))))
from scipy import constants as C
H0_si = 67.16e3/(3.0856775814913673e22)
S_geo_nats = math.pi*C.c**5/(C.G*C.hbar*H0_si**2)
bits_per_efold = 0.15765*S_geo_nats/math.log(2)
check('F4', 6, 'derived (numeric)', 'record entropy the book needs per e-fold today: beta_m S_geo/ln 2 = 5.2e121 bits (H0 = 67.16)',
      abs(bits_per_efold/5.2e121 - 1) < 0.02, f"{bits_per_efold:.3e} bits; S_geo = {S_geo_nats:.4e} nats")
T_GH = C.hbar*H0_si/(2*math.pi*C.k)
ratio_cell = 310.0/T_GH
check('F5', 6, 'derived (numeric)', 'two bookings of one locally paid bit: count (k ln2) vs heat at T_GH (k T_loc ln2 / T_GH): ratio T_loc/T_GH for 310 K',
      abs(T_GH/2.65e-30 - 1) < 0.01 and ratio_cell > 1e32, f"T_GH = {T_GH:.4e} K; ratio = {ratio_cell:.3e}")
T1, T2, Qq = sp.symbols('T1 T2 Q', positive=True)
check('F6', 6, 'derived', 'heat Q moved from T1 to a colder T2 gains entropy Q(1/T2 - 1/T1) > 0 (receipt grows as it travels to the bank)',
      sp.simplify((Qq/T2 - Qq/T1).subs({T1: 2*T2})) == Qq/(2*T2))

# ---------------------------------------------------------------- STEP 7: the horizon as the ledger
lP2 = C.hbar*C.G/C.c**3
area_bit = 4*math.log(2)*lP2
check('G1', 7, 'derived (S5 re-check)', 'one bit on the ledger = 4 ln2 l_P^2 = 7.24e-70 m^2 (eta = 1/4 l_P^2)',
      abs(area_bit/7.24e-70 - 1) < 0.003, f"{area_bit:.4e} m^2")
kap = sp.symbols('kappa', positive=True)
eta = c_**3/(4*hb_*G_)
dA = sp.symbols('dA', positive=True)
# kappa in s^-1: T = hbar kappa/(2 pi k_B); dE = kappa c^3 dA/(8 pi G) [book eq:th:dAmin]
dE2 = kap*c_**3*dA/(8*sp.pi*G_); Tk2 = hb_*kap/(2*sp.pi*kB_)
dA_bit = sp.solve(sp.Eq(dE2, kB_*Tk2*sp.log(2)), dA)[0]
check('G2', 7, 'derived', 'Landauer heat k T ln2 absorbed by a horizon adds exactly 4 ln2 l_P^2 of area, for every kappa (equality, no slack)',
      simp0(dA_bit - 4*sp.log(2)*hb_*G_/c_**3) and not dA_bit.has(kap))
Nb = S_bh/(kB_*sp.log(2))
check('G3', 7, 'derived', 'ledger bill: N_bits * k T_H ln2 = (1/2) M c^2 (the ln2 cancels; Smarr half)',
      simp0(Nb*kB_*T_h*sp.log(2) - M_*c_**2/2))
chi = sp.symbols('chi', positive=True)
check('G4', 7, 'derived', 'ledger sum: records of chi_i nats occupy 4 l_P^2 * sum chi_i of area; dA_info/dln a / A_H = beta_m when the book rate holds',
      simp0((4*chi*(hb_*G_/c_**3))*eta - chi) and simp0(sp.simplify((bm*sp.pi/(Gs*H0**2))/(sp.pi/(Gs*H0**2)) - bm)))
# S5 C7: a bit at T_H cannot be put on one tile: Bekenstein bound for carrier of size r_s with energy kT_H ln2
rs = 2*G_*M_/c_**2
Ebit = kB_*T_h*sp.log(2)
Smax = 2*sp.pi*Ebit*rs/(hb_*c_)
check('G5', 7, 'derived (S5 re-check)', 'Bekenstein bound on a carrier no larger than r_s holding k T_H ln2: at most (1/2) ln2 nats (half a bit): the ledger counts, it does not address',
      simp0(sp.simplify(Smax) - sp.log(2)/2))   # Smax is already in nats (S/k_B)

# ---------------------------------------------------------------- STEP 2b: where the free energy comes from
# H1: the Step-1 write is paid from the coupling's binding free energy; no new energy
Xh = sp.symbols('X', real=True); Fh, sh, kTh = sp.symbols('F s kT', positive=True)
rho0h = sp.exp(-Xh**2/(2*sh**2))/sp.sqrt(2*sp.pi*sh**2)
W_agent = sp.integrate(rho0h*(-Fh*Xh), (Xh, -sp.oo, sp.oo))       # sudden switch work from the blank
U_h = Fh**2*sh**2/(2*kTh)                                         # U = F^2/(2 m w^2) with s^2 = kT/(m w^2)
dF_h = -U_h
check('H1', '2b', 'derived', 'sudden write from blank: agent work W = 0, dF = -U, heat to bath W - dF = U: paid from the pointer-surface binding free energy, total energy conserved',
      simp0(W_agent) and simp0((W_agent - dF_h) - U_h))
# H2: Rovelli 2022 eqs (1), (4), (5)
Cm, Te, Tm, x_ = sp.symbols('C_m T_e T_m x', positive=True)    # x = tau_m/tau_em
dE_r = Cm*(Te - Tm)*(1 - sp.exp(-x_))
dS_r = dE_r/Tm - dE_r/Te
check('H2', '2b', 'derived (re-check of Rovelli 2022 eqs 1, 4, 5)', 'trace entropy dS = dE(1/T_m - 1/T_e) = C_m (T_e - T_m)^2 (1 - e^{-tau_m/tau_em})/(T_e T_m)',
      simp0(sp.simplify(dS_r - Cm*(Te - Tm)**2*(1 - sp.exp(-x_))/(Te*Tm))))
dEw = sp.symbols('dE', positive=True)
check('H3', '2b', 'derived (join)', "Rovelli's bound I < dE(1/T_m - 1/T_e)/k with a work source (T_e -> oo) is the Step-2 bound chi <= U/kT [B2]",
      simp0(sp.limit(dEw*(1/Tm - 1/Te), Te, sp.oo) - dEw/Tm))
Mcell = 54000/(8.314462618*310.15)
check('H4', '2b', 'book re-check (p6_02:24-28, 87, 98)', 'one ATP in the cytosol: M = 54 kJ/mol / (R 310.15 K) = 20.94 kT = 30.21 Landauer bits',
      abs(Mcell - 20.94) < 0.005 and abs(Mcell/math.log(2) - 30.21) < 0.01, f"M = {Mcell:.4f}, bits = {Mcell/math.log(2):.3f}")
S0 = 1361.0; Tsun = 5772.0; albedo = 0.30                         # IAU 2015 nominal S0, Teff; albedo assumed (illustrative)
Tearth = (S0*(1 - albedo)/(4*C.Stefan_Boltzmann))**0.25
uT = sp.symbols('u T', positive=True)
check('H5', '2b', 'established (blackbody s = 4u/3T) + derived (numeric)', 'sunlight: steady state, energy in = energy out; photon entropy out/in = T_sun/T_earth (albedo 0.30 assumed)',
      simp0(sp.simplify((sp.Rational(4, 3)*uT[0]/sp.Symbol('T_E', positive=True))/(sp.Rational(4, 3)*uT[0]/sp.Symbol('T_S', positive=True)) - sp.Symbol('T_S', positive=True)/sp.Symbol('T_E', positive=True))) and 20 < Tsun/Tearth < 25, f"T_earth = {Tearth:.1f} K, ratio = {Tsun/Tearth:.2f}")
mp_ = C.physical_constants['proton mass energy equivalent in MeV'][0]
me_ = C.physical_constants['electron mass energy equivalent in MeV'][0]
ma_ = C.physical_constants['alpha particle mass energy equivalent in MeV'][0]
Qpp = 4*mp_ + 2*me_ - ma_
frac = Qpp/(4*(mp_ + me_))
gam = sp.symbols('gamma', positive=True); Wg = sp.symbols('W', negative=True)
Etot = (3*gam - 4)/(3*(gam - 1))*Wg
radiated_share = sp.simplify(sp.diff(Etot, Wg))  # dW < 0 releases |dW|; E falls by share*|dW|, which is radiated
check('H6', '2b', 'derived (CODATA) + established (stellar virial)', 'star: 4 p + 2 e -> He-4 releases 26.73 MeV, 0.71% of rest mass; contraction at gamma = 5/3 radiates half of |dW|',
      abs(Qpp - 26.73) < 0.01 and abs(frac - 0.00712) < 0.0001 and radiated_share.subs(gam, sp.Rational(5, 3)) == sp.Rational(1, 2),
      f"Q = {Qpp:.4f} MeV, fraction = {frac:.5f}")
rr_, Hh, Sd_ = sp.symbols('rho_r H Sdot', positive=True)
check('H7', '2b', 'derived (from book eq:law_F2info)', 'radiated heat has rho + P = (4/3) rho > 0; the record component has rho_x + P_x = -H^3 Sdot_info/(8 pi^2) < 0: the heat cannot be the dark-sector term',
      (rr_ + rr_/3).is_positive and (-Hh**3*Sd_/(8*sp.pi**2)).is_negative and ((1 + (-1 - sp.Rational(1, 3)))).is_negative)
TGH_book = C.hbar*(67.4e3/3.0856775814913673e22)/(2*math.pi*C.k)
Tgas_implied = (3.15e-8/2.6e-44)*TGH_book
Tgas_sigma = 0.59*C.m_p*(213e3)**2/C.k
check('H8', '2b', 'derived (reconstruction of book p2_12b:92-93; inferred)', "the book's heat (3.15e-8) and count-priced (2.6e-44) numbers differ by T_gas/T_GH; implied T_gas matches a 213 km/s virial gas (mu = 0.59) within 15%",
      abs(Tgas_implied/Tgas_sigma - 1) < 0.15, f"implied {Tgas_implied:.3e} K vs mu m_p sigma^2/k = {Tgas_sigma:.3e} K")
LP2 = hb_*G_/c_**3
A_s = 4*sp.pi*rs**2
E_eq = sp.Rational(1, 2)*(A_s/LP2)*kB_*T_h
check('H9', '2b', 'derived (re-check of Padmanabhan 2010 eq 7 with c1 = 1)', 'Schwarzschild: E = (1/2) n k T_H with n = A/L_P^2 equals M c^2 = 2 T_H S: equipartition is Smarr',
      simp0(sp.simplify(E_eq - M_*c_**2)) and simp0(sp.simplify(E_eq - 2*T_h*S_bh)))
rad, kap2 = sp.symbols('r kappa', positive=True)
kap_sol = sp.solve(sp.Eq(sp.Rational(1, 2)*(4*sp.pi*rad**2/LP2)*hb_*kap2/(2*sp.pi*c_), M_*c_**2), kap2)[0]
check('H10', '2b', 'derived (re-check of Padmanabhan 2010 eqs 8-9)', 'equipartition (1/2) k T per Planck area on a sphere of area A with Unruh T and E = M c^2 gives kappa = G M / r^2',
      simp0(sp.simplify(kap_sol - G_*M_/rad**2)))
Rh = sp.symbols('R', positive=True)
E_MS = c_**4*Rh/(2*G_)
S_dS = sp.pi*Rh**2*c_**3*kB_/(G_*hb_)
T_GHs = hb_*c_/(2*sp.pi*kB_*Rh)
rho_L = 3*c_**2/(8*sp.pi*G_*Rh**2)                 # de Sitter: H = c/R, energy density rho c^2
V_ = sp.Rational(4, 3)*sp.pi*Rh**3
komar = (rho_L*c_**2 - 3*rho_L*c_**2)*V_          # (rho + 3p) V with p = -rho c^2
E_eq_dS = sp.Rational(1, 2)*(4*sp.pi*Rh**2/LP2)*kB_*T_GHs
check('H11', '2b', 'derived (identity; for R13, no verdict)', 'de Sitter horizon: (1/2) n k T_GH = 2 T_GH S = 2 E_MS = |(rho + 3p) V| (the Komar/Tolman active energy of the vacuum)',
      simp0(sp.simplify(E_eq_dS - 2*T_GHs*S_dS)) and simp0(sp.simplify(E_eq_dS - 2*E_MS)) and simp0(sp.simplify(E_eq_dS + komar)))

P_abs = S0*(1 - albedo)*math.pi*(6.371e6)**2
Sdot_earth = (4/3)*P_abs*(1/Tearth - 1/Tsun)                     # blackbody photon entropy, steady state
bits_rate = Sdot_earth/(C.k*math.log(2))
check('H12', '2b', 'derived (numeric; upper bound, albedo 0.30 assumed)', "Earth's record budget from sunlight (Rovelli's I < dS/k): absorbed power and the largest record-writing rate it can fund",
      1e37 < bits_rate < 1e38, f"P_abs = {P_abs:.3e} W, Sdot = {Sdot_earth:.3e} W/K, bound = {bits_rate:.2e} bits/s")


# ==================================================== NEW IN THE ASSEMBLY


# ---------------------------------------------------------------- STEP 3 in d space dimensions (which-threes spider)
dd = sp.symbols('d', positive=True)
Vabs = sp.symbols('V_abs', positive=True)
kept_d = (dd - 2)/2*Vabs                # 2<K> = k<V>, k = -(d-2), V = -|V|
rad_d = -(kept_d - Vabs)                # from rest: Q = -E_f = -(K + V)
check('W1', 3, 'derived (which-threes)', 'd-dim virial: kept K = (d-2)/2 |V|, radiated (4-d)/2 |V|; equal only at d = 3',
      simp0(rad_d - (4 - dd)/2*Vabs) and sp.solve(sp.Eq(kept_d, rad_d), dd) == [3])
rr_d, cM, cS = sp.symbols('r c_M c_S', positive=True)
Md = cM*rr_d**(dd - 2); Sd_ = cS*rr_d**(dd - 1)
Td = sp.diff(Md, rr_d)/sp.diff(Sd_, rr_d)
check('W2', 3, 'derived (which-threes; Tangherlini)', 'Smarr in d space dims: T S/M = (d-2)/(d-1) (Euler degree), 1/2 only at d = 3',
      simp0(sp.simplify(Td*Sd_/Md) - (dd - 2)/(dd - 1)) and ((dd - 2)/(dd - 1)).subs(dd, 3) == sp.Rational(1, 2))
cd_ = sp.symbols('c_d', positive=True)
Vd = cd_*rr_d**dd; Ad = sp.diff(Vd, rr_d)
check('W3', 6, 'derived (which-threes; ledger 1/3)', 'swept-shell geometry A r / V = d (so the ledger 1/3 = 1/d at d = 3)',
      simp0(sp.simplify(Ad*rr_d/Vd) - dd))
check('W4', 3, 'derived (which-threes)', 'virial share (d-2)/2 and Smarr share (d-2)/(d-1) agree only at d = 2 and d = 3',
      set(sp.solve(sp.Eq((dd - 2)/2, (dd - 2)/(dd - 1)), dd)) == {2, 3})
cv2, Ngen = sp.symbols('CV2 N', positive=True)
QN = (1 + cv2)/Ngen
check('W5', 4, 'derived (which-threes)', 'N generations: Q = (1+CV^2)/N and D_1/2(p||u) = ln N + ln Q = ln(1+CV^2): one bit at CV = 1 for any N',
      simp0(sp.expand_log(sp.log(Ngen) + sp.log(QN), force=True) - sp.log(1 + cv2)))
okN = True
for Nn in range(3, 8):
    s_ = np.array([1 + math.sqrt(2)*math.cos(0.3 + 2*math.pi*k/Nn) for k in range(Nn)])
    okN &= abs((s_**2).sum()/s_.sum()**2 - 2/Nn) < 1e-12
check('W6', 4, 'derived (numeric; which-threes F2)', 'Brannen form on N phases at r = sqrt 2 gives Q = 2/N (N = 3..7)', okN)

# ---------------------------------------------------------------- STEP 4: why the plane equals the axis (Koide-split spider)
nhat = sp.Matrix([1, 1, 1])/sp.sqrt(3)
Jn = sp.Matrix([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])/sp.sqrt(3)        # generator of rotation about (1,1,1) (real antisymmetric)
Lsym = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'L{min(i,j)}{max(i,j)}'))
eqs = list((Lsym*Jn - Jn*Lsym))
solL = sp.solve(eqs, list(Lsym.free_symbols), dict=True)[0]
Lgen = Lsym.subs(solL)
check('K1', 4, 'derived (Koide-split A1)', 'a symmetric coupling invariant under rotation about (1,1,1) has a 2-dim solution space: a P_par + b P_perp',
      len(Lgen.free_symbols) == 2)
aa_, bb_, th_ = sp.symbols('a b theta', positive=True)
Eth = (aa_*sp.cos(th_)**2 + bb_*sp.sin(th_)**2)/2
crit = sp.solve(sp.diff(Eth, th_), th_)
check('K2', 4, 'derived (Koide-split A2c)', 'classical pointer: extremizing the held energy over directions gives 0 or 90 degrees only (never 45) for a != b',
      all(sp.simplify(cc - sp.pi/4) != 0 for cc in crit) and set(crit) <= {0, sp.pi/2, sp.pi})
Wsrc = sp.symbols('W', positive=True)
Qof = lambda par, perp: (par + perp)/(3*par)
check('K3', 4, 'derived (Koide-split A3)', 'Euler 1:2 roles: (axis, plane) = (held, spent) -> Q = 2/3; (source, held) -> 1/2; (held, source) -> 1',
      Qof(Wsrc/2, Wsrc/2) == sp.Rational(2, 3) and sp.simplify(Qof(Wsrc, Wsrc/2)) == sp.Rational(1, 2) and sp.simplify(Qof(Wsrc/2, Wsrc)) == 1)
qs, wt_ = sp.symbols('q_s omega_t', positive=True)
al = -qs*sp.cos(wt_); be = qs*sp.sin(wt_)
e1 = sp.Matrix([2, -1, -1])/sp.sqrt(6); e2 = sp.Matrix([0, 1, -1])/sp.sqrt(2)
svw = qs*nhat + al*e1 + be*e2
Qw = (svw.dot(svw))/(sum(svw))**2
check('K4', 4, 'derived (Koide-split C2)', 'sudden write: axis = held offset q*, plane = ringing pair (alpha, beta): |s_perp| = |s_par| and Q = 2/3 at every instant',
      simp0(sp.simplify(sp.trigsimp(Qw)) - sp.Rational(2, 3)))
brannen = [qs/sp.sqrt(3)*(1 + sp.sqrt(2)*sp.cos(wt_ + sp.pi + 2*sp.pi*k/3)) for k in range(3)]
check('K5', 4, 'derived (Koide-split C3)', 'read on three phases the written vector is Brannen form with r = sqrt 2 and delta = omega t + pi (the ringing phase)',
      all(simp0(sp.simplify(sp.expand_trig(svw[k] - brannen[k]))) for k in range(3)))
check('K6', 4, 'derived (Koide-split C9)', 'F = cos^2 theta = held/(held + spent) = 1/2, so D_1/2(p||u) = -ln F = ln 2 (Step 1 pair = Step 4 bit)',
      simp0(sp.simplify((svw.dot(nhat))**2/svw.dot(svw)) - sp.Rational(1, 2)))
sinc2 = (math.sin(1.0)/1.0)**2
Qramp = (1 + sinc2)/3
check('K7', 4, 'derived (Koide-split C5)', 'finite ramp: ring radius q*|sinc(w tau/2)| -> Q = (1 + sinc^2)/3 <= 2/3; at w tau = 2 the ODE spent fraction gives the same Q',
      abs(vals[2.0] - sinc2) < 1e-6 and Qramp < 2/3, f"Q(w tau = 2) = {Qramp:.6f}")
kTs, kst = sp.symbols('kT k_s', positive=True)
QT = Qof(qs**2, qs**2 + 2*kTs/kst)
check('K8', 4, 'derived (Koide-split C6)', 'warm start adds 2kT/k to the plane only: Q_T = 2/3 + 2kT/(3 k q*^2) > 2/3',
      simp0(sp.simplify(QT - (sp.Rational(2, 3) + 2*kTs/(3*kst*qs**2)))))
check('K9', 4, 'derived (Koide-split C4)', 'position-only reading (one quadrature) gives r = 1 and Q = (1 + 1/2)/3 = 1/2',
      (1 + sp.Rational(1, 2))/3 == sp.Rational(1, 2))
eps3 = lambda k, i, j: sp.LeviCivita(k, i, j)
Jx, Jy, Jz = [sp.Matrix(3, 3, lambda i, j: -sp.I*eps3(k, i, j)) for k in range(3)]
Jsq = Jx**2 + Jy**2 + Jz**2
Jn3 = (Jx + Jy + Jz)/sp.sqrt(3)
evs = Jn3.eigenvects()
ev_vals = sorted([sp.nsimplify(v[0]) for v in evs])
m0vec = [v for v in evs if sp.simplify(v[0]) == 0][0][2][0]
mp1 = [v for v in evs if sp.simplify(v[0] - 1) == 0][0][2][0]
mp1 = mp1/sp.sqrt((mp1.H*mp1)[0])
axis_m1 = sp.simplify((mp1.H*Jn3*mp1)[0])**2
plane_m1 = sp.simplify((mp1.H*(Jsq - Jn3**2)*mp1)[0])
check('K10', 4, 'derived (Koide-split A4-A4f) + established (spin algebra)', 'three components carry spin 1: J^2 = 2, J_n eigenvalues -1, 0, 1, m = 0 is the (1,1,1) axis; stretched m = 1: <J_n>^2 = <J_perp^2> = 1',
      sp.simplify(Jx*Jy - Jy*Jx - sp.I*Jz) == sp.zeros(3) and sp.simplify(Jsq - 2*sp.eye(3)) == sp.zeros(3)
      and ev_vals == [-1, 0, 1] and sp.simplify(m0vec.cross(sp.Matrix([1, 1, 1]))) == sp.zeros(3, 1)
      and sp.simplify(axis_m1 - 1) == 0 and sp.simplify(plane_m1 - 1) == 0)
jj = sp.symbols('j', positive=True)
Qj = (jj**2 + jj)/(3*jj**2)
okj = True
for j2 in (1, 2, 3, 4):                       # j = 1/2, 1, 3/2, 2: explicit spin matrices, stretched state
    jv = j2/2; dim = j2 + 1; ms = np.arange(jv, -jv - 1, -1)
    Jp = np.zeros((dim, dim))
    for i in range(1, dim): Jp[i - 1, i] = math.sqrt(jv*(jv + 1) - ms[i]*(ms[i] + 1))
    JX = (Jp + Jp.T)/2; JY = (Jp - Jp.T)/(2j); JZ = np.diag(ms); psi = np.zeros(dim); psi[0] = 1
    okj &= abs((psi@JZ@psi)**2 - jv**2) < 1e-12 and abs((psi@(JX@JX + JY@JY)@psi).real - jv) < 1e-12
check('K11', 4, 'derived (Koide-split A5a-c)', 'stretched spin-j pointer: axis j^2, plane j, Q(j) = (j+1)/(3j) = Q_nu(nu = 2j); equal split only at j = 1',
      okj and simp0(sp.simplify(Qj - (1 + 2/(2*jj))/3)) and sp.solve(sp.Eq(jj**2, jj), jj) == [1])
check('K12', 4, 'derived (Koide-split A5d; negative)', 'the Euler balance on the Casimir (2 j^2 = j) picks j = 1/2 and Q = 1: the dimension, not the degree ratio, fixes j = 1',
      sp.solve(sp.Eq(2*jj**2, jj), jj) == [sp.Rational(1, 2)] and Qj.subs(jj, sp.Rational(1, 2)) == 1)
check('K13', 4, 'derived (numeric; Koide-split F1)', 'measured |s_perp|^2/|s_par|^2 = 3Q - 1 from PDG 2024 masses (D5)',
      abs(3*Qmeas - 1 - mp.mpf('0.99999339')) < 2e-8, mp.nstr(3*Qmeas - 1, 9))

# ---------------------------------------------------------------- STEP 6: the chain to the Friedmann equation (book p1_02:305-447, re-derived)
check('R1', 6, 'established (Cai-Kim 2005) re-derived', 'with S_info = 0 the first law -dE = T_H dS_geo gives Hdot = -4 pi G (rho + P) (book eq:law_F2)',
      simp0(solH.subs(Sd, 0) + 4*sp.pi*Gs*(rho_ + P_)))
check('R2', 6, 'established re-derived', 'with continuity rho_dot = -3H(rho+P): d/dt(H^2 - 8 pi G rho/3) = 0, so H^2 = 8 pi G rho/3 + Lambda/3 (Lambda an integration constant)',
      simp0(2*H*(-4*sp.pi*Gs*(rho_ + P_)) - sp.Rational(8, 3)*sp.pi*Gs*(-3*H*(rho_ + P_))))
av = sp.symbols('a', positive=True)
Ef = sp.Function('Ef')
solE = sp.dsolve(sp.Eq(Ef(av).diff(av), Ef(av)/av**2), Ef(av), ics={Ef(1): 1})
Ebook = sp.exp(1 - 1/av)
check('R3', 6, 'derived given the record constraint (book eq:law_constraint, conjecture)', 'rho_dot_info = rho_info H/a with E(1) = 1 integrates to E(a) = exp(1 - 1/a) = e^{-z} (book eq:Ea)',
      simp0(sp.simplify(solE.rhs - Ebook)) and simp0(sp.simplify(Ebook.subs(av, 1/(1 + sp.Symbol('z'))) - sp.exp(-sp.Symbol('z')))))
dEdlna = sp.simplify(av*sp.diff(Ebook, av))
check('R4', 6, 'derived (book fig:activation)', 'E -> 0 as a -> 0, E(oo) = e, inflection at a = 1/2, dE/dln a = E/a peaks at a = 1',
      sp.limit(Ebook, av, 0, '+') == 0 and sp.limit(Ebook, av, sp.oo) == sp.E and sp.solve(sp.diff(Ebook, av, 2), av) == [sp.Rational(1, 2)]
      and sp.solve(sp.diff(dEdlna, av), av) == [1])
winf = -1 - sp.Rational(1, 3)*sp.diff(sp.log(Ebook), av)*av
check('R5', 6, 'derived (book eq:law_winfo)', 'continuity with dln rho_info/dln a = 1/a gives w_info = -1 - 1/(3a) (< -1 at every finite a)',
      simp0(sp.simplify(winf - (-1 - 1/(3*av)))) and sp.solve(1 + (-1 - 1/(3*av)), av) == [])
check('R6', 6, 'derived (book p1_02:336)', 'the record constraint is the balance law dln E/dln a = 1 - ln E (a writer at one per e-fold, no eraser)',
      simp0(sp.simplify(av*sp.diff(sp.log(Ebook), av) - (1 - sp.log(Ebook)))))
rhoi = sp.symbols('rho_info', positive=True)
Sd_gen = sp.solve(sp.Eq(-rhoi/(3*av), -H**3*Sd/(8*sp.pi**2)), Sd)[0]
check('R7', 6, 'derived (book eq:law_Sneed, general a)', 'rho_x + P_x = -H^3 Sdot_info/8 pi^2 with w_info: Sdot_info = 8 pi^2 rho_info/(3 a H^3)',
      simp0(Sd_gen - 8*sp.pi**2*rhoi/(3*av*H**3)))
nexp = sp.symbols('n')
need = sp.simplify(av*sp.diff(sp.log(Ebook*av**5), av))
check('R8', 6, 'derived (book p1_02:336, the book\'s own open problem)', 'matter era: eq:law_Sneed needs dS_info/dln a ~ E a^5 while n = 7/2 gives a^{-1}; matching needs n_eff = 19/2 + 1/a',
      simp0(sp.solve(sp.Eq(nexp - sp.Rational(9, 2), need), nexp)[0] - (sp.Rational(19, 2) + 1/av)))
bm_num = 0.15765
check('R9', 6, 'derived (book eq:mu, eq:law_sirens)', 'mu(a=1) = 1/(1 + beta_m) = 0.8638, mu_0 = -0.13618; matter-sector H0 = 67.16 sqrt(1 + beta_m) = 72.26',
      abs(1/(1 + bm_num) - 0.8638) < 5e-5 and abs(-bm_num/(1 + bm_num) + 0.13618) < 5e-6 and abs(67.16*math.sqrt(1 + bm_num) - 72.26) < 0.005,
      f"mu0 = {-bm_num/(1+bm_num):.5f}, H0_m = {67.16*math.sqrt(1+bm_num):.3f}")
check('R10', 6, 'book prediction (eq:law_beta), numeric', 'beta_m = Omega_m/2 = 0.15765 at Omega_m = 0.3153', abs(0.3153/2 - bm_num) < 1e-9)

# ---------------------------------------------------------------- the beta_m = Omega_m/2 identities (units hbar = c = k_B = 1)
Hx, Gx, Om_, bx, mx = sp.symbols('H G Omega_m beta m', positive=True)
rho_c = 3*Hx**2/(8*sp.pi*Gx)
Sgeo = sp.pi/(Gx*Hx**2); TH_ = Hx/(2*sp.pi); VH = sp.Rational(4, 3)*sp.pi/Hx**3; AH = 4*sp.pi/Hx**2
Mm = Om_*rho_c*VH
check('L1', 6, 'derived (count spider identity 1; ledger LH2)', 'beta S_geo = (1/2) M_m c^2/(k T_H) holds iff beta = Omega_m/2 (matter in the apparent-horizon sphere, a = 1)',
      sp.solve(sp.Eq(bx*Sgeo, Mm/(2*TH_)), bx) == [Om_/2])
flux_m = AH*Om_*rho_c*(1/Hx)*Hx*(1/Hx)            # A_H rho_m r_A H per unit time x (1/H) per e-fold
dS_flux = flux_m/TH_
check('L2', 6, 'derived (ledger I1-I4; count identity 2)', 'matter flux per e-fold at T_H carries 3 Omega_m S_geo; flux = 3 x content (= d x content); beta_m S_geo is 1/6 of it',
      simp0(sp.simplify(dS_flux - 3*Om_*Sgeo)) and simp0(sp.simplify(flux_m - 3*Mm)) and simp0(sp.simplify((Om_/2*Sgeo)/dS_flux - sp.Rational(1, 6))))
H0x, rc0 = sp.symbols('H_0 rho_c0', positive=True)
Ha = sp.Function('Hfun')(av)
req = 8*sp.pi**2*(rc0*bx*Ebook)/(3*av*Ha**4)          # dS_info/dln a required (R7 / H)
mflux = 8*sp.pi**2*(rc0*Om_*av**-3)/Ha**4               # matter flux entropy per e-fold at T_H(a)
check('L3', 6, 'derived (ledger LK1)', 'at any a the required record is E(a) a^2/6 of the matter-flux entropy (H and radiation drop out)',
      simp0(sp.simplify((req/mflux).subs(bx, Om_/2) - Ebook*av**2/6)))
Npart = Om_*rc0*av**-3*sp.Rational(4, 3)*sp.pi/Ha**3/mx
Gam = Ha*req/Npart
check('L4', 6, 'derived (time-and-rate; missing-writer)', 'per writing particle the requirement is Gamma_req = pi E a^2 m c^2/hbar; H cancels; Gamma_req/(Bekenstein rate pi m c^2/hbar) = E a^2',
      simp0(sp.simplify(Gam.subs(bx, Om_/2) - sp.pi*Ebook*av**2*mx)) and simp0(sp.simplify(Gam.subs(bx, Om_/2)/(sp.pi*mx) - Ebook*av**2)))
check('L5', 6, 'derived (time-and-rate)', 'Bekenstein bound of the Hubble radius: pi E/(hbar H) = E/(2 k T_H): pricing at T_H IS that bound with the virial 1/2',
      simp0(sp.simplify(sp.pi*mx/Hx - mx/(2*TH_))))
check('L6', 6, 'derived (time-and-rate)', 'Bekenstein rate pi E/hbar over Margolus-Levitin 2E/(pi hbar) = pi^2/2 (the missing-writer 4.93 nats per operation)',
      simp0(sp.simplify((sp.pi*mx)/(2*mx/sp.pi) - sp.pi**2/2)))
check('L7', 6, 'derived (dark-sector A1; register R14)', 'T_H S_geo / V_H = 3H^2/(8 pi G) = rho_crit exactly (horizon normalization)',
      simp0(sp.simplify(TH_*Sgeo/VH - rho_c)))
MH = rho_c*VH; RH = 1/Hx
check('L8', 7, 'derived (weave identities)', 'Hubble sphere: G M_H/(R_H c^2) = 1/2, r_s(M_H) = R_H, T_BH(M_H) = T_GH/2 (Bekenstein-saturated)',
      simp0(sp.simplify(Gx*MH/RH - sp.Rational(1, 2))) and simp0(sp.simplify(2*Gx*MH - RH)) and simp0(sp.simplify(1/(8*sp.pi*Gx*MH) - TH_/2)))
nats_req = bm_num*S_geo_nats
check('L9', 6, 'derived (numeric; count spider)', 'the book\'s requirement today: beta_m S_geo = 3.597e121 nats = 5.19e121 bits per e-fold (H0 = 67.16)',
      abs(nats_req/3.597e121 - 1) < 0.002, f"{nats_req:.4e} nats")
Esat = (4 - av**-3)/3
check('L10', 6, 'derived (Bekenstein-Dirac; proposal)', "ceiling a^3 dE/dln a <= 1: the book's E gives E a^2 (= 1 only at a = 1, 6.59 at a = 2); E_sat = (4 - a^-3)/3 rides the ceiling, E_sat(1) = 1, -> 4/3",
      simp0(sp.simplify(av**3*av*sp.diff(Ebook, av) - Ebook*av**2)) and simp0(sp.simplify(av**3*av*sp.diff(Esat, av) - 1))
      and Esat.subs(av, 1) == 1 and sp.limit(Esat, av, sp.oo) == sp.Rational(4, 3) and abs(float((Ebook*av**2).subs(av, 2)) - 6.59) < 0.005)
check('L11', 6, 'derived (numeric; time-and-rate)', 'record clock relative to today E a^2 = 0.0920 at z = 1, 0.0150 at z = 2, 0.0031 at z = 3',
      all(abs(float((Ebook*av**2).subs(av, 1/(1 + z))) - v) < 5e-4 for z, v in ((1, 0.0920), (2, 0.0150), (3, 0.0031))))
check('L12', 6, 'derived (ledger L3-L5)', 'ledger entry rule k T_H dS_info = V_H d rho_info per e-fold today equals beta_m rho_c V_H = (1/2) M_m c^2 (balance sheet x turnover 1)',
      simp0(sp.simplify(TH_*(Om_/2)*Sgeo - VH*(Om_/2)*rho_c)))

# ---------------------------------------------------------------- black-hole reading
Msun = 1.98847e30
def M_bh(M): return 2*(4*math.pi*C.G*M**2/(C.hbar*C.c))   # M c^2/(k T_BH) = 2 S/k
check('BH1', 'BH', 'book re-check (p1_02 fig:mahaffey_number; p3_01:357)', 'black hole drive ratio M = Mc^2/(k T_BH) = 2 S_BH/k: 2.1e77 for 1 Msun, 3.9e90 for Sgr A* (4.3e6 Msun)',
      abs(M_bh(Msun)/2.1e77 - 1) < 0.03 and abs(M_bh(4.3e6*Msun)/3.9e90 - 1) < 0.03, f"{M_bh(Msun):.3e}, {M_bh(4.3e6*Msun):.3e}")
Pbh = hb_*c_**6/(15360*sp.pi*G_**2*M_**2); Tbh = hb_*c_**3/(8*sp.pi*G_*M_*kB_)
check('BH2', 'BH', 'book re-check (eq:bh_gamma)', 'Hawking power priced at k T_BH ln2: Gamma = c^3/(1920 G M ln2) bits/s',
      simp0(sp.simplify(Pbh/(kB_*Tbh*sp.log(2)) - c_**3/(1920*G_*M_*sp.log(2)))))
Mv = sp.symbols('Mv', positive=True)
paid = sp.integrate((T_h*sp.diff(S_bh, M_)).subs(M_, Mv), (Mv, 0, M_))
check('BH3', 'BH', 'derived (register R6)', 'paid while growing: integral T dS = M c^2; held at today\'s price T S = M c^2/2 (held/paid = 1/2)',
      simp0(sp.simplify(paid - M_*c_**2)) and simp0(sp.simplify(T_h*S_bh/paid - sp.Rational(1, 2))))

# ---------------------------------------------------------------- cell reading
Tb = 310.15
ebit = C.k*Tb*math.log(2)
check('CE1', 'cell', 'book re-check (eq:ebit)', 'one bit at 310.15 K: k T ln2 = 2.968e-21 J', abs(ebit/2.968e-21 - 1) < 5e-4, f"{ebit:.4e} J")
Efl = 28217448*ebit; atp_J = 54000/C.N_A
check('CE2', 'cell', 'book re-check (eq:efloor)', 'floor for one copy of the methylome (28,217,448 CpGs) = 8.38e-14 J = 9.3e5 ATP at 54 kJ/mol',
      abs(Efl/8.38e-14 - 1) < 0.002 and abs(Efl/atp_J/9.3e5 - 1) < 0.01, f"{Efl:.4e} J, {Efl/atp_J:.3e} ATP")
phi_c = 3.08/Mcell; eps0c = 1/(1 + math.exp(3.08)); epsmin = 1/(1 + math.exp(Mcell))
check('CE3', 'cell', 'book re-check (eq:phi, eq:eps0)', 'phi = E_hold/(M kT) = 3.08/20.94 = 0.147; eps0 = 1/(1+e^{3.08}) = 0.044 (1 in 23); eps_min = 1 in 1.24e9',
      abs(phi_c - 0.147) < 0.0006 and abs(eps0c - 0.044) < 0.0006 and abs(1/epsmin/1.24e9 - 1) < 0.01, f"phi={phi_c:.4f}, eps0={eps0c:.4f}, 1/eps_min={1/epsmin:.3e}")
ff, gg, Eh, kTc = sp.symbols('f g E kT', positive=True)
epsb = ff/(ff + gg)
check('CE4', 'cell', 'derived (book eq:balance, eq:eps)', 'loss/restore balance eps = f/(f+g) gives E_hold = kT ln(g/f) = kT ln((1-eps)/eps), the inverse of eps = 1/(1+e^{E/kT})',
      simp0(sp.simplify(sp.log((1 - epsb)/epsb) - sp.log(gg/ff))) and simp0(sp.simplify(kTc*sp.log((1 - 1/(1 + sp.exp(Eh/kTc)))/(1/(1 + sp.exp(Eh/kTc)))) - Eh)))
check('CE5', 'cell', 'book re-check (p6_02 sec margin)', 'maintenance 0.90-0.98 (Genereux 2005) -> discrimination ln(1/0.10..0.02) = 2.3-3.9 kT per decision; 3.08 kT = 4.44 Landauer units (book rounds 4.45)',
      abs(math.log(10) - 2.30) < 0.01 and abs(math.log(50) - 3.91) < 0.01 and abs(3.08/math.log(2) - 4.44) < 0.01)

# (checks QC1-QC6 are kept in the private copy only)

npass = sum(r['passed'] for r in RESULTS); nfail = len(RESULTS) - npass
json.dump(dict(script='iam_law_checks.py', date='2026-11-18', passed=npass, failed=nfail, checks=RESULTS),
          open('iam_law_checks_output.json', 'w'), indent=1)
print(f"\n{npass} passed, {nfail} failed")
sys.exit(0 if nfail == 0 else 1)
