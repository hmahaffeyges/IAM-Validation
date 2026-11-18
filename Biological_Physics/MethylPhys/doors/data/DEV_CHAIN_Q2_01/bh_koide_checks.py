import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy'], check=True)
#!/usr/bin/env python3
"""BH-KOIDE (register X11, R23; author 2026-11-18: 'did we consider the Koide formula for the black hole the same way we did for the horizon?').
Exploration; runs ON A GITHUB MACHINE (sympy). Established input: the Christodoulou-Ruffini mass formula (Christodoulou 1970; Christodoulou and
Ruffini 1971; Scholarpedia 'Kerr-Newman metric'): M^2 = (M_irr + Q^2/(4 M_irr))^2 + J^2/(4 M_irr^2), G = c = 1, M_irr^2 = A/(16 pi).
IAM identification under test (the Koide-split spider's reading of Koide's equal split: axis = held, plane = spent): the black hole's record
(the area, M_irr, which no process lowers) is the held part; rotation and charge energy (extractable by Penrose / Christodoulou processes) is the
spent/working part. Koide's condition Q = 2/3 is |held|^2 = |working|^2.
EXPECTATIONS, written by the lead by hand before any run:
  K1 the formula is an identity for Kerr-Newman with r+ = M + sqrt(M^2 - a^2 - Q^2), A = 4 pi (r+^2 + a^2)
  K2 extremal Kerr (a = M): M_irr^2 = M^2/2 and J^2/(4 M_irr^2) = M^2/2 (equal split of M^2: the Koide condition under the identification)
  K3 extremal Reissner-Nordstrom (Q = M): M_irr = M/2 and Q^2/(4 M_irr) = M/2 (equal split of M itself, linear, not of M^2)
  K4 maximum extractable fraction: Kerr 1 - 1/sqrt2 = 0.2929; RN 1/2
  K5 over all sub-extremal Kerr-Newman states, the held share M_irr^2/M^2 never falls below 1/4 (extremal RN) and equals 1/2 only on the
     extremal Kerr point among Q = 0 holes; reported: the curve of held share against spin, and where held = working along extremal KN
  K6 a plain Koide ratio sum(v)^2 form on v = (M_irr, Q^2/4M_irr, J/2M_irr) is reported only as a number (expected 1/2 at both extremal points);
     no claim rests on it (it is not the same construction as Koide's sqrt-mass vector)."""
import json, os, sympy as sp
M, a, Q = sp.symbols("M a Q", positive=True)
rp = M + sp.sqrt(M**2 - a**2 - Q**2); A = 4 * sp.pi * (rp**2 + a**2); Mi = sp.sqrt(A / (16 * sp.pi)); J = a * M
CR = (Mi + Q**2 / (4 * Mi))**2 + J**2 / (4 * Mi**2)
R = {}
def chk(i, desc, ok, val=None): R[i] = {"desc": desc, "pass": bool(ok), "value": None if val is None else str(val)}
for (av, qv) in ((sp.Rational(3, 10), sp.Rational(2, 5)), (sp.Rational(1, 2), sp.Rational(1, 3)), (sp.Rational(7, 10), sp.Rational(1, 10)), (0, sp.Rational(9, 10)), (sp.Rational(9, 10), 0)):
    v = sp.nsimplify(sp.simplify(CR.subs({a: av * M, Q: qv * M}) / M**2)); chk(f"K1_{av}_{qv}", f"CR formula = M^2 at a={av}M, Q={qv}M", sp.simplify(v - 1) == 0, v)
mk = sp.simplify(Mi.subs({a: M, Q: 0})); chk("K2a", "extremal Kerr: M_irr^2 = M^2/2", sp.simplify(mk**2 - M**2 / 2) == 0, mk**2)
chk("K2b", "extremal Kerr: J^2/(4 M_irr^2) = M^2/2", sp.simplify((M**2)**2 / (4 * mk**2) - M**2 / 2) == 0, sp.simplify((M**2)**2 / (4 * mk**2)))
mr = sp.simplify(Mi.subs({a: 0, Q: M})); chk("K3a", "extremal RN: M_irr = M/2", sp.simplify(mr - M / 2) == 0, mr)
chk("K3b", "extremal RN: Q^2/(4 M_irr) = M/2", sp.simplify(M**2 / (4 * mr) - M / 2) == 0, sp.simplify(M**2 / (4 * mr)))
chk("K4a", "Kerr max extractable 1 - 1/sqrt2", abs(float(1 - mk / M) - (1 - 2**-0.5)) < 1e-12, float(1 - mk / M))
chk("K4b", "RN max extractable 1/2", abs(float(1 - mr / M) - 0.5) < 1e-12, float(1 - mr / M))
import numpy as np
f = sp.lambdify((a, Q), (Mi**2 / M**2).subs(M, 1), "numpy")
grid = [(x, y) for x in np.linspace(0, 1, 201) for y in np.linspace(0, 1, 201) if x**2 + y**2 <= 1]
hs = [(float(f(x, y)), x, y) for x, y in grid]; mn = min(hs)
chk("K5a", "held share M_irr^2/M^2 >= 1/4 on all sub-extremal KN (minimum at extremal RN)", mn[0] >= 0.25 - 1e-9, mn)
kerr = [(round(x, 3), round(float(f(x, 0)), 6)) for x in np.linspace(0, 1, 11)]
ext = [(round(float(np.cos(t)), 4), round(float(np.sin(t)), 4), round(float(f(np.cos(t), np.sin(t))), 6)) for t in np.linspace(0, np.pi / 2, 11)]
chk("K5b", "held share along Q = 0 reaches 1/2 only at a = M", abs(float(f(1.0, 0)) - 0.5) < 1e-9 and all(float(f(x, 0)) > 0.5 for x in np.linspace(0, 0.99, 50)), kerr)
def kq(x, y):
    mi = float(f(x, y))**0.5; v = np.array([mi, y**2 / (4 * mi), x / (2 * mi)]); return float((v @ v) / v.sum()**2)
chk("K6", "plain sum-ratio |v|^2/(sum v)^2 on v = (M_irr, Q^2/4M_irr, J/2M_irr): extremal Kerr, extremal RN (number only)", True, (round(kq(1.0, 0.0), 6), round(kq(0.0, 1.0), 6)))
# KOIDE HALF (lead's hand identity 2026-11-18, written before the run): v = (sqrt m1, sqrt m2, sqrt m3); |v|^2 = sum m (total rest energy);
# axis part |v.n|^2 with n = (1,1,1)/sqrt3 equals (sum sqrt m)^2/3; so axis share = 1/(3Q), Q = sum m/(sum sqrt m)^2; Q = 2/3 gives exactly 1/2.
m1, m2, m3 = sp.symbols("m1 m2 m3", positive=True); v = sp.Matrix([sp.sqrt(m1), sp.sqrt(m2), sp.sqrt(m3)]); n = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
Qk = (m1 + m2 + m3) / (sp.sqrt(m1) + sp.sqrt(m2) + sp.sqrt(m3))**2; ax = (v.dot(n))**2
chk("KH1", "axis share |v.n|^2/|v|^2 = 1/(3Q) for any masses", sp.simplify(ax / (v.dot(v)) - 1 / (3 * Qk)) == 0)
chk("KH2", "Q = 2/3 gives axis share exactly 1/2 (held = half the rest energy)", sp.Rational(1, 3) / sp.Rational(2, 3) == sp.Rational(1, 2))
me, mmu, mtau = 0.51099895, 105.6583755, 1776.93   # PDG 2024 charged-lepton masses, MeV
Ql = (me + mmu + mtau) / (me**0.5 + mmu**0.5 + mtau**0.5)**2
chk("KH3", "charged leptons (PDG 2024): Q and axis share 1/(3Q)", abs(Ql - 2 / 3) < 1e-4, (round(Ql, 7), round(1 / (3 * Ql), 7)))
chk("KH4", "ledger: half of rest energy x 1/d (d = 3) = 1/6, the record term's share of matter's horizon-flux entropy", sp.Rational(1, 2) * sp.Rational(1, 3) == sp.Rational(1, 6))
mu_, md_, ms_ = 2.16, 4.70, 93.5; Qq = (mu_ + md_ + ms_) / (mu_**0.5 + md_**0.5 + ms_**0.5)**2
chk("KH5", "light quarks (u, d, s; PDG 2024 MS-bar): Q, for the record (Koide does not hold there)", True, round(Qq, 4))
# KH6 (lead's hand, before the run): N components: axis share = 1/(N Q); Q = (1 + CV^2)/N with CV = 1 (one canonical pair) gives 2/N, so the axis share is 1/2 for ANY N
for Nn in (2, 3, 4, 5):
    xs = sp.symbols(f"x1:{Nn+1}", positive=True); vv = sp.Matrix(xs); nn = sp.Matrix([1] * Nn) / sp.sqrt(Nn)
    QN = sum(x**2 for x in xs) / sum(xs)**2
    chk(f"KH6_N{Nn}", f"N = {Nn}: axis share = 1/(N Q)", sp.simplify((vv.dot(nn))**2 / vv.dot(vv) - 1 / (Nn * QN)) == 0)
    chk(f"KH6b_N{Nn}", f"N = {Nn}: Q = 2/N gives axis share 1/2", sp.Rational(1, Nn) / sp.Rational(2, Nn) == sp.Rational(1, 2))
# K7: Kerr across spin: Smarr heat share 2TS/M... book's TS/M = sqrt(1-chi^2)/2 vs Christodoulou-Ruffini held share M_irr^2/M^2 and M_irr/M
chi = sp.symbols("chi", nonnegative=True); rpk = 1 + sp.sqrt(1 - chi**2); Ak = 4 * sp.pi * (rpk**2 + chi**2); Sk = Ak / 4; Tk = sp.sqrt(1 - chi**2) / (2 * sp.pi * (rpk**2 + chi**2) / rpk) / 1
Tk = (rpk - 1) / (4 * sp.pi * rpk)   # Kerr Hawking temperature, M = 1: kappa/2pi with kappa = (r+ - r-)/(2(r+^2 + a^2)) = sqrt(1-chi^2)/(2 r+) ... r+^2 + a^2 = 2 r+
chk("K7a", "Kerr: T S / M = sqrt(1 - chi^2)/2 (Smarr heat share; book eq:bh_gamma form)", all(abs(float((Tk * Sk - sp.sqrt(1 - chi**2) / 2).subs(chi, c))) < 1e-12 for c in (0, 0.3, 0.6, 0.9, 0.99)))
row = [(c, round(float((Tk * Sk).subs(chi, c)), 6), round(float((Ak / (16 * sp.pi)).subs(chi, c)), 6)) for c in (0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0)]
chk("K7b", "Kerr spin table (chi, Smarr share TS/M, CR held share M_irr^2/M^2): Smarr 1/2 at chi = 0 where CR = 1; CR 1/2 at chi = 1 where Smarr = 0", abs(row[0][1] - 0.5) < 1e-9 and abs(row[-1][2] - 0.5) < 1e-9, row)
mc_, mb_, mt_ = 1.2730, 4.183, 172.57   # PDG 2024: m_c(m_c), m_b(m_b) MS-bar, m_t pole (GeV); verify against the PDG tables in the report
Qh = (mc_ + mb_ + mt_) / (mc_**0.5 + mb_**0.5 + mt_**0.5)**2
chk("KH7", "heavy quarks c, b, t: Q and axis share 1/(3Q) (literature ~0.669)", abs(Qh - 0.669) < 0.003, (round(Qh, 5), round(1 / (3 * Qh), 5)))
out = {"checks": R, "held_share_kerr": kerr, "held_share_extremal_KN_(a,Q,share)": ext, "passed": sum(c["pass"] for c in R.values()), "failed": sum(not c["pass"] for c in R.values())}
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/bh_koide.json", "w"), indent=1, default=str)
print(f"::notice::BH-KOIDE (with KH: Koide half) {out['passed']} pass {out['failed']} fail: " + json.dumps({k: (v['pass'], v['value']) for k, v in R.items() if not k.startswith('K1')}, default=str)[:1700])
print("::notice::BH-KOIDE held share along extremal KN (a, Q, share): " + json.dumps(ext))
