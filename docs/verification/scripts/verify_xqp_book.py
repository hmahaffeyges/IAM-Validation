#!/usr/bin/env python3
"""Book carriage of the quasiparticle-floor chapter (docs/book/part3/p3_02_xqp.tex): every equation and number, recomputed.
Corrections applied (docs/verification/PAPER_ERRATA.md XQ1-XQ8; particle/XQP_CHECK.md; particle/XQP_REFEREE_NOTE.md) and the
source checks made in this carriage. Sympy for the algebra, numpy for numbers.
Run: python docs/verification/scripts/verify_xqp_book.py > docs/verification/scripts/verify_xqp_book_output.txt"""
import numpy as np, sympy as sp, scipy.constants as C
from scipy.special import k1

k, e, hbar, me = C.k, C.e, C.hbar, C.m_e
ln2 = np.log(2)
D_ueV = 182.0                          # Delta_Al in ueV (BCS 1.764 k_B T_c with T_c = 1.20 K)
D = D_ueV * 1e-6 * e
p = print

p("== 0. Inputs")
p(f"BCS gap for T_c = 1.20 K: 1.764 k_B T_c = {1.764*k*1.20/e*1e6:.1f} ueV (used: {D_ueV} ueV)")

p("== 1. Temperature structure")
Tgap = D / k
p(f"T_gap = Delta/k_B = {Tgap:.3f} K;  T_CMB - T_gap = {2.725-Tgap:.3f} K;  T_gap - 15 mK = {Tgap-0.015:.3f} K")
for T in (0.010, 0.015, 0.020):
    a = D / (k * T)
    p(f"T = {T*1e3:.0f} mK: Delta/k_B T = {a:.1f};  exp(-Delta/k_B T) = 10^{-a/np.log(10):.1f}")

p("== 2. Thermal fraction x_th = sqrt(2 pi k_B T/Delta) exp(-Delta/k_B T)")
xth = lambda T: np.sqrt(2*np.pi*k*T/D) * np.exp(-D/(k*T))
for T in (0.015, 0.020, 0.100, 0.150):
    p(f"  x_th({T*1e3:.0f} mK) = {xth(T):.2e}")
Tg = np.linspace(0.05, 0.3, 200001); T7 = Tg[np.argmin(abs(np.log10(xth(Tg)) + 7))]
p(f"  x_th = 1e-7 at T = {T7*1e3:.0f} mK;  asymptotic form at k_B T = Delta (outside k_B T << Delta): sqrt(2 pi)/e = {np.sqrt(2*np.pi)/np.e:.3f}")
Dl, kT, nu0, z = sp.symbols('Delta kT nu0 z', positive=True)
K1asym = sp.sqrt(sp.pi/(2*z))*sp.exp(-z)            # K1(z) for z >> 1
nqp = 4*nu0*Dl*K1asym.subs(z, Dl/kT)                # n_qp = 4 nu0 int_D^inf E/sqrt(E^2-D^2) e^{-E/kT} dE = 4 nu0 Delta K1(Delta/kT)
xsym = sp.simplify(nqp/(2*nu0*Dl))
p(f"  sympy: x = n_qp/(2 nu0 Delta) = {xsym};  difference from sqrt(2 pi kT/Delta) e^(-Delta/kT): {sp.simplify(xsym - sp.sqrt(2*sp.pi*kT/Dl)*sp.exp(-Dl/kT))}")
E = sp.symbols('E', positive=True)
for T in (0.02, 0.15):
    zz = D/(k*T); p(f"  {T*1e3:.0f} mK: x from 2 K1(Delta/kT) = {2*k1(zz):.3e};  asymptote = {xth(T):.3e}")

p("== 3. London phase-disturbance volume")
r, lam, r0, u = sp.symbols('r lambda r_0 u', positive=True)
Veff = sp.integrate(4*sp.pi*sp.exp(-2*r/lam)*r**2, (r, 0, sp.oo))
p(f"  int_0^inf 4 pi e^(-2r/lam) r^2 dr = {sp.simplify(Veff)};  Gamma(3) = {sp.integrate(u**2*sp.exp(-u),(u,0,sp.oo))}")
G = sp.exp(-r/lam)/(4*sp.pi*r)
p(f"  (nabla^2 - 1/lam^2)[e^(-r/lam)/(4 pi r)] for r > 0 = {sp.simplify(sp.diff(r**2*sp.diff(G, r), r)/r**2 - G/lam**2)}")
Vy = sp.integrate(4*sp.pi*r**2*(r0*sp.exp(-(r-r0)/lam)/r)**2, (r, r0, sp.oo))
p(f"  full profile e^(-r/lam)/r normalised at a core radius r0: V = {sp.simplify(Vy)}")
lamL, xi0 = 50e-9, 1600e-9
p(f"  pi lam^3 at 50 nm = {np.pi*lamL**3:.3e} m^3 = {np.pi*0.05**3:.3e} um^3;  (xi0/lam)^3 = {(xi0/lamL)**3:.0f};  lam/xi0 = {lamL/xi0:.3f}")

p("== 4. Erasure mapping")
for g in (1e6, 10e6):
    p(f"  g/2pi = {g/1e6:.0f} MHz at omega_q/2pi = 5 GHz: g/omega_q = {g/5e9:.1e} rad")
tphi = hbar / D
p(f"  tau_phi = hbar/Delta = {tphi:.3e} s = {tphi*1e12:.2f} ps")
for tt in (1e-6, 30e-6, 100e-6):
    p(f"  tau_phi/tau_TLS at {tt*1e6:.0f} us = {tphi/tt:.1e}")
p(f"  Delta ln2 = {D_ueV*ln2:.1f} ueV;  2 Delta = {2*D_ueV:.0f} ueV;  2Delta/(Delta ln2) = {2/ln2:.3f};  k_B (20 mK) ln2 = {k*0.02*ln2/e*1e6:.2f} ueV")

p("== 5. Cooper-pair density")
a = 4.05e-10; ne = 4*3/a**3
EF = hbar**2*(3*np.pi**2*ne)**(2/3)/(2*me)
nu0_fe = 3*ne/(4*EF)
p(f"  n_e = {ne:.3e} m^-3;  n_e/2 = {ne/2:.3e} m^-3 = {ne/2*1e-18:.2e} um^-3")
p(f"  E_F = {EF/e:.2f} eV;  nu0 = 3 n_e/(4 E_F) = {nu0_fe*e:.3e} eV^-1 m^-3 = {nu0_fe*e*1e-24:.2e} um^-3 ueV^-1")
p(f"  n_cp = 2 nu0 Delta (free electron) = {2*nu0_fe*D*1e-18:.2e} um^-3")
nu0_R = 1.2e4
p(f"  nu0 = 1.2e4 um^-3 ueV^-1: n_cp = {2*nu0_R*D_ueV:.2e} um^-3 (182 ueV), {2*nu0_R*170:.2e} um^-3 (170 ueV)")
ncp = 4e6
p(f"  used n_cp = 4e6 um^-3;  (n_e/2)/n_cp = {ne/2*1e-18/ncp:.0f};  floor x = 1e-7 -> n_qp = {1e-7*ncp:.1f} um^-3")
p(f"  n_qp = 0.04 um^-3 (20 mK) with n_cp = {2*nu0_R*170:.2e}: x = {0.04/(2*nu0_R*170):.1e}")

p("== 6. Diffusion over a lifetime")
for Dd in (0.6, 2.25, 6.0):
    p(f"  D = {Dd} um^2/ns, tau_qp = 100 us: sqrt(D tau) = {np.sqrt(Dd*1e3*100):.0f} um")

p("== 7. Steady state on the island")
t, Y, Ns, tT, tq, V, n_cp = sp.symbols('t Y N tau_TLS tau_qp V n_cp', positive=True)
Nq = sp.Function('Nq')
sol = sp.dsolve(sp.Eq(Nq(t).diff(t), Y*Ns/tT - Nq(t)/tq), Nq(t), ics={Nq(0): 0})
Nss = sp.limit(sol.rhs, t, sp.oo); xexpr = sp.simplify(Nss/(V*n_cp))
p(f"  N_qp(t) = {sp.simplify(sol.rhs)};  steady state {Nss};  x = {xexpr};  Y = 2: {xexpr.subs(Y,2)}")
p(f"  x n_cp V tau_TLS/(N tau_qp) = {sp.simplify(xexpr*n_cp*V*tT/(Ns*tq))}")
tT_, tq_ = 30.0, 100.0
for Vv in (1e3, 1e4, 1e5):
    Nn = 1e-7*ncp*Vv*tT_/(2*tq_); x1 = 2*tq_/(tT_*ncp*Vv)
    p(f"  V = {Vv:.0e} um^3: sites for 1e-7 = {Nn:.0f};  x from one site = {x1:.2e};  with Y = ln2: {Nn*2/ln2:.0f} sites")
p(f"  example: x 1e-7, n_cp 4e6, V 1e4, tau_TLS 30, N 600, tau_qp 100 -> {1e-7*4e6*1e4*30/(600*100):.3f}")
for tt in (1, 10, 30, 100, 1000):
    p(f"  tau_TLS = {tt:>5} us: sites for 1e-7 on 1e4 um^3 = {1e-7*ncp*1e4*tt/(2*tq_):.0f}")
p(f"  pi lam^3 / (1e4 um^3) = {np.pi*0.05**3/1e4:.1e}")
for NV in (0.006, 0.06, 0.6):
    p(f"  N/V = {NV} um^-3: x = 1e-7 at tau_TLS = {2*NV*tq_/(ncp*1e-7):.0f} us")
P1 = 2*D/30e-6
p(f"  minimum heat per site with one pair per event: 2 Delta/tau_TLS = {P1:.2e} W;  600 sites: {600*P1:.2e} W")

p("== 8. Coherence ceiling")
w = 2*np.pi*5e9
for x in (1e-7, 1e-6):
    p(f"  x = {x:.0e}: T1 (Gamma = x omega_q) = {1/(x*w)*1e3:.3f} ms;  T1 (Catelani) = {1/((x/np.pi)*np.sqrt(2*w*D/hbar))*1e3:.3f} ms")
for T1 in (0.3e-3, 0.5e-3, 0.6e-3):
    p(f"  T1 = {T1*1e3:.1f} ms at 5 GHz requires x <= {np.pi/(T1*np.sqrt(2*w*D/hbar)):.1e} (Catelani form)")
