#!/usr/bin/env python3
"""Recomputes every calculated (\\calc / \\derived) number printed in Part 2 chapters p2_16 to p2_20 (survey predictions, lensing-dynamics, three-way clusters,
missing satellites, w(z) and the far future). Chain values, published
measurements and DESI fits are inputs, not recomputed. numpy, scipy. Run from docs/verification/scripts/."""
import csv, os, numpy as np
trap = getattr(np, "trapezoid", None) or np.trapz
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
Om, OL = 0.3153, 0.6847; bm = Om/2; H0 = 67.36
E = lambda a: np.exp(1-1/a); H2 = lambda a: Om*a**-3+OL
mu = lambda a: H2(a)/(H2(a)+bm*E(a)); Oma = lambda a: Om*a**-3/H2(a)
def grow(m):
    r = lambda l, y: [y[1], -(2-1.5*Om*np.exp(-3*l)/H2(np.exp(l)))*y[1]+1.5*Oma(np.exp(l))*m(np.exp(l))*y[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
L, I = grow(lambda a: 1.0), grow(mu)
D = lambda S, a: S.sol(np.log(a))[0]; f = lambda S, a: S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
print("A. coupling, growth, potential (Omega_m 0.3153, beta_m = Omega_m/2 = %.5f, same early amplitude)" % bm)
for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
    a = 1/(1+z)
    print(f"   z {z:3}: mu {mu(a):.4f}  1-mu {100*(1-mu(a)):5.2f}%  1/mu {1/mu(a):.3f}  dD/D {100*(D(I,a)/D(L,a)-1):6.2f}%  "
          f"dfs8/fs8 {100*(f(I,a)*D(I,a)/(f(L,a)*D(L,a))-1):6.2f}%  dPhi/Phi (Sigma=1) {100*(D(I,a)/D(L,a)-1):6.2f}%")
print("B. activation milestones: fraction q of 1 - mu(0) reached")
d0 = 1-mu(1.0); Gyr = 977.792/H0; t = lambda a: quad(lambda x: 1/(x*np.sqrt(H2(x))), 1e-8, a)[0]*Gyr; t0 = t(1)
for q in (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    a = brentq(lambda a: (1-mu(a))/d0-q, 0.05, 1); print(f"   {100*q:4.0f}%: a {a:.3f} z {1/a-1:.2f} lookback {t0-t(a):.1f} Gyr")
zz = np.linspace(0, 3, 30001); dmz = np.gradient(np.array([mu(1/(1+z)) for z in zz]), zz)
print(f"   |dmu/dz| largest at z = {zz[np.argmax(abs(dmz))]:.3f}; value {abs(dmz[0]):.4f} at z=0, {abs(dmz[np.searchsorted(zz,0.5)]):.4f} at z=0.5")
print("C. ISW source (1-f)D/a ratio and ISW-galaxy cross (source x D, uniform weight z 0.05-1.5)")
src = lambda S, a: (1-f(S, a))*D(S, a)/a
for z in (0.1, 0.5, 1.0): a = 1/(1+z); print(f"   z {z}: source ratio {src(I,a)/src(L,a):.4f}")
zs = np.linspace(0.05, 1.5, 300); aa = 1/(1+zs)
num = trap([src(I, a)*D(I, a) for a in aa], zs); den = trap([src(L, a)*D(L, a) for a in aa], zs)
print(f"   A_ISW (cross) = {num/den:.3f}")
print("D. sirens / matter-sector H0: 67.161 sqrt(1 + 0.15750) =", round(67.161*np.sqrt(1.1575), 2))
print("E. clusters, Level 1 form: R = 1/mu; slope; non-thermal C_NT = 1 + 0.20 (1+z)^0.2 at bin centres")
Rz = lambda z: 1/mu(1/(1+z)); h = 1e-4
print(f"   dR/dz at z 0.3 = {(Rz(0.3+h)-Rz(0.3-h))/(2*h):.3f}; at z 0 = {(Rz(h)-Rz(0))/h:.3f}")
for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8)):
    zc = (lo+hi)/2; C = 1+0.20*(1+zc)**0.2; dC = 0.20*0.2*(1+zc)**-0.8
    print(f"   bin {lo}-{hi} (z {zc:.3f}): IAM {Rz(zc):.3f}  C_NT {C:.4f}  IAM x C_NT {Rz(zc)*C:.3f}  dC_NT/dz {dC:+.3f}")
print("   Planck SZ: Level 1 sigma8 0.8139 -> 0.8014 =", f"{100*(0.8014/0.8139-1):.2f} %", "; Level 2 0.8087 -> 0.7998 =", f"{100*(0.7998/0.8087-1):.2f} %")
print("F. missing satellites: Eq. 11 M_min = 4 Om sigma^3/(G H0); Eq. 12 sigma_crit; Eq. 10 direct")
G = 6.674e-11; Msun = 1.98892e30; H0s = H0*1e3/3.0857e22
Mmin = lambda s: 4*Om*(s*1e3)**3/(G*H0s)/Msun
print(f"   M_min(4 km/s) = 10^{np.log10(Mmin(4)):.2f} Msun;  sigma_crit(M=10^8.4) = {((G*H0s*10**8.4*Msun)/(4*Om))**(1/3)/1e3:.2f} km/s")
print(f"   Eq. 10 with t_dyn = 1/H0: sqrt(6/pi) sigma^3/(G H0) = 10^{np.log10(np.sqrt(6/np.pi)*(4e3)**3/(G*H0s)/Msun):.2f} Msun; prefactors 4 Om = {4*Om:.2f}, sqrt(6/pi) = {np.sqrt(6/np.pi):.2f}")
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "observations", "data", "lvdb_dwarf_mw.csv")
rows = list(csv.DictReader(open(p)))
kin = [r for r in rows if r["vlos_sigma"] or r["vlos_sigma_ul"]]
val = lambda r: float(r["vlos_sigma"]) if r["vlos_sigma"] else float(r["vlos_sigma_ul"])
below = [r for r in kin if val(r) < 4]; res = [r for r in below if r["vlos_sigma"]]; ul = [r for r in below if not r["vlos_sigma"]]
res1s = [r for r in res if float(r["vlos_sigma"])+float(r["vlos_sigma_ep"]) < 4]
print(f"   census: {len(rows)} MW satellites; {len(kin)} with a dispersion or upper limit; {len(below)} below 4 km/s "
      f"({100*len(below)/len(kin):.0f} %); {sum(r['confirmed_galaxy']=='1' for r in below)} of them confirmed galaxies; {len(kin)-len(below)} at or above 4")
print(f"   below 4: {len(res)} resolved dispersions, {len(ul)} upper limits; resolved with sigma + 1 sigma error still below 4: {len(res1s)}")
print("   resolved, still below 4 at +1 sigma:")
for r in sorted(res1s, key=lambda r: float(r["vlos_sigma"])): print(f"     {r['name']}: {r['vlos_sigma']} +{r['vlos_sigma_ep']}/-{r['vlos_sigma_em']}")
print("   upper limits below 4:")
for r in sorted(ul, key=lambda r: float(r["vlos_sigma_ul"])): print(f"     {r['name']}: < {r['vlos_sigma_ul']}")
fe = [float(r["metallicity_spectroscopic_sigma"]) for r in below if r["metallicity_spectroscopic_sigma"]]
print(f"   resolved [Fe/H] spreads among them: {len(fe)}, range {min(fe):.2f}-{max(fe):.2f} dex")
print("G. w(z) of the informational term, rho_info / rho_Lambda, CPL image, rates in three clocks (H0 67.4, Om 0.315 as the paper)")
Om2, OL2, H02 = 0.315, 0.685, 67.4; Hb = lambda a: np.sqrt(Om2*a**-3+OL2); bm2 = 0.3153/2; Gyr2 = 977.792/H02
for z in (0, 0.5, 1, 2, 3):
    a = 1/(1+z); print(f"   z {z}: w_info {-1-1/(3*a):.3f}  rho_info/rho_L {bm2*E(a)/OL2:.4f}  share of (Lambda + info) {bm2*E(a)/(OL2+bm2*E(a)):.3f}")
print(f"   saturation: rho_info/rho_L -> {bm2*np.e/OL2:.3f}")
ra = minimize_scalar(lambda a: -E(a)/a**2, bounds=(0.05, 5), method="bounded").x
rl = minimize_scalar(lambda a: -E(a)/a, bounds=(0.05, 5), method="bounded").x
rt = minimize_scalar(lambda a: -E(a)/a*Hb(a), bounds=(0.05, 5), method="bounded").x
tb = lambda a: quad(lambda x: 1/(x*Hb(x)), 1e-8, a, limit=200)[0]*Gyr2
print(f"   dE/da peaks a {ra:.3f}; dE/dln a peaks a {rl:.3f}; dE/dt peaks a {rt:.3f} (z {1/rt-1:.2f}, {tb(1)-tb(rt):.1f} Gyr ago), "
      f"{E(rt)/rt*Hb(rt)/np.e/Gyr2*100:.2f} %/Gyr; today {Hb(1)/np.e/Gyr2*100:.3f} %/Gyr")
for nm, w0, s0, wa, sa in (("DESI+CMB+Pantheon+", -0.838, 0.055, -0.62, 0.21), ("DESI+CMB+Union3", -0.667, 0.088, -1.09, 0.29), ("DESI+CMB+DESY5", -0.752, 0.057, -0.86, 0.22)):
    print(f"   {nm}: w0 offset {abs(-4/3-w0)/s0:.1f} sigma; wa offset {abs(-1/3-wa)/sa:.1f} sigma")
print(f"   H_inf = {H02*np.sqrt(OL2):.2f}; H_m,inf = {H02*np.sqrt(OL2+bm2*np.e):.2f}")
print("H. E_G, CMB lensing, and the remaining arithmetic printed in the chapters (Omega_m 0.3153)")
for z in (0.0, 0.295, 0.3, 0.5, 1.0):
    a = 1/(1+z); print(f"   E_G = Omega_m0 Sigma / f, change at z {z}: {100*(f(L,a)/f(I,a)-1):+.2f} %")
zsrc = 1089; chi = lambda z: quad(lambda x: 1/np.sqrt(H2(1/(1+x))), 0, z)[0]; cs = chi(zsrc)
zg = np.linspace(0.02, 10, 500); ch = np.array([chi(z) for z in zg]); W = ((cs-ch)/cs*(1+zg))**2/np.sqrt(H2(1/(1+zg)))
rat = np.array([(D(I, 1/(1+z))/D(L, 1/(1+z)))**2 for z in zg])
print(f"   CMB lensing C_phiphi (Limber, kernel (chi_s-chi)/chi_s, power ~ D^2) IAM/LCDM = {trap(W*rat, zg)/trap(W, zg):.4f}")
Hm, Hg = 72.26, 67.16
print(f"   sirens: separation {Hm-Hg:.2f}; 3 sigma needs sigma(H0) <= {(Hm-Hg)/3:.2f} = {100*(Hm-Hg)/3/Hm:.1f} % of 72.26; SH0ES (73.04 +/- 1.04) offset {(73.04-Hm)/1.04:.2f} sigma")
print(f"   Level 2 sigma8 0.8087 -> 0.7998: {100*(0.7998/0.8087-1):.2f} %, {(0.8087-0.7998)/0.0059:.2f} sigma; S8 0.822 +/- 0.011 vs Planck 0.832: {(0.832-0.822)/0.011:.2f} sigma")
r_sat = bm2*np.e/OL2
print(f"   w(z) chapter: saturation share rho_info/(rho_L + rho_info) = {r_sat/(1+r_sat):.3f}; H_m/H today {np.sqrt(Hb(1)**2+bm2*E(1))/Hb(1):.3f}, a -> inf {np.sqrt(OL2+bm2*np.e)/np.sqrt(OL2):.3f}")
print(f"   Press-Schechter: |Delta ln n| = |nu^2 - 1| x 0.78 % < 0.78 % for nu < sqrt(2) = {np.sqrt(2):.3f}")
