"""candidates_compare.py  (private, proposals only; 2026-11-18)

Four candidate bookings of the record term, side by side, against the required rate
  dS_info/dlna = (pi c^5/hbar G) beta_m H0^2 E(a) / (a H^4)  [book eq:law_Sneed, p1_02:330-335; count doc 1.1]
  = 5.19e121 bits per e-fold today,
today and at z = 1, 2.  Each candidate is computed from published inputs with no fitted factor;
the bars were not changed after the numbers came in.  Every arithmetic step is in this file
(the book-rule halo sum lives in bulk_virial_count.py, imported here).

Inputs (all web-verified 2026-11-18 or read in the count document's sources):
  Sheth & Tormen 1999 / Press & Schechter 1974 mass functions, Eisenstein & Hu 1998 no-wiggle P(k),
    Planck 2018 parameters as the book uses them (p2_03:613), Bryan & Norman 1998 Delta_vir.
  Fukugita & Peebles 2004, ApJ 616, 643, Table 1: entry 4 primeval gravitational binding energy
    Omega = -10^-6.1 (U = K + W = W/2); 4.1 halos of galaxies -10^-7.2; 4.2 clusters -10^-6.9.
  Driver et al. 2012, MNRAS 427, 3244: intrinsic energy output 1.8e35 h W Mpc^-3 (z < 0.1).
  Madau & Dickinson 2014, ARA&A 52, 415, eq. 15: psi(z) = 0.015 (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6) Msun/yr/Mpc^3;
    BHAR = psi/3300 (their comparison figure, as read by the count document).
  Shankar et al. 2004: radiative efficiency ~0.09 (Soltan argument); 0.1 used as the round value.
  The count document (BOTTOM_UP_COUNT_2026-11-18.md, sections 3-4): count-booking ratios, quoted not recomputed.
"""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bulk_virial_count as B

c, hb, kB, G = B.c, B.hb, B.kB, B.G
Mpc, Msun, yr = B.Mpc, B.Msun, 3.15576e7
H0b = B.H0_book                      # 67.16, the book's value for the required rate
hb_ = 0.6716
Om, Or = B.Om, B.Or

def Eb(a): return np.sqrt(Om/a**3 + Or/a**4 + (1 - Om - Or))
def Hb(a): return H0b*Eb(a)
def T_GH(a): return hb*Hb(a)/(2*np.pi*kB)
def Vcom_H(a): return 4*np.pi/3*(c/Hb(a))**3/a**3          # comoving m^3 inside the apparent horizon
def required(a): return B.required_bits(a)
def to_bits(Q_J, a): return Q_J/(kB*T_GH(a)*np.log(2))
def psi(z): return 0.015*(1+z)**2.7/(1 + ((1+z)/2.9)**5.6)   # Msun/yr/Mpc^3 comoving

zs = (0.0, 1.0, 2.0)
res = {}

# (1) the book's rule: written energy per e-fold = d K_com/dlna (K = |W|/2), priced at k T_GH ln2.
#     (1i) amplitude and shape as totals over the horizon volume;
#     (1ii) shape of eq:Sdot taken literally (per unit horizon area, density in), normalised to (1i) at a = 1.
for mf in ("ST", "PS"):
    f = B.f_ST if mf == "ST" else B.f_PS
    r_i, shape_ii = [], []
    for z in zs:
        a = 1/(1+z)
        bits, Q, T, V, dK = B.bulk_bits(a, f)
        r_i.append(bits/required(a))
        AH = 4*np.pi*(c/(B.H0*B.Ez(a)))**2
        shape_ii.append((dK/a**3)/(T*AH))
    N0 = r_i[0]*required(1.0)
    r_ii = [N0*shape_ii[k]/shape_ii[0]/required(1/(1+zs[k])) for k in range(3)]
    res[f"(1) book rule, {mf}, horizon-volume total"] = r_i
    res[f"(1) book rule, {mf}, eq:Sdot per-area shape"] = r_ii
# NFW bracket for the binding (Mo, Mao & White 1998 f(c)), c = 5 to 10, relative to the uniform sphere 3/5
fW = lambda cc: cc*(1 - 1/(1+cc)**2 - 2*np.log(1+cc)/(1+cc))/(2*(cc/(1+cc) - np.log(1+cc))**2)
nfw_lo, nfw_hi = fW(5)/0.6, fW(10)/0.6
# Fukugita-Peebles stock cross-check (today only): K_stock/rho_c c^2 vs our ST stock
K0_ST, _ = B.K_comoving(1.0, B.f_ST)
stock_ST = K0_ST/(B.rho_c0*c**2)
fp_halo, fp_total = 10**-7.2 + 10**-6.9, 10**-6.1
res["(1) FP2004 halos 4.1+4.2 (stock x our dlnK/dlna)"] = [res["(1) book rule, ST, horizon-volume total"][0]*fp_halo/stock_ST, np.nan, np.nan]
res["(1) FP2004 entry 4 total (stock x our dlnK/dlna)"] = [res["(1) book rule, ST, horizon-volume total"][0]*fp_total/stock_ST, np.nan, np.nan]

# (2) plain count of records (count document, quoted): low / high today; BH alpha=1 / alpha=2 at z=1, 2
res["(2) count booking, low (quoted)"] = [8.0e-20, 5.1e-18, 4.7e-17]
res["(2) count booking, high/alt (quoted)"] = [2.9e-19, 3.9e-18, 2.7e-17]

# (3) starlight and black-hole heat at k T_GH ln2
L0 = 1.8e35*hb_/Mpc**3                        # W per comoving m^3 today (Driver 2012)
r_star, r_bh, r_bh_rad = [], [], []
for z in zs:
    a = 1/(1+z)
    L = L0*psi(z)/psi(0)                       # luminosity density tracks psi (count document's scaling)
    Q = L*Vcom_H(a)/Hb(a)                      # J per e-fold in the horizon volume
    r_star.append(to_bits(Q, a)/required(a))
    bhar = psi(z)/3300*Msun/yr/Mpc**3          # kg/s per comoving m^3
    Qbh = bhar*c**2*Vcom_H(a)/Hb(a)            # accreted rest energy per e-fold
    r_bh.append(to_bits(Qbh, a)/required(a))
    r_bh_rad.append(to_bits(0.1*Qbh, a)/required(a))
res["(3) starlight at kT_GH ln2"] = r_star
res["(3) BH accreted rest energy at kT_GH ln2"] = r_bh
res["(3) BH radiated (eps=0.1) at kT_GH ln2"] = r_bh_rad

# (4) half the matter's rest energy inside the apparent horizon at k T_GH (identity at a = 1)
r_half = []
for z in zs:
    a = 1/(1+z)
    Mm = Om*(3*H0b**2/(8*np.pi*G))*a**-3*4*np.pi/3*(c/Hb(a))**3
    r_half.append(to_bits(0.5*Mm*c**2, a)/required(a))
res["(4) (1/2) M_m c^2 at kT_GH per e-fold"] = r_half
# its shape: required / (1/2 M_m c^2 / kT) = E(a) a^2 (ledger_half_checks C3) -> ratio = 1/(E a^2)
for k, z in enumerate(zs):
    a = 1/(1+z)
    assert abs(r_half[k]*np.exp(1 - 1/a)*a**2 - 1) < 2e-3, (z, r_half[k])

if __name__ == "__main__":
    print(f"required bits/e-fold: z=0 {required(1.0):.4e}, z=1 {required(0.5):.4e}, z=2 {required(1/3):.4e}")
    print(f"ST stock K/rho_c c^2 = {stock_ST:.3e}; FP halos 4.1+4.2 = {fp_halo:.3e}; FP entry 4 = {fp_total:.3e}")
    print(f"NFW binding / uniform sphere: c=5 {nfw_lo:.2f}, c=10 {nfw_hi:.2f}")
    print(f"{'candidate':58s} {'z=0':>10s} {'z=1':>10s} {'z=2':>10s}   shape r(1)/r(0), r(2)/r(0)  (E(a) shape needs 1, 1)")
    for k, v in res.items():
        s1 = v[1]/v[0] if np.isfinite(v[1]) else np.nan
        s2 = v[2]/v[0] if np.isfinite(v[2]) else np.nan
        print(f"{k:58s} {v[0]:10.2e} {v[1]:10.2e} {v[2]:10.2e}   {s1:8.3g} {s2:8.3g}")
    import json
    json.dump({k: [float(x) for x in v] for k, v in res.items()} | {"nfw_factor": [nfw_lo, nfw_hi], "stock_ST": stock_ST},
              open("candidates_results.json", "w"), indent=1)
