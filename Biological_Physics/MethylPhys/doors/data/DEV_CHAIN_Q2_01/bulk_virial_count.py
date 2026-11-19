"""bulk_virial_count.py  (private, proposals only; 2026-11-18)

The book's own open calculation (p1_02:336): from the bulk, does structure formation write at the
required beta_m S_geo = 5.2e121 bits per e-fold today?  Under the book's rule:
  - records are written by virialization (p2_03:275-279; p1_02:586-588 'the energy a bound structure
    writes is its kinetic half, K = |W|/2');
  - each record priced at k T_H ln 2 at the HORIZON temperature (eq:Sdot p1_02:358, eq:dSdt p2_03:302);
  - counted over the horizon (Hubble) volume.
Written energy per e-fold = d K_com / d ln a  (K_com = kinetic energy of halos per comoving volume;
uniform sphere K/M = (3/10) G M / R_vir as the book uses at p2_12b:88).
Mass functions: Sheth-Tormen 1999 and Press-Schechter 1974; Eisenstein-Hu 1998 no-wiggle transfer;
Planck 2018 parameters as the book (p2_03:613): h=0.674, Omega_b h^2=0.0224, sigma8=0.811, n_s=0.965,
Omega_m = 0.3153.  R_vir from Bryan-Norman 1998 Delta_c (relative to rho_crit(z)).
Cross-check of the stock: Fukugita & Peebles 2004 (ApJ 616, 643) Table 1 entry 4, primeval gravitational
binding energy Omega = -10^-6.1 (U = K + W = W/2), entries 4.1 + 4.2 (halos of galaxies, clusters) = -10^-7.2 - 10^-6.9.
"""
import numpy as np
from scipy.integrate import quad, simpson

# constants (SI)
c, hb, kB, G = 299792458.0, 1.054571817e-34, 1.380649e-23, 6.67430e-11
Mpc, Msun = 3.0856775814913673e22, 1.98847e30
h, Om, Obh2, s8, ns = 0.674, 0.3153, 0.0224, 0.811, 0.965
Ob = Obh2/h**2
Or = 9.1e-5
OL = 1 - Om - Or
H0 = 100*h*1e3/Mpc
H0_book = 67.16e3/Mpc        # book p1_02:335 value for the required rate
rho_c0 = 3*H0**2/(8*np.pi*G)
rho_m0 = Om*rho_c0           # comoving matter density, kg/m^3

def Ez(a): return np.sqrt(Om/a**3 + Or/a**4 + OL)

# ---- Eisenstein & Hu 1998 no-wiggle transfer function (eqs 26-31) ----
def T_nw(k_hMpc):
    k = k_hMpc*h                      # 1/Mpc
    omh2, fb = Om*h**2, Ob/Om
    theta = 2.7255/2.7
    s = 44.5*np.log(9.83/omh2)/np.sqrt(1 + 10*(Obh2)**0.75)
    aG = 1 - 0.328*np.log(431*omh2)*fb + 0.38*np.log(22.3*omh2)*fb**2
    Gam = Om*h*(aG + (1 - aG)/(1 + (0.43*k*s)**4))
    q = k_hMpc*theta**2/Gam
    L0 = np.log(2*np.e + 1.8*q)
    C0 = 14.2 + 731/(1 + 62.5*q)
    return L0/(L0 + C0*q**2)

lnk = np.linspace(np.log(1e-5), np.log(1e4), 6000)
kk = np.exp(lnk)
Pk_shape = kk**ns*T_nw(kk)**2       # (h/Mpc units)

def W_th(x):
    x = np.asarray(x)
    out = np.where(x < 1e-3, 1 - x**2/10, 3*(np.sin(x) - x*np.cos(x))/np.maximum(x, 1e-30)**3)
    return out

def sigma2_shape(R):  # R in Mpc/h
    return simpson(kk**3*Pk_shape*W_th(kk*R)**2/(2*np.pi**2), x=lnk)

norm = s8**2/sigma2_shape(8.0)

# growth factor, normalised D(1)=1 (integral form, matter+Lambda; radiation negligible for a>0.1)
def D_unnorm(a):
    I = quad(lambda x: 1/(x*Ez(x))**3, 1e-8, a, limit=200)[0]
    return 2.5*Om*Ez(a)*I
D1 = D_unnorm(1.0)
def D(a): return D_unnorm(a)/D1

# mass grid (Msun/h)
lnM = np.linspace(np.log(1e8), np.log(1e16), 400)
M_h = np.exp(lnM)
rho_m0_h = rho_m0/(Msun/h)*(Mpc/h)**3          # (Msun/h)/(Mpc/h)^3
R_L = (3*M_h/(4*np.pi*rho_m0_h))**(1/3)         # Mpc/h
sig0 = np.sqrt(np.array([sigma2_shape(R)*norm for R in R_L]))
dlnsig_dlnM = np.gradient(np.log(sig0), lnM)
dc = 1.686

def f_ST(nu, A=0.3222, q=0.707, p=0.3):
    return A*np.sqrt(2*q/np.pi)*nu*(1 + (q*nu**2)**(-p))*np.exp(-q*nu**2/2)
def f_PS(nu):
    return np.sqrt(2/np.pi)*nu*np.exp(-nu**2/2)

def Delta_BN(a):
    x = (Om/a**3)/Ez(a)**2 - 1
    return 18*np.pi**2 + 82*x - 39*x**2

def K_comoving(a, fmf):
    """kinetic energy of halos per comoving volume (J/m^3), uniform sphere K = (3/10) G M^2 / R_vir"""
    nu = dc/(sig0*D(a))
    dndlnM = (rho_m0_h/M_h)*fmf(nu)*np.abs(dlnsig_dlnM)      # per (Mpc/h)^3 comoving
    M_SI = M_h*Msun/h
    rho_cz = rho_c0*Ez(a)**2
    Rvir = (3*M_SI/(4*np.pi*Delta_BN(a)*rho_cz))**(1/3)       # physical m
    K = 0.3*G*M_SI**2/Rvir
    return simpson(dndlnM*K, x=lnM)/(Mpc/h)**3, simpson(dndlnM*M_h, x=lnM)/rho_m0_h

def required_bits(a):   # book eq:law_Sneed general a, H0 = 67.16 as at p1_02:335 (book form, count doc 1.1)
    Hb = H0_book*np.sqrt(Om/a**3 + Or/a**4 + (1-Om-Or))
    bm = Om/2
    return (np.pi*c**5/(hb*G))*bm*H0_book**2*np.exp(1 - 1/a)/(a*Hb**4)/np.log(2)

def bulk_bits(a, fmf, dlna=0.02):
    Kp, _ = K_comoving(a*np.exp(dlna), fmf)
    Km, _ = K_comoving(a*np.exp(-dlna), fmf)
    dK = (Kp - Km)/(2*dlna)                        # J per comoving m^3 per e-fold
    Ha = H0*Ez(a)
    Vcom = 4*np.pi/3*(c/Ha)**3/a**3                # horizon volume in comoving m^3
    T = hb*Ha/(2*np.pi*kB)
    Q = dK*Vcom
    return Q/(kB*T*np.log(2)), Q, T, Vcom, dK

if __name__ == "__main__":
    # sanity: mass-function normalisation and collapsed fraction (book p2_03:615: f_coll(ST, >1e6) = 0.64)
    for name, fmf in (("ST", f_ST), ("PS", f_PS)):
        K0, fcoll = K_comoving(1.0, fmf)
        print(f"{name}: z=0 f_coll(>1e8 Msun/h) = {fcoll:.3f}; K_com/rho_crit c^2 = {K0/(rho_c0*c**2):.3e}")
    print(f"Fukugita-Peebles 2004 entry 4 |Omega_BE| = 10^-6.1 = {10**-6.1:.3e}; 4.1+4.2 = {10**-7.2+10**-6.9:.3e}")
    rows = []
    for z in (0.0, 1.0, 2.0):
        a = 1/(1+z)
        req = required_bits(a)
        out = {"z": z, "req": req}
        for name, fmf in (("ST", f_ST), ("PS", f_PS)):
            b, Q, T, V, dK = bulk_bits(a, fmf)
            out[name] = b; out[name+"_Q"] = Q; out["T"] = T
        rows.append(out)
        print(f"z={z:.0f}: T_GH={out['T']:.3e} K; required={req:.3e} bits/e-fold; "
              f"ST bulk={out['ST']:.3e} (ratio {out['ST']/req:.2e}; Q={out['ST_Q']:.2e} J); "
              f"PS bulk={out['PS']:.3e} (ratio {out['PS']/req:.2e})")
    # Fukugita-Peebles stock turned into a rate: per e-fold ~ |Omega_BE| rho_c c^2 V_H x (dlnK/dlna today from ST)
    a = 1.0
    K0, _ = K_comoving(1.0, f_ST)
    _, _, T0, V0, dK0 = bulk_bits(1.0, f_ST)
    slope = dK0/K0
    for lab, Omg in (("FP entry 4 (total)", 10**-6.1), ("FP 4.1+4.2 (halos)", 10**-7.2+10**-6.9)):
        Q = Omg*rho_c0*c**2*V0*slope
        print(f"{lab}: x dlnK/dlna(ST)={slope:.2f} -> {Q/(kB*T0*np.log(2)):.3e} bits/e-fold; ratio {Q/(kB*T0*np.log(2))/required_bits(1.0):.2e}")
    np.save("bulk_rows.npy", rows, allow_pickle=True)
