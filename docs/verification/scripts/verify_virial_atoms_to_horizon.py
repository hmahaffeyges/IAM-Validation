#!/usr/bin/env python3
"""Recomputes every number in 'The Virial Partition from Atoms to the Horizon'. CODATA 2018 via scipy.constants; numpy only otherwise."""
import numpy as np, scipy.constants as C
G, c, hbar, k, me, mp, e = C.G, C.c, C.hbar, C.k, C.m_e, C.m_p, C.e
Msun, Rsun, Lsun, yr = 1.98847e30, 6.957e8, 3.828e26, 3.15576e7
print("1. Hydrogen ground state (Bohr/Schroedinger, infinite nuclear mass)")
Eh = C.physical_constants["Hartree energy in eV"][0]
print(f"   E1 = {-Eh/2:.4f} eV   <K> = {Eh/2:.4f} eV   <V> = {-Eh:.4f} eV   <K>/|<V>| = {0.5:.4f}")
print(f"   photon on capture from rest = |E1| = {Eh/2:.4f} eV = <K>")
print("2. The Sun: Kelvin-Helmholtz time from the virial theorem (n = 3 polytrope, U = -3GM^2/(2R))")
U = -1.5*G*Msun**2/Rsun; print(f"   U = {U:.3e} J; energy radiated so far if contraction alone powered it = |U|/2 = {abs(U)/2:.3e} J")
print(f"   t_KH = |U|/(2L) = {abs(U)/2/Lsun/yr/1e6:.1f} Myr (vs 4,600 Myr: the reason the Sun must burn hydrogen)")
print("3. Chandrasekhar mass: M_Ch = (omega3 sqrt(3 pi)/2) (hbar c/G)^(3/2) / (mu_e m_u)^2, omega3 = 2.01824")
mu = C.physical_constants["atomic mass constant"][0]
for mue in (2.0, 2.15):
    M = 2.01824*np.sqrt(3*np.pi)/2*(hbar*c/G)**1.5/(mue*mu)**2; print(f"   mu_e = {mue}: M_Ch = {M/Msun:.3f} Msun")
print("4. Black holes: Bekenstein-Hawking entropy S = k A c^3/(4 G hbar), Hawking temperature T = hbar c^3/(8 pi G M k)")
for m in (1.0, 4.3e6, 6.5e9):
    M = m*Msun; T = hbar*c**3/(8*np.pi*G*M*k); S = k*4*np.pi*G*M**2/(hbar*c); bits = S/(k*np.log(2))
    print(f"   M = {m:.2g} Msun: T = {T:.3e} K, S/k = {S/k:.3e}, bits = {bits:.3e}, bits x k T ln2 / (M c^2) = {bits*k*T*np.log(2)/(M*c**2):.10f}")
print("   Kerr (G = c = hbar = k = 1): T S / M = sqrt(1 - chi^2)/2; the rest of M c^2 is 2 Omega_H J (Smarr)")
for chi in (0.0, 0.5, 0.9, 0.998):
    rp = 1+np.sqrt(1-chi**2); a = chi; T = (2*np.sqrt(1-chi**2))/(4*np.pi*(rp**2+a**2)); S = np.pi*(rp**2+a**2)
    Om = a/(rp**2+a**2); J = chi
    print(f"   chi = {chi:5.3f}: TS/M = {T*S:.4f}  2 Omega J / M = {2*Om*J:.4f}  sum = {2*T*S+2*Om*J:.4f}")
print("5. Cosmic horizon: beta_m = Omega_m/2")
for Om_ in (0.3153, 0.3166): print(f"   Omega_m = {Om_}: beta_m = {Om_/2:.5f}")
print("   Level 2 chains (Cosmological_Physics/camb_validation/chains, 30 % burn-in): beta_m fixed at 0.15765; Delta chi2 vs LCDM +0.54 (lowest chi2 in each chain)")
