#!/usr/bin/env python3
"""'Quantum Darwinism at Cosmological Scales' (Mar 2026): halo decoherence time, the n exponent algebra, BH bit rate, H_matter."""
import numpy as np, scipy.constants as C
hb,G,c,k=C.hbar,C.G,C.c,C.k; Ms=1.98847e30; kpc=3.0857e19
M=1e12*Ms; R=200*kpc; t=hb*R/(G*M*M); print(f"1. tau_D = hbar R/(G M^2), 1e12 Msun, 200 kpc: {t:.1e} s (paper 1e-70); Hubble time / tau = 1e{np.log10(4.35e17/t):.0f}")
print("2. Eq. 12 as printed: integrand exponents -3 + n - 3/2 + 1/2 - 1 = n - 5 -> S ~ a^(n-4); paper writes a^(n-9/2)")
print("   with the paper's a^(n-9/2): n - 9/2 = -1 gives n =", 9/2-1, "(paper prints 5/2)")
P=lambda m: hb*c**6/(15360*np.pi*G**2*m**2); T=lambda m: hb*c**3/(8*np.pi*G*m*k)
m=Ms; print(f"3. BH: P/(k T ln2) = {P(m)/(k*T(m)*np.log(2)):.4e} bits/s; c^3/(1920 G M ln2) = {c**3/(1920*G*m*np.log(2)):.4e}; radiation entropy rate (4/3)P/T is 4/3 of it")
print("4. H_matter = 67.16 sqrt(1 + 0.1575) =", round(67.16*np.sqrt(1.1575),2))
print("5. Bottom-up exactly (Press-Schechter, matter domination): I_dot = (rho_m/m_p) H dF/dln a, F = erfc(nu/sqrt2), nu = delta_c/(sigma(M_min) D)")
print("   => n_eff = d ln(dF/dln a)/d ln D = nu^2 - 1  (exact); 7/2 at nu = 2.121, 5/2 at nu = 1.871")
