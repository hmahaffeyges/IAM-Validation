#!/usr/bin/env python3
"""Recomputes 'Electron Rest Mass from Holographic Horizon Thermodynamics' (Feb 2026) Eq. 14. CODATA via scipy.constants."""
import numpy as np, scipy.constants as C
hbar,c,G,al,me=C.hbar,C.c,C.G,C.alpha,C.m_e; mP=np.sqrt(hbar*c/G); Mpc=3.0856775814913673e22
def m(H0, pref=(2*np.pi)**-0.1): return pref*(hbar*(H0*1e3/Mpc)*np.log(2)*mP**1.5/(al**2.5*c**2))**0.4
print("1. Eq. 14 with H0 = 67.4:", f"{m(67.4):.6e} kg  vs CODATA {me:.6e}  dev {1e6*(m(67.4)/me-1):+.1f} ppm")
print("   without the identified prefactor: ratio", round(m(67.4,1.0)/me,6), " -> prefactor needed", round(me/m(67.4,1.0),6), " (2pi)^-1/10 =", round((2*np.pi)**-0.1,6))
for H in (67.36,67.4,67.9,70.0,73.04): print(f"   H0 {H}: dev {1e6*(m(H)/me-1):+8.0f} ppm")
print("   Planck sigma(H0) 0.54 -> sigma(m_e)/m_e =", f"{0.4*0.54/67.36*1e6:.0f} ppm")
print("   H0 that makes Eq. 14 exact:", round(67.4*(me/m(67.4))**2.5,4))
print("2. Black hole: bits x k T_H ln2 / (M c^2) =", 0.5, "(Smarr; T_H S = Mc^2/2), so mc^2 = E_bit x N does not hold for the black hole; it holds with 1/2")
