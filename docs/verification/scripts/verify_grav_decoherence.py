#!/usr/bin/env python3
"""'Gravitational Decoherence from Dual-Sector Thermodynamics' (Feb 2026): tau_IAM = hbar (kB T)^2 ln2 / E_G^3, E_G = G m^2/R, silica rho = 2000."""
import numpy as np, scipy.constants as C
from scipy.optimize import brentq
hb,kB,G=C.hbar,C.k,C.G; rho=2000.
EG=lambda m: G*m*m/((3*m/(4*np.pi*rho))**(1/3)); tI=lambda m,T: hb*(kB*T)**2*np.log(2)/EG(m)**3; tPD=lambda m: hb/EG(m)
m=1e-12; print(f"1. m = 1e-12 kg, 10 mK: tau_IAM = {tI(m,0.01):.1f} s  (paper text 560 us; Fig. 1 label 4687 us);  tau_PD = {tPD(m)*1e6:.2f} us")
ms=np.logspace(-15,-10,50); print(f"2. slopes: tau_IAM ~ m^{np.polyfit(np.log(ms),np.log(tI(ms,0.01)),1)[0]:.2f} (paper m^-6); tau_PD ~ m^{np.polyfit(np.log(ms),np.log(tPD(ms)),1)[0]:.3f}")
print(f"   crossover at 10 mK: {10**brentq(lambda l: np.log(tI(10**l,0.01)/tPD(10**l)),-20,-5):.2e} kg (paper 2.3e-10)")
print("3. tau(20 mK)/tau(10 mK) =", tI(m,0.02)/tI(m,0.01), "; tau(40)/tau(10) =", tI(m,0.04)/tI(m,0.01))
eta=np.linspace(1e-4,5,500001); print("4. dE_q/deta = exp(1-1/eta)/eta^2 peaks at eta =", round(eta[np.argmax(np.exp(1-1/eta)/eta**2)],4))
print("5. Eq. 6 as written: d ln E_q/dt = const = 1/tau  =>  E_q = exp(t/tau) (exponential growth), not exp(1 - tau/t)")
print(f"6. P_IAM = E_G^3/(hbar kB T) at 1e-12 kg, 10 mK = {EG(m)**3/(hb*kB*0.01):.2e} W")
