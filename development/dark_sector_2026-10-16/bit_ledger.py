"""Dark-sector bit ledger (author question 2026-10-16, downtime; exploration, not a test). Planck 2018 (TT,TE,EE+lowE+lensing) and CODATA 2018.
For today, the dark-energy/matter crossover and the CMB: energy fractions, the Hubble horizon's bits (area / 4 ln2 Planck areas), and the energy
inside the horizon per bit against one Landauer cost k T ln2 at the horizon temperature (de Sitter, Gibbons-Hawking)."""
import numpy as np
from scipy.integrate import quad
c=2.99792458e8; G=6.67430e-11; hbar=1.054571817e-34; kB=1.380649e-23; Mpc=3.0856775814913673e22; yr=3.15576e7
H0=67.36e3/Mpc; h=0.6736; Ob=0.02237/h**2; Oc=0.1200/h**2; Om=Ob+Oc; Og=2.4728e-5/h**2; Or=Og*(1+0.2271*3.046); OL=1-Om-Or
lP2=hbar*G/c**3; ln2=np.log(2); E=lambda z: np.sqrt(Or*(1+z)**4+Om*(1+z)**3+OL)
def era(z):
    H=H0*E(z); u=3*H**2/(8*np.pi*G)*c**2; R=c/H; N=4*np.pi*R**2/(4*lP2)/ln2; Mc2=u*4/3*np.pi*R**3
    TGH=hbar*H/(2*np.pi*kB); TBH=hbar*c/(4*np.pi*R*kB); t=quad(lambda zz: 1/((1+zz)*H0*E(zz)),z,np.inf,limit=200)[0]/yr/1e9
    return dict(z=round(z,3),age_Gyr=t,DE=OL/E(z)**2,DM=Oc*(1+z)**3/E(z)**2,baryons=Ob*(1+z)**3/E(z)**2,radiation=Or*(1+z)**4/E(z)**2,u_J_m3=u,
                bits=N,energy_per_bit_J=Mc2/N,per_bit_over_kTdS_ln2=Mc2/N/(kB*TGH*ln2))
for z in (0.0,(OL/Om)**(1/3)-1,1089.92): print({k:(float(f"{v:.4g}") if isinstance(v,float) else v) for k,v in era(z).items()})
print("DE/matter today", round(OL/Om,4), "| observed DE / Planck-cutoff vacuum", f"{OL*3*H0**2/(8*np.pi*G)*c**2/(c**7/(hbar*G**2)):.4g}")
