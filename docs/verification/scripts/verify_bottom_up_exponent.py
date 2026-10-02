#!/usr/bin/env python3
"""Bottom-up exponent n of the information-production rate, from measured halo mass functions (2026-10-02).
Theory chapter: dS_info/dt = I_dot/(T_H A_H), with 1/T_H the Landauer factor, so I_dot is an ENERGY rate. IAM's Law + virial partition: the energy a
halo writes is its virial kinetic half K ~ G M^2/R_vir ~ M^(5/3) (Delta_c rho_crit)^(1/3). I_dot (physical) = a^-3 dU/dt, U(a) = int dn/dlnM K dlnM.
Top-down form I_dot ~ rho_m D^n f H  =>  n_eff = d ln[(dU/dlna)/f] / d ln D.  No mass threshold (U converges at low M). Planck 2018 chain parameters.
Requires: pip install colossus."""
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
P18={'flat':True,'H0':67.36,'Om0':0.3153,'Ob0':0.0493,'sigma8':0.8111,'ns':0.9649,'relspecies':True}
c=cosmology.setCosmology('p18chains',params=P18,persistence='')
lna=np.linspace(np.log(0.02),0,240); a=np.exp(lna); z=1/a-1
lnM=np.linspace(np.log(1e4),np.log(1e17),320); M=np.exp(lnM)
E=c.Ez(z); D=c.growthFactor(z); f=np.gradient(np.log(D),lna); Om=c.Om(z); Dc=18*np.pi**2+82*(Om-1)-39*(Om-1)**2
print("model       n_eff at z = 9    5    4    3    2    1    0 | crosses 7/2 at z | 5/2 at z | mean z 2.3-9")
for model,mdef in (("press74","fof"),("sheth99","fof"),("tinker08","200m")):
    dn=np.array([mass_function.massFunction(M,zz,q_in='M',q_out='dndlnM',mdef=mdef,model=model) for zz in z])
    U=np.trapezoid(dn*M**(5/3)*((Dc*E**2)[:,None])**(1/3),lnM,axis=1); ne=np.gradient(np.log(np.gradient(U,lna)/f),np.log(D))
    x=lambda zz: np.interp(-zz,-z,ne); cr=lambda v: z[np.where(np.diff(np.sign(ne-v)))[0]]; m=(a>=0.1)&(a<=0.3)
    print(f"{model:9s}  "+" ".join(f"{x(q):5.2f}" for q in (9,5,4,3,2,1,0.05))+f" | {cr(3.5).round(2)} | {cr(2.5).round(2)} | {ne[m].mean():.2f}")
print("Analytic check (Press-Schechter, matter domination, mass-weighted, threshold M_min): n_eff = nu_min^2 - 1")
