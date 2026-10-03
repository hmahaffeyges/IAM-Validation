#!/usr/bin/env python3
"""Recomputes 'Dark Energy or Sector Tension? IAM Confrontation with DESI Full-Shape Growth Rates and Joint Weak Lensing' (Mar 2026).
Exact coupling mu = H^2/(H^2 + beta_m E(a)), beta_m = Om/2, LCDM background, Sigma = 1. numpy, scipy."""
import numpy as np
from scipy.integrate import solve_ivp, quad
Om,OL=0.3153,0.6847; bm=Om/2
E=lambda a: np.exp(1-1/a); H2=lambda a: Om*a**-3+OL; mu=lambda a: H2(a)/(H2(a)+bm*E(a)); Oma=lambda a: Om*a**-3/H2(a)
def grow(m):
    def r(l,y):
        a=np.exp(l); return [y[1],-(2-1.5*Om*a**-3/H2(a))*y[1]+1.5*Oma(a)*m(a)*y[0]]
    return solve_ivp(r,(np.log(1e-3),0),[1e-3,1e-3],dense_output=True,rtol=1e-10,atol=1e-14)
L=grow(lambda a:1.0); I=grow(mu)
D=lambda S,a: S.sol(np.log(a))[0]; f=lambda S,a: S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
print("1. DESI DR2 (arXiv:2503.14738, eqs. 25-27) w0wa and crossing redshift a_x = 1 + (1+w0)/wa")
for nm,w0,wa in (("DESI+CMB",-0.42,-1.75),("DESI+CMB+Pantheon+",-0.838,-0.62),("DESI+CMB+Union3",-0.667,-1.09),("DESI+CMB+DESY5",-0.752,-0.86)):
    ax=1+(1+w0)/wa; print(f"   {nm:20s} w0 {w0:+.3f} wa {wa:+.2f}  z_cross {1/ax-1:.2f}")
print("   paper Table 5 labels: 'BAO+CMB' -0.838/-0.62 is the Pantheon+ row; Union3 -0.752/-0.82 and DESY5 -0.734/-1.05 match no DR2 row")
print("2. at the DESI DR1 tracer redshifts: 1 - mu (paper's 'suppression') vs the actual fsigma8 deficit (same early amplitude)")
for nm,z in (("BGS",0.295),("LRG1",0.510),("LRG2",0.706),("LRG3",0.919),("ELG2",1.317),("QSO",1.491),("today",0.0)):
    a=1/(1+z); r=f(I,a)*D(I,a)/(f(L,a)*D(L,a))
    print(f"   {nm:5s} z {z:5.3f}: 1-mu {100*(1-mu(a)):5.2f} %   fsigma8 deficit {100*(1-r):5.2f} %   sigma8(z) deficit {100*(1-D(I,a)/D(L,a)):4.2f} %   E_G change {100*(f(L,a)/f(I,a)-1):+5.2f} %")
print("3. CMB lensing C_phiphi (Limber, kernel (chi_s-chi)/chi_s, power ~ D^2): exact-form ratio")
zs=1089; c=1.0; chi=lambda z: quad(lambda x: 1/np.sqrt(H2(1/(1+x))),0,z)[0]; cs=chi(zs)
zg=np.linspace(0.02,10,500); ch=np.array([chi(z) for z in zg]); W=((cs-ch)/cs*(1+zg))**2/np.sqrt(H2(1/(1+zg)))
rat=np.array([(D(I,1/(1+z))/D(L,1/(1+z)))**2 for z in zg]); print(f"   C_phiphi IAM/LCDM ~ {np.trapezoid(W*rat,zg)/np.trapezoid(W,zg):.4f}  (paper: 'no change')")
print("4. H_matter = 67.16 sqrt(1 + beta_m) =", round(67.16*np.sqrt(1+0.15765),2), "; SH0ES offset", round((73.04-72.26)/1.04,2), "sigma")
print("5. beta_gamma/beta_m: paper < 8.5e-6 (from the 1.4e-6 bound, reversed-array bug); corrected beta_gamma < 0.0039 gives", round(0.0039/0.15765,4))

# ---------------------------------------------------------------------------------------------------------------
# Book carriage (2026-10-03): every remaining number of the chapter, recomputed. Sources read in full for the values:
# DESI 2024 V (JCAP 2025 09 008, arXiv:2411.12021) Appendix A eqs. A.1-A.12 (ShapeFit-only datavectors, Gaussian covariance);
# eBOSS DR16 cosmology (Alam et al. 2021, PRD 103 083533) Table III; KiDS-Legacy joint analysis (Stolzner et al. 2025, A&A 702 A169).
import sympy as sp
print("\n=== Book carriage checks ===")
b,Om_s=sp.symbols('beta_m Omega_m',positive=True)
mu_today=1/(1+b)                                    # mu(a=1): H^2 = H0^2 (Om+OL) = H0^2, E(1) = 1
print("6. sympy: mu(1) =",mu_today," mu0 = mu(1)-1 =",sp.simplify(mu_today-1)," at beta = Om/2, Om = 0.3153:",
      round(float((mu_today-1).subs(b,0.3153/2)),5), " mu(1) =",round(float(mu_today.subs(b,0.3153/2)),5))
w0,wa,a=sp.symbols('w0 wa a')
ax=sp.solve(sp.Eq(w0+wa*(1-a),-1),a)[0]; print("   sympy: w(a_x) = -1 gives a_x =",ax)
print("   E(a) < 0.1 for z >", round(float(1/(1/(1+sp.log(10)))-1),3))
print("7. Table of mu at the DESI DR1 tracer redshifts (1 - mu, fsigma8 deficit, sigma8(z) deficit)")
for nm,z in (("BGS",0.295),("LRG1",0.510),("LRG2",0.706),("LRG3",0.919),("ELG2",1.317),("QSO",1.491)):
    aa=1/(1+z); print(f"   {nm:5s} z {z}: mu {mu(aa):.4f}")
# DESI 2024 V Appendix A: f sigma_s8 and its variance C33 (units 1e-4); D_V/r_d and its variance C11
dsv={"BGS":(0.295,0.377174,88.510877,7.788174,1314.664401),"LRG1":(0.510,0.513635,41.295470,12.514437,541.309833),
     "LRG2":(0.706,0.483623,28.119682,15.675560,762.457717),"LRG3":(0.919,0.422164,22.370314,19.676985,847.499793),
     "ELG2":(1.317,0.376715,13.997473,23.861806,2342.506886),"QSO":(1.491,0.434858,19.785658,25.708520,None)}
leg={"6dFGS":(0.067,0.423,0.055),"MGS":(0.15,0.53,0.16),"BOSS z0.38":(0.38,0.500,0.047),"BOSS z0.51":(0.51,0.455,0.039),
     "eBOSS LRG":(0.70,0.448,0.043),"eBOSS ELG":(0.85,0.315,0.095),"eBOSS QSO":(1.48,0.462,0.045)}
def grow4(m):   # the chapter's integration: a_i = 1e-4, D = a, dD/dlna = a (growing mode), no radiation
    def r(l,y):
        aa=np.exp(l); return [y[1],-(2-1.5*Om*aa**-3/H2(aa))*y[1]+1.5*Oma(aa)*m(aa)*y[0]]
    return solve_ivp(r,(np.log(1e-4),0),[1e-4,1e-4],dense_output=True,method="DOP853",rtol=1e-11,atol=1e-14)
L4=grow4(lambda aa:1.0); I4=grow4(mu)
def fs8(S,z,s8): aa=1/(1+z); y=S.sol(np.log(aa)); return y[1]/y[0]*s8*y[0]/S.sol(0.0)[0]
print("8. sigma8 deficit from the ODE (same early amplitude):",round(100*(1-D(I4,1)/D(L4,1)),2),"%  ->  0.8087 x ratio =",round(0.8087*D(I4,1)/D(L4,1),4),
      "; Level 2 Boltzmann 0.7998/0.8087 =",round(100*(1-0.7998/0.8087),2),"%")
print("9. fsigma8 predictions, paper method f sigma8 D(z)/D(0) with sigma8 = 0.7998 (IAM, Run A) and 0.8087 (LCDM, Run C); pulls (obs - pred)/sigma")
c2I=c2L=0
for nm,(z,v,c33,dv,c11) in dsv.items():
    s=np.sqrt(c33*1e-4); pI=fs8(I4,z,0.7998); pL=fs8(L4,z,0.8087); c2I+=((v-pI)/s)**2; c2L+=((v-pL)/s)**2
    print(f"   {nm:5s} z {z}: f sigma_s8 {v:.3f} +- {s:.3f}   IAM {pI:.3f}  LCDM {pL:.3f}   pull {(v-pI)/s:+.2f} {(v-pL)/s:+.2f}")
print(f"   DESI six bins, diagonal chi2: IAM {c2I:.2f}  LCDM {c2L:.2f}")
c2I=c2L=0
for nm,(z,v,s) in leg.items():
    pI=fs8(I4,z,0.7998); pL=fs8(L4,z,0.8087); c2I+=((v-pI)/s)**2; c2L+=((v-pL)/s)**2
    print(f"   {nm:10s} z {z}: {v:.3f} +- {s:.3f}   IAM {pI:.3f}  LCDM {pL:.3f}   pull {(v-pI)/s:+.2f} {(v-pL)/s:+.2f}")
print(f"   legacy seven points, diagonal chi2: IAM {c2I:.2f}  LCDM {c2L:.2f}")
print("10. D_V/r_d, LCDM (Planck 2018: h 0.6736, Om 0.3153, r_d 147.09 Mpc) against DESI 2024 V Appendix A (ShapeFit-only)")
cH=299792.458/(67.36)
def DV(z):
    dm=cH*quad(lambda x:1/np.sqrt(H2(1/(1+x))),0,z)[0]; dh=cH/np.sqrt(H2(1/(1+z))); return (z*dm*dm*dh)**(1/3)
for nm,(z,v,c33,dv,c11) in dsv.items():
    p=DV(z)/147.09; print(f"   {nm:5s} z {z}: D_V/r_d DESI {dv:.2f}" + (f" +- {np.sqrt(c11*1e-4):.2f}" if c11 else "") + f"   LCDM {p:.2f}")
print("11. weak lensing against the Level 2 IAM S8 = 0.822 +- 0.011 (error on the side facing 0.822, in quadrature)")
for nm,v,up,dn in (("KiDS-1000 3x2pt (Heymans 2021)",0.766,0.020,0.014),("KiDS-1000 shear (Asgari 2021)",0.759,0.024,0.021),
                   ("KiDS-Legacy shear (Wright 2025)",0.815,0.016,0.021),("DES Y3 3x2pt",0.776,0.017,0.017),("HSC Y3 C_ell (Dalal 2023)",0.776,0.032,0.033),
                   ("KiDS-Legacy+DES Y3 shear+Pantheon+ +DESI Y1 BAO S8 (Stolzner 2025)",0.814,0.011,0.012),("LCDM Run C",0.830,0.011,0.011)):
    e=up if v<0.822 else dn; print(f"   {nm}: {v} -> {(v-0.822)/np.hypot(e,0.011):+.2f} sigma")
print("   sigma8 joint 0.802 (+0.022 -0.018) vs 0.7998 +- 0.0058:", round((0.802-0.7998)/np.hypot(0.018,0.0058),2),"sigma")
print("   sigma8 Om^0.25 (Run A) =",round(0.7998*0.3166**0.25,3)," vs Planck CMB lensing 0.589 +- 0.020:",round((0.7998*0.3166**0.25-0.589)/0.020,2),"sigma")
print("12. phantom-crossing redshifts of the DR2 fits and the coupling there")
for nm,w0_,wa_ in (("DESI+CMB",-0.42,-1.75),("+Pantheon+",-0.838,-0.62),("+Union3",-0.667,-1.09),("+DES Y5",-0.752,-0.86)):
    zx=1/(1+(1+w0_)/wa_)-1; aa=1/(1+zx); print(f"   {nm:10s} z_x {zx:.2f}: 1-mu {100*(1-mu(aa)):.1f} %  fsigma8 deficit {100*(1-f(I,aa)*D(I,aa)/(f(L,aa)*D(L,aa))):.2f} %")
print("13. beta_m consistency (fixed at 0.15765): Run A posterior Om/2 =",0.3166/2,"+-",0.0065/2," offset",round((0.3166/2-0.15765)/(0.0065/2),2),"sigma")
print("14. hypothetical 1 % fsigma8 per bin, 15 bins of width 0.1 over 0.1 < z < 1.6: IAM vs LCDM separation")
zc=np.arange(0.15,1.6,0.1); dd=np.array([100*(1-f(I,1/(1+z))*D(I,1/(1+z))/(f(L,1/(1+z))*D(L,1/(1+z)))) for z in zc])
print("   deficits",np.round(dd,2)," sqrt(sum (d/1%)^2) =",round(float(np.sqrt((dd**2).sum())),2),"sigma")
