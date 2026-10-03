"""Checks for the wave-2 carriage (Variational, Holographic, GRF essay, encoding-surface bones notes).
Every number written in the wave-2 insertion blocks or quoted in MANIFEST.md is printed here. CODATA 2018 constants."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import curve_fit
import sympy as sp

hbar=1.054571817e-34; c=2.99792458e8; G=6.67430e-11; kB=1.380649e-23; h=2*np.pi*hbar
Msun=1.98847e30; NA=6.02214076e23; R=8.314462618; eV=1.602176634e-19
TCMB=2.7255; Mpc=3.0856775814913673e22
def T_BH(M): return hbar*c**3/(8*np.pi*G*M*kB)
def TGH(H0kms): return hbar*(H0kms*1e3/Mpc)/(2*np.pi*kB)

print("[W1] exponent algebra (matter domination)")
a,n=sp.symbols('a n',positive=True)
with_area = sp.simplify(a**-3*a**n/(a**sp.Rational(-3,2)*a**3))      # dS/dlna with 1/(T_H A_H)
print("  dS/dlna with 1/A_H:", with_area, "-> dS/da exponent", sp.simplify(sp.log(with_area/a)/sp.log(a)))
print("  integral a^(n-11/2) = a^(n-9/2)/(n-9/2); a^-1 needs n =", sp.solve(sp.Eq(n-sp.Rational(9,2),-1),n))
no_area = sp.simplify(a**-3*a**n*a**sp.Rational(3,2))                  # rho D^n f H / T_H  * dt/da (dt=da/(aH))  -> per da: rho D^n /(a H)?
# Variational l.235-246: S=int Idot/T_H * da/(aH) with Idot ~ rho D^n f H : integrand = rho D^n f H/(H) * 1/(aH) = rho D^n/(aH)
var_integrand = sp.simplify(a**-3*a**n/(a*a**sp.Rational(-3,2)))
print("  Variational integrand (no 1/A_H):", var_integrand, "; a^-2 needs n =", sp.solve(sp.Eq(n-sp.Rational(5,2),-2),n))

print("[W2] Raychaudhuri/Clausius, Cai-Kim, w_info")
H,Hd,rho,P=sp.symbols('H Hdot rho P'); Gs=sp.symbols('G',positive=True)
lhs=4*sp.pi*(rho+P)/H**2; rhs=-Hd/(Gs*H**2)
print("  Cai-Kim:", sp.solve(sp.Eq(lhs,rhs),Hd))
w=sp.symbols('w'); print("  w_info from H/a+3H(1+w)=0:", sp.solve(sp.Eq(1/a+3*(1+w),0),w))
r=sp.symbols('r',positive=True); lam=sp.symbols('lambda_L',positive=True)
print("  V_eff = 4 pi int e^(-2r/l) r^2 dr =", sp.simplify(4*sp.pi*sp.integrate(sp.exp(-2*r/lam)*r**2,(r,0,sp.oo))))

print("[W3] numerical fits exp(alpha-beta/a) to cumulative integrals (source tables)")
Om,Or=0.315,9.1e-5; OL=1-Om-Or
Ef=lambda x: np.sqrt(Om*x**-3+Or*x**-4+OL)
def growth(x):
    # D'' in ln a; y=[D, dD/dlna]
    def f(lna,y):
        aa=np.exp(lna); E2=Om*aa**-3+Or*aa**-4+OL; dlnE=(-3*Om*aa**-3-4*Or*aa**-4)/(2*E2)
        Oma=Om*aa**-3/E2
        return [y[1], -(2+dlnE)*y[1]+1.5*Oma*y[0]]
    a0=1e-4; s=solve_ivp(f,[np.log(a0),np.log(x.max())],[a0,a0],dense_output=True,rtol=1e-9,atol=1e-12)
    D=s.sol(np.log(x))[0]; dD=s.sol(np.log(x))[1]; D1=s.sol(0.0)[0]
    return D/D1, dD/D
ag=np.logspace(-4,np.log10(2.0),6000); D,fg=growth(ag); E=Ef(ag); Oma=Om*ag**-3/E**2
def cum(integrand):
    I=np.concatenate([[0],np.cumsum(0.5*(integrand[1:]+integrand[:-1])*np.diff(ag))]); return I
def fit(I):
    m=(ag>=0.15)&(ag<=2.0); y=I[m]/np.interp(1.0,ag,I)
    p,_=curve_fit(lambda x,al,be: np.exp(al-be/x), ag[m], y, p0=[1,1]); 
    rr=np.corrcoef(y,np.exp(1-1/ag[m]))[0,1]; return p,rr
for nn in [2.0,2.5,3.5,4.0]:
    for lab,integ in [("Var: D^n Om(a) f/(T_H a)", D**nn*Oma*fg/(E*ag)),
                      ("Holo: D^n Om(a) f/(T_H A_H)", D**nn*Oma*fg/(E*E**-2)),
                      ("Holo-tab: D^n Om(a) f/T_H /(T_H A_H)", D**nn*Oma*fg/E/(E*E**-2))]:
        p,rr=fit(cum(integ)); print(f"  n={nn}: {lab:38s} alpha={p[0]:.3f} beta={p[1]:.3f} r={rr:.3f}")

print("[W4] cosmology numbers in the sources vs the book")
for x,mu,s in [(67.16,67.4,0.5),(72.26,73.04,1.04),(72.51,73.04,1.04)]: print(f"  |{x}-{mu}|/{s} = {abs(x-mu)/s:.2f} sigma")
print("  67.4*sqrt(1.1575) =",round(67.4*np.sqrt(1.1575),2),"; 67.16*sqrt(1+0.15765) =",round(67.16*np.sqrt(1.15765),2))
print("  Omega_m/2 (0.315) =",0.315/2,"; (0.3153) =",0.3153/2)
Omx=0.315; print("  w_eff(1) = 2(3-Om)/(3(Om-2)) =",round(2*(3-Omx)/(3*(Omx-2)),4))
print("  f_coll naive 0.315*0.62 =",round(0.315*0.62,4)," eta=1/(2*0.62)=",round(1/(2*0.62),3))

print("[W5] thermal regimes against the sky (T/T_CMB)")
T_sgr=T_BH(4.3e6*Msun)
for name,T in [("cell 310.15 K",310.15),("transistor 300 K",300.0),("transistor 350 K",350.0),("qubit stage 15 mK",0.015),
               ("BH 1 Msun",T_BH(Msun)),("Sgr A*",T_sgr),("cosmic horizon H0=67.36",TGH(67.36))]:
    print(f"  {name:24s} T={T:.3e} K  T/T_CMB={T/TCMB:.3e}  T_CMB/T={TCMB/T:.3e}")
print("  M_CMB = hbar c^3/(8 pi G kB T_CMB) =",f"{hbar*c**3/(8*np.pi*G*kB*TCMB):.3e}"," kg")
M=Msun; S=4*np.pi*G*M**2*kB/(hbar*c); print("  Mc^2/(kB T_BH) / (2 S/kB) =",f"{M*c**2/(kB*T_BH(M))/(2*S/kB):.12f}",
     "; Mc^2/(kB T ln2) =",f"{M*c**2/(kB*T_BH(M)*np.log(2)):.3e}","; S/(kB ln2) =",f"{S/(kB*np.log(2)):.3e}")
mP=np.sqrt(hbar*c/G); print("  8 pi (M/m_P)^2 / (Mc^2/kT) =",f"{8*np.pi*(M/mP)**2/(M*c**2/(kB*T_BH(M))):.12f}")
P=hbar*c**6/(15360*np.pi*G**2*M**2); print("  Gamma=P/(kT ln2) vs c^3/(1920 G M ln2):",f"{P/(kB*T_BH(M)*np.log(2)):.6e}",f"{c**3/(1920*G*M*np.log(2)):.6e}")

print("[W6] bones numbers")
print("  kT at 310 K =",f"{kB*310/eV*1e3:.2f}","meV ; RT =",f"{R*310.15/1e3:.3f}","kJ/mol ; n_ratio = 54000/(R*310.15) =",f"{54000/(R*310.15):.3f}",
      "; /ln2 =",f"{54000/(R*310.15)/np.log(2):.2f}")
print("  kB*310.15*ln2 =",f"{kB*310.15*np.log(2):.4e}"," J")
Tc=2.112/1.764
print("  2*Delta_Al/h with Delta=182 ueV:",f"{2*182e-6*eV/h/1e9:.1f}"," GHz")
sig=5.670374419e-8; print("  sigma T_CMB^4 =",f"{sig*TCMB**4:.4e}"," W/m^2")
x0=h*88e9/(kB*TCMB)
Pfrac=quad(lambda x: x**3/np.expm1(x),x0,200)[0]/(np.pi**4/15)
Nph=2*np.pi*(kB*TCMB/h)**3/c**2*quad(lambda x: x**2/np.expm1(x),x0,200)[0]
print("  CMB power fraction above 88 GHz =",f"{Pfrac:.4f}"," ; photon flux above 88 GHz =",f"{Nph:.3e}"," /s/m^2")
xq=np.log(2)*100e-6/(9.03e28*30e-6*np.pi*(50e-9)**3); print("  x_qp_min (tau_TLS=30us) =",f"{xq:.3e}","; tau_TLS for x_qp=1e-7:",
     f"{np.log(2)*100e-6/(9.03e28*1e-7*np.pi*(50e-9)**3)*1e6:.1f}"," us")

print("[W7] equation of state, mu, chi2 threshold, Planck distance")
from scipy.stats import chi2 as _chi2
Om=0.315; OL=1-Om; bm=Om/2
for z in [0,0.5,1]:
    aa=1/(1+z); Ea=np.exp(1-1/aa); wi=-1-1/(3*aa)
    weff=(-OL+bm*Ea*wi)/(OL+bm*Ea); E2=Om*aa**-3+OL
    print(f"  z={z}: w_info={wi:.3f} w_eff={weff:.3f} mu={E2/(E2+bm*Ea):.3f}")
print("  chi2 95% for 1 dof =",round(_chi2.ppf(0.95,1),3))
print("  |67.16-67.36|/0.54 =",round(abs(67.16-67.36)/0.54,2))

print("[W8] S_8 from the Level-2 sigma_8")
for Omx in (0.315,0.3153): print(f"  S8 = 0.800*sqrt({Omx}/0.3) = {0.800*np.sqrt(Omx/0.3):.3f}")
print("  Omega_m needed for S8=0.78 at sigma8=0.800:",round(0.3*(0.78/0.800)**2,3))
print("  x_qp*tau_TLS (pi lambda^3 form) =",f"{np.log(2)*100e-6/(9.03e28*np.pi*(50e-9)**3):.3e}"," s")
Ebar=Pfrac*sig*TCMB**4/Nph; print("  mean CMB photon energy above 88 GHz =",f"{Ebar/kB:.2f}"," K ; per 2Delta (364 ueV):",f"{Ebar/(2*182e-6*eV):.2f}")
