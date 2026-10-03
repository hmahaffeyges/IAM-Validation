"""Checks of the numbers computed in the Part 3 Quantum Order additions (2026-10-03).
Every input is a published value (cited in the chapter) or a constant; nothing is fitted."""
import math
kB=1.380649e-23; e=1.602176634e-19; h=6.62607015e-34
n_cp=4e6            # um^-3, book value n_cp = 2 nu0 Delta (p3_02)
Delta=182e-6*e      # J, book value for Al
out=[]
def rep(name,val,expect,tol):
    ok=abs(val-expect)<=tol*abs(expect); out.append((name,val,expect,ok)); print(f"{name:55s} {val:.4g}  (text {expect:.3g})  {'OK' if ok else 'FAIL'}")
# Riste 2013: n_qp = 0.04 um^-3 at 20 mK -> x_qp
rep("x_qp from Riste n_qp=0.04 um^-3",0.04/n_cp,1e-8,0.05)
# London volume pi*lambda^3 (lambda = 50 nm) and integral check
lam=0.05 # um
import numpy as np
r=np.linspace(0,2,400001); I=4*math.pi*np.trapezoid(np.exp(-2*r/lam)*r**2,r)
rep("4pi int e^{-2r/lam} r^2 dr / (pi lam^3)",I/(math.pi*lam**3),1.0,1e-4)
V=math.pi*lam**3; rep("V_eff = pi lambda^3 [um^3]",V,3.9e-4,0.02)
rep("pairs in V_eff",n_cp*V,1.6e3,0.03)
# thermal fraction at 100 mK
T=0.1; x=math.sqrt(2*math.pi*kB*T/Delta)*math.exp(-Delta/(kB*T)); print("x_th(100 mK) =",f"{x:.3g}"); out.append(("x_th(100mK)<1e-9",x,1e-9,x<1e-9))
# feedback loop: g = 2 phi tau_qp / tau_TLS < 1
tqp,tTLS=100.,30.; rep("phi_max = tau_TLS/(2 tau_qp)",tTLS/(2*tqp),0.15,0.12)
# colour-code vs surface-code threshold ratio
rep("surface 1e-2 / colour 2e-3",1e-2/2e-3,5,1e-9)
# Harrington: cosmic-ray correlated rate 1/592 s
print("Harrington rate per hour =",f"{3600/592:.2f}")
print("ALL OK" if all(o[3] for o in out) else "SOME FAIL")
