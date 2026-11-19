"""implied_history.py  (private, proposals only; 2026-11-18)

What each candidate would PREDICT for the record history if it were the writer, so that a future
growth test can separate them.  The amplitude today is the chains' measured one (beta_m = Omega_m/2 fixed,
18 converged chains), which by the identity A3 is (1/2) M_m c^2 / k T_GH per e-fold; it is NOT fitted per
candidate.  Each candidate supplies only its time shape s_c(a) = rate_c(a)/rate_c(1).

Ledger rule (ledger_half_checks B1-B4):  k T_GH dS_info/dlna = V_H d rho_info/dlna.
With rho_info(1) = beta_m rho_c0 and the identity, the implied activation obeys
     dE_c/dlna = h(a)^4 s_c(a),   E_c(1) = 1          (h = H/H0)
(check: the book's own shape s = E/(a h^4) returns dE/dlna = E/a, i.e. E = exp(1-1/a)).
Then mu(z) = 1/(1 + beta_m E_c / h^2)  (book eq:mu, p1_02:458).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import cumulative_trapezoid
import candidates_compare as CC
B = CC.B

Om = B.Om
bm = Om/2
h = lambda a: CC.Eb(a)
lna = np.linspace(np.log(0.2), 0.0, 321)
A = np.exp(lna)

def implied_E(s):
    g = h(A)**4*s
    # integrate from a=1 backwards: E(a) = 1 - int_{lna}^{0} g
    I = cumulative_trapezoid(g[::-1], -lna[::-1], initial=0.0)[::-1]   # int from lna to 0
    return 1 - I

shapes = {}
shapes["book E(a) (measured shape)"] = np.exp(1 - 1/A)/(A*h(A)**4)
# (1) book rule, ST, horizon-volume total: rate ~ dK/dlna * Vcom_H / T
BB = [B.bulk_bits(a, B.f_ST) for a in A]
r1 = np.array([b[0] for b in BB])
shapes["(1) virial kinetic half at kT_GH (ST)"] = r1/r1[-1]
# (1ii) eq:Sdot per-area shape
AH = 4*np.pi*(B.c/(B.H0*B.Ez(A)))**2
dK = np.array([b[4] for b in BB])
T = B.hb*B.H0*B.Ez(A)/(2*np.pi*B.kB)
s1ii = (dK/A**3)/(T*AH)
shapes["(1ii) virial, eq:Sdot per-area shape"] = s1ii/s1ii[-1]
# (3) starlight / BH accretion (both psi-shaped): rate ~ psi(z) Vcom_H / H / T
z = 1/A - 1
s3 = CC.psi(z)*CC.Vcom_H(A)/CC.Hb(A)/CC.T_GH(A)
shapes["(3) starlight or BH accretion at kT_GH"] = s3/s3[-1]
# (4a) half the rest energy of the matter inside the apparent horizon at each epoch
s4a = A**-3/h(A)**4               # (1/2)M_m(a)/T(a) normalised to 1 at a=1
shapes["(4a) 1/2 M_m(a) c^2 (matter inside horizon now)"] = s4a
# (4b) half the rest energy of a fixed comoving mass (today's horizon matter M_m0) at kT_GH(a)
s4b = 1/h(A)
shapes["(4b) 1/2 M_m0 c^2 (fixed comoving matter)"] = s4b/s4b[-1]

out = {}
for k, s in shapes.items():
    E = implied_E(s)
    def at(zz):
        return float(np.interp(np.log(1/(1+zz)), lna, E))
    Ez = {zz: at(zz) for zz in (0.5, 1.0, 2.0, 3.0)}
    mu = {zz: 1/(1 + bm*max(Ez[zz], 0.0)/h(1/(1+zz))**2) for zz in (0.5, 1.0)}  # no negative record: mu=1 before switch-on
    mu[0.0] = 1/(1 + bm)
    neg = np.where(E < 0)[0]
    if len(neg):
        i = neg[-1]
        l0 = lna[i] + (0 - E[i])*(lna[i+1] - lna[i])/(E[i+1] - E[i])
        z_on = float(np.exp(-l0) - 1)
    else:
        z_on = None
    # w_info(a=1) = -1 - (1/3) dlnE/dlna at a=1 ; slope at a=1 is h^4 s = 1 for all (identity), so w(1) = -4/3 for all
    slope_half = float(np.interp(np.log(0.5), lna, h(A)**4*s/np.maximum(E, 1e-30)))
    w_half = -1 - slope_half/3 if at(1.0) > 0 else float('nan')
    out[k] = dict(E=Ez, mu=mu, z_on=z_on, w_z1=w_half)

if __name__ == "__main__":
    print(f"{'candidate':50s} {'E(z=.5)':>8s} {'E(1)':>7s} {'E(2)':>7s} {'mu(.5)':>7s} {'mu(1)':>7s} {'w(z=1)':>7s}  z_switch_on")
    for k, v in out.items():
        print(f"{k:50s} {v['E'][0.5]:8.3f} {v['E'][1.0]:7.3f} {v['E'][2.0]:7.3f} {v['mu'][0.5]:7.4f} {v['mu'][1.0]:7.4f} {v['w_z1']:7.3f}  {v['z_on']}")
    E4a = implied_E(shapes["(4a) 1/2 M_m(a) c^2 (matter inside horizon now)"])
    print("4a analytic 4/3 - a^-3/3, max dev:", float(np.max(np.abs(E4a - (4/3 - A**-3/3)))))
    Ebk = implied_E(shapes["book E(a) (measured shape)"])
    print("book shape returns exp(1-1/a), max dev:", float(np.max(np.abs(Ebk - np.exp(1 - 1/A)))))
    import json
    json.dump(out, open("implied_history.json", "w"), indent=1, default=str)
