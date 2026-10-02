#!/usr/bin/env python3
"""x_qp paper (14 Apr 2026): the formula as printed, then with the field's n_cp, the pair-breaking yield, and the device volume.
Sources: n_cp = 2 nu0 Delta ~ 4e6 um^-3 (Wang et al. Nat. Commun. 5, 5836; arXiv:2402.15471; arXiv:2208.02790); D_qp in Al ~ 6 um^2/ns normal state
(arXiv:2402.15471), 22.5 cm^2/s measured (PRL 20, 1502); QP diffusion over >= a few um treated as uniform over electrodes (arXiv:2402.15471)."""
import numpy as np
ln2=np.log(2); lam=0.05; tq=100.0; tt=30.0          # um, us, us
ncp_paper=9.03e10; ncp_field=4e6                      # um^-3
Vs=np.pi*lam**3
x_paper=ln2*tq/(ncp_paper*tt*Vs); print(f"1. as printed: x = {x_paper:.3e}  (n_cp = 9.03e10 um^-3, all conduction electrons / 2)")
n_qp=x_paper*ncp_paper; print(f"   implied QP density at a site = {n_qp:.0f} um^-3; measured floor 1e-7 x 4e6 = {1e-7*ncp_field:.1f} um^-3 -> ratio {n_qp/(1e-7*ncp_field):.0f}")
print(f"   same formula with the field's n_cp: x = {ln2*tq/(ncp_field*tt*Vs):.2e}  (n_cp ratio {ncp_paper/ncp_field:.0f})")
print("2. energy: Delta ln2 = 126 ueV per erasure < 2 Delta = 364 ueV per broken pair (2 QPs). If each erasure breaks one pair, yield = 2 QPs, not ln2.")
print("3. volume: QPs diffuse sqrt(D tau_qp) over 100 us:", ", ".join(f"D {D} um^2/ns -> {np.sqrt(D*1e3*tq):.0f} um" for D in (0.6,2.25,6)))
print("   steady state over the island: x = Y * N_sites * tau_qp / (tau_TLS * n_cp * V)")
for Y,lab in ((ln2,"ln2 (energy-limited)"),(2.0,"2 (one pair per event)")):
    for V in (1e3,1e4,1e5):
        N=1e-7*ncp_field*V*tt/(Y*tq); print(f"   yield {lab:22s} V {V:7.0e} um^3: sites needed for x = 1e-7: {N:8.0f};  x per site {Y*tq/(tt*ncp_field*V):.1e}")
