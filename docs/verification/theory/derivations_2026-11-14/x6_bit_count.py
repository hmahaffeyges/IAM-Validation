"""X6 (desk derivation, 2026-11-14): can halo virialization supply the record the first law requires? Pricing each bit at T_H and using the
identity T_H S_geo = E inside the horizon (R1 step 3), bits written / S_geo = (energy written) / (E_horizon ln 2). The first law needs the record term
rho_info = beta_m rho_crit today, i.e. energy written / E_horizon = beta_m = 0.158 (up to ln 2). Compare the halo kinetic half and the horizon scale."""
import sympy as sp, math, os
H = os.path.dirname(os.path.abspath(__file__)); out = []
G, c, Hh = sp.symbols("G c H", positive=True)
R = c / Hh; rho = 3 * Hh**2 / (8 * sp.pi * G); M = sp.Rational(4, 3) * sp.pi * R**3 * rho
out.append(f"horizon scale: G M_H/(R_H c^2) = {sp.simplify(G * M / (R * c**2))} (exact, flat FRW): binding comparable to rest energy only at the horizon")
Om, fcoll = 0.3153, 0.5
for s_kms in (100, 200, 500, 1000):
    s2 = (s_kms / 299792.458) ** 2
    K_frac = Om * fcoll * 1.5 * s2                  # kinetic energy in collapsed halos over the critical energy, all epochs' halos still bound today
    out.append(f"halos, mass-weighted sigma {s_kms:4} km/s: K_halos / E_horizon = {K_frac:.2e}  vs beta_m 0.158: short by a factor {0.158 / K_frac:.1e}")
for lab, k in (("uniform sphere, all matter (3/5)", 0.6), ("virial half of that (3/10)", 0.3), ("book's Omega_m/2", 0.5)):
    out.append(f"horizon-scale binding of the matter sector, |W_m|/E_horizon ~ {lab}: {k} x Omega_m x (G M_H/R_H c^2 = 1/2) x 2 = {k * Om:.4f}")
out.append("Reading: halo-scale virialization (sigma^2/c^2 ~ 1e-7 to 1e-5) writes about a millionth of the record the first law requires. The book's "
           "beta_m = Omega_m/2 has the size of matter's binding at the HORIZON scale, where G M/(R c^2) = 1/2 exactly. So the count that can supply beta_m is the "
           "horizon-scale (cosmic) virial, not the sum over halos; halos add only a ~1e-6 correction. This contradicts the book's halo reading of beta_m "
           "(p2_02:59, eq:vc_decompose with f_coll, eta_vir) as the SOURCE of the amount, while supporting the size of Omega_m/2. Order-one factor (3/5, 3/10, 1/2) "
           "open: it is X6's real derivation.")
open(os.path.join(H, "x6_bit_count.md"), "w").write("# X6: who can supply beta_m (desk, sympy)\n\n" + "\n".join("- " + l for l in out) + "\n")
print("\n".join(out))

# Second direction: count BITS, not energy. The first law needs S_info = beta_m S_geo today (eq:law_Sneed, R1 step 3).
import math
lP = 1.616255e-35; cH = 299792458 / (67.4e3 / 3.0857e22)              # Hubble radius, m
S_geo = 4 * math.pi * cH**2 / (4 * lP**2)                                # nats
need = 0.158 * S_geo
known = {"baryons (protons + neutrons) inside the horizon": 1e80, "one record per particle in halos (book p4_14, N_p ~ M/m_p)": 5e79,
         "CMB photons' entropy": 2e89, "relic neutrinos' entropy": 2e89, "all supermassive black holes' entropy (Egan & Lineweaver 2010)": 3e104}
L2 = [f"second direction (bits): S_geo = {S_geo:.2e} nats; the record term needs S_info = 0.158 S_geo = {need:.2e}"]
for k, v in known.items(): L2.append(f"   {k}: ~{v:.0e} ({v / need:.0e} of what is needed)")
L2.append("Even the largest entropy known inside the horizon (supermassive black holes, ~1e104) is 1e-17 of the 1e121 the record term needs. So S_info at "
          "0.158 S_geo cannot be a count of records written by matter in the bulk; it is a horizon-scale quantity. Both directions (energy and bits) agree.")
open(os.path.join(H, "x6_bit_count.md"), "a").write("\n".join("- " + l for l in L2) + "\n"); print("\n".join(L2))
