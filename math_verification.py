import sympy as sp
import numpy as np

# Symbols from derivations (holographic + variational PDFs)
a, H0, Om, beta_m = sp.symbols('a H0 Om beta_m', positive=True)

# E(a) from holographic derivation (exp(1 - 1/a))
E = sp.exp(1 - 1/a)

# Modified H^2 from variational PDF: H^2 = H0^2 [Om a^{-3} + (1-Om) + beta_m E]
H2 = H0**2 * (Om * a**(-3) + (1 - Om) + beta_m * E)
print("Symbolic Modified Friedmann H^2:")
sp.pprint(H2)

# μ(a) from CAMB note / variational implications: μ(a) = H^2_LCDM / (H^2_LCDM + beta_m E H0^2)
H2_LCDM = H0**2 * (Om * a**(-3) + (1 - Om))
mu = H2_LCDM / (H2_LCDM + beta_m * E * H0**2)
print("\nSymbolic μ(a):")
sp.pprint(mu)

# w_info(a) from variational scalar field: w_info = -1 - 1/(3a)
w_info = -1 - 1/(3*a)
print("\nSymbolic w_info(a):")
sp.pprint(w_info)

# w_info at z=0: -1 -1/3 ≈-1.333
w_info_num = float(w_info.subs(a, 1))
print(f"Numerical w_info(1): {w_info_num:.3f}")
assert np.isclose(w_info_num, -1.333, rtol=1e-3), "w_info check failed"

# Numerical checks with IAM values (Om=0.31, beta_m=0.157, H0=1 for normalized H^2/H0^2)
vals = {Om: 0.31, beta_m: 0.157, H0: 1.0, a: 1.0}  # a=1 (z=0)

# H^2 / H0^2 at z=0: should be 1 + beta_m ≈1.157
H2_num = float(H2.subs(vals))
print(f"\nNumerical H^2 / H0^2 at a=1 (z=0): {H2_num:.3f}")
assert np.isclose(H2_num, 1 + 0.157, rtol=1e-3), "H^2 check failed"

# μ at z=0: should be 1 / (1 + beta_m) ≈0.864
mu_num = float(mu.subs(vals))
print(f"Numerical μ(1): {mu_num:.3f}")
assert np.isclose(mu_num, 0.864, rtol=1e-3), "μ check failed"

# w_info at z=0: -1 -1/3 ≈-1.333
w_info_num = float(w_info.subs(a=1))
print(f"Numerical w_info(1): {w_info_num:.3f}")
assert np.isclose(w_info_num, -1.333, rtol=1e-3), "w_info check failed"

# Effective Ω_m dilution at z=0: Om / (1 + beta_m) ≈0.31 / 1.157 ≈0.268
eff_Om = Om / (1 + beta_m)
eff_Om_num = float(eff_Om.subs(vals))
print(f"Effective Ω_m at z=0 (dilution): {eff_Om_num:.3f}")
assert np.isclose(eff_Om_num, 0.268, rtol=1e-3), "Dilution check failed"

# Quick table of μ(a) at various z (as in CAMB note)
z_vals = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
a_vals = 1 / (1 + np.array(z_vals))
mu_table = [float(mu.subs({a: aval, **vals})) for aval in a_vals]
print("\nμ(a) table (z, a, μ):")
for z, a_val, mu_val in zip(z_vals, a_vals, mu_table):
    print(f"z={z:.1f}, a={a_val:.3f}, μ={mu_val:.3f}")

print("\nAll checks passed!")
