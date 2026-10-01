# The exponent n — IAM Theory paper §5.3–6.4, line by line (2026-10-01)

| eq. | paper | check |
|---|---|---|
| 28 | İ ∝ ρ_m D(a)ⁿ f(a) H(a) | definition |
| 29 | dS_info/dt = İ / (T_H A_H) | definition: the record is accumulated per unit **time**, so the measure is fixed by the paper |
| 32–35 | matter era: H ∝ a^(−3/2), A_H = 4π/H² ∝ a³, T_H = H/2π ∝ a^(−3/2), D ∝ a, f ≈ 1 | ✓ |
| 36 | dS/d ln a = (dS/dt)/H ∝ ρ_m Dⁿ f /(T_H A_H) | ✓ (the H in İ cancels the 1/H) |
| 37 | a⁻³ · aⁿ / (a^(−3/2) · a³) = a^(n − 9/2) | ✓ |
| 38 | dS/da ∝ a^(n − 11/2) | ✓ |
| 39 | S ∝ ∫ a^(n − 11/2) da = a^(n − 9/2)/(n − 9/2) | ✓ |
| 40 | need S ∝ −1/a + const | ✓ |
| 41 | n − 9/2 = −1 ⟹ **n = 5/2** | **arithmetic: n − 9/2 = −1 gives n = 7/2.** With n = 5/2, Eq. 39 gives S ∝ a⁻², not a⁻¹ |

**Numerical check** (Eqs. 28–29 integrated with the full ΛCDM growth factor, Planck 2018; dS/d ln a ∝ a^p, exp(1 − 1/a) needs p = −1):

| n | p, matter era (a 0.01–0.1) | p, late (a 0.25–1) |
|---|---|---|
| 5/2 | −2.02 | −2.42 |
| 3 | −1.52 | −1.99 |
| 7/2 | **−1.02** | −1.57 |

So with the paper's own Eqs. 28–39, matter domination needs n = 7/2, and the Λ era pushes the effective value higher. That agrees with §6.6
("full ΛCDM shifts it to n_eff ≈ 3–4") and with the paper's Table 2 fit D^(7/2) → exp(0.95 − 1.05/a). (The 'ST n_eff ≈ 3.5' is not a literature value; see NBODY_TRACE.md.)

**Correction to my previous check:** I wrote that the result depends on a choice of measure (da, d ln a, dt). That was wrong. Eq. 29 fixes the
measure (per unit time, divided by T_H A_H), and my "per dt gives 2" left out the 1/(T_H A_H) factor.

**Not yet checked line by line:** the bottom-up route ("ν ∝ D^(−1/2) near M* gives D^(5/2)", in the Measurement/Bridge paper and the
Zurek paper). In standard Press–Schechter ν = δ_c/[σ(M) D] ∝ D⁻¹; that derivation must be read in full before it is printed either way.

**Decision (author, 2026-10-02):** change to 7/2. Done in p2_theory.tex and p2_virial.tex.
