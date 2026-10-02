# BOTTOM_UP_EXPONENT — the information-production exponent from measured halo mass functions (2026-10-02)
Script: `scripts/verify_bottom_up_exponent.py` (output beside it). Figure: `n_eff_bottom_up.png`.

## The question
Top-down (Theory chapter, Eqs. Idot–Sn): for E(a) = exp(1 − 1/a), the rate İ ∝ ρ_m D^n f H must have n = 7/2 in matter domination. Bottom-up: does the
rate at which collapsing structure writes, computed from measured halo statistics, behave as D^(7/2)?

## What the bottom-up rate must be (no choices left open)
- In dS_info/dt = İ/(T_H A_H) the factor 1/T_H is Landauer (bits per unit energy), so İ is an energy rate.
- IAM's Law with the virial partition: the energy a bound structure writes is its kinetic half, K ≈ ½|W| ∝ G M²/R_vir ∝ M^(5/3)(Δ_c ρ_crit)^(1/3).
- İ = a⁻³ dU/dt with U(a) = ∫ (dn/dlnM) K dlnM over all halos. U converges at low mass, so no mass threshold is needed.
- n_eff = d ln[(dU/dlna)/f] / d ln D, the exponent the top-down form uses.

## Result
| mass function | n_eff z = 9 | 5 | 4 | 3 | 2 | 1 | crosses 7/2 at z | crosses 5/2 at z | mean z 2.3–9 |
|---|---|---|---|---|---|---|---|---|---|
| Press–Schechter | 5.84 | 4.29 | 3.84 | 3.36 | 2.83 | 2.20 | 3.29 | 1.46 | 4.26 |
| Sheth–Tormen | 5.31 | 3.92 | 3.52 | 3.08 | 2.60 | 2.01 | 3.97 | 1.85 | 3.89 |
| Tinker 2008 | 5.78 | 4.31 | 3.89 | 3.44 | 2.95 | 2.40 | 3.15 | 1.19 | 4.29 |
- n_eff is not constant: it runs from 5.3–5.8 at z = 9 through 7/2 at **z = 3.2–4.0** to 2.0–2.4 at z = 1.
- Mean over the matter-dominated window z = 2.3–9: **3.9 (Sheth–Tormen), 4.3 (Press–Schechter, Tinker)**.
- Robust to the halo definition (Bryan–Norman vs 200 × mean density: ±0.03) and to σ8 = 0.77–0.85 (Sheth–Tormen mean 4.0–3.8).
- 5/2 is reached only at z ≈ 1.2–1.9, where Λ already matters and the matter-domination derivation does not apply.
- Mass-weighted ("one bit per particle") instead gives n_eff = ν_min² − 1 (Press–Schechter, exact) and depends on an arbitrary M_min; it is inconsistent
  with the Landauer factor 1/T_H and is not used.

## What it means
The bottom-up rate, built only from IAM's Law, the virial partition, measured σ(M) and three standard halo mass functions, brackets the top-down 7/2:
it equals 7/2 at z ≈ 3–4 and averages ≈ 4 over the matter era. It does not reproduce 5/2. The two directions now point at the same value to within the
running of n_eff; they are not an exact identity. Open: whether including mergers/re-virialisation (beyond the change in total halo kinetic energy)
or the ρ_crit(z) scaling of K flattens the running.
