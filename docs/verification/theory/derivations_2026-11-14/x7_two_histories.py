"""X7 (desk derivation, 2026-11-14). Two premises in the book each fix how the horizon's record entropy S_info grows with a.
(R) the rate accounting (p1_02 eq:rate, eq:Sdot; conjecture): dS/dt ~ rho_m D^n f H / (T_H A_H); with n = 7/2 it gives S_info ~ 1 - 1/a (matter era).
(F) the first law (p1_02 eq:law_F2info -> eq:law_Sneed; derived given Assumptions 1-2) with the record term rho_info = beta_m E(a) rho_crit,0 and
    w_info = -1 - 1/(3a): Sdot = 8 pi^2 rho_info / (3 a H^3), i.e. dS/d ln a = beta_m S_geo,0 E(a) / (a (H/H0)^4).
Question: do (R) and (F) describe the same S_info? Writes x7_two_histories.md."""
import sympy as sp, math, os
a, n = sp.symbols("a n", positive=True)
H = os.path.dirname(os.path.abspath(__file__)); out = []
# matter era exact scalings: H ~ a^-3/2, A_H ~ a^3, T_H ~ a^-3/2, D ~ a, f = 1, rho_m ~ a^-3
R_dlna = a**-3 * a**n * 1 / (a**sp.Rational(-3, 2) * a**3)          # (R): dS/dln a up to a constant
F_dlna = sp.exp(1 - 1 / a) / (a * (a**sp.Rational(-3, 2))**4)         # (F): dS/dln a up to a constant (H/H0 ~ a^-3/2)
out.append(f"(R) matter era: dS/d ln a ~ {sp.simplify(R_dlna)}; with n = 7/2: {sp.simplify(R_dlna.subs(n, sp.Rational(7, 2)))}")
out.append(f"(F) matter era: dS/d ln a ~ {sp.simplify(F_dlna)}")
ratio = sp.simplify(F_dlna / R_dlna.subs(n, sp.Rational(7, 2)))
out.append(f"(F)/(R) = {ratio}: not a constant, so the two premises give DIFFERENT histories of the same S_info")
# effective exponent (R) would need to mimic (F) at each a: n_eff = 9/2 + d ln(F)/d ln a
neff = sp.simplify(sp.Rational(9, 2) + a * sp.diff(sp.log(F_dlna), a))
out.append(f"n that (R) would need to reproduce (F): n_eff(a) = {neff}; at a = 0.25, 0.5, 1: " + ", ".join(f"{float(neff.subs(a, x)):.2f}" for x in (0.25, 0.5, 1)))
# full LCDM check, numerically (Planck: Om 0.3153), growth from the standard integral
Om, OL = 0.3153, 0.6847
Hh = lambda x: math.sqrt(Om * x**-3 + OL)
def D(x, N=4000):
    s = sum((((k + 0.5) / N * x) * Hh((k + 0.5) / N * x)) ** -3 for k in range(N)) * x / N
    return 2.5 * Om * Hh(x) * s
D1 = D(1)
def fgrow(x, h=1e-4): return (math.log(D(x * (1 + h))) - math.log(D(x * (1 - h)))) / (math.log(1 + h) - math.log(1 - h))
for nn in (3.5,):
    out.append(f"full LCDM, a: (R) with n = {nn} vs (F), each normalized at a = 1:")
    for x in (0.3, 0.5, 0.7, 1.0):
        r = (Om * x**-3) * (D(x) / D1)**nn * fgrow(x) * Hh(x) / (Hh(x) * Hh(x)**-2) / Hh(x)   # rho D^n f H / (T_H A_H) / H, T_H ~ H, A_H ~ H^-2
        r1 = Om * (1) * fgrow(1) * 1 / 1
        Fv = math.exp(1 - 1 / x) / (x * Hh(x)**4)
        out.append(f"   a = {x:.1f}: (R) {r / r1:9.4f}   (F) {Fv:9.4f}")
out.append("Reading: (R) with 7/2 makes the per-e-fold writing FALL with a in the matter era (1/a); (F) makes it RISE steeply (E a^5). Since the record "
           "term is rho_info ~ E(a) in both, and (F) is the first law applied to that term, the 7/2 obtained from (R) is not the writing history the "
           "first law requires. The book's two statements (eq:law_n from (R); eq:law_Sneed from (F)) are therefore not the same S_info.")
open(os.path.join(H, "x7_two_histories.md"), "w").write("# X7: the two writing histories (desk, sympy)\n\n" + "\n".join("- " + l for l in out) + "\n")
print("\n".join(out))

# Second check: feed (R)'s n = 7/2 history into the first law and see what record term comes out (matter era, exact).
# From eq:law_F2info with continuity: d(H^2)_x / d ln a = (G/pi) H^4 dS/d ln a, accumulated, not diluted (R1 step 4).
x = sp.symbols("x", positive=True)
integrand = (x**sp.Rational(-3, 2))**4 * (1 / x)               # H^4 * dS/dln a, H ~ a^-3/2, dS/dln a ~ 1/a
rec = sp.integrate(integrand / x, (x, a, 1))                   # contribution written between a and today (d ln a = dx/x)
L2 = [f"(R)+(F): written between a and today, matter era: integral of a^-7 d ln a = {sp.simplify(rec)}",
      "so the record term built from the 7/2 history is dominated by the EARLIEST writing and is nearly constant afterwards: Lambda-like, not E(a).",
      "Alternative reading (rate of a per-area density sigma, S = sigma A_H): S ~ a^3 (1 - 1/a + c) in the matter era, also not E a^5. Neither reading closes."]
open(os.path.join(H, "x7_two_histories.md"), "a").write("\n".join("- " + l for l in L2) + "\n"); print("\n".join(L2))
