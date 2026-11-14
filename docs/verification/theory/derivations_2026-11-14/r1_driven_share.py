"""R1 (register X3 / X6; ring thread R1), desk derivation, step by step, each algebra step checked by sympy (2026-11-14). Draft until
checked and approved for the book. Return path: eq:vi_identity (p1_04, the virial identity) <- eq:law_landauer <- eq:law.
Writes r1_driven_share.md (the derivation as it stands, with what is derived, what is measured, and what is open)."""
import sympy as sp, os
H = os.path.dirname(os.path.abspath(__file__))
K, W, S, Q, Win, n = sp.symbols("K W S Q W_in n", real=True)      # W = |W| > 0 (binding), S = surface/drive virial term
out, ok = [], []
def check(name, expr):
    r = sp.simplify(expr) == 0; ok.append(r); out.append(f"- check `{name}`: {'PASS' if r else 'FAIL'}"); return r

# A. Clausius' virial needs only a STATIONARY bounded system (time average of d/dt sum p.r vanishes), not equilibrium.
#    For V ~ r^-n with an external surface/drive term S (S = 3 P_s V for a confining pressure; S = 0 for an isolated system): 2K = n|W| + S.
Ksol = (n * W + S) / 2
out.append("**A. Stationary virial with a drive term (Clausius 1870; standard; the external-pressure form is textbook astrophysics).** 2K = n|W| + S.")
check("S = 0, n = 1 gives K = |W|/2 (the book's half)", Ksol.subs({S: 0, n: 1}) - W / 2)

# B. IAM's steps 1-3 (p1_04) for an ISOLATED system: Q = E_i - E_f with E_i = 0, E_f = K - |W|.
Q_iso = -(Ksol - W)
out.append("**B. IAM's steps 1-3, isolated (E_i = 0, no work from outside):** Q = |W| - K = E_Landauer.")
check("isolated, 1/r: Q = K = |W|/2 (eq:vi_identity)", (Q_iso - Ksol).subs({S: 0, n: 1}))

# C. Confined/driven: the surroundings do work W_in during binding. First law: Q = W_in - Delta E = W_in + |W| - K.
Q_dr = Win + W - Ksol
out.append("**C. Driven or confined: the surroundings do work W_in during binding.** First law: Q = W_in + |W| - K. Written share s = Q/|W|.")
s = sp.simplify((Q_dr / W).subs(n, 1))
out.append(f"  s = {s}  (n = 1)")
s_a = s.subs(Win, 0)                 # (a) the drive term is pressure that does no net work over a cycle (stationary confinement)
s_b = s.subs(Win, S)                 # (b) the infalling matter brings in work equal to the surface virial term
check("(a) W_in = 0: s = 1/2 - S/(2|W|)", s_a - (sp.Rational(1, 2) - S / (2 * W)))
check("(b) W_in = S: s = 1/2 + S/(2|W|)", s_b - (sp.Rational(1, 2) + S / (2 * W)))
out.append("  (a) W_in = 0 (pressure confines, no net work): s = 1/2 - S/(2|W|), BELOW one half.")
out.append("  (b) W_in = S (infall brings in work equal to the surface term): s = 1/2 + S/(2|W|), ABOVE one half.")
out.append("  The two bookkeepings move the share in OPPOSITE directions; which holds is a question about the halo's actual energy flow "
           "(what the infall delivers and where the heat goes), so it is NOT decided here.")

# D. The measurement the book already holds (p2_02_virial.tex, tab:vc_halo): raw 2K/|W| = 1.1-1.3 within r_vir, the excess attributed by the
#    book to the pressure of infalling matter. If 2K = |W| + S holds (Klypin et al.: within a few to 17 %), S/|W| = 2K/|W| - 1.
rows = []
for r in (1.1, 1.2, 1.3):
    Sw = r - 1
    rows.append((r, Sw, float(s_a.subs(S, Sw * W)), float(s_b.subs(S, Sw * W))))
out.append("**D. With the book's measured halo virial ratios (2K/|W| = 1.1-1.3; S/|W| = 2K/|W| - 1):**")
out.append("| 2K/|W| | S/|W| | share (a) | beta_m (a) = Omega_m s | share (b) | beta_m (b) |")
out.append("|---|---|---|---|---|---|")
for r, Sw, a, b in rows: out.append(f"| {r} | {Sw:.1f} | {a:.3f} | {0.3153 * a:.4f} | {b:.3f} | {0.3153 * b:.4f} |")
out.append("For comparison only, NOT a target: the settled value 0.15765; the cell's phi 0.147 (its own drive, M = 20.94); the free fit 0.155 +/- 0.029.")
out += ["", "**What this shows (derived):** the one-half is the stationary virial with no drive term; Clausius' theorem itself holds for driven "
        "stationary systems with a drive term S, and cosmic halos have a measured S of 10-30 % of |W|. So beta_m = Omega_m/2 is the S = 0 case, "
        "and a real halo population departs from it by an amount the book already has data for.",
        "**What is open:** (1) how the drive enters IAM's step 1 (W_in), which sets the SIGN of the shift: the two bookkeepings disagree; "
        "(2) how halo-scale shares add up to the cosmic beta_m (mass-function weighting and epoch: the book's eq:vc_decompose with f_coll, eta_vir); "
        "(3) which halos' S (raw or surface-corrected; the book: 'a property of the halo sample chosen, not a constant').",
        "**The cell (same ring thread):** the cell has no 1/r binding, so its share is set by first passage under the paid push (phi = E_hold/M, "
        "-> ln M/M for a strong push) rather than by a degree n. Common statement candidate: the kept share is the restoring law's own value "
        "(n/2 for a power law; first passage for a paid hold) shifted by the drive. NOT derived; the two are not shown to be one formula.",
        "**Guard:** no number above is fitted; beta_m is not moved toward any value. The sign question (1) is decided from the halo energy flow "
        "before any value is compared with data."]
open(os.path.join(H, "r1_driven_share.md"), "w").write("# R1: the virial share for driven systems (desk derivation, as it stands)\n\n" + "\n".join(out) + "\n")
print("\n".join(out)); print("ALL CHECKS PASS" if all(ok) else "SOME CHECK FAILED")
