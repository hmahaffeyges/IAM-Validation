#!/usr/bin/env python3
"""minicosmos_loop.py: the book's nine-step chain as ONE closed loop, from gravity back to gravity, with the record E(a) COMPUTED, not put in
(2026-11-15; register X7; exploration). Where IAM's Law is in it: step 4-6 below (each record costs k T ln2 at the surface it is written to, here
the cosmic horizon at the Gibbons-Hawking temperature; the record enters gravity only through the horizon's entropy, Cai-Kim first law).

  1-2  gravity and collapse      growth D(a): D'' + (2 + dlnH/dlna) D' = (3/2) Omega_m(a) mu(a) D, LCDM background (book p2_06)
  3    decoherence               instantaneous; the rate is the collapse rate (book p4_14 'Collapse sets the rate')
  4    Landauer at T_H, per area the record written per e-fold, dS_info/dln a ~ R(a), R from the book's own source models (p2_03 Table
                                 tab:th:record): Omega_m(a) f D^n, or Omega_m(a) dF/dln a for PS / ST collapsed fractions; amplitude fixed by
                                 eq:law_Sneed, dS_info/dln a (a=1) = beta_m S_geo(1)  (Bekenstein-Hawking capacity S_geo = pi/(G H^2))
  5    Jacobson / Cai-Kim         the record term, by one of three closures:
                                   A  first law (eq:law_F2info + continuity): d rho_x/dln a = 3 H^4 (dS_info/dln a)/(8 pi^2)   [no free constant]
                                   B  the book's numerics: E = I(a)/I(1), I = int R dln a                                      [p2_03 sec:numerics]
                                   C  the appendix postulate: E = exp((S - S(1))/S_0), S_0 free                               [app_C3, conjecture]
  6    feedback                  mu(a) = H^2/(H^2 + beta_m E(a) H0^2) weakens growth (book eq:l2_mu), closing the loop at step 1
Iterated until E stops changing. Self-tests reproduce the book before any reading. Usage: python3 minicosmos_loop.py  (writes xout/ if present)."""
import os, math, numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc

OM0 = 0.315; OR0 = 9.1e-5; OL0 = 1 - OM0 - OR0; BETA_M = OM0 / 2; DELTA_C = 1.686   # the book's sec:numerics background (with radiation)
LNA = np.linspace(np.log(1e-3), np.log(50.0), 1400); A = np.exp(LNA); I1 = int(np.argmin(abs(LNA)))
E2 = OM0 * A**-3 + OR0 * A**-4 + OL0                                   # (H/H0)^2, LCDM background (light's sector; book: background unchanged)
DLNH = -(1.5 * OM0 * A**-3 + 2 * OR0 * A**-4) / E2                           # dlnH/dlna
OMA = OM0 * A**-3 / E2                                   # Omega_m(a)
cumtrap = lambda y, x: np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])

def growth(E_rec, feedback=True):
    """Step 1-2 with step 6: D(a), f(a), normalized D(1) = 1."""
    mu = 1 / (1 + BETA_M * E_rec / E2) if feedback else np.ones_like(A)
    def rhs(x, y):
        i = np.interp(x, LNA, np.arange(len(LNA)))
        j = int(min(i, len(LNA) - 2)); w = i - j
        g = lambda v: v[j] * (1 - w) + v[j + 1] * w
        return [y[1], -(2 + g(DLNH)) * y[1] + 1.5 * g(OMA) * g(mu) * y[0]]
    s = solve_ivp(rhs, [LNA[0], LNA[-1]], [A[0], A[0]], t_eval=LNA, rtol=1e-8, atol=1e-12)
    D = s.y[0]; f = s.y[1] / D
    return D / D[I1], f

def F_coll(nu, kind):
    if kind == "PS": return erfc(nu / math.sqrt(2))
    q, p, Ast = 0.707, 0.3, 0.3222                       # Sheth-Tormen; F_ST(nu) = int_nu^inf f_ST(nu') dnu'/nu'
    x = np.linspace(1e-3, 12, 4000); fst = Ast * math.sqrt(2 * q / math.pi) * (1 + (q * x * x)**-p) * np.exp(-q * x * x / 2)
    tail = cumtrap((fst / x)[::-1], -x[::-1])[::-1]
    return np.interp(nu, x, tail)

def source(D, f, model):
    """Step 3-4: R(a), the record written per e-fold (the book table's source models)."""
    if model.startswith("D^"):
        n = float(eval(model[2:])); return OMA * f * D**n
    kind, sig = model.split("_"); nu = DELTA_C / (float(sig) * D)
    F = F_coll(nu, kind); return OMA * np.gradient(F, LNA)

def record(R, closure, S0_over_S1=1.0):
    """Step 5: the record term E(a) (normalized E(1) = 1) and S_info in nats (amplitude from eq:law_Sneed)."""
    S = cumtrap(R, LNA)
    if closure == "B":
        E = S / S[I1]
    elif closure == "A":
        rho = cumtrap(E2**2 * R, LNA); E = rho / rho[I1]       # d rho_x/dln a ~ H^4 dS/dln a
    else:
        E = np.exp((S - S[I1]) / (S0_over_S1 * S[I1]))
    S_geo = math.pi / E2                                         # in units of 1/(G H0^2)
    S_nats = S / R[I1] * BETA_M * math.pi                       # dS/dln a (1) = beta_m S_geo(1)
    return E, S_nats / S_geo

def loop(model, closure, feedback=True, S0_over_S1=1.0, iters=8):
    E = np.exp(1 - 1 / A)
    for _ in range(iters):
        D, f = growth(E, feedback); R = source(D, f, model)
        E_new, fill = record(R, closure, S0_over_S1)
        if np.max(abs(E_new - E)[A <= 10]) < 1e-6: E = E_new; break
        E = E_new
    return dict(E=E, D=D, f=f, R=R, fill=fill)

def fit_ab(E):
    # the book's own fit (docs/book/figscripts/fig_p2_theory_derivations.py, fig_record_fit): curve_fit of exp(alpha - beta/a) to E itself,
    # points uniform in ln a over 0.15 <= a <= 2.0
    from scipy.optimize import curve_fit
    m = (A >= 0.15) & (A <= 2.0)
    p, _ = curve_fit(lambda x, al, be: np.exp(al - be / x), A[m], E[m], p0=[1, 1]); return p[0], p[1]

def report(out):
    E = out["E"]; dE = np.gradient(E, LNA); a0, b = fit_ab(E)
    return dict(alpha=a0, beta=b, c_direct=math.log(E[-1] / E[I1]), peak_a=A[np.argmax(dE)], fill_today=out["fill"][I1], fill_end=out["fill"][-1],
                D0=out["D"][I1])

def selftest():
    ok = True
    r = report(loop("D^3.5", "B", feedback=False))
    t1 = abs(r["alpha"] - 0.93) < 0.03 and abs(r["beta"] - 1.02) < 0.03; ok &= t1
    print(("PASS" if t1 else "FAIL"), f"closure B, no feedback, D^3.5 reproduces the book table (alpha 0.93, beta 1.02): {r['alpha']:.3f}, {r['beta']:.3f}")
    D_on, _ = growth(np.exp(1 - 1 / A), True); D_off, _ = growth(np.zeros_like(A), True)
    # growth ratio at z = 0 for the same EARLY amplitude: renormalize both at a = 1e-3
    ratio = (D_on[I1] / D_on[0]) / (D_off[I1] / D_off[0])
    t2 = abs(ratio - 0.9922) < 0.0015; ok &= t2
    print(("PASS" if t2 else "FAIL"), f"E = exp(1 - 1/a), beta_m = Omega_m/2 reproduces the book's D ratio at z = 0 (0.9922): {ratio:.4f}")
    return ok

if __name__ == "__main__":
    L = ["minicosmos_loop: the book's chain closed, E(a) computed (exploration)", "SELF-TESTS"]
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ok = selftest()
    L += buf.getvalue().strip().split("\n")
    if not ok: L.append("SELF-TEST FAILED: no reading reported")
    else:
        L.append("RESULTS (c_direct = ln E(50)/E(1); fit as the book did, 0.15 <= a <= 2; canon c = 1, alpha = beta = 1)")
        for model in ("D^2.5", "D^3", "D^3.5", "D^4", "PS_1.0", "PS_1.2", "ST_1.0", "ST_1.2"):
            for closure in ("B", "A"):
                for fb in (False, True):
                    try:
                        r = report(loop(model, closure, fb))
                        L.append(f"  {model:7s} closure {closure} feedback {'on ' if fb else 'off'}: c {r['c_direct']:.3f} | fit alpha {r['alpha']:.3f} beta {r['beta']:.3f} | "
                                 f"dE/dlna peaks a {r['peak_a']:.2f} | fill today {r['fill_today']:.3f} end {r['fill_end']:.3f}")
                    except Exception as e:
                        L.append(f"  {model:7s} closure {closure} feedback {fb}: failed {type(e).__name__}: {str(e)[:100]}")
        L.append("  closure C (appendix postulate): c = (S(50) - S(1))/S_0 for any source; S_0 is free, so C cannot predict c. Examples, D^3.5 feedback on:")
        for s0 in (0.5, 1.0, 2.0):
            r = report(loop("D^3.5", "C", True, s0)); L.append(f"    S_0 = {s0} x S(1): c {r['c_direct']:.3f}, fit beta {r['beta']:.3f}")
    txt = "\n".join(L); print(txt)
    if os.path.isdir("xout") or os.environ.get("GITHUB_ACTIONS"):
        os.makedirs("xout", exist_ok=True); open("xout/minicosmos_loop_summary.txt", "w").write(txt + "\n")
