"""Ledger line 4 (PHY-083, desk): the candidate's odds = undriven restore time / driven restore time, a mark-width diffusion from the lost end
(x = 1, reflecting) to the restored end (x = 0, absorbing), with a potential U falling toward x = 0 by the total work w. The candidate uses a
constant push (U = w x). Same total work, other physically plausible profiles: does any give the ~0.87 of the constant-push odds that the
healthy reading implies? Mean first passage T = int_0^1 dy e^{U(y)} int_y^1 e^{-U(z)} dz (D = 1, mark width 1); undriven T0 = 1/2."""
import numpy as np, math, os, json, io, contextlib
H = os.path.dirname(os.path.abspath(__file__))
M = float(json.load(open(os.path.join(H, "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
_buf = io.StringIO(); _ctx = contextlib.redirect_stdout(_buf); _ctx.__enter__()
w = 2 * M
y = np.linspace(0, 1, 200001)
def T(U):
    u = U(y); inner = np.flip(np.cumsum(np.flip(np.exp(-(u - u.max())))) ) * (y[1] - y[0])   # int_y^1 e^{-U}, scaled by e^{U max}
    return float(np.trapezoid(np.exp(u - u.max()) * inner, y))
prof = {"constant push (candidate)": lambda x: w * x,
        "push strongest where the mark is lost (U = w x^2)": lambda x: w * x ** 2,
        "push strongest at the restore site (U = w(2x - x^2))": lambda x: w * (2 * x - x ** 2),
        "push only over the first half from the restore site": lambda x: w * np.minimum(2 * x, 1.0),
        "push only over the half nearest the lost end": lambda x: w * np.maximum(2 * x - 1, 0.0)}
base = None
for k, U in prof.items():
    o = 0.5 / T(U); base = base or o
    print(f"{k:55} odds {o:8.2f}  held error {1/(1+o):.5f}  relative to constant {o/base:.3f}")
print("healthy reading implies odds about", round(1 / 0.0509 - 1, 2), "=", round((1 / 0.0509 - 1) / base, 3), "of the constant push")
# Is the constant push the best any profile can do with the same total work? Power laws U = w x^p and 2,000 random monotone profiles.
rng = np.random.default_rng(20261113)
print("power-law profiles:", {p: round(0.5 / T(lambda x, p=p: w * x ** p) / base, 3) for p in (0.7, 0.85, 1.0, 1.15, 1.25, 1.4, 2.0)})
best = 0
for _ in range(2000):
    k = rng.integers(3, 12); steps = rng.dirichlet(np.ones(k)); knots = np.concatenate([[0], np.cumsum(steps)]); xs = np.linspace(0, 1, k + 1)
    r = 0.5 / T(lambda x, xs=xs, knots=knots: w * np.interp(x, xs, knots)) / base; best = max(best, r)
print("best of 2,000 random monotone profiles, relative to the constant push:", round(best, 4))
_ctx.__exit__(None, None, None); out = _buf.getvalue(); print(out, end="")
open(os.path.join(H, "geometry_profiles_summary.txt"), "w").write(out)
