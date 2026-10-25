"""DEV-ERASE-01, the graded maintenance test: the scorer (written and committed 2026-10-25 before any count exists; plan 14.1).
Bar 2: pooled held error (the one command's Stage Q2 count, *_q2.json) by condition, DMSO < 2 uM < 10 uM on both days and both suppliers.
Bar 3: E = ln((opp - iso) / iso) tile by tile (*_tiles.csv.gz, two independent halves); E(dose) against E(control) across tiles, the control's
E from half A, the errors-in-variables bias removed with half B as the instrument: slope = cov(E_dose, E_B) / cov(E_A, E_B), intercept =
mean(E_dose) - slope mean(E_A); 95 % intervals from 2,000 bootstrap draws over tiles (seed 20261025). Tiles kept: at least MIN_OPP
opportunities and MIN_ISO lone errors in each control half and in the dose. R if the slope's interval holds 1 and the intercept's lies below
0; Phi if the slope's interval lies below 1 and the intercept's holds 0; otherwise neither.
Usage: python3 graded_score.py DIR OUT.txt [CTRL_ID DOSE_ID ...]   (pairs given: bar 3 on each; none given: bars 2 and 3 on GSE184116)"""
import sys, os, glob, json, numpy as np, pandas as pd
MIN_OPP, MIN_ISO = 500, 5
D, OUT = sys.argv[1], sys.argv[2]; H = os.path.dirname(os.path.abspath(__file__))
def tiles(i):
    f = glob.glob(os.path.join(D, "**", f"{i}_tiles.csv.gz"), recursive=True); return pd.read_csv(f[0]).set_index(["chrom", "tile"]) if f else None
def E(iso, opp): return np.log((opp - iso) / iso)
def bar3(c, d):
    C, X = tiles(c), tiles(d)
    if C is None or X is None: return f"{c} against {d}: tiles missing", None
    J = C.join(X, rsuffix="_d", how="inner"); xo, xi = J.opp_a_d + J.opp_b_d, J.iso_a_d + J.iso_b_d
    k = (J.opp_a >= MIN_OPP) & (J.opp_b >= MIN_OPP) & (J.iso_a >= MIN_ISO) & (J.iso_b >= MIN_ISO) & (xo >= MIN_OPP) & (xi >= MIN_ISO)
    a, b, y = E(J.iso_a[k], J.opp_a[k]).to_numpy(), E(J.iso_b[k], J.opp_b[k]).to_numpy(), E(xi[k], xo[k]).to_numpy()
    def fit(ix): s = np.cov(y[ix], b[ix])[0, 1] / np.cov(a[ix], b[ix])[0, 1]; return s, y[ix].mean() - s * a[ix].mean()
    s, i0 = fit(np.arange(len(y))); g = np.random.default_rng(20261025)
    B = np.array([fit(g.integers(0, len(y), len(y))) for _ in range(2000)]); sl, sh = np.quantile(B[:, 0], [0.025, 0.975]); il, ih = np.quantile(B[:, 1], [0.025, 0.975])
    call = "R (restoring slowed)" if sl <= 1 <= sh and ih < 0 else ("Phi (holding fraction lowered)" if sh < 1 and il <= 0 <= ih else "neither")
    return (f"{c} against {d}: {int(k.sum())} tiles | slope {s:.4f} [{sl:.4f}, {sh:.4f}] | intercept {i0:+.4f} [{il:+.4f}, {ih:+.4f}] | {call}"), dict(slope=s, intercept=i0, call=call)
L = []
if len(sys.argv) > 3:
    for c, d in zip(sys.argv[3::2], sys.argv[4::2]): L.append(bar3(c, d)[0])
else:
    S = pd.read_csv(os.path.join(H, "graded_samples.csv")); Q = {}
    for f in glob.glob(os.path.join(D, "**", "*_q2.json"), recursive=True): q = json.load(open(f)); Q[q["gsm"]] = q
    S["held_err"] = S.gsm.map(lambda x: (Q.get(x) or {}).get("held_err")); S["held_opp"] = S.gsm.map(lambda x: (Q.get(x) or {}).get("held_opp"))
    L.append(f"counted {S.held_err.notna().sum()} of {len(S)} runs"); L += [f"{r.gsm} | {r.title} | held error {r.held_err} | opportunities {r.held_opp}" for r in S.itertuples()]
    ok = True
    for day in ("d6", "d14"):
        for sup in ("ChemieTek", "MedChemExpress"):
            def he(t):
                x = S[S.title.str.contains(t, regex=False) & S.title.str.contains(day, regex=False)]; x = x[x.held_err.notna()]
                return (x.held_err * x.held_opp).sum() / x.held_opp.sum() if len(x) else None
            v = [he("WT DMSO"), he(f"2uM GSK-3484862 {day} ({sup})"), he(f"10uM GSK-3484862 {day} ({sup})")]
            r = None not in v and v[0] < v[1] < v[2]; ok &= bool(r); L.append(f"bar 2, {day}, {sup}: DMSO {v[0]} < 2 uM {v[1]} < 10 uM {v[2]}: {'yes' if r else 'no'}")
    L.append(f"bar 2 (held error rises with dose on both days and both suppliers): {'met' if ok else 'not met'}")
    t, _ = bar3("SRR17261405", "SRR17261406"); L.append("bar 3, the deep pair (DMSO day 14 against 2 uM day 14): " + t)
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
