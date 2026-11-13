"""Scores DEV-METHIONINE-01's energy-or-supply test (PHY-083) as sealed: each line's held error (Stage Q2 held_err_length, the one command
run_sample.py --dev-count-only through the erase_test pipeline; counts in out/) depleted over its own control, against each route's
prediction mapped through the synthetic at the controls' own molecule lengths (methionine_synthetic_lengths_summary.txt). MZGC2 not scored
(sealed). Writes score_methionine_summary.txt."""
import os, re, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
S = open(os.path.join(H, "methionine_synthetic_lengths_summary.txt")).read()
read = {float(a): float(b) for a, b in re.findall(r"planted x([\d.]+): read x([\d.]+)", S)}
P = np.log([1.0] + sorted(read)); Rd = np.log([1.0] + [read[k] for k in sorted(read)]); mp = lambda x: float(np.exp(np.interp(np.log(x), P, Rd)))
PRED = {"MZGC1": dict(energy=(1.013, 1.016), supply=(1.33, 1.43), c="DRR322659", d="DRR322660"),
        "MZGC3": dict(energy=(1.057, 1.077), supply=(3.33, 5.00), c="DRR322663", d="DRR322664")}
q = lambda g: json.load(open(os.path.join(H, "out", f"{g}_q2.json")))
L = ["DEV-METHIONINE-01 score (as sealed; predictions in predictions_by_line_summary.txt, mapped through the synthetic at the controls' lengths)"]
for k, v in PRED.items():
    C, D = q(v["c"]), q(v["d"]); r = D["held_err_length"] / C["held_err_length"]
    e = tuple(round(mp(x), 3) for x in v["energy"]); s = tuple(round(mp(x), 3) for x in v["supply"])
    L.append(f"{k}: control {C['held_err_length']:.5f} (loose {C['loose_share']:.3f}), depleted {D['held_err_length']:.5f} (loose {D['loose_share']:.3f}); "
             f"ratio {r:.4f}; energy route {e[0]}-{e[1]} {'met' if e[0] <= r <= e[1] else 'not met'}; supply route {s[0]}-{s[1]} {'met' if s[0] <= r <= s[1] else 'not met'}")
open(os.path.join(H, "score_methionine_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
