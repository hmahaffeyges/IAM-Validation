#!/usr/bin/env python3
"""A8D (register A8, the history term), exploration, desk model; runs ON GITHUB (never the laptop). Version 3, 2026-11-16 ~2:50 AM Pacific.
v2 failed its own check (held error 0.23-0.26 vs A4's 0.0445): it read end-of-cycle states and used the assumed default card. v3 uses the
mini cell's law card (law_card: maintenance calibrated on the hairpin wild type, the writer solved to the law's held level) and its own
phase-averaged reading (class_state), and runs the cycle forward from a fresh cell.
CHECK FIRST: the held-marked steady state must reproduce the law card's held error (~0.045); if not, nothing below is reported as a result.
Then (unmarked side, where a copy can lock in a writing error): the held-unmarked class's false marks from a fresh cell, cycle by cycle, and
the cycles to level off; compared afterwards with the clock sites' memory-minus-naive gain (lead). The held side is shown only as its balance
(gate rule 25; the author: the cost is paid per operation). Nothing fitted.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell")
import minicell as mc

card = mc.law_card(); v = card.v; lag = v("lag")
o = mc.model(card); target = o["held_error_marked"]
out = {"law_card_held_error": target, "rows": []}

def forward(cls, start, n=60, phases=20):
    m, d, e = v(cls + ".m"), v(cls + ".d"), v(cls + ".e")
    T, _ = mc._cycle_matrix(m, d, e, lag)
    obs = [mc._cycle_matrix(m, d, e, lag, t_obs=(j + 0.5) / phases)[1] for j in range(phases)]
    p, lv = np.array(start, float), []
    for _ in range(n + 1):
        q = np.mean([p @ M for M in obs], axis=0); lv.append(q[2] + q[1] / 2); p = p @ T
    return lv, (m, d, e)

held, rates = forward("held_marked", [0.0, 0.0, 1.0])
err = [1 - x for x in held]; ok = abs(err[-1] - target) <= 0.1 * target
print(f"::notice::CHECK held-marked from a fresh cell reaches {err[-1]:.5f} after 60 cycles vs the law card's {target:.5f}: {'PASS' if ok else 'FAIL'} (rates m {rates[0]:.4f} d {rates[1]:.4g} e {rates[2]:.4g}, lag {lag})")
out["held_check"] = dict(pass_=ok, after=[round(x, 6) for x in err[:61:10]])
if ok:
    for cls in ("held_unmarked",):
        lv, r = forward(cls, [1.0, 0.0, 0.0])
        n90 = next((i for i, x in enumerate(lv) if x >= 0.9 * lv[-1]), None)
        out["rows"].append(dict(cls=cls, rates=r, false_marks=[round(x, 6) for x in lv[:61:5]], steady=round(lv[-1], 6), cycles_to_90pct=n90))
        print(f"::notice::{cls} (rates m {r[0]:.4f} d {r[1]:.4g} e {r[2]:.4g}): false marks after 0,5,10,20,40,60 cycles "
              f"{[round(lv[i], 5) for i in (0, 5, 10, 20, 40, 60)]}; steady {lv[-1]:.5f}; 90 % after {n90} cycles")
    print("::notice::compare (lead, register A8): clock sites minus the islands meter, memory minus naive 0.007-0.009 over ~10-15 counted divisions; islands meter ~0.004")
else:
    print("::notice::check failed: no result reported; the held-marked class does not reproduce the law card's level from a fresh cell")
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8d_history_forward.json", "w"), indent=1, default=float)
