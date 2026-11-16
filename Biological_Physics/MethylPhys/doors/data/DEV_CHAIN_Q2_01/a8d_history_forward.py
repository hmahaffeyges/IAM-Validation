#!/usr/bin/env python3
"""A8D (register A8, the history term), exploration, desk model; runs ON GITHUB (never the laptop). Version 5, 2026-11-16 ~3:35 AM Pacific (v4's solver capped the writer at 0.9; DEV_MINICELL_01 calibrates it at 0.95).
v3's 'PASS' was empty: it compared the card with itself (0.638 vs 0.638); the law card's kinetic writer rate held_marked.d is still the
assumed 0.05, not solved to the law's level. v4 anchors each class on a MEASURED or DERIVED level and only then runs forward:
- held-marked: solve the writer d (m 0.9661 and e 0.10 as the card calibrates them) so the phase-averaged level equals 1 - eps from the
  law's own ledger (ledger(card)['eps_hold'], ~0.044); CHECK: a fresh cell run forward reaches that level; report cycles to get there.
- held-unmarked: solve d (m, e as the card has them, ASSUMED) so the steady stray marks equal the measured writer level genome-wide,
  0.0138 (PHY-093, meter removed); then the false marks from a fresh cell, cycle by cycle; compared afterwards with the clock sites'
  memory-minus-naive gain 0.007-0.009 over ~10-15 counted divisions (lead). Results depend on the assumed m and e of that class: a lead.
The held side is reported only as its balance (gate rule 25; the author: the cost is paid per operation). Nothing fitted to the A8 data.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell")
import minicell as mc

card = mc.law_card(); v = card.v; lag = v("lag")
eps_law = float(mc.ledger(card)["eps_hold"])
lvl = lambda q: q[2] + q[1] / 2
def steady(m, d, e): return lvl(mc.class_state(m, d, e, lag))
def solve_d(m, e, target, lo=1e-6, hi=0.999):
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if steady(m, mid, e) < target: lo = mid
        else: hi = mid
    return (lo * hi) ** 0.5
def forward(m, d, e, start, n=60, phases=20):
    T, _ = mc._cycle_matrix(m, d, e, lag); obs = [mc._cycle_matrix(m, d, e, lag, t_obs=(j + 0.5) / phases)[1] for j in range(phases)]
    p, out = np.array(start, float), []
    for _ in range(n + 1): out.append(lvl(np.mean([p @ M for M in obs], axis=0))); p = p @ T
    return out
out = {"eps_law": eps_law}
mh, eh = v("held_marked.m"), v("held_marked.e"); dh = solve_d(mh, eh, 1 - eps_law)
held = [1 - x for x in forward(mh, dh, eh, [0.0, 0.0, 1.0])]
ok = abs(held[-1] - eps_law) <= 0.05 * eps_law
n90 = next((i for i, x in enumerate(held) if abs(x - eps_law) <= 0.1 * eps_law), None)
print(f"::notice::CHECK held-marked (m {mh} e {eh}, d solved {dh:.4g}): fresh cell -> {held[-1]:.5f} after 60 cycles vs the law's eps {eps_law:.5f}: {'PASS' if ok else 'FAIL'}; within 10 % after {n90} cycles")
out["held"] = dict(d=dh, after=[round(held[i], 6) for i in (0, 1, 2, 5, 10, 20, 60)], cycles_to_10pct=n90, pass_=ok)
if ok:
    mu, eu = v("held_unmarked.m"), v("held_unmarked.e"); du = solve_d(mu, eu, 0.0138)
    un = forward(mu, du, eu, [1.0, 0.0, 0.0]); n90u = next((i for i, x in enumerate(un) if x >= 0.9 * un[-1]), None)
    print(f"::notice::held-unmarked (m {mu} e {eu} ASSUMED, d solved {du:.4g} to the measured 0.0138): false marks after 0,5,10,15,20,40 cycles "
          f"{[round(un[i], 5) for i in (0, 5, 10, 15, 20, 40)]}; 90 % of steady after {n90u} cycles; gain cycles 5->20 {un[20] - un[5]:.4f}")
    out["unmarked"] = dict(d=du, after=[round(un[i], 6) for i in (0, 5, 10, 15, 20, 40, 60)], cycles_to_90pct=n90u)
    print("::notice::compare (lead, register A8): clock sites minus the islands meter, memory minus naive 0.007-0.009 over ~10-15 counted divisions")
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8d_history_forward.json", "w"), indent=1, default=float)
