#!/usr/bin/env python3
"""A8D (register A8, the history term), exploration, desk model; runs ON GITHUB (never the laptop). Revised 2026-11-16 ~2:20 AM after the
gate (repro_check rule 25: the held error's balance is independent of division; author 2026-10-13: the cost is paid per operation) and the full same-donor set (central memory CD4: largest held rise, no clock-site rise) agreed.
Question 1 (the UNMARKED side, where division can lock in a writing error): at sites every cell keeps unmarked (the mini cell's
held_unmarked class), how many false marks does each copy cycle leave, starting from a fresh cell, and when does it level off? Compared
afterwards with the clock sites' gain (memory minus naive, ~5-9e-4 per counted copy, lead).
Question 2 (the HELD side, as a check only): the mini cell's held-marked steady state must equal the balance f/(f+g) whatever the interval;
reported as the steady state and the number of cycles to reach it from a fresh cell, never as a term that grows with copy cycles.
Uses the mini cell's OWN cycle model (minicell._cycle_matrix) and its default card's class rates; nothing fitted.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell")
import minicell as mc

card = mc.default_card(); v = card.v
out = {"unmarked": [], "held_check": []}
level = lambda p: p[2] + p[1] / 2
for d in (0.001, 0.002, 0.004, 0.008):                       # writing on an empty strand per cycle (class default 0.004, assumed)
    for e in (0.05, 0.10, 0.20):                             # erasing (class default 0.10, assumed)
        for lag in (0.05, 0.10, 0.20):
            T, _ = mc._cycle_matrix(v("held_unmarked.m"), d, e, lag)
            p, h = np.array([1.0, 0.0, 0.0]), []
            for n in range(41): h.append(level(p)); p = p @ T
            w, vv = np.linalg.eig(T.T); ss = np.real(vv[:, np.argmin(abs(w - 1))]); ss = ss / ss.sum()
            out["unmarked"].append(dict(d=d, e=e, lag=lag, false_marks=[round(x, 6) for x in h[:21:5]], steady=round(level(ss), 6),
                                        gain_first_cycle=round(h[1] - h[0], 6)))
            if d == 0.004 and e == 0.10 and lag == 0.10:
                print(f"::notice::unmarked class (d {d}, e {e}, lag {lag}): false marks after 0,5,10,15,20 cycles {[round(x, 5) for x in h[:21:5]]}; steady {level(ss):.5f}")
m, d, e = v("held_marked.m"), v("held_marked.d"), v("held_marked.e")
for lag in (0.05, 0.10, 0.20):
    T, _ = mc._cycle_matrix(m, d, e, lag)
    p = np.array([0.0, 0.0, 1.0]); n90 = None; w, vv = np.linalg.eig(T.T); ss = np.real(vv[:, np.argmin(abs(w - 1))]); ss = ss / ss.sum()
    for n in range(200):
        if abs((1 - level(p)) - (1 - level(ss))) <= 0.1 * (1 - level(ss)): n90 = n; break
        p = p @ T
    out["held_check"].append(dict(lag=lag, steady_held_error=round(1 - level(ss), 6), cycles_to_within_10pct=n90))
    print(f"::notice::held-marked check (lag {lag}): steady held error {1 - level(ss):.5f}; reached within 10 % after {n90} cycles from a fresh cell")
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8d_history_forward.json", "w"), indent=1)
print("::notice::target (lead, register A8): clock sites, memory minus naive ~0.007-0.009, ~5-9e-4 per counted copy")
