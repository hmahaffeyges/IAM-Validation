#!/usr/bin/env python3
"""A8L (register A8), exploration on GitHub: sweep every cell-simulator lever and report which can make each healthy type's level (x1.39).
Energy levers with IAM on (law sets the level); kinetic levers with the law off (class_state sets it). Nothing fitted to data."""
import sys, os, math, json
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell"); import minicell as mc
TARGET = 1.39
card = mc.law_card().set(baseline_candidate=True)
def held(c): return float(mc.ledger(c)["eps_hold"])
base = held(card); rows = []
for k, lo, hi in (("dG_ATP", 0.9, 1.1), ("T_cell", 0.99, 1.01), ("SAM_cost", 0.75, 1.25)):
    v = card.v(k); a, b = held(card.set(**{k: v * lo})), held(card.set(**{k: v * hi}))
    el = math.log(b / a) / math.log(hi / lo); need = math.exp(math.log(TARGET) / abs(el)) if el else float("inf")
    rows.append(("energy (IAM on)", k, v, round(el, 3), round(need, 2)))
m, d, e, lag = (card.v(f"held_marked.{x}") for x in ("m", "d", "e")) + (card.v("lag"),)
def lvl(mm, dd, ee, ll):
    U, Hh, M = mc.class_state(mm, dd, ee, ll); return 1 - (M + Hh / 2)
b0 = lvl(m, d, e, lag)
for name, idx in (("held_marked.m (maintenance per cycle, as loss 1-m)", 0), ("held_marked.d (writing)", 1), ("held_marked.e (erasing)", 2), ("lag (copier lag)", 3)):
    p = [m, d, e, lag]
    def at(f):
        q = list(p)
        if idx == 0: q[0] = 1 - (1 - m) * f
        else: q[idx] = p[idx] * f
        return lvl(*q)
    a, b = at(0.8), at(1.25); el = math.log(b / a) / math.log(1.25 / 0.8)
    need = math.exp(math.log(TARGET) / abs(el)) if abs(el) > 1e-6 else float("inf")
    rows.append(("kinetic (law off)", name, round(p[idx], 4), round(el, 3), round(need, 2)))
rows.append(("meter", "h: share of in-progress losses read as marked (5hmC)", 0.0, "read = true x (1-h)", f"h = {1 - 1 / TARGET:.2f} between types"))
out = [f"A8L lever sweep: derived held error {base:.5f} (IAM on); kinetic level {b0:.5f} (law off, m {m}, d {d}, e {e}, lag {lag}); target x{TARGET}"]
out += [f"{r[0]:18s} | {r[1]:55s} | at {r[2]} | elasticity {r[3]} | factor needed {r[4]}" for r in rows]
os.makedirs("xout", exist_ok=True); open("xout/a8l_lever_sweep.txt", "w").write("\n".join(out) + "\n")
print("::notice::" + " || ".join(out)[:3800])
