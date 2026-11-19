import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8L v2 (register '### A8L plan' item 7; prediction written before this ran). Exploration on GitHub with the cell simulator on main.
One lever at a time, all others at their card values; each lever over its MEASURED span where the record has one, else x0.5-x2; read the held
channel and the islands (unmarked-side) channel; compare with A8's healthy spread x1.39 (0.77-1.07). Adds the levers entered after the 11-16 sweep
(eraser share, writer selection odds, meter conversion failure and T->C slip, 5hmC transient share) and runs plan section 6, the author's call-up
burst (reservist vs enlisted), with the law off so the kinetics carry the level. Nothing fitted to data."""
import sys, os, math, json
import numpy as np
sys.path.insert(0, "Biological_Physics/MethylPhys/minicell"); import minicell as mc
TARGET = 1.39
card = mc.law_card().set(baseline_candidate=True)
def eps_on(c): return float(mc.ledger(c)["eps_hold"])
def out2(c): return mc.model2(c)
rows = []
def row(group, lever, span, held_f, isl_f, note=""):
    rows.append(dict(group=group, lever=lever, span=span, held_factor=held_f, islands_factor=isl_f, can_make_spread=bool(held_f is not None and max(held_f, 1 / held_f) >= TARGET), note=note))
# 1. energy levers, IAM on, over measured spans
for k, lo, hi, src in (("dG_ATP", 53500.0, 61700.0, "tissue coin band 53.5-61.7 kJ/mol (F114)"), ("T_cell", 309.15, 311.15, "37 +/- 1 C (x0.5-x2 analogue not physiological)"),
                       ("SAM_cost", 1.5, 2.5, "x0.75-x1.25 (A8L v1)")):
    a, b = eps_on(card.set(**{k: lo})), eps_on(card.set(**{k: hi})); row("energy (IAM on)", k, [lo, hi], max(a, b) / min(a, b), 1.0, src)
# 2. eraser share (held channel), measured span 0-0.136
c0 = card.set(eraser_share=0.0); c1 = card.set(eraser_share=0.136)
h0, h1 = float(out2(c0)["held_error_with_eraser"]), float(out2(c1)["held_error_with_eraser"]); row("eraser", "eraser_share", [0.0, 0.136], h1 / h0, 1.0, "TET2/TET3 knockout shares (mouse)")
# 3. islands levers
def isl(c): return float(out2(c)["stray_mark_rate_as_read"])
for k, lo, hi, src in (("D_select", 47.4, 88.0, "two enzymes 47.4-80; monocytes 88"), ("conv_fail", 0.0022, 0.0088, "x0.5-x2 of the blood median"), ("tc_err", 0.00023, 0.00092, "x0.5-x2")):
    a, b = isl(card.set(**{k: lo})), isl(card.set(**{k: hi})); row("islands / meter", k, [lo, hi], 1.0, max(a, b) / min(a, b), src)
row("eraser", "eraser_stray_clean", [0.13, 0.23], 1.0, None, "not in the simulator's readings: NOT ASSESSED")
# 4. 5hmC read as held (meter), shares measured in blood 0.0122-0.0217, transient 0-1
e0 = eps_on(card)
for sh in (0.0122, 0.0217):
    r1 = mc.bisulfite_held_error(e0, sh, 1.0) / e0; row("meter (5hmC)", f"hmC_transient_fraction 0->1 at share {sh}", [0.0, 1.0], 1 / r1 if r1 > 0 else float("inf"), 1.0, "lowers the read only (types below 1)")
# 5. kinetic levers, law off (operating point UNCALIBRATED: stated)
m, d, e, lag = (card.v(f"held_marked.{x}") for x in ("m", "d", "e")) + (card.v("lag"),)
def lvl(mm, dd, ee, ll): U, Hh, M = mc.class_state(mm, dd, ee, ll); return 1 - (M + Hh / 2)
for name, idx in (("loss per cycle 1-m", 0), ("writing d", 1), ("erasing e", 2), ("lag", 3)):
    p = [m, d, e, lag]
    def at(f):
        q = list(p)
        if idx == 0: q[0] = 1 - (1 - m) * f
        else: q[idx] = p[idx] * f
        return lvl(*q)
    a, b = at(0.5), at(2.0); row("kinetic (law off, uncalibrated)", name, ["x0.5", "x2"], max(a, b) / min(a, b), 1.0, "operating point uncalibrated")
# 6. call-up burst (plan section 6), law off
T, _ = mc._cycle_matrix(m, d, e, lag)
w = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]; lam2 = float(w[1])
def held_of(p): return 1 - (p[2] + p[1] / 2)
wv, v = np.linalg.eig(T.T); p0 = np.real(v[:, np.argmin(abs(wv - 1))]); p0 = p0 / p0.sum(); base = held_of(p0)
burst = {}
for B in (2.0, 5.0):
    TB, _ = mc._cycle_matrix(m, min(d * B, 0.999), min(e * B, 0.999), lag)
    for nb in (1, 3, 10):
        p = p0.copy()
        for _ in range(nb): p = p @ TB
        trace = {"end_of_burst": held_of(p) / base}
        for k in range(1, 101):
            p = p @ T
            if k in (1, 5, 20, 100): trace[f"after_{k}"] = held_of(p) / base
        burst[f"x{B:g} for {nb} cycles"] = trace
rep = {}
for nbursts in (1, 3, 10):
    TB, _ = mc._cycle_matrix(m, min(d * 5, 0.999), min(e * 5, 0.999), lag); p = p0.copy()
    for _ in range(nbursts):
        for _ in range(3): p = p @ TB
        for _ in range(20): p = p @ T
    rep[f"{nbursts} call-ups (x5, 3 cycles, 20 rest)"] = held_of(p) / base
out = dict(target=TARGET, eps_law=e0, rows=rows, burst=dict(lambda2=lam2, relax_cycles=(-1 / math.log(lam2) if 0 < lam2 < 1 else None), base_held=base, single=burst, repeated=rep))
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8l_lever_sweep_v2.json", "w"), indent=1)
for r in rows: print("::notice::A8L2SWEEP " + json.dumps(r))
print("::notice::A8L2SWEEP BURST " + json.dumps(out["burst"]))
