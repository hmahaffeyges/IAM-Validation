#!/usr/bin/env python3
"""Forward check of the public simulators (runs on GitHub explore.yml, never the laptop), 2026-11-15. (1) The cosmos simulator's self-tests,
including the 2026-11-15 additions (matter's own black-hole half, Lambda's own-horizon temperature, the black-hole module). (2) The cell
simulator, law first (law_card, model2), with IAM on: the share kept per unit paid against the pump ceiling, and the stray-mark rate as the meter
reads it (writer 1/(1+D) + conversion failure + the T->C slip) beside what PHY-091 measured (referee blood 0.0191 as read, 0.0143 corrected; the
atlas laboratory ~0.007-0.008 corrected). Writes xout/sim_forward_summary.txt."""
import os, sys, io, contextlib, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "numpy"], check=False)
H = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(H, "..", "..", "..", "minicell")); sys.path.insert(0, MC)
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
import minicosmos as MO, minicell as MI
buf = io.StringIO()
with contextlib.redirect_stdout(buf): ok = MO.minicosmos_selftest()
L = ["(1) cosmos simulator self-tests:"] + ["  " + l for l in buf.getvalue().strip().split("\n")]
card = MI.law_card(); out = MI.model2(card)
g = lambda k: float(out[k]) if k in out else None
L += ["(2) cell simulator, law first, IAM on:",
      f"  M {g('M'):.3f}; E_hold {g('E_hold_kT'):.4f} kT; held error the law sets {g('eps_hold'):.5f}",
      f"  share kept per unit paid {g('held_per_paid'):.4f} against the pump ceiling {g('pump_ceiling'):.4f}: {'below' if g('held_per_paid') < g('pump_ceiling') else 'ABOVE'}",
      f"  writer's stray marks 1/(1+D) {g('stray_mark_rate'):.4f}; as the meter reads it {g('stray_mark_rate_as_read'):.4f}",
      f"  measured (PHY-091): referee blood as read 0.0191 (ratio {g('stray_mark_rate_as_read') / 0.0191:.3f}); writer corrected 0.0143 (ratio {g('stray_mark_rate') / 0.0143:.3f}); atlas corrected ~0.0075 (ratio {g('stray_mark_rate') / 0.0075:.3f})",
      f"  the D that the referee lab's corrected value implies: {1 / 0.0143 - 1:.1f}; the atlas lab's: {1 / 0.0075 - 1:.0f} (published 10-80, median 20; card uses {card.v('D_select')})"]
L.append("ALL COSMOS SELF-TESTS PASS" if ok else "COSMOS SELF-TESTS: SOME FAILED")
open(os.path.join(OUT, "sim_forward_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0 if ok else 1)
