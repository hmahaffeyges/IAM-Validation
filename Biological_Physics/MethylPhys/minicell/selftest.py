"""Self-tests for the mini simulators (run: python3 selftest.py). Every check must pass before any output is used or shown.
Checks: origin rules (nothing is called IAM without its relation), units and limits, the IAM switch (off: the law's readings absent and
nothing moves with temperature or drive), and the energy floor (the holding energy never below zero; the coin never below kT ln2)."""
import sys, math
sys.path.insert(0, "."); import simcore as sc, minicell as mc

def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); return ok

ok = True
try: sc.Reading(0.1, "IAM"); ok &= check("an IAM reading without a relation is refused", False)
except AssertionError: ok &= check("an IAM reading without a relation is refused", True)
try: sc.Reading(0.1, "established", ["hold_energy"]); ok &= check("a non-IAM reading naming an IAM relation is refused", False)
except AssertionError: ok &= check("a non-IAM reading naming an IAM relation is refused", True)

c = mc.law_card(); on = mc.model2(c); off = mc.model2(c.set(iam=False))
ok &= check("canon M = 20.94 at 37 C", abs(float(on["M"]) - 20.94) < 0.01)
ok &= check("canon eps0 = 0.0438 at 37 C (measured E_hold reproduced, not predicted)", abs(float(on["eps_hold"]) - 0.04376) < 2e-4)
ok &= check("kinetics deliver the law's level", abs(float(on["held_error_marked"]) - float(on["eps_hold"])) < 1e-5)
ok &= check("every reading carries an origin", all(getattr(v, "origin", None) in sc.ORIGINS for v in list(on.values()) + list(off.values())))
ok &= check("IAM off: no IAM readings", all(v.origin != "IAM" for v in off.values()))
hot = mc.model2(c.set(T_cell=313.15, iam=False))
ok &= check("IAM off: held error does not move with temperature", abs(float(hot["held_error_marked"]) - float(off["held_error_marked"])) < 1e-9)
hot_on = mc.model2(c.set(T_cell=313.15))
ok &= check("IAM on: warmer cell holds worse", float(hot_on["held_error_marked"]) > float(on["held_error_marked"]))
weak = mc.model2(c.set(dG_ATP=40000.0))
ok &= check("IAM on: a smaller coin holds worse", float(weak["held_error_marked"]) > float(on["held_error_marked"]))
ok &= check("limit: no delivery (phi -> 0) gives the coin flip", abs(float(mc.ledger(c.set(phi=1e-9))["eps_hold"]) - 0.5) < 1e-6)
ok &= check("floor: the coin exceeds kT ln2 (M > ln2)", float(on["M"]) > math.log(2))
ok &= check("floor: kT ln2 at 37 C = 2.968e-21 J", abs(float(on["landauer_floor_J"]) - 2.968e-21) < 1e-24)
# 2026-11-12: the baseline candidate switch (PHY-083)
cc = c.set(baseline_candidate=True); Lc = mc.ledger(cc)
ok &= check("candidate: eps = 1/(1+odds(2M)) = 0.04454 (canon eps0_candidate)", abs(float(Lc["eps_hold"]) - 0.04454) < 5e-6)
ok &= check("candidate: E_hold = 3.066 kT (canon E_hold_candidate)", abs(float(Lc["E_hold_kT"]) - 3.066) < 5e-4)
ok &= check("candidate readings name the driven passage", "driven_passage" in Lc["eps_hold"].relations)
ok &= check("candidate: a dearer SAM holds better", float(mc.ledger(cc.set(SAM_cost=2.1))["eps_hold"]) < float(Lc["eps_hold"]))
ok &= check("candidate off: measured phi as before", abs(float(mc.ledger(c)["E_hold_kT"]) - 0.1473 * float(mc.ledger(c)["M"])) < 1e-9)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
