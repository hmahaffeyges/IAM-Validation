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
# 2026-11-14: writer selection (A15) and the eraser share (A5)
w = mc.model2(cc.set(eraser_share=0.136))
ok &= check("eraser: stem-cell share lifts the candidate to 0.05154", abs(float(w["held_error_with_eraser"]) - 0.04454 / 0.864) < 5e-6)
ok &= check("eraser: mature B (share 0) leaves the thermal level", abs(float(mc.model2(cc)["held_error_with_eraser"]) - float(Lc["eps_hold"])) < 1e-12)
ok &= check("eraser reading is IAM and names the driven passage", w["held_error_with_eraser"].origin == "IAM" and "driven_passage" in w["held_error_with_eraser"].relations)
st = [float(mc.model2(cc.set(D_select=D))["stray_mark_rate"]) for D in (47.4, 80.0)]
ok &= check("writer: stray marks 0.0207-0.0123 for D 47.4-80 (A15)", abs(st[0] - 1 / 48.4) < 1e-12 and abs(st[1] - 1 / 81) < 1e-12)
ok &= check("writer: the stray-mark reading is not IAM", mc.model2(cc)["stray_mark_rate"].origin != "IAM")
ok &= check("IAM off: no eraser-on-candidate reading", "held_error_with_eraser" not in mc.model2(cc.set(iam=False)))
T = mc.law_card().set(baseline_candidate=True)
e0 = float(mc.ledger(T)["eps_hold"]); e1 = float(mc.ledger(T.set(T_cell=311.15))["eps_hold"])
ok &= check("derived passage: held error 0.04454 from M alone (A4)", abs(e0 - 0.04454) < 2e-5)
ok &= check("derived passage: one kelvin at the nucleus moves the held error by 0.30 % (A8 size test)", abs(e1 / e0 - 1.0030) < 3e-4)
lo = float(mc.ledger(T.set(dG_ATP=53500.0))["eps_hold"]); hi = float(mc.ledger(T.set(dG_ATP=61700.0))["eps_hold"])
ok &= check("derived passage: the tissue coin band 53.5-61.7 kJ/mol moves it at most x1.14", abs(lo / hi - 1.143) < 0.003)
ok &= check("England's bound at the derived error: ln(1/eps0) = 3.11 kT (E7)", abs(math.log(1 / e0) - 3.111) < 0.002)
lv = mc.law_card()["A_type_level"].value
ok &= check("each type's level is measured, not IAM, and spans more than the energy balance can make", mc.law_card()["A_type_level"].origin == "measured" and max(lv.values()) / min(lv.values()) > 1.143)
ok &= check("islands meter slope is measured and inside its bootstrap range", mc.law_card()["islands_meter_slope"].origin == "measured" and 0.69 <= mc.law_card()["islands_meter_slope"].value <= 1.22)
eps_t = 0.04454
ok &= check("5hmC: with no transient 5hmC the bisulfite reading is the true held error (A8H)", mc.bisulfite_held_error(eps_t, 0.02, 0.0) == eps_t)
ok &= check("5hmC: half of a 2 % share in transit lowers the reading by 0.01", abs(mc.bisulfite_held_error(eps_t, 0.02, 0.5) - (eps_t - 0.01)) < 1e-12)
f_n = mc.transient_fraction_needed(0.81, eps_t, 0.10)
ok &= check("5hmC: a neuron-like type (A 0.81, 5hmC 10 %) needs ~8.5 % of its 5hmC in transit", abs(f_n - 0.0846) < 0.001)
ok &= check("5hmC: a type reading above 1 cannot be explained by hiding", mc.transient_fraction_needed(1.06, eps_t, 0.02) is None)
lc = mc.law_card()
ok &= check("5hmC: blood shares are measured, the transient fraction is an open lever at 0", lc["hmC_share_blood"].origin == "measured" and lc["hmC_transient_fraction"].value == 0.0)
Lg = mc.ledger(mc.law_card())
ok &= check("speed ceilings for one coin: Bekenstein 3.85e15 bits/s, Margolus-Levitin 5.41e14/s, thermal 6.46e12/s (saturation)",
            abs(float(Lg["bekenstein_bits_s"]) / 3.85e15 - 1) < 0.005 and abs(float(Lg["margolus_levitin_ops_s"]) / 5.41e14 - 1) < 0.005 and abs(float(Lg["thermal_rate_s"]) / 6.46e12 - 1) < 0.005)
ok &= check("ceiling ratio 2 pi^2 M/ln2 = 596 at M = 20.94, equal to Bekenstein over thermal", abs(float(Lg["ceiling_ratio"]) - 596.3) < 0.5 and abs(float(Lg["ceiling_ratio"]) - float(Lg["bekenstein_bits_s"]) / float(Lg["thermal_rate_s"])) < 1e-6 * float(Lg["ceiling_ratio"]))
ok &= check("a restore's Landauer efficiency ln2/(2M) = 1.655 %", abs(float(Lg["restore_landauer_efficiency"]) - 0.01655) < 0.00005)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
