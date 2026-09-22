"""Kit test for lab_zero.py (PROC-PANEL-03): refuses small panels; UNSET is not reportable; a synthetic lab offset is recovered."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MethylPhys/chain")); import lab_zero as LZ
curve = LZ.load_curve(); random.seed(1)
ages = [random.choice(range(20, 90)) for _ in range(60)]
A = [1.0 + LZ.age_reference(g, curve) - 0.046 + random.gauss(0, 0.02) for g in ages]   # a UCLA-like lab at -0.046
try: LZ.compute_lab_zero(A[:39], ages[:39]); raise SystemExit("FAIL: accepted a 39-array panel")
except ValueError: pass
z = LZ.compute_lab_zero(A, ages, curve); assert abs(z + 0.046) < 0.012, f"FAIL: recovered {z:+.4f}, expected -0.046"
r = LZ.absolute_reading(0.95, 70); assert r["reportable"] is False and r["lab_zero"] == "UNSET"
r = LZ.absolute_reading(1.0 + LZ.age_reference(70, curve) - 0.046, 70, lab_zero=z); assert abs(r["A_abs"] - 1.0) < 0.012
print(f"test_lab_zero: PASS (recovered lab zero {z:+.4f} for a -0.046 lab; 39-panel refused; UNSET not reportable)")
