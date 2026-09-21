"""lab_zero.py — the third layer of the healthy reference (PROC-PANEL-03, 2026-09-20).

A single sample's absolute reading is  A'' = A_mapped - c(decade) - z_lab
where c is the reference healthy age curve (reference_age_curve_v1.json) and z_lab is the laboratory's
zero, measured ONCE on a panel of >= 40 healthy arrays run through the same Stage 1 and map:
    z_lab = median_i [ A_i - c(decade_i) ] - 1.0
Any age mix is acceptable for the panel (PROC-PANEL-03 P4''). Readings without a lab zero are labelled
lab_zero=UNSET and are NOT reportable as absolute (cf. scale=UNMAPPED, LESSON-SCALE-01).
Record: Testing_and_Code/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/.
"""
import json, os, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
CURVE_PATH = os.path.join(HERE, "Runtime Matrices", "A_Scoring_Module", "reference_age_curve_v1.json")
MIN_PANEL = 40

def load_curve(path=CURVE_PATH):
    return {int(k): float(v) for k, v in json.load(open(path))["curve"].items()}

def age_reference(age, curve=None):
    """c(decade) for an age; nearest available decade if the exact one is absent from the curve."""
    curve = curve or load_curve(); dec = int(age // 10 * 10)
    return curve[dec] if dec in curve else curve[min(curve, key=lambda k: abs(k - dec))]

def compute_lab_zero(panel_A, panel_ages, curve=None):
    """z_lab from a healthy panel. Refuses a panel smaller than MIN_PANEL."""
    if len(panel_A) < MIN_PANEL or len(panel_A) != len(panel_ages):
        raise ValueError(f"lab zero needs >= {MIN_PANEL} healthy arrays with ages (got {len(panel_A)})")
    curve = curve or load_curve()
    return statistics.median(a - age_reference(g, curve) for a, g in zip(panel_A, panel_ages)) - 1.0

def absolute_reading(A_mapped, age, lab_zero=None, curve=None):
    """Returns dict(A_abs, reportable, lab_zero). lab_zero=None -> UNSET, reportable False."""
    if lab_zero is None:
        return {"A_abs": None, "reportable": False, "lab_zero": "UNSET", "reason": "no laboratory zero (PROC-PANEL-03)"}
    return {"A_abs": A_mapped - age_reference(age, curve) - lab_zero, "reportable": True, "lab_zero": lab_zero}
