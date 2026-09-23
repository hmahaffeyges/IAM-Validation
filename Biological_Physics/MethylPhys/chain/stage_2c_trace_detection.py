#!/usr/bin/env python3
"""Stage 2c - is there any evidence of a trace class in this specimen?

PROC-SMALL-01, adopted 2026-09-23. A side channel: it adds a verdict and changes no existing number. The
composition, the immune gauge, the tiers and the sky are exactly as sealed.

Why this exists. The commissioned composition solve is a non-negative least squares fit, and a non-negativity
constraint pins a trace component at exactly zero: measured on 48 mixtures of real healthy blood, secretory
and cycling read 0.00 at every spike below 5 % on two of three donors, because a boundary solution cannot
respond to a small change. The information is in the data - the one donor whose solution sat in the interior
responded monotonically to every 0.25 % step - so the fix is a statistic that is not pinned.

What it computes, per class. Fit the class design WITHOUT that class (constrained, as the chain does), take
the residual, project the candidate class's atlas profile off the columns already in the fit, and regress one
on the other UNCONSTRAINED. The coefficient may go negative, which is what lets the statistic vary below the
boundary; a healthy donor sits at t = -14, not at zero. Addresses are weighted by inverse atlas posterior
variance, which is the whole of the improvement: unweighted, secretory is undetectable below 5 %; weighted,
it is detected at 2 % on every donor tested.

What it is allowed to say.
  * t >= the panel's threshold (the 95th percentile measured on 38 healthy donors who played no part in
    choosing the configuration) -> EVIDENCE OF EPITHELIAL-LIKE MATERIAL.
  * Naming the class requires about 5 %: at the 2 % limit a cycling spike lifts the secretory statistic
    marginally over its own threshold, so the classes are not separable there. The panel carries that caveat
    and this module repeats it in the verdict.
  * No A, no tier, no fraction below 5 %. A trace class cannot be SCORED in blood at any fraction a blood
    draw presents: its identity loci still carry ~99 % background, measured as A = 0.807 at zero spike
    against 0.994 for a pure specimen.
"""
import json
import os

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "Runtime Matrices", "trace_detection_panel_v1.json")
TRACE_CLASSES = ("secretory", "cycling")


def load_panel(path=None):
    with open(path or PANEL) as f:
        return json.load(f)


def _design(panel, betas):
    """R (addresses x classes), y, and inverse-variance weights, over the addresses this specimen has."""
    classes = panel["_provenance"]["classes"]
    addr = [a for a in panel["addresses"]
            if a in betas and isinstance(betas[a], (int, float)) and 0.0 <= betas[a] <= 1.0]
    if not addr:
        return None
    n, k = len(addr), len(classes)
    R = np.zeros((n, k))
    y = np.zeros(n)
    w = np.zeros(n)
    for i, a in enumerate(addr):
        means = panel["class_means"][a]
        row = [means[c] for c in classes if c in means]
        fill = float(np.mean(row)) if row else 0.5
        for j, c in enumerate(classes):
            R[i, j] = means.get(c, fill)
        sds = (panel["class_sds"].get(a) or {})
        v = [sds[c] ** 2 for c in classes if sds.get(c)]
        w[i] = 1.0 / float(np.mean(v)) if v else np.nan
        y[i] = float(betas[a])
    # Scale guard (2026-09-23). The thresholds were measured on RAW stage1_noob betas - the same input the
    # composition solver gets. Handing this panel the scale-mapped betas shifted the statistic by ~24 units
    # on a specimen whose true answer was "no evidence", so the mismatch is caught here rather than reported.
    guard = panel["_provenance"].get("scale_guard")
    if guard:
        mb = float(np.nanmean(y))
        if not (guard[0] <= mb <= guard[1]):
            return {"scale_mismatch": (mb, guard)}
    med = np.nanmedian(w[np.isfinite(w)]) if np.any(np.isfinite(w)) else 1.0
    w = np.where(np.isfinite(w) & (w > 0), w, med)
    w = w / np.mean(w)
    return {"R": R, "y": y, "w": w, "classes": classes, "n": n}


def detect(betas, panel=None, substrate=None):
    """-> {class: {t, threshold, detected, ...}, '_meta': {...}} for every trace class in the panel."""
    panel = panel or load_panel()
    d = _design(panel, betas)
    if isinstance(d, dict) and "scale_mismatch" in d:
        mb, guard = d["scale_mismatch"]
        return {"_meta": {"available": False,
                          "reason": (f"input betas are not on the scale these thresholds were measured on "
                                     f"(mean over the panel {mb:.4f}, expected {guard[0]:.3f}-{guard[1]:.3f}). "
                                     f"Stage 2c takes RAW stage1_noob betas, not the scale-mapped beta.")}}
    if d is None:
        return {"_meta": {"available": False,
                          "reason": "none of the panel's addresses are present in this specimen"}}
    R, y, w, classes, n = d["R"], d["y"], d["w"], d["classes"], d["n"]
    sw = np.sqrt(w)
    # Substrate guard (2026-09-23): the thresholds are a whole-blood measurement. On tissue the same panel
    # returned t = 0.6 and 3.8 for classes truly present at 12 % and 35 % - lower than a 5 % blood spike -
    # because the residual structure and the platform both differ. Anything else is reported uncalibrated.
    blood = substrate is None or str(substrate).lower().replace("_", " ").strip() in (
        "whole blood", "blood", "buffy coat", "peripheral blood")
    out = {"_meta": {"available": True, "n_addresses": n,
                     "calibrated_for": panel["_provenance"].get("substrate", "whole blood"),
                     "substrate_declared": substrate,
                     "calibrated_for_this_substrate": bool(blood),
                     "panel": os.path.basename(PANEL),
                     "panel_sha256_12": panel["_provenance"].get("sha256_12"),
                     "weighting": "inverse atlas posterior variance",
                     "attribution_requires": panel["_provenance"]["detection_limit"]
                     ["attribution_to_one_class"],
                     "detection_limit": panel["_provenance"]["detection_limit"]
                     ["epithelial_like_material_present"],
                     "cross_talk_caveat": panel["_provenance"]["cross_talk_caveat"],
                     "changes_no_existing_number": True}}
    for c in TRACE_CLASSES:
        if c not in classes or c not in panel["thresholds"]:
            continue
        j = classes.index(c)
        keep = [q for q in range(len(classes)) if q != j]
        A = R[:, keep] * sw[:, None]
        b = y * sw
        f0, _ = nnls(A, b)
        resid = b - A @ f0
        pc = R[:, j] * sw
        active = [i for i, v in enumerate(f0) if v > 1e-12]
        if active:
            Aa = A[:, active]
            coef, *_ = np.linalg.lstsq(Aa, pc, rcond=None)
            pt = pc - Aa @ coef
        else:
            pt = pc - pc.mean()
        denom = float(pt @ pt)
        thr = panel["thresholds"][c]["t95"]
        if denom <= 1e-12:
            out[c] = {"t": None, "threshold": thr, "detected": None,
                      "reason": "the class profile is collinear with the rest of the panel here"}
            continue
        beta = float(pt @ resid) / denom
        dof = max(n - len(active) - 1, 1)
        s2 = max(float(resid @ resid) / dof, 1e-18)
        t = beta / (s2 / denom) ** 0.5
        det = bool(t >= thr)
        out[c] = {"t": round(float(t), 3), "threshold": round(float(thr), 3),
                  "detected": (det if blood else None),
                  "null_median_t": round(panel["thresholds"][c]["null_median"], 2),
                  "verdict": (("EVIDENCE OF EPITHELIAL-LIKE MATERIAL - at this limit the class cannot be "
                               "named; attribution needs about 5 %" if det else
                               "no evidence above the healthy null") if blood else
                              ("UNCALIBRATED on this substrate - the threshold is a whole-blood measurement, "
                               "so no verdict is given; the statistic is printed for reference only")),
                  "reportable_fraction": False,
                  "reportable_A": False}
    return out


if __name__ == "__main__":
    import pickle
    import sys
    betas = pickle.load(open(sys.argv[1], "rb")) if len(sys.argv) > 1 else {}
    print(json.dumps(detect(betas), indent=1))
