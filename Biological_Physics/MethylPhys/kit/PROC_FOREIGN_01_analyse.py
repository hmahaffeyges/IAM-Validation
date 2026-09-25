#!/usr/bin/env python3
"""PROC-FOREIGN-01 step 2: the mixtures against the pre-registered bars."""
import json, collections
import numpy as np

SIG = 0.02092                      # sigma_pooled, the band's own width
D = json.load(open("handoff/foreign01.json"))["rows"]
prior = {r["gsm"]: r for r in json.load(open("handoff/band01_arrays.json"))["arrays"]}
base = {r["gsm"]: r for r in D if r["foreign_class"] == "none"}
FR = sorted({r["f"] for r in D if r["f"] > 0})
CLS = sorted({r["foreign_class"] for r in D if r["foreign_class"] != "none"})
print("hosts %d | foreign classes %s | fractions %s" % (len(base), CLS, FR))

# ---- B6 first: the unspiked reading must equal what PROC-BAND-01 published
d6 = [abs(base[g]["A"] - prior[g]["immune"]["A_abs"]) for g in base
      if base[g]["A"] is not None and prior.get(g, {}).get("immune", {}).get("A_abs") is not None]
print("\nB6 unspiked A'' vs PROC-BAND-01: %d compared, max |delta| = %.3e -> %s"
      % (len(d6), max(d6), "MET" if max(d6) <= 1e-9 else "NOT MET"))

# ---- the detector threshold, set on the unspiked arrays ALONE and before any spiked array is scored
fh = np.array([base[g]["foreign"] for g in base])
# The pre-registration fixed the RULE, not the quantile: "a threshold set to give <= 0.05 false-positive
# rate on the 318 unspiked healthy arrays ... set on the healthy arrays alone, before any spiked array is
# scored". The first implementation used the 95th percentile and OVERSHOT that constraint - 16 of 318 fire,
# 0.0503 - because with 318 arrays the achievable rates are k/318 and 0.05 is not among them. The threshold
# that satisfies the pre-registered constraint is the largest one firing on at most 15 of 318 (0.0472). It is
# still chosen from the healthy arrays alone with no reference to spiked performance; the sensitivity it
# yields is then measured, not selected (2026-09-25).
_sorted = np.sort(fh)
_k = int(np.floor(0.05 * len(fh)))          # 15 of 318
THR = float(_sorted[len(fh) - _k - 1]) + 1e-9
print("\ndetector = 1 - (immune + progenitor + stem_adult); healthy median %.4f, 95th pct %.4f = the threshold"
      % (np.median(fh), THR))
print("B5 fires on unspiked healthy: %.3f (bar <= 0.05) -> %s"
      % ((fh > THR).mean(), "MET" if (fh > THR).mean() <= 0.05 else "NOT MET"))

# ---- B1 / B2 / B4 per fraction
print("\n%-10s %-9s %9s %9s %9s %9s" % ("class", "f", "med|dA|", "/2sig", "tier flip", "detected"))
rows = collections.defaultdict(dict)
for cls in CLS:
    for f in FR:
        sel = [r for r in D if r["foreign_class"] == cls and r["f"] == f]
        dA, flip, det = [], [], []
        for r in sel:
            b = base.get(r["gsm"])
            if r["A"] is None or b is None or b["A"] is None:
                continue
            dA.append(abs(r["A"] - b["A"]))
            flip.append(r["tier"] != b["tier"])
            det.append(r["foreign"] > THR)
        md = float(np.median(dA)); fl = float(np.mean(flip)); dt = float(np.mean(det))
        dt_flip = float(np.mean([d for d, fp in zip(det, flip) if fp])) if any(flip) else float("nan")
        rows[cls][f] = {"med_abs_dA": md, "over_2sig": md / (2 * SIG), "tier_flip": fl,
                        "detected": dt, "detected_among_flipped": dt_flip, "n": len(dA)}
        print("%-10s %-9.2f %9.4f %9.2f %9.3f %9.3f" % (cls, f, md, md / (2 * SIG), fl, dt))

# B1: smallest f where median |dA| > 2 sigma for some class; B2: >=25% flip there
b1f = {cls: next((f for f in FR if rows[cls][f]["med_abs_dA"] > 2 * SIG), None) for cls in CLS}
print("\nB1 smallest f with median |dA| > 2 sigma (0.0418): %s" % b1f)
b1 = any(f is not None and f <= 0.20 for f in b1f.values())
print("   bar: some class at f <= 0.20 -> %s" % ("MET" if b1 else "NOT MET"))
b2 = {cls: (rows[cls][b1f[cls]]["tier_flip"] if b1f[cls] else None) for cls in CLS}
b2met = any(v is not None and v >= 0.25 for v in b2.values())
print("B2 tier flip at that f: %s | bar >= 0.25 -> %s"
      % ({k: (round(v, 3) if v is not None else None) for k, v in b2.items()}, "MET" if b2met else "NOT MET"))
# B3 sensitivity at the smallest f clearing both
cand = [(cls, f) for cls, f in b1f.items() if f is not None and rows[cls][f]["tier_flip"] >= 0.25]
b3 = {c: rows[c][f]["detected"] for c, f in cand}
b3met = bool(b3) and all(v >= 0.90 for v in b3.values())
print("B3 detector sensitivity at those points: %s | bar >= 0.90 -> %s"
      % ({k: round(v, 3) for k, v in b3.items()}, "MET" if b3met else "NOT MET"))
# B4 every f where >=10% flip: detector must catch >=90% OF THE FLIPPED
b4rows = [(cls, f, rows[cls][f]["tier_flip"], rows[cls][f]["detected_among_flipped"])
          for cls in CLS for f in FR if rows[cls][f]["tier_flip"] >= 0.10]
b4met = bool(b4rows) and all((d >= 0.90) for _, _, _, d in b4rows)
print("B4 at every f with >=10%% flipping, share of flipped hosts detected:")
for cls, f, fl, d in b4rows:
    print("   %-10s f=%.2f  flipped %.3f  of those detected %.3f  %s" % (cls, f, fl, d, "ok" if d >= 0.90 else "BELOW"))
print("   -> %s" % ("MET" if b4met else "NOT MET" if b4rows else "no f reached 10% flipping"))

json.dump({"threshold": THR, "healthy_fpr": float((fh > THR).mean()), "b6_max_dA": max(d6),
           "per_class": {c: {str(f): rows[c][f] for f in FR} for c in CLS},
           "b1_first_f": b1f, "b1": bool(b1), "b2": b2met, "b3": b3, "b3_met": bool(b3met),
           "b4_rows": [{"class": c, "f": f, "flip": fl, "detected_among_flipped": d} for c, f, fl, d in b4rows],
           "b4_met": bool(b4met), "sigma_pooled": SIG},
          open("handoff/foreign01_results.json", "w"), indent=1)
print("\nwrote handoff/foreign01_results.json")
