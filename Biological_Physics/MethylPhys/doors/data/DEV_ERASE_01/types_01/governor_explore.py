"""Governor lead (exploration on spent counts; 2026-11-13 evening; nothing sealed). P6 found two wild types holding alike while their
erasers differ. If the held error is REGULATED toward a common level with the eraser as one load (a governor holding frequency whatever the
load), then across every eraser knockout we hold, the wild types sit closer together than the knockouts, and ln(WT) moves less than one for
one with ln(KO): slope b of ln(WT) on ln(KO) below 1. If each eraser simply ADDS its share on top of an unregulated level, b is about 1 and
the wild types spread at least as much as the knockouts. Caution: laboratory offsets are common to a unit's wild type and knockout and pull
b toward 1, so b < 1 is evidence for regulation, b near 1 is not evidence against it.
Units: DEV_ERASE_01 recount (five stem-cell units, recount_units.csv), PHY-085 (blood progenitors, blood_01), PHY-086 P4 (mature B,
counts/) and P5 (iNKT, counts_p5/). Null test: b recomputed leaving each unit out. Writes governor_explore_summary.txt."""
import os, json, glob, csv
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, "..")
U = [(r["unit"], float(r["held_err_wt"]), float(r["held_err_ko"])) for r in csv.DictReader(open(os.path.join(E, "recount_01", "recount_units.csv")))]
B = {json.load(open(f))["gsm"]: json.load(open(f))["held_err_length"] for f in glob.glob(os.path.join(E, "blood_01", "*_count.json"))}
U += [("PHY-085 LSK", B["GSM7111686"], B["GSM7111688"]), ("PHY-085 Lin-neg", B["GSM7111689"], B["GSM7111691"])]
def arms(d, wt):
    C = [json.load(open(f)) for f in glob.glob(os.path.join(H, d, "*_count.json"))]
    return (np.mean([c["held_err_length"] for c in C if c["gsm"] in wt]), np.mean([c["held_err_length"] for c in C if c["gsm"] not in wt]))
U += [("P4 mature B", *arms("counts", ("GSM4907469", "GSM4907470"))), ("P5 iNKT", *arms("counts_p5", ("GSM1855585", "GSM1855586")))]
w, k = np.log([u[1] for u in U]), np.log([u[2] for u in U])
b = np.polyfit(k, w, 1)[0]
loo = [np.polyfit(np.delete(k, i), np.delete(w, i), 1)[0] for i in range(len(U))]
L = [f"{len(U)} eraser-knockout units; sd of ln(level) across units: wild type {w.std(ddof=1):.3f}, knockout {k.std(ddof=1):.3f}",
     f"slope of ln(WT) on ln(KO): b = {b:.3f} (regulation predicts below 1; added shares about 1); leave-one-out range {min(loo):.3f}-{max(loo):.3f}",
     ""] + [f"{u[0]:16} WT {u[1]:.5f} KO {u[2]:.5f} ratio {u[2] / u[1]:.3f}" for u in U]
w1, w2 = B["GSM7111686"], B["GSM7111689"]; k1, k2 = B["GSM7111688"], B["GSM7111691"]
L.append(f"Within one experiment (PHY-085, LSK over Lin-neg): wild type {w1 / w2:.3f}, knockout {k1 / k2:.3f}: the governor's direction, inside replicate noise; not decisive")
L.append("Exploration on spent counts from five laboratories; laboratory offsets dominate the spread and pull b toward 1.")
open(os.path.join(H, "governor_explore_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
