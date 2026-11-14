"""Is the laboratory offset a meter property? (exploration on spent counts, 2026-11-13; nothing sealed). Self-calibration the Planck way:
read each library from its interior calls only (three or more calls from a molecule end; chain Stage Q2 held_by_distance), away from the
end damage, and compare the referee laboratory (GSE173787, referee_counts) with the atlas laboratory (percall_out) type by type. If the
offset is the meter's end, the interior readings converge. Writes lab_offset_interior_summary.txt."""
import os, json, glob, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); TY = ("b cells", "cd4 t cells", "cd8 t cells", "monocytes")
CELL = {r["gsm"]: r["cell"].lower() for r in csv.DictReader(open(os.path.join(H, "referee_samples.csv")))}
def prof(x):
    d = {int(k): v for k, v in x["held_by_distance"].items()}
    rate = lambda ks: sum(d[k][0] for k in ks if k in d) / max(1, sum(d[k][1] for k in ks if k in d))
    return rate([1]), rate([2]), rate([3, 4, 5, 6])
R = {}
for f in glob.glob(os.path.join(H, "referee_counts", "*_count.json")):
    x = json.load(open(f)); R.setdefault(("referee", CELL[x["gsm"]]), []).append(prof(x))
for f in glob.glob(os.path.join(H, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_by_distance") and str(x.get("cell", "")).lower() in TY:
            R.setdefault(("atlas", str(x["cell"]).lower()), []).append(prof(x))
L = ["referee over atlas, by distance from the molecule end (1 = end call, interior = 3 or more calls in):"]
for t in TY:
    a, r = np.array(R[("atlas", t)]).mean(0), np.array(R[("referee", t)]).mean(0)
    L.append(f"{t:12} end {r[0] / a[0]:.3f}  next {r[1] / a[1]:.3f}  interior {r[2] / a[2]:.3f}   (atlas interior {a[2]:.4f}, referee {r[2]:.4f})")
L.append("Reading: the offset does not shrink in the interior, and it differs by type (CD8 most, CD4 least): not the meter's end, not one "
         "laboratory factor. CD8 T cells change most with donor age. Atlas donors: three, aged 50, 55, 56 (GEO); referee controls matched to new-onset MS"
         " (Ma et al. 2021 PNAS), ages in the supplement, not read: likely younger (lead).")
open(os.path.join(H, "lab_offset_interior_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
