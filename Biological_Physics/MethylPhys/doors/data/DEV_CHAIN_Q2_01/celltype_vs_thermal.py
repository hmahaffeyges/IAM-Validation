"""Register A8 / A16, exploration on spent counts only (2026-11-13; nothing sealed). For each cell type: its zero-loose reading against the
THERMAL level (the derived per-site loss, CANON eps0_candidate, no eraser) pushed through the counter at the type's own clustering
(forward_out fc_* at eraser ratio 1, interpolated on coupling). Then the two things that could move a type: an eraser share s (raises the
reading: thermal/(1 - s)) and the type's own coin (a larger coin lowers the thermal level). A type reading BELOW thermal cannot be explained
by any eraser share; at the canon coin it needs a larger coin (or a meter offset). For those, the coin that puts the type exactly at thermal
with s = 0 is the smallest coin consistent with it, compared with the measured tissue band 53.5-61.7 kJ/mol (CANON dG_ATP band).
Readings: the referee laboratory's four types (read like the atlas laboratory, from referee_counts; the sealed scorer's own-slope intercepts from referee_out.csv listed beside them) and the atlas laboratory's types
(percall_out: 1-3 samples per type, read at zero loose along IAM-A's slope b = 0.034625; neurons set aside). Writes celltype_vs_thermal.csv
and celltype_vs_thermal_summary.txt."""
import os, json, glob, math, csv
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); B = 0.034625
K = json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]
R, T = 8.314462618, K["T_cell"]["value"]; band = K["dG_ATP"].get("band", [53500.0, 61700.0])
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
e_site = lambda dG: 1 / (1 + odds(2 * dG / (R * T)))
e54 = e_site(54000.0)
G = [(pd.read_csv(f).coupling.mean(), pd.read_csv(f).held_err_length.mean()) for f in glob.glob(os.path.join(H, "forward_out", "fc_*.csv"))]
G.sort(); gc, gh = np.array([g[0] for g in G]), np.array([g[1] for g in G])
thermal = lambda c: float(np.interp(c, gc, gh))          # lone-loss through the counter, canon coin, no eraser
def coin_for(target_site):
    lo, hi = 40000.0, 90000.0
    for _ in range(80):
        m = (lo + hi) / 2
        if e_site(m) > target_site: lo = m
        else: hi = m
    return (lo + hi) / 2
rows = []
for r in csv.DictReader(open(os.path.join(H, "referee_out.csv"))):
    rows.append(dict(lab="referee (own slope)", cell=r["cell"], n=int(r["n"]), reading=float(r["intercept"]), coupling=float(r["coupling"])))
CELL = {r["gsm"]: r["cell"].lower() for r in csv.DictReader(open(os.path.join(H, "referee_samples.csv")))}
RF = {}
for f in glob.glob(os.path.join(H, "referee_counts", "*_count.json")):
    x = json.load(open(f)); RF.setdefault(CELL[x["gsm"]], []).append((x["held_err_length"] - B * x["loose_share"], x["coupling"]))
for cell, v in RF.items():   # the same reading as the atlas laboratory's: zero loose along IAM-A's slope
    rows.append(dict(lab="referee", cell=cell, n=len(v), reading=float(np.mean([q for q, _ in v])), coupling=float(np.mean([c for _, c in v]))))
A = {}
for f in glob.glob(os.path.join(H, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_err_length") is not None and "neuron" not in str(x.get("cell", "")).lower():
            A.setdefault(str(x["cell"]).lower(), []).append((x["held_err_length"] - B * x["loose_share"], x["coupling"]))
for cell, v in A.items():
    rows.append(dict(lab="atlas", cell=cell, n=len(v), reading=float(np.mean([a for a, _ in v])), coupling=float(np.mean([c for _, c in v]))))
for r in rows:
    th = thermal(r["coupling"]); r["thermal"] = th; r["ratio"] = r["reading"] / th
    r["eraser_share_if_canon_coin"] = 1 - th / r["reading"]               # negative = below thermal: no eraser share can explain it
    r["coin_kJ_if_no_eraser"] = coin_for(e54 * r["reading"] / th) / 1000   # the coin that puts it exactly at thermal with s = 0
    r["coin_in_band"] = band[0] / 1000 <= r["coin_kJ_if_no_eraser"] <= band[1] / 1000
O = pd.DataFrame(rows).sort_values(["lab", "ratio"]); O.to_csv(os.path.join(H, "celltype_vs_thermal.csv"), index=False)
a = O[O.lab == "atlas"]
L = [f"thermal through the counter at the canon coin: {gh.min():.5f}-{gh.max():.5f} over coupling {gc.min():.2f}-{gc.max():.2f}",
     f"atlas laboratory: {len(a)} types; below thermal {int((a.ratio < 1).sum())}; reading/thermal median {a.ratio.median():.3f} (range {a.ratio.min():.3f}-{a.ratio.max():.3f})",
     f"atlas: implied eraser share at the canon coin, for types above thermal: median {a[a.ratio >= 1].eraser_share_if_canon_coin.median():.3f}",
     f"atlas: coin that puts a type at thermal with no eraser: median {a.coin_kJ_if_no_eraser.median():.1f} kJ/mol; inside the measured band 53.5-61.7: {int(a.coin_in_band.sum())} of {len(a)}",
     "", "referee laboratory, read like the atlas (zero loose along IAM-A's slope; own-slope reading listed after):", O[O.lab.str.startswith("referee")][["cell", "n", "reading", "coupling", "thermal", "ratio", "eraser_share_if_canon_coin", "coin_kJ_if_no_eraser"]].round(4).to_string(index=False),
     "", "atlas, the four referee types:", a[a.cell.isin(["b cells", "cd4 t cells", "cd8 t cells", "monocytes"])][["cell", "n", "reading", "coupling", "thermal", "ratio", "coin_kJ_if_no_eraser"]].round(4).to_string(index=False),
     "", "Exploration on spent counts; nothing sealed. Thermal level and eraser share at the canon coin; types below thermal point at the coin (A16) or the meter."]
rng = np.random.default_rng(2); full = coarse = 0; N = 1000; TT = sorted(RF)
for _ in range(N):   # null test: random halves of the referee laboratory's libraries per type
    h = [{}, {}]
    for t in TT:
        v = np.array([q for q, _ in RF[t]]); i = rng.permutation(len(v)); k = len(v) // 2; h[0][t], h[1][t] = v[i[:k]].mean(), v[i[k:]].mean()
    full += sorted(TT, key=h[0].get) == sorted(TT, key=h[1].get)
    coarse += all(min(x["cd4 t cells"], x["cd8 t cells"]) > max(x["b cells"], x["monocytes"]) for x in h)
L.insert(4, f"null test (referee laboratory, {N} random half splits): the full four-type order repeats in {100 * full / N:.0f} %; "
            f"T cells above B cells and monocytes in both halves {100 * coarse / N:.0f} %")
open(os.path.join(H, "celltype_vs_thermal_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
