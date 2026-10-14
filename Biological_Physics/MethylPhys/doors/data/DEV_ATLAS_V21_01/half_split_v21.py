"""Atlas v2.1 decision 17 (readiness L19): the two donor halves for the half-atlases. Written and committed 2026-10-14 BEFORE it is run.
No methylation value of any CpG site is read: only genotype probes (rs), sample metadata and cell names.

UNITS (who counts as one donor). Samples are joined into one donor unit by any of these links (union-find):
 a. ARRAYS, genotype: rs probe betas called 0 (< 0.2), 1 (0.3-0.7), 2 (> 0.8), else missing; two arrays with >= 30 probes called in both and
    concordance >= 0.90 are one person (the identity check's committed rule, identity_check_v21.py). Covers v2's 113 arrays (Moss2018,
    Salas2018, Salas2022; S3 results/ATLAS_V21/halfsplit/v2_rs.parquet) and v2.1's 472 build arrays (identity_v21_rs.parquet).
 b. METADATA donor id where the source gives one: v2.1 build 'donor' (within its series); GSE63409 stem-cell arrays 'subject' (hsc_manifest,
    normal marrow only), which were not genotyped.
 c. LOYFER (WGBS): GEO gives no donor id, so samples with the same (lab, age, sex) are ONE unit. This can merge two people; it can never split
    one person across the halves, which is the property the noise null needs. Samples missing any of the three stand alone.
 d. Tian2023 pools and ENCODE samples: each its own unit (a pool cannot be split).
ASSIGNMENT (fixed; no fit, no methylation value). Units are taken in the order of sha256("ATLAS_V21_HALF_17|" + unit key). Each unit goes to
the half where, summed over the cells it contributes samples to, that half currently holds fewer units of those cells; a tie goes by the
hash's last bit. A unit is never in both halves.
CHECKS. (1) no unit in both halves (by construction, verified); (2) no array in half A has concordance >= 0.90 with an array in half B;
(3) per cell: units in A, units in B; a cell with one unit is listed as NOT MEASURED in the other half.
Writes half_split_v21.csv (one row per sample) and sealed_half_split_v21.json (rule, counts, single-unit cells, sha256 of the csv).
FIX 2026-10-14, after the first run and before sealing: the stem-cell names now follow v2 exactly (label + " (bone marrow)", normal
subjects only, as v2 and the twin job); the first run used the bare labels. No rule changed.
Usage: python3 half_split_v21.py V2_RS_PARQUET V21_RS_PARQUET
"""
import os, sys, json, hashlib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V2RS = pd.read_parquet(sys.argv[1]); V21RS = pd.read_parquet(sys.argv[2])
S2 = pd.read_csv(os.path.join(MP, "atlas/v2/inputs/roster_samples.csv")); S2 = S2[S2.qc].drop_duplicates(["cell", "sample"])
H = pd.read_csv(os.path.join(MP, "atlas/v2/inputs/hsc_manifest.csv")); H = H[(H.status == "ok") & (H.subject_status == "normal")]   # as v2 and the twin job (v21_twin_terms_box.py line 161)
B = pd.read_csv(os.path.join(HERE, "atlas_v21_build_samples.csv")); B = B[B.state != "OUT"]
I = pd.read_csv(os.path.join(HERE, "identity_v21_arrays.csv")); I = I[(I.set == "build") & (I.status == "ok")]
LG = {d["gsm"]: d for d in json.load(open(os.path.join(HERE, "half_split_inputs_loyfer_geo.json")))}
rows = []
for r in S2.itertuples(): rows.append(dict(sample=r.sample, source=r.source, cell=r.cell, platform=r.platform))
for r in H.itertuples():
    lab = r.label.replace(" of normal bone marrow", "") + " (bone marrow)"; rows.append(dict(sample=r.gsm, source="GSE63409", cell=lab, platform="array450k", subject=r.subject))
base = dict(zip(I["sample"], I["base"]))
for r in B.itertuples(): rows.append(dict(sample=r.sample, source=r.series, cell=r.cell, platform=r.platform, donor=r.donor, rs_key=base.get(r.sample)))
T = pd.DataFrame(rows); T["key"] = T.source + "|" + T["sample"].astype(str) + "|" + T.cell
assert not T.key.duplicated().any(), T[T.key.duplicated()].head()
par = {k: k for k in T.key}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def join(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[max(ra, rb)] = min(ra, rb)
def link_groups(g):
    for ks in g:
        ks = list(ks)
        for k in ks[1:]: join(ks[0], k)
# a. genotype
T["rs_key"] = np.where(T.source.isin(["Moss2018", "Salas2018", "Salas2022"]), T["sample"], T.get("rs_key"))
G = pd.concat([V2RS, V21RS.reindex(columns=V2RS.columns)]); G = G[~G.index.duplicated()]
arr = T.dropna(subset=["rs_key"]); arr = arr[arr.rs_key.isin(G.index)]
assert len(arr) >= 0.98 * T.rs_key.notna().sum(), ("genotype rows missing", len(arr), T.rs_key.notna().sum())
ids = sorted(set(arr.rs_key)); v = G.reindex(ids).values; C = np.full(v.shape, np.nan)
C[v < 0.2] = 0; C[(v >= 0.3) & (v <= 0.7)] = 1; C[v > 0.8] = 2; ok = ~np.isnan(C); same_pairs = []
for i in range(len(ids)):
    both = ok[i] & ok[i + 1:]; m = both.sum(1); s = ((C[i] == C[i + 1:]) & both).sum(1)
    for j in np.nonzero((m >= 30) & (s >= 0.90 * np.maximum(m, 1)))[0]: same_pairs.append((ids[i], ids[i + 1 + j]))
bykey = arr.groupby("rs_key").key.apply(list).to_dict()
for a, b in same_pairs: join(bykey[a][0], bykey[b][0])
link_groups(arr.groupby("rs_key").key.apply(list))
# b. metadata donor ids
if "donor" in T: link_groups(T.dropna(subset=["donor"]).assign(d=lambda x: x.source + "|" + x.donor.astype(str)).groupby("d").key.apply(list))
link_groups(T.dropna(subset=["subject"]).groupby("subject").key.apply(list))
# c. Loyfer (lab, age, sex)
def lkey(s):
    d = LG.get(s, {}); f = [d.get("lab"), d.get("age"), d.get("sex")]
    return None if any(x in (None, "", "NA") for x in f) else "|".join(f)
L = T[T.source == "Loyfer2023"].assign(lk=lambda x: x["sample"].map(lkey)).dropna(subset=["lk"])
link_groups(L.groupby("lk").key.apply(list))
T["unit"] = [find(k) for k in T.key]
# assignment
cells_of = T.groupby("unit").cell.apply(lambda s: sorted(set(s))).to_dict()
order = sorted(cells_of, key=lambda u: hashlib.sha256(("ATLAS_V21_HALF_17|" + u).encode()).hexdigest())
cnt = {h: {} for h in "AB"}; half = {}
for u in order:
    load = {h: sum(cnt[h].get(c, 0) for c in cells_of[u]) for h in "AB"}
    h = "A" if load["A"] < load["B"] else "B" if load["B"] < load["A"] else ("A" if int(hashlib.sha256(("ATLAS_V21_HALF_17|" + u).encode()).hexdigest(), 16) % 2 == 0 else "B")
    half[u] = h
    for c in cells_of[u]: cnt[h][c] = cnt[h].get(c, 0) + 1
T["half"] = T.unit.map(half)
# checks
assert T.groupby("unit").half.nunique().max() == 1
hk = dict(zip(arr.rs_key, arr.key)); cross = [(a, b) for a, b in same_pairs if T.set_index("key").half[hk[a]] != T.set_index("key").half[hk[b]]]
assert not cross, ("same person across halves", cross[:5])
per = T.groupby("cell")[["half", "unit"]].apply(lambda g: pd.Series({"units_A": g[g.half == "A"].unit.nunique(), "units_B": g[g.half == "B"].unit.nunique(),
                                                  "samples_A": int((g.half == "A").sum()), "samples_B": int((g.half == "B").sum())})).astype(int)
single = per[(per.units_A == 0) | (per.units_B == 0)]
out = os.path.join(HERE, "half_split_v21.csv")
T.drop(columns=["key"]).sort_values(["cell", "source", "sample"]).to_csv(out, index=False)
per.to_csv(os.path.join(HERE, "half_split_v21_per_cell.csv"))
seal = {"rule": "half_split_v21.py (docstring), committed before it ran", "samples": int(len(T)), "units": int(T.unit.nunique()),
        "genotype_same_person_pairs": len(same_pairs), "cells": int(len(per)), "cells_in_both_halves": int(len(per) - len(single)),
        "cells_NOT_MEASURED_in_one_half": {c: ("A" if r.units_A == 0 else "B") + " empty" for c, r in single.iterrows()},
        "loyfer_units_from_lab_age_sex": int(L.lk.nunique()), "half_split_csv_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
json.dump(seal, open(os.path.join(HERE, "sealed_half_split_v21.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in seal.items() if k != "cells_NOT_MEASURED_in_one_half"}), "| one-half cells:", len(single))
print(per.assign(b=lambda x: x.units_A - x.units_B).b.abs().describe()[["mean", "max"]].round(2).to_dict())
