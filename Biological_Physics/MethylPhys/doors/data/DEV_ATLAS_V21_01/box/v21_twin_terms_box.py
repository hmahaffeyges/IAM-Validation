#!/usr/bin/env python3
"""Atlas v2.1 twin test and source terms on the box (readiness L6, L7). Committed before it runs. No new method: atlas v2's own
scripts 09_roster.py (twin pairs, split-half null), 10_twin_test_sample_level.py (sample-level twin test) and 12_source_terms.py (Deming
source terms, array-lab offsets, the Loyfer bridge) are reproduced line for line; the ONLY generalization is where a sample's file is found
(a path per row instead of v2's fixed source-to-folder map), so v2.1's new array series can be read.

 0. publish_code first (results/ATLAS_V21/terms/code/).
 1. REPRODUCTION GATE, v2 mode: on v2's own sample table the code must give v2's committed records exactly - twins.csv (every pair: shared,
    separating, r to 4 places, twin), twin_null_splithalf.csv, 10_twin_test_sample_level.csv, and 12b_source_terms_bridged.json (Loyfer a, b
    to 1e-12; array-lab offsets). Any difference stops the job before v2.1 is computed.
 2. v2.1 mode: v2's samples + the 472 v2.1 build arrays (Stage 1 manifest; qc = call rate >= 0.93, v2's rule); each v2.1 GEO/ArrayExpress
    series is its own array source. Cell names: the v2 roster's exact name where one exists (case-insensitive), else the v2.1 name.
    Admitted: v2's admitted cells plus any v2.1 cell with >= 2 qc arrays (v2's entry rule). One correction to v2 script 10, applied in v2.1
    mode only and recorded: a pair 'not testable' is skipped (v2's loop fell through after recording it; no such pair arose in v2).
Writes twins_v21.csv, twin_null_v21.csv, twin_sample_level_v21.csv, source_terms_v21.json, v21_samples_all.csv, reproduction_gate_terms.json.
"""
import os, sys, json, itertools, time, numpy as np, pandas as pd, boto3
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
CODE = ["v21_twin_terms_box.py", "publish_code.py", "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json",
        "twins.csv", "twin_null_splithalf.csv", "10_twin_test_sample_level.csv", "12b_source_terms_bridged.json", "stage1_v21_manifest.csv"]
publish_code("results/ATLAS_V21/terms/code/", CODE, job="atlas v2.1 twin test and source terms", commit=os.environ.get("REPO_COMMIT"))
BUCKET = "methylphys-data-945451304272-us-west-2-an"; OUTP = "results/ATLAS_V21/terms/"; s3 = boto3.client("s3", region_name="us-west-2")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
R = "/home/ubuntu/data/atlas_sources"; TW = json.load(open("twin_family_thresholds_v1.json")); TR, SL, SD = TW["twin_r"], TW["min_separating_loci"], TW["sep_delta"]
B = pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C = pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet"); log("Loyfer loaded", B.shape)
SUBV2 = {"Moss2018": "moss2018/shards", "Salas2018": "blood/GSE110554/shards", "Salas2022": "blood/GSE167998/shards", "GSE63409": "hsc_gse63409/shards"}
def array_path(r):
    if r.get("path"): return r["path"]
    return f"{R}/{SUBV2[r.source]}/{r['sample']}.parquet"
def arr_load(r): return pd.read_parquet(array_path(r)).iloc[:, 0]

# ---------------- 09: profiles, split-half null, twin pairs (verbatim logic)
def run09(ok, admitted_order, entry_ok):
    prof = {}
    for cell, grp in ok.groupby("cell"):
        lw = grp[grp.source == "Loyfer2023"]["sample"].tolist()
        if lw: prof[cell] = ("WGBS", B[lw].where(C[lw] >= 10).mean(axis=1)); continue
        en = grp[grp.source == "ENCODE"]["sample"].tolist()
        if en: prof[cell] = ("WGBS", pd.concat([pd.read_parquet(x).pipe(lambda d: d["beta"].where(d["cov"] >= 10)) for x in en], axis=1).mean(axis=1)); continue
        tw = grp[grp.source == "Tian2023"]["sample"].tolist()
        if tw:
            d = pd.read_parquet(f"{R}/tian2023/{tw[0]}_array.parquet"); prof[cell] = ("WGBS", d["beta"].where(d["cov"] >= 10)); continue
        arr = [arr_load(r) for _, r in grp.iterrows() if r.platform == "array" and os.path.exists(array_path(r))]
        if arr: prof[cell] = ("array", pd.concat(arr, axis=1).mean(axis=1))
    admitted = [c for c in admitted_order if entry_ok(c) and c in prof]
    null = []
    for cell, grp in ok.groupby("cell"):
        lw = grp[grp.source == "Loyfer2023"]["sample"].tolist()
        if len(lw) >= 4:
            h1, h2 = lw[:len(lw) // 2], lw[len(lw) // 2:]; u = B[h1].where(C[h1] >= 10).mean(axis=1); v = B[h2].where(C[h2] >= 10).mean(axis=1)
        else:
            g2 = grp[grp.platform == "array"]
            if len(g2) < 4: continue
            rr = [r for _, r in g2.iterrows()]
            u = pd.concat([arr_load(r) for r in rr[:len(rr) // 2]], axis=1).mean(axis=1); v = pd.concat([arr_load(r) for r in rr[len(rr) // 2:]], axis=1).mean(axis=1)
        j = u.dropna().index.intersection(v.dropna().index)
        null.append(dict(cell=cell, shared=len(j), r=float(np.corrcoef(u[j], v[j])[0, 1]), separating=int((np.abs(u[j] - v[j]) > SD).sum()), frac=float((np.abs(u[j] - v[j]) > SD).mean())))
    NU = pd.DataFrame(null); NULL_FRAC = float(NU.frac.quantile(0.95))
    tw = []
    for a, b in itertools.combinations(admitted, 2):
        (pa, xa), (pb, xb) = prof[a], prof[b]
        j = xa.dropna().index.intersection(xb.dropna().index)
        if len(j) < 1000: continue
        u, v = xa.loc[j].values, xb.loc[j].values
        if pa != pb:
            if pa == "array": s, i = np.polyfit(u, v, 1); u = i + s * u
            else: s, i = np.polyfit(v, u, 1); v = i + s * v
        r = float(np.corrcoef(u, v)[0, 1]); sep = int((np.abs(u - v) > SD).sum()); frac = sep / len(j)
        tw.append(dict(a=a, b=b, platforms=f"{pa}/{pb}", shared=len(j), r=round(r, 4), separating=sep, sep_frac=round(frac, 5),
                       excess_over_null=round(frac / max(NULL_FRAC, 1e-9), 2), twin=(r > TR and frac <= NULL_FRAC * 1.5)))
    return pd.DataFrame(tw).sort_values("r", ascending=False), NU, NULL_FRAC

# ---------------- 10: sample-level twin test (verbatim logic; v2.1 mode skips a not-testable pair)
def run10(S, T, faithful):
    def samples(cell):
        g = S[S.cell == cell]; w = g[g.source == "Loyfer2023"]["sample"].tolist()
        if len(w) >= 2: return "WGBS", B[w].where(C[w] >= 10)
        arr = g[g.platform == "array"]
        if len(arr) >= 2: return "array", pd.concat([arr_load(r).rename(r["sample"]) for _, r in arr.iterrows()], axis=1)
        return None, None
    out = []
    for _, p in T[T.r >= 0.975].iterrows():
        pa, Xa = samples(p.a); pb, Xb = samples(p.b)
        if Xa is None or Xb is None or pa != pb:
            out.append(dict(a=p.a, b=p.b, r=p.r, platform=f"{pa}/{pb}", n_a=None, n_b=None, shared=None, nonoverlap_sep=None, twin=None, note="not testable (need >=2 samples each, same platform)"))
            continue   # v2 skips every untestable pair, in both modes (fixed 2026-10-14: this line was inverted, job 5115bd05)
        j = Xa.dropna(thresh=Xa.shape[1]).index.intersection(Xb.dropna(thresh=Xb.shape[1]).index)
        a = Xa.loc[j].values; b = Xb.loc[j].values
        sep = int(((a.min(1) - b.max(1)) > SD).sum() + ((b.min(1) - a.max(1)) > SD).sum())
        out.append(dict(a=p.a, b=p.b, r=p.r, platform=pa, n_a=Xa.shape[1], n_b=Xb.shape[1], shared=len(j), nonoverlap_sep=sep, twin=(p.r > TW["twin_r"] and sep < SL), note=""))
    return pd.DataFrame(out)

# ---------------- 12: source terms + bridge (verbatim logic)
def run12(S, ARR_ORDER):
    rng = np.random.default_rng(20260928); NL = 50000
    loci = np.sort(rng.choice(B.index[(C >= 10).mean(axis=1) > 0.9].values, NL, replace=False))
    def vec(r):
        if r.source == "Loyfer2023": return B[r["sample"]].where(C[r["sample"]] >= 10).reindex(loci)
        if r.source == "Tian2023": d = pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); return d["beta"].where(d["cov"] >= 10).reindex(loci)
        if r.source == "ENCODE": d = pd.read_parquet(r["sample"]); return d["beta"].where(d["cov"] >= 10).reindex(loci)
        return arr_load(r).reindex(loci)
    M = {}
    for (cell, src), g in S.groupby(["cell", "source"]): M[(cell, src)] = pd.concat([vec(r) for _, r in g.iterrows()], axis=1).mean(axis=1)
    ARR = [s for s in ARR_ORDER if any(k[1] == s for k in M)]; SEQ = sorted({k[1] for k in M} - set(ARR))
    cells = sorted({k[0] for k in M}); arr_mean = {c: pd.concat([M[(c, s)] for s in ARR if (c, s) in M], axis=1).mean(axis=1) for c in cells if any((c, s) in M for s in ARR)}
    def deming(x, y):
        sx, sy = np.var(x), np.var(y); sxy = np.cov(x, y)[0, 1]; b = (sy - sx + np.sqrt((sy - sx) ** 2 + 4 * sxy ** 2)) / (2 * sxy); return float(np.mean(y) - b * np.mean(x)), float(b)
    out = {"scale": "our Stage 1 array scale", "loci": NL, "array_sources": ARR, "sequencing": {}, "array": {}}
    for k in SEQ:
        ov = [c for c in cells if (c, k) in M and c in arr_mean]
        if not ov: out["sequencing"][k] = {"status": "NOT IDENTIFIED - no cell measured on both an array and this source"}; continue
        pairs = {c: pd.DataFrame({"x": arr_mean[c], "y": M[(c, k)]}).dropna() for c in ov}
        X = pd.concat(pairs.values()); a, b = deming(X.x.values, X.y.values); bs = []
        for _ in range(2000):
            pick = rng.choice(ov, len(ov), replace=True); Z = pd.concat([pairs[c] for c in pick]); bs.append(deming(Z.x.values, Z.y.values))
        bs = np.array(bs); res = X.y - (a + b * X.x)
        out["sequencing"][k] = {"status": "IDENTIFIED", "overlap_cells": ov, "n_pairs": int(len(X)), "a": a, "b": b, "a_ci95": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
            "b_ci95": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), "resid_sd": float(res.std()), "r": float(np.corrcoef(X.x, X.y)[0, 1]),
            "per_cell_b": {c: round(deming(p.x.values, p.y.values)[1], 4) for c, p in pairs.items() if len(p) > 100}}
    d = {}
    for k in ARR:
        diffs = []
        for c in cells:
            others = [s for s in ARR if s != k and (c, s) in M]
            if (c, k) in M and others: diffs.append(float((M[(c, k)] - pd.concat([M[(c, s)] for s in others], axis=1).mean(axis=1)).median()))
        d[k] = (float(np.median(diffs)) if diffs else None, len(diffs))
    okd = [v[0] for v in d.values() if v[0] is not None]; cen = float(np.mean(okd)) if okd else 0.0
    out["array"] = {k: ({"d": v[0] - cen, "overlap_cells": v[1]} if v[0] is not None else {"status": "NOT IDENTIFIED - shares no cell with another array lab"}) for k, v in d.items()}
    L = out["sequencing"].get("Loyfer2023", {})
    if L.get("status") == "IDENTIFIED":
        aL, bL = L["a"], L["b"]
        for k in SEQ:
            if k == "Loyfer2023" or out["sequencing"][k]["status"] == "IDENTIFIED": continue
            ov = [c for c in cells if (c, k) in M and (c, "Loyfer2023") in M]
            if not ov: out["sequencing"][k] = {"status": "NOT IDENTIFIED - shares no cell with our arrays or with Loyfer"}; continue
            X = pd.concat([pd.DataFrame({"x": M[(c, "Loyfer2023")], "y": M[(c, k)]}).dropna() for c in ov]); a1, b1 = deming(X.x.values, X.y.values)
            out["sequencing"][k] = {"status": "IDENTIFIED VIA LOYFER", "overlap_cells_with_loyfer": ov, "n_pairs": int(len(X)), "a_vs_loyfer": a1, "b_vs_loyfer": b1,
                "a": a1 + b1 * aL, "b": b1 * bL, "r_vs_loyfer": float(np.corrcoef(X.x, X.y)[0, 1])}
        for k, v in list(out["array"].items()):
            if "d" in v: continue
            ov = [c for c in cells if (c, k) in M and (c, "Loyfer2023") in M]
            if not ov: out["array"][k] = {"status": "NOT IDENTIFIED - shares no cell with another array lab or with Loyfer"}; continue
            diffs = [float((M[(c, k)] - (M[(c, "Loyfer2023")] - aL) / bL).median()) for c in ov]
            out["array"][k] = {"d_via_loyfer": float(np.median(diffs)), "overlap_cells_with_loyfer": ov, "note": "offset of this lab vs Loyfer mapped onto the array scale; not centred with the directly measured labs"}
    return out

# ================= 1. REPRODUCTION GATE (v2 mode)
S2 = pd.read_csv("roster_samples.csv"); RO = pd.read_csv("roster.csv", index_col=0)
ok2 = S2[S2.qc]
T2, NU2, NF2 = run09(ok2, list(RO.index), lambda c: str(RO.loc[c, "entry"]).startswith("ADMIT"))
G = {}
ref = pd.read_csv("twins.csv"); key = lambda D: D.set_index(["a", "b"])[["shared", "separating", "r", "twin"]].sort_index()
G["twins"] = bool(len(ref) == len(T2) and key(ref).equals(key(T2)))
refn = pd.read_csv("twin_null_splithalf.csv")
G["null"] = bool(len(refn) == len(NU2) and (refn.set_index("cell")[["shared", "separating"]].sort_index().equals(NU2.set_index("cell")[["shared", "separating"]].sort_index()))
                 and np.allclose(refn.set_index("cell").r.sort_index(), NU2.set_index("cell").r.sort_index(), atol=1e-12))
ref10 = pd.read_csv("10_twin_test_sample_level.csv"); P10 = run10(ok2, ref, faithful=True)
_n = lambda D: pd.to_numeric(D.nonoverlap_sep, errors="coerce").fillna(-1).astype(int).tolist()
G["twin_sample_level"] = bool(len(ref10) == len(P10) and ref10.a.tolist() == P10.a.tolist() and ref10.b.tolist() == P10.b.tolist() and _n(ref10) == _n(P10))
H = pd.read_csv("hsc_manifest.csv"); H = H[H.subject_status == "normal"]
S12 = pd.concat([S2, pd.DataFrame({"cell": H.label.str.replace(" of normal bone marrow", "", regex=False) + " (bone marrow)", "source": "GSE63409", "platform": "array", "sample": H.gsm, "qc": H.call_rate >= 0.93})])
adm = RO.index[RO.v2_status.astype(str).str.startswith("IN")].tolist(); S12 = S12[S12.qc & S12.cell.isin(adm)].copy()
O2 = run12(S12, list(SUBV2)); ref12 = json.load(open("12b_source_terms_bridged.json"))
LA, LR = O2["sequencing"]["Loyfer2023"], ref12["sequencing"]["Loyfer2023"]
G["source_terms"] = bool(abs(LA["a"] - LR["a"]) < 1e-12 and abs(LA["b"] - LR["b"]) < 1e-12 and
    all(abs(O2["array"][k].get("d", O2["array"][k].get("d_via_loyfer", 0)) - ref12["array"][k].get("d", ref12["array"][k].get("d_via_loyfer", 0))) < 1e-12 for k in ref12["array"] if k in O2["array"]))
json.dump(G, open("reproduction_gate_terms.json", "w"), indent=1); s3.upload_file("reproduction_gate_terms.json", BUCKET, OUTP + "reproduction_gate_terms.json")
log("REPRODUCTION GATE", G)
if not all(G.values()): log("STOPPED: v2 mode does not reproduce v2's records"); sys.exit(2)

# ================= 2. v2.1 mode
M1 = pd.read_csv("stage1_v21_manifest.csv"); M1 = M1[(M1.state != "OUT") & (M1.status == "ok")].copy()
name = {c.lower(): c for c in RO.index}
M1["cellname"] = M1.cell.map(lambda c: name.get(c.lower(), c))
add = pd.DataFrame({"cell": M1.cellname, "source": M1.series, "platform": "array", "sample": M1.base, "qc": pd.to_numeric(M1.call_rate) >= 0.93,
                    "metric": M1.call_rate, "path": [f"{R}/v21/shards/{b}.parquet" for b in M1.base], "v21_state": M1.state})
S21 = pd.concat([S2.assign(path=""), add], ignore_index=True); S21.to_csv("v21_samples_all.csv", index=False)
new_cells = sorted(set(add[add.qc].cell) - set(RO.index))
cnt = add[add.qc].groupby("cell").size()
order = list(RO.index) + new_cells
ent = lambda c: (str(RO.loc[c, "entry"]).startswith("ADMIT") if c in RO.index else int(cnt.get(c, 0)) >= 2)
T21, NU21, NF21 = run09(S21[S21.qc], order, ent); T21.to_csv("twins_v21.csv", index=False); NU21.to_csv("twin_null_v21.csv", index=False)
log("v2.1 twins: pairs", len(T21), "| twin pairs", int(T21.twin.sum()), "| null p95", round(NF21, 5))
P21 = run10(S21[S21.qc], T21, faithful=False); P21.to_csv("twin_sample_level_v21.csv", index=False)
log("v2.1 sample-level: pairs", len(P21), "| twins", int(P21.twin.fillna(False).astype(bool).sum()))
H2 = S12.copy(); H2["path"] = ""
adm21 = adm + [c for c in new_cells if ent(c)]
S12b = pd.concat([H2, add[add.qc & add.cell.isin(adm21)]], ignore_index=True)
O21 = run12(S12b, list(SUBV2) + sorted(set(add.source)))
json.dump(O21, open("source_terms_v21.json", "w"), indent=1)
log("v2.1 source terms: Loyfer", {k: O21["sequencing"].get("Loyfer2023", {}).get(k) for k in ("a", "b")}, "| array labs", {k: (round(v["d"], 5) if "d" in v else v.get("status", "")[:30]) for k, v in O21["array"].items()})
for f in ("twins_v21.csv", "twin_null_v21.csv", "twin_sample_level_v21.csv", "source_terms_v21.json", "v21_samples_all.csv"): s3.upload_file(f, BUCKET, OUTP + f)
open("DONE", "w").write("ok\n")
