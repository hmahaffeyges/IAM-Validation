"""Atlas v2.1 U2a, blind sites of the four sealed tissue tests (sealed_commission_v21.json U2; bar B2 of amendment_v21b.json; chain Stage A21
u2_share / u2_verdict). Written 2026-10-16 before any U2 reading. For each tissue: its exact sealed blind sites (sites_methylated +
sites_unmethylated); its panel, copied from its committed rule file (breast null_normals.py, colon colon_synthetic.py, liver liver_synthetic.py,
lung lung_synthetic.py); its anchor and tolerance from the sealed file. A panel cell is assessed when held-out arrays of its atlas cells
(heldout_cell_map_v21.json) number at least 3; its set mean at each blind site (all its held-out arrays pooled) is compared with the v2.1
atlas anchor value; share = fraction of blind sites within the tolerance. A tissue passes when an epithelial panel cell is assessed and >= 95 %
of its assessed panel cells reach 0.90. Panel cells with no held-out set are listed. Usage: python score_u2_v21.py <summary> <shards> <out> | planted <summary> <out>"""
import os, sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../..")); CH = os.path.join(R, "Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH)
import stage_atlas_commission as A21, stage_q2_selfref as Q2
FIB = ["colon fibroblasts", "heart fibroblasts"]; TB = dict(T=["cd4 t cells", "cd8 t cells"], B=["b cells"])
PANEL = {
 "breast": dict(epi=["luminal", "basal"], anchor=["breast luminal epithelium"], p=dict(luminal=["breast luminal epithelium"], basal=["breast basal epithelium"], adipocytes=["adipocytes"], fibroblast=FIB, endothelium=["vascular endothelium"], macrophage=["colon macrophages"], **TB)),
 "colon": dict(epi=["epithelium"], anchor=["colon epithelium"], p=dict(epithelium=["colon epithelium"], fibroblast=["colon fibroblasts"], macrophage=["colon macrophages"], smooth_muscle=["smooth muscle"], endothelium=["vascular endothelium"], neutrophil=["neutrophils"], adipocytes=["adipocytes"], **TB)),
 "liver": dict(epi=["hepatocyte"], anchor=["hepatocyte"], p=dict(hepatocyte=["hepatocyte"], fibroblast=FIB, macrophage=["colon macrophages"], smooth_muscle=["smooth muscle"], endothelium=["vascular endothelium"], neutrophil=["neutrophils"], monocyte=["monocytes"], nk=["nk cells"], adipocytes=["adipocytes"], **TB)),
 "lung": dict(epi=["alveolar", "bronchial"], anchor=["lung alveolar epithelium", "lung bronchus epithelium"], p=dict(alveolar=["lung alveolar epithelium"], bronchial=["lung bronchus epithelium"], fibroblast=FIB, macrophage=["colon macrophages"], smooth_muscle=["smooth muscle"], endothelium=["vascular endothelium"], neutrophil=["neutrophils"], monocyte=["monocytes"], adipocytes=["adipocytes"], **TB))}
SEAL = json.load(open(os.path.join(H, "sealed_commission_v21.json")))["U2_blind_sites"]["tests"]
def sites(t):
    J = json.load(open(os.path.join(R, SEAL[t]["sealed_sites_file"]))); return pd.Index([str(x) for x in J["sites_methylated"] + J["sites_unmethylated"]])
def anchors(summary):
    allsites = sorted(set().union(*[set(sites(t)) for t in PANEL])); cells = sorted({c for v in PANEL.values() for c in v["anchor"]})
    T = pq.read_table(summary, filters=[("cell", "in", cells)], columns=["cell", "cpg", "mu"]).to_pandas(); T = T[T.cpg.astype(str).isin(allsites)]
    P = T.pivot(index="cpg", columns="cell", values="mu"); P.index = P.index.astype(str)
    return {t: P.reindex(sites(t))[v["anchor"]].mean(axis=1) for t, v in PANEL.items()}
def score(AN, arrays, tissues=None):   # arrays: dict atlas cell -> list of pd.Series (beta by site)
    out = {}
    for t, v in PANEL.items():
        if tissues and t not in tissues: continue
        tol = SEAL[t]["tolerance"]; sh, n, listed = {}, {}, []
        for g, cells in v["p"].items():
            A = [a for c in cells for a in arrays.get(c, [])]
            if len(A) < 3: listed.append(g); continue
            m = pd.concat([a.reindex(AN[t].index) for a in A], axis=1).mean(axis=1, skipna=True)
            sh[g] = A21.u2_share(m.values, AN[t].values, tol); n[g] = len(A)
        epi = next((e for e in v["epi"] if e in sh), None); V = A21.u2_verdict(sh, epithelium=epi)
        out[t] = dict(V, shares={k: round(x, 4) for k, x in sh.items()}, arrays=n, listed=listed, tolerance=tol, sites=int(len(AN[t])))
    return out
if __name__ == "__main__" and sys.argv[1] == "planted":
    summary, out = sys.argv[2:4]; AN = anchors(summary); rng = np.random.default_rng(20261016); ok = []
    cells = sorted({c for v in PANEL.values() for g in v["p"].values() for c in g})
    for case, shift, want in (("near", 0.0, "COMMISSIONED"), ("shifted", 0.09, "FAILED")):
        got = {}
        for t, v in PANEL.items():
            arr = {c: [pd.Series(np.clip(AN[t].values + shift + rng.normal(0, 0.012, len(AN[t])), 0, 1), index=AN[t].index) for _ in range(4)] for g in v["p"].values() for c in g}
            got[t] = score(AN, arr, [t])[t]["verdict"]
        ok.append(all(x == want for x in got.values())); print(case, got, "want", want)
    L = [f"planted near anchor: every tissue COMMISSIONED; planted 0.09 shift: every tissue FAILED", "PLANTED U2 TEST: " + ("PASSED" if all(ok) else "FAILED")]
    open(os.path.join(H, "planted_u2_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print(L[-1]); sys.exit(0 if all(ok) else 3)
if __name__ == "__main__":
    summary, shards, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); AN = anchors(summary)
    S = pd.read_csv(os.path.join(H, "atlas_v21_heldout_samples.csv")); S = S[S.role.str.startswith("held-out")]
    I = pd.read_csv(os.path.join(H, "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")][["sample", "series", "base"]]
    cmap = json.load(open(os.path.join(H, "heldout_cell_map_v21.json")))["map"]; S = S.merge(I, on=["sample", "series"], how="left"); S["cell"] = S.cell_type.map(cmap)
    need = {c for v in PANEL.values() for g in v["p"].values() for c in g}; arr = {}
    for _, r in S[S.cell.isin(need) & S.base.notna()].iterrows():
        p = os.path.join(shards, r.base + ".parquet")
        if os.path.exists(p): b = pd.read_parquet(p).iloc[:, 0].astype(float); b.index = b.index.astype(str); arr.setdefault(r.cell, []).append(b)
    V = score(AN, arr); V["arrays_by_cell"] = {k: len(v) for k, v in arr.items()}
    V["U2"] = "COMMISSIONED for " + ", ".join(t for t in PANEL if V[t]["verdict"] == "COMMISSIONED") if any(V[t]["verdict"] == "COMMISSIONED" for t in PANEL) else "NOT COMMISSIONED"
    Q2.write_json(V, os.path.join(out, "u2_scored_v21.json"), manifest=["ATL-005"])
    for t in PANEL: print(t, V[t]["verdict"], V[t].get("share"), V[t]["shares"], "| arrays", V[t]["arrays"], "| listed", V[t]["listed"])
    print("U2:", V["U2"])
