"""DEV-ATLAS-USE-02 (ATL-067): atlas v2.1 head to head against Loyfer's own reference, on the same held-out arrays, jobs and bars as the
Moss head-to-head (DEV-ATLAS-USE-01, met). Written and sealed 2026-10-16 before Loyfer's projected reference is built or read.
use_tests_v21.py is NOT edited: its SHA-256 is checked against DEV_ATLAS_USE_01/sealed_use_v21.json and its functions (markers,
confirmed, mixtures, score_mix) are loaded unchanged, as in the Moss run. Differences from the Moss run, all forced by Loyfer's atlas:
 - Loyfer's 39 groups pool CD4 and CD8 T cells into Blood-T, so both atlases are read with one T-cell profile: v2.1's is the mean of its
   CD4 and CD8 profiles, and the CD4 and CD8 held-out arrays together are the T-cell arrays.
 - Loyfer has oligodendrocytes, so they are shared ground here; the wide area is microglia and astrocytes (cells Loyfer does not have).
 - Loyfer's Blood-Mono+Macro pools monocytes and macrophages; it stands for monocytes.
Bars (the Moss bars, unchanged): UT1 v2.1 confirmed share >= Loyfer's - 0.05; UT2A v2.1 median error <= 1.5 x Loyfer's; UT2B v2.1 median
error <= 0.5 x Loyfer's. Seeds 2026 and 2027. Planted test first.
Usage: python use_tests_loyfer_v21.py <atlas_v21_summary.parquet> <loyfer_array_reference.parquet> <shards> <out> | planted <summary> <out>"""
import os, sys, json, hashlib, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../..")); sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain"))
import stage_q2_selfref as Q2
U1 = os.path.join(H, "..", "DEV_ATLAS_USE_01"); SEAL = json.load(open(os.path.join(U1, "sealed_use_v21.json"))); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SC = os.path.join(U1, "use_tests_v21.py"); assert sha(SC) == SEAL["scorer_sha256"], "sealed scorer changed"
src = open(SC).read(); NS = {"__file__": SC}; exec(src[:src.index('if sys.argv[1] == "synthetic":')], NS)
markers, confirmed, mixtures, score_mix = NS["markers"], NS["confirmed"], NS["mixtures"], NS["score_mix"]
MAP = {"neutrophils": "Blood-Granul", "monocytes": "Blood-Mono+Macro", "b cells": "Blood-B", "t cells": "Blood-T", "nk cells": "Blood-NK",
       "colon epithelium": "Colon-Ep", "cortical neurons": "Neuron", "oligodendrocytes": "Oligodend"}
SHARED = list(MAP); NS["SHARED"] = SHARED; OUTSIDE = ["microglia", "astrocytes"]
LAB = {"neutrophils": "neutrophils", "monocytes": "monocytes", "b cells": "b cells", "cd4 t cells": "t cells", "cd8 t cells": "t cells", "nk cells": "nk cells",
       "colon epithelium": "colon epithelium", "cortical neurons": "cortical neurons", "oligodendrocytes": "oligodendrocytes", "microglia": "microglia",
       "astrocytes [triple-negative glia]": "astrocytes"}
EXCL = {"GSE132547": "D13 NOT MET (2026-10-16)"}
union = lambda mk: sorted(set().union(*[set(m.index) for m in mk.values()]))
def v21_ref(summary):
    cells = [c for c in SHARED if c != "t cells"] + ["cd4 t cells", "cd8 t cells"] + OUTSIDE
    T = pq.read_table(summary, filters=[("cell", "in", cells)], columns=["cell", "cpg", "mu"]).to_pandas()
    v = T.pivot(index="cpg", columns="cell", values="mu"); v.index = v.index.astype(str)
    v["t cells"] = v[["cd4 t cells", "cd8 t cells"]].mean(axis=1); return v[SHARED + OUTSIDE]
def loyfer_ref(path):
    L = pd.read_parquet(path); L.index = L.index.astype(str); lo = L[[MAP[c] for c in SHARED]]; lo.columns = SHARED; return lo
def load_arrays(shards):
    S = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv"))
    I = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")]
    arrays, listed = {}, []
    for lab, v in SEAL["heldout_samples"].items():
        cell = LAB.get(lab)
        if cell is None: listed.append([lab, "not used: " + EXCL.get(v["series"][0], "not in this test")]); continue
        s = S[S["sample"].isin(v["samples"]) & ~S.series.isin(list(EXCL))].merge(I[["sample", "series", "base"]], on=["sample", "series"], how="left")
        for _, r in s.iterrows():
            p = os.path.join(shards, f"{r.base}.parquet") if isinstance(r.base, str) else None
            if not p or not os.path.exists(p): listed.append([r["sample"], "no calibrated array"]); continue
            b = pd.read_parquet(p).iloc[:, 0].astype("float32"); b.index = b.index.astype(str); arrays.setdefault(cell, []).append(b)
    return arrays, listed
def run(v21, opp, arrays):
    sites = v21.dropna().index.intersection(opp.dropna().index)
    for L in arrays.values():
        for b in L: sites = sites.intersection(b.dropna().index)
    sites = pd.Index(sorted(sites)); v21 = v21.loc[sites]; opp = opp.loc[sites]
    A = {c: [b.reindex(sites) for b in L] for c, L in arrays.items()}
    held = pd.DataFrame({c: pd.concat(A[c], axis=1).mean(axis=1) for c in SHARED})
    cv = confirmed(markers(v21, SHARED), held); co = confirmed(markers(opp, SHARED), held)
    p_v, p_o = float(np.nanmean(list(cv.values()))), float(np.nanmean(list(co.values())))
    siv = union(markers(v21, SHARED)); sio = union(markers(opp, SHARED))
    MA = mixtures(A, SHARED, 200, np.random.default_rng(2026))
    ev, _ = score_mix([(t, x[siv]) for t, x in MA], v21.loc[siv], SHARED); eo, _ = score_mix([(t, x[sio]) for t, x in MA], opp.loc[sio], SHARED)
    ALL = SHARED + OUTSIDE; sia = union(markers(v21, ALL))
    MB = mixtures(A, SHARED, 200, np.random.default_rng(2027), outside=OUTSIDE)
    eb, ob = score_mix([(t, x[sia]) for t, x in MB], v21.loc[sia], ALL); eob, _ = score_mix([(t, x[sio]) for t, x in MB], opp.loc[sio], SHARED)
    rec = float(np.mean([abs(a - b) <= 0.05 for a, b in ob]))
    V = dict(sites=len(sites), arrays={c: len(L) for c, L in A.items()},
             UT1=dict(v21_share=p_v, loyfer_share=p_o, per_cell_v21=cv, per_cell_loyfer=co, verdict="MET" if p_v >= p_o - 0.05 else "NOT MET"),
             UT2A=dict(median_error_v21=float(np.median(ev)), median_error_loyfer=float(np.median(eo)), ratio=float(np.median(ev) / np.median(eo)),
                       verdict="MET" if np.median(ev) <= 1.5 * np.median(eo) else "NOT MET"),
             UT2B=dict(median_error_v21=float(np.median(eb)), median_error_loyfer=float(np.median(eob)), recovered_within_005=rec, outside_cells=OUTSIDE,
                       verdict="MET" if np.median(eb) <= 0.5 * np.median(eob) else "NOT MET"))
    V["lines"] = [f"sites {len(sites)} | arrays {V['arrays']}",
        f"UT1 markers confirmed: v2.1 {p_v:.3f}, Loyfer {p_o:.3f} -> {V['UT1']['verdict']}",
        f"UT2A shared-ground mixtures: median error v2.1 {np.median(ev):.4f}, Loyfer {np.median(eo):.4f}, ratio {V['UT2A']['ratio']:.2f} -> {V['UT2A']['verdict']}",
        f"UT2B wide area ({', '.join(OUTSIDE)}): median error v2.1 {np.median(eb):.4f}, Loyfer {np.median(eob):.4f}; v2.1 recovers the outside share within 0.05 in {rec:.2f} -> {V['UT2B']['verdict']}"]
    return V
if __name__ == "__main__" and sys.argv[1] == "planted":
    summary, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(5)
    v21 = v21_ref(summary).dropna(); v21 = v21.iloc[rng.choice(len(v21), 60000, replace=False)]
    opp = (v21[SHARED] + rng.normal(0, 0.02, (len(v21), len(SHARED)))).clip(0, 1)
    fake = lambda prof: {c: [pd.Series((prof[c] + rng.normal(0, 0.04, len(prof))).clip(0, 1).values, index=prof.index) for _ in range(6)] for c in SHARED + OUTSIDE}
    arr = fake(v21); Vt = run(v21, opp, arr); bad = v21.copy(); bad[OUTSIDE] = rng.permutation(bad[OUTSIDE].values); Vb = run(bad, opp, arr)
    ok = Vt["UT1"]["verdict"] == Vt["UT2A"]["verdict"] == Vt["UT2B"]["verdict"] == "MET" and Vb["UT2B"]["verdict"] == "NOT MET"
    L = ["true case: " + " | ".join(Vt["lines"][1:]), "scrambled outside profiles: " + Vb["lines"][3], "PLANTED USE TEST: " + ("PASSED" if ok else "FAILED")]
    Q2.write_json(dict(true=Vt, scrambled=Vb, passed=ok), os.path.join(out, "use_tests_loyfer_planted.json"), manifest=["ATL-067"])
    open(os.path.join(out, "use_tests_loyfer_planted.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0 if ok else 3)
elif __name__ == "__main__":
    summary, lref, shards, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    SL = json.load(open(os.path.join(H, "sealed_use_loyfer_v21.json"))); assert sha(os.path.abspath(__file__)) == SL["runner_sha256"], "runner changed after sealing"
    v21 = v21_ref(summary); opp = loyfer_ref(lref); arrays, listed = load_arrays(shards); V = run(v21, opp, arrays)
    V.update(test="DEV-ATLAS-USE-02", loyfer_reference_sha256=sha(lref), listed=listed[:50], n_listed=len(listed), excluded_sets=EXCL)
    Q2.write_json(V, os.path.join(out, "use_tests_loyfer_result.json"), manifest=["ATL-067"])
    open(os.path.join(out, "head_to_head_loyfer_summary.txt"), "w").write("\n".join(V["lines"]) + "\n"); print("\n".join(V["lines"]))
