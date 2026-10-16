"""DEV-ATLAS-USE-01 real run (ATL-060), chain runner written 2026-10-16 before any use-test value is computed. The sealed scorer
use_tests_v21.py is NOT edited: its SHA-256 and the Moss reference's are checked against sealed_use_v21.json, and its functions (markers,
confirmed, deconv, mixtures, score_mix) are loaded unchanged; only the loading of real data is added here. Data choices, fixed before reading:
 - references: v2.1 means (atlas_v21_summary.parquet, mu) for the 8 shared cells and the outside cells; Moss's columns renamed to v2.1 names
   by shared_cells_v21_to_moss. Sites: probes present in both references and measured (not missing) in every held-out array used (the seal).
 - held-out arrays: the sealed sample lists, each array's calibrated file from the identity record (status ok). Carried checks: GSE132547 is
   out (D13 NOT MET, 2026-10-16), so lung alveolar macrophages leave the outside set; GSE131989 is in (D14 MET).
 - UT1: held-out means per shared cell over all its arrays.
 - UT2A and UT2B: no one platform holds every cell's held-out arrays (neutrophils and the outside cells are EPIC only; B, CD4 and CD8 T
   cells and colon epithelium are 450K only; found on run 1, which stopped before any score), so mixtures combine both platforms on the
   common probe set the seal names. Seeds 2026 (UT2A) and 2027 (UT2B). Disclosed as a deviation from "one platform" in the docstring.
 - subtype rollup: v2.1 is used with the shared and outside cells only, so it returns no subtypes and the rollup has nothing to sum.
Planted test first (chain rule): the same loader and scoring on arrays and a "Moss" built from the v2.1 means plus noise; the true case must
pass UT1, UT2A and UT2B, and a v2.1 whose outside-cell profiles are scrambled must fail UT2B.
Usage: python use_tests_real_v21.py <atlas_v21_summary.parquet> <shards dir> <out dir> | planted <atlas_v21_summary.parquet> <out dir>"""
import os, sys, json, hashlib, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../..")); sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain"))
import stage_q2_selfref as Q2
SEAL = json.load(open(os.path.join(H, "sealed_use_v21.json"))); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SC = os.path.join(H, "use_tests_v21.py"); MO = os.path.join(H, "moss_reference_atlas.csv")
assert sha(SC) == SEAL["scorer_sha256"], "sealed scorer changed"; assert sha(MO) == SEAL["moss_reference_sha256"], "Moss reference changed"
src = open(SC).read(); NS = {"__file__": SC}; exec(src[:src.index('if sys.argv[1] == "synthetic":')], NS)
markers, confirmed, mixtures, score_mix = NS["markers"], NS["confirmed"], NS["mixtures"], NS["score_mix"]
MAP = SEAL["shared_cells_v21_to_moss"]; SHARED = list(MAP); NS["SHARED"] = SHARED
OUTSIDE_LAB = {k: v for k, v in SEAL["outside_cells_heldout_to_v21"].items() if v != "lung alveolar macrophages"}; OUTSIDE = list(OUTSIDE_LAB.values())
EXCL = {"GSE132547": "D13 NOT MET (2026-10-16)"}
union = lambda mk: sorted(set().union(*[set(m.index) for m in mk.values()]))
def refs(summary):
    M = pd.read_csv(MO, index_col=0); M.index = M.index.astype(str)
    M = M[~M.index.duplicated()]   # Moss lists a site once per cell it marks: 7,890 rows, 6,105 sites, every repeat identical (checked 2026-10-16; run 2 stopped on it before any score)
    moss = M[[MAP[c] for c in SHARED]]; moss.columns = SHARED
    T = pq.read_table(summary, filters=[("cell", "in", SHARED + OUTSIDE)], columns=["cell", "cpg", "mu"]).to_pandas()
    v21 = T.pivot(index="cpg", columns="cell", values="mu"); v21.index = v21.index.astype(str); return v21[SHARED + OUTSIDE], moss
def load_arrays(shards):
    S = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/atlas_v21_heldout_samples.csv"))
    I = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")]
    lab2cell = dict({c: c for c in SHARED}, **OUTSIDE_LAB); arrays, plat, listed = {}, {}, []
    for lab, v in SEAL["heldout_samples"].items():
        cell = lab2cell.get(lab)
        if cell is None: listed.append([lab, "not used: " + EXCL.get(v["series"][0], "outside cell dropped")]); continue
        s = S[S["sample"].isin(v["samples"]) & ~S.series.isin(list(EXCL))].merge(I[["sample", "series", "base"]], on=["sample", "series"], how="left")
        for _, r in s.iterrows():
            p = os.path.join(shards, f"{r.base}.parquet") if isinstance(r.base, str) else None
            if not p or not os.path.exists(p): listed.append([r["sample"], "no calibrated array"]); continue
            b = pd.read_parquet(p).iloc[:, 0].astype("float32"); b.index = b.index.astype(str)
            arrays.setdefault(cell, []).append(b); plat.setdefault(cell, []).append("EPIC" if len(b) > 600000 else "450K")
    return arrays, plat, listed
def run(v21, moss, arrays, plat):
    sites = v21.dropna().index.intersection(moss.dropna().index)
    for L in arrays.values():
        for b in L: sites = sites.intersection(b.dropna().index)
    sites = pd.Index(sorted(sites)); v21 = v21.loc[sites]; moss = moss.loc[sites]
    A = {c: [b.reindex(sites) for b in L] for c, L in arrays.items()}
    held = pd.DataFrame({c: pd.concat(A[c], axis=1).mean(axis=1) for c in SHARED})
    cv = confirmed(markers(v21, SHARED), held); cm = confirmed(markers(moss, SHARED), held)
    p_v, p_m = float(np.nanmean(list(cv.values()))), float(np.nanmean(list(cm.values())))
    siv = union(markers(v21, SHARED)); sim = union(markers(moss, SHARED))
    MA = mixtures(A, SHARED, 200, np.random.default_rng(2026))
    ev, _ = score_mix([(t, x[siv]) for t, x in MA], v21.loc[siv], SHARED); em, _ = score_mix([(t, x[sim]) for t, x in MA], moss.loc[sim], SHARED)
    ALL = SHARED + OUTSIDE; sia = union(markers(v21, ALL))
    MB = mixtures(A, SHARED, 200, np.random.default_rng(2027), outside=OUTSIDE)
    eb, ob = score_mix([(t, x[sia]) for t, x in MB], v21.loc[sia], ALL); emb, _ = score_mix([(t, x[sim]) for t, x in MB], moss.loc[sim], SHARED)
    rec = float(np.mean([abs(a - b) <= 0.05 for a, b in ob]))
    V = dict(sites=int(len(sites)), arrays={c: len(v) for c, v in A.items()}, platforms={c: pd.Series(v).value_counts().to_dict() for c, v in plat.items()},
             UT1=dict(v21_share=p_v, moss_share=p_m, per_cell_v21=cv, per_cell_moss=cm, verdict="MET" if p_v >= p_m - 0.05 else "NOT MET"),
             UT2A=dict(median_error_v21=float(np.median(ev)), median_error_moss=float(np.median(em)), ratio=float(np.median(ev) / np.median(em)),
                       verdict="MET" if np.median(ev) <= 1.5 * np.median(em) else "NOT MET"),
             UT2B=dict(median_error_v21=float(np.median(eb)), median_error_moss=float(np.median(emb)), recovered_within_005=rec, outside_cells=OUTSIDE,
                       verdict="MET" if np.median(eb) <= 0.5 * np.median(emb) else "NOT MET"))
    V["lines"] = [f"sites {V['sites']} | arrays {V['arrays']}",
        f"UT1 markers confirmed: v2.1 {p_v:.3f}, Moss {p_m:.3f} -> {V['UT1']['verdict']}",
        f"UT2A shared-ground mixtures: median error v2.1 {np.median(ev):.4f}, Moss {np.median(em):.4f}, ratio {V['UT2A']['ratio']:.2f} -> {V['UT2A']['verdict']}",
        f"UT2B wide area ({', '.join(OUTSIDE)}): median error v2.1 {np.median(eb):.4f}, Moss {np.median(emb):.4f}; v2.1 recovers the outside share within 0.05 in {rec:.2f} -> {V['UT2B']['verdict']}"]
    return V
if __name__ == "__main__" and sys.argv[1] == "planted":
    summary, out = sys.argv[2:4]; os.makedirs(out, exist_ok=True); v21, _ = refs(summary); v21 = v21.dropna()
    rng = np.random.default_rng(16); v21 = v21.iloc[np.sort(rng.choice(len(v21), min(60000, len(v21)), replace=False))]
    moss = (v21[SHARED] + rng.normal(0, 0.02, (len(v21), len(SHARED)))).clip(0, 1)
    def fake(prof):
        arr, plat = {}, {}
        for c in SHARED + OUTSIDE:
            arr[c] = [pd.Series((prof[c] + rng.normal(0, 0.04, len(prof))).clip(0, 1).values, index=prof.index) for _ in range(6)]
            plat[c] = ["450K"] * 6 if c in SHARED else ["EPIC"] * 6
        return arr, plat
    arr, plat = fake(v21); Vt = run(v21, moss, arr, plat)
    bad = v21.copy(); bad[OUTSIDE] = rng.permutation(bad[OUTSIDE].values)      # v2.1 with scrambled outside-cell profiles
    Vb = run(bad, moss, arr, plat)
    ok = Vt["UT1"]["verdict"] == Vt["UT2A"]["verdict"] == Vt["UT2B"]["verdict"] == "MET" and Vb["UT2B"]["verdict"] == "NOT MET"
    L = ["true case: " + " | ".join(Vt["lines"][1:]), "scrambled outside profiles: " + Vb["lines"][3], "PLANTED USE TEST: " + ("PASSED" if ok else "FAILED")]
    Q2.write_json(dict(true=Vt, scrambled=Vb, passed=ok), os.path.join(H, "use_tests_real_planted.json"), manifest=["ATL-060"])
    open(os.path.join(H, "use_tests_real_planted.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0 if ok else 3)
if __name__ == "__main__":
    summary, shards, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); v21, moss = refs(summary); arrays, plat, listed = load_arrays(shards)
    V = run(v21, moss, arrays, plat); V.update(test="DEV-ATLAS-USE-01", listed=listed[:50], n_listed=len(listed), excluded_sets=EXCL,
        deviations=["UT2A and UT2B mixtures combine 450K and EPIC arrays on the common probe set (no one platform holds all their cells)", "lung alveolar macrophages not used (D13 NOT MET)"])
    Q2.write_json(V, os.path.join(out, "use_tests_v21_result.json"), manifest=["ATL-060"])
    open(os.path.join(out, "head_to_head_v21_summary.txt"), "w").write("\n".join(V["lines"]) + "\n"); print("\n".join(V["lines"]))
