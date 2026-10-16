"""Atlas v2.1 U3, blood composition (sealed bar B3, amendment_v21b.json; scorer chain Stage A21 u3_neutrophil). Written 2026-10-16 before any
U3 array is read. The tool, as the sealed rule names it: "the v2.1 composition step built exactly as atlas_e was from v2, on the v2.1 means":
chain dev_comp_methods (DeconvV2, frozen SOLVER), whole-blood cells (DROP_WB out), only cells measured on arrays in v2.1 (final_samples_v21.csv,
platform 'array'), the eight blood groups, then the mixture-identifiability rule (twin_r of twin_family_thresholds_v1.json), the 6,000 neutrophil
identity sites excluded. Truth, fixed here before reading (the project's own earlier decisions):
  GSE127824 (24 cord bloods, flow counts): granulocytes / 100 against the atlas's granulocytes (neutrophils + eosinophils + basophils),
            as decision D6 maps CD15+ granulocytes to neutrophils.
  GSE167202 (COVID-negative emergency-department bloods, clinical differential): (segmented neutrophils + immature granulocytes, only samples whose
            six-part differential sums to 100 +/- 2, the committed count-quality rule of DEV-COMPOSITION-TRUTH-04) / 100 against atlas neutrophils
            (immature granulocytes are neutrophil-lineage cells); any other sample is listed, not scored (rule applied 2026-10-16 before reading:
            the raw table holds impossible sums, up to 145 %).
Bar per set: mean absolute error <= 0.02 and >= 95 % of samples within 0.05; and not worse than the current method (chain Stage A NNLS8) on
neutrophils on any set. Usage: python score_u3_v21.py <atlas_v21_summary.parquet> <shards dir> <out dir> | planted <summary> <out>"""
# Chain copy (2026-10-16) of doors/data/DEV_ATLAS_V21_01/score_u3_v21.py: the logic is unchanged; only where it finds its data folder.
import os, sys, json, gzip, re, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../..")); H = os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01")   # moved into the chain 2026-10-16 (rule 23b-e); data stay in the test folder
CH = HERE; sys.path.insert(0, CH)
import dev_comp_methods as CM, stage_atlas_commission as A21, stage_q2_selfref as Q2
BC = json.load(open(os.path.join(CH, "Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json"))); NS = pd.Index(BC["neutrophil_sites"])
THR = json.load(open(os.path.join(R, "Biological_Physics/MethylPhys/atlas/v2/inputs/twin_family_thresholds_v1.json")))
NEU8 = [g for g in BC["groups"] if g.lower().startswith("neu")][0]; GRAN8 = [g for g in BC["groups"] if g.lower()[:3] in ("neu", "eos", "bas")]
def wide(summary, cells):
    T = pq.read_table(summary, filters=[("cell", "in", list(cells))], columns=["cell", "cpg", "n_obs", "mu", "mu_sd", "mu_lo", "mu_hi", "donor_sd"]).to_pandas()
    ren = dict(mu="mean", mu_sd="sd", mu_lo="ci_lo", mu_hi="ci_hi", donor_sd="donor_sd", n_obs="n_obs"); out = []
    for k, v in ren.items():
        P = T.pivot(index="cpg", columns="cell", values=k); P.columns = [f"{c}_{v}" for c in P.columns]; out.append(P)
    A = pd.concat(out, axis=1); A.index = A.index.astype(str); return A
def build(summary):
    S = pd.read_csv(os.path.join(H, "final_samples_v21.csv"), keep_default_na=False); S = S[S.qc.astype(str) == "True"]
    cells = sorted(json.load(open(os.path.join(H, "stageB_v21_summary.json")))["measured_fraction_by_cell"]); arr = set(S[S.platform == "array"].cell)
    ARRb = [c for c in cells if c not in CM.DROP_WB and c in arr and CM.grp(c) in CM.BLOOD8]
    A = wide(summary, ARRb); D0 = CM.build_deconv(A, ARRb, NS); removed, _ = CM.mixture_twin(D0, ARRb, THR["twin_r"])
    ARRe = [c for c in ARRb if c not in removed]; Dv = CM.build_deconv(A, ARRe, NS)
    return Dv, dict(array_blood_cells=ARRb, mixture_removed=removed, cells=ARRe, solver=Dv.meta), A
def est(Dv, beta):
    g = CM.to_groups(Dv.deconvolve(beta, n_boot=0)["fractions"]); n8 = CM.NNLS8(BC).deconvolve(beta)["fractions"]
    return dict(atlas_NEU=g.get("NEU", 0.0), atlas_GRAN=g.get("GRAN", 0.0), current_NEU=n8[NEU8], current_GRAN=sum(n8[x] for x in GRAN8))
def truth():
    rows = []; t = gzip.open(os.path.join(H, "geo_snapshots/GSE127824.txt.gz"), "rt").read()
    for b in t.split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip(); m = re.search(r"!Sample_characteristics_ch1 = granulocytes: ([0-9.]+)", b)
        rows.append(dict(series="GSE127824", sample=gsm, truth=float(m.group(1)) / 100 if m else np.nan, compare="GRAN"))
    C = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_COMPOSITION_TRUTH_04/gse167202_cbc.csv"))
    for _, c in C.iterrows():
        parts = [c.segmented_neutrophils, c.immature_granulocytes_percent, c.lymphocyte_percent, c.monocyte_percent, c.eosinophils_percent, c.basophils_percent]
        good = all(pd.notna(x) for x in parts) and abs(sum(parts) - 100) <= 2   # the committed count-quality rule (cbc_summary.txt: six parts sum to 100 +/- 2)
        tv = (c.segmented_neutrophils + c.immature_granulocytes_percent) / 100 if good else np.nan
        rows.append(dict(series="GSE167202", sample=c.gsm, truth=tv, compare="NEU"))
    return pd.DataFrame(rows)
def verdict(T):
    out = {}
    for s, g in T.groupby("series"):
        g = g[g.status == "scored"]
        if len(g) < 20: out[s] = dict(verdict="NOT ASSESSED", n=len(g)); continue
        a = A21.u3_neutrophil(g.est_atlas, g.truth); c = A21.u3_neutrophil(g.est_current, g.truth)
        ok = a["met"] and a["mae"] <= c["mae"]; out[s] = dict(verdict="MET" if ok else "NOT MET", n=len(g), atlas=a, current=c, not_worse=bool(a["mae"] <= c["mae"]))
    sets = [v for v in out.values() if v["verdict"] != "NOT ASSESSED"]
    out["U3"] = "COMMISSIONED" if len(sets) >= 2 and all(v["verdict"] == "MET" for v in sets) else ("NOT ASSESSED" if len(sets) < 2 else "FAILED")
    return out
if __name__ == "__main__" and sys.argv[1] == "planted":
    summary, out = sys.argv[2:4]; os.makedirs(out, exist_ok=True); Dv, info, A = build(summary); rng = np.random.default_rng(20261016)
    loci = Dv.loci_all; mu = A.loc[loci, [f"{c}_mean" for c in Dv.cells]].values; rows = []
    for k in range(60):
        f = rng.dirichlet(np.ones(len(Dv.cells)) * 0.5); neu = [i for i, c in enumerate(Dv.cells) if CM.grp(c) == "NEU"]
        f = f * (1 - (w := rng.uniform(0.45, 0.80))) / f.sum(); f[neu] += w / len(neu)
        y = pd.Series(np.clip(mu @ f + rng.normal(0, 0.02, len(loci)), 0, 1), index=loci); e = est(Dv, y)
        tN = float(sum(f[i] for i, c in enumerate(Dv.cells) if CM.grp(c) == "NEU"))
        rows.append(dict(series="PLANTED_OK", sample=f"p{k}", truth=tN, est_atlas=e["atlas_NEU"], est_current=e["atlas_NEU"] + 0.01, status="scored"))
        rows.append(dict(series="PLANTED_BIASED", sample=f"b{k}", truth=tN + 0.06, est_atlas=e["atlas_NEU"], est_current=e["atlas_NEU"] + 0.07, status="scored"))
    T = pd.DataFrame(rows); V = verdict(T)
    want = dict(PLANTED_OK="MET", PLANTED_BIASED="NOT MET"); ok = all(V[k]["verdict"] == w for k, w in want.items())
    L = [f"{k}: {V[k]['verdict']} (want {w}), atlas MAE {V[k]['atlas']['mae']:.4f}, within 0.05 {V[k]['atlas']['within_005']:.3f}" for k, w in want.items()]
    L += [f"cells used: {len(info['cells'])} ({', '.join(info['cells'])}); removed as mixtures: {sorted(info['mixture_removed'])}; markers {info['solver']['n_markers']}", "PLANTED U3 TEST: " + ("PASSED" if ok else "FAILED")]
    open(os.path.join(H, "planted_u3_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0 if ok else 3)
if __name__ == "__main__":
    summary, shards, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); Dv, info, _ = build(summary)
    S = pd.read_csv(os.path.join(H, "atlas_v21_heldout_samples.csv")); S = S[S.role == "truth set"]
    I = pd.read_csv(os.path.join(H, "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")][["sample", "series", "base"]]
    T = S.merge(I, on=["sample", "series"], how="left").merge(truth(), on=["sample", "series"], how="left"); rows = []
    for _, r in T.iterrows():
        d = dict(series=r.series, sample=r["sample"], truth=r.truth, compare=r["compare"])
        if not isinstance(r.base, str) or not os.path.exists(os.path.join(shards, r.base + ".parquet")): rows.append(dict(d, status="listed: no calibrated array")); continue
        if pd.isna(r.truth): rows.append(dict(d, status="listed: no count")); continue
        b = pd.read_parquet(os.path.join(shards, r.base + ".parquet")).iloc[:, 0].astype(float); b.index = b.index.astype(str); e = est(Dv, b)
        rows.append(dict(d, status="scored", est_atlas=e["atlas_" + r["compare"]], est_current=e["current_" + r["compare"]]))
    R_ = pd.DataFrame(rows); V = verdict(R_); V["tool"] = info
    Q2.write_csv(R_, os.path.join(out, "u3_samples_v21.csv"), manifest=["ATL-005"]); Q2.write_json(V, os.path.join(out, "u3_scored_v21.json"), manifest=["ATL-005"])
    for k, v in V.items():
        if isinstance(v, dict) and "verdict" in v: print(k, v["verdict"], "n", v["n"], "| atlas", {x: round(y, 4) for x, y in v.get("atlas", {}).items() if x != "met"}, "| current", {x: round(y, 4) for x, y in v.get("current", {}).items() if x != "met"})
    print("U3:", V["U3"], "| cells", len(info["cells"]), "| removed", sorted(info["mixture_removed"]))
