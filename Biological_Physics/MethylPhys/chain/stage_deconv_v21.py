#!/usr/bin/env python3
"""Stage D21 - the deconvolution tool on IAM-Atlas v2.1 (release track B; DEVELOPMENT, not commissioned). Written 2026-10-16.
Job: given a mixed sample's array readings, report each cell's share with its error bar.

 reference   the atlas's best value per site and cell, and the spread a NEW person's cell has around it, as the fit itself defines it
             (version 2, 2026-10-16): sd^2 = (uncertainty of the average, (hi - lo)/3.92)^2 + the fit's donor spread (donor_sd)^2 + the
             measured laboratory spread (Stage A21 lab_sd)^2 + FLOOR, exactly the array predictive of Stage A21 used in U1.
             Version 1 (scored once, DEV-DECONV-V21-01) used only the average's range; it is kept as SPREAD="mean_only".
 markers     for each cell, the sites that separate it most from its nearest other cell in the panel, in both directions:
             z = |mu_c - mu_nearest| / sqrt(sd_c^2 + sd_nearest^2); top `per_cell` per direction, every panel cell measured there.
 solver      weighted non-negative least squares on the markers; each site's weight is 1 / (sum_k f_k^2 sd_k^2 + lab_sd^2), refined over
             three passes from equal shares (the atlas's own range plus a laboratory term, which the U1 diagnostic found the range lacks).
             Shares are reported as estimated and normalized to sum 1; the raw sum is reported.
 error bars  95 % interval of each share from `boot` refits, each on resampled marker sites AND on cell profiles redrawn from the
             atlas's spread (a new person's cells are not the average cell), version 2; version 1 resampled sites only.
Usage: import stage_deconv_v21 as D; R = D.reference(cells); M = D.markers(R); D.deconvolve(beta, R, M)
       python stage_deconv_v21.py planted [out] [six]   (six: the BLOOD6 panel; default the 12-cell BLOOD panel)     (synthetic test on planted mixtures built from the atlas itself)"""
import os, sys, json, numpy as np, pandas as pd
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__)); ATLAS = os.path.join(HERE, "..", "atlas", "v2.1")
LAB_SD = 0.02; PER_CELL = 50; BOOT = 200; SEED = 2026; SPREAD = os.environ.get("DECONV_SPREAD", "full")   # "full" (v2) or "mean_only" (v1, as scored once)
def reference(cells, atlas=ATLAS, spread=None):
    spread = spread or SPREAD; import pyarrow.parquet as pq
    """mu and sd DataFrames (sites x cells) for the named cells, from the compact copy; reads only those columns."""
    sites = pd.read_parquet(os.path.join(atlas, "sites.parquet")).cpg.astype(str).values; out = {}
    for k in ("mu", "mu_lo", "mu_hi"):
        parts = []
        for n in (1, 2, 3):
            p = os.path.join(atlas, f"{k}_part{n}.parquet")
            import pyarrow.parquet as pq
            cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c in cells]
            if cols: parts.append(pd.read_parquet(p, columns=cols))
        X = pd.concat(parts, axis=1)[list(cells)].astype("float32"); X = X.where(X != 65535) / 10000.0; X.index = sites; out[k] = X
    miss = [c for c in cells if c not in out["mu"].columns]; assert not miss, f"cells not in the atlas: {miss}"
    se = (out["mu_hi"] - out["mu_lo"]) / 3.92
    if spread == "mean_only": return dict(mu=out["mu"], sd=se, cells=list(cells), spread=spread)
    import stage_atlas_commission as A21
    ds = []
    for n in (1, 2, 3):
        p = os.path.join(atlas, f"donor_sd_part{n}.parquet"); cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c in cells]
        if cols: ds.append(pd.read_parquet(p, columns=cols))
    D = pd.concat(ds, axis=1)[list(cells)].astype("float32"); D = D.where(D != 65535) / 10000.0; D.index = out["mu"].index
    lab = A21.lab_sd()
    return dict(mu=out["mu"], sd=np.sqrt(se ** 2 + D ** 2 + lab ** 2 + A21.FLOOR), cells=list(cells), spread=spread, lab_sd=lab, floor=A21.FLOOR)
def markers(R, per_cell=PER_CELL):
    mu, sd = R["mu"], R["sd"]; ok = mu.notna().all(1) & sd.notna().all(1); mu, sd = mu[ok], sd[ok]; pick = set()
    for c in R["cells"]:
        others = [o for o in R["cells"] if o != c]
        d = mu[others].sub(mu[c], axis=0); s = np.sqrt(sd[others].pow(2).add(sd[c].pow(2), axis=0))
        hi = (-d / s).min(1); lo = (d / s).min(1)   # c above every other cell / below every other cell, by its nearest one
        pick |= set(hi.nlargest(per_cell).index) | set(lo.nlargest(per_cell).index)
    return sorted(pick)
def _solve(y, A, S, lab_sd, passes=3):
    K = A.shape[1]; f = np.full(K, 1.0 / K)
    for _ in range(passes):
        w = 1.0 / np.sqrt((S ** 2 * f[None, :] ** 2).sum(1) + lab_sd ** 2); f, _ = nnls(A * w[:, None], y * w)
    return f
def deconvolve(beta, R, M, lab_sd=LAB_SD, boot=BOOT, seed=SEED):
    """beta: pd.Series of readings by site id. Returns shares (normalized), raw sum, 95 % intervals, sites used."""
    y = pd.Series(beta).reindex(M).astype(float); m = y.notna().values; sites = np.array(M)[m]
    A = R["mu"].loc[sites].values.astype(float); S = R["sd"].loc[sites].values.astype(float); yy = y.values[m]
    full = R.get("spread", "mean_only") == "full"; lab_w = 0.0 if full else lab_sd   # v2: the laboratory term is already inside S
    f = _solve(yy, A, S, lab_w); tot = f.sum(); rng = np.random.default_rng(seed); B = [f / tot if tot > 0 else f]
    for _ in range(boot):
        i = rng.integers(0, len(yy), len(yy)); Ai = A[i]
        if full: Ai = np.clip(Ai + rng.normal(0, 1, Ai.shape) * S[i], 0, 1)   # this person's cells, drawn from the atlas's spread
        g = _solve(yy[i], Ai, S[i], lab_w); B.append(g / g.sum() if g.sum() > 0 else g)
    B = np.array(B); lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)   # boot=0: the point estimate only
    sh = f / tot if tot > 0 else f
    return dict(shares={c: float(v) for c, v in zip(R["cells"], sh)}, raw_sum=float(tot),
                interval_95={c: [float(a), float(b)] for c, a, b in zip(R["cells"], lo, hi)}, sites_used=int(m.sum()), lab_sd=R.get("lab_sd", lab_sd), spread=R.get("spread", "mean_only"), boot=boot)
# the blood panel: separate subsets only; the atlas's unsorted parent cells (b cells, cd4 t cells, cd8 t cells) contain their subsets and would
# compete with them in a mixture (U3's mixture rule removed them for the same reason)
BLOOD = ["neutrophils", "eosinophils", "basophils", "monocytes", "naive b cells", "memory b cells", "naive cd4 t cells", "memory cd4 t cells",
         "regulatory t cells", "naive cd8 t cells", "effector memory cd8 t cells", "nk cells"]
# the six-cell blood panel the held-out independent laboratories hold (DEV-DECONV-V21-01 is sealed on it)
BLOOD6 = ["neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells"]
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "planted":
    sys.path.insert(0, HERE); import stage_q2_selfref as Q2
    out = sys.argv[2] if len(sys.argv) > 2 else HERE
    import pyarrow.parquet as pq
    have = set(n for k in (1, 2, 3) for n in pq.ParquetFile(os.path.join(ATLAS, f"mu_part{k}.parquet")).schema_arrow.names)
    panel = BLOOD6 if (len(sys.argv) > 3 and sys.argv[3] == "six") else BLOOD
    cells = [c for c in panel if c in have]; R = reference(cells); M = markers(R); rng = np.random.default_rng(7)
    # truth is planted with the fit's OWN person-to-person spread (donor_sd) and a laboratory shift, whatever spread the tool assumes, so
    # a tool that treats the atlas average as exact is caught (version 1's flaw, 2026-10-16)
    RF = reference(cells, spread="full"); mu = RF["mu"].loc[M].values.astype(float)
    dsd = np.sqrt(np.clip(RF["sd"].loc[M].values.astype(float) ** 2 - RF["lab_sd"] ** 2 - RF["floor"], 0, None)); rows = []
    def sample(f, lab):   # a planted array: each cell's DNA drawn from the atlas's donor spread, mixed, plus one laboratory shift and array noise
        x = np.clip(mu + rng.normal(0, 1, mu.shape) * dsd, 0, 1) @ f + lab + rng.normal(0, 0.01, len(M)); return pd.Series(np.clip(x, 0, 1), index=M)
    T = {}
    pure = [deconvolve(sample(np.eye(len(cells))[k], 0.0), R, M, boot=50)["shares"][c] for k, c in enumerate(cells)]
    T["T1_pure"] = dict(min_share=float(min(pure)), met=bool(min(pure) >= 0.80))
    err = []; cov = []
    for j in range(40):
        f = rng.dirichlet(np.ones(len(cells))); r = deconvolve(sample(f, rng.normal(0, RF["lab_sd"])), R, M)
        e = np.array([r["shares"][c] for c in cells]) - f; err.append(np.abs(e).mean())
        cov.append(np.mean([r["interval_95"][c][0] <= f[k] <= r["interval_95"][c][1] for k, c in enumerate(cells)]))
    T["T2_mixtures"] = dict(n=40, spread=R["spread"], mae=float(np.mean(err)), interval_coverage=float(np.mean(cov)), met=bool(np.mean(err) <= 0.03 and np.mean(cov) >= 0.80))
    # T3 a planted 5 % share of one cell is seen: its interval excludes 0 in most mixtures
    seen = []
    for j in range(20):
        f = rng.dirichlet(np.ones(len(cells))); k = j % len(cells); f = f * 0.95 / (f.sum() - f[k] + 1e-12); f[k] = 0.05; f = f / f.sum()
        r = deconvolve(sample(f, 0.0), R, M); seen.append(r["interval_95"][cells[k]][0] > 0.0)
    T["T3_small_share"] = dict(n=20, seen=float(np.mean(seen)), met=bool(np.mean(seen) >= 0.70))
    V = dict(stage="D21 deconvolution (development)", cells=cells, markers=len(M), tests=T, verdict="PASSED" if all(t["met"] for t in T.values()) else "FAILED")
    Q2.write_json(V, os.path.join(out, "planted_deconv_v21.json"), manifest=["ATL-005"])
    L = [f"{k}: {v}" for k, v in T.items()] + [f"PLANTED DECONVOLUTION TEST: {V['verdict']}"]
    open(os.path.join(out, "planted_deconv_v21.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---------------------------------------------------------------- the sealed real test (DEV-DECONV-V21-01), run once
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "score":
    sys.path.insert(0, HERE); import stage_q2_selfref as Q2
    shards, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    DD = os.path.join(HERE, "..", "doors", "data"); SEAL = json.load(open(os.path.join(DD, "DEV_DECONV_V21_01", "sealed_deconv_v21.json")))
    cells = SEAL["panel"]; R = reference(cells); M = markers(R)
    I = pd.read_csv(os.path.join(DD, "DEV_ATLAS_V21_01", "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")]
    base = dict(zip(I["sample"], I["base"])); A = {}
    for c in cells:
        A[c] = []
        for smp in SEAL["samples"][c]:
            b = base.get(smp); p = os.path.join(shards, f"{b}.parquet") if isinstance(b, str) else None
            if p and os.path.exists(p): x = pd.read_parquet(p).iloc[:, 0].astype(float); x.index = x.index.astype(str); A[c].append(x.reindex(M))
    rng = np.random.default_rng(2026); rows = []
    for j in range(200):
        f = rng.dirichlet(np.ones(len(cells))); comp = [A[c][rng.integers(len(A[c]))] for c in cells]
        mix = sum(fk * x for fk, x in zip(f, comp)); r = deconvolve(mix, R, M)
        for k, c in enumerate(cells):
            lo, hi = r["interval_95"][c]; rows.append(dict(mixture=j, cell=c, truth=float(f[k]), share=r["shares"][c], lo=lo, hi=hi, inside=bool(lo <= f[k] <= hi), sites=r["sites_used"]))
    T = pd.DataFrame(rows); mae = float(T.assign(e=(T.share - T.truth).abs()).groupby("mixture").e.mean().mean()); cov = float(T.inside.mean())
    V = dict(test=SEAL["test"], arrays={c: len(v) for c, v in A.items()}, markers=len(M), B1_mean_error=mae, B1_met=bool(mae <= 0.03), B2_inside=cov, B2_met=bool(cov >= 0.80),
             per_cell_error={c: float((g.share - g.truth).abs().mean()) for c, g in T.groupby("cell")}, per_cell_inside={c: float(g.inside.mean()) for c, g in T.groupby("cell")})
    V["verdict"] = "COMMISSIONED" if V["B1_met"] and V["B2_met"] else "NOT MET"
    Q2.write_csv(T, os.path.join(out, "deconv_mixtures_v21.csv"), manifest=["ATL-005"]); Q2.write_json(V, os.path.join(out, "deconv_scored_v21.json"), manifest=["ATL-005"])
    print(json.dumps(V))

# ---------------------------------------------------------------- DIAGNOSTIC ONLY (2026-10-16): no bar, nothing rescored
# Each pure held-out array of the six cells (the DEV-DECONV-V21-01 lists) read alone, by two panels: the six parent profiles, and the
# atlas's subsets summed back into the six families. Shows where a cell's share goes when it is read as itself.
FAMILY = {"neutrophils": ["neutrophils"], "monocytes": ["monocytes"], "b cells": ["naive b cells", "memory b cells"],
          "cd4 t cells": ["naive cd4 t cells", "memory cd4 t cells", "regulatory t cells", "cytotoxic cd4 t cells (cd28-)"],
          "cd8 t cells": ["naive cd8 t cells", "cd8 central memory t cells", "effector memory cd8 t cells", "cd8 temra t cells"],
          "nk cells": ["nk cells", "cd56bright nk cells"]}
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "diag_pure":
    sys.path.insert(0, HERE); import stage_q2_selfref as Q2
    shards, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    DD = os.path.join(HERE, "..", "doors", "data"); SEAL = json.load(open(os.path.join(DD, "DEV_DECONV_V21_01", "sealed_deconv_v21.json")))
    I = pd.read_csv(os.path.join(DD, "DEV_ATLAS_V21_01", "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")]; base = dict(zip(I["sample"], I["base"]))
    panels = {"six parents": {c: [c] for c in SEAL["panel"]}, "subsets summed": FAMILY}; rows = []
    for pn, fam in panels.items():
        cells = [x for v in fam.values() for x in v]; R = reference(cells); M = markers(R)
        for c in SEAL["panel"]:
            for smp in SEAL["samples"][c]:
                b = base.get(smp); p = os.path.join(shards, f"{b}.parquet") if isinstance(b, str) else None
                if not (p and os.path.exists(p)): continue
                x = pd.read_parquet(p).iloc[:, 0].astype(float); x.index = x.index.astype(str); r = deconvolve(x, R, M, boot=0)
                rows.append(dict(panel=pn, cell=c, sample=smp, **{f: sum(r["shares"][k] for k in v) for f, v in fam.items()}))
    T = pd.DataFrame(rows); S = T.groupby(["panel", "cell"])[list(FAMILY)].mean().round(3)
    Q2.write_csv(T, os.path.join(out, "diag_pure_v21.csv"), manifest=["ATL-065"]); print(S.to_string())

# ---------------------------------------------------------------- DEV-DECONV-V21-02 (sealed 2026-10-16): version 2 on fresh mixtures
# Sample check, first (the instrument knows when it cannot read a sample): every pure array is read alone by the six-cell panel; an array
# passes when it reads at least SELF_MIN of its own cell. Failing arrays are listed and not used. A cell with fewer than 3 passing arrays is
# left out of the mixtures (listed). Then 200 mixtures (seed 2027) of the passing arrays of one laboratory, E-MTAB-2145, scored once.
SELF_MIN = 0.80
def sample_check(x, cell, R, M):
    r = deconvolve(x, R, M, boot=0); return dict(own=r["shares"][cell], passes=bool(r["shares"][cell] >= SELF_MIN))
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "score_fresh":
    sys.path.insert(0, HERE); import stage_q2_selfref as Q2
    shards, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    DD = os.path.join(HERE, "..", "doors", "data"); SEAL = json.load(open(os.path.join(DD, "DEV_DECONV_V21_01", "sealed_deconv_v21_02.json")))
    assert SPREAD == "full", "version 2 is sealed"; cells = SEAL["panel"]; R = reference(cells); M = markers(R)
    I = pd.read_csv(os.path.join(DD, "DEV_ATLAS_V21_01", "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")]; base = dict(zip(I["sample"], I["base"]))
    A, chk = {}, []
    for c, smps in SEAL["samples"].items():
        A[c] = []
        for smp in smps:
            b = base.get(smp); p = os.path.join(shards, f"{b}.parquet") if isinstance(b, str) else None
            if not (p and os.path.exists(p)): chk.append(dict(cell=c, sample=smp, own=None, passes=False, note="no calibrated array")); continue
            x = pd.read_parquet(p).iloc[:, 0].astype(float); x.index = x.index.astype(str); k = sample_check(x, c, R, M); chk.append(dict(cell=c, sample=smp, **k))
            if k["passes"]: A[c].append(x.reindex(M))
    use = [c for c in SEAL["samples"] if len(A[c]) >= 3]; rng = np.random.default_rng(2027); rows = []
    for j in range(200):
        f = rng.dirichlet(np.ones(len(use))); mix = sum(fk * A[c][rng.integers(len(A[c]))] for fk, c in zip(f, use)); r = deconvolve(mix, R, M)
        truth = {c: 0.0 for c in cells}; truth.update(dict(zip(use, f)))
        for c in cells:
            lo, hi = r["interval_95"][c]; rows.append(dict(mixture=j, cell=c, truth=float(truth[c]), share=r["shares"][c], lo=lo, hi=hi, inside=bool(lo <= truth[c] <= hi)))
    T = pd.DataFrame(rows); mae = float(T.assign(e=(T.share - T.truth).abs()).groupby("mixture").e.mean().mean()); cov = float(T.inside.mean())
    V = dict(test=SEAL["test"], sample_check=chk, cells_mixed=use, markers=len(M), B1_mean_error=mae, B1_met=bool(mae <= 0.03), B2_inside=cov, B2_met=bool(cov >= 0.80),
             per_cell_error={c: float((g.share - g.truth).abs().mean()) for c, g in T.groupby("cell")}, per_cell_inside={c: float(g.inside.mean()) for c, g in T.groupby("cell")})
    V["verdict"] = "COMMISSIONED" if V["B1_met"] and V["B2_met"] else "NOT MET"
    Q2.write_csv(T, os.path.join(out, "deconv_v2_mixtures.csv"), manifest=["ATL-065"]); Q2.write_json(V, os.path.join(out, "deconv_v2_scored.json"), manifest=["ATL-065"])
    print(json.dumps({k: V[k] for k in ("cells_mixed", "B1_mean_error", "B2_inside", "verdict")}))
