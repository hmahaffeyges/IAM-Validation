#!/usr/bin/env python3
"""dev_stages.py - DEVELOPMENT - not commissioned. Chain v3 development round 2 (2026-10-04): the stages that sit behind development flags
(doors/DEV_FLAGS_01.md). Nothing here changes a reading, a gauge or a tare; each function returns a record that run_sample.py writes under
bundle["development"][<stage>] with the label DEV_LABEL. Physics only: every line is set on the array's own noise or on same-run references;
nothing is fitted to other arrays.

  selftare_ii(beta, specimen)        DEV-SELFTARE-02   affine map of each probe design onto the reference arrays' scale from this array's own
                                                       low and high fixed-site anchors; Met-A re-read on the mapped betas
  site_uncertainty(beta)             DEV-TOOLKIT-ADDED-02  per-design, per-state SD of beta at this array's own fixed sites (v3 per-site uncertainty)
  direction(beta, ref, sites)        DEV-DIRECTION-02  D = mean(|r - 0.5| - |beta - 0.5|) at the identity sites; > 0 toward disorder
  trace_cell(beta)                   DEV-TOOLKIT-ADDED-02 3b  contamination of an isolated-neutrophil specimen by another blood group
  foreign_cell(beta, template)       DEV-TOOLKIT-ADDED-02 3c  a non-blood template in whole blood, line = 3 x this array's own standard error
  brightness(beta, specimen, comp)   DEV-TOOLKIT-ADDED-02 11b  95 % interval on Met-A from the per-site uncertainty
  sky(beta, fractions, atlas)        DEV-SKY-02        residual sky, within-chromosome block-shuffle null, look-elsewhere by simulation
  nilc_e / atlas_e(beta, atlas)      DEV-NILC-01 / DEV-ATLAS-EPIC-02 methods, as frozen
  percell_b(beta, fractions, atlas)  DEV-PERCELL-01    B-cell Met-A on the development floor
Runtime files (development, not frozen): Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json, dev_foreign_placenta_EPIC_v1.json."""
import json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RM = os.path.join(HERE, "Runtime Matrices"); DEVDIR = os.path.join(RM, "Development")
DEV_LABEL = "DEVELOPMENT - not commissioned"
BLOOD8 = ["NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"]
PARENTS = {"NEU": "neutrophils", "EOS": "eosinophils", "BASO": "basophils", "MONO": "monocytes", "B": "b cells", "NK": "nk cells",
           "CD4T": "cd4 t cells", "CD8T": "cd8 t cells"}
SKY_RUN = 50            # sites per run in the within-chromosome block shuffle (DEV-SKY-02)
SKY_NULL_N = 20         # shuffles for the band-power null
SKY_LEE_N = 100         # shuffles for the look-elsewhere statistic
_C = {}


def _H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))


def _json(name):
    if name not in _C:
        p = os.path.join(DEVDIR, name) if not os.path.isabs(name) else name
        _C[name] = json.load(open(p)) if os.path.exists(p) else None
    return _C[name]


def _bc():
    if "bc" not in _C: _C["bc"] = json.load(open(os.path.join(RM, "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    return _C["bc"]


def _rec(stage, **kw):
    return {"stage": stage, "label": DEV_LABEL, **kw}


# ---------------------------------------------------------------- self-tare on type II fixed sites (DEV-SELFTARE-02)
def anchors(beta, st=None):
    """This array's anchors: mean beta over each fixed-site set (I_low, I_high, II_low, II_high) it measures, with the count."""
    st = st or _json("dev_selftare_typeII_EPIC_v1.json")
    out = {}
    for k, sites in st["sets"].items():
        x = beta.reindex(sites).dropna(); out[k] = {"mean": float(x.mean()) if len(x) else None, "n": int(len(x))}
    return out


def selftare_map(beta, st=None):
    """beta' = Lr + (beta - L)(Ur - Lr)/(U - L) per design at the sites whose design is recorded; other sites unchanged. Returns (beta', record)."""
    st = st or _json("dev_selftare_typeII_EPIC_v1.json")
    if st is None: return None, {"reason": "dev_selftare_typeII_EPIC_v1.json not found"}
    an = anchors(beta, st); ra = st["ref_anchors"]; des = pd.Series(st["design"]); b2 = beta.copy(); maps = {}
    for d in ("I", "II"):
        L, U = an[f"{d}_low"]["mean"], an[f"{d}_high"]["mean"]; Lr, Ur = ra[f"{d}_low"], ra[f"{d}_high"]
        if L is None or U is None or U - L <= 0.1: maps[d] = None; continue
        s = des.index[des.values == d].intersection(b2.dropna().index)
        b2.loc[s] = (Lr + (b2.loc[s] - L) * (Ur - Lr) / (U - L)).clip(1e-6, 1 - 1e-6)
        maps[d] = {"L": round(L, 5), "U": round(U, 5), "L_ref": round(Lr, 5), "U_ref": round(Ur, 5), "slope": round((Ur - Lr) / (U - L), 5), "n_sites_mapped": int(len(s))}
    return b2, {"anchors": an, "maps": maps}


def selftare_ii(beta, specimen="whole blood"):
    """Met-A re-read after the type II self-tare: each design mapped onto the reference arrays' scale by this array's own fixed-site anchors (DEV-SELFTARE-02)."""
    import conductor_v3 as C3
    b2, info = selftare_map(beta)
    if b2 is None: return _rec("selftare_ii", status="NOT_RUN", **info)
    if specimen.lower() in C3.ISOLATED: m, _ = C3.stage_m_isolated(b2)
    else: m, _ = C3.stage_m_blood(b2, C3.stage_a_composition(b2))
    return _rec("selftare_ii", status="OK", A_selftared=m.get("A"), reason=m.get("reason"), **info,
                note="Met-A re-read on this array's betas mapped onto the reference arrays' scale by its own fixed-site anchors; the reading above is unchanged")


# ---------------------------------------------------------------- v3 per-site uncertainty (DEV-TOOLKIT-ADDED-02)
def site_uncertainty(beta, st=None):
    """SD of (beta - frozen reference value) over this array's own fixed sites, per design and state. Returns {I_low: sd, ...}."""
    st = st or _json("dev_selftare_typeII_EPIC_v1.json")
    if st is None: return None
    rv = pd.Series(st["ref_value"]); out = {}
    for k, sites in st["sets"].items():
        x = (beta.reindex(sites) - rv.reindex(sites)).dropna(); out[k] = float(x.std(ddof=1)) if len(x) > 20 else None
    return out


def s_for(sites, ref_values, su, st=None):
    """Per-site s_i for `sites` given their reference values (design from the runtime file; low state if ref < 0.5)."""
    st = st or _json("dev_selftare_typeII_EPIC_v1.json"); des = pd.Series(st["design"]).reindex(sites).fillna("II")
    rv = pd.Series(ref_values, index=sites)
    k = des.astype(str) + np.where(rv.values < 0.5, "_low", "_high")
    return pd.Series([su.get(x) or np.nan for x in k], index=sites).fillna(np.nanmax([v for v in su.values() if v] or [0.02]))


# ---------------------------------------------------------------- directional decomposition (DEV-DIRECTION-02)
def direction(beta, ref, sites=None):
    """D = mean over sites of |r - 0.5| - |beta - 0.5| (> 0: toward 0.5, toward disorder). ref: Series of reference values."""
    r = pd.Series(ref, dtype="float64"); s = r.index if sites is None else pd.Index(sites)
    x = beta.reindex(s); ok = x.notna() & r.reindex(s).notna()
    d = (r.reindex(s)[ok] - 0.5).abs() - (x[ok] - 0.5).abs()
    return {"D": float(d.mean()) if ok.sum() else None, "n_sites": int(ok.sum())}


def direction_record(beta, specimen, comp, refs_D=None):
    """Signed move toward or away from 0.5 at the neutrophil identity sites against the cell's own reference, tared on same-run references (DEV-DIRECTION-02)."""
    import conductor_v3 as C3
    B = _bc(); S = pd.Index(B["neutrophil_sites"]); P = {g: pd.Series(v, index=S, dtype="float64") for g, v in B["profiles_at_neutrophil_sites"].items()}
    if specimen.lower() in C3.ISOLATED: r = P["NEU"]
    else:
        fr = (comp or {}).get("fractions")
        if not fr: return _rec("direction", status="NOT_RUN", reason="no composition")
        r = sum(v * P[g] for g, v in fr.items() if g in P)
    o = direction(beta, r, S); o.update(reference="purified neutrophil profile" if specimen.lower() in C3.ISOLATED else "composition-matched expectation")
    if refs_D and len([x for x in refs_D if x is not None]) >= 3 and o["D"] is not None:
        rd = np.array([x for x in refs_D if x is not None], float); o["D_rel"] = o["D"] - float(np.median(rd)); o["ref_spread"] = float(np.std(rd, ddof=1))
        o["direction"] = ("toward disorder" if o["D_rel"] > 2 * o["ref_spread"] else "toward over-order" if o["D_rel"] < -2 * o["ref_spread"] else "no direction")
    else: o["direction"] = "untared: needs >= 3 same-run references with D"
    return _rec("direction", status="OK", **o)


# ---------------------------------------------------------------- 3b trace cell in an isolated-neutrophil specimen
def trace_cell(beta, su=None, line=3.0):
    """3b: another blood group in an isolated-neutrophil specimen, called above 3 x this array's own standard error (DEV-TOOLKIT-ADDED-02)."""
    B = _bc(); M = pd.DataFrame(B["mu_markers"], index=B["markers"])[B["groups"]]
    y = beta.reindex(M.index); ok = y.notna(); y, M = y[ok], M[ok]
    su = su or site_uncertainty(beta)
    if su is None: return _rec("trace_cell", status="NOT_RUN", reason="dev_selftare_typeII_EPIC_v1.json (per-site uncertainty) not found")
    s = s_for(M.index, M["NEU"].values, su); w = 1.0 / s.values ** 2
    r = (y - M["NEU"]).values; out = {}
    for g in B["groups"]:
        if g == "NEU": continue
        d = (M[g] - M["NEU"]).values; den = float((w * d * d).sum()); f = float((w * d * r).sum() / den)
        res = r - f * d; sig2 = float((w * res * res).sum() / max(len(r) - 1, 1)); se = float(np.sqrt(sig2 / den))
        out[g] = {"f": round(f, 5), "se": round(se, 5), "z": round(f / se, 2) if se > 0 else None, "called": bool(se > 0 and f > line * se)}
    called = [g for g, v in out.items() if v["called"]]
    return _rec("trace_cell", status="OK", line=f"f > {line} x this array's own standard error", candidates=out, called=called, n_markers=int(ok.sum()))


# ---------------------------------------------------------------- 3c foreign cell in whole blood
def foreign_cell(beta, template=None, su=None, line=3.0):
    """3c: a non-blood template in whole blood by NNLS with the blood groups, called above 3 x this array's own standard error (DEV-TOOLKIT-ADDED-02)."""
    from scipy.optimize import nnls
    T = template or _json("dev_foreign_placenta_EPIC_v1.json")
    if T is None: return _rec("foreign_cell", status="NOT_RUN", reason="no foreign template file")
    B = _bc(); sites = pd.Index(T["sites"]); prof = pd.DataFrame(T["blood_profiles"], index=sites)[B["groups"]]
    X = prof.copy(); X["FOREIGN"] = T["template"]; y = beta.reindex(sites); ok = y.notna() & X.notna().all(1)
    X, y = X[ok], y[ok]; su = su or site_uncertainty(beta)
    if su is None: return _rec("foreign_cell", status="NOT_RUN", reason="dev_selftare_typeII_EPIC_v1.json (per-site uncertainty) not found")
    s = s_for(X.index, X["FOREIGN"].values, su); w = np.sqrt(1.0 / s.values ** 2)
    f, _ = nnls(X.values * w[:, None], y.values * w); res = (y.values - X.values @ f) * w
    act = f > 0; sig2 = float((res ** 2).sum() / max(len(y) - act.sum(), 1)); Xw = X.values[:, act] * w[:, None]
    try: cov = sig2 * np.linalg.inv(Xw.T @ Xw); se_act = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError: se_act = np.full(act.sum(), np.nan)
    se = np.full(len(f), np.nan); se[act] = se_act; k = list(X.columns).index("FOREIGN")
    if not act[k]:   # at the boundary: standard error of the template's coefficient with it entered
        Xk = X.values * w[:, None]; m = act.copy(); m[k] = True
        try: se[k] = float(np.sqrt(sig2 * np.linalg.inv(Xk[:, m].T @ Xk[:, m])[list(np.where(m)[0]).index(k), list(np.where(m)[0]).index(k)]))
        except np.linalg.LinAlgError: pass
    ff, sf = float(f[k]), float(se[k])
    return _rec("foreign_cell", status="OK", template=T.get("name", "placenta"), f=round(ff, 5), se=round(sf, 5),
                called=bool(np.isfinite(sf) and sf > 0 and ff > line * sf), line=f"f > {line} x this array's own standard error", n_sites=int(ok.sum()))


# ---------------------------------------------------------------- 11b surface brightness: interval on Met-A
def brightness(beta, specimen, comp, n_draw=200, su=None, seed=20261004):
    """11b: 95 % interval on Met-A from the v3 per-site uncertainty (this array's own fixed-site noise), by Monte Carlo (DEV-TOOLKIT-ADDED-02)."""
    import conductor_v3 as C3, stage_m_met_a as SM
    su = su or site_uncertainty(beta); rng = np.random.default_rng(seed)
    if su is None: return _rec("brightness", status="NOT_RUN", reason="dev_selftare_typeII_EPIC_v1.json (per-site uncertainty) not found")
    B = _bc(); S = pd.Index(B["neutrophil_sites"]); x = beta.reindex(S); ok = x.notna(); xs = x[ok]
    P = {g: pd.Series(v, index=S, dtype="float64") for g, v in B["profiles_at_neutrophil_sites"].items()}
    if specimen.lower() in C3.ISOLATED: den = SM._floors()["platforms"]["EPIC"]["neutrophils"]["floor"]; ref = P["NEU"][ok]
    else:
        fr = (comp or {}).get("fractions")
        if not fr: return _rec("brightness", status="NOT_RUN", reason="no composition")
        e = sum(v * P[g] for g, v in fr.items() if g in P); ok = ok & e.notna(); xs = x[ok]; e = e[ok]; den = float(_H(e.values).mean()); ref = e
    s = s_for(xs.index, ref.values, su).values; A = []
    for _ in range(n_draw): A.append(float(_H(np.clip(xs.values + rng.normal(0, s), 1e-6, 1 - 1e-6)).mean() / den))
    A0 = float(_H(xs.values).mean() / den); lo, hi = np.percentile(A, [2.5, 97.5])
    return _rec("brightness", status="OK", A=round(A0, 4), interval_95=[round(float(lo), 4), round(float(hi), 4)], half_width=round(float(hi - lo) / 2, 4),
                per_site_sd=su, note="site noise from this array's own fixed sites, independent per site; array-wide offsets are not in this interval")


# ---------------------------------------------------------------- 11 / 12 sky with the block-shuffle null and look-elsewhere
def _mapping():
    if "map" not in _C:
        z = np.load(os.path.join(RM, "Patient_CMB", "iamatlas_cpg_to_healpix_nside128.npz"))
        m = pd.DataFrame({"pix": z["pixel"], "chr": z["chr_key"]}, index=z["cpg_id"].astype(str)); m = m[(m.pix >= 0) & ~m.index.duplicated()]
        _C["map"] = m.sort_values(["chr", "pix"], kind="mergesort")
    return _C["map"]


def residual_z(beta, fractions, AP):
    """z_i = (beta_i - sum_g f_g mu_g,i) / sqrt(sum_g f_g^2 (sd^2 + donor_sd^2) + 0.02^2) (DEV-SKY-01 wrapper). AP: atlas parent columns."""
    idx = beta.dropna().index.intersection(AP.index)
    E = sum(fractions[g] * AP[f"{c}_mean"].reindex(idx) for g, c in PARENTS.items())
    V = sum(fractions[g] ** 2 * (AP[f"{c}_sd"].reindex(idx) ** 2 + AP[f"{c}_donor_sd"].reindex(idx) ** 2) for g, c in PARENTS.items()) + 0.02 ** 2
    return ((beta.reindex(idx) - E) / np.sqrt(V)).dropna()


def _pixmap(zs, pix, npix):
    s = np.bincount(pix, weights=zs, minlength=npix); n = np.bincount(pix, minlength=npix); out = np.full(npix, np.nan); out[n > 0] = s[n > 0] / n[n > 0]
    return out


def block_shuffle_order(chr_keys, run=SKY_RUN, rng=None):
    """Permutation of positions (already in genomic order) that moves whole runs of `run` consecutive sites within each chromosome."""
    rng = rng or np.random.default_rng(); chr_keys = np.asarray(chr_keys); out = np.empty(len(chr_keys), dtype=np.int64)
    for c in np.unique(chr_keys):
        pos = np.where(chr_keys == c)[0]; runs = [pos[i:i + run] for i in range(0, len(pos), run)]
        out[pos] = np.concatenate([runs[j] for j in rng.permutation(len(runs))])
    return out


def sky_from_z(z, n_null=SKY_NULL_N, n_lee=SKY_LEE_N, seed=20261004, free_null=False):
    import sky_statistics as SS
    m = _mapping(); zz = z.reindex(m.index).dropna(); mm = m.loc[zz.index]; NPIX = 12 * 128 * 128
    pix = mm.pix.values.astype(np.int64); vals = zz.values; rng = np.random.default_rng(seed)
    sky = _pixmap(vals, pix, NPIX); cl, f_sky, good = SS.masked_spectrum(sky); bp = SS.bandpowers(cl)
    def shuf():
        o = block_shuffle_order(mm.chr.values, rng=rng); return SS.bandpowers(SS.masked_spectrum(_pixmap(vals[o], pix, NPIX))[0])
    nb = np.array([shuf() for _ in range(max(n_null, n_lee))]); mu = nb[:n_null].mean(0)
    T = float(np.max(bp / mu)); Tn = np.max(nb[:n_lee] / mu, axis=1); p = float((np.sum(Tn >= T) + 1) / (len(Tn) + 1))
    rec = {"f_sky": round(f_sky, 4), "n_pix": int(good.sum()), "n_sites": int(len(zz)), "bandpowers": bp.tolist(), "bands": SS.BANDS,
           "ratio_to_block_null": (bp / mu).round(4).tolist(), "lee_T": round(T, 4), "lee_p": round(p, 4), "structure_beyond_null": bool(p < 0.05),
           "null": f"within-chromosome block shuffle, runs of {SKY_RUN} sites, {n_null} shuffles for the band null, {n_lee} for look-elsewhere"}
    if free_null: rec["ratio_to_free_null"] = (bp / SS.shuffled_null(sky, n_perm=n_null, rng=rng).mean(0)).round(4).tolist()
    return rec


def _has_cols(path, cols):
    try:
        import pyarrow.parquet as pq; names = set(pq.read_schema(path).names); return all(c in names for c in cols)
    except Exception: return False


def load_atlas_parents(atlas_parquet):
    if "AP" not in _C:
        cols = [f"{c}_{s}" for c in PARENTS.values() for s in ("mean", "sd", "donor_sd")]
        try: A = pd.read_parquet(atlas_parquet, columns=cols + ["cpg_id"])
        except Exception: A = pd.read_parquet(atlas_parquet, columns=cols) if _has_cols(atlas_parquet, cols) else pd.read_parquet(atlas_parquet)
        if "cpg_id" in A.columns: A = A.set_index("cpg_id")
        A.index = A.index.astype(str); _C["AP"] = A[cols].copy()
    return _C["AP"]


def sky(beta, fractions, atlas_parquet):
    """Residual sky with the within-chromosome block-shuffle null and the look-elsewhere statistic by simulation (DEV-SKY-02)."""
    try: import healpy  # noqa: F401
    except ImportError: return _rec("sky", status="NOT_RUN", reason="healpy is not installed in this environment")
    if not atlas_parquet or not os.path.exists(atlas_parquet): return _rec("sky", status="NOT_RUN", reason="--atlas-v2 parquet not given or not found")
    if not fractions: return _rec("sky", status="NOT_RUN", reason="no composition")
    return _rec("sky", status="OK", **sky_from_z(residual_z(beta, fractions, load_atlas_parents(atlas_parquet))))


# ---------------------------------------------------------------- stage 4 NILC-e, stage 3 atlas_e, stage 5 B cells (round-1 methods, unchanged)
ATLAS_E_CELLS = ["basophils", "effector memory cd8 t cells", "eosinophils", "memory b cells", "memory cd4 t cells", "monocytes",
                 "naive b cells", "naive cd4 t cells", "naive cd8 t cells", "neutrophils", "nk cells", "regulatory t cells"]


def _atlas_e(atlas_parquet):
    if "AE" not in _C:
        import dev_comp_methods as CM
        A = CM.load_atlas(atlas_parquet); NS = pd.Index(_bc()["neutrophil_sites"])
        D = CM.build_deconv(A, ATLAS_E_CELLS, NS); Ab = CM.subset_atlas(A, ATLAS_E_CELLS); Ab = Ab.loc[Ab.index.difference(NS)]
        mu = Ab[[f"{c}_mean" for c in ATLAS_E_CELLS]]; v = Ab[[f"{c}_sd" for c in ATLAS_E_CELLS]].values ** 2 + Ab[[f"{c}_donor_sd" for c in ATLAS_E_CELLS]].values ** 2
        ok = mu.notna().all(1).values & np.isfinite(v).all(1); rg = (mu.max(1) - mu.min(1)).values; sel = ok & (rg >= 0.2)
        _C["AE"] = (D, CM.NILC(mu.index[sel], ATLAS_E_CELLS, mu.values[sel], v[sel], cov="atlas"), CM)
    return _C["AE"]


def atlas_e(beta, atlas_parquet):
    """Stage 3 atlas_e composition (12 array-measured circulating atlas v2 cells, DEV-ATLAS-EPIC-02 method), 8-group sums."""
    if not atlas_parquet or not os.path.exists(atlas_parquet): return _rec("atlas_e", status="NOT_RUN", reason="--atlas-v2 parquet not given or not found")
    D, _, CM = _atlas_e(atlas_parquet); a = D.deconvolve(beta, n_boot=0); g = CM.to_groups(a["fractions"])
    return _rec("atlas_e", status="OK", fractions={k: round(float(g.get(k, 0.0)), 5) for k in BLOOD8}, cell_fractions={k: round(float(v), 5) for k, v in a["fractions"].items()})


def nilc_e(beta, atlas_parquet):
    """Stage 4 NILC-e composition on the atlas_e templates (DEV-NILC-01 method), 8-group sums and the noise per fraction."""
    if not atlas_parquet or not os.path.exists(atlas_parquet): return _rec("nilc_e", status="NOT_RUN", reason="--atlas-v2 parquet not given or not found")
    _, N, CM = _atlas_e(atlas_parquet); n = N.deconvolve(beta); g = CM.to_groups(n["fractions"])
    return _rec("nilc_e", status="OK", fractions={k: round(float(g.get(k, 0.0)), 5) for k in BLOOD8}, sum=round(n["sum"], 4), noise_sd={k: round(v, 5) for k, v in n["noise_sd"].items()})


def percell_b(beta, fractions, atlas_parquet):
    """Stage 5 B-cell Met-A against the composition-matched expectation on the development B-cell floor (DEV-PERCELL-01 method)."""
    FD = json.load(open(os.path.join(RM, "Met_A_Floors", "metA_floors_v1_2_ALLCELLS_development.json")))["platforms"]["EPIC"]["b cells"]
    sites = pd.Index(FD["sites"])
    if not fractions: return _rec("percell_b", status="NOT_RUN", reason="no composition (isolated specimen)")
    if not atlas_parquet or not os.path.exists(atlas_parquet): return _rec("percell_b", status="NOT_RUN", reason="--atlas-v2 parquet not given or not found")
    AP = load_atlas_parents(atlas_parquet); e = sum(fractions[g] * AP[f"{c}_mean"].reindex(sites) for g, c in PARENTS.items())
    x = beta.reindex(sites); ok = x.notna() & e.notna()
    if x.notna().sum() < 0.9 * len(sites): return _rec("percell_b", status="NOT_RUN", reason=f"only {int(x.notna().sum())} of {len(sites)} B-cell sites measured on the array")
    return _rec("percell_b", status="OK", A_untared=round(float(_H(x[ok].values).mean() / _H(e[ok].values).mean()), 4), fraction_B=round(fractions.get("B", 0.0), 4),
                n_sites=int(ok.sum()), n_sites_measured=int(x.notna().sum()), n_sites_with_expectation=int(e.notna().sum()),
                floor="metA_floors_v1_2_ALLCELLS_development.json (EPIC b cells, development)",
                note="read against the composition-matched expectation (atlas v2 parent means x stage 2 fractions); Stage M's read line (fraction >= 0.20) is not applied here")


# ---------------------------------------------------------------- EPIC v2 through SeSAMe (DEV-EPIC-V2-01)
SESAME_R = os.path.join(HERE, "dev_epicv2_sesame.R")


def epicv2_calibrate(grn, red, rscript, prep="QCDPB"):
    """EPIC v2 IDAT pair -> beta Series on EPIC v1 probe names (replicate v2 probes of one CpG averaged). SeSAMe openSesame(prep), pOOBAH-masked
    probes dropped. Returns (beta, meta)."""
    import subprocess, tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="cpg_v2_")
    try:
        for src, ch in ((grn, "Grn"), (red, "Red")):
            dst = os.path.join(tmp, f"S_{ch}.idat" + (".gz" if str(src).endswith(".gz") else "")); shutil.copyfile(src, dst)
        out = os.path.join(tmp, "betas.csv")
        r = subprocess.run([rscript, SESAME_R, os.path.join(tmp, "S"), out, prep], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not os.path.exists(out): raise RuntimeError("sesame failed: " + (r.stderr or r.stdout)[-400:])
        d = pd.read_csv(out); d = d.dropna(subset=["beta"])
        d["cpg"] = d["probe"].astype(str).str.replace(r"_(?:TC|BC|TO|BO)\d{2}$", "", regex=True)
        n_v2 = int(len(d)); b = d.groupby("cpg")["beta"].mean().astype("float64"); b.index = b.index.astype(str)
        b = b[b.index.str.startswith("cg")]; used = open(out + ".prep").read().strip() if os.path.exists(out + ".prep") else prep
        return b, {"calibrator": f"SeSAMe: {used}", "n_v2_probes_detected": n_v2, "n_cpg_v1_names": int(len(b)),
                   "collapse": "replicate EPIC v2 probes of one CpG averaged into the EPIC v1 name"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
