"""stage_metaa_atlas.py - Met-A read against IAM-Atlas v2.1, for any atlas cell (DEVELOPMENT, DEV-METAA-ATLAS-01, written 2026-10-21).

NOT COMMISSIONED. Every record this module returns carries flag = "read against IAM-Atlas v2.1; not commissioned by our standards". The
commissioned Met-A (stage_m_met_a.py, conductor_v3.py) reads neutrophils against purified arrays of the same cell and is not changed by
this module. Purified arrays exist for few cell types; IAM-Atlas v2.1 holds a healthy reference with its uncertainty for all 79.

The instrument's job: a healthy sample of a cell, read against that cell in IAM-Atlas v2.1 and tared against healthy samples of the same
cell run the same way, reads Normal; a planted loss of the cell's pattern moves it by the shift the reading predicts.

  identity_sites(cell)   Met-A's own site rule (metA_floors_v1_2: healthy spread <= 0.05; mean 0.75-0.95 or 0.05-0.25; at most 3,000 per
                         side), applied to the atlas: spread = the atlas's healthy width (person-to-person spread, donor_sd, with the atlas's
                         own uncertainty, half its 95 % range over 1.96), mean = the atlas's best value. Where more than 3,000 sites qualify on
                         a side, the 3,000 of smallest healthy width are kept (DEV_METAA_ATLAS_01.md, design addition 1; fixed before reading).
  read(beta, cell)       A = mean H(beta) over the identity sites the sample measured / mean H(atlas best value) over the same sites, with
                         the shift a 1 % loss of the cell's pattern would make, and the detection-limit inputs. The atlas blends 21 sources
                         and sits slightly softer than one array (DEV-REFERENCES-01), so this A is not Normal-calibrated on its own: it is
                         printed "untared" until tare() reads it.
  tare(A, refs)          A_rel = A / median(A of at least 3 healthy samples of the same cell run the same way), the chain's same-run tare
                         (conductor_v3.stage_t_tare's rule); it cancels any reference offset, whatever its size or sign. Never centered on the
                         sample itself (that hides a loss at every site: DEV-SELFTARE-03).
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, "..", "atlas", "v2.1"); NA = 65535
FLAG = "read against IAM-Atlas v2.1; not commissioned by our standards (DEV-METAA-ATLAS-01)"
RULE = dict(spread_max=0.05, high=(0.75, 0.95), low=(0.05, 0.25), per_side=3000, coverage_min=0.90, normal=(0.95, 1.05), min_refs=3)
_cache = {}

def _H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))

def _col(stem, cell):
    """One cell's atlas column (best value mu, mu_lo, mu_hi or donor_sd), as a Series over the atlas's sites; NaN where the atlas has none."""
    import pyarrow.parquet as pq
    k = (stem, cell)
    if k in _cache: return _cache[k]
    if "ids" not in _cache: _cache["ids"] = pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist()
    for part in (1, 2, 3):
        p = os.path.join(ATLAS, f"{stem}_part{part}.parquet")
        if cell in pq.read_schema(p).names:
            u = pq.read_table(p, columns=[cell]).column(0).to_numpy(); v = u.astype("float64") / 1e4; v[u == NA] = np.nan
            _cache[k] = pd.Series(v, index=_cache["ids"]); return _cache[k]
    raise KeyError(f"'{cell}' is not a cell of IAM-Atlas v2.1")

def width(cell):
    """The healthy width at each site: the person-to-person spread and the atlas's own uncertainty, combined."""
    un = (_col("mu_hi", cell) - _col("mu_lo", cell)) / 3.92
    return np.sqrt(_col("donor_sd", cell) ** 2 + un ** 2)

def identity_sites(cell):
    """Met-A's site rule on the atlas (see the module text). Returns the site names, high side then low side."""
    k = ("sites", cell)
    if k in _cache: return _cache[k]
    mu, w = _col("mu", cell), width(cell); ok = mu.notna() & w.notna() & (w <= RULE["spread_max"])
    out = []
    for lo, hi in (RULE["high"], RULE["low"]):
        s = w[ok & (mu >= lo) & (mu <= hi)].sort_values(kind="mergesort")
        out += list(s.index[:RULE["per_side"]])
    _cache[k] = out; return out

def specific_sites(cell, others, delta):
    """Identity sites held by this cell and not by the others (development, 2026-10-22, to-do 39.9; written before any comparison): Met-A's site
    rule (healthy width at most spread_max, level in the held windows), and at each kept site the cell's atlas value differs by at least delta
    from every other cell named that has a value there; then the tightest per_side a side, as identity_sites. others: the other cells of the
    sample's panel. With related cells in the panel, few sites survive a large delta; the count is recorded, never padded."""
    k = ("specific", cell, tuple(sorted(others)), float(delta))
    if k in _cache: return _cache[k]
    mu, w = _col("mu", cell), width(cell); ok = mu.notna() & w.notna() & (w <= RULE["spread_max"])
    for o in others:
        mo = _col("mu", o).reindex(mu.index); ok &= mo.isna() | ((mu - mo).abs() >= delta)
    out = []
    for lo, hi in (RULE["high"], RULE["low"]):
        out += list(w[ok & (mu >= lo) & (mu <= hi)].sort_values(kind="mergesort").index[:RULE["per_side"]])
    _cache[k] = out; return out

N_EXACT = 400   # expected_H_counts: above this many reads the k-of-n bias is below 0.2 % of H and H(p) is used (stated)
def expected_H_counts(p, n):
    """The exact expected entropy of a level measured as k of n reads when the true level is p (2026-10-22, to-do 40.3; derived, nothing fitted):
    E[H(k/n)] = sum_k C(n, k) p^k (1 - p)^(n - k) H(k/n), k ~ Binomial(n, p). A level made from few reads is 0 or 1 more often than p, and H(0) =
    H(1) = 0, so the measured disorder of a healthy site at depth n is this, not H(p). p, n: arrays by site. Above N_EXACT reads, H(p)."""
    from scipy.stats import binom
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6); n = np.asarray(n, dtype="int64"); out = np.full(p.shape, np.nan)
    big = n > N_EXACT; out[big] = np.asarray(_H(pd.Series(p[big])), dtype="float64") if big.any() else out[big]
    for m in np.unique(n[(n >= 1) & ~big]):
        i = n == m; k = np.arange(m + 1); hk = np.asarray(_H(pd.Series(k / m)), dtype="float64")
        out[i] = binom.pmf(k[None, :], m, p[i][:, None]) @ hk
    return out

def read(beta, cell, specimen="isolated", counts=None):
    """Met-A against the atlas cell (untared). beta: Series probe -> beta from Stage 1."""
    S = identity_sites(cell); rec = dict(stage="M-atlas", reading="Met-A against IAM-Atlas v2.1", cell=cell, specimen=specimen, flag=FLAG,
                                         rule=RULE, n_identity_sites=len(S), A=None)
    if not S: rec["reason"] = f"no site of '{cell}' meets Met-A's site rule in the atlas"; return rec
    x = beta.reindex(S); mu = _col("mu", cell).reindex(S); ok = x.notna() & mu.notna()
    if ok.sum() < RULE["coverage_min"] * len(S):
        rec["reason"] = f"only {int(ok.sum())} of {len(S)} identity sites measured (at least {RULE['coverage_min']:.0%} needed)"; return rec
    href = float(_H(mu[ok]).mean())
    if counts is not None:   # sequencing: the healthy expectation at each site's own read count (40.3)
        _n = pd.Series(counts).reindex(mu[ok].index).fillna(0).astype("int64").to_numpy(); href = float(np.nanmean(expected_H_counts(mu[ok].to_numpy(), _n)))
        rec["expectation"] = "k of n reads at each site's own depth (expected_H_counts)"
    A = float(_H(x[ok]).mean() / href)
    xd = (x[ok] + 0.01 * (0.5 - mu[ok]))   # a 1 % loss of the cell's pattern: every identity site moved 1 % of the way to 0.5
    rec.update(A=round(A, 5), n_sites=int(ok.sum()), H_ref_atlas=round(href, 5),
               shift_per_1pct_loss=round(float(_H(xd).mean() / _H(x[ok]).mean() * A - A), 5),
               state="untared: read A_rel against same-run healthy references of the same cell (tare)")
    return rec

def tare(A, refs):
    """The same-run tare: A_rel = A / median(refs); spread of the references and the detection limit as conductor_v3.stage_t_tare."""
    r = [float(v) for v in refs if v is not None and np.isfinite(v)]
    out = dict(rule="A_rel = A / median(A of same-run healthy references of the same cell)", n_refs=len(r), flag=FLAG)
    if A is None or len(r) < RULE["min_refs"]:
        out.update(A_rel=None, reason=f"{len(r)} same-run healthy references of this cell given; at least {RULE['min_refs']} needed"); return out
    m = float(np.median(r)); a = A / m; sp = float(np.std(np.array(r) / m, ddof=1)) if len(r) > 1 else None
    out.update(A_rel=round(a, 5), reference_median=round(m, 5), reference_spread=round(sp, 5) if sp is not None else None,
               state="Normal" if RULE["normal"][0] <= a <= RULE["normal"][1] else ("above Normal" if a > RULE["normal"][1] else "below Normal"))
    return out

def plant_all(beta, cell, loss=0.01):
    """A CONSTRUCTED loss of the cell's whole pattern, for tests only (2026-10-22): every atlas site the array measured where the cell has a value,
    moved loss of the way to 0.5, x + loss (0.5 - mu). A real loss of hold would reach every held site, not only the identity sites."""
    b = beta.copy(); mu = _col("mu", cell).reindex(b.index); ok = b.notna() & mu.notna()
    b.loc[ok] = (b[ok] + loss * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6); return b

RUNS, RUN_LEN = 10, 30   # plant_runs: 10 runs of 30 neighboring identity sites (300 of 6,000), the shape of a clustered loss (stated)
def plant_runs(beta, cell, loss=0.05, seed=20261022):
    """A CONSTRUCTED clustered loss, for tests only (2026-10-22, to-do 38.1): RUNS runs of RUN_LEN neighboring identity sites of the cell, in genome
    order (the frozen sky order, chain/Runtime Matrices/Atlas_V21/sky_order_v21.npz), chosen by a seeded draw; each site moved loss of the way to
    0.5, x + loss (0.5 - mu). The loss is concentrated in those runs, as a loss that spreads along the genome would be."""
    import pyarrow.parquet as pq
    Z = np.load(os.path.join(HERE, "Runtime Matrices", "Atlas_V21", "sky_order_v21.npz"))
    if "ids" not in _cache: _cache["ids"] = pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist()
    io = np.array(_cache["ids"], dtype=object)[Z["order"]]; pos = pd.Series(np.arange(len(io)), index=io)
    S = [x for x in identity_sites(cell) if x in pos.index and x in beta.index and pd.notna(beta.get(x))]
    S = sorted(S, key=lambda x: pos[x]); n = len(S) // RUN_LEN; rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(n, size=min(RUNS, n), replace=False)); sites = [x for k in pick for x in S[k * RUN_LEN:(k + 1) * RUN_LEN]]
    b = beta.copy(); mu = _col("mu", cell).reindex(sites); ok = mu.notna()
    b.loc[mu[ok].index] = (b.loc[mu[ok].index] + loss * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6); return b

def plant(beta, cell, loss=0.01):
    """A CONSTRUCTED loss of the cell's pattern, for tests only (the rule read() predicts its shift by): every identity site of the cell this
    array measured moved loss of the way to 0.5, x + loss (0.5 - mu). Returns a new Series; the input is unchanged."""
    S = identity_sites(cell); b = beta.copy(); x = b.reindex(S); mu = _col("mu", cell).reindex(S); ok = x.notna() & mu.notna()
    b.loc[x[ok].index] = (x[ok] + loss * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6); return b


MIX_COVER = 0.99   # read_in_mix: a site counts when the cells with an atlas value there hold at least this share of the sample (operational, stated)
def read_in_mix(beta, shares, counts=None, specific=None):
    """Met-A for each cell of a mixed sample (whole blood), against IAM-Atlas v2.1 mixed in this sample's own cell shares (development, written
    2026-10-22; author: read it and flag it until a sealed test passes). shares: {atlas cell: share} from the IAM-Deconvolver (Stage D21).
    For each cell c: its identity sites (identity_sites, Met-A's rule on the atlas); the healthy expectation e = sum_k f_k mu_k at those sites,
    the atlas's best values mixed in this sample's shares; A_c = mean H(beta) / mean H(e) over the sites both measured; and the shift a 1 % loss
    of c's pattern would make at this sample's share of c, beta + f_c x 0.01 x (0.5 - mu_c) (the rule conductor_v3.stage_m_blood uses for
    neutrophils), so a small share shows as a small shift. Untared: the same-run tare against healthy samples of the same specimen is the step
    that makes A_c read 1 when healthy. Known from exploration (GRP-014, 2026-10-21): a loss at every one of a cell's sites is partly absorbed
    into the Deconvolver's share of that cell, so A_c reads a loss smaller than planted; the record says so."""
    cells = [c for c, f in shares.items() if f is not None]
    out = dict(stage="M-atlas-mix", reading="Met-A for each cell, against IAM-Atlas v2.1 in this sample's own cell shares", flag=FLAG, cells=[],
               shares_from="IAM-Deconvolver (Stage D21)",
               note="untared; a loss at every site of a cell is partly absorbed into that cell's share (GRP-014), so a loss reads smaller than planted")
    if not cells: out["reason"] = "no cell shares"; return out
    if specific is not None: out.update(sites_rule=f"specific_sites: Met-A's site rule and at least {specific} from every other panel cell (development, 39.9)")
    for c in cells:
        S = identity_sites(c) if specific is None else specific_sites(c, [k for k in cells if k != c], specific)
        r = dict(cell=c, share=round(float(shares[c]), 4), n_identity_sites=len(S), A=None)
        if len(S) < (1 if specific is None else 200): r["reason"] = f"{len(S)} sites meet the site rule" + ("" if specific is None else f" with {specific} from every other cell (at least 200 needed)"); out["cells"].append(r); continue
        # 2026-10-22: the expectation is mixed over the cells with an atlas value at each site, renormalized, and a site is kept when those cells
        # hold at least MIX_COVER of the shares (on the full 79-cell panel a single small cell without a value had dropped about half the sites)
        MU = pd.concat([_col("mu", k).reindex(S).rename(k) for k in cells], axis=1); F = pd.Series({k: float(shares[k]) for k in cells})
        have = MU.notna(); cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12)
        e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0); x = beta.reindex(S); mu = _col("mu", c).reindex(S)
        ok = x.notna() & e.notna() & mu.notna() & (cover >= MIX_COVER)
        if ok.sum() < RULE["coverage_min"] * len(S):
            r["reason"] = f"only {int(ok.sum())} of {len(S)} identity sites measured (at least {RULE['coverage_min']:.0%} needed)"; out["cells"].append(r); continue
        he = float(_H(e[ok]).mean())
        if counts is not None:   # sequencing: the expectation at each site's own read count (40.3)
            _n = pd.Series(counts).reindex(e[ok].index).fillna(0).astype("int64").to_numpy(); he = float(np.nanmean(expected_H_counts(e[ok].to_numpy(), _n)))
        A = float(_H(x[ok]).mean() / he); xd = (x[ok] + float(shares[c]) * 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        _hx = np.asarray(_H(x[ok]), dtype="float64"); _he = np.asarray(_H(e[ok]), dtype="float64")
        r["se_over_sites"] = round(float(np.sqrt(((_hx - A * _he) ** 2).sum()) / _he.sum()), 6)   # 2026-10-22: for the all-site comparison
        r.update(A=round(A, 5), n_sites=int(ok.sum()), H_expectation=round(he, 5), expectation_rule=f"mixed over the cells with an atlas value; site kept at >= {MIX_COVER:.0%} of the shares",
                 shift_per_1pct_loss=round(float(_H(xd).mean() / he - A), 5), state="untared")
        out["cells"].append(r)
    return out


def departure_in_mix(beta, shares, cell):
    """The per-site departure behind Met-A for one cell of a mixed sample (development, 2026-10-22, for the globe's layer for each cell): at the
    cell's identity sites, H(beta) - H(e), e the atlas mixed in this sample's shares by read_in_mix's rule (MIX_COVER). A pd.Series by site id,
    or None when the cell has no identity sites. Positive: more disorder than this mix of healthy cells holds there."""
    S = identity_sites(cell)
    if not S: return None
    cells = [c for c, f in shares.items() if f is not None]
    MU = pd.concat([_col("mu", k).reindex(S).rename(k) for k in cells], axis=1); F = pd.Series({k: float(shares[k]) for k in cells})
    have = MU.notna(); cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12)
    e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0); x = beta.reindex(S)
    ok = x.notna() & e.notna() & (cover >= MIX_COVER)
    return pd.Series(np.asarray(_H(x[ok]), dtype="float64") - np.asarray(_H(e[ok]), dtype="float64"), index=x[ok].index)   # site ids kept


def read_in_mix_allsites(beta, shares):
    """Met-A for each cell over EVERY IAM-Atlas v2.1 site (development, written 2026-10-22 before any comparison is run; author: "Met-A should be
    able to read 800K sites so long as they correspond to the cells in the atlas"). For cell c, each site s counts by how much a loss of c's
    pattern would move it, the chain's own shift rule (conductor_v3.stage_m_blood): w_s = f_c x |0.5 - mu_c,s|. Nothing is chosen or fitted:
    a site c does not hold firmly, or where c is a small share, counts for little by itself. A_c = sum w H(beta) / sum w H(e), e the atlas mixed
    in the sample's own shares (read_in_mix's rule, MIX_COVER); also the shift a 1 % loss of c would make, and the reading's own standard error
    over sites (weighted, from the per-site ratios), so the two versions can be compared by how far a planted loss moves each in its own error."""
    cells = [c for c, f in shares.items() if f is not None]
    out = dict(stage="M-atlas-mix-allsites", reading="Met-A for each cell over every atlas site, each site weighted by the shift a loss of that cell would make",
               flag=FLAG, cells=[], weight_rule="w = share x |0.5 - mu| (conductor_v3.stage_m_blood's shift)")
    if not cells: out["reason"] = "no cell shares"; return out
    x = pd.Series(beta, dtype="float64"); idx = x.dropna().index
    MU = pd.concat([_col("mu", k).reindex(idx).rename(k) for k in cells], axis=1); F = pd.Series({k: float(shares[k]) for k in cells})
    have = MU.notna(); cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12)
    e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0)
    ok = e.notna() & (cover >= MIX_COVER); xs, es = x.reindex(idx)[ok], e[ok]
    Hx = np.asarray(_H(xs), dtype="float64"); He = np.asarray(_H(es), dtype="float64")
    for c in cells:
        mu = MU[c][ok]; m = mu.notna().to_numpy(); f = float(shares[c]); w = f * np.abs(0.5 - mu.to_numpy()[m]); hx, he = Hx[m], He[m]
        r = dict(cell=c, share=round(f, 4), n_sites=int(m.sum()), A=None)
        if m.sum() < 1000 or w.sum() <= 0: r["reason"] = f"only {int(m.sum())} sites with a value"; out["cells"].append(r); continue
        A = float((w * hx).sum() / (w * he).sum())
        xd = np.clip(xs.to_numpy()[m] + f * 0.01 * (0.5 - mu.to_numpy()[m]), 1e-6, 1 - 1e-6)
        sh = float((w * np.asarray(_H(pd.Series(xd)), dtype="float64")).sum() / (w * he).sum() - A)
        # standard error over sites: the weighted sum's linearized error, from each site's departure from the reading
        dv = hx - A * he; se = float(np.sqrt((w ** 2 * dv ** 2).sum()) / (w * he).sum())
        neff = float(w.sum() ** 2 / (w ** 2).sum())
        r.update(A=round(A, 5), shift_per_1pct_loss=round(sh, 5), se_over_sites=round(se, 6), effective_sites=int(neff), state="untared")
        out["cells"].append(r)
    return out


CS_BLOCK = 10   # the commissioned C-Score's block: 10 neighboring sites (conductor_v3, clustering_block)
CS_FLAG_SHARE = 0.01   # every cell found is scored; below 1 % of the sample its C is flagged, because its departure map is then mostly the other cells' (stated, not derived)

def _clustering(z, w=CS_BLOCK):
    """The commissioned C-Score's statistic (conductor_v3._clustering), unchanged: the departures in genome order, the variance of the means of
    blocks of w neighboring sites (x sqrt(w)) over the variance of the sites. Departures independent from site to site give 1; departures that run
    along the genome give more. It does not depend on the size of the departures, only on how they are laid out."""
    o = np.asarray(z, dtype="float64"); o = o[np.isfinite(o)]; nb = len(o) // w
    if nb < 10 or float(np.var(o)) == 0.0: return None
    b = o[:nb * w].reshape(nb, w).mean(1) * np.sqrt(w); return float(np.var(b) / np.var(o))

def _genome_order(ids):
    """Site ids sorted in genome order: the order of IAM-Atlas v2.1's sites.parquet, the order plant_runs and the globe use."""
    import pyarrow.parquet as pq
    if "ids" not in _cache: _cache["ids"] = pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist()
    if "pos" not in _cache: _cache["pos"] = pd.Series(np.arange(len(_cache["ids"])), index=_cache["ids"])
    p = _cache["pos"].reindex(pd.Index(ids)); k = p.notna().to_numpy(); return list(pd.Index(ids)[k][np.argsort(p.to_numpy()[k])])

def cscore_in_mix(beta, shares, deps=None):
    """The C-Score for each cell found in a mixed sample, and for the whole genome (development, written 2026-10-23 before it runs; plan 41.2,
    41.3; author: 'I wish we had the C-Scores wired in ... they could potentially shine here'). For each cell c with a share of at least
    any share the IAM-Deconvolver found (the instrument says what is there; below CS_FLAG_SHARE the C is flagged, not withheld): its departure map at its identity sites (departure_in_mix: H(beta) - H(e), e the atlas mixed in this sample's own shares, the map
    the globe's layer for c draws), in genome order, read by the commissioned C-Score's statistic (_clustering, blocks of 10). Whole genome: the
    same at every atlas site the sample measured, e mixed by read_in_mix's rule. Untared: the commissioned C-Score reads after a same-run tare
    against healthy samples; here healthy samples of the same series are the comparison until a tare for these readings is built."""
    cells = [c for c, f in shares.items() if f is not None]
    out = dict(stage="C-atlas-mix", reading="C-Score for each cell and for the whole genome, against IAM-Atlas v2.1 in this sample's own shares",
               flag=FLAG, block_sites=CS_BLOCK, flag_share=CS_FLAG_SHARE, cells=[], note="untared; healthy samples of the same series are the comparison")
    if not cells: out["reason"] = "no cell shares"; return out
    for c in cells:
        f = float(shares[c]); r = dict(cell=c, share=round(f, 4), C=None)
        if f <= 0: continue   # a cell the Deconvolver did not find
        if f < CS_FLAG_SHARE: r["share_flag"] = f"share below {CS_FLAG_SHARE}: the departure map at its sites is mostly the other cells'"
        d = (deps or {}).get(c)
        if d is None: d = departure_in_mix(beta, shares, c)
        d = None if d is None else pd.Series(d, dtype="float64").dropna()
        if d is None or len(d) < 10 * CS_BLOCK: r["reason"] = "too few identity sites with a departure"; out["cells"].append(r); continue
        C = _clustering(d.reindex(_genome_order(d.index)).to_numpy()); r.update(C=None if C is None else round(C, 4), n_sites=int(len(d))); out["cells"].append(r)
    x = pd.Series(beta, dtype="float64").dropna(); idx = x.index
    MU = pd.concat([_col("mu", k).reindex(idx).rename(k) for k in cells], axis=1); F = pd.Series({k: float(shares[k]) for k in cells})
    have = MU.notna(); cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12)
    e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0); ok = e.notna() & (cover >= MIX_COVER)
    dw = pd.Series(np.asarray(_H(x[ok]), dtype="float64") - np.asarray(_H(e[ok]), dtype="float64"), index=x[ok].index)
    C = _clustering(dw.reindex(_genome_order(dw.index)).to_numpy()) if len(dw) >= 10 * CS_BLOCK else None
    out["whole_genome"] = dict(C=None if C is None else round(C, 4), n_sites=int(len(dw)))
    return out



# ---- cancer's own territories (2026-10-23; DEVELOPMENT, plan 48; author: 'we know what is missing and we can find it') ---------------------
# Met-A's identity sites are by rule at healthy 0.05-0.25 or 0.75-0.95 in the cell, so neither place early cancer acts is read by it. Two fixed
# rules on IAM-Atlas v2.1, over the sites measured in all 79 cell types; nothing is chosen by hand and nothing is fitted:
TERR_RULES = dict(islands="unmarked in every healthy cell type: the highest of the 79 values below 0.2 (where Polycomb-island gains fall)",
                  domains="marked in most and loosely held in some: mean of the 79 at least 0.75, none below 0.2, at least one between 0.2 and 0.7")

def territory_sets():
    """The two site sets, computed from the atlas column by column (memory near one column at a time) and cached."""
    if "terr" in _cache: return _cache["terr"]
    import pyarrow.parquet as pq
    ids = None; S = MX = MN = C = MID = None
    for part in (1, 2, 3):
        t = pq.read_table(os.path.join(ATLAS, f"mu_part{part}.parquet"))
        for c in t.column_names:
            u = t.column(c).to_numpy(); ok = u != NA; x = u / 1e4
            if S is None:
                n = len(u); S = np.zeros(n); MX = np.full(n, -1.0); MN = np.full(n, 2.0); C = np.zeros(n, int); MID = np.zeros(n, int)
            S += np.where(ok, x, 0.0); C += ok; MX = np.where(ok, np.maximum(MX, x), MX); MN = np.where(ok, np.minimum(MN, x), MN); MID += ok & (x >= 0.2) & (x < 0.7)
        del t
        if ids is None: ids = pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist()
    full = C == C.max(); mu = S / np.maximum(C, 1); ids = np.asarray(ids, dtype=object)
    out = dict(islands=list(ids[full & (MX < 0.2)]), domains=list(ids[full & (mu >= 0.75) & (MN >= 0.2) & (MID >= 1)]))
    _cache["terr"] = out; return out

def territory_reading(beta, shares):
    """The sample's disorder at each territory against IAM-Atlas v2.1 mixed in its own shares (read_in_mix's mix rule): A = mean H(beta) over
    mean H(e); also the mean level change (beta - e: positive a gain, negative a loss), the reading's standard error over sites, and how much of
    the set this array measures. Development, flagged; healthy is read first and its spread sets any line, not this function."""
    x = pd.Series(beta, dtype="float64"); cells = [c for c, f in shares.items() if f]; F = pd.Series({k: float(shares[k]) for k in cells})
    out = dict(stage="M-territories", reading="disorder at cancer's own territories, against the atlas in this sample's mix", flag=FLAG, rules=TERR_RULES, sets={})
    if not cells: out["reason"] = "no cell shares"; return out
    for name, S in territory_sets().items():
        idx = pd.Index(S).intersection(x.dropna().index)
        MU = pd.concat([_col("mu", k).reindex(idx).rename(k) for k in cells], axis=1); have = MU.notna()
        cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12); e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0)
        ok = e.notna() & (cover >= MIX_COVER); xs, es = x.reindex(idx)[ok], e[ok]; r = dict(set_size=len(S), measured=int(ok.sum()), A=None)
        if ok.sum() < 500: r["reason"] = f"only {int(ok.sum())} of {len(S)} sites measured"; out["sets"][name] = r; continue
        hx = np.asarray(_H(xs), dtype="float64"); he = np.asarray(_H(es), dtype="float64"); A = float(hx.mean() / he.mean())
        se = float(np.std(hx - A * he, ddof=1) / np.sqrt(len(hx)) / he.mean()); d = (xs - es).to_numpy()
        r.update(A=round(A, 5), se_over_sites=round(se, 6), level_change=round(float(d.mean()), 5), level_change_se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 6),
                 coverage=round(ok.sum() / len(S), 4), state="untared")
        out["sets"][name] = r
    return out

def plant_territory(beta, name, frac, size, seed=20261023):
    """TESTS ONLY: a constructed change at one territory: at a seeded fraction frac of its measured sites, islands gain (beta + size(1 - beta))
    and domains lose (beta (1 - size)). Returns the planted values and how many sites moved."""
    b = pd.Series(beta, dtype="float64").copy(); S = pd.Index(territory_sets()[name]).intersection(b.dropna().index)
    pick = np.random.default_rng(seed).random(len(S)) < frac; s = S[pick]
    b.loc[s] = (b.loc[s] + size * (1 - b.loc[s])) if name == "islands" else (b.loc[s] * (1 - size))
    return b, int(pick.sum())



def read_whole(beta, shares):
    """Met-A for the whole sample over EVERY IAM-Atlas v2.1 site (development, 2026-10-23; author: 'adjust Met-A to read from the entire
    IAM-Atlas'). One reading per sample: e = the atlas mixed in the sample's own shares (read_in_mix's rule, MIX_COVER); each site counts by how
    firmly a healthy sample of this mix holds it, w = |0.5 - e| (a site held near 0 or 1 counts most, a half-held one least; nothing chosen,
    nothing fitted); A = sum w H(beta) / sum w H(e). Reported with the shift a 1 % loss of the whole held pattern would make, the reading's
    standard error over sites, the effective number of sites, and the same reading split by where the sites lie: Met-A's identity sites of any
    cell found, the islands and the domains (territory_sets), and the rest. The per-cell readings say which cell; this one says whether the sample
    as a whole has departed."""
    x = pd.Series(beta, dtype="float64"); cells = [c for c, f in shares.items() if f]; F = pd.Series({k: float(shares[k]) for k in cells})
    out = dict(stage="M-whole", reading="Met-A for the whole sample over every atlas site, against the atlas in this sample's mix", flag=FLAG,
               weight_rule="w = |0.5 - e|, e the atlas in the sample's shares", A=None)
    if not cells: out["reason"] = "no cell shares"; return out
    idx = x.dropna().index
    MU = pd.concat([_col("mu", k).reindex(idx).rename(k) for k in cells], axis=1); have = MU.notna()
    cover = have.mul(F, axis=1).sum(1) / max(F.sum(), 1e-12); e = MU.fillna(0.0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1).where(cover > 0)
    ok = e.notna() & (cover >= MIX_COVER); xs, es = x.reindex(idx)[ok], e[ok]
    if ok.sum() < 10000: out["reason"] = f"only {int(ok.sum())} sites with an expectation"; return out
    w = np.abs(0.5 - es.to_numpy()); hx = np.asarray(_H(xs), dtype="float64"); he = np.asarray(_H(es), dtype="float64")
    def one(m):
        if m.sum() < 500: return None
        A = float((w[m] * hx[m]).sum() / (w[m] * he[m]).sum()); dv = hx[m] - A * he[m]
        return dict(A=round(A, 5), se_over_sites=round(float(np.sqrt((w[m] ** 2 * dv ** 2).sum()) / (w[m] * he[m]).sum()), 6), sites=int(m.sum()))
    allm = np.ones(len(xs), bool); r = one(allm)
    xd = np.clip(xs.to_numpy() + 0.01 * (0.5 - es.to_numpy()), 1e-6, 1 - 1e-6)
    sh = float((w * np.asarray(_H(pd.Series(xd)), dtype="float64")).sum() / (w * he).sum() - r["A"])
    ident = set().union(*[set(identity_sites(c)) for c in cells]); T = territory_sets(); isl, dom = set(T["islands"]), set(T["domains"]); ix = xs.index
    mi = np.fromiter((i in ident for i in ix), bool, len(ix)); ms = np.fromiter((i in isl for i in ix), bool, len(ix)); md = np.fromiter((i in dom for i in ix), bool, len(ix))
    out.update(A=r["A"], se_over_sites=r["se_over_sites"], sites=r["sites"], shift_per_1pct_loss=round(sh, 5),
               effective_sites=int(w.sum() ** 2 / (w ** 2).sum()), state="untared",
               by_place={"identity sites of the cells found": one(mi), "islands": one(ms), "domains": one(md), "the rest": one(~mi & ~ms & ~md)})
    return out
