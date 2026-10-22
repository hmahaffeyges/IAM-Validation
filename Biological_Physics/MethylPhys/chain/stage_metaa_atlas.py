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

def read(beta, cell, specimen="isolated"):
    """Met-A against the atlas cell (untared). beta: Series probe -> beta from Stage 1."""
    S = identity_sites(cell); rec = dict(stage="M-atlas", reading="Met-A against IAM-Atlas v2.1", cell=cell, specimen=specimen, flag=FLAG,
                                         rule=RULE, n_identity_sites=len(S), A=None)
    if not S: rec["reason"] = f"no site of '{cell}' meets Met-A's site rule in the atlas"; return rec
    x = beta.reindex(S); mu = _col("mu", cell).reindex(S); ok = x.notna() & mu.notna()
    if ok.sum() < RULE["coverage_min"] * len(S):
        rec["reason"] = f"only {int(ok.sum())} of {len(S)} identity sites measured (at least {RULE['coverage_min']:.0%} needed)"; return rec
    href = float(_H(mu[ok]).mean()); A = float(_H(x[ok]).mean() / href)
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

def plant(beta, cell, loss=0.01):
    """A CONSTRUCTED loss of the cell's pattern, for tests only (the rule read() predicts its shift by): every identity site of the cell this
    array measured moved loss of the way to 0.5, x + loss (0.5 - mu). Returns a new Series; the input is unchanged."""
    S = identity_sites(cell); b = beta.copy(); x = b.reindex(S); mu = _col("mu", cell).reindex(S); ok = x.notna() & mu.notna()
    b.loc[x[ok].index] = (x[ok] + loss * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6); return b


def read_in_mix(beta, shares):
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
    for c in cells:
        S = identity_sites(c); r = dict(cell=c, share=round(float(shares[c]), 4), n_identity_sites=len(S), A=None)
        if not S: r["reason"] = "no site meets Met-A's site rule in the atlas"; out["cells"].append(r); continue
        e = sum(float(shares[k]) * _col("mu", k).reindex(S) for k in cells); x = beta.reindex(S); mu = _col("mu", c).reindex(S)
        ok = x.notna() & e.notna() & mu.notna()
        if ok.sum() < RULE["coverage_min"] * len(S):
            r["reason"] = f"only {int(ok.sum())} of {len(S)} identity sites measured (at least {RULE['coverage_min']:.0%} needed)"; out["cells"].append(r); continue
        he = float(_H(e[ok]).mean()); A = float(_H(x[ok]).mean() / he); xd = (x[ok] + float(shares[c]) * 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        r.update(A=round(A, 5), n_sites=int(ok.sum()), H_expectation=round(he, 5),
                 shift_per_1pct_loss=round(float(_H(xd).mean() / he - A), 5), state="untared")
        out["cells"].append(r)
    return out
