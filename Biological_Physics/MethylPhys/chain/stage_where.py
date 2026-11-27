"""The Locator (instrument 22): where the sample changed, place by place, and the name of each place.

Every Law Sphere unit (a CpG island, or a block of 50 CpGs between islands) is read on two sides from the counts the Law Sphere already makes
from the sample's molecules:
  loss  held errors: marks missing at sites a held molecule keeps marked (err out of opp, IAM-A's molecule rule);
  gain  stray marks: marks present at sites a molecule keeps unmarked (gerr out of gopp).
Expected count at a unit: E = opp x s x r, where s is the sample's own overall rate on that side (so depth and the sample's general level divide
out) and r is the unit's healthy relative rate, the atlas cell types weighted by the sample's Deconvolver shares (types without a healthy
reference, such as megakaryocytes, drop out and the rest are renormalized). The healthy reference holds r for 57 cell types over 237,850 units,
built from 156 healthy atlas libraries (each read self-referenced, so its own rate divides out).
The null is MEASURED, not assumed: each healthy library was read against the reference built without it (leave one out); in bins of E the
spread of (obs - E) / sqrt(E) gives a variance inflation phi, and a unit's p-value is the upper tail of a negative binomial with mean E and
variance phi x E (Poisson where phi <= 1). Places are called by Benjamini and Hochberg at a 5 % false discovery rate, per sample and per side.
Each called place is named from the positions reference v2 (chromosome, hg19 coordinates, CpG island, gene relation, CpG context, array probes).
Detection limit per sample and side: at the median expected count of the units tested, the smallest count that would pass the cut, as a
multiple of the expected count. It falls with depth.
Built and tested before entering the chain: no false places in planted or unplanted reads of the known-answer library (165,285 loss and 46,973
gain units); 9 of 10 planted gain islands named; loss blocks named from about 4 to 6 times their healthy loss at that depth.
"""
import os, json, numpy as np

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Runtime Matrices", "Where")
RATES = "where_rates_v2_part{}.npz"; RATE_PARTS = 8; NULL = "where_null_v2.json"; POSITIONS = "where_units_v2.parquet"
EB = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, 1e12])
FDR = 0.05
TOP_IN_BUNDLE = 300
NAME_COLS = ["unit", "chr", "start", "end", "kind", "cpgs", "array_probes", "island", "nearest_gene_refseq", "relation_refseq", "tss_distance_refseq",
             "promoter_genes_refseq", "body_genes_refseq", "cpg_context"]
_REF = {}


def load_ref():
    """The healthy relative rates, the measured null and the positions table (read once)."""
    if not _REF:
        # the rates are split by groups of cell types (19 per part) so each file stays under the repository's upload limit; every part holds every unit
        P = [np.load(os.path.join(REF_DIR, RATES.format(k)), allow_pickle=False) for k in range(1, RATE_PARTS + 1)]
        assert all(np.array_equal(p["unit"], P[0]["unit"]) for p in P), "rate parts disagree on their units"
        _REF.update(unit=P[0]["unit"], types=[str(t) for p in P for t in p["types"]], loss=np.concatenate([p["loss"] for p in P]), gain=np.concatenate([p["gain"] for p in P]),
                    null=json.load(open(os.path.join(REF_DIR, NULL), encoding="utf-8")))
    return _REF


def pval(obs, E, bins):
    """Upper-tail p of an observed count: negative binomial with mean E and variance phi x E, phi measured on the healthy leave-one-out null of
    the same expected-count bin (phi <= 1: Poisson). P(X >= obs)."""
    from scipy.stats import nbinom, poisson
    p = np.ones(len(obs)); b = np.digitize(E, EB) - 1
    for k, B in enumerate(bins):
        m = b == k
        if not m.any(): continue
        phi = B.get("phi") or 1.0
        if phi <= 1.0001: p[m] = poisson.sf(obs[m] - 1, E[m])
        else: r = E[m] / (phi - 1); p[m] = nbinom.sf(obs[m] - 1, r, r / (r + E[m]))
    return np.clip(p, 1e-300, 1)


def bh(p, q=FDR):
    """Benjamini and Hochberg (1995): which p-values pass at false discovery rate q, the q-value of each, and the p cut used."""
    n = len(p); o = np.argsort(p); k = np.nonzero(p[o] <= q * np.arange(1, n + 1) / n)[0]
    keep = np.zeros(n, bool); cut = q * ((k[-1] + 1) if len(k) else 1) / max(n, 1)
    if len(k): keep[o[:k[-1] + 1]] = True
    qv = np.empty(n); qv[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return keep, np.minimum(qv, 1), cut


def detection_limit(E, bins, cut):
    """At the median expected count of the tested units: the smallest count whose p passes the cut, and that count over the expected count."""
    em = float(np.median(E)) if len(E) else 0.0
    if em <= 0: return dict(median_expected=em, count_needed=None, times_expected=None)
    hi = int(max(10, em * 200)); xs = np.arange(0, hi + 1, dtype=float)
    ok = np.nonzero(pval(xs, np.full(len(xs), em), bins) <= cut)[0]
    if not len(ok): return dict(median_expected=round(em, 3), count_needed=None, times_expected=None)
    return dict(median_expected=round(em, 3), count_needed=int(xs[ok[0]]), times_expected=round(float(xs[ok[0]] / em), 2))


def read(C, shares):
    """C: the Law Sphere's counts by unit index (err, opp, gerr, gopp); shares: {cell type: share} from the Deconvolver.
    Returns, per side, the units tested, the places called (unit, observed, expected, z, q, p), the p cut and the detection limit."""
    R = load_ref(); u = R["unit"]; n = len(C["err"])
    w = np.array([float(shares.get(t, 0.0)) for t in R["types"]]); used = w.sum()
    if used <= 0: return dict(status="NOT_RUN", reason="no Deconvolver share falls on a cell type with a healthy reference")
    w = w / used
    inr = u < n; out = dict(status="OK", fdr=FDR, share_with_reference=round(float(used / max(sum(shares.values()), 1e-12)), 4))
    for nm, e, o in (("loss", "err", "opp"), ("gain", "gerr", "gopp")):
        obs = np.zeros(len(u)); opp = np.zeros(len(u)); obs[inr] = np.asarray(C[e])[u[inr]]; opp[inr] = np.asarray(C[o])[u[inr]]
        Rt = R[nm].astype(np.float64); W = np.where(np.isfinite(Rt), w[:, None], 0.0); r = np.nansum(np.nan_to_num(Rt) * W, 0) / np.maximum(W.sum(0), 1e-12)
        # the sample's own rate over the reference's units only, as the reference and its measured null were built (edit 266; edit 259 took it
        # over every unit the Law Sphere counts, about 592,000, which lowered the gain side's rate and the expected counts)
        ref_u = inr & (W.sum(0) > 0); s = float(obs[ref_u].sum() / max(opp[ref_u].sum(), 1)); E = opp * s * r; m = ref_u & (E > 0)
        bins = R["null"]["bins"][nm]; p = np.ones(len(u)); p[m] = pval(obs[m], E[m], bins)
        keep, qv, cut = bh(p[m]); idx = np.where(m)[0]; P = idx[keep]
        z = (obs[P] - E[P]) / np.sqrt(E[P]); order = np.argsort(-z); qm = dict(zip(idx, qv))
        out[nm] = dict(tested=int(m.sum()), called=int(len(P)), p_cut=float(cut), sample_rate=s, detection_limit=detection_limit(E[m], bins, cut),
                       places=[dict(unit=int(u[i]), observed=int(obs[i]), expected=round(float(E[i]), 3), times_expected=round(float(obs[i] / E[i]), 2),
                                    z=round(float(zz), 2), q=float(qm[i]), p=float(p[i])) for i, zz in zip(P[order], z[order])])
    return out


def name(out):
    """Adds each called place's name from the positions reference v2 (reads only the rows of the called units)."""
    import pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    units = sorted({pl["unit"] for nm in ("loss", "gain") for pl in (out.get(nm) or {}).get("places", [])})
    if not units: return out
    T = pq.read_table(os.path.join(REF_DIR, POSITIONS), columns=NAME_COLS, filters=[("unit", "in", units)]).to_pandas().set_index("unit")
    for nm in ("loss", "gain"):
        S = out.get(nm) or {}; blind = 0
        for pl in S.get("places", []):
            if pl["unit"] in T.index:
                row = T.loc[pl["unit"]]; pl.update({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()})
                pl["array_visible"] = bool(row["array_probes"] > 0); blind += int(row["array_probes"] == 0)
        if S.get("places") is not None: S["array_blind_called"] = blind
    return out


def run(C, shares, places_csv=None):
    """The Locator in the one button: read, name, write the full place list beside the report (places_csv), keep the top places in the bundle."""
    out = read(C, shares)
    if out.get("status") != "OK": return out
    out = name(out)
    if places_csv:
        import pandas as pd
        rows = [dict(side=nm, **pl) for nm in ("loss", "gain") for pl in out[nm]["places"]]
        pd.DataFrame(rows).to_csv(places_csv, index=False); out["places_file"] = os.path.basename(places_csv)
    for nm in ("loss", "gain"):
        if len(out[nm]["places"]) > TOP_IN_BUNDLE: out[nm]["places"] = out[nm]["places"][:TOP_IN_BUNDLE]; out[nm]["places_in_bundle"] = TOP_IN_BUNDLE
    out["reference"] = dict(rates=RATES, null=NULL, positions=POSITIONS, types=len(load_ref()["types"]), units=int(len(load_ref()["unit"])),
                            libraries=load_ref()["null"].get("libraries"))
    return out

# ---- edit 277 (finding F304): a genetic variant at a CpG reads, through bisulfite, as a lost mark on every molecule that carries it ----
VARIANT_TOP = 0.6     # the one or two CpGs that carry at least this share of a place's changed calls (operational, from F304: 0.92 at one CpG)
VARIANT_T_MIN = 0.35  # and read changed on at least this share of the molecules covering them (one copy about 0.5, both copies about 1)

def variant_check(pat_path, out, isl_first, isl_last, block, min_calls=4, min_meth=0.75):
    """Edit 277: for each place the Locator named, the calls of the sample's own molecules CpG by CpG (one pass over the molecule file, only the
    named places' CpGs). Loss side: T calls on held molecules (at least min_calls calls, at least min_meth of them C); gain side: C calls on unmarked
    molecules (at most 1 - min_meth C). A place where the top one or two CpGs carry at least VARIANT_TOP of the changed calls, each read changed on
    at least VARIANT_T_MIN of the molecules covering it, is marked 'probable genetic variant at a CpG' (the CpGs printed) and counted apart from the
    changes: a variant changes one CpG on every molecule that carries it, a change of methylation spreads over the place."""
    import gzip, bisect, collections
    nI = len(isl_first); R = []
    for side in ("loss", "gain"):
        for p in (out.get(side) or {}).get("places") or []:
            u = int(p["unit"]); a, b = (int(isl_first[u]), int(isl_last[u])) if u < nI else ((u - nI) * block + 1, (u - nI + 1) * block)
            R.append((a, b, side, p))
    if not R: return out
    R.sort(key=lambda r: r[0]); starts = [r[0] for r in R]; ch = collections.defaultdict(collections.Counter); cov = collections.defaultdict(collections.Counter)
    with gzip.open(pat_path, "rt") as f:
        for line in f:
            q = line.split("\t")
            if len(q) < 4: continue
            s, pat, w = int(q[1]), q[2], int(q[3]); e = s + len(pat) - 1
            j = bisect.bisect_right(starts, e) - 1
            while j >= 0 and R[j][1] >= s - 2000:
                a, b, side, p = R[j]
                if a <= e and b >= s:
                    n = sum(1 for x in pat if x != "."); nc = pat.count("C")
                    keep = (n >= min_calls) and ((nc >= min_meth * n) if side == "loss" else (nc <= (1 - min_meth) * n))
                    if keep:
                        bad = "T" if side == "loss" else "C"
                        for k, x in enumerate(pat):
                            i = s + k
                            if a <= i <= b and x in "CT": cov[(side, p["unit"])][i] += w; ch[(side, p["unit"])][i] += w if x == bad else 0
                j -= 1
    for side in ("loss", "gain"):
        o = out.get(side) or {}; nv = 0
        for p in o.get("places") or []:
            c, v = ch[(side, p["unit"])], cov[(side, p["unit"])]; tot = sum(c.values())
            top = sorted(c, key=lambda i: -c[i])[:2]; top = [i for i in top if v[i] and c[i] / v[i] >= VARIANT_T_MIN]
            p["changed_calls"] = tot; p["top_cpgs"] = [dict(cpg=i, changed=c[i], covered=v[i]) for i in sorted(c, key=lambda i: -c[i])[:2]]
            if tot and top and sum(c[i] for i in top) >= VARIANT_TOP * tot:
                p["probable_variant"] = ("probable genetic variant at a CpG: " + ", ".join(f"CpG {i} changed on {c[i]} of {v[i]} molecules" for i in top) +
                                         "; a variant changes one CpG on every molecule that carries it, so this place is not counted as a change of methylation"); nv += 1
        if o: o["probable_variants"] = nv; o["called_changes"] = int(o.get("called") or 0) - nv
    out["variant_check"] = dict(rule=f"top one or two CpGs carry at least {VARIANT_TOP:.0%} of the place's changed calls, each changed on at least {VARIANT_T_MIN:.0%} of the molecules covering it", source="the sample's own molecules, named places only")
    return out
