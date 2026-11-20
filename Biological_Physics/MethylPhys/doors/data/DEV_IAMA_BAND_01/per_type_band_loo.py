"""DEV-IAMA-BAND-01 / COM-IAMA-054 (2026-11-20, exploration on spent data). Per-type IAM-A band vs the pooled band, LIKE FOR LIKE, leave one donor out (2026-11-20; supersedes the preliminary numbers in bands_compare.py,
which scored the pooled band v2 in-sample). Data: per_type_band_loo_inputs.csv: each IAM Sequencing Atlas library's A_derived from the one-button joint run (derived_line_v1), its cell type, and
the GEO fields used as the donor stand-in (lab, age, sex; GEO gives no donor id). Run: python3 per_type_band_loo.py [inputs.csv] [out_dir].
For each library L of type t, everything from L's donor stand-in is removed, then:
  POOLED band (band v2's own rule, rebuilt): 2.5th-97.5th percentile of A_derived over all remaining libraries of all types.
  PER-TYPE band: centre m_t = median A_derived of the remaining libraries of type t (n_t >= 2); relative half-width w = t_{0.975, df} * s * sqrt(1 + 1/n_t),
    s = the pooled within-type relative SD (each library's A over its type mean, minus 1), pooled over all types after removing L's donor, df = sum (n_t - 1).
    (Variant B: s from type t alone, df = n_t - 1; reported too.)
  Inside = lo <= A_L <= hi. Expected about 95 % for a calibrated 95 % band.
SENSITIVITY per type t (calculated): the smallest held-error rise r flagged when the library sits exactly at its type's centre m_t. A_derived is held error
  over a prediction that does not depend on held error, so a rise r multiplies A by (1 + r): per-type r = w; pooled r = hi / m_t - 1 (and a fall: 1 - lo / m_t).
Outputs: per_type_band_loo_libraries.csv (one row per library), private/a8/bands_loo_types.csv (one row per type), private/a8/bands_loo_summary.json."""
import json, csv, os, numpy as np
from scipy.stats import t as T
import sys; IN = sys.argv[1] if len(sys.argv) > 1 else "per_type_band_loo_inputs.csv"; OUTD = sys.argv[2] if len(sys.argv) > 2 else "."
D = list(csv.DictReader(open(IN)))
meta = {r["gsm"]: r for r in D}
donor = lambda g: (meta[g].get("lab"), meta[g].get("age"), meta[g].get("sex")) if g in meta else ("?", g)
L = {r["gsm"]: r["cell"] for r in D}; TR = {r["gsm"]: {"A_derived": r["A_derived"]} for r in D}
REF = [(g, L[g], float(TR[g]["A_derived"])) for g in sorted(L) if g in TR and TR[g].get("A_derived") is not None]
print("libraries with a type and A_derived:", len(REF), "of", len(L), "| types:", len({c for _, c, _ in REF}))
rows = []
for g, c, A in REF:
    rest = [(gg, cc, a) for gg, cc, a in REF if donor(gg) != donor(g)]
    allA = np.array([a for _, _, a in rest]); plo, phi = np.percentile(allA, [2.5, 97.5])
    own = np.array([a for _, cc, a in rest if cc == c]); n = len(own)
    res, df = [], 0
    for cc in {x for _, x, _ in rest}:
        v = np.array([a for _, x, a in rest if x == cc])
        if len(v) >= 2: res += list(v / v.mean() - 1); df += len(v) - 1
    s = float(np.sqrt(np.sum(np.square(res)) / df))
    r = dict(gsm=g, cell=c, A=round(A, 5), donor="/".join(map(str, donor(g))), donor_libs_removed=len(REF) - len(rest), pooled_lo=round(plo, 4), pooled_hi=round(phi, 4),
             pooled_inside=bool(plo <= A <= phi), type_n=n)
    if n >= 2:
        m = float(np.median(own)); w = float(T.ppf(0.975, df) * s * np.sqrt(1 + 1 / n)); sB = float(np.std(own / own.mean(), ddof=1)); wB = float(T.ppf(0.975, n - 1) * sB * np.sqrt(1 + 1 / n))
        r.update(type_centre=round(m, 4), type_halfwidth=round(w, 4), type_lo=round(m * (1 - w), 4), type_hi=round(m * (1 + w), 4), type_inside=bool(m * (1 - w) <= A <= m * (1 + w)),
                 z_type=round((A / m - 1) / (w / T.ppf(0.975, df)), 2), typeB_halfwidth=round(wB, 4), typeB_inside=bool(m * (1 - wB) <= A <= m * (1 + wB)),
                 pooled_rise_needed=round(phi / m - 1, 4), pooled_fall_needed=round(1 - plo / m, 4), pooled_sd=round(s, 4), pooled_df=df)
    rows.append(r)
with open(OUTD + "/per_type_band_loo_libraries.csv", "w", newline="") as f:
    W = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)); W.writeheader(); W.writerows(rows)
S = [r for r in rows if "type_inside" in r]
types = {}
for r in S: types.setdefault(r["cell"], []).append(r)
TT = []
for c, v in sorted(types.items()):
    TT.append(dict(cell=c, libraries=len(v), centre_median=round(float(np.median([x["type_centre"] for x in v])), 4), type_halfwidth=round(float(np.median([x["type_halfwidth"] for x in v])), 4),
                   type_inside=sum(x["type_inside"] for x in v), pooled_inside=sum(x["pooled_inside"] for x in v),
                   rise_flagged_type=round(float(np.median([x["type_halfwidth"] for x in v])), 4), rise_flagged_pooled=round(float(np.median([x["pooled_rise_needed"] for x in v])), 4),
                   fall_flagged_pooled=round(float(np.median([x["pooled_fall_needed"] for x in v])), 4)))
with open(OUTD + "/per_type_band_loo_types.csv", "w", newline="") as f:
    W = csv.DictWriter(f, fieldnames=list(TT[0])); W.writeheader(); W.writerows(TT)
out = dict(libraries_with_A=len(REF), scored=len(S), not_scored_reason="type has fewer than 2 libraries once the library's donor stand-in is removed",
           not_scored=sorted({r["cell"] for r in rows if "type_inside" not in r}),
           per_type=dict(inside=sum(r["type_inside"] for r in S), share=round(np.mean([r["type_inside"] for r in S]), 3), halfwidth_median=round(float(np.median([r["type_halfwidth"] for r in S])), 4),
                         pooled_sd=round(float(np.median([r["pooled_sd"] for r in S])), 4), df=int(np.median([r["pooled_df"] for r in S]))),
           per_type_variantB=dict(inside=sum(r["typeB_inside"] for r in S), halfwidth_median=round(float(np.median([r["typeB_halfwidth"] for r in S])), 4)),
           pooled_loo=dict(inside=sum(r["pooled_inside"] for r in S), share=round(np.mean([r["pooled_inside"] for r in S]), 3), lo_median=round(float(np.median([r["pooled_lo"] for r in S])), 4),
                           hi_median=round(float(np.median([r["pooled_hi"] for r in S])), 4)),
           sensitivity=dict(per_type_rise_median=round(float(np.median([t["rise_flagged_type"] for t in TT])), 4),
                            pooled_rise_median=round(float(np.median([t["rise_flagged_pooled"] for t in TT])), 4),
                            pooled_rise_range=[round(min(t["rise_flagged_pooled"] for t in TT), 4), round(max(t["rise_flagged_pooled"] for t in TT), 4)],
                            types_where_pooled_catches_10pct=sum(t["rise_flagged_pooled"] <= 0.10 for t in TT), types_where_type_catches_10pct=sum(t["rise_flagged_type"] <= 0.10 for t in TT), types=len(TT)),
           outside_per_type=[(r["gsm"], r["cell"], r["A"], r["type_lo"], r["type_hi"], r["z_type"]) for r in S if not r["type_inside"]],
           outside_pooled=[(r["gsm"], r["cell"], r["A"]) for r in S if not r["pooled_inside"]],
           band_v2_in_sample_check=dict(lo=0.8744415, hi=1.178255125, inside=sum(0.8744415 <= r["A"] <= 1.178255125 for r in S)))
json.dump(out, open(OUTD + "/per_type_band_loo_summary.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
