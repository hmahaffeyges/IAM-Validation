"""stage_groupa.py - Group-A: has a whole group moved? (commissioned 2026-10-20, doors/COMMISSIONING_NOTE_GROUPA.md)

The instrument's job: two groups of healthy people read as no different after the tare, and a group shift planted into healthy arrays is found.

What it does. One person's whole-blood Met-A is too noisy to call a small change (healthy people differ by about 2 % after the tare), but the
average of a group is not. Group-A takes each person's Met-A reading from the one command (run_sample.py, unchanged), tares each reading
against healthy arrays run the same way (same slide when it holds at least 3 healthy references, else the same series; the C-Score band
test's tare, word for word), and compares the two groups' tared means (Welch: difference, standard error, t). Nothing is fitted.

  groups differ        |t| >= 2
  no difference found  |t| <  2, with the smallest difference this design could have found printed (2 x the standard error)

Groups must be run in the SAME series (same laboratory, same run): across laboratories the untared readings differ by several percent
(DEV-GROUP-01). Commissioned for whole blood, EPIC v1, two groups within one series; anything else is read and labeled preliminary.

Usage:
  python3 stage_groupa.py collect BUNDLE_DIR SAMPLES.csv OUT.csv [READING]   one row per person from run_sample.py bundles (<id>_bundle.json);
                                                                     READING met_a (arrays, default) or, on sequencing, sorted, whole, islands;
                                                                     SAMPLES.csv needs id, slide, and your group column (optional: healthy_reference)
  python3 stage_groupa.py compare READINGS.csv GROUP_COLUMN A B      tare, compare group A with group B, print the call
  python3 stage_groupa.py collect-cells BUNDLE_DIR SAMPLES.csv OUT.csv        development: one row per person and cell found
  python3 stage_groupa.py compare-cells CELLS.csv GROUP_COLUMN A B           development: per cell, against the atlas alone, no tare
  python3 stage_groupa.py compare-cells-sites CELLS.csv VALUES_DIR GROUP_COLUMN A B [--plant CELL:LOSS:HALF_COLUMN]
                                                                     development: which cell changed, site by site (saved values, run_sample.py --keep-values)
"""
import os, sys, json, numpy as np, pandas as pd
GROUPA_COMMISSIONING = {"status": "COMMISSIONED", "date": "2026-10-20", "reading": "Group-A", "platform": "EPIC v1", "specimens": ["whole blood"],
    "scope": "two groups within one series, each person read by run_sample.py and tared against same-run healthy references",
    "rule": "groups differ when |t| >= 2", "record": "doors/COMMISSIONING_NOTE_GROUPA.md",
    "evidence": "DEV-GROUPA-COMMISSION-01, 278 healthy controls of GSE318173: women vs men 0.36 % apart (1.2 SE); a planted 1 % loss found, +2.00 % at 7.3 SE"}
PRELIMINARY = "This score is preliminary until Group-A is officially commissioned on {what}."

def tare(r, healthy):
    import conductor_v3 as C
    hh = r[healthy & r.A.notna()]; out = []
    for _, x in r.iterrows():
        if pd.isna(x.A): out.append((None, None, 0)); continue
        res = None
        for scope in ("slide", "series"):
            ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)]
            if len(ref) >= 3: res = (float(x.A / ref.A.median()), scope, len(ref)); break
        out.append(res or (None, None, 0))
    r["A_rel"], r["ref_scope"], r["n_refs"] = zip(*out)
    cr = []
    for _, x in r.iterrows():
        v = None
        if pd.notna(x.C):
            for scope in ("slide", "series"):
                ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)].C.dropna()
                if len(ref) >= 3: v = float(x.C / ref.median()); break
        cr.append(v)
    r["C_rel"] = cr
    r["state"] = r.A_rel.map(lambda a: None if pd.isna(a) else ("Normal" if 0.95 <= a <= 1.05 else ("above Normal" if a > 1.05 else "below Normal")))
    return r

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    d = float(a.mean() - b.mean()); return dict(n1=len(a), n2=len(b), mean1=round(float(a.mean()), 5), mean2=round(float(b.mean()), 5), difference=round(d, 5),
                                                 rel_difference=round(d / float(np.concatenate([a, b]).mean()), 5), se=round(se, 5), t=round(d / se, 3))

def compare(rows, group_col, a, b, healthy=None, specimen="whole blood", platform="EPIC v1"):
    """rows: one row per person with gsm, A (untared Met-A), C (optional), slide, group_col. healthy: boolean Series of the arrays the tare may
    use as references (default: every array). Returns the comparison, the call and, when needed, the preliminary label."""
    r = rows.copy()
    if "C" not in r: r["C"] = np.nan
    r = tare(r, pd.Series(True, index=r.index) if healthy is None else healthy.reindex(r.index).fillna(False).astype(bool))
    x, y = r[r[group_col] == a].A_rel.dropna(), r[r[group_col] == b].A_rel.dropna()
    why = []
    if len(x) < 3 or len(y) < 3: why.append(f"group {a!r} has {len(x)} tared readings and group {b!r} has {len(y)} (at least 3 each required)")
    untared = int(r.A.notna().sum() - r.A_rel.notna().sum())
    if why: return dict(call=None, reason="; ".join(why), untared=untared, rows=r)
    s = welch(x, y); s["call"] = "groups differ" if abs(s["t"]) >= 2 else "no difference found"
    s["smallest_findable_difference"] = round(2 * s["se"], 5); s["untared"] = untared
    s["ref_scope"] = r.ref_scope.value_counts().to_dict()
    if specimen != "whole blood" or platform != "EPIC v1":
        s["commissioning"] = PRELIMINARY.format(what=f"{specimen}, {platform}")
    s["rows"] = r; return s

READINGS = {   # 2026-11-02 (sequencing atlas checklist B9, development): which reading of the bundle a group is compared on
    "met_a": lambda b: (b.get("met_a") or {}, b.get("met_a_cscore") or {}),   # arrays (commissioned): whole-blood Met-A and its C-Score
    "sorted": lambda b: ((b.get("development") or {}).get("metaa_atlas") or {}, ((b.get("cscore_each_cell") or {}).get("whole_genome") or {})),
    "whole": lambda b: (b.get("met_a_whole") or {}, ((b.get("cscore_each_cell") or {}).get("whole_genome") or {})),
    "islands": lambda b: ((((b.get("territories") or {}).get("sets") or {}).get("islands")) or {}, ((b.get("cscore_each_cell") or {}).get("whole_genome") or {}))}

def _find(bundle_dir, i):
    p = os.path.join(bundle_dir, f"{i}_bundle.json")
    if os.path.exists(p): return p
    p = os.path.join(bundle_dir, f"{i}.json")   # 2026-11-10: run_series.py's pass-1 bundles are named GSM.json
    if os.path.exists(p): return p
    import glob as _g   # 2026-11-02: sequencing runs land each bundle in its own folder
    f = sorted(_g.glob(os.path.join(bundle_dir, "**", f"{i}_bundle.json"), recursive=True)); return f[-1] if f else None

def collect(bundle_dir, samples_csv, out_csv, reading="met_a"):
    """reading: met_a (arrays, the commissioned reading) or, on sequencing (development), sorted (sorted-cell Met-A against the sequencing
    atlas), whole (whole-sample Met-A) or islands (the islands territory). A sample list without a slide column is tared within its series."""
    S = pd.read_csv(samples_csv); L = []
    for _, s in S.iterrows():
        p = _find(bundle_dir, s["id"]); b = json.load(open(p)) if p else {}
        m, c = READINGS[reading](b); C = c.get("C")
        row = dict(s, gsm=s["id"], A=m.get("A"), C=C, reading=reading, withheld=None if m.get("A") is not None else (m.get("reason") or ("no bundle" if not b else "no reason recorded")))
        if "slide" not in row or pd.isna(row.get("slide")): row["slide"] = "series"
        L.append(row)
    pd.DataFrame(L).to_csv(out_csv, index=False); return len(L)


def collect_cells(bundle_dir, samples_csv, out_csv):
    """2026-11-10 (development; re-commissioning against IAM-Atlas v2.1 alone): one row per person and per cell the IAM-Deconvolver found
    (the bundle's cells_found rows), with that cell's share and its Met-A against the atlas (each cell divided by its own healthy prediction).
    No tare: healthy comes only from the atlas, and two groups of one series share the series' own line, which cancels in their difference."""
    S = pd.read_csv(samples_csv); L = []
    for _, s in S.iterrows():
        p = _find(bundle_dir, s["id"]); b = json.load(open(p)) if p else {}
        cf = (b.get("cells_found") or {}).get("cells") or []
        if not cf: L.append(dict(s, gsm=s["id"], cell=None, share=None, A=None, C=None, withheld="no bundle" if not b else "no cells found")); continue
        for c in cf:
            L.append(dict(s, gsm=s["id"], cell=c.get("cell"), share=c.get("share"), A=c.get("met_a"), C=c.get("c_score"),
                          withheld=None if c.get("met_a") is not None else (c.get("met_a_note") or "no reading")))
    pd.DataFrame(L).to_csv(out_csv, index=False); return len(L)

def compare_cells(rows, group_col, a, b, min_share=0.05, min_n=3):
    """2026-11-10 (development): Group-A per cell found, against the atlas alone (no tare). For each cell read in at least min_n people of
    each group at a share of at least min_share, the two groups' Met-A means are compared (welch). Returns one result per cell."""
    r = rows[rows.A.notna() & (rows.share.fillna(0) >= min_share)]; out = {}
    for cell, g in r.groupby("cell"):
        x, y = g[g[group_col] == a].A, g[g[group_col] == b].A
        if len(x) < min_n or len(y) < min_n: continue
        s = welch(x, y); s["call"] = "groups differ" if abs(s["t"]) >= 2 else "no difference found"
        s["smallest_findable_difference"] = round(2 * s["se"], 5); s["share_median"] = round(float(g.share.median()), 4); out[cell] = s
    return out


WEIGHTED = True   # compare_cells_sites: each site weighted by its own measured noise in the series (False: equal weights, the first reading)
SITES_SHARE_MIN, SITES_PEOPLE_MIN, SITES_BOOT = 0.03, 0.5, 200   # compare_cells_sites: a cell enters when it is at least 3 % of the sample in at least half the people; 200 people resamples (stated)

def compare_cells_sites(rows, values_dir, group_col, a, b, plant=None, seed=20261110):
    """2026-11-10 (development; Group-A per cell against IAM-Atlas v2.1 alone, the site-by-site comparison). Every cell's row in one array
    blends that cell with its neighbors in proportion to their DNA (the share synthetic), so a change in one cell moves every row; this reads
    which cell changed. At every site of the cells found (each at least SITES_SHARE_MIN of the sample in at least SITES_PEOPLE_MIN of the
    people), each person's measured value is x_i = sum_k F_ik a_k + g_i sum_k F_ik d_k, with F_ik the shares the IAM-Deconvolver found (scaled
    to sum 1 over those cells), g_i the person's group (centered), a_k each cell's value in this series (shared by both groups, as the series'
    array line is) and d_k each cell's difference between the groups: the established cell-type-specific comparison, solved by least squares
    at each site, as Welch's mean difference is an estimate; nothing about healthy is estimated, and the atlas enters only as each cell's
    template. Each cell's difference is read against its own pattern at its own identity sites: departure = sum d (0.5 - mu) / sum (0.5 - mu)^2
    over the cell's identity sites (a 1 % loss of that cell's pattern reads +1 %). Standard errors from SITES_BOOT resamples of people within
    each group (seeded). Sites: those every person measured. plant=(cell, loss, half_col): TESTS ONLY, the people with half_col == 'planted'
    get stage_metaa_atlas.plant(values, cell, loss x that person's share of the cell), as run_series.py --plant neutrophil_atlas plants."""
    import stage_metaa_atlas as SA
    r = rows[rows.cell.notna()].copy(); P = r.pivot_table(index="gsm", columns="cell", values="share", aggfunc="first").fillna(0.0)
    cells = [c for c in P.columns if (P[c] >= SITES_SHARE_MIN).mean() >= SITES_PEOPLE_MIN]
    grp = r.groupby("gsm")[group_col].first().astype(str); keep = [g for g in P.index if grp.get(g) in (str(a), str(b))]
    sites = sorted(set().union(*[set(SA.identity_sites(c)) for c in cells])); X = {}
    for g in keep:
        f = os.path.join(values_dir, g + "_betas.parquet")
        if not os.path.exists(f): continue
        v = pd.read_parquet(f); v = (v.set_index(v.columns[0]).iloc[:, 0] if v.shape[1] > 1 else v.iloc[:, 0]).astype("float64")
        if plant and str(r[r.gsm == g][plant[2]].iloc[0]) == "planted": v = SA.plant(v, plant[0], loss=plant[1] * float(P.loc[g, plant[0]]))
        X[g] = v.reindex(sites)
    if len(X) < 6: return dict(reason=f"only {len(X)} people with saved values")
    Xm = pd.DataFrame(X).T.dropna(axis=1); ppl = list(Xm.index); F = P.loc[ppl, cells]; F = F.div(F.sum(1), axis=0).to_numpy()
    gA = (grp.reindex(ppl) == str(a)).to_numpy().astype(float); Xv = Xm.to_numpy(); K = len(cells)
    T = {c: (0.5 - SA._col("mu", c).reindex(Xm.columns)).where(Xm.columns.isin(SA.identity_sites(c))).to_numpy() for c in cells}
    # each site's own noise in this series (2026-11-10, the matched filter weighted by each pixel's measured noise, as the CMB maps are): the
    # residual variance across the people of the full-data fit at that site, fixed for the resamples; the sites' weights are 1 / variance
    D0 = np.column_stack([F, (gA - gA.mean())[:, None] * F]); res0 = Xv - D0 @ np.linalg.lstsq(D0, Xv, rcond=None)[0]
    W = 1.0 / np.maximum(res0.var(0, ddof=D0.shape[1]) if len(ppl) > D0.shape[1] + 1 else res0.var(0), 1e-8)
    if not WEIGHTED: W = np.ones_like(W)
    def est(ix):
        D = np.column_stack([F[ix], (gA[ix] - gA[ix].mean())[:, None] * F[ix]]); B = np.linalg.lstsq(D, Xv[ix], rcond=None)[0][K:]
        out = []
        for k, c in enumerate(cells):
            t = T[c]; ok = np.isfinite(t); out.append(float((B[k][ok] * W[ok]) @ t[ok] / ((W[ok] * t[ok]) @ t[ok])) if ok.any() else np.nan)
        return np.array(out)
    i0 = np.arange(len(ppl)); d0 = est(i0); rng = np.random.default_rng(seed); ia, ib = i0[gA == 1], i0[gA == 0]
    bs = np.array([est(np.r_[rng.choice(ia, len(ia)), rng.choice(ib, len(ib))]) for _ in range(SITES_BOOT)]); se = bs.std(0)
    res = {}
    for k, c in enumerate(cells):
        t = d0[k] / se[k] if se[k] > 0 else np.nan
        res[c] = dict(departure_pct=round(100 * d0[k], 4), se_pct=round(100 * se[k], 4), t=round(float(t), 2), call="groups differ" if abs(t) >= 2 else "no difference found",
                      smallest_findable_pct=round(200 * se[k], 4), share_median=round(float(P.loc[ppl, c].median()), 4), identity_sites=int(np.isfinite(T[c]).sum()))
    return dict(flag="DEVELOPMENT - Group-A per cell, site by site, against IAM-Atlas v2.1 alone", people=len(ppl), n_a=int(gA.sum()), n_b=int((1 - gA).sum()),
                sites=int(Xm.shape[1]), weighting="each site by 1 / its residual variance across the people" if WEIGHTED else "equal", cells=res, planted=(dict(cell=plant[0], loss=plant[1], half=plant[2]) if plant else None))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) >= 5 and sys.argv[1] == "collect":
        rd = sys.argv[5] if len(sys.argv) >= 6 else "met_a"
        print(f"collected {collect(sys.argv[2], sys.argv[3], sys.argv[4], rd)} people into {sys.argv[4]} ({rd})")
    elif len(sys.argv) >= 5 and sys.argv[1] == "compare":
        R = pd.read_csv(sys.argv[2]); g, a, b = sys.argv[3], sys.argv[4], sys.argv[5]
        for col in (g,): R[col] = R[col].astype(str)
        h = R["healthy_reference"].astype(bool) if "healthy_reference" in R else None
        rd = str(R["reading"].dropna().iloc[0]) if "reading" in R and R["reading"].notna().any() else "met_a"
        o = compare(R, g, a, b, h, **({} if rd == "met_a" else dict(specimen=f"sorted cells ({rd} reading)", platform="whole-genome bisulfite sequencing")))
        if o.get("call") is None: print("Group-A: no call -", o["reason"]); sys.exit(2)
        print(f"Group-A ({GROUPA_COMMISSIONING['status']} {GROUPA_COMMISSIONING['date']}, {GROUPA_COMMISSIONING['scope']})\n"
              f"  {a}: tared mean {o['mean1']} (n {o['n1']})   {b}: tared mean {o['mean2']} (n {o['n2']})\n"
              f"  difference {100*o['rel_difference']:+.2f} %, {o['t']} standard errors -> {o['call']}\n"
              f"  smallest difference this design could have found: {100*o['smallest_findable_difference']:.2f} % (2 x the standard error)\n"
              f"  tare references: {o['ref_scope']}; readings with no tare: {o['untared']}" + (f"\n  {o['commissioning']}" if o.get("commissioning") else ""))
    elif len(sys.argv) >= 5 and sys.argv[1] == "collect-cells":
        print(f"collected {collect_cells(sys.argv[2], sys.argv[3], sys.argv[4])} cell rows into {sys.argv[4]} (cells_found, against the atlas, no tare)")
    elif len(sys.argv) >= 6 and sys.argv[1] == "compare-cells":
        R = pd.read_csv(sys.argv[2]); R[sys.argv[3]] = R[sys.argv[3]].astype(str)
        for cell, o in compare_cells(R, sys.argv[3], sys.argv[4], sys.argv[5]).items():
            print(f"Group-A (development, per cell, against IAM-Atlas v2.1 alone) {cell} (median share {o['share_median']}): {sys.argv[4]} {o['mean1']} (n {o['n1']}), "
                  f"{sys.argv[5]} {o['mean2']} (n {o['n2']}): {100*o['rel_difference']:+.2f} %, {o['t']} SE -> {o['call']}; smallest findable {100*o['smallest_findable_difference']:.2f} %")
    elif len(sys.argv) >= 7 and sys.argv[1] == "compare-cells-sites":
        R = pd.read_csv(sys.argv[2]); pl = None
        if len(sys.argv) >= 9 and sys.argv[7] == "--plant": c_, l_, h_ = sys.argv[8].split(":"); pl = (c_, float(l_), h_)
        o = compare_cells_sites(R, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], plant=pl)
        print(json.dumps(o, indent=1))
    else:
        print(__doc__); sys.exit(1)
