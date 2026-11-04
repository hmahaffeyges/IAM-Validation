"""The A-score database (OTH-051): Biological_Physics/MethylPhys/ASCORES/<instrument>.csv and Biological_Physics/MethylPhys/ASCORES/README.md, generated from CANON/ascore_sources.json. Every row is
read from a committed per-sample file; nothing is typed by hand. Run by checkpoint.sh; checked_push.sh refuses drift.
Usage: python3 CANON/ascore_db.py [--check]"""
import os, sys, json, io, re
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "ASCORES")
def load(s):
    f = os.path.join(ROOT, s["file"])
    if s.get("dict_rows"): d = pd.DataFrame.from_dict(json.load(open(f, encoding="utf-8"))[s["dict_rows"]], orient="index").rename_axis(s["sample"]).reset_index()
    else: d = pd.DataFrame(json.load(open(f, encoding="utf-8"))[s["rows"]]) if s.get("rows") else pd.read_csv(f, low_memory=False)
    if s.get("query"): d = d.query(s["query"])
    d = d[pd.to_numeric(d[s["value"]], errors="coerce").notna()]
    smp = d[s["sample_cols"]].astype(str).agg("_".join, axis=1) if s.get("sample_cols") else d[s["sample"]].astype(str)
    def col(fixed, c, m):
        if c: v = d[c].astype(str); return v.map(lambda x: (m or {}).get(x, x))
        return pd.Series(fixed, index=d.index)
    return pd.DataFrame({"instrument": s["instrument"], "source": s["id"], "sample": smp.values,
        "specimen": col(s.get("specimen"), s.get("specimen_col"), s.get("specimen_map")).str.lower().values,
        "condition": col(s.get("condition"), s.get("condition_col"), s.get("condition_map")).values,
        "platform": col(s.get("platform"), s.get("platform_col"), None).values, "reading": s["kind"],
        "value": pd.to_numeric(d[s["value"]]).round(6).values, "status": s["status"], "record": s["record"],
        "line": json.load(open(f, encoding="utf-8"))[s["line"]["from_file"]] if s.get("line") else None})
LINES = {}
INSTR = {"IAM-A": "IAM-A", "coupling-A": "Coupling-A", "Met-A": "Met-A", "Met-A sorted cell (atlas)": "Met-A", "Met-A whole sample": "Met-A",
         "Met-A whole sample, by place": "Met-A", "Met-A each cell": "Met-A", "territory": "Territories", "C-Score whole genome": "C-Score",
         "C-Score each cell": "C-Score", "IAM-A in the mix": "IAM-A", "globe changed blocks": "Globe", "information sphere": "Information sphere"}
def load_rows(s):
    """2026-10-29 (author: every A score logged, with where Normal begins and ends): a rows table from CANON/bundle_rows.py, one row per sample
    and reading, with the band each gauge used and the reading's state, all copied from the one command's bundles."""
    d = pd.read_csv(os.path.join(ROOT, s["file"]), low_memory=False); d = d[pd.to_numeric(d["A"], errors="coerce").notna() & d.reading.isin(list(INSTR))]
    for c in ("band_lo", "band_hi", "state", "flag", "chain_commit", "cancer_type", "stage"):
        if c not in d: d[c] = None
    dev = d.flag.fillna("").str.contains("DEVELOPMENT|not commissioned", case=False)
    return pd.DataFrame({"instrument": d.reading.map(INSTR).values, "source": s["id"], "sample": d.gsm.astype(str).values,
        "specimen": d.specimen.fillna("").astype(str).str.lower().values, "condition": d.condition.fillna(s.get("condition", "")).astype(str).values,
        "platform": d.platform.fillna("").astype(str).str.replace(r"^sequencing.*", "sequencing", regex=True).values, "reading": (d.reading + " | " + d.place.fillna("").astype(str)).values,
        "value": pd.to_numeric(d.A).round(6).values, "status": [("development (flagged in the bundle)" if x else s["status"]) for x in dev],
        "record": s["record"], "line": None, "band_lo": pd.to_numeric(d.band_lo, errors="coerce").values, "band_hi": pd.to_numeric(d.band_hi, errors="coerce").values,
        "state": d.state.fillna("").astype(str).values, "chain_commit": d.chain_commit.fillna("").astype(str).values,
        "cancer_type": d.cancer_type.fillna(s.get("cancer_type", "")).astype(str).values, "stage": d.stage.fillna(s.get("stage", "")).astype(str).values})
def bands(T):
    """Where Normal begins and ends, for each instrument, reading, cell and platform: the band the gauge used (from the bundles), and the spread of
    the healthy readings themselves (2.5th to 97.5th percentile; at least 3 healthy samples), so a band can be checked against healthy."""
    R = []; T = T.copy(); sp = T.reading.astype(str).str.contains(r" \| ")
    T["cell"] = T.specimen.astype(str).str.replace(r"^(sorted|isolated|purified) ", "", regex=True); T.loc[sp, "cell"] = T.reading[sp].astype(str).str.split(" | ", regex=False).str[1]
    T["reading"] = T.reading.astype(str).str.split(" | ", regex=False).str[0]
    T["chain_commit"] = T["chain_commit"].fillna("") if "chain_commit" in T else ""   # readings of different chain versions are never pooled
    for (ins, rd, cl, pl, cc), g in T.groupby(["instrument", "reading", "cell", "platform", "chain_commit"], sort=True):
        h = g[g.condition.astype(str).str.lower() == "healthy"].value   # healthy-labeled samples only (a control line or cord blood is not the healthy reference)
        lo = g.band_lo.dropna() if "band_lo" in g else pd.Series(dtype=float); hi = g.band_hi.dropna() if "band_hi" in g else pd.Series(dtype=float)
        ln = (LINES or {}).get(rd) or {}   # a commissioned line for this reading (ascore_sources.json "lines")
        if not len(lo) and "lo" in ln: lo = pd.Series([ln["lo"]])
        if not len(hi) and "hi" in ln: hi = pd.Series([ln["hi"]])
        R.append(dict(instrument=ins, reading=rd, cell=cl, platform=pl, chain_commit=cc, gauge_band_lo=round(lo.median(), 4) if len(lo) else None, gauge_band_hi=round(hi.median(), 4) if len(hi) else None,
                      healthy_samples=int(len(h)), healthy_lo=round(h.quantile(.025), 4) if len(h) >= 3 else None, healthy_median=round(h.median(), 4) if len(h) else None,
                      healthy_hi=round(h.quantile(.975), 4) if len(h) >= 3 else None, all_samples=int(len(g)), status="; ".join(sorted(set(g.status.astype(str))))[:120]))
    return pd.DataFrame(R)
GA_BLOCK = re.compile(r"^== (?P<cmp>.+?), (?P<reading>[a-z ]+) \((?P<use>scored|recorded)\)\n.*?\n\s+(?P<g1>\S+): tared mean (?P<m1>[\d.]+) \(n (?P<n1>\d+)\)\s+(?P<g2>\S+): tared mean (?P<m2>[\d.]+) \(n (?P<n2>\d+)\)\n\s+difference (?P<d>[+-]?[\d.]+) %, (?P<t>[+-]?[\d.]+) standard errors -> (?P<verdict>[^\n]+)", re.M)
def groupa(S):
    """2026-11-04 (author: every score of every instrument in the A-score document): Group-A's scores. Group-A reads a group, not a sample, so
    its table has one row per committed comparison: the two groups' tared means, the difference and its standard errors, copied from the
    committed output (the printed Group-A blocks, or the commissioning JSON's bars)."""
    R = []
    for s in S.get("group_a", []):
        f = os.path.join(ROOT, s["file"]); base = dict(source=s["id"], specimen=s.get("specimen", ""), platform=s.get("platform", ""), status=s["status"], record=s["record"])
        if s.get("json_bars"):
            d = json.load(open(f, encoding="utf-8"))
            for b, lab in s["json_bars"].items():
                x = d.get(b) or {}
                if "mean1" not in x: continue
                R.append(dict(base, comparison=lab, reading=s.get("reading", ""), group1=s.get("groups", {}).get(b, ["1", "2"])[0], n1=x["n1"], mean1=x["mean1"],
                              group2=s.get("groups", {}).get(b, ["1", "2"])[1], n2=x["n2"], mean2=x["mean2"], difference_pct=round(100 * x["rel_difference"], 2),
                              t=x["t"], verdict="met" if x.get("met") else "not met", use="scored"))
        else:
            for m in GA_BLOCK.finditer(open(f, encoding="utf-8").read()):
                R.append(dict(base, comparison=m["cmp"], reading=m["reading"], group1=m["g1"], n1=int(m["n1"]), mean1=float(m["m1"]), group2=m["g2"], n2=int(m["n2"]),
                              mean2=float(m["m2"]), difference_pct=float(m["d"]), t=float(m["t"]), verdict=m["verdict"].strip(), use=m["use"]))
        for nc in s.get("no_call", []): R.append(dict(base, comparison=nc["comparison"], reading=nc.get("reading", ""), verdict="no call: " + nc["why"], use="recorded"))
    return pd.DataFrame(R, columns=["source", "comparison", "reading", "group1", "n1", "mean1", "group2", "n2", "mean2", "difference_pct", "t", "verdict", "use",
                                    "specimen", "platform", "status", "record"])

def build():
    S = json.load(open(os.path.join(ROOT, "CANON", "ascore_sources.json"), encoding="utf-8")); global LINES; LINES = S.get("lines", {})
    T = pd.concat([x.astype({'line': float}) for x in ((load_rows(s) if s.get("rows_table") else load(s)) for s in S["sources"])], ignore_index=True); files = {}
    b = io.StringIO(); bands(T).to_csv(b, index=False, lineterminator="\n"); files["Bands.csv"] = b.getvalue()
    for ins, g in T.groupby("instrument", sort=False):
        b = io.StringIO(); g.to_csv(b, index=False, lineterminator="\n"); files[re.sub(r"[^A-Za-z0-9]+", "_", ins) + ".csv"] = b.getvalue()
    L = ["# The A-score database", "", "_Generated from `CANON/ascore_sources.json` by `CANON/ascore_db.py`; do not edit by hand._", "",
         "One table per instrument, one row per sample reading, each read from a committed file and linked to its record. It is how we learn",
         "where healthy samples and tumors sit on each instrument's gauge, and whether a line of concern shows before the tumor line",
         "(manifest PHY-069). Readings of different kinds (an untared A, a tared A, a tumor-over-own-normal ratio, a held error) are kept",
         "apart and are never pooled.", "",
         "**Past the healthy line** counts the samples beyond the healthy line of the reading they were made with: Met-A's Normal band (0.95 to",
         "1.05) and the C-Score's upper limit on the chain's tare (1.152) from their commissioning, and, for a tumor read against the same",
         "patient's normal tissue, the line its sealed test set from pairs of normal tissue, where chance alone puts about 5 % of tumors past it.",
         "Healthy rows show how rarely a healthy sample crosses its line; tumor rows show how often a tumor does. A line is never applied to a",
         "reading made another way (an earlier tare), so those rows show none.", "",
         "| instrument | reading | specimen | condition | samples | median | middle 90 percent | past the healthy line | file |", "|---|---|---|---|---|---|---|---|---|"]
    T["cancer_type"] = T["cancer_type"].fillna("") if "cancer_type" in T else ""   # 2026-11-03: each kind of cancer kept apart, for our own look; nothing declared
    for (ins, rd, sp, cd, ct), g in T.groupby(["instrument", "reading", "specimen", "condition", "cancer_type"], sort=False):
        cd = cd + (f" ({ct})" if ct else "")
        v = g.value; f = re.sub(r"[^A-Za-z0-9]+", "_", ins) + ".csv"
        ln = S.get("lines", {}).get(rd)
        if g.line.notna().any(): k = int((v > g.line).sum()); past = f"{k} of {len(g)} ({100 * k / len(g):.0f} %) above {g.line.iloc[0]:.3f}"
        elif ln: k = int(((v < ln.get("lo", -1e9)) | (v > ln.get("hi", 1e9))).sum()); past = f"{k} of {len(g)} ({100 * k / len(g):.0f} %)"
        else: past = "no line set for this reading"
        L.append(f"| {ins} | {rd} | {sp} | {cd} | {len(g)} | {v.median():.5g} | {v.quantile(.05):.5g} to {v.quantile(.95):.5g} | {past} | [{f}]({f}) |")
    L += ["", f"{len(T)} readings from {T.source.nunique()} sources. Status and record for every row are in the tables.", "",
          "**Where Normal begins and ends:** [Bands.csv](Bands.csv) lists, for each instrument, reading, cell and platform, the band the gauge drew",
          "(copied from the bundles) beside the spread of the healthy readings themselves (2.5th to 97.5th percentile, with how many healthy samples).", ""]
    G = groupa(S)
    if len(G):
        b = io.StringIO(); G.to_csv(b, index=False, lineterminator="\n"); files["Group_A.csv"] = b.getvalue()
        L += ["**Group-A** reads a group, not a sample: [Group_A.csv](Group_A.csv) holds one row per committed comparison (the two groups' tared means,",
              "the difference and its standard errors).", "", "| source | comparison | reading | groups (n) | difference | standard errors | verdict | specimen |", "|---|---|---|---|---|---|---|---|"]
        for _, r in G.iterrows():
            gr = f"{r.group1} ({r.n1:.0f}) vs {r.group2} ({r.n2:.0f})" if pd.notna(r.n1) else ""
            df = f"{r.difference_pct:+.2f} %" if pd.notna(r.difference_pct) else ""; tt = f"{r.t:.2f}" if pd.notna(r.t) else ""
            L.append(f"| {r.source} | {r.comparison} | {r.reading} | {gr} | {df} | {tt} | {r.verdict} | {r.specimen} |")
        L.append("")
    files["README.md"] = "\n".join(L); return files
if __name__ == "__main__":
    F = build()
    if "--check" in sys.argv:
        bad = [k for k, v in F.items() if not os.path.exists(os.path.join(OUT, k)) or open(os.path.join(OUT, k), encoding="utf-8").read() != v]
        print("ascore_db: " + ("ok" if not bad else "REFUSED: differs: " + ", ".join(bad) + " (run python3 CANON/ascore_db.py)")); sys.exit(1 if bad else 0)
    os.makedirs(OUT, exist_ok=True)
    for k, v in F.items(): open(os.path.join(OUT, k), "w", encoding="utf-8").write(v)
    print("ascore_db: wrote", ", ".join(F))
