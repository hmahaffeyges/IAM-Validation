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
def build():
    S = json.load(open(os.path.join(ROOT, "CANON", "ascore_sources.json"), encoding="utf-8"))
    T = pd.concat([x.astype({'line': float}) for x in (load(s) for s in S["sources"])], ignore_index=True); files = {}
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
    for (ins, rd, sp, cd), g in T.groupby(["instrument", "reading", "specimen", "condition"], sort=False):
        v = g.value; f = re.sub(r"[^A-Za-z0-9]+", "_", ins) + ".csv"
        ln = S.get("lines", {}).get(rd)
        if g.line.notna().any(): k = int((v > g.line).sum()); past = f"{k} of {len(g)} ({100 * k / len(g):.0f} %) above {g.line.iloc[0]:.3f}"
        elif ln: k = int(((v < ln.get("lo", -1e9)) | (v > ln.get("hi", 1e9))).sum()); past = f"{k} of {len(g)} ({100 * k / len(g):.0f} %)"
        else: past = "no line set for this reading"
        L.append(f"| {ins} | {rd} | {sp} | {cd} | {len(g)} | {v.median():.4g} | {v.quantile(.05):.4g} to {v.quantile(.95):.4g} | {past} | [{f}]({f}) |")
    L += ["", f"{len(T)} readings from {T.source.nunique()} sources. Status and record for every row are in the tables.", ""]
    files["README.md"] = "\n".join(L); return files
if __name__ == "__main__":
    F = build()
    if "--check" in sys.argv:
        bad = [k for k, v in F.items() if not os.path.exists(os.path.join(OUT, k)) or open(os.path.join(OUT, k), encoding="utf-8").read() != v]
        print("ascore_db: " + ("ok" if not bad else "REFUSED: differs: " + ", ".join(bad) + " (run python3 CANON/ascore_db.py)")); sys.exit(1 if bad else 0)
    os.makedirs(OUT, exist_ok=True)
    for k, v in F.items(): open(os.path.join(OUT, k), "w", encoding="utf-8").write(v)
    print("ascore_db: wrote", ", ".join(F))
