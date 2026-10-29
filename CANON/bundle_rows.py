"""One row per sample and reading from the one command's bundles (2026-10-25; author: the distance to the ceiling must be captured for every
run, as the A-score database is). Reads every bundle under DIR (run_sample.py --bundle files) and writes a table with, for each sample: the
whole-sample Met-A, each cell's Met-A, and each territory, with A and the ceiling fields the chain wrote (A_sat, distance_to_ceiling,
headroom_used, turnover_share). Nothing is computed here; every value is copied from a bundle. The table is committed with its record and
listed in CANON/ascore_sources.json, so the A-score database reads it.
Usage: python3 CANON/bundle_rows.py DIR OUT.csv [LABELS.csv with gsm and any of series, condition]"""
import sys, os, json, glob, csv
D, OUT = sys.argv[1], sys.argv[2]
LAB = {r["gsm"]: r for r in csv.DictReader(open(sys.argv[3]))} if len(sys.argv) > 3 else {}
F = ["gsm", "series", "condition", "specimen", "platform", "reading", "place", "share", "A", "A_sat", "distance_to_ceiling", "headroom_used", "turnover_share", "band_lo", "band_hi", "state", "flag", "chain_commit"]
rows = []
def add(g, B, reading, place, A, c, share=None):
    if not isinstance(c, dict) or c.get("distance_to_ceiling") is None: return
    l = LAB.get(g, {}); rows.append(dict(gsm=g, series=l.get("series", l.get("gse", l.get("lab", ""))), condition=l.get("condition", l.get("genotype", "")), specimen=B.get("specimen", ""),
        platform=B.get("array_type") or B.get("platform") or "", reading=reading, place=place, share=share, A=A,
        chain_commit=(B.get("versions") or {}).get("chain_commit", ""), **{k: c.get(k) for k in ("A_sat", "distance_to_ceiling", "headroom_used", "turnover_share")}))
for f in sorted(glob.glob(os.path.join(D, "**", "*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json", "_q2.json", "_timing.json", "_stageq.json")): continue
    try: B = json.load(open(f))
    except Exception: continue
    if not isinstance(B, dict) or "met_a_whole" not in B: continue
    g = os.path.basename(f).replace("_bundle.json", "").replace(".json", ""); m = B.get("met_a_whole") or {}
    add(g, B, "Met-A whole sample", "whole sample", m.get("A"), m.get("ceiling"))
    for k, v in (m.get("by_place") or {}).items(): add(g, B, "Met-A whole sample, by place", k, v.get("A"), v.get("ceiling"))
    for c in (B.get("met_a_each_cell") or {}).get("cells") or []: add(g, B, "Met-A each cell", c.get("cell"), c.get("A"), c.get("ceiling"), c.get("share"))
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): add(g, B, "territory", k, (v.get("ceiling") or {}).get("A", v.get("A")), v.get("ceiling"))
    # 2026-10-29 (author: every A score in the A-score database, with where Normal begins and ends): the readings without a ceiling block too,
    # each with the band its gauge used and its state, copied from the bundle
    def band(v):
        b = v.get("band") if isinstance(v, dict) else None
        return (b.get("lo"), b.get("hi")) if isinstance(b, dict) else (None, None)
    def add2(reading, place, v, lo=None, hi=None, share=None):
        if not isinstance(v, dict) or v.get("A") is None: return
        bl, bh = band(v); l = LAB.get(g, {})
        rows.append(dict(gsm=g, series=l.get("series", l.get("gse", l.get("lab", ""))), condition=l.get("condition", l.get("genotype", "")), specimen=B.get("specimen", ""),
                         platform=B.get("array_type") or B.get("platform") or "", reading=reading, place=place, share=share, A=v.get("A"),
                         band_lo=bl if lo is None else lo, band_hi=bh if hi is None else hi, state=str(v.get("state") or "")[:80], flag=str(v.get("flag") or v.get("label") or "")[:80], chain_commit=(B.get("versions") or {}).get("chain_commit", "")))
    add2("IAM-A", (B.get("iam_a") or {}).get("cell", ""), B.get("iam_a"))
    add2("coupling-A", (B.get("coupling_a") or {}).get("cell", ""), B.get("coupling_a"))
    add2("Met-A", (B.get("met_a") or {}).get("cell", ""), B.get("met_a"))
    add2("Met-A sorted cell (atlas)", ((B.get("development") or {}).get("metaa_atlas") or {}).get("cell", ""), (B.get("development") or {}).get("metaa_atlas"))
    w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is not None: add2("C-Score whole genome", "whole genome", dict(A=w["C"], state="within its limit" if w.get("within") else "above its limit"), hi=w.get("limit"))
    for c in (B.get("cscore_each_cell") or {}).get("cells") or []:
        if c.get("C") is not None: add2("C-Score each cell", c.get("cell"), dict(A=c["C"], state="within its limit" if c.get("within") else "above its limit"), hi=c.get("limit"))
with open(OUT, "w", newline="") as h: w = csv.DictWriter(h, fieldnames=F, lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"bundle_rows: {len({r['gsm'] for r in rows})} samples, {len(rows)} rows -> {OUT}")
