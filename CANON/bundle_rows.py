"""One row per sample and reading from the one command's bundles (2026-10-25; author: the distance to the ceiling must be captured for every
run, as the A-score database is). Reads every bundle under DIR (run_sample.py --bundle files) and writes a table with, for each sample: the
whole-sample Met-A, each cell's Met-A, and each territory, with A and the ceiling fields the chain wrote (A_sat, distance_to_ceiling,
headroom_used, turnover_share). Nothing is computed here; every value is copied from a bundle. The table is committed with its record and
listed in CANON/ascore_sources.json, so the A-score database reads it.
Usage: python3 CANON/bundle_rows.py DIR OUT.csv [LABELS.csv with gsm and any of series, condition]"""
import sys, os, json, glob, csv
D, OUT = sys.argv[1], sys.argv[2]
LAB = {r["gsm"]: r for r in csv.DictReader(open(sys.argv[3]))} if len(sys.argv) > 3 else {}
F = ["gsm", "series", "condition", "specimen", "platform", "reading", "place", "share", "A", "A_sat", "distance_to_ceiling", "headroom_used", "turnover_share"]
rows = []
def add(g, B, reading, place, A, c, share=None):
    if not isinstance(c, dict) or c.get("distance_to_ceiling") is None: return
    l = LAB.get(g, {}); rows.append(dict(gsm=g, series=l.get("series", l.get("gse", l.get("lab", ""))), condition=l.get("condition", l.get("genotype", "")), specimen=B.get("specimen", ""),
        platform=B.get("array_type") or B.get("platform") or "", reading=reading, place=place, share=share, A=A,
        **{k: c.get(k) for k in ("A_sat", "distance_to_ceiling", "headroom_used", "turnover_share")}))
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
with open(OUT, "w", newline="") as h: w = csv.DictWriter(h, fieldnames=F, lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"bundle_rows: {len({r['gsm'] for r in rows})} samples, {len(rows)} rows -> {OUT}")
