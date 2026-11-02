"""The globe and information sphere on sequencing, the outside healthy samples read both ways (bundles gl_GSM_seq.json against the IAM
Sequencing Atlas, gl_GSM_v21.json with --dev-globe-v21; every other setting the same). Bar: the median number of changed sites, and of
changed territories, no higher against the sequencing atlas than against v2.1. Recorded: the Fisher-Rao distance both ways, by cell.
Usage: python3 globe_outside_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, re, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "gl_GSM*_seq.json"), recursive=True)):
    g = re.search(r"GSM\d+", os.path.basename(f)).group(0); f2 = f.replace("_seq.json", "_v21.json")
    if not os.path.exists(f2): continue
    r = dict(gsm=g, specimen=json.load(open(f)).get("specimen"))
    for lab, ff in (("seq", f), ("v21", f2)):
        B = json.load(open(ff)); gl = B.get("globe") or {}; s = B.get("info_sphere") or {}
        r.update({f"sites_changed_{lab}": gl.get("sites_changed"), f"territories_changed_{lab}": s.get("territories_changed"), f"fr_rms_{lab}": s.get("fisher_rao_rms_rad"),
                  f"reference_{lab}": gl.get("reference"), f"includes_self_{lab}": bool(gl.get("reference_includes_this_sample"))})
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False); m = D.median(numeric_only=True)
b1 = m.sites_changed_seq <= m.sites_changed_v21; b2 = m.territories_changed_seq <= m.territories_changed_v21
L = [f"samples read both ways: {len(D)} (any flagged as an atlas input: {int(D.includes_self_seq.sum())})",
     f"median changed sites: sequencing atlas {m.sites_changed_seq:g}, v2.1 {m.sites_changed_v21:g}; median changed territories: {m.territories_changed_seq:g}, {m.territories_changed_v21:g}",
     f"median Fisher-Rao distance from the own healthy self: {m.fr_rms_seq:.5f}, {m.fr_rms_v21:.5f}"] + \
    [f"  {c}: n {len(x)}, changed sites {x.sites_changed_seq.median():g} vs {x.sites_changed_v21.median():g}, Fisher-Rao {x.fr_rms_seq.median():.5f} vs {x.fr_rms_v21.median():.5f}" for c, x in D.groupby("specimen")] + \
    [f"bar 1 changed sites no higher: {'met' if b1 else 'not met'}", f"bar 2 changed territories no higher: {'met' if b2 else 'not met'}", f"globe on sequencing: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
