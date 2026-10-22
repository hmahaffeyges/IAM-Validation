"""DEV-TISSUE-GLOBE-01, step 4 (2026-10-22; development, a picture, nothing scored; to-do step 28). The tumor pairs on the Globe Explorer:
the same pairs as run 1 (tissue_pairs_globe.py: the first 3 sealed pairs of DEV-TISSUE-COLON-01, GSE131013, and of DEV-TISSUE-BREAST-01,
TCGA-BRCA; both tests already scored, so these arrays are spent and nothing here can change their results) and its control (two different
people's normal colon). Each pair is drawn by the chain (report_globe_v4.build_pair, the globe's block rule, unchanged) from arrays calibrated
by chain Stage 1, and all seven go into one Explorer globe, one group per pair, written by the Explorer's own writer (globe_viewer.write_files).
Usage: python3 explorer_pairs_04.py OUTDIR   (writes OUTDIR/explorer/tumor-pairs.json, its pictures, explorer_pairs_04_summary.txt and a
.tar.gz of the explorer folder)"""
import os, sys, json, tarfile, runpy
H_ = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H_, "..", "..", "..")); CH = os.path.join(MP, "chain")
sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import report_globe_v4 as G, globe_viewer as V
out = sys.argv[1]; E = os.path.join(out, "explorer"); os.makedirs(E, exist_ok=True)
# the pairs, the downloads and Stage 1 exactly as run 1 defines them (its own code, read up to its job list)
src = open(os.path.join(H_, "tissue_pairs_globe.py"), encoding="utf-8").read(); src = src[:src.index("res = {}")]
ns = {"__file__": os.path.join(H_, "tissue_pairs_globe.py")}; sys.argv = [sys.argv[0], out]; exec(compile(src, "tissue_pairs_globe.py", "exec"), ns)
groups, L = [], []
for tag, title, fs, fr, rn in ns["jobs"]:
    try:
        r = G.build_pair(fs(), fr(), rn, None, title); s = r["spec"]
        for g in s["groups"]:
            g["name"] = title.replace(" against the same person's adjacent normal", "") + f" ({tag.split('_', 1)[1].replace('_', ' ')})" + ("" if len(s["groups"]) == 1 else ": " + g["name"])
            groups.append(g)
        L.append(f"{tag}: {r['blocks_changed']} of {r['blocks']} blocks changed, {r['sites_compared']:,} sites compared")
    except Exception as e: L.append(f"{tag}: not drawn ({type(e).__name__}: {e})"[:300])
    print(L[-1], flush=True)
about = ("<b>Development: not commissioned by our standards.</b> Each tumor read against the same person's own adjacent normal tissue, so no other "
         "person enters the comparison; the last globe is a control, two different people's normal colon, to show what a difference between "
         "people looks like. The change is fixed by the measurement (the globe's block rule) and its count never moves; the slider only fades "
         "the unchanged background. These pairs come from the colon and breast tissue tests, already scored, so these pictures cannot change "
         "their results. In breast the tumor lights far more than the two-person control; in colon two people's normal tissue already differs "
         "nearly as much, which is why the person's own normal is the reference that counts.")
spec = V.dedupe(dict(id="tumor-pairs", name="Tumors against the same person's own normal tissue (development)", groups=groups, about=about))
s = V.write_files(spec, E, "../art/explorer"); json.dump(s, open(os.path.join(E, "tumor-pairs.json"), "w"), indent=1)
L.append(f"globe tumor-pairs: {len(groups)} groups, {len(s['assets'])} pictures")
open(os.path.join(out, "explorer_pairs_04_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
with tarfile.open(os.path.join(out, "explorer_pairs_04.tar.gz"), "w:gz") as t: t.add(E, arcname="explorer")
