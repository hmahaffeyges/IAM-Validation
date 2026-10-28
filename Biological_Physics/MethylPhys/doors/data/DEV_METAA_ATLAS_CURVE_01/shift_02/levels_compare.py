"""DEV-METAA-ATLAS-CURVE-01, the shift's cause (written before the rerun). For each sequencing input read by the one command: the sample's and
the atlas column's median level at the column's methylated and unmethylated identity sites (chain field levels_at_identity). Usage:
python3 levels_compare.py BUNDLE_DIR OUT.txt"""
import sys, os, json, glob, csv, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); S = list(csv.DictReader(open(os.path.join(H, "..", "shift_01", "seq_inputs.csv"))))
L = ["cell | sample | Met-A | methylated sites: sample, atlas | unmethylated sites: sample, atlas"]; up = []; dn = []
for r in S:
    f = glob.glob(os.path.join(sys.argv[1], "**", r["gsm"] + "_bundle.json"), recursive=True)
    if not f: L.append(f"{r['cell']} | {r['gsm']} | missing"); continue
    m = (json.load(open(f[0])).get("development") or {}).get("metaa_atlas") or {}; v = m.get("levels_at_identity") or {}
    if v.get("sample_median_methylated") is None: L.append(f"{r['cell']} | {r['gsm']} | {m.get('A')} | not recorded"); continue
    up.append(v["sample_median_methylated"] - v["atlas_median_methylated"]); dn.append(v["sample_median_unmethylated"] - v["atlas_median_unmethylated"])
    L.append(f"{r['cell']} | {r['gsm']} | {m.get('A')} | {v['sample_median_methylated']}, {v['atlas_median_methylated']} | {v['sample_median_unmethylated']}, {v['atlas_median_unmethylated']}")
if up:
    L += ["", f"median (sample minus atlas) at methylated sites {st.median(up):+.4f}, at unmethylated sites {st.median(dn):+.4f}, n {len(up)}",
          f"the blur hypothesis (sequencing more extreme than the atlas at both): {'supported' if st.median(up) > 0 and st.median(dn) < 0 else 'not supported'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L[-3:]))
