"""PHY-094 canary rule (design/PHY-094.json canary.instrument_check), run by CANON/pipeline_next.py on the canary's artifact directory.
The count reads a molecule file kept by PHY-084 (patfrom; no alignment). Met when: the chain's report is present with the clock-site count
(count_sites.cell = epitoc2, at least 200 opportunities on the stray-mark side), and the genome-wide stray-mark rate agrees with PHY-091's
aligned-file count of the same library (min/max >= 0.85; conv_01/a17_read.csv). Exit 0 = met."""
import sys, os, glob, json, csv
d = sys.argv[1]; H = os.path.dirname(os.path.abspath(__file__))
qj = glob.glob(os.path.join(d, "**", "*_q2.json"), recursive=True)
if not qj: print("canary report missing"); sys.exit(1)
c = json.load(open(qj[0])); cs = c.get("count_sites") or {}; go = cs.get("gain_opp") or 0
A = {r["gsm"]: float(r["gain_err"]) for r in csv.DictReader(open(os.path.join(H, "a17_read.csv"))) if r.get("gain_err")}
g = c.get("gsm"); a, b = c.get("gain_err"), A.get(g)
agree = (min(a, b) / max(a, b)) if a and b else 0.0
ok = cs.get("cell") == "epitoc2" and go >= 200 and agree >= 0.85
print(f"{g}: clock-site opportunities {go}, clock-site stray rate {cs.get('gain_err')}, genome-wide {a} vs PHY-091 {b} (agreement {agree:.3f}): {'MET' if ok else 'NOT MET'}")
sys.exit(0 if ok else 1)
