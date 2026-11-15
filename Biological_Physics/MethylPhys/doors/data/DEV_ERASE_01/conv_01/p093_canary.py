"""PHY-093 canary rule (design/PHY-093.json canary.instrument_check), run by CANON/pipeline_next.py on the canary's artifact directory:
quality-failed <= 0.05, mapped >= 0.95, seconds per pair <= 0.0015, and the islands count present with at least 500 opportunities on the
stray-mark side (count_sites.gain_opp, run_sample.py --dev-count-sites islands). Exit 0 = met."""
import sys, os, glob, json, re
d = sys.argv[1]
fs = glob.glob(os.path.join(d, "**", "*_flagstat.txt"), recursive=True); qj = glob.glob(os.path.join(d, "**", "*_q2.json"), recursive=True)
if not fs or not qj: print("canary files missing"); sys.exit(1)
t = open(fs[0]).read(); tot = int(re.search(r"^(\d+) \+ (\d+) in total", t, re.M).group(1)); qcf = int(re.search(r"^\d+ \+ (\d+) in total", t, re.M).group(1))
mp = float(re.search(r"mapped \(([\d.]+)%", t).group(1)) / 100
c = json.load(open(qj[0])); cs = c.get("count_sites") or {}; go = cs.get("gain_opp") or 0
tj = glob.glob(os.path.join(d, "**", "*_timing.json"), recursive=True)
spp = (json.load(open(tj[0]))["align_s"] / max(json.load(open(tj[0]))["pairs"], 1)) if tj else 0.0
ok = qcf / max(tot + qcf, 1) <= 0.05 and mp >= 0.95 and spp <= 0.0015 and cs.get("cell") == "islands" and go >= 500
print(f"quality-failed {qcf / max(tot + qcf, 1):.4f}, mapped {mp:.4f}, s/pair {spp:.5f}, islands opportunities {go}: {'MET' if ok else 'NOT MET'}"); sys.exit(0 if ok else 1)
