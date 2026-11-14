"""PHY-090 canary rule (design/PHY-090.json canary.instrument_check), computed from the canary's own committed files (GitHub run
43564679932, GSM1866966, 2 million pairs). Writes canary_summary.txt."""
import os, re, json
H = os.path.dirname(os.path.abspath(__file__)); f = open(os.path.join(H, "GSM1866966_flagstat.txt")).read()
p, q = map(int, re.search(r"(\d+) \+ (\d+) in total", f).groups()); mp = float(re.search(r"mapped \(([\d.]+)%", f).group(1)) / 100
T = json.load(open(os.path.join(H, "GSM1866966_timing.json"))); spp = T["align_s"] / T["pairs"]; qf = q / (p + q)
L = [f"quality-failed {qf:.3f} (limit 0.05), mapped {mp:.3f} (minimum 0.95), s per pair {spp:.5f} (limit 0.0015): "
     f"{'canary rule met' if qf <= 0.05 and mp >= 0.95 and spp <= 0.0015 else 'canary rule NOT met'}; fetch {T['fetch_s']} s, align {T['align_s']} s"]
open(os.path.join(H, "canary_summary.txt"), "w").write("\n".join(L) + "\n"); print(L[0])
