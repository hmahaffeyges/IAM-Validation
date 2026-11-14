"""PHY-092 canary rule (design/PHY-092.json): the recount's tile file has >= 900 tiles with a mean call. Exit 0 = met."""
import sys, os, glob, gzip, csv
f = glob.glob(os.path.join(sys.argv[1], "**", "*_tiles.csv.gz"), recursive=True)
if not f: print("no tile file"); sys.exit(1)
n = sum(1 for r in csv.DictReader(gzip.open(f[0], "rt")) if r.get("mean_call"))
print(f"{n} tiles with a mean call: {'MET' if n >= 900 else 'NOT MET'}"); sys.exit(0 if n >= 900 else 1)
