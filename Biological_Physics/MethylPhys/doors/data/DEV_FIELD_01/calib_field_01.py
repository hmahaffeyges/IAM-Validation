"""DEV-FIELD-01: GSE287331 IDATs through the chain's Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), unchanged, written
2026-10-13 AFTER the seal (sealed_field_01.json, commit a37fde82) and before any array is read.

Grn/Red paired by GSM prefix from the GEO RAW tar. One process per array, a pool of workers. For each array writes the FULL beta table
(parquet, probe names as the index - the reader asserts names are present, LESSONS: values without names read blank) and a small
table at the 6,000 sealed sites only, which is all the scorer reads. Usage: python3 calib_field_01.py RAWDIR FULLOUT SITESOUT WORKERS
"""
import glob, json, os, re, sys
import pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CHAIN", os.path.join(HERE, "..", "..", "..", "chain")))
from stage_1_idat_calibration import calibrate_idat_to_beta
RAW, FULL, SUB, NW = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SITES = json.load(open(os.environ.get("SEAL", os.path.join(HERE, "sealed_field_01.json"))))["sites"]
os.makedirs(FULL, exist_ok=True); os.makedirs(SUB, exist_ok=True)
pairs = {}
for f in glob.glob(os.path.join(RAW, "GSM*_Grn.idat*")) + glob.glob(os.path.join(RAW, "GSM*_Red.idat*")):
    g = re.match(r"(GSM\d+)", os.path.basename(f)).group(1); pairs.setdefault(g, {})["Grn" if "_Grn" in f else "Red"] = f
def one(g):
    out = os.path.join(SUB, g + ".parquet")
    if os.path.exists(out): return g, "done"
    p = pairs[g]
    if len(p) != 2: return g, "missing idat"
    o = calibrate_idat_to_beta(p["Grn"], p["Red"], verbose=False); b = o[0] if isinstance(o, tuple) else o
    b = pd.Series(b); assert str(b.index[0]).startswith("cg"), f"{g}: betas have no probe names"
    pd.DataFrame({"beta": b}).to_parquet(os.path.join(FULL, g + ".parquet"))
    pd.DataFrame({"beta": b.reindex(SITES)}).to_parquet(out)
    return g, f"probes {int(b.notna().sum())}, sealed sites {int(b.reindex(SITES).notna().sum())}"
if __name__ == "__main__":
    G = sorted(pairs); print("arrays", len(G), flush=True)
    print(*one(G[0]), flush=True)                      # one serially first, so the manifest is in place before the pool starts
    with Pool(NW) as P:
        for g, msg in P.imap_unordered(one, G[1:]): print(g, msg, flush=True)
