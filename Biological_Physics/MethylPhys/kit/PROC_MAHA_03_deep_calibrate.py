#!/usr/bin/env python3
"""PROC-MAHA-03 deep arm: calibrate GSE87571 CHIP BY CHIP, caching after every chip.

Chip-complete ordering matters: B1 is a between-chip variance test, so a partly-processed chip is worth less than
a whole one. Chips are taken in sorted barcode order (deterministic, not cherry-picked) and the cache is written
after each, so the analysis can run on whatever is finished and nothing is lost to an interruption.
Measured cost: 26 s per array single-threaded, so the full 729 is about 5 hours; 4 threads per chip.
"""
import os, sys, re, json, glob, time, pickle, lzma, gzip, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "stage1")
from concurrent.futures import ThreadPoolExecutor
from stage_1_idat_calibration import calibrate_idat_to_beta

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
LOCI = set(json.load(open(glob.glob(CH + "/**/iamatlas_gauge_identity_loci_v1_0.json", recursive=True)[0]))["immune"]["loci"])
CACHE = "results/percell/stage1_betamean_GSE87571.json"
os.makedirs("results/percell", exist_ok=True)

pairs = {}
for p in glob.glob("idats_gse87571/*.idat.gz"):
    m = re.match(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz", os.path.basename(p))
    if m: pairs.setdefault(m.group(1), {})[m.group(4)] = p
with gzip.open("geo/GSE87571_series_matrix.txt.gz", "rt", errors="replace") as f:
    gsms = None; chips = None
    for line in f:
        if line.startswith("!Sample_geo_accession"): gsms = re.findall(r'"([^"]+)"', line)
        elif line.startswith("!Sample_supplementary_file") and "_R0" in line:
            got = [re.search(r"(\d{9,12})_(R0\dC0\d)", v) for v in re.findall(r'"([^"]+)"', line)]
            if any(got): chips = [m.group(1) if m else None for m in got]
        elif line.startswith("!series_matrix_table_begin"): break
chip_of = {g: chips[i] for i, g in enumerate(gsms) if chips[i]}

by_chip = collections.defaultdict(list)
for g, d in pairs.items():
    if len(d) == 2 and g in chip_of: by_chip[chip_of[g]].append(g)

done = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
print(f"{len(by_chip)} chips, {sum(len(v) for v in by_chip.values())} arrays with a pair | already cached: {len(done)}", flush=True)


def one(g):
    try:
        beta, _ = calibrate_idat_to_beta(pairs[g]["Grn"], pairs[g]["Red"])
        s = beta.iloc[:, 0] if hasattr(beta, "columns") else beta
        s = s[[i for i in s.index if i in LOCI]].dropna()
        return (g, float(s.mean()) if len(s) >= 1000 else None, len(s))
    except Exception as e:
        return (g, None, f"ERR {type(e).__name__}")


t0 = time.time(); nchip = 0
for chip in sorted(by_chip):
    todo = [g for g in by_chip[chip] if g not in done]
    if not todo:
        nchip += 1; continue
    with ThreadPoolExecutor(max_workers=4) as ex:
        for g, val, nl in ex.map(one, todo):
            if val is not None: done[g] = val
    tmp = CACHE + ".tmp"
    json.dump(done, open(tmp, "w")); os.replace(tmp, CACHE)   # atomic, so a reader never sees a half-written file
    nchip += 1
    el = time.time() - t0
    print(f"  chip {chip} done ({len(by_chip[chip])} arrays) | {nchip}/{len(by_chip)} chips, {len(done)} arrays "
          f"cached, {el/60:.1f} min elapsed, ~{(el/max(len(done),1))*(sum(len(v) for v in by_chip.values())-len(done))/60:.0f} min left",
          flush=True)
print(f"CALIBRATION DONE: {len(done)} arrays across {nchip} chips in {(time.time()-t0)/60:.1f} min", flush=True)
