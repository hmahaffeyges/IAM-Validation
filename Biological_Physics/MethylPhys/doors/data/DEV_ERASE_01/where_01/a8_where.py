"""PHY-092 (exploration; register A8; design/PHY-092.json): where the mature B-cell knockouts' extra loose stretches sit. Reads only chain outputs: each
library's tiles (<GSM>_tiles.csv.gz from run_sample.py --dev-count-only --dev-count-tiles: chrom, tile, ..., calls, mean_call), recounted from the
molecules saved for PHY-086 P4 (pipeline items p092-<GSM>). A tile is loose below mean call 0.70 (the counter's own cut). Questions, written first:
  (1) null test: the two wild types disagree on few tiles (share of tiles loose in exactly one wild type);
  (2) do the tiles loose in both knockouts but in neither wild type sit in contiguous runs along each chromosome (domains, as when division outruns
      the copier) or scattered? Statistic: mean run length of such tiles, against 2,000 within-chromosome shuffles (one-sided p);
  (3) were they already graded in the wild type (wild-type mean call of those tiles against all held tiles)?
Usage: a8_where.py TILES_DIR [OUT]  (TILES_DIR holds the four <GSM>_tiles.csv.gz). Writes a8_where_summary.txt."""
import os, sys, gzip, csv, random, statistics as S
WT, KO, CUT, MIN = ("GSM4907469", "GSM4907470"), ("GSM4907471", "GSM4907472"), 0.70, 200
def load(d, g):
    T = {}
    with gzip.open(os.path.join(d, f"{g}_tiles.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            if r.get("mean_call") and int(r["calls"]) >= MIN: T[(r["chrom"], int(r["tile"]))] = float(r["mean_call"])
    return T
def runs(keys):
    by = {}
    for c, t in keys: by.setdefault(c, []).append(t)
    L = []
    for c, ts in by.items():
        ts.sort(); n = 1
        for a, b in zip(ts, ts[1:]):
            if b == a + 1: n += 1
            else: L.append(n); n = 1
        L.append(n)
    return S.mean(L) if L else 0.0
def main(d, out=None):
    T = {g: load(d, g) for g in WT + KO}; common = set.intersection(*(set(v) for v in T.values()))
    lo = {g: {k for k in common if T[g][k] < CUT} for g in T}
    one_wt = len(lo[WT[0]] ^ lo[WT[1]]) / len(common)
    ko_only = (lo[KO[0]] & lo[KO[1]]) - (lo[WT[0]] | lo[WT[1]])
    obs = runs(ko_only); rnd = random.Random(7); bychr = {}
    for c, t in common: bychr.setdefault(c, []).append(t)
    n_by = {}
    for c, t in ko_only: n_by[c] = n_by.get(c, 0) + 1
    null = []
    for _ in range(2000):
        pick = [(c, t) for c, n in n_by.items() for t in rnd.sample(bychr[c], n)]; null.append(runs(pick))
    p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    held = [k for k in common if all(T[g][k] >= CUT for g in WT)]
    wt_ko = [S.mean(T[g][k] for g in WT) for k in ko_only]; wt_held = [S.mean(T[g][k] for g in WT) for k in held]
    L = [f"tiles in all four libraries (>= {MIN} calls): {len(common)}",
         f"(1) null test: tiles loose in exactly one wild type: {one_wt:.4f} of tiles",
         f"(2) tiles loose in both knockouts, neither wild type: {len(ko_only)} ({len(ko_only) / len(common):.4f}); mean run length {obs:.2f} vs shuffled "
         f"{S.mean(null):.2f} (95 % {sorted(null)[1899]:.2f}); one-sided p = {p:.4f} -> {'CLUSTERED (domains)' if p < 0.01 else 'not clustered'}",
         f"(3) wild-type mean call of those tiles {S.mean(wt_ko) if wt_ko else float('nan'):.3f} vs all held tiles {S.mean(wt_held):.3f}"]
    open(os.path.join(out or d, "a8_where_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); return obs, p
if __name__ == "__main__": main(*sys.argv[1:3])
