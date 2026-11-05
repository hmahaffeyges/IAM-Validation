"""Are BLUEPRINT's public methylation tracks plain read counts? (diagnostic on spent libraries; nothing scored)

The chain reads a BLUEPRINT library from two GRCh38 bigWig tracks, a per-CpG level and its coverage, and treats level x coverage as the
methylated read count (stage_seq_values.values_from_bigwig). Every sequencing reading counts disorder from those counts, so they have to be
counts. This checks it on chromosomes 21 and 22 of the libraries named, read over the network (pyBigWig, the chain's requirement):

  1. the share of CpGs whose level x coverage lies within 0.01 of a whole number (1.0 if the level is methylated reads over coverage);
  2. at coverage exactly 10, how many distinct levels appear (11 at most for counts: 0, 0.1, ..., 1.0);
  (a level that is smoothed, shrunk toward its neighbors or a model estimate fails both).

Usage: python3 blueprint_track_counts.py SAMPLES.csv N OUT.csv OUT.txt   (SAMPLES.csv: blueprint_healthy_test_samples.csv; first N rows)
"""
import sys, numpy as np, pandas as pd, pyBigWig

def local(u):
    """A local copy of a track (pyBigWig is not always built to open web addresses): downloaded with retries, removed after reading."""
    import subprocess, tempfile, os
    if not u.startswith(("http://", "https://", "ftp://")): return u, False
    f = os.path.join(tempfile.gettempdir(), os.path.basename(u))
    subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-delay", "10", "-o", f, u], check=True); return f, True

def one(call_bw, cov_bw, chroms=("chr21", "chr22")):
    import os
    (call_bw, d1), (cov_bw, d2) = local(call_bw), local(cov_bw)
    lv, cv = [], []
    bc, bv = pyBigWig.open(call_bw), pyBigWig.open(cov_bw)
    for ch in chroms:
        if ch not in bc.chroms(): continue
        L = {a: v for a, b, v in (bc.intervals(ch) or [])}; C = {a: v for a, b, v in (bv.intervals(ch) or [])}
        for a, v in L.items():
            if a in C and C[a] > 0: lv.append(v); cv.append(C[a])
    bc.close(); bv.close()
    for f, d in ((call_bw, d1), (cov_bw, d2)):
        if d: os.remove(f)
    lv, cv = np.array(lv), np.array(cv)
    m = lv * cv; whole = float(np.mean(np.abs(m - np.round(m)) < 0.01))
    # added after the first run (60 % whole, falling with depth): is the level a count rounded to a fixed number of decimals? For each
    # precision, the share of levels on that grid, and the share where the nearest whole count, divided by coverage and rounded to the
    # same precision, gives the level back (1.0 = the count is recovered exactly from the track)
    grid = {}
    for dp in (2, 3, 4):
        on = float(np.mean(np.abs(lv * 10 ** dp - np.round(lv * 10 ** dp)) < 1e-3))
        back = float(np.mean(np.abs(np.round(np.round(m) / cv, dp) - np.round(lv, dp)) < 10 ** -(dp + 2)))
        grid[f"on_grid_{dp}dp"] = round(on, 4); grid[f"count_recovered_{dp}dp"] = round(back, 4)
    at10 = lv[cv == 10]; distinct10 = int(len(np.unique(np.round(at10, 6))))
    return dict(cpgs=int(len(lv)), coverage_median=float(np.median(cv)), coverage_is_whole=float(np.mean(np.abs(cv - np.round(cv)) < 1e-6)),
                level_x_coverage_whole=round(whole, 4), **grid, distinct_levels_at_coverage_10=distinct10, cpgs_at_coverage_10=int(len(at10)))

if __name__ == "__main__":
    S = pd.read_csv(sys.argv[1]).head(int(sys.argv[2])); rows = []
    for _, r in S.iterrows():
        rec = dict(gsm=r.gsm, atlas_cell=r.atlas_cell); rec.update(one(r.call_bw, r.cov_bw)); rows.append(rec); print(rec, flush=True)
    T = pd.DataFrame(rows); T.to_csv(sys.argv[3], index=False)
    L = [f"BLUEPRINT tracks, chromosomes 21 and 22, {len(T)} libraries (diagnostic, spent libraries, nothing scored)",
         f"level x coverage within 0.01 of a whole number: median share {T.level_x_coverage_whole.median():.4f} (range {T.level_x_coverage_whole.min():.4f} to {T.level_x_coverage_whole.max():.4f})",
         f"coverage a whole number: median share {T.coverage_is_whole.median():.4f}",
         *[f"levels on the {dp}-decimal grid: median share {T[f'on_grid_{dp}dp'].median():.4f}; nearest whole count gives the level back: {T[f'count_recovered_{dp}dp'].median():.4f}" for dp in (2, 3, 4)],
         f"distinct levels at coverage 10: median {T.distinct_levels_at_coverage_10.median():.0f} (counts allow at most 11)"]
    open(sys.argv[4], "w").write("\n".join(L) + "\n"); print("\n".join(L))
