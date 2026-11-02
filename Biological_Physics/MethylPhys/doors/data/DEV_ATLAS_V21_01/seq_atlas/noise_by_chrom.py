"""Noise map 1 (donor half A minus half B) of the IAM Sequencing Atlas by chromosome, per cell, from the noise columns built by the one
command (run_sample.py --build seq-noise-cols). Nothing scored. Usage: python3 noise_by_chrom.py NOISE_COLS_DIR OUT.csv OUT.txt"""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_metaa_atlas as SA
D = sys.argv[1]; ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe").chrom_hg19.reindex(ids).to_numpy()
auto = np.isin(K, [f"chr{i}" for i in range(1, 23)]); X = K == "chrX"; rows = []
for k in (1, 2, 3):
    A, B = (pq.read_table(os.path.join(D, f"half{h}_part{k}.parquet")) for h in ("A", "B"))
    for c in A.column_names:
        a = A.column(c).to_numpy().astype(float); b = B.column(c).to_numpy().astype(float); d = np.where((a == 65535) | (b == 65535), np.nan, (a - b) / 1e4)
        if not np.isfinite(d).any(): continue
        rows.append(dict(cell=c, mean_autosomes=np.nanmean(d[auto]), ms_autosomes=np.nanmean(d[auto] ** 2), mean_chrX=np.nanmean(d[X]), ms_chrX=np.nanmean(d[X] ** 2)))
R = pd.DataFrame(rows).round(5); R["chrX_over_autosomes"] = (R.ms_chrX / R.ms_autosomes).round(3); R.sort_values("cell").to_csv(sys.argv[2], index=False)
L = [f"cells with both donor halves: {len(R)}; chrX mean square over the autosomes' (median) {R.chrX_over_autosomes.median():.3f}; cells above 2: {int((R.chrX_over_autosomes > 2).sum())}",
     f"largest autosome offset between donor halves: {R.loc[R.mean_autosomes.abs().idxmax(), 'cell']} {R.mean_autosomes.abs().max():.4f}; median cell {R.mean_autosomes.abs().median():.4f}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
